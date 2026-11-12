from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from src.poomsae_scoring.contracts import ScoringContractError
from src.poomsae_scoring.fixation import FixationParameters
from src.poomsae_scoring.fixation_annotation import ANNOTATION_JS, build_independent_annotation_html
from src.poomsae_scoring.fixation_audit import attach_camera_audit, build_fixation_audit
from src.poomsae_scoring.fixation_experiments import build_fixation_sensitivity
from src.poomsae_scoring.automatic_segmentation import detect_automatic_segments
from src.poomsae_scoring.fixation_validation import (
    annotation_template, build_fixation_benchmark, match_events, recording_binding, validate_annotation, validate_dataset_split,
)


def _report():
    return {"frame_count": 180, "fps": 60.0,
            "bindings": {"pose": {"sha256": "a" * 64}, "videos": {"camera-a": {"sha256": "b" * 64}}},
            "fixation_candidates": [
                {"candidate_episode_id": 0, "peak_frame": 30, "completion_candidate_frame": 60,
                 "fixation_frame": 66, "hold_end_frame": 75, "status": "hold_observed"},
                {"candidate_episode_id": 1, "peak_frame": 100, "completion_candidate_frame": 120,
                 "fixation_frame": None, "hold_end_frame": None, "status": "insufficient_evidence"},
            ]}


def _event(identifier="E1", kind="hold_onset", start=65, end=67):
    return {"event_id": identifier, "kind": kind, "start_frame": start, "end_frame": end, "note": ""}


def _annotation(report=None):
    report = report or _report()
    data = annotation_template(recording_binding(report))
    data.update(reviewer="rater-a", athlete_id="athlete-a", session_id="session-a", reviewed_at="2026-10-18T10:00:00Z",
                blinded_to_predictions=True, coverage_complete=True, events=[_event()])
    return data


def test_annotation_identity_ignores_detector_and_pose_but_not_recording():
    original = _report()
    changed = deepcopy(original)
    changed["fixation_candidates"][0]["fixation_frame"] = 69
    changed["bindings"]["pose"]["sha256"] = "c" * 64
    changed["bindings"]["videos"]["camera-a"]["path"] = "relocated/video.mp4"
    assert recording_binding(original) == recording_binding(changed)
    changed["bindings"]["videos"]["camera-a"]["sha256"] = "d" * 64
    assert recording_binding(original) != recording_binding(changed)
    changed = deepcopy(original)
    changed["fps"] = 30
    assert recording_binding(original) != recording_binding(changed)


def test_same_reviewer_repeated_rounds_measure_repeatability_not_two_experts():
    report = _report()
    first = _annotation(report)
    second = deepcopy(first)
    second["annotation_round"] = "tur-2"
    second["events"][0].update(start_frame=70, end_frame=72)
    result = build_fixation_benchmark(report, [first, second], [100])
    assert result["reviewer_count"] == 1 and result["annotation_count"] == 2
    pair = result["reviewer_agreement"][0]
    assert pair["comparison_kind"] == "intra_rater_repeatability"
    assert pair["annotation_rounds"] == ["tur-1", "tur-2"]
    assert pair["metrics"]["hold_onset"][0]["timing_mae_frames"] == 3
    assert result["deployment_gate"]["external_accuracy_verified"] is False
    second["annotation_round"] = "tur-1"
    with pytest.raises(ScoringContractError):
        build_fixation_benchmark(report, [first, second])


def test_legacy_annotation_is_readable_and_has_an_explicit_legacy_round():
    report = _report()
    annotation = _annotation(report)
    annotation["schema_version"] = 1
    annotation.pop("annotation_round")
    validate_annotation(annotation, recording_binding(report))
    result = build_fixation_benchmark(report, [annotation])
    assert result["per_reviewer"][0]["annotation_round"] == "legacy-1"


@pytest.mark.parametrize("mutation", ["foreign", "frame_bool", "outside", "duplicate", "contradiction", "uncertain", "incomplete_type", "reviewer"])
def test_annotation_rejects_invalid_or_contradictory_labels(mutation):
    report = _report()
    data = _annotation(report)
    if mutation == "foreign":
        data["binding"]["recording_id"] = "foreign"
    elif mutation == "frame_bool":
        data["events"][0]["start_frame"] = True
    elif mutation == "outside":
        data["coverage"]["start_frame"] = 70
    elif mutation == "duplicate":
        data["events"].append(_event(identifier="E2"))
    elif mutation == "contradiction":
        data["events"].append(_event(identifier="E2", kind="no_hold"))
    elif mutation == "uncertain":
        data["events"].append(_event(identifier="E2", kind="uncertain"))
    elif mutation == "incomplete_type":
        data["coverage_complete"] = "true"
    else:
        data["reviewer"] = " "
    with pytest.raises(ScoringContractError):
        validate_annotation(data, recording_binding(report))


def test_matching_maximizes_count_before_nearest_distance():
    refs = [_event("R1", start=0, end=0), _event("R2", start=5, end=5)]
    predictions = [{"event_id": "P1", "frame": 5}, {"event_id": "P2", "frame": 10}]
    result = match_events(predictions, refs, 5)
    assert result["true_positives"] == 2
    assert result["timing_mae_frames"] == 5
    duplicate = match_events(predictions, refs[1:], 5)
    assert duplicate["true_positives"] == 1 and duplicate["false_positives"] == 1
    assert duplicate["timing_mae_frames"] == 0


def test_matching_intervals_and_no_predictions_are_not_perfect_accuracy():
    refs = [_event(start=10, end=15)]
    exact = match_events([{"event_id": "P", "frame": 12}], refs, 0)
    assert exact["timing_mae_frames"] == 0
    missed = match_events([], refs, 6)
    assert missed["precision"] is None and missed["recall"] == 0 and missed["f1"] == 0
    empty = match_events([], [], 6)
    assert empty["f1"] is None and empty["timing_mae_frames"] is None
    first = match_events([_event("P", start=2, end=8)], refs, 2)
    reverse = match_events([_event("P", start=10, end=15)], [_event("R", start=2, end=8)], 2)
    assert first["timing_mae_frames"] == reverse["timing_mae_frames"] == 2


def test_benchmark_semantics_uncertainty_and_review_completeness():
    report = _report()
    annotation = _annotation(report)
    annotation["events"] += [_event("E2", "completion", 59, 61), _event("E3", "uncertain", 118, 125)]
    result = build_fixation_benchmark(report, [annotation], [0])
    row = result["per_reviewer"][0]
    assert row["excluded_uncertain_frames"] == 8
    assert row["evaluable_frames"] == 172
    assert row["metrics"]["completion"][0]["true_positives"] == 1
    assert row["metrics"]["completion"][0]["ignored_predictions"] == 1
    assert row["metrics"]["hold_onset"][0]["true_positives"] == 1
    assert row["metrics"]["hold_end"][0]["false_positives"] == 1
    assert result["deployment_gate"]["automatic_deployment_allowed"] is False
    annotation["coverage_complete"] = False
    pending = build_fixation_benchmark(report, [annotation])
    assert pending["per_reviewer"] == [] and pending["incomplete_reviewers"] == ["rater-a"]
    assert pending["status"] == "awaiting_independent_annotations"
    assert build_fixation_benchmark(report, [])["per_reviewer"] == []


def test_unknown_event_region_excludes_nearby_predictions_at_each_tolerance():
    report = _report()
    report["fixation_candidates"][1]["completion_candidate_frame"] = 116
    annotation = _annotation(report)
    annotation["events"] = [_event("unknown", "uncertain", 118, 125)]
    metrics = build_fixation_benchmark(report, [annotation], [0, 100])["per_reviewer"][0]["metrics"]["completion"]
    assert metrics[0]["false_positives"] == 2
    assert metrics[1]["false_positives"] == 1
    assert metrics[1]["ignored_predictions"] == 1
    assert metrics[1]["excluded_uncertainty_guard_frames"] == 20


def test_no_hold_interval_counts_false_positive_and_no_consensus_is_invented():
    report = _report()
    a = _annotation(report)
    a["events"] = [_event(kind="no_hold", start=60, end=80)]
    b = _annotation(report)
    b["reviewer"] = "rater-b"
    b["events"][0].update(start_frame=80, end_frame=82)
    result = build_fixation_benchmark(report, [a, b], [0])
    assert result["per_reviewer"][0]["metrics"]["hold_onset"][0]["false_positives"] == 1
    assert result["reviewer_agreement"][0]["metrics"]["hold_onset"][0]["false_negatives"] == 1
    assert result["deployment_gate"]["external_accuracy_verified"] is False
    b["coverage"]["start_frame"] = 70
    different_scope = build_fixation_benchmark(report, [a, b])["reviewer_agreement"][0]
    assert different_scope["status"] == "different_review_scopes" and different_scope["metrics"] == {}
    b["reviewer"] = " RATER-A "
    with pytest.raises(ScoringContractError):
        build_fixation_benchmark(report, [a, b])


@pytest.mark.parametrize("mutation", ["duplicate", "boolean_frame", "status", "end", "nan_tolerance"])
def test_benchmark_rejects_malformed_predictions(mutation):
    report = _report()
    tolerance = [100]
    if mutation == "duplicate":
        report["fixation_candidates"][1]["candidate_episode_id"] = 0
    elif mutation == "boolean_frame":
        report["fixation_candidates"][0]["fixation_frame"] = True
    elif mutation == "status":
        report["fixation_candidates"][0]["status"] = "insufficient_evidence"
    elif mutation == "end":
        report["fixation_candidates"][0]["hold_end_frame"] = 65
    else:
        tolerance = [float("nan")]
    with pytest.raises(ScoringContractError):
        build_fixation_benchmark(report, [], tolerance)


@pytest.mark.parametrize("leak", [None, "athlete_id", "session_id", "recording_id"])
def test_dataset_split_prevents_same_athlete_session_or_recording_leakage(leak):
    rows = [{"recording_id": str(i), "athlete_id": f"a{i}", "session_id": f"s{i}", "split": split}
            for i, split in enumerate(("development", "validation", "test"))]
    if leak:
        rows[2][leak] = rows[0][leak]
        with pytest.raises(ScoringContractError):
            validate_dataset_split({"schema_version": 1, "recordings": rows})
    else:
        audit = validate_dataset_split({"schema_version": 1, "recordings": rows})
        assert audit["status"] == "declared_groups_separated"
        assert audit["athlete_count"] == 3


def test_camera_audit_keeps_missing_and_multiple_person_rows_ambiguous(tmp_path):
    path = tmp_path / "measurements.csv"
    path.write_text('frame_idx,camera_id,joint_idx,x,y,score,valid\n0,A,5,1,2,0.9,True\n1,A,5,2,3,0.8,True\n1,A,5,4,5,0.7,True\n', encoding="utf-8")
    audit = {"frame_count": 3, "candidates": [{"window": [0, 2]}]}
    attach_camera_audit(audit, path, ["A"])
    evidence = audit["candidates"][0]["camera_2d"]["A"]["left_shoulder"]
    assert evidence["valid_finite_frames"] == 1
    assert evidence["ambiguous_frames"] == 1
    assert evidence["missing_frames"] == 1
    assert evidence["score"]["median"] == .9


def test_audit_preserves_reliability_rejection_and_explains_derived_hand_gap():
    points = np.zeros((180, 133, 3))
    for hip, knee, ankle in ((11, 13, 15), (12, 14, 16)):
        points[:, hip, 2] = .9
        points[:, knee, 2] = .5
        points[:, ankle, 2] = .1
    points[:, 5, 0], points[:, 6, 0] = -.2, .2
    points[:, 11, 0], points[:, 12, 0] = -.2, .2
    valid = np.ones((180, 133), bool)
    valid[60:101, 11] = False
    pose = {"keypoints_3d_world": points, "sample_fps": 60, "used_cameras": np.full((180, 133), 2),
            "reprojection_error": np.ones((180, 133)), "reliability_valid_mask": valid,
            "reliability_rejection_reasons": (~valid).astype(int)}
    report = {"frame_count": 180, "fps": 60, "detector": {"fixation": {"parameters": vars(FixationParameters())}},
              "fixation_candidates": [{"candidate_episode_id": 0, "completion_candidate_frame": 60, "peak_frame": 30,
                                        "end_frame": 100, "status": "insufficient_evidence", "components": {}}]}
    audit = build_fixation_audit(pose, report)
    row = audit["candidates"][0]
    assert row["components"]["left_hand"]["valid_ratio"] == 0
    assert row["components"]["right_hand"]["valid_ratio"] == 0
    assert row["components"]["torso"]["valid_ratio"] == 1
    joint = row["joints"]["left_hip"]
    assert joint["rejected_frame_counts"]["upstream_reliability_rejected"] == 41
    assert joint["rejected_frame_counts"]["insufficient_cameras"] == 0
    assert joint["upstream_reason_bits"]["base_quality"] == 41
    json.dumps(audit, allow_nan=False)


def test_blind_page_contains_no_model_proposals_or_existing_labels():
    page = build_independent_annotation_html(recording_binding(_report()), {"cam < a": "a%20b.mp4"})
    assert "cam &lt; a" in page and "requestVideoFrameCallback" in page
    assert "fixation_candidates" not in page and "completion_candidate_frame" not in page
    assert "review_binding" not in page and "source_report" not in page


def test_parameter_sensitivity_keeps_quality_episodes_and_labels_out_of_search():
    n = 180
    points = np.zeros((n, 133, 3))
    for joint, height in ((11, .9), (12, .9), (13, .5), (14, .5), (15, .1), (16, .1), (5, 1.5), (6, 1.5)):
        points[:, joint, 2] = height
    points[:, 5, 0] = points[:, 11, 0] = -.2
    points[:, 6, 0] = points[:, 12, 0] = .2
    points[:, :, 0] += (np.clip((np.arange(n) - 30) / 30, 0, 1) * .4)[:, None]
    pose = {"keypoints_3d_world": points, "sample_fps": 60,
            "reliability_valid_mask": np.ones((n, 133), bool),
            "used_cameras": np.full((n, 133), 2), "reprojection_error": np.ones((n, 133))}
    detection = detect_automatic_segments(points, fps=60, quality=pose, movements=[{"movement_id": "M01", "phases": ["preparation", "execution", "fixation"]}])
    report = {"frame_count": n, "fps": 60, "detector": {"fixation": {"parameters": vars(FixationParameters())}},
              "candidate_episodes": detection["candidate_episodes"], "fixation_candidates": detection["fixation_candidates"]}
    study = build_fixation_sensitivity(pose, report)
    assert study["baseline_reproduced"] and study["configuration_count"] == 9
    assert not study["quality_gate_relaxed"] and not study["automatic_parameter_selection"]
    assert all(v["parameters"]["maximum_reprojection_px"] == 15 and v["parameters"]["minimum_cameras"] == 2 for v in study["variants"])
    report["reference_comparison"] = {"untrusted_manual_frame": 150}
    assert build_fixation_sensitivity(pose, report) == study
    report["fixation_candidates"][0]["fixation_frame"] += 1
    with pytest.raises(ScoringContractError):
        build_fixation_sensitivity(pose, report)


def test_actual_annotation_javascript_export_and_foreign_import(tmp_path):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is optional")
    template = annotation_template(recording_binding(_report()))
    setup = r"""
const assert = require('node:assert/strict');
const elements = new Map();
const make = () => ({value:'',checked:false,textContent:'',dataset:{},children:[],appendChild(x){this.children.push(x);},replaceChildren(){this.children=[];},addEventListener(){},click(){}});
const document = {getElementById(id){if(!elements.has(id))elements.set(id,make());return elements.get(id);},querySelectorAll(){return [];},createElement(){return make();}};
let exported=null;
const URL={createObjectURL(blob){exported=blob;return 'blob:local';},revokeObjectURL(){}};
const store=new Map(),localStorage={setItem:(k,v)=>store.set(k,v),getItem:k=>store.get(k)};
const setTimeout=()=>{};
"""
    initialize = f"document.getElementById('annotation-data').textContent={json.dumps(json.dumps({'template': template}))};\n"
    assertions = r"""
(async()=>{
const $=id=>document.getElementById(id);
for(const [id,value] of Object.entries({reviewer:'judge-a',annotation_round:'tur-1',athlete_id:'athlete-a',session_id:'session-a','coverage-start':'0','coverage-end':'179','event-start':'65','event-end':'67','event-kind':'hold_onset'})) $(id).value=value;
$('blinded_to_predictions').checked=true;
$('save-event').onclick();
assert.equal($('events').children.length,1);
$('save-event').onclick();
assert.equal($('events').children.length,1); // duplicate overlapping event rejected
$('coverage_complete').checked=true;
$('export').onclick();
assert.ok(exported);
const payload=JSON.parse(await exported.text());
$('reviewer').value='other-rater';
$('export').onclick();
assert.match($('status').textContent,/İnceleyen veya tur değişti/);
$('reviewer').value='judge-a';
$('annotation_round').value='tur-2';
$('export').onclick();
assert.match($('status').textContent,/İnceleyen veya tur değişti/);
$('annotation_round').value='tur-1';
const foreign=JSON.parse(JSON.stringify(payload));foreign.binding.recording_id='foreign';
await $('import').onchange({target:{files:[{text:async()=>JSON.stringify(foreign)}],value:'x'}});
assert.match($('status').textContent,/eşleşmiyor/);
$('export').onclick();
assert.equal(JSON.parse(await exported.text()).events.length,1);
$('new-review').onclick();
assert.equal($('events').children.length,0);
$('reviewer').value='judge-a';$('annotation_round').value='tur-2';
$('load-local').onclick();
assert.match($('status').textContent,/bulunamadı/);
$('annotation_round').value='tur-1';$('load-local').onclick();
assert.equal($('events').children.length,1);
console.log(JSON.stringify(payload));
})().catch(error=>{console.error(error);process.exitCode=1;});
"""
    path = Path(tmp_path) / "annotation-ui.js"
    path.write_text(setup + initialize + ANNOTATION_JS + assertions, encoding="utf-8")
    result = subprocess.run([node, str(path)], capture_output=True, text=True, encoding="utf-8")
    assert result.returncode == 0, result.stderr
    validate_annotation(json.loads(result.stdout), recording_binding(_report()))
