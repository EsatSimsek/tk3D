"""Physical counterexamples, label independence and annotation binding checks."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from src.poomsae_scoring.automatic_segmentation import (
    build_automatic_segmentation_diagnostics, detect_automatic_segments,
)
from src.poomsae_scoring.contracts import ScoringContractError, load_movement_timeline, load_poomsae_spec
from src.poomsae_scoring.fixation import (
    FixationParameters, build_fixation_review_binding, build_fixation_timing_diagnostics,
    evaluate_fixation_candidates, prepare_fixation_signals,
    validate_fixation_review_export,
)


def _pose(fps=60, seconds=3):
    frame = np.zeros((133, 3))
    for left, right, height in ((5, 6, 1.5), (7, 8, 1.2), (9, 10, 1.0),
                                (11, 12, .9), (13, 14, .5), (15, 16, .1)):
        frame[left] = [-.2, 0, height]
        frame[right] = [.2, 0, height]
    return np.repeat(frame[None], int(fps * seconds), axis=0)


def _quality(points):
    return {"reliability_valid_mask": np.ones(points.shape[:2], dtype=bool),
            "used_cameras": np.full(points.shape[:2], 2),
            "reprojection_error": np.ones(points.shape[:2])}


def _detect(points, fps=60, quality=None, count=1):
    return detect_automatic_segments(
        points, fps=fps, quality=_quality(points) if quality is None else quality,
        movements=[{"movement_id": f"M{i + 1:02}", "phases": ["preparation", "execution", "fixation"]} for i in range(count)],
    )


def _translation(points, fps):
    times = np.arange(len(points)) / fps
    points[:, :, 0] += np.clip(times - .5, 0, .5)[:, None] * .6
    return points


@pytest.mark.parametrize("fps", [30, 60, 120])
def test_observed_stop_has_time_based_support_and_no_future_end_anchor(fps):
    result = _detect(_translation(_pose(fps), fps), fps)
    event = result["fixation_candidates"][0]
    assert event["status"] == "hold_observed"
    assert 1.0 <= event["fixation_frame"] / fps <= 1.12
    assert (event["confirmation_frame"] - event["fixation_frame"]) / fps >= .10 - 1e-9
    assert event["confidence_probability"] is None


def test_continuous_motion_cannot_become_fixation_at_recording_end():
    points = _pose()
    points[:, :, 0] += np.arange(len(points))[:, None] * .01
    result = _detect(points)
    assert result["fixation_candidates"]
    assert all(row["fixation_frame"] is None for row in result["fixation_candidates"])
    assert result["fixation_candidates"][-1]["status"] == "recording_truncated"


def test_static_recording_is_not_a_completed_technique():
    result = _detect(_pose())
    assert result["fixation_candidates"] == []
    json.dumps(result["fixation_detector"], allow_nan=False)


def test_all_invalid_quality_abstains_instead_of_claiming_valid_signal():
    points = _translation(_pose(), 60)
    quality = _quality(points)
    quality["reliability_valid_mask"][:] = False
    result = _detect(points, quality=quality)
    assert result["valid_signal_ratio"] == 0
    assert result["fixation_candidates"] == []


@pytest.mark.parametrize("change", ["one_camera", "bad_reprojection", "hidden_hand", "missing_metadata", "stride"])
def test_required_evidence_is_not_replaced_by_other_stationary_joints(change):
    points = _translation(_pose(), 60)
    quality = _quality(points)
    if change == "one_camera":
        quality["used_cameras"][:, 9] = 1
    elif change == "bad_reprojection":
        quality["reprojection_error"][:, 9] = 100
    elif change == "hidden_hand":
        quality["reliability_valid_mask"][:, 9] = False
    elif change == "missing_metadata":
        quality = {}
    else:
        quality["inference_stride"] = 2
    result = _detect(points, quality=quality)
    assert result["fixation_candidates"]
    assert all(row["fixation_frame"] is None for row in result["fixation_candidates"])
    assert all(row["status"] == "insufficient_evidence" for row in result["fixation_candidates"])


def test_moving_hand_cannot_be_diluted_by_still_feet_and_torso():
    points = _translation(_pose(), 60)
    points[:, 9, 1] += np.maximum(np.arange(len(points)) / 60 - 1, 0) * .5
    result = _detect(points)
    assert all(row["fixation_frame"] is None for row in result["fixation_candidates"])


def test_late_correction_does_not_replace_the_first_supported_hold():
    points = _translation(_pose(), 60)
    points[:, 9, 1] += np.clip(np.arange(len(points)) / 60 - 1.5, 0, .2) * .5
    signals = prepare_fixation_signals(points, 60, quality=_quality(points))
    events = evaluate_fixation_candidates(signals, [
        {"candidate_episode_id": 0, "start_frame": 30, "peak_frame": 40, "end_frame": 62}
    ])
    event = events[0]
    assert event["fixation_frame"] < 80
    assert 90 <= event["later_motion_frame"] <= 96
    assert event["completion_candidate_frame"] == 63


def test_missing_samples_do_not_fill_a_short_hold():
    points = _translation(_pose(seconds=1.2), 60)
    quality = _quality(points)
    quality["reliability_valid_mask"][65:69, 9] = False
    result = _detect(points, quality=quality)
    assert all(row["fixation_frame"] is None for row in result["fixation_candidates"])


def test_missing_pelvis_preserves_absolute_hand_evidence_but_cannot_confirm_hold():
    points = _translation(_pose(), 60)
    quality = _quality(points)
    quality["reliability_valid_mask"][60:, 11:13] = False
    signals = prepare_fixation_signals(points, 60, quality=quality)
    events = evaluate_fixation_candidates(signals, [
        {"candidate_episode_id": 0, "start_frame": 30, "peak_frame": 40, "end_frame": 65}
    ])
    diagnostic = build_fixation_timing_diagnostics(signals, events)[0]
    assert diagnostic["hands"]["left_hand"] == {"absolute_valid_ratio": 1.0, "pelvis_relative_valid_ratio": 0.0}
    assert "left_hand" in diagnostic["missing_components"]
    assert events[0]["fixation_frame"] is None
    assert diagnostic["speed_unsmoothed_onset_frame"] is None
    json.dumps(diagnostic, allow_nan=False)


@pytest.mark.parametrize("interruption", ["motion", "missing", "later_hold"])
def test_unsmoothed_timing_check_does_not_hide_motion_or_adopt_later_hold(interruption):
    signals = prepare_fixation_signals(_translation(_pose(), 60), 60, quality=_quality(_pose()))
    events = evaluate_fixation_candidates(signals, [
        {"candidate_episode_id": 0, "start_frame": 30, "peak_frame": 40, "end_frame": 65}
    ])
    event = events[0]
    start = event["fixation_frame"]
    before = deepcopy(event)
    values = signals["raw_channels"]["left_hand"]
    if interruption == "later_hold":
        event["hold_end_frame"] = start + 6
        before = deepcopy(event)
    values[start + 2] = np.nan if interruption == "missing" else 100.0
    row = build_fixation_timing_diagnostics(signals, events)[0]
    assert events[0] == before
    if interruption == "later_hold":
        assert row["speed_unsmoothed_onset_frame"] is None
        assert row["status"] == "not_confirmed_without_speed_smoothing"
    else:
        assert row["speed_unsmoothed_onset_frame"] == start + 3
        assert row["shift_frames"] == 3
        assert row["confirmation_frame"] >= start + 9
    assert row["replaces_automatic_proposal"] is False


def test_normalization_is_invariant_to_scale_translation_and_rotation():
    points = _translation(_pose(), 60)
    expected = _detect(points)["fixation_candidates"][0]["fixation_frame"]
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    changed = (points @ rotation.T) * 1.6 + np.array([4, -2, 1])
    assert _detect(changed)["fixation_candidates"][0]["fixation_frame"] == expected


def test_expected_count_changes_neither_event_detection_nor_hold_times():
    points = _translation(_pose(), 60)
    a, b = _detect(points, count=1), _detect(points, count=6)
    assert a["fixation_candidates"] == b["fixation_candidates"]
    assert b["segments"] == []
    assert b["selection"]["status"] == "ambiguous_episode_count"


def _limb_dip_sequence(head_moves):
    """Two movements whose limbs pause mid-movement; the head keeps moving through the pause.

    Frames: rest, [limb burst 25, limb pause 20, limb burst 25] x2 separated by 90 rest frames.
    """
    fps = 60
    points = _pose(fps, 6)
    limb_step = np.zeros(len(points))
    head_step = np.zeros(len(points))
    for start in (60, 220):
        limb_step[start:start + 25] = .6 / fps
        limb_step[start + 45:start + 70] = .6 / fps
        head_step[start:start + 70] = .9 / fps
    points[:, 5:17, 0] += np.cumsum(limb_step)[:, None]
    points[:, 0:5, 0] += np.cumsum(head_step if head_moves else limb_step)[:, None]
    return points


def test_head_motion_through_limb_pause_keeps_one_movement_episode():
    # Regression: limb-only (5:17) energy split each movement at the limb pause and
    # made the episode count ambiguous; BODY-17 energy keeps the head evidence.
    points = _limb_dip_sequence(head_moves=True)
    limb_only = np.linalg.norm(np.diff(points[:, 5:17], axis=0), axis=-1).mean(axis=1) * 60
    assert np.all(limb_only[60 + 26:60 + 44] == 0)  # the fixture really has a limb-only valley
    result = _detect(points, count=2)
    assert len(result["candidate_episodes"]) == 2
    first = result["candidate_episodes"][0]
    assert first["start_frame"] <= 86 and first["end_frame"] >= 104  # spans the limb pause
    assert result["selection"]["status"] == "exact_episode_count"
    assert [segment["movement_id"] for segment in result["segments"]] == ["M01", "M02"]
    assert 58 <= result["segments"][0]["start_frame"] <= 64
    assert 218 <= result["segments"][1]["start_frame"] <= 224
    assert result["parameters"]["signal_joint_indices"] == list(range(17))
    assert np.all(np.isfinite(result["motion_energy"][60 + 26:60 + 44]))
    assert np.all(result["motion_energy"][60 + 26:60 + 44] > result["thresholds"]["movement_onset_mps"])


def test_ambiguous_episode_clusters_still_select_nothing():
    # Without head motion every limb pause is a real valley: 4 episodes in two equal
    # clusters for 2 expected movements must fail closed, not pick one by fallback.
    result = _detect(_limb_dip_sequence(head_moves=False), count=2)
    assert len(result["candidate_episodes"]) == 4
    assert result["selection"]["status"] == "ambiguous_episode_count"
    assert result["selection"]["cluster_sizes"] == [2, 2]
    assert result["selection"]["selected_candidate_episode_ids"] == []
    assert result["segments"] == []
    assert len(result["fixation_candidates"]) == 4


def test_manual_anchor_changes_do_not_change_automatic_evidence():
    root = Path(__file__).resolve().parents[1]
    spec = load_poomsae_spec(root / "config/scoring/poomsae/taegeuk_1_jang_v0_draft.yaml")
    timeline = load_movement_timeline(root / "config/scoring/timelines/poomsae1_zed2i_rgbd_rerun_20260802_draft.yaml", spec)
    points = _translation(_pose(seconds=741 / 60), 60)
    pose = {"keypoints_3d_world": points, "sample_fps": 60, **_quality(points)}
    changed = deepcopy(timeline)
    for segment in changed["segments"]:
        segment["anchors"]["fixation"] += 1
    a, _ = build_automatic_segmentation_diagnostics(pose, spec, timeline)
    b, _ = build_automatic_segmentation_diagnostics(pose, spec, changed)
    assert a["fixation_candidates"] == b["fixation_candidates"]
    assert a["detector"] == b["detector"]
    assert a["segments"] == b["segments"]
    json.dumps(a, allow_nan=False)


@pytest.mark.parametrize("field,value", [("minimum_hold_sec", 0), ("peak_speed_fraction", 1),
    ("minimum_cameras", True), ("maximum_reprojection_px", float("nan"))])
def test_invalid_parameters_rejected(field, value):
    with pytest.raises(ScoringContractError):
        FixationParameters(**{field: value})


@pytest.mark.parametrize("field,value", [("timestamps_sec", [0, 0]), ("inference_stride", 0),
    ("reliability_valid_mask", [[1]]), ("used_cameras", [[2]])])
def test_invalid_quality_contract_rejected(field, value):
    points = _pose()
    quality = _quality(points)
    quality[field] = value
    with pytest.raises(ScoringContractError):
        _detect(points, quality=quality)


def _review():
    report = {"frame_count": 180, "selected_scope_movement_ids": ["M01"],
              "fixation_candidates": [{"candidate_episode_id": 0, "fixation_frame": 63}]}
    binding = {"review_id": "content-bound-test"}
    payload = {"schema_version": 1, "kind": "fixation_review", "binding": binding,
               "decisions": {"0": {"decision": "corrected", "frame": 65, "movement_id": "M01",
                                   "reviewer": "synthetic", "reviewed_at": "2026-10-17"}}}
    return payload, report, binding


def test_correction_preserves_automatic_proposal():
    payload, report, binding = _review()
    assert validate_fixation_review_export(payload, report, binding) == payload
    assert report["fixation_candidates"][0]["fixation_frame"] == 63


@pytest.mark.parametrize("mutation", ["binding", "unknown", "frame", "bool", "accepted", "reviewer", "no_hold", "movement"])
def test_annotation_cannot_masquerade_as_original_or_foreign_review(mutation):
    payload, report, binding = _review()
    item = payload["decisions"]["0"]
    if mutation == "binding":
        payload["binding"] = {"review_id": "different"}
    elif mutation == "unknown":
        payload["decisions"]["999"] = deepcopy(item)
    elif mutation == "frame":
        item["frame"] = 180
    elif mutation == "bool":
        item["frame"] = True
    elif mutation == "accepted":
        item["decision"] = "accepted"
    elif mutation == "reviewer":
        item["reviewer"] = ""
    elif mutation == "movement":
        item["movement_id"] = "M99"
    else:
        item["decision"] = "no_hold"
    with pytest.raises(ScoringContractError):
        validate_fixation_review_export(payload, report, binding)


def test_unlabelled_recording_produces_the_same_candidates_and_review():
    from src.poomsae_scoring.fixation_review import build_fixation_review_html
    root = Path(__file__).resolve().parents[1]
    spec = load_poomsae_spec(root / "config/scoring/poomsae/taegeuk_1_jang_v0_draft.yaml")
    points = _translation(_pose(), 60)
    pose = {"keypoints_3d_world": points, "sample_fps": 60, **_quality(points)}
    report, rows = build_automatic_segmentation_diagnostics(pose, spec)
    assert report["movement_timeline_id"] is None
    assert report["reference_comparison"]["movements"] == []
    assert report["fixation_candidates"] == _detect(points)["fixation_candidates"]
    assert len(rows) == len(points)
    assert "fixation_left_hand_speed" in rows[0]
    page = build_fixation_review_html(report, {"A < B": "camera%20a.mp4"}, {"review_id": "test"})
    assert '"fixation_review_binding": {"review_id": "test"}' in page
    assert "binding:fixationBinding" in page
    assert "A &lt; B" in page
    assert "Elle işaretler kesin doğru kabul edilmez" in page
    assert "data-fixation-frame" in page
    assert "manual_labels_used_for_fixation_detection_or_thresholds" in page
    json.dumps(report, allow_nan=False)


def test_fixation_review_binding_tracks_source_and_candidates():
    report = {"bindings": {"pose": {"sha256": "a"}}, "fixation_candidates": [{"fixation_frame": 12}]}
    binding = build_fixation_review_binding(report)
    assert len(binding["review_id"]) == 64
    assert binding["proposal_sha256"] == binding["review_id"]
    assert binding == build_fixation_review_binding(deepcopy(report))
    changed = deepcopy(report)
    changed["fixation_candidates"][0]["fixation_frame"] = 13
    assert build_fixation_review_binding(changed) != binding
    changed = deepcopy(report)
    changed["bindings"]["pose"]["sha256"] = "b"
    assert build_fixation_review_binding(changed) != binding
    report["review_binding"] = binding
    with pytest.raises(ScoringContractError):
        build_fixation_review_binding(report)
