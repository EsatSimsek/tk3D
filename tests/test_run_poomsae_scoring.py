from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

from src.poomsae_scoring.application import (
    WorkflowError,
    _load_profile,
    _output_paths,
    _portable_pose_path,
    _transfer_timeline_binding,
)
from src.poomsae_scoring import load_poomsae_spec


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "config" / "scoring" / "poomsae" / "taegeuk_1_jang_v0_draft.yaml"


def _write_pose(path: Path, *, timestamps: list[float], run_id: str) -> None:
    path.write_text(
        json.dumps(
            {
                "session_id": "poomsae_test",
                "run_id": run_id,
                "inference_stride": 1,
                "sample_fps": 2.0,
                "frame_indices": [0, 1, 2],
                "timestamps_sec": timestamps,
            }
        ),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_timeline(path: Path, reference_pose: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 2,
                "timeline_id": "timeline-test",
                "frame_count": 3,
                "fps": 2.0,
                "source_binding": {
                    "session_id": "poomsae_test",
                    "run_id": "reference",
                    "pose_file": str(reference_pose),
                    "pose_file_sha256": _sha256(reference_pose),
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_project_one_command_profile_is_valid() -> None:
    profile = _load_profile(ROOT / "config" / "scoring" / "profiles" / "poomsae1_trimmed.yaml")

    assert profile["profile_id"] == "poomsae1_trimmed"
    assert profile["processing"]["stride"] == 1
    assert len(profile["videos"]) == 2
    assert {video["camera_id"] for video in profile["videos"]} == {"zed_35151067", "zed_37137479"}


def test_one_command_output_contract_includes_integrated_diagnostics(tmp_path: Path) -> None:
    outputs = _output_paths(tmp_path / "run")

    assert outputs["categorical_diagnostics"].name == "categorical_diagnostics_report.json"
    assert outputs["technical_conformance"].name == "technical_conformance_report.json"
    assert outputs["presentation_diagnostics"].name == "presentation_diagnostics_report.json"
    assert outputs["automatic_segmentation"].name == "automatic_segmentation_report.json"
    assert outputs["automatic_segmentation_signal"].name == "automatic_segmentation_signal.csv"
    assert outputs["browser_review_video_manifest"].name == "browser_review_video_manifest.json"
    assert outputs["run_history_json"].name == "run_history_report.json"
    assert outputs["run_history_html"].name == "run_history.html"


def test_timeline_transfer_requires_identical_video_time_axis(tmp_path: Path) -> None:
    reference = tmp_path / "reference.json"
    new = tmp_path / "new.json"
    timeline = tmp_path / "timeline.yaml"
    _write_pose(reference, timestamps=[0.0, 0.5, 1.0], run_id="reference")
    _write_pose(new, timestamps=[0.0, 0.5, 1.0], run_id="new-run")
    _write_timeline(timeline, reference)

    transferred = _transfer_timeline_binding(
        timeline_path=timeline,
        reference_pose_path=reference,
        new_pose_path=new,
        session_id="poomsae_test",
        run_id="new-run",
    )

    assert transferred["source_binding"]["run_id"] == "new-run"
    assert transferred["source_binding"]["pose_file_sha256"] == _sha256(new)
    assert transferred["timeline_id"].endswith("new-run")
    assert transferred["source_binding"]["pose_file"] == _portable_pose_path(new)


def test_portable_pose_path_preserves_external_absolute_path() -> None:
    external = Path("C:/Windows/Temp/tk3d-external-pose.json")

    assert _portable_pose_path(external) == external.resolve().as_posix()


def test_timeline_transfer_fails_closed_when_timestamps_change(tmp_path: Path) -> None:
    reference = tmp_path / "reference.json"
    new = tmp_path / "new.json"
    timeline = tmp_path / "timeline.yaml"
    _write_pose(reference, timestamps=[0.0, 0.5, 1.0], run_id="reference")
    _write_pose(new, timestamps=[0.0, 0.6, 1.0], run_id="new-run")
    _write_timeline(timeline, reference)

    with pytest.raises(WorkflowError, match="timestamps differ"):
        _transfer_timeline_binding(
            timeline_path=timeline,
            reference_pose_path=reference,
            new_pose_path=new,
            session_id="poomsae_test",
            run_id="new-run",
        )


def _prefix_timeline_yaml(
    path: Path,
    *,
    segment_lengths: list[int],
    gap_frames: list[int],
    fps: float = 60.0,
) -> None:
    """Write a valid partial MovementTimeline whose inter-segment gaps are exact."""
    spec = load_poomsae_spec(SPEC_PATH)
    segments = []
    frame = 0
    for index, length in enumerate(segment_lengths):
        movement = spec["movements"][index]
        start = frame
        end = start + length - 1
        segments.append(
            {
                "sequence_index": index + 1,
                "movement_id": movement["movement_id"],
                "start_frame": start,
                "end_frame": end,
                "anchors": {
                    phase: start + offset + 1
                    for offset, phase in enumerate(movement["phases"])
                },
                "confidence": 1.0,
                "label_status": "confirmed",
            }
        )
        frame = end + 1
        if index < len(gap_frames):
            frame += gap_frames[index]
    payload = {
        "schema_version": 2,
        "timeline_id": "runner-stage-test",
        "poomsae_id": spec["poomsae_id"],
        "poomsae_version": spec["version"],
        "status": "draft",
        "label_source": "manual",
        "frame_index_space": "sample_index",
        "frame_count": frame,
        "fps": fps,
        "source_binding": {
            "session_id": "runner-stage-session",
            "run_id": "runner-stage-run",
            "pose_file": "outputs/runner-stage/pose.json",
            "pose_file_sha256": None,
        },
        "coverage": {
            "recording_scope": "partial_sequence",
            "observed_movement_ids": [item["movement_id"] for item in segments],
            "missing_movement_ids": [
                movement["movement_id"] for movement in spec["movements"][len(segments):]
            ],
            "source_end_reason": "synthetic runner stage fixture",
        },
        "segments": segments,
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _run_script(script: str, *args: str | Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(ROOT)
    if existing_pythonpath:
        env["PYTHONPATH"] += os.pathsep + existing_pythonpath
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), *[str(item) for item in args]],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=env,
        check=False,  # the failure cases below assert on returncode themselves
    )


def test_run_outputs_include_integrated_categorical_and_presentation_stages(tmp_path: Path) -> None:
    outputs = _output_paths(tmp_path)

    assert outputs["categorical_diagnostics"].name == "categorical_diagnostics_report.json"
    assert outputs["presentation_diagnostics"].name == "presentation_diagnostics_report.json"
    assert outputs["categorical_diagnostics"].parent == tmp_path / "json"
    assert outputs["presentation_diagnostics"].parent == tmp_path / "json"


def test_runner_feeds_derived_observations_into_the_accuracy_stage() -> None:
    """The observation stage is worthless if its output never reaches the accuracy stage."""
    source = (ROOT / "src" / "poomsae_scoring" / "application.py").read_text(encoding="utf-8")
    derive_at = source.index("run_categorical_poomsae_diagnostics.py")
    accuracy_at = source.index("build_source_bound_accuracy_decisions.py")

    assert derive_at < accuracy_at, "observations must be derived before accuracy consumes them"
    accuracy_stage = source[accuracy_at:source.index("build_poomsae_evidence_events.py")]
    assert '"--observations"' in accuracy_stage
    assert 'outputs["categorical_diagnostics"]' in accuracy_stage


def test_derive_observations_script_emits_pause_and_manifest(tmp_path: Path) -> None:
    timeline = tmp_path / "timeline.yaml"
    output = tmp_path / "observations.json"
    # 240 empty frames at 60 fps is a 4.0 s pause after M01; the second gap is 1.0 s.
    _prefix_timeline_yaml(timeline, segment_lengths=[30, 30, 30], gap_frames=[240, 60])

    result = _run_script(
        "derive_poomsae_categorical_observations.py",
        "--poomsae-spec",
        SPEC_PATH,
        "--timeline",
        timeline,
        "--output-json",
        output,
    )

    assert result.returncode == 0, result.stderr
    observations = json.loads(output.read_text(encoding="utf-8"))
    assert len(observations) == 1
    assert observations[0]["movement_id"] == "M01"
    assert observations[0]["event_kind"] == "pause_at_least_3_sec"
    assert observations[0]["confirmation_method"] == "duration_measurement"
    assert observations[0]["measurement"]["duration_sec"] == pytest.approx(4.0)

    manifest = json.loads((tmp_path / "observations_manifest.json").read_text(encoding="utf-8"))
    assert manifest["pause_threshold_sec"] == 3.0
    assert manifest["observation_count"] == 1
    assert "wrong_action" in manifest["not_derived_event_kinds"]
    assert set(manifest["bindings"]) == {"poomsae_spec", "movement_timeline"}


def test_derive_observations_script_refuses_to_overwrite(tmp_path: Path) -> None:
    timeline = tmp_path / "timeline.yaml"
    output = tmp_path / "observations.json"
    _prefix_timeline_yaml(timeline, segment_lengths=[30, 30], gap_frames=[240])
    output.write_text("[]", encoding="utf-8")

    result = _run_script(
        "derive_poomsae_categorical_observations.py",
        "--poomsae-spec",
        SPEC_PATH,
        "--timeline",
        timeline,
        "--output-json",
        output,
    )

    assert result.returncode != 0
    assert "refusing to overwrite" in result.stderr


def test_presentation_script_writes_a_report_that_claims_no_score(tmp_path: Path) -> None:
    timeline = tmp_path / "timeline.yaml"
    diagnostics = tmp_path / "wholebody.json"
    output = tmp_path / "presentation.json"
    _prefix_timeline_yaml(timeline, segment_lengths=[60, 60], gap_frames=[60])
    spec = load_poomsae_spec(SPEC_PATH)
    timeline_payload = yaml.safe_load(timeline.read_text(encoding="utf-8"))
    diagnostics.write_text(
        json.dumps(
            {
                "status": "wholebody_diagnostics_only",
                "poomsae": {"poomsae_id": spec["poomsae_id"], "version": spec["version"]},
                "movement_timeline_id": timeline_payload["timeline_id"],
                "movements": [
                    {
                        "movement_id": movement_id,
                        "metrics": [
                            {
                                "metric_id": "executing_wrist_peak_speed_body_scale_per_sec",
                                "value": 4.5,
                                "unit": "body_scale/sec",
                            }
                        ],
                    }
                    for movement_id in ("M01", "M02")
                ],
            }
        ),
        encoding="utf-8",
    )

    result = _run_script(
        "build_poomsae_presentation_diagnostics.py",
        "--wholebody-diagnostics",
        diagnostics,
        "--poomsae-spec",
        SPEC_PATH,
        "--timeline",
        timeline,
        "--output-json",
        output,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "presentation_diagnostic_only"
    assert report["total_score"] is None
    assert report["safety_contract"]["score_claim_allowed"] is False
    rhythm = report["components"]["rhythm_and_tempo"]["metrics"]
    assert rhythm["movement_duration_sec"]["sample_count"] == 2
    assert rhythm["transition_gap_sec"]["median"] == pytest.approx(1.0)
    assert set(report["bindings"]) == {
        "wholebody_diagnostics",
        "poomsae_spec",
        "movement_timeline",
    }


def test_presentation_script_rejects_a_non_wholebody_report(tmp_path: Path) -> None:
    timeline = tmp_path / "timeline.yaml"
    diagnostics = tmp_path / "not_wholebody.json"
    output = tmp_path / "presentation.json"
    _prefix_timeline_yaml(timeline, segment_lengths=[60, 60], gap_frames=[60])
    diagnostics.write_text(json.dumps({"status": "something_else", "movements": []}), encoding="utf-8")

    result = _run_script(
        "build_poomsae_presentation_diagnostics.py",
        "--wholebody-diagnostics",
        diagnostics,
        "--poomsae-spec",
        SPEC_PATH,
        "--timeline",
        timeline,
        "--output-json",
        output,
    )

    assert result.returncode != 0
    assert not output.exists()


def test_repo_relative_posix_keeps_outside_paths_absolute() -> None:
    inside = ROOT / "config" / "model_config.yaml"
    with tempfile.NamedTemporaryFile(suffix=".json") as stream:
        outside = Path(stream.name)
        assert _portable_pose_path(inside) == "config/model_config.yaml"
        assert _portable_pose_path(outside) == outside.resolve().as_posix()


def _write_synthetic_pose(path: Path, frame_count: int) -> None:
    """A pose file shaped like the real one, with a couple of joints deliberately unseen."""
    import numpy as np

    rng = np.random.default_rng(20260824)
    keypoints = rng.normal(size=(frame_count, 133, 3)).tolist()
    valid = np.ones((frame_count, 133), dtype=bool)
    valid[:, 7] = False          # never observed -> must stay null in the template
    valid[:, 8] = False
    valid[:120, 9] = False       # seen too rarely near the first anchor
    path.write_text(
        json.dumps({"keypoints_3d_world": keypoints, "reliability_valid_mask": valid.tolist()}),
        encoding="utf-8",
    )


def test_reference_template_script_covers_only_the_labelled_movements(tmp_path: Path) -> None:
    timeline = tmp_path / "timeline.yaml"
    pose = tmp_path / "pose.json"
    output = tmp_path / "templates.json"
    _prefix_timeline_yaml(timeline, segment_lengths=[60, 60], gap_frames=[30])
    frame_count = yaml.safe_load(timeline.read_text(encoding="utf-8"))["frame_count"]
    _write_synthetic_pose(pose, frame_count)

    result = _run_script(
        "build_poomsae_reference_templates.py",
        "--pose", pose,
        "--poomsae-spec", SPEC_PATH,
        "--timeline", timeline,
        "--output-json", output,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "reference_pose_templates"
    assert payload["coverage"]["covered_movement_ids"] == ["M01", "M02"]
    assert payload["coverage"]["template_count"] == 2
    assert payload["coverage"]["expected_movement_count"] == 18
    assert "M18" in payload["coverage"]["missing_movement_ids"]
    # The limits must travel with the file, not live only in someone's memory.
    assert any("Single athlete" in line for line in payload["limitations"])

    first = payload["templates"][0]
    assert len(first["mean_pose"]) == 133
    assert first["mean_pose"][7] == [None, None, None]   # never observed
    assert first["mean_pose"][8] == [None, None, None]
    assert all(value is not None for value in first["mean_pose"][0])
    assert first["valid_joint_count"] < first["total_joint_count"]


def test_reference_template_script_refuses_an_automatic_timeline(tmp_path: Path) -> None:
    """Templates built from automatic labels would let the alignment grade its own work."""
    timeline = tmp_path / "timeline.yaml"
    pose = tmp_path / "pose.json"
    output = tmp_path / "templates.json"
    _prefix_timeline_yaml(timeline, segment_lengths=[60, 60], gap_frames=[30])
    payload = yaml.safe_load(timeline.read_text(encoding="utf-8"))
    payload["label_source"] = "automatic"
    timeline.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    _write_synthetic_pose(pose, payload["frame_count"])

    result = _run_script(
        "build_poomsae_reference_templates.py",
        "--pose", pose,
        "--poomsae-spec", SPEC_PATH,
        "--timeline", timeline,
        "--output-json", output,
    )

    assert result.returncode != 0
    assert "hand-labelled" in result.stderr
    assert not output.exists()


def test_reference_template_script_rejects_a_pose_of_the_wrong_length(tmp_path: Path) -> None:
    timeline = tmp_path / "timeline.yaml"
    pose = tmp_path / "pose.json"
    output = tmp_path / "templates.json"
    _prefix_timeline_yaml(timeline, segment_lengths=[60, 60], gap_frames=[30])
    frame_count = yaml.safe_load(timeline.read_text(encoding="utf-8"))["frame_count"]
    _write_synthetic_pose(pose, frame_count - 10)

    result = _run_script(
        "build_poomsae_reference_templates.py",
        "--pose", pose,
        "--poomsae-spec", SPEC_PATH,
        "--timeline", timeline,
        "--output-json", output,
    )

    assert result.returncode != 0
    assert "do not describe the same recording" in result.stderr
