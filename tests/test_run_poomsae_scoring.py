from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from scripts.run_poomsae_scoring import (
    WorkflowError,
    _load_profile,
    _output_paths,
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
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), *[str(item) for item in args]],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,  # the failure cases below assert on returncode themselves
    )


def test_run_outputs_include_the_new_observation_and_presentation_stages(tmp_path: Path) -> None:
    outputs = _output_paths(tmp_path)

    assert outputs["categorical_observations"].name == "automatic_categorical_observations.json"
    assert outputs["presentation_diagnostics"].name == "presentation_diagnostics_report.json"
    assert outputs["categorical_observations"].parent == tmp_path / "json"
    assert outputs["presentation_diagnostics"].parent == tmp_path / "json"


def test_runner_feeds_derived_observations_into_the_accuracy_stage() -> None:
    """The observation stage is worthless if its output never reaches the accuracy stage."""
    source = (ROOT / "scripts" / "run_poomsae_scoring.py").read_text(encoding="utf-8")
    derive_at = source.index("derive_poomsae_categorical_observations.py")
    accuracy_at = source.index("build_source_bound_accuracy_decisions.py")
    presentation_at = source.index("build_poomsae_presentation_diagnostics.py")

    assert derive_at < accuracy_at, "observations must be derived before accuracy consumes them"
    accuracy_stage = source[accuracy_at:presentation_at]
    assert '"--observations"' in accuracy_stage
    assert 'outputs["categorical_observations"]' in accuracy_stage


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
    diagnostics.write_text(
        json.dumps(
            {
                "status": "wholebody_diagnostics_only",
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


def test_repo_relative_posix_keeps_outside_paths_absolute(tmp_path: Path) -> None:
    from scripts.run_poomsae_scoring import _repo_relative_posix

    inside = ROOT / "config" / "model_config.yaml"
    outside = tmp_path / "pose.json"
    outside.write_text("{}", encoding="utf-8")

    assert _repo_relative_posix(inside) == "config/model_config.yaml"
    assert _repo_relative_posix(outside) == outside.resolve().as_posix()
