from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from src.poomsae_scoring.application import (
    WorkflowError,
    _load_profile,
    _output_paths,
    _portable_pose_path,
    _run_stage,
    _transfer_timeline_binding,
)
from src.poomsae_scoring import load_poomsae_spec
from src.run_outputs import initialize_run_state, mark_run_running
from src.poomsae_scoring import application


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


def test_failed_subprocess_stage_marks_existing_run_failed(tmp_path: Path, monkeypatch) -> None:
    run_root = tmp_path / "runs" / "failed-stage"
    run_root.mkdir(parents=True)
    initialize_run_state(run_root, "session-test", "failed-stage")
    mark_run_running(run_root, "session-test", "failed-stage")

    def fail_stage(*args, **kwargs):
        raise subprocess.CalledProcessError(7, args[0])

    monkeypatch.setattr(subprocess, "run", fail_stage)
    with pytest.raises(WorkflowError, match="synthetic stage failed with exit code 7"):
        _run_stage(
            "synthetic stage",
            "scripts/does_not_matter.py",
            failure_context=(run_root, "session-test", "failed-stage"),
        )

    state = json.loads((run_root / "run_state.json").read_text(encoding="utf-8"))
    assert state["status"] == "failed"
    assert state["error"] == "synthetic stage failed with exit code 7."


@pytest.mark.parametrize("error", [OSError("disk full"), ValueError("invalid JSON"), KeyboardInterrupt()])
@pytest.mark.parametrize("failure_at", ["snapshot", "video_resolution", "subprocess"])
def test_workflow_marks_all_analysis_failures_and_cancellation_failed(tmp_path, monkeypatch, error, failure_at):
    profile = {key: str(tmp_path / key) for key in application._PATH_KEYS}
    profile["output_root"] = str(tmp_path)
    profile["videos"] = []
    monkeypatch.setattr(application, "_resolve_profile", lambda value: tmp_path / "profile.yaml")
    monkeypatch.setattr(application, "_load_profile", lambda path: profile)
    monkeypatch.setattr(application, "_verify_profile_bindings", lambda *args: None)
    monkeypatch.setattr(application, "load_session", lambda path: SimpleNamespace(session_id="session-test"))
    monkeypatch.setattr(application, "_read_json", lambda path: {"session_id": "session-test"})
    marker = tmp_path / "session-test" / "latest_run.json"
    marker.parent.mkdir()
    marker.write_text('{"run_id":"previous-success"}', encoding="utf-8")
    original_marker = marker.read_bytes()

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(application, "_snapshot_configuration", fail if failure_at == "snapshot" else lambda **kwargs: {})
    monkeypatch.setattr(application, "_resolve_videos", fail if failure_at == "video_resolution" else lambda videos: [])
    monkeypatch.setattr(application.subprocess, "run", fail)
    with pytest.raises(type(error)) as caught:
        application.run_workflow(profile_value="test", process_video=False, requested_run_id="failure-test")
    assert caught.value is error
    state = json.loads((marker.parent / "runs/failure-test/run_state.json").read_text(encoding="utf-8"))
    assert state["status"] == "failed"
    assert type(error).__name__ in state["error"]
    assert marker.read_bytes() == original_marker


def test_lifecycle_write_failure_preserves_original_exception_and_existing_run(tmp_path, monkeypatch):
    profile = {key: str(tmp_path / key) for key in application._PATH_KEYS}
    profile["output_root"] = str(tmp_path)
    monkeypatch.setattr(application, "_resolve_profile", lambda value: tmp_path / "profile.yaml")
    monkeypatch.setattr(application, "_load_profile", lambda path: profile)
    monkeypatch.setattr(application, "_verify_profile_bindings", lambda *args: None)
    monkeypatch.setattr(application, "load_session", lambda path: SimpleNamespace(session_id="session-test"))
    monkeypatch.setattr(application, "_read_json", lambda path: {"session_id": "session-test"})
    original = ValueError("original analysis failure")

    def fail_snapshot(**kwargs):
        raise original

    def fail_state(*args):
        raise OSError("state storage unavailable")

    monkeypatch.setattr(application, "_snapshot_configuration", fail_snapshot)
    monkeypatch.setattr(application, "mark_run_failed", fail_state)
    with pytest.raises(ValueError) as caught:
        application.run_workflow(profile_value="test", process_video=False, requested_run_id="failure-test")
    assert caught.value is original
    assert "Could not persist" in original.__notes__[0]
    state_path = tmp_path / "session-test/runs/failure-test/run_state.json"
    previous_state = state_path.read_bytes()
    with pytest.raises(WorkflowError, match="already exists"):
        application.run_workflow(profile_value="test", process_video=False, requested_run_id="failure-test")
    assert state_path.read_bytes() == previous_state


def test_video_subprocess_failure_is_recorded_after_child_creates_run(tmp_path, monkeypatch):
    profile = {key: str(tmp_path / key) for key in application._PATH_KEYS}
    profile.update(output_root=str(tmp_path), processing={"stride": 1, "smoothing_window": 3, "progress_every": 10})
    monkeypatch.setattr(application, "_resolve_profile", lambda value: tmp_path / "profile.yaml")
    monkeypatch.setattr(application, "_load_profile", lambda path: profile)
    monkeypatch.setattr(application, "_verify_profile_bindings", lambda *args: None)
    monkeypatch.setattr(application, "_verify_process_inputs", lambda *args: None)
    monkeypatch.setattr(application, "load_session", lambda path: SimpleNamespace(session_id="session-test"))
    monkeypatch.setattr(application, "_read_json", lambda path: {"session_id": "session-test"})
    run_root = tmp_path / "session-test/runs/video-failure"

    def child(command, **kwargs):
        if "run_vitpose_multiview_3d.py" in command[1]:
            assert not run_root.exists()
            run_root.mkdir(parents=True)
            raise subprocess.CalledProcessError(9, command)

    monkeypatch.setattr(application.subprocess, "run", child)
    with pytest.raises(WorkflowError, match="exit code 9"):
        application.run_workflow(profile_value="test", process_video=True, requested_run_id="video-failure")
    state = json.loads((run_root / "run_state.json").read_text(encoding="utf-8"))
    assert state["status"] == "failed"
    assert not (run_root.parent.parent / "latest_run.json").exists()


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


def test_run_outputs_include_the_session_bound_direction_reference(tmp_path: Path) -> None:
    outputs = _output_paths(tmp_path)

    assert outputs["athlete_direction_reference"].name == "athlete_local_direction_reference.json"
    assert outputs["athlete_direction_reference_status"].name == "athlete_local_direction_reference_status.json"
    assert outputs["athlete_direction_reference"].parent == tmp_path / "json"
    assert outputs["athlete_direction_reference_status"].parent == tmp_path / "json"


def test_runner_derives_the_direction_reference_before_the_accuracy_diagnostics_stage() -> None:
    """The 17 direction-bound rules stay closed unless the reference is built first and passed on."""
    source = (ROOT / "src" / "poomsae_scoring" / "application.py").read_text(encoding="utf-8")
    derive_at = source.index("build_athlete_local_direction_reference.py")
    diagnostics_at = source.index("run_technical_accuracy_diagnostics.py")

    assert derive_at < diagnostics_at, "the reference must exist before diagnostics consume it"
    diagnostics_stage = source[diagnostics_at:source.index("analyze_poomsae_movement_evidence.py")]
    assert "*direction_args" in diagnostics_stage
    assert '"--direction-reference"' in source[derive_at:diagnostics_at]
    assert 'outputs["athlete_direction_reference"].is_file()' in source[derive_at:diagnostics_at]


def test_runner_feeds_derived_observations_into_the_accuracy_stage() -> None:
    """The observation stage is worthless if its output never reaches the accuracy stage."""
    source = (ROOT / "src" / "poomsae_scoring" / "application.py").read_text(encoding="utf-8")
    derive_at = source.index("run_categorical_poomsae_diagnostics.py")
    accuracy_at = source.index("build_source_bound_accuracy_decisions.py")

    assert derive_at < accuracy_at, "observations must be derived before accuracy consumes them"
    accuracy_stage = source[accuracy_at:source.index("build_poomsae_evidence_events.py")]
    assert '"--observations"' in accuracy_stage
    assert 'outputs["categorical_diagnostics"]' in accuracy_stage


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


def test_evidence_events_script_can_receive_alignment_anomalies() -> None:
    """Alignment doubts are worthless if they never reach the report."""
    source = (ROOT / "scripts" / "build_poomsae_evidence_events.py").read_text(encoding="utf-8")

    assert '"--alignment-anomalies"' in source
    assert "alignment_anomalies=alignment_anomalies" in source
    # A report built for another timeline must not be attached to this one.
    assert 'produced for a different timeline' in source


def test_automatic_timeline_draft_is_not_part_of_the_scoring_run() -> None:
    """The draft is a proposal for a person; scoring must never consume it unreviewed."""
    application = (ROOT / "src" / "poomsae_scoring" / "application.py").read_text(encoding="utf-8")
    draft = ROOT / "scripts" / "build_poomsae_automatic_timeline_draft.py"

    assert draft.is_file()
    assert "build_poomsae_automatic_timeline_draft" not in application
    body = draft.read_text(encoding="utf-8")
    assert "hand-labelled" in body  # templates must not come from an automatic timeline
    assert "Review and correct it before scoring uses it." in body


def test_anchor_review_sheet_defaults_cover_the_measured_error() -> None:
    """A window narrower than the measured error would hide the frames worth checking."""
    source = (ROOT / "scripts" / "build_poomsae_anchor_review_sheet.py").read_text(encoding="utf-8")
    application = (ROOT / "src" / "poomsae_scoring" / "application.py").read_text(encoding="utf-8")

    assert '"--radius-frames"' in source and "default=15" in source
    assert '"--step-frames"' in source and "default=5" in source
    # Reviewing a draft is a human step, never part of the scoring run.
    assert "build_poomsae_anchor_review_sheet" not in application


def test_anchor_review_sheet_renders_one_strip_per_movement(tmp_path: Path) -> None:
    import numpy as np

    cv2 = pytest.importorskip("cv2")
    spec = load_poomsae_spec(SPEC_PATH)
    span, movements = 60, 2
    frame_count = span * movements
    video = tmp_path / "camera.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 60.0, (64, 48))
    if not writer.isOpened():
        pytest.skip("no MJPG writer available in this OpenCV build")
    for index in range(frame_count):
        writer.write(np.full((48, 64, 3), index % 255, dtype=np.uint8))
    writer.release()
    if int(cv2.VideoCapture(str(video)).get(cv2.CAP_PROP_FRAME_COUNT)) != frame_count:
        pytest.skip("this OpenCV build does not report an exact frame count")

    timeline = tmp_path / "timeline.yaml"
    _prefix_timeline_yaml(timeline, segment_lengths=[span, span], gap_frames=[])
    payload = yaml.safe_load(timeline.read_text(encoding="utf-8"))
    assert payload["frame_count"] == frame_count
    # Put every fixation anchor mid-span so the full window fits inside the recording.
    for segment in payload["segments"]:
        middle = (segment["start_frame"] + segment["end_frame"]) // 2
        segment["anchors"][sorted(segment["anchors"], key=segment["anchors"].get)[-1]] = middle
    timeline.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    output = tmp_path / "sheet.html"

    result = _run_script(
        "build_poomsae_anchor_review_sheet.py",
        "--timeline", timeline,
        "--poomsae-spec", SPEC_PATH,
        "--camera", f"cam_a={video}",
        "--output-html", output,
    )

    assert result.returncode == 0, result.stderr
    page = output.read_text(encoding="utf-8")
    for movement in spec["movements"][:movements]:
        assert movement["movement_id"] in page
    # Seven thumbnails per movement: -15 to +15 in steps of five.
    assert page.count("data:image/jpeg;base64,") == 7 * movements
    assert page.count("class='proposed'") == movements
    assert "hiçbir kesinti veya puan iddiası taşımaz" in page


def test_anchor_review_sheet_can_draw_the_skeleton_without_a_recording(tmp_path: Path) -> None:
    """The camera files live on the laboratory machine; review must not require them."""
    import numpy as np

    pytest.importorskip("cv2")
    span, movements = 60, 2
    frame_count = span * movements
    timeline = tmp_path / "timeline.yaml"
    _prefix_timeline_yaml(timeline, segment_lengths=[span, span], gap_frames=[])
    payload = yaml.safe_load(timeline.read_text(encoding="utf-8"))
    for segment in payload["segments"]:
        middle = (segment["start_frame"] + segment["end_frame"]) // 2
        segment["anchors"][sorted(segment["anchors"], key=segment["anchors"].get)[-1]] = middle
    timeline.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    body = {
        0: [0.0, 0.05, 1.72], 1: [-0.03, 0.04, 1.75], 2: [0.03, 0.04, 1.75],
        3: [-0.07, 0.0, 1.74], 4: [0.07, 0.0, 1.74], 5: [-0.20, 0.0, 1.45],
        6: [0.20, 0.0, 1.45], 7: [-0.35, 0.05, 1.20], 8: [0.35, 0.05, 1.20],
        9: [-0.42, 0.15, 1.00], 10: [0.42, 0.15, 1.00], 11: [-0.14, 0.0, 0.95],
        12: [0.14, 0.0, 0.95], 13: [-0.16, 0.03, 0.52], 14: [0.16, -0.03, 0.52],
        15: [-0.16, 0.12, 0.06], 16: [0.16, -0.12, 0.06],
    }
    points = np.full((frame_count, 133, 3), np.nan)
    for index in range(frame_count):
        for joint, value in body.items():
            points[index, joint] = value
    pose = tmp_path / "pose.json"
    pose.write_text(
        json.dumps({"keypoints_3d_world": np.where(np.isnan(points), None, points).tolist()}),
        encoding="utf-8",
    )
    output = tmp_path / "sheet.html"

    result = _run_script(
        "build_poomsae_anchor_review_sheet.py",
        "--timeline", timeline,
        "--poomsae-spec", SPEC_PATH,
        "--pose", pose,
        "--output-html", output,
    )

    assert result.returncode == 0, result.stderr
    page = output.read_text(encoding="utf-8")
    assert page.count("data:image/jpeg;base64,") == 7 * movements
    # The page must say plainly that a drawn skeleton is not camera evidence.
    assert "kamera görüntüsü değil" in page


def test_anchor_review_sheet_needs_a_camera_or_a_pose_file(tmp_path: Path) -> None:
    timeline = tmp_path / "timeline.yaml"
    _prefix_timeline_yaml(timeline, segment_lengths=[30], gap_frames=[])

    result = _run_script(
        "build_poomsae_anchor_review_sheet.py",
        "--timeline", timeline,
        "--poomsae-spec", SPEC_PATH,
        "--output-html", tmp_path / "sheet.html",
    )

    assert result.returncode != 0
    assert "at least one --camera or a --pose" in (result.stdout + result.stderr)


def test_diagnostic_scripts_can_run_without_the_package_installed() -> None:
    """Every script that imports src must put the repository root on the search path."""
    for name in (
        "run_technical_accuracy_diagnostics.py",
        "run_wholebody_poomsae_diagnostics.py",
        "build_poomsae_evidence_events.py",
        "build_poomsae_anchor_review_sheet.py",
        "build_poomsae_automatic_timeline_draft.py",
        "build_athlete_local_direction_reference.py",
    ):
        source = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert "sys.path.insert(0, str(ROOT))" in source, name


def test_judge_questionnaire_asks_only_about_values_no_referee_has_signed() -> None:
    """The questionnaire is the profile's own list of what it still does not know."""
    from src.poomsae_scoring import load_technical_accuracy_profile
    from scripts.build_judge_threshold_questionnaire import collect_open_questions

    profile_path = ROOT / "config" / "scoring" / "engineering" / "taegeuk_1_wholebody_diagnostics_v3.yaml"
    profile = load_technical_accuracy_profile(profile_path)
    questions = collect_open_questions(profile)

    asked = {row["metric_id"] for row in questions["priority"] + questions["deferred"]}
    assert asked == set(profile["thresholds"])  # nothing is signed yet, so everything is open

    # A referee's answer removes the question, and only that question.
    raw = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    raw["thresholds"]["torso_lean_p95_deg"]["judge_source"] = {
        "origin": "judge_supplied_validated_threshold",
        "judge_name": "Test Referee",
        "judge_credential": "WT international poomsae referee, 1st class",
        "decision_date": "2026-09-10",
        "approval_reference": "TK3D-JUDGE-REVIEW-2026-09-10",
        "score_effect": "deduction_candidate",
        "deduction_points": 0.1,
    }
    raw["judge_validated_rules"] = ["torso_lean_p95_deg"]
    from src.poomsae_scoring import validate_technical_accuracy_profile

    signed = collect_open_questions(validate_technical_accuracy_profile(raw))
    still_asked = {row["metric_id"] for row in signed["priority"] + signed["deferred"]}
    assert still_asked == asked - {"torso_lean_p95_deg"}


def test_judge_questionnaire_puts_answerable_questions_first() -> None:
    """A question whose answer works immediately is worth more meeting time than one that waits."""
    from src.poomsae_scoring import load_technical_accuracy_profile
    from scripts.build_judge_threshold_questionnaire import collect_open_questions

    profile = load_technical_accuracy_profile(
        ROOT / "config" / "scoring" / "engineering" / "taegeuk_1_wholebody_diagnostics_v3.yaml"
    )
    questions = collect_open_questions(profile)

    assert questions["priority"], "an active rule with an unsigned threshold must be asked first"
    assert all(row["status"] == "active_diagnostic" for row in questions["priority"])
    assert all(row["status"] != "active_diagnostic" for row in questions["deferred"])
    # A deferred question must say what else is missing, or the reader cannot judge its priority.
    assert all(row["blocking_reason"] for row in questions["deferred"])
    # The stance ranges are asked because a landing tolerance sits on top of them.
    assert {row["stance"] for row in questions["stance_ranges"]} == {"ap_seogi", "ap_gubi"}


def test_judge_questionnaire_claims_nothing_and_stays_out_of_the_scoring_run(tmp_path: Path) -> None:
    application = (ROOT / "src" / "poomsae_scoring" / "application.py").read_text(encoding="utf-8")
    assert "build_judge_threshold_questionnaire" not in application

    output = tmp_path / "hakem_sorulari.html"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_judge_threshold_questionnaire.py"),
            "--profile",
            str(ROOT / "config" / "scoring" / "engineering" / "taegeuk_1_wholebody_diagnostics_v3.yaml"),
            "--output-html",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    page = output.read_text(encoding="utf-8")
    assert "kesinti ve puan içermez" in page  # the page states its own limits
    assert "Bölüm 1 &mdash; öncelikli" in page
    assert "Bölüm 2 &mdash; vakit kalırsa" in page
    assert "judge_supplied_validated_threshold" in page  # how an answer is recorded

    # An existing page is never overwritten; a stale answer sheet must not disappear silently.
    again = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_judge_threshold_questionnaire.py"),
            "--profile",
            str(ROOT / "config" / "scoring" / "engineering" / "taegeuk_1_wholebody_diagnostics_v3.yaml"),
            "--output-html",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert again.returncode != 0
    assert "üzerine yazılmayacak" in again.stderr
