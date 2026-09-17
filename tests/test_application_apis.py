from __future__ import annotations

import json
from pathlib import Path

import pytest
import cv2

from src import multiview_application
from src.poomsae_scoring import application as poomsae_application
from src.artifact_io import load_json_object
from src.run_outputs import create_run_output_tree, mark_run_complete


def test_multiview_application_rejects_invalid_sampling_before_io() -> None:
    options = multiview_application.MultiviewRunOptions(
        session="missing.yaml",
        stride=0,
    )

    with pytest.raises(ValueError, match="stride"):
        multiview_application.run_multiview_pose(options)


def test_poomsae_application_returns_explicit_result(monkeypatch, tmp_path: Path) -> None:
    summary_path = tmp_path / "run" / "json" / "poomsae_scoring_summary.json"
    summary_path.parent.mkdir(parents=True)
    summary = {"run": {"root": str(tmp_path / "run")}, "status": "diagnostics_only_no_score"}
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    monkeypatch.setattr(poomsae_application, "run_workflow", lambda **kwargs: summary_path)

    result = poomsae_application.run_poomsae_analysis(profile_value="fixture")

    assert result.summary_path == summary_path
    assert result.run_root == tmp_path / "run"
    assert result.summary == summary


@pytest.mark.parametrize("error", [OSError("snapshot unavailable"), KeyboardInterrupt(), SystemExit("stopped")])
def test_multiview_snapshot_failure_records_failed_state_and_preserves_latest(monkeypatch, tmp_path, error) -> None:
    _, previous = create_run_output_tree(tmp_path, "session_001", "previous")
    marker = mark_run_complete(tmp_path, "session_001", "previous", previous["root"])
    marker_before = marker.read_bytes()

    def fail_snapshot(*args, **kwargs):
        raise error

    monkeypatch.setattr(multiview_application, "snapshot_file", fail_snapshot)
    options = multiview_application.MultiviewRunOptions(
        session="data/session_001/session.yaml", output_root=tmp_path,
        run_id="failure", allow_approximate_calibration=True,
    )
    with pytest.raises(type(error)) as caught:
        multiview_application.run_multiview_pose(options)

    assert caught.value is error
    state = load_json_object(tmp_path / "session_001/runs/failure/run_state.json")
    assert state["status"] == "failed"
    assert type(error).__name__ in state["error"]
    assert marker.read_bytes() == marker_before


def test_multiview_existing_run_is_never_marked_failed(monkeypatch, tmp_path) -> None:
    _, existing = create_run_output_tree(tmp_path, "session_001", "existing")
    state_path = existing["root"] / "run_state.json"
    before = state_path.read_bytes()
    options = multiview_application.MultiviewRunOptions(
        session="data/session_001/session.yaml", output_root=tmp_path,
        run_id="existing", allow_approximate_calibration=True,
    )
    with pytest.raises(FileExistsError):
        multiview_application.run_multiview_pose(options)
    assert state_path.read_bytes() == before


@pytest.mark.parametrize("failure_at", ["second_camera", "metadata", "model"])
def test_multiview_closes_cameras_on_startup_error_or_cancellation(monkeypatch, tmp_path, failure_at) -> None:
    error = KeyboardInterrupt() if failure_at == "model" else OSError("camera failure")
    captures = []

    class Capture:
        def __init__(self, path):
            if failure_at == "second_camera" and captures:
                raise error
            self.released = 0
            captures.append(self)

        def isOpened(self):
            return True

        def get(self, prop):
            if failure_at == "metadata":
                raise error
            return {cv2.CAP_PROP_FPS: 30.0, cv2.CAP_PROP_FRAME_WIDTH: 640.0, cv2.CAP_PROP_FRAME_HEIGHT: 480.0}[prop]

        def release(self):
            self.released += 1

    def fail_model(config):
        raise error

    monkeypatch.setattr(multiview_application, "snapshot_file", lambda *args: {"sha256": "a" * 64})
    monkeypatch.setattr(multiview_application, "model_provenance", lambda *args: {})
    monkeypatch.setattr(multiview_application.cv2, "VideoCapture", Capture)
    monkeypatch.setattr(multiview_application, "ViTPose2DEstimator", fail_model)
    options = multiview_application.MultiviewRunOptions(
        session="data/session_001/session.yaml", output_root=tmp_path,
        run_id="failure", allow_approximate_calibration=True,
    )

    with pytest.raises(type(error)) as caught:
        multiview_application.run_multiview_pose(options)

    assert caught.value is error
    assert captures and all(capture.released == 1 for capture in captures)
    assert load_json_object(tmp_path / "session_001/runs/failure/run_state.json")["status"] == "failed"


def test_multiview_state_write_failure_keeps_original_error(monkeypatch, tmp_path) -> None:
    error = OSError("original snapshot failure")

    def fail_snapshot(*args):
        raise error

    def fail_state(*args):
        raise OSError("state unavailable")

    monkeypatch.setattr(multiview_application, "snapshot_file", fail_snapshot)
    monkeypatch.setattr(multiview_application, "mark_run_failed", fail_state)
    options = multiview_application.MultiviewRunOptions(
        session="data/session_001/session.yaml", output_root=tmp_path,
        run_id="failure", allow_approximate_calibration=True,
    )
    with pytest.raises(OSError) as caught:
        multiview_application.run_multiview_pose(options)
    assert caught.value is error
    assert "Could not persist failed run state" in error.__notes__[0]


def test_multiview_allowed_diagnostic_output_is_terminal_but_not_latest(monkeypatch, tmp_path) -> None:
    def diagnostic_result(options, context, resources):
        run_id, paths = create_run_output_tree(tmp_path, "session", "diagnostic")
        context.owned_run = (paths["root"], "session", run_id)
        return multiview_application.MultiviewRunResult(
            session_id="session", run_id=run_id, run_root=paths["root"],
            main_3d_path=paths["json"] / "pose.json", quality_path=paths["json"] / "quality.json",
            manifest_path=paths["json"] / "manifest.json", quality_passed=False,
        )

    monkeypatch.setattr(multiview_application, "_execute_multiview_pose", diagnostic_result)
    result = multiview_application.run_multiview_pose(
        multiview_application.MultiviewRunOptions(session="unused.yaml", allow_low_quality_output=True)
    )
    assert result.quality_passed is False
    assert load_json_object(result.run_root / "run_state.json")["status"] == "failed"
    assert not (tmp_path / "session/latest_run.json").exists()
