from __future__ import annotations

from src.preflight import has_errors, preflight_summary, run_preflight
from src.video_io import load_session


def test_preflight_warns_when_demo_files_are_missing() -> None:
    session = load_session("data/session_001/session.yaml")
    model_config = {
        "pose2d": {
            "config_path": "models/vitpose/ViTPose_huge_wholebody_256x192.py",
            "checkpoint_path": "weights/vitpose_huge_wholebody_256x192.pth",
        },
        "pose3d_single_view": {
            "enabled": True,
            "config_path": "models/rtmw3d/rtmw3d-x.py",
            "checkpoint_path": "weights/rtmw3d-x.pth",
        },
    }

    issues = run_preflight(
        session=session,
        model_config=model_config,
        videos_required=False,
        calibration_videos_required=False,
        model_files_required=False,
    )
    summary = preflight_summary(issues)

    assert not has_errors(issues)
    assert summary["status"] == "passed"
    assert summary["warning_count"] >= 3


def test_preflight_can_be_strict_for_live_inputs() -> None:
    session = load_session("data/session_001/session.yaml")
    issues = run_preflight(
        session=session,
        model_config={"pose2d": {}, "pose3d_single_view": {"enabled": False}},
        videos_required=True,
        calibration_videos_required=True,
        model_files_required=True,
    )

    assert has_errors(issues)
    assert preflight_summary(issues)["status"] == "failed"
