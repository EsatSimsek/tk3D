from __future__ import annotations

from src.video_io import load_session
from src.video_probe import probe_session_videos, video_probe_summary


def test_video_probe_reports_missing_demo_videos() -> None:
    session = load_session("data/session_001/session.yaml")
    probes = probe_session_videos(session)
    summary = video_probe_summary(probes)

    assert summary["camera_count"] == 3
    assert summary["opened_count"] == 0
    assert summary["all_opened"] is False
    assert all(not probe.exists for probe in probes)
