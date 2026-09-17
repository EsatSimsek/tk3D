from __future__ import annotations

import json

import cv2
import pytest

from src.video_io import load_session
from src.video_probe import probe_session_videos, probe_video, video_probe_summary


def test_video_probe_reports_missing_demo_videos() -> None:
    session = load_session("data/session_001/session.yaml")
    probes = probe_session_videos(session)
    summary = video_probe_summary(probes)

    assert summary["camera_count"] == 3
    assert summary["opened_count"] == 0
    assert summary["all_opened"] is False
    assert all(not probe.exists for probe in probes)


@pytest.fixture
def captured_video(tmp_path, monkeypatch):
    video_path = tmp_path / "sample.mp4"
    video_path.touch()

    class Capture:
        def __init__(self):
            self.properties = {
                cv2.CAP_PROP_FRAME_WIDTH: 1280.0,
                cv2.CAP_PROP_FRAME_HEIGHT: 720.0,
                cv2.CAP_PROP_FPS: 60.0,
                cv2.CAP_PROP_FRAME_COUNT: 741.0,
            }
            self.released = False

        def isOpened(self):
            return True

        def get(self, property_id):
            return self.properties[property_id]

        def release(self):
            self.released = True

    capture = Capture()
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: capture)
    return video_path, capture


def test_video_probe_preserves_valid_metadata(captured_video) -> None:
    video_path, capture = captured_video

    probe = probe_video("cam", video_path)

    assert (probe.width, probe.height, probe.fps, probe.frame_count) == (1280, 720, 60.0, 741)
    assert probe.duration_sec == pytest.approx(12.35)
    assert capture.released


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf"), -1.0, 0.0])
def test_video_probe_reports_invalid_metadata_as_unavailable(captured_video, invalid) -> None:
    video_path, capture = captured_video
    capture.properties = {key: invalid for key in capture.properties}

    probe = probe_video("cam", video_path)
    summary = video_probe_summary([probe])

    assert probe.opened is True
    assert (probe.width, probe.height, probe.fps, probe.frame_count, probe.duration_sec) == (None,) * 5
    assert summary["fps_min"] is None
    assert summary["frame_count_min"] is None
    json.dumps(summary, allow_nan=False)
    assert capture.released
