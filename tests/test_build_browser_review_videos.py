from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts.build_browser_review_videos import build_browser_review_videos


def _write_mjpeg(path: Path, *, frame_count: int = 12, fps: float = 24.0) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (96, 64),
    )
    assert writer.isOpened()
    for index in range(frame_count):
        frame = np.full((64, 96, 3), index * 10, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_browser_review_video_transcode_preserves_timeline_and_dimensions(tmp_path: Path) -> None:
    camera_a = tmp_path / "camera_a.avi"
    camera_b = tmp_path / "camera_b.avi"
    _write_mjpeg(camera_a)
    _write_mjpeg(camera_b)
    output_dir = tmp_path / "browser"
    manifest_path = tmp_path / "browser_manifest.json"

    manifest = build_browser_review_videos(
        [("camera_a", camera_a), ("camera_b", camera_b)],
        output_dir,
        manifest_path,
    )

    assert manifest["codec"] == "h264"
    assert manifest["timeline_contract"]["frames_dropped"] is False
    assert len(manifest["videos"]) == 2
    for item in manifest["videos"]:
        assert item["input"]["frame_count"] == item["output"]["frame_count"] == 12
        assert item["input"]["fps"] == item["output"]["fps"] == 24.0
        assert (item["output"]["width"], item["output"]["height"]) == (96, 64)
        assert item["output"]["mime_type"] == "video/mp4"
        assert Path(item["output"]["path"]).is_file()
    stored = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert stored == manifest


def test_browser_review_video_transcode_refuses_duplicate_camera_ids(tmp_path: Path) -> None:
    camera = tmp_path / "camera.avi"
    _write_mjpeg(camera, frame_count=2)

    with pytest.raises(ValueError, match="camera ids must be unique"):
        build_browser_review_videos(
            [("camera", camera), ("camera", camera)],
            tmp_path / "browser",
            tmp_path / "manifest.json",
        )
