from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2

from .data_structures import Session


@dataclass(slots=True)
class VideoProbe:
    camera_id: str
    path: str
    exists: bool
    opened: bool
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    frame_count: int | None = None
    duration_sec: float | None = None


def probe_session_videos(session: Session) -> list[VideoProbe]:
    return [probe_video(camera.camera_id, camera.video_path) for camera in session.cameras]


def probe_video(camera_id: str, path: str | Path) -> VideoProbe:
    video_path = Path(path)
    if not video_path.exists():
        return VideoProbe(camera_id=camera_id, path=str(video_path), exists=False, opened=False)

    capture = cv2.VideoCapture(str(video_path))
    try:
        opened = bool(capture.isOpened())
        if not opened:
            return VideoProbe(camera_id=camera_id, path=str(video_path), exists=True, opened=False)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = frame_count / fps if fps > 0 else None
        return VideoProbe(
            camera_id=camera_id,
            path=str(video_path),
            exists=True,
            opened=True,
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            duration_sec=duration_sec,
        )
    finally:
        capture.release()


def video_probe_summary(probes: list[VideoProbe]) -> dict[str, Any]:
    opened = [probe for probe in probes if probe.opened]
    fps_values = [probe.fps for probe in opened if probe.fps is not None and probe.fps > 0]
    frame_counts = [probe.frame_count for probe in opened if probe.frame_count is not None]
    return {
        "camera_count": len(probes),
        "opened_count": len(opened),
        "all_opened": len(opened) == len(probes),
        "fps_min": min(fps_values) if fps_values else None,
        "fps_max": max(fps_values) if fps_values else None,
        "frame_count_min": min(frame_counts) if frame_counts else None,
        "frame_count_max": max(frame_counts) if frame_counts else None,
        "videos": [asdict(probe) for probe in probes],
    }
