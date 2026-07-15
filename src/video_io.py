from __future__ import annotations

import math
from pathlib import Path
import re
from typing import Iterator

import yaml

from .data_structures import CameraView, Frame, Session


_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")


def load_session(session_path: str | Path) -> Session:
    path = Path(session_path).resolve()
    with path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)

    if not isinstance(raw, dict):
        raise ValueError("Session YAML root must be a mapping")
    session_id = _validated_id(raw.get("session_id"), "session_id")
    raw_cameras = raw.get("cameras")
    if not isinstance(raw_cameras, list) or not raw_cameras:
        raise ValueError("Session must define at least one camera")
    camera_ids = [_validated_id(item.get("camera_id") if isinstance(item, dict) else None, "camera_id") for item in raw_cameras]
    if len(camera_ids) != len(set(camera_ids)):
        raise ValueError("Session camera_id values must be unique")
    fps = raw.get("fps")
    if fps is not None and (not math.isfinite(float(fps)) or float(fps) <= 0.0):
        raise ValueError("Session fps must be positive when provided")

    root_dir = path.parent
    sync = raw.get("sync", {})
    offsets = sync.get("offsets", {})
    time_offsets = sync.get("offsets_sec", {})
    cameras = [
        CameraView(
            camera_id=item["camera_id"],
            video_path=(root_dir / item["video_path"]).resolve(),
            calibration_video_path=(root_dir / item["calibration_video_path"]).resolve()
            if item.get("calibration_video_path")
            else None,
            frame_offset=int(offsets.get(item["camera_id"], 0)),
            time_offset_sec=float(time_offsets.get(item["camera_id"], 0.0)),
        )
        for item in raw_cameras
    ]
    return Session(
        session_id=session_id,
        task_name=raw.get("task_name", "unknown_task"),
        root_dir=root_dir,
        cameras=cameras,
        fps=float(fps) if fps is not None else None,
        sync_method=str(sync.get("method", "timestamp")),
    )


def ensure_output_tree(output_root: str | Path, session_id: str) -> dict[str, Path]:
    _validated_id(session_id, "session_id")
    root = Path(output_root).resolve() / session_id
    paths = {
        "root": root,
        "videos": root / "videos",
        "figures": root / "figures",
        "csv": root / "csv",
        "json": root / "json",
        "calibration": root / "calibration",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def iter_video_frames(video_path: str | Path, stride: int = 1) -> Iterator[tuple[int, float, object]]:
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    frame_idx = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_idx % stride == 0:
                timestamp_sec = frame_idx / fps if fps > 0 else 0.0
                yield frame_idx, timestamp_sec, frame
            frame_idx += 1
    finally:
        capture.release()


def frame_record(frame_idx: int, fps: float | None, camera_id: str) -> Frame:
    timestamp_sec = frame_idx / fps if fps else 0.0
    return Frame(frame_idx=frame_idx, timestamp_sec=timestamp_sec, camera_id=camera_id)


def _validated_id(value: object, label: str) -> str:
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
        raise ValueError(f"{label} must contain only letters, numbers, dot, underscore, or hyphen")
    return value
