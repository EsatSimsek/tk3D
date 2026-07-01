from __future__ import annotations

from pathlib import Path
from typing import Iterator

import yaml

from .data_structures import CameraView, Frame, Session


def load_session(session_path: str | Path) -> Session:
    path = Path(session_path).resolve()
    with path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)

    root_dir = path.parent
    offsets = raw.get("sync", {}).get("offsets", {})
    cameras = [
        CameraView(
            camera_id=item["camera_id"],
            video_path=(root_dir / item["video_path"]).resolve(),
            calibration_video_path=(root_dir / item["calibration_video_path"]).resolve()
            if item.get("calibration_video_path")
            else None,
            frame_offset=int(offsets.get(item["camera_id"], 0)),
        )
        for item in raw["cameras"]
    ]
    return Session(
        session_id=raw["session_id"],
        task_name=raw.get("task_name", "unknown_task"),
        root_dir=root_dir,
        cameras=cameras,
        fps=raw.get("fps"),
    )


def ensure_output_tree(output_root: str | Path, session_id: str) -> dict[str, Path]:
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
