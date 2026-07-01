from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .data_structures import PersonPose2D


def draw_pose2d(frame: np.ndarray, pose: PersonPose2D, color: tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    output = frame.copy()
    for point, valid in zip(pose.keypoints_xy, pose.valid_mask):
        if valid and np.all(np.isfinite(point)):
            cv2.circle(output, tuple(np.round(point).astype(int)), 2, color, -1)
    return output


def write_placeholder_overlay_video(path: str | Path, size: tuple[int, int] = (1280, 720), fps: float = 30.0) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(target), fourcc, fps, size)
    blank = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    writer.write(blank)
    writer.release()
