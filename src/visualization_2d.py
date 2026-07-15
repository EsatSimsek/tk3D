from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .data_structures import PersonPose2D
from .visualization_3d import COCO_BODY_EDGES


def draw_pose2d(
    frame: np.ndarray,
    pose: PersonPose2D,
    color: tuple[int, int, int] = (0, 255, 0),
    edge_color: tuple[int, int, int] = (255, 180, 40),
) -> np.ndarray:
    output = frame.copy()
    valid_points = np.asarray(pose.valid_mask, dtype=bool) & np.all(np.isfinite(pose.keypoints_xy), axis=1)
    for start, end in COCO_BODY_EDGES:
        if start >= pose.keypoints_xy.shape[0] or end >= pose.keypoints_xy.shape[0]:
            continue
        if valid_points[start] and valid_points[end]:
            p1 = tuple(np.round(pose.keypoints_xy[start]).astype(int))
            p2 = tuple(np.round(pose.keypoints_xy[end]).astype(int))
            cv2.line(output, p1, p2, edge_color, 2, lineType=cv2.LINE_AA)
    for point, valid in zip(pose.keypoints_xy, valid_points):
        if valid:
            cv2.circle(output, tuple(np.round(point).astype(int)), 3, color, -1, lineType=cv2.LINE_AA)
    return output


def write_placeholder_overlay_video(path: str | Path, size: tuple[int, int] = (1280, 720), fps: float = 30.0) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(target), fourcc, fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {target}")
    blank = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    writer.write(blank)
    writer.release()
