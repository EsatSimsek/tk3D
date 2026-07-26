from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .data_structures import PersonPose2D
from .visualization_3d import (
    COCO_BODY_EDGES,
    COCO_FOOT_EDGES,
    COCO_HAND_EDGES,
)


def draw_pose2d(
    frame: np.ndarray,
    pose: PersonPose2D,
    color: tuple[int, int, int] = (0, 255, 0),
    edge_color: tuple[int, int, int] = (255, 180, 40),
    draw_wholebody_points: bool = False,
    draw_hands: bool = True,
    draw_feet: bool = True,
    provenance: np.ndarray | None = None,
    crossview_color: tuple[int, int, int] = (0, 165, 255),
    image_guided_color: tuple[int, int, int] = (255, 0, 255),
) -> np.ndarray:
    output = frame.copy()
    valid_points = np.asarray(pose.valid_mask, dtype=bool) & np.all(np.isfinite(pose.keypoints_xy), axis=1)
    provenance_values = (
        np.zeros(pose.keypoints_xy.shape[0], dtype=np.uint8)
        if provenance is None
        else np.asarray(provenance, dtype=np.uint8)
    )
    if provenance_values.shape != (pose.keypoints_xy.shape[0],):
        raise ValueError(
            f"provenance must have shape {(pose.keypoints_xy.shape[0],)}, "
            f"got {provenance_values.shape}"
        )
    coordinate_shift = 4
    coordinate_scale = 1 << coordinate_shift
    line_thickness = max(int(round(min(frame.shape[:2]) / 360.0)), 2)
    point_radius = max(int(round(min(frame.shape[:2]) / 270.0)), 3)
    detail_line_thickness = max(line_thickness - 1, 1)
    detail_point_radius = max(point_radius - 1, 2)
    edges = list(COCO_BODY_EDGES)
    if draw_feet:
        edges.extend(COCO_FOOT_EDGES)
    if draw_hands:
        edges.extend(COCO_HAND_EDGES)
    for start, end in edges:
        if start >= pose.keypoints_xy.shape[0] or end >= pose.keypoints_xy.shape[0]:
            continue
        if valid_points[start] and valid_points[end]:
            p1 = tuple(np.round(pose.keypoints_xy[start] * coordinate_scale).astype(int))
            p2 = tuple(np.round(pose.keypoints_xy[end] * coordinate_scale).astype(int))
            current_edge_color = (
                crossview_color
                if provenance_values[start] == 2 or provenance_values[end] == 2
                else image_guided_color
                if provenance_values[start] == 1 or provenance_values[end] == 1
                else edge_color
            )
            cv2.line(
                output,
                p1,
                p2,
                current_edge_color,
                line_thickness if (start, end) in COCO_BODY_EDGES else detail_line_thickness,
                lineType=cv2.LINE_AA,
                shift=coordinate_shift,
            )
    if draw_wholebody_points:
        point_indices = range(pose.keypoints_xy.shape[0])
    else:
        point_indices = list(range(min(17, pose.keypoints_xy.shape[0])))
        if draw_feet:
            point_indices.extend(
                range(17, min(23, pose.keypoints_xy.shape[0])),
            )
        if draw_hands and pose.keypoints_xy.shape[0] > 91:
            point_indices.extend(
                range(91, min(133, pose.keypoints_xy.shape[0])),
            )
    for point_idx in point_indices:
        point = pose.keypoints_xy[point_idx]
        valid = valid_points[point_idx]
        if valid:
            center = tuple(np.round(point * coordinate_scale).astype(int))
            current_color = (
                crossview_color
                if provenance_values[point_idx] == 2
                else image_guided_color
                if provenance_values[point_idx] == 1
                else color
            )
            cv2.circle(
                output,
                center,
                (point_radius if point_idx < 17 else detail_point_radius) * coordinate_scale,
                current_color,
                -1,
                lineType=cv2.LINE_AA,
                shift=coordinate_shift,
            )
    if np.any(provenance_values == 2):
        cv2.putText(
            output,
            "orange: cross-view recovered",
            (18, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            crossview_color,
            2,
            cv2.LINE_AA,
        )
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
