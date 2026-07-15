from __future__ import annotations

import numpy as np

from src.data_structures import COCO_BODY_JOINTS
from src.pose_reliability import (
    REJECTION_BONE_LENGTH,
    REJECTION_TEMPORAL,
    filter_unreliable_pose,
)


def _stable_pose(frame_count: int = 9) -> np.ndarray:
    points = np.zeros((frame_count, 133, 3), dtype=float)
    idx = COCO_BODY_JOINTS
    points[:, idx["left_shoulder"]] = [-0.2, 0.0, 1.5]
    points[:, idx["right_shoulder"]] = [0.2, 0.0, 1.5]
    points[:, idx["left_elbow"]] = [-0.45, 0.0, 1.25]
    points[:, idx["right_elbow"]] = [0.45, 0.0, 1.25]
    points[:, idx["left_wrist"]] = [-0.65, 0.0, 1.0]
    points[:, idx["right_wrist"]] = [0.65, 0.0, 1.0]
    points[:, idx["left_hip"]] = [-0.15, 0.0, 1.0]
    points[:, idx["right_hip"]] = [0.15, 0.0, 1.0]
    points[:, idx["left_knee"]] = [-0.15, 0.0, 0.55]
    points[:, idx["right_knee"]] = [0.15, 0.0, 0.55]
    points[:, idx["left_ankle"]] = [-0.15, 0.0, 0.1]
    points[:, idx["right_ankle"]] = [0.15, 0.0, 0.1]
    return points


def test_reliability_filter_rejects_single_bone_and_temporal_spike() -> None:
    points = _stable_pose()
    idx = COCO_BODY_JOINTS
    points[4, idx["left_wrist"]] = [-2.0, 0.0, 2.0]
    valid = np.ones(points.shape[:2], dtype=bool)
    scores = np.full(points.shape[:2], 0.9)
    scores[4, idx["left_wrist"]] = 0.4

    result = filter_unreliable_pose(
        points,
        valid,
        np.arange(points.shape[0], dtype=float) / 30.0,
        scores,
    )

    assert not result.valid_mask[4, idx["left_wrist"]]
    reason = result.rejection_reasons[4, idx["left_wrist"]]
    assert reason & REJECTION_BONE_LENGTH
    assert result.summary["bone_length_rejection_count"] >= 1


def test_reliability_filter_temporal_limit_scales_for_sparse_samples() -> None:
    points = _stable_pose()
    idx = COCO_BODY_JOINTS
    points[:, idx["left_wrist"], 0] += np.linspace(0.0, 1.0, points.shape[0])
    valid = np.ones(points.shape[:2], dtype=bool)
    sparse_timestamps = np.arange(points.shape[0], dtype=float) * 0.5

    result = filter_unreliable_pose(points, valid, sparse_timestamps)

    assert not np.any(result.rejection_reasons[:, idx["left_wrist"]] & REJECTION_TEMPORAL)
