from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .data_structures import COCO_BODY_JOINT_INDICES, COCO_BODY_JOINTS


REJECTION_BASE_QUALITY = 1
REJECTION_BONE_LENGTH = 2
REJECTION_TEMPORAL = 4

_BODY_BONES: tuple[tuple[int, int], ...] = tuple(
    (COCO_BODY_JOINTS[first], COCO_BODY_JOINTS[second])
    for first, second in (
        ("left_shoulder", "right_shoulder"),
        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"),
        ("left_shoulder", "left_hip"),
        ("right_shoulder", "right_hip"),
        ("left_hip", "right_hip"),
        ("left_hip", "left_knee"),
        ("left_knee", "left_ankle"),
        ("right_hip", "right_knee"),
        ("right_knee", "right_ankle"),
    )
)


@dataclass(slots=True)
class PoseReliabilityResult:
    keypoints_3d: np.ndarray
    valid_mask: np.ndarray
    rejection_reasons: np.ndarray
    summary: dict[str, Any]


def filter_unreliable_pose(
    keypoints_3d: np.ndarray,
    base_valid_mask: np.ndarray,
    timestamps_sec: np.ndarray,
    confidence: np.ndarray | None = None,
    *,
    max_bone_relative_deviation: float = 0.25,
    max_bone_absolute_deviation_m: float = 0.08,
    min_temporal_residual_m: float = 0.08,
    max_temporal_acceleration_mps2: float = 70.0,
    minimum_bone_samples: int = 5,
) -> PoseReliabilityResult:
    """Reject anatomically or temporally implausible body joints without GT data."""
    points = np.asarray(keypoints_3d, dtype=float)
    supplied_valid = np.asarray(base_valid_mask, dtype=bool)
    timestamps = np.asarray(timestamps_sec, dtype=float)
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError(f"Expected keypoints [frames, joints, 3], got {points.shape}")
    if supplied_valid.shape != points.shape[:2]:
        raise ValueError(f"base_valid_mask must have shape {points.shape[:2]}")
    if timestamps.shape != (points.shape[0],):
        raise ValueError(f"timestamps_sec must have shape {(points.shape[0],)}")
    if np.any(np.diff(timestamps) <= 0.0):
        raise ValueError("timestamps_sec must be strictly increasing")
    if max_bone_relative_deviation <= 0.0 or max_bone_absolute_deviation_m <= 0.0:
        raise ValueError("bone deviation limits must be positive")
    if min_temporal_residual_m <= 0.0 or max_temporal_acceleration_mps2 <= 0.0:
        raise ValueError("temporal reliability limits must be positive")
    if minimum_bone_samples < 1:
        raise ValueError("minimum_bone_samples must be positive")

    if confidence is None:
        scores = np.ones(points.shape[:2], dtype=float)
    else:
        scores = np.asarray(confidence, dtype=float)
        if scores.shape != points.shape[:2]:
            raise ValueError(f"confidence must have shape {points.shape[:2]}")
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    finite = np.all(np.isfinite(points), axis=-1)
    initial_valid = supplied_valid & finite
    valid = initial_valid.copy()
    reasons = np.zeros(points.shape[:2], dtype=np.uint8)
    reasons[~initial_valid] |= REJECTION_BASE_QUALITY

    body_indices = np.asarray([index for index in COCO_BODY_JOINT_INDICES if index < points.shape[1]])
    body_set = set(body_indices.tolist())
    bones = [(first, second) for first, second in _BODY_BONES if first in body_set and second in body_set]
    degrees = {index: 0 for index in body_set}
    for first, second in bones:
        degrees[first] += 1
        degrees[second] += 1

    bone_rejected = np.zeros_like(valid)
    reference_lengths: dict[str, float] = {}
    for first, second in bones:
        observed = initial_valid[:, first] & initial_valid[:, second]
        lengths = np.linalg.norm(points[:, first] - points[:, second], axis=-1)
        usable = lengths[observed & np.isfinite(lengths)]
        if usable.size < minimum_bone_samples:
            continue
        reference = float(np.median(usable))
        if not np.isfinite(reference) or reference <= 1e-6:
            continue
        reference_lengths[f"{first}-{second}"] = reference
        tolerance = max(max_bone_absolute_deviation_m, max_bone_relative_deviation * reference)
        bad_frames = np.flatnonzero(observed & (np.abs(lengths - reference) > tolerance))
        for frame_idx in bad_frames:
            endpoint = _less_reliable_endpoint(
                first,
                second,
                scores[frame_idx],
                degrees,
            )
            bone_rejected[frame_idx, endpoint] = True
    valid[bone_rejected] = False
    reasons[bone_rejected] |= REJECTION_BONE_LENGTH

    temporal_rejected = np.zeros_like(valid)
    for frame_idx in range(1, points.shape[0] - 1):
        previous_dt = float(timestamps[frame_idx] - timestamps[frame_idx - 1])
        next_dt = float(timestamps[frame_idx + 1] - timestamps[frame_idx])
        span = previous_dt + next_dt
        if previous_dt <= 0.0 or next_dt <= 0.0 or span <= 0.0:
            continue
        alpha = previous_dt / span
        expected = points[frame_idx - 1] + alpha * (points[frame_idx + 1] - points[frame_idx - 1])
        residual = np.linalg.norm(points[frame_idx] - expected, axis=-1)
        tolerance = max(
            min_temporal_residual_m,
            0.5 * max_temporal_acceleration_mps2 * previous_dt * next_dt,
        )
        neighbor_valid = valid[frame_idx - 1] & valid[frame_idx] & valid[frame_idx + 1]
        temporal_rejected[frame_idx] = neighbor_valid & (residual > tolerance)
        temporal_rejected[frame_idx, np.setdiff1d(np.arange(points.shape[1]), body_indices)] = False
    valid[temporal_rejected] = False
    reasons[temporal_rejected] |= REJECTION_TEMPORAL

    body_total = int(points.shape[0] * body_indices.size)
    input_valid_count = int(np.count_nonzero(initial_valid[:, body_indices])) if body_total else 0
    output_valid_count = int(np.count_nonzero(valid[:, body_indices])) if body_total else 0
    summary = {
        "algorithm": "tk3d_anatomical_temporal_reliability_v1",
        "body_point_count": body_total,
        "input_valid_body_point_count": input_valid_count,
        "output_valid_body_point_count": output_valid_count,
        "input_valid_body_ratio": input_valid_count / body_total if body_total else 0.0,
        "output_valid_body_ratio": output_valid_count / body_total if body_total else 0.0,
        "base_quality_rejection_count": int(
            np.count_nonzero((reasons[:, body_indices] & REJECTION_BASE_QUALITY) != 0)
        ),
        "bone_length_rejection_count": int(
            np.count_nonzero((reasons[:, body_indices] & REJECTION_BONE_LENGTH) != 0)
        ),
        "temporal_rejection_count": int(
            np.count_nonzero((reasons[:, body_indices] & REJECTION_TEMPORAL) != 0)
        ),
        "reference_bone_lengths_m": reference_lengths,
        "thresholds": {
            "max_bone_relative_deviation": max_bone_relative_deviation,
            "max_bone_absolute_deviation_m": max_bone_absolute_deviation_m,
            "min_temporal_residual_m": min_temporal_residual_m,
            "max_temporal_acceleration_mps2": max_temporal_acceleration_mps2,
            "minimum_bone_samples": minimum_bone_samples,
        },
    }
    filtered = np.where(valid[..., None], points, np.nan)
    return PoseReliabilityResult(filtered, valid, reasons, summary)


def _less_reliable_endpoint(
    first: int,
    second: int,
    frame_scores: np.ndarray,
    degrees: dict[int, int],
) -> int:
    first_score = float(frame_scores[first])
    second_score = float(frame_scores[second])
    if abs(first_score - second_score) > 0.10:
        return first if first_score < second_score else second
    first_degree = degrees.get(first, 0)
    second_degree = degrees.get(second, 0)
    if first_degree != second_degree:
        return first if first_degree < second_degree else second
    return second
