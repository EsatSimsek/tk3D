from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import savgol_filter

from .data_structures import PersonPose2D
from .temporal_smoothing import robust_savgol_segment


@dataclass(frozen=True, slots=True)
class Pose2DStabilizationConfig:
    """Zero-phase offline stabilization for already inferred 2D poses."""

    enabled: bool = True
    window_size: int = 9
    polynomial_order: int = 2
    min_outlier_distance_px: float = 6.0
    body_joint_count: int = 133

    def __post_init__(self) -> None:
        if self.window_size < 1 or self.window_size % 2 == 0:
            raise ValueError("window_size must be a positive odd integer")
        if self.polynomial_order < 1:
            raise ValueError("polynomial_order must be positive")
        if self.window_size > 1 and self.polynomial_order >= self.window_size:
            raise ValueError("polynomial_order must be smaller than window_size")
        if not np.isfinite(self.min_outlier_distance_px) or self.min_outlier_distance_px < 0.0:
            raise ValueError("min_outlier_distance_px must be finite and non-negative")
        if self.body_joint_count < 1:
            raise ValueError("body_joint_count must be positive")


def stabilize_pose2d_sequence(
    poses: list[PersonPose2D],
    config: Pose2DStabilizationConfig | None = None,
) -> list[PersonPose2D]:
    """Stabilize body joints without temporal lag or identity mixing.

    A centered Savitzky-Golay fit suppresses frame-to-frame detector/heatmap
    noise while preserving linear and quadratic motion. Each tracked identity
    and each contiguous valid run is processed independently.
    """
    if not poses:
        return []
    cfg = config or Pose2DStabilizationConfig()
    _validate_pose_sequence(poses)
    if not cfg.enabled or cfg.window_size <= 1:
        return [_copy_pose(pose) for pose in poses]

    xy = np.stack([np.asarray(pose.keypoints_xy, dtype=float) for pose in poses], axis=0)
    scores = np.stack([np.asarray(pose.scores, dtype=float) for pose in poses], axis=0)
    valid = np.stack([np.asarray(pose.valid_mask, dtype=bool) for pose in poses], axis=0)
    valid &= np.all(np.isfinite(xy), axis=-1)
    person_ids = np.asarray([pose.person_id for pose in poses], dtype=int)
    joint_count = min(cfg.body_joint_count, xy.shape[1])
    stabilized = xy.copy()

    for joint_idx in range(joint_count):
        joint_valid = valid[:, joint_idx]
        for start, end in _identity_valid_runs(joint_valid, person_ids):
            stabilized[start:end, joint_idx] = robust_savgol_segment(
                xy[start:end, joint_idx],
                window_size=cfg.window_size,
                polynomial_order=cfg.polynomial_order,
                min_outlier_distance=cfg.min_outlier_distance_px,
            )

    output: list[PersonPose2D] = []
    for frame_idx, pose in enumerate(poses):
        output.append(
            PersonPose2D(
                camera_id=pose.camera_id,
                frame_idx=pose.frame_idx,
                keypoints_xy=stabilized[frame_idx].copy(),
                scores=scores[frame_idx].copy(),
                valid_mask=valid[frame_idx].copy(),
                person_id=pose.person_id,
            )
        )
    return output


def pose2d_stability_metrics(
    raw_poses: list[PersonPose2D],
    stabilized_poses: list[PersonPose2D],
    body_joint_count: int = 17,
) -> dict[str, float | int | None]:
    """Measure high-frequency body-joint motion before and after stabilization."""
    if len(raw_poses) != len(stabilized_poses):
        raise ValueError("raw_poses and stabilized_poses must have the same length")
    if not raw_poses:
        return {
            "frame_count": 0,
            "comparable_body_samples": 0,
            "raw_high_frequency_ratio": None,
            "stabilized_high_frequency_ratio": None,
            "high_frequency_reduction_percent": None,
            "median_adjustment_px": None,
            "p95_adjustment_px": None,
        }
    _validate_pose_sequence(raw_poses)
    _validate_pose_sequence(stabilized_poses)
    raw_xy = np.stack([pose.keypoints_xy for pose in raw_poses], axis=0).astype(float)
    stable_xy = np.stack([pose.keypoints_xy for pose in stabilized_poses], axis=0).astype(float)
    raw_valid = np.stack([pose.valid_mask for pose in raw_poses], axis=0).astype(bool)
    stable_valid = np.stack([pose.valid_mask for pose in stabilized_poses], axis=0).astype(bool)
    joint_count = min(body_joint_count, raw_xy.shape[1])
    valid = (
        raw_valid[:, :joint_count]
        & stable_valid[:, :joint_count]
        & np.all(np.isfinite(raw_xy[:, :joint_count]), axis=-1)
        & np.all(np.isfinite(stable_xy[:, :joint_count]), axis=-1)
    )
    scales = _body_scales(raw_xy[:, :joint_count], valid)
    raw_energy = _high_frequency_ratio(raw_xy[:, :joint_count], valid, scales)
    stable_energy = _high_frequency_ratio(stable_xy[:, :joint_count], valid, scales)
    adjustments = np.linalg.norm(
        stable_xy[:, :joint_count] - raw_xy[:, :joint_count],
        axis=-1,
    )
    usable_adjustments = adjustments[valid & np.isfinite(adjustments)]
    reduction = None
    if raw_energy is not None and raw_energy > 1e-12 and stable_energy is not None:
        reduction = 100.0 * (raw_energy - stable_energy) / raw_energy
    return {
        "frame_count": len(raw_poses),
        "comparable_body_samples": int(np.count_nonzero(valid)),
        "raw_high_frequency_ratio": raw_energy,
        "stabilized_high_frequency_ratio": stable_energy,
        "high_frequency_reduction_percent": reduction,
        "median_adjustment_px": (float(np.median(usable_adjustments)) if usable_adjustments.size else None),
        "p95_adjustment_px": (float(np.percentile(usable_adjustments, 95.0)) if usable_adjustments.size else None),
    }


def _identity_valid_runs(
    valid: np.ndarray,
    person_ids: np.ndarray,
) -> list[tuple[int, int]]:
    usable = np.asarray(valid, dtype=bool)
    if usable.shape != person_ids.shape:
        raise ValueError("valid and person_ids must have matching shapes")
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index in range(usable.size):
        continues_identity = start is not None and index > start and person_ids[index] == person_ids[index - 1]
        if usable[index] and (start is None or continues_identity):
            if start is None:
                start = index
            continue
        if start is not None:
            runs.append((start, index))
            start = index if usable[index] else None
    if start is not None:
        runs.append((start, usable.size))
    return runs


def _true_runs(valid: np.ndarray) -> list[tuple[int, int]]:
    mask = np.asarray(valid, dtype=bool).reshape(-1)
    padded = np.r_[False, mask, False]
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(start), int(end)) for start, end in changes.reshape(-1, 2)]


def _validate_pose_sequence(poses: list[PersonPose2D]) -> None:
    camera_ids = {pose.camera_id for pose in poses}
    if len(camera_ids) != 1:
        raise ValueError("All poses must belong to the same camera")
    frame_indices = [int(pose.frame_idx) for pose in poses]
    if any(second < first for first, second in zip(frame_indices, frame_indices[1:])):
        raise ValueError("Pose frame indices must be non-decreasing")
    shapes = {np.asarray(pose.keypoints_xy).shape for pose in poses}
    if len(shapes) != 1:
        raise ValueError("All poses must use the same keypoint shape")


def _copy_pose(pose: PersonPose2D) -> PersonPose2D:
    return PersonPose2D(
        camera_id=pose.camera_id,
        frame_idx=pose.frame_idx,
        keypoints_xy=np.asarray(pose.keypoints_xy, dtype=float).copy(),
        scores=np.asarray(pose.scores, dtype=float).copy(),
        valid_mask=np.asarray(pose.valid_mask, dtype=bool).copy(),
        person_id=pose.person_id,
    )


def _body_scales(xy: np.ndarray, valid: np.ndarray) -> np.ndarray:
    scales = np.full(xy.shape[0], np.nan, dtype=float)
    for frame_idx in range(xy.shape[0]):
        points = xy[frame_idx][valid[frame_idx]]
        if points.shape[0] >= 2:
            scales[frame_idx] = np.linalg.norm(np.max(points, axis=0) - np.min(points, axis=0))
    finite = scales[np.isfinite(scales) & (scales > 1e-6)]
    fallback = float(np.median(finite)) if finite.size else 1.0
    return np.where(np.isfinite(scales) & (scales > 1e-6), scales, fallback)


def _high_frequency_ratio(
    xy: np.ndarray,
    valid: np.ndarray,
    scales: np.ndarray,
) -> float | None:
    if xy.shape[0] < 5:
        return None
    residuals: list[np.ndarray] = []
    for joint_idx in range(xy.shape[1]):
        for start, end in _true_runs(valid[:, joint_idx]):
            segment = xy[start:end, joint_idx]
            if segment.shape[0] < 5:
                continue
            trend_window = min(11, segment.shape[0] if segment.shape[0] % 2 else segment.shape[0] - 1)
            if trend_window < 5:
                continue
            trend = savgol_filter(
                segment,
                window_length=trend_window,
                polyorder=min(2, trend_window - 1),
                axis=0,
                mode="interp",
            )
            residual = np.linalg.norm(segment - trend, axis=-1)
            residuals.append(residual / scales[start:end])
    if not residuals:
        return None
    return float(np.median(np.concatenate(residuals)))
