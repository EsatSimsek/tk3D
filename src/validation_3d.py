from __future__ import annotations

import numpy as np

from .data_structures import Validation

def validate_triangulation(
    keypoints_3d_world: np.ndarray,
    reprojection_error: np.ndarray,
    max_reprojection_error_px: float = 25.0,
) -> Validation:
    valid_xyz = np.all(np.isfinite(keypoints_3d_world), axis=-1)
    valid_error = np.isfinite(reprojection_error) & (reprojection_error <= max_reprojection_error_px)
    valid = valid_xyz & valid_error

    frame_valid_ratio = np.mean(valid, axis=1) if valid.size else np.array([])
    joint_valid_ratio = np.mean(valid, axis=0) if valid.size else np.array([])
    mean_error = _safe_nanmean_axis1(reprojection_error) if reprojection_error.size else np.array([])

    warnings: list[str] = []
    if frame_valid_ratio.size and _safe_nanmean(frame_valid_ratio) < 0.50:
        warnings.append("mean_frame_valid_ratio_below_0_50")
    if mean_error.size and _safe_nanmean(mean_error) > max_reprojection_error_px:
        warnings.append("mean_reprojection_error_above_threshold")

    return Validation(
        frame_valid_ratio=frame_valid_ratio,
        joint_valid_ratio=joint_valid_ratio,
        mean_reprojection_error_px=mean_error,
        warnings=warnings,
    )

def quality_summary(
    keypoints_3d_world: np.ndarray,
    triangulation_score: np.ndarray,
    reprojection_error: np.ndarray,
    used_cameras: np.ndarray,
    validation: Validation,
) -> dict[str, float | int | list[str]]:
    valid_xyz = np.all(np.isfinite(keypoints_3d_world), axis=-1)
    valid_joint_count = int(np.sum(np.any(valid_xyz, axis=0)))
    valid_frame_count = int(np.sum(np.any(valid_xyz, axis=1)))
    return {
        "frame_count": int(keypoints_3d_world.shape[0]),
        "keypoint_count": int(keypoints_3d_world.shape[1]),
        "valid_frame_count": valid_frame_count,
        "valid_joint_count": valid_joint_count,
        "mean_frame_valid_ratio": _safe_nanmean(validation.frame_valid_ratio),
        "mean_joint_valid_ratio": _safe_nanmean(validation.joint_valid_ratio),
        "mean_triangulation_score": _safe_nanmean(triangulation_score[triangulation_score > 0]),
        "mean_reprojection_error_px": _safe_nanmean(reprojection_error),
        "max_reprojection_error_px": _safe_nanmax(reprojection_error),
        "mean_used_cameras": _safe_nanmean(used_cameras[used_cameras > 0]),
        "warnings": validation.warnings,
    }

def _safe_nanmean(values: np.ndarray) -> float:
    if values.size == 0 or not np.any(np.isfinite(values)):
        return float("nan")
    return float(np.nanmean(values))

def _safe_nanmean_axis1(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    counts = np.sum(finite, axis=1)
    sums = np.nansum(values, axis=1)
    return np.divide(sums, counts, out=np.full(values.shape[0], np.nan, dtype=float), where=counts > 0)

def _safe_nanmax(values: np.ndarray) -> float:
    if values.size == 0 or not np.any(np.isfinite(values)):
        return float("nan")
    return float(np.nanmax(values))
