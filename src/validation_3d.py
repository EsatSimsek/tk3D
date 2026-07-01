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
    mean_error = np.nanmean(reprojection_error, axis=1) if reprojection_error.size else np.array([])

    warnings: list[str] = []
    if frame_valid_ratio.size and float(np.nanmean(frame_valid_ratio)) < 0.50:
        warnings.append("mean_frame_valid_ratio_below_0_50")
    if mean_error.size and float(np.nanmean(mean_error)) > max_reprojection_error_px:
        warnings.append("mean_reprojection_error_above_threshold")

    return Validation(
        frame_valid_ratio=frame_valid_ratio,
        joint_valid_ratio=joint_valid_ratio,
        mean_reprojection_error_px=mean_error,
        warnings=warnings,
    )
