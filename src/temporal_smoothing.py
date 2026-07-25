from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter


def robust_savgol_keypoints(
    values: np.ndarray,
    *,
    window_size: int,
    polynomial_order: int,
    valid_mask: np.ndarray | None = None,
    min_outlier_distance: float = 0.0,
    joint_indices: range | list[int] | tuple[int, ...] | None = None,
) -> np.ndarray:
    """Apply robust zero-phase smoothing to generic keypoint trajectories."""
    points = np.asarray(values, dtype=float)
    if points.ndim != 3:
        raise ValueError(f"Expected [frames, joints, coordinates], got {points.shape}")
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd integer")
    if polynomial_order < 1:
        raise ValueError("polynomial_order must be positive")
    if window_size > 1 and polynomial_order >= window_size:
        raise ValueError("polynomial_order must be smaller than window_size")
    if not np.isfinite(min_outlier_distance) or min_outlier_distance < 0.0:
        raise ValueError("min_outlier_distance must be finite and non-negative")
    finite = np.all(np.isfinite(points), axis=-1)
    if valid_mask is None:
        valid = finite
    else:
        supplied = np.asarray(valid_mask, dtype=bool)
        if supplied.shape != points.shape[:2]:
            raise ValueError(f"valid_mask must have shape {points.shape[:2]}, got {supplied.shape}")
        valid = supplied & finite
    output = np.where(valid[..., None], points, np.nan)
    if window_size <= 1 or points.shape[0] < 3:
        return output

    indices = range(points.shape[1]) if joint_indices is None else joint_indices
    for joint_idx in indices:
        if not 0 <= int(joint_idx) < points.shape[1]:
            raise ValueError(f"joint index out of range: {joint_idx}")
        for start, end in true_runs(valid[:, int(joint_idx)]):
            output[start:end, int(joint_idx)] = robust_savgol_segment(
                points[start:end, int(joint_idx)],
                window_size=window_size,
                polynomial_order=polynomial_order,
                min_outlier_distance=min_outlier_distance,
            )
    return output


def robust_savgol_segment(
    values: np.ndarray,
    *,
    window_size: int,
    polynomial_order: int,
    min_outlier_distance: float,
) -> np.ndarray:
    segment = np.asarray(values, dtype=float)
    length = segment.shape[0]
    effective_window = min(window_size, length if length % 2 else length - 1)
    if effective_window <= polynomial_order or effective_window < 3:
        return segment.copy()
    first_pass = savgol_filter(
        segment,
        window_length=effective_window,
        polyorder=min(polynomial_order, effective_window - 1),
        axis=0,
        mode="interp",
    )
    if min_outlier_distance <= 0.0 or length < 5:
        return first_pass

    residual = np.linalg.norm(segment - first_pass, axis=-1)
    center = float(np.median(residual))
    mad = float(np.median(np.abs(residual - center)))
    robust_sigma = 1.4826 * mad
    threshold = max(min_outlier_distance, center + 4.0 * robust_sigma)
    outliers = residual > threshold
    if not np.any(outliers):
        return first_pass
    cleaned = segment.copy()
    cleaned[outliers] = first_pass[outliers]
    return savgol_filter(
        cleaned,
        window_length=effective_window,
        polyorder=min(polynomial_order, effective_window - 1),
        axis=0,
        mode="interp",
    )


def true_runs(valid: np.ndarray) -> list[tuple[int, int]]:
    mask = np.asarray(valid, dtype=bool).reshape(-1)
    padded = np.r_[False, mask, False]
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(start), int(end)) for start, end in changes.reshape(-1, 2)]
