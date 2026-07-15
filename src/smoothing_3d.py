from __future__ import annotations

import numpy as np


def moving_average_nan(keypoints_3d: np.ndarray, window_size: int = 5) -> np.ndarray:
    """NaN-aware temporal moving average for 3D keypoint sequences.

    Uses vectorized cumulative-sum operations instead of a Python loop,
    giving O(n) performance regardless of window size.
    """
    if window_size <= 1:
        return keypoints_3d.copy()
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd")

    radius = window_size // 2
    keypoints_3d = np.asarray(keypoints_3d, dtype=float)
    frame_count = keypoints_3d.shape[0]

    finite = np.isfinite(keypoints_3d)
    safe = np.where(finite, keypoints_3d, 0.0)

    # Cumulative sums along the frame axis for O(n) sliding window
    cum_sum = np.cumsum(safe, axis=0)
    cum_count = np.cumsum(finite.astype(float), axis=0)

    # Prepend a zero row so that the subtraction works for all indices
    zero_shape = (1,) + keypoints_3d.shape[1:]
    cum_sum = np.concatenate([np.zeros(zero_shape, dtype=float), cum_sum], axis=0)
    cum_count = np.concatenate([np.zeros(zero_shape, dtype=float), cum_count], axis=0)

    # Window boundaries (1-indexed because of the prepended zero row)
    starts = np.clip(np.arange(frame_count) - radius, 0, frame_count)
    ends = np.clip(np.arange(frame_count) + radius + 1, 0, frame_count)

    window_sums = cum_sum[ends] - cum_sum[starts]
    window_counts = cum_count[ends] - cum_count[starts]

    smoothed = np.divide(
        window_sums,
        window_counts,
        out=np.full_like(keypoints_3d, np.nan, dtype=float),
        where=window_counts > 0,
    )
    return smoothed


def moving_average_pose(
    keypoints_3d: np.ndarray,
    window_size: int = 5,
    valid_mask: np.ndarray | None = None,
    min_valid_samples: int = 1,
) -> np.ndarray:
    """Smooth complete XYZ observations without mixing partial coordinates."""
    values = np.asarray(keypoints_3d, dtype=float)
    if values.ndim != 3 or values.shape[-1] != 3:
        raise ValueError(f"Expected [frames, joints, 3], got {values.shape}")
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd")
    valid = np.all(np.isfinite(values), axis=-1)
    if valid_mask is not None:
        supplied = np.asarray(valid_mask, dtype=bool)
        if supplied.shape != valid.shape:
            raise ValueError(f"valid_mask must have shape {valid.shape}, got {supplied.shape}")
        valid &= supplied
    if window_size <= 1:
        return np.where(valid[..., None], values, np.nan)
    if min_valid_samples < 1:
        raise ValueError("min_valid_samples must be at least 1")
    if values.shape[0] < window_size:
        return np.where(valid[..., None], values, np.nan)
    safe = np.where(valid[..., None], values, 0.0)
    radius = window_size // 2
    frame_count = values.shape[0]
    cumulative = np.concatenate([np.zeros((1,) + safe.shape[1:]), np.cumsum(safe, axis=0)], axis=0)
    counts = np.concatenate([np.zeros((1,) + valid.shape[1:]), np.cumsum(valid.astype(float), axis=0)], axis=0)
    starts = np.clip(np.arange(frame_count) - radius, 0, frame_count)
    ends = np.clip(np.arange(frame_count) + radius + 1, 0, frame_count)
    sums = cumulative[ends] - cumulative[starts]
    sample_counts = counts[ends] - counts[starts]
    return np.divide(
        sums,
        sample_counts[..., None],
        out=np.full_like(values, np.nan),
        where=(sample_counts >= min_valid_samples)[..., None],
    )
