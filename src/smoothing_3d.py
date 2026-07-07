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
