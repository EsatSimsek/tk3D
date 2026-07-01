from __future__ import annotations

import numpy as np


def moving_average_nan(keypoints_3d: np.ndarray, window_size: int = 5) -> np.ndarray:
    if window_size <= 1:
        return keypoints_3d.copy()
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd")

    radius = window_size // 2
    smoothed = np.full_like(keypoints_3d, np.nan, dtype=float)
    frame_count = keypoints_3d.shape[0]
    for frame_idx in range(frame_count):
        start = max(0, frame_idx - radius)
        end = min(frame_count, frame_idx + radius + 1)
        window = keypoints_3d[start:end]
        finite = np.isfinite(window)
        counts = np.sum(finite, axis=0)
        sums = np.nansum(window, axis=0)
        smoothed[frame_idx] = np.divide(
            sums,
            counts,
            out=np.full_like(sums, np.nan, dtype=float),
            where=counts > 0,
        )
    return smoothed
