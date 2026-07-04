from __future__ import annotations

import numpy as np


def angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom == 0 or not np.isfinite(denom):
        return float("nan")
    cosine = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def segment_length(a: np.ndarray, b: np.ndarray) -> float:
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return float("nan")
    return float(np.linalg.norm(a - b))


def center_of_mass_proxy(keypoints_3d: np.ndarray, joint_indices: list[int]) -> np.ndarray:
    selected = keypoints_3d[joint_indices]
    if selected.size == 0 or not np.any(np.isfinite(selected)):
        return np.array([np.nan, np.nan, np.nan], dtype=float)
    sums = np.nansum(selected, axis=0)
    counts = np.sum(np.isfinite(selected), axis=0)
    return np.divide(sums, counts, out=np.full(3, np.nan, dtype=float), where=counts > 0)
