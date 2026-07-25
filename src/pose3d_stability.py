from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import savgol_filter

from .data_structures import COCO_BODY_JOINTS


ANGLE_SPECS: dict[str, tuple[str, str, str]] = {
    "left_elbow": ("left_shoulder", "left_elbow", "left_wrist"),
    "right_elbow": ("right_shoulder", "right_elbow", "right_wrist"),
    "left_shoulder": ("left_elbow", "left_shoulder", "left_hip"),
    "right_shoulder": ("right_elbow", "right_shoulder", "right_hip"),
    "left_hip": ("left_shoulder", "left_hip", "left_knee"),
    "right_hip": ("right_shoulder", "right_hip", "right_knee"),
    "left_knee": ("left_hip", "left_knee", "left_ankle"),
    "right_knee": ("right_hip", "right_knee", "right_ankle"),
}


def pose3d_stability_metrics(
    raw_keypoints_3d: np.ndarray,
    stabilized_keypoints_3d: np.ndarray,
    body_joint_count: int = 17,
) -> dict[str, Any]:
    """Quantify 3D joint and angle jitter without changing scoring logic."""
    raw = np.asarray(raw_keypoints_3d, dtype=float)
    stable = np.asarray(stabilized_keypoints_3d, dtype=float)
    if raw.shape != stable.shape:
        raise ValueError("raw and stabilized 3D arrays must have the same shape")
    if raw.ndim != 3 or raw.shape[-1] != 3:
        raise ValueError(f"Expected [frames, joints, 3], got {raw.shape}")

    joint_count = min(body_joint_count, raw.shape[1])
    raw_body = raw[:, :joint_count]
    stable_body = stable[:, :joint_count]
    valid = np.all(np.isfinite(raw_body), axis=-1) & np.all(
        np.isfinite(stable_body),
        axis=-1,
    )
    scales = _body_scales(raw_body, valid)
    raw_joint_hf = _joint_high_frequency_ratio(raw_body, valid, scales)
    stable_joint_hf = _joint_high_frequency_ratio(stable_body, valid, scales)
    adjustment_m = np.linalg.norm(stable_body - raw_body, axis=-1)
    usable_adjustments = adjustment_m[valid & np.isfinite(adjustment_m)]

    angle_metrics: dict[str, dict[str, float | int | None]] = {}
    for name, joint_names in ANGLE_SPECS.items():
        indices = tuple(COCO_BODY_JOINTS[joint_name] for joint_name in joint_names)
        if any(index >= joint_count for index in indices):
            continue
        first, center, last = indices
        angle_valid = valid[:, first] & valid[:, center] & valid[:, last]
        raw_angles = joint_angles_degrees(
            raw_body[:, first],
            raw_body[:, center],
            raw_body[:, last],
        )
        stable_angles = joint_angles_degrees(
            stable_body[:, first],
            stable_body[:, center],
            stable_body[:, last],
        )
        angle_valid &= np.isfinite(raw_angles) & np.isfinite(stable_angles)
        raw_angle_hf = _scalar_high_frequency_median(raw_angles, angle_valid)
        stable_angle_hf = _scalar_high_frequency_median(stable_angles, angle_valid)
        angle_metrics[name] = {
            "valid_frame_count": int(np.count_nonzero(angle_valid)),
            "raw_high_frequency_median_deg": raw_angle_hf,
            "stabilized_high_frequency_median_deg": stable_angle_hf,
            "reduction_percent": _reduction_percent(raw_angle_hf, stable_angle_hf),
        }

    raw_angle_values = [
        metric["raw_high_frequency_median_deg"]
        for metric in angle_metrics.values()
        if metric["raw_high_frequency_median_deg"] is not None
    ]
    stable_angle_values = [
        metric["stabilized_high_frequency_median_deg"]
        for metric in angle_metrics.values()
        if metric["stabilized_high_frequency_median_deg"] is not None
    ]
    raw_angle_median = float(np.median(np.asarray(raw_angle_values, dtype=float))) if raw_angle_values else None
    stable_angle_median = (
        float(np.median(np.asarray(stable_angle_values, dtype=float))) if stable_angle_values else None
    )
    return {
        "frame_count": int(raw.shape[0]),
        "comparable_body_samples": int(np.count_nonzero(valid)),
        "raw_joint_high_frequency_ratio": raw_joint_hf,
        "stabilized_joint_high_frequency_ratio": stable_joint_hf,
        "joint_high_frequency_reduction_percent": _reduction_percent(
            raw_joint_hf,
            stable_joint_hf,
        ),
        "median_adjustment_mm": (1000.0 * float(np.median(usable_adjustments)) if usable_adjustments.size else None),
        "p95_adjustment_mm": (
            1000.0 * float(np.percentile(usable_adjustments, 95.0)) if usable_adjustments.size else None
        ),
        "raw_angle_high_frequency_median_deg": raw_angle_median,
        "stabilized_angle_high_frequency_median_deg": stable_angle_median,
        "angle_high_frequency_reduction_percent": _reduction_percent(
            raw_angle_median,
            stable_angle_median,
        ),
        "angles": angle_metrics,
    }


def joint_angles_degrees(
    first: np.ndarray,
    center: np.ndarray,
    last: np.ndarray,
) -> np.ndarray:
    first_vector = np.asarray(first, dtype=float) - np.asarray(center, dtype=float)
    last_vector = np.asarray(last, dtype=float) - np.asarray(center, dtype=float)
    denominator = np.linalg.norm(first_vector, axis=-1) * np.linalg.norm(
        last_vector,
        axis=-1,
    )
    cosine = np.divide(
        np.sum(first_vector * last_vector, axis=-1),
        denominator,
        out=np.full(denominator.shape, np.nan, dtype=float),
        where=denominator > 1e-12,
    )
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def _body_scales(points: np.ndarray, valid: np.ndarray) -> np.ndarray:
    scales = np.full(points.shape[0], np.nan, dtype=float)
    for frame_idx in range(points.shape[0]):
        frame_points = points[frame_idx][valid[frame_idx]]
        if frame_points.shape[0] >= 2:
            scales[frame_idx] = np.linalg.norm(
                np.max(frame_points, axis=0) - np.min(frame_points, axis=0),
            )
    finite = scales[np.isfinite(scales) & (scales > 1e-9)]
    fallback = float(np.median(finite)) if finite.size else 1.0
    return np.where(np.isfinite(scales) & (scales > 1e-9), scales, fallback)


def _joint_high_frequency_ratio(
    points: np.ndarray,
    valid: np.ndarray,
    scales: np.ndarray,
) -> float | None:
    residuals: list[np.ndarray] = []
    for joint_idx in range(points.shape[1]):
        for start, end in _true_runs(valid[:, joint_idx]):
            segment = points[start:end, joint_idx]
            trend = _smooth_trend(segment)
            if trend is None:
                continue
            residuals.append(
                np.linalg.norm(segment - trend, axis=-1) / scales[start:end],
            )
    if not residuals:
        return None
    return float(np.median(np.concatenate(residuals)))


def _scalar_high_frequency_median(
    values: np.ndarray,
    valid: np.ndarray,
) -> float | None:
    residuals: list[np.ndarray] = []
    for start, end in _true_runs(valid):
        segment = np.asarray(values[start:end], dtype=float)
        trend = _smooth_trend(segment)
        if trend is not None:
            residuals.append(np.abs(segment - trend))
    if not residuals:
        return None
    return float(np.median(np.concatenate(residuals)))


def _smooth_trend(values: np.ndarray) -> np.ndarray | None:
    if values.shape[0] < 5:
        return None
    window = min(
        11,
        values.shape[0] if values.shape[0] % 2 else values.shape[0] - 1,
    )
    if window < 5:
        return None
    return savgol_filter(
        values,
        window_length=window,
        polyorder=min(2, window - 1),
        axis=0,
        mode="interp",
    )


def _true_runs(valid: np.ndarray) -> list[tuple[int, int]]:
    mask = np.asarray(valid, dtype=bool).reshape(-1)
    padded = np.r_[False, mask, False]
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(start), int(end)) for start, end in changes.reshape(-1, 2)]


def _reduction_percent(
    raw_value: float | None,
    stabilized_value: float | None,
) -> float | None:
    if (
        raw_value is None
        or stabilized_value is None
        or not np.isfinite(raw_value)
        or not np.isfinite(stabilized_value)
        or raw_value <= 1e-12
    ):
        return None
    return 100.0 * (raw_value - stabilized_value) / raw_value
