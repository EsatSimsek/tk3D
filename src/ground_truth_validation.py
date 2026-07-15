from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


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

BONE_SPECS: tuple[tuple[str, str], ...] = (
    ("left_shoulder", "right_shoulder"),
    ("left_hip", "right_hip"),
    ("left_shoulder", "left_elbow"),
    ("right_shoulder", "right_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_elbow", "right_wrist"),
    ("left_hip", "left_knee"),
    ("right_hip", "right_knee"),
    ("left_knee", "left_ankle"),
    ("right_knee", "right_ankle"),
)

DEFAULT_THRESHOLDS: dict[str, float] = {
    "min_evaluated_frames": 300,
    "min_valid_joint_ratio": 0.95,
    "max_mpjpe_mm": 50.0,
    "max_p95_error_mm": 100.0,
    "max_root_relative_mpjpe_mm": 50.0,
    "min_pck_100mm": 0.95,
    "max_angle_mae_deg": 5.0,
    "max_velocity_mae_mps": 0.20,
    "max_acceleration_mae_mps2": 2.0,
    "max_bone_length_cv_percent": 3.0,
}


@dataclass(slots=True)
class GroundTruthValidationResult:
    report: dict[str, Any]
    frame_rows: list[dict[str, Any]]
    joint_rows: list[dict[str, Any]]
    angle_rows: list[dict[str, Any]]


def evaluate_ground_truth_3d(
    predicted_m: np.ndarray,
    ground_truth_m: np.ndarray,
    joint_names: list[str] | tuple[str, ...],
    fps: float,
    thresholds: dict[str, float] | None = None,
    bootstrap_samples: int = 1000,
    random_seed: int = 20260715,
) -> GroundTruthValidationResult:
    """Compare synchronized 3D poses in TK3D analysis coordinates and meters.

    Global MPJPE is the primary geometry metric. Root-relative and Procrustes-
    aligned values are diagnostic and must not replace the global metric because
    their alignment can hide calibration, scale, or world-position errors.
    """
    predicted = np.asarray(predicted_m, dtype=float)
    truth = np.asarray(ground_truth_m, dtype=float)
    names = list(joint_names)
    if predicted.shape != truth.shape or predicted.ndim != 3 or predicted.shape[-1] != 3:
        raise ValueError(f"Expected matching [frames, joints, 3] arrays, got {predicted.shape} and {truth.shape}")
    if predicted.shape[1] != len(names) or len(set(names)) != len(names):
        raise ValueError("joint_names must be unique and match the joint dimension")
    if predicted.shape[0] == 0:
        raise ValueError("At least one synchronized frame is required")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive")
    if bootstrap_samples < 0:
        raise ValueError("bootstrap_samples must be non-negative")

    limits = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        limits.update({key: float(value) for key, value in thresholds.items()})
    _validate_thresholds(limits)

    valid = np.all(np.isfinite(predicted), axis=-1) & np.all(np.isfinite(truth), axis=-1)
    errors_m = np.full(valid.shape, np.nan, dtype=float)
    errors_m[valid] = np.linalg.norm(predicted[valid] - truth[valid], axis=-1)
    root_relative_errors = _root_relative_errors(predicted, truth, names, valid)
    pa_errors = _procrustes_errors(predicted, truth, valid)
    frame_rows = _frame_rows(errors_m, root_relative_errors, valid)
    joint_rows = _joint_rows(errors_m, valid, names)
    angle_rows = _angle_rows(predicted, truth, names)

    velocity_mae = _derivative_mae(predicted, truth, valid, fps=fps, order=1)
    acceleration_mae = _derivative_mae(predicted, truth, valid, fps=fps, order=2)
    bone_cv = _bone_length_cv_percent(predicted, names)
    mpjpe_ci = _bootstrap_frame_mean_ci(errors_m, bootstrap_samples, random_seed)
    angle_errors = np.asarray(
        [row["mae_deg"] for row in angle_rows if np.isfinite(row["mae_deg"])], dtype=float
    )

    summary: dict[str, Any] = {
        "coordinate_system_requirement": "tk3d_analysis: meters, x right, y forward, z up",
        "evaluation_frame_count": int(predicted.shape[0]),
        "evaluated_joint_count": int(predicted.shape[1]),
        "evaluated_point_count": int(np.sum(valid)),
        "valid_joint_ratio": float(np.mean(valid)),
        "mpjpe_mm": 1000.0 * _nanmean(errors_m),
        "median_error_mm": 1000.0 * _nanpercentile(errors_m, 50),
        "p95_error_mm": 1000.0 * _nanpercentile(errors_m, 95),
        "root_relative_mpjpe_mm": 1000.0 * _nanmean(root_relative_errors),
        "pa_mpjpe_mm": 1000.0 * _nanmean(pa_errors),
        "pck_50mm": _pck(errors_m, 0.05),
        "pck_100mm": _pck(errors_m, 0.10),
        "angle_mae_deg": _nanmean(angle_errors),
        "velocity_mae_mps": velocity_mae,
        "acceleration_mae_mps2": acceleration_mae,
        "bone_length_cv_percent": bone_cv,
        "mpjpe_95ci_mm": [1000.0 * value for value in mpjpe_ci],
        "bootstrap_samples": int(bootstrap_samples),
    }
    gates = _quality_gates(summary, limits)
    failed = [name for name, passed in gates.items() if not passed]
    summary.update(
        {
            "status": "passed_for_scoring_validation" if not failed else "failed_ground_truth_quality_gate",
            "thresholds": limits,
            "quality_gates": gates,
            "failed_gates": failed,
            "metric_interpretation": {
                "mpjpe_mm": "Primary global 3D error; includes calibration and world-position error.",
                "root_relative_mpjpe_mm": "Pelvis-centered pose error; hides global translation error.",
                "pa_mpjpe_mm": "Similarity-aligned diagnostic; hides translation, rotation, and scale error.",
            },
        }
    )
    return GroundTruthValidationResult(summary, frame_rows, joint_rows, angle_rows)


def _root_relative_errors(
    predicted: np.ndarray,
    truth: np.ndarray,
    names: list[str],
    valid: np.ndarray,
) -> np.ndarray:
    output = np.full(valid.shape, np.nan, dtype=float)
    if "left_hip" not in names or "right_hip" not in names:
        return output
    left, right = names.index("left_hip"), names.index("right_hip")
    root_valid = valid[:, left] & valid[:, right]
    if not np.any(root_valid):
        return output
    predicted_root = 0.5 * (predicted[:, left] + predicted[:, right])
    truth_root = 0.5 * (truth[:, left] + truth[:, right])
    relative_predicted = predicted - predicted_root[:, None, :]
    relative_truth = truth - truth_root[:, None, :]
    relative_valid = valid & root_valid[:, None]
    output[relative_valid] = np.linalg.norm(
        relative_predicted[relative_valid] - relative_truth[relative_valid], axis=-1
    )
    return output


def _procrustes_errors(predicted: np.ndarray, truth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    output = np.full(valid.shape, np.nan, dtype=float)
    for frame_idx in range(predicted.shape[0]):
        mask = valid[frame_idx]
        if int(np.sum(mask)) < 3:
            continue
        source = predicted[frame_idx, mask]
        target = truth[frame_idx, mask]
        source_center = np.mean(source, axis=0)
        target_center = np.mean(target, axis=0)
        source_zero = source - source_center
        target_zero = target - target_center
        denominator = float(np.sum(source_zero**2))
        if denominator <= 1e-12:
            continue
        u, singular_values, vt = np.linalg.svd(source_zero.T @ target_zero)
        rotation = u @ vt
        if np.linalg.det(rotation) < 0:
            u[:, -1] *= -1
            singular_values[-1] *= -1
            rotation = u @ vt
        scale = float(np.sum(singular_values) / denominator)
        aligned = scale * source_zero @ rotation + target_center
        output[frame_idx, mask] = np.linalg.norm(aligned - target, axis=-1)
    return output


def _frame_rows(errors: np.ndarray, root_errors: np.ndarray, valid: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for frame_idx in range(errors.shape[0]):
        rows.append(
            {
                "frame_idx": frame_idx,
                "valid_joint_ratio": float(np.mean(valid[frame_idx])),
                "mpjpe_mm": 1000.0 * _nanmean(errors[frame_idx]),
                "p95_error_mm": 1000.0 * _nanpercentile(errors[frame_idx], 95),
                "root_relative_mpjpe_mm": 1000.0 * _nanmean(root_errors[frame_idx]),
            }
        )
    return rows


def _joint_rows(errors: np.ndarray, valid: np.ndarray, names: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for joint_idx, name in enumerate(names):
        joint_errors = errors[:, joint_idx]
        rows.append(
            {
                "joint_idx": joint_idx,
                "joint_name": name,
                "valid_ratio": float(np.mean(valid[:, joint_idx])),
                "mpjpe_mm": 1000.0 * _nanmean(joint_errors),
                "median_error_mm": 1000.0 * _nanpercentile(joint_errors, 50),
                "p95_error_mm": 1000.0 * _nanpercentile(joint_errors, 95),
                "pck_50mm": _pck(joint_errors, 0.05),
                "pck_100mm": _pck(joint_errors, 0.10),
            }
        )
    return rows


def _angle_rows(predicted: np.ndarray, truth: np.ndarray, names: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    index = {name: idx for idx, name in enumerate(names)}
    for angle_name, spec in ANGLE_SPECS.items():
        if any(name not in index for name in spec):
            continue
        first, center, last = (index[name] for name in spec)
        predicted_angles = _angles(predicted[:, first], predicted[:, center], predicted[:, last])
        truth_angles = _angles(truth[:, first], truth[:, center], truth[:, last])
        differences = np.abs(predicted_angles - truth_angles)
        rows.append(
            {
                "angle_name": angle_name,
                "valid_frame_count": int(np.sum(np.isfinite(differences))),
                "mae_deg": _nanmean(differences),
                "p95_error_deg": _nanpercentile(differences, 95),
            }
        )
    return rows


def _angles(first: np.ndarray, center: np.ndarray, last: np.ndarray) -> np.ndarray:
    first_vector = first - center
    last_vector = last - center
    denominator = np.linalg.norm(first_vector, axis=-1) * np.linalg.norm(last_vector, axis=-1)
    dot = np.sum(first_vector * last_vector, axis=-1)
    cosine = np.divide(dot, denominator, out=np.full(dot.shape, np.nan), where=denominator > 1e-12)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def _derivative_mae(
    predicted: np.ndarray,
    truth: np.ndarray,
    valid: np.ndarray,
    fps: float,
    order: int,
) -> float:
    if predicted.shape[0] <= order:
        return float("nan")
    predicted_derivative = predicted.copy()
    truth_derivative = truth.copy()
    derivative_valid = valid.copy()
    for _ in range(order):
        predicted_derivative = np.diff(predicted_derivative, axis=0) * fps
        truth_derivative = np.diff(truth_derivative, axis=0) * fps
        derivative_valid = derivative_valid[1:] & derivative_valid[:-1]
    error = np.linalg.norm(predicted_derivative - truth_derivative, axis=-1)
    return _nanmean(np.where(derivative_valid, error, np.nan))


def _bone_length_cv_percent(predicted: np.ndarray, names: list[str]) -> float:
    index = {name: idx for idx, name in enumerate(names)}
    coefficients: list[float] = []
    for first_name, second_name in BONE_SPECS:
        if first_name not in index or second_name not in index:
            continue
        lengths = np.linalg.norm(predicted[:, index[first_name]] - predicted[:, index[second_name]], axis=-1)
        finite = lengths[np.isfinite(lengths) & (lengths > 1e-9)]
        if finite.size >= 2:
            coefficients.append(100.0 * float(np.std(finite) / np.mean(finite)))
    return float(np.median(coefficients)) if coefficients else float("nan")


def _bootstrap_frame_mean_ci(errors: np.ndarray, samples: int, seed: int) -> tuple[float, float]:
    frame_means = np.asarray([_nanmean(row) for row in errors], dtype=float)
    frame_means = frame_means[np.isfinite(frame_means)]
    if frame_means.size == 0:
        return float("nan"), float("nan")
    if samples == 0 or frame_means.size == 1:
        mean = float(np.mean(frame_means))
        return mean, mean
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=float)
    for sample_idx in range(samples):
        means[sample_idx] = float(np.mean(rng.choice(frame_means, size=frame_means.size, replace=True)))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _quality_gates(summary: dict[str, Any], limits: dict[str, float]) -> dict[str, bool]:
    return {
        "enough_frames": summary["evaluation_frame_count"] >= limits["min_evaluated_frames"],
        "valid_joint_ratio": summary["valid_joint_ratio"] >= limits["min_valid_joint_ratio"],
        "mpjpe": _at_most(summary["mpjpe_mm"], limits["max_mpjpe_mm"]),
        "p95_error": _at_most(summary["p95_error_mm"], limits["max_p95_error_mm"]),
        "root_relative_mpjpe": _at_most(
            summary["root_relative_mpjpe_mm"], limits["max_root_relative_mpjpe_mm"]
        ),
        "pck_100mm": _at_least(summary["pck_100mm"], limits["min_pck_100mm"]),
        "angle_mae": _at_most(summary["angle_mae_deg"], limits["max_angle_mae_deg"]),
        "velocity_mae": _at_most(summary["velocity_mae_mps"], limits["max_velocity_mae_mps"]),
        "acceleration_mae": _at_most(
            summary["acceleration_mae_mps2"], limits["max_acceleration_mae_mps2"]
        ),
        "bone_length_stability": _at_most(
            summary["bone_length_cv_percent"], limits["max_bone_length_cv_percent"]
        ),
    }


def _validate_thresholds(limits: dict[str, float]) -> None:
    if limits["min_evaluated_frames"] < 1:
        raise ValueError("min_evaluated_frames must be at least 1")
    for key in ("min_valid_joint_ratio", "min_pck_100mm"):
        if not 0.0 <= limits[key] <= 1.0:
            raise ValueError(f"{key} must be between 0 and 1")
    for key, value in limits.items():
        if key not in {"min_valid_joint_ratio", "min_pck_100mm"} and value < 0:
            raise ValueError(f"{key} must be non-negative")


def _pck(errors_m: np.ndarray, threshold_m: float) -> float:
    finite = np.asarray(errors_m, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.mean(finite <= threshold_m)) if finite.size else float("nan")


def _nanmean(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.mean(finite)) if finite.size else float("nan")


def _nanpercentile(values: np.ndarray, percentile: float) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.percentile(finite, percentile)) if finite.size else float("nan")


def _at_most(value: Any, maximum: float) -> bool:
    return bool(np.isfinite(value) and float(value) <= maximum)


def _at_least(value: Any, minimum: float) -> bool:
    return bool(np.isfinite(value) and float(value) >= minimum)
