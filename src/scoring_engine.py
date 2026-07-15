from __future__ import annotations

from typing import Any

import numpy as np

from .biomechanics_3d import center_of_mass_proxy
from .data_structures import COCO_BODY_JOINTS


def build_provisional_score(
    keypoints_3d: np.ndarray,
    biomechanics_rows: list[dict[str, Any]],
    frame_quality_rows: list[dict[str, Any]],
    segment_rows: list[dict[str, Any]],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    """Build an explainable technical baseline, never an official poomsae score."""
    points = np.asarray(keypoints_3d, dtype=float)
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError(f"Expected keypoints [frames, joints, 3], got {points.shape}")
    if len(biomechanics_rows) != points.shape[0] or len(frame_quality_rows) != points.shape[0]:
        raise ValueError("Biomechanics and frame-quality rows must align with keypoint frames")

    lean_limit = max(float(thresholds.get("trunk_lean_warn_deg", 10.0)), 1e-6)
    knee_min = float(thresholds.get("knee_angle_front_stance_min_deg", 130.0))
    balance_min = float(thresholds.get("balance_min_score", 0.70))
    bone_stability = _bone_length_stability(points)
    frame_scores: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for frame_idx, (biomechanics, quality) in enumerate(zip(biomechanics_rows, frame_quality_rows, strict=True)):
        reliable = bool(quality.get("ready_for_scoring", False))
        lean = _finite_float(biomechanics.get("torso_lean_deg"))
        knees = [
            value for value in (
                _finite_float(biomechanics.get("left_knee_deg")),
                _finite_float(biomechanics.get("right_knee_deg")),
            )
            if value is not None
        ]
        posture = max(0.0, 1.0 - abs(lean) / lean_limit) if lean is not None else 0.0
        knee_score = float(np.mean([min(max(value / max(knee_min, 1e-6), 0.0), 1.0) for value in knees])) if knees else 0.0
        balance = _balance_proxy(points[frame_idx])
        components = {
            "posture": posture,
            "lower_body": knee_score,
            "balance": balance,
            "measurement_stability": bone_stability,
        }
        technical = 100.0 * (
            0.30 * posture + 0.30 * knee_score + 0.25 * balance + 0.15 * bone_stability
        )
        score = technical if reliable else 0.0
        frame_scores.append(
            {
                "frame_idx": frame_idx,
                "reliable": reliable,
                "score": score,
                **{f"{name}_score": value for name, value in components.items()},
            }
        )
        if not reliable:
            errors.append(_error(frame_idx, "unreliable_3d_frame", "quality", "3D quality gates did not pass"))
        if lean is not None and abs(lean) > lean_limit:
            errors.append(_error(frame_idx, "excessive_torso_lean", "posture", f"|torso lean| {abs(lean):.1f}° > {lean_limit:.1f}°"))
        if knees and min(knees) < knee_min:
            errors.append(_error(frame_idx, "knee_extension_below_baseline", "lower_body", f"minimum knee angle {min(knees):.1f}° < {knee_min:.1f}°"))
        if balance < balance_min:
            errors.append(_error(frame_idx, "balance_proxy_below_baseline", "balance", f"balance proxy {balance:.2f} < {balance_min:.2f}"))

    step_scores = _score_segments(frame_scores, segment_rows)
    reliable_scores = [row["score"] for row in frame_scores if row["reliable"]]
    return {
        "status": "provisional_not_official",
        "score_name": "TK3D provisional technical baseline",
        "overall_score": float(np.mean(reliable_scores)) if reliable_scores else None,
        "reliable_frame_ratio": float(np.mean([row["reliable"] for row in frame_scores])) if frame_scores else 0.0,
        "component_weights": {"posture": 0.30, "lower_body": 0.30, "balance": 0.25, "measurement_stability": 0.15},
        "frame_scores": frame_scores,
        "step_scores": step_scores,
        "errors": errors,
        "limitations": [
            "This is not an official World Taekwondo or referee score.",
            "Named poomsae phases and technique-specific targets require an approved reference template.",
            "The balance metric is a kinematic center-of-mass/support proxy; force-plate kinetics are not available.",
        ],
    }


def _balance_proxy(frame: np.ndarray) -> float:
    idx = COCO_BODY_JOINTS
    required = [idx["left_ankle"], idx["right_ankle"], idx["left_shoulder"], idx["right_shoulder"]]
    if not np.all(np.isfinite(frame[required])):
        return 0.0
    com = center_of_mass_proxy(frame)
    if not np.all(np.isfinite(com)):
        return 0.0
    ankles = frame[[idx["left_ankle"], idx["right_ankle"]], :2]
    segment = ankles[1] - ankles[0]
    length_sq = float(np.dot(segment, segment))
    if length_sq <= 1e-12:
        return 0.0
    projection = float(np.clip(np.dot(com[:2] - ankles[0], segment) / length_sq, 0.0, 1.0))
    nearest = ankles[0] + projection * segment
    distance = float(np.linalg.norm(com[:2] - nearest))
    shoulder_width = float(np.linalg.norm(frame[idx["left_shoulder"]] - frame[idx["right_shoulder"]]))
    return float(np.exp(-distance / max(shoulder_width, 1e-6)))


def _bone_length_stability(points: np.ndarray) -> float:
    idx = COCO_BODY_JOINTS
    pairs = [
        (idx["left_shoulder"], idx["left_elbow"]),
        (idx["right_shoulder"], idx["right_elbow"]),
        (idx["left_elbow"], idx["left_wrist"]),
        (idx["right_elbow"], idx["right_wrist"]),
        (idx["left_hip"], idx["left_knee"]),
        (idx["right_hip"], idx["right_knee"]),
        (idx["left_knee"], idx["left_ankle"]),
        (idx["right_knee"], idx["right_ankle"]),
    ]
    coefficients: list[float] = []
    for first, second in pairs:
        lengths = np.linalg.norm(points[:, first] - points[:, second], axis=1)
        finite = lengths[np.isfinite(lengths) & (lengths > 1e-6)]
        if finite.size >= 2:
            coefficients.append(float(np.std(finite) / max(np.mean(finite), 1e-6)))
    return float(np.exp(-10.0 * np.median(coefficients))) if coefficients else 0.0


def _score_segments(frame_scores: list[dict[str, Any]], segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, segment in enumerate(segments):
        start = max(int(segment.get("start_frame", 0)), 0)
        end = min(int(segment.get("end_frame", start)), len(frame_scores) - 1)
        selected = [row for row in frame_scores[start : end + 1] if row["reliable"]] if end >= start else []
        rows.append(
            {
                "step_id": index,
                "label": segment.get("label", "motion_candidate"),
                "start_frame": start,
                "end_frame": end,
                "score": float(np.mean([row["score"] for row in selected])) if selected else None,
                "status": "needs_reference_label" if segment.get("label") != "pending_motion" else segment.get("status"),
            }
        )
    return rows


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _error(frame_idx: int, code: str, category: str, description: str) -> dict[str, Any]:
    return {"frame_idx": frame_idx, "code": code, "category": category, "description": description}
