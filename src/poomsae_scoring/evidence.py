from __future__ import annotations

from typing import Any

import numpy as np

from src.biomechanics_3d import angle_deg
from src.coordinate_system import ANALYSIS_COORDINATE_SYSTEM
from src.data_structures import COCO_BODY_JOINTS
from src.poomsae_scoring.contracts import (
    ScoringContractError,
    validate_movement_timeline,
    validate_poomsae_spec,
)
from src.scoring_metrics import stance_length_forward, stance_width_lateral, torso_lean_from_vertical_deg


ANGLE_SPECS: dict[str, tuple[str, str, str]] = {
    "left_elbow_deg": ("left_shoulder", "left_elbow", "left_wrist"),
    "right_elbow_deg": ("right_shoulder", "right_elbow", "right_wrist"),
    "left_knee_deg": ("left_hip", "left_knee", "left_ankle"),
    "right_knee_deg": ("right_hip", "right_knee", "right_ankle"),
}


def build_movement_evidence(
    pose_payload: dict[str, Any],
    poomsae_spec: dict[str, Any],
    movement_timeline: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Measure source-bound movement anchors without generating deductions."""
    spec = validate_poomsae_spec(poomsae_spec)
    timeline = validate_movement_timeline(movement_timeline, spec)
    arrays = _pose_arrays(pose_payload, timeline)
    movements = {movement["movement_id"]: movement for movement in spec["movements"]}
    rows: list[dict[str, Any]] = []
    segment_reports: list[dict[str, Any]] = []

    for segment in timeline["segments"]:
        movement = movements[segment["movement_id"]]
        start = segment["start_frame"]
        end = segment["end_frame"]
        frame_slice = slice(start, end + 1)
        segment_valid_ratio = _finite_mean(arrays["valid_mask"][frame_slice, :17])
        segment_report = {
            "movement_id": movement["movement_id"],
            "sequence_index": movement["sequence_index"],
            "display_name": movement["display_name"],
            "start_frame": start,
            "end_frame": end,
            "duration_sec": (end - start + 1) / timeline["fps"],
            "label_status": segment["label_status"],
            "label_confidence": segment["confidence"],
            "body17_valid_ratio": segment_valid_ratio,
            "median_reprojection_error_px": _finite_median(arrays["reprojection"][frame_slice, :17]),
            "median_triangulation_score": _finite_median(arrays["triangulation"][frame_slice, :17]),
            "median_used_cameras": _finite_median(arrays["used_cameras"][frame_slice, :17]),
            "anchor_count": len(segment["anchors"]),
        }
        segment_reports.append(segment_report)
        for phase_id, frame_index in segment["anchors"].items():
            rows.append(
                _anchor_row(
                    arrays,
                    movement_id=movement["movement_id"],
                    sequence_index=movement["sequence_index"],
                    phase_id=phase_id,
                    frame_index=frame_index,
                    fps=timeline["fps"],
                    label_status=segment["label_status"],
                )
            )

    observed_anchor_count = sum(row["evidence_status"] == "observed" for row in rows)
    partial_anchor_count = sum(row["evidence_status"] == "partially_observed" for row in rows)
    missing_ids = list(timeline["coverage"]["missing_movement_ids"])
    is_partial = timeline["coverage"]["recording_scope"] == "partial_sequence"
    report = {
        "schema_version": 1,
        "status": "measurement_evidence_only",
        "scoring_status": "not_scored_partial_recording" if is_partial else "not_scored_source_thresholds_inactive",
        "accuracy_score": None,
        "deductions": [],
        "judge_calibrated": False,
        "official_scoring_ready": False,
        "poomsae": {"poomsae_id": spec["poomsae_id"], "version": spec["version"]},
        "timeline": {
            "timeline_id": timeline["timeline_id"],
            "frame_count": timeline["frame_count"],
            "fps": timeline["fps"],
            "labeled_movement_count": len(timeline["segments"]),
            "expected_movement_count": len(spec["movements"]),
            "recording_scope": timeline["coverage"]["recording_scope"],
            "missing_movement_ids": missing_ids,
            "source_end_reason": timeline["coverage"]["source_end_reason"],
        },
        "source_pose": {
            "session_id": pose_payload.get("session_id"),
            "run_id": pose_payload.get("run_id"),
            "coordinate_system": pose_payload["coordinate_system"],
            "shape": list(arrays["keypoints"].shape),
        },
        "observability": {
            "anchor_count": len(rows),
            "observed_anchor_count": observed_anchor_count,
            "partially_observed_anchor_count": partial_anchor_count,
            "not_measurable_anchor_count": len(rows) - observed_anchor_count - partial_anchor_count,
            "observed_anchor_ratio": observed_anchor_count / len(rows) if rows else 0.0,
            "policy": {
                "observed": "BODY-17 valid ratio >= 0.90 and median used cameras >= 2",
                "partially_observed": "BODY-17 valid ratio >= 0.70",
                "effect_on_score": "None; this report cannot create deductions.",
            },
        },
        "segments": segment_reports,
        "interpretation": (
            "Measurements are replayable evidence for the labeled part of the recording. "
            + (
                "The recording does not contain the complete Poomsae, and "
                if is_partial
                else "The recording coverage is complete, but "
            )
            + "no source-approved numeric error thresholds are active, so no Accuracy score or deduction is produced."
        ),
    }
    return _json_safe(report), [_json_safe(row) for row in rows]


def _pose_arrays(pose_payload: dict[str, Any], timeline: dict[str, Any]) -> dict[str, np.ndarray]:
    if not isinstance(pose_payload, dict):
        raise ScoringContractError("pose payload must be a mapping")
    if pose_payload.get("coordinate_system") != ANALYSIS_COORDINATE_SYSTEM:
        raise ScoringContractError("pose coordinate system must be canonical TK3D x=right, y=forward, z=up in meters")
    keypoints = np.asarray(pose_payload.get("keypoints_3d_world"), dtype=float)
    expected_shape = (timeline["frame_count"], 133, 3)
    if keypoints.shape != expected_shape:
        raise ScoringContractError(f"pose keypoints_3d_world must have shape {expected_shape}, got {keypoints.shape}")
    arrays = {
        "keypoints": keypoints,
        "valid_mask": np.asarray(pose_payload.get("reliability_valid_mask"), dtype=bool),
        "reprojection": np.asarray(pose_payload.get("reprojection_error"), dtype=float),
        "triangulation": np.asarray(pose_payload.get("triangulation_score"), dtype=float),
        "used_cameras": np.asarray(pose_payload.get("used_cameras"), dtype=float),
    }
    expected_matrix_shape = expected_shape[:2]
    for name, values in arrays.items():
        if name == "keypoints":
            continue
        if values.shape != expected_matrix_shape:
            raise ScoringContractError(f"pose {name} must have shape {expected_matrix_shape}, got {values.shape}")
    finite_xyz = np.all(np.isfinite(keypoints), axis=-1)
    arrays["valid_mask"] &= finite_xyz
    return arrays


def _anchor_row(
    arrays: dict[str, np.ndarray],
    *,
    movement_id: str,
    sequence_index: int,
    phase_id: str,
    frame_index: int,
    fps: float,
    label_status: str,
) -> dict[str, Any]:
    frame = arrays["keypoints"][frame_index, :17].copy()
    valid = arrays["valid_mask"][frame_index, :17]
    frame[~valid] = np.nan
    valid_ratio = float(np.mean(valid))
    median_cameras = _finite_median(arrays["used_cameras"][frame_index, :17])
    if valid_ratio >= 0.90 and median_cameras is not None and median_cameras >= 2.0:
        evidence_status = "observed"
    elif valid_ratio >= 0.70:
        evidence_status = "partially_observed"
    else:
        evidence_status = "not_measurable"
    row: dict[str, Any] = {
        "movement_id": movement_id,
        "sequence_index": sequence_index,
        "phase_id": phase_id,
        "frame_index": frame_index,
        "timestamp_sec": frame_index / fps,
        "label_status": label_status,
        "evidence_status": evidence_status,
        "valid_body17_ratio": valid_ratio,
        "median_reprojection_error_px": _finite_median(arrays["reprojection"][frame_index, :17]),
        "median_triangulation_score": _finite_median(arrays["triangulation"][frame_index, :17]),
        "median_used_cameras": median_cameras,
        "torso_lean_deg": torso_lean_from_vertical_deg(frame),
        "stance_width_m": stance_width_lateral(frame),
        "stance_length_m": stance_length_forward(frame),
        "body_scale_m": _body_scale_m(frame),
        "stance_span_ratio": _stance_span_ratio(frame),
        "left_wrist_height_torso_ratio": _wrist_height_torso_ratio(frame, "left_wrist"),
        "right_wrist_height_torso_ratio": _wrist_height_torso_ratio(frame, "right_wrist"),
    }
    for metric_id, (first, center, last) in ANGLE_SPECS.items():
        row[metric_id] = angle_deg(
            frame[COCO_BODY_JOINTS[first]],
            frame[COCO_BODY_JOINTS[center]],
            frame[COCO_BODY_JOINTS[last]],
        )
    return row


def _body_scale_m(frame: np.ndarray) -> float | None:
    leg_lengths = []
    for side in ("left", "right"):
        hip = frame[COCO_BODY_JOINTS[f"{side}_hip"]]
        knee = frame[COCO_BODY_JOINTS[f"{side}_knee"]]
        ankle = frame[COCO_BODY_JOINTS[f"{side}_ankle"]]
        if np.all(np.isfinite([hip, knee, ankle])):
            leg_lengths.append(float(np.linalg.norm(hip - knee) + np.linalg.norm(knee - ankle)))
    return float(np.mean(leg_lengths)) if leg_lengths else None


def _stance_span_ratio(frame: np.ndarray) -> float | None:
    left = frame[COCO_BODY_JOINTS["left_ankle"]]
    right = frame[COCO_BODY_JOINTS["right_ankle"]]
    scale = _body_scale_m(frame)
    if scale is None or scale <= 0.0 or not np.all(np.isfinite([left, right])):
        return None
    return float(np.linalg.norm(left - right) / scale)


def _wrist_height_torso_ratio(frame: np.ndarray, wrist_name: str) -> float | None:
    left_shoulder = frame[COCO_BODY_JOINTS["left_shoulder"]]
    right_shoulder = frame[COCO_BODY_JOINTS["right_shoulder"]]
    left_hip = frame[COCO_BODY_JOINTS["left_hip"]]
    right_hip = frame[COCO_BODY_JOINTS["right_hip"]]
    wrist = frame[COCO_BODY_JOINTS[wrist_name]]
    if not np.all(np.isfinite([left_shoulder, right_shoulder, left_hip, right_hip, wrist])):
        return None
    shoulder_mid = (left_shoulder + right_shoulder) / 2.0
    hip_mid = (left_hip + right_hip) / 2.0
    torso_height = float(shoulder_mid[2] - hip_mid[2])
    if torso_height <= 1e-6:
        return None
    return float((wrist[2] - hip_mid[2]) / torso_height)


def _finite_mean(values: np.ndarray) -> float | None:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.mean(finite)) if finite.size else None


def _finite_median(values: np.ndarray) -> float | None:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.median(finite)) if finite.size else None


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value
