from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.data_structures import (
    COCO_BODY_JOINTS,
    COCO_FACE_INDICES,
    COCO_FOOT_JOINTS,
    COCO_LEFT_HAND_INDICES,
    COCO_RIGHT_HAND_INDICES,
    COCO_WHOLEBODY_KEYPOINTS,
    coco_hand_joint,
)
from src.poomsae_scoring.contracts import (
    ScoringContractError,
    _UniqueKeyLoader,
    validate_movement_timeline,
    validate_poomsae_spec,
)


GROUPS: dict[str, tuple[int, ...]] = {
    "body17": tuple(range(17)),
    "feet": tuple(range(17, 23)),
    "face": COCO_FACE_INDICES,
    "left_hand": COCO_LEFT_HAND_INDICES,
    "right_hand": COCO_RIGHT_HAND_INDICES,
}

THRESHOLD_KEYS = {
    "torso_lean_p95_deg_max",
    "shoulder_hip_twist_deg_max",
    "head_torso_yaw_mismatch_deg_max",
    "expected_side_dominance_ratio_min",
    "hikite_hip_distance_body_scale_max",
    "wrist_forearm_alignment_deg_max",
    "fist_closure_ratio_max",
    "simultaneity_sec_max",
    "fixation_wrist_jitter_body_scale_max",
    "weight_transfer_body_scale_min",
    "path_efficiency_punch_max",
    "path_efficiency_block_max",
    "ap_seogi_span_ratio_min",
    "ap_seogi_span_ratio_max",
    "ap_seogi_front_knee_deg_min",
    "ap_seogi_front_knee_deg_max",
    "ap_gubi_span_ratio_min",
    "ap_gubi_span_ratio_max",
    "ap_gubi_front_knee_deg_min",
    "ap_gubi_front_knee_deg_max",
    "punch_wrist_height_torso_ratio_min",
    "punch_wrist_height_torso_ratio_max",
    "block_wrist_height_torso_ratio_min",
    "block_wrist_height_torso_ratio_max",
    "punch_elbow_deg_min",
    "punch_elbow_deg_max",
    "block_elbow_deg_min",
    "block_elbow_deg_max",
}


def load_wholebody_diagnostic_profile(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise ScoringContractError(f"WholeBody diagnostic profile not found: {source}")
    try:
        payload = yaml.load(source.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except yaml.YAMLError as exc:
        raise ScoringContractError(f"invalid WholeBody diagnostic YAML: {exc}") from exc
    return validate_wholebody_diagnostic_profile(payload)


def validate_wholebody_diagnostic_profile(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ScoringContractError("WholeBody diagnostic profile must be a mapping")
    data = deepcopy(payload)
    expected = {
        "schema_version",
        "profile_id",
        "version",
        "status",
        "poomsae_id",
        "required_keypoint_count",
        "score_policy",
        "quality_gates",
        "thresholds",
        "disclaimer",
    }
    _exact_keys(data, expected, "WholeBody diagnostic profile")
    if data["schema_version"] != 1 or data["status"] != "diagnostic_only_unvalidated":
        raise ScoringContractError("WholeBody diagnostic profile must be schema 1 diagnostic_only_unvalidated")
    for key in ("profile_id", "version", "poomsae_id", "disclaimer"):
        _nonempty(data[key], key)
    if data["required_keypoint_count"] != COCO_WHOLEBODY_KEYPOINTS:
        raise ScoringContractError("WholeBody diagnostics require exactly 133 keypoints")
    policy = data["score_policy"]
    _exact_keys(
        policy,
        {"numeric_score_enabled", "insufficient_coverage_blocks_score", "minimum_metric_coverage"},
        "WholeBody score_policy",
    )
    if policy["numeric_score_enabled"] is not False or policy["insufficient_coverage_blocks_score"] is not True:
        raise ScoringContractError("WholeBody diagnostics must remain fail-closed with numeric scoring disabled")
    policy["minimum_metric_coverage"] = _probability(
        policy["minimum_metric_coverage"], "minimum_metric_coverage"
    )
    gates = data["quality_gates"]
    _exact_keys(
        gates,
        {
            "min_used_cameras",
            "max_reprojection_error_px",
            "min_segment_group_valid_ratio",
            "anchor_window_radius_frames",
            "min_anchor_window_valid_samples",
            "face_eye_separation_body_scale_min",
            "face_eye_separation_body_scale_max",
            "hand_palm_length_body_scale_min",
            "hand_palm_length_body_scale_max",
            "foot_length_body_scale_min",
            "foot_length_body_scale_max",
        },
        "WholeBody quality_gates",
    )
    gates["min_used_cameras"] = _positive(gates["min_used_cameras"], "min_used_cameras")
    gates["max_reprojection_error_px"] = _positive(
        gates["max_reprojection_error_px"], "max_reprojection_error_px"
    )
    gates["min_segment_group_valid_ratio"] = _probability(
        gates["min_segment_group_valid_ratio"], "min_segment_group_valid_ratio"
    )
    gates["anchor_window_radius_frames"] = _nonnegative_integer(
        gates["anchor_window_radius_frames"], "anchor_window_radius_frames"
    )
    gates["min_anchor_window_valid_samples"] = _positive_integer(
        gates["min_anchor_window_valid_samples"], "min_anchor_window_valid_samples"
    )
    window_size = 2 * gates["anchor_window_radius_frames"] + 1
    if gates["min_anchor_window_valid_samples"] > window_size:
        raise ScoringContractError("min_anchor_window_valid_samples cannot exceed the anchor window size")
    plausibility_prefixes = (
        "face_eye_separation_body_scale",
        "hand_palm_length_body_scale",
        "foot_length_body_scale",
    )
    for prefix in plausibility_prefixes:
        gates[f"{prefix}_min"] = _positive(gates[f"{prefix}_min"], f"{prefix}_min")
        gates[f"{prefix}_max"] = _positive(gates[f"{prefix}_max"], f"{prefix}_max")
        if gates[f"{prefix}_min"] >= gates[f"{prefix}_max"]:
            raise ScoringContractError(f"{prefix}_min must be lower than {prefix}_max")
    if not isinstance(data["thresholds"], dict):
        raise ScoringContractError("WholeBody thresholds must be a mapping")
    _exact_keys(data["thresholds"], THRESHOLD_KEYS, "WholeBody thresholds")
    data["thresholds"] = {
        key: _nonnegative(value, f"thresholds.{key}") for key, value in data["thresholds"].items()
    }
    range_prefixes = (
        "ap_seogi_span_ratio",
        "ap_seogi_front_knee_deg",
        "ap_gubi_span_ratio",
        "ap_gubi_front_knee_deg",
        "punch_wrist_height_torso_ratio",
        "block_wrist_height_torso_ratio",
        "punch_elbow_deg",
        "block_elbow_deg",
    )
    for prefix in range_prefixes:
        if data["thresholds"][f"{prefix}_min"] > data["thresholds"][f"{prefix}_max"]:
            raise ScoringContractError(f"thresholds.{prefix}_min cannot exceed {prefix}_max")
    return data


def build_wholebody_diagnostics(
    pose_payload: dict[str, Any],
    poomsae_spec: dict[str, Any],
    movement_timeline: dict[str, Any],
    diagnostic_profile: dict[str, Any],
) -> dict[str, Any]:
    spec = validate_poomsae_spec(poomsae_spec)
    timeline = validate_movement_timeline(movement_timeline, spec)
    profile = validate_wholebody_diagnostic_profile(diagnostic_profile)
    if profile["poomsae_id"] != spec["poomsae_id"]:
        raise ScoringContractError("WholeBody profile Poomsae id does not match")
    arrays = _pose_arrays(pose_payload, timeline, profile["quality_gates"])
    movements = {movement["movement_id"]: movement for movement in spec["movements"]}
    movement_reports = [
        _movement_diagnostics(movements[segment["movement_id"]], segment, arrays, timeline["fps"], profile)
        for segment in timeline["segments"]
    ]
    metrics = [metric for movement in movement_reports for metric in movement["metrics"]]
    thresholded = [metric for metric in metrics if metric["screening_status"] != "measured_diagnostic_only"]
    measurable = [metric for metric in thresholded if metric["screening_status"] != "not_measurable"]
    candidates = [metric for metric in thresholded if metric["screening_status"] == "review_candidate"]
    not_measurable = [metric for metric in thresholded if metric["screening_status"] == "not_measurable"]
    coverage = len(measurable) / len(thresholded) if thresholded else 0.0
    coverage_gate = coverage >= profile["score_policy"]["minimum_metric_coverage"]
    return _json_safe(
        {
            "schema_version": 1,
            "status": "wholebody_diagnostics_only",
            "scoring_status": "not_scored_diagnostic_candidates_only",
            "accuracy_score": None,
            "partial_engineering_trial_score": None,
            "deductions": [],
            "numeric_score_enabled": False,
            "keypoint_contract": {
                "name": "COCO-WholeBody",
                "keypoint_count": COCO_WHOLEBODY_KEYPOINTS,
                "groups_used": list(GROUPS),
                "layout": {
                    "body17": "0:17",
                    "feet": "17:23",
                    "face": "23:91 (68-point face layout)",
                    "left_hand": "91:112 (21-point hand layout)",
                    "right_hand": "112:133 (21-point hand layout)",
                },
            },
            "profile": {
                "profile_id": profile["profile_id"],
                "version": profile["version"],
                "status": profile["status"],
                "disclaimer": profile["disclaimer"],
            },
            "poomsae": {"poomsae_id": spec["poomsae_id"], "version": spec["version"]},
            "movement_timeline_id": timeline["timeline_id"],
            "coverage": {
                "thresholded_metric_count": len(thresholded),
                "measurable_metric_count": len(measurable),
                "not_measurable_metric_count": len(thresholded) - len(measurable),
                "measurement_coverage_ratio": coverage,
                "minimum_required_ratio": profile["score_policy"]["minimum_metric_coverage"],
                "coverage_gate_passed": coverage_gate,
            },
            "summary": {
                "movement_count": len(movement_reports),
                "metric_count": len(metrics),
                "review_candidate_count": len(candidates),
                "within_screening_range_count": sum(
                    metric["screening_status"] == "within_screening_range" for metric in thresholded
                ),
                "not_measurable_count": len(thresholded) - len(measurable),
                "candidate_family_counts": dict(Counter(metric["family"] for metric in candidates)),
                "not_measurable_reason_counts": dict(
                    Counter(
                        metric["measurement_evidence"]["not_measurable_reason"]
                        for metric in not_measurable
                    )
                ),
                "movement_group_quality_gate_failure_count": sum(
                    len(movement["group_quality_gate_failures"]) for movement in movement_reports
                ),
            },
            "candidate_events": [
                {
                    "event_id": f"WB-{metric['movement_id']}-{metric['metric_id']}",
                    "movement_id": metric["movement_id"],
                    "family": metric["family"],
                    "metric_id": metric["metric_id"],
                    "criterion_id": metric["criterion_id"],
                    "value": metric["value"],
                    "screening_rule": metric["screening_rule"],
                    "threshold_margin": metric["threshold_margin"],
                    "normalized_threshold_margin": metric["normalized_threshold_margin"],
                    "measurement_evidence": metric["measurement_evidence"],
                    "decision_status": "review_candidate_not_deduction",
                    "human_review_status": "pending",
                    "rule_eligibility": "blocked_unvalidated_screening_threshold",
                    "score_effect": None,
                }
                for metric in candidates
            ],
            "movements": movement_reports,
            "interpretation": (
                "Every candidate requires video review. Within-screening-range is not a correctness label. "
                "WholeBody diagnostics cannot create a numeric Accuracy score or WT deduction."
            ),
        }
    )


def _movement_diagnostics(
    movement: dict[str, Any],
    segment: dict[str, Any],
    arrays: dict[str, np.ndarray],
    fps: float,
    profile: dict[str, Any],
) -> dict[str, Any]:
    start, end = segment["start_frame"], segment["end_frame"]
    frames = np.arange(start, end + 1)
    group_coverage = {
        name: float(np.mean(arrays["quality_mask"][start : end + 1, list(indices)]))
        for name, indices in GROUPS.items()
    }
    minimum_group_coverage = profile["quality_gates"]["min_segment_group_valid_ratio"]
    group_gate_failures = [
        name for name, ratio in group_coverage.items() if ratio < minimum_group_coverage
    ]
    executing_side, technique_id = _primary_technique(movement)
    other_side = "right" if executing_side == "left" else "left"
    prep = segment["anchors"].get("preparation", start)
    execution = _execution_anchor(segment)
    fixation = segment["anchors"].get("fixation", end)
    scale = _median_body_scale(arrays, frames)
    metrics: list[dict[str, Any]] = []
    t = profile["thresholds"]
    radius = profile["quality_gates"]["anchor_window_radius_frames"]
    minimum_samples = profile["quality_gates"]["min_anchor_window_valid_samples"]
    fixation_frames = np.arange(max(start, fixation - radius), min(end, fixation + radius) + 1)

    def fixation_metric(function: Any, *args: Any) -> float | None:
        return _window_median(
            function,
            arrays,
            fixation,
            start,
            end,
            radius,
            minimum_samples,
            *args,
        )

    def fixation_estimate(function: Any, *args: Any) -> tuple[float | None, float | None, int]:
        return _window_estimate(
            function,
            arrays,
            fixation,
            start,
            end,
            radius,
            minimum_samples,
            *args,
        )

    metrics.append(_upper_metric(movement, "balance", "torso_lean_p95_deg", _torso_lean_p95(arrays, fixation_frames), t["torso_lean_p95_deg_max"], "deg"))
    back_foot_yaw = fixation_estimate(
        _back_foot_yaw_to_stance_direction,
        movement["stance"],
        scale,
        profile["quality_gates"],
    )
    metrics.append(
        _with_uncertainty(
            _diagnostic_metric(
                movement,
                "stance",
                "back_foot_yaw_to_stance_direction_deg",
                back_foot_yaw[0],
                "deg",
            ),
            uncertainty_95=back_foot_yaw[1],
            sample_count=back_foot_yaw[2],
            direction_basis="back_ankle_to_front_ankle",
        )
    )
    metrics.append(_upper_metric(movement, "rotation", "shoulder_hip_twist_deg", fixation_metric(_shoulder_hip_twist), t["shoulder_hip_twist_deg_max"], "deg"))
    metrics.append(_upper_metric(movement, "gaze", "head_torso_yaw_mismatch_deg", fixation_metric(_head_torso_yaw_mismatch, scale, profile["quality_gates"]), t["head_torso_yaw_mismatch_deg_max"], "deg"))
    dominance = _expected_side_dominance(
        arrays,
        prep,
        execution,
        executing_side,
        other_side,
        start,
        end,
        radius,
        minimum_samples,
    )
    metrics.append(_lower_metric(movement, "wrong_side", "expected_side_dominance_ratio", dominance, t["expected_side_dominance_ratio_min"], "ratio"))
    hikite = fixation_metric(_hikite_distance_ratio, other_side, scale)
    metrics.append(_upper_metric(movement, "hikite", "reaction_hand_hip_distance_ratio", hikite, t["hikite_hip_distance_body_scale_max"], "body_scale"))
    alignment = fixation_metric(_wrist_forearm_alignment, executing_side, scale, profile["quality_gates"])
    metrics.append(_upper_metric(movement, "hand", "wrist_forearm_alignment_deg", alignment, t["wrist_forearm_alignment_deg_max"], "deg"))
    closure = fixation_metric(_fist_closure_ratio, executing_side, scale, profile["quality_gates"])
    metrics.append(_upper_metric(movement, "hand", "fist_closure_ratio", closure, t["fist_closure_ratio_max"], "ratio"))
    simultaneity = _settle_time_difference(arrays, prep, fixation, executing_side, _stance_side(movement["stance"]), fps)
    metrics.append(_upper_metric(movement, "timing", "hand_foot_settle_difference_sec", simultaneity, t["simultaneity_sec_max"], "sec"))
    jitter = _fixation_jitter_ratio(arrays, fixation, end, executing_side, scale)
    metrics.append(_upper_metric(movement, "fixation", "fixation_wrist_jitter_ratio", jitter, t["fixation_wrist_jitter_body_scale_max"], "body_scale"))
    weight_transfer = _weight_transfer_ratio(
        arrays,
        prep,
        fixation,
        movement["stance"],
        scale,
        start,
        end,
        radius,
        minimum_samples,
    )
    metrics.append(_lower_metric(movement, "weight_transfer", "pelvis_weight_transfer_ratio", weight_transfer, t["weight_transfer_body_scale_min"], "body_scale"))
    path_efficiency = _path_efficiency(arrays, prep, execution, executing_side)
    path_max = t["path_efficiency_punch_max"] if technique_id == "momtong_jireugi" else t["path_efficiency_block_max"]
    metrics.append(_upper_metric(movement, "trajectory", "executing_wrist_path_efficiency", path_efficiency, path_max, "ratio"))
    peak_speed = _peak_speed_normalized(arrays, prep, execution, executing_side, scale, fps)
    metrics.append(_diagnostic_metric(movement, "kinematics", "executing_wrist_peak_speed_body_scale_per_sec", peak_speed, "body_scale/sec"))

    stance_span = fixation_metric(_stance_span_ratio, scale)
    knee = fixation_metric(_front_knee_angle, _stance_side(movement["stance"]))
    if "ap_seogi" in movement["stance"]:
        span_range = [t["ap_seogi_span_ratio_min"], t["ap_seogi_span_ratio_max"]]
        knee_range = [t["ap_seogi_front_knee_deg_min"], t["ap_seogi_front_knee_deg_max"]]
    else:
        span_range = [t["ap_gubi_span_ratio_min"], t["ap_gubi_span_ratio_max"]]
        knee_range = [t["ap_gubi_front_knee_deg_min"], t["ap_gubi_front_knee_deg_max"]]
    metrics.append(_range_metric(movement, "stance", "stance_span_ratio", stance_span, span_range, "body_scale"))
    metrics.append(_range_metric(movement, "stance", "front_knee_deg", knee, knee_range, "deg"))
    wrist_height = fixation_metric(_wrist_height_ratio, executing_side)
    if technique_id == "momtong_jireugi":
        height_range = [t["punch_wrist_height_torso_ratio_min"], t["punch_wrist_height_torso_ratio_max"]]
        metrics.append(
            _range_metric(
                movement,
                "technique",
                "executing_wrist_height_torso_ratio",
                wrist_height,
                height_range,
                "torso_height",
            )
        )
    elif technique_id == "arae_makki":
        height_range = [t["block_wrist_height_torso_ratio_min"], t["block_wrist_height_torso_ratio_max"]]
        metrics.append(
            _range_metric(
                movement,
                "technique",
                "executing_wrist_height_torso_ratio",
                wrist_height,
                height_range,
                "torso_height",
            )
        )
    else:
        metrics.append(
            _diagnostic_metric(
                movement,
                "technique",
                "executing_wrist_height_torso_ratio",
                wrist_height,
                "torso_height",
            )
        )
    elbow_estimate = fixation_estimate(_executing_elbow_angle, executing_side)
    elbow = elbow_estimate[0]
    if technique_id == "momtong_jireugi":
        elbow_range = [t["punch_elbow_deg_min"], t["punch_elbow_deg_max"]]
        metrics.append(
            _with_uncertainty(
                _range_metric(
                    movement,
                    "technique",
                    "executing_elbow_deg",
                    elbow,
                    elbow_range,
                    "deg",
                ),
                uncertainty_95=elbow_estimate[1],
                sample_count=elbow_estimate[2],
            )
        )
    elif technique_id == "arae_makki":
        metrics.append(
            _with_uncertainty(
                _diagnostic_metric(
                    movement,
                    "technique",
                    "executing_elbow_deg",
                    elbow,
                    "deg",
                ),
                uncertainty_95=elbow_estimate[1],
                sample_count=elbow_estimate[2],
            )
        )
    else:
        metrics.append(
            _with_uncertainty(
                _diagnostic_metric(
                    movement,
                    "technique",
                    "executing_elbow_deg",
                    elbow,
                    "deg",
                ),
                uncertainty_95=elbow_estimate[1],
                sample_count=elbow_estimate[2],
            )
        )

    if technique_id == "arae_makki":
        fist_to_thigh = fixation_estimate(
            _arae_fist_to_thigh_ratio,
            executing_side,
            scale,
            profile["quality_gates"],
        )
        metrics.append(
            _with_uncertainty(
                _diagnostic_metric(
                    movement,
                    "technique",
                    "arae_fist_to_thigh_fist_ratio",
                    fist_to_thigh[0],
                    "fist_width",
                ),
                uncertainty_95=fist_to_thigh[1],
                sample_count=fist_to_thigh[2],
            )
        )

    if any(technique["technique_id"] == "ap_chagi" for technique in movement["techniques"]):
        metrics.extend(
            _ap_chagi_diagnostics(
                movement,
                segment,
                arrays,
                scale,
                fps,
                profile["quality_gates"],
            )
        )

    _annotate_metrics(
        metrics,
        movement,
        segment,
        technique_id,
        executing_side,
        other_side,
        group_coverage,
        minimum_group_coverage,
        radius,
        timeline_fps=fps,
    )
    return {
        "movement_id": movement["movement_id"],
        "display_name": movement["display_name"],
        "stance": movement["stance"],
        "technique_id": technique_id,
        "technique_ids": [technique["technique_id"] for technique in movement["techniques"]],
        "executing_side": executing_side,
        "start_frame": start,
        "end_frame": end,
        "anchors": {"preparation": prep, "execution": execution, "fixation": fixation},
        "group_quality_coverage": group_coverage,
        "group_quality_gate_failures": group_gate_failures,
        "minimum_group_quality_coverage": minimum_group_coverage,
        "anchor_window_radius_frames": radius,
        "min_anchor_window_valid_samples": minimum_samples,
        "body_scale_m": scale,
        "metrics": metrics,
    }


def _pose_arrays(payload: dict[str, Any], timeline: dict[str, Any], gates: dict[str, float]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ScoringContractError("pose payload must be a mapping")
    keypoints = np.asarray(payload.get("keypoints_3d_world"), dtype=float)
    expected = (timeline["frame_count"], COCO_WHOLEBODY_KEYPOINTS, 3)
    if keypoints.shape != expected:
        raise ScoringContractError(f"WholeBody keypoints must have shape {expected}, got {keypoints.shape}")
    valid = np.asarray(payload.get("reliability_valid_mask"), dtype=bool)
    cameras = np.asarray(payload.get("used_cameras"), dtype=float)
    reprojection = np.asarray(payload.get("reprojection_error"), dtype=float)
    for name, value in (("reliability_valid_mask", valid), ("used_cameras", cameras), ("reprojection_error", reprojection)):
        if value.shape != expected[:2]:
            raise ScoringContractError(f"{name} must have shape {expected[:2]}")
    finite = np.all(np.isfinite(keypoints), axis=-1)
    quality = (
        valid
        & finite
        & (cameras >= gates["min_used_cameras"])
        & np.isfinite(reprojection)
        & (reprojection <= gates["max_reprojection_error_px"])
    )
    points = keypoints.copy()
    points[~quality] = np.nan
    return {
        "points": points,
        "quality_mask": quality,
        "cameras": cameras,
        "reprojection": reprojection,
        "min_trajectory_valid_ratio": gates["min_segment_group_valid_ratio"],
    }


def _primary_technique(movement: dict[str, Any]) -> tuple[str, str]:
    for technique in movement["techniques"]:
        if technique["side"] in {"left", "right"} and technique["technique_id"] != "ap_chagi":
            return technique["side"], technique["technique_id"]
    raise ScoringContractError(f"no scorable sided technique in {movement['movement_id']}")


def _execution_anchor(segment: dict[str, Any]) -> int:
    for name in ("execution", "punch_execution", "kick_apex"):
        if name in segment["anchors"]:
            return segment["anchors"][name]
    return segment["end_frame"]


def _stance_side(stance: str) -> str:
    if stance.startswith("left_"):
        return "left"
    if stance.startswith("right_"):
        return "right"
    raise ScoringContractError(f"stance side is unknown: {stance}")


def _stance_family(stance: str) -> str:
    if stance.endswith("_ap_seogi"):
        return "ap_seogi"
    if stance.endswith("_ap_gubi"):
        return "ap_gubi"
    raise ScoringContractError(f"stance family is unknown: {stance}")


def _p(arrays: dict[str, Any], frame: int, index: int) -> np.ndarray | None:
    point = arrays["points"][frame, index]
    return point if np.all(np.isfinite(point)) else None


def _p_window(
    arrays: dict[str, Any],
    center: int,
    index: int,
    start: int,
    end: int,
    radius: int,
    minimum_samples: int,
) -> np.ndarray | None:
    low, high = max(start, center - radius), min(end, center + radius)
    values = arrays["points"][low : high + 1, index]
    valid = np.all(np.isfinite(values), axis=1)
    if np.count_nonzero(valid) < minimum_samples:
        return None
    return np.median(values[valid], axis=0)


def _window_median(
    function: Any,
    arrays: dict[str, Any],
    center: int,
    start: int,
    end: int,
    radius: int,
    minimum_samples: int,
    *args: Any,
) -> float | None:
    values = []
    for frame in range(max(start, center - radius), min(end, center + radius) + 1):
        value = function(arrays, frame, *args)
        if value is not None and np.isfinite(value):
            values.append(float(value))
    return float(np.median(values)) if len(values) >= minimum_samples else None


def _window_estimate(
    function: Any,
    arrays: dict[str, Any],
    center: int,
    start: int,
    end: int,
    radius: int,
    minimum_samples: int,
    *args: Any,
) -> tuple[float | None, float | None, int]:
    values = []
    for frame in range(max(start, center - radius), min(end, center + radius) + 1):
        value = function(arrays, frame, *args)
        if value is not None and np.isfinite(value):
            values.append(float(value))
    if len(values) < minimum_samples:
        return None, None, len(values)
    samples = np.asarray(values, dtype=float)
    median = float(np.median(samples))
    mad = float(np.median(np.abs(samples - median)))
    uncertainty_95 = 1.96 * 1.4826 * mad
    return median, uncertainty_95, len(values)


def _distance(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    return None if a is None or b is None else float(np.linalg.norm(a - b))


def _angle(a: np.ndarray | None, center: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or center is None or b is None:
        return None
    first, second = a - center, b - center
    denom = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denom <= 1e-9:
        return None
    return float(np.degrees(np.arccos(np.clip(np.dot(first, second) / denom, -1.0, 1.0))))


def _vector_angle(
    first: np.ndarray | None,
    second: np.ndarray | None,
    *,
    undirected: bool,
) -> float | None:
    if first is None or second is None:
        return None
    a, b = np.asarray(first[:2]), np.asarray(second[:2])
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-9:
        return None
    angle = float(np.degrees(np.arccos(np.clip(np.dot(a, b) / denom, -1.0, 1.0))))
    return min(angle, 180.0 - angle) if undirected else angle


def _median_body_scale(arrays: dict[str, Any], frames: np.ndarray) -> float | None:
    values = []
    for frame in frames:
        lengths = []
        for side in ("left", "right"):
            hip = _p(arrays, int(frame), COCO_BODY_JOINTS[f"{side}_hip"])
            knee = _p(arrays, int(frame), COCO_BODY_JOINTS[f"{side}_knee"])
            ankle = _p(arrays, int(frame), COCO_BODY_JOINTS[f"{side}_ankle"])
            first, second = _distance(hip, knee), _distance(knee, ankle)
            if first is not None and second is not None:
                lengths.append(first + second)
        if lengths:
            values.append(float(np.mean(lengths)))
    return float(np.median(values)) if values else None


def _torso_lean_p95(arrays: dict[str, Any], frames: np.ndarray) -> float | None:
    values = []
    for frame in frames:
        shoulders = [_p(arrays, int(frame), COCO_BODY_JOINTS[f"{side}_shoulder"]) for side in ("left", "right")]
        hips = [_p(arrays, int(frame), COCO_BODY_JOINTS[f"{side}_hip"]) for side in ("left", "right")]
        if any(point is None for point in shoulders + hips):
            continue
        shoulder_mid = (shoulders[0] + shoulders[1]) / 2.0
        hip_mid = (hips[0] + hips[1]) / 2.0
        vector = shoulder_mid - hip_mid
        denom = float(np.linalg.norm(vector))
        if denom > 1e-9:
            values.append(float(np.degrees(np.arccos(np.clip(vector[2] / denom, -1.0, 1.0)))))
    return float(np.percentile(values, 95)) if values else None


def _back_foot_yaw_to_stance_direction(
    arrays: dict[str, Any],
    frame: int,
    stance: str,
    scale: float | None,
    gates: dict[str, Any],
) -> float | None:
    if scale is None or scale <= 0:
        return None
    front_side = _stance_side(stance)
    back_side = "right" if front_side == "left" else "left"
    front_ankle = _p(arrays, frame, COCO_BODY_JOINTS[f"{front_side}_ankle"])
    back_ankle = _p(arrays, frame, COCO_BODY_JOINTS[f"{back_side}_ankle"])
    back_heading = _foot_heading(arrays, frame, back_side, scale, gates)
    if front_ankle is None or back_ankle is None or back_heading is None:
        return None
    stance_direction = front_ankle - back_ankle
    if float(np.linalg.norm(stance_direction[:2]) / scale) < 0.10:
        return None
    return _vector_angle(back_heading, stance_direction, undirected=False)


def _shoulder_hip_twist(arrays: dict[str, Any], frame: int) -> float | None:
    ls, rs = (_p(arrays, frame, COCO_BODY_JOINTS[name]) for name in ("left_shoulder", "right_shoulder"))
    lh, rh = (_p(arrays, frame, COCO_BODY_JOINTS[name]) for name in ("left_hip", "right_hip"))
    return _vector_angle(
        None if ls is None or rs is None else rs - ls,
        None if lh is None or rh is None else rh - lh,
        undirected=True,
    )


def _head_torso_yaw_mismatch(
    arrays: dict[str, Any],
    frame: int,
    scale: float | None,
    gates: dict[str, Any],
) -> float | None:
    left_eye_indices = [23 + index for index in range(36, 42)]
    right_eye_indices = [23 + index for index in range(42, 48)]
    left_eye = [_p(arrays, frame, index) for index in left_eye_indices]
    right_eye = [_p(arrays, frame, index) for index in right_eye_indices]
    ls, rs = (_p(arrays, frame, COCO_BODY_JOINTS[name]) for name in ("left_shoulder", "right_shoulder"))
    if scale is None or scale <= 0 or any(point is None for point in left_eye + right_eye) or ls is None or rs is None:
        return None
    eye_line = np.mean(right_eye, axis=0) - np.mean(left_eye, axis=0)
    eye_ratio = float(np.linalg.norm(eye_line) / scale)
    if not gates["face_eye_separation_body_scale_min"] <= eye_ratio <= gates["face_eye_separation_body_scale_max"]:
        return None
    return _vector_angle(eye_line, rs - ls, undirected=True)


def _expected_side_dominance(
    arrays: dict[str, Any],
    prep: int,
    execution: int,
    expected: str,
    other: str,
    start: int,
    end: int,
    radius: int,
    minimum_samples: int,
) -> float | None:
    expected_distance = _distance(
        _p_window(
            arrays,
            prep,
            COCO_BODY_JOINTS[f"{expected}_wrist"],
            start,
            end,
            radius,
            minimum_samples,
        ),
        _p_window(
            arrays,
            execution,
            COCO_BODY_JOINTS[f"{expected}_wrist"],
            start,
            end,
            radius,
            minimum_samples,
        ),
    )
    other_distance = _distance(
        _p_window(
            arrays,
            prep,
            COCO_BODY_JOINTS[f"{other}_wrist"],
            start,
            end,
            radius,
            minimum_samples,
        ),
        _p_window(
            arrays,
            execution,
            COCO_BODY_JOINTS[f"{other}_wrist"],
            start,
            end,
            radius,
            minimum_samples,
        ),
    )
    if expected_distance is None or other_distance is None:
        return None
    return float(expected_distance / max(other_distance, 1e-6))


def _hikite_distance_ratio(arrays: dict[str, Any], frame: int, side: str, scale: float | None) -> float | None:
    if scale is None or scale <= 0:
        return None
    wrist = _p(arrays, frame, coco_hand_joint(side, "wrist"))
    hip = _p(arrays, frame, COCO_BODY_JOINTS[f"{side}_hip"])
    distance = _distance(wrist, hip)
    return None if distance is None else distance / scale


def _wrist_forearm_alignment(
    arrays: dict[str, Any],
    frame: int,
    side: str,
    scale: float | None,
    gates: dict[str, Any],
) -> float | None:
    elbow = _p(arrays, frame, COCO_BODY_JOINTS[f"{side}_elbow"])
    wrist = _p(arrays, frame, COCO_BODY_JOINTS[f"{side}_wrist"])
    middle = _p(arrays, frame, coco_hand_joint(side, "middle_mcp"))
    if not _plausible_palm_scale(wrist, middle, scale, gates):
        return None
    angle = _angle(elbow, wrist, middle)
    return None if angle is None else abs(180.0 - angle)


def _fist_closure_ratio(
    arrays: dict[str, Any],
    frame: int,
    side: str,
    scale: float | None,
    gates: dict[str, Any],
) -> float | None:
    palm_names = ("wrist", "thumb_cmc", "thumb_mcp", "index_mcp", "middle_mcp", "ring_mcp", "pinky_mcp")
    distal_names = (
        "thumb_ip",
        "thumb_tip",
        "index_pip",
        "index_dip",
        "index_tip",
        "middle_pip",
        "middle_dip",
        "middle_tip",
        "ring_pip",
        "ring_dip",
        "ring_tip",
        "pinky_pip",
        "pinky_dip",
        "pinky_tip",
    )
    palm_points = [_p(arrays, frame, coco_hand_joint(side, name)) for name in palm_names]
    distal_points = [_p(arrays, frame, coco_hand_joint(side, name)) for name in distal_names]
    if any(point is None for point in palm_points + distal_points):
        return None
    wrist = palm_points[0]
    middle = palm_points[4]
    palm_scale = _distance(wrist, middle)
    if palm_scale is None or palm_scale <= 1e-9 or not _plausible_palm_scale(wrist, middle, scale, gates):
        return None
    palm_center = np.mean(palm_points, axis=0)
    return float(
        np.mean([np.linalg.norm(point - palm_center) for point in distal_points]) / palm_scale
    )


def _plausible_palm_scale(
    wrist: np.ndarray | None,
    middle: np.ndarray | None,
    scale: float | None,
    gates: dict[str, Any],
) -> bool:
    palm_length = _distance(wrist, middle)
    if scale is None or scale <= 0 or palm_length is None:
        return False
    ratio = palm_length / scale
    return gates["hand_palm_length_body_scale_min"] <= ratio <= gates["hand_palm_length_body_scale_max"]


def _trajectory(arrays: dict[str, Any], start: int, end: int, index: int) -> tuple[np.ndarray, np.ndarray]:
    values = arrays["points"][start : end + 1, index]
    valid = np.all(np.isfinite(values), axis=1)
    return values, valid


def _settle_frame(values: np.ndarray, valid: np.ndarray, start_frame: int) -> int | None:
    if values.shape[0] < 3:
        return None
    speed = np.linalg.norm(np.diff(values, axis=0), axis=1)
    speed[~(valid[1:] & valid[:-1])] = np.nan
    finite = speed[np.isfinite(speed)]
    if finite.size < 2:
        return None
    threshold = max(float(np.nanmax(speed)) * 0.20, 1e-5)
    active = np.flatnonzero(np.isfinite(speed) & (speed >= threshold))
    return None if active.size == 0 else start_frame + int(active[-1]) + 1


def _settle_time_difference(
    arrays: dict[str, Any], start: int, end: int, hand_side: str, foot_side: str, fps: float
) -> float | None:
    wrist, wrist_valid = _trajectory(arrays, start, end, COCO_BODY_JOINTS[f"{hand_side}_wrist"])
    ankle, ankle_valid = _trajectory(arrays, start, end, COCO_BODY_JOINTS[f"{foot_side}_ankle"])
    minimum = arrays["min_trajectory_valid_ratio"]
    if float(np.mean(wrist_valid)) < minimum or float(np.mean(ankle_valid)) < minimum:
        return None
    wrist_settle = _settle_frame(wrist, wrist_valid, start)
    ankle_settle = _settle_frame(ankle, ankle_valid, start)
    if wrist_settle is None or ankle_settle is None:
        return None
    return abs(wrist_settle - ankle_settle) / fps


def _fixation_jitter_ratio(
    arrays: dict[str, Any], start: int, end: int, side: str, scale: float | None
) -> float | None:
    if scale is None or scale <= 0 or end <= start:
        return None
    values, valid = _trajectory(arrays, start, end, COCO_BODY_JOINTS[f"{side}_wrist"])
    if float(np.mean(valid)) < arrays["min_trajectory_valid_ratio"]:
        return None
    values = values[valid]
    if values.shape[0] < 3:
        return None
    center = np.median(values, axis=0)
    return float(np.percentile(np.linalg.norm(values - center, axis=1), 95) / scale)


def _weight_transfer_ratio(
    arrays: dict[str, Any],
    prep: int,
    fixation: int,
    stance: str,
    scale: float | None,
    start: int,
    end: int,
    radius: int,
    minimum_samples: int,
) -> float | None:
    if scale is None or scale <= 0:
        return None
    side = _stance_side(stance)
    other = "right" if side == "left" else "left"
    front = _p_window(
        arrays,
        fixation,
        COCO_BODY_JOINTS[f"{side}_ankle"],
        start,
        end,
        radius,
        minimum_samples,
    )
    rear = _p_window(
        arrays,
        fixation,
        COCO_BODY_JOINTS[f"{other}_ankle"],
        start,
        end,
        radius,
        minimum_samples,
    )
    prep_hips = [
        _p_window(
            arrays,
            prep,
            COCO_BODY_JOINTS[f"{item}_hip"],
            start,
            end,
            radius,
            minimum_samples,
        )
        for item in ("left", "right")
    ]
    fix_hips = [
        _p_window(
            arrays,
            fixation,
            COCO_BODY_JOINTS[f"{item}_hip"],
            start,
            end,
            radius,
            minimum_samples,
        )
        for item in ("left", "right")
    ]
    if front is None or rear is None or any(point is None for point in prep_hips + fix_hips):
        return None
    axis = front[:2] - rear[:2]
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-9:
        return None
    displacement = np.mean(fix_hips, axis=0)[:2] - np.mean(prep_hips, axis=0)[:2]
    # Keep direction: movement away from the declared front foot must not be
    # accepted as positive weight transfer merely because its magnitude is large.
    return float(np.dot(displacement, axis / norm)) / scale


def _path_efficiency(arrays: dict[str, Any], start: int, end: int, side: str) -> float | None:
    values, valid = _trajectory(arrays, start, end, COCO_BODY_JOINTS[f"{side}_wrist"])
    if float(np.mean(valid)) < arrays["min_trajectory_valid_ratio"]:
        return None
    adjacent_valid = valid[1:] & valid[:-1]
    if np.count_nonzero(adjacent_valid) < 2:
        return None
    path = float(np.sum(np.linalg.norm(np.diff(values, axis=0)[adjacent_valid], axis=1)))
    valid_indices = np.flatnonzero(valid)
    direct = float(np.linalg.norm(values[valid_indices[-1]] - values[valid_indices[0]]))
    return None if direct <= 1e-6 else path / direct


def _peak_speed_normalized(
    arrays: dict[str, Any], start: int, end: int, side: str, scale: float | None, fps: float
) -> float | None:
    if scale is None or scale <= 0:
        return None
    values, valid = _trajectory(arrays, start, end, COCO_BODY_JOINTS[f"{side}_wrist"])
    if float(np.mean(valid)) < arrays["min_trajectory_valid_ratio"]:
        return None
    speed = np.linalg.norm(np.diff(values, axis=0), axis=1) * fps
    speed[~(valid[1:] & valid[:-1])] = np.nan
    return None if not np.any(np.isfinite(speed)) else float(np.nanpercentile(speed, 95) / scale)


def _stance_span_ratio(arrays: dict[str, Any], frame: int, scale: float | None) -> float | None:
    if scale is None or scale <= 0:
        return None
    left = _p(arrays, frame, COCO_BODY_JOINTS["left_ankle"])
    right = _p(arrays, frame, COCO_BODY_JOINTS["right_ankle"])
    if left is None or right is None:
        return None
    return float(np.linalg.norm(left[:2] - right[:2]) / scale)


def _front_knee_angle(arrays: dict[str, Any], frame: int, side: str) -> float | None:
    return _angle(
        _p(arrays, frame, COCO_BODY_JOINTS[f"{side}_hip"]),
        _p(arrays, frame, COCO_BODY_JOINTS[f"{side}_knee"]),
        _p(arrays, frame, COCO_BODY_JOINTS[f"{side}_ankle"]),
    )


def _wrist_height_ratio(arrays: dict[str, Any], frame: int, side: str) -> float | None:
    shoulders = [_p(arrays, frame, COCO_BODY_JOINTS[f"{item}_shoulder"]) for item in ("left", "right")]
    hips = [_p(arrays, frame, COCO_BODY_JOINTS[f"{item}_hip"]) for item in ("left", "right")]
    wrist = _p(arrays, frame, coco_hand_joint(side, "wrist"))
    if wrist is None or any(point is None for point in shoulders + hips):
        return None
    shoulder_mid, hip_mid = np.mean(shoulders, axis=0), np.mean(hips, axis=0)
    torso_height = float(shoulder_mid[2] - hip_mid[2])
    return None if torso_height <= 1e-6 else float((wrist[2] - hip_mid[2]) / torso_height)


def _executing_elbow_angle(arrays: dict[str, Any], frame: int, side: str) -> float | None:
    return _angle(
        _p(arrays, frame, COCO_BODY_JOINTS[f"{side}_shoulder"]),
        _p(arrays, frame, COCO_BODY_JOINTS[f"{side}_elbow"]),
        _p(arrays, frame, COCO_BODY_JOINTS[f"{side}_wrist"]),
    )


def _arae_fist_to_thigh_ratio(
    arrays: dict[str, Any],
    frame: int,
    side: str,
    scale: float | None,
    gates: dict[str, Any],
) -> float | None:
    if scale is None or scale <= 0:
        return None
    index_mcp = _p(arrays, frame, coco_hand_joint(side, "index_mcp"))
    middle_mcp = _p(arrays, frame, coco_hand_joint(side, "middle_mcp"))
    ring_mcp = _p(arrays, frame, coco_hand_joint(side, "ring_mcp"))
    pinky_mcp = _p(arrays, frame, coco_hand_joint(side, "pinky_mcp"))
    hip = _p(arrays, frame, COCO_BODY_JOINTS[f"{side}_hip"])
    knee = _p(arrays, frame, COCO_BODY_JOINTS[f"{side}_knee"])
    hand_points = [index_mcp, middle_mcp, ring_mcp, pinky_mcp]
    if any(point is None for point in hand_points) or hip is None or knee is None:
        return None
    fist_width = _distance(index_mcp, pinky_mcp)
    if fist_width is None:
        return None
    fist_width_ratio = fist_width / scale
    if not (
        gates["hand_palm_length_body_scale_min"]
        <= fist_width_ratio
        <= gates["hand_palm_length_body_scale_max"]
    ):
        return None
    fist_center = np.mean(hand_points, axis=0)
    thigh = knee - hip
    denominator = float(np.dot(thigh, thigh))
    if denominator <= 1e-9:
        return None
    projection = float(np.clip(np.dot(fist_center - hip, thigh) / denominator, 0.0, 1.0))
    closest = hip + projection * thigh
    return float(np.linalg.norm(fist_center - closest) / fist_width)


def _ap_chagi_diagnostics(
    movement: dict[str, Any],
    segment: dict[str, Any],
    arrays: dict[str, Any],
    scale: float | None,
    fps: float,
    gates: dict[str, Any],
) -> list[dict[str, Any]]:
    kick = next(
        technique for technique in movement["techniques"] if technique["technique_id"] == "ap_chagi"
    )
    side = kick["side"]
    support_side = "right" if side == "left" else "left"
    start, end = segment["start_frame"], segment["end_frame"]
    apex = segment["anchors"].get("kick_apex")
    rechamber = segment["anchors"].get("rechamber")
    preparation = segment["anchors"].get("preparation", start)
    landing = segment["anchors"].get("landing")
    punch = segment["anchors"].get("punch_execution")
    radius = gates["anchor_window_radius_frames"]
    minimum_samples = gates["min_anchor_window_valid_samples"]

    def at_anchor(anchor: int | None, function: Any, *args: Any) -> float | None:
        if anchor is None:
            return None
        return _window_median(
            function,
            arrays,
            anchor,
            start,
            end,
            radius,
            minimum_samples,
            *args,
        )

    extension = at_anchor(apex, _front_knee_angle, side)
    height = at_anchor(apex, _ankle_height_body_scale_ratio, side, scale)
    rechamber_angle = at_anchor(rechamber, _front_knee_angle, side)
    support_pivot = _support_foot_pivot(
        arrays,
        preparation,
        apex,
        support_side,
        scale,
        start,
        end,
        radius,
        minimum_samples,
        gates,
    )
    landing_to_punch = (
        None if landing is None or punch is None or punch < landing else (punch - landing) / fps
    )
    return [
        _diagnostic_metric(
            movement,
            "kick",
            "kick_knee_extension_deg",
            extension,
            "deg",
        ),
        _diagnostic_metric(
            movement,
            "kick",
            "kick_ankle_height_body_scale_ratio",
            height,
            "body_scale",
        ),
        _diagnostic_metric(
            movement,
            "kick",
            "kick_rechamber_knee_deg",
            rechamber_angle,
            "deg",
        ),
        _diagnostic_metric(
            movement,
            "kick",
            "support_foot_pivot_deg",
            support_pivot,
            "deg",
        ),
        _diagnostic_metric(
            movement,
            "timing",
            "kick_landing_to_punch_sec",
            landing_to_punch,
            "sec",
        ),
    ]


def _ankle_height_body_scale_ratio(
    arrays: dict[str, Any],
    frame: int,
    side: str,
    scale: float | None,
) -> float | None:
    if scale is None or scale <= 0:
        return None
    ankle = _p(arrays, frame, COCO_BODY_JOINTS[f"{side}_ankle"])
    hips = [_p(arrays, frame, COCO_BODY_JOINTS[f"{item}_hip"]) for item in ("left", "right")]
    if ankle is None or any(point is None for point in hips):
        return None
    hip_mid = np.mean(hips, axis=0)
    return float((ankle[2] - hip_mid[2]) / scale)


def _foot_heading(
    arrays: dict[str, Any],
    frame: int,
    side: str,
    scale: float | None,
    gates: dict[str, Any],
) -> np.ndarray | None:
    if scale is None or scale <= 0:
        return None
    heel = _p(arrays, frame, COCO_FOOT_JOINTS[f"{side}_heel"])
    big = _p(arrays, frame, COCO_FOOT_JOINTS[f"{side}_big_toe"])
    small = _p(arrays, frame, COCO_FOOT_JOINTS[f"{side}_small_toe"])
    if heel is None or big is None or small is None:
        return None
    vector = (big + small) / 2.0 - heel
    ratio = float(np.linalg.norm(vector) / scale)
    if not gates["foot_length_body_scale_min"] <= ratio <= gates["foot_length_body_scale_max"]:
        return None
    return vector


def _support_foot_pivot(
    arrays: dict[str, Any],
    preparation: int,
    apex: int | None,
    side: str,
    scale: float | None,
    start: int,
    end: int,
    radius: int,
    minimum_samples: int,
    gates: dict[str, Any],
) -> float | None:
    if apex is None:
        return None
    values = []
    for prep_frame in range(max(start, preparation - radius), min(end, preparation + radius) + 1):
        prep_heading = _foot_heading(arrays, prep_frame, side, scale, gates)
        if prep_heading is None:
            continue
        for apex_frame in range(max(start, apex - radius), min(end, apex + radius) + 1):
            apex_heading = _foot_heading(arrays, apex_frame, side, scale, gates)
            angle = _vector_angle(prep_heading, apex_heading, undirected=False)
            if angle is not None:
                values.append(angle)
    return float(np.median(values)) if len(values) >= minimum_samples else None


def _annotate_metrics(
    metrics: list[dict[str, Any]],
    movement: dict[str, Any],
    segment: dict[str, Any],
    technique_id: str,
    executing_side: str,
    other_side: str,
    group_coverage: dict[str, float],
    minimum_group_coverage: float,
    radius: int,
    *,
    timeline_fps: float,
) -> None:
    fixation = segment["anchors"].get("fixation", segment["end_frame"])
    execution = _execution_anchor(segment)
    execution_scope = {
        "expected_side_dominance_ratio",
        "hand_foot_settle_difference_sec",
        "executing_wrist_path_efficiency",
        "executing_wrist_peak_speed_body_scale_per_sec",
    }
    body_only = {
        "torso_lean_p95_deg",
        "shoulder_hip_twist_deg",
        "expected_side_dominance_ratio",
        "hand_foot_settle_difference_sec",
        "fixation_wrist_jitter_ratio",
        "pelvis_weight_transfer_ratio",
        "executing_wrist_path_efficiency",
        "executing_wrist_peak_speed_body_scale_per_sec",
        "stance_span_ratio",
        "front_knee_deg",
    }
    for metric in metrics:
        metric_id = metric["metric_id"]
        criterion_id = {
            "torso_lean_p95_deg": "balance.torso_vertical",
            "back_foot_yaw_to_stance_direction_deg": "stance.foot_direction",
            "shoulder_hip_twist_deg": "rotation.shoulder_hip",
            "head_torso_yaw_mismatch_deg": "gaze.direction",
            "expected_side_dominance_ratio": "technique.side",
            "reaction_hand_hip_distance_ratio": "technique.hikite.position",
            "wrist_forearm_alignment_deg": "technique.hand.wrist_alignment",
            "fist_closure_ratio": "technique.hand.fist",
            "hand_foot_settle_difference_sec": "timing.hand_foot.simultaneity",
            "fixation_wrist_jitter_ratio": "technique.fixation.stability",
            "pelvis_weight_transfer_ratio": "stance.weight_transfer",
            "executing_wrist_path_efficiency": "technique.trajectory",
            "executing_wrist_peak_speed_body_scale_per_sec": "presentation.kinematics",
            "stance_span_ratio": f"stance.{_stance_family(movement['stance'])}.span",
            "front_knee_deg": f"stance.{_stance_family(movement['stance'])}.front_knee",
            "executing_wrist_height_torso_ratio": f"technique.{technique_id}.height",
            "executing_elbow_deg": f"technique.{technique_id}.elbow",
            "arae_fist_to_thigh_fist_ratio": "technique.arae_makki.height",
            "kick_knee_extension_deg": "technique.ap_chagi.knee_extension",
            "kick_ankle_height_body_scale_ratio": "technique.ap_chagi.height",
            "kick_rechamber_knee_deg": "technique.ap_chagi.rechamber",
            "support_foot_pivot_deg": "technique.ap_chagi.support_foot_pivot",
            "kick_landing_to_punch_sec": "timing.kick_landing_punch.sequence",
        }[metric_id]
        metric["criterion_id"] = criterion_id
        if metric_id in {"kick_knee_extension_deg", "kick_ankle_height_body_scale_ratio"}:
            anchor = segment["anchors"].get("kick_apex", segment["end_frame"])
            low = max(segment["start_frame"], anchor - radius)
            high = min(segment["end_frame"], anchor + radius)
            scope = "kick_apex_window"
        elif metric_id == "kick_rechamber_knee_deg":
            anchor = segment["anchors"].get("rechamber", segment["end_frame"])
            low = max(segment["start_frame"], anchor - radius)
            high = min(segment["end_frame"], anchor + radius)
            scope = "kick_rechamber_window"
        elif metric_id == "support_foot_pivot_deg":
            low = segment["anchors"].get("preparation", segment["start_frame"])
            high = segment["anchors"].get("kick_apex", segment["end_frame"])
            anchor, scope = high, "preparation_to_kick_apex"
        elif metric_id == "kick_landing_to_punch_sec":
            low = segment["anchors"].get("landing", segment["start_frame"])
            high = segment["anchors"].get("punch_execution", segment["end_frame"])
            anchor, scope = high, "landing_to_punch_execution"
        elif metric_id in execution_scope:
            low = segment["anchors"].get("preparation", segment["start_frame"])
            high, anchor, scope = execution, execution, "preparation_to_execution"
        else:
            low = max(segment["start_frame"], fixation - radius)
            high = min(segment["end_frame"], fixation + radius)
            anchor, scope = fixation, "fixation_window"

        if metric_id == "back_foot_yaw_to_stance_direction_deg":
            required_groups = ["body17", "feet"]
        elif metric_id == "support_foot_pivot_deg":
            required_groups = ["feet"]
        elif metric_id == "head_torso_yaw_mismatch_deg":
            required_groups = ["body17", "face"]
        elif metric_id in {
            "wrist_forearm_alignment_deg",
            "fist_closure_ratio",
            "executing_wrist_height_torso_ratio",
            "arae_fist_to_thigh_fist_ratio",
        }:
            required_groups = ["body17", f"{executing_side}_hand"]
        elif metric_id == "reaction_hand_hip_distance_ratio":
            required_groups = ["body17", f"{other_side}_hand"]
        elif metric_id in body_only:
            required_groups = ["body17"]
        elif metric_id in {
            "kick_knee_extension_deg",
            "kick_ankle_height_body_scale_ratio",
            "kick_rechamber_knee_deg",
            "kick_landing_to_punch_sec",
        }:
            required_groups = ["body17"]
        else:
            required_groups = ["body17"]

        failed_groups = [
            group for group in required_groups if group_coverage[group] < minimum_group_coverage
        ]
        not_measurable_reason = None
        if metric["screening_status"] == "not_measurable":
            not_measurable_reason = (
                "segment_group_coverage_below_minimum"
                if failed_groups
                else "insufficient_or_implausible_required_keypoint_evidence"
            )
        metric["measurement_evidence"] = {
            "scope": scope,
            "anchor_frame": anchor,
            "start_frame": low,
            "end_frame": high,
            "anchor_time_sec": anchor / timeline_fps,
            "start_time_sec": low / timeline_fps,
            "end_time_sec": high / timeline_fps,
            "required_groups": required_groups,
            "required_group_coverage": {group: group_coverage[group] for group in required_groups},
            "failed_group_quality_gates": failed_groups,
            "not_measurable_reason": not_measurable_reason,
        }


def _metric_base(movement: dict[str, Any], family: str, metric_id: str, value: float | None, unit: str) -> dict[str, Any]:
    return {
        "movement_id": movement["movement_id"],
        "family": family,
        "metric_id": metric_id,
        "value": value,
        "unit": unit,
    }


def _upper_metric(
    movement: dict[str, Any], family: str, metric_id: str, value: float | None, maximum: float, unit: str
) -> dict[str, Any]:
    status = "not_measurable" if value is None else ("review_candidate" if value > maximum else "within_screening_range")
    margin = None if value is None else value - maximum
    normalized = None if margin is None else margin / max(abs(maximum), 1e-9)
    return {
        **_metric_base(movement, family, metric_id, value, unit),
        "screening_status": status,
        "screening_rule": {"operator": "max", "value": maximum},
        "threshold_margin": margin,
        "normalized_threshold_margin": normalized,
    }


def _lower_metric(
    movement: dict[str, Any], family: str, metric_id: str, value: float | None, minimum: float, unit: str
) -> dict[str, Any]:
    status = "not_measurable" if value is None else ("review_candidate" if value < minimum else "within_screening_range")
    margin = None if value is None else minimum - value
    normalized = None if margin is None else margin / max(abs(minimum), 1e-9)
    return {
        **_metric_base(movement, family, metric_id, value, unit),
        "screening_status": status,
        "screening_rule": {"operator": "min", "value": minimum},
        "threshold_margin": margin,
        "normalized_threshold_margin": normalized,
    }


def _range_metric(
    movement: dict[str, Any], family: str, metric_id: str, value: float | None, limits: list[float], unit: str
) -> dict[str, Any]:
    status = "not_measurable" if value is None else ("within_screening_range" if limits[0] <= value <= limits[1] else "review_candidate")
    if value is None:
        margin = normalized = None
    elif value < limits[0]:
        margin = limits[0] - value
        normalized = margin / max(abs(limits[0]), 1e-9)
    elif value > limits[1]:
        margin = value - limits[1]
        normalized = margin / max(abs(limits[1]), 1e-9)
    else:
        margin = -min(value - limits[0], limits[1] - value)
        normalized = margin / max(abs(limits[1] - limits[0]), 1e-9)
    return {
        **_metric_base(movement, family, metric_id, value, unit),
        "screening_status": status,
        "screening_rule": {"operator": "range", "value": limits},
        "threshold_margin": margin,
        "normalized_threshold_margin": normalized,
    }


def _diagnostic_metric(
    movement: dict[str, Any], family: str, metric_id: str, value: float | None, unit: str
) -> dict[str, Any]:
    return {
        **_metric_base(movement, family, metric_id, value, unit),
        "screening_status": "measured_diagnostic_only",
        "screening_rule": None,
        "threshold_margin": None,
        "normalized_threshold_margin": None,
    }


def _with_uncertainty(
    metric: dict[str, Any],
    *,
    uncertainty_95: float | None,
    sample_count: int,
    **metadata: Any,
) -> dict[str, Any]:
    metric["uncertainty_95"] = uncertainty_95
    metric["sample_count"] = sample_count
    metric.update(metadata)
    return metric


def _exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        actual = set(value) if isinstance(value, dict) else set()
        raise ScoringContractError(
            f"{label} keys are invalid; missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
        )


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ScoringContractError(f"{label} must be a non-empty string")
    return value


def _nonnegative(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ScoringContractError(f"{label} must be numeric") from exc
    if not np.isfinite(result) or result < 0:
        raise ScoringContractError(f"{label} must be finite and non-negative")
    return result


def _positive(value: Any, label: str) -> float:
    result = _nonnegative(value, label)
    if result <= 0:
        raise ScoringContractError(f"{label} must be positive")
    return result


def _probability(value: Any, label: str) -> float:
    result = _nonnegative(value, label)
    if result > 1:
        raise ScoringContractError(f"{label} must be at most 1")
    return result


def _nonnegative_integer(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ScoringContractError(f"{label} must be a non-negative integer")
    return value


def _positive_integer(value: Any, label: str) -> int:
    result = _nonnegative_integer(value, label)
    if result == 0:
        raise ScoringContractError(f"{label} must be positive")
    return result


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
