from __future__ import annotations

from typing import Any, Callable

import numpy as np

from src.data_structures import (
    COCO_BODY_JOINTS,
    COCO_FOOT_JOINTS,
    COCO_HAND_LOCAL_JOINTS,
    COCO_LEFT_HAND_OFFSET,
    COCO_RIGHT_HAND_OFFSET,
)


def measure_observable_accuracy_metrics(
    arrays: dict[str, Any],
    contract: dict[str, Any],
    segment: dict[str, Any],
    profile: dict[str, Any],
    direction: dict[str, Any] | None,
    fps: float,
) -> dict[str, dict[str, Any]]:
    """Measure score-neutral geometry; correctness remains in the rule layer."""
    start, end = int(segment["start_frame"]), int(segment["end_frame"])
    anchors = segment["anchors"]
    fixation = int(anchors.get("fixation", end))
    preparation = int(anchors.get("preparation", start))
    execution = int(
        next(
            (anchors[name] for name in ("execution", "punch_execution", "kick_apex") if name in anchors),
            fixation,
        )
    )
    radius = int(profile["quality_gates"]["anchor_window_radius_frames"])
    fix_frames = _window(fixation, start, end, radius)
    prep_frames = _window(preparation, start, end, radius)
    exec_frames = _window(execution, start, end, radius)
    after_frames = np.arange(fixation, end + 1, dtype=int)
    transition_frames = np.arange(start, fixation + 1, dtype=int)
    active = contract["active_arm"]
    reaction = contract["chamber_or_reaction_arm"]
    lead = contract["lead_leg"]
    rear = contract["rear_or_support_leg"]
    kick_side = contract.get("kicking_leg")
    result: dict[str, dict[str, Any]] = {}

    def put(metric_id: str, value: Any, unit: str, *, reason: str | None = None) -> None:
        measured = _finite_value(value)
        result[metric_id] = {
            "value": value if measured else None,
            "unit": unit,
            "quality_status": "measured" if measured else "unmeasurable",
            "evidence": {
                "start_frame": int(start),
                "end_frame": int(end),
                "fixation_frame": fixation,
                "measurement_implementation": "wholebody_133_geometry_v3",
            },
            "source_metric_id": None,
            "not_measurable_reason": None if measured else (reason or "insufficient_valid_landmark_evidence"),
        }

    body_scale = _median_scalar(arrays, fix_frames, _body_scale)
    shoulder_width = _median_scalar(arrays, fix_frames, _shoulder_width)
    lead_leg_scale = _median_scalar(arrays, fix_frames, lambda a, f: _limb_scale(a, f, "leg", lead))
    active_arm_scale = _median_scalar(arrays, fix_frames, lambda a, f: _limb_scale(a, f, "arm", active))
    stance = _stance_geometry(arrays, fix_frames, lead, rear)
    head_quality = _group_quality(arrays, fix_frames, _head_indices())
    torso_quality = _group_quality(arrays, fix_frames, _torso_indices())
    lower_quality = _group_quality(arrays, fix_frames, _lower_indices())
    upper_quality = _group_quality(arrays, fix_frames, _upper_indices())
    hand_quality = _group_quality(arrays, fix_frames, _hand_indices(active))
    reaction_hand_quality = _group_quality(arrays, fix_frames, _hand_indices(reaction))
    foot_quality = _group_quality(arrays, fix_frames, _foot_indices(lead) + _foot_indices(rear))

    put("head_face_geometry_quality", head_quality, "ratio")
    put("head_orientation_unmeasurable_reason", "none" if head_quality is not None and head_quality >= 0.75 else None, "reason_code")
    put("torso_geometry_quality", torso_quality, "ratio")
    put("stance_geometry_quality", _minimum_finite(lower_quality, foot_quality), "ratio")
    put("lower_body_geometry_quality", lower_quality, "ratio")
    put("upper_body_geometry_quality", upper_quality, "ratio")
    put("hand_geometry_quality", _minimum_finite(hand_quality, reaction_hand_quality), "ratio")
    put("phase_evidence_quality", _phase_quality(anchors), "ratio")
    put("transition_geometry_quality", _group_quality(arrays, transition_frames, _lower_indices()), "ratio")
    put("fixation_evidence_quality", _group_quality(arrays, after_frames, tuple(range(17))), "ratio")

    put("torso_forward_backward_lean_deg", _median_scalar(arrays, fix_frames, _torso_forward_lean), "deg")
    put("shoulder_pelvis_roll_difference_deg", _median_scalar(arrays, fix_frames, _roll_difference), "deg")
    put("shoulder_height_asymmetry_body_ratio", _median_scalar(arrays, fix_frames, lambda a, f: _height_asymmetry(a, f, "shoulder")), "body_ratio")
    put("pelvis_height_asymmetry_body_ratio", _median_scalar(arrays, fix_frames, lambda a, f: _height_asymmetry(a, f, "hip")), "body_ratio")
    put("torso_fixation_orientation_dispersion_p95_deg", _orientation_dispersion(arrays, after_frames, "shoulder"), "deg")
    put("pelvis_fixation_orientation_dispersion_p95_deg", _orientation_dispersion(arrays, after_frames, "hip"), "deg")
    put("torso_post_fixation_drift_deg", _orientation_drift(arrays, after_frames, "shoulder"), "deg")
    put("pelvis_post_fixation_drift_deg", _orientation_drift(arrays, after_frames, "hip"), "deg")
    put("torso_translation_after_fixation_body_ratio", _centre_dispersion(arrays, after_frames, "shoulder"), "body_ratio")
    put("pelvis_translation_after_fixation_body_ratio", _centre_dispersion(arrays, after_frames, "hip"), "body_ratio")
    put("body_height_change_during_fixation_ratio", _body_height_change(arrays, after_frames), "body_ratio")

    put("stance_length_leg_ratio", _safe_ratio(stance.get("length"), lead_leg_scale), "leg_length")
    put("stance_width_shoulder_ratio", _safe_ratio(stance.get("width"), shoulder_width), "shoulder_width")
    stance_match = _range_match(_safe_ratio(stance.get("length"), lead_leg_scale), contract["stance_length_expectation"]) and _range_match(
        _safe_ratio(stance.get("width"), shoulder_width), contract["stance_width_expectation"]
    )
    put("expected_stance_type_match", stance_match, "bool")
    lead_displacement = _joint_displacement(arrays, prep_frames, fix_frames, f"{lead}_ankle", lead_leg_scale)
    rear_displacement = _joint_displacement(arrays, prep_frames, fix_frames, f"{rear}_ankle", lead_leg_scale)
    put("lead_leg_side_match", lead in {"left", "right"}, "bool")
    put("moving_foot_side_match", _side_displacement_match(contract["expected_moving_foot"], lead, lead_displacement, rear_displacement), "bool")
    put("expected_support_or_pivot_foot_match", _side_displacement_match(contract["expected_pivot_or_support_foot"], rear, rear_displacement, lead_displacement), "bool")
    put("moving_foot_match", result["moving_foot_side_match"]["value"], "bool")
    put("support_or_pivot_foot_match", result["expected_support_or_pivot_foot_match"]["value"], "bool")
    put("pivot_foot_displacement_body_ratio", rear_displacement, "leg_length")
    put("support_foot_displacement_body_ratio", rear_displacement, "leg_length")
    put("foot_landing_position_error_body_ratio", _stance_range_error(contract, stance, lead_leg_scale, shoulder_width), "leg_length")
    put("foot_crossing_margin_body_ratio", _safe_ratio(stance.get("crossing_margin"), body_scale), "body_ratio")
    put("heel_alignment_error_body_ratio", _safe_ratio(stance.get("heel_alignment"), body_scale), "body_ratio")
    put("toe_alignment_error_body_ratio", _safe_ratio(stance.get("toe_alignment"), body_scale), "body_ratio")
    put("front_back_foot_order_match", stance.get("front_back_order_match"), "bool")
    put("stance_depth_or_pelvis_height_leg_ratio", _pelvis_height_leg_ratio(arrays, fix_frames, lead_leg_scale), "leg_length")
    put("inter_foot_yaw_difference_deg", _foot_yaw_difference(arrays, fix_frames, lead, rear), "deg")
    front_foot_dispersion = _foot_dispersion(arrays, after_frames, lead, lead_leg_scale)
    rear_foot_dispersion = _foot_dispersion(arrays, after_frames, rear, lead_leg_scale)
    fixation_foot_dispersion = (
        max(front_foot_dispersion, rear_foot_dispersion)
        if front_foot_dispersion is not None and rear_foot_dispersion is not None
        else None
    )
    put("foot_fixation_slip_body_ratio", fixation_foot_dispersion, "leg_length")
    put("stance_fixation_dispersion_body_ratio", fixation_foot_dispersion, "leg_length")

    front_knee = _median_scalar(arrays, fix_frames, lambda a, f: _leg_angle(a, f, lead))
    rear_knee = _median_scalar(arrays, fix_frames, lambda a, f: _leg_angle(a, f, rear))
    put("front_knee_flexion_deg", front_knee, "deg")
    put("rear_knee_flexion_deg", rear_knee, "deg")
    knee_expectation = contract["knee_angle_or_alignment_expectation"]
    knee_match = None if knee_expectation is None else _range_match(front_knee, knee_expectation["front_included_angle_deg"]) and _range_match(rear_knee, knee_expectation["rear_included_angle_deg"])
    put("expected_knee_flexion_range_match", knee_match, "bool")
    put("knee_over_foot_alignment_ratio", _median_scalar(arrays, fix_frames, lambda a, f: _knee_foot_alignment(a, f, lead)), "leg_length")
    put("knee_valgus_varus_proxy_deg", _median_scalar(arrays, fix_frames, lambda a, f: _knee_plane_angle(a, f, lead)), "deg")
    put("hip_knee_ankle_plane_deviation_ratio", _median_scalar(arrays, fix_frames, lambda a, f: _knee_plane_deviation(a, f, lead)), "leg_length")
    put("lead_leg_identity_match", lead in {"left", "right"}, "bool")
    put("rear_leg_identity_match", rear in {"left", "right"} and rear != lead, "bool")
    put("lower_body_left_right_swap_state", False if lead in {"left", "right"} and rear != lead else None, "bool")
    put("pelvis_height_stance_match", _range_match(front_knee, knee_expectation["front_included_angle_deg"] if knee_expectation else None), "bool")
    put("lower_body_fixation_dispersion", _lower_dispersion(arrays, after_frames, lead_leg_scale), "leg_length")

    put("active_arm_side_match", _arm_side_match(arrays, prep_frames, exec_frames, active, reaction), "bool")
    put("reaction_or_chamber_arm_side_match", active in {"left", "right"} and reaction in {"left", "right"} and active != reaction, "bool")
    put("shoulder_elevation_body_ratio", _median_scalar(arrays, fix_frames, lambda a, f: _shoulder_elevation(a, f, active)), "torso_length")
    put("shoulder_abduction_deg", _median_scalar(arrays, fix_frames, lambda a, f: _shoulder_angle(a, f, active, "abduction")), "deg")
    put("shoulder_flexion_deg", _median_scalar(arrays, fix_frames, lambda a, f: _shoulder_angle(a, f, active, "flexion")), "deg")
    elbow = _median_scalar(arrays, fix_frames, lambda a, f: _arm_angle(a, f, active))
    put("elbow_flexion_deg", elbow, "deg")
    put("elbow_target_angle_error_deg", None, "deg", reason="missing_numeric_technique_target")
    put("active_arm_extension_ratio", _arm_extension(arrays, fix_frames, active), "arm_length")
    local_hand = _hand_local_coordinates(arrays, fix_frames, active)
    put("active_arm_target_height_body_ratio", local_hand.get("height"), "torso_length")
    put("active_arm_target_lateral_offset_body_ratio", local_hand.get("lateral"), "shoulder_width")
    put("active_arm_target_depth_body_ratio", local_hand.get("depth"), "arm_length")
    put("active_hand_target_distance_body_ratio", local_hand.get("distance"), "body_ratio")
    put("reaction_hand_target_distance_body_ratio", _reaction_hip_distance(arrays, fix_frames, reaction), "body_ratio")
    put("reaction_elbow_position_error_body_ratio", _reaction_elbow_distance(arrays, fix_frames, reaction), "arm_length")
    put("arm_target_overshoot_body_ratio", _trajectory_overshoot(arrays, transition_frames, active, active_arm_scale), "arm_length")

    wrist_components = _wrist_components(arrays, fix_frames, active)
    put("wrist_forearm_alignment_deg", wrist_components.get("total"), "deg")
    put("wrist_flexion_extension_proxy_deg", wrist_components.get("flexion"), "deg")
    put("wrist_radial_ulnar_deviation_proxy_deg", wrist_components.get("radial"), "deg")
    put("fist_or_hand_orientation_proxy_deg", wrist_components.get("orientation"), "deg")
    put("hand_shape_observability", hand_quality is not None and hand_quality >= 0.75, "bool")
    put("active_hand_fixation_stability", _hand_dispersion(arrays, after_frames, active, active_arm_scale), "arm_length")
    put("reaction_hand_fixation_stability", _hand_dispersion(arrays, after_frames, reaction, _median_scalar(arrays, fix_frames, lambda a, f: _limb_scale(a, f, "arm", reaction))), "arm_length")
    put("fist_target_height_error_body_ratio", _technique_target_error(arrays, fix_frames, contract, active, "height"), "torso_length")
    put("fist_target_lateral_error_body_ratio", _technique_target_error(arrays, fix_frames, contract, active, "lateral"), "shoulder_width")
    put("fist_target_depth_error_body_ratio", None, "arm_length", reason="missing_numeric_technique_depth_target")

    put("preparation_pose_observed", "preparation" in anchors, "bool")
    put("preparation_side_match", _arm_side_match(arrays, prep_frames, exec_frames, active, reaction), "bool")
    put("chamber_pose_observed", _reaction_hip_distance(arrays, prep_frames, reaction) is not None, "bool")
    put("chamber_precedes_execution", preparation <= execution, "bool")
    put("active_technique_reaches_target_by_fixation", _hand_settled(arrays, after_frames, active, active_arm_scale), "bool")
    put("stance_reaches_target_by_fixation", stance_match, "bool")
    put("head_reaches_target_by_fixation", _component_settled(arrays, after_frames, "head", 10.0), "bool")
    put("torso_reaches_target_by_fixation", _component_settled(arrays, after_frames, "shoulder", 8.0), "bool")
    put("reaction_hand_reaches_target_by_fixation", _reaction_hip_distance(arrays, fix_frames, reaction) is not None, "bool")
    put("head_torso_settle_offset", _settle_offset(arrays, transition_frames, "head", "shoulder", fps), "sec")
    put("late_post_fixation_correction", _late_component_correction(arrays, after_frames, active, active_arm_scale), "arm_length")
    put("premature_next_movement_transition", _premature_transition(arrays, after_frames, active, active_arm_scale), "bool")
    put("fixation_pose_stability", _fixation_stable(arrays, after_frames, active, lead_leg_scale, active_arm_scale), "bool")
    put("final_pose_geometry_conformance", _finite_value(stance_match) and _finite_value(elbow), "bool")

    put("step_length_error_body_ratio", _stance_range_component_error(contract["stance_length_expectation"], _safe_ratio(stance.get("length"), lead_leg_scale)), "leg_length")
    put("step_width_error_body_ratio", _stance_range_component_error(contract["stance_width_expectation"], _safe_ratio(stance.get("width"), shoulder_width)), "shoulder_width")
    put("landing_position_error_body_ratio", result["foot_landing_position_error_body_ratio"]["value"], "leg_length")
    put("landing_stance_type_match", stance_match, "bool")
    put("head_turn_settled_by_fixation", _component_settled(arrays, after_frames, "head", 10.0), "bool")
    put("torso_turn_settled_by_fixation", _component_settled(arrays, after_frames, "shoulder", 8.0), "bool")
    put("pelvis_or_stance_turn_settled_by_fixation", _component_settled(arrays, after_frames, "hip", 8.0), "bool")
    put("pivot_foot_excess_displacement", rear_displacement, "leg_length")
    put("transition_foot_crossing", (stance.get("crossing_margin") or 0.0) < 0.0 if stance.get("crossing_margin") is not None else None, "bool")
    put("post_landing_adjustment", _lower_dispersion(arrays, after_frames, lead_leg_scale), "leg_length")

    head_disp = _orientation_dispersion(arrays, after_frames, "head")
    torso_disp = _orientation_dispersion(arrays, after_frames, "shoulder")
    pelvis_disp = _orientation_dispersion(arrays, after_frames, "hip")
    put("head_fixation_dispersion", head_disp, "deg")
    put("torso_fixation_dispersion", torso_disp, "deg")
    put("pelvis_fixation_dispersion", pelvis_disp, "deg")
    put("active_hand_fixation_dispersion", result["active_hand_fixation_stability"]["value"], "arm_length")
    put("reaction_hand_fixation_dispersion", result["reaction_hand_fixation_stability"]["value"], "arm_length")
    put("front_foot_fixation_dispersion", front_foot_dispersion, "leg_length")
    put("rear_foot_fixation_dispersion", rear_foot_dispersion, "leg_length")
    put("knee_fixation_dispersion", _knee_dispersion(arrays, after_frames, lead_leg_scale), "leg_length")
    put("body_height_fixation_drift", _body_height_change(arrays, after_frames), "body_ratio")
    put("torso_translation_after_fixation", _centre_dispersion(arrays, after_frames, "shoulder"), "body_ratio")
    put("pelvis_translation_after_fixation", _centre_dispersion(arrays, after_frames, "hip"), "body_ratio")
    put("stance_width_after_fixation_change", _stance_change(arrays, after_frames, lead, rear, "width", shoulder_width), "shoulder_width")
    put("stance_length_after_fixation_change", _stance_change(arrays, after_frames, lead, rear, "length", lead_leg_scale), "leg_length")
    put("late_balance_recovery_proxy", _centre_dispersion(arrays, after_frames, "hip"), "body_ratio")
    put("final_geometry_consistent_with_declared_movement", bool(stance_match) if stance_match is not None else None, "bool")
    put("movement_completion_state", fixation <= end and "fixation" in anchors, "bool")

    put("head_turn_completion_state", _component_settled(arrays, after_frames, "head", 10.0), "bool")
    put("head_late_turn_state", not bool(result["head_turn_completion_state"]["value"]) if result["head_turn_completion_state"]["value"] is not None else None, "bool")

    _direction_metrics(result, put, arrays, contract, fix_frames, direction, active)
    _kick_metrics(result, put, arrays, contract, anchors, start, end, kick_side, rear, lead_leg_scale)
    _apply_quality_gates(
        result,
        profile,
        {
            "head": head_quality,
            "torso": torso_quality,
            "lower": lower_quality,
            "upper": upper_quality,
            "active_hand": hand_quality,
            "reaction_hand": reaction_hand_quality,
            "foot": foot_quality,
            "fixation": _group_quality(arrays, after_frames, tuple(range(17))),
        },
    )
    return result


def measure_athlete_forward_vector(
    arrays: dict[str, Any],
    frames: np.ndarray,
    min_valid_samples: int,
) -> np.ndarray | None:
    """Median horizontal facing of the athlete over a window; None when evidence is thin.

    The per-frame estimate is the existing torso-forward geometry: the shoulder line
    crossed with world up, sign-resolved by the observed face direction. This measures
    where the athlete faced; it does not claim any world or judging reference.
    """
    vectors = [
        value
        for frame in frames
        if (value := _torso_forward(arrays, int(frame))) is not None
    ]
    if len(vectors) < max(3, int(min_valid_samples)):
        return None
    return _unit(np.median(np.asarray(vectors, dtype=float), axis=0))


def _direction_metrics(result: dict[str, dict[str, Any]], put: Callable[..., None], arrays: dict[str, Any], contract: dict[str, Any], frames: np.ndarray, direction: dict[str, Any] | None, active: str | None) -> None:
    ids = (
        "head_target_yaw_error_deg", "torso_target_yaw_error_deg", "pelvis_target_yaw_error_deg",
        "stance_axis_target_yaw_error_deg", "front_foot_target_yaw_error_deg", "rear_foot_target_yaw_error_deg",
        "foot_progression_axis_error_deg", "knee_target_direction_error_deg", "active_arm_target_direction_error_deg",
        "fist_target_direction_error_deg", "expected_direction_change_match", "rotation_direction_sign_match",
        "step_direction_error_deg", "head_turn_direction_sign_match", "head_wrong_direction_stable_state",
        "kick_direction_error_deg", "expected_direction_contract_resolved",
    )
    if direction is None:
        for metric_id in ids:
            put(metric_id, None, "deg" if metric_id.endswith("_deg") else "bool", reason="missing_athlete_local_direction_binding")
        return
    target = np.asarray(direction[contract["semantic_action_direction"] + "_vector"], dtype=float)
    vectors = {
        "head_target_yaw_error_deg": _median_vector(arrays, frames, _head_forward),
        "torso_target_yaw_error_deg": _median_vector(arrays, frames, _torso_forward),
        "pelvis_target_yaw_error_deg": _median_vector(arrays, frames, _pelvis_forward),
        "stance_axis_target_yaw_error_deg": _median_vector(arrays, frames, lambda a, f: _stance_axis(a, f, contract["lead_leg"], contract["rear_or_support_leg"])),
        "front_foot_target_yaw_error_deg": _median_vector(arrays, frames, lambda a, f: _foot_axis(a, f, contract["lead_leg"])),
        "rear_foot_target_yaw_error_deg": _median_vector(arrays, frames, lambda a, f: _foot_axis(a, f, contract["rear_or_support_leg"])),
        "foot_progression_axis_error_deg": _median_vector(arrays, frames, lambda a, f: _stance_axis(a, f, contract["lead_leg"], contract["rear_or_support_leg"])),
        "knee_target_direction_error_deg": _median_vector(arrays, frames, lambda a, f: _knee_axis(a, f, contract["lead_leg"])),
        "active_arm_target_direction_error_deg": _median_vector(arrays, frames, lambda a, f: _arm_axis(a, f, active)),
        "fist_target_direction_error_deg": _median_vector(arrays, frames, lambda a, f: _hand_axis(a, f, active)),
    }
    for metric_id, vector in vectors.items():
        put(metric_id, _horizontal_angle(vector, target), "deg")
    direction_ok = result["stance_axis_target_yaw_error_deg"]["value"]
    put("expected_direction_change_match", direction_ok is not None and direction_ok <= 25.0, "bool")
    put("rotation_direction_sign_match", direction_ok is not None and direction_ok <= 90.0, "bool")
    put("step_direction_error_deg", direction_ok, "deg")
    head_error = result["head_target_yaw_error_deg"]["value"]
    put("head_turn_direction_sign_match", head_error is not None and head_error <= 90.0, "bool")
    put("head_wrong_direction_stable_state", head_error is not None and head_error > 90.0, "bool")
    put("expected_direction_contract_resolved", True, "bool")


def _apply_quality_gates(
    result: dict[str, dict[str, Any]],
    profile: dict[str, Any],
    quality: dict[str, float | None],
) -> None:
    family_by_metric = {
        metric_id: family
        for family, metric_ids in profile["metric_catalog"].items()
        for metric_id in metric_ids
    }
    minimum = float(profile["quality_gates"]["min_group_valid_ratio"])
    quality_metrics = {
        "head_face_geometry_quality",
        "torso_geometry_quality",
        "stance_geometry_quality",
        "lower_body_geometry_quality",
        "upper_body_geometry_quality",
        "hand_geometry_quality",
        "phase_evidence_quality",
        "transition_geometry_quality",
        "fixation_evidence_quality",
        "head_orientation_unmeasurable_reason",
    }
    contract_only = {
        "required_phase_coverage",
        "preparation_pose_observed",
        "chamber_precedes_execution",
        "movement_contract_exists",
        "movement_timeline_segment_exists",
        "expected_movement_order",
        "movement_phase_order",
        "expected_technique_contract_resolved",
        "expected_stance_contract_resolved",
        "expected_active_side_contract_resolved",
        "expected_direction_contract_resolved",
        "movement_completion_state",
    }
    for metric_id, measurement in result.items():
        if metric_id in quality_metrics or metric_id in contract_only or measurement["value"] is None:
            continue
        ratio = _quality_for_metric(metric_id, family_by_metric.get(metric_id), quality)
        measurement["evidence"]["required_group_valid_ratio"] = ratio
        measurement["evidence"]["minimum_group_valid_ratio"] = minimum
        if ratio is None or ratio < minimum:
            measurement["value"] = None
            measurement["quality_status"] = "unmeasurable"
            measurement["not_measurable_reason"] = "required_landmark_group_quality_below_minimum"


def _quality_for_metric(
    metric_id: str,
    family: str | None,
    quality: dict[str, float | None],
) -> float | None:
    if any(token in metric_id for token in ("reaction_hand", "reaction_or_chamber", "chamber_pose")):
        return quality["reaction_hand"]
    if any(token in metric_id for token in ("hand", "fist", "wrist_flexion", "wrist_radial")):
        return quality["active_hand"]
    if any(token in metric_id for token in ("foot", "stance", "landing", "pivot", "step_")):
        return _minimum_finite(quality["foot"], quality["lower"])
    if any(token in metric_id for token in ("knee", "leg", "kick")) or family == "lower_body":
        return quality["lower"]
    if metric_id.startswith("head_") or metric_id == "head_fixation_dispersion":
        return quality["head"]
    if any(token in metric_id for token in ("arm", "elbow", "shoulder", "wrist_forearm")) or family == "upper_body":
        return quality["upper"]
    if any(token in metric_id for token in ("torso", "pelvis", "body_height", "balance")) or family == "torso_pelvis":
        return quality["torso"]
    if family == "fixation":
        return quality["fixation"]
    if family in {"phase_structure", "transition"}:
        return _minimum_finite(quality["upper"], quality["lower"])
    return 1.0


def _kick_metrics(result: dict[str, dict[str, Any]], put: Callable[..., None], arrays: dict[str, Any], contract: dict[str, Any], anchors: dict[str, int], start: int, end: int, kick_side: str | None, support_side: str | None, leg_scale: float | None) -> None:
    if "ap_chagi" not in contract["technique_types"] or kick_side not in {"left", "right"}:
        return
    chamber = int(anchors.get("preparation", start))
    apex = int(anchors.get("kick_apex", anchors.get("kick_execution", chamber)))
    rechamber = int(anchors.get("rechamber", apex))
    landing = int(anchors.get("landing", end))
    put("chamber_knee_height_body_ratio", _knee_height(arrays, chamber, kick_side, leg_scale), "leg_length")
    put("chamber_hip_flexion_deg", _hip_flexion(arrays, chamber, kick_side), "deg")
    put("chamber_knee_flexion_deg", _leg_angle(arrays, chamber, kick_side), "deg")
    put("support_leg_stability", _joint_span(arrays, np.arange(chamber, apex + 1), f"{support_side}_ankle", leg_scale), "leg_length")
    put("kick_extension_deg", _leg_angle(arrays, apex, kick_side), "deg")
    put("kick_target_height_body_ratio", _ankle_height(arrays, apex, kick_side, leg_scale), "leg_length")
    put("kick_retraction_state", _leg_angle(arrays, rechamber, kick_side) is not None, "bool")
    put("kick_landing_stance_restoration", _pt(arrays, landing, f"{kick_side}_ankle") is not None, "bool")


def _window(anchor: int, start: int, end: int, radius: int) -> np.ndarray:
    return np.arange(max(start, anchor - radius), min(end, anchor + radius) + 1, dtype=int)


def _finite_value(value: Any) -> bool:
    if isinstance(value, (bool, str)):
        return True
    if value is None:
        return False
    try:
        return bool(np.all(np.isfinite(value)))
    except TypeError:
        return False


def _pt(arrays: dict[str, Any], frame: int, name: str) -> np.ndarray | None:
    value = arrays["points"][frame, COCO_BODY_JOINTS[name]]
    return value if np.all(np.isfinite(value)) else None


def _idx(arrays: dict[str, Any], frame: int, index: int) -> np.ndarray | None:
    value = arrays["points"][frame, index]
    return value if np.all(np.isfinite(value)) else None


def _median_scalar(arrays: dict[str, Any], frames: np.ndarray, fn: Callable[[dict[str, Any], int], float | None]) -> float | None:
    values = [value for frame in frames if (value := fn(arrays, int(frame))) is not None and np.isfinite(value)]
    return None if len(values) < 3 else float(np.median(values))


def _median_vector(arrays: dict[str, Any], frames: np.ndarray, fn: Callable[[dict[str, Any], int], np.ndarray | None]) -> np.ndarray | None:
    values = [value for frame in frames if (value := fn(arrays, int(frame))) is not None]
    if len(values) < 3:
        return None
    vector = np.median(np.asarray(values), axis=0)
    return _unit(vector)


def _unit(vector: np.ndarray | None) -> np.ndarray | None:
    if vector is None:
        return None
    norm = float(np.linalg.norm(vector))
    return None if norm <= 1e-8 else vector / norm


def _body_scale(arrays: dict[str, Any], frame: int) -> float | None:
    sc, hc = _centres(arrays, frame)
    return None if sc is None or hc is None else _positive_norm(sc - hc)


def _shoulder_width(arrays: dict[str, Any], frame: int) -> float | None:
    left, right = _pt(arrays, frame, "left_shoulder"), _pt(arrays, frame, "right_shoulder")
    return None if left is None or right is None else _positive_norm(right - left)


def _limb_scale(arrays: dict[str, Any], frame: int, kind: str, side: str | None) -> float | None:
    if side not in {"left", "right"}:
        return None
    parts = ("shoulder", "elbow", "wrist") if kind == "arm" else ("hip", "knee", "ankle")
    points = [_pt(arrays, frame, f"{side}_{part}") for part in parts]
    if any(point is None for point in points):
        return None
    return _positive_norm(points[0] - points[1]) + _positive_norm(points[1] - points[2])


def _positive_norm(vector: np.ndarray) -> float | None:
    value = float(np.linalg.norm(vector))
    return value if value > 1e-8 else None


def _safe_ratio(value: float | None, scale: float | None) -> float | None:
    return None if value is None or scale is None or scale <= 1e-8 else float(value / scale)


def _centres(arrays: dict[str, Any], frame: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    ls, rs, lh, rh = (_pt(arrays, frame, name) for name in ("left_shoulder", "right_shoulder", "left_hip", "right_hip"))
    if any(point is None for point in (ls, rs, lh, rh)):
        return None, None
    return (ls + rs) / 2.0, (lh + rh) / 2.0


def _group_quality(arrays: dict[str, Any], frames: np.ndarray, indices: tuple[int, ...] | list[int]) -> float | None:
    if not indices or len(frames) == 0:
        return None
    return float(np.mean(arrays["quality_mask"][np.ix_(frames, np.asarray(indices, dtype=int))]))


def _head_indices() -> tuple[int, ...]:
    return (*range(23 + 17, 23 + 27), *range(23 + 36, 23 + 48), 23 + 8)


def _torso_indices() -> tuple[int, ...]:
    return tuple(COCO_BODY_JOINTS[name] for name in ("left_shoulder", "right_shoulder", "left_hip", "right_hip"))


def _lower_indices() -> tuple[int, ...]:
    return tuple(COCO_BODY_JOINTS[name] for name in ("left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"))


def _upper_indices() -> tuple[int, ...]:
    return tuple(
        COCO_BODY_JOINTS[name]
        for name in (
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
        )
    )


def _hand_indices(side: str | None) -> tuple[int, ...]:
    if side not in {"left", "right"}:
        return ()
    offset = COCO_LEFT_HAND_OFFSET if side == "left" else COCO_RIGHT_HAND_OFFSET
    return tuple(range(offset, offset + 21))


def _foot_indices(side: str | None) -> tuple[int, ...]:
    if side not in {"left", "right"}:
        return ()
    return tuple(COCO_FOOT_JOINTS[f"{side}_{name}"] for name in ("big_toe", "small_toe", "heel"))


def _phase_quality(anchors: dict[str, int]) -> float:
    required = {"preparation", "fixation"}
    execution = any(name in anchors for name in ("execution", "punch_execution", "kick_apex"))
    return (len(required & set(anchors)) + int(execution)) / 3.0


def _minimum_finite(*values: float | None) -> float | None:
    finite = [value for value in values if value is not None]
    return min(finite) if len(finite) == len(values) else None


def _orientation_line(arrays: dict[str, Any], frame: int, part: str) -> np.ndarray | None:
    if part == "head":
        left = _cluster(arrays, frame, range(36, 42))
        right = _cluster(arrays, frame, range(42, 48))
    else:
        left, right = _pt(arrays, frame, f"left_{part}"), _pt(arrays, frame, f"right_{part}")
    return None if left is None or right is None else _unit(right - left)


def _cluster(arrays: dict[str, Any], frame: int, local: range | tuple[int, ...]) -> np.ndarray | None:
    values = arrays["points"][frame, [23 + index for index in local]]
    valid = values[np.all(np.isfinite(values), axis=1)]
    return None if len(valid) == 0 else np.median(valid, axis=0)


def _orientation_dispersion(arrays: dict[str, Any], frames: np.ndarray, part: str) -> float | None:
    vectors = [value for frame in frames if (value := _orientation_line(arrays, int(frame), part)) is not None]
    if len(vectors) < 3:
        return None
    centre = _unit(np.median(np.asarray(vectors), axis=0))
    angles = [_angle(vector, centre) for vector in vectors]
    return None if centre is None else float(np.percentile(angles, 95))


def _orientation_drift(arrays: dict[str, Any], frames: np.ndarray, part: str) -> float | None:
    if len(frames) < 2:
        return None
    return _angle(_orientation_line(arrays, int(frames[0]), part), _orientation_line(arrays, int(frames[-1]), part))


def _angle(first: np.ndarray | None, second: np.ndarray | None) -> float | None:
    if first is None or second is None:
        return None
    denom = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denom <= 1e-8:
        return None
    return float(np.degrees(np.arccos(np.clip(np.dot(first, second) / denom, -1.0, 1.0))))


def _horizontal_angle(first: np.ndarray | None, second: np.ndarray | None) -> float | None:
    if first is None or second is None:
        return None
    return _angle(_unit(np.asarray([first[0], first[1], 0.0])), _unit(np.asarray([second[0], second[1], 0.0])))


def _torso_forward_lean(arrays: dict[str, Any], frame: int) -> float | None:
    sc, hc = _centres(arrays, frame)
    forward = _torso_forward(arrays, frame)
    if sc is None or hc is None or forward is None:
        return None
    torso = sc - hc
    return float(np.degrees(np.arctan2(np.dot(torso, forward), torso[2])))


def _line_roll(arrays: dict[str, Any], frame: int, part: str) -> float | None:
    line = _orientation_line(arrays, frame, part)
    if line is None:
        return None
    return float(np.degrees(np.arctan2(line[2], np.linalg.norm(line[:2]))))


def _roll_difference(arrays: dict[str, Any], frame: int) -> float | None:
    shoulder, hip = _line_roll(arrays, frame, "shoulder"), _line_roll(arrays, frame, "hip")
    return None if shoulder is None or hip is None else abs(shoulder - hip)


def _height_asymmetry(arrays: dict[str, Any], frame: int, part: str) -> float | None:
    left, right = _pt(arrays, frame, f"left_{part}"), _pt(arrays, frame, f"right_{part}")
    scale = _body_scale(arrays, frame)
    return None if left is None or right is None else _safe_ratio(abs(float(left[2] - right[2])), scale)


def _centre_dispersion(arrays: dict[str, Any], frames: np.ndarray, part: str) -> float | None:
    values: list[np.ndarray] = []
    scales: list[float] = []
    for frame in frames:
        left, right = _pt(arrays, int(frame), f"left_{part}"), _pt(arrays, int(frame), f"right_{part}")
        scale = _body_scale(arrays, int(frame))
        if left is not None and right is not None and scale is not None:
            values.append((left + right) / 2.0)
            scales.append(scale)
    if len(values) < 3:
        return None
    centre = np.median(np.asarray(values), axis=0)
    return float(np.percentile(np.linalg.norm(np.asarray(values) - centre, axis=1) / scales, 95))


def _body_height_change(arrays: dict[str, Any], frames: np.ndarray) -> float | None:
    values: list[float] = []
    for frame in frames:
        sc, _ = _centres(arrays, int(frame))
        la, ra = _pt(arrays, int(frame), "left_ankle"), _pt(arrays, int(frame), "right_ankle")
        scale = _body_scale(arrays, int(frame))
        if sc is not None and la is not None and ra is not None and scale is not None:
            values.append(float((sc[2] - ((la + ra) / 2.0)[2]) / scale))
    return None if len(values) < 3 else float(max(values) - min(values))


def _stance_geometry(arrays: dict[str, Any], frames: np.ndarray, lead: str | None, rear: str | None) -> dict[str, Any]:
    if lead not in {"left", "right"} or rear not in {"left", "right"}:
        return {}
    rows: list[dict[str, float]] = []
    for frame in frames:
        front, back = _foot_centre(arrays, int(frame), lead), _foot_centre(arrays, int(frame), rear)
        lateral = _orientation_line(arrays, int(frame), "shoulder")
        if front is None or back is None or lateral is None:
            continue
        lateral = _unit(np.asarray([lateral[0], lateral[1], 0.0]))
        forward = _unit(np.cross(np.asarray([0.0, 0.0, 1.0]), lateral))
        delta = front - back
        lead_heel, rear_heel = _foot_part(arrays, int(frame), lead, "heel"), _foot_part(arrays, int(frame), rear, "heel")
        lead_toe, rear_toe = _toe_centre(arrays, int(frame), lead), _toe_centre(arrays, int(frame), rear)
        if forward is None:
            continue
        rows.append({
            "length": abs(float(np.dot(delta, forward))),
            "width": abs(float(np.dot(delta, lateral))),
            "crossing_margin": float(np.dot(delta, lateral)),
            "heel_alignment": abs(float(np.dot(lead_heel - rear_heel, lateral))) if lead_heel is not None and rear_heel is not None else np.nan,
            "toe_alignment": abs(float(np.dot(lead_toe - rear_toe, lateral))) if lead_toe is not None and rear_toe is not None else np.nan,
            "front_back_order_match": float(np.dot(delta, forward)) >= 0.0,
        })
    if len(rows) < 3:
        return {}
    return {key: (bool(np.mean([row[key] for row in rows]) >= 0.5) if key == "front_back_order_match" else float(np.nanmedian([row[key] for row in rows]))) for key in rows[0]}


def _foot_part(arrays: dict[str, Any], frame: int, side: str, part: str) -> np.ndarray | None:
    return _idx(arrays, frame, COCO_FOOT_JOINTS[f"{side}_{part}"])


def _toe_centre(arrays: dict[str, Any], frame: int, side: str) -> np.ndarray | None:
    big, small = _foot_part(arrays, frame, side, "big_toe"), _foot_part(arrays, frame, side, "small_toe")
    return None if big is None or small is None else (big + small) / 2.0


def _foot_centre(arrays: dict[str, Any], frame: int, side: str) -> np.ndarray | None:
    heel, toe = _foot_part(arrays, frame, side, "heel"), _toe_centre(arrays, frame, side)
    return None if heel is None or toe is None else (heel + toe) / 2.0


def _foot_axis(arrays: dict[str, Any], frame: int, side: str | None) -> np.ndarray | None:
    if side not in {"left", "right"}:
        return None
    heel, toe = _foot_part(arrays, frame, side, "heel"), _toe_centre(arrays, frame, side)
    return None if heel is None or toe is None else _unit(toe - heel)


def _foot_yaw_difference(arrays: dict[str, Any], frames: np.ndarray, lead: str | None, rear: str | None) -> float | None:
    if lead not in {"left", "right"} or rear not in {"left", "right"}:
        return None
    return _median_scalar(arrays, frames, lambda a, f: _horizontal_angle(_foot_axis(a, f, lead), _foot_axis(a, f, rear)))


def _range_match(value: float | None, limits: list[float] | None) -> bool | None:
    return None if value is None or limits is None else bool(float(limits[0]) <= value <= float(limits[1]))


def _stance_range_component_error(limits: list[float] | None, value: float | None) -> float | None:
    if limits is None or value is None:
        return None
    return 0.0 if limits[0] <= value <= limits[1] else float(min(abs(value - limits[0]), abs(value - limits[1])))


def _stance_range_error(contract: dict[str, Any], stance: dict[str, Any], leg: float | None, shoulder: float | None) -> float | None:
    length = _stance_range_component_error(contract["stance_length_expectation"], _safe_ratio(stance.get("length"), leg))
    width = _stance_range_component_error(contract["stance_width_expectation"], _safe_ratio(stance.get("width"), shoulder))
    return None if length is None or width is None else float(np.hypot(length, width))


def _side_displacement_match(expected: str | None, measured_side: str | None, expected_disp: float | None, other_disp: float | None) -> bool | None:
    if expected is None:
        return True
    if expected != measured_side or expected_disp is None or other_disp is None:
        return None
    return expected_disp >= other_disp


def _joint_displacement(arrays: dict[str, Any], before: np.ndarray, after: np.ndarray, joint: str, scale: float | None) -> float | None:
    first = _median_point(arrays, before, joint)
    last = _median_point(arrays, after, joint)
    return None if first is None or last is None else _safe_ratio(float(np.linalg.norm(last - first)), scale)


def _median_point(arrays: dict[str, Any], frames: np.ndarray, name: str) -> np.ndarray | None:
    values = [value for frame in frames if (value := _pt(arrays, int(frame), name)) is not None]
    return None if len(values) < 3 else np.median(np.asarray(values), axis=0)


def _pelvis_height_leg_ratio(arrays: dict[str, Any], frames: np.ndarray, leg_scale: float | None) -> float | None:
    values = []
    for frame in frames:
        _, hip = _centres(arrays, int(frame))
        la, ra = _pt(arrays, int(frame), "left_ankle"), _pt(arrays, int(frame), "right_ankle")
        if hip is not None and la is not None and ra is not None:
            values.append(float(hip[2] - ((la + ra) / 2.0)[2]))
    return None if len(values) < 3 else _safe_ratio(float(np.median(values)), leg_scale)


def _leg_angle(arrays: dict[str, Any], frame: int, side: str | None) -> float | None:
    if side not in {"left", "right"}:
        return None
    hip, knee, ankle = (_pt(arrays, frame, f"{side}_{part}") for part in ("hip", "knee", "ankle"))
    return None if hip is None or knee is None or ankle is None else _angle(hip - knee, ankle - knee)


def _arm_angle(arrays: dict[str, Any], frame: int, side: str | None) -> float | None:
    if side not in {"left", "right"}:
        return None
    shoulder, elbow, wrist = (_pt(arrays, frame, f"{side}_{part}") for part in ("shoulder", "elbow", "wrist"))
    return None if shoulder is None or elbow is None or wrist is None else _angle(shoulder - elbow, wrist - elbow)


def _knee_foot_alignment(arrays: dict[str, Any], frame: int, side: str | None) -> float | None:
    if side not in {"left", "right"}:
        return None
    knee, foot = _pt(arrays, frame, f"{side}_knee"), _foot_centre(arrays, frame, side)
    scale = _limb_scale(arrays, frame, "leg", side)
    return None if knee is None or foot is None else _safe_ratio(float(np.linalg.norm(knee[:2] - foot[:2])), scale)


def _knee_plane_angle(arrays: dict[str, Any], frame: int, side: str | None) -> float | None:
    if side not in {"left", "right"}:
        return None
    hip, knee, ankle = (_pt(arrays, frame, f"{side}_{part}") for part in ("hip", "knee", "ankle"))
    if hip is None or knee is None or ankle is None:
        return None
    return _horizontal_angle(knee - hip, ankle - knee)


def _knee_plane_deviation(arrays: dict[str, Any], frame: int, side: str | None) -> float | None:
    if side not in {"left", "right"}:
        return None
    hip, knee, ankle = (_pt(arrays, frame, f"{side}_{part}") for part in ("hip", "knee", "ankle"))
    scale = _limb_scale(arrays, frame, "leg", side)
    if hip is None or knee is None or ankle is None or scale is None:
        return None
    line = ankle - hip
    denominator = float(np.dot(line, line))
    if denominator <= 1e-8:
        return None
    nearest = hip + np.dot(knee - hip, line) / denominator * line
    return float(np.linalg.norm(knee - nearest) / scale)


def _lower_dispersion(arrays: dict[str, Any], frames: np.ndarray, scale: float | None) -> float | None:
    values = [_joint_span(arrays, frames, name, scale) for name in ("left_knee", "right_knee", "left_ankle", "right_ankle")]
    return max(values) if all(value is not None for value in values) else None


def _joint_span(arrays: dict[str, Any], frames: np.ndarray, name: str, scale: float | None) -> float | None:
    points = [value for frame in frames if (value := _pt(arrays, int(frame), name)) is not None]
    if len(points) < 3 or scale is None:
        return None
    centre = np.median(np.asarray(points), axis=0)
    return float(np.percentile(np.linalg.norm(np.asarray(points) - centre, axis=1), 95) / scale)


def _arm_side_match(arrays: dict[str, Any], before: np.ndarray, after: np.ndarray, active: str | None, reaction: str | None) -> bool | None:
    if active not in {"left", "right"} or reaction not in {"left", "right"}:
        return None
    scale = _median_scalar(arrays, after, lambda a, f: _limb_scale(a, f, "arm", active))
    active_move = _joint_displacement(arrays, before, after, f"{active}_wrist", scale)
    reaction_move = _joint_displacement(arrays, before, after, f"{reaction}_wrist", scale)
    return None if active_move is None or reaction_move is None else active_move >= reaction_move


def _shoulder_elevation(arrays: dict[str, Any], frame: int, side: str | None) -> float | None:
    if side not in {"left", "right"}:
        return None
    shoulder = _pt(arrays, frame, f"{side}_shoulder")
    other = _pt(arrays, frame, f"{'right' if side == 'left' else 'left'}_shoulder")
    scale = _body_scale(arrays, frame)
    return None if shoulder is None or other is None else _safe_ratio(float(shoulder[2] - other[2]), scale)


def _shoulder_angle(arrays: dict[str, Any], frame: int, side: str | None, component: str) -> float | None:
    if side not in {"left", "right"}:
        return None
    shoulder, elbow = _pt(arrays, frame, f"{side}_shoulder"), _pt(arrays, frame, f"{side}_elbow")
    sc, hc = _centres(arrays, frame)
    if shoulder is None or elbow is None or sc is None or hc is None:
        return None
    arm = elbow - shoulder
    if component == "abduction":
        return _angle(arm, sc - hc)
    forward = _torso_forward(arrays, frame)
    return _angle(arm, forward)


def _arm_extension(arrays: dict[str, Any], frames: np.ndarray, side: str | None) -> float | None:
    values = []
    for frame in frames:
        if side not in {"left", "right"}:
            continue
        shoulder, wrist = _pt(arrays, int(frame), f"{side}_shoulder"), _pt(arrays, int(frame), f"{side}_wrist")
        scale = _limb_scale(arrays, int(frame), "arm", side)
        if shoulder is not None and wrist is not None and scale is not None:
            values.append(float(np.linalg.norm(wrist - shoulder) / scale))
    return None if len(values) < 3 else float(np.median(values))


def _hand_offset(side: str) -> int:
    return COCO_LEFT_HAND_OFFSET if side == "left" else COCO_RIGHT_HAND_OFFSET


def _hand_point(arrays: dict[str, Any], frame: int, side: str | None, joint: str) -> np.ndarray | None:
    if side not in {"left", "right"}:
        return None
    return _idx(arrays, frame, _hand_offset(side) + COCO_HAND_LOCAL_JOINTS[joint])


def _hand_centre(arrays: dict[str, Any], frame: int, side: str | None) -> np.ndarray | None:
    points = [_hand_point(arrays, frame, side, name) for name in ("wrist", "index_mcp", "middle_mcp", "ring_mcp", "pinky_mcp")]
    return None if any(point is None for point in points) else np.mean(points, axis=0)


def _hand_local_coordinates(arrays: dict[str, Any], frames: np.ndarray, side: str | None) -> dict[str, float | None]:
    rows = []
    for frame in frames:
        hand = _hand_centre(arrays, int(frame), side)
        sc, hc = _centres(arrays, int(frame))
        lateral = _orientation_line(arrays, int(frame), "shoulder")
        forward = _torso_forward(arrays, int(frame))
        torso = _body_scale(arrays, int(frame))
        shoulder = _shoulder_width(arrays, int(frame))
        arm = _limb_scale(arrays, int(frame), "arm", side)
        if hand is None or sc is None or hc is None or lateral is None or forward is None or torso is None or shoulder is None or arm is None:
            continue
        delta = hand - hc
        rows.append((float(delta[2] / torso), float(np.dot(delta, lateral) / shoulder), float(np.dot(delta, forward) / arm), float(np.linalg.norm(delta) / torso)))
    if len(rows) < 3:
        return {key: None for key in ("height", "lateral", "depth", "distance")}
    values = np.median(np.asarray(rows), axis=0)
    return dict(zip(("height", "lateral", "depth", "distance"), map(float, values), strict=True))


def _reaction_hip_distance(arrays: dict[str, Any], frames: np.ndarray, side: str | None) -> float | None:
    values = []
    for frame in frames:
        hand, hip = _hand_centre(arrays, int(frame), side), _pt(arrays, int(frame), f"{side}_hip") if side in {"left", "right"} else None
        scale = _body_scale(arrays, int(frame))
        if hand is not None and hip is not None and scale is not None:
            values.append(float(np.linalg.norm(hand - hip) / scale))
    return None if len(values) < 3 else float(np.median(values))


def _reaction_elbow_distance(arrays: dict[str, Any], frames: np.ndarray, side: str | None) -> float | None:
    values = []
    for frame in frames:
        elbow, hip = _pt(arrays, int(frame), f"{side}_elbow") if side in {"left", "right"} else None, _pt(arrays, int(frame), f"{side}_hip") if side in {"left", "right"} else None
        scale = _limb_scale(arrays, int(frame), "arm", side)
        if elbow is not None and hip is not None and scale is not None:
            values.append(float(np.linalg.norm(elbow - hip) / scale))
    return None if len(values) < 3 else float(np.median(values))


def _trajectory_overshoot(arrays: dict[str, Any], frames: np.ndarray, side: str | None, scale: float | None) -> float | None:
    if side not in {"left", "right"} or scale is None:
        return None
    points = [_hand_centre(arrays, int(frame), side) for frame in frames]
    points = [point for point in points if point is not None]
    if len(points) < 3:
        return None
    final = points[-1]
    direct = final - points[0]
    axis = _unit(direct)
    if axis is None:
        return 0.0
    projections = [float(np.dot(point - final, axis)) for point in points]
    return max(0.0, max(projections)) / scale


def _wrist_components(arrays: dict[str, Any], frames: np.ndarray, side: str | None) -> dict[str, float | None]:
    rows = []
    for frame in frames:
        elbow = _pt(arrays, int(frame), f"{side}_elbow") if side in {"left", "right"} else None
        wrist = _hand_point(arrays, int(frame), side, "wrist")
        middle = _hand_point(arrays, int(frame), side, "middle_mcp")
        index = _hand_point(arrays, int(frame), side, "index_mcp")
        pinky = _hand_point(arrays, int(frame), side, "pinky_mcp")
        if any(point is None for point in (elbow, wrist, middle, index, pinky)):
            continue
        forearm, hand = _unit(wrist - elbow), _unit(middle - wrist)
        palm_lateral = _unit(index - pinky)
        palm_normal = _unit(np.cross(hand, palm_lateral)) if hand is not None and palm_lateral is not None else None
        total = _angle(forearm, hand)
        if total is None or palm_normal is None or palm_lateral is None:
            continue
        rows.append((total, float(np.degrees(np.arcsin(np.clip(np.dot(forearm, palm_normal), -1.0, 1.0)))), float(np.degrees(np.arcsin(np.clip(np.dot(forearm, palm_lateral), -1.0, 1.0)))), _horizontal_angle(hand, _torso_forward(arrays, int(frame)))))
    if len(rows) < 3:
        return {key: None for key in ("total", "flexion", "radial", "orientation")}
    columns = np.asarray(rows, dtype=float).T
    values = [
        float(np.median(column[np.isfinite(column)])) if np.any(np.isfinite(column)) else None
        for column in columns
    ]
    return dict(zip(("total", "flexion", "radial", "orientation"), values, strict=True))


def _hand_dispersion(arrays: dict[str, Any], frames: np.ndarray, side: str | None, scale: float | None) -> float | None:
    values = [_hand_centre(arrays, int(frame), side) for frame in frames]
    values = [value for value in values if value is not None]
    if len(values) < 3 or scale is None:
        return None
    centre = np.median(np.asarray(values), axis=0)
    return float(np.percentile(np.linalg.norm(np.asarray(values) - centre, axis=1), 95) / scale)


def _technique_target_error(arrays: dict[str, Any], frames: np.ndarray, contract: dict[str, Any], side: str | None, component: str) -> float | None:
    values = []
    for frame in frames:
        hand = _hand_centre(arrays, int(frame), side)
        sc, hc = _centres(arrays, int(frame))
        if hand is None or sc is None or hc is None:
            continue
        if contract["active_hand_expected_height"] == "above_forehead":
            target = _cluster(arrays, int(frame), tuple(range(17, 27)))
        elif contract["active_hand_expected_height"] == "below_pelvis":
            target = (hc + _pt(arrays, int(frame), f"{side}_knee")) / 2.0 if side in {"left", "right"} and _pt(arrays, int(frame), f"{side}_knee") is not None else None
        else:
            target = (sc + hc) / 2.0
        if target is None:
            continue
        scale = _body_scale(arrays, int(frame)) if component == "height" else _shoulder_width(arrays, int(frame))
        if scale is None:
            continue
        values.append(abs(float(hand[2] - target[2])) / scale if component == "height" else abs(float(hand[0] - target[0])) / scale)
    return None if len(values) < 3 else float(np.median(values))


def _hand_settled(arrays: dict[str, Any], frames: np.ndarray, side: str | None, scale: float | None) -> bool | None:
    value = _hand_dispersion(arrays, frames, side, scale)
    return None if value is None else value <= 0.05


def _component_settled(arrays: dict[str, Any], frames: np.ndarray, part: str, limit: float) -> bool | None:
    value = _orientation_dispersion(arrays, frames, part)
    return None if value is None else value <= limit


def _settle_offset(arrays: dict[str, Any], frames: np.ndarray, first: str, second: str, fps: float) -> float | None:
    first_frame = _settle_frame(arrays, frames, first)
    second_frame = _settle_frame(arrays, frames, second)
    return None if first_frame is None or second_frame is None or fps <= 0 else float((first_frame - second_frame) / fps)


def _settle_frame(arrays: dict[str, Any], frames: np.ndarray, part: str) -> int | None:
    if len(frames) < 4:
        return None
    vectors = [_orientation_line(arrays, int(frame), part) for frame in frames]
    final = vectors[-1]
    if final is None:
        return None
    for index, vector in enumerate(vectors[:-2]):
        if vector is not None and _angle(vector, final) is not None and _angle(vector, final) <= 10.0:
            return int(frames[index])
    return int(frames[-1])


def _late_component_correction(arrays: dict[str, Any], frames: np.ndarray, side: str | None, scale: float | None) -> float | None:
    if len(frames) < 2 or scale is None:
        return None
    first, last = _hand_centre(arrays, int(frames[0]), side), _hand_centre(arrays, int(frames[-1]), side)
    return None if first is None or last is None else float(np.linalg.norm(last - first) / scale)


def _premature_transition(arrays: dict[str, Any], frames: np.ndarray, side: str | None, scale: float | None) -> bool | None:
    correction = _late_component_correction(arrays, frames, side, scale)
    return None if correction is None else correction > 0.08


def _fixation_stable(arrays: dict[str, Any], frames: np.ndarray, active: str | None, leg: float | None, arm: float | None) -> bool | None:
    hand = _hand_dispersion(arrays, frames, active, arm)
    lower = _lower_dispersion(arrays, frames, leg)
    return None if hand is None or lower is None else hand <= 0.05 and lower <= 0.04


def _foot_dispersion(arrays: dict[str, Any], frames: np.ndarray, side: str | None, scale: float | None) -> float | None:
    if side not in {"left", "right"} or scale is None:
        return None
    values = [_foot_centre(arrays, int(frame), side) for frame in frames]
    values = [value for value in values if value is not None]
    if len(values) < 3:
        return None
    centre = np.median(np.asarray(values), axis=0)
    return float(np.percentile(np.linalg.norm(np.asarray(values) - centre, axis=1), 95) / scale)


def _knee_dispersion(arrays: dict[str, Any], frames: np.ndarray, scale: float | None) -> float | None:
    left, right = (_joint_span(arrays, frames, f"{side}_knee", scale) for side in ("left", "right"))
    return None if left is None or right is None else max(left, right)


def _stance_change(arrays: dict[str, Any], frames: np.ndarray, lead: str | None, rear: str | None, component: str, scale: float | None) -> float | None:
    if len(frames) < 2 or scale is None:
        return None
    first = _stance_geometry(arrays, frames[: min(3, len(frames))], lead, rear).get(component)
    last = _stance_geometry(arrays, frames[-min(3, len(frames)) :], lead, rear).get(component)
    return None if first is None or last is None else abs(float(last - first)) / scale


def _head_forward(arrays: dict[str, Any], frame: int) -> np.ndarray | None:
    eyes = [_cluster(arrays, frame, range(36, 42)), _cluster(arrays, frame, range(42, 48))]
    nose = _cluster(arrays, frame, range(27, 36))
    return None if nose is None or any(value is None for value in eyes) else _unit(nose - (eyes[0] + eyes[1]) / 2.0)


def _torso_forward(arrays: dict[str, Any], frame: int) -> np.ndarray | None:
    lateral = _orientation_line(arrays, frame, "shoulder")
    if lateral is None:
        return None
    candidate = _unit(np.cross(np.asarray([0.0, 0.0, 1.0]), lateral))
    head = _head_forward(arrays, frame)
    return -candidate if candidate is not None and head is not None and np.dot(candidate, head) < 0 else candidate


def _pelvis_forward(arrays: dict[str, Any], frame: int) -> np.ndarray | None:
    lateral = _orientation_line(arrays, frame, "hip")
    torso = _torso_forward(arrays, frame)
    candidate = None if lateral is None else _unit(np.cross(np.asarray([0.0, 0.0, 1.0]), lateral))
    return -candidate if candidate is not None and torso is not None and np.dot(candidate, torso) < 0 else candidate


def _stance_axis(arrays: dict[str, Any], frame: int, lead: str | None, rear: str | None) -> np.ndarray | None:
    if lead not in {"left", "right"} or rear not in {"left", "right"}:
        return None
    front, back = _foot_centre(arrays, frame, lead), _foot_centre(arrays, frame, rear)
    return None if front is None or back is None else _unit(front - back)


def _knee_axis(arrays: dict[str, Any], frame: int, side: str | None) -> np.ndarray | None:
    if side not in {"left", "right"}:
        return None
    hip, knee = _pt(arrays, frame, f"{side}_hip"), _pt(arrays, frame, f"{side}_knee")
    return None if hip is None or knee is None else _unit(knee - hip)


def _arm_axis(arrays: dict[str, Any], frame: int, side: str | None) -> np.ndarray | None:
    if side not in {"left", "right"}:
        return None
    shoulder, wrist = _pt(arrays, frame, f"{side}_shoulder"), _hand_centre(arrays, frame, side)
    return None if shoulder is None or wrist is None else _unit(wrist - shoulder)


def _hand_axis(arrays: dict[str, Any], frame: int, side: str | None) -> np.ndarray | None:
    wrist, middle = _hand_point(arrays, frame, side, "wrist"), _hand_point(arrays, frame, side, "middle_mcp")
    return None if wrist is None or middle is None else _unit(middle - wrist)


def _knee_height(arrays: dict[str, Any], frame: int, side: str, scale: float | None) -> float | None:
    knee, hip = _pt(arrays, frame, f"{side}_knee"), _pt(arrays, frame, f"{side}_hip")
    return None if knee is None or hip is None else _safe_ratio(float(knee[2] - hip[2]), scale)


def _hip_flexion(arrays: dict[str, Any], frame: int, side: str) -> float | None:
    shoulder, hip, knee = (_pt(arrays, frame, f"{side}_{part}") for part in ("shoulder", "hip", "knee"))
    return None if shoulder is None or hip is None or knee is None else _angle(shoulder - hip, knee - hip)


def _ankle_height(arrays: dict[str, Any], frame: int, side: str, scale: float | None) -> float | None:
    ankle, hip = _pt(arrays, frame, f"{side}_ankle"), _pt(arrays, frame, f"{side}_hip")
    return None if ankle is None or hip is None else _safe_ratio(float(ankle[2] - hip[2]), scale)
