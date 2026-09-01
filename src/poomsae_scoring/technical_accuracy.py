from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from src.data_structures import (
    COCO_BODY_JOINTS,
    COCO_BODY_JOINT_NAMES,
    COCO_FACE_INDICES,
    COCO_FOOT_JOINTS,
    COCO_HAND_LOCAL_JOINTS,
    COCO_LEFT_HAND_OFFSET,
    COCO_RIGHT_HAND_OFFSET,
    COCO_WHOLEBODY_KEYPOINTS,
)
from src.poomsae_scoring.contracts import (
    ScoringContractError,
    load_yaml_mapping,
    validate_movement_timeline,
    validate_poomsae_spec,
)
from src.poomsae_scoring.wholebody_diagnostics import _pose_arrays
from src.poomsae_scoring.technical_accuracy_metrics import (
    measure_athlete_forward_vector,
    measure_observable_accuracy_metrics,
)


RULE_STATES = {
    "active_diagnostic",
    "measurement_only",
    "blocked_missing_reference",
    "unmeasurable",
    "not_observable_with_current_pipeline",
}
UNITS = {
    "bool",
    "deg",
    "ratio",
    "sec",
    "body_ratio",
    "leg_length",
    "arm_length",
    "shoulder_width",
    "torso_length",
    "state",
    "reason_code",
}
POLICY_KEYS = {
    "category",
    "numeric_score_enabled",
    "deduction_enabled",
    "decision_status",
    "rule_eligibility",
    "provenance",
    "score_effect",
    "deduction_points",
}
THRESHOLD_POLICY_KEYS = {
    "origin",
    "status",
    "scope",
    "version_date",
    "rationale",
    "no_score_semantics",
}
TOP_LEVEL_KEYS = {
    "schema_version",
    "profile_id",
    "version",
    "status",
    "poomsae_id",
    "required_keypoint_count",
    "policy",
    "quality_gates",
    "threshold_policy",
    "thresholds",
    "stance_contracts",
    "technique_contracts",
    "metric_catalog",
    "active_rules",
    "direction_bound_rules",
    "not_observable_rules",
    "failure_behavior",
    "skip_reason_codes",
    "disclaimer",
}
QUALITY_KEYS = {
    "min_used_cameras",
    "max_reprojection_error_px",
    "min_valid_samples",
    "anchor_window_radius_frames",
    "min_group_valid_ratio",
    "max_normalized_frame_jump",
}
THRESHOLD_KEYS = {"operator", "value", "uncertainty_band", "unit"}
OPERATORS = {"max", "abs_max", "min", "range"}
ACTIVE_EVALUATORS = {
    "head_torso_relative_yaw_deg",
    "head_roll_relative_shoulders_deg",
    "head_pitch_relative_torso_deg",
    "head_fixation_orientation_dispersion_p95_deg",
    "head_post_fixation_drift_deg",
    "torso_pelvis_relative_yaw_deg",
    "torso_lean_p95_deg",
    "torso_lateral_bend_deg",
    "shoulder_roll_deg",
    "pelvis_roll_deg",
    "torso_fixation_orientation_dispersion_p95_deg",
    "pelvis_fixation_orientation_dispersion_p95_deg",
    "torso_post_fixation_drift_deg",
    "pelvis_post_fixation_drift_deg",
    "foot_fixation_slip_body_ratio",
    "stance_fixation_dispersion_body_ratio",
    "elbow_flare_body_ratio",
    "shoulder_elbow_wrist_plane_deviation_ratio",
    "active_arm_fixation_dispersion_body_ratio",
    "reaction_arm_fixation_dispersion_body_ratio",
    "arm_late_correction_body_ratio",
    "wrist_forearm_alignment_deg",
    "active_hand_fixation_stability",
    "reaction_hand_fixation_stability",
    "active_hand_stance_settle_offset",
    "required_phase_coverage",
    "movement_contract_exists",
    "movement_timeline_segment_exists",
    "expected_movement_order",
    "movement_phase_order",
    "expected_technique_contract_resolved",
    "expected_stance_contract_resolved",
    "expected_active_side_contract_resolved",
}


def load_technical_accuracy_profile(path: str | Path) -> dict[str, Any]:
    return validate_technical_accuracy_profile(
        load_yaml_mapping(path, label="technical accuracy diagnostic profile")
    )


def validate_technical_accuracy_profile(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ScoringContractError("technical accuracy profile must be a mapping")
    data = deepcopy(payload)
    _exact_keys(data, TOP_LEVEL_KEYS, "technical accuracy profile")
    if data["schema_version"] != 3 or data["status"] != "diagnostic_only_unvalidated":
        raise ScoringContractError("technical accuracy profile must be schema 3 diagnostic_only_unvalidated")
    if data["required_keypoint_count"] != COCO_WHOLEBODY_KEYPOINTS:
        raise ScoringContractError("technical accuracy diagnostics require exactly 133 keypoints")
    for key in ("profile_id", "version", "poomsae_id", "failure_behavior", "disclaimer"):
        _nonempty(data[key], key)

    policy = data["policy"]
    _exact_keys(policy, POLICY_KEYS, "technical accuracy policy")
    required_policy = {
        "category": "technical_accuracy_diagnostic",
        "numeric_score_enabled": False,
        "deduction_enabled": False,
        "decision_status": "review_candidate_not_deduction",
        "rule_eligibility": "blocked_unvalidated_screening_threshold",
        "provenance": "self_authored_temporary_accuracy_rule",
        "score_effect": None,
        "deduction_points": None,
    }
    if policy != required_policy:
        raise ScoringContractError("technical accuracy policy must preserve exact no-score semantics")

    gates = data["quality_gates"]
    _exact_keys(gates, QUALITY_KEYS, "technical accuracy quality_gates")
    for key in ("min_used_cameras", "max_reprojection_error_px", "max_normalized_frame_jump"):
        gates[key] = _positive(gates[key], key)
    gates["min_valid_samples"] = _positive_integer(gates["min_valid_samples"], "min_valid_samples")
    gates["anchor_window_radius_frames"] = _nonnegative_integer(
        gates["anchor_window_radius_frames"], "anchor_window_radius_frames"
    )
    gates["min_group_valid_ratio"] = _probability(gates["min_group_valid_ratio"], "min_group_valid_ratio")

    threshold_policy = data["threshold_policy"]
    _exact_keys(threshold_policy, THRESHOLD_POLICY_KEYS, "threshold_policy")
    if threshold_policy["origin"] != policy["provenance"] or threshold_policy["no_score_semantics"] is not True:
        raise ScoringContractError("temporary threshold policy must be self-authored and score-neutral")
    for key in THRESHOLD_POLICY_KEYS - {"no_score_semantics"}:
        _nonempty(threshold_policy[key], f"threshold_policy.{key}")

    if not isinstance(data["thresholds"], dict):
        raise ScoringContractError("thresholds must be a mapping")
    for metric_id, threshold in data["thresholds"].items():
        _nonempty(metric_id, "threshold metric id")
        if not isinstance(threshold, dict):
            raise ScoringContractError(f"threshold {metric_id} must be a mapping")
        _exact_keys(threshold, THRESHOLD_KEYS, f"threshold {metric_id}")
        if threshold["operator"] not in OPERATORS:
            raise ScoringContractError(f"invalid threshold operator for {metric_id}")
        if threshold["unit"] not in UNITS:
            raise ScoringContractError(f"invalid threshold unit for {metric_id}: {threshold['unit']}")
        threshold["uncertainty_band"] = _nonnegative(
            threshold["uncertainty_band"], f"thresholds.{metric_id}.uncertainty_band"
        )
        if threshold["operator"] == "range":
            values = threshold["value"]
            if not isinstance(values, list) or len(values) != 2:
                raise ScoringContractError(f"range threshold {metric_id} must contain two values")
            low, high = (_finite(value, f"thresholds.{metric_id}.value") for value in values)
            if low > high:
                raise ScoringContractError(f"threshold range is inverted for {metric_id}")
            threshold["value"] = [low, high]
        else:
            threshold["value"] = _nonnegative(threshold["value"], f"thresholds.{metric_id}.value")

    catalog = data["metric_catalog"]
    if not isinstance(catalog, dict) or not catalog:
        raise ScoringContractError("metric_catalog must be a non-empty mapping")
    metric_family: dict[str, str] = {}
    for family, metric_ids in catalog.items():
        _nonempty(family, "rule family")
        if not isinstance(metric_ids, list) or not metric_ids:
            raise ScoringContractError(f"metric_catalog.{family} must be a non-empty list")
        for metric_id in metric_ids:
            metric_id = _nonempty(metric_id, f"metric_catalog.{family}")
            if metric_id in metric_family:
                raise ScoringContractError(f"duplicate rule/metric id: {metric_id}")
            metric_family[metric_id] = family

    active = _id_set(data["active_rules"], metric_family, "active_rules")
    direction = _id_set(data["direction_bound_rules"], metric_family, "direction_bound_rules")
    not_observable = _id_set(data["not_observable_rules"], metric_family, "not_observable_rules")
    if active & direction or active & not_observable or direction & not_observable:
        raise ScoringContractError("rule state lists must not overlap")
    if active != ACTIVE_EVALUATORS:
        raise ScoringContractError(
            "active rules and implemented evaluator inventory differ: "
            f"missing={sorted(active - ACTIVE_EVALUATORS)}, "
            f"undefined={sorted(ACTIVE_EVALUATORS - active)}"
        )
    thresholdless_active = {
        "required_phase_coverage",
        "movement_contract_exists",
        "movement_timeline_segment_exists",
        "expected_movement_order",
        "movement_phase_order",
        "expected_technique_contract_resolved",
        "expected_stance_contract_resolved",
        "expected_active_side_contract_resolved",
    }
    missing_thresholds = active - set(data["thresholds"]) - thresholdless_active
    if missing_thresholds:
        raise ScoringContractError(f"active rules lack configured thresholds: {sorted(missing_thresholds)}")
    unknown_thresholds = set(data["thresholds"]) - set(metric_family)
    if unknown_thresholds:
        raise ScoringContractError(f"thresholds reference unknown metrics: {sorted(unknown_thresholds)}")

    _validate_contract_tables(data["stance_contracts"], data["technique_contracts"])
    reasons = data["skip_reason_codes"]
    if not isinstance(reasons, list) or len(set(reasons)) != len(reasons) or not all(isinstance(x, str) and x for x in reasons):
        raise ScoringContractError("skip_reason_codes must be unique non-empty strings")
    data["resolved_rules"] = [
        _resolved_rule(data, metric_id, family, active, direction, not_observable)
        for metric_id, family in metric_family.items()
    ]
    return data


def validate_athlete_local_direction_reference(
    payload: dict[str, Any] | None,
    *,
    expected_session_id: str | None = None,
    expected_pose_sha256: str | None = None,
) -> dict[str, Any] | None:
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise ScoringContractError("athlete-local direction reference must be a mapping")
    expected = {
        "schema_version",
        "session_id",
        "reference_pose_sha256",
        "gravity_up_vector",
        "initial_forward_vector",
        "basis_source",
        "provenance",
        "quality_status",
    }
    _exact_keys(payload, expected, "athlete-local direction reference")
    if payload["schema_version"] != 1:
        raise ScoringContractError("athlete-local direction reference schema must be 1")
    if expected_session_id is not None and payload["session_id"] != expected_session_id:
        raise ScoringContractError("athlete-local direction session binding mismatch")
    if expected_pose_sha256 is not None and payload["reference_pose_sha256"] != expected_pose_sha256:
        raise ScoringContractError("athlete-local direction pose binding mismatch")
    if payload["basis_source"] not in {"manually_declared_session_bound", "derived_session_bound"}:
        raise ScoringContractError("athlete-local direction basis_source is invalid")
    if payload["quality_status"] != "validated_diagnostic_reference":
        raise ScoringContractError("athlete-local direction quality must be validated_diagnostic_reference")
    up = _unit_vector(payload["gravity_up_vector"], "gravity_up_vector")
    forward_raw = np.asarray(payload["initial_forward_vector"], dtype=float)
    if forward_raw.shape != (3,) or not np.all(np.isfinite(forward_raw)):
        raise ScoringContractError("initial_forward_vector must be a finite 3-vector")
    forward_horizontal = forward_raw - np.dot(forward_raw, up) * up
    if np.linalg.norm(forward_horizontal) <= 1e-8:
        raise ScoringContractError("initial_forward_vector is degenerate after horizontal projection")
    forward = forward_horizontal / np.linalg.norm(forward_horizontal)
    left = np.cross(up, forward)
    if np.linalg.norm(left) <= 1e-8:
        raise ScoringContractError("athlete-local basis is degenerate")
    left /= np.linalg.norm(left)
    if abs(float(np.dot(forward, up))) > 1e-6 or abs(float(np.dot(left, up))) > 1e-6 or abs(float(np.dot(left, forward))) > 1e-6:
        raise ScoringContractError("athlete-local basis is not orthogonal and horizontal")
    return {
        **deepcopy(payload),
        "gravity_up_vector": up.tolist(),
        "initial_forward_vector": forward.tolist(),
        "initial_backward_vector": (-forward).tolist(),
        "initial_left_vector": left.tolist(),
        "initial_right_vector": (-left).tolist(),
        "handedness": "right_handed_z_up_with_left=cross(up,forward)",
        "production_calibration_claim": False,
    }


DIRECTION_REFERENCE_ANCHOR_MOVEMENT = "M01"
DIRECTION_REFERENCE_ANCHOR_PHASE = "preparation"
WORLD_UP_VECTOR = (0.0, 0.0, 1.0)


def derive_athlete_local_direction_reference(
    pose_payload: dict[str, Any],
    poomsae_spec: dict[str, Any],
    movement_timeline: dict[str, Any],
    profile_payload: dict[str, Any],
    *,
    anchor_movement_id: str = DIRECTION_REFERENCE_ANCHOR_MOVEMENT,
    anchor_phase: str = DIRECTION_REFERENCE_ANCHOR_PHASE,
) -> dict[str, Any]:
    """Derive the session-bound athlete-local direction reference from the ready stance.

    The reference is measured, never assumed: the athlete's facing is taken as the
    median torso-forward geometry inside the anchor window of the opening movement,
    projected horizontally against the pipeline's gravity-aligned world up axis. The
    result is a diagnostic reference bound to one session and one pose hash; it is not
    a production calibration and it never claims a world or judging axis.

    Always returns an envelope. When the geometry cannot be measured the envelope
    carries ``status="not_derived"`` with a profile skip reason and no reference, so
    direction-bound rules stay fail-closed instead of running on invented geometry.
    """
    spec = validate_poomsae_spec(poomsae_spec)
    timeline = validate_movement_timeline(movement_timeline, spec)
    profile = (
        profile_payload
        if isinstance(profile_payload, dict) and "resolved_rules" in profile_payload
        else validate_technical_accuracy_profile(profile_payload)
    )
    if profile["poomsae_id"] != spec["poomsae_id"]:
        raise ScoringContractError("technical accuracy profile Poomsae id does not match")
    gates = profile["quality_gates"]
    source_binding = timeline["source_binding"]
    envelope: dict[str, Any] = {
        "schema_version": 1,
        "status": "not_derived",
        "reason": None,
        "binding": {
            "session_id": source_binding["session_id"],
            "reference_pose_sha256": source_binding["pose_file_sha256"],
            "movement_timeline_id": timeline["timeline_id"],
        },
        "anchor": {
            "movement_id": anchor_movement_id,
            "phase": anchor_phase,
            "anchor_frame": None,
            "window": None,
            "sample_count": 0,
        },
        "measurement": "median_torso_forward_horizontal",
        "production_calibration_claim": False,
        "reference": None,
    }

    segment = next(
        (item for item in timeline["segments"] if item["movement_id"] == anchor_movement_id),
        None,
    )
    if segment is None:
        envelope["reason"] = "movement_not_present_in_timeline"
        return envelope
    anchors = segment["anchors"]
    if anchor_phase not in anchors:
        envelope["reason"] = "movement_contract_incomplete"
        return envelope

    arrays = _pose_arrays(
        pose_payload,
        timeline,
        {
            "min_used_cameras": gates["min_used_cameras"],
            "max_reprojection_error_px": gates["max_reprojection_error_px"],
            "min_segment_group_valid_ratio": gates["min_group_valid_ratio"],
        },
    )
    anchor_frame = int(anchors[anchor_phase])
    radius = int(gates["anchor_window_radius_frames"])
    start = max(int(segment["start_frame"]), anchor_frame - radius)
    end = min(int(segment["end_frame"]), anchor_frame + radius)
    frames = np.arange(start, end + 1, dtype=int)
    envelope["anchor"].update(
        {"anchor_frame": anchor_frame, "window": [int(start), int(end)], "sample_count": int(len(frames))}
    )

    forward = measure_athlete_forward_vector(arrays, frames, int(gates["min_valid_samples"]))
    if forward is None:
        envelope["reason"] = "insufficient_valid_samples"
        return envelope

    up = np.asarray(WORLD_UP_VECTOR, dtype=float)
    horizontal = forward - float(np.dot(forward, up)) * up
    if not np.all(np.isfinite(horizontal)) or float(np.linalg.norm(horizontal)) <= 1e-8:
        envelope["reason"] = "degenerate_body_axis"
        return envelope
    horizontal = horizontal / float(np.linalg.norm(horizontal))

    reference = {
        "schema_version": 1,
        "session_id": source_binding["session_id"],
        "reference_pose_sha256": source_binding["pose_file_sha256"],
        "gravity_up_vector": [float(value) for value in up],
        "initial_forward_vector": [float(value) for value in horizontal],
        "basis_source": "derived_session_bound",
        "provenance": (
            f"derived from {anchor_movement_id} {anchor_phase} median torso-forward geometry; "
            "session-bound diagnostic reference, not production calibration"
        ),
        "quality_status": "validated_diagnostic_reference",
    }
    validate_athlete_local_direction_reference(
        reference,
        expected_session_id=source_binding["session_id"],
        expected_pose_sha256=source_binding["pose_file_sha256"],
    )
    envelope["status"] = "derived"
    envelope["reference"] = reference
    return envelope


def build_technical_accuracy_diagnostics(
    pose_payload: dict[str, Any],
    poomsae_spec: dict[str, Any],
    movement_timeline: dict[str, Any],
    profile_payload: dict[str, Any],
    wholebody_diagnostics: dict[str, Any],
    *,
    direction_reference: dict[str, Any] | None = None,
) -> dict[str, Any]:
    spec = validate_poomsae_spec(poomsae_spec)
    timeline = validate_movement_timeline(movement_timeline, spec)
    profile = (
        profile_payload
        if isinstance(profile_payload, dict) and "resolved_rules" in profile_payload
        else validate_technical_accuracy_profile(profile_payload)
    )
    if profile["poomsae_id"] != spec["poomsae_id"]:
        raise ScoringContractError("technical accuracy profile Poomsae id does not match")
    if wholebody_diagnostics.get("status") != "wholebody_diagnostics_only":
        raise ScoringContractError("technical accuracy diagnostics require canonical WholeBody diagnostics")
    if wholebody_diagnostics.get("movement_timeline_id") != timeline["timeline_id"]:
        raise ScoringContractError("WholeBody diagnostic timeline binding mismatch")
    gates = profile["quality_gates"]
    arrays = _pose_arrays(
        pose_payload,
        timeline,
        {
            "min_used_cameras": gates["min_used_cameras"],
            "max_reprojection_error_px": gates["max_reprojection_error_px"],
            "min_segment_group_valid_ratio": gates["min_group_valid_ratio"],
        },
    )
    source_binding = timeline["source_binding"]
    direction = validate_athlete_local_direction_reference(
        direction_reference,
        expected_session_id=source_binding["session_id"],
        expected_pose_sha256=source_binding["pose_file_sha256"],
    )
    contracts = resolve_movement_accuracy_contracts(spec, profile)
    observed_segments = {segment["movement_id"]: segment for segment in timeline["segments"]}
    old_movements = {movement["movement_id"]: movement for movement in wholebody_diagnostics.get("movements", [])}
    rules = profile["resolved_rules"]
    coverage: list[dict[str, Any]] = []
    movement_reports: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []

    for contract in contracts:
        movement_id = contract["movement_id"]
        segment = observed_segments.get(movement_id)
        old = old_movements.get(movement_id)
        measurements = (
            _movement_measurements(
                arrays,
                contract,
                segment,
                old,
                profile,
                direction,
                float(timeline["fps"]),
            )
            if segment
            else {}
        )
        results: list[dict[str, Any]] = []
        for rule in rules:
            applies = _rule_applies(rule, contract)
            result = _evaluate_rule(rule, contract, segment, measurements, direction, profile) if applies else _not_applicable(rule, contract)
            results.append(result)
            coverage.append(_coverage_row(result, applies))
            if result["decision_status"] == profile["policy"]["decision_status"]:
                candidates.append(result)
        counts = Counter(result["state"] for result in results if result["applies"])
        movement_reports.append(
            {
                "movement_id": movement_id,
                "movement_label": contract["movement_label"],
                "present_in_current_timeline": segment is not None,
                "contract": contract,
                "summary": {
                    "applicable_rule_count": sum(result["applies"] for result in results),
                    "measured_rule_count": sum(result["measured"] for result in results),
                    "evaluated_rule_count": sum(result["evaluated"] for result in results),
                    "in_range_rule_count": sum(result["evaluation"] == "within_screening_range" for result in results),
                    "temporary_candidate_count": sum(result["evaluation"] == "out_of_range" for result in results),
                    "boundary_count": sum(result["evaluation"] == "boundary_uncertain" for result in results),
                    "measurement_only_count": counts["measurement_only"],
                    "blocked_count": counts["blocked_missing_reference"],
                    "unmeasurable_count": counts["unmeasurable"],
                    "not_observable_count": counts["not_observable_with_current_pipeline"],
                },
                "rules": results,
            }
        )

    state_counts = Counter(row["state"] for row in coverage if row["applies"])
    landmark_inventory = _build_landmark_inventory(rules)
    return _json_safe(
        {
            "schema_version": 1,
            "status": "technical_accuracy_diagnostics_only",
            "category": "technical_accuracy_diagnostic",
            "scoring_status": "not_scored_diagnostic_candidates_only",
            "accuracy_score": None,
            "presentation_score": None,
            "total_score": None,
            "deductions": [],
            "numeric_score_enabled": False,
            "deduction_enabled": False,
            "official_accuracy_claim_allowed": False,
            "movement_timeline_id": timeline["timeline_id"],
            "poomsae": {"poomsae_id": spec["poomsae_id"], "version": spec["version"], "status": spec["status"]},
            "profile": {
                "profile_id": profile["profile_id"],
                "version": profile["version"],
                "status": profile["status"],
                "rule_count": len(rules),
                "threshold_policy": profile["threshold_policy"],
                "disclaimer": profile["disclaimer"],
            },
            "direction_reference": direction,
            "direction_reference_status": "validated_session_bound_diagnostic" if direction else "missing_fail_closed",
            "summary": {
                "movement_contract_count": len(contracts),
                "observed_movement_count": len(observed_segments),
                "configuration_only_movement_count": len(contracts) - len(observed_segments),
                "rule_count": len(rules),
                "active_diagnostic_rule_count": sum(rule["status"] == "active_diagnostic" for rule in rules),
                "measurement_only_rule_count": sum(rule["status"] == "measurement_only" for rule in rules),
                "blocked_missing_reference_rule_count": sum(rule["status"] == "blocked_missing_reference" for rule in rules),
                "not_observable_rule_count": sum(rule["status"] == "not_observable_with_current_pipeline" for rule in rules),
                "implemented_measurement_evaluator_rule_count": sum(
                    rule["measurement_evaluator_status"] == "implemented" for rule in rules
                ),
                "evaluator_not_implemented_rule_count": sum(
                    rule["measurement_evaluator_status"] == "not_implemented" for rule in rules
                ),
                "landmark_inventory_count": len(landmark_inventory),
                "landmarks_declared_by_any_rule_count": sum(
                    row["declared_rule_count"] > 0 for row in landmark_inventory
                ),
                "landmarks_declared_by_active_rule_count": sum(
                    row["active_diagnostic_rule_count"] > 0 for row in landmark_inventory
                ),
                "coverage_state_counts": dict(state_counts),
                "temporary_candidate_count": len(candidates),
                "score_effect_count": 0,
                "deduction_effect_count": 0,
            },
            "rule_inventory": rules,
            "landmark_inventory": landmark_inventory,
            "movement_contracts": contracts,
            "coverage_matrix": coverage,
            "candidate_events": candidates,
            "movements": movement_reports,
            "limitations": [
                "M07-M18 are configuration/synthetic scope only because the active recording ends after M06.",
                "head_orientation_proxy is not actual eye gaze or visual attention.",
                "pelvis/body-centre proxies are not centre of mass, pressure, or weight distribution.",
                "WholeBody hand, face, and foot points do not have the same depth/global-optimizer validation as BODY-17.",
                "Temporary thresholds are unvalidated engineering hypotheses and cannot change any score or deduction.",
            ],
        }
    )


def resolve_movement_accuracy_contracts(spec_payload: dict[str, Any], profile_payload: dict[str, Any]) -> list[dict[str, Any]]:
    spec = validate_poomsae_spec(spec_payload)
    profile = (
        profile_payload
        if isinstance(profile_payload, dict) and "resolved_rules" in profile_payload
        else validate_technical_accuracy_profile(profile_payload)
    )
    contracts: list[dict[str, Any]] = []
    previous_direction: str | None = None
    for movement in spec["movements"]:
        stance_family = "ap_seogi" if movement["stance"].endswith("_ap_seogi") else "ap_gubi" if movement["stance"].endswith("_ap_gubi") else None
        stance = profile["stance_contracts"].get(stance_family) if stance_family else None
        sided = [technique for technique in movement["techniques"] if technique["side"] in {"left", "right"}]
        action = next((technique for technique in sided if technique["technique_id"] != "ap_chagi"), sided[0] if sided else None)
        kick = next((technique for technique in sided if technique["technique_id"] == "ap_chagi"), None)
        technique_contracts = [profile["technique_contracts"].get(item["technique_id"]) for item in movement["techniques"]]
        active_side = None if action is None else action["side"]
        lead_leg = movement["stance"].split("_", 1)[0] if movement["stance"].startswith(("left_", "right_")) else None
        direction_changed = previous_direction is not None and movement["direction"] != previous_direction
        has_turn = "turn" in movement["phases"] or direction_changed
        moving_foot = lead_leg if any(phase in movement["phases"] for phase in ("step", "landing", "weight_transfer")) or has_turn else None
        support_foot = None if moving_foot is None else ("right" if moving_foot == "left" else "left")
        primary_contract = next((item for item in technique_contracts if item), None)
        contracts.append(
            {
                "movement_id": movement["movement_id"],
                "movement_label": movement["display_name"],
                "technique_types": [item["technique_id"] for item in movement["techniques"]],
                "stance_type": movement["stance"],
                "semantic_action_direction": movement["direction"],
                "previous_semantic_action_direction": previous_direction,
                "direction_changes_from_previous": direction_changed,
                "active_arm": active_side,
                "kicking_leg": None if kick is None else kick["side"],
                "chamber_or_reaction_arm": None if active_side is None else ("right" if active_side == "left" else "left"),
                "lead_leg": lead_leg,
                "rear_or_support_leg": None if lead_leg is None else ("right" if lead_leg == "left" else "left"),
                "expected_moving_foot": moving_foot,
                "expected_pivot_or_support_foot": support_foot,
                "preparation_requirements": [item["preparation"] for item in technique_contracts if item],
                "execution_requirements": [f"execute_{item['technique_id']}_toward_semantic_direction" for item in movement["techniques"]],
                "fixation_requirements": ["declared_stance_and_technique_geometry_settled", "score_neutral_diagnostic_evidence_only"],
                "head_target_orientation": movement["direction"],
                "torso_target_orientation": movement["direction"],
                "pelvis_or_stance_orientation": movement["direction"],
                "active_hand_target_region": None if primary_contract is None else primary_contract["target_region"],
                "active_hand_expected_height": None if primary_contract is None else primary_contract["target_height"],
                "active_hand_expected_lateral_position": None if primary_contract is None else primary_contract["lateral_position"],
                "active_hand_expected_depth": None if primary_contract is None else primary_contract["depth"],
                "elbow_relationship": None if primary_contract is None else primary_contract["elbow_relationship"],
                "wrist_or_hand_relationship": None if primary_contract is None else primary_contract["wrist_relationship"],
                "chamber_hand_target_region": None if primary_contract is None else primary_contract["reaction_hand"],
                "stance_length_expectation": None if stance is None else stance["length_leg_ratio"],
                "stance_width_expectation": None if stance is None else stance["width_shoulder_ratio"],
                "foot_angle_expectation": None if stance is None else {"target_uncertainty_deg": stance["foot_target_yaw_uncertainty_deg"]},
                "knee_angle_or_alignment_expectation": None if stance is None else {
                    "front_included_angle_deg": stance["front_knee_included_angle_deg"],
                    "rear_included_angle_deg": stance["rear_knee_included_angle_deg"],
                },
                "permitted_transition_motion": "head_torso_pelvis_and_feet_may_rotate_before_fixation" if has_turn else "same_direction_continuation_without_required_new_turn",
                "required_fixation_stability": "configured_component_specific_dispersion_limits",
                "evaluation_phases": movement["phases"],
                "evidence_quality_requirements": ["exact_133_points", "finite_required_landmarks", "minimum_camera_and_reprojection_gates", "minimum_valid_samples"],
                "contract_resolution_status": "resolved" if stance is not None and all(item is not None for item in technique_contracts) and active_side is not None else "incomplete",
                "blocking_reasons": [
                    reason
                    for reason, blocked in (
                        ("unsupported_stance", stance is None),
                        ("unsupported_technique", any(item is None for item in technique_contracts)),
                        ("missing_active_side", active_side is None),
                    )
                    if blocked
                ],
            }
        )
        previous_direction = movement["direction"]
    expected_ids = [f"M{index:02d}" for index in range(1, 19)]
    if [item["movement_id"] for item in contracts] != expected_ids:
        raise ScoringContractError("Taegeuk 1 accuracy contracts must resolve exactly M01-M18 in order")
    return contracts


def evaluate_temporary_threshold(value: Any, threshold: dict[str, Any] | None) -> str:
    if value is None:
        return "unmeasurable"
    if threshold is None:
        if not isinstance(value, (bool, np.bool_)):
            return "unmeasurable"
        return "within_screening_range" if bool(value) else "out_of_range"
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        return "unmeasurable"
    numeric = float(value)
    if not np.isfinite(numeric):
        return "unmeasurable"
    uncertainty = float(threshold["uncertainty_band"])
    operator = threshold["operator"]
    limit = threshold["value"]
    if operator in {"max", "abs_max"}:
        numeric = abs(numeric) if operator == "abs_max" else numeric
        if numeric - uncertainty > float(limit):
            return "out_of_range"
        if numeric + uncertainty > float(limit):
            return "boundary_uncertain"
        return "within_screening_range"
    if operator == "min":
        if numeric + uncertainty < float(limit):
            return "out_of_range"
        if numeric - uncertainty < float(limit):
            return "boundary_uncertain"
        return "within_screening_range"
    low, high = map(float, limit)
    if numeric + uncertainty < low or numeric - uncertainty > high:
        return "out_of_range"
    if numeric - uncertainty < low or numeric + uncertainty > high:
        return "boundary_uncertain"
    return "within_screening_range"


def _movement_measurements(
    arrays: dict[str, Any],
    contract: dict[str, Any],
    segment: dict[str, Any],
    wholebody: dict[str, Any] | None,
    profile: dict[str, Any],
    direction: dict[str, Any] | None,
    fps: float,
) -> dict[str, dict[str, Any]]:
    old_metrics = {item["metric_id"]: item for item in (wholebody or {}).get("metrics", [])}
    aliases = {
        "head_torso_relative_yaw_deg": "head_torso_yaw_mismatch_deg",
        "torso_pelvis_relative_yaw_deg": "shoulder_hip_twist_deg",
        "torso_lean_p95_deg": "torso_lean_p95_deg",
        "wrist_forearm_alignment_deg": "wrist_forearm_alignment_deg",
        "active_hand_stance_settle_offset": "hand_foot_settle_difference_sec",
        "active_arm_target_height_body_ratio": "executing_wrist_height_torso_ratio",
        "elbow_flexion_deg": "executing_elbow_deg",
        "reaction_hand_target_distance_body_ratio": "reaction_hand_hip_distance_ratio",
        "stance_length_leg_ratio": "stance_span_ratio",
        "front_knee_flexion_deg": "front_knee_deg",
    }
    result = measure_observable_accuracy_metrics(
        arrays,
        contract,
        segment,
        profile,
        direction,
        fps,
    )
    for target, source in aliases.items():
        metric = old_metrics.get(source)
        if metric is not None:
            result[target] = {
                "value": metric.get("value"),
                "unit": _unit_for(target),
                "quality_status": "measured" if metric.get("value") is not None else "unmeasurable",
                "evidence": metric.get("measurement_evidence"),
                "source_metric_id": source,
            }

    start, end = segment["start_frame"], segment["end_frame"]
    fixation = segment["anchors"].get("fixation", end)
    radius = profile["quality_gates"]["anchor_window_radius_frames"]
    frames = np.arange(max(start, fixation - radius), min(end, fixation + radius) + 1)
    after = np.arange(fixation, end + 1)
    active = contract["active_arm"]
    reaction = contract["chamber_or_reaction_arm"]
    rear = contract["rear_or_support_leg"]
    computed = {
        "head_roll_relative_shoulders_deg": _window_scalar(arrays, frames, _head_roll),
        "head_pitch_relative_torso_deg": _window_scalar(arrays, frames, _head_pitch_unsigned),
        "head_fixation_orientation_dispersion_p95_deg": _orientation_dispersion(
            arrays, after, "head"
        ),
        "head_post_fixation_drift_deg": _orientation_drift(arrays, after, "head"),
        "torso_lateral_bend_deg": _window_scalar(arrays, frames, _torso_lateral_bend),
        "shoulder_roll_deg": _window_scalar(arrays, frames, lambda a, f: _line_roll(a, f, "shoulder")),
        "pelvis_roll_deg": _window_scalar(arrays, frames, lambda a, f: _line_roll(a, f, "hip")),
        "shoulder_height_asymmetry_body_ratio": _window_scalar(arrays, frames, lambda a, f: _height_asymmetry(a, f, "shoulder")),
        "pelvis_height_asymmetry_body_ratio": _window_scalar(arrays, frames, lambda a, f: _height_asymmetry(a, f, "hip")),
        "elbow_flare_body_ratio": _arm_plane_deviation(arrays, frames, active),
        "shoulder_elbow_wrist_plane_deviation_ratio": _arm_plane_deviation(arrays, frames, active),
        "active_arm_fixation_dispersion_body_ratio": _joint_dispersion(arrays, after, f"{active}_wrist", "arm", active),
        "reaction_arm_fixation_dispersion_body_ratio": _joint_dispersion(arrays, after, f"{reaction}_wrist", "arm", reaction),
        "arm_late_correction_body_ratio": _late_correction(arrays, after, f"{active}_wrist", "arm", active),
        "rear_knee_flexion_deg": _window_scalar(arrays, frames, lambda a, f: _joint_angle(a, f, rear)),
        "shoulder_pelvis_roll_difference_deg": _window_scalar(arrays, frames, _shoulder_pelvis_roll_difference),
        "torso_translation_after_fixation_body_ratio": _centre_dispersion(arrays, after, "shoulder"),
        "pelvis_translation_after_fixation_body_ratio": _centre_dispersion(arrays, after, "hip"),
        "body_height_change_during_fixation_ratio": _body_height_change(arrays, after),
        "torso_fixation_orientation_dispersion_p95_deg": _orientation_dispersion(
            arrays, after, "shoulder"
        ),
        "pelvis_fixation_orientation_dispersion_p95_deg": _orientation_dispersion(
            arrays, after, "hip"
        ),
        "torso_post_fixation_drift_deg": _orientation_drift(arrays, after, "shoulder"),
        "pelvis_post_fixation_drift_deg": _orientation_drift(arrays, after, "hip"),
    }
    for metric_id, value in computed.items():
        result[metric_id] = {
            "value": value,
            "unit": _unit_for(metric_id),
            "quality_status": "measured" if value is not None else "unmeasurable",
            "evidence": {"start_frame": int(frames[0]), "end_frame": int(frames[-1]), "minimum_valid_samples": profile["quality_gates"]["min_valid_samples"]},
            "source_metric_id": None,
        }
    required = {"preparation", "fixation"}
    execution_present = any(name in segment["anchors"] for name in ("execution", "punch_execution", "kick_apex"))
    result["required_phase_coverage"] = _bool_measurement(required.issubset(segment["anchors"]) and execution_present)
    result["movement_contract_exists"] = _bool_measurement(True)
    result["movement_timeline_segment_exists"] = _bool_measurement(True)
    result["expected_movement_order"] = _bool_measurement(segment["sequence_index"] == int(contract["movement_id"][1:]))
    anchors = list(segment["anchors"].values())
    result["movement_phase_order"] = _bool_measurement(anchors == sorted(anchors))
    result["expected_technique_contract_resolved"] = _bool_measurement(not any(reason == "unsupported_technique" for reason in contract["blocking_reasons"]))
    result["expected_stance_contract_resolved"] = _bool_measurement(not any(reason == "unsupported_stance" for reason in contract["blocking_reasons"]))
    result["expected_active_side_contract_resolved"] = _bool_measurement(contract["active_arm"] in {"left", "right"})
    _apply_measurement_quality_gates(result, arrays, segment, profile)
    return result


def _apply_measurement_quality_gates(
    measurements: dict[str, dict[str, Any]],
    arrays: dict[str, Any],
    segment: dict[str, Any],
    profile: dict[str, Any],
) -> None:
    rules = {rule["metric_id"]: rule for rule in profile["resolved_rules"]}
    quality_metrics = {
        metric_id
        for metric_id in measurements
        if metric_id.endswith("_geometry_quality")
        or metric_id.endswith("_evidence_quality")
        or metric_id.endswith("_unmeasurable_reason")
    }
    minimum = float(profile["quality_gates"]["min_group_valid_ratio"])
    frames = np.arange(int(segment["start_frame"]), int(segment["end_frame"]) + 1, dtype=int)
    for metric_id, measurement in measurements.items():
        if metric_id in quality_metrics or measurement.get("value") is None:
            continue
        rule = rules.get(metric_id)
        if rule is None or not rule["required_landmarks"]:
            continue
        indices = np.asarray(rule["required_landmarks"], dtype=int)
        ratio = float(np.mean(arrays["quality_mask"][np.ix_(frames, indices)]))
        evidence = measurement.get("evidence")
        if not isinstance(evidence, dict):
            evidence = {}
            measurement["evidence"] = evidence
        evidence["required_landmark_valid_ratio"] = ratio
        evidence["minimum_required_landmark_valid_ratio"] = minimum
        if ratio < minimum:
            measurement["value"] = None
            measurement["quality_status"] = "unmeasurable"
            measurement["not_measurable_reason"] = "required_landmark_group_quality_below_minimum"


def _evaluate_rule(
    rule: dict[str, Any],
    contract: dict[str, Any],
    segment: dict[str, Any] | None,
    measurements: dict[str, dict[str, Any]],
    direction: dict[str, Any] | None,
    profile: dict[str, Any],
) -> dict[str, Any]:
    base = _result_base(rule, contract)
    if segment is None:
        return {**base, "measured": False, "evaluated": False, "state": "blocked_missing_reference", "evaluation": "not_evaluated", "value": None, "quality_status": "blocked", "skip_or_block_reason": "movement_not_present_in_timeline"}
    if rule["status"] == "not_observable_with_current_pipeline":
        return {**base, "measured": False, "evaluated": False, "state": rule["status"], "evaluation": "not_evaluated", "value": None, "quality_status": "not_observable", "skip_or_block_reason": "measurement_not_observable"}
    if rule["status"] == "blocked_missing_reference" and direction is None:
        return {**base, "measured": False, "evaluated": False, "state": rule["status"], "evaluation": "not_evaluated", "value": None, "quality_status": "blocked", "skip_or_block_reason": "missing_athlete_local_direction_binding"}
    measurement = measurements.get(rule["metric_id"])
    if measurement is None or measurement.get("value") is None:
        reason = "evaluator_not_implemented" if measurement is None else measurement.get(
            "not_measurable_reason", "insufficient_valid_samples"
        )
        missing_reference = isinstance(reason, str) and reason.startswith("missing_numeric_")
        return {
            **base,
            "measured": False,
            "evaluated": False,
            "state": "blocked_missing_reference" if missing_reference else "unmeasurable",
            "evaluation": "not_evaluated",
            "value": None,
            "quality_status": "blocked" if missing_reference else "unmeasurable",
            "skip_or_block_reason": reason,
        }
    direction_bound_evaluable = rule["status"] == "blocked_missing_reference" and direction is not None
    if rule["status"] != "active_diagnostic" and not direction_bound_evaluable:
        return {**base, **measurement, "measured": True, "evaluated": False, "state": "measurement_only", "evaluation": "measurement_only", "skip_or_block_reason": None}
    evaluation = evaluate_temporary_threshold(measurement["value"], rule["threshold"])
    decision = profile["policy"]["decision_status"] if evaluation == "out_of_range" else "no_candidate"
    return {
        **base,
        **measurement,
        "measured": True,
        "evaluated": True,
        "state": "active_diagnostic",
        "evaluation": evaluation,
        "decision_status": decision,
        "skip_or_block_reason": None,
    }


def _not_applicable(rule: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    return {**_result_base(rule, contract), "applies": False, "measured": False, "evaluated": False, "state": rule["status"], "evaluation": "not_applicable", "value": None, "quality_status": "not_applicable", "skip_or_block_reason": None}


def _result_base(rule: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "movement_id": contract["movement_id"],
        "movement_label": contract["movement_label"],
        "technique": contract["technique_types"],
        "stance": contract["stance_type"],
        "phase_or_window": rule["evaluation_phase"],
        "rule_id": rule["rule_id"],
        "metric_id": rule["metric_id"],
        "criterion_id": rule["criterion_id"],
        "rule_family": rule["rule_family"],
        "applies": True,
        "expected_value_or_range": rule["expected_value_or_range"],
        "threshold": rule["threshold"],
        "uncertainty": None if rule["threshold"] is None else rule["threshold"]["uncertainty_band"],
        "unit": rule["unit"],
        "decision_status": "no_candidate",
        "rule_eligibility": rule["rule_eligibility"],
        "provenance": rule["origin"],
        "score_effect": None,
        "deduction_points": None,
        "deduction_enabled": False,
        "numeric_score_enabled": False,
    }


def _coverage_row(result: dict[str, Any], applies: bool) -> dict[str, Any]:
    return {
        "movement_id": result["movement_id"],
        "rule_id": result["rule_id"],
        "metric_id": result["metric_id"],
        "criterion_id": result["criterion_id"],
        "rule_family": result["rule_family"],
        "applies": applies,
        "measured": result["measured"],
        "evaluated": result["evaluated"],
        "state": result["state"],
        "blocking_reason": result.get("skip_or_block_reason"),
    }


def _resolved_rule(profile: dict[str, Any], metric_id: str, family: str, active: set[str], direction: set[str], not_observable: set[str]) -> dict[str, Any]:
    if metric_id in active:
        status = "active_diagnostic"
    elif metric_id in direction:
        status = "blocked_missing_reference"
    elif metric_id in not_observable:
        status = "not_observable_with_current_pipeline"
    else:
        status = "measurement_only"
    threshold = deepcopy(profile["thresholds"].get(metric_id))
    unit = threshold["unit"] if threshold else _unit_for(metric_id)
    return {
        "rule_id": f"TK3D-T1-V3-{metric_id.upper().replace('_', '-')}",
        "metric_id": metric_id,
        "criterion_id": f"technical_accuracy.{family}.{metric_id}",
        "rule_family": family,
        "label": metric_id.replace("_", " "),
        "description": f"Temporary WholeBody-133 technical-accuracy diagnostic for {metric_id}.",
        "applicable_movements": [f"M{index:02d}" for index in range(1, 19)],
        "applicable_techniques": ["all_declared_taegeuk_1_techniques"],
        "evaluation_phase": "phase_specific_or_fixation",
        "aggregation": "robust_window_median_or_explicit_state",
        "expected_value_or_range": None if threshold is None else threshold["value"],
        "threshold": threshold,
        "uncertainty_band": None if threshold is None else threshold["uncertainty_band"],
        "unit": unit,
        "required_landmarks": _required_landmarks(metric_id),
        "minimum_valid_samples": profile["quality_gates"]["min_valid_samples"],
        "required_reference_frame": "athlete_local_direction" if metric_id in direction else "gravity_relative_or_body_relative",
        "evidence_quality_requirements": ["exact_133_points", "finite_required_landmarks", "camera_and_reprojection_gate", "minimum_valid_samples"],
        "status": status,
        "origin": profile["policy"]["provenance"],
        "rationale": profile["threshold_policy"]["rationale"],
        "score_effect": None,
        "deduction_enabled": False,
        "failure_behavior": profile["failure_behavior"],
        "skip_reason_codes": profile["skip_reason_codes"],
        "decision_status": profile["policy"]["decision_status"],
        "rule_eligibility": profile["policy"]["rule_eligibility"],
        "numeric_score_enabled": False,
        "deduction_points": None,
        "measurement_evaluator_status": (
            "not_applicable_pipeline_limit"
            if status == "not_observable_with_current_pipeline"
            else "implemented"
        ),
        "measurement_evaluator_id": (
            None
            if status == "not_observable_with_current_pipeline"
            else f"wholebody_133_geometry_v3.{family}"
        ),
    }


def _rule_applies(rule: dict[str, Any], contract: dict[str, Any]) -> bool:
    if rule["rule_family"] == "kick" and "ap_chagi" not in contract["technique_types"]:
        return False
    return contract["movement_id"] in rule["applicable_movements"]


def _validate_contract_tables(stance: Any, techniques: Any) -> None:
    if not isinstance(stance, dict) or set(stance) != {"ap_seogi", "ap_gubi"}:
        raise ScoringContractError("stance_contracts must define ap_seogi and ap_gubi")
    stance_keys = {"length_leg_ratio", "width_shoulder_ratio", "front_knee_included_angle_deg", "rear_knee_included_angle_deg", "foot_target_yaw_uncertainty_deg"}
    for name, contract in stance.items():
        _exact_keys(contract, stance_keys, f"stance contract {name}")
        for key in stance_keys - {"foot_target_yaw_uncertainty_deg"}:
            values = contract[key]
            if not isinstance(values, list) or len(values) != 2 or not all(np.isfinite(values)) or values[0] > values[1]:
                raise ScoringContractError(f"invalid range {name}.{key}")
        _nonnegative(contract["foot_target_yaw_uncertainty_deg"], f"{name}.foot_target_yaw_uncertainty_deg")
    technique_keys = {"target_region", "target_height", "lateral_position", "depth", "elbow_relationship", "wrist_relationship", "preparation", "reaction_hand", "allowed_torso_pelvis_yaw_deg"}
    required_techniques = {
        "arae_makki",
        "momtong_jireugi",
        "momtong_an_makki",
        "eolgul_makki",
        "ap_chagi",
        "kihap",
    }
    if not isinstance(techniques, dict) or set(techniques) != required_techniques:
        raise ScoringContractError("technique_contracts must define every Taegeuk 1 technique")
    for name, contract in techniques.items():
        _exact_keys(contract, technique_keys, f"technique contract {name}")
        for key in technique_keys - {"allowed_torso_pelvis_yaw_deg"}:
            _nonempty(contract[key], f"{name}.{key}")
        _nonnegative(contract["allowed_torso_pelvis_yaw_deg"], f"{name}.allowed_torso_pelvis_yaw_deg")


def _id_set(values: Any, catalog: dict[str, str], label: str) -> set[str]:
    if not isinstance(values, list) or len(values) != len(set(values)):
        raise ScoringContractError(f"{label} must be a unique list")
    result = set(values)
    unknown = result - set(catalog)
    if unknown:
        raise ScoringContractError(f"{label} contains unknown metrics: {sorted(unknown)}")
    return result


def _required_landmarks(metric_id: str) -> list[int]:
    body = COCO_BODY_JOINTS
    torso = [body[name] for name in ("left_shoulder", "right_shoulder", "left_hip", "right_hip")]
    arms = [
        body[name]
        for name in (
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
        )
    ]
    legs = [
        body[name]
        for name in (
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
        )
    ]
    feet = list(COCO_FOOT_JOINTS.values())
    hands = list(range(COCO_LEFT_HAND_OFFSET, COCO_WHOLEBODY_KEYPOINTS))
    palm_points = [
        offset + COCO_HAND_LOCAL_JOINTS[name]
        for offset in (COCO_LEFT_HAND_OFFSET, COCO_RIGHT_HAND_OFFSET)
        for name in ("wrist", "index_mcp", "middle_mcp", "ring_mcp", "pinky_mcp")
    ]
    eye_clusters = list(range(23 + 36, 23 + 48))
    brow_and_chin = list(range(23 + 17, 23 + 27)) + [23 + 8]
    contract_only = {
        "required_phase_coverage",
        "movement_contract_exists",
        "movement_timeline_segment_exists",
        "expected_movement_order",
        "movement_phase_order",
        "expected_technique_contract_resolved",
        "expected_stance_contract_resolved",
        "expected_active_side_contract_resolved",
        "expected_direction_contract_resolved",
    }
    if metric_id in contract_only or metric_id in {
        "centre_of_mass",
        "centre_of_pressure",
        "actual_weight_distribution",
        "ground_reaction_force",
        "impact_power",
        "muscle_tension",
    }:
        return []
    if metric_id == "head_face_geometry_quality":
        return sorted(
            {
                body["nose"],
                body["left_eye"],
                body["right_eye"],
                body["left_ear"],
                body["right_ear"],
                *COCO_FACE_INDICES,
                *torso,
            }
        )
    if metric_id.startswith("head_") or metric_id == "actual_eye_gaze":
        required = eye_clusters + torso
        if "pitch" in metric_id:
            required += brow_and_chin
        return sorted(set(required))
    if metric_id in {"foot_fixation_slip_body_ratio", "stance_fixation_dispersion_body_ratio"}:
        return sorted(set(legs + feet + torso))
    if metric_id in {"active_hand_fixation_stability", "reaction_hand_fixation_stability"}:
        return sorted(set(arms + palm_points + torso))
    if metric_id == "active_hand_stance_settle_offset":
        return sorted(set(arms + legs + torso))
    if any(token in metric_id for token in ("foot", "stance", "knee", "leg", "landing", "pivot")):
        return sorted(set(legs + feet + torso))
    if any(token in metric_id for token in ("arm", "elbow", "wrist", "hand", "fist", "chamber")):
        if any(token in metric_id for token in ("hand", "fist")):
            return sorted(set(arms + hands + torso))
        return sorted(set(arms + torso))
    return torso


def _build_landmark_inventory(rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Enumerate all 133 points without pretending each point is an error criterion."""
    by_index: dict[int, list[dict[str, Any]]] = {index: [] for index in range(COCO_WHOLEBODY_KEYPOINTS)}
    for rule in rules:
        for index in rule["required_landmarks"]:
            by_index[index].append(rule)

    rows: list[dict[str, Any]] = []
    for index in range(COCO_WHOLEBODY_KEYPOINTS):
        region, name = _landmark_identity(index)
        linked = by_index[index]
        statuses = sorted({rule["status"] for rule in linked})
        active = [rule for rule in linked if rule["status"] == "active_diagnostic"]
        if active:
            usage_status = "declared_by_active_diagnostic_rule"
            reason = None
        elif linked:
            usage_status = "declared_by_non_active_rule_only"
            reason = "no_reliable_active_threshold_or_evaluator_for_this_landmark_role"
        else:
            usage_status = "not_declared_by_any_rule"
            reason = "no_independent_technically_defensible_accuracy_criterion"
        rows.append(
            {
                "landmark_index": index,
                "landmark_name": name,
                "region": region,
                "usage_status": usage_status,
                "blocking_or_scope_reason": reason,
                "declared_rule_count": len(linked),
                "active_diagnostic_rule_count": len(active),
                "rule_statuses": statuses,
                "rule_ids": [rule["rule_id"] for rule in linked],
                "active_diagnostic_rule_ids": [rule["rule_id"] for rule in active],
            }
        )
    return rows


def _landmark_identity(index: int) -> tuple[str, str]:
    if index < len(COCO_BODY_JOINT_NAMES):
        return "body", COCO_BODY_JOINT_NAMES[index]
    foot_name = next((name for name, value in COCO_FOOT_JOINTS.items() if value == index), None)
    if foot_name is not None:
        return "foot", foot_name
    if index in COCO_FACE_INDICES:
        local = index - COCO_FACE_INDICES[0]
        return "face", f"face_{local:02d}_{_face_region(local)}"
    side = "left" if index < COCO_RIGHT_HAND_OFFSET else "right"
    offset = COCO_LEFT_HAND_OFFSET if side == "left" else COCO_RIGHT_HAND_OFFSET
    local = index - offset
    joint_name = next(name for name, value in COCO_HAND_LOCAL_JOINTS.items() if value == local)
    return f"{side}_hand", f"{side}_{joint_name}"


def _face_region(local_index: int) -> str:
    if local_index <= 16:
        return "jaw_contour"
    if local_index <= 26:
        return "eyebrow"
    if local_index <= 35:
        return "nose"
    if local_index <= 47:
        return "eye"
    if local_index <= 59:
        return "outer_lip"
    return "inner_lip"


def _unit_for(metric_id: str) -> str:
    if metric_id.endswith("_deg"):
        return "deg"
    if metric_id.endswith("_sec") or metric_id.endswith("_offset"):
        return "sec"
    if metric_id.endswith("_unmeasurable_reason"):
        return "reason_code"
    if metric_id.endswith("_geometry_quality") or metric_id.endswith("_evidence_quality"):
        return "ratio"
    if any(token in metric_id for token in ("state", "match", "observed", "exists", "resolved", "coverage", "order", "completion", "observability")):
        return "state"
    if "body_ratio" in metric_id or metric_id.endswith("_ratio"):
        return "body_ratio"
    if "dispersion" in metric_id or "stability" in metric_id or "correction" in metric_id or "displacement" in metric_id or "drift" in metric_id or "slip" in metric_id:
        return "body_ratio"
    return "ratio"


def _bool_measurement(value: bool) -> dict[str, Any]:
    return {"value": bool(value), "unit": "bool", "quality_status": "measured", "evidence": None, "source_metric_id": None}


def _window_scalar(arrays: dict[str, Any], frames: np.ndarray, function: Any) -> float | None:
    values = [function(arrays, int(frame)) for frame in frames]
    finite = np.asarray([value for value in values if value is not None and np.isfinite(value)], dtype=float)
    return None if len(finite) < 3 else float(np.median(finite))


def _point(arrays: dict[str, Any], frame: int, name: str) -> np.ndarray | None:
    index = COCO_BODY_JOINTS[name]
    value = arrays["points"][frame, index]
    return value if np.all(np.isfinite(value)) else None


def _line_roll(arrays: dict[str, Any], frame: int, part: str) -> float | None:
    left, right = _point(arrays, frame, f"left_{part}"), _point(arrays, frame, f"right_{part}")
    if left is None or right is None:
        return None
    vector = right - left
    horizontal = float(np.linalg.norm(vector[:2]))
    return None if horizontal <= 1e-8 else float(np.degrees(np.arctan2(vector[2], horizontal)))


def _head_roll(arrays: dict[str, Any], frame: int) -> float | None:
    left = _face_cluster(arrays, frame, range(36, 42))
    right = _face_cluster(arrays, frame, range(42, 48))
    if left is None or right is None:
        return None
    eye_roll = _roll_vector(right - left)
    shoulder_roll = _line_roll(arrays, frame, "shoulder")
    return None if eye_roll is None or shoulder_roll is None else eye_roll - shoulder_roll


def _head_pitch_unsigned(arrays: dict[str, Any], frame: int) -> float | None:
    brow = _face_cluster(arrays, frame, range(17, 27))
    chin = _face_cluster(arrays, frame, [8])
    ls, rs, lh, rh = (
        _point(arrays, frame, name)
        for name in ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
    )
    if brow is None or chin is None or any(point is None for point in (ls, rs, lh, rh)):
        return None
    face_up = brow - chin
    torso_up = (ls + rs) / 2.0 - (lh + rh) / 2.0
    return _angle_between(face_up, torso_up)


def _face_cluster(arrays: dict[str, Any], frame: int, local_indices: Any) -> np.ndarray | None:
    values = arrays["points"][frame, [23 + int(index) for index in local_indices]]
    valid = values[np.all(np.isfinite(values), axis=1)]
    return None if len(valid) < 1 else np.median(valid, axis=0)


def _orientation_vector(arrays: dict[str, Any], frame: int, part: str) -> np.ndarray | None:
    if part == "head":
        left = _face_cluster(arrays, frame, range(36, 42))
        right = _face_cluster(arrays, frame, range(42, 48))
    else:
        left = _point(arrays, frame, f"left_{part}")
        right = _point(arrays, frame, f"right_{part}")
    if left is None or right is None:
        return None
    vector = right - left
    norm = float(np.linalg.norm(vector))
    return None if norm <= 1e-8 else vector / norm


def _orientation_dispersion(arrays: dict[str, Any], frames: np.ndarray, part: str) -> float | None:
    vectors = [
        vector
        for frame in frames
        if (vector := _orientation_vector(arrays, int(frame), part)) is not None
    ]
    if len(vectors) < 3:
        return None
    centre = np.median(np.asarray(vectors), axis=0)
    norm = float(np.linalg.norm(centre))
    if norm <= 1e-8:
        return None
    centre /= norm
    angles = [_angle_between(vector, centre) for vector in vectors]
    finite = [angle for angle in angles if angle is not None]
    return None if len(finite) < 3 else float(np.percentile(finite, 95))


def _orientation_drift(arrays: dict[str, Any], frames: np.ndarray, part: str) -> float | None:
    if len(frames) < 2:
        return None
    first = _orientation_vector(arrays, int(frames[0]), part)
    last = _orientation_vector(arrays, int(frames[-1]), part)
    return None if first is None or last is None else _angle_between(first, last)


def _angle_between(first: np.ndarray, second: np.ndarray) -> float | None:
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 1e-8:
        return None
    cosine = float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _roll_vector(vector: np.ndarray) -> float | None:
    horizontal = float(np.linalg.norm(vector[:2]))
    return None if horizontal <= 1e-8 else float(np.degrees(np.arctan2(vector[2], horizontal)))


def _torso_lateral_bend(arrays: dict[str, Any], frame: int) -> float | None:
    ls, rs, lh, rh = (_point(arrays, frame, name) for name in ("left_shoulder", "right_shoulder", "left_hip", "right_hip"))
    if any(point is None for point in (ls, rs, lh, rh)):
        return None
    shoulder_centre, hip_centre = (ls + rs) / 2.0, (lh + rh) / 2.0
    torso = shoulder_centre - hip_centre
    lateral = rs - ls
    lateral[2] = 0.0
    if np.linalg.norm(torso) <= 1e-8 or np.linalg.norm(lateral) <= 1e-8:
        return None
    lateral /= np.linalg.norm(lateral)
    return float(np.degrees(np.arctan2(np.dot(torso, lateral), torso[2])))


def _height_asymmetry(arrays: dict[str, Any], frame: int, part: str) -> float | None:
    left, right = _point(arrays, frame, f"left_{part}"), _point(arrays, frame, f"right_{part}")
    scale = _body_scale(arrays, frame)
    return None if left is None or right is None or scale is None else abs(float(left[2] - right[2])) / scale


def _shoulder_pelvis_roll_difference(arrays: dict[str, Any], frame: int) -> float | None:
    shoulder, pelvis = _line_roll(arrays, frame, "shoulder"), _line_roll(arrays, frame, "hip")
    return None if shoulder is None or pelvis is None else abs(shoulder - pelvis)


def _body_scale(arrays: dict[str, Any], frame: int) -> float | None:
    ls, rs, lh, rh = (_point(arrays, frame, name) for name in ("left_shoulder", "right_shoulder", "left_hip", "right_hip"))
    if any(point is None for point in (ls, rs, lh, rh)):
        return None
    value = float(np.linalg.norm((ls + rs) / 2.0 - (lh + rh) / 2.0))
    return value if value > 1e-8 else None


def _limb_scale(arrays: dict[str, Any], frame: int, kind: str, side: str | None) -> float | None:
    if side not in {"left", "right"}:
        return None
    names = (f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist") if kind == "arm" else (f"{side}_hip", f"{side}_knee", f"{side}_ankle")
    a, b, c = (_point(arrays, frame, name) for name in names)
    if any(point is None for point in (a, b, c)):
        return None
    value = float(np.linalg.norm(a - b) + np.linalg.norm(b - c))
    return value if value > 1e-8 else None


def _joint_dispersion(arrays: dict[str, Any], frames: np.ndarray, joint: str, kind: str, side: str | None) -> float | None:
    values = [(_point(arrays, int(frame), joint), _limb_scale(arrays, int(frame), kind, side)) for frame in frames]
    points = np.asarray([point for point, scale in values if point is not None and scale is not None], dtype=float)
    scales = np.asarray([scale for point, scale in values if point is not None and scale is not None], dtype=float)
    if len(points) < 3:
        return None
    centre = np.median(points, axis=0)
    return float(np.percentile(np.linalg.norm(points - centre, axis=1) / scales, 95))


def _late_correction(arrays: dict[str, Any], frames: np.ndarray, joint: str, kind: str, side: str | None) -> float | None:
    if len(frames) < 2:
        return None
    first, last = _point(arrays, int(frames[0]), joint), _point(arrays, int(frames[-1]), joint)
    scale = _limb_scale(arrays, int(frames[0]), kind, side)
    return None if first is None or last is None or scale is None else float(np.linalg.norm(last - first) / scale)


def _paired_dispersion(arrays: dict[str, Any], frames: np.ndarray, joint: str, side_a: str | None, side_b: str | None, kind: str) -> float | None:
    values = [_joint_dispersion(arrays, frames, f"{side}_{joint}", kind, side) for side in (side_a, side_b)]
    finite = [value for value in values if value is not None]
    return max(finite) if len(finite) == 2 else None


def _arm_plane_deviation(arrays: dict[str, Any], frames: np.ndarray, side: str | None) -> float | None:
    if side not in {"left", "right"}:
        return None
    values: list[float] = []
    for frame in frames:
        shoulder, elbow, wrist = (_point(arrays, int(frame), f"{side}_{name}") for name in ("shoulder", "elbow", "wrist"))
        scale = _limb_scale(arrays, int(frame), "arm", side)
        if shoulder is None or elbow is None or wrist is None or scale is None:
            continue
        line = wrist - shoulder
        denom = float(np.dot(line, line))
        if denom <= 1e-8:
            continue
        nearest = shoulder + np.dot(elbow - shoulder, line) / denom * line
        values.append(float(np.linalg.norm(elbow - nearest) / scale))
    return None if len(values) < 3 else float(np.median(values))


def _joint_angle(arrays: dict[str, Any], frame: int, side: str | None) -> float | None:
    if side not in {"left", "right"}:
        return None
    hip, knee, ankle = (_point(arrays, frame, f"{side}_{name}") for name in ("hip", "knee", "ankle"))
    if hip is None or knee is None or ankle is None:
        return None
    a, b = hip - knee, ankle - knee
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return None if denom <= 1e-8 else float(np.degrees(np.arccos(np.clip(np.dot(a, b) / denom, -1.0, 1.0))))


def _centre_dispersion(arrays: dict[str, Any], frames: np.ndarray, part: str) -> float | None:
    points: list[np.ndarray] = []
    scales: list[float] = []
    for frame in frames:
        left, right = _point(arrays, int(frame), f"left_{part}"), _point(arrays, int(frame), f"right_{part}")
        scale = _body_scale(arrays, int(frame))
        if left is not None and right is not None and scale is not None:
            points.append((left + right) / 2.0)
            scales.append(scale)
    if len(points) < 3:
        return None
    values = np.asarray(points)
    centre = np.median(values, axis=0)
    return float(np.percentile(np.linalg.norm(values - centre, axis=1) / np.asarray(scales), 95))


def _body_height_change(arrays: dict[str, Any], frames: np.ndarray) -> float | None:
    heights: list[float] = []
    scales: list[float] = []
    for frame in frames:
        shoulders = [_point(arrays, int(frame), name) for name in ("left_shoulder", "right_shoulder")]
        ankles = [_point(arrays, int(frame), name) for name in ("left_ankle", "right_ankle")]
        scale = _body_scale(arrays, int(frame))
        if all(point is not None for point in shoulders + ankles) and scale is not None:
            heights.append(float(((shoulders[0] + shoulders[1]) / 2.0)[2] - ((ankles[0] + ankles[1]) / 2.0)[2]))
            scales.append(scale)
    return None if len(heights) < 3 else float((max(heights) - min(heights)) / np.median(scales))


def _unit_vector(value: Any, label: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)) or np.linalg.norm(vector) <= 1e-8:
        raise ScoringContractError(f"{label} must be a finite non-degenerate 3-vector")
    return vector / np.linalg.norm(vector)


def _exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        raise ScoringContractError(f"{label} keys must be exactly {sorted(expected)}")


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ScoringContractError(f"{label} must be a non-empty string")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ScoringContractError(f"{label} must be finite")
    result = float(value)
    if not np.isfinite(result):
        raise ScoringContractError(f"{label} must be finite")
    return result


def _nonnegative(value: Any, label: str) -> float:
    result = _finite(value, label)
    if result < 0:
        raise ScoringContractError(f"{label} must be non-negative")
    return result


def _positive(value: Any, label: str) -> float:
    result = _finite(value, label)
    if result <= 0:
        raise ScoringContractError(f"{label} must be positive")
    return result


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ScoringContractError(f"{label} must be a positive integer")
    return value


def _nonnegative_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ScoringContractError(f"{label} must be a non-negative integer")
    return value


def _probability(value: Any, label: str) -> float:
    result = _finite(value, label)
    if not 0.0 <= result <= 1.0:
        raise ScoringContractError(f"{label} must be between 0 and 1")
    return result


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
