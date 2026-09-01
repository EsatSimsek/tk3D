from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from src.data_structures import COCO_BODY_JOINTS, COCO_WHOLEBODY_KEYPOINTS
from src.poomsae_scoring.contracts import ScoringContractError, load_yaml_mapping
from src.poomsae_scoring.technical_accuracy import (
    build_technical_accuracy_diagnostics,
    evaluate_temporary_threshold,
    resolve_movement_accuracy_contracts,
    validate_technical_accuracy_profile,
)


VALIDATION_TOP_LEVEL_KEYS = {
    "schema_version",
    "validation_id",
    "status",
    "target_profile_id",
    "required_keypoint_count",
    "synthetic_fixture_id",
    "classification_cases",
    "geometry_scenarios",
    "disclaimer",
}
VALIDATION_CASES = {
    "pass",
    "boundary",
    "opposite_boundary",
    "fail",
    "opposite_fail",
    "missing",
    "nan",
    "positive_infinity",
    "negative_infinity",
    "wrong_type",
}
SCENARIO_KEYS = {"scenario_id", "fixture_mutation", "expectation", "target_metrics"}
MUTATIONS = {
    "none",
    "invalidate_face",
    "invalidate_hands",
    "invalidate_feet",
    "lower_camera_count",
    "raise_reprojection_error",
    "collapse_face",
    "inject_post_fixation_drift",
    "mirror_left_right",
    "reduce_to_body17",
}
EXPECTATIONS = {
    "complete_score_neutral_contract",
    "target_metrics_unmeasurable_without_candidate",
    "target_metrics_increase_from_baseline",
    "mirror_involution_and_stability_invariance",
    "scoring_contract_error",
    "direction_rules_blocked",
    "direction_rules_evaluated_when_measurable",
}
SCENARIO_CONTRACTS = {
    "baseline_full_133": ("none", "complete_score_neutral_contract", ()),
    "missing_face_evidence": (
        "invalidate_face",
        "target_metrics_unmeasurable_without_candidate",
        (
            "head_roll_relative_shoulders_deg",
            "head_pitch_relative_torso_deg",
            "head_fixation_orientation_dispersion_p95_deg",
            "head_post_fixation_drift_deg",
        ),
    ),
    "missing_hand_evidence": (
        "invalidate_hands",
        "target_metrics_unmeasurable_without_candidate",
        ("active_hand_fixation_stability", "reaction_hand_fixation_stability"),
    ),
    "missing_foot_evidence": (
        "invalidate_feet",
        "target_metrics_unmeasurable_without_candidate",
        ("foot_fixation_slip_body_ratio", "stance_fixation_dispersion_body_ratio"),
    ),
    "insufficient_camera_evidence": (
        "lower_camera_count",
        "target_metrics_unmeasurable_without_candidate",
        (
            "head_roll_relative_shoulders_deg",
            "torso_lateral_bend_deg",
            "active_hand_fixation_stability",
            "foot_fixation_slip_body_ratio",
        ),
    ),
    "excessive_reprojection_error": (
        "raise_reprojection_error",
        "target_metrics_unmeasurable_without_candidate",
        (
            "head_roll_relative_shoulders_deg",
            "torso_lateral_bend_deg",
            "active_hand_fixation_stability",
            "foot_fixation_slip_body_ratio",
        ),
    ),
    "degenerate_face_geometry": (
        "collapse_face",
        "target_metrics_unmeasurable_without_candidate",
        (
            "head_roll_relative_shoulders_deg",
            "head_pitch_relative_torso_deg",
            "head_fixation_orientation_dispersion_p95_deg",
        ),
    ),
    "post_fixation_drift": (
        "inject_post_fixation_drift",
        "target_metrics_increase_from_baseline",
        (
            "head_post_fixation_drift_deg",
            "active_hand_fixation_stability",
            "foot_fixation_slip_body_ratio",
        ),
    ),
    "left_right_mirror": (
        "mirror_left_right",
        "mirror_involution_and_stability_invariance",
        (
            "head_fixation_orientation_dispersion_p95_deg",
            "foot_fixation_slip_body_ratio",
            "active_hand_fixation_stability",
        ),
    ),
    "body17_contract_rejected": ("reduce_to_body17", "scoring_contract_error", ()),
    "missing_direction_binding": ("none", "direction_rules_blocked", ()),
    "valid_direction_binding": ("none", "direction_rules_evaluated_when_measurable", ()),
}


def load_rule_accuracy_validation_profile(path: str | Path) -> dict[str, Any]:
    return validate_rule_accuracy_validation_profile(
        load_yaml_mapping(path, label="technical accuracy rule validation profile")
    )


def validate_rule_accuracy_validation_profile(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ScoringContractError("rule accuracy validation profile must be a mapping")
    data = deepcopy(payload)
    _exact_keys(data, VALIDATION_TOP_LEVEL_KEYS, "rule accuracy validation profile")
    if data["schema_version"] != 1 or data["status"] != "synthetic_engineering_validation_only":
        raise ScoringContractError("rule accuracy validation profile must be schema 1 synthetic engineering validation only")
    if data["required_keypoint_count"] != COCO_WHOLEBODY_KEYPOINTS:
        raise ScoringContractError("rule accuracy validation requires exactly 133 keypoints")
    for key in ("validation_id", "target_profile_id", "synthetic_fixture_id", "disclaimer"):
        if not isinstance(data[key], str) or not data[key].strip():
            raise ScoringContractError(f"{key} must be a non-empty string")
    cases = data["classification_cases"]
    if not isinstance(cases, list) or set(cases) != VALIDATION_CASES or len(cases) != len(VALIDATION_CASES):
        raise ScoringContractError("classification_cases must contain the exact unique validation case inventory")
    scenarios = data["geometry_scenarios"]
    if not isinstance(scenarios, list) or not scenarios:
        raise ScoringContractError("geometry_scenarios must be a non-empty list")
    ids: set[str] = set()
    for scenario in scenarios:
        if not isinstance(scenario, dict):
            raise ScoringContractError("each geometry scenario must be a mapping")
        _exact_keys(scenario, SCENARIO_KEYS, "geometry scenario")
        scenario_id = scenario["scenario_id"]
        if not isinstance(scenario_id, str) or not scenario_id or scenario_id in ids:
            raise ScoringContractError("geometry scenario ids must be unique non-empty strings")
        ids.add(scenario_id)
        if scenario["fixture_mutation"] not in MUTATIONS:
            raise ScoringContractError(f"unsupported fixture mutation: {scenario['fixture_mutation']}")
        if scenario["expectation"] not in EXPECTATIONS:
            raise ScoringContractError(f"unsupported geometry expectation: {scenario['expectation']}")
        if not isinstance(scenario["target_metrics"], list) or not all(
            isinstance(metric_id, str) and metric_id for metric_id in scenario["target_metrics"]
        ):
            raise ScoringContractError("geometry target_metrics must be a list of non-empty strings")
        expected_contract = SCENARIO_CONTRACTS.get(scenario_id)
        actual_contract = (
            scenario["fixture_mutation"],
            scenario["expectation"],
            tuple(scenario["target_metrics"]),
        )
        if expected_contract is not None and actual_contract != expected_contract:
            raise ScoringContractError(f"geometry scenario contract mismatch: {scenario_id}")
    if ids != set(SCENARIO_CONTRACTS):
        raise ScoringContractError(
            "geometry scenario inventory must be exact; "
            f"missing={sorted(set(SCENARIO_CONTRACTS) - ids)}, "
            f"unknown={sorted(ids - set(SCENARIO_CONTRACTS))}"
        )
    return data


def build_synthetic_wholebody_pose(frame_count: int) -> dict[str, Any]:
    if not isinstance(frame_count, int) or frame_count <= 0:
        raise ScoringContractError("synthetic fixture frame_count must be a positive integer")
    points = np.zeros((frame_count, COCO_WHOLEBODY_KEYPOINTS, 3), dtype=float)
    base = {
        "nose": [0.0, 0.0, 1.75],
        "left_eye": [-0.03, 0.0, 1.78],
        "right_eye": [0.03, 0.0, 1.78],
        "left_shoulder": [-0.20, 0.0, 1.45],
        "right_shoulder": [0.20, 0.0, 1.45],
        "left_elbow": [-0.35, 0.0, 1.20],
        "right_elbow": [0.35, 0.0, 1.20],
        "left_wrist": [-0.45, 0.0, 1.05],
        "right_wrist": [0.45, 0.0, 1.05],
        "left_hip": [-0.14, 0.0, 0.95],
        "right_hip": [0.14, 0.0, 0.95],
        "left_knee": [-0.14, 0.05, 0.52],
        "right_knee": [0.14, -0.05, 0.52],
        "left_ankle": [-0.14, 0.15, 0.08],
        "right_ankle": [0.14, -0.15, 0.08],
    }
    for name, value in base.items():
        points[:, COCO_BODY_JOINTS[name], :] = value
    points[:, 17:23, :] = np.asarray(
        [
            [-0.14, 0.28, 0.04],
            [-0.10, 0.28, 0.04],
            [-0.14, 0.10, 0.04],
            [0.14, -0.02, 0.04],
            [0.10, -0.02, 0.04],
            [0.14, -0.20, 0.04],
        ]
    )
    points[:, 23:91, :] = [0.0, 0.0, 1.75]
    points[:, 23 + 8, :] = [0.0, 0.0, 1.68]
    points[:, 23 + 17 : 23 + 22, :] = [-0.03, 0.0, 1.82]
    points[:, 23 + 22 : 23 + 27, :] = [0.03, 0.0, 1.82]
    points[:, 23 + 27 : 23 + 36, :] = [0.0, 0.03, 1.76]
    points[:, 23 + 36 : 23 + 42, :] = [-0.03, 0.0, 1.78]
    points[:, 23 + 42 : 23 + 48, :] = [0.03, 0.0, 1.78]
    for offset, sign in ((91, -1.0), (112, 1.0)):
        wrist = np.asarray([0.45 * sign, 0.0, 1.05])
        points[:, offset, :] = wrist
        for local_index in range(1, 21):
            finger = min(4, (local_index - 1) // 4)
            depth = 0.025 * (1 + (local_index - 1) % 4)
            points[:, offset + local_index, :] = wrist + [0.008 * (finger - 2), depth, 0.0]
    phase = 0.002 * np.sin(np.linspace(0.0, 12.0 * np.pi, frame_count))
    points[:, 23 + 36 : 23 + 42, 2] += phase[:, None]
    points[:, 23 + 42 : 23 + 48, 2] -= phase[:, None]
    points[:, 91:133, 1] += phase[:, None]
    points[:, 17:23, 1] += phase[:, None]
    shape = (frame_count, COCO_WHOLEBODY_KEYPOINTS)
    return {
        "keypoints_3d_world": points.tolist(),
        "reliability_valid_mask": np.ones(shape, dtype=bool).tolist(),
        "used_cameras": np.full(shape, 2, dtype=int).tolist(),
        "reprojection_error": np.full(shape, 2.0).tolist(),
    }


def build_synthetic_wholebody_diagnostics(timeline: dict[str, Any]) -> dict[str, Any]:
    metrics = {
        "head_torso_yaw_mismatch_deg": (5.0, "deg"),
        "shoulder_hip_twist_deg": (5.0, "deg"),
        "torso_lean_p95_deg": (5.0, "deg"),
        "wrist_forearm_alignment_deg": (5.0, "deg"),
        "hand_foot_settle_difference_sec": (0.05, "sec"),
        "fixation_wrist_jitter_ratio": (0.01, "body_scale"),
        "executing_wrist_height_torso_ratio": (0.5, "torso_height"),
        "executing_elbow_deg": (165.0, "deg"),
        "reaction_hand_hip_distance_ratio": (0.1, "body_scale"),
        "stance_span_ratio": (0.5, "body_scale"),
        "front_knee_deg": (165.0, "deg"),
    }
    return {
        "status": "wholebody_diagnostics_only",
        "movement_timeline_id": timeline["timeline_id"],
        "movements": [
            {
                "movement_id": segment["movement_id"],
                "metrics": [
                    {
                        "metric_id": metric_id,
                        "value": value,
                        "unit": unit,
                        "measurement_evidence": {"scope": "synthetic_rule_validation"},
                    }
                    for metric_id, (value, unit) in metrics.items()
                ],
            }
            for segment in timeline["segments"]
        ],
    }


def build_rule_accuracy_validation(
    profile_payload: dict[str, Any],
    poomsae_spec: dict[str, Any],
    movement_timeline: dict[str, Any],
    validation_payload: dict[str, Any],
) -> dict[str, Any]:
    profile = (
        profile_payload
        if isinstance(profile_payload, dict) and "resolved_rules" in profile_payload
        else validate_technical_accuracy_profile(profile_payload)
    )
    validation = validate_rule_accuracy_validation_profile(validation_payload)
    if validation["target_profile_id"] != profile["profile_id"]:
        raise ScoringContractError("validation target profile binding mismatch")
    known_metrics = {rule["metric_id"] for rule in profile["resolved_rules"]}
    unknown_targets = {
        metric_id
        for scenario in validation["geometry_scenarios"]
        for metric_id in scenario["target_metrics"]
        if metric_id not in known_metrics
    }
    if unknown_targets:
        raise ScoringContractError(f"validation scenarios reference unknown metrics: {sorted(unknown_targets)}")

    inventory = [_inventory_row(rule) for rule in profile["resolved_rules"]]
    classifications = _classification_rows(profile, validation["classification_cases"])
    base_pose = build_synthetic_wholebody_pose(int(movement_timeline["frame_count"]))
    wholebody = build_synthetic_wholebody_diagnostics(movement_timeline)
    baseline = build_technical_accuracy_diagnostics(
        base_pose, poomsae_spec, movement_timeline, profile, wholebody
    )
    scenarios = [
        _run_geometry_scenario(
            scenario,
            base_pose,
            baseline,
            poomsae_spec,
            movement_timeline,
            profile,
            wholebody,
        )
        for scenario in validation["geometry_scenarios"]
    ]
    landmarks = [
        {**row, "passed": row["declared_rule_count"] > 0}
        for row in baseline["landmark_inventory"]
    ]
    passed = all(row["passed"] for row in inventory + landmarks + classifications + scenarios)
    state_counts = {
        state: sum(row["configured_state"] == state for row in inventory)
        for state in (
            "active_diagnostic",
            "measurement_only",
            "blocked_missing_reference",
            "not_observable_with_current_pipeline",
        )
    }
    return {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "validation_id": validation["validation_id"],
        "validation_scope": validation["status"],
        "target_profile_id": profile["profile_id"],
        "synthetic_fixture_id": validation["synthetic_fixture_id"],
        "required_keypoint_count": COCO_WHOLEBODY_KEYPOINTS,
        "scientific_accuracy_claim_allowed": False,
        "judge_calibration_claim_allowed": False,
        "score_effect": None,
        "deduction_effect": None,
        "readiness": {
            "synthetic_contract_validation_ready": passed,
            "external_rule_accuracy_ready": False,
            "judge_calibrated_ready": False,
            "production_threshold_ready": False,
            "official_scoring_ready": False,
            "blockers": [
                "no_rule_level_expert_labelled_positive_negative_dataset",
                "no_blinded_judge_agreement_study",
                "no_external_precision_recall_or_false_positive_analysis",
                "temporary_thresholds_not_calibrated_on_independent_data",
            ],
        },
        "summary": {
            "rule_count": len(inventory),
            "rule_inventory_passed_count": sum(row["passed"] for row in inventory),
            "active_rule_count": sum(row["configured_state"] == "active_diagnostic" for row in inventory),
            "rule_state_counts": state_counts,
            "landmark_count": len(landmarks),
            "landmark_passed_count": sum(row["passed"] for row in landmarks),
            "classification_case_count": len(classifications),
            "classification_passed_count": sum(row["passed"] for row in classifications),
            "geometry_scenario_count": len(scenarios),
            "geometry_scenario_passed_count": sum(row["passed"] for row in scenarios),
        },
        "rule_validation_inventory": inventory,
        "landmark_validation_inventory": landmarks,
        "classification_cases": classifications,
        "geometry_scenarios": scenarios,
        "limitations": [
            validation["disclaimer"],
            "Synthetic fixtures exercise software contracts, boundaries, missing evidence, and geometric sensitivity only.",
            "A rule passing this harness remains an unvalidated review diagnostic until external labelled data and judge agreement exist.",
        ],
    }


def _inventory_row(rule: dict[str, Any]) -> dict[str, Any]:
    status = rule["status"]
    landmarks = rule["required_landmarks"]
    landmarks_valid = (
        isinstance(landmarks, list)
        and len(landmarks) == len(set(landmarks))
        and all(isinstance(index, int) and 0 <= index < COCO_WHOLEBODY_KEYPOINTS for index in landmarks)
    )
    expected_evaluator = (
        "not_applicable_pipeline_limit"
        if status == "not_observable_with_current_pipeline"
        else "implemented"
    )
    passed = (
        rule["measurement_evaluator_status"] == expected_evaluator
        and rule["score_effect"] is None
        and rule["deduction_points"] is None
        and rule["numeric_score_enabled"] is False
        and rule["deduction_enabled"] is False
        and landmarks_valid
    )
    depth = {
        "active_diagnostic": "threshold_classification_and_fail_closed",
        "measurement_only": "evaluator_contract_and_score_neutrality",
        "blocked_missing_reference": "reference_binding_and_fail_closed",
        "not_observable_with_current_pipeline": "pipeline_limit_and_score_neutrality",
    }[status]
    return {
        "rule_id": rule["rule_id"],
        "metric_id": rule["metric_id"],
        "rule_family": rule["rule_family"],
        "configured_state": status,
        "validation_depth": depth,
        "required_landmark_count": len(landmarks),
        "required_landmarks_valid": landmarks_valid,
        "measurement_evaluator_status": rule["measurement_evaluator_status"],
        "score_neutral": rule["score_effect"] is None and rule["deduction_points"] is None,
        "passed": passed,
    }


def _classification_rows(profile: dict[str, Any], cases: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rule in profile["resolved_rules"]:
        if rule["status"] != "active_diagnostic":
            continue
        for case in cases:
            value, expected, applicable = _classification_input(rule["threshold"], case)
            actual = evaluate_temporary_threshold(value, rule["threshold"]) if applicable else "not_applicable"
            rows.append(
                {
                    "rule_id": rule["rule_id"],
                    "metric_id": rule["metric_id"],
                    "operator": None if rule["threshold"] is None else rule["threshold"]["operator"],
                    "case_id": case,
                    "input_value": value if isinstance(value, (bool, int, float)) and np.isfinite(value) else None,
                    "input_kind": _input_kind(value),
                    "applicable": applicable,
                    "expected_evaluation": expected,
                    "actual_evaluation": actual,
                    "passed": actual == expected,
                }
            )
    return rows


def _classification_input(threshold: dict[str, Any] | None, case: str) -> tuple[Any, str, bool]:
    special = {
        "missing": (None, "unmeasurable", True),
        "nan": (float("nan"), "unmeasurable", True),
        "positive_infinity": (float("inf"), "unmeasurable", True),
        "negative_infinity": (float("-inf"), "unmeasurable", True),
        "wrong_type": ("not-a-number", "unmeasurable", True),
    }
    if case in special:
        return special[case]
    if threshold is None:
        if case in {"boundary", "opposite_boundary", "opposite_fail"}:
            return None, "not_applicable", False
        value = case == "pass"
        return value, "within_screening_range" if value else "out_of_range", True
    operator = threshold["operator"]
    uncertainty = float(threshold["uncertainty_band"])
    limit = threshold["value"]
    epsilon = max(1e-6, uncertainty + 1.0)
    if operator == "range":
        low, high = map(float, limit)
        values = {
            "pass": (low + high) / 2.0,
            "boundary": high,
            "opposite_boundary": low,
            "fail": high + epsilon,
            "opposite_fail": low - epsilon,
        }
    elif operator in {"max", "abs_max"}:
        high = float(limit)
        values = {
            "pass": max(0.0, high - uncertainty - 1e-3),
            "boundary": high,
            "fail": high + epsilon,
        }
        if operator == "abs_max":
            values.update({"opposite_boundary": -high, "opposite_fail": -(high + epsilon)})
    else:
        low = float(limit)
        values = {"pass": low + uncertainty + 1e-3, "boundary": low, "fail": low - epsilon}
    if case not in values:
        return None, "not_applicable", False
    expected = {
        "pass": "within_screening_range",
        "boundary": "boundary_uncertain" if uncertainty > 0 else "within_screening_range",
        "opposite_boundary": "boundary_uncertain" if uncertainty > 0 else "within_screening_range",
        "fail": "out_of_range",
        "opposite_fail": "out_of_range",
    }[case]
    return values[case], expected, True


def _run_geometry_scenario(
    scenario: dict[str, Any],
    base_pose: dict[str, Any],
    baseline: dict[str, Any],
    spec: dict[str, Any],
    timeline: dict[str, Any],
    profile: dict[str, Any],
    wholebody: dict[str, Any],
) -> dict[str, Any]:
    pose = _mutate_pose(base_pose, scenario["fixture_mutation"], timeline, spec, profile)
    expectation = scenario["expectation"]
    details: dict[str, Any] = {}
    try:
        direction = _direction_reference(timeline) if expectation == "direction_rules_evaluated_when_measurable" else None
        report = build_technical_accuracy_diagnostics(
            pose, spec, timeline, profile, wholebody, direction_reference=direction
        )
    except ScoringContractError as exc:
        passed = expectation == "scoring_contract_error"
        details = {"raised": type(exc).__name__, "message": str(exc)}
    else:
        passed, details = _evaluate_geometry_expectation(
            expectation, scenario["target_metrics"], report, baseline, pose, base_pose, profile
        )
    return {
        "scenario_id": scenario["scenario_id"],
        "fixture_mutation": scenario["fixture_mutation"],
        "expectation": expectation,
        "target_metrics": scenario["target_metrics"],
        "passed": passed,
        "details": details,
    }


def _evaluate_geometry_expectation(
    expectation: str,
    target_metrics: list[str],
    report: dict[str, Any],
    baseline: dict[str, Any],
    pose: dict[str, Any],
    base_pose: dict[str, Any],
    profile: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    rules = _movement_rule_map(report, "M01")
    baseline_rules = _movement_rule_map(baseline, "M01")
    if expectation == "complete_score_neutral_contract":
        all_runtime_rules = [
            rule for movement in report["movements"] for rule in movement["rules"]
        ]
        measurement_only = [rule for rule in all_runtime_rules if rule["state"] == "measurement_only"]
        passed = (
            report["total_score"] is None
            and report["deductions"] == []
            and report["summary"]["rule_count"] == len(profile["resolved_rules"])
            and report["summary"]["landmark_inventory_count"] == COCO_WHOLEBODY_KEYPOINTS
            and report["summary"]["landmarks_declared_by_any_rule_count"] == COCO_WHOLEBODY_KEYPOINTS
            and report["summary"]["score_effect_count"] == 0
            and report["summary"]["deduction_effect_count"] == 0
            and all(
                rule["score_effect"] is None
                and rule["deduction_points"] is None
                and rule["numeric_score_enabled"] is False
                and rule["deduction_enabled"] is False
                for rule in all_runtime_rules
            )
            and all(
                rule["evaluated"] is False and rule["decision_status"] == "no_candidate"
                for rule in measurement_only
            )
        )
        return passed, {
            "rule_count": report["summary"]["rule_count"],
            "runtime_rule_result_count": len(all_runtime_rules),
            "measurement_only_result_count": len(measurement_only),
            "score_effect_count": report["summary"]["score_effect_count"],
            "deduction_effect_count": report["summary"]["deduction_effect_count"],
        }
    if expectation == "target_metrics_unmeasurable_without_candidate":
        observations = {
            metric_id: {
                "baseline_state": baseline_rules[metric_id]["state"],
                "baseline_value": baseline_rules[metric_id]["value"],
                "state": rules[metric_id]["state"],
                "decision_status": rules[metric_id]["decision_status"],
                "reason": rules[metric_id].get("skip_or_block_reason"),
            }
            for metric_id in target_metrics
        }
        passed = all(
            item["baseline_value"] is not None
            and item["baseline_state"] in {"active_diagnostic", "measurement_only"}
            and item["state"] == "unmeasurable"
            and item["decision_status"] == "no_candidate"
            for item in observations.values()
        )
        return passed, {"observations": observations}
    if expectation == "target_metrics_increase_from_baseline":
        comparisons = {
            metric_id: {
                "baseline": baseline_rules[metric_id]["value"],
                "mutated": rules[metric_id]["value"],
            }
            for metric_id in target_metrics
        }
        passed = all(
            item["baseline"] is not None
            and item["mutated"] is not None
            and float(item["mutated"]) > float(item["baseline"]) + 1e-9
            for item in comparisons.values()
        )
        return passed, {"comparisons": comparisons}
    if expectation == "mirror_involution_and_stability_invariance":
        twice = _mirror_pose(pose)
        involution_by_array = {
            key: bool(np.allclose(np.asarray(base_pose[key]), np.asarray(twice[key])))
            for key in (
                "keypoints_3d_world",
                "reliability_valid_mask",
                "used_cameras",
                "reprojection_error",
            )
        }
        comparisons = {
            metric_id: {
                "baseline": baseline_rules[metric_id]["value"],
                "mirrored": rules[metric_id]["value"],
            }
            for metric_id in target_metrics
        }
        invariant = all(
            item["baseline"] is not None
            and item["mirrored"] is not None
            and np.isclose(float(item["baseline"]), float(item["mirrored"]), atol=1e-9)
            for item in comparisons.values()
        )
        nontrivial = all(abs(float(item["baseline"])) > 1e-9 for item in comparisons.values())
        return bool(all(involution_by_array.values()) and invariant and nontrivial), {
            "mirror_involution": all(involution_by_array.values()),
            "mirror_involution_by_array": involution_by_array,
            "nonzero_baseline_metrics": nontrivial,
            "comparisons": comparisons,
        }
    direction_ids = set(profile["direction_bound_rules"])
    direction_rows = [
        row
        for row in report["coverage_matrix"]
        if row["movement_id"] == "M01" and row["metric_id"] in direction_ids and row["applies"]
    ]
    if expectation == "direction_rules_blocked":
        passed = bool(direction_rows) and all(
            row["state"] == "blocked_missing_reference"
            and row["blocking_reason"] == "missing_athlete_local_direction_binding"
            for row in direction_rows
        )
        return passed, {"checked_rule_count": len(direction_rows)}
    if expectation == "direction_rules_evaluated_when_measurable":
        passed = bool(direction_rows) and not any(
            row["blocking_reason"] == "missing_athlete_local_direction_binding" for row in direction_rows
        ) and all(row["measured"] and row["evaluated"] for row in direction_rows)
        return passed, {
            "checked_rule_count": len(direction_rows),
            "evaluated_rule_count": sum(row["measured"] and row["evaluated"] for row in direction_rows),
        }
    return False, {"error": "expected ScoringContractError was not raised"}


def _mutate_pose(
    base_pose: dict[str, Any],
    mutation: str,
    timeline: dict[str, Any],
    spec: dict[str, Any],
    profile: dict[str, Any],
) -> dict[str, Any]:
    if mutation == "mirror_left_right":
        return _mirror_pose(base_pose)
    pose = deepcopy(base_pose)
    if mutation == "none":
        return pose
    if mutation == "reduce_to_body17":
        for key in ("keypoints_3d_world", "reliability_valid_mask", "used_cameras", "reprojection_error"):
            pose[key] = np.asarray(pose[key])[:, :17].tolist()
        return pose
    if mutation in {"invalidate_face", "invalidate_hands", "invalidate_feet"}:
        limits = {"invalidate_face": (23, 91), "invalidate_hands": (91, 133), "invalidate_feet": (17, 23)}
        start, end = limits[mutation]
        valid = np.asarray(pose["reliability_valid_mask"], dtype=bool)
        valid[:, start:end] = False
        pose["reliability_valid_mask"] = valid.tolist()
        return pose
    if mutation == "lower_camera_count":
        pose["used_cameras"] = np.ones_like(np.asarray(pose["used_cameras"], dtype=int)).tolist()
        return pose
    if mutation == "raise_reprojection_error":
        value = float(profile["quality_gates"]["max_reprojection_error_px"]) + 1.0
        pose["reprojection_error"] = np.full_like(
            np.asarray(pose["reprojection_error"], dtype=float), value
        ).tolist()
        return pose
    points = np.asarray(pose["keypoints_3d_world"], dtype=float)
    if mutation == "collapse_face":
        points[:, 23:91, :] = points[:, 23, :][:, None, :]
    elif mutation == "inject_post_fixation_drift":
        segment = timeline["segments"][0]
        fixation = int(segment["anchors"]["fixation"])
        end = int(segment["end_frame"])
        frames = np.arange(fixation, end + 1, dtype=int)
        alpha = np.linspace(0.0, 0.25, len(frames))
        points[frames, 23 + 36 : 23 + 42, 2] += alpha[:, None]
        points[frames, 23 + 42 : 23 + 48, 2] -= alpha[:, None]
        contract = resolve_movement_accuracy_contracts(spec, profile)[0]
        hand_start = 91 if contract["active_arm"] == "left" else 112
        points[frames, hand_start : hand_start + 21, 1] += alpha[:, None]
        foot_start = 17 if contract["lead_leg"] == "left" else 20
        points[frames, foot_start : foot_start + 3, 1] += alpha[:, None]
    else:
        raise ScoringContractError(f"unsupported fixture mutation: {mutation}")
    pose["keypoints_3d_world"] = points.tolist()
    return pose


def _mirror_pose(payload: dict[str, Any]) -> dict[str, Any]:
    pose = deepcopy(payload)
    permutation = np.arange(COCO_WHOLEBODY_KEYPOINTS)
    pairs = [
        (1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16),
        (17, 20), (18, 21), (19, 22),
    ]
    pairs.extend((91 + index, 112 + index) for index in range(21))
    face_pairs = [
        (0, 16), (1, 15), (2, 14), (3, 13), (4, 12), (5, 11), (6, 10), (7, 9),
        (17, 26), (18, 25), (19, 24), (20, 23), (21, 22),
        (31, 35), (32, 34), (36, 45), (37, 44), (38, 43), (39, 42), (40, 47), (41, 46),
        (48, 54), (49, 53), (50, 52), (55, 59), (56, 58), (60, 64), (61, 63), (65, 67),
    ]
    pairs.extend((23 + left, 23 + right) for left, right in face_pairs)
    for left, right in pairs:
        permutation[left], permutation[right] = permutation[right], permutation[left]
    for key in ("keypoints_3d_world", "reliability_valid_mask", "used_cameras", "reprojection_error"):
        array = np.asarray(pose[key])[:, permutation].copy()
        if key == "keypoints_3d_world":
            array[:, :, 0] *= -1.0
        pose[key] = array.tolist()
    return pose


def _movement_rule_map(report: dict[str, Any], movement_id: str) -> dict[str, dict[str, Any]]:
    movement = next(item for item in report["movements"] if item["movement_id"] == movement_id)
    return {rule["metric_id"]: rule for rule in movement["rules"]}


def _direction_reference(timeline: dict[str, Any]) -> dict[str, Any]:
    binding = timeline["source_binding"]
    return {
        "schema_version": 1,
        "session_id": binding["session_id"],
        "reference_pose_sha256": binding["pose_file_sha256"],
        "gravity_up_vector": [0.0, 0.0, 1.0],
        "initial_forward_vector": [0.0, 1.0, 0.0],
        "basis_source": "manually_declared_session_bound",
        "provenance": "synthetic validation binding; not production calibration",
        "quality_status": "validated_diagnostic_reference",
    }


def _input_kind(value: Any) -> str:
    if value is None:
        return "missing"
    if isinstance(value, (bool, np.bool_)):
        return "boolean"
    if not isinstance(value, (int, float, np.integer, np.floating)):
        return f"invalid_type:{type(value).__name__}"
    numeric = float(value)
    if np.isnan(numeric):
        return "nan"
    if np.isposinf(numeric):
        return "positive_infinity"
    if np.isneginf(numeric):
        return "negative_infinity"
    return "finite_numeric"


def _exact_keys(payload: dict[str, Any], expected: set[str], label: str) -> None:
    if set(payload) != expected:
        raise ScoringContractError(
            f"{label} keys must be exactly {sorted(expected)}; got {sorted(payload)}"
        )
