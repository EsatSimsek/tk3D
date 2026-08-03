from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Any

import numpy as np

from src.poomsae_scoring.contracts import (
    ScoringContractError,
    validate_engineering_profile,
    validate_movement_timeline,
    validate_poomsae_spec,
    validate_rule_pack,
)


def build_partial_engineering_trial(
    rule_pack: dict[str, Any],
    poomsae_spec: dict[str, Any],
    movement_timeline: dict[str, Any],
    engineering_profile: dict[str, Any],
    evidence_report: dict[str, Any],
    anchor_measurements: list[dict[str, Any]],
) -> dict[str, Any]:
    """Create a fail-closed partial trial score from explicitly non-official thresholds."""
    pack = validate_rule_pack(rule_pack)
    spec = validate_poomsae_spec(poomsae_spec)
    timeline = validate_movement_timeline(movement_timeline, spec)
    profile = validate_engineering_profile(engineering_profile)
    _validate_bindings(pack, spec, timeline, profile, evidence_report)
    if not isinstance(anchor_measurements, list):
        raise ScoringContractError("anchor_measurements must be a list")

    row_map = {
        (row.get("movement_id"), row.get("phase_id")): row
        for row in anchor_measurements
        if isinstance(row, dict)
    }
    segments = {segment["movement_id"]: segment for segment in timeline["segments"]}
    movements = {movement["movement_id"]: movement for movement in spec["movements"]}
    measurements: list[dict[str, Any]] = []

    for movement_id in profile["scope"]["movement_ids"]:
        movement = movements[movement_id]
        segment = segments[movement_id]
        for criterion in profile["criteria"]:
            if criterion["criterion_id"] not in movement["measurable_criteria"]:
                continue
            if not _selector_matches(criterion["selector"], movement):
                continue
            row = row_map.get((movement_id, criterion["phase_id"]))
            measurements.append(_evaluate_criterion(movement, segment, criterion, row, profile["quality_gates"]))

    candidate_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for measurement in measurements:
        if measurement["measurement_status"] == "candidate":
            key = (measurement["movement_id"], measurement["family"], measurement["phase_id"])
            candidate_groups[key].append(measurement)

    diagnostic_candidate_groups = 0
    for group in candidate_groups.values():
        ordered = sorted(group, key=lambda item: (-item["normalized_deviation"], item["criterion_id"]))
        selected = ordered[0]
        selected["trial_application_status"] = "review_candidate_no_score"
        selected["trial_deduction_points"] = 0.0
        diagnostic_candidate_groups += 1
        for duplicate in ordered[1:]:
            duplicate["trial_application_status"] = "not_applied_duplicate_family_phase"
            duplicate["trial_deduction_points"] = 0.0

    for measurement in measurements:
        if "trial_application_status" not in measurement:
            measurement["trial_application_status"] = "not_applicable"
            measurement["trial_deduction_points"] = 0.0

    reference_score = profile["scope"]["reference_accuracy_score"]
    not_measurable = [item for item in measurements if item["measurement_status"] == "not_measurable"]
    candidates = [item for item in measurements if item["measurement_status"] == "candidate"]
    measurable_count = len(measurements) - len(not_measurable)
    measurement_coverage_ratio = measurable_count / len(measurements) if measurements else 0.0
    return _json_safe(
        {
            "schema_version": 1,
            "status": "deprecated_body17_screening_no_score",
            "score_name": "TK3D deprecated BODY-17 screening diagnostics",
            "score_scope": {
                "recording_scope": timeline["coverage"]["recording_scope"],
                "observed_movement_ids": list(timeline["coverage"]["observed_movement_ids"]),
                "missing_movement_ids": list(timeline["coverage"]["missing_movement_ids"]),
                "comparable_to_complete_poomsae": False,
            },
            "accuracy_score": None,
            "applied_deductions": [],
            "partial_engineering_trial_score": None,
            "trial_score_confidence": "not_scored",
            "measurement_coverage_ratio": measurement_coverage_ratio,
            "reference_accuracy_score": reference_score,
            "total_engineering_trial_deduction": None,
            "trial_deductions": [],
            "judge_calibrated": False,
            "official_scoring_ready": False,
            "thresholds_official": False,
            "automatic_major_detection": {
                "enabled": False,
                "reason": "A numeric pose deviation cannot establish the WT wrong-action definition for a major mistake.",
                "candidate_count": 0,
            },
            "rule_pack": {"rule_pack_id": pack["rule_pack_id"], "version": pack["version"]},
            "poomsae_spec": {"poomsae_id": spec["poomsae_id"], "version": spec["version"]},
            "movement_timeline_id": timeline["timeline_id"],
            "engineering_profile": {
                "profile_id": profile["profile_id"],
                "version": profile["version"],
                "status": profile["status"],
                "threshold_origin": profile["provenance"]["threshold_origin"],
                "disclaimer": profile["provenance"]["disclaimer"],
            },
            "quality_gates": deepcopy(profile["quality_gates"]),
            "measurements": measurements,
            "summary": {
                "measurement_count": len(measurements),
                "measurable_count": measurable_count,
                "pass_count": sum(item["measurement_status"] == "pass" for item in measurements),
                "candidate_count": len(candidates),
                "not_measurable_count": len(not_measurable),
                "selected_trial_deduction_count": 0,
                "diagnostic_candidate_family_count": diagnostic_candidate_groups,
                "deduplicated_candidate_count": len(candidates) - diagnostic_candidate_groups,
            },
            "interpretation": (
                "The former BODY-17 numeric trial is disabled because it missed WholeBody technique errors and "
                "treated undetected errors as correctness. Measurements remain diagnostic candidates only; no "
                "Accuracy score or deduction is produced."
            ),
        }
    )


def _validate_bindings(
    pack: dict[str, Any],
    spec: dict[str, Any],
    timeline: dict[str, Any],
    profile: dict[str, Any],
    evidence: dict[str, Any],
) -> None:
    if profile["rule_pack_id"] != pack["rule_pack_id"] or spec["rule_pack_id"] != pack["rule_pack_id"]:
        raise ScoringContractError("engineering profile, PoomsaeSpec and RulePack must match")
    if profile["poomsae_id"] != spec["poomsae_id"]:
        raise ScoringContractError("engineering profile Poomsae id does not match")
    if timeline["coverage"]["recording_scope"] != profile["scope"]["recording_scope"]:
        raise ScoringContractError("engineering profile recording scope does not match timeline")
    if profile["scope"]["movement_ids"] != timeline["coverage"]["observed_movement_ids"]:
        raise ScoringContractError("engineering profile movements must exactly match observed timeline movements")
    if not np.isclose(
        profile["scope"]["reference_accuracy_score"],
        pack["scoring"]["accuracy"]["initial_score"],
        atol=1e-9,
    ):
        raise ScoringContractError("engineering reference score must match the active RulePack")
    if evidence.get("timeline", {}).get("timeline_id") != timeline["timeline_id"]:
        raise ScoringContractError("engineering evidence must reference the exact MovementTimeline")
    if evidence.get("scoring_status") != "not_scored_partial_recording":
        raise ScoringContractError("engineering trial requires partial, no-score evidence")


def _selector_matches(selector: dict[str, Any], movement: dict[str, Any]) -> bool:
    if selector["field"] == "all":
        return True
    if selector["field"] == "stance":
        return movement["stance"] in selector["values"]
    technique_ids = {technique["technique_id"] for technique in movement["techniques"]}
    return bool(technique_ids.intersection(selector["values"]))


def _evaluate_criterion(
    movement: dict[str, Any],
    segment: dict[str, Any],
    criterion: dict[str, Any],
    row: dict[str, Any] | None,
    gates: dict[str, Any],
) -> dict[str, Any]:
    base = {
        "event_id": f"ENG-{movement['movement_id']}-{criterion['criterion_id']}",
        "deduplication_key": f"{movement['movement_id']}:{criterion['family']}:{criterion['phase_id']}",
        "movement_id": movement["movement_id"],
        "phase_id": criterion["phase_id"],
        "frame_index": None if row is None else row.get("frame_index"),
        "criterion_id": criterion["criterion_id"],
        "family": criterion["family"],
        "metric_id": criterion["metric"],
        "acceptable_range": list(criterion["acceptable_range"]),
        "deduction_kind_candidate": criterion["deduction_kind"],
        "decision_status": "engineering_candidate_only",
        "rationale": criterion["rationale"],
    }
    quality_reasons = _quality_reasons(row, segment, gates)
    value = None if row is None else _metric_value(criterion["metric"], movement, criterion, row)
    if value is None:
        quality_reasons.append("metric_not_measurable")
    if quality_reasons:
        return {
            **base,
            "measurement_status": "not_measurable",
            "measured_value": value,
            "normalized_deviation": None,
            "evidence_status": None if row is None else row.get("evidence_status"),
            "evidence_confidence_proxy": 0.0,
            "quality_gate_reasons": sorted(set(quality_reasons)),
        }

    lower, upper = criterion["acceptable_range"]
    deviation = max(lower - value, 0.0, value - upper)
    width = max(upper - lower, 1e-9)
    normalized_deviation = deviation / width
    return {
        **base,
        "measurement_status": "pass" if deviation == 0.0 else "candidate",
        "measured_value": value,
        "normalized_deviation": normalized_deviation,
        "evidence_status": row["evidence_status"],
        "evidence_confidence_proxy": min(float(row["valid_body17_ratio"]), float(segment["confidence"])),
        "quality_gate_reasons": [],
    }


def _quality_reasons(
    row: dict[str, Any] | None,
    segment: dict[str, Any],
    gates: dict[str, Any],
) -> list[str]:
    if row is None:
        return ["phase_measurement_missing"]
    reasons: list[str] = []
    if row.get("evidence_status") != "observed":
        reasons.append("anchor_not_fully_observed")
    if _float(row.get("valid_body17_ratio")) < gates["min_body17_valid_ratio"]:
        reasons.append("body17_valid_ratio_below_gate")
    if _float(row.get("median_used_cameras")) < gates["min_used_cameras"]:
        reasons.append("camera_evidence_below_gate")
    reprojection = _optional_float(row.get("median_reprojection_error_px"))
    if reprojection is None or reprojection > gates["max_median_reprojection_error_px"]:
        reasons.append("reprojection_error_above_gate")
    if float(segment["confidence"]) < gates["min_label_confidence"]:
        reasons.append("timeline_label_confidence_below_gate")
    return reasons


def _metric_value(
    metric_id: str,
    movement: dict[str, Any],
    criterion: dict[str, Any],
    row: dict[str, Any],
) -> float | None:
    if metric_id in {"torso_lean_deg", "stance_span_ratio"}:
        return _optional_float(row.get(metric_id))
    if metric_id == "front_knee_deg":
        side = _stance_side(movement["stance"])
        return _optional_float(row.get(f"{side}_knee_deg"))
    if metric_id in {"executing_elbow_deg", "executing_wrist_height_torso_ratio"}:
        side = _technique_side(movement, criterion["selector"]["values"])
        suffix = "elbow_deg" if metric_id == "executing_elbow_deg" else "wrist_height_torso_ratio"
        return _optional_float(row.get(f"{side}_{suffix}"))
    raise ScoringContractError(f"unsupported engineering metric: {metric_id}")


def _stance_side(stance: str) -> str:
    if stance.startswith("left_"):
        return "left"
    if stance.startswith("right_"):
        return "right"
    raise ScoringContractError(f"stance side cannot be resolved: {stance}")


def _technique_side(movement: dict[str, Any], technique_ids: list[str]) -> str:
    for technique in movement["techniques"]:
        if technique["technique_id"] in technique_ids and technique["side"] in {"left", "right"}:
            return technique["side"]
    raise ScoringContractError(f"executing side cannot be resolved for {movement['movement_id']}")


def _trial_deduction(measurement: dict[str, Any], pack: dict[str, Any]) -> dict[str, Any]:
    rule = pack["scoring"]["accuracy"]["deductions"]["minor"]
    return {
        "event_id": measurement["event_id"],
        "deduplication_key": measurement["deduplication_key"],
        "movement_id": measurement["movement_id"],
        "phase_id": measurement["phase_id"],
        "family": measurement["family"],
        "criterion_id": measurement["criterion_id"],
        "metric_id": measurement["metric_id"],
        "measured_value": measurement["measured_value"],
        "acceptable_range": measurement["acceptable_range"],
        "deduction_kind": "minor",
        "deduction_points": rule["amount"],
        "wt_amount_rule_id": rule["rule_id"],
        "threshold_origin": "engineering_hypothesis_not_official_rule",
    }


def _float(value: Any) -> float:
    result = _optional_float(value)
    return float("-inf") if result is None else result


def _optional_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


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
