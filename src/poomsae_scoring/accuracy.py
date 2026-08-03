from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from src.poomsae_scoring.contracts import (
    ScoringContractError,
    validate_movement_timeline,
    validate_poomsae_spec,
    validate_rule_pack,
)


def evaluate_accuracy(
    rule_pack: dict[str, Any],
    poomsae_spec: dict[str, Any],
    movement_timeline: dict[str, Any],
    events: list[dict[str, Any]],
    *,
    minimum_confidence: float = 0.80,
) -> dict[str, Any]:
    """Apply source-backed Accuracy deductions to confirmed, observable events."""
    pack = validate_rule_pack(rule_pack)
    spec = validate_poomsae_spec(poomsae_spec)
    timeline = validate_movement_timeline(movement_timeline, spec, require_complete=True)
    if pack["status"] != "active":
        raise ScoringContractError("Accuracy scoring requires an active RulePack")
    if spec["rule_pack_id"] != pack["rule_pack_id"]:
        raise ScoringContractError("PoomsaeSpec rule_pack_id does not match the RulePack")
    minimum_confidence = _probability(minimum_confidence, "minimum_confidence")
    if not isinstance(events, list):
        raise ScoringContractError("Accuracy events must be a list")

    deduction_rules = pack["scoring"]["accuracy"]["deductions"]
    intervals = {segment["movement_id"]: segment for segment in timeline["segments"]}
    movement_phases = {movement["movement_id"]: set(movement["phases"]) for movement in spec["movements"]}
    movement_criteria = {
        movement["movement_id"]: set(movement["measurable_criteria"])
        for movement in spec["movements"]
    }
    applied: list[dict[str, Any]] = []
    not_applied: list[dict[str, Any]] = []
    seen_event_ids: set[str] = set()
    seen_deduplication_keys: set[str] = set()

    for raw_event in events:
        event = _validate_event(raw_event, deduction_rules)
        if event["event_id"] in seen_event_ids:
            raise ScoringContractError(f"duplicate Accuracy event_id: {event['event_id']}")
        seen_event_ids.add(event["event_id"])

        reason = _non_application_reason(
            event,
            intervals,
            movement_phases,
            movement_criteria,
            frame_count=timeline["frame_count"],
            minimum_confidence=minimum_confidence,
            seen_deduplication_keys=seen_deduplication_keys,
        )
        rule = deduction_rules[event["deduction_kind"]]
        result = {
            **event,
            "rule_id": rule["rule_id"],
            "deduction_points": rule["amount"],
            "rule_definition": rule["definition"],
            "rule_source_refs": list(rule["source_refs"]),
        }
        if reason is not None:
            not_applied.append({**result, "application_status": "not_applied", "reason": reason})
            continue
        seen_deduplication_keys.add(event["deduplication_key"])
        applied.append({**result, "application_status": "applied", "reason": None})

    initial_score = pack["scoring"]["accuracy"]["initial_score"]
    total_deduction = float(sum(item["deduction_points"] for item in applied))
    final_score = float(max(0.0, initial_score - total_deduction))
    return {
        "schema_version": 1,
        "score_name": "TK3D rule-based provisional Accuracy",
        "status": "rule_based_provisional",
        "judge_calibrated": False,
        "official_scoring_ready": False,
        "rule_pack": {"rule_pack_id": pack["rule_pack_id"], "version": pack["version"]},
        "poomsae_spec": {"poomsae_id": spec["poomsae_id"], "version": spec["version"]},
        "movement_timeline_id": timeline["timeline_id"],
        "initial_accuracy_score": initial_score,
        "total_deduction": total_deduction,
        "accuracy_score": final_score,
        "minimum_confidence": minimum_confidence,
        "applied_deductions": applied,
        "not_applied_events": not_applied,
        "summary": {
            "event_count": len(events),
            "applied_count": len(applied),
            "not_applied_count": len(not_applied),
            "movement_count": len(timeline["segments"]),
        },
        "interpretation": (
            "This is a source-backed provisional Accuracy score. It is not judge-calibrated or an official result."
        ),
    }


def _validate_event(event: dict[str, Any], deduction_rules: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(event, dict):
        raise ScoringContractError("each Accuracy event must be a mapping")
    required = {
        "event_id",
        "deduplication_key",
        "deduction_kind",
        "movement_id",
        "phase_id",
        "start_frame",
        "end_frame",
        "evidence_status",
        "decision_status",
        "confidence",
        "metric_id",
        "criterion_id",
        "description",
        "rule_confirmation",
    }
    if set(event) != required:
        raise ScoringContractError(
            "Accuracy event keys are invalid; "
            f"missing={sorted(required - set(event))}, unexpected={sorted(set(event) - required)}"
        )
    normalized = deepcopy(event)
    for key in (
        "event_id",
        "deduplication_key",
        "deduction_kind",
        "metric_id",
        "criterion_id",
        "description",
    ):
        if not isinstance(normalized[key], str) or not normalized[key].strip():
            raise ScoringContractError(f"Accuracy event {key} must be a non-empty string")
    if normalized["deduction_kind"] not in deduction_rules:
        raise ScoringContractError(f"unsupported deduction_kind: {normalized['deduction_kind']}")
    if normalized["movement_id"] is not None and (
        not isinstance(normalized["movement_id"], str) or not normalized["movement_id"].strip()
    ):
        raise ScoringContractError("Accuracy event movement_id must be null or a non-empty string")
    if normalized["phase_id"] is not None and (
        not isinstance(normalized["phase_id"], str) or not normalized["phase_id"].strip()
    ):
        raise ScoringContractError("Accuracy event phase_id must be null or a non-empty string")
    for name in ("start_frame", "end_frame"):
        if not isinstance(normalized[name], int) or isinstance(normalized[name], bool) or normalized[name] < 0:
            raise ScoringContractError(f"Accuracy event {name} must be a non-negative integer")
    if normalized["start_frame"] > normalized["end_frame"]:
        raise ScoringContractError("Accuracy event start_frame cannot be after end_frame")
    if normalized["evidence_status"] not in {"observed", "partially_observed", "inferred", "not_measurable"}:
        raise ScoringContractError("Accuracy event evidence_status is unsupported")
    if normalized["decision_status"] not in {"confirmed_by_rule", "candidate", "rejected"}:
        raise ScoringContractError("Accuracy event decision_status is unsupported")
    normalized["confidence"] = _probability(normalized["confidence"], "Accuracy event confidence")
    confirmation = normalized["rule_confirmation"]
    if not isinstance(confirmation, dict) or set(confirmation) != {
        "rule_id",
        "source_ref",
        "confirmation_method",
        "review_record_id",
    }:
        raise ScoringContractError("Accuracy event rule_confirmation keys are invalid")
    for key in ("rule_id", "source_ref", "review_record_id"):
        if not isinstance(confirmation[key], str) or not confirmation[key].strip():
            raise ScoringContractError(f"Accuracy event rule_confirmation.{key} must be a non-empty string")
    if confirmation["confirmation_method"] not in {
        "manual_rule_review",
        "deterministic_sequence_violation",
        "observed_restart",
    }:
        raise ScoringContractError("Accuracy event confirmation_method is unsupported")
    rule = deduction_rules[normalized["deduction_kind"]]
    if confirmation["rule_id"] != rule["rule_id"]:
        raise ScoringContractError("Accuracy event confirmation rule_id does not match deduction_kind")
    if confirmation["source_ref"] not in rule["source_refs"]:
        raise ScoringContractError("Accuracy event confirmation source_ref is not authorized by the RulePack")
    if normalized["deduction_kind"] == "restart" and confirmation["confirmation_method"] != "observed_restart":
        raise ScoringContractError("restart events require observed_restart confirmation")
    if normalized["deduction_kind"] != "restart" and confirmation["confirmation_method"] == "observed_restart":
        raise ScoringContractError("observed_restart confirmation is only valid for restart events")
    return normalized


def _non_application_reason(
    event: dict[str, Any],
    intervals: dict[str, dict[str, Any]],
    movement_phases: dict[str, set[str]],
    movement_criteria: dict[str, set[str]],
    *,
    frame_count: int,
    minimum_confidence: float,
    seen_deduplication_keys: set[str],
) -> str | None:
    if event["decision_status"] != "confirmed_by_rule":
        return "decision_not_confirmed"
    if event["evidence_status"] != "observed":
        return "insufficient_independent_evidence"
    if event["confidence"] < minimum_confidence:
        return "confidence_below_threshold"
    if event["deduplication_key"] in seen_deduplication_keys:
        return "duplicate_error_event"
    if event["end_frame"] >= frame_count:
        return "event_outside_timeline"
    if event["deduction_kind"] == "restart":
        if event["movement_id"] is not None or event["phase_id"] is not None:
            return "restart_must_be_performance_scoped"
        return None
    movement_id = event["movement_id"]
    if movement_id not in intervals:
        return "unknown_movement"
    if event["phase_id"] is None or event["phase_id"] not in movement_phases[movement_id]:
        return "unknown_movement_phase"
    if event["criterion_id"] not in movement_criteria[movement_id]:
        return "metric_not_authorized_for_movement"
    interval = intervals[movement_id]
    if event["start_frame"] < interval["start_frame"] or event["end_frame"] > interval["end_frame"]:
        return "event_outside_movement_interval"
    return None


def _probability(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ScoringContractError(f"{label} must be numeric") from exc
    if not np.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ScoringContractError(f"{label} must be finite and between 0 and 1")
    return result
