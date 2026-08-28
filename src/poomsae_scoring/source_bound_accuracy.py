from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
from typing import Any

import numpy as np

from src.poomsae_scoring.contracts import (
    ScoringContractError,
    load_yaml_mapping,
    validate_movement_timeline,
    validate_poomsae_spec,
)


DECISION_STATUSES = {
    "within_source_range",
    "boundary_uncertain",
    "confirmed_source_bound_minor",
    "not_measurable",
    "not_applicable",
}


def load_source_bound_accuracy_profile(path: str | Path) -> dict[str, Any]:
    return validate_source_bound_accuracy_profile(
        load_yaml_mapping(path, label="source-bound Accuracy profile")
    )


def validate_source_bound_accuracy_profile(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ScoringContractError("Source-bound Accuracy profile must be a mapping")
    data = deepcopy(payload)
    _exact_keys(
        data,
        {
            "schema_version",
            "profile_id",
            "version",
            "status",
            "poomsae_id",
            "decision_policy",
            "numeric_rules",
            "categorical_rules",
            "disclaimer",
        },
        "Source-bound Accuracy profile",
    )
    if data["schema_version"] != 1:
        raise ScoringContractError("Source-bound Accuracy schema_version must be 1")
    if data["status"] != "provisional_historical_geometry":
        raise ScoringContractError("Source-bound Accuracy profile status is unsupported")
    for key in ("profile_id", "version", "poomsae_id", "disclaimer"):
        _nonempty(data[key], key)
    policy = data["decision_policy"]
    _exact_keys(
        policy,
        {
            "minor_points",
            "major_points",
            "restart_points",
            "minimum_observation_confidence",
            "boundary_policy",
            "numeric_geometry_can_create_major",
            "partial_timeline_score_enabled",
        },
        "decision_policy",
    )
    for key in ("minor_points", "major_points", "restart_points"):
        policy[key] = _positive(policy[key], f"decision_policy.{key}")
    policy["minimum_observation_confidence"] = _probability(
        policy["minimum_observation_confidence"], "minimum_observation_confidence"
    )
    if policy["boundary_policy"] != "require_95pct_interval_wholly_outside":
        raise ScoringContractError("source-bound decisions require the conservative 95% interval policy")
    if policy["numeric_geometry_can_create_major"] is not False:
        raise ScoringContractError("numeric geometry must never create a major deduction")
    if policy["partial_timeline_score_enabled"] is not False:
        raise ScoringContractError("partial timelines must not produce an Accuracy score")
    if not isinstance(data["numeric_rules"], list) or not data["numeric_rules"]:
        raise ScoringContractError("numeric_rules must be a non-empty list")
    numeric_ids: set[str] = set()
    for rule in data["numeric_rules"]:
        _validate_numeric_rule(rule, numeric_ids)
    if not isinstance(data["categorical_rules"], list) or not data["categorical_rules"]:
        raise ScoringContractError("categorical_rules must be a non-empty list")
    categorical_ids: set[str] = set()
    for rule in data["categorical_rules"]:
        _validate_categorical_rule(rule, categorical_ids)
    return data


def build_source_bound_accuracy_decisions(
    wholebody_diagnostics: dict[str, Any],
    poomsae_spec: dict[str, Any],
    movement_timeline: dict[str, Any],
    profile: dict[str, Any],
    observations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    spec = validate_poomsae_spec(poomsae_spec)
    timeline = validate_movement_timeline(movement_timeline, spec)
    rules = validate_source_bound_accuracy_profile(profile)
    if rules["poomsae_id"] != spec["poomsae_id"]:
        raise ScoringContractError("Source-bound profile Poomsae id does not match")
    if wholebody_diagnostics.get("status") != "wholebody_diagnostics_only":
        raise ScoringContractError("source-bound decisions require a WholeBody diagnostics report")
    _validate_diagnostic_binding(wholebody_diagnostics, spec, timeline)
    movement_specs = {item["movement_id"]: item for item in spec["movements"]}
    metric_movements = wholebody_diagnostics.get("movements")
    if not isinstance(metric_movements, list):
        raise ScoringContractError("WholeBody diagnostics movements must be a list")

    decisions: list[dict[str, Any]] = []
    for movement_report in metric_movements:
        movement_id = movement_report.get("movement_id")
        movement = movement_specs.get(movement_id)
        if movement is None:
            raise ScoringContractError(f"unknown diagnostics movement: {movement_id}")
        metrics = {item.get("metric_id"): item for item in movement_report.get("metrics", [])}
        for rule in rules["numeric_rules"]:
            if not _numeric_rule_applies(rule, movement):
                continue
            metric = metrics.get(rule["metric_id"])
            decisions.append(_numeric_decision(rule, movement, metric))

    applied_numeric = _deduplicate_confirmed_numeric(decisions, rules["decision_policy"]["minor_points"])
    categorical = _categorical_decisions(
        observations or [],
        rules,
        movement_specs,
        timeline,
    )
    applied_categorical = [item for item in categorical if item["application_status"] == "applied"]
    provisional_total = float(sum(item["deduction_points"] for item in applied_numeric + applied_categorical))
    complete_score_eligible = (
        spec["status"] == "active"
        and timeline["status"] == "complete"
        and timeline["coverage"]["recording_scope"] == "complete_performance"
    )
    report = _json_safe(
        {
            "schema_version": 1,
            "status": "source_bound_accuracy_decisions",
            "result_kind": "provisional_observed_scope_deduction_analysis",
            "scoring_status": (
                "eligible_for_separate_full_accuracy_evaluation"
                if complete_score_eligible
                else "observed_scope_only_no_accuracy_score"
            ),
            "accuracy_evaluation_status": (
                "eligible_not_evaluated"
                if complete_score_eligible
                else "not_eligible_incomplete_evidence"
            ),
            "accuracy_score_unavailable_reason": (
                "separate_full_accuracy_evaluation_not_run"
                if complete_score_eligible
                else "incomplete_or_inactive_scoring_evidence"
            ),
            "accuracy_score": None,
            "official_score_status": "not_available",
            "official_score": None,
            "provisional_deduction_status": "observed_scope_only_not_official",
            "profile": {"profile_id": rules["profile_id"], "version": rules["version"]},
            "poomsae": {"poomsae_id": spec["poomsae_id"], "version": spec["version"]},
            "timeline_id": timeline["timeline_id"],
            "timeline_scope": timeline["coverage"]["recording_scope"],
            "observed_scope_provisional_deduction_total": provisional_total,
            "numeric_decisions": decisions,
            "categorical_decisions": categorical,
            "applied_observed_scope_deductions": applied_numeric + applied_categorical,
            "summary": {
                "numeric_decision_count": len(decisions),
                "confirmed_numeric_minor_count": len(applied_numeric),
                "boundary_uncertain_count": sum(item["decision_status"] == "boundary_uncertain" for item in decisions),
                "not_measurable_count": sum(item["decision_status"] == "not_measurable" for item in decisions),
                "applied_categorical_count": len(applied_categorical),
            },
            "safety_contract": {
                "numeric_geometry_can_create_major": False,
                "boundary_overlap_creates_deduction": False,
                "partial_timeline_creates_accuracy_score": False,
                "historical_geometry_is_current_wt_attachment": False,
            },
            "interpretation": rules["disclaimer"],
        }
    )
    return validate_source_bound_accuracy_result(report)


def validate_source_bound_accuracy_result(payload: dict[str, Any]) -> dict[str, Any]:
    data = dict(payload)
    if data.get("schema_version") != 1 or data.get("status") != "source_bound_accuracy_decisions":
        raise ScoringContractError("Source-bound result contract/version is invalid")
    if data.get("result_kind") != "provisional_observed_scope_deduction_analysis":
        raise ScoringContractError("Source-bound result_kind must remain explicitly provisional")
    evaluation_status = data.get("accuracy_evaluation_status")
    if evaluation_status not in {"eligible_not_evaluated", "not_eligible_incomplete_evidence"}:
        raise ScoringContractError("Source-bound accuracy_evaluation_status is invalid")
    if data.get("accuracy_score") is not None:
        raise ScoringContractError("This source-bound decision contract cannot emit an Accuracy score")
    if data.get("official_score_status") != "not_available" or data.get("official_score") is not None:
        raise ScoringContractError("An official score cannot appear in a source-bound decision package")
    if data.get("provisional_deduction_status") != "observed_scope_only_not_official":
        raise ScoringContractError("Observed deductions must remain explicitly provisional")
    provisional = data.get("observed_scope_provisional_deduction_total")
    if isinstance(provisional, bool) or not isinstance(provisional, (int, float)) or not math.isfinite(provisional):
        raise ScoringContractError("Observed-scope provisional deduction total must be finite")
    return data


def derive_categorical_observations(
    poomsae_spec: dict[str, Any],
    movement_timeline: dict[str, Any],
    *,
    pause_threshold_sec: float = 3.0,
    minimum_confidence: float = 0.80,
) -> list[dict[str, Any]]:
    """Emit automatic categorical observations that timeline data alone can prove.

    Only supports ``pause_at_least_3_sec`` for now: WT Article 16-1.2.3 defines a
    3-second pause during movements as a major deduction, and gap length between
    two labelled segments is a deterministic timeline measurement. The function
    reports each bounded inter-segment gap that meets or exceeds the WT minimum
    as an observation attached to the movement that immediately precedes the gap
    (which is the movement WT considers 'held' for that long).

    Wrong-action / wrong-stance detection is intentionally NOT auto-derived
    here: the timeline validator already rejects timelines whose segment order
    does not match the PoomsaeSpec, and segments do not yet carry an observed
    stance field, so any 'sequence violation' inferred from timeline alone
    would be tautologically empty. Those categorical rules still require a
    human observation payload with ``manual_video_review``.
    """
    spec = validate_poomsae_spec(poomsae_spec)
    timeline = validate_movement_timeline(movement_timeline, spec)
    if pause_threshold_sec < 3.0:
        raise ScoringContractError("pause_threshold_sec cannot be below the WT 3-second rule")
    if not 0.0 <= minimum_confidence <= 1.0:
        raise ScoringContractError("minimum_confidence must be between 0 and 1")
    fps = float(timeline["fps"])
    if fps <= 0:
        raise ScoringContractError("timeline fps must be positive")
    threshold_frames = int(np.ceil(pause_threshold_sec * fps))
    segments = timeline["segments"]
    observations: list[dict[str, Any]] = []
    for index, segment in enumerate(segments[:-1]):
        next_segment = segments[index + 1]
        gap_start = int(segment["end_frame"]) + 1
        gap_end_exclusive = int(next_segment["start_frame"])
        gap_end = gap_end_exclusive - 1
        gap_length = gap_end_exclusive - gap_start
        if gap_length < threshold_frames:
            continue
        # Pin the observation inside the preceding segment so the existing
        # boundary check (start_frame within movement interval) accepts it, but
        # keep the true duration measurement for the pause guard.
        anchor_frame = int(segment["end_frame"])
        duration_sec = float(gap_length) / fps
        confidence = min(float(segment["confidence"]), float(next_segment["confidence"]))
        independently_reviewed = confidence >= minimum_confidence and timeline["label_source"] in {
            "manual",
            "manual_reviewed_automatic",
        } and all(
            item["label_status"] == "confirmed" for item in (segment, next_segment)
        )
        observations.append(
            {
                "observation_id": f"AUTO-PAUSE-{segment['movement_id']}-{gap_start:06d}",
                "event_kind": "pause_at_least_3_sec",
                "movement_id": segment["movement_id"],
                "start_frame": anchor_frame,
                "end_frame": anchor_frame,
                "evidence_status": "observed" if independently_reviewed else "inferred",
                "confidence": confidence,
                "description": (
                    f"Automatic gap of {duration_sec:.2f}s after {segment['movement_id']} "
                    f"(frames {gap_start}-{gap_end}); WT 2024 Article 16-1.2.3 pauses"
                    " of at least 3 seconds count as a major deduction."
                ),
                "measurement": {"duration_sec": duration_sec},
                "confirmation_method": "duration_measurement",
            }
        )
    return observations


def _validate_diagnostic_binding(
    diagnostics: dict[str, Any],
    spec: dict[str, Any],
    timeline: dict[str, Any],
) -> None:
    expected_poomsae = {"poomsae_id": spec["poomsae_id"], "version": spec["version"]}
    if diagnostics.get("poomsae") != expected_poomsae:
        raise ScoringContractError("WholeBody diagnostics Poomsae binding does not match")
    if diagnostics.get("movement_timeline_id") != timeline["timeline_id"]:
        raise ScoringContractError("WholeBody diagnostics timeline binding does not match")


def _numeric_decision(rule: dict[str, Any], movement: dict[str, Any], metric: dict[str, Any] | None) -> dict[str, Any]:
    base = {
        "movement_id": movement["movement_id"],
        "rule_id": rule["rule_id"],
        "error_unit": rule["error_unit"],
        "metric_id": rule["metric_id"],
        "deduction_kind": "minor",
        "deduction_points": None,
        "application_status": "not_applied",
        "source": deepcopy(rule["source"]),
        "description": rule["description"],
        "sample_count": None if metric is None else metric.get("sample_count"),
        "measurement_evidence": (None if metric is None else deepcopy(metric.get("measurement_evidence"))),
    }
    if metric is None or metric.get("value") is None:
        return {
            **base,
            "decision_status": "not_measurable",
            "measured_value": None,
            "uncertainty_95": None,
            "effective_uncertainty_95": None,
            "measurement_interval_95": None,
            "rule_operator": rule["operator"],
            "rule_limits": rule["limits"],
            "boundary_guard": rule["boundary_guard"],
            "reason": "required_metric_not_measurable",
        }
    value = float(metric["value"])
    measured_uncertainty = metric.get("uncertainty_95")
    measured_uncertainty = 0.0 if measured_uncertainty is None else float(measured_uncertainty)
    uncertainty = max(measured_uncertainty, float(rule["uncertainty_floor"]))
    interval = [value - uncertainty, value + uncertainty]
    status, reason = _interval_decision(rule["operator"], rule["limits"], interval, rule["boundary_guard"])
    return {
        **base,
        "decision_status": status,
        "measured_value": value,
        "uncertainty_95": measured_uncertainty,
        "effective_uncertainty_95": uncertainty,
        "measurement_interval_95": interval,
        "rule_operator": rule["operator"],
        "rule_limits": rule["limits"],
        "boundary_guard": rule["boundary_guard"],
        "reason": reason,
    }


def _interval_decision(
    operator: str,
    limits: list[float],
    interval: list[float],
    boundary_guard: float,
) -> tuple[str, str]:
    low, high = interval
    if operator == "max":
        maximum = limits[0]
        if low > maximum + boundary_guard:
            return "confirmed_source_bound_minor", "entire_95pct_interval_above_guarded_source_maximum"
        if high <= maximum - boundary_guard:
            return "within_source_range", "entire_95pct_interval_within_guarded_source_maximum"
        return "boundary_uncertain", "95pct_interval_overlaps_guarded_source_boundary"
    lower, upper = limits
    if high < lower - boundary_guard or low > upper + boundary_guard:
        return "confirmed_source_bound_minor", "entire_95pct_interval_outside_guarded_source_range"
    if low >= lower + boundary_guard and high <= upper - boundary_guard:
        return "within_source_range", "entire_95pct_interval_within_guarded_source_range"
    return "boundary_uncertain", "95pct_interval_overlaps_guarded_source_boundary"


def _deduplicate_confirmed_numeric(decisions: list[dict[str, Any]], minor_points: float) -> list[dict[str, Any]]:
    applied: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in decisions:
        if item["decision_status"] != "confirmed_source_bound_minor":
            continue
        key = (item["movement_id"], item["error_unit"])
        if key in seen:
            item["application_status"] = "not_applied_duplicate_error_unit"
            item["reason"] = "correlated_metric_deduplicated"
            continue
        seen.add(key)
        item["application_status"] = "applied"
        item["deduction_points"] = minor_points
        applied.append(item)
    return applied


def _categorical_decisions(
    observations: list[dict[str, Any]],
    profile: dict[str, Any],
    movements: dict[str, dict[str, Any]],
    timeline: dict[str, Any],
) -> list[dict[str, Any]]:
    if not isinstance(observations, list):
        raise ScoringContractError("categorical observations must be a list")
    rules = {item["event_kind"]: item for item in profile["categorical_rules"]}
    intervals = {item["movement_id"]: item for item in timeline["segments"]}
    seen_ids: set[str] = set()
    seen_units: set[tuple[str | None, str]] = set()
    results = []
    for raw in observations:
        observation = _validate_observation(raw, rules)
        if observation["observation_id"] in seen_ids:
            raise ScoringContractError(f"duplicate observation_id: {observation['observation_id']}")
        seen_ids.add(observation["observation_id"])
        rule = rules[observation["event_kind"]]
        reason = _categorical_rejection_reason(
            observation,
            rule,
            movements,
            intervals,
            timeline["frame_count"],
            profile["decision_policy"]["minimum_observation_confidence"],
        )
        unit = (observation["movement_id"], rule["error_unit"])
        if reason is None and unit in seen_units:
            reason = "duplicate_error_unit"
        if reason is None:
            seen_units.add(unit)
        points_key = "restart_points" if rule["deduction_kind"] == "restart" else "major_points"
        results.append(
            {
                **observation,
                "rule_id": rule["rule_id"],
                "error_unit": rule["error_unit"],
                "deduction_kind": rule["deduction_kind"],
                "deduction_points": (profile["decision_policy"][points_key] if reason is None else None),
                "application_status": "applied" if reason is None else "not_applied",
                "reason": reason,
                "source_ref": rule["source_ref"],
            }
        )
    return results


def _categorical_rejection_reason(
    observation: dict[str, Any],
    rule: dict[str, Any],
    movements: dict[str, dict[str, Any]],
    intervals: dict[str, dict[str, Any]],
    frame_count: int,
    minimum_confidence: float,
) -> str | None:
    if observation["evidence_status"] != "observed":
        return "not_directly_observed"
    if observation["confidence"] < minimum_confidence:
        return "confidence_below_threshold"
    if observation["end_frame"] >= frame_count:
        return "outside_timeline"
    if rule["deduction_kind"] == "restart":
        return None if observation["movement_id"] is None else "restart_must_be_performance_scoped"
    movement_id = observation["movement_id"]
    if movement_id not in movements or movement_id not in intervals:
        return "unknown_or_unobserved_movement"
    interval = intervals[movement_id]
    if observation["start_frame"] < interval["start_frame"] or observation["end_frame"] > interval["end_frame"]:
        return "outside_movement_interval"
    if observation["event_kind"] == "pause_at_least_3_sec":
        measurement = observation["measurement"]
        if measurement is None or measurement.get("duration_sec", 0.0) < 3.0:
            return "pause_duration_below_3_sec"
    return None


def _validate_observation(observation: dict[str, Any], rules: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not isinstance(observation, dict):
        raise ScoringContractError("each categorical observation must be a mapping")
    required = {
        "observation_id",
        "event_kind",
        "movement_id",
        "start_frame",
        "end_frame",
        "evidence_status",
        "confidence",
        "description",
        "measurement",
        "confirmation_method",
    }
    _exact_keys(observation, required, "categorical observation")
    result = deepcopy(observation)
    for key in ("observation_id", "event_kind", "description"):
        _nonempty(result[key], f"observation.{key}")
    if result["event_kind"] not in rules:
        raise ScoringContractError(f"unsupported categorical event_kind: {result['event_kind']}")
    if result["confirmation_method"] not in rules[result["event_kind"]]["confirmation_methods"]:
        raise ScoringContractError("confirmation_method is not authorized for categorical rule")
    if result["movement_id"] is not None:
        _nonempty(result["movement_id"], "observation.movement_id")
    for key in ("start_frame", "end_frame"):
        if not isinstance(result[key], int) or isinstance(result[key], bool) or result[key] < 0:
            raise ScoringContractError(f"observation.{key} must be a non-negative integer")
    if result["start_frame"] > result["end_frame"]:
        raise ScoringContractError("observation start_frame cannot exceed end_frame")
    if result["evidence_status"] not in {"observed", "inferred", "not_measurable"}:
        raise ScoringContractError("unsupported categorical evidence_status")
    result["confidence"] = _probability(result["confidence"], "observation.confidence")
    if result["event_kind"] == "pause_at_least_3_sec":
        if not isinstance(result["measurement"], dict) or set(result["measurement"]) != {"duration_sec"}:
            raise ScoringContractError("pause observation requires only measurement.duration_sec")
        result["measurement"]["duration_sec"] = _nonnegative(
            result["measurement"]["duration_sec"], "measurement.duration_sec"
        )
    elif result["measurement"] is not None:
        raise ScoringContractError("non-pause categorical observation measurement must be null")
    return result


def _validate_numeric_rule(rule: dict[str, Any], seen: set[str]) -> None:
    _exact_keys(
        rule,
        {
            "rule_id",
            "error_unit",
            "stances",
            "technique_ids",
            "metric_id",
            "operator",
            "limits",
            "uncertainty_floor",
            "boundary_guard",
            "unit",
            "deduction_kind",
            "source",
            "description",
        },
        "numeric rule",
    )
    _unique_id(rule["rule_id"], seen, "numeric rule_id")
    for key in ("error_unit", "metric_id", "unit", "description"):
        _nonempty(rule[key], f"numeric_rule.{key}")
    for key in ("stances", "technique_ids"):
        if not isinstance(rule[key], list) or not all(isinstance(item, str) and item for item in rule[key]):
            raise ScoringContractError(f"numeric_rule.{key} must be a string list")
    if not rule["stances"] and not rule["technique_ids"]:
        raise ScoringContractError("numeric rule must constrain a stance or technique")
    if rule["operator"] not in {"max", "range"}:
        raise ScoringContractError("numeric rule operator must be max or range")
    expected_limits = 1 if rule["operator"] == "max" else 2
    if not isinstance(rule["limits"], list) or len(rule["limits"]) != expected_limits:
        raise ScoringContractError("numeric rule limits do not match operator")
    rule["limits"] = [_finite(item, "numeric_rule.limits") for item in rule["limits"]]
    if rule["operator"] == "range" and rule["limits"][0] >= rule["limits"][1]:
        raise ScoringContractError("numeric range lower limit must be below upper limit")
    rule["uncertainty_floor"] = _nonnegative(rule["uncertainty_floor"], "numeric_rule.uncertainty_floor")
    rule["boundary_guard"] = _nonnegative(rule["boundary_guard"], "numeric_rule.boundary_guard")
    if rule["deduction_kind"] != "minor":
        raise ScoringContractError("numeric source rules may create minor deductions only")
    _validate_rule_source(rule["source"])


def _validate_rule_source(source: dict[str, Any]) -> None:
    _exact_keys(
        source,
        {"source_id", "content_sha256", "pages", "authority_status"},
        "numeric rule source",
    )
    for key in ("source_id", "content_sha256", "authority_status"):
        _nonempty(source[key], f"source.{key}")
    if source["authority_status"] != "historical_official_not_current_attachment":
        raise ScoringContractError("numeric geometry must be labelled as historical, not current")
    if not isinstance(source["pages"], list) or not source["pages"]:
        raise ScoringContractError("source.pages must be a non-empty list")


def _validate_categorical_rule(rule: dict[str, Any], seen: set[str]) -> None:
    _exact_keys(
        rule,
        {
            "rule_id",
            "event_kind",
            "error_unit",
            "deduction_kind",
            "source_ref",
            "confirmation_methods",
        },
        "categorical rule",
    )
    _unique_id(rule["rule_id"], seen, "categorical rule_id")
    for key in ("event_kind", "error_unit", "source_ref"):
        _nonempty(rule[key], f"categorical_rule.{key}")
    if rule["deduction_kind"] not in {"major", "restart"}:
        raise ScoringContractError("categorical deduction_kind must be major or restart")
    if (
        not isinstance(rule["confirmation_methods"], list)
        or not rule["confirmation_methods"]
        or not all(isinstance(item, str) and item for item in rule["confirmation_methods"])
    ):
        raise ScoringContractError("categorical confirmation_methods must be a string list")


def _numeric_rule_applies(rule: dict[str, Any], movement: dict[str, Any]) -> bool:
    stance_match = not rule["stances"] or movement["stance"] in rule["stances"]
    techniques = {item["technique_id"] for item in movement["techniques"]}
    technique_match = not rule["technique_ids"] or bool(techniques & set(rule["technique_ids"]))
    return stance_match and technique_match


def _exact_keys(value: Any, expected: set[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        actual = set(value) if isinstance(value, dict) else set()
        raise ScoringContractError(
            f"{label} keys are invalid; missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
        )


def _unique_id(value: Any, seen: set[str], label: str) -> None:
    identifier = _nonempty(value, label)
    if identifier in seen:
        raise ScoringContractError(f"duplicate {label}: {identifier}")
    seen.add(identifier)


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ScoringContractError(f"{label} must be a non-empty string")
    return value


def _finite(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ScoringContractError(f"{label} must be numeric") from exc
    if not np.isfinite(result):
        raise ScoringContractError(f"{label} must be finite")
    return result


def _positive(value: Any, label: str) -> float:
    result = _finite(value, label)
    if result <= 0:
        raise ScoringContractError(f"{label} must be positive")
    return result


def _nonnegative(value: Any, label: str) -> float:
    result = _finite(value, label)
    if result < 0:
        raise ScoringContractError(f"{label} must be non-negative")
    return result


def _probability(value: Any, label: str) -> float:
    result = _finite(value, label)
    if not 0.0 <= result <= 1.0:
        raise ScoringContractError(f"{label} must be between 0 and 1")
    return result


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
