"""Movement-level technical conformance diagnostics for labelled Poomsae scope.

This layer fuses existing WholeBody measurements with categorical technique and
stance screening.  It deliberately does not create a score or deduction.  The
result says only what the available, provenance-bound measurements support for
each labelled movement.
"""

from __future__ import annotations

from copy import deepcopy
from math import isfinite
from typing import Any

from src.poomsae_scoring.contracts import (
    ScoringContractError,
    validate_movement_timeline,
    validate_poomsae_spec,
)


TECHNICAL_CONFORMANCE_STATUS = "technical_conformance_diagnostic_only"


def build_technical_conformance(
    wholebody_diagnostics: dict[str, Any],
    categorical_diagnostics: dict[str, Any],
    poomsae_spec: dict[str, Any],
    movement_timeline: dict[str, Any],
) -> dict[str, Any]:
    """Fuse movement identity, geometry and control evidence without scoring."""
    spec = validate_poomsae_spec(poomsae_spec)
    timeline = validate_movement_timeline(movement_timeline, spec)
    _validate_bindings(wholebody_diagnostics, categorical_diagnostics, spec, timeline)

    movement_specs = {item["movement_id"]: item for item in spec["movements"]}
    wholebody_by_id = _unique_by_movement(
        wholebody_diagnostics.get("movements"), "WholeBody diagnostics"
    )
    checks_by_id: dict[str, list[dict[str, Any]]] = {}
    checks = categorical_diagnostics.get("checks")
    if not isinstance(checks, list):
        raise ScoringContractError("categorical diagnostics checks must be a list")
    for check in checks:
        movement_id = check.get("movement_id")
        if movement_id not in movement_specs:
            raise ScoringContractError(
                f"technical conformance references unknown categorical movement: {movement_id}"
            )
        checks_by_id.setdefault(movement_id, []).append(check)

    reports: list[dict[str, Any]] = []
    for segment in timeline["segments"]:
        movement_id = segment["movement_id"]
        source_report = wholebody_by_id.get(movement_id)
        if source_report is None:
            raise ScoringContractError(
                f"technical conformance is missing WholeBody movement: {movement_id}"
            )
        reports.append(
            _movement_report(
                movement_specs[movement_id],
                segment,
                source_report,
                checks_by_id.get(movement_id, []),
                timeline["label_source"],
            )
        )

    status_ids = (
        "mismatch_candidate",
        "review_candidate",
        "ambiguous",
        "consistent_within_measured_scope",
        "not_measurable",
    )
    return {
        "schema_version": 1,
        "status": TECHNICAL_CONFORMANCE_STATUS,
        "scoring_status": "diagnostic_only_no_score_or_deduction",
        "poomsae": {"poomsae_id": spec["poomsae_id"], "version": spec["version"]},
        "movement_timeline_id": timeline["timeline_id"],
        "recording_scope": timeline["coverage"]["recording_scope"],
        "summary": {
            "movement_count": len(reports),
            **{
                f"{status}_count": sum(item["conformance_status"] == status for item in reports)
                for status in status_ids
            },
            "review_required_count": sum(item["review_required"] for item in reports),
            "expected_criterion_count": sum(
                item["criterion_coverage"]["expected_count"] for item in reports
            ),
            "measurable_criterion_count": sum(
                item["criterion_coverage"]["measurable_count"] for item in reports
            ),
            "threshold_evaluable_criterion_count": sum(
                item["criterion_coverage"]["threshold_evaluable_count"] for item in reports
            ),
        },
        "movements": reports,
        "safety_contract": {
            "official_conformance_claim_allowed": False,
            "score_claim_allowed": False,
            "automatic_deduction_allowed": False,
            "consistent_status_is_limited_to_measured_scope": True,
            "inferred_identity_mismatch_requires_confirmation": True,
        },
        "interpretation": (
            "Hareket sonuçları yalnız etiketli kayıt kapsamındaki mevcut 3B ölçümlerin "
            "birleşik teknik taramasıdır. 'Ölçülen kapsam içinde uyumlu' ifadesi eksik veya "
            "tanısal-only ölçütlerin doğru olduğunu kanıtlamaz; sonuçlar puan ya da otomatik "
            "kesinti oluşturmaz."
        ),
    }


def _movement_report(
    movement: dict[str, Any],
    segment: dict[str, Any],
    wholebody: dict[str, Any],
    categorical_checks: list[dict[str, Any]],
    timeline_label_source: str,
) -> dict[str, Any]:
    timeline_confidence = float(segment["confidence"])
    metrics_by_criterion: dict[str, list[dict[str, Any]]] = {}
    metrics = wholebody.get("metrics")
    if not isinstance(metrics, list):
        raise ScoringContractError(
            f"WholeBody movement metrics must be a list: {movement['movement_id']}"
        )
    for metric in metrics:
        criterion_id = metric.get("criterion_id")
        if isinstance(criterion_id, str):
            metrics_by_criterion.setdefault(criterion_id, []).append(metric)

    criterion_results = [
        _criterion_result(
            criterion_id,
            metrics_by_criterion.get(criterion_id, []),
            timeline_confidence,
        )
        for criterion_id in movement["measurable_criteria"]
    ]
    identity_checks = [
        _identity_check(check, timeline_confidence)
        for check in categorical_checks
        if check.get("event_kind") in {"wrong_action", "wrong_stance"}
    ]
    identity_kinds = [item["event_kind"] for item in identity_checks]
    if sorted(identity_kinds) != ["wrong_action", "wrong_stance"]:
        raise ScoringContractError(
            f"technical conformance requires one action and one stance check: {movement['movement_id']}"
        )

    review_criteria = [
        item["criterion_id"]
        for item in criterion_results
        if item["status"] == "review_candidate"
    ]
    boundary_criteria = [
        item["criterion_id"]
        for item in criterion_results
        if item["status"] == "boundary_uncertain"
    ]
    mismatch_checks = [
        item["check_id"] for item in identity_checks if item["status"] == "mismatch_candidate"
    ]
    ambiguous_checks = [
        item["check_id"] for item in identity_checks if item["status"] == "ambiguous"
    ]
    threshold_evaluable = sum(
        item["status"]
        in {"within_screening_range", "review_candidate", "boundary_uncertain"}
        for item in criterion_results
    )

    if segment["label_status"] != "confirmed":
        conformance_status = "ambiguous"
        reason = "movement_timeline_label_is_not_confirmed"
    elif mismatch_checks:
        conformance_status = "mismatch_candidate"
        reason = "inferred_technique_or_stance_identity_mismatch"
    elif review_criteria:
        conformance_status = "review_candidate"
        reason = "one_or_more_measured_criteria_outside_screening_range"
    elif ambiguous_checks or boundary_criteria:
        conformance_status = "ambiguous"
        reason = "identity_or_numeric_boundary_is_ambiguous"
    elif threshold_evaluable == 0:
        conformance_status = "not_measurable"
        reason = "no_thresholded_criterion_was_evaluable"
    else:
        conformance_status = "consistent_within_measured_scope"
        reason = "no_conflict_found_in_evaluable_screening_criteria"

    supporting_confidences = [
        float(item["evidence_confidence"])
        for item in criterion_results
        if item["status"] != "not_measurable" and item["evidence_confidence"] is not None
    ]
    supporting_confidences.extend(
        float(item["fused_confidence"])
        for item in identity_checks
        if item["status"] not in {"not_measurable", "unsupported"}
    )
    fused_confidence = min([timeline_confidence, *supporting_confidences])

    expected_count = len(criterion_results)
    measurable_count = sum(item["status"] != "not_measurable" for item in criterion_results)
    return {
        "movement_id": movement["movement_id"],
        "display_name": movement["display_name"],
        "expected_stance": movement["stance"],
        "expected_techniques": deepcopy(movement["techniques"]),
        "start_frame": segment["start_frame"],
        "end_frame": segment["end_frame"],
        "anchor_frame": segment["anchors"].get("fixation", segment["end_frame"]),
        "timeline_label": {
            "status": segment["label_status"],
            "source": timeline_label_source,
            "confidence": timeline_confidence,
        },
        "conformance_status": conformance_status,
        "reason": reason,
        "fused_evidence_confidence": round(fused_confidence, 6),
        "review_required": conformance_status
        in {"mismatch_candidate", "review_candidate", "ambiguous"},
        "identity_checks": identity_checks,
        "criterion_coverage": {
            "expected_count": expected_count,
            "measurable_count": measurable_count,
            "threshold_evaluable_count": threshold_evaluable,
            "diagnostic_only_count": sum(
                item["status"] == "measured_diagnostic_only" for item in criterion_results
            ),
            "not_measurable_count": expected_count - measurable_count,
            "measurement_coverage_ratio": round(measurable_count / expected_count, 6)
            if expected_count
            else 0.0,
        },
        "review_triggers": {
            "identity_mismatch_check_ids": mismatch_checks,
            "criterion_ids": review_criteria,
            "boundary_uncertain_criterion_ids": boundary_criteria,
            "ambiguous_identity_check_ids": ambiguous_checks,
        },
        "aspects": _aggregate_aspects(criterion_results),
        "criteria": criterion_results,
    }


def _criterion_result(
    criterion_id: str,
    metrics: list[dict[str, Any]],
    timeline_confidence: float,
) -> dict[str, Any]:
    evaluations = [_metric_evaluation(metric, timeline_confidence) for metric in metrics]
    statuses = [item["status"] for item in evaluations]
    if "review_candidate" in statuses:
        status = "review_candidate"
    elif "boundary_uncertain" in statuses:
        status = "boundary_uncertain"
    elif "within_screening_range" in statuses:
        status = "within_screening_range"
    elif "measured_diagnostic_only" in statuses:
        status = "measured_diagnostic_only"
    else:
        status = "not_measurable"
    confidences = [
        float(item["evidence_confidence"])
        for item in evaluations
        if item["evidence_confidence"] is not None
    ]
    return {
        "criterion_id": criterion_id,
        "aspect": _aspect_for_criterion(criterion_id),
        "status": status,
        "evidence_confidence": round(min(confidences), 6) if confidences else None,
        "metrics": evaluations,
    }


def _metric_evaluation(metric: dict[str, Any], timeline_confidence: float) -> dict[str, Any]:
    value = _optional_finite(metric.get("value"), "metric value")
    uncertainty = _optional_finite(metric.get("uncertainty_95"), "metric uncertainty")
    if uncertainty is not None and uncertainty < 0:
        raise ScoringContractError("metric uncertainty must be non-negative")
    evidence = metric.get("measurement_evidence")
    if evidence is not None and not isinstance(evidence, dict):
        raise ScoringContractError("measurement_evidence must be a mapping")
    evidence = evidence or {}
    confidence = None if value is None else _evidence_confidence(evidence, timeline_confidence)
    rule = metric.get("screening_rule")
    if value is None or metric.get("screening_status") == "not_measurable":
        status = "not_measurable"
        interval = None
    elif rule is None:
        status = "measured_diagnostic_only"
        interval = [value, value] if uncertainty is None else [value - uncertainty, value + uncertainty]
    else:
        interval = [value, value] if uncertainty is None else [value - uncertainty, value + uncertainty]
        relation = _interval_relation(interval, rule)
        status = {
            "inside": "within_screening_range",
            "outside": "review_candidate",
            "overlap": "boundary_uncertain",
        }[relation]
    return {
        "metric_id": metric.get("metric_id"),
        "value": value,
        "unit": metric.get("unit"),
        "uncertainty_95": uncertainty,
        "comparison_interval": interval,
        "screening_rule": deepcopy(rule),
        "source_screening_status": metric.get("screening_status"),
        "status": status,
        "evidence_confidence": None if confidence is None else round(confidence, 6),
        "measurement_evidence": deepcopy(evidence),
    }


def _identity_check(check: dict[str, Any], timeline_confidence: float) -> dict[str, Any]:
    status = check.get("status")
    if status not in {
        "consistent",
        "mismatch_candidate",
        "ambiguous",
        "not_measurable",
        "unsupported",
    }:
        raise ScoringContractError("categorical check status is unsupported")
    raw_confidence = _optional_finite(check.get("confidence"), "categorical confidence")
    raw_confidence = 0.0 if raw_confidence is None else raw_confidence
    if not 0.0 <= raw_confidence <= 1.0:
        raise ScoringContractError("categorical confidence must be between zero and one")
    evidence = check.get("evidence") or []
    if not isinstance(evidence, list):
        raise ScoringContractError("categorical check evidence must be a list")
    coverage_factors = [
        _evidence_confidence(item.get("measurement_evidence") or {}, timeline_confidence)
        for item in evidence
    ]
    fused = min([timeline_confidence, raw_confidence, *coverage_factors])
    return {
        "check_id": check.get("check_id"),
        "event_kind": check.get("event_kind"),
        "status": status,
        "expected_label": check.get("expected_label"),
        "alternate_label": check.get("alternate_label"),
        "reason": check.get("reason"),
        "source_confidence": raw_confidence,
        "fused_confidence": round(fused, 6),
        "evidence": deepcopy(evidence),
    }


def _evidence_confidence(evidence: dict[str, Any], timeline_confidence: float) -> float:
    factors = [timeline_confidence]
    group_coverage = evidence.get("required_group_coverage") or {}
    if not isinstance(group_coverage, dict):
        raise ScoringContractError("required_group_coverage must be a mapping")
    for value in group_coverage.values():
        numeric = float(value)
        if not isfinite(numeric) or not 0.0 <= numeric <= 1.0:
            raise ScoringContractError("required group coverage must be a finite probability")
        factors.append(numeric)

    sample_counts = evidence.get("required_joint_sample_counts") or {}
    if not isinstance(sample_counts, dict):
        raise ScoringContractError("required_joint_sample_counts must be a mapping")
    start = evidence.get("start_frame")
    end = evidence.get("end_frame")
    if sample_counts and isinstance(start, int) and isinstance(end, int) and end >= start:
        window_size = end - start + 1
        for value in sample_counts.values():
            numeric = float(value)
            if not isfinite(numeric) or numeric < 0:
                raise ScoringContractError("required joint sample counts must be finite and non-negative")
            factors.append(min(numeric / window_size, 1.0))
    return min(factors)


def _interval_relation(interval: list[float], rule: dict[str, Any]) -> str:
    operator = rule.get("operator")
    limits = rule.get("value")
    low, high = interval
    if operator == "max":
        limit = _required_finite(limits, "maximum screening limit")
        return "inside" if high <= limit else "outside" if low > limit else "overlap"
    if operator == "min":
        limit = _required_finite(limits, "minimum screening limit")
        return "inside" if low >= limit else "outside" if high < limit else "overlap"
    if operator == "range" and isinstance(limits, list) and len(limits) == 2:
        lower, upper = (
            _required_finite(limits[0], "lower screening limit"),
            _required_finite(limits[1], "upper screening limit"),
        )
        if lower > upper:
            raise ScoringContractError("screening range limits are inverted")
        if low >= lower and high <= upper:
            return "inside"
        if high < lower or low > upper:
            return "outside"
        return "overlap"
    raise ScoringContractError("unsupported screening rule in technical conformance")


def _aggregate_aspects(criteria: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aspects: list[dict[str, Any]] = []
    for aspect_id in ("posture_and_stance", "technique_execution", "timing_and_control"):
        selected = [item for item in criteria if item["aspect"] == aspect_id]
        statuses = [item["status"] for item in selected]
        if "review_candidate" in statuses:
            status = "review_candidate"
        elif "boundary_uncertain" in statuses:
            status = "boundary_uncertain"
        elif "within_screening_range" in statuses:
            status = "within_screening_range"
        elif "measured_diagnostic_only" in statuses:
            status = "measured_diagnostic_only"
        else:
            status = "not_measurable"
        aspects.append(
            {
                "aspect_id": aspect_id,
                "status": status,
                "criterion_count": len(selected),
                "measurable_count": sum(item["status"] != "not_measurable" for item in selected),
                "review_candidate_count": sum(item["status"] == "review_candidate" for item in selected),
            }
        )
    return aspects


def _aspect_for_criterion(criterion_id: str) -> str:
    if criterion_id.startswith(("balance.", "stance.", "rotation.", "gaze.")):
        return "posture_and_stance"
    if criterion_id.startswith(("timing.",)) or criterion_id in {
        "technique.fixation.stability",
        "stance.weight_transfer",
        "technique.trajectory",
    }:
        return "timing_and_control"
    return "technique_execution"


def _validate_bindings(
    wholebody: dict[str, Any],
    categorical: dict[str, Any],
    spec: dict[str, Any],
    timeline: dict[str, Any],
) -> None:
    if wholebody.get("status") != "wholebody_diagnostics_only":
        raise ScoringContractError("technical conformance requires WholeBody diagnostics")
    if categorical.get("status") != "categorical_diagnostics_only":
        raise ScoringContractError("technical conformance requires categorical diagnostics")
    expected_poomsae = {"poomsae_id": spec["poomsae_id"], "version": spec["version"]}
    if wholebody.get("poomsae") != expected_poomsae or categorical.get("poomsae") != expected_poomsae:
        raise ScoringContractError("technical conformance Poomsae binding does not match")
    if wholebody.get("movement_timeline_id") != timeline["timeline_id"]:
        raise ScoringContractError("WholeBody diagnostics timeline binding does not match")
    if categorical.get("movement_timeline_id") != timeline["timeline_id"]:
        raise ScoringContractError("categorical diagnostics timeline binding does not match")
    categorical_profile = categorical.get("profile") or {}
    wholebody_profile = wholebody.get("profile") or {}
    if not isinstance(categorical_profile, dict) or not isinstance(wholebody_profile, dict):
        raise ScoringContractError("diagnostic profile bindings must be mappings")
    binding_keys = ("profile_id", "version")
    if any(categorical_profile.get(key) != wholebody_profile.get(key) for key in binding_keys):
        raise ScoringContractError("categorical and WholeBody diagnostic profile bindings do not match")


def _unique_by_movement(raw: Any, label: str) -> dict[str, dict[str, Any]]:
    if not isinstance(raw, list):
        raise ScoringContractError(f"{label} movements must be a list")
    result: dict[str, dict[str, Any]] = {}
    for item in raw:
        movement_id = item.get("movement_id")
        if not isinstance(movement_id, str) or movement_id in result:
            raise ScoringContractError(f"{label} movement ids must be unique strings")
        result[movement_id] = item
    return result


def _optional_finite(value: Any, label: str) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    if not isfinite(numeric):
        raise ScoringContractError(f"{label} must be finite or null")
    return numeric


def _required_finite(value: Any, label: str) -> float:
    numeric = float(value)
    if not isfinite(numeric):
        raise ScoringContractError(f"{label} must be finite")
    return numeric
