"""Fail-closed categorical Poomsae diagnostics from existing 3D measurements."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from src.poomsae_scoring.contracts import (
    ScoringContractError,
    validate_movement_timeline,
    validate_poomsae_spec,
)
from src.poomsae_scoring.source_bound_accuracy import derive_categorical_observations
from src.poomsae_scoring.wholebody_diagnostics import validate_wholebody_diagnostic_profile


def build_categorical_diagnostics(
    wholebody_diagnostics: dict[str, Any],
    poomsae_spec: dict[str, Any],
    movement_timeline: dict[str, Any],
    diagnostic_profile: dict[str, Any],
) -> dict[str, Any]:
    """Build pause, wrong-action and wrong-stance diagnostic observations.

    Technique and stance mismatches are deliberately emitted as ``inferred``.
    They are useful review candidates, but cannot create a source-bound major
    deduction without an independently observed confirmation.
    """
    spec = validate_poomsae_spec(poomsae_spec)
    timeline = validate_movement_timeline(movement_timeline, spec)
    profile = validate_wholebody_diagnostic_profile(diagnostic_profile)
    if profile["poomsae_id"] != spec["poomsae_id"]:
        raise ScoringContractError("WholeBody profile Poomsae id does not match")
    _validate_bindings(wholebody_diagnostics, spec, timeline, profile)

    movement_specs = {item["movement_id"]: item for item in spec["movements"]}
    reports = wholebody_diagnostics.get("movements")
    if not isinstance(reports, list):
        raise ScoringContractError("WholeBody diagnostics movements must be a list")
    segments = {item["movement_id"]: item for item in timeline["segments"]}
    checks: list[dict[str, Any]] = []
    observations = derive_categorical_observations(spec, timeline)

    for report in reports:
        movement_id = report.get("movement_id")
        movement = movement_specs.get(movement_id)
        segment = segments.get(movement_id)
        if movement is None or segment is None:
            raise ScoringContractError(f"categorical diagnostics reference unknown movement: {movement_id}")
        metrics = {item.get("metric_id"): item for item in report.get("metrics", [])}
        stance_check = _stance_check(movement, segment, metrics, profile["thresholds"])
        action_check = _action_check(movement, segment, metrics, profile["thresholds"])
        checks.extend((stance_check, action_check))
        for check in (stance_check, action_check):
            if check["status"] == "mismatch_candidate":
                observations.append(_observation_from_check(check))

    counts = {
        status: sum(item["status"] == status for item in checks)
        for status in ("consistent", "mismatch_candidate", "ambiguous", "not_measurable", "unsupported")
    }
    return {
        "schema_version": 1,
        "status": "categorical_diagnostics_only",
        "scoring_status": "inferred_candidates_not_automatic_deductions",
        "poomsae": {"poomsae_id": spec["poomsae_id"], "version": spec["version"]},
        "movement_timeline_id": timeline["timeline_id"],
        "profile": {"profile_id": profile["profile_id"], "version": profile["version"]},
        "summary": {
            "check_count": len(checks),
            "mismatch_candidate_count": counts["mismatch_candidate"],
            "consistent_count": counts["consistent"],
            "ambiguous_count": counts["ambiguous"],
            "not_measurable_count": counts["not_measurable"],
            "unsupported_count": counts["unsupported"],
            "pause_observation_count": sum(
                item["event_kind"] == "pause_at_least_3_sec" for item in observations
            ),
        },
        "checks": checks,
        "observations": observations,
        "safety_contract": {
            "kinematic_mismatch_is_direct_observation": False,
            "inferred_candidate_can_create_deduction": False,
            "human_or_validated_classifier_confirmation_required": True,
        },
        "interpretation": (
            "Yanlış hareket ve duruş sonuçları mevcut 3B mühendislik ölçümlerinden çıkarılan "
            "inceleme adaylarıdır. Doğrudan gözlem veya doğrulanmış sınıflandırıcı olmadığından "
            "otomatik WT kesintisi oluşturmazlar."
        ),
    }


def _validate_bindings(
    diagnostics: dict[str, Any],
    spec: dict[str, Any],
    timeline: dict[str, Any],
    profile: dict[str, Any],
) -> None:
    if diagnostics.get("status") != "wholebody_diagnostics_only":
        raise ScoringContractError("categorical diagnostics require a WholeBody diagnostics report")
    expected_poomsae = {"poomsae_id": spec["poomsae_id"], "version": spec["version"]}
    if diagnostics.get("poomsae") != expected_poomsae:
        raise ScoringContractError("WholeBody diagnostics Poomsae binding does not match")
    if diagnostics.get("movement_timeline_id") != timeline["timeline_id"]:
        raise ScoringContractError("WholeBody diagnostics timeline binding does not match")
    diagnostic_profile = diagnostics.get("profile", {})
    if diagnostic_profile.get("profile_id") != profile["profile_id"] or diagnostic_profile.get(
        "version"
    ) != profile["version"]:
        raise ScoringContractError("WholeBody diagnostic profile binding does not match")


def _stance_check(
    movement: dict[str, Any],
    segment: dict[str, Any],
    metrics: dict[str, dict[str, Any]],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    expected = _stance_family(movement["stance"])
    if expected not in {"ap_seogi", "ap_gubi"}:
        return _unsupported_check(movement, segment, "wrong_stance", expected)
    alternate = "ap_gubi" if expected == "ap_seogi" else "ap_seogi"
    return _profile_check(
        movement=movement,
        segment=segment,
        event_kind="wrong_stance",
        expected_label=expected,
        alternate_label=alternate,
        metric_ranges={
            "stance_span_ratio": (
                _range(thresholds, expected, "span_ratio"),
                _range(thresholds, alternate, "span_ratio"),
            ),
            "front_knee_deg": (
                _range(thresholds, expected, "front_knee_deg"),
                _range(thresholds, alternate, "front_knee_deg"),
            ),
        },
        metrics=metrics,
    )


def _action_check(
    movement: dict[str, Any],
    segment: dict[str, Any],
    metrics: dict[str, dict[str, Any]],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    technique_ids = [item["technique_id"] for item in movement["techniques"]]
    supported = [item for item in technique_ids if item in {"momtong_jireugi", "arae_makki"}]
    unsupported_companions = [
        item
        for item in technique_ids
        if item not in {"momtong_jireugi", "arae_makki", "kihap"}
    ]
    if len(supported) != 1 or unsupported_companions:
        return _unsupported_check(
            movement,
            segment,
            "wrong_action",
            "+".join(technique_ids),
        )
    expected = supported[0]
    alternate = "arae_makki" if expected == "momtong_jireugi" else "momtong_jireugi"
    expected_prefix = "punch" if expected == "momtong_jireugi" else "block"
    alternate_prefix = "block" if expected_prefix == "punch" else "punch"
    return _profile_check(
        movement=movement,
        segment=segment,
        event_kind="wrong_action",
        expected_label=expected,
        alternate_label=alternate,
        metric_ranges={
            "executing_wrist_height_torso_ratio": (
                _range(thresholds, expected_prefix, "wrist_height_torso_ratio"),
                _range(thresholds, alternate_prefix, "wrist_height_torso_ratio"),
            ),
            "executing_elbow_deg": (
                _range(thresholds, expected_prefix, "elbow_deg"),
                _range(thresholds, alternate_prefix, "elbow_deg"),
            ),
        },
        metrics=metrics,
    )


def _profile_check(
    *,
    movement: dict[str, Any],
    segment: dict[str, Any],
    event_kind: str,
    expected_label: str,
    alternate_label: str,
    metric_ranges: dict[str, tuple[tuple[float, float], tuple[float, float]]],
    metrics: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    evidence: list[dict[str, Any]] = []
    for metric_id, (expected_range, alternate_range) in metric_ranges.items():
        metric = metrics.get(metric_id)
        if metric is None or metric.get("value") is None:
            return _not_measurable_check(
                movement, segment, event_kind, expected_label, alternate_label, metric_id
            )
        value = float(metric["value"])
        raw_uncertainty = metric.get("uncertainty_95")
        uncertainty = None if raw_uncertainty is None else float(raw_uncertainty)
        interval = (
            (value, value)
            if uncertainty is None
            else (value - uncertainty, value + uncertainty)
        )
        evidence.append(
            {
                "metric_id": metric_id,
                "value": value,
                "unit": metric.get("unit"),
                "uncertainty_95": uncertainty,
                "comparison_interval": list(interval),
                "comparison_uses_uncertainty": uncertainty is not None,
                "expected_range": list(expected_range),
                "alternate_range": list(alternate_range),
                "expected_fit": _in_range(value, expected_range),
                "alternate_fit": _in_range(value, alternate_range),
                "wholly_outside_expected": _outside(interval, expected_range),
                "measurement_evidence": deepcopy(metric.get("measurement_evidence")),
            }
        )

    expected_fits = [item["expected_fit"] for item in evidence]
    alternate_fits = [item["alternate_fit"] for item in evidence]
    strong_mismatches = sum(item["wholly_outside_expected"] for item in evidence)
    if all(expected_fits):
        status, confidence, reason = "consistent", 0.85, "measurements_fit_expected_profile"
    elif all(alternate_fits) and strong_mismatches >= 1:
        status = "mismatch_candidate"
        uncertainty_count = sum(item["comparison_uses_uncertainty"] for item in evidence)
        confidence = min(0.95, 0.62 + 0.1 * strong_mismatches + 0.05 * uncertainty_count)
        reason = "alternate_profile_fits_and_expected_range_is_excluded"
    else:
        status, confidence, reason = "ambiguous", 0.5, "profiles_overlap_or_evidence_conflicts"
    return {
        "check_id": f"CAT-{event_kind.upper()}-{movement['movement_id']}",
        "movement_id": movement["movement_id"],
        "event_kind": event_kind,
        "status": status,
        "expected_label": expected_label,
        "alternate_label": alternate_label,
        "confidence": confidence,
        "reason": reason,
        "start_frame": segment["start_frame"],
        "end_frame": segment["end_frame"],
        "anchor_frame": segment["anchors"].get("fixation", segment["end_frame"]),
        "evidence": evidence,
    }


def _observation_from_check(check: dict[str, Any]) -> dict[str, Any]:
    values = ", ".join(
        f"{item['metric_id']}={item['value']:.3f}" for item in check["evidence"]
    )
    return {
        "observation_id": check["check_id"],
        "event_kind": check["event_kind"],
        "movement_id": check["movement_id"],
        "start_frame": check["anchor_frame"],
        "end_frame": check["anchor_frame"],
        "evidence_status": "inferred",
        "confidence": check["confidence"],
        "description": (
            f"3B kinematic screening: expected {check['expected_label']}, "
            f"alternate {check['alternate_label']} fits more clearly ({values})."
        ),
        "measurement": None,
        "confirmation_method": "kinematic_screening",
    }


def _unsupported_check(
    movement: dict[str, Any],
    segment: dict[str, Any],
    event_kind: str,
    expected_label: str,
) -> dict[str, Any]:
    return {
        "check_id": f"CAT-{event_kind.upper()}-{movement['movement_id']}",
        "movement_id": movement["movement_id"],
        "event_kind": event_kind,
        "status": "unsupported",
        "expected_label": expected_label,
        "alternate_label": None,
        "confidence": 0.0,
        "reason": "no_validated_alternative_profile_for_expected_label",
        "start_frame": segment["start_frame"],
        "end_frame": segment["end_frame"],
        "anchor_frame": segment["anchors"].get("fixation", segment["end_frame"]),
        "evidence": [],
    }


def _not_measurable_check(
    movement: dict[str, Any],
    segment: dict[str, Any],
    event_kind: str,
    expected_label: str,
    alternate_label: str,
    missing_metric: str,
) -> dict[str, Any]:
    result = _unsupported_check(movement, segment, event_kind, expected_label)
    result.update(
        {
            "status": "not_measurable",
            "alternate_label": alternate_label,
            "reason": f"required_metric_not_measurable:{missing_metric}",
        }
    )
    return result


def _range(
    thresholds: dict[str, float],
    prefix: str,
    metric: str,
) -> tuple[float, float]:
    return (
        float(thresholds[f"{prefix}_{metric}_min"]),
        float(thresholds[f"{prefix}_{metric}_max"]),
    )


def _in_range(value: float, limits: tuple[float, float]) -> bool:
    return limits[0] <= value <= limits[1]


def _outside(interval: tuple[float, float], limits: tuple[float, float]) -> bool:
    return interval[1] < limits[0] or interval[0] > limits[1]


def _stance_family(stance: str) -> str:
    if stance.endswith("ap_seogi"):
        return "ap_seogi"
    if stance.endswith("ap_gubi"):
        return "ap_gubi"
    return stance
