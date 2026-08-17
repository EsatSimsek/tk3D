"""Presentation diagnostic layer for Poomsae scoring.

This module does NOT produce a WT Presentation score. Presentation (6.0 total,
split into speed & power / rhythm & tempo / expression of energy) is defined by
the rule book as a judge's qualitative assessment; the sensor pipeline cannot
replicate that. What this layer produces is the kinematic proxy substrate that a
future calibration layer would need if judge labels ever became available:

- per-recording measurements aggregated from the existing WholeBody-133
  diagnostics report and the labelled movement timeline;
- component-level rollups (speed_and_power, rhythm_and_tempo,
  expression_of_energy) with sample counts, medians and dispersion measures;
- explicit contract flags forbidding score claims until judge calibration is in
  place (``judge_calibrated=False``, ``not_judge_validated=True``,
  ``total_score=None``).

Everything is derived from data that is already computed and provenance-bound in
other stages, so this module introduces no new tolerance values and no new
external sources.
"""
from __future__ import annotations

import statistics
from typing import Any

from src.poomsae_scoring.contracts import (
    ScoringContractError,
    validate_movement_timeline,
    validate_poomsae_spec,
)


PRESENTATION_STATUS = "presentation_diagnostic_only"

# Metric ids we roll up into each presentation component. These names must exist
# in the WholeBody diagnostics report; we do not invent new measurements here.
SPEED_METRICS = ("executing_wrist_peak_speed_body_scale_per_sec",)
RHYTHM_METRICS: tuple[str, ...] = ()  # rhythm is derived from timeline, not per-movement metrics
ENERGY_METRICS = (
    "fixation_wrist_jitter_ratio",
    "torso_lean_p95_deg",
    "head_torso_yaw_mismatch_deg",
    "shoulder_hip_twist_deg",
)


def build_presentation_diagnostics(
    wholebody_diagnostics: dict[str, Any],
    poomsae_spec: dict[str, Any],
    movement_timeline: dict[str, Any],
) -> dict[str, Any]:
    """Aggregate wholebody + timeline evidence into a presentation diagnostic report.

    The output is a diagnostic panel, not a score. Every component carries the
    raw sample values, a robust central measure (median), a dispersion measure
    (interquartile range), and the sample count so a future calibrator can pick
    its own reduction. No component contains a score, no total is claimed.
    """
    spec = validate_poomsae_spec(poomsae_spec)
    timeline = validate_movement_timeline(movement_timeline, spec)
    if wholebody_diagnostics.get("status") != "wholebody_diagnostics_only":
        raise ScoringContractError(
            "presentation diagnostics require a WholeBody diagnostics report"
        )

    movement_reports = wholebody_diagnostics.get("movements")
    if not isinstance(movement_reports, list):
        raise ScoringContractError("WholeBody diagnostics movements must be a list")

    speed_component = _aggregate_metric_component(movement_reports, SPEED_METRICS)
    rhythm_component = _rhythm_and_tempo_component(timeline)
    energy_component = _aggregate_metric_component(movement_reports, ENERGY_METRICS)

    recording_scope = timeline["coverage"]["recording_scope"]
    return {
        "schema_version": 1,
        "status": PRESENTATION_STATUS,
        "poomsae": {"poomsae_id": spec["poomsae_id"], "version": spec["version"]},
        "timeline_id": timeline["timeline_id"],
        "timeline_scope": recording_scope,
        "judge_calibrated": False,
        "not_judge_validated": True,
        "total_score": None,
        "components": {
            "speed_and_power": speed_component,
            "rhythm_and_tempo": rhythm_component,
            "expression_of_energy": energy_component,
        },
        "safety_contract": {
            "score_claim_allowed": False,
            "judge_calibration_required_for_score": True,
            "partial_recording_can_produce_score": False,
            "kinematic_proxy_is_not_force_measurement": True,
        },
        "interpretation": (
            "Bu rapor WT Presentation puanı üretmez. Sadece kaynak-bağlı 3B pozdan"
            " çıkarılan kinematic proxy ölçümleridir; hakem kalibrasyonu"
            " sağlanmadan Presentation skoru iddia edilemez."
        ),
    }


def _aggregate_metric_component(
    movement_reports: list[dict[str, Any]],
    metric_ids: tuple[str, ...],
) -> dict[str, Any]:
    """Roll a set of per-movement metric ids into a component summary."""
    per_metric: dict[str, dict[str, Any]] = {}
    for metric_id in metric_ids:
        samples: list[float] = []
        unit: str | None = None
        for movement_report in movement_reports:
            movement_id = movement_report.get("movement_id")
            for metric in movement_report.get("metrics", []):
                if metric.get("metric_id") != metric_id:
                    continue
                value = metric.get("value")
                if value is None:
                    continue
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    continue
                samples.append(numeric)
                if unit is None:
                    unit = metric.get("unit")
                _ = movement_id  # kept for clarity when we later want per-movement rows
        per_metric[metric_id] = _summarize(samples, unit)
    return {
        "metrics": per_metric,
        "measurable_metric_count": sum(1 for m in per_metric.values() if m["sample_count"] > 0),
        "requested_metric_count": len(metric_ids),
    }


def _rhythm_and_tempo_component(timeline: dict[str, Any]) -> dict[str, Any]:
    """Derive tempo/rhythm proxies straight from timeline segment durations and gaps."""
    fps = float(timeline["fps"])
    segments = timeline["segments"]
    durations = [
        (int(segment["end_frame"]) - int(segment["start_frame"]) + 1) / fps
        for segment in segments
    ]
    gaps = []
    for index in range(1, len(segments)):
        gap_frames = int(segments[index]["start_frame"]) - int(segments[index - 1]["end_frame"]) - 1
        gaps.append(max(gap_frames, 0) / fps)
    metrics = {
        "movement_duration_sec": _summarize(durations, "sec"),
        "transition_gap_sec": _summarize(gaps, "sec"),
    }
    return {
        "metrics": metrics,
        "measurable_metric_count": sum(1 for m in metrics.values() if m["sample_count"] > 0),
        "requested_metric_count": len(metrics),
    }


def _summarize(samples: list[float], unit: str | None) -> dict[str, Any]:
    """Return a robust rollup for a single metric across all its samples."""
    count = len(samples)
    if count == 0:
        return {
            "sample_count": 0,
            "median": None,
            "min": None,
            "max": None,
            "interquartile_range": None,
            "unit": unit,
        }
    ordered = sorted(samples)
    median = statistics.median(ordered)
    if count >= 4:
        try:
            quartiles = statistics.quantiles(ordered, n=4, method="inclusive")
            iqr = float(quartiles[2] - quartiles[0])
        except statistics.StatisticsError:
            iqr = None
    else:
        iqr = None
    return {
        "sample_count": count,
        "median": float(median),
        "min": float(ordered[0]),
        "max": float(ordered[-1]),
        "interquartile_range": iqr,
        "unit": unit,
    }
