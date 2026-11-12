from __future__ import annotations

from copy import deepcopy
import math
from typing import Any

from src.data_structures import (
    COCO_BODY_JOINT_NAMES,
    COCO_BODY_JOINTS,
    COCO_FOOT_JOINTS,
    COCO_HAND_LOCAL_JOINTS,
    COCO_LEFT_HAND_OFFSET,
    COCO_RIGHT_HAND_OFFSET,
    COCO_WHOLEBODY_KEYPOINTS,
    coco_hand_joint,
)
from src.poomsae_scoring.contracts import (
    ScoringContractError,
    validate_movement_timeline,
    validate_poomsae_spec,
)
from src.poomsae_scoring.timeline_review import timeline_review_status


_STATUS_PRESENTATION = {
    "confirmed_source_bound_minor": ("confirmed_deduction_candidate", "red", "Kesinti adayı"),
    "boundary_uncertain": ("uncertain_no_deduction", "amber", "Sınır belirsiz"),
    "not_measurable": ("not_measurable_no_deduction", "gray", "Ölçülemedi"),
    "within_source_range": ("within_source_range", "green", "Kaynak aralığında"),
    "not_applicable": ("not_applicable", "blue", "Uygulanamaz"),
    "diagnostic_review_candidate": ("diagnostic_review_candidate", "blue", "Teşhis adayı — puan yok"),
    "timeline_unverified": ("timeline_unverified", "amber", "Faz onayı yok — puan yok"),
}

# Shown next to (never instead of) each event's own result and correction text.
PHASE_APPROVAL_NOTE = (
    "Faz onayı yok; ölçüm taslak zaman aralığına bağlı. Kesinti değerlendirilmedi. "
    "Önce iki kameradan hareketi ve değerlendirme aralığını inceleyin."
)
_PHASE_APPROVAL_RESULT = "Faz onayı yok; ölçüm taslak zaman aralığına bağlı. Kesinti değerlendirilmedi."
VISUAL_GEOMETRY_UNDEFINED = "görsel geometri tanımsız"
_KNOWN_THRESHOLD_OPERATORS = {"range", "max", "min", "abs_max"}


def build_decision_evidence_events(
    accuracy_decisions: dict[str, Any],
    poomsae_spec: dict[str, Any],
    movement_timeline: dict[str, Any],
    wholebody_diagnostics: dict[str, Any] | None = None,
    alignment_anomalies: list[dict[str, Any]] | None = None,
    technical_accuracy_diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert scoring decisions into immutable, camera-renderable evidence events.

    ``alignment_anomalies`` comes from :func:`build_automatic_timeline_report` and is
    optional because a hand-labelled timeline has no alignment step and therefore no
    anomalies. Anomalies become blue review candidates carrying no deduction: they
    report that the segment-to-movement match was uncertain, not that the athlete
    made an error.

    Presentation (titles, descriptions, units, highlighted joints, geometry kind) is
    derived from data the input reports already carry; nothing here measures,
    thresholds or decides.
    """
    spec = validate_poomsae_spec(poomsae_spec)
    timeline = validate_movement_timeline(movement_timeline, spec)
    if accuracy_decisions.get("status") != "source_bound_accuracy_decisions":
        raise ScoringContractError("decision evidence requires a source-bound Accuracy report")
    if accuracy_decisions.get("timeline_id") != timeline["timeline_id"]:
        raise ScoringContractError("Accuracy decisions and MovementTimeline must match")

    movements = {item["movement_id"]: item for item in spec["movements"]}
    segments = {item["movement_id"]: item for item in timeline["segments"]}
    frame_count = int(timeline["frame_count"])
    wholebody_units = _wholebody_metric_units(wholebody_diagnostics)
    events: list[dict[str, Any]] = []
    numeric = accuracy_decisions.get("numeric_decisions")
    if not isinstance(numeric, list):
        raise ScoringContractError("numeric_decisions must be a list")
    for index, decision in enumerate(numeric, start=1):
        movement_id = decision.get("movement_id")
        if movement_id not in movements or movement_id not in segments:
            raise ScoringContractError(f"numeric decision references an unobserved movement: {movement_id}")
        status = decision.get("decision_status")
        if not isinstance(status, str) or not status:
            raise ScoringContractError(f"numeric decision has no decision_status: {decision.get('rule_id')}")
        event = _numeric_event(
            index=index,
            decision=decision,
            movement=movements[movement_id],
            segment=segments[movement_id],
            fps=timeline["fps"],
            wholebody_units=wholebody_units,
        )
        events.append(event)

    categorical = accuracy_decisions.get("categorical_decisions")
    if not isinstance(categorical, list):
        raise ScoringContractError("categorical_decisions must be a list")
    for index, decision in enumerate(categorical, start=1):
        events.append(_categorical_event(index, decision, movements, timeline["fps"]))

    diagnostic_count = 0
    if wholebody_diagnostics is not None:
        if wholebody_diagnostics.get("status") != "wholebody_diagnostics_only":
            raise ScoringContractError("diagnostic evidence requires a WholeBody diagnostics report")
        if wholebody_diagnostics.get("movement_timeline_id") != timeline["timeline_id"]:
            raise ScoringContractError("WholeBody diagnostics and MovementTimeline must match")
        candidates = wholebody_diagnostics.get("candidate_events")
        if not isinstance(candidates, list):
            raise ScoringContractError("WholeBody candidate_events must be a list")
        for index, candidate in enumerate(candidates, start=1):
            movement_id = candidate.get("movement_id")
            if movement_id not in movements or movement_id not in segments:
                raise ScoringContractError(
                    f"WholeBody candidate references an unobserved movement: {movement_id}"
                )
            events.append(
                _diagnostic_event(
                    index=index,
                    candidate=candidate,
                    movement=movements[movement_id],
                    segment=segments[movement_id],
                    fps=timeline["fps"],
                    frame_count=frame_count,
                )
            )
        diagnostic_count = len(candidates)

    technical_accuracy_count = 0
    if technical_accuracy_diagnostics is not None:
        if technical_accuracy_diagnostics.get("status") != "technical_accuracy_diagnostics_only":
            raise ScoringContractError(
                "technical accuracy evidence requires a technical accuracy diagnostics report"
            )
        if technical_accuracy_diagnostics.get("movement_timeline_id") != timeline["timeline_id"]:
            raise ScoringContractError(
                "technical accuracy diagnostics and MovementTimeline must match"
            )
        candidates = technical_accuracy_diagnostics.get("candidate_events")
        if not isinstance(candidates, list):
            raise ScoringContractError("technical accuracy candidate_events must be a list")
        rules = _records_by(technical_accuracy_diagnostics.get("rule_inventory"), "rule_id")
        contracts = _records_by(technical_accuracy_diagnostics.get("movement_contracts"), "movement_id")
        for index, candidate in enumerate(candidates, start=1):
            movement_id = candidate.get("movement_id")
            if movement_id not in movements or movement_id not in segments:
                raise ScoringContractError(
                    f"technical accuracy candidate references an unobserved movement: {movement_id}"
                )
            events.append(
                _technical_accuracy_diagnostic_event(
                    index=index,
                    candidate=candidate,
                    movement=movements[movement_id],
                    segment=segments[movement_id],
                    fps=timeline["fps"],
                    frame_count=frame_count,
                    rule=rules.get(candidate.get("rule_id")) or {},
                    contract=contracts.get(movement_id) or {},
                )
            )
        technical_accuracy_count = len(candidates)

    anomaly_count = 0
    if alignment_anomalies is not None:
        if not isinstance(alignment_anomalies, list):
            raise ScoringContractError("alignment_anomalies must be a list")
        for index, anomaly in enumerate(alignment_anomalies, start=1):
            events.append(
                _alignment_anomaly_event(
                    index=index,
                    anomaly=anomaly,
                    movements=movements,
                    frame_count=timeline["frame_count"],
                    fps=timeline["fps"],
                )
            )
        anomaly_count = len(alignment_anomalies)

    for event in events:
        review = timeline_review_status(timeline, movement_id=event.get("movement_id"))
        event["timeline_review"] = review
        if review["status"] != "verified":
            explanation = event.setdefault("user_explanation", {})
            if event["decision_status"] != "diagnostic_review_candidate":
                if event["decision_status"] != "timeline_unverified":
                    # The metric-specific result described a status that no longer
                    # applies; the correction text stays metric-specific.
                    explanation["result"] = _PHASE_APPROVAL_RESULT
                event["decision_status"] = "timeline_unverified"
                event["display_status"], event["display_color"], event["display_label"] = _STATUS_PRESENTATION["timeline_unverified"]
            else:
                event["display_label"] += " · Faz onayı yok"
            event["application_status"] = "review_only"
            event["deduction_points"] = None
            explanation["phase_approval_note"] = PHASE_APPROVAL_NOTE
    counts = {key: sum(event["decision_status"] == key for event in events) for key in _STATUS_PRESENTATION}
    return {
        "schema_version": 1,
        "status": "decision_evidence_events",
        "timeline_id": timeline["timeline_id"],
        "frame_count": timeline["frame_count"],
        "fps": timeline["fps"],
        "timeline_review": timeline_review_status(timeline),
        "measurement_space": "tk3d_world_3d",
        "camera_overlay_space": "observed_vitpose_2d_visual_trace_only",
        "camera_overlay_warning": (
            "Numeric decisions are calculated in calibrated 3D. Camera drawings use the stored "
            "ViTPose 2D observations only as a visual trace and are never rescored as 2D angles."
        ),
        "summary": {
            "event_count": len(events),
            "confirmed_deduction_candidate_count": counts["confirmed_source_bound_minor"],
            "boundary_uncertain_count": counts["boundary_uncertain"],
            "not_measurable_count": sum(
                event["decision_status"] == "not_measurable"
                or (event["event_kind"] == "numeric_source_bound_decision" and event["measurement"]["value"] is None)
                for event in events
            ),
            "within_source_range_count": counts["within_source_range"],
            "timeline_unverified_count": sum(event["timeline_review"]["status"] != "verified" for event in events),
            "categorical_event_count": len(categorical),
            "diagnostic_review_candidate_count": diagnostic_count,
            "technical_accuracy_review_candidate_count": technical_accuracy_count,
            "alignment_anomaly_count": anomaly_count,
            "presentation_warning_count": sum(bool(event.get("presentation_warnings")) for event in events),
            "visual_geometry_undefined_count": sum(
                (event.get("visual_geometry") or {}).get("visual_status") == VISUAL_GEOMETRY_UNDEFINED
                for event in events
            ),
        },
        "events": events,
    }


def _records_by(records: Any, key: str) -> dict[str, dict[str, Any]]:
    if not isinstance(records, list):
        return {}
    return {item[key]: item for item in records if isinstance(item, dict) and isinstance(item.get(key), str)}


def _wholebody_metric_units(report: dict[str, Any] | None) -> dict[tuple[str, str], str]:
    """Units of the WholeBody measurements that source-bound decisions are made on."""
    units: dict[tuple[str, str], str] = {}
    if not isinstance(report, dict):
        return units
    for movement in report.get("movements") or []:
        if not isinstance(movement, dict):
            continue
        for metric in movement.get("metrics") or []:
            if isinstance(metric, dict) and isinstance(metric.get("unit"), str) and metric.get("unit"):
                units[(str(movement.get("movement_id")), str(metric.get("metric_id")))] = metric["unit"]
    return units


_ALIGNMENT_ANOMALY_TEXT = {
    "unmatched_movement": (
        "Bu hareket için eşleşen segment bulunamadı",
        "Videoyu bu aralıkta izleyip hareketin yapılıp yapılmadığını doğrulayın; "
        "yapıldıysa segment tespiti kaçırmış demektir.",
    ),
    "unmatched_segment": (
        "Bu kare aralığı hiçbir harekete oturmadı",
        "Fazladan bir hareket, tekrar ya da hazırlık/bitiş duruşu olabilir; "
        "videodan doğrulayın.",
    ),
    "segment_spans_multiple_movements": (
        "Tek segment birden fazla hareketi kapsıyor",
        "Sporcu iki hareketi arada durmadan birleştirmiş olabilir; "
        "segment sınırını videodan kontrol edin.",
    ),
    "ambiguous_match": (
        "Eşleşme belirsizlik bandında",
        "Eşleşme sporcu lehine korundu; doğru hareket olup olmadığını videodan teyit edin.",
    ),
    "movement_split_across_segments": (
        "Hareket birden fazla segmente bölünmüş tespit edildi",
        "Parçalar tek aralığa birleştirildi; birleşik aralığın hareketi doğru kapsayıp "
        "kapsamadığını videodan kontrol edin.",
    ),
}


def _alignment_anomaly_event(
    *,
    index: int,
    anomaly: dict[str, Any],
    movements: dict[str, Any],
    frame_count: int,
    fps: float,
) -> dict[str, Any]:
    """Turn one alignment anomaly into a blue, point-free review event.

    An anomaly about a movement that never got a segment has no frame window of its
    own, so the window is left null rather than invented; the renderer must skip such
    an event instead of drawing it at a guessed position.
    """
    issue = anomaly.get("issue")
    if issue not in _ALIGNMENT_ANOMALY_TEXT:
        raise ScoringContractError(f"unsupported alignment anomaly issue: {issue}")
    movement_id = anomaly.get("movement_id")
    if movement_id is not None and movement_id not in movements:
        raise ScoringContractError(f"alignment anomaly references an unknown movement: {movement_id}")
    movement = movements.get(movement_id) if movement_id else None
    start = anomaly.get("start_frame")
    end = anomaly.get("end_frame")
    window = None
    if start is not None and end is not None:
        start = _frame(start, 0)
        end = _frame(end, start)
        if not 0 <= start <= end < frame_count:
            raise ScoringContractError(f"alignment anomaly window is outside the timeline: {issue}")
        window = {
            "start_frame": start,
            "anchor_frame": start,
            "end_frame": end,
            "start_time_sec": start / fps,
            "anchor_time_sec": start / fps,
            "end_time_sec": end / fps,
            "scope": "alignment_segment",
        }
    title, correction = _ALIGNMENT_ANOMALY_TEXT[issue]
    label = movement_id or f"SEG{anomaly.get('segment_id')}"
    cost = anomaly.get("cost")
    unit = "pose_distance" if cost is not None else None
    return {
        "event_id": f"ALN-{index:03d}-{label}-{issue}",
        "event_kind": "alignment_anomaly_review_candidate",
        "movement_id": movement_id,
        "movement_name": movement["display_name"] if movement else None,
        "technique_id": movement["techniques"][0]["technique_id"] if movement else None,
        "metric_id": None,
        "rule_id": None,
        "description": title,
        "decision_status": "diagnostic_review_candidate",
        "display_status": "diagnostic_review_candidate",
        "display_label": "Hizalama şüphesi — puan yok",
        "display_color": "blue",
        "application_status": "review_only",
        "deduction_kind": None,
        "deduction_points": None,
        "measurement": {
            "value": cost,
            "unit": unit,
            "unit_label": None if unit is None else unit_label(unit),
            "unit_source": None if unit is None else "alignment_cost",
            "uncertainty_95": None,
            "interval_95": None,
            "rule_operator": None,
            "rule_limits": [],
            "boundary_guard": None,
            "sample_count": None,
        },
        "user_explanation": {
            "title": title,
            "observed": str(anomaly.get("detail") or title),
            "measured": "Hizalama maliyeti yok." if cost is None else format_measured_text(cost, unit),
            "correction": correction,
            "authority": (
                "Bu bir kesinti değildir. Otomatik hizalamanın kendi belirsizliğini "
                "bildirir; sebebi sporcu hatası da olabilir, segment tespitinin hatası da."
            ),
        },
        "evidence_window": window,
        "visual_geometry": None,
        "alignment": {
            "issue": issue,
            "segment_id": anomaly.get("segment_id"),
            "competing_movement_id": anomaly.get("competing_movement_id"),
        },
    }


def _diagnostic_event(
    *,
    index: int,
    candidate: dict[str, Any],
    movement: dict[str, Any],
    segment: dict[str, Any],
    fps: float,
    frame_count: int,
    presentation: dict[str, Any] | None = None,
    window: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    warnings = list(warnings or [])
    if window is None:
        evidence = candidate.get("measurement_evidence")
        if not isinstance(evidence, dict):
            raise ScoringContractError("WholeBody candidate must contain measurement_evidence")
        anchor = _frame(evidence.get("anchor_frame"), segment["end_frame"])
        start = _frame(evidence.get("start_frame"), anchor)
        end = _frame(evidence.get("end_frame"), anchor)
        start, anchor, end = _bounded_window(
            start, anchor, end, segment, frame_count, warnings, label=str(candidate.get("metric_id"))
        )
        window = {"start_frame": start, "anchor_frame": anchor, "end_frame": end, "scope": evidence.get("scope")}
    start, anchor, end = window["start_frame"], window["anchor_frame"], window["end_frame"]
    metric_id = str(candidate.get("metric_id"))
    screening_rule = candidate.get("screening_rule") or {}
    unit, unit_source = _declared_unit(candidate.get("unit"), "candidate")
    if presentation is None:
        presentation = _wholebody_presentation(candidate, movement, unit)
    visual = presentation["visual"]
    visual["unit"] = unit
    explanation = _diagnostic_explanation(candidate, movement, unit, presentation)
    event = {
        "event_id": f"DIA-{index:03d}-{movement['movement_id']}-{metric_id}",
        "event_kind": "wholebody_diagnostic_review_candidate",
        "movement_id": movement["movement_id"],
        "movement_name": movement["display_name"],
        "technique_id": movement["techniques"][0]["technique_id"],
        "metric_id": candidate.get("metric_id"),
        "rule_id": None,
        "description": presentation["description"],
        "decision_status": "diagnostic_review_candidate",
        "display_status": "diagnostic_review_candidate",
        "display_label": "Teşhis adayı — puan yok",
        "display_color": "blue",
        "application_status": "review_only",
        "deduction_kind": None,
        "deduction_points": None,
        "measurement": {
            "value": candidate.get("value"),
            "unit": unit,
            "unit_label": unit_label(unit),
            "unit_source": unit_source,
            "uncertainty_95": candidate.get("uncertainty_95"),
            "interval_95": None,
            "rule_operator": screening_rule.get("operator"),
            "rule_limits": _screening_limits(screening_rule),
            "expected_boolean": screening_rule.get("expected_boolean"),
            "boundary_guard": None,
            "sample_count": candidate.get("sample_count"),
        },
        "user_explanation": explanation,
        "presentation_source": presentation["sources"],
        "evidence_window": {
            "start_frame": start,
            "anchor_frame": anchor,
            "end_frame": end,
            "start_time_sec": start / fps,
            "anchor_time_sec": anchor / fps,
            "end_time_sec": end / fps,
            "scope": window.get("scope"),
        },
        "visual_geometry": visual,
        "source": None,
        "reason": "engineering_screening_threshold_exceeded",
        "review": {
            "status": "pending",
            "allowed_values": ["confirmed", "rejected", "uncertain"],
        },
    }
    if window.get("subwindows"):
        event["evidence_window"]["subwindows"] = window["subwindows"]
    if window.get("anchor_source"):
        event["evidence_window"]["anchor_source"] = window["anchor_source"]
    if warnings:
        event["presentation_warnings"] = warnings
    return event


def _bounded_window(
    start: int,
    anchor: int,
    end: int,
    segment: dict[str, Any],
    frame_count: int,
    warnings: list[str],
    *,
    label: str,
) -> tuple[int, int, int]:
    """Keep a candidate's own window; clip only to the video, warn outside the segment.

    An inverted window or one that does not overlap the video at all is malformed and
    fails. A window reaching past the movement segment is a legitimate measurement
    (for example a post-fixation tail or a rule measured in another phase), so it is
    kept and marked instead of failing the whole run.
    """
    if start > end:
        raise ScoringContractError(f"diagnostic evidence window is inverted: {label}")
    if end < 0 or start >= frame_count:
        raise ScoringContractError(f"diagnostic evidence window is outside the timeline: {label}")
    if start < 0 or end >= frame_count:
        start, end = max(start, 0), min(end, frame_count - 1)
        warnings.append("evidence_window_clipped_to_video_timeline")
    if not segment["start_frame"] <= start <= end <= segment["end_frame"]:
        warnings.append("evidence_window_extends_beyond_movement_segment")
    if not start <= anchor <= end:
        anchor = min(max(anchor, start), end)
        warnings.append("evidence_anchor_clamped_into_window")
    return start, anchor, end


def _technical_window(
    candidate: dict[str, Any],
    evidence: dict[str, Any],
    segment: dict[str, Any],
    frame_count: int,
    warnings: list[str],
) -> dict[str, Any]:
    """Use the candidate's own evaluation window and anchor; the timeline is the fallback."""
    raw_windows = evidence.get("window")
    if isinstance(raw_windows, dict):
        raw_windows = [raw_windows]
    subwindows = [
        {"name": item.get("name"), "start_frame": int(item["start_frame"]), "end_frame": int(item["end_frame"])}
        for item in (raw_windows if isinstance(raw_windows, list) else [])
        if isinstance(item, dict) and item.get("start_frame") is not None and item.get("end_frame") is not None
    ]
    fixation = segment.get("anchors", {}).get("fixation", segment["end_frame"])
    if subwindows:
        start = min(item["start_frame"] for item in subwindows)
        end = max(item["end_frame"] for item in subwindows)
    else:
        start = _frame(evidence.get("start_frame"), max(segment["start_frame"], fixation - 5))
        end = _frame(evidence.get("end_frame"), min(segment["end_frame"], fixation + 5))
    if evidence.get("anchor_frame") is not None:
        anchor, anchor_source = int(evidence["anchor_frame"]), "candidate_anchor_frame"
    elif evidence.get("fixation_frame") is not None:
        anchor, anchor_source = int(evidence["fixation_frame"]), "candidate_fixation_frame"
    else:
        anchor, anchor_source = fixation, "timeline_fixation_anchor"
    start, anchor, end = _bounded_window(
        start, anchor, end, segment, frame_count, warnings, label=str(candidate.get("metric_id"))
    )
    names = [str(item["name"]) for item in subwindows if item.get("name")]
    scope = evidence.get("scope") or ("+".join(names) if names else candidate.get("phase_or_window"))
    window = {"start_frame": start, "anchor_frame": anchor, "end_frame": end, "scope": scope,
              "anchor_source": anchor_source}
    if len(subwindows) > 1:
        window["subwindows"] = subwindows
    return window


def _technical_accuracy_diagnostic_event(
    *,
    index: int,
    candidate: dict[str, Any],
    movement: dict[str, Any],
    segment: dict[str, Any],
    fps: float,
    frame_count: int,
    rule: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, Any]:
    if candidate.get("decision_status") != "review_candidate_not_deduction":
        raise ScoringContractError(
            "technical accuracy evidence accepts only score-neutral review candidates"
        )
    if candidate.get("score_effect") is not None or candidate.get("deduction_points") is not None:
        raise ScoringContractError("technical accuracy candidate must remain score-neutral")
    raw_evidence = candidate.get("evidence")
    evidence = raw_evidence if isinstance(raw_evidence, dict) else {}
    warnings: list[str] = []
    window = _technical_window(candidate, evidence, segment, frame_count, warnings)
    threshold = candidate.get("threshold") or {}
    value = candidate.get("value")
    unit = candidate.get("unit")
    if threshold:
        screening_rule = _technical_screening_rule(threshold, warnings)
        reason = "unvalidated_technical_accuracy_screening_threshold_exceeded"
    elif isinstance(value, bool):
        expected = candidate.get("expected_boolean")
        if type(expected) is not bool:
            raise ScoringContractError("technical accuracy expected_boolean must be boolean")
        if value == expected:
            raise ScoringContractError("boolean review candidate already satisfies its expectation")
        if unit != "bool":
            warnings.append(f"boolean_value_with_non_bool_unit:{unit}")
        screening_rule = {"operator": "bool_true" if expected else "bool_false", "expected_boolean": expected}
        reason = "unvalidated_technical_accuracy_boolean_condition_failed"
    elif isinstance(value, str) and value:
        # A state- or reason-code-valued rule: the measurement layer already decided it
        # is a candidate; the presentation shows the state and its declared expectation.
        expected_state = candidate.get("expected_value_or_range")
        screening_rule = {"operator": "state", "expected_state": expected_state}
        warnings.append("state_valued_candidate_without_numeric_threshold")
        reason = "unvalidated_technical_accuracy_state_condition_failed"
    else:
        raise ScoringContractError(
            "thresholdless technical accuracy candidates must be boolean conditions"
        )
    adapted = {
        "movement_id": candidate.get("movement_id"),
        "metric_id": candidate.get("metric_id"),
        "criterion_id": candidate.get("criterion_id"),
        "family": candidate.get("rule_family"),
        "value": value,
        "unit": unit,
        "uncertainty_95": candidate.get("uncertainty"),
        "sample_count": None,
        "screening_rule": screening_rule,
    }
    declared_unit, _ = _declared_unit(unit, "candidate")
    presentation = _technical_presentation(
        candidate=candidate,
        rule=rule,
        contract=contract,
        movement=movement,
        unit=declared_unit,
        screening_rule=screening_rule,
        window=window,
    )
    event = _diagnostic_event(
        index=index,
        candidate=adapted,
        movement=movement,
        segment=segment,
        fps=fps,
        frame_count=frame_count,
        presentation=presentation,
        window=window,
        warnings=warnings,
    )
    event["event_id"] = (
        f"TAD-{index:03d}-{movement['movement_id']}-{candidate.get('metric_id')}"
    )
    event["event_kind"] = "technical_accuracy_diagnostic_review_candidate"
    event["rule_id"] = candidate.get("rule_id")
    event["criterion_id"] = candidate.get("criterion_id")
    event["rule_family"] = candidate.get("rule_family")
    event["provenance"] = candidate.get("provenance")
    event["rule_eligibility"] = candidate.get("rule_eligibility")
    event["reason"] = reason
    if screening_rule.get("operator") == "state":
        event["measurement"]["expected_state"] = screening_rule.get("expected_state")
    return event


def _screening_limits(screening_rule: dict[str, Any]) -> list[float]:
    if "range" in screening_rule:
        return [float(value) for value in screening_rule["range"]]
    value = screening_rule.get("value")
    if isinstance(value, list):
        return [float(item) for item in value if _is_number(item)]
    if _is_number(value):
        return [float(value)]
    return []


def _is_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


def _technical_screening_rule(threshold: dict[str, Any], warnings: list[str]) -> dict[str, Any]:
    operator = threshold.get("operator")
    value = threshold.get("value")
    if not isinstance(operator, str) or not operator:
        raise ScoringContractError("technical accuracy threshold has no operator")
    if operator == "range":
        if not isinstance(value, list) or len(value) != 2:
            raise ScoringContractError("technical accuracy range threshold must contain two limits")
        limits = [_finite_rule_limit(item) for item in value]
        if limits[0] > limits[1]:
            raise ScoringContractError("technical accuracy range threshold is inverted")
        return {"operator": "range", "range": limits}
    if operator not in _KNOWN_THRESHOLD_OPERATORS:
        # A new operator is not malformed input; the presentation layer simply cannot
        # phrase it. Its limits are shown raw and the event is marked.
        warnings.append(f"unrecognized_threshold_operator:{operator}")
        raw = value if isinstance(value, list) else [value]
        return {"operator": operator, "value": [float(item) for item in raw if _is_number(item)],
                "raw_value": deepcopy(value)}
    return {"operator": operator, "value": _finite_rule_limit(value)}


def _finite_rule_limit(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ScoringContractError("technical accuracy threshold limit must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ScoringContractError("technical accuracy threshold limit must be finite")
    return numeric


def _diagnostic_explanation(
    candidate: dict[str, Any],
    movement: dict[str, Any],
    unit: str | None,
    presentation: dict[str, Any],
) -> dict[str, str]:
    rule = candidate.get("screening_rule") or {}
    limits = _screening_limits(rule)
    value = candidate.get("value")
    operator = rule.get("operator")
    expected = format_expected_text(
        operator, limits, unit, expected_boolean=rule.get("expected_boolean"),
        expected_state=rule.get("expected_state"), raw_value=rule.get("raw_value"),
    )
    measured = format_measured_text(value, unit, operator=operator)
    comparison = _comparison_text(value, None, operator, limits, unit)
    return {
        "title": presentation["title"],
        "expected": expected,
        "measured": measured,
        "interval": "Bu teşhis metriğinde kaynak-bağlı %95 kesinti aralığı yok.",
        "comparison": comparison,
        "result": "Sonuç: yalnız inceleme adayıdır; Accuracy puanı düşürülmedi.",
        "correction": presentation["correction"],
        "source_note": (
            "Mühendislik tarama eşiği; güncel WT kesinti toleransı değildir. "
            f"WholeBody-133 ölçümü, {movement['movement_id']} sabitlenme/faz kanıtı."
        ),
    }


_EXPLICIT_DIAGNOSTIC_TEXT = {
    "fist_closure_ratio": (
        "YUMRUK KAPALILIK GEOMETRİSİ",
        "21 el noktasında parmak uçlarını avuç merkezine yaklaştırarak yumruğu sıkı kapat.",
    ),
    "wrist_forearm_alignment_deg": (
        "EL BİLEĞİ–ÖN KOL HİZASI",
        "Bileği ön kol doğrultusuyla hizala; içe veya dışa kırılmayı azalt.",
    ),
    "head_torso_yaw_mismatch_deg": (
        "BAŞ/YÜZ–GÖVDE YÖN FARKI",
        "Baş ve yüz yönünü hareketin hedef yönüyle tekrar kontrol et.",
    ),
    "hand_foot_settle_difference_sec": (
        "EL–AYAK BİTİŞ EŞZAMANLILIĞI",
        "Teknik elinin ve duruş ayağının sabitlenmesini aynı anda tamamla.",
    ),
    "fixation_wrist_jitter_ratio": (
        "TEKNİK BİTİŞ KARARLILIĞI",
        "Teknik bittikten sonra bileği sabitle; artçı salınımı azalt.",
    ),
    "pelvis_weight_transfer_ratio": (
        "AĞIRLIK AKTARIMI",
        "Adım ve teknik boyunca pelvis aktarımını duruş yönünde tamamla.",
    ),
}

_EXPLICIT_SOURCE_BOUND_TEXT = {
    "back_foot_yaw_to_stance_direction_deg": (
        "ARKA AYAK ACISI",
        "Arka ayağı duruş yönüne yaklaştır; ayak açısını izin verilen sınıra indir.",
    ),
    "arae_fist_to_thigh_fist_ratio": (
        "ARAE-MAKKI YUMRUK-UYLUK MESAFESI",
        "Blok bitişinde yumruğu uyluğa, hedeflenen yumruk-genişliği aralığına getir.",
    ),
    "executing_elbow_deg": (
        "UYGULAYAN DIRSEK ACISI",
        "Omuz-dirsek-bilek hizasını hedef açı aralığına getir.",
    ),
}

# Mirrors the ``unit`` of the three source-bound accuracy rules
# (config/scoring/accuracy/taegeuk_1_source_bound_v1.yaml). Used only when neither the
# decision nor the WholeBody measurement it was made on declares a unit.
_SOURCE_BOUND_RULE_UNITS = {
    "back_foot_yaw_to_stance_direction_deg": "deg",
    "arae_fist_to_thigh_fist_ratio": "fist_width",
    "executing_elbow_deg": "deg",
}


def _derived_correction(title: str) -> str:
    return (
        f"Kurala özel düzeltme metni profilde tanımlı değil; “{title}” ölçümünü iki kameradan "
        "beklenen değerle karşılaştırın."
    )


def _numeric_event(
    *,
    index: int,
    decision: dict[str, Any],
    movement: dict[str, Any],
    segment: dict[str, Any],
    fps: float,
    wholebody_units: dict[tuple[str, str], str],
) -> dict[str, Any]:
    status = decision["decision_status"]
    warnings: list[str] = []
    if status in _STATUS_PRESENTATION:
        display_status, color, label = _STATUS_PRESENTATION[status]
    else:
        # A status the presentation layer does not know: keep it verbatim, never
        # recolour it as a known meaning, and say so on the event.
        display_status, color, label = (
            "unrecognized_decision_status", "gray", f"Tanınmayan karar durumu: {status}",
        )
        warnings.append(f"unrecognized_decision_status:{status}")
    evidence = decision.get("measurement_evidence")
    if not isinstance(evidence, dict):
        evidence = {}
    anchor = _frame(evidence.get("anchor_frame"), segment["end_frame"])
    start = _frame(evidence.get("start_frame"), anchor)
    end = _frame(evidence.get("end_frame"), anchor)
    if not segment["start_frame"] <= start <= anchor <= end <= segment["end_frame"]:
        # Source-bound decisions can carry deductions; a window outside the movement is
        # a contract break between scoring and evidence, not a presentation variant.
        raise ScoringContractError(f"decision evidence window is outside {movement['movement_id']}")
    metric_id = decision["metric_id"]
    if decision.get("unit"):
        unit, unit_source = str(decision["unit"]), "decision"
    elif (movement["movement_id"], metric_id) in wholebody_units:
        unit, unit_source = wholebody_units[(movement["movement_id"], metric_id)], "wholebody_measurement"
    elif metric_id in _SOURCE_BOUND_RULE_UNITS:
        unit, unit_source = _SOURCE_BOUND_RULE_UNITS[metric_id], "source_bound_rule_unit_map"
    else:
        unit, unit_source = None, "not_declared"
    explicit = _explicit_visual_geometry(metric_id, movement)
    if explicit is not None:
        visual = explicit
        visual["derivation"] = "explicit_metric_map"
    else:
        visual = _derive_visual(
            metric_id=metric_id,
            unit=unit,
            family=None,
            landmarks=_evidence_landmarks(evidence),
            sides=_metric_sides(_tokens(metric_id), {}, _technique_side(movement)),
            title=_metric_explanation(metric_id)[0],
            landmark_source="measurement_evidence.required_joint_sample_counts",
        )
    visual["unit"] = unit
    explanation = _user_explanation(decision, unit)
    if explanation["title"] == metric_id.upper():
        explanation["title"] = _derived_title(metric_id)[0]
        explanation["correction"] = _derived_correction(explanation["title"])
    visual.setdefault("label", explanation["title"])
    event = {
        "event_id": f"NUM-{index:03d}-{movement['movement_id']}-{metric_id}",
        "event_kind": "numeric_source_bound_decision",
        "movement_id": movement["movement_id"],
        "movement_name": movement["display_name"],
        "technique_id": movement["techniques"][0]["technique_id"],
        "metric_id": metric_id,
        "rule_id": decision["rule_id"],
        "description": decision["description"],
        "decision_status": status,
        "conditional_geometry_status": decision.get("conditional_geometry_status"),
        "display_status": display_status,
        "display_label": label,
        "display_color": color,
        "application_status": decision.get("application_status"),
        "deduction_kind": decision.get("deduction_kind"),
        "deduction_points": decision.get("deduction_points"),
        "measurement": {
            "value": decision.get("measured_value"),
            "unit": unit,
            "unit_label": unit_label(unit),
            "unit_source": unit_source,
            "uncertainty_95": decision.get("effective_uncertainty_95"),
            "interval_95": deepcopy(decision.get("measurement_interval_95")),
            "rule_operator": decision.get("rule_operator"),
            "rule_limits": deepcopy(decision.get("rule_limits")),
            "boundary_guard": decision.get("boundary_guard"),
            "sample_count": decision.get("sample_count"),
        },
        "user_explanation": explanation,
        "evidence_window": {
            "start_frame": start,
            "anchor_frame": anchor,
            "end_frame": end,
            "start_time_sec": start / fps,
            "anchor_time_sec": anchor / fps,
            "end_time_sec": end / fps,
            "scope": evidence.get("scope"),
        },
        "visual_geometry": visual,
        "source": deepcopy(decision.get("source")),
        "reason": decision.get("reason"),
        "review": {
            "status": "pending",
            "allowed_values": ["confirmed", "rejected", "uncertain"],
        },
    }
    if warnings:
        event["presentation_warnings"] = warnings
    return event


def _user_explanation(decision: dict[str, Any], unit: str | None) -> dict[str, str]:
    measurement = decision.get("measured_value")
    interval = decision.get("measurement_interval_95")
    limits = decision.get("rule_limits") or []
    operator = decision.get("rule_operator")
    metric_id = decision["metric_id"]
    title, correction = _metric_explanation(metric_id)
    expected = _expected_text(operator, limits, unit)
    if measurement is None:
        missing = (decision.get("measurement_evidence") or {}).get("missing_required_joints") or []
        missing_text = "" if not missing else f" Eksik/kesintili kritik noktalar: {', '.join(missing)}."
        measured = f"Olcum yapilamadi.{missing_text}"
    else:
        measured = format_measured_text(measurement, unit)
    interval_text = (
        "Belirsizlik araligi hesaplanamadi."
        if not isinstance(interval, list) or len(interval) != 2
        else f"%95 olası aralık: {float(interval[0]):.2f} - {float(interval[1]):.2f} {unit_label(unit)}"
    )
    comparison = _comparison_text(measurement, interval, operator, limits, unit)
    result = _result_text(decision, expected)
    source = decision.get("source") or {}
    pages = source.get("pages") or []
    page_text = ", ".join(str(page) for page in pages) if pages else "belirtilmedi"
    authority = source.get("authority_status")
    if authority == "historical_official_not_current_attachment":
        source_text = f"2014 tarihli resmi fakat tarihsel kilavuz, sayfa {page_text}; guncel WT eki degildir."
    else:
        source_text = f"{source.get('source_id') or 'kaynak belirtilmedi'}, sayfa {page_text}."
    return {
        "title": title,
        "expected": expected,
        "measured": measured,
        "interval": interval_text,
        "comparison": comparison,
        "result": result,
        "correction": correction,
        "source_note": source_text,
    }


def _metric_explanation(metric_id: str) -> tuple[str, str]:
    return _EXPLICIT_SOURCE_BOUND_TEXT.get(
        metric_id, (metric_id.upper(), "Teknik konumu kaynak araligina getir.")
    )


def _expected_text(operator: Any, limits: list[Any], unit: str | None) -> str:
    label = unit_label(unit)
    if operator == "bool_true":
        return "Olması gereken: teknik koşul sağlanmalı (beklenen: Evet)."
    if operator == "bool_false":
        return "Olması gereken: hata koşulu bulunmamalı (beklenen: Hayır)."
    if operator == "max" and limits:
        return f"Olması gereken: en fazla {float(limits[0]):.2f} {label}."
    if operator == "abs_max" and limits:
        return f"Olması gereken: mutlak değer en fazla {float(limits[0]):.2f} {label}."
    if operator == "min" and limits:
        return f"Olması gereken: en az {float(limits[0]):.2f} {label}."
    if operator == "range" and len(limits) == 2:
        return f"Olması gereken: {float(limits[0]):.2f} - {float(limits[1]):.2f} {label}."
    return "Olması gereken kaynakta sayısal olarak tanımlı değil."


def format_expected_text(
    operator: Any,
    limits: list[Any],
    unit: str | None,
    *,
    expected_boolean: Any = None,
    expected_state: Any = None,
    raw_value: Any = None,
) -> str:
    """Single formatting path for an expectation, used by events, cards and HTML."""
    if operator == "state":
        if expected_state is None:
            return "Olması gereken durum profilde tanımlı değil."
        return f"Olması gereken durum: {expected_state}."
    if (
        isinstance(operator, str)
        and operator not in _KNOWN_THRESHOLD_OPERATORS | {"bool_true", "bool_false"}
    ):
        shown = raw_value if raw_value is not None else limits
        return (
            f"Olması gereken: operatör “{operator}”, değer {shown} {unit_label(unit)} "
            "(sunum katmanı bu operatörü tanımıyor; ham gösterim)."
        )
    if operator is None and isinstance(expected_boolean, bool):
        operator = "bool_true" if expected_boolean else "bool_false"
    return (
        _expected_text(operator, limits, unit)
        .replace("Olmasi", "Olması")
        .replace("mutlak değer", "mutlak değer")
        .replace("sayisal olarak tanimli degil", "sayısal olarak tanımlı değil")
    )


def format_measured_text(value: Any, unit: str | None, *, operator: Any = None) -> str:
    """Single formatting path for a measured value, used by events, cards and HTML."""
    if value is None:
        return "Ölçüm yapılamadı."
    if isinstance(value, bool):
        if operator == "bool_false":
            return "Evet — hata koşulu mevcut." if value else "Hayır — hata koşulu yok."
        return "Evet — koşul sağlandı." if value else "Hayır — koşul sağlanmadı."
    if isinstance(value, str):
        return f"Durum: {value}"
    if not _is_number(value):
        return f"Ölçüm sayısal değil: {value}"
    return f"{float(value):.2f} {unit_label(unit)}"


def _comparison_text(
    value: Any,
    interval: Any,
    operator: Any,
    limits: list[Any],
    unit: str | None,
) -> str:
    if operator in {"bool_true", "bool_false"}:
        if not isinstance(value, bool):
            return "Koşul karşılaştırılamadı; boolean ölçüm yok."
        return "Koşul sağlandı." if value == (operator == "bool_true") else "Koşul sağlanmadı; inceleme gerekiyor."
    if operator == "state":
        return "Durum farkı sayısal değildir; ölçüm modülü bu durumu inceleme adayı olarak bildirdi."
    if isinstance(operator, str) and operator not in _KNOWN_THRESHOLD_OPERATORS and operator:
        return "Fark hesaplanmadı; operatör sunum katmanında tanımlı değil."
    if value is None or not limits:
        return "Fark hesaplanamadı; gerekli eklem kanıtı yetersiz."
    if not _is_number(value):
        return "Fark hesaplanamadı; ölçüm sayısal değil."
    label = unit_label(unit)
    numeric = float(value)
    if operator == "max":
        difference = numeric - float(limits[0])
        if difference > 0:
            minimum_excess = None
            if isinstance(interval, list) and len(interval) == 2:
                minimum_excess = max(0.0, float(interval[0]) - float(limits[0]))
            suffix = "" if minimum_excess is None else f"; %95'e göre en az {minimum_excess:.2f} fazla"
            return f"Fark: üst sınırdan {difference:.2f} {label} fazla{suffix}."
        return f"Fark: üst sınırın {abs(difference):.2f} {label} altında."
    if operator == "abs_max":
        magnitude = abs(numeric)
        difference = magnitude - float(limits[0])
        if difference > 0:
            return f"Fark: mutlak değer üst sınırdan {difference:.2f} {label} fazla."
        return f"Fark: mutlak değer üst sınırın {abs(difference):.2f} {label} altında."
    if operator == "min":
        difference = float(limits[0]) - numeric
        if difference > 0:
            return f"Fark: alt sınırdan {difference:.2f} {label} eksik."
        return f"Fark: alt sınırın {abs(difference):.2f} {label} üstünde."
    if operator == "range" and len(limits) == 2:
        low, high = float(limits[0]), float(limits[1])
        if numeric < low:
            return f"Fark: alt sınırdan {low - numeric:.2f} {label} eksik."
        if numeric > high:
            return f"Fark: üst sınırdan {numeric - high:.2f} {label} fazla."
        return "Fark: ölçüm hedef aralığın içinde."
    return "Fark kaynak operatörü nedeniyle hesaplanamadı."


def _result_text(decision: dict[str, Any], expected: str) -> str:
    status = decision["decision_status"]
    points = decision.get("deduction_points")
    if status == "timeline_unverified":
        return _PHASE_APPROVAL_RESULT
    if status == "confirmed_source_bound_minor":
        return f"Sonuç: belirsizlik aralığı da sınır dışında; küçük hata adayı (-{float(points):g})."
    if status == "boundary_uncertain":
        return "Sonuç: %95 aralık karar sınırına temas ediyor; kesinti uygulanmadı."
    if status == "not_measurable":
        return "Sonuç: güvenilir 3B ölçüm yok; kesinti uygulanmadı."
    if status == "within_source_range":
        return f"Sonuç: {expected.removeprefix('Olması gereken: ')} Kesinti yok."
    if status not in _STATUS_PRESENTATION:
        return f"Sonuç: “{status}” karar durumu sunum katmanında tanımlı değil; karar raporuna bakın."
    return "Sonuç: bu karar puana uygulanmadı."


_UNIT_LABELS = {
    "bool": "koşul",
    "deg": "derece",
    "fist_width": "yumruk genişliği",
    "ratio": "oran",
    "sec": "saniye",
    "body_scale": "vücut ölçeği",
    "body_scale/sec": "vücut ölçeği/saniye",
    "torso_height": "gövde yüksekliği",
    "torso_length": "gövde boyu",
    "arm_length": "kol boyu",
    "leg_length": "bacak boyu",
    "shoulder_width": "omuz genişliği",
    "state": "durum",
    "reason_code": "neden kodu",
    "pose_distance": "poz uzaklığı",
    "event": "olay",
}


def unit_label(unit: str | None) -> str:
    """Turkish unit label; an undeclared or unknown unit is shown raw and marked."""
    if unit is None or unit == "":
        return "[birim bildirilmemiş]"
    return _UNIT_LABELS.get(str(unit), f"[ham birim: {unit}]")


def _unit_label(unit: str | None) -> str:
    return unit_label(unit)


def _declared_unit(unit: Any, source: str) -> tuple[str | None, str]:
    if isinstance(unit, str) and unit:
        return unit, source
    return None, "not_declared"


def _categorical_event(
    index: int,
    decision: dict[str, Any],
    movements: dict[str, dict[str, Any]],
    fps: float,
) -> dict[str, Any]:
    movement_id = decision.get("movement_id")
    movement = movements.get(movement_id) if movement_id is not None else None
    applied = decision.get("application_status") == "applied"
    inferred = decision.get("evidence_status") == "inferred"
    start, end = int(decision["start_frame"]), int(decision["end_frame"])
    anchor = (start + end) // 2
    return {
        "event_id": f"CAT-{index:03d}-{decision['observation_id']}",
        "event_kind": "categorical_observation",
        "categorical_kind": decision.get("event_kind"),
        "movement_id": movement_id,
        "movement_name": None if movement is None else movement["display_name"],
        "technique_id": None if movement is None else movement["techniques"][0]["technique_id"],
        "metric_id": None,
        "rule_id": decision["rule_id"],
        "description": decision["description"],
        "decision_status": "confirmed_source_bound_minor" if applied else "not_applicable",
        "display_status": "confirmed_deduction_candidate" if applied else "not_applicable",
        "display_label": (
            "Kesinti adayı" if applied else "Çıkarım — kesinti yok" if inferred else "Uygulanmadı"
        ),
        "display_color": "red" if applied else "blue",
        "application_status": decision.get("application_status"),
        "deduction_kind": decision.get("deduction_kind"),
        "deduction_points": decision.get("deduction_points"),
        "measurement": deepcopy(decision.get("measurement")),
        "evidence_window": {
            "start_frame": start,
            "anchor_frame": anchor,
            "end_frame": end,
            "start_time_sec": start / fps,
            "anchor_time_sec": anchor / fps,
            "end_time_sec": end / fps,
            "scope": "kinematic_screening" if inferred else "direct_observation",
        },
        "visual_geometry": {"kind": "movement_region", "joint_indices": [], "unit": "event"},
        "source": {"source_ref": decision.get("source_ref")},
        "reason": decision.get("reason"),
        "review": {"status": "pending", "allowed_values": ["confirmed", "rejected", "uncertain"]},
    }


# --------------------------------------------------------------------------------------
# Presentation derivation. Everything below turns what a report already declares (rule
# family, metric id, unit, threshold, movement contract, evaluation window, required
# landmarks) into words and a drawable joint set. It never measures or decides.
# --------------------------------------------------------------------------------------

_FAMILY_TR = {
    "head_orientation": "Baş yönelimi",
    "torso_pelvis": "Gövde ve pelvis",
    "stance_feet": "Duruş ve ayaklar",
    "lower_body": "Alt beden",
    "kick": "Tekme",
    "upper_body": "Üst beden ve kol",
    "wrist_hand": "El bileği ve el",
    "phase_structure": "Faz yapısı",
    "transition": "Geçiş",
    "fixation": "Sabitlenme",
    "sequence": "Hareket sırası ve kontrat",
    "pipeline_limits": "Ölçüm hattı sınırı",
    "balance": "Denge",
    "hand": "El",
    "gaze": "Bakış",
    "timing": "Zamanlama",
    "stance": "Duruş",
    "technique": "Teknik",
}

# Word glossary for metric-id tokens (a vocabulary, not a per-rule table). Unknown
# tokens stay verbatim and are reported in ``untranslated_tokens``.
_TOKEN_TR = {
    "head": "baş", "face": "yüz", "torso": "gövde", "pelvis": "pelvis", "shoulder": "omuz",
    "shoulders": "omuzlar", "elbow": "dirsek", "wrist": "el bileği", "forearm": "ön kol",
    "hand": "el", "fist": "yumruk", "arm": "kol", "knee": "diz", "foot": "ayak", "feet": "ayaklar",
    "heel": "topuk", "toe": "ayak ucu", "ankle": "ayak bileği", "hip": "kalça", "leg": "bacak",
    "stance": "duruş", "step": "adım", "body": "vücut", "pose": "poz", "kick": "tekme",
    "target": "hedef", "yaw": "yön (yaw)", "roll": "yana yatma (roll)", "pitch": "eğim (pitch)",
    "error": "hatası", "difference": "farkı", "deviation": "sapması", "dispersion": "dağılımı",
    "drift": "kayması", "stability": "kararlılığı", "alignment": "hizası", "flexion": "bükülmesi",
    "extension": "açılması", "lean": "eğilmesi", "bend": "bükülmesi", "fixation": "sabitlenme",
    "orientation": "yönelim", "direction": "yön", "position": "konum", "lateral": "yanal",
    "height": "yükseklik", "depth": "derinlik", "distance": "mesafesi", "width": "genişlik",
    "length": "uzunluk", "axis": "ekseni", "active": "aktif", "reaction": "reaksiyon",
    "chamber": "hazırlık", "front": "ön", "rear": "arka", "back": "arka", "lead": "ön",
    "support": "destek", "pivot": "pivot", "moving": "hareket eden", "side": "taraf",
    "match": "uyumu", "settle": "oturma", "offset": "zaman farkı", "late": "geç",
    "correction": "düzeltme", "premature": "erken", "next": "sonraki", "movement": "hareket",
    "transition": "geçişi", "wrong": "yanlış", "stable": "kalıcı", "state": "durumu",
    "radial": "radyal", "ulnar": "ulnar", "proxy": "(yaklaşık)", "p95": "(p95)",
    "expected": "beklenen", "type": "türü", "phase": "faz", "order": "sırası",
    "coverage": "kapsamı", "contract": "kontratı", "resolved": "çözümlendi", "exists": "mevcut",
    "segment": "segment", "timeline": "zaman çizelgesi", "technique": "teknik",
    "required": "gerekli", "turn": "dönüş", "completion": "tamamlanma", "sign": "işaret",
    "geometry": "geometri", "quality": "kalitesi", "unmeasurable": "ölçülemezlik",
    "reason": "nedeni", "relative": "göreli", "progression": "ilerleme", "crossing": "çaprazlama",
    "margin": "payı", "landing": "iniş", "displacement": "yer değiştirme", "slip": "kayma",
    "change": "değişimi", "translation": "ötelenmesi", "over": "üzerinde", "valgus": "valgus",
    "varus": "varus", "plane": "düzlem", "identity": "kimliği", "swap": "yer değiştirme",
    "left": "sol", "right": "sağ", "observed": "gözlendi", "preparation": "hazırlık",
    "precedes": "önce gelir", "execution": "uygulama", "reaches": "ulaşır", "by": "itibarıyla",
    "final": "son", "conformance": "uygunluğu", "evidence": "kanıt", "rotation": "dönüş",
    "landmark": "nokta", "elevation": "yükselmesi", "abduction": "abdüksiyonu", "flare": "açılması",
    "overshoot": "aşması", "angle": "açısı", "retraction": "geri çekme", "restoration": "geri kazanımı",
    "hold": "tutuş", "jitter": "titreşimi", "weight": "ağırlık", "transfer": "aktarımı",
    "closure": "kapanması", "mismatch": "uyumsuzluğu", "consistent": "tutarlı", "with": "ile",
    "declared": "bildirilen", "excess": "fazla", "settled": "oturmuş", "recovery": "toparlanma",
    "balance": "denge", "actual": "gerçek", "eye": "göz", "gaze": "bakışı", "centre": "merkezi",
    "center": "merkezi", "mass": "kütle", "pressure": "basınç", "distribution": "dağılımı",
    "ground": "zemin", "force": "kuvveti", "impact": "darbe", "power": "gücü", "muscle": "kas",
    "tension": "gerginliği", "shape": "şekli", "observability": "gözlenebilirliği",
    "kicking": "tekme", "executing": "uygulayan", "during": "sırasında", "after": "sonrası",
    "asymmetry": "asimetrisi", "range": "aralık", "adjustment": "ayarlaması",
}
_DROP_TOKENS = {"deg", "sec", "of", "to", "or", "and", "is"}
_BIGRAMS = {
    ("post", "fixation"): "sabitlenme sonrası",
    ("post", "landing"): "iniş sonrası",
    ("forward", "backward"): "öne-arkaya",
    ("inter", "foot"): "ayaklar arası",
    ("lower", "body"): "alt beden",
    ("upper", "body"): "üst beden",
    ("radial", "ulnar"): "radyal-ulnar",
    ("body", "ratio"): "",
    ("leg", "ratio"): "(bacak oranı)",
    ("shoulder", "ratio"): "(omuz oranı)",
    ("torso", "ratio"): "(gövde oranı)",
    ("body", "height"): "vücut yüksekliği",
}
_BODY_PART_TR = {"baş", "yüz", "gövde", "pelvis", "omuz", "omuzlar", "dirsek", "el bileği", "ön kol",
                 "el", "yumruk", "kol", "diz", "ayak", "ayaklar", "topuk", "kalça", "bacak", "duruş"}
_DATIVE = {"gövde": "gövdeye", "omuzlar": "omuzlara", "omuz": "omuza", "pelvis": "pelvise",
           "baş": "başa", "kalça": "kalçaya", "ayak": "ayağa", "duruş": "duruşa"}

_ROLE_TR = {
    "active_arm": "aktif kol",
    "reaction_arm": "reaksiyon kolu",
    "lead_leg": "ön bacak",
    "rear_leg": "arka bacak",
    "support_leg": "destek bacağı",
    "kicking_leg": "tekme bacağı",
}
_SIDE_TR = {"left": "sol", "right": "sağ"}
_WINDOW_TR = {
    "fixation_anchor": "sabitlenme çevresi",
    "post_fixation_hold": "sabitlenme sonrası tutuş",
    "segment_start_to_fixation": "hareket başından sabitlenmeye",
    "post_fixation_tail_to_segment_end": "sabitlenmeden segment sonuna",
    "preparation_anchor": "hazırlık çevresi",
    "fixation_window": "sabitlenme",
    "fixation_to_segment_end": "sabitlenmeden segment sonuna",
    "preparation_to_fixation": "hazırlıktan sabitlenmeye",
    "phase_specific_or_fixation": "faza özgü veya sabitlenme",
}

_ARM_LANDMARK_PARTS = {"shoulder", "elbow", "wrist", "hand"}
_LEG_LANDMARK_PARTS = {"hip", "knee", "ankle", "foot"}
_ARM_TOKENS = {"hand", "fist", "wrist", "elbow", "forearm", "arm"}
_TOKEN_PARTS = {
    "head": {"face", "head"}, "face": {"face", "head"}, "eye": {"face", "head"},
    "torso": {"shoulder", "hip"}, "shoulder": {"shoulder"}, "shoulders": {"shoulder"},
    "pelvis": {"hip"}, "hip": {"hip"},
    "elbow": {"elbow"}, "wrist": {"wrist", "hand"}, "forearm": {"elbow", "wrist"},
    "hand": {"wrist", "hand"}, "fist": {"wrist", "hand"}, "arm": {"shoulder", "elbow", "wrist"},
    "knee": {"knee"}, "leg": {"hip", "knee", "ankle"}, "foot": {"ankle", "foot"},
    "feet": {"ankle", "foot"}, "heel": {"foot"}, "toe": {"foot"}, "ankle": {"ankle"},
    "stance": {"ankle", "foot"}, "step": {"ankle", "foot"}, "kick": {"hip", "knee", "ankle"},
}
_ANGLE_TOKENS = {"flexion", "extension", "alignment", "angle", "deviation"}
_ORIENTATION_TOKENS = {"yaw", "roll", "pitch", "lean", "bend", "direction", "orientation", "deviation"}
_STABILITY_TOKENS = {"stability", "dispersion", "drift", "jitter", "slip", "translation",
                     "correction", "settle", "offset", "change"}
_PAIR_TOKENS = {"distance", "width", "length", "margin", "crossing", "inter"}
_STATE_UNITS = {"bool", "state", "reason_code"}
_HAND_ANCHORS = ("index_mcp", "middle_mcp", "pinky_mcp")
_FACE_ANCHORS = (36, 39, 42, 45, 30, 8)
_MAX_DRAWABLE_POINTS = 10


def _tokens(metric_id: str) -> list[str]:
    return [token for token in str(metric_id).lower().split("_") if token]


def _derived_title(metric_id: str) -> tuple[str, list[str]]:
    """Turkish title from metric-id words; returns (title, untranslated tokens)."""
    tokens = _tokens(metric_id)
    if tokens and tokens[-1] == "ratio":
        tokens = tokens[:-1] + ["ratio"]
    words: list[str] = []
    untranslated: list[str] = []
    index = 0
    while index < len(tokens):
        pair = tuple(tokens[index:index + 2])
        if len(pair) == 2 and pair in _BIGRAMS:
            if _BIGRAMS[pair]:
                words.append(_BIGRAMS[pair])
            index += 2
            continue
        token = tokens[index]
        index += 1
        if token in _DROP_TOKENS or (token == "ratio" and index == len(tokens)):
            continue
        if token in _TOKEN_TR:
            words.append(_TOKEN_TR[token])
        else:
            words.append(token)
            untranslated.append(token)
    if "göreli" in words:
        split = words.index("göreli")
        head, reference = words[:split], words[split + 1:]
        if reference:
            noun = " ".join(reference)
            words = head + [f"({_DATIVE.get(noun, noun)} göre)"]
        else:
            words = head + ["(göreli)"]
    merged: list[str] = []
    for word in words:
        if merged and merged[-1] in _BODY_PART_TR and word in _BODY_PART_TR:
            merged[-1] = f"{merged[-1]}–{word}"
        else:
            merged.append(word)
    text = " ".join(merged).strip() or str(metric_id)
    return _sentence_case(text), untranslated


def _sentence_case(text: str) -> str:
    if not text:
        return text
    first = {"i": "İ", "ı": "I"}.get(text[0], text[0].upper())
    return first + text[1:]


def _catalog_text(rule: dict[str, Any], metric_id: str) -> tuple[str | None, str | None]:
    """Profile catalog label/description, ignoring the machine-generated placeholders."""
    label = rule.get("label")
    description = rule.get("description")
    machine_label = metric_id.replace("_", " ")
    usable_label = label if isinstance(label, str) and label.strip() and label.strip() != machine_label else None
    usable_description = (
        description
        if isinstance(description, str) and description.strip() and metric_id not in description
        else None
    )
    return usable_label, usable_description


def _technique_side(movement: dict[str, Any]) -> str:
    side = movement["techniques"][0].get("side")
    return side if side in {"left", "right"} else "left"


def _metric_sides(tokens: list[str], contract: dict[str, Any], technique_side: str) -> dict[str, Any]:
    """Which arm/leg the metric names, resolved through the movement contract.

    A role word (active, reaction, front/lead, rear/back, support/pivot, kick) selects
    the contract side. An arm-part word without a role uses the active arm, the same
    convention the WholeBody evaluator applies. No role word means both sides.
    """
    token_set = set(tokens)
    active = contract.get("active_arm") if contract.get("active_arm") in _SIDE_TR else technique_side
    reaction = contract.get("chamber_or_reaction_arm")
    arm_side = arm_role = None
    if "reaction" in token_set and reaction in _SIDE_TR:
        arm_side, arm_role = reaction, "reaction_arm"
    elif "active" in token_set or "executing" in token_set:
        arm_side, arm_role = active, "active_arm"
    elif token_set & _ARM_TOKENS:
        arm_side, arm_role = active, "active_arm"
    leg_side = leg_role = None
    if token_set & {"front", "lead"} and contract.get("lead_leg") in _SIDE_TR:
        leg_side, leg_role = contract["lead_leg"], "lead_leg"
    elif token_set & {"rear", "back"} and contract.get("rear_or_support_leg") in _SIDE_TR:
        leg_side, leg_role = contract["rear_or_support_leg"], "rear_leg"
    elif token_set & {"support", "pivot"}:
        side = contract.get("expected_pivot_or_support_foot") or contract.get("rear_or_support_leg")
        if side in _SIDE_TR:
            leg_side, leg_role = side, "support_leg"
    elif token_set & {"kick", "kicking"} or ("chamber" in token_set and "knee" in token_set):
        if contract.get("kicking_leg") in _SIDE_TR:
            leg_side, leg_role = contract["kicking_leg"], "kicking_leg"
    return {"arm_side": arm_side, "arm_role": arm_role, "leg_side": leg_side, "leg_role": leg_role}


def _landmark_info(index: int) -> tuple[str, str | None, int | None]:
    """(part, side, local index) for a WholeBody-133 landmark."""
    if 0 <= index < len(COCO_BODY_JOINT_NAMES):
        name = COCO_BODY_JOINT_NAMES[index]
        if index <= 4:
            return "head", None, None
        side, _, part = name.partition("_")
        return part, side, None
    for name, foot_index in COCO_FOOT_JOINTS.items():
        if foot_index == index:
            return "foot", name.split("_", 1)[0], None
    if 23 <= index < COCO_LEFT_HAND_OFFSET:
        return "face", None, index - 23
    if COCO_LEFT_HAND_OFFSET <= index < COCO_RIGHT_HAND_OFFSET:
        return "hand", "left", index - COCO_LEFT_HAND_OFFSET
    if COCO_RIGHT_HAND_OFFSET <= index < COCO_WHOLEBODY_KEYPOINTS:
        return "hand", "right", index - COCO_RIGHT_HAND_OFFSET
    return "unknown", None, None


_HAND_LOCAL_NAMES = {local: name for name, local in COCO_HAND_LOCAL_JOINTS.items()}


def _landmark_index_from_name(name: str) -> int | None:
    if name in COCO_BODY_JOINTS:
        return COCO_BODY_JOINTS[name]
    if name in COCO_FOOT_JOINTS:
        return COCO_FOOT_JOINTS[name]
    side, _, local = name.partition("_")
    if side in _SIDE_TR and local in COCO_HAND_LOCAL_JOINTS and local != "wrist":
        return coco_hand_joint(side, local)
    return None


def _evidence_landmarks(evidence: dict[str, Any]) -> list[int]:
    counts = evidence.get("required_joint_sample_counts")
    if not isinstance(counts, dict):
        return []
    indices = [_landmark_index_from_name(str(name)) for name in counts]
    return [index for index in indices if index is not None]


def _side_allows(part: str, side: str | None, sides: dict[str, Any]) -> bool:
    if side is None:
        return True
    if part in _ARM_LANDMARK_PARTS and sides.get("arm_side") is not None:
        return side == sides["arm_side"]
    if part in _LEG_LANDMARK_PARTS and part != "hip" and sides.get("leg_side") is not None:
        return side == sides["leg_side"]
    if part == "hip" and sides.get("leg_side") is not None and sides.get("arm_side") is None:
        return side == sides["leg_side"]
    return True


def _reduce_landmarks(indices: list[int]) -> list[int]:
    """Collapse 21-point hands and 68-point faces to a few anchor points."""
    result: list[int] = []
    hands: dict[str, list[int]] = {}
    face: list[int] = []
    for index in indices:
        part, side, local = _landmark_info(index)
        if part == "hand":
            hands.setdefault(str(side), []).append(int(local))
        elif part == "face":
            face.append(int(local))
        elif index not in result:
            result.append(index)
    for side, locals_ in hands.items():
        anchors = [COCO_HAND_LOCAL_JOINTS[name] for name in _HAND_ANCHORS if COCO_HAND_LOCAL_JOINTS[name] in locals_]
        chosen = anchors or sorted(set(locals_))[:1]
        offset = COCO_LEFT_HAND_OFFSET if side == "left" else COCO_RIGHT_HAND_OFFSET
        result.extend(offset + local for local in chosen)
    if face:
        anchors = [local for local in _FACE_ANCHORS if local in face]
        if not anchors:
            ordered = sorted(set(face))
            anchors = sorted({ordered[0], ordered[len(ordered) // 2], ordered[-1]})
        result.extend(23 + local for local in anchors)
    return result[:_MAX_DRAWABLE_POINTS]


def _side_index(name: str, side: str) -> int | None:
    if side not in _SIDE_TR:
        return None
    if name in _HAND_LOCAL_NAMES.values() and name != "wrist":
        return coco_hand_joint(side, name)
    key = f"{side}_{name}"
    return COCO_BODY_JOINTS.get(key, COCO_FOOT_JOINTS.get(key))


def _explicit_visual_geometry(metric_id: str, movement: dict[str, Any]) -> dict[str, Any] | None:
    """Hand-checked drawings that are better than the derived ones; None when absent."""
    side = _technique_side(movement)
    if metric_id == "executing_elbow_deg":
        return {
            "kind": "joint_angle",
            "joint_indices": [
                COCO_BODY_JOINTS[f"{side}_shoulder"],
                COCO_BODY_JOINTS[f"{side}_elbow"],
                COCO_BODY_JOINTS[f"{side}_wrist"],
            ],
            "vertex_joint_index": COCO_BODY_JOINTS[f"{side}_elbow"],
            "label": "DIRSEK",
        }
    if metric_id == "back_foot_yaw_to_stance_direction_deg":
        front_side = "left" if movement["stance"].startswith("left_") else "right"
        back_side = "right" if front_side == "left" else "left"
        return {
            "kind": "foot_direction_angle",
            "joint_indices": [
                COCO_BODY_JOINTS[f"{front_side}_ankle"],
                COCO_BODY_JOINTS[f"{back_side}_ankle"],
                COCO_FOOT_JOINTS[f"{back_side}_heel"],
                COCO_FOOT_JOINTS[f"{back_side}_big_toe"],
                COCO_FOOT_JOINTS[f"{back_side}_small_toe"],
            ],
            "front_side": front_side,
            "back_side": back_side,
        }
    if metric_id == "arae_fist_to_thigh_fist_ratio":
        return {
            "kind": "fist_to_thigh_distance",
            "joint_indices": [
                coco_hand_joint(side, "index_mcp"),
                coco_hand_joint(side, "middle_mcp"),
                coco_hand_joint(side, "ring_mcp"),
                coco_hand_joint(side, "pinky_mcp"),
                COCO_BODY_JOINTS[f"{side}_hip"],
                COCO_BODY_JOINTS[f"{side}_knee"],
            ],
            "side": side,
        }
    if metric_id == "fist_closure_ratio":
        return {
            "kind": "hand_shape",
            "joint_indices": [coco_hand_joint(side, name) for name in (
                "wrist",
                "thumb_tip",
                "index_mcp",
                "index_tip",
                "middle_mcp",
                "middle_tip",
                "ring_mcp",
                "ring_tip",
                "pinky_mcp",
                "pinky_tip",
            )],
            "side": side,
        }
    if metric_id == "wrist_forearm_alignment_deg":
        return {
            "kind": "joint_angle",
            "joint_indices": [
                COCO_BODY_JOINTS[f"{side}_elbow"],
                COCO_BODY_JOINTS[f"{side}_wrist"],
                coco_hand_joint(side, "middle_mcp"),
            ],
            "vertex_joint_index": COCO_BODY_JOINTS[f"{side}_wrist"],
            "label": "EL BILEGI",
        }
    if metric_id == "head_torso_yaw_mismatch_deg":
        return {
            "kind": "head_torso_direction",
            "joint_indices": [
                *range(23 + 36, 23 + 48),
                COCO_BODY_JOINTS["left_shoulder"],
                COCO_BODY_JOINTS["right_shoulder"],
            ],
        }
    if metric_id == "hand_foot_settle_difference_sec":
        stance_side = "left" if movement["stance"].startswith("left_") else "right"
        return {
            "kind": "highlight_joints",
            "joint_indices": [
                COCO_BODY_JOINTS[f"{side}_wrist"],
                COCO_BODY_JOINTS[f"{stance_side}_ankle"],
            ],
        }
    if metric_id == "fixation_wrist_jitter_ratio":
        return {
            "kind": "highlight_joints",
            "joint_indices": [COCO_BODY_JOINTS[f"{side}_wrist"]],
        }
    if metric_id == "pelvis_weight_transfer_ratio":
        return {
            "kind": "highlight_joints",
            "joint_indices": [COCO_BODY_JOINTS["left_hip"], COCO_BODY_JOINTS["right_hip"]],
        }
    return None


def _visual_geometry(metric_id: str, movement: dict[str, Any]) -> dict[str, Any]:
    """Backwards-compatible entry point: explicit drawing or an explicit undefined marker."""
    explicit = _explicit_visual_geometry(metric_id, movement)
    if explicit is not None:
        explicit["derivation"] = "explicit_metric_map"
        return explicit
    return _undefined_visual("no explicit drawing and no landmark data supplied")


def _undefined_visual(reason: str) -> dict[str, Any]:
    return {
        "kind": "undefined",
        "joint_indices": [],
        "visual_status": VISUAL_GEOMETRY_UNDEFINED,
        "undefined_reason": reason,
        "derivation": "undefined",
    }


def _group(indices: list[int], available: set[int]) -> list[int]:
    return [index for index in indices if index in available]


def _derive_visual(
    *,
    metric_id: str,
    unit: str | None,
    family: str | None,
    landmarks: list[int],
    sides: dict[str, Any],
    title: str,
    landmark_source: str,
) -> dict[str, Any]:
    tokens = _tokens(metric_id)
    token_set = set(tokens)
    if family == "pipeline_limits":
        return {**_undefined_visual("pipeline limit: quantity is not observable from WholeBody pose"),
                "label": title}
    valid = [int(index) for index in landmarks if isinstance(index, int) and 0 <= index < COCO_WHOLEBODY_KEYPOINTS]
    if not valid:
        return {**_undefined_visual(f"no landmarks declared in {landmark_source}"), "label": title}
    sided = [index for index in valid if _side_allows(*_landmark_info(index)[:2], sides)]
    focus_parts = set().union(*(_TOKEN_PARTS[token] for token in tokens if token in _TOKEN_PARTS)) if any(
        token in _TOKEN_PARTS for token in tokens
    ) else set()
    focused = [index for index in sided if _landmark_info(index)[0] in focus_parts] if focus_parts else []
    candidates = focused or sided
    drawable = _reduce_landmarks(candidates)
    available = set(valid)
    base = {
        "derivation": "derived_from_rule_landmarks",
        "landmark_source": landmark_source,
        "sides": {key: value for key, value in sides.items() if value is not None},
        "label": title,
    }
    if not drawable:
        return {**_undefined_visual("declared landmarks do not include the side named by the movement contract"),
                **{k: v for k, v in base.items() if k != "derivation"}}
    if unit in _STATE_UNITS:
        return {**base, "kind": "boolean_state", "joint_indices": drawable,
                "geometry_note": "Durum kuralı: açı veya mesafe çizilmez; yalnız ilgili eklemler işaretlenir."}
    arm, leg = sides.get("arm_side"), sides.get("leg_side")
    if unit == "deg" and token_set & _ANGLE_TOKENS:
        for joint, side, triplet in (
            ("knee", leg, ("hip", "knee", "ankle")),
            ("elbow", arm, ("shoulder", "elbow", "wrist")),
            ("wrist", arm, ("elbow", "wrist", "middle_mcp")),
        ):
            if joint in token_set and side is not None:
                indices = [_side_index(name, side) for name in triplet]
                if all(index is not None and index in available for index in indices):
                    return {**base, "kind": "joint_angle", "joint_indices": indices,
                            "vertex_joint_index": indices[1]}
    if unit == "deg" and token_set & _ORIENTATION_TOKENS:
        segments = _orientation_segments(tokens, sides, available)
        if segments:
            joints = sorted({index for segment in segments for part in ("from", "to") for index in segment[part]})
            return {**base, "kind": "segment_direction", "segments": segments,
                    "joint_indices": joints[: 2 * _MAX_DRAWABLE_POINTS]}
    if token_set & _STABILITY_TOKENS:
        return {**base, "kind": "point_stability", "joint_indices": drawable}
    if token_set & _PAIR_TOKENS:
        pair = _pair_groups(tokens, sides, available)
        if pair is not None:
            return {**base, "kind": "pair_distance", "pair": {"from": pair[0], "to": pair[1]},
                    "joint_indices": sorted(set(pair[0]) | set(pair[1]))}
    return {**base, "kind": "target_position", "joint_indices": drawable}


def _orientation_segments(tokens: list[str], sides: dict[str, Any], available: set[int]) -> list[dict[str, Any]]:
    token_set = set(tokens)
    segments: list[dict[str, Any]] = []

    def add(name: str, first: list[int], second: list[int]) -> None:
        first, second = _group(first, available), _group(second, available)
        if first and second and not any(item["name"] == name for item in segments):
            segments.append({"name": name, "from": first, "to": second})

    body = COCO_BODY_JOINTS
    vertical = bool(token_set & {"lean", "pitch", "bend"})
    arm_sides = [sides["arm_side"]] if sides.get("arm_side") else ["left", "right"]
    leg_sides = [sides["leg_side"]] if sides.get("leg_side") else ["left", "right"]
    for token in tokens:
        if token in {"head", "face"}:
            add("eye_line", list(range(23 + 36, 23 + 42)), list(range(23 + 42, 23 + 48)))
            if not any(item["name"] == "eye_line" for item in segments):
                add("eye_line", [body["left_eye"]], [body["right_eye"]])
        elif token == "torso":
            if vertical:
                add("torso_axis", [body["left_hip"], body["right_hip"]],
                    [body["left_shoulder"], body["right_shoulder"]])
            else:
                add("shoulder_line", [body["left_shoulder"]], [body["right_shoulder"]])
        elif token in {"shoulder", "shoulders"}:
            add("shoulder_line", [body["left_shoulder"]], [body["right_shoulder"]])
        elif token in {"pelvis", "hip"}:
            add("hip_line", [body["left_hip"]], [body["right_hip"]])
        elif token in {"stance", "step"}:
            add("stance_axis", [body["left_ankle"]], [body["right_ankle"]])
        elif token in {"foot", "feet", "toe", "heel"}:
            for side in leg_sides:
                add(f"{side}_foot_axis", [COCO_FOOT_JOINTS[f"{side}_heel"]],
                    [COCO_FOOT_JOINTS[f"{side}_big_toe"], COCO_FOOT_JOINTS[f"{side}_small_toe"]])
        elif token == "knee":
            for side in leg_sides:
                add(f"{side}_thigh", [body[f"{side}_hip"]], [body[f"{side}_knee"]])
        elif token in _ARM_TOKENS:
            for side in arm_sides:
                if token in {"fist", "hand"}:
                    add(f"{side}_hand_axis", [body[f"{side}_wrist"]],
                        [coco_hand_joint(side, name) for name in _HAND_ANCHORS])
                add(f"{side}_forearm", [body[f"{side}_elbow"]], [body[f"{side}_wrist"]])
    return segments


def _pair_groups(tokens: list[str], sides: dict[str, Any], available: set[int]) -> tuple[list[int], list[int]] | None:
    token_set = set(tokens)
    body = COCO_BODY_JOINTS
    if token_set & {"stance", "step", "feet", "inter", "crossing"}:
        first = _group([body["left_ankle"], COCO_FOOT_JOINTS["left_heel"]], available)
        second = _group([body["right_ankle"], COCO_FOOT_JOINTS["right_heel"]], available)
        return (first, second) if first and second else None
    named = [token for token in tokens if token in _TOKEN_PARTS]
    if len(named) >= 2 and named[0] != named[1]:
        groups = []
        for token in named[:2]:
            parts = _TOKEN_PARTS[token]
            group = [index for index in sorted(available)
                     if _landmark_info(index)[0] in parts and _side_allows(*_landmark_info(index)[:2], sides)]
            groups.append(_reduce_landmarks(group))
        if groups[0] and groups[1]:
            return groups[0], groups[1]
    return None


def _window_phrase(window: dict[str, Any] | None) -> str:
    if not window:
        return ""
    scope = str(window.get("scope") or "")
    names = [part for part in scope.split("+") if part]
    label = " + ".join(_WINDOW_TR.get(name, name) for name in names) or "kanıt penceresi"
    return f"{label} penceresinde (kare {window['start_frame']}–{window['end_frame']})"


def _side_phrase(sides: dict[str, Any]) -> str:
    parts = []
    for side_key, role_key in (("arm_side", "arm_role"), ("leg_side", "leg_role")):
        if sides.get(side_key) and sides.get(role_key):
            parts.append(f"{_ROLE_TR[sides[role_key]]}: {_SIDE_TR[sides[side_key]]}")
    return ", ".join(parts)


def _description_sentence(
    *,
    family: str | None,
    title: str,
    sides: dict[str, Any],
    window: dict[str, Any] | None,
    expected: str,
) -> str:
    family_text = _FAMILY_TR.get(str(family), str(family)) if family else "Teşhis"
    side_text = _side_phrase(sides)
    window_text = _window_phrase(window)
    where = f", {window_text} ölçülür" if window_text else ""
    who = f" ({side_text})" if side_text else ""
    expectation = expected.replace("Olması gereken: ", "").replace("Olması gereken: ", "").rstrip(".")
    return f"{family_text} kuralı — {title}{who}{where}; beklenen: {expectation}."


def _technical_presentation(
    *,
    candidate: dict[str, Any],
    rule: dict[str, Any],
    contract: dict[str, Any],
    movement: dict[str, Any],
    unit: str | None,
    screening_rule: dict[str, Any],
    window: dict[str, Any] | None,
) -> dict[str, Any]:
    metric_id = str(candidate.get("metric_id"))
    family = candidate.get("rule_family") or rule.get("rule_family")
    tokens = _tokens(metric_id)
    sides = _metric_sides(tokens, contract, _technique_side(movement))
    catalog_label, catalog_description = _catalog_text(rule, metric_id)
    untranslated: list[str] = []
    if metric_id in _EXPLICIT_DIAGNOSTIC_TEXT:
        title, correction = _EXPLICIT_DIAGNOSTIC_TEXT[metric_id]
        title_source = "explicit_metric_map"
    elif catalog_label:
        title, title_source = catalog_label, "profile_catalog"
        correction = _derived_correction(title)
    else:
        title, untranslated = _derived_title(metric_id)
        title_source = "derived_from_metric_tokens"
        correction = _derived_correction(title)
    expected = format_expected_text(
        screening_rule.get("operator"), _screening_limits(screening_rule), unit,
        expected_boolean=screening_rule.get("expected_boolean"),
        expected_state=screening_rule.get("expected_state"), raw_value=screening_rule.get("raw_value"),
    )
    if catalog_description:
        description, description_source = catalog_description, "profile_catalog"
    else:
        description = _description_sentence(family=family, title=title, sides=sides, window=window, expected=expected)
        description_source = "derived_from_family_window_and_threshold"
    explicit = _explicit_visual_geometry(metric_id, movement)
    if explicit is not None:
        visual = {**explicit, "derivation": "explicit_metric_map"}
        visual.setdefault("label", title)
    else:
        landmarks = rule.get("required_landmarks") if isinstance(rule.get("required_landmarks"), list) else []
        visual = _derive_visual(
            metric_id=metric_id, unit=unit, family=family, landmarks=landmarks, sides=sides, title=title,
            landmark_source="rule_inventory.required_landmarks",
        )
    sources = {"title": title_source, "description": description_source,
               "visual": visual.get("derivation")}
    if untranslated:
        sources["untranslated_tokens"] = untranslated
    return {"title": title, "description": description, "correction": correction,
            "visual": visual, "sources": sources}


def _wholebody_presentation(candidate: dict[str, Any], movement: dict[str, Any], unit: str | None) -> dict[str, Any]:
    metric_id = str(candidate.get("metric_id"))
    tokens = _tokens(metric_id)
    sides = _metric_sides(tokens, {}, _technique_side(movement))
    untranslated: list[str] = []
    if metric_id in _EXPLICIT_DIAGNOSTIC_TEXT:
        title, correction = _EXPLICIT_DIAGNOSTIC_TEXT[metric_id]
        title_source = "explicit_metric_map"
    else:
        title, untranslated = _derived_title(metric_id)
        title_source = "derived_from_metric_tokens"
        correction = _derived_correction(title)
    rule = candidate.get("screening_rule") or {}
    expected = format_expected_text(rule.get("operator"), _screening_limits(rule), unit,
                                    expected_boolean=rule.get("expected_boolean"))
    evidence = candidate.get("measurement_evidence") if isinstance(candidate.get("measurement_evidence"), dict) else {}
    window = None
    if evidence.get("start_frame") is not None and evidence.get("end_frame") is not None:
        window = {"start_frame": evidence["start_frame"], "end_frame": evidence["end_frame"],
                  "scope": evidence.get("scope")}
    description = _description_sentence(family=candidate.get("family"), title=title, sides=sides,
                                        window=window, expected=expected)
    explicit = _explicit_visual_geometry(metric_id, movement)
    if explicit is not None:
        visual = {**explicit, "derivation": "explicit_metric_map"}
        visual.setdefault("label", title)
    else:
        visual = _derive_visual(
            metric_id=metric_id, unit=unit, family=candidate.get("family"),
            landmarks=_evidence_landmarks(evidence), sides=sides, title=title,
            landmark_source="measurement_evidence.required_joint_sample_counts",
        )
    sources = {"title": title_source, "description": "derived_from_family_window_and_threshold",
               "visual": visual.get("derivation")}
    if untranslated:
        sources["untranslated_tokens"] = untranslated
    return {"title": title, "description": description, "correction": correction,
            "visual": visual, "sources": sources}


def describe_rule_presentation(
    rule: dict[str, Any],
    movement: dict[str, Any],
    contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Presentation a rule-inventory entry would get, without a candidate (coverage audits)."""
    threshold = rule.get("threshold") or {}
    if threshold.get("operator") in _KNOWN_THRESHOLD_OPERATORS:
        screening_rule = _technical_screening_rule(threshold, [])
    elif isinstance(rule.get("expected_boolean"), bool):
        screening_rule = {"operator": "bool_true" if rule["expected_boolean"] else "bool_false",
                          "expected_boolean": rule["expected_boolean"]}
    else:
        screening_rule = {"operator": "state" if rule.get("unit") in _STATE_UNITS else None}
    unit, _ = _declared_unit(rule.get("unit"), "rule")
    return _technical_presentation(
        candidate={"metric_id": rule.get("metric_id"), "rule_family": rule.get("rule_family")},
        rule=rule, contract=contract or {}, movement=movement, unit=unit,
        screening_rule=screening_rule, window=None,
    )


def _frame(value: Any, fallback: int) -> int:
    return fallback if value is None else int(value)
