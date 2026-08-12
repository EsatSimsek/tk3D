from __future__ import annotations

from copy import deepcopy
from typing import Any

from src.data_structures import COCO_BODY_JOINTS, COCO_FOOT_JOINTS, coco_hand_joint
from src.poomsae_scoring.contracts import (
    ScoringContractError,
    validate_movement_timeline,
    validate_poomsae_spec,
)


_STATUS_PRESENTATION = {
    "confirmed_source_bound_minor": ("confirmed_deduction_candidate", "red", "Kesinti adayı"),
    "boundary_uncertain": ("uncertain_no_deduction", "amber", "Sınır belirsiz"),
    "not_measurable": ("not_measurable_no_deduction", "gray", "Ölçülemedi"),
    "within_source_range": ("within_source_range", "green", "Kaynak aralığında"),
    "not_applicable": ("not_applicable", "blue", "Uygulanamaz"),
}


def build_decision_evidence_events(
    accuracy_decisions: dict[str, Any],
    poomsae_spec: dict[str, Any],
    movement_timeline: dict[str, Any],
) -> dict[str, Any]:
    """Convert scoring decisions into immutable, camera-renderable evidence events."""
    spec = validate_poomsae_spec(poomsae_spec)
    timeline = validate_movement_timeline(movement_timeline, spec)
    if accuracy_decisions.get("status") != "source_bound_accuracy_decisions":
        raise ScoringContractError("decision evidence requires a source-bound Accuracy report")
    if accuracy_decisions.get("timeline_id") != timeline["timeline_id"]:
        raise ScoringContractError("Accuracy decisions and MovementTimeline must match")

    movements = {item["movement_id"]: item for item in spec["movements"]}
    segments = {item["movement_id"]: item for item in timeline["segments"]}
    events: list[dict[str, Any]] = []
    numeric = accuracy_decisions.get("numeric_decisions")
    if not isinstance(numeric, list):
        raise ScoringContractError("numeric_decisions must be a list")
    for index, decision in enumerate(numeric, start=1):
        movement_id = decision.get("movement_id")
        if movement_id not in movements or movement_id not in segments:
            raise ScoringContractError(f"numeric decision references an unobserved movement: {movement_id}")
        status = decision.get("decision_status")
        if status not in _STATUS_PRESENTATION:
            raise ScoringContractError(f"unsupported decision evidence status: {status}")
        event = _numeric_event(
            index=index,
            decision=decision,
            movement=movements[movement_id],
            segment=segments[movement_id],
            fps=timeline["fps"],
        )
        events.append(event)

    categorical = accuracy_decisions.get("categorical_decisions")
    if not isinstance(categorical, list):
        raise ScoringContractError("categorical_decisions must be a list")
    for index, decision in enumerate(categorical, start=1):
        events.append(_categorical_event(index, decision, movements, timeline["fps"]))

    counts = {key: sum(event["decision_status"] == key for event in events) for key in _STATUS_PRESENTATION}
    return {
        "schema_version": 1,
        "status": "decision_evidence_events",
        "timeline_id": timeline["timeline_id"],
        "frame_count": timeline["frame_count"],
        "fps": timeline["fps"],
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
            "not_measurable_count": counts["not_measurable"],
            "within_source_range_count": counts["within_source_range"],
            "categorical_event_count": len(categorical),
        },
        "events": events,
    }


def _numeric_event(
    *,
    index: int,
    decision: dict[str, Any],
    movement: dict[str, Any],
    segment: dict[str, Any],
    fps: float,
) -> dict[str, Any]:
    status = decision["decision_status"]
    display_status, color, label = _STATUS_PRESENTATION[status]
    evidence = decision.get("measurement_evidence")
    if not isinstance(evidence, dict):
        evidence = {}
    anchor = _frame(evidence.get("anchor_frame"), segment["end_frame"])
    start = _frame(evidence.get("start_frame"), anchor)
    end = _frame(evidence.get("end_frame"), anchor)
    if not segment["start_frame"] <= start <= anchor <= end <= segment["end_frame"]:
        raise ScoringContractError(f"decision evidence window is outside {movement['movement_id']}")
    visual = _visual_geometry(decision["metric_id"], movement)
    return {
        "event_id": f"NUM-{index:03d}-{movement['movement_id']}-{decision['metric_id']}",
        "event_kind": "numeric_source_bound_decision",
        "movement_id": movement["movement_id"],
        "movement_name": movement["display_name"],
        "technique_id": movement["techniques"][0]["technique_id"],
        "metric_id": decision["metric_id"],
        "rule_id": decision["rule_id"],
        "description": decision["description"],
        "decision_status": status,
        "display_status": display_status,
        "display_label": label,
        "display_color": color,
        "application_status": decision.get("application_status"),
        "deduction_kind": decision.get("deduction_kind"),
        "deduction_points": decision.get("deduction_points"),
        "measurement": {
            "value": decision.get("measured_value"),
            "unit": visual["unit"],
            "uncertainty_95": decision.get("effective_uncertainty_95"),
            "interval_95": deepcopy(decision.get("measurement_interval_95")),
            "rule_operator": decision.get("rule_operator"),
            "rule_limits": deepcopy(decision.get("rule_limits")),
            "boundary_guard": decision.get("boundary_guard"),
            "sample_count": decision.get("sample_count"),
        },
        "user_explanation": _user_explanation(decision, visual),
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


def _user_explanation(decision: dict[str, Any], visual: dict[str, Any]) -> dict[str, str]:
    measurement = decision.get("measured_value")
    interval = decision.get("measurement_interval_95")
    limits = decision.get("rule_limits") or []
    operator = decision.get("rule_operator")
    unit = visual["unit"]
    metric_id = decision["metric_id"]
    title, correction = _metric_explanation(metric_id)
    expected = _expected_text(operator, limits, unit)
    measured = "Olcum yapilamadi." if measurement is None else f"{float(measurement):.2f} {_unit_label(unit)}"
    interval_text = (
        "Belirsizlik araligi hesaplanamadi."
        if not isinstance(interval, list) or len(interval) != 2
        else f"%95 olasi aralik: {float(interval[0]):.2f} - {float(interval[1]):.2f} {_unit_label(unit)}"
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
    if metric_id == "back_foot_yaw_to_stance_direction_deg":
        return (
            "ARKA AYAK ACISI",
            "Arka ayagi durus yonune yaklastir; ayak acisini izin verilen sinira indir.",
        )
    if metric_id == "arae_fist_to_thigh_fist_ratio":
        return (
            "ARAE-MAKKI YUMRUK-UYLUK MESAFESI",
            "Blok bitisinde yumrugu uyluga, hedeflenen yumruk-genisligi araligina getir.",
        )
    if metric_id == "executing_elbow_deg":
        return (
            "UYGULAYAN DIRSEK ACISI",
            "Omuz-dirsek-bilek hizasini hedef aci araligina getir.",
        )
    return metric_id.upper(), "Teknik konumu kaynak araligina getir."


def _expected_text(operator: Any, limits: list[Any], unit: str) -> str:
    label = _unit_label(unit)
    if operator == "max" and limits:
        return f"Olmasi gereken: en fazla {float(limits[0]):.2f} {label}."
    if operator == "range" and len(limits) == 2:
        return f"Olmasi gereken: {float(limits[0]):.2f} - {float(limits[1]):.2f} {label}."
    return "Olmasi gereken kaynakta sayisal olarak tanimli degil."


def _comparison_text(
    value: Any,
    interval: Any,
    operator: Any,
    limits: list[Any],
    unit: str,
) -> str:
    if value is None or not limits:
        return "Fark hesaplanamadi; gerekli eklem kaniti yetersiz."
    label = _unit_label(unit)
    numeric = float(value)
    if operator == "max":
        difference = numeric - float(limits[0])
        if difference > 0:
            minimum_excess = None
            if isinstance(interval, list) and len(interval) == 2:
                minimum_excess = max(0.0, float(interval[0]) - float(limits[0]))
            suffix = "" if minimum_excess is None else f"; %95'e gore en az {minimum_excess:.2f} fazla"
            return f"Fark: ust sinirdan {difference:.2f} {label} fazla{suffix}."
        return f"Fark: ust sinirin {abs(difference):.2f} {label} altinda."
    if operator == "range" and len(limits) == 2:
        low, high = float(limits[0]), float(limits[1])
        if numeric < low:
            return f"Fark: alt sinirdan {low - numeric:.2f} {label} eksik."
        if numeric > high:
            return f"Fark: ust sinirdan {numeric - high:.2f} {label} fazla."
        return "Fark: olcum hedef araligin icinde."
    return "Fark kaynak operatoru nedeniyle hesaplanamadi."


def _result_text(decision: dict[str, Any], expected: str) -> str:
    status = decision["decision_status"]
    points = decision.get("deduction_points")
    if status == "confirmed_source_bound_minor":
        return f"Sonuc: belirsizlik araligi da sinir disinda; kucuk hata adayi (-{float(points):g})."
    if status == "boundary_uncertain":
        return "Sonuc: %95 aralik karar sinirina temas ediyor; kesinti uygulanmadi."
    if status == "not_measurable":
        return "Sonuc: guvenilir 3B olcum yok; kesinti uygulanmadi."
    if status == "within_source_range":
        return f"Sonuc: {expected.removeprefix('Olmasi gereken: ')} Kesinti yok."
    return "Sonuc: bu karar puana uygulanmadi."


def _unit_label(unit: str) -> str:
    return {"deg": "derece", "fist_width": "yumruk genisligi"}.get(unit, unit)


def _categorical_event(
    index: int,
    decision: dict[str, Any],
    movements: dict[str, dict[str, Any]],
    fps: float,
) -> dict[str, Any]:
    movement_id = decision.get("movement_id")
    movement = movements.get(movement_id) if movement_id is not None else None
    applied = decision.get("application_status") == "applied"
    start, end = int(decision["start_frame"]), int(decision["end_frame"])
    anchor = (start + end) // 2
    return {
        "event_id": f"CAT-{index:03d}-{decision['observation_id']}",
        "event_kind": "categorical_observation",
        "movement_id": movement_id,
        "movement_name": None if movement is None else movement["display_name"],
        "technique_id": None if movement is None else movement["techniques"][0]["technique_id"],
        "metric_id": None,
        "rule_id": decision["rule_id"],
        "description": decision["description"],
        "decision_status": "confirmed_source_bound_minor" if applied else "not_applicable",
        "display_status": "confirmed_deduction_candidate" if applied else "not_applicable",
        "display_label": "Kesinti adayı" if applied else "Uygulanmadı",
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
            "scope": "direct_observation",
        },
        "visual_geometry": {"kind": "movement_region", "joint_indices": [], "unit": "event"},
        "source": {"source_ref": decision.get("source_ref")},
        "reason": decision.get("reason"),
        "review": {"status": "pending", "allowed_values": ["confirmed", "rejected", "uncertain"]},
    }


def _visual_geometry(metric_id: str, movement: dict[str, Any]) -> dict[str, Any]:
    side = movement["techniques"][0]["side"]
    if side not in {"left", "right"}:
        side = "left"
    if metric_id == "executing_elbow_deg":
        return {
            "kind": "joint_angle",
            "joint_indices": [
                COCO_BODY_JOINTS[f"{side}_shoulder"],
                COCO_BODY_JOINTS[f"{side}_elbow"],
                COCO_BODY_JOINTS[f"{side}_wrist"],
            ],
            "vertex_joint_index": COCO_BODY_JOINTS[f"{side}_elbow"],
            "unit": "deg",
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
            "unit": "deg",
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
            "unit": "fist_width",
        }
    return {"kind": "highlight_joints", "joint_indices": [], "unit": "unknown"}


def _frame(value: Any, fallback: int) -> int:
    return fallback if value is None else int(value)
