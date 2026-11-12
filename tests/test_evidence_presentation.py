"""Presentation-only contracts for decision evidence events, cards, video labels and HTML rows.

Every fixture here is synthetic; none of these values is a measurement or a threshold.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from src.data_structures import COCO_BODY_JOINTS, COCO_LEFT_HAND_OFFSET, COCO_RIGHT_HAND_OFFSET
from src.poomsae_scoring import (
    ScoringContractError,
    build_decision_evidence_events,
    build_run_history,
    load_movement_timeline,
    load_poomsae_spec,
)
from src.poomsae_scoring import decision_evidence as de
from src.poomsae_scoring import error_video
from src.poomsae_scoring.review_report import _decision_event_row, _wholebody_metric_cell
from src.poomsae_scoring.video_cards import _card


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "config" / "scoring" / "poomsae" / "taegeuk_1_jang_v0_draft.yaml"
TIMELINE_PATH = ROOT / "config" / "scoring" / "timelines" / "poomsae1_zed2i_rgbd_rerun_20260802_draft.yaml"
ARM = [5, 6, 7, 8, 9, 10, 11, 12]
LEFT_HAND = list(range(COCO_LEFT_HAND_OFFSET, COCO_RIGHT_HAND_OFFSET))
RIGHT_HAND = list(range(COCO_RIGHT_HAND_OFFSET, 133))
EYES = list(range(23 + 36, 23 + 48))
LEGS = [5, 6, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
# Same shape as the technical report's movement_contracts entry for M01 (left ap-seogi,
# left arae-makki): active arm left, reaction arm right, lead leg left, rear leg right.
M01_CONTRACT = {
    "movement_id": "M01", "active_arm": "left", "chamber_or_reaction_arm": "right",
    "lead_leg": "left", "rear_or_support_leg": "right", "kicking_leg": None,
    "expected_moving_foot": "left", "expected_pivot_or_support_foot": "right",
}


@pytest.fixture(scope="module")
def spec_timeline():
    spec = load_poomsae_spec(SPEC_PATH)
    return spec, load_movement_timeline(TIMELINE_PATH, spec)


def _decisions(timeline: dict, numeric: list | None = None) -> dict:
    return {
        "status": "source_bound_accuracy_decisions",
        "timeline_id": timeline["timeline_id"],
        "numeric_decisions": numeric or [],
        "categorical_decisions": [],
    }


def _rule(metric_id: str, family: str, unit: str, landmarks: list[int], **extra) -> dict:
    rule = {
        "rule_id": f"SYN-{metric_id}",
        "metric_id": metric_id,
        "rule_family": family,
        # Placeholders exactly as the v3 profile emits them today.
        "label": metric_id.replace("_", " "),
        "description": f"Temporary WholeBody-133 technical-accuracy diagnostic for {metric_id}.",
        "unit": unit,
        "required_landmarks": landmarks,
        "evaluation_phase": "phase_specific_or_fixation",
    }
    rule.update(extra)
    return rule


def _candidate(metric_id: str, family: str, unit: str | None, value, threshold=None, **extra) -> dict:
    candidate = {
        "movement_id": "M01",
        "rule_id": f"SYN-{metric_id}",
        "metric_id": metric_id,
        "criterion_id": f"technical_accuracy.{family}.{metric_id}",
        "rule_family": family,
        "phase_or_window": "phase_specific_or_fixation",
        "value": value,
        "unit": unit,
        "threshold": threshold,
        "uncertainty": None,
        "decision_status": "review_candidate_not_deduction",
        "score_effect": None,
        "deduction_points": None,
        "evidence": {
            "start_frame": 140, "end_frame": 229, "fixation_frame": 218,
            "window": {"name": "fixation_anchor", "start_frame": 213, "end_frame": 223},
        },
    }
    candidate.update(extra)
    return candidate


def _technical(timeline: dict, candidates: list[dict], rules: list[dict]) -> dict:
    return {
        "status": "technical_accuracy_diagnostics_only",
        "movement_timeline_id": timeline["timeline_id"],
        "candidate_events": candidates,
        "rule_inventory": rules,
        "movement_contracts": [M01_CONTRACT],
    }


def _events(spec_timeline, candidates, rules, numeric=None, wholebody=None) -> list[dict]:
    spec, timeline = spec_timeline
    return build_decision_evidence_events(
        _decisions(timeline, numeric), spec, timeline, wholebody,
        technical_accuracy_diagnostics=_technical(timeline, candidates, rules),
    )["events"]


# 1. Derived titles, descriptions, joints and geometry kinds -----------------------------

def test_unmapped_rules_get_turkish_title_description_and_contract_side_joints(spec_timeline) -> None:
    rules = [
        _rule("reaction_hand_fixation_stability", "wrist_hand", "arm_length", ARM + LEFT_HAND + RIGHT_HAND),
        _rule("rear_knee_flexion_deg", "lower_body", "deg", LEGS),
        _rule("head_target_yaw_error_deg", "head_orientation", "deg", [5, 6, 11, 12, *EYES]),
        _rule("moving_foot_side_match", "stance_feet", "bool", LEGS, expected_boolean=True),
    ]
    candidates = [
        _candidate("reaction_hand_fixation_stability", "wrist_hand", "arm_length", 0.08,
                   {"operator": "max", "value": 0.05}),
        _candidate("rear_knee_flexion_deg", "lower_body", "deg", 150.0, {"operator": "range", "value": [155, 180]}),
        _candidate("head_target_yaw_error_deg", "head_orientation", "deg", 40.0, {"operator": "max", "value": 25.0}),
        _candidate("moving_foot_side_match", "stance_feet", "bool", False, None, expected_boolean=True),
    ]
    hand, knee, head, foot = _events(spec_timeline, candidates, rules)

    for event in (hand, knee, head, foot):
        title = event["user_explanation"]["title"]
        assert title and title != event["metric_id"].upper() and event["metric_id"] not in title
        assert event["description"] != title and title in event["description"]
        assert event["visual_geometry"]["joint_indices"]
        assert event["presentation_source"]["title"] == "derived_from_metric_tokens"
    assert hand["user_explanation"]["title"] == "Reaksiyon el sabitlenme kararlılığı"
    assert "reaksiyon kolu: sağ" in hand["description"] and "kare 213–223" in hand["description"]
    # Reaction arm of M01 is the right arm: no left-side point, hand reduced to anchors.
    joints = hand["visual_geometry"]["joint_indices"]
    assert hand["visual_geometry"]["kind"] == "point_stability"
    assert not set(joints) & ({5, 7, 9} | set(LEFT_HAND))
    assert len([index for index in joints if index in RIGHT_HAND]) <= 3
    assert knee["visual_geometry"]["kind"] == "joint_angle"
    assert knee["visual_geometry"]["joint_indices"] == [
        COCO_BODY_JOINTS["right_hip"], COCO_BODY_JOINTS["right_knee"], COCO_BODY_JOINTS["right_ankle"],
    ]
    assert knee["user_explanation"]["title"] == "Arka diz bükülmesi"
    assert head["visual_geometry"]["kind"] == "segment_direction"
    assert [segment["name"] for segment in head["visual_geometry"]["segments"]] == ["eye_line"]
    assert foot["visual_geometry"]["kind"] == "boolean_state"
    # Correction text is metric-specific, never one shared sentence.
    assert len({event["user_explanation"]["correction"] for event in (hand, knee, head, foot)}) == 4


def test_explicit_mapping_and_profile_catalog_text_take_precedence(spec_timeline) -> None:
    rules = [
        _rule("wrist_forearm_alignment_deg", "wrist_hand", "deg", ARM),
        _rule("brand_new_metric_deg", "upper_body", "deg", ARM, label="Yeni kol ölçütü",
              description="Profil kataloğundaki açıklama."),
    ]
    candidates = [
        _candidate("wrist_forearm_alignment_deg", "wrist_hand", "deg", 30.0, {"operator": "max", "value": 20.0}),
        _candidate("brand_new_metric_deg", "upper_body", "deg", 30.0, {"operator": "max", "value": 20.0}),
    ]
    wrist, new = _events(spec_timeline, candidates, rules)
    assert wrist["user_explanation"]["title"] == "EL BİLEĞİ–ÖN KOL HİZASI"
    assert wrist["visual_geometry"]["derivation"] == "explicit_metric_map"
    assert wrist["visual_geometry"]["label"] == "EL BILEGI"
    assert new["user_explanation"]["title"] == "Yeni kol ölçütü"
    assert new["description"] == "Profil kataloğundaki açıklama."
    assert new["presentation_source"]["title"] == "profile_catalog"


def test_undrawable_rules_are_marked_instead_of_silently_empty(spec_timeline) -> None:
    rules = [
        _rule("movement_phase_order", "sequence", "bool", [], expected_boolean=True),
        _rule("centre_of_pressure", "pipeline_limits", "ratio", [11, 12]),
    ]
    candidates = [
        _candidate("movement_phase_order", "sequence", "bool", False, None, expected_boolean=True),
        _candidate("centre_of_pressure", "pipeline_limits", "ratio", 2.0, {"operator": "max", "value": 1.0}),
    ]
    for event in _events(spec_timeline, candidates, rules):
        visual = event["visual_geometry"]
        assert visual["joint_indices"] == []
        assert visual["kind"] == "undefined"
        assert visual["visual_status"] == "görsel geometri tanımsız"
        assert visual["undefined_reason"]
        assert event["user_explanation"]["title"] != event["metric_id"].upper()


# 2. Units ----------------------------------------------------------------------------

@pytest.mark.parametrize(
    "unit,label",
    [("arm_length", "kol boyu"), ("leg_length", "bacak boyu"), ("shoulder_width", "omuz genişliği"),
     ("torso_length", "gövde boyu"), ("body_scale/sec", "vücut ölçeği/saniye"), ("state", "durum"),
     ("reason_code", "neden kodu"), ("deg", "derece"), ("fist_width", "yumruk genişliği")],
)
def test_every_report_unit_has_a_turkish_label(unit: str, label: str) -> None:
    assert de.unit_label(unit) == label


def test_unknown_or_missing_units_are_marked_raw_never_unknown_or_ratio(spec_timeline) -> None:
    assert de.unit_label("newton_metre") == "[ham birim: newton_metre]"
    assert de.unit_label(None) == "[birim bildirilmemiş]"
    spec, timeline = spec_timeline
    wholebody = {
        "status": "wholebody_diagnostics_only",
        "movement_timeline_id": timeline["timeline_id"],
        "candidate_events": [{
            "movement_id": "M01", "metric_id": "new_wholebody_metric", "family": "balance",
            "value": 1.2, "screening_rule": {"operator": "max", "value": 1.0},
            "measurement_evidence": {"scope": "fixation_window", "start_frame": 213,
                                     "anchor_frame": 218, "end_frame": 223,
                                     "required_joint_sample_counts": {"left_hip": 11, "right_hip": 11}},
        }],
        "movements": [{"movement_id": "M01", "metrics": [
            {"metric_id": "new_source_bound_metric", "unit": "torso_length"},
        ]}],
    }
    numeric = [{
        "movement_id": "M01", "rule_id": "SYN-NUM", "metric_id": "new_source_bound_metric",
        "decision_status": "within_source_range", "deduction_points": None,
        "application_status": "not_applied", "description": "Sentetik kaynak kuralı.",
        "measured_value": 0.3, "rule_operator": "max", "rule_limits": [0.5],
        "measurement_evidence": {"scope": "fixation_window", "start_frame": 213, "anchor_frame": 218,
                                 "end_frame": 223, "required_joint_sample_counts": {"left_wrist": 11}},
    }]
    report = build_decision_evidence_events(_decisions(timeline, numeric), spec, timeline, wholebody)
    number, diagnostic = report["events"]
    assert number["measurement"]["unit"] == "torso_length"
    assert number["measurement"]["unit_source"] == "wholebody_measurement"
    assert number["user_explanation"]["measured"] == "0.30 gövde boyu"
    assert number["visual_geometry"]["joint_indices"] == [COCO_BODY_JOINTS["left_wrist"]]
    assert diagnostic["measurement"]["unit"] is None
    assert diagnostic["measurement"]["unit_source"] == "not_declared"
    assert "[birim bildirilmemiş]" in diagnostic["user_explanation"]["measured"]
    assert "oran" not in diagnostic["user_explanation"]["measured"]
    assert diagnostic["visual_geometry"]["kind"] == "target_position"


# 3. HTML rows ------------------------------------------------------------------------

def test_html_row_uses_event_texts_for_booleans_and_abs_max(spec_timeline) -> None:
    rules = [
        _rule("premature_next_movement_transition", "phase_structure", "bool", ARM, expected_boolean=False),
        _rule("head_roll_relative_shoulders_deg", "head_orientation", "deg", [5, 6, *EYES]),
    ]
    candidates = [
        _candidate("premature_next_movement_transition", "phase_structure", "bool", True, None,
                   expected_boolean=False),
        _candidate("head_roll_relative_shoulders_deg", "head_orientation", "deg", -30.0,
                   {"operator": "abs_max", "value": 15.0}),
    ]
    boolean, absolute = _events(spec_timeline, candidates, rules)
    boolean_row = _decision_event_row(boolean)
    assert "3B Evet — hata koşulu mevcut." in boolean_row
    assert "beklenen: hata koşulu bulunmamalı (beklenen: Hayır)" in boolean_row
    assert "1.00" not in boolean_row
    absolute_row = _decision_event_row(absolute)
    assert "beklenen: mutlak değer en fazla 15.00 derece" in absolute_row
    assert "-30.00 derece" in absolute_row
    assert "Faz onayı yok" in absolute_row  # phase note shown next to, not instead of, the correction
    assert "Kurala özel düzeltme metni" in absolute_row


def test_wholebody_metric_cell_formats_booleans_and_abs_max() -> None:
    cell = _wholebody_metric_cell({
        "metric_id": "synthetic_flag", "value": True, "unit": "bool", "screening_status": "review_candidate",
        "screening_rule": {"operator": "bool_false", "expected_boolean": False}, "measurement_evidence": {},
    })
    assert "Evet" in cell and "1.00" not in cell and "Hayır" in cell
    cell = _wholebody_metric_cell({
        "metric_id": "synthetic_roll", "value": -14.0, "unit": "deg", "screening_status": "review_candidate",
        "screening_rule": {"operator": "abs_max", "value": 12.0}, "measurement_evidence": {},
    })
    assert "mutlak değer en fazla 12.00 derece" in cell and "-14.00 derece" in cell


# 4. Video overlay label ----------------------------------------------------------------

def test_joint_angle_overlay_label_comes_from_the_event(monkeypatch) -> None:
    import numpy as np

    captured: list[str] = []
    monkeypatch.setattr(error_video, "_label", lambda image, point, text, color, **_: captured.append(text))
    event = {
        "event_id": "E", "user_explanation": {"title": "El bileği radyal sapma"},
        "visual_geometry": {"kind": "joint_angle", "joint_indices": [7, 9, 100], "vertex_joint_index": 9},
    }
    observations = {("cam", 0, index): (10.0 * index, 20.0, 0.9) for index in (7, 9, 100)}
    error_video._draw_geometry(np.zeros((540, 960, 3), np.uint8), event, "cam", 0, observations,
                               (1.0, 1.0), (0, 0, 255), 1)
    assert captured == ["[1] EL BILEGI RADYAL SAPMA"]
    assert all("DIRSEK" not in text for text in captured)


def test_new_geometry_kinds_draw_on_the_camera_pane() -> None:
    import numpy as np

    observations = {("cam", 0, index): (100.0 + 40 * index, 200.0, 0.9) for index in (5, 6, 11, 12)}
    for visual in (
        {"kind": "segment_direction", "joint_indices": [5, 6], "segments": [{"name": "s", "from": [5], "to": [6]}]},
        {"kind": "pair_distance", "joint_indices": [11, 12], "pair": {"from": [11], "to": [12]}},
        {"kind": "point_stability", "joint_indices": [5, 6]},
    ):
        pane = np.zeros((540, 960, 3), np.uint8)
        error_video._draw_geometry(pane, {"event_id": "E", "visual_geometry": visual}, "cam", 0,
                                   observations, (1.0, 1.0), (0, 0, 255), 1)
        assert pane.any(), visual["kind"]


# 5. Phase-approval note ----------------------------------------------------------------

def test_phase_note_is_a_separate_field_and_card_renders_it(spec_timeline) -> None:
    rules = [_rule("torso_target_yaw_error_deg", "torso_pelvis", "deg", [5, 6, 11, 12])]
    event = _events(spec_timeline, [
        _candidate("torso_target_yaw_error_deg", "torso_pelvis", "deg", 30.0, {"operator": "max", "value": 25.0}),
    ], rules)[0]
    explanation = event["user_explanation"]
    assert explanation["phase_approval_note"] == de.PHASE_APPROVAL_NOTE
    assert explanation["correction"] != de.PHASE_APPROVAL_NOTE
    assert "Faz onayı" not in explanation["correction"]
    import json

    card = _card(json.dumps(event, sort_keys=True, ensure_ascii=False), 1, 600, 436)
    assert card.size == (600, 436)


# 6. Robustness and the candidate's own window -------------------------------------------

def test_candidate_window_and_anchor_come_from_the_candidate(spec_timeline) -> None:
    rules = [_rule("torso_target_yaw_error_deg", "torso_pelvis", "deg", [5, 6, 11, 12])]
    candidate = _candidate("torso_target_yaw_error_deg", "torso_pelvis", "deg", 30.0,
                           {"operator": "max", "value": 25.0})
    candidate["evidence"] = {"start_frame": 140, "end_frame": 229, "fixation_frame": 200,
                             "window": {"name": "fixation_anchor", "start_frame": 195, "end_frame": 205}}
    window = _events(spec_timeline, [candidate], rules)[0]["evidence_window"]
    # The timeline's M01 fixation anchor is 218; the candidate was measured around 200.
    assert (window["start_frame"], window["anchor_frame"], window["end_frame"]) == (195, 200, 205)
    assert window["anchor_source"] == "candidate_fixation_frame"


def test_legitimate_variants_become_marked_events_instead_of_failing(spec_timeline) -> None:
    rules = [
        _rule("premature_next_movement_transition", "phase_structure", "bool", ARM),
        _rule("future_operator_metric", "upper_body", "deg", ARM),
        _rule("head_late_turn_state", "head_orientation", "state", [5, 6, *EYES]),
        _rule("boolean_without_unit", "phase_structure", None, ARM),
    ]
    past_segment = _candidate("premature_next_movement_transition", "phase_structure", "bool", True, None,
                              expected_boolean=False)
    past_segment["evidence"] = {"window": {"name": "post_fixation_tail", "start_frame": 218, "end_frame": 260},
                                "fixation_frame": 218}
    candidates = [
        past_segment,
        _candidate("future_operator_metric", "upper_body", "deg", 3.0, {"operator": "outside", "value": [1.0, 2.0]}),
        _candidate("head_late_turn_state", "head_orientation", "state", "late", None,
                   expected_value_or_range="settled"),
        _candidate("boolean_without_unit", "phase_structure", None, True, None, expected_boolean=False),
    ]
    tail, operator, state, unitless = _events(spec_timeline, candidates, rules)
    assert tail["evidence_window"]["end_frame"] == 260
    assert "evidence_window_extends_beyond_movement_segment" in tail["presentation_warnings"]
    assert "unrecognized_threshold_operator:outside" in operator["presentation_warnings"]
    assert "outside" in operator["user_explanation"]["expected"]
    assert operator["measurement"]["rule_limits"] == [1.0, 2.0]
    assert state["user_explanation"]["measured"] == "Durum: late"
    assert "settled" in state["user_explanation"]["expected"]
    assert state["visual_geometry"]["kind"] == "boolean_state"
    assert "boolean_value_with_non_bool_unit:None" in unitless["presentation_warnings"]
    for event in (tail, operator, state, unitless):
        assert event["deduction_points"] is None
        assert event["decision_status"] == "diagnostic_review_candidate"


@pytest.mark.parametrize(
    "mutation,message",
    [
        (lambda c: c["evidence"].update(window={"start_frame": 230, "end_frame": 220}), "inverted"),
        (lambda c: c["evidence"].update(window={"start_frame": 5000, "end_frame": 5010}), "outside the timeline"),
        (lambda c: c.update(threshold={"operator": "range", "value": [30, 10]}), "inverted"),
        (lambda c: c.update(threshold={"value": 3.0}), "no operator"),
        (lambda c: c.update(threshold={"operator": "max", "value": float("nan")}), "finite"),
        (lambda c: c.update(threshold=None, value=4.0), "thresholdless"),
    ],
)
def test_malformed_candidates_still_fail_loudly(spec_timeline, mutation, message) -> None:
    rules = [_rule("torso_target_yaw_error_deg", "torso_pelvis", "deg", [5, 6, 11, 12])]
    candidate = _candidate("torso_target_yaw_error_deg", "torso_pelvis", "deg", 30.0,
                           {"operator": "max", "value": 25.0})
    mutation(candidate)
    with pytest.raises(ScoringContractError, match=message):
        _events(spec_timeline, [candidate], rules)


def test_unknown_numeric_decision_status_is_shown_raw_not_recoloured(spec_timeline) -> None:
    spec, timeline = spec_timeline
    numeric = [{
        "movement_id": "M01", "rule_id": "SYN-NEW-STATUS", "metric_id": "back_foot_yaw_to_stance_direction_deg",
        "decision_status": "confirmed_source_bound_major", "deduction_points": None,
        "application_status": "not_applied", "description": "Sentetik.", "measured_value": 40.0,
        "rule_operator": "max", "rule_limits": [30.0],
        "measurement_evidence": {"start_frame": 213, "anchor_frame": 218, "end_frame": 223},
    }]
    from timeline_review_support import reviewed_timeline

    reviewed = reviewed_timeline(deepcopy(timeline))
    event = build_decision_evidence_events(_decisions(reviewed, numeric), spec, reviewed)["events"][0]
    assert event["decision_status"] == "confirmed_source_bound_major"
    assert event["display_status"] == "unrecognized_decision_status"
    assert event["display_color"] == "gray"
    assert "unrecognized_decision_status:confirmed_source_bound_major" in event["presentation_warnings"]
    assert "tanımlı değil" in event["user_explanation"]["result"]


# 7. Smaller items ------------------------------------------------------------------------

def test_timeline_unverified_events_sort_before_blue_diagnostics() -> None:
    window = {"start_frame": 0, "anchor_frame": 0, "end_frame": 0}
    events = [
        {"event_id": "A-DIAG", "event_kind": "wholebody_diagnostic_review_candidate",
         "decision_status": "diagnostic_review_candidate", "evidence_window": window},
        {"event_id": "B-CAT", "event_kind": "categorical_observation",
         "decision_status": "timeline_unverified", "evidence_window": window},
    ]
    indexed, _, _ = error_video._index_events(events, 1)
    assert [event["event_id"] for event in indexed[0]] == ["B-CAT", "A-DIAG"]


def test_run_history_never_uses_an_unfinished_run_as_baseline() -> None:
    def summary(run_id: str) -> dict:
        return {
            "schema_version": 1, "workflow": "tk3d_source_bound_poomsae_scoring_v1",
            "status": "partial_sequence_decisions_generated", "mode": "score_verified_pose",
            "profile_id": "profile", "run": {"run_id": run_id, "root": f"C:/runs/{run_id}"},
            "coverage": {"selected_scope_id": "scope"}, "results": {"rule_scoring_ready": False},
            "bindings": {"pose": {"sha256": "a" * 64}},
        }

    report = build_run_history(summary("current"), [
        {"summary": summary("completed-old"), "modified_at_utc": "2026-01-01T00:00:00+00:00",
         "run_state_status": "completed"},
        {"summary": summary("failed-new"), "modified_at_utc": "2026-02-01T00:00:00+00:00",
         "run_state_status": "failed"},
        {"summary": summary("legacy"), "modified_at_utc": "2025-01-01T00:00:00+00:00"},
    ])
    assert report["summary"]["baseline_run_id"] == "completed-old"
    assert report["summary"]["skipped_unfinished_run_count"] == 1


def test_review_page_does_not_interpolate_event_ids_into_css_selectors() -> None:
    import inspect

    from src.poomsae_scoring import review_report

    source = inspect.getsource(review_report)
    assert '[data-review-event="${' not in source
    assert "item.dataset.reviewEvent === eventId" in source


def test_overlay_labels_stay_below_the_camera_title_strip_and_inside_the_frame() -> None:
    import numpy as np

    from src.poomsae_scoring import error_video

    image = np.zeros((200, 300, 3), dtype=np.uint8)
    error_video._label(image, (290, 5), "BAS EGIM (PITCH) GOVDEYE GORE", (255, 255, 255))
    rows, cols = np.nonzero(image.max(axis=2))
    assert rows.min() > error_video._TITLE_STRIP_HEIGHT
    assert cols.max() < image.shape[1]
