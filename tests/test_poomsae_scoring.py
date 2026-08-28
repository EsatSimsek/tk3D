from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np
import pytest

from src.data_structures import coco_hand_joint
from src.poomsae_scoring import (
    ScoringContractError,
    assess_accuracy_readiness,
    build_movement_evidence,
    build_categorical_diagnostics,
    build_partial_engineering_trial,
    build_review_html,
    build_presentation_diagnostics,
    build_source_bound_accuracy_decisions,
    build_technical_conformance,
    build_wholebody_diagnostics,
    derive_categorical_observations,
    evaluate_accuracy,
    inspect_source_intake,
    load_movement_timeline,
    load_engineering_profile,
    load_poomsae_spec,
    load_rule_pack,
    load_source_bound_accuracy_profile,
    load_wholebody_diagnostic_profile,
    overlay_state_for_frame,
    validate_movement_timeline,
    validate_engineering_profile,
    validate_poomsae_spec,
    validate_rule_pack,
    validate_source_intake,
    validate_source_bound_accuracy_profile,
    validate_source_bound_accuracy_result,
    validate_wholebody_diagnostic_profile,
)


ROOT = Path(__file__).resolve().parents[1]
RULE_PACK_PATH = ROOT / "config" / "scoring" / "rules" / "wt_recognized_2024-09-30.yaml"
DRAFT_SPEC_PATH = ROOT / "config" / "scoring" / "poomsae" / "taegeuk_1_jang_v0_draft.yaml"
DRAFT_TIMELINE_PATH = ROOT / "config" / "scoring" / "timelines" / "poomsae1_zed2i_rgbd_rerun_20260802_draft.yaml"
ENGINEERING_PROFILE_PATH = ROOT / "config" / "scoring" / "engineering" / "taegeuk_1_m01_m06_v1.yaml"
WHOLEBODY_PROFILE_PATH = ROOT / "config" / "scoring" / "engineering" / "taegeuk_1_wholebody_diagnostics_v2.yaml"
SOURCE_BOUND_ACCURACY_PATH = ROOT / "config" / "scoring" / "accuracy" / "taegeuk_1_source_bound_v1.yaml"


def _source() -> dict:
    return {
        "source_id": "test_source",
        "authority": "Test Authority",
        "title": "Test Poomsae Specification",
        "url": "https://example.test/poomsae",
        "effective_date": "2026-08-02",
        "accessed_at": "2026-08-02",
        "language": "English",
        "access": "public",
        "content_sha256": None,
        "sections": ["movement table"],
    }


def _movement(sequence_index: int) -> dict:
    return {
        "movement_id": f"M{sequence_index:02d}",
        "sequence_index": sequence_index,
        "display_name": f"Test movement {sequence_index}",
        "direction": "forward",
        "stance": "test_stance",
        "techniques": [{"technique_id": "test_technique", "side": "left"}],
        "phases": ["preparation", "execution", "apex"],
        "measurable_criteria": ["test.metric"],
        "source_refs": ["test_source#movement-table"],
    }


def _active_spec() -> dict:
    return validate_poomsae_spec(
        {
            "schema_version": 1,
            "poomsae_id": "test_poomsae",
            "version": "1.0.0",
            "status": "active",
            "display_name": "Test Poomsae",
            "rule_pack_id": "wt_recognized_2024-09-30",
            "sequence_status": "active",
            "source_documents": [_source()],
            "movements": [_movement(1), _movement(2)],
            "blocked_reasons": [],
        }
    )


def _complete_timeline() -> dict:
    return {
        "schema_version": 2,
        "timeline_id": "test_timeline",
        "poomsae_id": "test_poomsae",
        "poomsae_version": "1.0.0",
        "status": "complete",
        "label_source": "manual",
        "frame_index_space": "sample_index",
        "frame_count": 30,
        "fps": 30.0,
        "source_binding": {
            "session_id": "test_session",
            "run_id": "test_run",
            "pose_file": "outputs/test_session/runs/test_run/json/pose.json",
            "pose_file_sha256": "a" * 64,
        },
        "coverage": {
            "recording_scope": "complete_performance",
            "observed_movement_ids": ["M01", "M02"],
            "missing_movement_ids": [],
            "source_end_reason": None,
        },
        "segments": [
            {
                "sequence_index": 1,
                "movement_id": "M01",
                "start_frame": 0,
                "end_frame": 10,
                "anchors": {"apex": 5},
                "confidence": 1.0,
                "label_status": "confirmed",
            },
            {
                "sequence_index": 2,
                "movement_id": "M02",
                "start_frame": 11,
                "end_frame": 29,
                "anchors": {"apex": 20},
                "confidence": 1.0,
                "label_status": "confirmed",
            },
        ],
    }


def _event(
    event_id: str,
    deduction_kind: str,
    movement_id: str | None,
    start_frame: int,
    end_frame: int,
    **overrides,
) -> dict:
    payload = {
        "event_id": event_id,
        "deduplication_key": f"{movement_id}:{event_id}",
        "deduction_kind": deduction_kind,
        "movement_id": movement_id,
        "phase_id": None if deduction_kind == "restart" else "apex",
        "start_frame": start_frame,
        "end_frame": end_frame,
        "evidence_status": "observed",
        "decision_status": "confirmed_by_rule",
        "confidence": 0.95,
        "metric_id": "test.metric",
        "criterion_id": "test.metric",
        "description": "Synthetic rule-confirmed event",
        "rule_confirmation": {
            "rule_id": {
                "minor": "WT-2024-09-30-A16-1.2.2",
                "major": "WT-2024-09-30-A16-1.2.3",
                "restart": "WT-2024-09-30-A16-EXPLANATION-3",
            }[deduction_kind],
            "source_ref": {
                "minor": "wt_poomsae_rules_2024-09-30#Article16-1.2.2",
                "major": "wt_poomsae_rules_2024-09-30#Article16-1.2.3",
                "restart": "wt_poomsae_rules_2024-09-30#Article16-Explanation3",
            }[deduction_kind],
            "confirmation_method": "observed_restart" if deduction_kind == "restart" else "manual_rule_review",
            "review_record_id": f"review:{event_id}",
        },
    }
    payload.update(overrides)
    return payload


def test_project_rule_pack_uses_source_backed_accuracy_values() -> None:
    pack = load_rule_pack(RULE_PACK_PATH)

    assert pack["rule_pack_id"] == "wt_recognized_2024-09-30"
    assert pack["version"] == "1.1.0"
    assert pack["status"] == "active"
    assert pack["scoring"]["accuracy"]["initial_score"] == 4.0
    assert pack["scoring"]["accuracy"]["deductions"]["minor"]["amount"] == 0.1
    assert pack["scoring"]["accuracy"]["deductions"]["major"]["amount"] == 0.3
    assert pack["scoring"]["accuracy"]["deductions"]["restart"]["amount"] == 0.6
    final_deductions = pack["scoring"]["final_score_deductions"]
    assert final_deductions["time_violation"]["amount"] == 0.3
    assert final_deductions["boundary_crossing"]["amount"] == 0.3
    assert final_deductions["boundary_crossing"]["source_ambiguity"]


def test_taegeuk_1_draft_contains_source_transcribed_sequence_but_stays_fail_closed() -> None:
    spec = load_poomsae_spec(DRAFT_SPEC_PATH)

    assert spec["status"] == "draft"
    assert spec["sequence_status"] == "source_transcribed"
    assert len(spec["movements"]) == 18
    assert spec["movements"][13]["movement_id"] == "M14"
    assert len(spec["movements"][13]["techniques"]) == 2
    assert spec["blocked_reasons"]
    historical = next(
        source for source in spec["source_documents"] if source["source_id"] == "wtf_poomsae_scoring_guidelines_2014"
    )
    assert historical["content_sha256"] == ("46686f9a58c7d3dbc50bd9a1d284d9f6d2a56dbab6f70c365c39f029f8e01c8f")
    assert any("historical" in reason.lower() for reason in spec["blocked_reasons"])
    assert any(ref.startswith("wtf_poomsae_scoring_guidelines_2014#") for ref in spec["movements"][0]["source_refs"])
    assert all(movement["measurable_criteria"] for movement in spec["movements"])
    assert {
        "technique.ap_chagi.knee_extension",
        "technique.ap_chagi.rechamber",
        "technique.ap_chagi.support_foot_pivot",
        "timing.kick_landing_punch.sequence",
    }.issubset(spec["movements"][13]["measurable_criteria"])
    assert "audio.kihap.timing" in spec["movements"][17]["measurable_criteria"]

    empty_complete_timeline = {
        **_complete_timeline(),
        "poomsae_id": spec["poomsae_id"],
        "poomsae_version": spec["version"],
        "coverage": {
            "recording_scope": "complete_performance",
            "observed_movement_ids": [movement["movement_id"] for movement in spec["movements"]],
            "missing_movement_ids": [],
            "source_end_reason": None,
        },
        "segments": [
            {
                "sequence_index": movement["sequence_index"],
                "movement_id": movement["movement_id"],
                "start_frame": 2 * (movement["sequence_index"] - 1),
                "end_frame": 2 * movement["sequence_index"] - 1,
                "anchors": {movement["phases"][-1]: 2 * movement["sequence_index"] - 1},
                "confidence": 1.0,
                "label_status": "confirmed",
            }
            for movement in spec["movements"]
        ],
        "frame_count": 36,
    }
    with pytest.raises(ScoringContractError, match="complete MovementTimeline"):
        validate_movement_timeline(empty_complete_timeline, spec)


def test_project_timeline_draft_binds_exact_rgbd_pose_artifact() -> None:
    spec = load_poomsae_spec(DRAFT_SPEC_PATH)
    timeline = load_movement_timeline(DRAFT_TIMELINE_PATH, spec)

    assert timeline["status"] == "draft"
    assert timeline["frame_count"] == 741
    assert timeline["fps"] == 60.0
    assert len(timeline["segments"]) == 6
    assert timeline["segments"][0]["movement_id"] == "M01"
    assert timeline["segments"][-1]["movement_id"] == "M06"
    assert all(segment["label_status"] == "confirmed" for segment in timeline["segments"])
    assert timeline["coverage"]["recording_scope"] == "partial_sequence"
    assert timeline["coverage"]["missing_movement_ids"] == [f"M{index:02d}" for index in range(7, 19)]
    assert timeline["source_binding"]["pose_file_sha256"] == (
        "a3098284bed5cf83bea7e0d7488fe52d82f97b7d06977219b4c8f5eeccaf5947"
    )


def test_project_engineering_profile_is_explicitly_non_official_and_minor_only() -> None:
    profile = load_engineering_profile(ENGINEERING_PROFILE_PATH)

    assert profile["status"] == "provisional_engineering"
    assert profile["scope"]["movement_ids"] == [f"M{index:02d}" for index in range(1, 7)]
    assert profile["provenance"]["threshold_origin"] == "engineering_hypothesis_not_official_rule"
    assert profile["policy"]["automatic_major_detection"] is False
    assert {criterion["deduction_kind"] for criterion in profile["criteria"]} == {"minor"}


BOUND_POSE_PATH = (
    ROOT
    / "outputs"
    / "poomsae_1_zed2i_20260731_trimmed"
    / "runs"
    / "poomsae1-zed2i-rgbd-gated-ultra-rerun-20260802"
    / "json"
    / "vitpose_session_3d.json"
)


def test_project_accuracy_readiness_is_blocked_by_source_and_timeline_gaps(tmp_path: Path) -> None:
    """Readiness blockers must be scientific contract gaps, not unavailable local data."""
    spec = load_poomsae_spec(DRAFT_SPEC_PATH)
    timeline = load_movement_timeline(DRAFT_TIMELINE_PATH, spec)
    stand_in = tmp_path / "json" / "vitpose_session_3d.json"
    stand_in.parent.mkdir(parents=True)
    stand_in.write_bytes(b'{"session_id": "readiness-fixture"}')
    timeline["source_binding"]["pose_file"] = str(stand_in.relative_to(tmp_path).as_posix())
    timeline["source_binding"]["pose_file_sha256"] = hashlib.sha256(
        stand_in.read_bytes()
    ).hexdigest()

    report = assess_accuracy_readiness(
        load_rule_pack(RULE_PACK_PATH),
        spec,
        timeline,
        workspace_root=tmp_path,
    )

    assert report["status"] == "blocked"
    assert report["rule_scoring_ready"] is False
    assert report["pose_binding"]["status"] == "verified"
    blocker_codes = {blocker["code"] for blocker in report["blockers"]}
    assert blocker_codes == {
        "movement_sequence_not_active",
        "movement_timeline_not_complete",
        "partial_source_recording",
        "poomsae_spec_has_source_gaps",
        "poomsae_spec_not_active",
        "wholebody_diagnostics_not_provided",
    }


def test_project_accuracy_readiness_reports_missing_pose_separately(tmp_path: Path) -> None:
    spec = load_poomsae_spec(DRAFT_SPEC_PATH)
    report = assess_accuracy_readiness(
        load_rule_pack(RULE_PACK_PATH),
        spec,
        load_movement_timeline(DRAFT_TIMELINE_PATH, spec),
        workspace_root=tmp_path,
    )

    assert report["status"] == "blocked"
    assert report["rule_scoring_ready"] is False
    assert report["pose_binding"]["status"] == "pose_file_missing"
    blocker_codes = {blocker["code"] for blocker in report["blockers"]}
    assert blocker_codes == {
        "movement_sequence_not_active",
        "movement_timeline_not_complete",
        "partial_source_recording",
        "poomsae_spec_has_source_gaps",
        "poomsae_spec_not_active",
        "pose_file_missing",
        "wholebody_diagnostics_not_provided",
    }

@pytest.mark.skipif(
    not BOUND_POSE_PATH.is_file(),
    reason="the bound ZED pose artifact lives under gitignored outputs/ and is absent here",
)
def test_project_accuracy_readiness_verifies_the_real_bound_pose_when_present() -> None:
    """Runs only on a machine that actually holds the recording this timeline binds to.

    This is the check the previous version of the test above was really making: that the
    committed MovementTimeline still points at the real artifact and that its SHA-256
    still matches. It cannot run on a clean checkout, so it skips instead of failing.
    """
    spec = load_poomsae_spec(DRAFT_SPEC_PATH)
    report = assess_accuracy_readiness(
        load_rule_pack(RULE_PACK_PATH),
        spec,
        load_movement_timeline(DRAFT_TIMELINE_PATH, spec),
        workspace_root=ROOT,
    )

    assert report["pose_binding"]["status"] == "verified"
    assert report["pose_binding"]["actual_sha256"] == report["pose_binding"]["expected_sha256"]


def test_project_overlay_labels_partial_recording_without_inventing_missing_movements() -> None:
    spec = load_poomsae_spec(DRAFT_SPEC_PATH)
    timeline = load_movement_timeline(DRAFT_TIMELINE_PATH, spec)

    assert overlay_state_for_frame(spec, timeline, 139) is None
    first = overlay_state_for_frame(spec, timeline, 205)
    assert first is not None
    assert first["movement_id"] == "M01"
    assert first["phase_id"] == "execution"
    sixth = overlay_state_for_frame(spec, timeline, 728)
    assert sixth is not None
    assert sixth["movement_id"] == "M06"
    assert sixth["phase_id"] == "fixation"
    assert sixth["expected_movement_count"] == 18


def test_review_html_exposes_partial_coverage_without_inventing_a_score() -> None:
    spec = load_poomsae_spec(DRAFT_SPEC_PATH)
    timeline = load_movement_timeline(DRAFT_TIMELINE_PATH, spec)
    evidence = {
        "timeline": {"timeline_id": timeline["timeline_id"]},
        "observability": {"anchor_count": 24, "observed_anchor_ratio": 22 / 24},
        "segments": [
            {
                "movement_id": segment["movement_id"],
                "body17_valid_ratio": 0.98,
                "median_reprojection_error_px": 2.5,
                "median_used_cameras": 2.0,
            }
            for segment in timeline["segments"]
        ],
        "interpretation": "Measurement evidence only.",
    }
    readiness = {
        "movement_timeline": {"timeline_id": timeline["timeline_id"]},
        "pose_binding": {"status": "verified"},
        "blockers": [{"code": "partial_source_recording", "message": "Partial recording."}],
    }
    engineering_trial = {
        "movement_timeline_id": timeline["timeline_id"],
        "status": "deprecated_body17_screening_no_score",
        "partial_engineering_trial_score": None,
        "reference_accuracy_score": 4.0,
        "trial_deductions": [],
        "summary": {
            "measurement_count": 30,
            "measurable_count": 25,
            "candidate_count": 2,
            "not_measurable_count": 5,
        },
        "measurement_coverage_ratio": 25 / 30,
        "engineering_profile": {"disclaimer": "Engineering hypothesis only."},
    }
    wholebody = {
        "movement_timeline_id": timeline["timeline_id"],
        "accuracy_score": None,
        "summary": {"review_candidate_count": 1},
        "coverage": {
            "measurable_metric_count": 65,
            "thresholded_metric_count": 90,
            "measurement_coverage_ratio": 65 / 90,
            "coverage_gate_passed": False,
        },
        "candidate_events": [
            {
                "movement_id": "M01",
                "family": "fixation",
                "metric_id": "fixation_wrist_jitter_ratio",
                "value": 0.07,
            }
        ],
    }
    categorical = {
        "status": "categorical_diagnostics_only",
        "movement_timeline_id": timeline["timeline_id"],
        "summary": {
            "check_count": 2,
            "mismatch_candidate_count": 1,
            "consistent_count": 1,
            "ambiguous_count": 0,
            "not_measurable_count": 0,
            "unsupported_count": 0,
        },
        "checks": [
            {
                "movement_id": "M01",
                "event_kind": "wrong_stance",
                "status": "mismatch_candidate",
                "expected_label": "ap_seogi",
                "alternate_label": "ap_gubi",
                "anchor_frame": timeline["segments"][0]["anchors"]["fixation"],
                "evidence": [],
            },
            {
                "movement_id": "M01",
                "event_kind": "wrong_action",
                "status": "consistent",
                "expected_label": "arae_makki",
                "alternate_label": "momtong_jireugi",
                "anchor_frame": timeline["segments"][0]["anchors"]["fixation"],
                "evidence": [],
            },
        ],
        "interpretation": "Inferred candidates only.",
    }
    presentation = {
        "status": "presentation_diagnostic_only",
        "timeline_id": timeline["timeline_id"],
        "total_score": None,
        "components": {
            "speed_and_power": {
                "requested_metric_count": 1,
                "measurable_metric_count": 1,
                "metrics": {
                    "peak_speed": {
                        "sample_count": 6,
                        "median": 3.2,
                        "interquartile_range": 0.4,
                        "unit": "body_scale/sec",
                    }
                },
            }
        },
        "interpretation": "No judge-calibrated score.",
    }
    technical_conformance = {
        "status": "technical_conformance_diagnostic_only",
        "movement_timeline_id": timeline["timeline_id"],
        "summary": {
            "movement_count": 1,
            "review_required_count": 1,
            "mismatch_candidate_count": 0,
            "review_candidate_count": 1,
            "ambiguous_count": 0,
            "consistent_within_measured_scope_count": 0,
            "measurable_criterion_count": 1,
            "expected_criterion_count": 2,
        },
        "movements": [
            {
                "movement_id": "M01",
                "display_name": "Sol ap-seogi ve sol arae-makki",
                "conformance_status": "review_candidate",
                "fused_evidence_confidence": 0.82,
                "anchor_frame": timeline["segments"][0]["anchors"]["fixation"],
                "reason": "one_or_more_measured_criteria_outside_screening_range",
                "criterion_coverage": {
                    "measurable_count": 1,
                    "expected_count": 2,
                    "threshold_evaluable_count": 1,
                },
                "aspects": [
                    {
                        "aspect_id": "timing_and_control",
                        "status": "review_candidate",
                    }
                ],
                "identity_checks": [],
                "criteria": [
                    {
                        "criterion_id": "technique.fixation.stability",
                        "status": "review_candidate",
                        "evidence_confidence": 0.82,
                        "metrics": [
                            {
                                "metric_id": "fixation_wrist_jitter_ratio",
                                "value": 0.07,
                                "unit": "body_scale",
                            }
                        ],
                    }
                ],
            }
        ],
        "interpretation": "Measured scope only; no score or automatic deduction.",
    }
    automatic_segmentation = {
        "status": "automatic_segmentation_diagnostic_only",
        "movement_timeline_id": timeline["timeline_id"],
        "summary": {
            "detected_candidate_count": 7,
            "selected_movement_count": 6,
            "expected_movement_count": 6,
            "start_boundary_mae_frames": 11.3,
            "end_boundary_mae_frames": 10.3,
            "phase_anchor_mae_frames": 6.0,
            "phase_anchor_max_error_frames": 19,
        },
        "segments": [
            {
                "movement_id": "M01",
                "start_frame": 146,
                "end_frame": 236,
                "anchors": {"preparation": 146, "execution": 202, "fixation": 213},
            }
        ],
        "reference_comparison": {
            "summary": {"phase_anchor_mae_sec": 0.1},
            "movements": [
                {
                    "movement_id": "M01",
                    "status": "compared",
                    "start": {
                        "proposed_frame": 146,
                        "reference_frame": 140,
                        "delta_frames": 6,
                    },
                    "phases": {
                        "fixation": {
                            "proposed_frame": 213,
                            "reference_frame": 218,
                            "delta_frames": -5,
                        }
                    },
                }
            ],
        },
        "interpretation": "Diagnostic boundaries only.",
    }

    rendered = build_review_html(
        spec,
        timeline,
        evidence,
        readiness,
        {
            "ZED 35151067": "../videos/a.mp4",
            "ZED 37137479": "../videos/b.mp4",
            "Gelecek kamera": "../videos/c.mp4",
        },
        engineering_trial_report=engineering_trial,
        wholebody_diagnostics_report=wholebody,
        categorical_diagnostics_report=categorical,
        presentation_diagnostics_report=presentation,
        technical_conformance_report=technical_conformance,
        automatic_segmentation_report=automatic_segmentation,
    )

    assert "Kısmi kayıt · 6/18" in rendered
    assert "Accuracy</span><b>Hesaplanmadı" in rendered
    assert "M06" in rendered and "M18" in rendered
    assert "../videos/a.mp4" in rendered and "../videos/b.mp4" in rendered
    assert "../videos/c.mp4" in rendered and "3 kamera" in rendered
    assert "partial_source_recording" in rendered
    assert "3 kamerayı oynat" in rendered
    assert 'type="video/mp4"' in rendered
    assert 'preload="auto"' in rendered
    assert "Hedef kareler yükleniyor" in rendered
    assert "Video hazırlanıyor" in rendered
    assert "BODY-17 sayısal denemesi iptal edildi" in rendered
    assert "WholeBody-133 hata inceleme adayları" in rendered
    assert "fixation_wrist_jitter_ratio" in rendered
    assert "Skor yok" in rendered
    assert "Yanlış hareket ve yanlış duruş teşhisi" in rendered
    assert "Uyuşmazlık adayı" in rendered
    assert "Presentation kinematik göstergeleri" in rendered
    assert "M01–M06 hareket bazlı teknik uygunluk" in rendered
    assert "Birleşik kanıt güveni %82" in rendered
    assert "metric-filter" in rendered
    assert "metric-filter-status" in rendered
    assert "clear-review" in rendered
    assert "Otomatik hareket ve faz sınırı doğrulaması" in rendered
    assert "6.00 kare" in rendered
    assert "Oto fixation" in rendered


def test_movement_evidence_reports_measurements_without_scoring() -> None:
    spec = _active_spec()
    timeline = _complete_timeline()
    keypoints = [[[0.0, 0.0, 0.0] for _ in range(133)] for _ in range(30)]
    for frame in keypoints:
        frame[5] = [0.2, 0.0, 1.4]
        frame[6] = [-0.2, 0.0, 1.4]
        frame[7] = [0.35, 0.0, 1.1]
        frame[8] = [-0.35, 0.0, 1.1]
        frame[9] = [0.45, 0.0, 0.9]
        frame[10] = [-0.45, 0.0, 0.9]
        frame[11] = [0.15, 0.0, 0.9]
        frame[12] = [-0.15, 0.0, 0.9]
        frame[13] = [0.15, 0.2, 0.5]
        frame[14] = [-0.15, -0.2, 0.5]
        frame[15] = [0.15, 0.4, 0.0]
        frame[16] = [-0.15, -0.4, 0.0]
    matrix = [[1.0 for _ in range(133)] for _ in range(30)]
    cameras = [[2 for _ in range(133)] for _ in range(30)]
    payload = {
        "session_id": "test_session",
        "run_id": "test_run",
        "coordinate_system": {
            "name": "tk3d_analysis",
            "unit": "meter",
            "axes": {"x": "right", "y": "forward", "z": "up"},
            "handedness": "right",
        },
        "keypoints_3d_world": keypoints,
        "reliability_valid_mask": [[True for _ in range(133)] for _ in range(30)],
        "reprojection_error": matrix,
        "triangulation_score": matrix,
        "used_cameras": cameras,
    }

    report, rows = build_movement_evidence(payload, spec, timeline)

    assert report["status"] == "measurement_evidence_only"
    assert report["scoring_status"] == "not_scored_source_thresholds_inactive"
    assert report["accuracy_score"] is None
    assert report["deductions"] == []
    assert report["observability"]["observed_anchor_ratio"] == 1.0
    assert len(rows) == 2
    assert all(row["evidence_status"] == "observed" for row in rows)
    assert all("stance_span_ratio" in row and "body_scale_m" in row for row in rows)


def test_partial_engineering_trial_gates_quality_and_deduplicates_families() -> None:
    pack = load_rule_pack(RULE_PACK_PATH)
    spec = load_poomsae_spec(DRAFT_SPEC_PATH)
    timeline = load_movement_timeline(DRAFT_TIMELINE_PATH, spec)
    profile = load_engineering_profile(ENGINEERING_PROFILE_PATH)
    rows = []
    for movement_id in timeline["coverage"]["observed_movement_ids"]:
        rows.append(
            {
                "movement_id": movement_id,
                "phase_id": "fixation",
                "frame_index": 100,
                "evidence_status": "observed",
                "valid_body17_ratio": 0.99,
                "median_reprojection_error_px": 2.0,
                "median_used_cameras": 2.0,
                "torso_lean_deg": 5.0,
                "stance_span_ratio": 0.4 if movement_id in {"M01", "M02", "M03", "M04"} else 0.6,
                "left_knee_deg": 170.0 if movement_id != "M05" else 150.0,
                "right_knee_deg": 170.0,
                "left_elbow_deg": 170.0,
                "right_elbow_deg": 170.0,
                "left_wrist_height_torso_ratio": 0.3,
                "right_wrist_height_torso_ratio": 0.3,
            }
        )
    rows[0]["stance_span_ratio"] = 0.1
    rows[0]["left_knee_deg"] = 140.0
    rows[2]["evidence_status"] = "partially_observed"
    rows[2]["valid_body17_ratio"] = 0.88
    evidence = {
        "timeline": {"timeline_id": timeline["timeline_id"]},
        "scoring_status": "not_scored_partial_recording",
    }

    report = build_partial_engineering_trial(pack, spec, timeline, profile, evidence, rows)

    assert report["status"] == "deprecated_body17_screening_no_score"
    assert report["accuracy_score"] is None
    assert report["applied_deductions"] == []
    assert report["partial_engineering_trial_score"] is None
    assert report["total_engineering_trial_deduction"] is None
    assert report["summary"] == {
        "measurement_count": 30,
        "measurable_count": 25,
        "pass_count": 23,
        "candidate_count": 2,
        "not_measurable_count": 5,
        "selected_trial_deduction_count": 0,
        "diagnostic_candidate_family_count": 1,
        "deduplicated_candidate_count": 1,
    }
    assert report["automatic_major_detection"]["enabled"] is False
    assert report["measurement_coverage_ratio"] == pytest.approx(25 / 30)
    assert report["trial_score_confidence"] == "not_scored"
    assert report["trial_deductions"] == []
    assert any(item["trial_application_status"] == "review_candidate_no_score" for item in report["measurements"])
    m03 = [item for item in report["measurements"] if item["movement_id"] == "M03"]
    assert {item["measurement_status"] for item in m03} == {"not_measurable"}
    assert all(item["trial_deduction_points"] == 0.0 for item in m03)


def test_wholebody_diagnostics_use_133_points_and_fail_closed() -> None:
    spec = load_poomsae_spec(DRAFT_SPEC_PATH)
    timeline = load_movement_timeline(DRAFT_TIMELINE_PATH, spec)
    profile = load_wholebody_diagnostic_profile(WHOLEBODY_PROFILE_PATH)
    frame_count = timeline["frame_count"]
    points = np.zeros((frame_count, 133, 3), dtype=float)
    points[..., 2] = 0.8

    body_points = {
        5: [0.20, 0.0, 1.40],
        6: [-0.20, 0.0, 1.40],
        7: [0.35, 0.0, 1.10],
        8: [-0.35, 0.0, 1.10],
        9: [0.45, 0.0, 0.95],
        10: [-0.45, 0.0, 0.95],
        11: [0.15, 0.0, 0.90],
        12: [-0.15, 0.0, 0.90],
        13: [0.15, 0.2, 0.48],
        14: [-0.15, 0.2, 0.48],
        15: [0.15, 0.4, 0.04],
        16: [-0.15, 0.4, 0.04],
        17: [0.15, 0.62, 0.02],
        18: [0.12, 0.60, 0.02],
        19: [0.15, 0.40, 0.02],
        20: [-0.15, 0.62, 0.02],
        21: [-0.12, 0.60, 0.02],
        22: [-0.15, 0.40, 0.02],
    }
    for index, value in body_points.items():
        points[:, index] = value
    points[:, 59:65] = [0.06, 0.0, 1.62]
    points[:, 65:71] = [-0.06, 0.0, 1.62]
    for side, wrist in (("left", [0.45, 0.0, 0.95]), ("right", [-0.45, 0.0, 0.95])):
        for local_index in range(21):
            points[:, coco_hand_joint(side, "wrist") + local_index] = [
                wrist[0] + 0.002 * local_index,
                wrist[1] + 0.004 * local_index,
                wrist[2],
            ]
    first_fixation = timeline["segments"][0]["anchors"]["fixation"]
    points[first_fixation, 17:23] = np.nan

    payload = {
        "keypoints_3d_world": points,
        "reliability_valid_mask": np.ones((frame_count, 133), dtype=bool),
        "used_cameras": np.full((frame_count, 133), 2),
        "reprojection_error": np.ones((frame_count, 133), dtype=float),
    }
    report = build_wholebody_diagnostics(payload, spec, timeline, profile)

    assert report["status"] == "wholebody_diagnostics_only"
    assert report["numeric_score_enabled"] is False
    assert report["accuracy_score"] is None
    assert report["partial_engineering_trial_score"] is None
    assert report["deductions"] == []
    assert report["keypoint_contract"]["keypoint_count"] == 133
    assert report["keypoint_contract"]["groups_used"] == [
        "body17",
        "feet",
        "face",
        "left_hand",
        "right_hand",
    ]
    assert all(event["decision_status"] == "review_candidate_not_deduction" for event in report["candidate_events"])
    assert all(event["score_effect"] is None for event in report["candidate_events"])
    assert all(
        event["rule_eligibility"] == "blocked_unvalidated_screening_threshold" for event in report["candidate_events"]
    )
    first_foot_yaw = next(
        metric
        for metric in report["movements"][0]["metrics"]
        if metric["metric_id"] == "back_foot_yaw_to_stance_direction_deg"
    )
    assert first_foot_yaw["screening_status"] == "measured_diagnostic_only"
    assert first_foot_yaw["measurement_evidence"]["scope"] == "fixation_window"
    assert first_foot_yaw["measurement_evidence"]["required_groups"] == ["body17", "feet"]
    assert first_foot_yaw["direction_basis"] == "back_ankle_to_front_ankle"
    assert first_foot_yaw["sample_count"] >= 3
    assert first_foot_yaw["uncertainty_95"] is not None
    first_metrics = {
        metric["metric_id"]: metric for metric in report["movements"][0]["metrics"]
    }
    assert first_metrics["hand_foot_settle_difference_sec"]["measurement_evidence"]["scope"] == "preparation_to_fixation"
    assert first_metrics["fixation_wrist_jitter_ratio"]["measurement_evidence"]["scope"] == "fixation_to_segment_end"
    assert first_metrics["pelvis_weight_transfer_ratio"]["measurement_evidence"]["scope"] == "preparation_and_fixation_windows"
    not_measurable = [
        metric
        for movement in report["movements"]
        for metric in movement["metrics"]
        if metric["screening_status"] == "not_measurable"
    ]
    assert not_measurable
    assert all(
        "required_joint_sample_counts" in metric["measurement_evidence"]
        and "missing_required_joints" in metric["measurement_evidence"]
        for metric in not_measurable
    )


def test_fixation_pose_completion_handles_non_simultaneous_joint_gaps() -> None:
    from src.poomsae_scoring.wholebody_diagnostics import _window_pose_torso_lean

    points = np.full((11, 17, 3), np.nan, dtype=float)
    critical = {
        5: [0.20, 0.0, 1.40],
        6: [-0.20, 0.0, 1.40],
        11: [0.15, 0.0, 0.90],
        12: [-0.15, 0.0, 0.90],
    }
    for index, value in critical.items():
        points[:, index] = value
    for frame in range(points.shape[0]):
        points[frame, tuple(critical)[frame % len(critical)]] = np.nan
    assert all(
        not np.all(np.isfinite(points[frame, list(critical)]))
        for frame in range(points.shape[0])
    )
    arrays = {"points": points}

    value = _window_pose_torso_lean(
        arrays,
        center=5,
        start=0,
        end=10,
        radius=5,
        minimum_samples=3,
    )

    assert value == pytest.approx(0.0)


def test_wholebody_full_prefix_adds_compound_kick_diagnostics_without_thresholds() -> None:
    spec = load_poomsae_spec(DRAFT_SPEC_PATH)
    profile = load_wholebody_diagnostic_profile(WHOLEBODY_PROFILE_PATH)
    observed = spec["movements"][:14]
    frame_count = 12 * len(observed)
    segments = []
    for movement in observed:
        start = 12 * (movement["sequence_index"] - 1)
        anchors = {phase: start + offset + 1 for offset, phase in enumerate(movement["phases"])}
        segments.append(
            {
                "sequence_index": movement["sequence_index"],
                "movement_id": movement["movement_id"],
                "start_frame": start,
                "end_frame": start + 11,
                "anchors": anchors,
                "confidence": 1.0,
                "label_status": "confirmed",
            }
        )
    timeline = validate_movement_timeline(
        {
            "schema_version": 2,
            "timeline_id": "compound-kick-test",
            "poomsae_id": spec["poomsae_id"],
            "poomsae_version": spec["version"],
            "status": "draft",
            "label_source": "manual",
            "frame_index_space": "sample_index",
            "frame_count": frame_count,
            "fps": 60.0,
            "source_binding": {
                "session_id": "compound-test",
                "run_id": "compound-test-run",
                "pose_file": "outputs/compound-test/pose.json",
                "pose_file_sha256": None,
            },
            "coverage": {
                "recording_scope": "partial_sequence",
                "observed_movement_ids": [movement["movement_id"] for movement in observed],
                "missing_movement_ids": [movement["movement_id"] for movement in spec["movements"][14:]],
                "source_end_reason": "synthetic prefix",
            },
            "segments": segments,
        },
        spec,
    )
    points = np.zeros((frame_count, 133, 3), dtype=float)
    body_points = {
        5: [0.20, 0.0, 1.40],
        6: [-0.20, 0.0, 1.40],
        7: [0.35, 0.0, 1.10],
        8: [-0.35, 0.0, 1.10],
        9: [0.45, 0.0, 0.95],
        10: [-0.45, 0.0, 0.95],
        11: [0.15, 0.0, 0.90],
        12: [-0.15, 0.0, 0.90],
        13: [0.15, 0.2, 0.48],
        14: [-0.15, 0.2, 0.48],
        15: [0.15, 0.4, 0.04],
        16: [-0.15, 0.4, 0.04],
        17: [0.15, 0.62, 0.02],
        18: [0.12, 0.60, 0.02],
        19: [0.15, 0.40, 0.02],
        20: [-0.15, 0.62, 0.02],
        21: [-0.12, 0.60, 0.02],
        22: [-0.15, 0.40, 0.02],
    }
    for index, value in body_points.items():
        points[:, index] = value
    payload = {
        "keypoints_3d_world": points,
        "reliability_valid_mask": np.ones((frame_count, 133), dtype=bool),
        "used_cameras": np.full((frame_count, 133), 2),
        "reprojection_error": np.ones((frame_count, 133), dtype=float),
    }
    report = build_wholebody_diagnostics(payload, spec, timeline, profile)
    m14 = next(item for item in report["movements"] if item["movement_id"] == "M14")
    kick_metrics = [metric for metric in m14["metrics"] if metric["family"] == "kick"]
    landing_metric = next(metric for metric in m14["metrics"] if metric["metric_id"] == "kick_landing_to_punch_sec")

    assert m14["technique_ids"] == ["ap_chagi", "momtong_jireugi"]
    assert {metric["metric_id"] for metric in kick_metrics} == {
        "kick_knee_extension_deg",
        "kick_ankle_height_body_scale_ratio",
        "kick_rechamber_knee_deg",
        "support_foot_pivot_deg",
    }
    assert all(metric["screening_status"] == "measured_diagnostic_only" for metric in kick_metrics)
    assert all(metric["screening_rule"] is None for metric in kick_metrics)
    assert landing_metric["screening_status"] == "measured_diagnostic_only"
    assert landing_metric["criterion_id"] == "timing.kick_landing_punch.sequence"


def test_wholebody_eolgul_makki_emits_measurable_forehead_distance() -> None:
    spec = load_poomsae_spec(DRAFT_SPEC_PATH)
    profile = load_wholebody_diagnostic_profile(WHOLEBODY_PROFILE_PATH)
    observed = spec["movements"][:13]  # prefix through M13 (left eolgul-makki)
    frame_count = 12 * len(observed)
    segments = []
    for movement in observed:
        start = 12 * (movement["sequence_index"] - 1)
        anchors = {phase: start + offset + 1 for offset, phase in enumerate(movement["phases"])}
        segments.append(
            {
                "sequence_index": movement["sequence_index"],
                "movement_id": movement["movement_id"],
                "start_frame": start,
                "end_frame": start + 11,
                "anchors": anchors,
                "confidence": 1.0,
                "label_status": "confirmed",
            }
        )
    timeline = validate_movement_timeline(
        {
            "schema_version": 2,
            "timeline_id": "eolgul-forehead-test",
            "poomsae_id": spec["poomsae_id"],
            "poomsae_version": spec["version"],
            "status": "draft",
            "label_source": "manual",
            "frame_index_space": "sample_index",
            "frame_count": frame_count,
            "fps": 60.0,
            "source_binding": {
                "session_id": "eolgul-test",
                "run_id": "eolgul-test-run",
                "pose_file": "outputs/eolgul-test/pose.json",
                "pose_file_sha256": None,
            },
            "coverage": {
                "recording_scope": "partial_sequence",
                "observed_movement_ids": [movement["movement_id"] for movement in observed],
                "missing_movement_ids": [movement["movement_id"] for movement in spec["movements"][13:]],
                "source_end_reason": "synthetic prefix",
            },
            "segments": segments,
        },
        spec,
    )
    points = np.zeros((frame_count, 133, 3), dtype=float)
    # Body BODY-17 (legs give body_scale ~= 0.86 m).
    body_points = {
        5: [0.20, 0.0, 1.40],
        6: [-0.20, 0.0, 1.40],
        7: [0.35, 0.0, 1.10],
        8: [-0.35, 0.0, 1.10],
        9: [0.45, 0.0, 0.95],
        10: [-0.45, 0.0, 0.95],
        11: [0.15, 0.0, 0.90],
        12: [-0.15, 0.0, 0.90],
        13: [0.15, 0.2, 0.48],
        14: [-0.15, 0.2, 0.48],
        15: [0.15, 0.4, 0.04],
        16: [-0.15, 0.4, 0.04],
    }
    for index, value in body_points.items():
        points[:, index] = value
    # Face: eyes (iBUG 36-47 -> 59-70) separated ~0.08 m; eyebrows (iBUG 17-26 -> 40-49)
    # form the directly-measured forehead centre at the origin, z=1.58.
    for index in range(23 + 36, 23 + 42):  # left eye cluster
        points[:, index] = [0.04, 0.0, 1.52]
    for index in range(23 + 42, 23 + 48):  # right eye cluster
        points[:, index] = [-0.04, 0.0, 1.52]
    for index in range(23 + 17, 23 + 27):  # eyebrow line -> forehead centre
        points[:, index] = [0.0, 0.0, 1.58]
    # Left block fist ~0.88 fist-widths from the forehead centre.
    left_hand = {96: [0.09, 0.05, 1.58], 100: [0.063, 0.05, 1.58], 104: [0.037, 0.05, 1.58], 108: [0.01, 0.05, 1.58]}
    for index, value in left_hand.items():
        points[:, index] = value
    payload = {
        "keypoints_3d_world": points,
        "reliability_valid_mask": np.ones((frame_count, 133), dtype=bool),
        "used_cameras": np.full((frame_count, 133), 2),
        "reprojection_error": np.ones((frame_count, 133), dtype=float),
    }
    report = build_wholebody_diagnostics(payload, spec, timeline, profile)
    m13 = next(item for item in report["movements"] if item["movement_id"] == "M13")
    forehead = next(
        metric for metric in m13["metrics"] if metric["metric_id"] == "eolgul_fist_to_forehead_fist_ratio"
    )
    assert forehead["criterion_id"] == "technique.eolgul_makki.forehead_distance"
    assert forehead["value"] is not None
    assert 0.5 <= forehead["value"] <= 1.5
    assert forehead["screening_status"] == "measured_diagnostic_only"


def test_source_bound_accuracy_uses_uncertainty_and_keeps_partial_score_null() -> None:
    spec = load_poomsae_spec(DRAFT_SPEC_PATH)
    timeline = load_movement_timeline(DRAFT_TIMELINE_PATH, spec)
    profile = load_source_bound_accuracy_profile(SOURCE_BOUND_ACCURACY_PATH)
    diagnostics = {
        "status": "wholebody_diagnostics_only",
        "poomsae": {"poomsae_id": spec["poomsae_id"], "version": spec["version"]},
        "movement_timeline_id": timeline["timeline_id"],
        "movements": [
            {
                "movement_id": segment["movement_id"],
                "metrics": (
                    [
                        {
                            "metric_id": "back_foot_yaw_to_stance_direction_deg",
                            "value": 32.0,
                            "uncertainty_95": 1.0,
                        },
                        {
                            "metric_id": "arae_fist_to_thigh_fist_ratio",
                            "value": 2.5,
                            "uncertainty_95": 0.1,
                        },
                    ]
                    if segment["movement_id"] == "M01"
                    else []
                ),
            }
            for segment in timeline["segments"]
        ],
    }

    report = build_source_bound_accuracy_decisions(
        diagnostics,
        spec,
        timeline,
        profile,
    )

    m01 = [item for item in report["numeric_decisions"] if item["movement_id"] == "M01"]
    foot = next(item for item in m01 if item["metric_id"].startswith("back_foot"))
    arae = next(item for item in m01 if item["metric_id"].startswith("arae_fist"))
    assert foot["decision_status"] == "boundary_uncertain"
    assert foot["effective_uncertainty_95"] == 5.0
    assert foot["deduction_points"] is None
    assert arae["decision_status"] == "confirmed_source_bound_minor"
    assert arae["deduction_points"] == 0.1
    assert report["observed_scope_provisional_deduction_total"] == 0.1
    assert report["accuracy_score"] is None
    assert report["scoring_status"] == "observed_scope_only_no_accuracy_score"
    assert report["result_kind"] == "provisional_observed_scope_deduction_analysis"
    assert report["accuracy_evaluation_status"] == "not_eligible_incomplete_evidence"
    assert report["official_score_status"] == "not_available"
    assert report["provisional_deduction_status"] == "observed_scope_only_not_official"


def test_categorical_diagnostics_find_kinematic_mismatches_without_deduction() -> None:
    spec = load_poomsae_spec(DRAFT_SPEC_PATH)
    timeline = load_movement_timeline(DRAFT_TIMELINE_PATH, spec)
    diagnostic_profile = load_wholebody_diagnostic_profile(WHOLEBODY_PROFILE_PATH)
    accuracy_profile = load_source_bound_accuracy_profile(SOURCE_BOUND_ACCURACY_PATH)
    evidence = {
        "scope": "fixation_window",
        "anchor_frame": timeline["segments"][0]["anchors"]["fixation"],
        "anchor_time_sec": timeline["segments"][0]["anchors"]["fixation"] / timeline["fps"],
    }

    def metric(metric_id: str, value: float, unit: str) -> dict:
        return {
            "metric_id": metric_id,
            "value": value,
            "unit": unit,
            "uncertainty_95": 0.01,
            "measurement_evidence": evidence,
        }

    diagnostics = {
        "status": "wholebody_diagnostics_only",
        "poomsae": {"poomsae_id": spec["poomsae_id"], "version": spec["version"]},
        "movement_timeline_id": timeline["timeline_id"],
        "profile": {
            "profile_id": diagnostic_profile["profile_id"],
            "version": diagnostic_profile["version"],
        },
        "movements": [
            {
                "movement_id": "M01",
                "metrics": [
                    metric("stance_span_ratio", 0.80, "body_scale"),
                    metric("front_knee_deg", 140.0, "deg"),
                    metric("executing_wrist_height_torso_ratio", 0.70, "torso_ratio"),
                    metric("executing_elbow_deg", 175.0, "deg"),
                ],
            }
        ],
    }

    report = build_categorical_diagnostics(
        diagnostics,
        spec,
        timeline,
        diagnostic_profile,
    )

    assert report["summary"]["mismatch_candidate_count"] == 2
    assert {item["event_kind"] for item in report["observations"]} == {
        "wrong_action",
        "wrong_stance",
    }
    assert all(item["evidence_status"] == "inferred" for item in report["observations"])

    decisions = build_source_bound_accuracy_decisions(
        diagnostics,
        spec,
        timeline,
        accuracy_profile,
        report["observations"],
    )
    assert decisions["summary"]["applied_categorical_count"] == 0
    assert all(
        item["reason"] == "not_directly_observed"
        for item in decisions["categorical_decisions"]
    )


def test_technical_conformance_fuses_identity_uncertainty_and_evidence_quality() -> None:
    spec = load_poomsae_spec(DRAFT_SPEC_PATH)
    timeline = load_movement_timeline(DRAFT_TIMELINE_PATH, spec)
    poomsae_binding = {"poomsae_id": spec["poomsae_id"], "version": spec["version"]}

    def movement_report(movement_id: str, value: float, uncertainty: float) -> dict:
        segment = next(item for item in timeline["segments"] if item["movement_id"] == movement_id)
        anchor = segment["anchors"]["fixation"]
        return {
            "movement_id": movement_id,
            "metrics": [
                {
                    "metric_id": "torso_lean_p95_deg",
                    "criterion_id": "balance.torso_vertical",
                    "value": value,
                    "unit": "deg",
                    "uncertainty_95": uncertainty,
                    "screening_status": "review_candidate" if value > 10.0 else "within_screening_range",
                    "screening_rule": {"operator": "max", "value": 10.0},
                    "measurement_evidence": {
                        "start_frame": anchor - 4,
                        "end_frame": anchor + 5,
                        "required_group_coverage": {"body17": 0.92},
                        "required_joint_sample_counts": {
                            "left_shoulder": 9,
                            "right_shoulder": 10,
                        },
                    },
                }
            ],
        }

    wholebody = {
        "status": "wholebody_diagnostics_only",
        "poomsae": poomsae_binding,
        "movement_timeline_id": timeline["timeline_id"],
        "profile": {"profile_id": "test-profile", "version": "1.0.0"},
        "movements": [
            movement_report(segment["movement_id"], 14.0 if index == 1 else 5.0, 5.0 if index == 1 else 1.0)
            for index, segment in enumerate(timeline["segments"])
        ],
    }
    categorical = {
        "status": "categorical_diagnostics_only",
        "poomsae": poomsae_binding,
        "movement_timeline_id": timeline["timeline_id"],
        "profile": {"profile_id": "test-profile", "version": "1.0.0"},
        "checks": [
            {
                "check_id": f"CAT-{event_kind.upper()}-{segment['movement_id']}",
                "movement_id": segment["movement_id"],
                "event_kind": event_kind,
                "status": (
                    "mismatch_candidate"
                    if segment["movement_id"] == "M01" and event_kind == "wrong_action"
                    else "consistent"
                ),
                "expected_label": "expected",
                "alternate_label": "alternate",
                "reason": "test_reason",
                "confidence": 0.8 if segment["movement_id"] == "M01" else 0.85,
                "evidence": [],
            }
            for segment in timeline["segments"]
            for event_kind in ("wrong_action", "wrong_stance")
        ],
    }

    report = build_technical_conformance(wholebody, categorical, spec, timeline)

    assert report["status"] == "technical_conformance_diagnostic_only"
    assert report["safety_contract"]["score_claim_allowed"] is False
    assert report["safety_contract"]["automatic_deduction_allowed"] is False
    by_id = {item["movement_id"]: item for item in report["movements"]}
    assert by_id["M01"]["conformance_status"] == "mismatch_candidate"
    assert by_id["M01"]["fused_evidence_confidence"] == 0.8
    assert by_id["M02"]["conformance_status"] == "ambiguous"
    metric = by_id["M02"]["criteria"][0]["metrics"][0]
    assert metric["comparison_interval"] == [9.0, 19.0]
    assert metric["status"] == "boundary_uncertain"
    assert by_id["M03"]["conformance_status"] == "consistent_within_measured_scope"
    assert by_id["M03"]["criterion_coverage"]["measurable_count"] == 1


def test_technical_conformance_rejects_mismatched_categorical_binding() -> None:
    spec = load_poomsae_spec(DRAFT_SPEC_PATH)
    timeline = load_movement_timeline(DRAFT_TIMELINE_PATH, spec)
    poomsae_binding = {"poomsae_id": spec["poomsae_id"], "version": spec["version"]}
    wholebody = {
        "status": "wholebody_diagnostics_only",
        "poomsae": poomsae_binding,
        "movement_timeline_id": timeline["timeline_id"],
        "movements": [],
    }
    categorical = {
        "status": "categorical_diagnostics_only",
        "poomsae": poomsae_binding,
        "movement_timeline_id": "other-timeline",
        "checks": [],
    }

    with pytest.raises(ScoringContractError, match="categorical diagnostics timeline binding"):
        build_technical_conformance(wholebody, categorical, spec, timeline)


def test_presentation_diagnostics_aggregates_components_without_producing_a_score() -> None:
    spec = load_poomsae_spec(DRAFT_SPEC_PATH)
    timeline = load_movement_timeline(DRAFT_TIMELINE_PATH, spec)
    diagnostics = {
        "status": "wholebody_diagnostics_only",
        "poomsae": {"poomsae_id": spec["poomsae_id"], "version": spec["version"]},
        "movement_timeline_id": timeline["timeline_id"],
        "movements": [
            {
                "movement_id": seg["movement_id"],
                "metrics": [
                    {
                        "metric_id": "executing_wrist_peak_speed_body_scale_per_sec",
                        "value": 4.5 + index * 0.3,
                        "unit": "body_scale/sec",
                    },
                    {
                        "metric_id": "torso_lean_p95_deg",
                        "value": 8.0 + index,
                        "unit": "deg",
                    },
                    {
                        "metric_id": "fixation_wrist_jitter_ratio",
                        "value": 0.02 + index * 0.005,
                        "unit": "body_scale",
                    },
                    {
                        "metric_id": "head_torso_yaw_mismatch_deg",
                        "value": 12.0 + index * 2,
                        "unit": "deg",
                    },
                    {
                        "metric_id": "shoulder_hip_twist_deg",
                        "value": 15.0 + index,
                        "unit": "deg",
                    },
                ],
            }
            for index, seg in enumerate(timeline["segments"])
        ],
    }

    report = build_presentation_diagnostics(diagnostics, spec, timeline)

    assert report["status"] == "presentation_diagnostic_only"
    assert report["total_score"] is None
    assert report["judge_calibrated"] is False
    assert report["not_judge_validated"] is True
    assert report["safety_contract"]["score_claim_allowed"] is False
    assert report["safety_contract"]["kinematic_proxy_is_not_force_measurement"] is True

    speed = report["components"]["speed_and_power"]
    assert speed["measurable_metric_count"] == 1
    peak = speed["metrics"]["executing_wrist_peak_speed_body_scale_per_sec"]
    assert peak["sample_count"] == len(timeline["segments"])
    assert peak["median"] is not None
    assert peak["unit"] == "body_scale/sec"

    rhythm = report["components"]["rhythm_and_tempo"]
    assert "movement_duration_sec" in rhythm["metrics"]
    assert "transition_gap_sec" in rhythm["metrics"]
    assert rhythm["metrics"]["movement_duration_sec"]["sample_count"] == len(timeline["segments"])

    energy = report["components"]["expression_of_energy"]
    assert energy["measurable_metric_count"] == 4
    for metric_id in ("fixation_wrist_jitter_ratio", "torso_lean_p95_deg", "head_torso_yaw_mismatch_deg", "shoulder_hip_twist_deg"):
        assert energy["metrics"][metric_id]["sample_count"] == len(timeline["segments"])


def test_presentation_diagnostics_reject_non_wholebody_input() -> None:
    spec = load_poomsae_spec(DRAFT_SPEC_PATH)
    timeline = load_movement_timeline(DRAFT_TIMELINE_PATH, spec)
    with pytest.raises(ScoringContractError, match="WholeBody diagnostics report"):
        build_presentation_diagnostics({"status": "something_else"}, spec, timeline)


def test_downstream_diagnostics_reject_mismatched_provenance() -> None:
    spec = load_poomsae_spec(DRAFT_SPEC_PATH)
    timeline = load_movement_timeline(DRAFT_TIMELINE_PATH, spec)
    profile = load_source_bound_accuracy_profile(SOURCE_BOUND_ACCURACY_PATH)
    diagnostics = {
        "status": "wholebody_diagnostics_only",
        "poomsae": {"poomsae_id": spec["poomsae_id"], "version": spec["version"]},
        "movement_timeline_id": "different-timeline",
        "movements": [],
    }

    with pytest.raises(ScoringContractError, match="timeline binding"):
        build_presentation_diagnostics(diagnostics, spec, timeline)
    with pytest.raises(ScoringContractError, match="timeline binding"):
        build_source_bound_accuracy_decisions(diagnostics, spec, timeline, profile)


def test_derive_categorical_observations_flags_only_gaps_at_or_above_three_seconds() -> None:
    spec = load_poomsae_spec(DRAFT_SPEC_PATH)
    fps = 60.0
    segments = []
    frame = 0
    # M01: 30-frame segment then a 4.17s gap (250 frames @ 60fps) → should fire.
    # M02: 30-frame segment then a 1s gap (60 frames) → should not fire.
    # M03: 30-frame segment then 60 frames of trailing time → should not fire.
    gap_frames = [250, 60, 60]
    for index, movement in enumerate(spec["movements"][:3]):
        start = frame
        end = start + 30
        segments.append(
            {
                "sequence_index": index + 1,
                "movement_id": movement["movement_id"],
                "start_frame": start,
                "end_frame": end,
                "anchors": {
                    phase: start + offset + 1 for offset, phase in enumerate(movement["phases"])
                },
                "confidence": 1.0,
                "label_status": "confirmed",
            }
        )
        frame = end + gap_frames[index]
    timeline = validate_movement_timeline(
        {
            "schema_version": 2,
            "timeline_id": "pause-auto-test",
            "poomsae_id": spec["poomsae_id"],
            "poomsae_version": spec["version"],
            "status": "draft",
            "label_source": "manual",
            "frame_index_space": "sample_index",
            "frame_count": frame,
            "fps": fps,
            "source_binding": {
                "session_id": "pause-test",
                "run_id": "pause-test-run",
                "pose_file": "outputs/pause-test/pose.json",
                "pose_file_sha256": None,
            },
            "coverage": {
                "recording_scope": "partial_sequence",
                "observed_movement_ids": [seg["movement_id"] for seg in segments],
                "missing_movement_ids": [
                    movement["movement_id"] for movement in spec["movements"][3:]
                ],
                "source_end_reason": "synthetic",
            },
            "segments": segments,
        },
        spec,
    )

    observations = derive_categorical_observations(spec, timeline)

    assert len(observations) == 1
    observation = observations[0]
    assert observation["event_kind"] == "pause_at_least_3_sec"
    assert observation["movement_id"] == "M01"
    assert observation["confirmation_method"] == "duration_measurement"
    assert observation["evidence_status"] == "observed"
    assert 4.15 <= observation["measurement"]["duration_sec"] <= 4.20
    assert observation["start_frame"] == segments[0]["end_frame"]

    profile = load_source_bound_accuracy_profile(SOURCE_BOUND_ACCURACY_PATH)
    diagnostics = {
        "status": "wholebody_diagnostics_only",
        "poomsae": {"poomsae_id": spec["poomsae_id"], "version": spec["version"]},
        "movement_timeline_id": timeline["timeline_id"],
        "movements": [{"movement_id": seg["movement_id"], "metrics": []} for seg in segments],
    }
    report = build_source_bound_accuracy_decisions(
        diagnostics, spec, timeline, profile, observations=observations
    )
    assert report["summary"]["applied_categorical_count"] == 1
    applied_pause = next(
        item
        for item in report["categorical_decisions"]
        if item["event_kind"] == "pause_at_least_3_sec"
    )
    assert applied_pause["application_status"] == "applied"
    assert applied_pause["deduction_points"] == 0.3
    assert report["observed_scope_provisional_deduction_total"] == 0.3


def test_source_bound_accuracy_eolgul_forehead_rule_fires_and_gates_boundary() -> None:
    spec = load_poomsae_spec(DRAFT_SPEC_PATH)
    timeline = load_movement_timeline(DRAFT_TIMELINE_PATH, spec)
    profile = load_source_bound_accuracy_profile(SOURCE_BOUND_ACCURACY_PATH)

    def diagnostics_with_eolgul(value: float, uncertainty: float) -> dict:
        return {
            "status": "wholebody_diagnostics_only",
            "poomsae": {"poomsae_id": spec["poomsae_id"], "version": spec["version"]},
            "movement_timeline_id": timeline["timeline_id"],
            "movements": [
                {
                    "movement_id": "M13",
                    "metrics": [
                        {
                            "metric_id": "eolgul_fist_to_forehead_fist_ratio",
                            "value": value,
                            "uncertainty_95": uncertainty,
                        }
                    ],
                }
            ],
        }

    out_of_range = build_source_bound_accuracy_decisions(
        diagnostics_with_eolgul(3.0, 0.1), spec, timeline, profile
    )
    eolgul = next(
        item
        for item in out_of_range["numeric_decisions"]
        if item["rule_id"] == "HIST-2014-EOLGUL-FIST-FOREHEAD-ONE"
    )
    assert eolgul["movement_id"] == "M13"
    assert eolgul["decision_status"] == "confirmed_source_bound_minor"
    assert eolgul["deduction_points"] == 0.1

    within_range = build_source_bound_accuracy_decisions(
        diagnostics_with_eolgul(1.0, 0.05), spec, timeline, profile
    )
    eolgul_ok = next(
        item
        for item in within_range["numeric_decisions"]
        if item["rule_id"] == "HIST-2014-EOLGUL-FIST-FOREHEAD-ONE"
    )
    assert eolgul_ok["decision_status"] == "within_source_range"
    assert eolgul_ok["deduction_points"] is None
    # Numeric geometry can never escalate to a major deduction.
    assert eolgul["deduction_kind"] == "minor"


def test_source_bound_accuracy_recognizes_complete_performance_scope_name() -> None:
    spec = load_poomsae_spec(DRAFT_SPEC_PATH)
    spec["status"] = "active"
    spec["sequence_status"] = "active"
    spec["blocked_reasons"] = []
    spec = validate_poomsae_spec(spec)
    movement_ids = [movement["movement_id"] for movement in spec["movements"]]
    segments = []
    for movement in spec["movements"]:
        start = 10 * (movement["sequence_index"] - 1)
        anchors = {
            phase: start + min(offset + 1, 8)
            for offset, phase in enumerate(movement["phases"])
        }
        segments.append(
            {
                "sequence_index": movement["sequence_index"],
                "movement_id": movement["movement_id"],
                "start_frame": start,
                "end_frame": start + 9,
                "anchors": anchors,
                "confidence": 1.0,
                "label_status": "confirmed",
            }
        )
    timeline = validate_movement_timeline(
        {
            "schema_version": 2,
            "timeline_id": "complete-taegeuk1-test",
            "poomsae_id": spec["poomsae_id"],
            "poomsae_version": spec["version"],
            "status": "complete",
            "label_source": "manual",
            "frame_index_space": "sample_index",
            "frame_count": 180,
            "fps": 60.0,
            "source_binding": {
                "session_id": "test-session",
                "run_id": "test-run",
                "pose_file": "outputs/test-session/runs/test-run/json/pose.json",
                "pose_file_sha256": "a" * 64,
            },
            "coverage": {
                "recording_scope": "complete_performance",
                "observed_movement_ids": movement_ids,
                "missing_movement_ids": [],
                "source_end_reason": None,
            },
            "segments": segments,
        },
        spec,
    )
    profile = load_source_bound_accuracy_profile(SOURCE_BOUND_ACCURACY_PATH)
    diagnostics = {
        "status": "wholebody_diagnostics_only",
        "poomsae": {"poomsae_id": spec["poomsae_id"], "version": spec["version"]},
        "movement_timeline_id": timeline["timeline_id"],
        "movements": [
            {"movement_id": movement_id, "metrics": []} for movement_id in movement_ids
        ],
    }

    report = build_source_bound_accuracy_decisions(diagnostics, spec, timeline, profile)

    assert report["scoring_status"] == "eligible_for_separate_full_accuracy_evaluation"
    assert report["accuracy_evaluation_status"] == "eligible_not_evaluated"
    assert report["accuracy_score"] is None


def test_source_bound_result_rejects_official_score_contradiction() -> None:
    payload = {
        "schema_version": 1,
        "status": "source_bound_accuracy_decisions",
        "result_kind": "provisional_observed_scope_deduction_analysis",
        "accuracy_evaluation_status": "eligible_not_evaluated",
        "accuracy_score": None,
        "official_score_status": "not_available",
        "official_score": 9.7,
        "provisional_deduction_status": "observed_scope_only_not_official",
        "observed_scope_provisional_deduction_total": 0.0,
    }

    with pytest.raises(ScoringContractError, match="official score"):
        validate_source_bound_accuracy_result(payload)


def test_source_bound_accuracy_major_requires_explicit_categorical_observation() -> None:
    spec = load_poomsae_spec(DRAFT_SPEC_PATH)
    timeline = load_movement_timeline(DRAFT_TIMELINE_PATH, spec)
    profile = load_source_bound_accuracy_profile(SOURCE_BOUND_ACCURACY_PATH)
    diagnostics = {
        "status": "wholebody_diagnostics_only",
        "poomsae": {"poomsae_id": spec["poomsae_id"], "version": spec["version"]},
        "movement_timeline_id": timeline["timeline_id"],
        "movements": [{"movement_id": segment["movement_id"], "metrics": []} for segment in timeline["segments"]],
    }
    first = timeline["segments"][0]
    observations = [
        {
            "observation_id": "obs-wrong-action-m01",
            "event_kind": "wrong_action",
            "movement_id": "M01",
            "start_frame": first["start_frame"],
            "end_frame": first["end_frame"],
            "evidence_status": "observed",
            "confidence": 0.95,
            "description": "Sequence identity differs from the required action.",
            "measurement": None,
            "confirmation_method": "deterministic_sequence_violation",
        }
    ]

    report = build_source_bound_accuracy_decisions(
        diagnostics,
        spec,
        timeline,
        profile,
        observations,
    )

    event = report["categorical_decisions"][0]
    assert event["application_status"] == "applied"
    assert event["deduction_kind"] == "major"
    assert event["deduction_points"] == 0.3
    assert report["accuracy_score"] is None


def test_source_bound_profile_rejects_numeric_major_deductions() -> None:
    profile = load_source_bound_accuracy_profile(SOURCE_BOUND_ACCURACY_PATH)
    profile["numeric_rules"][0]["deduction_kind"] = "major"
    with pytest.raises(ScoringContractError, match="minor deductions only"):
        validate_source_bound_accuracy_profile(profile)


def test_source_intake_never_auto_activates_and_blocks_historical_numeric_tolerances(
    tmp_path: Path,
) -> None:
    source = tmp_path / "candidate.pdf"
    source.write_bytes(b"%PDF-1.4\nsynthetic source\n")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "source_id": "candidate_rule_source",
        "status": "candidate",
        "authority": "Test Official Authority",
        "title": "Candidate scoring source",
        "document_date": "2026-08-03",
        "effective_date": "2026-08-03",
        "retrieved_at": "2026-08-03",
        "authority_tier": "current_official_rule",
        "language": "English",
        "access": "user_supplied",
        "local_path": str(source),
        "expected_sha256": digest,
        "intended_uses": ["numeric_tolerance"],
        "activation_request": {
            "requested": True,
            "requested_claims": ["ap-seogi rear foot tolerance"],
            "numeric_thresholds_requested": True,
        },
        "notes": "Synthetic contract test.",
    }
    report = inspect_source_intake(validate_source_intake(manifest), workspace_root=tmp_path)

    assert report["status"] == "ready_for_manual_activation_review"
    assert report["activation"]["automatic_activation_allowed"] is False
    assert report["activation"]["ready_for_manual_review"] is True
    assert report["source"]["hash_status"] == "match"

    historical = deepcopy(manifest)
    historical["authority_tier"] = "historical_official"
    blocked = inspect_source_intake(historical, workspace_root=tmp_path)
    blocker_codes = {item["code"] for item in blocked["activation"]["blockers"]}

    assert blocked["status"] == "candidate_blocked"
    assert "numeric_threshold_authority_insufficient" in blocker_codes
    assert blocked["activation"]["automatic_activation_allowed"] is False


def test_wholebody_diagnostics_reject_body17_only_payload() -> None:
    spec = load_poomsae_spec(DRAFT_SPEC_PATH)
    timeline = load_movement_timeline(DRAFT_TIMELINE_PATH, spec)
    profile = load_wholebody_diagnostic_profile(WHOLEBODY_PROFILE_PATH)
    frame_count = timeline["frame_count"]
    payload = {
        "keypoints_3d_world": np.zeros((frame_count, 17, 3)),
        "reliability_valid_mask": np.ones((frame_count, 17), dtype=bool),
        "used_cameras": np.full((frame_count, 17), 2),
        "reprojection_error": np.ones((frame_count, 17)),
    }

    with pytest.raises(ScoringContractError, match="exactly 133|shape"):
        build_wholebody_diagnostics(payload, spec, timeline, profile)


def test_wholebody_profile_rejects_inverted_ranges() -> None:
    profile = load_wholebody_diagnostic_profile(WHOLEBODY_PROFILE_PATH)
    profile["thresholds"]["ap_seogi_span_ratio_min"] = 0.8
    profile["thresholds"]["ap_seogi_span_ratio_max"] = 0.2

    with pytest.raises(ScoringContractError, match="cannot exceed"):
        validate_wholebody_diagnostic_profile(profile)


def test_engineering_profile_rejects_automatic_major_detection() -> None:
    profile = load_engineering_profile(ENGINEERING_PROFILE_PATH)
    profile["policy"]["automatic_major_detection"] = True

    with pytest.raises(ScoringContractError, match="major detection"):
        validate_engineering_profile(profile)


def test_accuracy_readiness_verifies_complete_bound_artifact(tmp_path: Path) -> None:
    pose_path = tmp_path / "outputs" / "test_session" / "runs" / "test_run" / "json" / "pose.json"
    pose_path.parent.mkdir(parents=True)
    pose_path.write_text('{"keypoints_3d_world": []}\n', encoding="utf-8")
    timeline = _complete_timeline()
    timeline["source_binding"]["pose_file_sha256"] = hashlib.sha256(pose_path.read_bytes()).hexdigest()
    diagnostics = {
        "movement_timeline_id": timeline["timeline_id"],
        "keypoint_contract": {
            "keypoint_count": 133,
            "groups_used": ["body17", "feet", "face", "left_hand", "right_hand"],
        },
        "bindings": {"pose": {"sha256": timeline["source_binding"]["pose_file_sha256"]}},
        "numeric_score_enabled": False,
        "accuracy_score": None,
        "coverage": {
            "coverage_gate_passed": True,
            "measurement_coverage_ratio": 0.95,
            "minimum_required_ratio": 0.90,
        },
    }

    report = assess_accuracy_readiness(
        load_rule_pack(RULE_PACK_PATH),
        _active_spec(),
        timeline,
        workspace_root=tmp_path,
        wholebody_diagnostics=diagnostics,
    )

    assert report["status"] == "ready"
    assert report["rule_scoring_ready"] is True
    assert report["judge_calibrated_ready"] is False
    assert report["official_scoring_ready"] is False
    assert report["pose_binding"]["status"] == "verified"
    assert report["wholebody_diagnostic_binding"]["status"] == "verified"
    assert report["blockers"] == []


def test_accuracy_readiness_fails_closed_when_wholebody_diagnostics_are_missing(tmp_path: Path) -> None:
    pose_path = tmp_path / "outputs" / "test_session" / "runs" / "test_run" / "json" / "pose.json"
    pose_path.parent.mkdir(parents=True)
    pose_path.write_text('{"keypoints_3d_world": []}\n', encoding="utf-8")
    timeline = _complete_timeline()
    timeline["source_binding"]["pose_file_sha256"] = hashlib.sha256(pose_path.read_bytes()).hexdigest()

    report = assess_accuracy_readiness(
        load_rule_pack(RULE_PACK_PATH),
        _active_spec(),
        timeline,
        workspace_root=tmp_path,
    )

    assert report["status"] == "blocked"
    assert report["rule_scoring_ready"] is False
    assert report["component_states"]["wholebody_diagnostics"] == "wholebody_diagnostics_not_provided"
    assert {blocker["code"] for blocker in report["blockers"]} == {
        "wholebody_diagnostics_not_provided"
    }


def test_accuracy_readiness_blocks_low_wholebody_coverage(tmp_path: Path) -> None:
    pose_path = tmp_path / "outputs" / "test_session" / "runs" / "test_run" / "json" / "pose.json"
    pose_path.parent.mkdir(parents=True)
    pose_path.write_text('{"keypoints_3d_world": []}\n', encoding="utf-8")
    timeline = _complete_timeline()
    pose_sha256 = hashlib.sha256(pose_path.read_bytes()).hexdigest()
    timeline["source_binding"]["pose_file_sha256"] = pose_sha256
    diagnostics = {
        "movement_timeline_id": timeline["timeline_id"],
        "keypoint_contract": {
            "keypoint_count": 133,
            "groups_used": ["body17", "feet", "face", "left_hand", "right_hand"],
        },
        "bindings": {"pose": {"sha256": pose_sha256}},
        "numeric_score_enabled": False,
        "accuracy_score": None,
        "coverage": {
            "coverage_gate_passed": False,
            "measurement_coverage_ratio": 0.67,
            "minimum_required_ratio": 0.90,
        },
    }

    report = assess_accuracy_readiness(
        load_rule_pack(RULE_PACK_PATH),
        _active_spec(),
        timeline,
        workspace_root=tmp_path,
        wholebody_diagnostics=diagnostics,
    )

    assert report["status"] == "blocked"
    assert report["wholebody_diagnostic_binding"]["status"] == "wholebody_metric_coverage_below_minimum"
    assert {blocker["code"] for blocker in report["blockers"]} == {"wholebody_metric_coverage_below_minimum"}


def test_accuracy_readiness_rejects_changed_pose_artifact(tmp_path: Path) -> None:
    pose_path = tmp_path / "outputs" / "test_session" / "runs" / "test_run" / "json" / "pose.json"
    pose_path.parent.mkdir(parents=True)
    pose_path.write_text("changed\n", encoding="utf-8")

    report = assess_accuracy_readiness(
        load_rule_pack(RULE_PACK_PATH),
        _active_spec(),
        _complete_timeline(),
        workspace_root=tmp_path,
    )

    assert report["status"] == "blocked"
    assert report["pose_binding"]["status"] == "pose_sha256_mismatch"
    assert {blocker["code"] for blocker in report["blockers"]} == {
        "pose_sha256_mismatch",
        "wholebody_diagnostics_not_provided",
    }


def test_accuracy_engine_applies_minor_and_major_deductions() -> None:
    result = evaluate_accuracy(
        load_rule_pack(RULE_PACK_PATH),
        _active_spec(),
        _complete_timeline(),
        [
            _event("E01", "minor", "M01", 4, 6),
            _event("E02", "major", "M02", 18, 22),
        ],
    )

    assert result["status"] == "rule_based_provisional"
    assert result["judge_calibrated"] is False
    assert result["official_scoring_ready"] is False
    assert result["initial_accuracy_score"] == 4.0
    assert result["total_deduction"] == pytest.approx(0.4)
    assert result["accuracy_score"] == pytest.approx(3.6)
    assert result["summary"] == {
        "event_count": 2,
        "applied_count": 2,
        "not_applied_count": 0,
        "movement_count": 2,
    }


def test_accuracy_engine_applies_restart_as_performance_event() -> None:
    result = evaluate_accuracy(
        load_rule_pack(RULE_PACK_PATH),
        _active_spec(),
        _complete_timeline(),
        [_event("RESTART", "restart", None, 0, 0)],
    )

    assert result["accuracy_score"] == pytest.approx(3.4)
    assert result["applied_deductions"][0]["rule_id"] == "WT-2024-09-30-A16-EXPLANATION-3"


def test_accuracy_engine_does_not_deduct_without_observed_confirmed_evidence() -> None:
    duplicate = _event("E05", "minor", "M01", 4, 6, deduplication_key="same-error")
    events = [
        _event("E01", "minor", "M01", 4, 6, decision_status="candidate"),
        _event("E02", "minor", "M01", 4, 6, evidence_status="inferred"),
        _event("E03", "minor", "M01", 4, 6, confidence=0.5),
        _event("E04", "minor", "M01", 4, 6, deduplication_key="same-error"),
        duplicate,
        _event("E06", "major", "M01", 11, 12),
        _event("E07", "restart", None, 30, 30),
        _event("E08", "minor", "M01", 4, 6, phase_id="not_a_real_phase"),
        _event("E09", "minor", "M01", 4, 6, criterion_id="unknown.metric"),
    ]

    result = evaluate_accuracy(
        load_rule_pack(RULE_PACK_PATH),
        _active_spec(),
        _complete_timeline(),
        events,
    )

    assert result["accuracy_score"] == pytest.approx(3.9)
    assert result["summary"]["applied_count"] == 1
    reasons = {item["reason"] for item in result["not_applied_events"]}
    assert reasons == {
        "decision_not_confirmed",
        "insufficient_independent_evidence",
        "confidence_below_threshold",
        "duplicate_error_event",
        "event_outside_movement_interval",
        "event_outside_timeline",
        "unknown_movement_phase",
        "metric_not_authorized_for_movement",
    }


def test_accuracy_engine_rejects_duplicate_event_ids() -> None:
    event = _event("E01", "minor", "M01", 4, 6)

    with pytest.raises(ScoringContractError, match="duplicate Accuracy event_id"):
        evaluate_accuracy(
            load_rule_pack(RULE_PACK_PATH),
            _active_spec(),
            _complete_timeline(),
            [event, deepcopy(event)],
        )


def test_accuracy_engine_rejects_forged_rule_confirmation() -> None:
    event = _event("E01", "minor", "M01", 4, 6)
    event["rule_confirmation"]["rule_id"] = "forged-rule"

    with pytest.raises(ScoringContractError, match="rule_id does not match"):
        evaluate_accuracy(
            load_rule_pack(RULE_PACK_PATH),
            _active_spec(),
            _complete_timeline(),
            [event],
        )


def test_complete_timeline_must_match_poomsae_order() -> None:
    timeline = _complete_timeline()
    timeline["segments"][0]["movement_id"] = "M02"

    with pytest.raises(ScoringContractError, match="movement order"):
        validate_movement_timeline(timeline, _active_spec(), require_complete=True)


def test_timeline_rejects_shared_boundary_frame() -> None:
    timeline = _complete_timeline()
    timeline["segments"][1]["start_frame"] = timeline["segments"][0]["end_frame"]

    with pytest.raises(ScoringContractError, match="overlap"):
        validate_movement_timeline(timeline, _active_spec(), require_complete=True)


def test_timeline_coverage_must_match_observed_segments() -> None:
    timeline = _complete_timeline()
    timeline["coverage"]["recording_scope"] = "partial_sequence"
    timeline["coverage"]["observed_movement_ids"] = ["M01"]
    timeline["coverage"]["missing_movement_ids"] = ["M02"]
    timeline["coverage"]["source_end_reason"] = "Synthetic partial recording"
    timeline["status"] = "draft"

    with pytest.raises(ScoringContractError, match="exactly match timeline segments"):
        validate_movement_timeline(timeline, _active_spec())


def test_rule_pack_loader_rejects_duplicate_yaml_keys(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.yaml"
    duplicate.write_text("schema_version: 1\nschema_version: 1\n", encoding="utf-8")

    with pytest.raises(ScoringContractError, match="duplicate key"):
        load_rule_pack(duplicate)


def test_rule_pack_accepts_optional_categorical_examples() -> None:
    pack = load_rule_pack(RULE_PACK_PATH)
    deductions = pack["scoring"]["accuracy"]["deductions"]

    examples = deductions["major"]["categorical_examples"]
    assert len(examples) == 8
    assert {item["measurable"] for item in examples} <= {
        "pose_only",
        "pose_and_spec",
        "audio_required",
    }
    assert "categorical_examples" not in deductions["minor"]
    assert "categorical_examples" not in deductions["restart"]


def test_rule_pack_rejects_unsupported_measurable_label(tmp_path: Path) -> None:
    broken = tmp_path / "broken_measurable.yaml"
    broken.write_text(
        RULE_PACK_PATH.read_text(encoding="utf-8").replace(
            "measurable: pose_only", "measurable: telepathy_required", 1
        ),
        encoding="utf-8",
    )

    with pytest.raises(ScoringContractError, match="measurable is unsupported"):
        load_rule_pack(broken)


def test_rule_pack_rejects_duplicate_categorical_example_ids(tmp_path: Path) -> None:
    broken = tmp_path / "duplicate_example_id.yaml"
    broken.write_text(
        RULE_PACK_PATH.read_text(encoding="utf-8").replace("id: MAJ-02", "id: MAJ-01", 1),
        encoding="utf-8",
    )

    with pytest.raises(ScoringContractError, match="duplicate categorical example id"):
        load_rule_pack(broken)


def test_rule_pack_rejects_invalid_final_score_deduction_frequency(tmp_path: Path) -> None:
    broken = tmp_path / "invalid_final_frequency.yaml"
    broken.write_text(
        RULE_PACK_PATH.read_text(encoding="utf-8").replace("frequency: per_performance", "frequency: per_frame", 1),
        encoding="utf-8",
    )

    with pytest.raises(ScoringContractError, match="frequency must be per_performance"):
        load_rule_pack(broken)


def test_rule_pack_rejects_missing_final_score_deduction(tmp_path: Path) -> None:
    payload = load_rule_pack(RULE_PACK_PATH)
    del payload["scoring"]["final_score_deductions"]["boundary_crossing"]

    with pytest.raises(ScoringContractError, match="must contain exactly"):
        validate_rule_pack(payload)


def _prefix_timeline_with_gaps(
    spec: dict,
    segment_lengths: list[int],
    gap_frames: list[int],
    *,
    fps: float = 60.0,
    trailing_frames: int = 0,
    confidences: list[float] | None = None,
    timeline_id: str = "gap-edge-test",
) -> dict:
    """Validated prefix timeline whose inter-segment gaps are exact frame counts.

    ``gap_frames[i]`` is the empty-frame count between segment ``i`` and segment
    ``i + 1``; ``trailing_frames`` extends ``frame_count`` past the last segment
    so trailing-pause behaviour can be exercised deterministically.
    """
    segments = []
    frame = 0
    for index, length in enumerate(segment_lengths):
        movement = spec["movements"][index]
        start = frame
        end = start + length - 1
        segments.append(
            {
                "sequence_index": index + 1,
                "movement_id": movement["movement_id"],
                "start_frame": start,
                "end_frame": end,
                "anchors": {
                    phase: start + offset + 1
                    for offset, phase in enumerate(movement["phases"])
                },
                "confidence": 1.0 if confidences is None else confidences[index],
                "label_status": "confirmed",
            }
        )
        frame = end + 1
        if index < len(gap_frames):
            frame += gap_frames[index]
    frame_count = frame + trailing_frames
    observed = [segment["movement_id"] for segment in segments]
    return validate_movement_timeline(
        {
            "schema_version": 2,
            "timeline_id": timeline_id,
            "poomsae_id": spec["poomsae_id"],
            "poomsae_version": spec["version"],
            "status": "draft",
            "label_source": "manual",
            "frame_index_space": "sample_index",
            "frame_count": frame_count,
            "fps": fps,
            "source_binding": {
                "session_id": "gap-edge-session",
                "run_id": "gap-edge-run",
                "pose_file": "outputs/gap-edge/pose.json",
                "pose_file_sha256": None,
            },
            "coverage": {
                "recording_scope": "partial_sequence",
                "observed_movement_ids": observed,
                "missing_movement_ids": [
                    movement["movement_id"]
                    for movement in spec["movements"][len(segments):]
                ],
                "source_end_reason": "synthetic gap-edge prefix",
            },
            "segments": segments,
        },
        spec,
    )


def test_derive_categorical_observations_boundary_is_inclusive_at_threshold() -> None:
    spec = load_poomsae_spec(DRAFT_SPEC_PATH)
    # 180 frames @ 60fps is exactly 3.00s (fires, >= semantics); 179 frames is not.
    timeline = _prefix_timeline_with_gaps(spec, [30, 30, 30], [180, 179])

    observations = derive_categorical_observations(spec, timeline)

    assert len(observations) == 1
    observation = observations[0]
    assert observation["movement_id"] == "M01"
    assert observation["event_kind"] == "pause_at_least_3_sec"
    assert observation["measurement"]["duration_sec"] == pytest.approx(3.0)
    assert observation["observation_id"].startswith("AUTO-PAUSE-M01-")


def test_derive_categorical_observations_ignores_unbounded_trailing_gap() -> None:
    spec = load_poomsae_spec(DRAFT_SPEC_PATH)
    # Adjacent segments (zero gap) then 240 empty trailing frames (4.0s @ 60fps).
    timeline = _prefix_timeline_with_gaps(spec, [30, 30], [0], trailing_frames=240)

    observations = derive_categorical_observations(spec, timeline)

    assert observations == []


def test_derive_categorical_observations_threshold_parameter_and_validation() -> None:
    spec = load_poomsae_spec(DRAFT_SPEC_PATH)
    timeline = _prefix_timeline_with_gaps(spec, [30, 30], [90])  # 1.5s gap

    assert derive_categorical_observations(spec, timeline) == []

    for bad_threshold in (0.0, 1.0, 2.99):
        with pytest.raises(ScoringContractError, match="3-second rule"):
            derive_categorical_observations(spec, timeline, pause_threshold_sec=bad_threshold)
    with pytest.raises(ScoringContractError, match="minimum_confidence"):
        derive_categorical_observations(spec, timeline, minimum_confidence=1.1)


def test_derive_categorical_observations_preserves_conservative_confidence() -> None:
    spec = load_poomsae_spec(DRAFT_SPEC_PATH)
    timeline = _prefix_timeline_with_gaps(
        spec, [30, 30], [200], trailing_frames=200, confidences=[0.55, 0.95]
    )

    observations = derive_categorical_observations(spec, timeline)

    assert [obs["movement_id"] for obs in observations] == ["M01"]
    assert observations[0]["confidence"] == pytest.approx(0.55)


def test_presentation_diagnostics_rejects_non_list_movements() -> None:
    spec = load_poomsae_spec(DRAFT_SPEC_PATH)
    timeline = load_movement_timeline(DRAFT_TIMELINE_PATH, spec)

    with pytest.raises(ScoringContractError, match="movements must be a list"):
        build_presentation_diagnostics(
            {
                "status": "wholebody_diagnostics_only",
                "poomsae": {"poomsae_id": spec["poomsae_id"], "version": spec["version"]},
                "movement_timeline_id": timeline["timeline_id"],
                "movements": "not-a-list",
            },
            spec,
            timeline,
        )


def test_presentation_diagnostics_skips_unusable_values_and_summarizes_sparse_samples() -> None:
    spec = load_poomsae_spec(DRAFT_SPEC_PATH)
    timeline = _prefix_timeline_with_gaps(spec, [60], [])  # single 1.0s segment
    diagnostics = {
        "status": "wholebody_diagnostics_only",
        "poomsae": {"poomsae_id": spec["poomsae_id"], "version": spec["version"]},
        "movement_timeline_id": timeline["timeline_id"],
        "movements": [
            {
                "movement_id": "M01",
                "metrics": [
                    {
                        "metric_id": "executing_wrist_peak_speed_body_scale_per_sec",
                        "value": None,
                        "unit": "body_scale/sec",
                    },
                    {
                        "metric_id": "torso_lean_p95_deg",
                        "value": "not-a-number",
                        "unit": "deg",
                    },
                    {
                        "metric_id": "fixation_wrist_jitter_ratio",
                        "value": 0.02,
                        "unit": "body_scale",
                    },
                ],
            }
        ],
    }

    report = build_presentation_diagnostics(diagnostics, spec, timeline)

    speed = report["components"]["speed_and_power"]
    assert speed["measurable_metric_count"] == 0
    peak = speed["metrics"]["executing_wrist_peak_speed_body_scale_per_sec"]
    assert peak["sample_count"] == 0
    assert peak["median"] is None
    assert peak["interquartile_range"] is None

    energy = report["components"]["expression_of_energy"]
    assert energy["measurable_metric_count"] == 1
    jitter = energy["metrics"]["fixation_wrist_jitter_ratio"]
    assert jitter["sample_count"] == 1
    assert jitter["median"] == pytest.approx(0.02)
    assert jitter["interquartile_range"] is None  # fewer than 4 samples
    assert energy["metrics"]["torso_lean_p95_deg"]["sample_count"] == 0  # garbage skipped

    rhythm = report["components"]["rhythm_and_tempo"]
    duration = rhythm["metrics"]["movement_duration_sec"]
    assert duration["sample_count"] == 1
    assert duration["median"] == pytest.approx(1.0)
    assert rhythm["metrics"]["transition_gap_sec"]["sample_count"] == 0  # no transitions
    assert rhythm["measurable_metric_count"] == 1


def test_eolgul_forehead_ratio_fails_closed_on_quality_gates() -> None:
    from src.poomsae_scoring.wholebody_diagnostics import _eolgul_fist_to_forehead_ratio

    profile = load_wholebody_diagnostic_profile(WHOLEBODY_PROFILE_PATH)
    gates = profile["quality_gates"]
    scale = 0.86

    def scene() -> dict:
        points = np.full((1, 133, 3), np.nan)
        left_hand = {
            96: [0.09, 0.05, 1.58],
            100: [0.063, 0.05, 1.58],
            104: [0.037, 0.05, 1.58],
            108: [0.01, 0.05, 1.58],
        }
        for index, value in left_hand.items():
            points[0, index] = value
        for index in range(23 + 36, 23 + 42):  # left eye cluster
            points[0, index] = [0.04, 0.0, 1.52]
        for index in range(23 + 42, 23 + 48):  # right eye cluster
            points[0, index] = [-0.04, 0.0, 1.52]
        for index in range(23 + 17, 23 + 27):  # eyebrow line -> forehead centre
            points[0, index] = [0.0, 0.0, 1.58]
        return {"points": points}

    baseline = _eolgul_fist_to_forehead_ratio(scene(), 0, "left", scale, gates)
    assert baseline is not None
    assert 0.5 <= baseline <= 1.5

    # Missing or degenerate body scale.
    assert _eolgul_fist_to_forehead_ratio(scene(), 0, "left", None, gates) is None
    assert _eolgul_fist_to_forehead_ratio(scene(), 0, "left", 0.0, gates) is None

    # Any missing fist landmark fails closed.
    missing_hand = scene()
    missing_hand["points"][0, 100] = np.nan
    assert _eolgul_fist_to_forehead_ratio(missing_hand, 0, "left", scale, gates) is None

    # Zero-width fist (all knuckles collapsed) fails closed.
    degenerate_fist = scene()
    for index in (96, 100, 104, 108):
        degenerate_fist["points"][0, index] = [0.05, 0.05, 1.58]
    assert _eolgul_fist_to_forehead_ratio(degenerate_fist, 0, "left", scale, gates) is None

    # Fist width outside the hand quality gate fails closed.
    giant_fist = scene()
    giant_fist["points"][0, 96] = [0.5, 0.05, 1.58]
    giant_fist["points"][0, 108] = [-0.5, 0.05, 1.58]
    assert _eolgul_fist_to_forehead_ratio(giant_fist, 0, "left", scale, gates) is None

    # Any missing eye landmark fails closed.
    missing_eye = scene()
    missing_eye["points"][0, 23 + 40] = np.nan
    assert _eolgul_fist_to_forehead_ratio(missing_eye, 0, "left", scale, gates) is None

    # Eye separation outside the face quality gate fails closed.
    wide_eyes = scene()
    for index in range(23 + 36, 23 + 42):
        wide_eyes["points"][0, index] = [0.4, 0.0, 1.52]
    for index in range(23 + 42, 23 + 48):
        wide_eyes["points"][0, index] = [-0.4, 0.0, 1.52]
    assert _eolgul_fist_to_forehead_ratio(wide_eyes, 0, "left", scale, gates) is None

    # Fewer than 4 valid eyebrow points fails closed.
    sparse_brows = scene()
    for index in range(23 + 17, 23 + 24):  # leave only 3 of 10 brow points valid
        sparse_brows["points"][0, index] = np.nan
    assert _eolgul_fist_to_forehead_ratio(sparse_brows, 0, "left", scale, gates) is None


def test_automatic_timeline_feeds_accuracy_and_presentation_end_to_end() -> None:
    from src.poomsae_scoring.sequence_alignment import build_automatic_movement_timeline
    from src.synthetic_poses import build_frame

    spec = load_poomsae_spec(DRAFT_SPEC_PATH)
    profile = load_source_bound_accuracy_profile(SOURCE_BOUND_ACCURACY_PATH)
    expected_poses = [
        build_frame(60 + 6 * index, 60 + 6 * index)
        for index in range(len(spec["movements"]))
    ]
    # Segment mean poses exactly match the first three expected poses; the gap
    # between segment 0 and 1 is 240 frames (4.0s @ 60fps) so the automatic
    # pause detector must fire on M01 and nowhere else.
    segments = [
        {"segment_id": 0, "start_frame": 0, "end_frame": 59, "mean_pose": build_frame(60, 60)},
        {"segment_id": 1, "start_frame": 300, "end_frame": 359, "mean_pose": build_frame(66, 66)},
        {"segment_id": 2, "start_frame": 370, "end_frame": 429, "mean_pose": build_frame(72, 72)},
    ]

    timeline = build_automatic_movement_timeline(
        segments=segments,
        expected_poses=expected_poses,
        poomsae_spec=spec,
        frame_count=500,
        fps=60.0,
        source_binding={
            "session_id": "chain-session",
            "run_id": "chain-run",
            "pose_file": "outputs/chain/pose.json",
            "pose_file_sha256": None,
        },
        timeline_id="auto-chain-test",
    )
    assert timeline["label_source"] == "automatic"
    assert timeline["coverage"]["observed_movement_ids"] == ["M01", "M02", "M03"]
    assert timeline["coverage"]["recording_scope"] == "partial_sequence"

    observations = derive_categorical_observations(spec, timeline)
    assert len(observations) == 1
    assert observations[0]["movement_id"] == "M01"
    assert observations[0]["measurement"]["duration_sec"] == pytest.approx(4.0)
    assert observations[0]["evidence_status"] == "inferred"

    diagnostics = {
        "status": "wholebody_diagnostics_only",
        "poomsae": {"poomsae_id": spec["poomsae_id"], "version": spec["version"]},
        "movement_timeline_id": timeline["timeline_id"],
        "movements": [
            {"movement_id": segment["movement_id"], "metrics": []}
            for segment in timeline["segments"]
        ],
    }
    accuracy = build_source_bound_accuracy_decisions(
        diagnostics, spec, timeline, profile, observations=observations
    )
    assert accuracy["accuracy_score"] is None
    assert accuracy["scoring_status"] == "observed_scope_only_no_accuracy_score"
    assert accuracy["summary"]["applied_categorical_count"] == 0
    assert accuracy["observed_scope_provisional_deduction_total"] == pytest.approx(0.0)

    presentation = build_presentation_diagnostics(diagnostics, spec, timeline)
    assert presentation["total_score"] is None
    assert presentation["safety_contract"]["score_claim_allowed"] is False
    rhythm = presentation["components"]["rhythm_and_tempo"]
    assert rhythm["metrics"]["movement_duration_sec"]["sample_count"] == 3
    assert rhythm["metrics"]["transition_gap_sec"]["sample_count"] == 2
    assert rhythm["metrics"]["transition_gap_sec"]["max"] == pytest.approx(4.0)
