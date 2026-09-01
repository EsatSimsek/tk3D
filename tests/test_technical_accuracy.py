from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from src.data_structures import (
    COCO_BODY_JOINTS,
    COCO_FACE_INDICES,
    COCO_FOOT_JOINTS,
    COCO_LEFT_HAND_OFFSET,
    COCO_WHOLEBODY_KEYPOINTS,
)
from src.poomsae_scoring import ScoringContractError, load_movement_timeline, load_poomsae_spec
from src.poomsae_scoring.technical_accuracy import (
    ACTIVE_EVALUATORS,
    build_technical_accuracy_diagnostics,
    derive_athlete_local_direction_reference,
    evaluate_temporary_threshold,
    load_technical_accuracy_profile,
    resolve_movement_accuracy_contracts,
    validate_athlete_local_direction_reference,
    validate_technical_accuracy_profile,
)
from src.poomsae_scoring.technical_accuracy_metrics import measure_observable_accuracy_metrics


ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = ROOT / "config" / "scoring" / "engineering" / "taegeuk_1_wholebody_diagnostics_v3.yaml"
SPEC_PATH = ROOT / "config" / "scoring" / "poomsae" / "taegeuk_1_jang_v0_draft.yaml"
TIMELINE_PATH = ROOT / "config" / "scoring" / "timelines" / "poomsae1_zed2i_rgbd_rerun_20260802_draft.yaml"


def test_v3_profile_has_strict_complete_score_neutral_rule_inventory() -> None:
    profile = load_technical_accuracy_profile(PROFILE_PATH)
    rules = profile["resolved_rules"]

    assert len(rules) == 174
    assert {rule["metric_id"] for rule in rules if rule["status"] == "active_diagnostic"} == ACTIVE_EVALUATORS
    assert sum(rule["status"] == "active_diagnostic" for rule in rules) == 33
    assert sum(rule["status"] == "measurement_only" for rule in rules) == 116
    assert sum(rule["status"] == "blocked_missing_reference" for rule in rules) == 17
    assert sum(rule["status"] == "not_observable_with_current_pipeline" for rule in rules) == 8
    for rule in rules:
        assert rule["metric_id"] != rule["criterion_id"]
        assert rule["score_effect"] is None
        assert rule["deduction_points"] is None
        assert rule["deduction_enabled"] is False
        assert rule["numeric_score_enabled"] is False
        assert rule["evidence_quality_requirements"]
        assert rule["failure_behavior"] == "emit_state_and_reason_without_candidate"
        if rule["status"] == "not_observable_with_current_pipeline":
            assert rule["measurement_evaluator_status"] == "not_applicable_pipeline_limit"
            assert rule["measurement_evaluator_id"] is None
        else:
            assert rule["measurement_evaluator_status"] == "implemented"
            assert rule["measurement_evaluator_id"]


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (lambda profile: profile.update({"unknown": 1}), "keys must be exactly"),
        (lambda profile: profile["thresholds"]["torso_lean_p95_deg"].update({"unit": "pixels"}), "invalid threshold unit"),
        (lambda profile: profile["thresholds"]["torso_lean_p95_deg"].update({"value": -1}), "non-negative"),
        (lambda profile: profile["metric_catalog"]["head_orientation"].append("head_target_yaw_error_deg"), "duplicate rule"),
        (lambda profile: profile["active_rules"].pop(), "implemented evaluator inventory differ"),
    ],
)
def test_v3_profile_rejects_invalid_configuration(mutator, match: str) -> None:
    raw = _raw_profile()
    mutator(raw)
    with pytest.raises(ScoringContractError, match=match):
        validate_technical_accuracy_profile(raw)


def test_every_active_threshold_has_pass_boundary_fail_and_fail_closed_behavior() -> None:
    profile = load_technical_accuracy_profile(PROFILE_PATH)
    rules = {rule["metric_id"]: rule for rule in profile["resolved_rules"]}
    boolean_rules = ACTIVE_EVALUATORS - set(profile["thresholds"])

    for metric_id in ACTIVE_EVALUATORS - boolean_rules:
        threshold = rules[metric_id]["threshold"]
        operator = threshold["operator"]
        value = threshold["value"]
        uncertainty = threshold["uncertainty_band"]
        if operator == "range":
            passing = (value[0] + value[1]) / 2.0
            boundary = value[1]
            failing = value[1] + uncertainty + 1.0
        elif operator == "min":
            passing = value + uncertainty + 1.0
            boundary = value
            failing = max(0.0, value - uncertainty - 1.0)
        else:
            passing = max(0.0, value - uncertainty - 1e-3)
            boundary = value
            failing = value + uncertainty + 1.0
        assert evaluate_temporary_threshold(passing, threshold) == "within_screening_range"
        assert evaluate_temporary_threshold(boundary, threshold) == "boundary_uncertain"
        assert evaluate_temporary_threshold(failing, threshold) == "out_of_range"
        assert evaluate_temporary_threshold(None, threshold) == "unmeasurable"
        assert evaluate_temporary_threshold(float("nan"), threshold) == "unmeasurable"
        assert evaluate_temporary_threshold(float("inf"), threshold) == "unmeasurable"
        assert evaluate_temporary_threshold(float("-inf"), threshold) == "unmeasurable"
        assert evaluate_temporary_threshold("5", threshold) == "unmeasurable"
        assert evaluate_temporary_threshold(True, threshold) == "unmeasurable"
    for metric_id in boolean_rules:
        assert evaluate_temporary_threshold(True, None) == "within_screening_range", metric_id
        assert evaluate_temporary_threshold(False, None) == "out_of_range", metric_id
        assert evaluate_temporary_threshold(None, None) == "unmeasurable", metric_id
        assert evaluate_temporary_threshold("true", None) == "unmeasurable", metric_id


def test_athlete_local_direction_reference_is_session_bound_orthogonal_and_fail_closed() -> None:
    reference = {
        "schema_version": 1,
        "session_id": "session-a",
        "reference_pose_sha256": "a" * 64,
        "gravity_up_vector": [0.0, 0.0, 1.0],
        "initial_forward_vector": [0.0, 2.0, 0.1],
        "basis_source": "manually_declared_session_bound",
        "provenance": "manual diagnostic declaration; not production calibration",
        "quality_status": "validated_diagnostic_reference",
    }
    validated = validate_athlete_local_direction_reference(
        reference,
        expected_session_id="session-a",
        expected_pose_sha256="a" * 64,
    )
    assert validated is not None
    assert np.dot(validated["initial_forward_vector"], validated["gravity_up_vector"]) == pytest.approx(0.0)
    assert validated["production_calibration_claim"] is False
    assert validate_athlete_local_direction_reference(None) is None

    invalid = deepcopy(reference)
    invalid["initial_forward_vector"] = [0.0, 0.0, 2.0]
    with pytest.raises(ScoringContractError, match="degenerate after horizontal projection"):
        validate_athlete_local_direction_reference(invalid)


def test_all_m01_m18_contracts_resolve_without_inventing_spec_techniques() -> None:
    profile = load_technical_accuracy_profile(PROFILE_PATH)
    spec = load_poomsae_spec(SPEC_PATH)
    contracts = resolve_movement_accuracy_contracts(spec, profile)

    assert [item["movement_id"] for item in contracts] == [f"M{index:02d}" for index in range(1, 19)]
    assert all(item["contract_resolution_status"] == "resolved" for item in contracts)
    assert contracts[13]["technique_types"] == ["ap_chagi", "momtong_jireugi"]
    assert contracts[15]["technique_types"] == ["ap_chagi", "momtong_jireugi"]
    assert contracts[17]["technique_types"] == ["momtong_jireugi", "kihap"]


def test_real_scope_shape_report_keeps_m07_m18_blocked_and_every_candidate_score_neutral() -> None:
    profile = load_technical_accuracy_profile(PROFILE_PATH)
    spec = load_poomsae_spec(SPEC_PATH)
    timeline = load_movement_timeline(TIMELINE_PATH, spec)
    pose = _synthetic_pose(timeline["frame_count"])
    wholebody = _synthetic_wholebody(timeline)

    report = build_technical_accuracy_diagnostics(pose, spec, timeline, profile, wholebody)

    assert report["total_score"] is None
    assert report["presentation_score"] is None
    assert report["deductions"] == []
    assert report["summary"]["movement_contract_count"] == 18
    assert report["summary"]["observed_movement_count"] == 6
    assert report["summary"]["configuration_only_movement_count"] == 12
    assert report["summary"]["landmark_inventory_count"] == COCO_WHOLEBODY_KEYPOINTS
    assert report["summary"]["landmarks_declared_by_any_rule_count"] == COCO_WHOLEBODY_KEYPOINTS
    inventory = report["landmark_inventory"]
    assert [row["landmark_index"] for row in inventory] == list(range(COCO_WHOLEBODY_KEYPOINTS))
    assert all(row["declared_rule_count"] > 0 for row in inventory)
    assert all(
        inventory[index]["active_diagnostic_rule_count"] > 0
        for index in COCO_FOOT_JOINTS.values()
    )
    active_hand_local = {0, 5, 9, 13, 17}
    assert all(
        (inventory[index]["active_diagnostic_rule_count"] > 0)
        == ((index - offset) in active_hand_local)
        for offset in (COCO_LEFT_HAND_OFFSET, COCO_LEFT_HAND_OFFSET + 21)
        for index in range(offset, offset + 21)
    )
    assert any(
        inventory[index]["active_diagnostic_rule_count"] > 0 for index in COCO_FACE_INDICES
    )
    assert report["summary"]["landmarks_declared_by_active_rule_count"] == 51
    assert report["summary"]["implemented_measurement_evaluator_rule_count"] == 166
    assert report["summary"]["evaluator_not_implemented_rule_count"] == 0
    assert not any(
        row["blocking_reason"] == "evaluator_not_implemented"
        for row in report["coverage_matrix"]
        if row["applies"] and row["movement_id"] in {f"M{index:02d}" for index in range(1, 7)}
    )
    assert all(not movement["present_in_current_timeline"] for movement in report["movements"][6:])
    assert all(
        row["blocking_reason"] == "movement_not_present_in_timeline"
        for row in report["coverage_matrix"]
        if row["movement_id"] in {f"M{index:02d}" for index in range(7, 19)} and row["applies"]
    )
    for candidate in report["candidate_events"]:
        assert candidate["decision_status"] == "review_candidate_not_deduction"
        assert candidate["score_effect"] is None
        assert candidate["deduction_points"] is None
    serialized = json.dumps(report, allow_nan=False)
    assert "NaN" not in serialized and "Infinity" not in serialized


def test_technical_accuracy_rejects_body17_only_input() -> None:
    profile = load_technical_accuracy_profile(PROFILE_PATH)
    spec = load_poomsae_spec(SPEC_PATH)
    timeline = load_movement_timeline(TIMELINE_PATH, spec)
    pose = _synthetic_pose(timeline["frame_count"])
    pose["keypoints_3d_world"] = np.zeros((timeline["frame_count"], 17, 3)).tolist()

    with pytest.raises(ScoringContractError, match="exactly 133|shape"):
        build_technical_accuracy_diagnostics(pose, spec, timeline, profile, _synthetic_wholebody(timeline))


def test_direction_bound_rules_evaluate_only_with_valid_session_local_basis() -> None:
    profile = load_technical_accuracy_profile(PROFILE_PATH)
    spec = load_poomsae_spec(SPEC_PATH)
    timeline = load_movement_timeline(TIMELINE_PATH, spec)
    pose = _synthetic_pose(timeline["frame_count"])
    binding = timeline["source_binding"]
    reference = {
        "schema_version": 1,
        "session_id": binding["session_id"],
        "reference_pose_sha256": binding["pose_file_sha256"],
        "gravity_up_vector": [0.0, 0.0, 1.0],
        "initial_forward_vector": [0.0, 1.0, 0.0],
        "basis_source": "manually_declared_session_bound",
        "provenance": "synthetic test binding; not production calibration",
        "quality_status": "validated_diagnostic_reference",
    }

    report = build_technical_accuracy_diagnostics(
        pose,
        spec,
        timeline,
        profile,
        _synthetic_wholebody(timeline),
        direction_reference=reference,
    )

    direction_ids = set(profile["direction_bound_rules"])
    observed = [
        row
        for row in report["coverage_matrix"]
        if row["movement_id"] == "M01" and row["metric_id"] in direction_ids
    ]
    assert observed
    assert not any(row["blocking_reason"] == "missing_athlete_local_direction_binding" for row in observed)
    assert any(row["measured"] and row["evaluated"] for row in observed)
    assert report["summary"]["score_effect_count"] == 0
    assert report["total_score"] is None

    foreign = deepcopy(reference)
    foreign["session_id"] = "another-session"
    with pytest.raises(ScoringContractError, match="session binding mismatch"):
        build_technical_accuracy_diagnostics(
            pose,
            spec,
            timeline,
            profile,
            _synthetic_wholebody(timeline),
            direction_reference=foreign,
        )


def test_direction_reference_is_derived_from_ready_stance_and_stays_session_bound() -> None:
    profile = load_technical_accuracy_profile(PROFILE_PATH)
    spec = load_poomsae_spec(SPEC_PATH)
    timeline = load_movement_timeline(TIMELINE_PATH, spec)
    pose = _synthetic_pose(timeline["frame_count"])
    binding = timeline["source_binding"]

    envelope = derive_athlete_local_direction_reference(pose, spec, timeline, profile)

    assert envelope["status"] == "derived"
    assert envelope["reason"] is None
    assert envelope["production_calibration_claim"] is False
    assert envelope["anchor"]["movement_id"] == "M01"
    assert envelope["anchor"]["phase"] == "preparation"
    reference = envelope["reference"]
    assert reference["basis_source"] == "derived_session_bound"
    assert reference["quality_status"] == "validated_diagnostic_reference"
    assert reference["session_id"] == binding["session_id"]
    assert reference["reference_pose_sha256"] == binding["pose_file_sha256"]
    # The synthetic athlete faces +y; the derived basis must stay horizontal and unit.
    forward = np.asarray(reference["initial_forward_vector"], dtype=float)
    assert forward == pytest.approx([0.0, 1.0, 0.0], abs=1e-6)
    assert np.linalg.norm(forward) == pytest.approx(1.0)
    assert np.dot(forward, reference["gravity_up_vector"]) == pytest.approx(0.0)

    report = build_technical_accuracy_diagnostics(
        pose,
        spec,
        timeline,
        profile,
        _synthetic_wholebody(timeline),
        direction_reference=reference,
    )
    assert report["direction_reference_status"] == "validated_session_bound_diagnostic"
    assert report["total_score"] is None


def test_direction_reference_fails_closed_without_measurable_ready_stance() -> None:
    profile = load_technical_accuracy_profile(PROFILE_PATH)
    spec = load_poomsae_spec(SPEC_PATH)
    timeline = load_movement_timeline(TIMELINE_PATH, spec)

    missing_anchor = derive_athlete_local_direction_reference(
        _synthetic_pose(timeline["frame_count"]), spec, timeline, profile, anchor_phase="kick_apex"
    )
    assert missing_anchor["status"] == "not_derived"
    assert missing_anchor["reason"] == "movement_contract_incomplete"
    assert missing_anchor["reference"] is None

    blind = _synthetic_pose(timeline["frame_count"])
    points = np.asarray(blind["keypoints_3d_world"], dtype=float)
    points[:, [COCO_BODY_JOINTS["left_shoulder"], COCO_BODY_JOINTS["right_shoulder"]], :] = np.nan
    blind["keypoints_3d_world"] = points.tolist()

    unmeasurable = derive_athlete_local_direction_reference(blind, spec, timeline, profile)
    assert unmeasurable["status"] == "not_derived"
    assert unmeasurable["reason"] == "insufficient_valid_samples"
    assert unmeasurable["reference"] is None
    assert unmeasurable["reason"] in profile["skip_reason_codes"]


def test_detailed_hand_and_foot_quality_fail_closed_with_specific_evidence_reason() -> None:
    profile = load_technical_accuracy_profile(PROFILE_PATH)
    spec = load_poomsae_spec(SPEC_PATH)
    timeline = load_movement_timeline(TIMELINE_PATH, spec)
    pose = _synthetic_pose(timeline["frame_count"])
    valid = np.asarray(pose["reliability_valid_mask"], dtype=bool)
    valid[:, 17:23] = False
    valid[:, 91:133] = False
    pose["reliability_valid_mask"] = valid.tolist()

    report = build_technical_accuracy_diagnostics(
        pose,
        spec,
        timeline,
        profile,
        _synthetic_wholebody(timeline),
    )
    rules = {rule["metric_id"]: rule for rule in report["movements"][0]["rules"]}
    for metric_id in (
        "active_hand_fixation_stability",
        "reaction_hand_fixation_stability",
        "foot_fixation_slip_body_ratio",
        "stance_fixation_dispersion_body_ratio",
    ):
        assert rules[metric_id]["state"] == "unmeasurable"
        assert rules[metric_id]["skip_or_block_reason"] == "insufficient_valid_landmark_evidence"
        assert rules[metric_id]["decision_status"] == "no_candidate"


def test_missing_numeric_technique_target_is_blocked_not_mislabeled_as_bad_evidence() -> None:
    profile = load_technical_accuracy_profile(PROFILE_PATH)
    spec = load_poomsae_spec(SPEC_PATH)
    timeline = load_movement_timeline(TIMELINE_PATH, spec)
    report = build_technical_accuracy_diagnostics(
        _synthetic_pose(timeline["frame_count"]),
        spec,
        timeline,
        profile,
        _synthetic_wholebody(timeline),
    )
    rules = {rule["metric_id"]: rule for rule in report["movements"][0]["rules"]}
    for metric_id in ("elbow_target_angle_error_deg", "fist_target_depth_error_body_ratio"):
        assert rules[metric_id]["state"] == "blocked_missing_reference"
        assert rules[metric_id]["quality_status"] == "blocked"
        assert rules[metric_id]["skip_or_block_reason"].startswith("missing_numeric_")


def test_declared_kick_contract_has_geometry_evaluators_without_current_video_claim() -> None:
    profile = load_technical_accuracy_profile(PROFILE_PATH)
    spec = load_poomsae_spec(SPEC_PATH)
    contract = resolve_movement_accuracy_contracts(spec, profile)[13]
    pose = _synthetic_pose(30)
    arrays = {
        "points": np.asarray(pose["keypoints_3d_world"], dtype=float),
        "quality_mask": np.asarray(pose["reliability_valid_mask"], dtype=bool),
    }
    segment = {
        "start_frame": 0,
        "end_frame": 29,
        "anchors": {
            "preparation": 3,
            "kick_execution": 7,
            "kick_apex": 10,
            "rechamber": 14,
            "landing": 18,
            "punch_execution": 21,
            "fixation": 24,
        },
    }
    measurements = measure_observable_accuracy_metrics(
        arrays,
        contract,
        segment,
        profile,
        None,
        60.0,
    )
    expected = {
        "chamber_knee_height_body_ratio",
        "chamber_hip_flexion_deg",
        "chamber_knee_flexion_deg",
        "support_leg_stability",
        "kick_extension_deg",
        "kick_target_height_body_ratio",
        "kick_retraction_state",
        "kick_landing_stance_restoration",
        "kick_direction_error_deg",
    }
    assert expected <= set(measurements)
    assert contract["movement_id"] == "M14"
    assert contract["kicking_leg"] == "right"


def _raw_profile() -> dict:
    import yaml

    return yaml.safe_load(PROFILE_PATH.read_text(encoding="utf-8"))


def _synthetic_pose(frame_count: int) -> dict:
    points = np.zeros((frame_count, 133, 3), dtype=float)
    body = COCO_BODY_JOINTS
    base = {
        "nose": [0.0, 0.0, 1.75],
        "left_eye": [-0.03, 0.0, 1.78],
        "right_eye": [0.03, 0.0, 1.78],
        "left_shoulder": [-0.20, 0.0, 1.45],
        "right_shoulder": [0.20, 0.0, 1.45],
        "left_elbow": [-0.35, 0.0, 1.20],
        "right_elbow": [0.35, 0.0, 1.20],
        "left_wrist": [-0.45, 0.0, 1.05],
        "right_wrist": [0.45, 0.0, 1.05],
        "left_hip": [-0.14, 0.0, 0.95],
        "right_hip": [0.14, 0.0, 0.95],
        "left_knee": [-0.14, 0.05, 0.52],
        "right_knee": [0.14, -0.05, 0.52],
        "left_ankle": [-0.14, 0.15, 0.08],
        "right_ankle": [0.14, -0.15, 0.08],
    }
    for name, value in base.items():
        points[:, body[name], :] = value
    points[:, 17:23, :] = np.array([[-0.14, 0.28, 0.04], [-0.10, 0.28, 0.04], [-0.14, 0.10, 0.04], [0.14, -0.02, 0.04], [0.10, -0.02, 0.04], [0.14, -0.20, 0.04]])
    points[:, 23:91, :] = [0.0, 0.0, 1.75]
    points[:, 23 + 8, :] = [0.0, 0.0, 1.68]
    points[:, 23 + 17 : 23 + 22, :] = [-0.03, 0.0, 1.82]
    points[:, 23 + 22 : 23 + 27, :] = [0.03, 0.0, 1.82]
    points[:, 23 + 27 : 23 + 36, :] = [0.0, 0.03, 1.76]
    points[:, 23 + 36 : 23 + 42, :] = [-0.03, 0.0, 1.78]
    points[:, 23 + 42 : 23 + 48, :] = [0.03, 0.0, 1.78]
    for offset, sign in ((91, -1.0), (112, 1.0)):
        wrist = np.array([0.45 * sign, 0.0, 1.05])
        points[:, offset, :] = wrist
        for local_index in range(1, 21):
            finger = min(4, (local_index - 1) // 4)
            depth = 0.025 * (1 + (local_index - 1) % 4)
            points[:, offset + local_index, :] = wrist + [0.008 * (finger - 2), depth, 0.0]
    shape = (frame_count, 133)
    return {
        "keypoints_3d_world": points.tolist(),
        "reliability_valid_mask": np.ones(shape, dtype=bool).tolist(),
        "used_cameras": np.full(shape, 2, dtype=int).tolist(),
        "reprojection_error": np.full(shape, 2.0).tolist(),
    }


def _synthetic_wholebody(timeline: dict) -> dict:
    metrics = {
        "head_torso_yaw_mismatch_deg": (5.0, "deg"),
        "shoulder_hip_twist_deg": (5.0, "deg"),
        "torso_lean_p95_deg": (5.0, "deg"),
        "wrist_forearm_alignment_deg": (5.0, "deg"),
        "hand_foot_settle_difference_sec": (0.05, "sec"),
        "fixation_wrist_jitter_ratio": (0.01, "body_scale"),
        "executing_wrist_height_torso_ratio": (0.5, "torso_height"),
        "executing_elbow_deg": (165.0, "deg"),
        "reaction_hand_hip_distance_ratio": (0.1, "body_scale"),
        "stance_span_ratio": (0.5, "body_scale"),
        "front_knee_deg": (165.0, "deg"),
    }
    return {
        "status": "wholebody_diagnostics_only",
        "movement_timeline_id": timeline["timeline_id"],
        "movements": [
            {
                "movement_id": segment["movement_id"],
                "metrics": [
                    {"metric_id": metric_id, "value": value, "unit": unit, "measurement_evidence": {"scope": "synthetic"}}
                    for metric_id, (value, unit) in metrics.items()
                ],
            }
            for segment in timeline["segments"]
        ],
    }
