"""Metric registry, landmark declarations and known-geometry evaluator tests.

These tests pin the geometry of each evaluator family to a pose whose answer is
known by construction (angle, distance, side, axis, sign), so a left/right swap, a
wrong lean axis or a flipped roll sign fails here even when every synthetic
threshold-classification case still passes.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from test_technical_accuracy import _arrays, _raw_profile, _rotate_points, _synthetic_pose, _synthetic_wholebody

from src.data_structures import COCO_BODY_JOINTS, COCO_LEFT_HAND_OFFSET, COCO_RIGHT_HAND_OFFSET
from src.poomsae_scoring import ScoringContractError, load_movement_timeline, load_poomsae_spec
from src.poomsae_scoring import technical_accuracy as accuracy
from src.poomsae_scoring import technical_accuracy_metrics as metrics
from src.poomsae_scoring.technical_accuracy import (
    build_technical_accuracy_diagnostics,
    load_technical_accuracy_profile,
    resolve_movement_accuracy_contracts,
    validate_athlete_local_direction_reference,
    validate_technical_accuracy_profile,
)
from src.poomsae_scoring.technical_accuracy_metrics import (
    CONTRACT_FACT,
    FIXATION_ANCHOR,
    KICK_PHASE_ANCHORS,
    METRIC_REGISTRY,
    MetricSpec,
    measure_observable_accuracy_metrics,
    resolve_metric_landmarks,
)
from src.poomsae_scoring.technical_accuracy_validation import _mirror_pose
from src.poomsae_scoring.fixation import prepare_fixation_signals


ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = ROOT / "config" / "scoring" / "engineering" / "taegeuk_1_wholebody_diagnostics_v3.yaml"
SPEC_PATH = ROOT / "config" / "scoring" / "poomsae" / "taegeuk_1_jang_v0_draft.yaml"
TIMELINE_PATH = ROOT / "config" / "scoring" / "timelines" / "poomsae1_zed2i_rgbd_rerun_20260802_draft.yaml"
BODY = COCO_BODY_JOINTS
OTHER = {"left": "right", "right": "left"}
SEGMENT = {"start_frame": 0, "end_frame": 39, "sequence_index": 1, "anchors": {"preparation": 5, "execution": 15, "fixation": 22}}


@pytest.fixture(scope="module")
def profile() -> dict:
    return load_technical_accuracy_profile(PROFILE_PATH)


@pytest.fixture(scope="module")
def contracts(profile: dict) -> list[dict]:
    return resolve_movement_accuracy_contracts(load_poomsae_spec(SPEC_PATH), profile)


def _points(frame_count: int = 40) -> np.ndarray:
    return np.asarray(_synthetic_pose(frame_count)["keypoints_3d_world"], dtype=float)


def _pose_from(points: np.ndarray) -> dict:
    pose = _synthetic_pose(points.shape[0])
    pose["keypoints_3d_world"] = points.tolist()
    return pose


def _direction() -> dict:
    return validate_athlete_local_direction_reference({
        "schema_version": 1, "session_id": "synthetic", "reference_pose_sha256": "a" * 64,
        "gravity_up_vector": [0.0, 0.0, 1.0], "initial_forward_vector": [0.0, 1.0, 0.0],
        "basis_source": "manually_declared_session_bound", "provenance": "synthetic test binding",
        "quality_status": "validated_diagnostic_reference",
    })


# --- Known geometry per evaluator family -------------------------------------


@pytest.mark.parametrize("side", ["left", "right"])
def test_knee_and_elbow_included_angles_are_measured_on_the_named_side(side: str, profile: dict, contracts: list[dict]) -> None:
    points = _points()
    x = -1.0 if side == "left" else 1.0
    # Both legs: hip above knee above ankle; only ``side`` bends to a 120 deg included angle.
    for leg_side, sign in (("left", -1.0), ("right", 1.0)):
        points[:, BODY[f"{leg_side}_hip"]] = [0.14 * sign, 0.0, 0.95]
        points[:, BODY[f"{leg_side}_knee"]] = [0.14 * sign, 0.0, 0.52]
        points[:, BODY[f"{leg_side}_ankle"]] = [0.14 * sign, 0.0, 0.08]
        points[:, BODY[f"{leg_side}_shoulder"]] = [0.20 * sign, 0.0, 1.45]
        points[:, BODY[f"{leg_side}_elbow"]] = [0.20 * sign, 0.0, 1.20]
        points[:, BODY[f"{leg_side}_wrist"]] = [0.20 * sign, 0.0, 0.95]
    knee = np.array([0.14 * x, 0.0, 0.52])
    points[:, BODY[f"{side}_ankle"]] = knee + 0.44 * np.array([0.0, np.sin(np.deg2rad(120)), np.cos(np.deg2rad(120))])
    elbow = np.array([0.20 * x, 0.0, 1.20])
    points[:, BODY[f"{side}_wrist"]] = elbow + 0.25 * np.array([0.0, 1.0, 0.0])  # 90 deg elbow
    arrays = {"points": points}

    assert metrics._leg_angle(arrays, 0, side) == pytest.approx(120.0, abs=1e-6)
    assert metrics._leg_angle(arrays, 0, OTHER[side]) == pytest.approx(180.0, abs=1e-6)
    assert metrics._arm_angle(arrays, 0, side) == pytest.approx(90.0, abs=1e-6)
    assert metrics._arm_angle(arrays, 0, OTHER[side]) == pytest.approx(180.0, abs=1e-6)

    # Through the measurement module the front knee is the lead leg's and the elbow the active arm's.
    contract = next(item for item in contracts if item["lead_leg"] == side and item["active_arm"] == side)
    result = measure_observable_accuracy_metrics(_arrays(_pose_from(points)), contract, SEGMENT, profile, None, 30.0)
    assert result["front_knee_flexion_deg"]["value"] == pytest.approx(120.0, abs=1e-6)
    assert result["rear_knee_flexion_deg"]["value"] == pytest.approx(180.0, abs=1e-6)
    assert result["elbow_flexion_deg"]["value"] == pytest.approx(90.0, abs=1e-6)


def _upper_body_indices() -> list[int]:
    # Everything above the hips: head, shoulders, arms, face and hands (hips and legs stay).
    return [*range(0, 11), *range(23, 133)]


def test_forward_lean_and_lateral_bend_are_measured_about_their_own_axes() -> None:
    hip_centre = np.array([0.0, 0.0, 0.95])
    forward_lean = _points()
    # Rotating about +x by -10 deg moves the shoulders towards +y, the athlete's front.
    _rotate_points(forward_lean, _upper_body_indices(), hip_centre, np.array([1.0, 0.0, 0.0]), -10.0)
    arrays = {"points": forward_lean}
    assert metrics._torso_forward_lean(arrays, 0) == pytest.approx(10.0, abs=1e-6)
    assert accuracy._torso_lateral_bend(arrays, 0) == pytest.approx(0.0, abs=1e-6)

    side_bend = _points()
    # Rotating about +y by +10 deg moves the shoulders towards +x, the athlete's right.
    _rotate_points(side_bend, _upper_body_indices(), hip_centre, np.array([0.0, 1.0, 0.0]), 10.0)
    arrays = {"points": side_bend}
    assert accuracy._torso_lateral_bend(arrays, 0) == pytest.approx(10.0, abs=1e-6)
    assert metrics._torso_forward_lean(arrays, 0) == pytest.approx(0.0, abs=1e-6)


def test_roll_and_yaw_signs_follow_the_athletes_sides(profile: dict, contracts: list[dict]) -> None:
    points = _points()
    head = [*metrics.FACE_RIGHT_EYE_INDICES, *metrics.FACE_LEFT_EYE_INDICES, *range(23 + 17, 23 + 36), 23 + 8]
    forward = np.array([0.0, 1.0, 0.0])
    # Head tilted towards the athlete's LEFT shoulder: the right eye rises -> +10.
    tilted = points.copy()
    _rotate_points(tilted, head, np.array([0.0, 0.0, 1.78]), forward, -10.0)
    assert accuracy._head_roll({"points": tilted}, 0) == pytest.approx(10.0, abs=1e-6)
    tilted_other = points.copy()
    _rotate_points(tilted_other, head, np.array([0.0, 0.0, 1.78]), forward, 10.0)
    assert accuracy._head_roll({"points": tilted_other}, 0) == pytest.approx(-10.0, abs=1e-6)
    # Right shoulder raised: positive shoulder roll; left raised: negative.
    raised = points.copy()
    raised[:, BODY["right_shoulder"], 2] += 0.4 * np.tan(np.deg2rad(5.0))
    assert accuracy._line_roll({"points": raised}, 0, "shoulder") == pytest.approx(5.0, abs=1e-6)
    raised = points.copy()
    raised[:, BODY["left_shoulder"], 2] += 0.4 * np.tan(np.deg2rad(5.0))
    assert accuracy._line_roll({"points": raised}, 0, "shoulder") == pytest.approx(-5.0, abs=1e-6)
    # Head yawed 30 deg towards the athlete's right (+x): forward vector points right,
    # and the direction-bound yaw error is 30 deg.
    turned = points.copy()
    head_and_ears = [*head, BODY["nose"], BODY["left_ear"], BODY["right_ear"]]
    _rotate_points(turned, head_and_ears, np.array([0.0, 0.0, 1.78]), np.array([0.0, 0.0, 1.0]), -30.0)
    head_forward = metrics._head_forward({"points": turned}, 0)
    assert head_forward[0] > 0
    contract = next(item for item in contracts if item["semantic_action_direction"] == "forward") if any(
        item["semantic_action_direction"] == "forward" for item in contracts
    ) else None
    if contract is not None:
        result = measure_observable_accuracy_metrics(_arrays(_pose_from(turned)), contract, SEGMENT, profile, _direction(), 30.0)
        assert result["head_target_yaw_error_deg"]["value"] == pytest.approx(30.0, abs=1e-6)


@pytest.mark.parametrize("yaw_deg", [0.0, 37.0, -120.0])
def test_stance_length_and_width_are_separated_in_a_rotated_body(yaw_deg: float) -> None:
    points = _points(5)
    length, width = 0.50, 0.30
    # Lead (left) foot forward and to the left, rear (right) foot behind and to the right.
    for side, along, across in (("left", length / 2, -width / 2), ("right", -length / 2, width / 2)):
        centre = np.array([across, along, 0.04])
        for name, offset in (("heel", -0.10), ("big_toe", 0.10), ("small_toe", 0.10)):
            points[:, metrics.COCO_FOOT_JOINTS[f"{side}_{name}"]] = centre + [0.0, offset, 0.0]
    _rotate_points(points, list(range(133)), np.zeros(3), np.array([0.0, 0.0, 1.0]), yaw_deg)
    radians = np.deg2rad(yaw_deg)
    # The movement target rotates with the scene (+y rotated by the same yaw).
    target = np.array([-np.sin(radians), np.cos(radians), 0.0])
    stance = metrics._stance_geometry({"points": points}, np.arange(5), "left", "right", target)
    assert stance["length"] == pytest.approx(length, abs=1e-9)
    assert stance["width"] == pytest.approx(width, abs=1e-9)
    assert stance["front_back_order_match"] is True
    # A torso turned 30 deg against the stance (walking-stance block) must not leak
    # stance length into width: the frame is the movement target, not the torso.
    twisted = points.copy()
    _rotate_points(twisted, list(range(0, 17)) + list(range(23, 133)), np.zeros(3), np.array([0.0, 0.0, 1.0]), 30.0)
    turned = metrics._stance_geometry({"points": twisted}, np.arange(5), "left", "right", target)
    assert turned["length"] == pytest.approx(length, abs=1e-9)
    assert turned["width"] == pytest.approx(width, abs=1e-9)
    # No direction reference, no frame: no stance values.
    assert metrics._stance_geometry({"points": points}, np.arange(5), "left", "right", None) == {}


def test_dispersion_of_a_known_oscillation(profile: dict) -> None:
    frames = np.arange(21)
    phase = np.linspace(0.0, 4.0 * np.pi, len(frames))
    amplitude = 0.02
    points = _points(len(frames))
    offset = amplitude * np.sin(phase)
    hand = slice(COCO_LEFT_HAND_OFFSET, COCO_LEFT_HAND_OFFSET + 21)
    points[:, hand, 1] += offset[:, None]
    expected = float(np.percentile(np.abs(offset - np.median(offset)), 95))
    assert metrics._hand_dispersion({"points": points}, frames, "left", 1.0) == pytest.approx(expected, abs=1e-12)
    # The other hand did not move.
    assert metrics._hand_dispersion({"points": points}, frames, "right", 1.0) == pytest.approx(0.0, abs=1e-12)

    # Shoulder-line yaw oscillating by +-4 deg: p95 angular dispersion from the median line.
    yaw = 4.0 * np.sin(phase)
    for index, angle in enumerate(yaw):
        rotation = np.deg2rad(angle)
        for name, sign in (("left_shoulder", -1.0), ("right_shoulder", 1.0)):
            points[index, BODY[name]] = [0.20 * sign * np.cos(rotation), 0.20 * sign * np.sin(rotation), 1.45]
    expected_deg = float(np.percentile(np.abs(yaw - np.median(yaw)), 95))
    measured = metrics._orientation_dispersion({"points": points}, frames, "shoulder")
    assert measured == pytest.approx(expected_deg, abs=0.05)


def test_side_match_booleans_follow_which_side_actually_moved(profile: dict, contracts: list[dict]) -> None:
    before, after = np.arange(0, 5), np.arange(10, 15)
    for moved, expected in (("left", True), ("right", False)):
        points = _points(15)
        points[10:, BODY[f"{moved}_wrist"], 1] += 0.3
        assert metrics._arm_side_match({"points": points}, before, after, "left", "right", noise_floor=0.05) is expected

    contract = next(item for item in contracts if item["expected_moving_foot"] is not None)
    lead, rear = contract["lead_leg"], contract["rear_or_support_leg"]
    for moved, expected in ((lead, True), (rear, False)):
        points = _points()
        points[12:, BODY[f"{moved}_ankle"], 1] += 0.25
        result = measure_observable_accuracy_metrics(_arrays(_pose_from(points)), contract, SEGMENT, profile, None, 30.0)
        assert result["moving_foot_side_match"]["value"] is expected


# --- Mirror symmetry ----------------------------------------------------------

# Measurement-module values whose sign is defined on the body's left->right axis.
SIGNED_UNDER_MIRROR = {
    "active_arm_target_lateral_offset_body_ratio",  # + towards the athlete's right
    # Sign follows the palm-normal handedness (cross product), not anatomy: the same
    # anatomical flexion of the left and right wrist has opposite signs. Reported as a
    # finding; the active screen is abs_max so the evaluation is unaffected.
    "wrist_flexion_extension_proxy_deg",
}


def test_mirroring_left_right_flips_signed_metrics_and_keeps_unsigned_ones(profile: dict, contracts: list[dict]) -> None:
    pose = _synthetic_pose(40)
    points = np.asarray(pose["keypoints_3d_world"], dtype=float)
    points += np.random.default_rng(3).normal(0.0, 0.01, points.shape)
    points[:, BODY["right_shoulder"], 2] += 0.03
    pose["keypoints_3d_world"] = points.tolist()
    mirrored = _mirror_pose(pose)
    contract = contracts[1]
    mirrored_contract = {key: OTHER.get(value, value) if isinstance(value, str) else value for key, value in contract.items()}

    original = measure_observable_accuracy_metrics(_arrays(pose), contract, SEGMENT, profile, None, 30.0)
    flipped = measure_observable_accuracy_metrics(_arrays(mirrored), mirrored_contract, SEGMENT, profile, None, 30.0)
    checked = 0
    for metric_id, measurement in original.items():
        value, other = measurement["value"], flipped[metric_id]["value"]
        if isinstance(value, float):
            expected = -value if metric_id in SIGNED_UNDER_MIRROR else value
            assert other == pytest.approx(expected, abs=1e-6), metric_id
            checked += 1
        else:
            assert other == value, metric_id
    assert checked >= 40
    assert all(abs(original[metric_id]["value"]) > 1e-6 for metric_id in SIGNED_UNDER_MIRROR)

    arrays, mirror_arrays = _arrays(pose), _arrays(mirrored)
    for signed in (accuracy._head_roll, accuracy._torso_lateral_bend,
                   lambda a, f: accuracy._line_roll(a, f, "shoulder"), lambda a, f: accuracy._line_roll(a, f, "hip")):
        assert abs(signed(arrays, 20)) > 1e-6
        assert signed(mirror_arrays, 20) == pytest.approx(-signed(arrays, 20), abs=1e-9)
    for unsigned in (accuracy._head_pitch_unsigned, metrics._torso_forward_lean):
        assert unsigned(mirror_arrays, 20) == pytest.approx(unsigned(arrays, 20), abs=1e-9)


# --- Registry contract ---------------------------------------------------------


def test_registry_rejects_records_without_unit_or_landmarks() -> None:
    with pytest.raises(ValueError, match="no valid unit"):
        MetricSpec("new_metric_deg", "", ("torso",), FIXATION_ANCHOR, "torso", "metrics")
    with pytest.raises(ValueError, match="must declare its landmarks"):
        MetricSpec("new_metric_deg", "deg", None, FIXATION_ANCHOR, "torso", "metrics")
    with pytest.raises(ValueError, match="declares no landmarks"):
        MetricSpec("new_metric_deg", "deg", (), FIXATION_ANCHOR, "torso", "metrics")
    with pytest.raises(ValueError, match="unknown landmark"):
        MetricSpec("new_metric_deg", "deg", ("elbow:pinky",), FIXATION_ANCHOR, "torso", "metrics")
    with pytest.raises(ValueError, match="value kind"):
        MetricSpec("new_metric_deg", "deg", ("torso",), FIXATION_ANCHOR, "torso", "metrics", kind="boolean")
    with pytest.raises(ValueError, match="duplicate"):
        metrics.build_metric_registry([METRIC_REGISTRY["shoulder_roll_deg"]] * 2)


def test_profile_and_registry_must_match_exactly(monkeypatch) -> None:
    raw = _raw_profile()
    raw["metric_catalog"]["upper_body"].append("catalogued_without_record_deg")
    raw["measurement_only_reasons"]["catalogued_without_record_deg"] = "test"
    with pytest.raises(ScoringContractError, match="no metric registry record"):
        validate_technical_accuracy_profile(raw)

    monkeypatch.setitem(
        METRIC_REGISTRY, "registered_without_yaml_deg",
        MetricSpec("registered_without_yaml_deg", "deg", ("torso",), FIXATION_ANCHOR, "torso", "metrics"),
    )
    with pytest.raises(ScoringContractError, match="missing from metric_catalog"):
        validate_technical_accuracy_profile(_raw_profile())


def test_evaluator_status_is_true_only_for_ids_an_evaluator_produces(profile: dict, contracts: list[dict]) -> None:
    points = _points()
    # One wrist and one ankle travel well above the noise floor so side matches resolve.
    points[12:, [BODY["left_wrist"], BODY["left_ankle"]], 1] += 0.3
    pose = _pose_from(points)
    arrays = _arrays(pose)
    # As in the pipeline: the hold detector's signals come with the pose arrays.
    arrays["fixation_signals"] = prepare_fixation_signals(points, 30.0, quality=pose)
    kick_segment = {"start_frame": 0, "end_frame": 39, "sequence_index": 14, "anchors": {
        "preparation": 3, "kick_execution": 7, "kick_apex": 10, "rechamber": 14, "landing": 18,
        "punch_execution": 21, "fixation": 24}}
    produced: dict[str, set] = {}
    for contract in contracts:
        segment = kick_segment if "ap_chagi" in contract["technique_types"] else SEGMENT
        for direction in (None, _direction()):
            for metric_id, measurement in measure_observable_accuracy_metrics(arrays, contract, segment, profile, direction, 30.0).items():
                produced.setdefault(metric_id, set()).add(measurement["value"] is not None)
    module_ids = {metric_id for metric_id, spec in METRIC_REGISTRY.items() if spec.producer == "metrics"}
    placeholders = {metric_id for metric_id, spec in METRIC_REGISTRY.items() if spec.producer == "placeholder"}
    assert set(produced) == module_ids | placeholders
    assert not [metric_id for metric_id in sorted(module_ids) if True not in produced[metric_id]]
    assert all(produced[metric_id] == {False} for metric_id in placeholders)
    assert placeholders == {"elbow_target_angle_error_deg", "fist_target_depth_error_body_ratio", "kick_direction_error_deg"}
    for rule in profile["resolved_rules"]:
        spec = METRIC_REGISTRY[rule["metric_id"]]
        if rule["status"] != "not_observable_with_current_pipeline":
            assert (rule["measurement_evaluator_status"] == "implemented") == spec.implemented


WINDOW_NAMES = {
    "fixation_anchor": metrics.FIXATION_ANCHOR,
    "post_fixation_hold": metrics.POST_FIXATION_HOLD,
    "post_fixation_tail_to_segment_end": metrics.POST_FIXATION_TAIL,
    "segment_start_to_fixation": metrics.SEGMENT_START_TO_FIXATION,
    "preparation_anchor": metrics.PREPARATION_ANCHOR,
    ("preparation_anchor", "fixation_anchor"): metrics.PREPARATION_AND_FIXATION_ANCHORS,
    ("preparation_anchor", "execution_anchor"): metrics.PREPARATION_AND_EXECUTION_ANCHORS,
}


def test_registry_window_matches_the_window_the_evaluator_reports(profile: dict, contracts: list[dict]) -> None:
    arrays = _arrays(_synthetic_pose(40))
    for contract in (contracts[0], contracts[13]):
        for metric_id, measurement in measure_observable_accuracy_metrics(arrays, contract, SEGMENT, profile, _direction(), 30.0).items():
            window = (measurement.get("evidence") or {}).get("window")
            registered = METRIC_REGISTRY[metric_id].window
            if window is None:
                assert registered in {CONTRACT_FACT, KICK_PHASE_ANCHORS} or METRIC_REGISTRY[metric_id].producer == "placeholder", metric_id
            else:
                key = window["name"] if isinstance(window, dict) else tuple(item["name"] for item in window)
                assert WINDOW_NAMES[key] == registered, metric_id


# --- Landmark declarations (step B) --------------------------------------------


def _measure_all(points: np.ndarray, contract: dict, profile: dict, segment: dict) -> dict:
    arrays = {"points": points, "quality_mask": np.ones(points.shape[:2], dtype=bool)}
    return {
        metric_id: measurement["value"]
        for metric_id, measurement in accuracy._movement_measurements(
            arrays, contract, segment, None, profile, _direction(), 30.0
        ).items()
    }


def test_landmarks_no_evaluator_reads_are_not_needed_by_any_declaration(profile: dict, contracts: list[dict]) -> None:
    """Completeness: blanking any undeclared landmark leaves the metric unchanged.

    The quality mask stays valid so only the evaluators' own reads are tested; a
    metric that reads a point it does not declare changes value or vanishes here.
    """
    base = _points()
    base += np.random.default_rng(11).normal(0.0, 0.004, base.shape)
    kick_segment = {"start_frame": 0, "end_frame": 39, "sequence_index": 14, "anchors": {
        "preparation": 3, "kick_execution": 7, "kick_apex": 10, "rechamber": 14, "landing": 18,
        "punch_execution": 21, "fixation": 24}}
    checked = {contract["movement_id"]: contract for contract in (contracts[0], contracts[1], contracts[13])}
    offenders = []
    for contract in checked.values():
        segment = kick_segment if "ap_chagi" in contract["technique_types"] else SEGMENT
        reference = _measure_all(base, contract, profile, segment)
        declared = {metric_id: set(resolve_metric_landmarks(metric_id, contract)) for metric_id in reference}
        for index in range(133):
            blanked = base.copy()
            blanked[:, index] = np.nan
            values = _measure_all(blanked, contract, profile, segment)
            for metric_id, value in reference.items():
                if index in declared[metric_id] or METRIC_REGISTRY[metric_id].quality_group == "quality_metric":
                    continue
                same = value == values[metric_id] or (
                    isinstance(value, float) and isinstance(values[metric_id], float)
                    and np.isclose(value, values[metric_id], atol=1e-9)
                )
                if not same:
                    offenders.append((contract["movement_id"], metric_id, index))
    assert not offenders, offenders[:20]


def test_audited_declarations_cover_what_the_evaluators_read(contracts: list[dict]) -> None:
    kick = next(contract for contract in contracts if "ap_chagi" in contract["technique_types"])
    side = kick["kicking_leg"]
    leg = {BODY[f"{side}_{part}"] for part in ("hip", "knee", "ankle")}
    assert leg <= set(resolve_metric_landmarks("kick_extension_deg", kick))
    assert {BODY[f"{side}_knee"], BODY[f"{side}_hip"]} <= set(resolve_metric_landmarks("chamber_hip_flexion_deg", kick))
    assert not {BODY["left_elbow"], BODY["right_elbow"]} & set(resolve_metric_landmarks("chamber_hip_flexion_deg", kick))
    m01 = contracts[0]
    lead = m01["lead_leg"]
    assert {BODY[f"{lead}_knee"], BODY[f"{lead}_ankle"]} <= set(resolve_metric_landmarks("step_length_error_body_ratio", m01))
    assert {BODY["left_ankle"], BODY["right_ankle"]} <= set(resolve_metric_landmarks("body_height_change_during_fixation_ratio", m01))
    assert set(resolve_metric_landmarks("head_target_yaw_error_deg", m01)) == {BODY["nose"], BODY["left_ear"], BODY["right_ear"]}
    # Torso facing comes from the shoulder labels alone; the face is not read (AD-049).
    assert not set(metrics._UNSIDED_GROUPS["face_eyes"]) & set(resolve_metric_landmarks("torso_forward_backward_lean_deg", m01))
    active = m01["active_arm"]
    offset = COCO_LEFT_HAND_OFFSET if active == "left" else COCO_RIGHT_HAND_OFFSET
    other_offset = COCO_RIGHT_HAND_OFFSET if active == "left" else COCO_LEFT_HAND_OFFSET
    for metric_id in ("wrist_flexion_extension_proxy_deg", "fist_or_hand_orientation_proxy_deg", "active_hand_fixation_stability"):
        landmarks = set(resolve_metric_landmarks(metric_id, m01))
        assert offset + 9 in landmarks, metric_id  # active middle MCP
        assert not landmarks & set(range(other_offset, other_offset + 21)), metric_id


def test_occluded_reaction_hand_no_longer_discards_the_active_hand_measurement(profile: dict) -> None:
    spec = load_poomsae_spec(SPEC_PATH)
    timeline = load_movement_timeline(TIMELINE_PATH, spec)
    pose = _synthetic_pose(timeline["frame_count"])
    contract = resolve_movement_accuracy_contracts(spec, profile)[0]
    reaction_offset = COCO_LEFT_HAND_OFFSET if contract["chamber_or_reaction_arm"] == "left" else COCO_RIGHT_HAND_OFFSET
    valid = np.asarray(pose["reliability_valid_mask"], dtype=bool)
    valid[:, reaction_offset:reaction_offset + 21] = False
    pose["reliability_valid_mask"] = valid.tolist()
    report = build_technical_accuracy_diagnostics(pose, spec, timeline, profile, _synthetic_wholebody(timeline))
    rules = {rule["metric_id"]: rule for rule in report["movements"][0]["rules"]}
    for metric_id in ("active_hand_fixation_stability", "fist_target_height_error_body_ratio", "late_post_fixation_correction"):
        assert rules[metric_id]["measured"] is True, metric_id
        assert rules[metric_id]["evidence"]["required_landmark_valid_ratio"] == 1.0
    assert rules["reaction_hand_fixation_stability"]["state"] == "unmeasurable"


# --- Worked example: adding one rule -------------------------------------------


def _active_wrist_height_above_pelvis(arrays: dict, frame: int, contract: dict) -> float | None:
    """Hypothetical rule: active wrist height above the hip centre, in torso lengths."""
    side = contract["active_arm"]
    if side not in {"left", "right"}:
        return None
    wrist = arrays["points"][frame, BODY[f"{side}_wrist"]]
    shoulder_centre, hip_centre = metrics._centres(arrays, frame)
    if shoulder_centre is None or not np.all(np.isfinite(wrist)):
        return None
    torso = float(np.linalg.norm(shoulder_centre - hip_centre))
    return None if torso <= 1e-8 else float((wrist[2] - hip_centre[2]) / torso)


def test_worked_example_adding_one_rule_flows_to_a_report_row(monkeypatch) -> None:
    metric_id = "example_active_wrist_height_above_pelvis_torso_ratio"
    # 1. One registry record: unit, landmarks by role, window, gate, evaluator.
    monkeypatch.setitem(METRIC_REGISTRY, metric_id, MetricSpec(
        metric_id, "torso_length", ("wrist:active", "torso"), FIXATION_ANCHOR, "rule_layer",
        "frame_function", frame_value=_active_wrist_height_above_pelvis,
    ))
    # 2. YAML entries: catalog family, status, threshold (same unit as the record).
    raw = _raw_profile()
    raw["metric_catalog"]["upper_body"].append(metric_id)
    raw["active_rules"].append(metric_id)
    raw["thresholds"][metric_id] = {"operator": "max", "value": 0.5, "uncertainty_band": 0.02, "unit": "torso_length"}
    profile = validate_technical_accuracy_profile(raw)

    spec = load_poomsae_spec(SPEC_PATH)
    timeline = load_movement_timeline(TIMELINE_PATH, spec)
    report = build_technical_accuracy_diagnostics(
        _synthetic_pose(timeline["frame_count"]), spec, timeline, profile, _synthetic_wholebody(timeline)
    )
    inventory = {rule["metric_id"]: rule for rule in report["rule_inventory"]}
    assert inventory[metric_id]["measurement_evaluator_status"] == "implemented"
    assert inventory[metric_id]["status"] == "active_diagnostic"
    row = next(rule for rule in report["movements"][0]["rules"] if rule["metric_id"] == metric_id)
    # Synthetic pose: wrist 1.05 m, hip centre 0.95 m, torso length 0.50 m -> 0.20.
    assert row["value"] == pytest.approx(0.20, abs=1e-9)
    assert row["unit"] == "torso_length"
    assert row["state"] == "active_diagnostic"
    assert row["evaluation"] == "within_screening_range"
    assert row["decision_status"] == "no_candidate"
    assert row["evidence"]["required_landmark_valid_ratio"] == 1.0
    assert report["summary"]["rule_count"] == len(raw_catalog_ids(raw))

    # A threshold in another unit than the record is refused at load.
    wrong = deepcopy(raw)
    wrong["thresholds"][metric_id]["unit"] = "arm_length"
    with pytest.raises(ScoringContractError, match="does not match the measurement unit"):
        validate_technical_accuracy_profile(wrong)


def raw_catalog_ids(raw: dict) -> list[str]:
    return [metric_id for ids in raw["metric_catalog"].values() for metric_id in ids]


# --- One landmark gate: every read landmark, over the frames actually read ----


def _rear_knee_case(profile: dict, contracts: list[dict], invalid: dict[str, range]) -> dict:
    contract = next(item for item in contracts if item["rear_or_support_leg"] == "right")
    arrays = _arrays(_pose_from(_points()))
    for joint, frames in invalid.items():
        arrays["quality_mask"][list(frames), BODY[joint]] = False
    return measure_observable_accuracy_metrics(arrays, contract, SEGMENT, profile, None, 30.0)["rear_knee_flexion_deg"]


def test_a_poor_required_landmark_is_not_rescued_by_well_observed_others(profile: dict, contracts: list[dict]) -> None:
    # Fixation 22, radius 5: the metric reads frames 17..27. The rear knee is valid on
    # 6 of those 11 frames (0.55); averaged with hip and ankle it would be 0.85.
    measurement = _rear_knee_case(profile, contracts, {"right_knee": range(17, 22)})
    assert measurement["value"] is None
    assert measurement["not_measurable_reason"] == "required_landmark_quality_below_minimum"
    gate = measurement["evidence"]["landmark_gate"]
    assert gate["weakest_landmark_index"] == BODY["right_knee"]
    assert gate["weakest_landmark_valid_ratio"] == pytest.approx(6 / 11)
    assert (gate["gated_start_frame"], gate["gated_end_frame"]) == (17, 27)
    assert gate["gated_frames_source"] == "evidence_window"


def test_occlusion_outside_the_read_frames_does_not_discard_the_measurement(profile: dict, contracts: list[dict]) -> None:
    outside = [*range(0, 17), *range(28, 40)]
    # Over the whole segment knee and ankle would be valid on 11 of 40 frames.
    measurement = _rear_knee_case(profile, contracts, {"right_knee": outside, "right_ankle": outside})
    assert measurement["value"] is not None
    gate = measurement["evidence"]["landmark_gate"]
    assert gate["weakest_landmark_valid_ratio"] == pytest.approx(1.0)
    assert set(gate["landmark_valid_ratios"]) == {str(BODY[f"right_{name}"]) for name in ("hip", "knee", "ankle")}


def test_gate_is_idempotent_and_records_the_minimum(profile: dict, contracts: list[dict]) -> None:
    contract = next(item for item in contracts if item["rear_or_support_leg"] == "right")
    arrays = _arrays(_pose_from(_points()))
    arrays["quality_mask"][20:23, BODY["right_hip"]] = False
    result = measure_observable_accuracy_metrics(arrays, contract, SEGMENT, profile, None, 30.0)
    before = deepcopy(result["rear_knee_flexion_deg"])
    metrics.apply_landmark_quality_gate(result, arrays, SEGMENT, profile, contract)
    assert result["rear_knee_flexion_deg"] == before
    assert before["evidence"]["required_landmark_valid_ratio"] == pytest.approx(8 / 11)
    assert before["evidence"]["minimum_required_landmark_valid_ratio"] == pytest.approx(0.75)


def test_final_pose_conformance_is_false_when_the_stance_does_not_match(profile: dict, contracts: list[dict]) -> None:
    # Refined Poomsae run, M05: the stance did not match but the final pose was reported
    # as conforming, because the metric only checked that values existed.
    points = _points()
    points[:, BODY["left_ankle"], 0] -= 0.6
    points[:, 17:20, 0] -= 0.6
    arrays = _arrays(_pose_from(points))
    checked = 0
    for contract in contracts:
        if contract["stance_width_expectation"] is None:
            continue
        result = measure_observable_accuracy_metrics(arrays, contract, SEGMENT, profile, _direction(), 30.0)
        stance = result["expected_stance_type_match"]["value"]
        conformance = result["final_pose_geometry_conformance"]["value"]
        if stance is False:
            checked += 1
            assert conformance is False
    assert checked > 0


def _with_signals(points: np.ndarray) -> dict:
    pose = _pose_from(points)
    arrays = _arrays(pose)
    arrays["fixation_signals"] = prepare_fixation_signals(points, 30.0, quality=pose)
    return arrays


def test_without_an_observed_hold_stability_metrics_are_withheld_not_flagged(profile: dict, contracts: list[dict]) -> None:
    # Refined Poomsae run, M01/M03/M05: the blocking hand keeps moving after the technique
    # ends, so there is no hold; "unstable fixation" would re-report that motion.
    points = _points()
    times = np.arange(points.shape[0]) / 30.0
    moving = np.clip(times - 0.1, 0, None) * 0.6
    points[:, BODY["left_wrist"], 1] += moving
    points[:, 91:112, 1] += moving[:, None]
    result = measure_observable_accuracy_metrics(_with_signals(points), contracts[0], SEGMENT, profile, _direction(), 30.0)
    assert result["fixation_hold_observed"]["value"] is False
    assert result["fixation_hold_onset_delay_sec"]["value"] is None
    stability = result["active_hand_fixation_stability"]
    assert stability["value"] is None
    assert stability["not_measurable_reason"] == metrics.HOLD_NOT_OBSERVED_REASON
    assert stability["evidence"]["hold_gate"]["withheld_value"] is not None
    # Measurements read at the fixation anchor are unaffected.
    assert result["elbow_flexion_deg"]["value"] is not None


def test_stability_window_starts_at_the_observed_hold_onset(profile: dict, contracts: list[dict]) -> None:
    # Refined Poomsae run, M02/M04: the hold starts 12-19 frames after the technique ends.
    points = _points()
    times = np.arange(points.shape[0]) / 30.0
    moving = np.clip(times - 0.1, 0, 0.9) * 0.6  # moves until frame 30, then holds
    points[:, BODY["left_wrist"], 1] += moving
    points[:, 91:112, 1] += moving[:, None]
    result = measure_observable_accuracy_metrics(_with_signals(points), contracts[0], SEGMENT, profile, _direction(), 30.0)
    assert result["fixation_hold_observed"]["value"] is True
    onset = SEGMENT["anchors"]["fixation"] + round(result["fixation_hold_onset_delay_sec"]["value"] * 30.0)
    assert onset >= 30
    window = result["active_hand_fixation_stability"]["evidence"]["window"]
    assert window["starts_at"] == "observed_hold_onset"
    assert window["start_frame"] == onset
    assert result["active_hand_fixation_stability"]["value"] is not None
