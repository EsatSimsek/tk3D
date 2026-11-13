from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from src.data_structures import (
    COCO_BODY_JOINTS,
    COCO_FACE_INDICES,
    COCO_FOOT_JOINTS,
    COCO_HAND_LOCAL_JOINTS,
    COCO_LEFT_HAND_OFFSET,
    COCO_RIGHT_HAND_OFFSET,
)
from src.poomsae_scoring.contracts import ScoringContractError


# The 68-point face block (WholeBody 23..90) follows the iBUG-300W layout, which is
# labelled from the viewer's side: face-local 36..41 is the athlete's RIGHT eye and
# 42..47 the LEFT eye (on the real ZED recording the 36..41 cluster sits ~4 mm from
# body right_eye and ~63 mm from body left_eye). Every eye-line consumer reads these
# two constants so the sides are named once, the same way as body joints 1 and 2.
FACE_OFFSET = COCO_FACE_INDICES[0]
FACE_RIGHT_EYE_LOCAL = tuple(range(36, 42))
FACE_LEFT_EYE_LOCAL = tuple(range(42, 48))
FACE_RIGHT_EYE_INDICES = tuple(FACE_OFFSET + index for index in FACE_RIGHT_EYE_LOCAL)
FACE_LEFT_EYE_INDICES = tuple(FACE_OFFSET + index for index in FACE_LEFT_EYE_LOCAL)

# ---------------------------------------------------------------------------
# Metric registry: the single record per technical-accuracy metric id.
#
# Everything that used to be repeated by hand (unit table, boolean/contract-only
# sets, quality-group substring rules, required-landmark token matching, evaluator
# inventory, contextual threshold tables, WholeBody alias table) is derived from
# these records. The diagnostic profile YAML keeps only status, thresholds and
# expectations; at load the profile catalog and this registry must match exactly.
#
# ``torso_length`` is the shoulder-centre to hip-centre distance (``_body_scale``);
# several metric ids still carry a historical ``_body_ratio`` suffix, but the value
# is divided by the torso length.
# ---------------------------------------------------------------------------

POST_FIXATION_HOLD = "post_fixation_hold"
POST_FIXATION_TAIL = "post_fixation_tail_to_segment_end"
FIXATION_ANCHOR = "fixation_anchor"
PREPARATION_ANCHOR = "preparation_anchor"
PREPARATION_AND_FIXATION_ANCHORS = "preparation_and_fixation_anchors"
PREPARATION_AND_EXECUTION_ANCHORS = "preparation_and_execution_anchors"
PREPARATION_TO_FIXATION = "preparation_to_fixation"
SEGMENT_START_TO_FIXATION = "segment_start_to_fixation"
KICK_PHASE_ANCHORS = "kick_phase_single_anchors"
CONTRACT_FACT = "contract_or_anchor_fact"
NOT_OBSERVED = "not_observed_by_pipeline"

METRIC_WINDOWS = frozenset({
    FIXATION_ANCHOR, POST_FIXATION_HOLD, POST_FIXATION_TAIL, PREPARATION_ANCHOR,
    PREPARATION_AND_FIXATION_ANCHORS, PREPARATION_AND_EXECUTION_ANCHORS,
    PREPARATION_TO_FIXATION, SEGMENT_START_TO_FIXATION, KICK_PHASE_ANCHORS,
    CONTRACT_FACT, NOT_OBSERVED,
})
METRIC_UNIT_VOCABULARY = frozenset({
    "bool", "deg", "ratio", "sec", "leg_length", "arm_length", "shoulder_width",
    "torso_length", "state", "reason_code",
})
VALUE_KINDS = frozenset({"numeric", "boolean", "state"})

# Group gates applied inside the measurement module (fixation-anchor group quality).
# ``quality_metric`` ids report evidence quality and are never gated; ``contract``
# ids are contract/anchor facts; ``ungated`` keeps the historical ratio of 1.0;
# ``rule_layer`` ids are produced outside the measurement module and only meet the
# rule-layer landmark gate; ``not_measured`` ids are never produced.
MEASUREMENT_QUALITY_GROUPS = frozenset({
    "head", "torso", "lower", "upper", "active_hand", "reaction_hand", "stance",
    "fixation", "phase", "ungated", "quality_metric", "contract",
})
QUALITY_GROUPS = MEASUREMENT_QUALITY_GROUPS | {"rule_layer", "not_measured"}

# Who produces the id. ``placeholder`` emits the id but never a value (no numeric
# target or no maths yet), so it does not count as an implemented evaluator.
PRODUCERS = frozenset({
    "metrics", "placeholder", "rule_layer", "contract", "wholebody_alias",
    "frame_function", "none",
})
IMPLEMENTED_PRODUCERS = frozenset({"metrics", "rule_layer", "contract", "wholebody_alias", "frame_function"})
APPLICABILITY = frozenset({"all", "kick", "expected_moving_foot"})

_ROLE_FIELDS = {
    "active": "active_arm",
    "reaction": "chamber_or_reaction_arm",
    "lead": "lead_leg",
    "rear": "rear_or_support_leg",
    "kick": "kicking_leg",
}
_BODY_PARTS = ("shoulder", "elbow", "wrist", "hip", "knee", "ankle", "eye", "ear")
_SIDED_GROUPS: dict[str, Callable[[str], tuple[int, ...]]] = {
    **{part: (lambda side, part=part: (COCO_BODY_JOINTS[f"{side}_{part}"],)) for part in _BODY_PARTS},
    "arm": lambda side: tuple(COCO_BODY_JOINTS[f"{side}_{part}"] for part in ("shoulder", "elbow", "wrist")),
    "leg": lambda side: tuple(COCO_BODY_JOINTS[f"{side}_{part}"] for part in ("hip", "knee", "ankle")),
    "foot": lambda side: tuple(COCO_FOOT_JOINTS[f"{side}_{part}"] for part in ("big_toe", "small_toe", "heel")),
    "hand": lambda side: tuple(range(_hand_offset(side), _hand_offset(side) + 21)),
    "palm": lambda side: tuple(
        _hand_offset(side) + COCO_HAND_LOCAL_JOINTS[name]
        for name in ("wrist", "index_mcp", "middle_mcp", "ring_mcp", "pinky_mcp")
    ),
    **{
        f"hand_{joint}": (lambda side, joint=joint: (_hand_offset(side) + COCO_HAND_LOCAL_JOINTS[joint],))
        for joint in ("wrist", "index_mcp", "middle_mcp", "ring_mcp", "pinky_mcp")
    },
}
_UNSIDED_GROUPS: dict[str, tuple[int, ...]] = {
    "torso": tuple(COCO_BODY_JOINTS[name] for name in ("left_shoulder", "right_shoulder", "left_hip", "right_hip")),
    "nose_body": (COCO_BODY_JOINTS["nose"],),
    "body17": tuple(range(17)),
    "face_all": tuple(COCO_FACE_INDICES),
    "face_eyes": tuple(sorted(FACE_RIGHT_EYE_INDICES + FACE_LEFT_EYE_INDICES)),
    "face_nose": tuple(FACE_OFFSET + index for index in range(27, 36)),
    "face_brows": tuple(FACE_OFFSET + index for index in range(17, 27)),
    "face_chin": (FACE_OFFSET + 8,),
}


def _hand_offset(side: str) -> int:
    return COCO_LEFT_HAND_OFFSET if side == "left" else COCO_RIGHT_HAND_OFFSET


def _role_sides(role: str, contract: dict[str, Any] | None) -> tuple[str, ...]:
    if role == "both":
        return ("left", "right")
    if role in {"left", "right"}:
        return (role,)
    if role not in _ROLE_FIELDS:
        raise ValueError(f"unknown landmark role: {role}")
    if contract is None:
        # Contract-independent view: every side the role can take.
        return ("left", "right")
    side = contract.get(_ROLE_FIELDS[role])
    return (side,) if side in {"left", "right"} else ()


def _resolve_landmark_token(token: str, contract: dict[str, Any] | None) -> tuple[int, ...]:
    if token in _UNSIDED_GROUPS:
        return _UNSIDED_GROUPS[token]
    group, _, role = token.partition(":")
    if group == "technique_target":
        # The fist height / lateral target of the active hand depends on the
        # technique contract: brows above the forehead, the active knee below the
        # pelvis, otherwise the torso centre (already read through ``torso``).
        height = None if contract is None else contract.get("active_hand_expected_height")
        indices: tuple[int, ...] = ()
        if contract is None or height == "above_forehead":
            indices += _UNSIDED_GROUPS["face_brows"]
        if contract is None or height == "below_pelvis":
            indices += tuple(COCO_BODY_JOINTS[f"{side}_knee"] for side in _role_sides(role, contract))
        return indices
    if group not in _SIDED_GROUPS or not role:
        raise ValueError(f"unknown landmark token: {token}")
    return tuple(index for side in _role_sides(role, contract) for index in _SIDED_GROUPS[group](side))


@dataclass(frozen=True)
class MetricSpec:
    """One registered technical-accuracy metric.

    ``landmarks`` are tokens such as ``"torso"``, ``"arm:active"`` or ``"foot:lead"``;
    a role (active / reaction / lead / rear / kick) is resolved against the movement
    contract, so a record declares exactly the points its evaluator reads for that
    movement's sides. An empty tuple is allowed only for contract/anchor facts, for
    placeholders without maths and for ids the pipeline cannot observe.
    """

    metric_id: str
    unit: str
    landmarks: tuple[str, ...] | None
    window: str
    quality_group: str
    producer: str
    kind: str = ""
    applies_to: str = "all"
    contextual_range: tuple[Any, ...] | None = None
    alias_of: str | None = None
    frame_value: Callable[[dict[str, Any], int, dict[str, Any]], float | None] | None = None

    def __post_init__(self) -> None:
        label = f"metric registry record {self.metric_id!r}"
        if not isinstance(self.metric_id, str) or not self.metric_id:
            raise ValueError("metric registry record needs a non-empty metric_id")
        if self.unit not in METRIC_UNIT_VOCABULARY:
            raise ValueError(f"{label} has no valid unit: {self.unit!r}")
        if self.landmarks is None or not isinstance(self.landmarks, tuple):
            raise ValueError(f"{label} must declare its landmarks (use () only for contract facts)")
        for token in self.landmarks:
            _resolve_landmark_token(token, None)
        kind = self.kind or ("boolean" if self.unit == "bool" else "state" if self.unit in {"state", "reason_code"} else "numeric")
        object.__setattr__(self, "kind", kind)
        if kind not in VALUE_KINDS or (kind == "boolean") != (self.unit == "bool"):
            raise ValueError(f"{label} value kind {kind!r} does not match unit {self.unit!r}")
        if self.window not in METRIC_WINDOWS:
            raise ValueError(f"{label} has unknown evaluation window {self.window!r}")
        if self.quality_group not in QUALITY_GROUPS:
            raise ValueError(f"{label} has unknown quality group {self.quality_group!r}")
        if self.producer not in PRODUCERS:
            raise ValueError(f"{label} has unknown producer {self.producer!r}")
        if self.applies_to not in APPLICABILITY:
            raise ValueError(f"{label} has unknown applicability {self.applies_to!r}")
        if not self.landmarks and self.window != CONTRACT_FACT and self.producer not in {"placeholder", "none"}:
            raise ValueError(f"{label} reads geometry but declares no landmarks")
        in_module = self.producer in {"metrics", "placeholder"}
        if in_module != (self.quality_group in MEASUREMENT_QUALITY_GROUPS):
            raise ValueError(f"{label}: quality group {self.quality_group!r} does not fit producer {self.producer!r}")
        if self.producer == "none" and self.quality_group != "not_measured":
            raise ValueError(f"{label}: never-produced ids use quality group 'not_measured'")
        if self.producer in {"rule_layer", "contract", "wholebody_alias", "frame_function"} and self.quality_group != "rule_layer":
            raise ValueError(f"{label}: ids produced outside the measurement module use quality group 'rule_layer'")
        if (self.producer == "wholebody_alias") != bool(self.alias_of):
            raise ValueError(f"{label}: alias_of is required exactly for wholebody_alias producers")
        if (self.producer == "frame_function") != callable(self.frame_value):
            raise ValueError(f"{label}: frame_value is required exactly for frame_function producers")
        if self.producer == "frame_function" and self.window not in {FIXATION_ANCHOR, POST_FIXATION_HOLD}:
            raise ValueError(f"{label}: frame functions run on the fixation anchor or the post-fixation hold")
        if self.contextual_range is not None:
            if kind != "numeric" or self.contextual_range[0] not in {"stance", "technique"}:
                raise ValueError(f"{label}: contextual ranges are numeric stance or technique ranges")

    @property
    def implemented(self) -> bool:
        return self.producer in IMPLEMENTED_PRODUCERS


def resolve_metric_landmarks(metric: str | MetricSpec, contract: dict[str, Any] | None = None) -> list[int]:
    """Landmark indices the metric's evaluator reads for this contract's sides.

    With ``contract=None`` every role resolves to both sides, i.e. the set of points
    the rule may read in any movement.
    """
    spec = metric if isinstance(metric, MetricSpec) else METRIC_REGISTRY[metric]
    return sorted({index for token in spec.landmarks or () for index in _resolve_landmark_token(token, contract)})


def build_metric_registry(records: Any) -> dict[str, MetricSpec]:
    registry: dict[str, MetricSpec] = {}
    for record in records:
        if not isinstance(record, MetricSpec):
            raise ValueError("metric registry entries must be MetricSpec records")
        if record.metric_id in registry:
            raise ValueError(f"duplicate metric registry record: {record.metric_id}")
        registry[record.metric_id] = record
    return registry


def measurement_evaluator_status(spec: MetricSpec, rule_status: str) -> str:
    if rule_status == "not_observable_with_current_pipeline":
        return "not_applicable_pipeline_limit"
    return "implemented" if spec.implemented else "not_implemented"


class _RegistryUnits(Mapping[str, str]):
    """Live ``metric_id -> unit`` view of the registry (kept for existing callers)."""

    def __getitem__(self, metric_id: str) -> str:
        return METRIC_REGISTRY[metric_id].unit

    def __iter__(self) -> Iterator[str]:
        return iter(METRIC_REGISTRY)

    def __len__(self) -> int:
        return len(METRIC_REGISTRY)


def _m(metric_id: str, unit: str, landmarks: tuple[str, ...], window: str, group: str, producer: str = "metrics", **extra: Any) -> MetricSpec:
    return MetricSpec(metric_id, unit, landmarks, window, group, producer, **extra)


# Landmark declarations: exactly the points each evaluator reads, by contract role.
# Torso-forward helpers (``_torso_forward``) take the forward direction, sign
# included, from the anatomical left/right shoulder line only; the face is not read.
_T = ("torso",)
_EYES, _NOSE, _BROWS, _CHIN = "face_eyes", "face_nose", "face_brows", "face_chin"
_FORWARD_SIGN = ("shoulder:both",)
# Head facing reads the body nose and both ears (see ``_head_forward``).
_HEAD_FACING = ("nose_body", "ear:both")
_STANCE_GEOMETRY = ("foot:lead", "foot:rear", "shoulder:both")
_FOOT_DISPLACEMENT = ("ankle:lead", "ankle:rear", "leg:lead")
_ARM_DISPLACEMENT = ("arm:active", "wrist:reaction")
_LOWER_SPAN = ("knee:both", "ankle:both", "leg:lead")
_ACTIVE_PALM = ("palm:active", "arm:active")
_HAND_LOCAL = ("palm:active", "torso", "arm:active")
_WRIST_FRAME = ("elbow:active", "hand_wrist:active", "hand_index_mcp:active", "hand_middle_mcp:active", "hand_pinky_mcp:active")
_NONE: tuple[str, ...] = ()
_FIX, _HOLD, _TAIL, _CONTRACT = FIXATION_ANCHOR, POST_FIXATION_HOLD, POST_FIXATION_TAIL, CONTRACT_FACT
_PF, _PE = PREPARATION_AND_FIXATION_ANCHORS, PREPARATION_AND_EXECUTION_ANCHORS
_SF, _KICK = SEGMENT_START_TO_FIXATION, KICK_PHASE_ANCHORS
_RL = "rule_layer"

_RECORDS: tuple[MetricSpec, ...] = (
    # head_orientation
    _m("head_target_yaw_error_deg", "deg", _HEAD_FACING, _FIX, "head"),
    _m("head_torso_relative_yaw_deg", "deg", (*_HEAD_FACING, "shoulder:both", "leg:both"), _FIX, _RL, "wholebody_alias", alias_of="head_torso_yaw_mismatch_deg"),
    _m("head_roll_relative_shoulders_deg", "deg", (_EYES, "shoulder:both"), _FIX, _RL, "rule_layer"),
    _m("head_pitch_relative_torso_deg", "deg", (_BROWS, _CHIN, "torso"), _FIX, _RL, "rule_layer"),
    _m("head_fixation_orientation_dispersion_p95_deg", "deg", (_EYES,), _HOLD, _RL, "rule_layer"),
    _m("head_post_fixation_drift_deg", "deg", (_EYES,), _HOLD, _RL, "rule_layer"),
    _m("head_turn_completion_state", "bool", (_EYES,), _HOLD, "head"),
    _m("head_turn_direction_sign_match", "bool", _HEAD_FACING, _FIX, "head"),
    _m("head_wrong_direction_stable_state", "bool", _HEAD_FACING, _FIX, "head"),
    _m("head_late_turn_state", "bool", (_EYES,), _HOLD, "head"),
    _m("head_orientation_unmeasurable_reason", "reason_code", (_BROWS, _EYES, _CHIN), _FIX, "quality_metric"),
    _m("head_face_geometry_quality", "ratio", (_BROWS, _EYES, _CHIN), _FIX, "quality_metric"),
    # torso_pelvis
    _m("torso_target_yaw_error_deg", "deg", _FORWARD_SIGN, _FIX, "torso"),
    _m("pelvis_target_yaw_error_deg", "deg", ("hip:both", *_FORWARD_SIGN), _FIX, "torso"),
    _m("stance_axis_target_yaw_error_deg", "deg", ("foot:lead", "foot:rear"), _FIX, "stance"),
    _m("torso_pelvis_relative_yaw_deg", "deg", _T, _FIX, _RL, "wholebody_alias", alias_of="shoulder_hip_twist_deg"),
    _m("torso_lean_p95_deg", "deg", _T, _FIX, _RL, "wholebody_alias", alias_of="torso_lean_p95_deg"),
    _m("torso_forward_backward_lean_deg", "deg", ("torso",), _FIX, "torso"),
    _m("torso_lateral_bend_deg", "deg", _T, _FIX, _RL, "rule_layer"),
    _m("shoulder_roll_deg", "deg", ("shoulder:both",), _FIX, _RL, "rule_layer"),
    _m("pelvis_roll_deg", "deg", ("hip:both",), _FIX, _RL, "rule_layer"),
    _m("shoulder_pelvis_roll_difference_deg", "deg", _T, _FIX, "upper"),
    _m("shoulder_height_asymmetry_body_ratio", "torso_length", _T, _FIX, "upper"),
    _m("pelvis_height_asymmetry_body_ratio", "torso_length", _T, _FIX, "torso"),
    _m("torso_fixation_orientation_dispersion_p95_deg", "deg", ("shoulder:both",), _HOLD, "torso"),
    _m("pelvis_fixation_orientation_dispersion_p95_deg", "deg", ("hip:both",), _HOLD, "torso"),
    _m("torso_post_fixation_drift_deg", "deg", ("shoulder:both",), _HOLD, "torso"),
    _m("pelvis_post_fixation_drift_deg", "deg", ("hip:both",), _HOLD, "torso"),
    _m("torso_translation_after_fixation_body_ratio", "torso_length", _T, _HOLD, "torso"),
    _m("pelvis_translation_after_fixation_body_ratio", "torso_length", _T, _HOLD, "torso"),
    _m("body_height_change_during_fixation_ratio", "torso_length", ("torso", "ankle:both"), _HOLD, "torso"),
    _m("torso_geometry_quality", "ratio", _T, _FIX, "quality_metric"),
    # stance_feet
    _m("expected_stance_type_match", "bool", (*_STANCE_GEOMETRY, "leg:lead"), _FIX, "stance"),
    _m("lead_leg_side_match", "bool", _NONE, _CONTRACT, "lower"),
    _m("moving_foot_side_match", "bool", _FOOT_DISPLACEMENT, _PF, "stance", applies_to="expected_moving_foot"),
    _m("expected_support_or_pivot_foot_match", "bool", _FOOT_DISPLACEMENT, _PF, "stance", applies_to="expected_moving_foot"),
    _m("stance_length_leg_ratio", "leg_length", (*_STANCE_GEOMETRY, "leg:lead"), _FIX, "stance", contextual_range=("stance", "length_leg_ratio")),
    _m("stance_width_shoulder_ratio", "shoulder_width", _STANCE_GEOMETRY, _FIX, "stance", contextual_range=("stance", "width_shoulder_ratio")),
    _m("front_foot_target_yaw_error_deg", "deg", ("foot:lead",), _FIX, "stance"),
    _m("rear_foot_target_yaw_error_deg", "deg", ("foot:rear",), _FIX, "stance"),
    _m("inter_foot_yaw_difference_deg", "deg", ("foot:lead", "foot:rear"), _FIX, "stance"),
    _m("foot_progression_axis_error_deg", "deg", ("foot:lead", "foot:rear"), _FIX, "stance"),
    _m("foot_crossing_margin_body_ratio", "torso_length", ("foot:lead", "foot:rear", "torso"), _FIX, "stance"),
    _m("heel_alignment_error_body_ratio", "torso_length", ("foot:lead", "foot:rear", "torso"), _FIX, "ungated"),
    _m("toe_alignment_error_body_ratio", "torso_length", ("foot:lead", "foot:rear", "torso"), _FIX, "ungated"),
    _m("front_back_foot_order_match", "bool", ("foot:lead", "foot:rear", *_FORWARD_SIGN), _FIX, "stance"),
    _m("foot_landing_position_error_body_ratio", "leg_length", (*_STANCE_GEOMETRY, "leg:lead"), _FIX, "stance"),
    _m("pivot_foot_displacement_body_ratio", "leg_length", ("ankle:rear", "leg:lead"), _PF, "stance"),
    _m("support_foot_displacement_body_ratio", "leg_length", ("ankle:rear", "leg:lead"), _PF, "stance"),
    _m("foot_fixation_slip_body_ratio", "leg_length", ("foot:lead", "foot:rear", "leg:lead"), _HOLD, "stance"),
    _m("stance_fixation_dispersion_body_ratio", "leg_length", ("foot:lead", "foot:rear", "leg:lead"), _HOLD, "stance"),
    _m("stance_depth_or_pelvis_height_leg_ratio", "leg_length", ("torso", "ankle:both", "leg:lead"), _FIX, "stance"),
    _m("stance_geometry_quality", "ratio", ("leg:both", "foot:lead", "foot:rear"), _FIX, "quality_metric"),
    # lower_body
    _m("front_knee_flexion_deg", "deg", ("leg:lead",), _FIX, "lower", contextual_range=("stance", "front_knee_included_angle_deg")),
    _m("rear_knee_flexion_deg", "deg", ("leg:rear",), _FIX, "lower", contextual_range=("stance", "rear_knee_included_angle_deg")),
    _m("expected_knee_flexion_range_match", "bool", ("leg:lead", "leg:rear"), _FIX, "lower"),
    _m("knee_over_foot_alignment_ratio", "leg_length", ("leg:lead", "foot:lead"), _FIX, "stance"),
    _m("knee_valgus_varus_proxy_deg", "deg", ("leg:lead",), _FIX, "lower"),
    _m("knee_target_direction_error_deg", "deg", ("hip:lead", "knee:lead"), _FIX, "lower"),
    _m("hip_knee_ankle_plane_deviation_ratio", "leg_length", ("leg:lead",), _FIX, "lower"),
    _m("lead_leg_identity_match", "bool", _NONE, _CONTRACT, "lower"),
    _m("rear_leg_identity_match", "bool", _NONE, _CONTRACT, "lower"),
    _m("lower_body_left_right_swap_state", "bool", _NONE, _CONTRACT, "lower"),
    _m("pelvis_height_stance_match", "bool", ("leg:lead",), _FIX, "stance"),
    _m("lower_body_fixation_dispersion", "leg_length", _LOWER_SPAN, _HOLD, "lower"),
    _m("lower_body_geometry_quality", "ratio", ("leg:both",), _FIX, "quality_metric"),
    # kick
    _m("chamber_knee_height_body_ratio", "leg_length", ("hip:kick", "knee:kick", "leg:lead"), _KICK, "lower", applies_to="kick"),
    _m("chamber_hip_flexion_deg", "deg", ("shoulder:kick", "hip:kick", "knee:kick"), _KICK, "ungated", applies_to="kick"),
    _m("chamber_knee_flexion_deg", "deg", ("leg:kick",), _KICK, "lower", applies_to="kick"),
    _m("support_leg_stability", "leg_length", ("ankle:rear", "leg:lead"), _KICK, "lower", applies_to="kick"),
    _m("kick_extension_deg", "deg", ("leg:kick",), _KICK, "lower", applies_to="kick"),
    _m("kick_target_height_body_ratio", "leg_length", ("ankle:kick", "hip:kick", "leg:lead"), _KICK, "lower", applies_to="kick"),
    _m("kick_direction_error_deg", "deg", _NONE, _FIX, "lower", "placeholder", applies_to="kick"),
    _m("kick_retraction_state", "bool", ("leg:kick",), _KICK, "lower", applies_to="kick"),
    _m("kick_landing_stance_restoration", "bool", ("ankle:kick",), _KICK, "stance", applies_to="kick"),
    # upper_body
    _m("active_arm_side_match", "bool", _ARM_DISPLACEMENT, _PE, "upper"),
    _m("reaction_or_chamber_arm_side_match", "bool", _NONE, _CONTRACT, "reaction_hand"),
    _m("shoulder_elevation_body_ratio", "torso_length", _T, _FIX, "upper"),
    _m("shoulder_abduction_deg", "deg", ("torso", "elbow:active"), _FIX, "upper"),
    _m("shoulder_flexion_deg", "deg", ("torso", "elbow:active"), _FIX, "upper"),
    _m("elbow_flexion_deg", "deg", ("arm:active",), _FIX, "upper", contextual_range=("technique",)),
    _m("elbow_target_angle_error_deg", "deg", _NONE, _FIX, "upper", "placeholder"),
    _m("elbow_flare_body_ratio", "arm_length", ("arm:active",), _FIX, _RL, "rule_layer"),
    _m("shoulder_elbow_wrist_plane_deviation_ratio", "arm_length", ("arm:active",), _FIX, _RL, "rule_layer"),
    _m("active_arm_extension_ratio", "arm_length", ("arm:active",), _FIX, "upper", contextual_range=("technique",)),
    _m("active_arm_target_direction_error_deg", "deg", ("shoulder:active", "palm:active"), _FIX, "upper"),
    _m("active_arm_target_height_body_ratio", "torso_length", _HAND_LOCAL, _FIX, "upper"),
    _m("active_arm_target_lateral_offset_body_ratio", "shoulder_width", _HAND_LOCAL, _FIX, "upper"),
    _m("active_arm_target_depth_body_ratio", "arm_length", _HAND_LOCAL, _FIX, "upper"),
    _m("active_hand_target_distance_body_ratio", "torso_length", _HAND_LOCAL, _FIX, "active_hand"),
    _m("reaction_hand_target_distance_body_ratio", "torso_length", ("palm:reaction", "torso"), _FIX, "reaction_hand"),
    _m("reaction_elbow_position_error_body_ratio", "arm_length", ("arm:reaction", "hip:reaction"), _FIX, "upper"),
    _m("active_arm_fixation_dispersion_body_ratio", "arm_length", ("arm:active",), _HOLD, _RL, "rule_layer"),
    _m("reaction_arm_fixation_dispersion_body_ratio", "arm_length", ("arm:reaction",), _HOLD, _RL, "rule_layer"),
    _m("arm_late_correction_body_ratio", "arm_length", ("arm:active",), _HOLD, _RL, "rule_layer"),
    _m("arm_target_overshoot_body_ratio", "arm_length", _ACTIVE_PALM, _SF, "upper"),
    _m("upper_body_geometry_quality", "ratio", ("arm:both",), _FIX, "quality_metric"),
    # wrist_hand
    _m("wrist_forearm_alignment_deg", "deg", ("elbow:active", "wrist:active", "hand_middle_mcp:active", "leg:both"), _FIX, _RL, "wholebody_alias", alias_of="wrist_forearm_alignment_deg"),
    _m("wrist_flexion_extension_proxy_deg", "deg", _WRIST_FRAME, _FIX, "active_hand"),
    _m("wrist_radial_ulnar_deviation_proxy_deg", "deg", _WRIST_FRAME, _FIX, "active_hand"),
    _m("fist_or_hand_orientation_proxy_deg", "deg", (*_WRIST_FRAME, *_FORWARD_SIGN), _FIX, "active_hand"),
    _m("fist_target_direction_error_deg", "deg", ("hand_wrist:active", "hand_middle_mcp:active"), _FIX, "active_hand"),
    _m("fist_target_height_error_body_ratio", "torso_length", ("palm:active", "torso", "technique_target:active"), _FIX, "active_hand"),
    _m("fist_target_lateral_error_body_ratio", "shoulder_width", ("palm:active", "torso", "technique_target:active"), _FIX, "active_hand"),
    _m("fist_target_depth_error_body_ratio", "arm_length", _NONE, _FIX, "active_hand", "placeholder"),
    _m("hand_shape_observability", "bool", ("hand:active",), _FIX, "active_hand"),
    _m("active_hand_fixation_stability", "arm_length", _ACTIVE_PALM, _HOLD, "active_hand"),
    _m("reaction_hand_fixation_stability", "arm_length", ("palm:reaction", "arm:reaction"), _HOLD, "reaction_hand"),
    _m("hand_geometry_quality", "ratio", ("hand:active", "hand:reaction"), _FIX, "quality_metric"),
    # phase_structure
    _m("required_phase_coverage", "bool", _NONE, _CONTRACT, _RL, "contract"),
    _m("preparation_pose_observed", "bool", _NONE, _CONTRACT, "contract"),
    _m("preparation_side_match", "bool", _ARM_DISPLACEMENT, _PE, "phase"),
    _m("chamber_pose_observed", "bool", ("palm:reaction", "torso"), PREPARATION_ANCHOR, "reaction_hand"),
    _m("chamber_precedes_execution", "bool", _NONE, _CONTRACT, "contract"),
    _m("active_technique_reaches_target_by_fixation", "bool", _ACTIVE_PALM, _HOLD, "phase"),
    _m("stance_reaches_target_by_fixation", "bool", (*_STANCE_GEOMETRY, "leg:lead"), _FIX, "stance"),
    _m("head_reaches_target_by_fixation", "bool", (_EYES,), _HOLD, "head"),
    _m("torso_reaches_target_by_fixation", "bool", ("shoulder:both",), _HOLD, "torso"),
    _m("reaction_hand_reaches_target_by_fixation", "bool", ("palm:reaction", "torso"), _FIX, "reaction_hand"),
    _m("active_hand_stance_settle_offset", "sec", ("wrist:active", "ankle:lead"), PREPARATION_TO_FIXATION, _RL, "wholebody_alias", alias_of="hand_foot_settle_difference_sec"),
    _m("head_torso_settle_offset", "sec", (_EYES, "shoulder:both"), _SF, "head"),
    _m("late_post_fixation_correction", "arm_length", _ACTIVE_PALM, _HOLD, "phase"),
    _m("premature_next_movement_transition", "bool", _ACTIVE_PALM, _TAIL, "phase"),
    _m("fixation_pose_stability", "bool", (*_ACTIVE_PALM, *_LOWER_SPAN), _HOLD, "phase"),
    _m("final_pose_geometry_conformance", "bool", (*_STANCE_GEOMETRY, "leg:lead", "arm:active"), _FIX, "phase"),
    _m("phase_evidence_quality", "ratio", _NONE, _CONTRACT, "quality_metric"),
    # transition
    _m("expected_direction_change_match", "bool", ("foot:lead", "foot:rear"), _FIX, "phase"),
    _m("rotation_direction_sign_match", "bool", ("foot:lead", "foot:rear"), _FIX, "phase"),
    _m("moving_foot_match", "bool", _FOOT_DISPLACEMENT, _PF, "stance"),
    _m("support_or_pivot_foot_match", "bool", _FOOT_DISPLACEMENT, _PF, "stance"),
    _m("step_direction_error_deg", "deg", ("foot:lead", "foot:rear"), _FIX, "stance"),
    _m("step_length_error_body_ratio", "leg_length", (*_STANCE_GEOMETRY, "leg:lead"), _FIX, "stance"),
    _m("step_width_error_body_ratio", "shoulder_width", _STANCE_GEOMETRY, _FIX, "stance"),
    _m("landing_position_error_body_ratio", "leg_length", (*_STANCE_GEOMETRY, "leg:lead"), _FIX, "stance"),
    _m("landing_stance_type_match", "bool", (*_STANCE_GEOMETRY, "leg:lead"), _FIX, "stance"),
    _m("head_turn_settled_by_fixation", "bool", (_EYES,), _HOLD, "head"),
    _m("torso_turn_settled_by_fixation", "bool", ("shoulder:both",), _HOLD, "torso"),
    _m("pelvis_or_stance_turn_settled_by_fixation", "bool", ("hip:both",), _HOLD, "stance"),
    _m("pivot_foot_excess_displacement", "leg_length", ("ankle:rear", "leg:lead"), _PF, "stance"),
    _m("transition_foot_crossing", "bool", _STANCE_GEOMETRY, _FIX, "stance"),
    _m("post_landing_adjustment", "leg_length", _LOWER_SPAN, _HOLD, "stance"),
    _m("transition_geometry_quality", "ratio", ("leg:both",), _SF, "quality_metric"),
    # fixation
    _m("head_fixation_dispersion", "deg", (_EYES,), _HOLD, "head"),
    _m("torso_fixation_dispersion", "deg", ("shoulder:both",), _HOLD, "torso"),
    _m("pelvis_fixation_dispersion", "deg", ("hip:both",), _HOLD, "torso"),
    _m("active_hand_fixation_dispersion", "arm_length", _ACTIVE_PALM, _HOLD, "active_hand"),
    _m("reaction_hand_fixation_dispersion", "arm_length", ("palm:reaction", "arm:reaction"), _HOLD, "reaction_hand"),
    _m("front_foot_fixation_dispersion", "leg_length", ("foot:lead", "leg:lead"), _HOLD, "stance"),
    _m("rear_foot_fixation_dispersion", "leg_length", ("foot:rear", "leg:lead"), _HOLD, "stance"),
    _m("knee_fixation_dispersion", "leg_length", ("knee:both", "leg:lead"), _HOLD, "lower"),
    _m("body_height_fixation_drift", "torso_length", ("torso", "ankle:both"), _HOLD, "torso"),
    _m("torso_translation_after_fixation", "torso_length", _T, _HOLD, "torso"),
    _m("pelvis_translation_after_fixation", "torso_length", _T, _HOLD, "torso"),
    _m("stance_width_after_fixation_change", "shoulder_width", _STANCE_GEOMETRY, _HOLD, "stance"),
    _m("stance_length_after_fixation_change", "leg_length", (*_STANCE_GEOMETRY, "leg:lead"), _HOLD, "stance"),
    _m("late_balance_recovery_proxy", "torso_length", _T, _HOLD, "torso"),
    _m("fixation_evidence_quality", "ratio", ("body17",), _HOLD, "quality_metric"),
    # sequence
    _m("movement_contract_exists", "bool", _NONE, _CONTRACT, _RL, "contract"),
    _m("movement_timeline_segment_exists", "bool", _NONE, _CONTRACT, _RL, "contract"),
    _m("expected_movement_order", "bool", _NONE, _CONTRACT, _RL, "contract"),
    _m("movement_phase_order", "bool", _NONE, _CONTRACT, _RL, "contract"),
    _m("expected_technique_contract_resolved", "bool", _NONE, _CONTRACT, _RL, "contract"),
    _m("expected_stance_contract_resolved", "bool", _NONE, _CONTRACT, _RL, "contract"),
    _m("expected_active_side_contract_resolved", "bool", _NONE, _CONTRACT, _RL, "contract"),
    _m("expected_direction_contract_resolved", "bool", _NONE, _CONTRACT, "contract"),
    _m("final_geometry_consistent_with_declared_movement", "bool", (*_STANCE_GEOMETRY, "leg:lead"), _FIX, "ungated"),
    _m("movement_completion_state", "bool", _NONE, _CONTRACT, "contract"),
    # pipeline_limits: never measured; the unit only names the kind of quantity.
    _m("actual_eye_gaze", "state", _NONE, NOT_OBSERVED, "not_measured", "none"),
    _m("centre_of_mass", "state", _NONE, NOT_OBSERVED, "not_measured", "none"),
    _m("centre_of_pressure", "state", _NONE, NOT_OBSERVED, "not_measured", "none"),
    _m("actual_weight_distribution", "state", _NONE, NOT_OBSERVED, "not_measured", "none"),
    _m("ground_reaction_force", "state", _NONE, NOT_OBSERVED, "not_measured", "none"),
    _m("impact_power", "state", _NONE, NOT_OBSERVED, "not_measured", "none"),
    _m("muscle_tension", "state", _NONE, NOT_OBSERVED, "not_measured", "none"),
)

METRIC_REGISTRY: dict[str, MetricSpec] = build_metric_registry(_RECORDS)
# The unit each metric's maths produces; the rule layer reports it and refuses a
# threshold written in any other unit. A live view of the registry.
METRIC_UNITS: Mapping[str, str] = _RegistryUnits()


def _profile_limits(profile: dict[str, Any]) -> dict[str, float]:
    """Read every decision limit from the profile instead of repeating it in code.

    Each value below already exists in the diagnostic profile. Copying one into
    this module would let a referee change the profile while a boolean rule kept
    answering from the stale copy, so the copies are resolved here once.
    """
    thresholds = profile["thresholds"]
    return {
        "head_settled_deg": float(thresholds["head_fixation_orientation_dispersion_p95_deg"]["value"]),
        "torso_settled_deg": float(thresholds["torso_fixation_orientation_dispersion_p95_deg"]["value"]),
        "pelvis_settled_deg": float(thresholds["pelvis_fixation_orientation_dispersion_p95_deg"]["value"]),
        "stance_direction_deg": float(thresholds["stance_axis_target_yaw_error_deg"]["value"]),
        "late_correction_ratio": float(thresholds["late_post_fixation_correction"]["value"]),
        "hand_still_ratio": float(thresholds["active_hand_fixation_stability"]["value"]),
        "foot_still_ratio": float(thresholds["foot_fixation_slip_body_ratio"]["value"]),
        "group_quality_ratio": float(profile["quality_gates"]["min_group_valid_ratio"]),
    }


# Where each "did not move" floor comes from, so the evidence can name it.
NOISE_FLOOR_SOURCES = {
    "foot_still_ratio": ("thresholds.foot_fixation_slip_body_ratio.value", "leg_length"),
    "hand_still_ratio": ("thresholds.active_hand_fixation_stability.value", "arm_length"),
}


def post_fixation_hold_frames(segment: dict[str, Any], profile: dict[str, Any]) -> np.ndarray:
    """Frames fixation .. min(segment end, fixation + N), N from the profile.

    The segment runs until one frame before the next movement's preparation, so the
    tail after fixation can contain the athlete already leaving the pose. Every
    post-fixation stability, drift and correction metric uses this bounded window.
    """
    end = int(segment["end_frame"])
    fixation = int(segment["anchors"].get("fixation", end))
    length = int(profile["quality_gates"]["post_fixation_hold_window_frames"])
    return np.arange(fixation, min(end, fixation + length) + 1, dtype=int)


def describe_frame_window(name: str, frames: np.ndarray, **extra: Any) -> dict[str, Any]:
    return {
        "name": name,
        "start_frame": int(frames[0]) if len(frames) else None,
        "end_frame": int(frames[-1]) if len(frames) else None,
        "frame_count": int(len(frames)),
        **extra,
    }


def measure_observable_accuracy_metrics(
    arrays: dict[str, Any],
    contract: dict[str, Any],
    segment: dict[str, Any],
    profile: dict[str, Any],
    direction: dict[str, Any] | None,
    fps: float,
) -> dict[str, dict[str, Any]]:
    """Measure score-neutral geometry; correctness remains in the rule layer."""
    start, end = int(segment["start_frame"]), int(segment["end_frame"])
    anchors = segment["anchors"]
    fixation = int(anchors.get("fixation", end))
    preparation = int(anchors.get("preparation", start))
    execution = int(
        next(
            (anchors[name] for name in ("execution", "punch_execution", "kick_apex") if name in anchors),
            fixation,
        )
    )
    radius = int(profile["quality_gates"]["anchor_window_radius_frames"])
    hold_length = int(profile["quality_gates"]["post_fixation_hold_window_frames"])
    fix_frames = _window(fixation, start, end, radius)
    prep_frames = _window(preparation, start, end, radius)
    exec_frames = _window(execution, start, end, radius)
    # Every post-fixation stability / drift / correction metric uses the bounded hold.
    hold_frames = post_fixation_hold_frames(segment, profile)
    # Only "left the pose before the next movement started" needs the whole tail.
    tail_frames = np.arange(fixation, end + 1, dtype=int)
    transition_frames = np.arange(start, fixation + 1, dtype=int)
    windows = {
        FIXATION_ANCHOR: describe_frame_window(FIXATION_ANCHOR, fix_frames, radius_frames=radius),
        "preparation_anchor": describe_frame_window("preparation_anchor", prep_frames, radius_frames=radius),
        "execution_anchor": describe_frame_window("execution_anchor", exec_frames, radius_frames=radius),
        POST_FIXATION_HOLD: describe_frame_window(
            POST_FIXATION_HOLD,
            hold_frames,
            configured_length_frames=hold_length,
            source="quality_gates.post_fixation_hold_window_frames",
        ),
        POST_FIXATION_TAIL: describe_frame_window(POST_FIXATION_TAIL, tail_frames),
        "segment_start_to_fixation": describe_frame_window("segment_start_to_fixation", transition_frames),
    }
    active = contract["active_arm"]
    reaction = contract["chamber_or_reaction_arm"]
    lead = contract["lead_leg"]
    rear = contract["rear_or_support_leg"]
    kick_side = contract.get("kicking_leg")
    limits = _profile_limits(profile)
    result: dict[str, dict[str, Any]] = {}

    def put(
        metric_id: str,
        value: Any,
        *,
        window: str | tuple[str, ...] | None = None,
        reason: str | None = None,
        evidence: dict[str, Any] | None = None,
    ) -> None:
        measured = _finite_value(value)
        record = {
            "start_frame": int(start),
            "end_frame": int(end),
            "fixation_frame": fixation,
            "measurement_implementation": "wholebody_133_geometry_v3",
        }
        if window is not None:
            names = (window,) if isinstance(window, str) else window
            record["window"] = windows[names[0]] if len(names) == 1 else [windows[name] for name in names]
        record.update(evidence or {})
        result[metric_id] = {
            "value": value if measured else None,
            "unit": METRIC_UNITS[metric_id],
            "quality_status": "measured" if measured else "unmeasurable",
            "evidence": record,
            "source_metric_id": None,
            "not_measurable_reason": None if measured else (reason or "insufficient_valid_landmark_evidence"),
        }

    def displacement_floor(name: str, expected: float | None, other: float | None) -> tuple[float, dict[str, Any]]:
        floor = limits[name]
        source, unit = NOISE_FLOOR_SOURCES[name]
        applied = expected is not None and other is not None and max(expected, other) <= floor
        return floor, {
            "noise_floor": {
                "value": floor,
                "unit": unit,
                "source": source,
                "applied": applied,
                "expected_side_displacement": expected,
                "other_side_displacement": other,
            }
        }

    body_scale = _median_scalar(arrays, fix_frames, _body_scale)
    shoulder_width = _median_scalar(arrays, fix_frames, _shoulder_width)
    lead_leg_scale = _median_scalar(arrays, fix_frames, lambda a, f: _limb_scale(a, f, "leg", lead))
    active_arm_scale = _median_scalar(arrays, fix_frames, lambda a, f: _limb_scale(a, f, "arm", active))
    stance = _stance_geometry(arrays, fix_frames, lead, rear)
    head_quality = _group_quality(arrays, fix_frames, _head_indices())
    torso_quality = _group_quality(arrays, fix_frames, _torso_indices())
    lower_quality = _group_quality(arrays, fix_frames, _lower_indices())
    upper_quality = _group_quality(arrays, fix_frames, _upper_indices())
    hand_quality = _group_quality(arrays, fix_frames, _hand_indices(active))
    reaction_hand_quality = _group_quality(arrays, fix_frames, _hand_indices(reaction))
    foot_quality = _group_quality(arrays, fix_frames, _foot_indices(lead) + _foot_indices(rear))
    hold_quality = _group_quality(arrays, hold_frames, tuple(range(17)))

    put("head_face_geometry_quality", head_quality, window=FIXATION_ANCHOR)
    put("head_orientation_unmeasurable_reason", "none" if head_quality is not None and head_quality >= limits["group_quality_ratio"] else None, window=FIXATION_ANCHOR)
    put("torso_geometry_quality", torso_quality, window=FIXATION_ANCHOR)
    put("stance_geometry_quality", _minimum_finite(lower_quality, foot_quality), window=FIXATION_ANCHOR)
    put("lower_body_geometry_quality", lower_quality, window=FIXATION_ANCHOR)
    put("upper_body_geometry_quality", upper_quality, window=FIXATION_ANCHOR)
    put("hand_geometry_quality", _minimum_finite(hand_quality, reaction_hand_quality), window=FIXATION_ANCHOR)
    put("phase_evidence_quality", _phase_quality(anchors))
    put("transition_geometry_quality", _group_quality(arrays, transition_frames, _lower_indices()), window="segment_start_to_fixation")
    put("fixation_evidence_quality", hold_quality, window=POST_FIXATION_HOLD)

    put("torso_forward_backward_lean_deg", _median_scalar(arrays, fix_frames, _torso_forward_lean), window=FIXATION_ANCHOR)
    put("shoulder_pelvis_roll_difference_deg", _median_scalar(arrays, fix_frames, _roll_difference), window=FIXATION_ANCHOR)
    put("shoulder_height_asymmetry_body_ratio", _median_scalar(arrays, fix_frames, lambda a, f: _height_asymmetry(a, f, "shoulder")), window=FIXATION_ANCHOR)
    put("pelvis_height_asymmetry_body_ratio", _median_scalar(arrays, fix_frames, lambda a, f: _height_asymmetry(a, f, "hip")), window=FIXATION_ANCHOR)
    put("torso_fixation_orientation_dispersion_p95_deg", _orientation_dispersion(arrays, hold_frames, "shoulder"), window=POST_FIXATION_HOLD)
    put("pelvis_fixation_orientation_dispersion_p95_deg", _orientation_dispersion(arrays, hold_frames, "hip"), window=POST_FIXATION_HOLD)
    put("torso_post_fixation_drift_deg", _orientation_drift(arrays, hold_frames, "shoulder"), window=POST_FIXATION_HOLD)
    put("pelvis_post_fixation_drift_deg", _orientation_drift(arrays, hold_frames, "hip"), window=POST_FIXATION_HOLD)
    put("torso_translation_after_fixation_body_ratio", _centre_dispersion(arrays, hold_frames, "shoulder"), window=POST_FIXATION_HOLD)
    put("pelvis_translation_after_fixation_body_ratio", _centre_dispersion(arrays, hold_frames, "hip"), window=POST_FIXATION_HOLD)
    put("body_height_change_during_fixation_ratio", _body_height_change(arrays, hold_frames), window=POST_FIXATION_HOLD)

    put("stance_length_leg_ratio", _safe_ratio(stance.get("length"), lead_leg_scale), window=FIXATION_ANCHOR)
    put("stance_width_shoulder_ratio", _safe_ratio(stance.get("width"), shoulder_width), window=FIXATION_ANCHOR)
    stance_match = _all_known(_range_match(_safe_ratio(stance.get("length"), lead_leg_scale), contract["stance_length_expectation"]), _range_match(
        _safe_ratio(stance.get("width"), shoulder_width), contract["stance_width_expectation"]
    ))
    put("expected_stance_type_match", stance_match, window=FIXATION_ANCHOR)
    lead_displacement = _joint_displacement(arrays, prep_frames, fix_frames, f"{lead}_ankle", lead_leg_scale)
    rear_displacement = _joint_displacement(arrays, prep_frames, fix_frames, f"{rear}_ankle", lead_leg_scale)
    put("lead_leg_side_match", lead in {"left", "right"})
    foot_floor, foot_floor_evidence = displacement_floor("foot_still_ratio", lead_displacement, rear_displacement)
    below_floor_reason = "displacement_below_noise_floor" if foot_floor_evidence["noise_floor"]["applied"] else None
    put(
        "moving_foot_side_match",
        _side_displacement_match(contract["expected_moving_foot"], lead, lead_displacement, rear_displacement, noise_floor=foot_floor),
        window=("preparation_anchor", FIXATION_ANCHOR), reason=below_floor_reason, evidence=foot_floor_evidence,
    )
    put(
        "expected_support_or_pivot_foot_match",
        _side_displacement_match(contract["expected_pivot_or_support_foot"], rear, rear_displacement, lead_displacement, support=True, noise_floor=foot_floor),
        window=("preparation_anchor", FIXATION_ANCHOR), reason=below_floor_reason, evidence=foot_floor_evidence,
    )
    put("moving_foot_match", result["moving_foot_side_match"]["value"], window=("preparation_anchor", FIXATION_ANCHOR), reason=below_floor_reason, evidence=foot_floor_evidence)
    put("support_or_pivot_foot_match", result["expected_support_or_pivot_foot_match"]["value"], window=("preparation_anchor", FIXATION_ANCHOR), reason=below_floor_reason, evidence=foot_floor_evidence)
    put("pivot_foot_displacement_body_ratio", rear_displacement, window=("preparation_anchor", FIXATION_ANCHOR))
    put("support_foot_displacement_body_ratio", rear_displacement, window=("preparation_anchor", FIXATION_ANCHOR))
    put("foot_landing_position_error_body_ratio", _stance_range_error(contract, stance, lead_leg_scale, shoulder_width), window=FIXATION_ANCHOR)
    put("foot_crossing_margin_body_ratio", _safe_ratio(stance.get("crossing_margin"), body_scale), window=FIXATION_ANCHOR)
    put("heel_alignment_error_body_ratio", _safe_ratio(stance.get("heel_alignment"), body_scale), window=FIXATION_ANCHOR)
    put("toe_alignment_error_body_ratio", _safe_ratio(stance.get("toe_alignment"), body_scale), window=FIXATION_ANCHOR)
    put("front_back_foot_order_match", stance.get("front_back_order_match"), window=FIXATION_ANCHOR)
    put("stance_depth_or_pelvis_height_leg_ratio", _pelvis_height_leg_ratio(arrays, fix_frames, lead_leg_scale), window=FIXATION_ANCHOR)
    put("inter_foot_yaw_difference_deg", _foot_yaw_difference(arrays, fix_frames, lead, rear), window=FIXATION_ANCHOR)
    front_foot_dispersion = _foot_dispersion(arrays, hold_frames, lead, lead_leg_scale)
    rear_foot_dispersion = _foot_dispersion(arrays, hold_frames, rear, lead_leg_scale)
    fixation_foot_dispersion = (
        max(front_foot_dispersion, rear_foot_dispersion)
        if front_foot_dispersion is not None and rear_foot_dispersion is not None
        else None
    )
    put("foot_fixation_slip_body_ratio", fixation_foot_dispersion, window=POST_FIXATION_HOLD)
    put("stance_fixation_dispersion_body_ratio", fixation_foot_dispersion, window=POST_FIXATION_HOLD)

    front_knee = _median_scalar(arrays, fix_frames, lambda a, f: _leg_angle(a, f, lead))
    rear_knee = _median_scalar(arrays, fix_frames, lambda a, f: _leg_angle(a, f, rear))
    put("front_knee_flexion_deg", front_knee, window=FIXATION_ANCHOR)
    put("rear_knee_flexion_deg", rear_knee, window=FIXATION_ANCHOR)
    knee_expectation = contract["knee_angle_or_alignment_expectation"]
    knee_match = None if knee_expectation is None else _all_known(_range_match(front_knee, knee_expectation["front_included_angle_deg"]), _range_match(rear_knee, knee_expectation["rear_included_angle_deg"]))
    put("expected_knee_flexion_range_match", knee_match, window=FIXATION_ANCHOR)
    put("knee_over_foot_alignment_ratio", _median_scalar(arrays, fix_frames, lambda a, f: _knee_foot_alignment(a, f, lead)), window=FIXATION_ANCHOR)
    put("knee_valgus_varus_proxy_deg", _median_scalar(arrays, fix_frames, lambda a, f: _knee_plane_angle(a, f, lead)), window=FIXATION_ANCHOR)
    put("hip_knee_ankle_plane_deviation_ratio", _median_scalar(arrays, fix_frames, lambda a, f: _knee_plane_deviation(a, f, lead)), window=FIXATION_ANCHOR)
    put("lead_leg_identity_match", lead in {"left", "right"})
    put("rear_leg_identity_match", rear in {"left", "right"} and rear != lead)
    put("lower_body_left_right_swap_state", False if lead in {"left", "right"} and rear != lead else None)
    put("pelvis_height_stance_match", _range_match(front_knee, knee_expectation["front_included_angle_deg"] if knee_expectation else None), window=FIXATION_ANCHOR)
    lower_dispersion = _lower_dispersion(arrays, hold_frames, lead_leg_scale)
    put("lower_body_fixation_dispersion", lower_dispersion, window=POST_FIXATION_HOLD)

    arm_floor, arm_floor_evidence = displacement_floor("hand_still_ratio", *_arm_side_displacements(arrays, prep_frames, exec_frames, active, reaction))
    arm_side = _arm_side_match(arrays, prep_frames, exec_frames, active, reaction, noise_floor=arm_floor)
    arm_reason = "displacement_below_noise_floor" if arm_floor_evidence["noise_floor"]["applied"] else None
    put("active_arm_side_match", arm_side, window=("preparation_anchor", "execution_anchor"), reason=arm_reason, evidence=arm_floor_evidence)
    put("reaction_or_chamber_arm_side_match", active in {"left", "right"} and reaction in {"left", "right"} and active != reaction)
    put("shoulder_elevation_body_ratio", _median_scalar(arrays, fix_frames, lambda a, f: _shoulder_elevation(a, f, active)), window=FIXATION_ANCHOR)
    put("shoulder_abduction_deg", _median_scalar(arrays, fix_frames, lambda a, f: _shoulder_angle(a, f, active, "abduction")), window=FIXATION_ANCHOR)
    put("shoulder_flexion_deg", _median_scalar(arrays, fix_frames, lambda a, f: _shoulder_angle(a, f, active, "flexion")), window=FIXATION_ANCHOR)
    elbow = _median_scalar(arrays, fix_frames, lambda a, f: _arm_angle(a, f, active))
    put("elbow_flexion_deg", elbow, window=FIXATION_ANCHOR)
    put("elbow_target_angle_error_deg", None, reason="missing_numeric_technique_target")
    put("active_arm_extension_ratio", _arm_extension(arrays, fix_frames, active), window=FIXATION_ANCHOR)
    local_hand = _hand_local_coordinates(arrays, fix_frames, active)
    put("active_arm_target_height_body_ratio", local_hand.get("height"), window=FIXATION_ANCHOR)
    put("active_arm_target_lateral_offset_body_ratio", local_hand.get("lateral"), window=FIXATION_ANCHOR)
    put("active_arm_target_depth_body_ratio", local_hand.get("depth"), window=FIXATION_ANCHOR)
    put("active_hand_target_distance_body_ratio", local_hand.get("distance"), window=FIXATION_ANCHOR)
    put("reaction_hand_target_distance_body_ratio", _reaction_hip_distance(arrays, fix_frames, reaction), window=FIXATION_ANCHOR)
    put("reaction_elbow_position_error_body_ratio", _reaction_elbow_distance(arrays, fix_frames, reaction), window=FIXATION_ANCHOR)
    put("arm_target_overshoot_body_ratio", _trajectory_overshoot(arrays, transition_frames, active, active_arm_scale), window="segment_start_to_fixation")

    # wrist_forearm_alignment_deg is not measured here: its single implementation is the
    # palm-plausibility-gated WholeBody measurement bound in the rule layer.
    wrist_components = _wrist_components(arrays, fix_frames, active)
    put("wrist_flexion_extension_proxy_deg", wrist_components.get("flexion"), window=FIXATION_ANCHOR)
    put("wrist_radial_ulnar_deviation_proxy_deg", wrist_components.get("radial"), window=FIXATION_ANCHOR)
    put("fist_or_hand_orientation_proxy_deg", wrist_components.get("orientation"), window=FIXATION_ANCHOR)
    put("hand_shape_observability", hand_quality is not None and hand_quality >= limits["group_quality_ratio"], window=FIXATION_ANCHOR)
    active_hand_dispersion = _hand_dispersion(arrays, hold_frames, active, active_arm_scale)
    reaction_hand_dispersion = _hand_dispersion(arrays, hold_frames, reaction, _median_scalar(arrays, fix_frames, lambda a, f: _limb_scale(a, f, "arm", reaction)))
    put("active_hand_fixation_stability", active_hand_dispersion, window=POST_FIXATION_HOLD)
    put("reaction_hand_fixation_stability", reaction_hand_dispersion, window=POST_FIXATION_HOLD)
    put("fist_target_height_error_body_ratio", _technique_target_error(arrays, fix_frames, contract, active, "height"), window=FIXATION_ANCHOR)
    put("fist_target_lateral_error_body_ratio", _technique_target_error(arrays, fix_frames, contract, active, "lateral"), window=FIXATION_ANCHOR)
    put("fist_target_depth_error_body_ratio", None, reason="missing_numeric_technique_depth_target")

    put("preparation_pose_observed", "preparation" in anchors)
    put("preparation_side_match", arm_side, window=("preparation_anchor", "execution_anchor"), reason=arm_reason, evidence=arm_floor_evidence)
    put("chamber_pose_observed", _reaction_hip_distance(arrays, prep_frames, reaction) is not None, window="preparation_anchor")
    put("chamber_precedes_execution", preparation <= execution)
    put("active_technique_reaches_target_by_fixation", None if active_hand_dispersion is None else active_hand_dispersion <= limits["hand_still_ratio"], window=POST_FIXATION_HOLD)
    put("stance_reaches_target_by_fixation", stance_match, window=FIXATION_ANCHOR)
    put("head_reaches_target_by_fixation", _component_settled(arrays, hold_frames, "head", limits["head_settled_deg"]), window=POST_FIXATION_HOLD)
    put("torso_reaches_target_by_fixation", _component_settled(arrays, hold_frames, "shoulder", limits["torso_settled_deg"]), window=POST_FIXATION_HOLD)
    put("reaction_hand_reaches_target_by_fixation", _reaction_hip_distance(arrays, fix_frames, reaction) is not None, window=FIXATION_ANCHOR)
    put("head_torso_settle_offset", _settle_offset(arrays, transition_frames, "head", "shoulder", fps, limits["head_settled_deg"], limits["torso_settled_deg"]), window="segment_start_to_fixation")
    put("late_post_fixation_correction", _late_component_correction(arrays, hold_frames, active, active_arm_scale), window=POST_FIXATION_HOLD)
    put("premature_next_movement_transition", _premature_transition(arrays, tail_frames, active, active_arm_scale, limits["late_correction_ratio"]), window=POST_FIXATION_TAIL)
    put("fixation_pose_stability", None if active_hand_dispersion is None or lower_dispersion is None else active_hand_dispersion <= limits["hand_still_ratio"] and lower_dispersion <= limits["foot_still_ratio"], window=POST_FIXATION_HOLD)
    put("final_pose_geometry_conformance", _finite_value(stance_match) and _finite_value(elbow), window=FIXATION_ANCHOR)

    put("step_length_error_body_ratio", _stance_range_component_error(contract["stance_length_expectation"], _safe_ratio(stance.get("length"), lead_leg_scale)), window=FIXATION_ANCHOR)
    put("step_width_error_body_ratio", _stance_range_component_error(contract["stance_width_expectation"], _safe_ratio(stance.get("width"), shoulder_width)), window=FIXATION_ANCHOR)
    put("landing_position_error_body_ratio", result["foot_landing_position_error_body_ratio"]["value"], window=FIXATION_ANCHOR)
    put("landing_stance_type_match", stance_match, window=FIXATION_ANCHOR)
    head_settled = _component_settled(arrays, hold_frames, "head", limits["head_settled_deg"])
    put("head_turn_settled_by_fixation", head_settled, window=POST_FIXATION_HOLD)
    put("torso_turn_settled_by_fixation", _component_settled(arrays, hold_frames, "shoulder", limits["torso_settled_deg"]), window=POST_FIXATION_HOLD)
    put("pelvis_or_stance_turn_settled_by_fixation", _component_settled(arrays, hold_frames, "hip", limits["pelvis_settled_deg"]), window=POST_FIXATION_HOLD)
    put("pivot_foot_excess_displacement", rear_displacement, window=("preparation_anchor", FIXATION_ANCHOR))
    put("transition_foot_crossing", (stance.get("crossing_margin") or 0.0) < 0.0 if stance.get("crossing_margin") is not None else None, window=FIXATION_ANCHOR)
    put("post_landing_adjustment", lower_dispersion, window=POST_FIXATION_HOLD)

    put("head_fixation_dispersion", _orientation_dispersion(arrays, hold_frames, "head"), window=POST_FIXATION_HOLD)
    put("torso_fixation_dispersion", _orientation_dispersion(arrays, hold_frames, "shoulder"), window=POST_FIXATION_HOLD)
    put("pelvis_fixation_dispersion", _orientation_dispersion(arrays, hold_frames, "hip"), window=POST_FIXATION_HOLD)
    put("active_hand_fixation_dispersion", active_hand_dispersion, window=POST_FIXATION_HOLD)
    put("reaction_hand_fixation_dispersion", reaction_hand_dispersion, window=POST_FIXATION_HOLD)
    put("front_foot_fixation_dispersion", front_foot_dispersion, window=POST_FIXATION_HOLD)
    put("rear_foot_fixation_dispersion", rear_foot_dispersion, window=POST_FIXATION_HOLD)
    put("knee_fixation_dispersion", _knee_dispersion(arrays, hold_frames, lead_leg_scale), window=POST_FIXATION_HOLD)
    put("body_height_fixation_drift", result["body_height_change_during_fixation_ratio"]["value"], window=POST_FIXATION_HOLD)
    put("torso_translation_after_fixation", result["torso_translation_after_fixation_body_ratio"]["value"], window=POST_FIXATION_HOLD)
    put("pelvis_translation_after_fixation", result["pelvis_translation_after_fixation_body_ratio"]["value"], window=POST_FIXATION_HOLD)
    put("stance_width_after_fixation_change", _stance_change(arrays, hold_frames, lead, rear, "width", shoulder_width), window=POST_FIXATION_HOLD)
    put("stance_length_after_fixation_change", _stance_change(arrays, hold_frames, lead, rear, "length", lead_leg_scale), window=POST_FIXATION_HOLD)
    put("late_balance_recovery_proxy", result["pelvis_translation_after_fixation_body_ratio"]["value"], window=POST_FIXATION_HOLD)
    put("final_geometry_consistent_with_declared_movement", bool(stance_match) if stance_match is not None else None, window=FIXATION_ANCHOR)
    put("movement_completion_state", fixation <= end and "fixation" in anchors)

    put("head_turn_completion_state", head_settled, window=POST_FIXATION_HOLD)
    put("head_late_turn_state", not bool(head_settled) if head_settled is not None else None, window=POST_FIXATION_HOLD)

    _direction_metrics(result, put, arrays, contract, fix_frames, direction, active, limits)
    _kick_metrics(result, put, arrays, contract, anchors, start, end, kick_side, rear, lead_leg_scale)
    apply_landmark_quality_gate(result, arrays, segment, profile, contract)
    return result


def measure_athlete_forward_vector(
    arrays: dict[str, Any],
    frames: np.ndarray,
    min_valid_samples: int,
) -> np.ndarray | None:
    """Median horizontal facing of the athlete over a window; None when evidence is thin.

    The per-frame estimate is the existing torso-forward geometry: the shoulder line
    crossed with world up, sign-resolved by the observed face direction. This measures
    where the athlete faced; it does not claim any world or judging reference.
    """
    vectors = [
        value
        for frame in frames
        if (value := _torso_forward(arrays, int(frame))) is not None
    ]
    if len(vectors) < max(3, int(min_valid_samples)):
        return None
    return _unit(np.median(np.asarray(vectors, dtype=float), axis=0))


def _direction_metrics(result: dict[str, dict[str, Any]], put: Callable[..., None], arrays: dict[str, Any], contract: dict[str, Any], frames: np.ndarray, direction: dict[str, Any] | None, active: str | None, limits: dict[str, float]) -> None:
    ids = (
        "head_target_yaw_error_deg", "torso_target_yaw_error_deg", "pelvis_target_yaw_error_deg",
        "stance_axis_target_yaw_error_deg", "front_foot_target_yaw_error_deg", "rear_foot_target_yaw_error_deg",
        "foot_progression_axis_error_deg", "knee_target_direction_error_deg", "active_arm_target_direction_error_deg",
        "fist_target_direction_error_deg", "expected_direction_change_match", "rotation_direction_sign_match",
        "step_direction_error_deg", "head_turn_direction_sign_match", "head_wrong_direction_stable_state",
        "kick_direction_error_deg", "expected_direction_contract_resolved",
    )
    if direction is None:
        for metric_id in ids:
            put(metric_id, None, reason="missing_athlete_local_direction_binding")
        return
    target = np.asarray(direction[contract["semantic_action_direction"] + "_vector"], dtype=float)
    vectors = {
        "head_target_yaw_error_deg": _median_vector(arrays, frames, _head_forward),
        "torso_target_yaw_error_deg": _median_vector(arrays, frames, _torso_forward),
        "pelvis_target_yaw_error_deg": _median_vector(arrays, frames, _pelvis_forward),
        "stance_axis_target_yaw_error_deg": _median_vector(arrays, frames, lambda a, f: _stance_axis(a, f, contract["lead_leg"], contract["rear_or_support_leg"])),
        "front_foot_target_yaw_error_deg": _median_vector(arrays, frames, lambda a, f: _foot_axis(a, f, contract["lead_leg"])),
        "rear_foot_target_yaw_error_deg": _median_vector(arrays, frames, lambda a, f: _foot_axis(a, f, contract["rear_or_support_leg"])),
        "foot_progression_axis_error_deg": _median_vector(arrays, frames, lambda a, f: _stance_axis(a, f, contract["lead_leg"], contract["rear_or_support_leg"])),
        "knee_target_direction_error_deg": _median_vector(arrays, frames, lambda a, f: _knee_axis(a, f, contract["lead_leg"])),
        "active_arm_target_direction_error_deg": _median_vector(arrays, frames, lambda a, f: _arm_axis(a, f, active)),
        "fist_target_direction_error_deg": _median_vector(arrays, frames, lambda a, f: _hand_axis(a, f, active)),
    }
    window = "fixation_anchor"
    for metric_id, vector in vectors.items():
        put(metric_id, _horizontal_angle(vector, target), window=window)
    direction_ok = result["stance_axis_target_yaw_error_deg"]["value"]
    put(
        "expected_direction_change_match",
        None if direction_ok is None else direction_ok <= limits["stance_direction_deg"],
        window=window,
    )
    put("rotation_direction_sign_match", None if direction_ok is None else direction_ok <= 90.0, window=window)
    put("step_direction_error_deg", direction_ok, window=window)
    head_error = result["head_target_yaw_error_deg"]["value"]
    put("head_turn_direction_sign_match", None if head_error is None else head_error <= 90.0, window=window)
    put("head_wrong_direction_stable_state", None if head_error is None else head_error > 90.0, window=window)
    put("expected_direction_contract_resolved", True)


def gated_frames(measurement: dict[str, Any], segment: dict[str, Any]) -> tuple[np.ndarray, str]:
    """Frames the measurement actually read, from its recorded evidence window.

    A measurement that records no window falls back to its whole movement segment,
    which is never more permissive than the frames it read.
    """
    start, end = int(segment["start_frame"]), int(segment["end_frame"])
    evidence = measurement.get("evidence")
    window = evidence.get("window") if isinstance(evidence, dict) else None
    windows = window if isinstance(window, list) else [window] if isinstance(window, dict) else []
    frames: set[int] = set()
    for item in windows:
        first, last = item.get("start_frame"), item.get("end_frame")
        if isinstance(first, int) and isinstance(last, int) and first <= last:
            frames.update(range(max(start, first), min(end, last) + 1))
    if frames:
        return np.asarray(sorted(frames), dtype=int), "evidence_window"
    return np.arange(start, end + 1, dtype=int), "movement_segment_no_window_recorded"


def apply_landmark_quality_gate(
    measurements: dict[str, dict[str, Any]],
    arrays: dict[str, Any],
    segment: dict[str, Any],
    profile: dict[str, Any],
    contract: dict[str, Any],
) -> None:
    """The one landmark gate for every technical-accuracy measurement.

    Every landmark the metric's evaluator reads for this movement's sides must
    itself reach ``min_group_valid_ratio`` over the frames the measurement read.
    Well-observed extra points cannot rescue a poorly observed required landmark,
    and occlusion outside the read frames does not discard the measurement.
    The per-frame quality mask and the ratio itself are unchanged. Idempotent.
    """
    minimum = float(profile["quality_gates"]["min_group_valid_ratio"])
    for metric_id, measurement in measurements.items():
        spec = METRIC_REGISTRY.get(metric_id)
        if spec is None:
            raise ScoringContractError(f"measurement has no registry record: {metric_id}")
        if spec.quality_group == "quality_metric" or measurement.get("value") is None:
            continue
        landmarks = resolve_metric_landmarks(spec, contract)
        if not landmarks:
            continue
        frames, source = gated_frames(measurement, segment)
        ratios = np.mean(arrays["quality_mask"][np.ix_(frames, np.asarray(landmarks, dtype=int))], axis=0)
        weakest = int(np.argmin(ratios))
        evidence = measurement.get("evidence")
        if not isinstance(evidence, dict):
            evidence = {}
            measurement["evidence"] = evidence
        evidence["landmark_gate"] = {
            "rule": "every_required_landmark_at_least_minimum_over_read_frames",
            "gated_start_frame": int(frames[0]),
            "gated_end_frame": int(frames[-1]),
            "gated_frame_count": int(len(frames)),
            "gated_frames_source": source,
            "landmark_valid_ratios": {str(index): float(ratio) for index, ratio in zip(landmarks, ratios, strict=True)},
            "weakest_landmark_index": int(landmarks[weakest]),
            "weakest_landmark_valid_ratio": float(ratios[weakest]),
            "minimum_required_ratio": minimum,
        }
        evidence["required_landmark_indices"] = landmarks
        evidence["required_landmark_valid_ratio"] = float(ratios[weakest])
        evidence["minimum_required_landmark_valid_ratio"] = minimum
        if float(ratios[weakest]) < minimum:
            measurement["value"] = None
            measurement["quality_status"] = "unmeasurable"
            measurement["not_measurable_reason"] = "required_landmark_quality_below_minimum"


def _kick_metrics(result: dict[str, dict[str, Any]], put: Callable[..., None], arrays: dict[str, Any], contract: dict[str, Any], anchors: dict[str, int], start: int, end: int, kick_side: str | None, support_side: str | None, leg_scale: float | None) -> None:
    if "ap_chagi" not in contract["technique_types"] or kick_side not in {"left", "right"}:
        return
    chamber = int(anchors.get("preparation", start))
    apex = int(anchors.get("kick_apex", anchors.get("kick_execution", chamber)))
    rechamber = int(anchors.get("rechamber", apex))
    landing = int(anchors.get("landing", end))
    put("chamber_knee_height_body_ratio", _knee_height(arrays, chamber, kick_side, leg_scale))
    put("chamber_hip_flexion_deg", _hip_flexion(arrays, chamber, kick_side))
    put("chamber_knee_flexion_deg", _median_scalar(arrays, _window(chamber, start, end, 2), lambda a, f: _leg_angle(a, f, kick_side)) if "preparation" in anchors else None, reason="missing_kick_preparation_anchor")
    put("support_leg_stability", _joint_span(arrays, np.arange(chamber, apex + 1), f"{support_side}_ankle", leg_scale))
    put("kick_extension_deg", _median_scalar(arrays, _window(apex, start, end, 2), lambda a, f: _leg_angle(a, f, kick_side)) if "kick_apex" in anchors or "kick_execution" in anchors else None, reason="missing_kick_execution_anchor")
    if not ("preparation" in anchors and ("kick_apex" in anchors or "kick_execution" in anchors)):
        put("support_leg_stability", None, reason="missing_kick_phase_anchors")
    put("kick_target_height_body_ratio", _ankle_height(arrays, apex, kick_side, leg_scale))
    put("kick_retraction_state", _leg_angle(arrays, rechamber, kick_side) is not None)
    put("kick_landing_stance_restoration", _pt(arrays, landing, f"{kick_side}_ankle") is not None)


def _window(anchor: int, start: int, end: int, radius: int) -> np.ndarray:
    return np.arange(max(start, anchor - radius), min(end, anchor + radius) + 1, dtype=int)


def _finite_value(value: Any) -> bool:
    if isinstance(value, (bool, str)):
        return True
    if value is None:
        return False
    try:
        return bool(np.all(np.isfinite(value)))
    except TypeError:
        return False


def _pt(arrays: dict[str, Any], frame: int, name: str) -> np.ndarray | None:
    value = arrays["points"][frame, COCO_BODY_JOINTS[name]]
    return value if np.all(np.isfinite(value)) else None


def _idx(arrays: dict[str, Any], frame: int, index: int) -> np.ndarray | None:
    value = arrays["points"][frame, index]
    return value if np.all(np.isfinite(value)) else None


def _median_scalar(arrays: dict[str, Any], frames: np.ndarray, fn: Callable[[dict[str, Any], int], float | None]) -> float | None:
    values = [value for frame in frames if (value := fn(arrays, int(frame))) is not None and np.isfinite(value)]
    return None if len(values) < 3 else float(np.median(values))


def _median_vector(arrays: dict[str, Any], frames: np.ndarray, fn: Callable[[dict[str, Any], int], np.ndarray | None]) -> np.ndarray | None:
    values = [value for frame in frames if (value := fn(arrays, int(frame))) is not None]
    if len(values) < 3:
        return None
    vector = np.median(np.asarray(values), axis=0)
    return _unit(vector)


def _unit(vector: np.ndarray | None) -> np.ndarray | None:
    if vector is None:
        return None
    norm = float(np.linalg.norm(vector))
    return None if norm <= 1e-8 else vector / norm


def _body_scale(arrays: dict[str, Any], frame: int) -> float | None:
    sc, hc = _centres(arrays, frame)
    return None if sc is None or hc is None else _positive_norm(sc - hc)


def _shoulder_width(arrays: dict[str, Any], frame: int) -> float | None:
    left, right = _pt(arrays, frame, "left_shoulder"), _pt(arrays, frame, "right_shoulder")
    return None if left is None or right is None else _positive_norm(right - left)


def _limb_scale(arrays: dict[str, Any], frame: int, kind: str, side: str | None) -> float | None:
    if side not in {"left", "right"}:
        return None
    parts = ("shoulder", "elbow", "wrist") if kind == "arm" else ("hip", "knee", "ankle")
    points = [_pt(arrays, frame, f"{side}_{part}") for part in parts]
    if any(point is None for point in points):
        return None
    first, second = _positive_norm(points[0] - points[1]), _positive_norm(points[1] - points[2])
    return None if first is None or second is None else first + second


def _positive_norm(vector: np.ndarray) -> float | None:
    value = float(np.linalg.norm(vector))
    return value if value > 1e-8 else None


def _safe_ratio(value: float | None, scale: float | None) -> float | None:
    return None if value is None or scale is None or scale <= 1e-8 else float(value / scale)


def _centres(arrays: dict[str, Any], frame: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    ls, rs, lh, rh = (_pt(arrays, frame, name) for name in ("left_shoulder", "right_shoulder", "left_hip", "right_hip"))
    if any(point is None for point in (ls, rs, lh, rh)):
        return None, None
    return (ls + rs) / 2.0, (lh + rh) / 2.0


def _group_quality(arrays: dict[str, Any], frames: np.ndarray, indices: tuple[int, ...] | list[int]) -> float | None:
    if not indices or len(frames) == 0:
        return None
    return float(np.mean(arrays["quality_mask"][np.ix_(frames, np.asarray(indices, dtype=int))]))


def _head_indices() -> tuple[int, ...]:
    return (*range(23 + 17, 23 + 27), *range(23 + 36, 23 + 48), 23 + 8)


def _torso_indices() -> tuple[int, ...]:
    return tuple(COCO_BODY_JOINTS[name] for name in ("left_shoulder", "right_shoulder", "left_hip", "right_hip"))


def _lower_indices() -> tuple[int, ...]:
    return tuple(COCO_BODY_JOINTS[name] for name in ("left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"))


def _upper_indices() -> tuple[int, ...]:
    return tuple(
        COCO_BODY_JOINTS[name]
        for name in (
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
        )
    )


def _hand_indices(side: str | None) -> tuple[int, ...]:
    if side not in {"left", "right"}:
        return ()
    offset = COCO_LEFT_HAND_OFFSET if side == "left" else COCO_RIGHT_HAND_OFFSET
    return tuple(range(offset, offset + 21))


def _foot_indices(side: str | None) -> tuple[int, ...]:
    if side not in {"left", "right"}:
        return ()
    return tuple(COCO_FOOT_JOINTS[f"{side}_{name}"] for name in ("big_toe", "small_toe", "heel"))


def _phase_quality(anchors: dict[str, int]) -> float:
    required = {"preparation", "fixation"}
    execution = any(name in anchors for name in ("execution", "punch_execution", "kick_apex"))
    return (len(required & set(anchors)) + int(execution)) / 3.0


def _minimum_finite(*values: float | None) -> float | None:
    finite = [value for value in values if value is not None]
    return min(finite) if len(finite) == len(values) else None


def _orientation_line(arrays: dict[str, Any], frame: int, part: str) -> np.ndarray | None:
    if part == "head":
        left = _cluster(arrays, frame, FACE_LEFT_EYE_LOCAL)
        right = _cluster(arrays, frame, FACE_RIGHT_EYE_LOCAL)
    else:
        left, right = _pt(arrays, frame, f"left_{part}"), _pt(arrays, frame, f"right_{part}")
    return None if left is None or right is None else _unit(right - left)


def _cluster(arrays: dict[str, Any], frame: int, local: range | tuple[int, ...]) -> np.ndarray | None:
    values = arrays["points"][frame, [23 + index for index in local]]
    valid = values[np.all(np.isfinite(values), axis=1)]
    return None if len(valid) == 0 else np.median(valid, axis=0)


def _orientation_dispersion(arrays: dict[str, Any], frames: np.ndarray, part: str) -> float | None:
    vectors = [value for frame in frames if (value := _orientation_line(arrays, int(frame), part)) is not None]
    if len(vectors) < 3:
        return None
    centre = _unit(np.median(np.asarray(vectors), axis=0))
    angles = [_angle(vector, centre) for vector in vectors]
    return None if centre is None else float(np.percentile(angles, 95))


def _orientation_drift(arrays: dict[str, Any], frames: np.ndarray, part: str) -> float | None:
    if len(frames) < 2:
        return None
    return _angle(_orientation_line(arrays, int(frames[0]), part), _orientation_line(arrays, int(frames[-1]), part))


def _angle(first: np.ndarray | None, second: np.ndarray | None) -> float | None:
    if first is None or second is None:
        return None
    denom = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denom <= 1e-8:
        return None
    return float(np.degrees(np.arccos(np.clip(np.dot(first, second) / denom, -1.0, 1.0))))


def _horizontal_angle(first: np.ndarray | None, second: np.ndarray | None) -> float | None:
    if first is None or second is None:
        return None
    return _angle(_unit(np.asarray([first[0], first[1], 0.0])), _unit(np.asarray([second[0], second[1], 0.0])))


def _torso_forward_lean(arrays: dict[str, Any], frame: int) -> float | None:
    sc, hc = _centres(arrays, frame)
    forward = _torso_forward(arrays, frame)
    if sc is None or hc is None or forward is None:
        return None
    torso = sc - hc
    return float(np.degrees(np.arctan2(np.dot(torso, forward), torso[2])))


def _line_roll(arrays: dict[str, Any], frame: int, part: str) -> float | None:
    line = _orientation_line(arrays, frame, part)
    if line is None:
        return None
    return float(np.degrees(np.arctan2(line[2], np.linalg.norm(line[:2]))))


def _roll_difference(arrays: dict[str, Any], frame: int) -> float | None:
    shoulder, hip = _line_roll(arrays, frame, "shoulder"), _line_roll(arrays, frame, "hip")
    return None if shoulder is None or hip is None else abs(shoulder - hip)


def _height_asymmetry(arrays: dict[str, Any], frame: int, part: str) -> float | None:
    left, right = _pt(arrays, frame, f"left_{part}"), _pt(arrays, frame, f"right_{part}")
    scale = _body_scale(arrays, frame)
    return None if left is None or right is None else _safe_ratio(abs(float(left[2] - right[2])), scale)


def _centre_dispersion(arrays: dict[str, Any], frames: np.ndarray, part: str) -> float | None:
    values: list[np.ndarray] = []
    scales: list[float] = []
    for frame in frames:
        left, right = _pt(arrays, int(frame), f"left_{part}"), _pt(arrays, int(frame), f"right_{part}")
        scale = _body_scale(arrays, int(frame))
        if left is not None and right is not None and scale is not None:
            values.append((left + right) / 2.0)
            scales.append(scale)
    if len(values) < 3:
        return None
    centre = np.median(np.asarray(values), axis=0)
    return float(np.percentile(np.linalg.norm(np.asarray(values) - centre, axis=1) / scales, 95))


def _body_height_change(arrays: dict[str, Any], frames: np.ndarray) -> float | None:
    values: list[float] = []
    for frame in frames:
        sc, _ = _centres(arrays, int(frame))
        la, ra = _pt(arrays, int(frame), "left_ankle"), _pt(arrays, int(frame), "right_ankle")
        scale = _body_scale(arrays, int(frame))
        if sc is not None and la is not None and ra is not None and scale is not None:
            values.append(float((sc[2] - ((la + ra) / 2.0)[2]) / scale))
    return None if len(values) < 3 else float(max(values) - min(values))


def _stance_geometry(arrays: dict[str, Any], frames: np.ndarray, lead: str | None, rear: str | None) -> dict[str, Any]:
    if lead not in {"left", "right"} or rear not in {"left", "right"}:
        return {}
    rows: list[dict[str, float]] = []
    for frame in frames:
        front, back = _foot_centre(arrays, int(frame), lead), _foot_centre(arrays, int(frame), rear)
        lateral = _orientation_line(arrays, int(frame), "shoulder")
        if front is None or back is None or lateral is None:
            continue
        lateral = _unit(np.asarray([lateral[0], lateral[1], 0.0]))
        if lateral is None:
            continue
        forward = _torso_forward(arrays, int(frame))
        delta = front - back
        lead_heel, rear_heel = _foot_part(arrays, int(frame), lead, "heel"), _foot_part(arrays, int(frame), rear, "heel")
        lead_toe, rear_toe = _toe_centre(arrays, int(frame), lead), _toe_centre(arrays, int(frame), rear)
        if forward is None:
            continue
        rows.append({
            "length": abs(float(np.dot(delta, forward))),
            "width": abs(float(np.dot(delta, lateral))),
            "crossing_margin": float(np.dot(delta, lateral)) * (-1.0 if lead == "left" else 1.0),
            "heel_alignment": abs(float(np.dot(lead_heel - rear_heel, lateral))) if lead_heel is not None and rear_heel is not None else np.nan,
            "toe_alignment": abs(float(np.dot(lead_toe - rear_toe, lateral))) if lead_toe is not None and rear_toe is not None else np.nan,
            "front_back_order_match": float(np.dot(delta, forward)) >= 0.0,
        })
    if len(rows) < 3:
        return {}
    return {key: (bool(np.mean([row[key] for row in rows]) >= 0.5) if key == "front_back_order_match" else float(np.nanmedian([row[key] for row in rows]))) for key in rows[0]}


def _foot_part(arrays: dict[str, Any], frame: int, side: str, part: str) -> np.ndarray | None:
    return _idx(arrays, frame, COCO_FOOT_JOINTS[f"{side}_{part}"])


def _toe_centre(arrays: dict[str, Any], frame: int, side: str) -> np.ndarray | None:
    big, small = _foot_part(arrays, frame, side, "big_toe"), _foot_part(arrays, frame, side, "small_toe")
    return None if big is None or small is None else (big + small) / 2.0


def _foot_centre(arrays: dict[str, Any], frame: int, side: str) -> np.ndarray | None:
    heel, toe = _foot_part(arrays, frame, side, "heel"), _toe_centre(arrays, frame, side)
    return None if heel is None or toe is None else (heel + toe) / 2.0


def _foot_axis(arrays: dict[str, Any], frame: int, side: str | None) -> np.ndarray | None:
    if side not in {"left", "right"}:
        return None
    heel, toe = _foot_part(arrays, frame, side, "heel"), _toe_centre(arrays, frame, side)
    return None if heel is None or toe is None else _unit(toe - heel)


def _foot_yaw_difference(arrays: dict[str, Any], frames: np.ndarray, lead: str | None, rear: str | None) -> float | None:
    if lead not in {"left", "right"} or rear not in {"left", "right"}:
        return None
    return _median_scalar(arrays, frames, lambda a, f: _horizontal_angle(_foot_axis(a, f, lead), _foot_axis(a, f, rear)))


def _range_match(value: float | None, limits: list[float] | None) -> bool | None:
    return None if value is None or limits is None else bool(float(limits[0]) <= value <= float(limits[1]))


def _stance_range_component_error(limits: list[float] | None, value: float | None) -> float | None:
    if limits is None or value is None:
        return None
    return 0.0 if limits[0] <= value <= limits[1] else float(min(abs(value - limits[0]), abs(value - limits[1])))


def _stance_range_error(contract: dict[str, Any], stance: dict[str, Any], leg: float | None, shoulder: float | None) -> float | None:
    length = _stance_range_component_error(contract["stance_length_expectation"], _safe_ratio(stance.get("length"), leg))
    width = _stance_range_component_error(contract["stance_width_expectation"], _safe_ratio(stance.get("width"), shoulder))
    return None if length is None or width is None else float(np.hypot(length, width))


def _all_known(*values: bool | None) -> bool | None:
    return None if any(value is None for value in values) else all(values)


def _side_displacement_match(
    expected: str | None,
    measured_side: str | None,
    expected_disp: float | None,
    other_disp: float | None,
    *,
    noise_floor: float,
    support: bool = False,
) -> bool | None:
    """Which side moved more; None when neither side moved more than the noise floor.

    ``noise_floor`` is the profile's own "did not move" limit in the displacement's
    unit. Comparing two displacements that are both inside it would turn jitter
    into a left/right verdict, so the comparison is reported as unmeasurable.
    """
    if expected is None:
        return None
    if expected != measured_side or expected_disp is None or other_disp is None:
        return None
    if max(expected_disp, other_disp) <= noise_floor:
        return None
    if abs(expected_disp - other_disp) <= 1e-8:
        return None
    return expected_disp <= other_disp if support else expected_disp >= other_disp


def _joint_displacement(arrays: dict[str, Any], before: np.ndarray, after: np.ndarray, joint: str, scale: float | None) -> float | None:
    first = _median_point(arrays, before, joint)
    last = _median_point(arrays, after, joint)
    return None if first is None or last is None else _safe_ratio(float(np.linalg.norm(last - first)), scale)


def _median_point(arrays: dict[str, Any], frames: np.ndarray, name: str) -> np.ndarray | None:
    values = [value for frame in frames if (value := _pt(arrays, int(frame), name)) is not None]
    return None if len(values) < 3 else np.median(np.asarray(values), axis=0)


def _pelvis_height_leg_ratio(arrays: dict[str, Any], frames: np.ndarray, leg_scale: float | None) -> float | None:
    values = []
    for frame in frames:
        _, hip = _centres(arrays, int(frame))
        la, ra = _pt(arrays, int(frame), "left_ankle"), _pt(arrays, int(frame), "right_ankle")
        if hip is not None and la is not None and ra is not None:
            values.append(float(hip[2] - ((la + ra) / 2.0)[2]))
    return None if len(values) < 3 else _safe_ratio(float(np.median(values)), leg_scale)


def _leg_angle(arrays: dict[str, Any], frame: int, side: str | None) -> float | None:
    if side not in {"left", "right"}:
        return None
    hip, knee, ankle = (_pt(arrays, frame, f"{side}_{part}") for part in ("hip", "knee", "ankle"))
    return None if hip is None or knee is None or ankle is None else _angle(hip - knee, ankle - knee)


def _arm_angle(arrays: dict[str, Any], frame: int, side: str | None) -> float | None:
    if side not in {"left", "right"}:
        return None
    shoulder, elbow, wrist = (_pt(arrays, frame, f"{side}_{part}") for part in ("shoulder", "elbow", "wrist"))
    return None if shoulder is None or elbow is None or wrist is None else _angle(shoulder - elbow, wrist - elbow)


def _knee_foot_alignment(arrays: dict[str, Any], frame: int, side: str | None) -> float | None:
    if side not in {"left", "right"}:
        return None
    knee, foot = _pt(arrays, frame, f"{side}_knee"), _foot_centre(arrays, frame, side)
    scale = _limb_scale(arrays, frame, "leg", side)
    return None if knee is None or foot is None else _safe_ratio(float(np.linalg.norm(knee[:2] - foot[:2])), scale)


def _knee_plane_angle(arrays: dict[str, Any], frame: int, side: str | None) -> float | None:
    if side not in {"left", "right"}:
        return None
    hip, knee, ankle = (_pt(arrays, frame, f"{side}_{part}") for part in ("hip", "knee", "ankle"))
    if hip is None or knee is None or ankle is None:
        return None
    return _horizontal_angle(knee - hip, ankle - knee)


def _knee_plane_deviation(arrays: dict[str, Any], frame: int, side: str | None) -> float | None:
    if side not in {"left", "right"}:
        return None
    hip, knee, ankle = (_pt(arrays, frame, f"{side}_{part}") for part in ("hip", "knee", "ankle"))
    scale = _limb_scale(arrays, frame, "leg", side)
    if hip is None or knee is None or ankle is None or scale is None:
        return None
    line = ankle - hip
    denominator = float(np.dot(line, line))
    if denominator <= 1e-8:
        return None
    nearest = hip + np.dot(knee - hip, line) / denominator * line
    return float(np.linalg.norm(knee - nearest) / scale)


def _lower_dispersion(arrays: dict[str, Any], frames: np.ndarray, scale: float | None) -> float | None:
    values = [_joint_span(arrays, frames, name, scale) for name in ("left_knee", "right_knee", "left_ankle", "right_ankle")]
    return max(values) if all(value is not None for value in values) else None


def _joint_span(arrays: dict[str, Any], frames: np.ndarray, name: str, scale: float | None) -> float | None:
    points = [value for frame in frames if (value := _pt(arrays, int(frame), name)) is not None]
    if len(points) < 3 or scale is None:
        return None
    centre = np.median(np.asarray(points), axis=0)
    return float(np.percentile(np.linalg.norm(np.asarray(points) - centre, axis=1), 95) / scale)


def _arm_side_displacements(arrays: dict[str, Any], before: np.ndarray, after: np.ndarray, active: str | None, reaction: str | None) -> tuple[float | None, float | None]:
    if active not in {"left", "right"} or reaction not in {"left", "right"}:
        return None, None
    scale = _median_scalar(arrays, after, lambda a, f: _limb_scale(a, f, "arm", active))
    active_move = _joint_displacement(arrays, before, after, f"{active}_wrist", scale)
    reaction_move = _joint_displacement(arrays, before, after, f"{reaction}_wrist", scale)
    return active_move, reaction_move


def _arm_side_match(arrays: dict[str, Any], before: np.ndarray, after: np.ndarray, active: str | None, reaction: str | None, *, noise_floor: float) -> bool | None:
    active_move, reaction_move = _arm_side_displacements(arrays, before, after, active, reaction)
    if active_move is None or reaction_move is None or max(active_move, reaction_move) <= noise_floor:
        return None
    return active_move >= reaction_move


def _shoulder_elevation(arrays: dict[str, Any], frame: int, side: str | None) -> float | None:
    if side not in {"left", "right"}:
        return None
    shoulder = _pt(arrays, frame, f"{side}_shoulder")
    other = _pt(arrays, frame, f"{'right' if side == 'left' else 'left'}_shoulder")
    scale = _body_scale(arrays, frame)
    return None if shoulder is None or other is None else _safe_ratio(float(shoulder[2] - other[2]), scale)


def _trunk_axes(arrays: dict[str, Any], frame: int) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Trunk-fixed (down, athlete-right, forward) axes from shoulders and hips only.

    ``down`` runs from the shoulder centre to the hip centre; ``right`` is the
    shoulder line (left -> right shoulder) made orthogonal to it; ``forward`` is
    up x right, so the frame is right-handed and follows the trunk when it leans
    or turns. No world axis is used.
    """
    ls, rs = _pt(arrays, frame, "left_shoulder"), _pt(arrays, frame, "right_shoulder")
    sc, hc = _centres(arrays, frame)
    if ls is None or rs is None or sc is None or hc is None:
        return None
    down = _unit(hc - sc)
    if down is None:
        return None
    line = rs - ls
    right = _unit(line - np.dot(line, down) * down)
    if right is None:
        return None
    forward = _unit(np.cross(-down, right))
    return None if forward is None else (down, right, forward)


def _shoulder_angle(arrays: dict[str, Any], frame: int, side: str | None, component: str) -> float | None:
    """Anatomical shoulder flexion or abduction of the humerus, in degrees.

    Both are 0 with the upper arm hanging along the trunk. Flexion is the humerus
    angle in the trunk's sagittal plane (down/forward; +90 = raised forward,
    negative = extension behind the trunk). Abduction is the angle in the trunk's
    frontal plane (down/outward; +90 = raised sideways, negative = adduction
    across the body). A component is unmeasurable when the humerus is
    perpendicular to its plane, because its projection then has no direction.
    """
    if side not in {"left", "right"}:
        return None
    shoulder, elbow = _pt(arrays, frame, f"{side}_shoulder"), _pt(arrays, frame, f"{side}_elbow")
    axes = _trunk_axes(arrays, frame)
    if shoulder is None or elbow is None or axes is None:
        return None
    arm = _unit(elbow - shoulder)
    if arm is None:
        return None
    down, right, forward = axes
    if component == "flexion":
        second = forward
    elif component == "abduction":
        second = right if side == "right" else -right
    else:
        raise ValueError(f"unknown shoulder angle component: {component}")
    along_down, along_second = float(np.dot(arm, down)), float(np.dot(arm, second))
    if float(np.hypot(along_down, along_second)) <= 1e-8:
        return None
    return float(np.degrees(np.arctan2(along_second, along_down)))


def _arm_extension(arrays: dict[str, Any], frames: np.ndarray, side: str | None) -> float | None:
    values = []
    for frame in frames:
        if side not in {"left", "right"}:
            continue
        shoulder, wrist = _pt(arrays, int(frame), f"{side}_shoulder"), _pt(arrays, int(frame), f"{side}_wrist")
        scale = _limb_scale(arrays, int(frame), "arm", side)
        if shoulder is not None and wrist is not None and scale is not None:
            values.append(float(np.linalg.norm(wrist - shoulder) / scale))
    return None if len(values) < 3 else float(np.median(values))


def _hand_offset(side: str) -> int:
    return COCO_LEFT_HAND_OFFSET if side == "left" else COCO_RIGHT_HAND_OFFSET


def _hand_point(arrays: dict[str, Any], frame: int, side: str | None, joint: str) -> np.ndarray | None:
    if side not in {"left", "right"}:
        return None
    return _idx(arrays, frame, _hand_offset(side) + COCO_HAND_LOCAL_JOINTS[joint])


def _hand_centre(arrays: dict[str, Any], frame: int, side: str | None) -> np.ndarray | None:
    points = [_hand_point(arrays, frame, side, name) for name in ("wrist", "index_mcp", "middle_mcp", "ring_mcp", "pinky_mcp")]
    return None if any(point is None for point in points) else np.mean(points, axis=0)


def _hand_local_coordinates(arrays: dict[str, Any], frames: np.ndarray, side: str | None) -> dict[str, float | None]:
    rows = []
    for frame in frames:
        hand = _hand_centre(arrays, int(frame), side)
        sc, hc = _centres(arrays, int(frame))
        lateral = _orientation_line(arrays, int(frame), "shoulder")
        forward = _torso_forward(arrays, int(frame))
        torso = _body_scale(arrays, int(frame))
        shoulder = _shoulder_width(arrays, int(frame))
        arm = _limb_scale(arrays, int(frame), "arm", side)
        if hand is None or sc is None or hc is None or lateral is None or forward is None or torso is None or shoulder is None or arm is None:
            continue
        delta = hand - hc
        rows.append((float(delta[2] / torso), float(np.dot(delta, lateral) / shoulder), float(np.dot(delta, forward) / arm), float(np.linalg.norm(delta) / torso)))
    if len(rows) < 3:
        return {key: None for key in ("height", "lateral", "depth", "distance")}
    values = np.median(np.asarray(rows), axis=0)
    return dict(zip(("height", "lateral", "depth", "distance"), map(float, values), strict=True))


def _reaction_hip_distance(arrays: dict[str, Any], frames: np.ndarray, side: str | None) -> float | None:
    values = []
    for frame in frames:
        hand, hip = _hand_centre(arrays, int(frame), side), _pt(arrays, int(frame), f"{side}_hip") if side in {"left", "right"} else None
        scale = _body_scale(arrays, int(frame))
        if hand is not None and hip is not None and scale is not None:
            values.append(float(np.linalg.norm(hand - hip) / scale))
    return None if len(values) < 3 else float(np.median(values))


def _reaction_elbow_distance(arrays: dict[str, Any], frames: np.ndarray, side: str | None) -> float | None:
    values = []
    for frame in frames:
        elbow, hip = _pt(arrays, int(frame), f"{side}_elbow") if side in {"left", "right"} else None, _pt(arrays, int(frame), f"{side}_hip") if side in {"left", "right"} else None
        scale = _limb_scale(arrays, int(frame), "arm", side)
        if elbow is not None and hip is not None and scale is not None:
            values.append(float(np.linalg.norm(elbow - hip) / scale))
    return None if len(values) < 3 else float(np.median(values))


def _trajectory_overshoot(arrays: dict[str, Any], frames: np.ndarray, side: str | None, scale: float | None) -> float | None:
    if side not in {"left", "right"} or scale is None:
        return None
    points = [_hand_centre(arrays, int(frame), side) for frame in frames]
    points = [point for point in points if point is not None]
    if len(points) < 3:
        return None
    final = points[-1]
    direct = final - points[0]
    axis = _unit(direct)
    if axis is None:
        return 0.0
    projections = [float(np.dot(point - final, axis)) for point in points]
    return max(0.0, max(projections)) / scale


def _wrist_components(arrays: dict[str, Any], frames: np.ndarray, side: str | None) -> dict[str, float | None]:
    rows = []
    for frame in frames:
        elbow = _pt(arrays, int(frame), f"{side}_elbow") if side in {"left", "right"} else None
        wrist = _hand_point(arrays, int(frame), side, "wrist")
        middle = _hand_point(arrays, int(frame), side, "middle_mcp")
        index = _hand_point(arrays, int(frame), side, "index_mcp")
        pinky = _hand_point(arrays, int(frame), side, "pinky_mcp")
        if any(point is None for point in (elbow, wrist, middle, index, pinky)):
            continue
        forearm, hand = _unit(wrist - elbow), _unit(middle - wrist)
        palm_lateral = _unit(index - pinky)
        palm_normal = _unit(np.cross(hand, palm_lateral)) if hand is not None and palm_lateral is not None else None
        total = _angle(forearm, hand)
        if total is None or palm_normal is None or palm_lateral is None:
            continue
        rows.append((total, float(np.degrees(np.arcsin(np.clip(np.dot(forearm, palm_normal), -1.0, 1.0)))), float(np.degrees(np.arcsin(np.clip(np.dot(forearm, palm_lateral), -1.0, 1.0)))), _horizontal_angle(hand, _torso_forward(arrays, int(frame)))))
    if len(rows) < 3:
        return {key: None for key in ("total", "flexion", "radial", "orientation")}
    columns = np.asarray(rows, dtype=float).T
    values = [
        float(np.median(column[np.isfinite(column)])) if np.any(np.isfinite(column)) else None
        for column in columns
    ]
    return dict(zip(("total", "flexion", "radial", "orientation"), values, strict=True))


def _hand_dispersion(arrays: dict[str, Any], frames: np.ndarray, side: str | None, scale: float | None) -> float | None:
    values = [_hand_centre(arrays, int(frame), side) for frame in frames]
    values = [value for value in values if value is not None]
    if len(values) < 3 or scale is None:
        return None
    centre = np.median(np.asarray(values), axis=0)
    return float(np.percentile(np.linalg.norm(np.asarray(values) - centre, axis=1), 95) / scale)


def _technique_target_error(arrays: dict[str, Any], frames: np.ndarray, contract: dict[str, Any], side: str | None, component: str) -> float | None:
    values = []
    for frame in frames:
        hand = _hand_centre(arrays, int(frame), side)
        sc, hc = _centres(arrays, int(frame))
        if hand is None or sc is None or hc is None:
            continue
        if contract["active_hand_expected_height"] == "above_forehead":
            target = _cluster(arrays, int(frame), tuple(range(17, 27)))
        elif contract["active_hand_expected_height"] == "below_pelvis":
            target = (hc + _pt(arrays, int(frame), f"{side}_knee")) / 2.0 if side in {"left", "right"} and _pt(arrays, int(frame), f"{side}_knee") is not None else None
        else:
            target = (sc + hc) / 2.0
        if target is None:
            continue
        scale = _body_scale(arrays, int(frame)) if component == "height" else _shoulder_width(arrays, int(frame))
        if scale is None:
            continue
        lateral = _orientation_line(arrays, int(frame), "shoulder")
        if component != "height" and lateral is None:
            continue
        values.append(abs(float(hand[2] - target[2])) / scale if component == "height" else abs(float(np.dot(hand - target, lateral))) / scale)
    return None if len(values) < 3 else float(np.median(values))


def _component_settled(arrays: dict[str, Any], frames: np.ndarray, part: str, limit: float) -> bool | None:
    value = _orientation_dispersion(arrays, frames, part)
    return None if value is None else value <= limit


def _settle_offset(
    arrays: dict[str, Any],
    frames: np.ndarray,
    first: str,
    second: str,
    fps: float,
    first_limit: float,
    second_limit: float,
) -> float | None:
    first_frame = _settle_frame(arrays, frames, first, first_limit)
    second_frame = _settle_frame(arrays, frames, second, second_limit)
    return None if first_frame is None or second_frame is None or fps <= 0 else float((first_frame - second_frame) / fps)


def _settle_frame(arrays: dict[str, Any], frames: np.ndarray, part: str, limit: float) -> int | None:
    if len(frames) < 4:
        return None
    vectors = [_orientation_line(arrays, int(frame), part) for frame in frames]
    final = vectors[-1]
    if final is None:
        return None
    for index, vector in enumerate(vectors[:-2]):
        if vector is not None and _angle(vector, final) is not None and _angle(vector, final) <= limit:
            return int(frames[index])
    return int(frames[-1])


def _late_component_correction(arrays: dict[str, Any], frames: np.ndarray, side: str | None, scale: float | None) -> float | None:
    if len(frames) < 2 or scale is None:
        return None
    first, last = _hand_centre(arrays, int(frames[0]), side), _hand_centre(arrays, int(frames[-1]), side)
    return None if first is None or last is None else float(np.linalg.norm(last - first) / scale)


def _premature_transition(arrays: dict[str, Any], frames: np.ndarray, side: str | None, scale: float | None, limit: float) -> bool | None:
    correction = _late_component_correction(arrays, frames, side, scale)
    return None if correction is None else correction > limit


def _foot_dispersion(arrays: dict[str, Any], frames: np.ndarray, side: str | None, scale: float | None) -> float | None:
    if side not in {"left", "right"} or scale is None:
        return None
    values = [_foot_centre(arrays, int(frame), side) for frame in frames]
    values = [value for value in values if value is not None]
    if len(values) < 3:
        return None
    centre = np.median(np.asarray(values), axis=0)
    return float(np.percentile(np.linalg.norm(np.asarray(values) - centre, axis=1), 95) / scale)


def _knee_dispersion(arrays: dict[str, Any], frames: np.ndarray, scale: float | None) -> float | None:
    left, right = (_joint_span(arrays, frames, f"{side}_knee", scale) for side in ("left", "right"))
    return None if left is None or right is None else max(left, right)


def _stance_change(arrays: dict[str, Any], frames: np.ndarray, lead: str | None, rear: str | None, component: str, scale: float | None) -> float | None:
    if len(frames) < 2 or scale is None:
        return None
    first = _stance_geometry(arrays, frames[: min(3, len(frames))], lead, rear).get(component)
    last = _stance_geometry(arrays, frames[-min(3, len(frames)) :], lead, rear).get(component)
    return None if first is None or last is None else abs(float(last - first)) / scale


def _head_forward(arrays: dict[str, Any], frame: int) -> np.ndarray | None:
    """Horizontal head facing: body nose relative to the midpoint of the two ears.

    The face eye clusters are not used: when the athlete is side-on to the cameras
    they collapse in 3D (about 2-3 cm apart instead of ~6 cm), and a nose-minus-eyes
    vector then points anywhere. The nose-to-ear-midpoint baseline is ~10 cm and
    stayed within 10 deg of the torso facing on every observed movement.
    """
    nose = _pt(arrays, frame, "nose")
    left_ear, right_ear = _pt(arrays, frame, "left_ear"), _pt(arrays, frame, "right_ear")
    if nose is None or left_ear is None or right_ear is None:
        return None
    facing = nose - (left_ear + right_ear) / 2.0
    return _unit(np.asarray([facing[0], facing[1], 0.0]))


def _torso_forward(arrays: dict[str, Any], frame: int) -> np.ndarray | None:
    """Horizontal facing from the anatomical shoulder line: up x (right - left).

    The sign comes from the left/right labels alone. Resolving it with the face
    (nose minus eye midpoint) flipped the torso by 180 deg whenever the head was
    turned or seen in profile, because that vector is short and often nearly
    perpendicular to the shoulder normal.
    """
    lateral = _orientation_line(arrays, frame, "shoulder")
    if lateral is None:
        return None
    return _unit(np.cross(np.asarray([0.0, 0.0, 1.0]), lateral))


def _pelvis_forward(arrays: dict[str, Any], frame: int) -> np.ndarray | None:
    lateral = _orientation_line(arrays, frame, "hip")
    torso = _torso_forward(arrays, frame)
    candidate = None if lateral is None else _unit(np.cross(np.asarray([0.0, 0.0, 1.0]), lateral))
    return -candidate if candidate is not None and torso is not None and np.dot(candidate, torso) < 0 else candidate


def _stance_axis(arrays: dict[str, Any], frame: int, lead: str | None, rear: str | None) -> np.ndarray | None:
    if lead not in {"left", "right"} or rear not in {"left", "right"}:
        return None
    front, back = _foot_centre(arrays, frame, lead), _foot_centre(arrays, frame, rear)
    return None if front is None or back is None else _unit(front - back)


def _knee_axis(arrays: dict[str, Any], frame: int, side: str | None) -> np.ndarray | None:
    if side not in {"left", "right"}:
        return None
    hip, knee = _pt(arrays, frame, f"{side}_hip"), _pt(arrays, frame, f"{side}_knee")
    return None if hip is None or knee is None else _unit(knee - hip)


def _arm_axis(arrays: dict[str, Any], frame: int, side: str | None) -> np.ndarray | None:
    if side not in {"left", "right"}:
        return None
    shoulder, wrist = _pt(arrays, frame, f"{side}_shoulder"), _hand_centre(arrays, frame, side)
    return None if shoulder is None or wrist is None else _unit(wrist - shoulder)


def _hand_axis(arrays: dict[str, Any], frame: int, side: str | None) -> np.ndarray | None:
    wrist, middle = _hand_point(arrays, frame, side, "wrist"), _hand_point(arrays, frame, side, "middle_mcp")
    return None if wrist is None or middle is None else _unit(middle - wrist)


def _knee_height(arrays: dict[str, Any], frame: int, side: str, scale: float | None) -> float | None:
    knee, hip = _pt(arrays, frame, f"{side}_knee"), _pt(arrays, frame, f"{side}_hip")
    return None if knee is None or hip is None else _safe_ratio(float(knee[2] - hip[2]), scale)


def _hip_flexion(arrays: dict[str, Any], frame: int, side: str) -> float | None:
    shoulder, hip, knee = (_pt(arrays, frame, f"{side}_{part}") for part in ("shoulder", "hip", "knee"))
    return None if shoulder is None or hip is None or knee is None else _angle(shoulder - hip, knee - hip)


def _ankle_height(arrays: dict[str, Any], frame: int, side: str, scale: float | None) -> float | None:
    ankle, hip = _pt(arrays, frame, f"{side}_ankle"), _pt(arrays, frame, f"{side}_hip")
    return None if ankle is None or hip is None else _safe_ratio(float(ankle[2] - hip[2]), scale)
