from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import yaml


class ScoringContractError(ValueError):
    """Raised when a scoring source or runtime artifact violates its contract."""


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"duplicate key: {key}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def load_rule_pack(path: str | Path) -> dict[str, Any]:
    return validate_rule_pack(_load_yaml_mapping(path))


def load_poomsae_spec(path: str | Path) -> dict[str, Any]:
    return validate_poomsae_spec(_load_yaml_mapping(path))


def load_movement_timeline(
    path: str | Path,
    poomsae_spec: dict[str, Any],
    *,
    require_complete: bool = False,
) -> dict[str, Any]:
    return validate_movement_timeline(
        _load_yaml_mapping(path),
        poomsae_spec,
        require_complete=require_complete,
    )


def load_engineering_profile(path: str | Path) -> dict[str, Any]:
    return validate_engineering_profile(_load_yaml_mapping(path))

def validate_rule_pack(payload: dict[str, Any]) -> dict[str, Any]:
    data = deepcopy(_require_mapping(payload, "rule pack"))
    _require_exact_keys(
        data,
        {
            "schema_version",
            "rule_pack_id",
            "version",
            "status",
            "authority",
            "discipline",
            "effective_date",
            "accessed_at",
            "source_documents",
            "scoring",
        },
        "rule pack",
    )
    if data["schema_version"] != 1:
        raise ScoringContractError("rule pack schema_version must be 1")
    _require_identifier(data["rule_pack_id"], "rule_pack_id")
    _require_nonempty_string(data["version"], "rule pack version")
    if data["status"] not in {"draft", "active", "retired"}:
        raise ScoringContractError("rule pack status must be draft, active or retired")
    _require_nonempty_string(data["authority"], "rule pack authority")
    if data["discipline"] != "recognized_poomsae":
        raise ScoringContractError("only recognized_poomsae rule packs are supported")
    _require_nonempty_string(data["effective_date"], "rule pack effective_date")
    _require_nonempty_string(data["accessed_at"], "rule pack accessed_at")

    source_ids = _validate_sources(data["source_documents"], "rule pack")
    scoring = _require_mapping(data["scoring"], "rule pack scoring")
    optional_presentation = scoring.pop("presentation", None)
    optional_final_deductions = scoring.pop("final_score_deductions", None)
    _require_exact_keys(scoring, {"total_score", "accuracy", "presentation_reserved_score"}, "rule pack scoring")
    total_score = _finite_nonnegative(scoring["total_score"], "total_score")
    presentation_score = _finite_nonnegative(scoring["presentation_reserved_score"], "presentation_reserved_score")

    accuracy = _require_mapping(scoring["accuracy"], "accuracy")
    _require_exact_keys(accuracy, {"initial_score", "deductions"}, "accuracy")
    initial_score = _finite_nonnegative(accuracy["initial_score"], "accuracy.initial_score")
    if not np.isclose(initial_score + presentation_score, total_score, atol=1e-9):
        raise ScoringContractError("accuracy.initial_score + presentation_reserved_score must equal total_score")
    if optional_presentation is not None:
        presentation = _require_mapping(optional_presentation, "presentation")
        _require_exact_keys(
            presentation,
            {"total_score", "scoring_mode", "scoring_note", "components"},
            "presentation",
        )
        _finite_nonnegative(presentation["total_score"], "presentation.total_score")
        _require_nonempty_string(presentation["scoring_mode"], "presentation.scoring_mode")
        _require_nonempty_string(presentation["scoring_note"], "presentation.scoring_note")
        components = _require_mapping(presentation["components"], "presentation.components")
        required_components = {"speed_and_power", "rhythm_and_tempo", "expression_of_energy"}
        if set(components) != required_components:
            raise ScoringContractError(
                "presentation.components must contain exactly speed_and_power, rhythm_and_tempo and expression_of_energy"
            )
        component_total = 0.0
        for name, raw_component in components.items():
            component = _require_mapping(raw_component, f"presentation.components.{name}")
            _require_exact_keys(component, {"score", "source_refs"}, f"presentation.components.{name}")
            component_total += _finite_nonnegative(component["score"], f"presentation.components.{name}.score")
            _validate_source_refs(component["source_refs"], source_ids, f"presentation.components.{name}.source_refs")
        if not np.isclose(component_total, presentation_score, atol=1e-9):
            raise ScoringContractError(
                "presentation.components scores must sum to presentation_reserved_score"
            )
        if not np.isclose(presentation["total_score"], presentation_score, atol=1e-9):
            raise ScoringContractError(
                "presentation.total_score must equal presentation_reserved_score"
            )

    deductions = _require_mapping(accuracy["deductions"], "accuracy.deductions")
    required_deductions = {"minor", "major", "restart"}
    if set(deductions) != required_deductions:
        raise ScoringContractError("accuracy.deductions must contain exactly minor, major and restart")
    normalized_deductions: dict[str, dict[str, Any]] = {}
    for kind, raw_rule in deductions.items():
        rule = _require_mapping(raw_rule, f"accuracy.deductions.{kind}")
        examples = rule.pop("categorical_examples", None)
        _require_exact_keys(
            rule,
            {"rule_id", "amount", "scope", "definition", "source_refs"},
            f"accuracy.deductions.{kind}",
        )
        _require_identifier(rule["rule_id"], f"{kind}.rule_id")
        amount = _finite_nonnegative(rule["amount"], f"{kind}.amount")
        if amount <= 0.0:
            raise ScoringContractError(f"{kind}.amount must be positive")
        if rule["scope"] not in {"individual_movement", "performance"}:
            raise ScoringContractError(f"{kind}.scope is unsupported")
        _require_nonempty_string(rule["definition"], f"{kind}.definition")
        _validate_source_refs(rule["source_refs"], source_ids, f"{kind}.source_refs")
        if examples is not None:
            if not isinstance(examples, list) or not examples:
                raise ScoringContractError(f"{kind}.categorical_examples must be a non-empty list")
            seen_example_ids: set[str] = set()
            normalized_examples: list[dict[str, Any]] = []
            for index, raw_example in enumerate(examples, start=1):
                label = f"{kind}.categorical_examples[{index}]"
                example = _require_mapping(raw_example, label)
                _require_exact_keys(example, {"id", "text", "measurable", "source_ref"}, label)
                example_id = _require_identifier(example["id"], f"{label}.id")
                if example_id in seen_example_ids:
                    raise ScoringContractError(f"duplicate categorical example id: {example_id}")
                seen_example_ids.add(example_id)
                _require_nonempty_string(example["text"], f"{label}.text")
                if example["measurable"] not in {"pose_only", "pose_and_spec", "audio_required"}:
                    raise ScoringContractError(f"{label}.measurable is unsupported")
                _validate_source_refs([example["source_ref"]], source_ids, f"{label}.source_ref")
                normalized_examples.append(example)
            rule["categorical_examples"] = normalized_examples
        normalized_deductions: dict[str, dict[str, Any]] = {}
    for kind, raw_rule in deductions.items():
        rule = _require_mapping(raw_rule, f"accuracy.deductions.{kind}")
        examples = rule.pop("categorical_examples", None)
        _require_exact_keys(
            rule,
            {"rule_id", "amount", "scope", "definition", "source_refs"},
            f"accuracy.deductions.{kind}",
        )
        _require_identifier(rule["rule_id"], f"{kind}.rule_id")
        amount = _finite_nonnegative(rule["amount"], f"{kind}.amount")
        if amount <= 0.0:
            raise ScoringContractError(f"{kind}.amount must be positive")
        if rule["scope"] not in {"individual_movement", "performance"}:
            raise ScoringContractError(f"{kind}.scope is unsupported")
        _require_nonempty_string(rule["definition"], f"{kind}.definition")
        _validate_source_refs(rule["source_refs"], source_ids, f"{kind}.source_refs")
        if examples is not None:
            if not isinstance(examples, list) or not examples:
                raise ScoringContractError(f"{kind}.categorical_examples must be a non-empty list")
            seen_example_ids: set[str] = set()
            normalized_examples: list[dict[str, Any]] = []
            for index, raw_example in enumerate(examples, start=1):
                label = f"{kind}.categorical_examples[{index}]"
                example = _require_mapping(raw_example, label)
                _require_exact_keys(example, {"id", "text", "measurable", "source_ref"}, label)
                example_id = _require_identifier(example["id"], f"{label}.id")
                if example_id in seen_example_ids:
                    raise ScoringContractError(f"duplicate categorical example id: {example_id}")
                seen_example_ids.add(example_id)
                _require_nonempty_string(example["text"], f"{label}.text")
                if example["measurable"] not in {"pose_only", "pose_and_spec", "audio_required"}:
                    raise ScoringContractError(f"{label}.measurable is unsupported")
                _validate_source_refs([example["source_ref"]], source_ids, f"{label}.source_ref")
                normalized_examples.append(example)
            rule["categorical_examples"] = normalized_examples
        normalized_deductions[kind] = {**rule, "amount": amount}

    data["scoring"] = {
        "total_score": total_score,
        "accuracy": {"initial_score": initial_score, "deductions": normalized_deductions},
        "presentation_reserved_score": presentation_score,
    }
    return data

def validate_poomsae_spec(payload: dict[str, Any]) -> dict[str, Any]:
    data = deepcopy(_require_mapping(payload, "PoomsaeSpec"))
    _require_exact_keys(
        data,
        {
            "schema_version",
            "poomsae_id",
            "version",
            "status",
            "display_name",
            "rule_pack_id",
            "sequence_status",
            "source_documents",
            "movements",
            "blocked_reasons",
        },
        "PoomsaeSpec",
    )
    if data["schema_version"] != 1:
        raise ScoringContractError("PoomsaeSpec schema_version must be 1")
    _require_identifier(data["poomsae_id"], "poomsae_id")
    _require_nonempty_string(data["version"], "PoomsaeSpec version")
    _require_nonempty_string(data["display_name"], "PoomsaeSpec display_name")
    _require_identifier(data["rule_pack_id"], "PoomsaeSpec rule_pack_id")
    if data["status"] not in {"draft", "active", "retired"}:
        raise ScoringContractError("PoomsaeSpec status must be draft, active or retired")
    if data["sequence_status"] not in {"source_inventory_incomplete", "source_transcribed", "active"}:
        raise ScoringContractError("PoomsaeSpec sequence_status is unsupported")
    source_ids = _validate_sources(data["source_documents"], "PoomsaeSpec")
    blocked_reasons = _require_string_list(data["blocked_reasons"], "PoomsaeSpec blocked_reasons")
    movements = data["movements"]
    if not isinstance(movements, list):
        raise ScoringContractError("PoomsaeSpec movements must be a list")

    normalized_movements: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for expected_index, raw_movement in enumerate(movements, start=1):
        movement = _require_mapping(raw_movement, f"movement {expected_index}")
        _require_exact_keys(
            movement,
            {
                "movement_id",
                "sequence_index",
                "display_name",
                "direction",
                "stance",
                "techniques",
                "phases",
                "measurable_criteria",
                "source_refs",
            },
            f"movement {expected_index}",
        )
        movement_id = _require_identifier(movement["movement_id"], f"movement {expected_index} id")
        if movement_id in seen_ids:
            raise ScoringContractError(f"duplicate movement_id: {movement_id}")
        seen_ids.add(movement_id)
        if movement["sequence_index"] != expected_index:
            raise ScoringContractError("movement sequence_index values must be contiguous and start at 1")
        for name in ("display_name", "direction", "stance"):
            _require_nonempty_string(movement[name], f"{movement_id}.{name}")
        if not isinstance(movement["techniques"], list) or not movement["techniques"]:
            raise ScoringContractError(f"{movement_id}.techniques must be a non-empty list")
        _require_string_list(movement["phases"], f"{movement_id}.phases", allow_empty=False)
        movement["measurable_criteria"] = _require_string_list(
            movement["measurable_criteria"],
            f"{movement_id}.measurable_criteria",
        )
        _validate_source_refs(movement["source_refs"], source_ids, f"{movement_id}.source_refs")
        normalized_movements.append(movement)

    if data["status"] == "active":
        if data["sequence_status"] != "active" or not normalized_movements:
            raise ScoringContractError("active PoomsaeSpec requires an active, non-empty movement sequence")
        if blocked_reasons:
            raise ScoringContractError("active PoomsaeSpec cannot have blocked_reasons")
    if data["sequence_status"] == "active" and data["status"] != "active":
        raise ScoringContractError("active sequence_status requires active PoomsaeSpec status")

    data["movements"] = normalized_movements
    data["blocked_reasons"] = blocked_reasons
    return data


def validate_engineering_profile(payload: dict[str, Any]) -> dict[str, Any]:
    data = deepcopy(_require_mapping(payload, "EngineeringToleranceProfile"))
    _require_exact_keys(
        data,
        {
            "schema_version",
            "profile_id",
            "version",
            "status",
            "rule_pack_id",
            "poomsae_id",
            "scope",
            "provenance",
            "quality_gates",
            "policy",
            "criteria",
        },
        "EngineeringToleranceProfile",
    )
    if data["schema_version"] != 1:
        raise ScoringContractError("EngineeringToleranceProfile schema_version must be 1")
    _require_identifier(data["profile_id"], "engineering profile_id")
    _require_nonempty_string(data["version"], "engineering profile version")
    if data["status"] != "provisional_engineering":
        raise ScoringContractError("engineering profile status must be provisional_engineering")
    _require_identifier(data["rule_pack_id"], "engineering profile rule_pack_id")
    _require_identifier(data["poomsae_id"], "engineering profile poomsae_id")

    scope = _require_mapping(data["scope"], "engineering profile scope")
    _require_exact_keys(
        scope,
        {"recording_scope", "movement_ids", "reference_accuracy_score"},
        "engineering profile scope",
    )
    if scope["recording_scope"] != "partial_sequence":
        raise ScoringContractError("engineering profile currently supports only partial_sequence")
    scope["movement_ids"] = _require_string_list(
        scope["movement_ids"], "engineering profile scope.movement_ids", allow_empty=False
    )
    scope["reference_accuracy_score"] = _finite_positive(
        scope["reference_accuracy_score"], "engineering profile reference_accuracy_score"
    )

    provenance = _require_mapping(data["provenance"], "engineering profile provenance")
    _require_exact_keys(
        provenance,
        {"threshold_origin", "rule_semantics_refs", "technique_semantics", "disclaimer"},
        "engineering profile provenance",
    )
    if provenance["threshold_origin"] != "engineering_hypothesis_not_official_rule":
        raise ScoringContractError("engineering threshold_origin must remain explicitly non-official")
    provenance["rule_semantics_refs"] = _require_string_list(
        provenance["rule_semantics_refs"], "engineering rule_semantics_refs", allow_empty=False
    )
    _require_nonempty_string(provenance["technique_semantics"], "engineering technique_semantics")
    _require_nonempty_string(provenance["disclaimer"], "engineering disclaimer")

    gates = _require_mapping(data["quality_gates"], "engineering quality_gates")
    _require_exact_keys(
        gates,
        {
            "min_body17_valid_ratio",
            "min_used_cameras",
            "max_median_reprojection_error_px",
            "min_label_confidence",
        },
        "engineering quality_gates",
    )
    gates["min_body17_valid_ratio"] = _finite_probability(
        gates["min_body17_valid_ratio"], "min_body17_valid_ratio"
    )
    gates["min_used_cameras"] = _finite_positive(gates["min_used_cameras"], "min_used_cameras")
    gates["max_median_reprojection_error_px"] = _finite_positive(
        gates["max_median_reprojection_error_px"], "max_median_reprojection_error_px"
    )
    gates["min_label_confidence"] = _finite_probability(
        gates["min_label_confidence"], "min_label_confidence"
    )

    policy = _require_mapping(data["policy"], "engineering policy")
    _require_exact_keys(
        policy,
        {
            "numeric_deviations_candidate_kind",
            "deduplicate_by",
            "automatic_major_detection",
            "score_field",
        },
        "engineering policy",
    )
    if policy["numeric_deviations_candidate_kind"] != "minor":
        raise ScoringContractError("numeric engineering deviations may only create minor candidates")
    if policy["deduplicate_by"] != "movement_family_phase":
        raise ScoringContractError("engineering candidates must deduplicate by movement_family_phase")
    if policy["automatic_major_detection"] is not False:
        raise ScoringContractError("automatic major detection must remain disabled")
    if policy["score_field"] != "partial_engineering_trial_score":
        raise ScoringContractError("engineering score_field must be partial_engineering_trial_score")

    criteria = data["criteria"]
    if not isinstance(criteria, list) or not criteria:
        raise ScoringContractError("engineering criteria must be a non-empty list")
    seen_ids: set[str] = set()
    normalized_criteria: list[dict[str, Any]] = []
    for raw_criterion in criteria:
        criterion = _require_mapping(raw_criterion, "engineering criterion")
        _require_exact_keys(
            criterion,
            {
                "criterion_id",
                "family",
                "selector",
                "phase_id",
                "metric",
                "acceptable_range",
                "deduction_kind",
                "rationale",
            },
            "engineering criterion",
        )
        criterion_id = _require_identifier(criterion["criterion_id"], "engineering criterion_id")
        if criterion_id in seen_ids:
            raise ScoringContractError(f"duplicate engineering criterion_id: {criterion_id}")
        seen_ids.add(criterion_id)
        _require_identifier(criterion["family"], f"{criterion_id}.family")
        _require_identifier(criterion["phase_id"], f"{criterion_id}.phase_id")
        _require_identifier(criterion["metric"], f"{criterion_id}.metric")
        if criterion["deduction_kind"] != "minor":
            raise ScoringContractError("engineering numeric criteria may only use minor deductions")
        _require_nonempty_string(criterion["rationale"], f"{criterion_id}.rationale")
        selector = _require_mapping(criterion["selector"], f"{criterion_id}.selector")
        _require_exact_keys(selector, {"field", "values"}, f"{criterion_id}.selector")
        if selector["field"] not in {"all", "stance", "technique"}:
            raise ScoringContractError(f"{criterion_id}.selector.field is unsupported")
        selector["values"] = _require_string_list(selector["values"], f"{criterion_id}.selector.values")
        if selector["field"] == "all" and selector["values"]:
            raise ScoringContractError(f"{criterion_id} all selector cannot have values")
        if selector["field"] != "all" and not selector["values"]:
            raise ScoringContractError(f"{criterion_id} selector values cannot be empty")
        limits = criterion["acceptable_range"]
        if not isinstance(limits, list) or len(limits) != 2:
            raise ScoringContractError(f"{criterion_id}.acceptable_range must have exactly two values")
        lower = _finite_nonnegative(limits[0], f"{criterion_id}.acceptable_range lower")
        upper = _finite_positive(limits[1], f"{criterion_id}.acceptable_range upper")
        if lower > upper:
            raise ScoringContractError(f"{criterion_id}.acceptable_range must be ordered")
        normalized_criteria.append({**criterion, "selector": selector, "acceptable_range": [lower, upper]})

    data["scope"] = scope
    data["provenance"] = provenance
    data["quality_gates"] = gates
    data["policy"] = policy
    data["criteria"] = normalized_criteria
    return data


def validate_movement_timeline(
    payload: dict[str, Any],
    poomsae_spec: dict[str, Any],
    *,
    require_complete: bool = False,
) -> dict[str, Any]:
    spec = validate_poomsae_spec(poomsae_spec)
    data = deepcopy(_require_mapping(payload, "MovementTimeline"))
    _require_exact_keys(
        data,
        {
            "schema_version",
            "timeline_id",
            "poomsae_id",
            "poomsae_version",
            "status",
            "label_source",
            "frame_index_space",
            "frame_count",
            "fps",
            "source_binding",
            "coverage",
            "segments",
        },
        "MovementTimeline",
    )
    if data["schema_version"] != 2:
        raise ScoringContractError("MovementTimeline schema_version must be 2")
    _require_identifier(data["timeline_id"], "timeline_id")
    if data["poomsae_id"] != spec["poomsae_id"] or data["poomsae_version"] != spec["version"]:
        raise ScoringContractError("MovementTimeline must bind to the exact PoomsaeSpec id and version")
    if data["status"] not in {"draft", "complete"}:
        raise ScoringContractError("MovementTimeline status must be draft or complete")
    if data["label_source"] not in {"manual", "automatic", "manual_reviewed_automatic"}:
        raise ScoringContractError("MovementTimeline label_source is unsupported")
    if data["frame_index_space"] != "sample_index":
        raise ScoringContractError("MovementTimeline frame_index_space must be sample_index")
    if not isinstance(data["frame_count"], int) or data["frame_count"] <= 0:
        raise ScoringContractError("MovementTimeline frame_count must be a positive integer")
    data["fps"] = _finite_positive(data["fps"], "MovementTimeline fps")
    source_binding = _require_mapping(data["source_binding"], "MovementTimeline source_binding")
    _require_exact_keys(
        source_binding,
        {"session_id", "run_id", "pose_file", "pose_file_sha256"},
        "MovementTimeline source_binding",
    )
    _require_identifier(source_binding["session_id"], "source_binding.session_id")
    _require_identifier(source_binding["run_id"], "source_binding.run_id")
    _require_nonempty_string(source_binding["pose_file"], "source_binding.pose_file")
    pose_sha256 = source_binding["pose_file_sha256"]
    if pose_sha256 is not None and not _is_sha256(pose_sha256):
        raise ScoringContractError("source_binding.pose_file_sha256 must be null or a 64-character hex digest")
    segments = data["segments"]
    if not isinstance(segments, list):
        raise ScoringContractError("MovementTimeline segments must be a list")

    spec_movements = {item["movement_id"]: item for item in spec["movements"]}
    normalized_segments: list[dict[str, Any]] = []
    previous_end = -1
    for expected_index, raw_segment in enumerate(segments, start=1):
        segment = _require_mapping(raw_segment, f"timeline segment {expected_index}")
        _require_exact_keys(
            segment,
            {
                "sequence_index",
                "movement_id",
                "start_frame",
                "end_frame",
                "anchors",
                "confidence",
                "label_status",
            },
            f"timeline segment {expected_index}",
        )
        if segment["sequence_index"] != expected_index:
            raise ScoringContractError("timeline sequence_index values must be contiguous and start at 1")
        movement_id = segment["movement_id"]
        if movement_id not in spec_movements:
            raise ScoringContractError(f"timeline references unknown movement_id: {movement_id}")
        if spec_movements[movement_id]["sequence_index"] != expected_index:
            raise ScoringContractError("timeline movement order must match PoomsaeSpec")
        start = _frame_index(segment["start_frame"], "start_frame", data["frame_count"])
        end = _frame_index(segment["end_frame"], "end_frame", data["frame_count"])
        if start > end:
            raise ScoringContractError("timeline frames must satisfy start_frame <= end_frame")
        if start <= previous_end:
            raise ScoringContractError("timeline segments cannot move backward or overlap")
        previous_end = end
        anchors = _require_mapping(segment["anchors"], f"{movement_id}.anchors")
        movement_phases = spec_movements[movement_id]["phases"]
        unexpected_anchors = set(anchors) - set(movement_phases)
        if unexpected_anchors:
            raise ScoringContractError(
                f"{movement_id}.anchors references unknown phases: {sorted(unexpected_anchors)}"
            )
        normalized_anchors: dict[str, int] = {}
        previous_anchor = start
        for phase_id in movement_phases:
            if phase_id not in anchors:
                continue
            anchor_frame = _frame_index(
                anchors[phase_id],
                f"{movement_id}.anchors.{phase_id}",
                data["frame_count"],
            )
            if not start <= anchor_frame <= end:
                raise ScoringContractError(f"{movement_id} anchor frames must fall inside the movement interval")
            if anchor_frame < previous_anchor:
                raise ScoringContractError(f"{movement_id} anchor frames must follow phase order")
            previous_anchor = anchor_frame
            normalized_anchors[phase_id] = anchor_frame
        confidence = _finite_probability(segment["confidence"], "timeline confidence")
        if segment["label_status"] not in {"confirmed", "provisional", "ambiguous"}:
            raise ScoringContractError("timeline label_status is unsupported")
        normalized_segments.append(
            {
                **segment,
                "start_frame": start,
                "end_frame": end,
                "anchors": normalized_anchors,
                "confidence": confidence,
            }
        )

    coverage = _require_mapping(data["coverage"], "MovementTimeline coverage")
    _require_exact_keys(
        coverage,
        {"recording_scope", "observed_movement_ids", "missing_movement_ids", "source_end_reason"},
        "MovementTimeline coverage",
    )
    if coverage["recording_scope"] not in {"complete_performance", "partial_sequence"}:
        raise ScoringContractError("MovementTimeline coverage.recording_scope is unsupported")
    observed_ids = _require_string_list(
        coverage["observed_movement_ids"],
        "MovementTimeline coverage.observed_movement_ids",
    )
    missing_ids = _require_string_list(
        coverage["missing_movement_ids"],
        "MovementTimeline coverage.missing_movement_ids",
    )
    expected_ids = [movement["movement_id"] for movement in spec["movements"]]
    segment_ids = [segment["movement_id"] for segment in normalized_segments]
    if observed_ids != segment_ids:
        raise ScoringContractError("coverage.observed_movement_ids must exactly match timeline segments")
    if len(set(observed_ids + missing_ids)) != len(observed_ids) + len(missing_ids):
        raise ScoringContractError("MovementTimeline coverage movement ids cannot repeat")
    if observed_ids + missing_ids != expected_ids:
        raise ScoringContractError("MovementTimeline coverage must partition the ordered PoomsaeSpec movements")
    if coverage["recording_scope"] == "complete_performance":
        if missing_ids or coverage["source_end_reason"] is not None:
            raise ScoringContractError("complete_performance coverage cannot have missing movements or an end reason")
    else:
        if not observed_ids or not missing_ids:
            raise ScoringContractError("partial_sequence coverage requires observed and missing movements")
        _require_nonempty_string(coverage["source_end_reason"], "coverage.source_end_reason")

    is_complete = (
        data["status"] == "complete"
        and spec["status"] == "active"
        and pose_sha256 is not None
        and len(normalized_segments) == len(spec["movements"])
        and coverage["recording_scope"] == "complete_performance"
        and all(segment["label_status"] == "confirmed" for segment in normalized_segments)
        and all(segment["anchors"] for segment in normalized_segments)
    )
    if data["status"] == "complete" and not is_complete:
        raise ScoringContractError("complete MovementTimeline requires every active spec movement with confirmed labels")
    if require_complete and not is_complete:
        raise ScoringContractError("Accuracy scoring requires a complete MovementTimeline and active PoomsaeSpec")
    data["segments"] = normalized_segments
    data["coverage"] = {
        **coverage,
        "observed_movement_ids": observed_ids,
        "missing_movement_ids": missing_ids,
    }
    return data


def _load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.exists():
        raise ScoringContractError(f"scoring contract not found: {source}")
    try:
        payload = yaml.load(source.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except yaml.YAMLError as exc:
        raise ScoringContractError(f"invalid scoring YAML: {exc}") from exc
    return _require_mapping(payload, str(source))


def _validate_sources(value: Any, label: str) -> set[str]:
    if not isinstance(value, list) or not value:
        raise ScoringContractError(f"{label} source_documents must be a non-empty list")
    source_ids: set[str] = set()
    for index, raw_source in enumerate(value, start=1):
        source = _require_mapping(raw_source, f"{label} source {index}")
        _require_exact_keys(
            source,
            {
                "source_id",
                "authority",
                "title",
                "url",
                "effective_date",
                "accessed_at",
                "language",
                "access",
                "content_sha256",
                "sections",
            },
            f"{label} source {index}",
        )
        source_id = _require_identifier(source["source_id"], f"{label} source_id")
        if source_id in source_ids:
            raise ScoringContractError(f"duplicate source_id: {source_id}")
        source_ids.add(source_id)
        for name in ("authority", "title", "url", "effective_date", "accessed_at", "language"):
            _require_nonempty_string(source[name], f"{source_id}.{name}")
        if source["access"] not in {"public", "paid", "restricted"}:
            raise ScoringContractError(f"{source_id}.access must be public, paid or restricted")
        content_sha256 = source["content_sha256"]
        if content_sha256 is not None and not _is_sha256(content_sha256):
            raise ScoringContractError(f"{source_id}.content_sha256 must be null or a 64-character hex digest")
        _require_string_list(source["sections"], f"{source_id}.sections", allow_empty=False)
    return source_ids


def _validate_source_refs(value: Any, source_ids: set[str], label: str) -> None:
    refs = _require_string_list(value, label, allow_empty=False)
    for ref in refs:
        source_id = ref.split("#", maxsplit=1)[0]
        if source_id not in source_ids:
            raise ScoringContractError(f"{label} references unknown source_id: {source_id}")


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ScoringContractError(f"{label} must be a mapping")
    return value


def _require_exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ScoringContractError(
            f"{label} keys are invalid; missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
        )


def _require_nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ScoringContractError(f"{label} must be a non-empty string")
    return value


def _require_identifier(value: Any, label: str) -> str:
    result = _require_nonempty_string(value, label)
    if any(character.isspace() for character in result):
        raise ScoringContractError(f"{label} cannot contain whitespace")
    return result


def _require_string_list(value: Any, label: str, *, allow_empty: bool = True) -> list[str]:
    if not isinstance(value, list) or (not allow_empty and not value):
        raise ScoringContractError(f"{label} must be {'a non-empty' if not allow_empty else 'a'} list")
    for item in value:
        _require_nonempty_string(item, label)
    return value


def _finite_nonnegative(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ScoringContractError(f"{label} must be numeric") from exc
    if not np.isfinite(result) or result < 0.0:
        raise ScoringContractError(f"{label} must be finite and non-negative")
    return result


def _finite_positive(value: Any, label: str) -> float:
    result = _finite_nonnegative(value, label)
    if result <= 0.0:
        raise ScoringContractError(f"{label} must be positive")
    return result


def _finite_probability(value: Any, label: str) -> float:
    result = _finite_nonnegative(value, label)
    if result > 1.0:
        raise ScoringContractError(f"{label} must be between 0 and 1")
    return result


def _frame_index(value: Any, label: str, frame_count: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < frame_count:
        raise ScoringContractError(f"{label} must be an integer in [0, {frame_count - 1}]")
    return value


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )
