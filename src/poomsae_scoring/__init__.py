"""Rule-driven, explainable Poomsae scoring contracts and engines."""

from src.poomsae_scoring.accuracy import evaluate_accuracy
from src.poomsae_scoring.application import PoomsaeAnalysisResult, run_poomsae_analysis
from src.poomsae_scoring.automatic_segmentation import (
    build_automatic_segmentation_diagnostics,
    compare_segments_to_reference,
    detect_automatic_segments,
)
from src.poomsae_scoring.contracts import (
    ScoringContractError,
    load_engineering_profile,
    load_movement_timeline,
    load_poomsae_spec,
    load_rule_pack,
    validate_engineering_profile,
    validate_movement_timeline,
    validate_poomsae_spec,
    validate_rule_pack,
)
from src.poomsae_scoring.evidence import build_movement_evidence
from src.poomsae_scoring.decision_evidence import build_decision_evidence_events
from src.poomsae_scoring.categorical_diagnostics import build_categorical_diagnostics
from src.poomsae_scoring.engineering_trial import build_partial_engineering_trial
from src.poomsae_scoring.presentation import build_presentation_diagnostics
from src.poomsae_scoring.technical_conformance import build_technical_conformance
from src.poomsae_scoring.technical_accuracy import (
    build_technical_accuracy_diagnostics,
    derive_athlete_local_direction_reference,
    evaluate_temporary_threshold,
    load_technical_accuracy_profile,
    resolve_movement_accuracy_contracts,
    validate_athlete_local_direction_reference,
    validate_technical_accuracy_profile,
)
from src.poomsae_scoring.readiness import assess_accuracy_readiness
from src.poomsae_scoring.segmentation import (  # noqa: E402
    detect_movement_segments,
)
from src.poomsae_scoring.sequence_alignment import (
    align_segments_to_movements,
    build_automatic_movement_timeline,
    build_automatic_timeline_report,
)
from src.poomsae_scoring.review_report import build_review_html
from src.poomsae_scoring.run_history import build_run_history, build_run_history_html
from src.poomsae_scoring.source_intake import (
    inspect_source_intake,
    load_source_intake,
    validate_source_intake,
)
from src.poomsae_scoring.source_bound_accuracy import (
    build_source_bound_accuracy_decisions,
    derive_categorical_observations,
    load_source_bound_accuracy_profile,
    validate_source_bound_accuracy_result,
    validate_source_bound_accuracy_profile,
)
from src.poomsae_scoring.wholebody_diagnostics import (
    build_wholebody_diagnostics,
    load_wholebody_diagnostic_profile,
    validate_wholebody_diagnostic_profile,
)
from src.poomsae_scoring.overlay import overlay_state_for_frame, render_movement_overlay

__all__ = [
    "ScoringContractError",
    "PoomsaeAnalysisResult",
    "align_segments_to_movements",
    "assess_accuracy_readiness",
    "build_automatic_segmentation_diagnostics",
    "build_automatic_movement_timeline",
    "build_automatic_timeline_report",
    "detect_movement_segments",
    "build_movement_evidence",
    "build_presentation_diagnostics",
    "build_technical_conformance",
    "build_technical_accuracy_diagnostics",
    "derive_athlete_local_direction_reference",
    "build_decision_evidence_events",
    "build_categorical_diagnostics",
    "build_partial_engineering_trial",
    "build_review_html",
    "build_run_history",
    "build_run_history_html",
    "build_source_bound_accuracy_decisions",
    "build_wholebody_diagnostics",
    "compare_segments_to_reference",
    "detect_automatic_segments",
    "derive_categorical_observations",
    "evaluate_accuracy",
    "load_movement_timeline",
    "load_engineering_profile",
    "load_poomsae_spec",
    "load_rule_pack",
    "load_source_bound_accuracy_profile",
    "load_source_intake",
    "load_wholebody_diagnostic_profile",
    "load_technical_accuracy_profile",
    "overlay_state_for_frame",
    "render_movement_overlay",
    "run_poomsae_analysis",
    "inspect_source_intake",
    "validate_movement_timeline",
    "validate_engineering_profile",
    "validate_poomsae_spec",
    "validate_rule_pack",
    "validate_source_bound_accuracy_result",
    "validate_source_bound_accuracy_profile",
    "validate_source_intake",
    "validate_wholebody_diagnostic_profile",
    "validate_technical_accuracy_profile",
    "validate_athlete_local_direction_reference",
    "resolve_movement_accuracy_contracts",
    "evaluate_temporary_threshold",
]
