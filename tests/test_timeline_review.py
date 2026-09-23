"""Trust regressions: labels are declarations, recorded review is scoped content."""
from copy import deepcopy

import pytest

from src.poomsae_scoring import (
    ScoringContractError, build_source_bound_accuracy_decisions,
    load_movement_timeline, load_poomsae_spec, load_source_bound_accuracy_profile,
    validate_movement_timeline,
)
from src.poomsae_scoring.timeline_review import timeline_review_digest, timeline_review_status
from timeline_review_support import reviewed_timeline
from test_poomsae_scoring import (
    DRAFT_SPEC_PATH, DRAFT_TIMELINE_PATH, SOURCE_BOUND_ACCURACY_PATH,
    _active_spec, _complete_timeline,
)


def _draft():
    spec = load_poomsae_spec(DRAFT_SPEC_PATH)
    return spec, load_movement_timeline(DRAFT_TIMELINE_PATH, spec)


def test_legacy_declarations_are_preserved_but_never_prove_review():
    _, timeline = _draft()
    original = deepcopy(timeline)
    assert timeline["label_source"] == "manual"
    assert all(item["label_status"] == "confirmed" for item in timeline["segments"])
    assert timeline_review_status(timeline)["reason"] == "missing_review_record"
    assert timeline == original


def test_content_bound_review_is_not_identity_or_accuracy_authentication():
    spec, timeline = _draft()
    reviewed = reviewed_timeline(timeline)
    loaded = validate_movement_timeline(reviewed, spec)
    status = timeline_review_status(loaded)
    assert status["status"] == "verified"
    assert status["reviewer_identity_authenticated"] is False
    assert status["independent_accuracy_verified"] is False
    assert timeline_review_digest(loaded) == timeline_review_digest(reviewed)


@pytest.mark.parametrize("field", ["anchor", "start", "end", "pose", "fps", "timeline_id", "scope_ids"])
def test_changed_content_cannot_inherit_an_old_review(field):
    _, timeline = _draft()
    reviewed = reviewed_timeline(timeline)
    if field == "anchor":
        reviewed["segments"][0]["anchors"]["fixation"] += 1
    elif field in {"start", "end"}:
        reviewed["segments"][0][field + "_frame"] += 1
    elif field == "pose":
        reviewed["source_binding"]["pose_file_sha256"] = "c" * 64
    elif field == "fps":
        reviewed["fps"] = 30
    elif field == "timeline_id":
        reviewed["timeline_id"] += "-copy"
    else:
        reviewed["coverage"]["observed_movement_ids"].reverse()
    assert timeline_review_status(reviewed)["reason"] == "review_timeline_digest_mismatch"


def test_partial_review_does_not_authorize_other_phases_boundaries_or_movements():
    _, timeline = _draft()
    reviewed = reviewed_timeline(timeline)
    reviewed["review"]["scope"] = [{"movement_id": "M01", "boundaries_reviewed": False, "phases": ["fixation"]}]
    assert timeline_review_status(reviewed, "M01", "fixation")["status"] == "verified"
    for movement, phase in [(None, None), ("M01", None), ("M01", "execution"), ("M02", "fixation")]:
        assert timeline_review_status(reviewed, movement, phase)["status"] == "unverified"
    reviewed["review"]["scope"][0]["boundaries_reviewed"] = True
    reviewed["review"]["scope"][0]["phases"] = list(reviewed["segments"][0]["anchors"])
    assert timeline_review_status(reviewed, "M01")["status"] == "verified"
    assert timeline_review_status(reviewed)["status"] == "unverified"


@pytest.mark.parametrize("field,value", [
    ("reviewed_at", "2026-09-21T10:00:00"),
    ("reviewed_at", "2026-02-30T10:00:00Z"),
    ("method", "automatic"),
    ("schema_version", True),
    ("timeline_sha256", "bad"),
    ("scope", []),
    ("reviewer", {"name": "", "role": "test"}),
    ("evidence", {"reference": "", "sha256": "a" * 64}),
])
def test_malformed_review_is_rejected_without_modifying_inputs(field, value):
    spec, timeline = _draft()
    reviewed = reviewed_timeline(timeline)
    reviewed["review"][field] = value
    with pytest.raises(ScoringContractError):
        validate_movement_timeline(reviewed, spec)
    assert timeline_review_status(reviewed)["status"] == "unverified"


def test_complete_legacy_data_reads_but_cannot_enter_accuracy():
    timeline = _complete_timeline()
    timeline.pop("review")
    assert validate_movement_timeline(timeline, _active_spec())["status"] == "complete"
    with pytest.raises(ScoringContractError, match="content-bound video review"):
        validate_movement_timeline(timeline, _active_spec(), require_complete=True)
    assert validate_movement_timeline(reviewed_timeline(timeline), _active_spec(), require_complete=True)


def test_high_confidence_and_large_angle_never_override_missing_phase_review():
    spec, timeline = _draft()
    for segment in timeline["segments"]:
        segment["confidence"] = 1.0
    diagnostics = {
        "status": "wholebody_diagnostics_only",
        "poomsae": {"poomsae_id": spec["poomsae_id"], "version": spec["version"]},
        "movement_timeline_id": timeline["timeline_id"],
        "movements": [{"movement_id": "M01", "metrics": [{
            "metric_id": "back_foot_yaw_to_stance_direction_deg", "value": 175.0, "uncertainty_95": 1.0,
        }]}],
    }
    observations = [{
        "observation_id": "test-wrong-action", "event_kind": "wrong_action",
        "movement_id": "M01", "start_frame": 218, "end_frame": 218,
        "evidence_status": "observed", "confidence": 1.0, "description": "synthetic observation",
        "measurement": None, "confirmation_method": "manual_video_review",
    }]
    profile = load_source_bound_accuracy_profile(SOURCE_BOUND_ACCURACY_PATH)
    result = build_source_bound_accuracy_decisions(diagnostics, spec, timeline, profile, observations)
    assert result["observed_scope_provisional_deduction_total"] is None
    assert result["provisional_deduction_status"] == "not_evaluated_unverified_timeline"
    assert result["applied_observed_scope_deductions"] == []
    assert all(item["decision_status"] == "timeline_unverified" for item in result["numeric_decisions"])
    measured = next(item for item in result["numeric_decisions"] if item["measured_value"] is not None)
    assert measured["measured_value"] == 175.0
    assert measured["conditional_geometry_status"] == "confirmed_source_bound_minor"
    assert result["categorical_decisions"][0]["deduction_points"] is None
    assert result["categorical_decisions"][0]["reason"] == "movement_timeline_review_unverified"
