from __future__ import annotations

from src.poomsae_scoring import build_run_history, build_run_history_html


def _summary(
    run_id: str,
    *,
    pose_sha: str,
    wholebody_measurable: int,
    technical_measurable: int,
    not_measurable: int,
    rule_ready: bool,
    profile_id: str = "profile",
) -> dict:
    return {
        "schema_version": 1,
        "workflow": "tk3d_source_bound_poomsae_scoring_v1",
        "status": "partial_sequence_decisions_generated",
        "mode": "score_verified_pose",
        "profile_id": profile_id,
        "run": {"run_id": run_id, "root": f"C:/outputs/runs/{run_id}"},
        "coverage": {"selected_scope_id": "current_recording_m01_m06"},
        "results": {
            "wholebody_measurable_metric_count": wholebody_measurable,
            "wholebody_thresholded_metric_count": 100,
            "technical_conformance_measurable_criterion_count": technical_measurable,
            "technical_conformance_expected_criterion_count": 100,
            "not_measurable_count": not_measurable,
            "boundary_uncertain_count": 1,
            "diagnostic_review_candidate_count": 4,
            "categorical_mismatch_candidate_count": 1,
            "technical_conformance_review_required_count": 3,
            "observed_scope_provisional_deduction_total": 0.4,
            "rule_scoring_ready": rule_ready,
        },
        "bindings": {"pose": {"sha256": pose_sha}},
    }


def test_run_history_flags_only_same_pose_measurement_regressions() -> None:
    baseline = _summary(
        "baseline",
        pose_sha="a" * 64,
        wholebody_measurable=80,
        technical_measurable=75,
        not_measurable=2,
        rule_ready=True,
    )
    current = _summary(
        "current",
        pose_sha="a" * 64,
        wholebody_measurable=70,
        technical_measurable=65,
        not_measurable=3,
        rule_ready=False,
    )

    report = build_run_history(
        current,
        [
            {
                "summary": baseline,
                "summary_path": "C:/baseline.json",
                "modified_at_utc": "2026-08-22T00:00:00+00:00",
            }
        ],
    )

    assert report["comparison"]["comparison_kind"] == "same_pose_regression"
    assert {item["code"] for item in report["comparison"]["alerts"]} == {
        "wholebody_measurement_coverage_decreased",
        "technical_criterion_coverage_decreased",
        "not_measurable_count_increased",
        "rule_scoring_readiness_regressed",
    }
    rendered = build_run_history_html(report)
    assert "Koşu geçmişi ve karşılaştırma" in rendered
    assert "WholeBody ölçüm kapsamı" in rendered
    assert "baseline" in rendered and "current" in rendered


def test_run_history_different_pose_is_context_only_and_incompatible_profile_is_skipped() -> None:
    current = _summary(
        "current",
        pose_sha="a" * 64,
        wholebody_measurable=70,
        technical_measurable=65,
        not_measurable=3,
        rule_ready=False,
    )
    other_pose = _summary(
        "other-pose",
        pose_sha="b" * 64,
        wholebody_measurable=90,
        technical_measurable=90,
        not_measurable=1,
        rule_ready=True,
    )
    incompatible = _summary(
        "other-profile",
        pose_sha="a" * 64,
        wholebody_measurable=90,
        technical_measurable=90,
        not_measurable=1,
        rule_ready=True,
        profile_id="other-profile",
    )

    report = build_run_history(
        current,
        [
            {"summary": incompatible, "modified_at_utc": "2026-08-20T00:00:00+00:00"},
            {"summary": other_pose, "modified_at_utc": "2026-08-21T00:00:00+00:00"},
        ],
    )

    assert report["summary"]["compatible_prior_count"] == 1
    assert report["comparison"]["comparison_kind"] == "different_pose_context_only"
    assert report["comparison"]["alerts"] == []
