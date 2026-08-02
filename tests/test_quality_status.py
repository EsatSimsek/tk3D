from __future__ import annotations

from src.quality_status import external_accuracy_not_evaluated, internal_sensor_consistency_status


def test_unmeasured_external_accuracy_never_inherits_historical_metrics() -> None:
    status = external_accuracy_not_evaluated()
    assert status["status"] == "not_evaluated_for_this_run"
    assert status["evaluated"] is False
    assert status["metrics"] is None
    assert status["historical_benchmark_inherited"] is False
    assert status["required_for_internal_provisional_scoring"] is False


def test_zed_depth_is_classified_as_internal_sensor_evidence() -> None:
    status = external_accuracy_not_evaluated()
    assert status["zed_depth_is_external_ground_truth"] is False


def test_internal_sensor_consistency_keeps_external_claim_false() -> None:
    status = internal_sensor_consistency_status(
        internal_geometry_passed=True,
        zed_depth_fusion_report={
            "applied": True,
            "final_output_used": True,
            "final_acceptance_gate": {"passed": True},
        },
    )

    assert status["status"] == "passed"
    assert status["zed_depth_vs_rgb_gate_passed"] is True
    assert status["independent_external_accuracy_claim"] is False


def test_failed_depth_gate_cannot_pass_internal_sensor_consistency() -> None:
    status = internal_sensor_consistency_status(
        internal_geometry_passed=True,
        zed_depth_fusion_report={
            "applied": True,
            "final_output_used": False,
            "final_acceptance_gate": {"passed": False},
        },
    )

    assert status["status"] == "failed"
