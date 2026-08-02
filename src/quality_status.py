from __future__ import annotations

from typing import Any


def external_accuracy_not_evaluated() -> dict[str, Any]:
    """Return a run-local status that cannot inherit metrics from another run."""
    return {
        "schema_version": 1,
        "status": "not_evaluated_for_this_run",
        "evaluated": False,
        "ground_truth_source": None,
        "metrics": None,
        "historical_benchmark_inherited": False,
        "zed_depth_is_external_ground_truth": False,
        "required_for_internal_provisional_scoring": False,
        "interpretation": (
            "ZED RGB, stereo depth, confidence, calibration, timestamps, and IMU are "
            "evidence inside the system under test. They measure internal sensor consistency, "
            "not independent external 3D accuracy. External accuracy is informational and is "
            "not required for internally quality-gated provisional scoring."
        ),
    }


def internal_sensor_consistency_status(
    *,
    internal_geometry_passed: bool,
    zed_depth_fusion_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe run-local checks without presenting them as external accuracy."""
    depth_report = zed_depth_fusion_report or {}
    depth_applied = bool(depth_report.get("applied"))
    depth_gate = depth_report.get("final_acceptance_gate")
    depth_gate_passed = (
        bool(depth_gate.get("passed")) if depth_applied and isinstance(depth_gate, dict) else None
    )
    depth_output_used = bool(depth_report.get("final_output_used")) if depth_applied else False
    passed = bool(
        internal_geometry_passed
        and (not depth_applied or (depth_gate_passed and depth_output_used))
    )
    return {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "internal_geometry_passed": bool(internal_geometry_passed),
        "zed_depth_fusion_applied": depth_applied,
        "zed_depth_vs_rgb_gate_passed": depth_gate_passed,
        "zed_depth_output_used": depth_output_used,
        "independent_external_accuracy_claim": False,
    }
