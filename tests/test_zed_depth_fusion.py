from __future__ import annotations

import numpy as np

from src.data_structures import CameraCalibration
from src.zed_depth_fusion import (
    ZedDepthFusionConfig,
    final_depth_fusion_acceptance_gate,
    fuse_depth_constraints,
    robust_depth_patch_sample,
)


def _camera() -> CameraCalibration:
    intrinsic = np.asarray([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]])
    return CameraCalibration(
        camera_id="zed_test",
        image_size=(100, 80),
        intrinsic_matrix=intrinsic,
        distortion_coefficients=np.zeros(5),
        rotation_vector=np.zeros(3),
        translation_vector=np.zeros(3),
        projection_matrix=intrinsic @ np.hstack([np.eye(3), np.zeros((3, 1))]),
    )


def test_depth_patch_uses_confident_cluster_near_expected_joint_depth() -> None:
    depth = np.full((15, 15), 6.0, dtype=float)
    confidence = np.zeros_like(depth)
    depth[5:10, 5:10] = 3.02
    depth[7, 7] = 8.0
    confidence[7, 7] = 90.0
    config = ZedDepthFusionConfig(patch_radius_px=3, min_patch_samples=4)
    sample = robust_depth_patch_sample(depth, confidence, np.asarray([7.0, 7.0]), 3.0, config)
    assert sample is not None
    assert abs(sample.depth_m - 3.02) < 1e-9
    assert sample.confidence == 0.0
    assert sample.pixel_count >= 20


def test_depth_patch_rejects_background_outside_expected_surface_gate() -> None:
    depth = np.full((9, 9), 6.0, dtype=float)
    confidence = np.zeros_like(depth)
    config = ZedDepthFusionConfig(patch_radius_px=2, min_patch_samples=4)
    assert robust_depth_patch_sample(depth, confidence, np.asarray([4.0, 4.0]), 3.0, config) is None


def test_depth_constraint_is_auxiliary_and_correction_is_bounded_by_baseline_prior() -> None:
    config = ZedDepthFusionConfig(baseline_weight=1.0, depth_weight=0.35)
    baseline = np.asarray([0.2, -0.1, 3.0])
    fused = fuse_depth_constraints(baseline, [(_camera(), 3.2, 1.0)], config)
    assert np.allclose(fused[:2], baseline[:2])
    assert baseline[2] < fused[2] < 3.2


def test_final_gate_accepts_small_tradeoff_with_better_acceleration() -> None:
    rgb = _optimization_report(reprojection_median=5.0, reprojection_p95=9.0, acceleration_p95=12.0, bone_cv=1.7)
    depth = _optimization_report(
        reprojection_median=5.01,
        reprojection_p95=9.05,
        acceleration_p95=11.5,
        bone_cv=1.72,
    )
    gate = final_depth_fusion_acceptance_gate(rgb, depth, ZedDepthFusionConfig())
    assert gate["passed"] is True
    assert gate["fallback_to_rgb_reference"] is False


def test_final_gate_falls_back_when_depth_candidate_degrades_acceleration() -> None:
    rgb = _optimization_report(reprojection_median=5.0, reprojection_p95=9.0, acceleration_p95=12.0, bone_cv=1.7)
    depth = _optimization_report(
        reprojection_median=5.0,
        reprojection_p95=9.0,
        acceleration_p95=20.0,
        bone_cv=1.7,
    )
    gate = final_depth_fusion_acceptance_gate(rgb, depth, ZedDepthFusionConfig())
    assert gate["passed"] is False
    assert "p95_acceleration_degraded" in gate["failures"]


def _optimization_report(
    reprojection_median: float,
    reprojection_p95: float,
    acceleration_p95: float,
    bone_cv: float,
) -> dict:
    return {
        "fallback_used": False,
        "after": {
            "reprojection_error_px": {"median": reprojection_median, "p95": reprojection_p95},
            "acceleration_mps2": {"p95": acceleration_p95},
            "mean_bone_length_cv_percent": bone_cv,
        },
    }
