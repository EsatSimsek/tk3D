from __future__ import annotations

import numpy as np

from src.data_structures import COCO_BODY_JOINTS
from src.multiview_pose_optimization import (
    PROVENANCE_TEMPORALLY_RECOVERED,
    GlobalPoseOptimizationConfig,
    optimize_body_sequence,
)
from src.synthetic_data import build_synthetic_calibrations, project_world_sequence
from src.triangulation import triangulate_frame


def _rigid_body_sequence(frame_count: int) -> np.ndarray:
    base = np.zeros((17, 3), dtype=float)
    idx = COCO_BODY_JOINTS
    base[idx["nose"]] = [0.0, 1.72, 3.0]
    base[idx["left_eye"]] = [-0.04, 1.76, 3.0]
    base[idx["right_eye"]] = [0.04, 1.76, 3.0]
    base[idx["left_ear"]] = [-0.09, 1.73, 3.0]
    base[idx["right_ear"]] = [0.09, 1.73, 3.0]
    base[idx["left_shoulder"]] = [-0.23, 1.48, 3.0]
    base[idx["right_shoulder"]] = [0.23, 1.48, 3.0]
    base[idx["left_elbow"]] = [-0.45, 1.25, 3.02]
    base[idx["right_elbow"]] = [0.45, 1.25, 3.02]
    base[idx["left_wrist"]] = [-0.62, 1.03, 3.05]
    base[idx["right_wrist"]] = [0.62, 1.03, 3.05]
    base[idx["left_hip"]] = [-0.16, 1.00, 3.0]
    base[idx["right_hip"]] = [0.16, 1.00, 3.0]
    base[idx["left_knee"]] = [-0.16, 0.55, 3.02]
    base[idx["right_knee"]] = [0.16, 0.55, 3.02]
    base[idx["left_ankle"]] = [-0.16, 0.10, 3.05]
    base[idx["right_ankle"]] = [0.16, 0.10, 3.05]

    output = np.full((frame_count, 133, 3), np.nan, dtype=float)
    for frame in range(frame_count):
        phase = 2.0 * np.pi * frame / max(frame_count - 1, 1)
        translation = np.asarray(
            [0.12 * np.sin(phase), 0.025 * np.cos(phase), 0.04 * np.sin(phase * 0.5)]
        )
        output[frame, :17] = base + translation
    return output


def _noisy_problem(frame_count: int = 24):
    rng = np.random.default_rng(27)
    calibrations = build_synthetic_calibrations()
    ground_truth = _rigid_body_sequence(frame_count)
    poses = project_world_sequence(ground_truth, calibrations)
    left_wrist = COCO_BODY_JOINTS["left_wrist"]
    for frame_poses in poses.values():
        for camera_id, pose in frame_poses.items():
            pose.keypoints_xy[:17] += rng.normal(0.0, 2.25, size=(17, 2))
            if camera_id == "cam_side":
                pose.keypoints_xy[:17] += np.asarray([8.0, -5.0])
    for frame in range(9, 12):
        for camera_id in ("cam_front", "cam_back"):
            poses[frame][camera_id].valid_mask[left_wrist] = False
            poses[frame][camera_id].scores[left_wrist] = 0.0
            poses[frame][camera_id].keypoints_xy[left_wrist] = np.nan

    triangulated = [
        triangulate_frame(
            frame,
            poses[frame],
            calibrations,
            min_views=2,
            min_keypoint_score=0.30,
            max_reprojection_error_px=25.0,
        )
        for frame in range(frame_count)
    ]
    initial = np.stack([pose.keypoints_3d_world for pose in triangulated])
    confidence = np.stack([pose.triangulation_score for pose in triangulated])
    reprojection_error = np.stack([pose.reprojection_error for pose in triangulated])
    used_cameras = np.stack([pose.used_cameras for pose in triangulated])
    valid = (
        np.all(np.isfinite(initial), axis=-1)
        & np.isfinite(reprojection_error)
        & (reprojection_error <= 25.0)
        & (confidence >= 0.05)
        & (used_cameras >= 2)
    )
    return ground_truth, poses, calibrations, initial, confidence, valid


def test_global_optimizer_reduces_3d_error_recovers_short_gap_and_downweights_bad_camera() -> None:
    ground_truth, poses, calibrations, initial, confidence, valid = _noisy_problem()
    config = GlobalPoseOptimizationConfig(
        minimum_bone_samples=5,
        outer_iterations=2,
        max_solver_evaluations=25,
        max_p95_correction_m=0.50,
    )

    result = optimize_body_sequence(
        initial,
        valid,
        confidence,
        np.arange(initial.shape[0]),
        np.arange(initial.shape[0], dtype=float) / 30.0,
        poses,
        calibrations,
        np.eye(4),
        config,
    )

    left_wrist = COCO_BODY_JOINTS["left_wrist"]
    common = valid[:, :17] & np.all(np.isfinite(result.keypoints_3d[:, :17]), axis=-1)
    before_error = np.mean(np.linalg.norm(initial[:, :17][common] - ground_truth[:, :17][common], axis=1))
    after_error = np.mean(
        np.linalg.norm(result.keypoints_3d[:, :17][common] - ground_truth[:, :17][common], axis=1)
    )

    assert result.applied
    assert after_error < before_error
    assert np.all(
        result.provenance[9:12, left_wrist] == PROVENANCE_TEMPORALLY_RECOVERED
    )
    assert np.all(np.isfinite(result.keypoints_3d[9:12, left_wrist]))
    assert result.camera_weights["cam_side"] < result.camera_weights["cam_front"]
    assert result.camera_weights["cam_side"] < result.camera_weights["cam_back"]
    assert (
        result.report["after"]["mean_bone_length_cv_percent"]
        < result.report["before"]["mean_bone_length_cv_percent"]
    )


def test_global_optimizer_falls_back_without_overwriting_initial_data_when_gate_fails() -> None:
    _, poses, calibrations, initial, confidence, valid = _noisy_problem(frame_count=16)
    config = GlobalPoseOptimizationConfig(
        minimum_bone_samples=4,
        outer_iterations=1,
        max_solver_evaluations=10,
        max_p95_correction_m=1e-6,
    )

    result = optimize_body_sequence(
        initial,
        valid,
        confidence,
        np.arange(initial.shape[0]),
        np.arange(initial.shape[0], dtype=float) / 30.0,
        poses,
        calibrations,
        np.eye(4),
        config,
    )

    assert not result.applied
    assert result.report["fallback_used"]
    assert "body_correction_too_large" in result.report["fallback_reason"]
    np.testing.assert_allclose(result.keypoints_3d, initial, equal_nan=True)
