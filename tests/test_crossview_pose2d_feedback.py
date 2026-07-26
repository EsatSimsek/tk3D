from __future__ import annotations

import cv2
import numpy as np

from scripts.diagnose_camera_consistency import _assess_temporal_shift
from src.crossview_pose2d_feedback import (
    CrossView2DFeedbackConfig,
    build_feedback_plan,
    decide_guided_candidate,
    project_source_point_distorted,
)
from src.data_structures import (
    COCO_BODY_JOINTS,
    COCO_WHOLEBODY_KEYPOINTS,
    CameraCalibration,
    PersonPose2D,
)
from src.vitpose_plus_runtime import ViTPosePlusWholeBodyInferencer


def test_guided_heatmap_decode_selects_image_peak_near_prior() -> None:
    runtime = ViTPosePlusWholeBodyInferencer.__new__(ViTPosePlusWholeBodyInferencer)
    runtime.heatmap_offsets_xy = np.zeros((COCO_WHOLEBODY_KEYPOINTS, 2), dtype=float)
    heatmaps = np.zeros((COCO_WHOLEBODY_KEYPOINTS, 64, 48), dtype=float)
    joint_idx = COCO_BODY_JOINTS["left_wrist"]
    yy, xx = np.mgrid[0:64, 0:48]
    heatmaps[joint_idx] = (
        0.95 * np.exp(-((xx - 10.0) ** 2 + (yy - 20.0) ** 2) / 3.0)
        + 0.55 * np.exp(-((xx - 30.0) ** 2 + (yy - 30.0) ** 2) / 3.0)
    )
    crop = (0.0, 0.0, 470.0, 630.0)
    priors = np.full((COCO_WHOLEBODY_KEYPOINTS, 2), np.nan, dtype=float)
    prior_valid = np.zeros(COCO_WHOLEBODY_KEYPOINTS, dtype=bool)
    priors[joint_idx] = [300.0, 300.0]
    prior_valid[joint_idx] = True

    global_xy, _ = runtime._decode_heatmaps(heatmaps, crop)
    guided_xy, guided_scores = runtime._decode_heatmaps_near_priors(
        heatmaps,
        crop,
        priors,
        prior_valid,
        search_radius_px=35.0,
    )

    assert np.linalg.norm(guided_xy[joint_idx] - priors[joint_idx]) < 15.0
    assert np.linalg.norm(global_xy[joint_idx] - priors[joint_idx]) > 150.0
    assert guided_scores[joint_idx] > 0.45


def test_guided_candidate_requires_image_and_geometry_improvement() -> None:
    config = CrossView2DFeedbackConfig()
    accepted = decide_guided_candidate(
        observed_xy=np.array([200.0, 100.0]),
        prior_xy=np.array([100.0, 100.0]),
        guided_xy=np.array([105.0, 102.0]),
        guided_score=0.55,
        unconstrained_score=0.90,
        config=config,
    )
    weak_image = decide_guided_candidate(
        observed_xy=np.array([200.0, 100.0]),
        prior_xy=np.array([100.0, 100.0]),
        guided_xy=np.array([105.0, 102.0]),
        guided_score=0.05,
        unconstrained_score=0.90,
        config=config,
    )
    wrong_place = decide_guided_candidate(
        observed_xy=np.array([200.0, 100.0]),
        prior_xy=np.array([100.0, 100.0]),
        guided_xy=np.array([170.0, 100.0]),
        guided_score=0.80,
        unconstrained_score=0.90,
        config=config,
    )

    assert accepted.accepted
    assert not weak_image.accepted
    assert weak_image.reason == "insufficient_image_score"
    assert not wrong_place.accepted
    assert wrong_place.reason == "candidate_too_far_from_crossview_prior"


def test_periodic_motion_does_not_create_an_automatic_camera_offset() -> None:
    assessment = _assess_temporal_shift(
        geometric_best={"shift_frames": 89, "median_error_px": 15.0},
        geometric_zero={"shift_frames": 0, "median_error_px": 51.0},
        motion_sync={
            "best_shift": {"shift_frames": 179, "correlation": 0.80},
            "zero_shift_correlation": 0.44,
        },
    )

    assert assessment["status"] == "ambiguous_periodic_motion"
    assert assessment["recommended_frame_offset"] is None


def test_feedback_plan_excludes_target_camera_and_localizes_arm_failure() -> None:
    calibrations = {
        f"c{index:02d}": _calibration(f"c{index:02d}", tx=0.35 * (index - 3))
        for index in range(1, 6)
    }
    points_3d = np.full((1, COCO_WHOLEBODY_KEYPOINTS, 3), np.nan, dtype=float)
    for joint_idx in range(17):
        points_3d[0, joint_idx] = [
            -0.35 + 0.04 * joint_idx,
            -0.20 + 0.025 * (joint_idx % 5),
            5.0 + 0.02 * (joint_idx % 3),
        ]
    poses: dict[str, PersonPose2D] = {}
    corrupted = {
        COCO_BODY_JOINTS["left_shoulder"],
        COCO_BODY_JOINTS["right_shoulder"],
        COCO_BODY_JOINTS["left_elbow"],
        COCO_BODY_JOINTS["right_elbow"],
        COCO_BODY_JOINTS["left_wrist"],
        COCO_BODY_JOINTS["right_wrist"],
    }
    for camera_id, calibration in calibrations.items():
        xy = np.full((COCO_WHOLEBODY_KEYPOINTS, 2), np.nan, dtype=float)
        scores = np.zeros(COCO_WHOLEBODY_KEYPOINTS, dtype=float)
        valid = np.zeros(COCO_WHOLEBODY_KEYPOINTS, dtype=bool)
        for joint_idx in range(17):
            xy[joint_idx] = project_source_point_distorted(
                points_3d[0, joint_idx],
                calibration,
            )
            scores[joint_idx] = 0.9
            valid[joint_idx] = True
        if camera_id == "c05":
            for joint_idx in corrupted:
                xy[joint_idx] += np.array([120.0, -35.0])
        poses[camera_id] = PersonPose2D(camera_id, 0, xy, scores, valid)

    plan = build_feedback_plan(
        np.array([0]),
        {0: poses},
        points_3d,
        calibrations,
        CrossView2DFeedbackConfig(min_supporting_views=4),
    )

    for joint_idx in corrupted:
        assert plan.target_mask["c05"][0, joint_idx]
        expected = project_source_point_distorted(points_3d[0, joint_idx], calibrations["c05"])
        assert np.linalg.norm(plan.priors_xy["c05"][0, joint_idx] - expected) < 1e-4
        assert plan.supporting_views["c05"][0, joint_idx] == 4
    assert not np.any(plan.target_mask["c01"])
    assert (
        plan.report["cameras"]["c05"]["classification"]
        == "self_occlusion_or_2d_joint_detection"
    )


def _calibration(camera_id: str, tx: float) -> CameraCalibration:
    intrinsic = np.array(
        [[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    rotation = np.eye(3, dtype=float)
    translation = np.array([tx, 0.0, 0.0], dtype=float)
    rotation_vector, _ = cv2.Rodrigues(rotation)
    return CameraCalibration(
        camera_id=camera_id,
        image_size=(1920, 1080),
        intrinsic_matrix=intrinsic,
        distortion_coefficients=np.zeros(5, dtype=float),
        rotation_vector=rotation_vector.reshape(3),
        translation_vector=translation,
        projection_matrix=intrinsic @ np.hstack([rotation, translation[:, None]]),
    )
