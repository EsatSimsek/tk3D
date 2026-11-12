from __future__ import annotations

import numpy as np
import pytest

from src.calibration_refinement import (
    calibration_consistency_report,
    refine_relative_rotation,
    rotate_about_centre,
    signed_epipolar_distances,
)
from src.synthetic_data import build_synthetic_calibrations, build_synthetic_world_sequence, project_world_sequence


def _scene(frame_count: int = 40):
    calibrations = build_synthetic_calibrations()
    pair = {key: calibrations[key] for key in ("cam_front", "cam_side")}
    world = build_synthetic_world_sequence(frame_count, valid_joint_count=33)
    poses = project_world_sequence(world, pair)
    return pair, poses


def _correspondences(poses):
    first, second, frames = [], [], []
    for frame, by_camera in poses.items():
        a, b = by_camera["cam_front"], by_camera["cam_side"]
        usable = a.valid_mask & b.valid_mask
        first.append(a.keypoints_xy[usable])
        second.append(b.keypoints_xy[usable])
        frames.append(np.full(int(usable.sum()), frame))
    return np.concatenate(first), np.concatenate(second), np.concatenate(frames)


def test_true_calibration_has_no_epipolar_bias_and_passes_the_run_check() -> None:
    pair, poses = _scene()
    a, b, _ = _correspondences(poses)
    distances = signed_epipolar_distances(pair["cam_front"], pair["cam_side"], a, b)
    assert np.max(np.abs(distances)) < 1e-6
    report = calibration_consistency_report(
        poses, pair, min_keypoint_score=0.8, max_abs_signed_median_px=3.0, min_correspondences=100
    )
    assert report["passed"] is True
    assert report["pairs"][0]["evaluated"] is True


def test_a_moved_camera_creates_a_same_signed_bias_and_fails_the_run_check() -> None:
    pair, poses = _scene()
    stale = dict(pair)
    stale["cam_side"] = rotate_about_centre(pair["cam_side"], np.deg2rad([0.6, 0.0, 0.0]))
    report = calibration_consistency_report(
        poses, stale, min_keypoint_score=0.8, max_abs_signed_median_px=3.0, min_correspondences=100
    )
    pair_report = report["pairs"][0]
    assert report["passed"] is False
    assert abs(pair_report["signed_median_px"]) > 3.0
    assert pair_report["same_sign_fraction"] > 0.95


def test_too_few_correspondences_are_reported_not_failed() -> None:
    pair, poses = _scene(frame_count=2)
    report = calibration_consistency_report(
        poses, pair, min_keypoint_score=0.8, max_abs_signed_median_px=3.0, min_correspondences=10_000
    )
    assert report["passed"] is True
    assert report["pairs"][0]["evaluated"] is False


def test_refinement_recovers_a_known_rotation_on_held_out_frames_and_keeps_centres() -> None:
    pair, poses = _scene()
    stale = rotate_about_centre(pair["cam_side"], np.deg2rad([0.6, -0.3, 0.4]))
    a, b, frames = _correspondences(poses)
    refined, report = refine_relative_rotation(pair["cam_front"], stale, a, b, frames % 2 == 0)
    assert abs(report["held_out_before"]["signed_median_px"]) > 1.0
    assert abs(report["held_out_after"]["signed_median_px"]) < 0.05
    assert report["held_out_after"]["abs_median_px"] < 0.05
    assert report["independent_accuracy_claim"] is False

    def centre(calibration):
        import cv2

        rotation, _ = cv2.Rodrigues(np.asarray(calibration.rotation_vector, float).reshape(3, 1))
        return -rotation.T @ np.asarray(calibration.translation_vector, float)

    assert centre(refined) == pytest.approx(centre(pair["cam_side"]), abs=1e-9)
    assert np.asarray(refined.intrinsic_matrix) == pytest.approx(np.asarray(pair["cam_side"].intrinsic_matrix))


def test_refinement_requires_a_proper_train_split() -> None:
    pair, poses = _scene()
    a, b, _ = _correspondences(poses)
    with pytest.raises(ValueError):
        refine_relative_rotation(pair["cam_front"], pair["cam_side"], a, b, np.ones(len(a), bool))
