from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.data_structures import PersonPose2D
from src.exporter import export_keypoints2d_csv, export_session_json
from src.biomechanics_3d import angle_deg
from src.pose3d_stability import pose3d_stability_metrics
from src.smoothing_3d import (
    moving_average_nan,
    moving_average_pose,
    robust_savgol_pose,
    smooth_pose_sequence,
)
from src.validation_3d import validate_triangulation


def test_moving_average_keeps_all_nan_joints_without_warning() -> None:
    keypoints = np.full((3, 133, 3), np.nan, dtype=float)
    keypoints[:, 0, 0] = [0.0, 2.0, 4.0]

    smoothed = moving_average_nan(keypoints, window_size=3)

    assert np.isfinite(smoothed[:, 0, 0]).all()
    assert np.isnan(smoothed[:, 1]).all()
    np.testing.assert_allclose(smoothed[:, 0, 0], [1.0, 2.0, 3.0])


def test_validation_flags_low_validity() -> None:
    keypoints = np.full((2, 133, 3), np.nan, dtype=float)
    errors = np.full((2, 133), np.nan, dtype=float)
    keypoints[:, 0] = [0.0, 0.0, 1.0]
    errors[:, 0] = 2.0

    validation = validate_triangulation(keypoints, errors)

    assert validation.frame_valid_ratio.shape == (2,)
    assert "mean_frame_valid_ratio_below_0_50" in validation.warnings


def test_session_json_replaces_nan_with_null(tmp_path) -> None:
    output_path = tmp_path / "session_3d.json"
    export_session_json({"value": np.array([1.0, np.nan])}, output_path)

    raw = output_path.read_text(encoding="utf-8")
    assert "NaN" not in raw
    assert json.loads(raw) == {"value": [1.0, None]}


def test_keypoints2d_export_preserves_tracked_person_identity(tmp_path) -> None:
    pose = PersonPose2D(
        camera_id="cam",
        frame_idx=3,
        keypoints_xy=np.zeros((133, 2), dtype=float),
        scores=np.ones(133, dtype=float),
        valid_mask=np.ones(133, dtype=bool),
        person_id=17,
    )
    output_path = tmp_path / "keypoints.csv"

    export_keypoints2d_csv({3: {"cam": pose}}, output_path)
    exported = pd.read_csv(output_path)

    assert set(exported["person_id"]) == {17}


def test_validation_all_nan_errors_without_runtime_warning() -> None:
    keypoints = np.full((2, 133, 3), np.nan, dtype=float)
    errors = np.full((2, 133), np.nan, dtype=float)

    validation = validate_triangulation(keypoints, errors)

    assert validation.mean_reprojection_error_px.shape == (2,)
    assert np.isnan(validation.mean_reprojection_error_px).all()


def test_pose_smoothing_does_not_flatten_sequences_shorter_than_window() -> None:
    keypoints = np.zeros((2, 133, 3), dtype=float)
    keypoints[1, :, 0] = 1.0

    smoothed = moving_average_pose(keypoints, window_size=5)

    np.testing.assert_array_equal(smoothed, keypoints)


def test_window_one_still_applies_validity_mask() -> None:
    keypoints = np.ones((2, 3, 3), dtype=float)
    valid = np.ones((2, 3), dtype=bool)
    valid[0, 1] = False

    filtered = moving_average_pose(keypoints, window_size=1, valid_mask=valid)

    assert np.isnan(filtered[0, 1]).all()
    np.testing.assert_array_equal(filtered[1], keypoints[1])


def test_robust_savgol_reduces_joint_angle_jitter_without_phase_lag() -> None:
    frame_count = 41
    keypoints = np.full((frame_count, 17, 3), np.nan, dtype=float)
    truth = np.empty(frame_count, dtype=float)
    raw_angles = np.empty(frame_count, dtype=float)
    for frame_idx in range(frame_count):
        target_angle = 140.0 + 8.0 * np.sin(frame_idx / 12.0)
        truth[frame_idx] = target_angle
        radians = np.radians(180.0 - target_angle)
        hip = np.asarray([0.0, 0.0, 0.0])
        knee = np.asarray([0.0, -1.0, 0.0])
        ankle = knee + np.asarray([np.sin(radians), -np.cos(radians), 0.0])
        jitter = 0.025 if frame_idx % 2 else -0.025
        knee = knee + np.asarray([jitter, 0.0, 0.0])
        ankle = ankle + np.asarray([-jitter, jitter, 0.0])
        keypoints[frame_idx, 11] = hip
        keypoints[frame_idx, 13] = knee
        keypoints[frame_idx, 15] = ankle
        raw_angles[frame_idx] = angle_deg(hip, knee, ankle)

    valid = np.all(np.isfinite(keypoints), axis=-1)
    smoothed = robust_savgol_pose(
        keypoints,
        window_size=7,
        polynomial_order=2,
        valid_mask=valid,
        min_outlier_distance_m=0.04,
    )
    stable_angles = np.asarray([angle_deg(frame[11], frame[13], frame[15]) for frame in smoothed])

    raw_mae = float(np.mean(np.abs(raw_angles[3:-3] - truth[3:-3])))
    stable_mae = float(np.mean(np.abs(stable_angles[3:-3] - truth[3:-3])))
    assert stable_mae < 0.35 * raw_mae


def test_robust_savgol_preserves_invalid_gap() -> None:
    keypoints = np.zeros((11, 2, 3), dtype=float)
    keypoints[:, 0, 0] = np.arange(11, dtype=float)
    valid = np.ones((11, 2), dtype=bool)
    valid[5, 0] = False

    smoothed = smooth_pose_sequence(
        keypoints,
        method="robust_savgol",
        window_size=7,
        polynomial_order=2,
        valid_mask=valid,
    )

    assert np.isnan(smoothed[5, 0]).all()
    np.testing.assert_allclose(
        smoothed[[0, 4, 6, 10], 0, 0],
        [0.0, 4.0, 6.0, 10.0],
        atol=1e-12,
    )


def test_pose3d_stability_report_quantifies_angle_jitter_reduction() -> None:
    frame_count = 41
    raw = np.full((frame_count, 17, 3), np.nan, dtype=float)
    for frame_idx in range(frame_count):
        target_angle = 140.0 + 8.0 * np.sin(frame_idx / 12.0)
        radians = np.radians(180.0 - target_angle)
        hip = np.asarray([0.0, 0.0, 0.0])
        knee = np.asarray([0.0, -1.0, 0.0])
        ankle = knee + np.asarray([np.sin(radians), -np.cos(radians), 0.0])
        jitter = 0.025 if frame_idx % 2 else -0.025
        raw[frame_idx, 11] = hip
        raw[frame_idx, 13] = knee + np.asarray([jitter, 0.0, 0.0])
        raw[frame_idx, 15] = ankle + np.asarray([-jitter, jitter, 0.0])

    valid = np.all(np.isfinite(raw), axis=-1)
    stable = robust_savgol_pose(
        raw,
        window_size=7,
        polynomial_order=2,
        valid_mask=valid,
        min_outlier_distance_m=0.04,
    )
    report = pose3d_stability_metrics(raw, stable)

    assert report["comparable_body_samples"] == frame_count * 3
    assert report["joint_high_frequency_reduction_percent"] > 50.0
    assert report["angles"]["left_knee"]["reduction_percent"] > 50.0
