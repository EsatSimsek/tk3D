from __future__ import annotations

import json

import numpy as np

from src.exporter import export_session_json
from src.smoothing_3d import moving_average_nan, moving_average_pose
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
