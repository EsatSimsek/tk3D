from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.aist_calibration import find_aist_camera_setting, load_aist_camera_calibrations
from src.camera_calibration import calibration_report, match_synchronized_detections
from src.data_structures import CameraCalibration

def test_load_aist_camera_calibrations(tmp_path: Path) -> None:
    cameras_dir = tmp_path / "cameras"
    cameras_dir.mkdir()
    (cameras_dir / "mapping.txt").write_text("seq_cAll setting1\n", encoding="utf-8")
    camera_payload = [
        {
            "name": "c01",
            "size": [1920, 1080],
            "matrix": [[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]],
            "distortions": [0.1, 0.0, 0.0, 0.0, 0.0],
            "rotation": [0.0, 0.0, 0.0],
            "translation": [1.0, 2.0, 3.0],
        },
        {
            "name": "c02",
            "size": [1920, 1080],
            "matrix": [[1100.0, 0.0, 960.0], [0.0, 1100.0, 540.0], [0.0, 0.0, 1.0]],
            "distortions": [0.0, 0.0, 0.0, 0.0, 0.0],
            "rotation": [0.0, 0.0, 0.1],
            "translation": [4.0, 5.0, 6.0],
        },
    ]
    (cameras_dir / "setting1.json").write_text(json.dumps(camera_payload), encoding="utf-8")

    assert find_aist_camera_setting("seq_cAll", cameras_dir) == "setting1"
    calibrations = load_aist_camera_calibrations("seq_cAll", cameras_dir, camera_ids=["c02"])

    assert len(calibrations) == 1
    assert calibrations[0].camera_id == "c02"
    assert calibrations[0].image_size == (1920, 1080)
    assert calibrations[0].projection_matrix.shape == (3, 4)
    assert calibrations[0].translation_vector.tolist() == [4.0, 5.0, 6.0]

def test_calibration_report_ignores_missing_reprojection_errors() -> None:
    calibration = CameraCalibration(
        camera_id="c01",
        image_size=(1920, 1080),
        intrinsic_matrix=np.eye(3),
        distortion_coefficients=np.zeros(5),
        rotation_vector=np.zeros(3),
        translation_vector=np.zeros(3),
        projection_matrix=np.zeros((3, 4)),
        reprojection_error_px=None,
    )

    report = calibration_report([calibration])

    assert report["camera_count"] == 1
    assert report["mean_reprojection_error_px"] is None


def test_calibration_detections_match_nearest_timestamps_without_reuse() -> None:
    detections = {
        "c01": {0.000: np.asarray([[1.0]]), 0.100: np.asarray([[2.0]])},
        "c02": {0.004: np.asarray([[3.0]]), 0.096: np.asarray([[4.0]])},
        "c03": {0.002: np.asarray([[5.0]]), 0.103: np.asarray([[6.0]])},
    }

    matches = match_synchronized_detections(detections, "c01", tolerance_sec=0.01)

    assert len(matches) == 2
    assert matches[0]["c02"].item() == 3.0
    assert matches[1]["c03"].item() == 6.0
