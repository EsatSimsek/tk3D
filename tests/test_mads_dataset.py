from __future__ import annotations

import cv2
import numpy as np
from scipy.io import savemat

from src.coordinate_system import mads_world_to_analysis, transform_points
from src.mads_dataset import (
    MADS_DEPTH_JOINT_NAMES,
    MADS_MULTIVIEW_JOINT_NAMES,
    discover_mads_sequences,
    load_mads_camera_calibration,
    load_mads_ground_truth,
    resolve_mads_roots,
)


def test_mads_coordinate_transform_converts_mm_y_up_to_meter_z_up() -> None:
    source = np.asarray([[[1000.0, 2000.0, 3000.0]]])

    converted = transform_points(source, mads_world_to_analysis())

    np.testing.assert_allclose(converted, [[[1.0, -3.0, 2.0]]])
    assert np.linalg.det(mads_world_to_analysis()[:3, :3]) > 0.0


def test_load_mads_ground_truth_reads_matlab_cell_pose_sequence(tmp_path) -> None:
    cells = np.empty((1, 2), dtype=object)
    cells[0, 0] = np.arange(45, dtype=float).reshape(15, 3)
    cells[0, 1] = np.arange(45, 90, dtype=float).reshape(15, 3)
    source = tmp_path / "Kata_F2_GT.mat"
    savemat(source, {"GTpose2": cells})

    points = load_mads_ground_truth(source, expected_joint_count=len(MADS_MULTIVIEW_JOINT_NAMES))

    assert points.shape == (2, 15, 3)
    assert points[1, -1, -1] == 89.0


def test_load_mads_camera_calibration_builds_opencv_projection(tmp_path) -> None:
    intrinsic = np.asarray([[400.0, 0.0, 256.0], [0.0, 410.0, 192.0], [0.0, 0.0, 1.0]])
    rotation_vector = np.asarray([0.1, -0.2, 0.05])
    translation = np.asarray([10.0, 20.0, 3000.0])
    source = tmp_path / "Calib_Cam0.mat"
    savemat(
        source,
        {
            "KK": intrinsic,
            "kc": np.zeros((5, 1)),
            "om_ext": rotation_vector.reshape(1, 3),
            "T_ext": translation.reshape(1, 3),
        },
    )

    calibration = load_mads_camera_calibration(source, "C0", image_size=(512, 384))

    rotation, _ = cv2.Rodrigues(rotation_vector)
    np.testing.assert_allclose(calibration.projection_matrix, intrinsic @ np.hstack([rotation, translation[:, None]]))
    assert calibration.camera_id == "C0"
    assert calibration.image_size == (512, 384)


def test_discover_mads_multiview_and_depth_sequences(tmp_path) -> None:
    multiview = tmp_path / "MADS_multiview" / "MADS" / "multi_view_data" / "Kata"
    depth = tmp_path / "MADS_depth" / "MADS" / "depth_data" / "Kata"
    multiview.mkdir(parents=True)
    depth.mkdir(parents=True)
    for camera in range(3):
        (multiview / f"Kata_F2_C{camera}.avi").touch()
    (multiview / "Kata_F2_GT.mat").touch()
    for camera in ("Left", "Right"):
        (depth / f"Kata_F2_{camera}.avi").touch()
    (depth / "Kata_F2_GT.mat").touch()
    (depth / "Kata_F2_depthMaps.mat").touch()

    sequences = discover_mads_sequences(resolve_mads_roots(tmp_path))

    assert [(item.modality, item.action, item.sequence) for item in sequences] == [
        ("depth", "Kata", "F2"),
        ("multiview", "Kata", "F2"),
    ]
    assert len(sequences[0].auxiliary_paths) == 1
    assert len(MADS_DEPTH_JOINT_NAMES) == 19
