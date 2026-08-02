from __future__ import annotations

import numpy as np

from src.zed_svo import (
    absolute_camera_pose,
    calibration_from_camera_pose,
    common_timestamp_timeline,
    nearest_timestamp_indices,
    timestamp_mapping_report,
)


def test_timestamp_resampling_preserves_common_duration_and_reports_gap_reuse() -> None:
    period = 10_000_000
    camera_a = np.asarray([0, period, 2 * period, 4 * period, 5 * period], dtype=np.int64)
    camera_b = np.asarray([1_000_000, 11_000_000, 31_000_000, 41_000_000, 51_000_000], dtype=np.int64)
    timeline = common_timestamp_timeline([camera_a, camera_b], fps=100.0)
    assert timeline.tolist() == [1_000_000, 11_000_000, 21_000_000, 31_000_000, 41_000_000]
    mapping = nearest_timestamp_indices(camera_a, timeline)
    assert mapping.tolist() == [0, 1, 2, 3, 3]
    report = timestamp_mapping_report(camera_a, timeline, mapping, fps=100.0)
    assert report["output_frame_count"] == 5
    assert report["source_frames_reused"] == 1
    assert report["output_frames_from_reused_source"] == 1


def test_absolute_camera_pose_applies_documented_imu_rotation_order() -> None:
    pose = np.eye(4)
    pose[:3, 3] = [1.0, 2.0, 3.0]
    imu = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    absolute = absolute_camera_pose(pose, imu, override_gravity=False)
    assert np.allclose(absolute[:3, :3], imu)
    assert np.allclose(absolute[:3, 3], [1.0, 2.0, 3.0])
    assert np.allclose(absolute_camera_pose(pose, None, override_gravity=True), pose)


def test_camera_pose_is_inverted_for_world_projection() -> None:
    camera_to_world = np.eye(4)
    camera_to_world[:3, 3] = [1.0, 0.0, 0.0]
    intrinsic = np.asarray([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]])
    calibration = calibration_from_camera_pose(
        "zed_1",
        (100, 80),
        intrinsic,
        np.zeros(5),
        camera_to_world,
    )
    assert np.allclose(calibration.translation_vector, [-1.0, 0.0, 0.0])
    world_point = np.asarray([1.0, 2.0, 0.0, 1.0])
    projected = calibration.projection_matrix @ world_point
    assert np.allclose(projected[:2] / projected[2], [50.0, 40.0])
