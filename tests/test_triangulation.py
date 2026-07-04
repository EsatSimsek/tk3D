from __future__ import annotations

import numpy as np

from src.synthetic_data import build_synthetic_calibrations, build_synthetic_world_sequence, project_world_sequence
from src.triangulation import stack_triangulated, triangulate_frame

def test_triangulate_frame_recovers_synthetic_world_points() -> None:
    calibrations = build_synthetic_calibrations()
    world = build_synthetic_world_sequence(frame_count=1, valid_joint_count=10)
    projected = project_world_sequence(world, calibrations)

    pose = triangulate_frame(
        frame_idx=0,
        poses_by_camera=projected[0],
        calibrations=calibrations,
        min_views=2,
    )

    np.testing.assert_allclose(pose.keypoints_3d_world[:10], world[0, :10], atol=1e-6)
    assert np.all(pose.used_cameras[:10] == 3)
    assert np.nanmax(pose.reprojection_error[:10]) < 1e-6

def test_stack_triangulated_handles_empty_input() -> None:
    arrays = stack_triangulated([])

    assert arrays["keypoints_3d_world"].shape == (0, 133, 3)
    assert arrays["triangulation_score"].shape == (0, 133)
    assert arrays["reprojection_error"].shape == (0, 133)
    assert arrays["used_cameras"].shape == (0, 133)
