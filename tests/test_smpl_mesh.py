from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from src.smpl_mesh import find_smpl_model_file, load_aist_smpl_motion, selected_frame_indices, split_smpl_pose


def test_find_smpl_model_file(tmp_path: Path) -> None:
    smpl_dir = tmp_path / "smpl"
    smpl_dir.mkdir()
    model = smpl_dir / "SMPL_MALE.pkl"
    model.write_bytes(b"placeholder")

    assert find_smpl_model_file(smpl_dir, "MALE") == model


def test_find_smpl_model_file_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        find_smpl_model_file(tmp_path / "smpl", "MALE")


def test_load_aist_smpl_motion(tmp_path: Path) -> None:
    motion_path = tmp_path / "motion.pkl"
    payload = {
        "smpl_poses": np.zeros((2, 24, 3), dtype=np.float32),
        "smpl_scaling": np.array([1.2], dtype=np.float32),
        "smpl_trans": np.ones((2, 3), dtype=np.float32),
    }
    with motion_path.open("wb") as file:
        pickle.dump(payload, file)

    motion = load_aist_smpl_motion(motion_path)

    assert motion.poses.shape == (2, 24, 3)
    assert motion.scaling.tolist() == [pytest.approx(1.2)]
    assert motion.translation.shape == (2, 3)


def test_selected_frame_indices() -> None:
    assert selected_frame_indices(total_frames=10, max_frames=3, stride=2) == [0, 2, 4]

def test_split_smpl_pose_flat_72() -> None:
    poses = np.arange(144, dtype=np.float32).reshape(2, 72)

    global_orient, body_pose = split_smpl_pose(poses)

    assert global_orient.shape == (2, 3)
    assert body_pose.shape == (2, 69)
    np.testing.assert_array_equal(global_orient[0], poses[0, :3])
    np.testing.assert_array_equal(body_pose[0], poses[0, 3:])


def test_split_smpl_pose_joint_axis_angle() -> None:
    poses = np.arange(2 * 24 * 3, dtype=np.float32).reshape(2, 24, 3)

    global_orient, body_pose = split_smpl_pose(poses)

    assert global_orient.shape == (2, 3)
    assert body_pose.shape == (2, 69)
    np.testing.assert_array_equal(global_orient[0], poses[0, 0])
    np.testing.assert_array_equal(body_pose[0], poses[0, 1:].reshape(-1))

