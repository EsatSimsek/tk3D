from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class AISTSMPLMotion:
    poses: np.ndarray
    scaling: np.ndarray
    translation: np.ndarray


def find_smpl_model_file(smpl_dir: str | Path, gender: str = "MALE") -> Path:
    root = Path(smpl_dir)
    gender = gender.upper()
    candidates = [
        root / f"SMPL_{gender}.pkl",
        root / "SMPL" / f"SMPL_{gender}.pkl",
        root / gender.lower() / f"SMPL_{gender}.pkl",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    expected = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"SMPL model file not found. Expected one of: {expected}")


def load_aist_smpl_motion(motion_path: str | Path) -> AISTSMPLMotion:
    path = Path(motion_path)
    if not path.exists():
        raise FileNotFoundError(f"AIST++ SMPL motion file not found: {path}")
    with path.open("rb") as file:
        raw: dict[str, Any] = pickle.load(file)
    return AISTSMPLMotion(
        poses=np.asarray(raw["smpl_poses"], dtype=np.float32),
        scaling=np.asarray(raw["smpl_scaling"], dtype=np.float32).reshape(-1),
        translation=np.asarray(raw["smpl_trans"], dtype=np.float32),
    )


def split_smpl_pose(poses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    poses = np.asarray(poses, dtype=np.float32)
    if poses.ndim == 2 and poses.shape[1] == 72:
        return poses[:, :3], poses[:, 3:]
    if poses.ndim == 3 and poses.shape[1:] == (24, 3):
        return poses[:, 0, :], poses[:, 1:, :].reshape(poses.shape[0], -1)
    raise ValueError(f"Unsupported SMPL pose shape: {poses.shape}")
def selected_frame_indices(total_frames: int, max_frames: int | None, stride: int) -> list[int]:
    stride = max(int(stride), 1)
    indices = list(range(0, total_frames, stride))
    if max_frames is not None:
        indices = indices[: max(int(max_frames), 0)]
    return indices

