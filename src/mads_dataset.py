from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.io import loadmat

from .data_structures import CameraCalibration


MADS_MULTIVIEW_JOINT_NAMES: tuple[str, ...] = (
    "neck",
    "pelvis",
    "left_hip",
    "left_knee",
    "left_ankle",
    "right_hip",
    "right_knee",
    "right_ankle",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "head",
)

MADS_DEPTH_JOINT_NAMES: tuple[str, ...] = (
    "neck",
    "pelvis",
    "left_hip",
    "left_knee",
    "left_ankle",
    "left_foot",
    "right_hip",
    "right_knee",
    "right_ankle",
    "right_foot",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "left_hand",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "right_hand",
    "head",
)

_MULTIVIEW_VIDEO = re.compile(r"^(?P<action>[^_]+)_(?P<sequence>.+)_C(?P<camera>[0-2])\.avi$")
_DEPTH_VIDEO = re.compile(r"^(?P<action>[^_]+)_(?P<sequence>.+)_(?P<camera>Left|Right)\.avi$")


@dataclass(slots=True)
class MadsRoots:
    dataset_root: Path
    multiview_data: Path
    depth_data: Path | None


@dataclass(slots=True)
class MadsSequence:
    modality: str
    action: str
    sequence: str
    videos: dict[str, Path]
    ground_truth_path: Path
    auxiliary_paths: list[Path]


def resolve_mads_roots(dataset_root: str | Path) -> MadsRoots:
    root = Path(dataset_root).resolve()
    multiview = root / "MADS_multiview" / "MADS" / "multi_view_data"
    depth = root / "MADS_depth" / "MADS" / "depth_data"
    if not multiview.is_dir():
        raise FileNotFoundError(f"Extracted MADS multi-view data not found: {multiview}")
    return MadsRoots(root, multiview, depth if depth.is_dir() else None)


def discover_mads_sequences(roots: MadsRoots) -> list[MadsSequence]:
    sequences = _discover_modality(roots.multiview_data, "multiview", _MULTIVIEW_VIDEO)
    if roots.depth_data is not None:
        sequences.extend(_discover_modality(roots.depth_data, "depth", _DEPTH_VIDEO))
    return sorted(sequences, key=lambda item: (item.modality, item.action, item.sequence))


def load_mads_ground_truth(path: str | Path, expected_joint_count: int | None = None) -> np.ndarray:
    source = Path(path)
    payload = loadmat(source, squeeze_me=True, struct_as_record=False)
    if "GTpose2" not in payload:
        raise ValueError(f"MADS ground-truth file does not contain GTpose2: {source}")
    cells = np.ravel(np.asarray(payload["GTpose2"], dtype=object))
    if cells.size == 0:
        raise ValueError(f"MADS ground-truth file is empty: {source}")
    poses = [np.asarray(cell, dtype=float) for cell in cells]
    shapes = {pose.shape for pose in poses}
    if len(shapes) != 1:
        raise ValueError(f"MADS ground-truth frames have inconsistent shapes: {sorted(shapes)}")
    points = np.stack(poses, axis=0)
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError(f"MADS ground truth must be [frames, joints, 3], got {points.shape}")
    if expected_joint_count is not None and points.shape[1] != expected_joint_count:
        raise ValueError(
            f"Expected {expected_joint_count} MADS joints in {source.name}, got {points.shape[1]}"
        )
    return points


def load_mads_camera_calibration(
    path: str | Path,
    camera_id: str,
    image_size: tuple[int, int],
) -> CameraCalibration:
    source = Path(path)
    payload = loadmat(source, squeeze_me=True, struct_as_record=False)
    required = {"KK", "kc", "om_ext", "T_ext"}
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"MADS calibration {source} is missing fields: {missing}")
    intrinsic = np.asarray(payload["KK"], dtype=float).reshape(3, 3)
    distortion = np.asarray(payload["kc"], dtype=float).reshape(-1)
    rotation_vector = np.asarray(payload["om_ext"], dtype=float).reshape(3)
    translation_vector = np.asarray(payload["T_ext"], dtype=float).reshape(3)
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    projection = intrinsic @ np.hstack([rotation_matrix, translation_vector[:, None]])
    return CameraCalibration(
        camera_id=camera_id,
        image_size=image_size,
        intrinsic_matrix=intrinsic,
        distortion_coefficients=distortion,
        rotation_vector=rotation_vector,
        translation_vector=translation_vector,
        projection_matrix=projection,
        reprojection_error_px=None,
    )


def probe_mads_video(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open MADS video: {source}")
    try:
        return {
            "path": str(source.resolve()),
            "size_bytes": source.stat().st_size,
            "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            "fps": float(capture.get(cv2.CAP_PROP_FPS) or 0.0),
            "image_size": [
                int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            ],
        }
    finally:
        capture.release()


def _discover_modality(root: Path, modality: str, pattern: re.Pattern[str]) -> list[MadsSequence]:
    grouped: dict[tuple[str, str], dict[str, Path]] = {}
    for video in root.glob("*/*.avi"):
        match = pattern.match(video.name)
        if match:
            key = (match.group("action"), match.group("sequence"))
            grouped.setdefault(key, {})[match.group("camera")] = video.resolve()

    output: list[MadsSequence] = []
    for (action, sequence), videos in grouped.items():
        action_root = root / action
        ground_truth = action_root / f"{action}_{sequence}_GT.mat"
        if not ground_truth.exists():
            raise FileNotFoundError(f"MADS ground truth missing for {action}/{sequence}: {ground_truth}")
        auxiliary: list[Path] = []
        if modality == "depth":
            depth_maps = action_root / f"{action}_{sequence}_depthMaps.mat"
            if depth_maps.exists():
                auxiliary.append(depth_maps.resolve())
        output.append(MadsSequence(modality, action, sequence, videos, ground_truth.resolve(), auxiliary))
    return output
