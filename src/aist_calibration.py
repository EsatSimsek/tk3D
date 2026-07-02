from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .data_structures import CameraCalibration


def find_aist_camera_setting(sequence_name: str, cameras_dir: str | Path) -> str:
    mapping_path = Path(cameras_dir) / "mapping.txt"
    if not mapping_path.exists():
        raise FileNotFoundError(f"AIST++ mapping.txt not found: {mapping_path}")

    with mapping_path.open("r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 2 and parts[0] == sequence_name:
                return parts[1]
    raise KeyError(f"AIST++ sequence not found in mapping.txt: {sequence_name}")


def load_aist_camera_calibrations(
    sequence_name: str,
    cameras_dir: str | Path,
    camera_ids: list[str] | None = None,
) -> list[CameraCalibration]:
    cameras_root = Path(cameras_dir)
    setting_name = find_aist_camera_setting(sequence_name, cameras_root)
    setting_path = cameras_root / f"{setting_name}.json"
    if not setting_path.exists():
        raise FileNotFoundError(f"AIST++ setting file not found: {setting_path}")

    with setting_path.open("r", encoding="utf-8") as file:
        raw_cameras = json.load(file)

    selected = set(camera_ids or [])
    calibrations = []
    for raw in raw_cameras:
        camera_id = raw["name"]
        if selected and camera_id not in selected:
            continue
        calibrations.append(aist_camera_to_calibration(raw))

    missing = selected.difference({camera.camera_id for camera in calibrations})
    if missing:
        raise KeyError(f"AIST++ setting {setting_name} missing cameras: {sorted(missing)}")
    return calibrations


def aist_camera_to_calibration(raw: dict) -> CameraCalibration:
    camera_id = raw["name"]
    image_size = tuple(int(value) for value in raw["size"])
    intrinsic = np.asarray(raw["matrix"], dtype=float)
    distortion = np.asarray(raw["distortions"], dtype=float)
    rotation_vector = np.asarray(raw["rotation"], dtype=float).reshape(3)
    translation_vector = np.asarray(raw["translation"], dtype=float).reshape(3)
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    projection = intrinsic @ np.hstack([rotation_matrix, translation_vector.reshape(3, 1)])

    return CameraCalibration(
        camera_id=camera_id,
        image_size=(image_size[0], image_size[1]),
        intrinsic_matrix=intrinsic,
        distortion_coefficients=distortion,
        rotation_vector=rotation_vector,
        translation_vector=translation_vector,
        projection_matrix=projection,
        reprojection_error_px=None,
    )
