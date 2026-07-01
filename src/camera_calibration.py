from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .data_structures import CameraCalibration
from .video_io import iter_video_frames


def checkerboard_object_points(pattern_size: tuple[int, int], square_size_m: float) -> np.ndarray:
    cols, rows = pattern_size
    points = np.zeros((rows * cols, 3), np.float32)
    points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    points *= float(square_size_m)
    return points


def collect_checkerboard_points(
    video_path: str | Path,
    pattern_size: tuple[int, int],
    square_size_m: float,
    frame_stride: int,
) -> tuple[list[np.ndarray], list[np.ndarray], tuple[int, int]]:
    object_template = checkerboard_object_points(pattern_size, square_size_m)
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    for _, _, frame in iter_video_frames(video_path, stride=frame_stride):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        image_size = (gray.shape[1], gray.shape[0])
        found, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        if not found:
            continue
        refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        object_points.append(object_template.copy())
        image_points.append(refined)

    if image_size is None:
        raise FileNotFoundError(f"No frames found in calibration video: {video_path}")
    return object_points, image_points, image_size


def calibrate_single_camera(
    camera_id: str,
    video_path: str | Path,
    pattern_size: tuple[int, int],
    square_size_m: float,
    frame_stride: int = 10,
    min_valid_frames: int = 12,
) -> CameraCalibration:
    object_points, image_points, image_size = collect_checkerboard_points(
        video_path=video_path,
        pattern_size=pattern_size,
        square_size_m=square_size_m,
        frame_stride=frame_stride,
    )
    if len(object_points) < min_valid_frames:
        raise ValueError(
            f"{camera_id}: need at least {min_valid_frames} checkerboard detections, got {len(object_points)}"
        )

    rms, intrinsic, distortion, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )
    rotation_vector = rvecs[0]
    translation_vector = tvecs[0]
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    extrinsic = np.hstack([rotation_matrix, translation_vector.reshape(3, 1)])
    projection = intrinsic @ extrinsic

    return CameraCalibration(
        camera_id=camera_id,
        image_size=image_size,
        intrinsic_matrix=intrinsic,
        distortion_coefficients=distortion,
        rotation_vector=rotation_vector.reshape(-1),
        translation_vector=translation_vector.reshape(-1),
        projection_matrix=projection,
        reprojection_error_px=float(rms),
    )


def save_calibrations(calibrations: list[CameraCalibration], output_path: str | Path) -> None:
    payload = {"cameras": [camera.to_json_dict() for camera in calibrations]}
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def load_calibrations(path: str | Path) -> dict[str, CameraCalibration]:
    with Path(path).open("r", encoding="utf-8") as file:
        raw: dict[str, Any] = json.load(file)
    return {
        item["camera_id"]: CameraCalibration.from_json_dict(item)
        for item in raw.get("cameras", [])
    }


def calibration_report(calibrations: list[CameraCalibration]) -> dict[str, Any]:
    return {
        "camera_count": len(calibrations),
        "mean_reprojection_error_px": float(
            np.nanmean([cam.reprojection_error_px for cam in calibrations])
        )
        if calibrations
        else None,
        "cameras": [
            {
                "camera_id": cam.camera_id,
                "image_size": cam.image_size,
                "reprojection_error_px": cam.reprojection_error_px,
            }
            for cam in calibrations
        ],
    }
