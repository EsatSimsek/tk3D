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


def collect_checkerboard_detections(
    video_path: str | Path,
    pattern_size: tuple[int, int],
    square_size_m: float,
    frame_stride: int,
    frame_offset: int = 0,
) -> tuple[list[np.ndarray], list[np.ndarray], tuple[int, int], dict[int, np.ndarray]]:
    object_template = checkerboard_object_points(pattern_size, square_size_m)
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    detections_by_global_frame: dict[int, np.ndarray] = {}
    image_size: tuple[int, int] | None = None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    for local_frame_idx, _, frame in iter_video_frames(video_path, stride=frame_stride):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        image_size = (gray.shape[1], gray.shape[0])
        found, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        if not found:
            continue
        refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        object_points.append(object_template.copy())
        image_points.append(refined)
        detections_by_global_frame[int(local_frame_idx) + int(frame_offset)] = refined

    if image_size is None:
        raise FileNotFoundError(f"No frames found in calibration video: {video_path}")
    return object_points, image_points, image_size, detections_by_global_frame


def calibrate_multiview_cameras(
    camera_videos: dict[str, str | Path],
    frame_offsets: dict[str, int],
    pattern_size: tuple[int, int],
    square_size_m: float,
    frame_stride: int = 10,
    min_valid_frames: int = 12,
    min_common_frames: int = 3,
    reference_camera_id: str | None = None,
) -> list[CameraCalibration]:
    if len(camera_videos) < 2:
        raise ValueError("Multi-view calibration requires at least two cameras.")
    object_template = checkerboard_object_points(pattern_size, square_size_m)
    intrinsics: dict[str, np.ndarray] = {}
    distortions: dict[str, np.ndarray] = {}
    image_sizes: dict[str, tuple[int, int]] = {}
    rms_errors: dict[str, float] = {}
    detections: dict[str, dict[int, np.ndarray]] = {}

    for camera_id, video_path in camera_videos.items():
        object_points, image_points, image_size, detections_by_frame = collect_checkerboard_detections(
            video_path=video_path,
            pattern_size=pattern_size,
            square_size_m=square_size_m,
            frame_stride=frame_stride,
            frame_offset=int(frame_offsets.get(camera_id, 0)),
        )
        if len(object_points) < min_valid_frames:
            raise ValueError(
                f"{camera_id}: need at least {min_valid_frames} checkerboard detections, got {len(object_points)}"
            )
        rms, intrinsic, distortion, _, _ = cv2.calibrateCamera(object_points, image_points, image_size, None, None)
        intrinsics[camera_id] = intrinsic
        distortions[camera_id] = distortion
        image_sizes[camera_id] = image_size
        rms_errors[camera_id] = float(rms)
        detections[camera_id] = detections_by_frame

    reference_id = reference_camera_id or next(iter(camera_videos))
    if reference_id not in camera_videos:
        raise ValueError(f"Reference camera is not part of calibration set: {reference_id}")
    common_frames = sorted(set.intersection(*(set(items) for items in detections.values())))
    if len(common_frames) < min_common_frames:
        raise ValueError(
            f"Need at least {min_common_frames} synchronized checkerboard detections across all cameras, "
            f"got {len(common_frames)}"
        )

    relative_poses: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {camera_id: [] for camera_id in camera_videos}
    for global_frame_idx in common_frames:
        board_to_camera: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for camera_id in camera_videos:
            ok, rvec, tvec = cv2.solvePnP(
                object_template,
                detections[camera_id][global_frame_idx],
                intrinsics[camera_id],
                distortions[camera_id],
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not ok:
                break
            rotation, _ = cv2.Rodrigues(rvec)
            board_to_camera[camera_id] = (rotation, tvec.reshape(3))
        if len(board_to_camera) != len(camera_videos):
            continue

        ref_rotation, ref_translation = board_to_camera[reference_id]
        for camera_id, (rotation, translation) in board_to_camera.items():
            if camera_id == reference_id:
                relative_poses[camera_id].append((np.eye(3, dtype=float), np.zeros(3, dtype=float)))
                continue
            relative_rotation = rotation @ ref_rotation.T
            relative_translation = translation - relative_rotation @ ref_translation
            relative_poses[camera_id].append((relative_rotation, relative_translation))

    if any(len(poses) < min_common_frames for poses in relative_poses.values()):
        raise ValueError("Not enough valid solvePnP poses after synchronized checkerboard matching.")

    calibrations: list[CameraCalibration] = []
    for camera_id in camera_videos:
        rotation = _average_rotation([pose[0] for pose in relative_poses[camera_id]])
        translation = np.median(np.asarray([pose[1] for pose in relative_poses[camera_id]], dtype=float), axis=0)
        projection = intrinsics[camera_id] @ np.hstack([rotation, translation.reshape(3, 1)])
        rotation_vector, _ = cv2.Rodrigues(rotation)
        calibrations.append(
            CameraCalibration(
                camera_id=camera_id,
                image_size=image_sizes[camera_id],
                intrinsic_matrix=intrinsics[camera_id],
                distortion_coefficients=distortions[camera_id],
                rotation_vector=rotation_vector.reshape(-1),
                translation_vector=translation.reshape(-1),
                projection_matrix=projection,
                reprojection_error_px=rms_errors[camera_id],
            )
        )
    return calibrations

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
    best_idx = representative_extrinsic_index(object_points, image_points, intrinsic, distortion, rvecs, tvecs)
    rotation_vector = rvecs[best_idx]
    translation_vector = tvecs[best_idx]
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


def representative_extrinsic_index(
    object_points: list[np.ndarray],
    image_points: list[np.ndarray],
    intrinsic: np.ndarray,
    distortion: np.ndarray,
    rvecs: tuple[np.ndarray, ...],
    tvecs: tuple[np.ndarray, ...],
) -> int:
    if not rvecs or not tvecs:
        raise ValueError("No extrinsic estimates were produced by camera calibration.")
    errors = np.asarray(
        [
            _view_reprojection_error(object_point, image_point, intrinsic, distortion, rvec, tvec)
            for object_point, image_point, rvec, tvec in zip(object_points, image_points, rvecs, tvecs)
        ],
        dtype=float,
    )
    centers = np.asarray([_camera_center_world(rvec, tvec) for rvec, tvec in zip(rvecs, tvecs)], dtype=float)
    finite_centers = np.all(np.isfinite(centers), axis=1)
    finite_errors = np.isfinite(errors)
    if not np.any(finite_errors):
        return 0
    if np.any(finite_centers):
        median_center = np.median(centers[finite_centers], axis=0)
        center_distance = np.linalg.norm(centers - median_center, axis=1)
        center_distance[~np.isfinite(center_distance)] = np.inf
        error_scale = np.nanmedian(errors[finite_errors]) or 1.0
        distance_scale = np.nanmedian(center_distance[np.isfinite(center_distance)]) or 1.0
        combined = (errors / max(error_scale, 1e-9)) + (center_distance / max(distance_scale, 1e-9))
        combined[~finite_errors] = np.inf
        return int(np.argmin(combined))
    return int(np.nanargmin(errors))


def _view_reprojection_error(
    object_points: np.ndarray,
    image_points: np.ndarray,
    intrinsic: np.ndarray,
    distortion: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> float:
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, intrinsic, distortion)
    observed = np.asarray(image_points, dtype=float).reshape(-1, 2)
    projected = projected.reshape(-1, 2)
    return float(np.mean(np.linalg.norm(projected - observed, axis=1)))


def _camera_center_world(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    rotation, _ = cv2.Rodrigues(rvec)
    return (-rotation.T @ np.asarray(tvec, dtype=float).reshape(3, 1)).reshape(3)


def _average_rotation(rotations: list[np.ndarray]) -> np.ndarray:
    if not rotations:
        raise ValueError("Cannot average an empty rotation set.")
    accumulator = np.sum(np.asarray(rotations, dtype=float), axis=0)
    u, _, vt = np.linalg.svd(accumulator)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    return rotation

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
    errors = np.asarray(
        [cam.reprojection_error_px for cam in calibrations if cam.reprojection_error_px is not None],
        dtype=float,
    )
    mean_error = float(np.nanmean(errors)) if errors.size and np.any(np.isfinite(errors)) else None
    return {
        "camera_count": len(calibrations),
        "mean_reprojection_error_px": mean_error,
        "cameras": [
            {
                "camera_id": cam.camera_id,
                "image_size": cam.image_size,
                "reprojection_error_px": cam.reprojection_error_px,
            }
            for cam in calibrations
        ],
    }
