from __future__ import annotations

import cv2
import numpy as np

from .data_structures import COCO_WHOLEBODY_KEYPOINTS, CameraCalibration, PersonPose2D
from .triangulation import triangulate_frame


def build_synthetic_calibrations() -> dict[str, CameraCalibration]:
    intrinsic = np.array(
        [
            [1200.0, 0.0, 640.0],
            [0.0, 1200.0, 360.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    cameras = {
        "cam_front": (np.eye(3), np.array([0.0, 0.0, 0.0])),
        "cam_back": (_rotation_y(np.deg2rad(-8.0)), np.array([-0.65, 0.0, 0.04])),
        "cam_side": (_rotation_y(np.deg2rad(10.0)), np.array([0.62, 0.0, 0.03])),
    }
    return {
        camera_id: _calibration_from_rt(camera_id, intrinsic, rotation, translation)
        for camera_id, (rotation, translation) in cameras.items()
    }


def build_synthetic_world_sequence(frame_count: int, valid_joint_count: int = 33) -> np.ndarray:
    keypoints = np.full((frame_count, COCO_WHOLEBODY_KEYPOINTS, 3), np.nan, dtype=float)
    for frame_idx in range(frame_count):
        phase = frame_idx / max(frame_count - 1, 1)
        for joint_idx in range(valid_joint_count):
            lateral = (joint_idx % 5 - 2) * 0.055
            vertical = 0.70 + (joint_idx // 5) * 0.045
            forward = 3.0 + 0.18 * np.sin(phase * np.pi * 2.0 + joint_idx * 0.2)
            keypoints[frame_idx, joint_idx] = [
                lateral + 0.05 * np.sin(phase * np.pi * 2.0),
                vertical + 0.03 * np.cos(phase * np.pi * 2.0 + joint_idx * 0.1),
                forward,
            ]
    return keypoints


def project_world_sequence(
    keypoints_3d_world: np.ndarray,
    calibrations: dict[str, CameraCalibration],
    score: float = 0.95,
) -> dict[int, dict[str, PersonPose2D]]:
    frames: dict[int, dict[str, PersonPose2D]] = {}
    for frame_idx in range(keypoints_3d_world.shape[0]):
        frames[frame_idx] = {}
        for camera_id, calibration in calibrations.items():
            xy = np.full((COCO_WHOLEBODY_KEYPOINTS, 2), np.nan, dtype=float)
            scores = np.zeros(COCO_WHOLEBODY_KEYPOINTS, dtype=float)
            valid = np.zeros(COCO_WHOLEBODY_KEYPOINTS, dtype=bool)
            for joint_idx, point_3d in enumerate(keypoints_3d_world[frame_idx]):
                projected = project_point(point_3d, calibration.projection_matrix)
                if projected is None:
                    continue
                xy[joint_idx] = projected
                scores[joint_idx] = score
                valid[joint_idx] = True
            frames[frame_idx][camera_id] = PersonPose2D(
                camera_id=camera_id,
                frame_idx=frame_idx,
                keypoints_xy=xy,
                scores=scores,
                valid_mask=valid,
            )
    return frames


def build_synthetic_triangulation_result(frame_count: int) -> dict[str, np.ndarray]:
    calibrations = build_synthetic_calibrations()
    world = build_synthetic_world_sequence(frame_count)
    projected = project_world_sequence(world, calibrations)
    triangulated = [
        triangulate_frame(
            frame_idx=frame_idx,
            poses_by_camera=poses_by_camera,
            calibrations=calibrations,
            min_views=2,
        )
        for frame_idx, poses_by_camera in projected.items()
    ]
    return {
        "keypoints_3d_world": np.stack([pose.keypoints_3d_world for pose in triangulated], axis=0),
        "triangulation_score": np.stack([pose.triangulation_score for pose in triangulated], axis=0),
        "reprojection_error": np.stack([pose.reprojection_error for pose in triangulated], axis=0),
        "used_cameras": np.stack([pose.used_cameras for pose in triangulated], axis=0),
        "poses_2d_by_frame": projected,
    }


def project_point(point_3d: np.ndarray, projection_matrix: np.ndarray) -> np.ndarray | None:
    if not np.all(np.isfinite(point_3d)):
        return None
    homogeneous = np.array([point_3d[0], point_3d[1], point_3d[2], 1.0], dtype=float)
    projected = projection_matrix @ homogeneous
    if abs(projected[2]) < 1e-12:
        return None
    return (projected[:2] / projected[2]).astype(float)


def _calibration_from_rt(
    camera_id: str,
    intrinsic: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> CameraCalibration:
    projection = intrinsic @ np.hstack([rotation, translation.reshape(3, 1)])
    rotation_vector, _ = cv2.Rodrigues(rotation)
    return CameraCalibration(
        camera_id=camera_id,
        image_size=(1280, 720),
        intrinsic_matrix=intrinsic.copy(),
        distortion_coefficients=np.zeros(5, dtype=float),
        rotation_vector=rotation_vector.reshape(-1),
        translation_vector=translation.astype(float),
        projection_matrix=projection,
        reprojection_error_px=0.0,
    )


def _rotation_y(angle_rad: float) -> np.ndarray:
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    return np.array(
        [
            [cos_a, 0.0, sin_a],
            [0.0, 1.0, 0.0],
            [-sin_a, 0.0, cos_a],
        ],
        dtype=float,
    )
