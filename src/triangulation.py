from __future__ import annotations

import cv2
import numpy as np

from .data_structures import COCO_WHOLEBODY_KEYPOINTS, CameraCalibration, PersonPose2D, TriangulatedPose3D


def triangulate_frame(
    frame_idx: int,
    poses_by_camera: dict[str, PersonPose2D],
    calibrations: dict[str, CameraCalibration],
    min_views: int = 2,
) -> TriangulatedPose3D:
    keypoints_3d = np.full((COCO_WHOLEBODY_KEYPOINTS, 3), np.nan, dtype=float)
    triangulation_score = np.zeros(COCO_WHOLEBODY_KEYPOINTS, dtype=float)
    reprojection_error = np.full(COCO_WHOLEBODY_KEYPOINTS, np.nan, dtype=float)
    used_cameras = np.zeros(COCO_WHOLEBODY_KEYPOINTS, dtype=int)

    for joint_idx in range(COCO_WHOLEBODY_KEYPOINTS):
        camera_ids: list[str] = []
        points_2d: list[np.ndarray] = []
        projection_mats: list[np.ndarray] = []
        scores: list[float] = []

        for camera_id, pose in poses_by_camera.items():
            if camera_id not in calibrations or not pose.valid_mask[joint_idx]:
                continue
            point = pose.keypoints_xy[joint_idx]
            if not np.all(np.isfinite(point)):
                continue
            camera_ids.append(camera_id)
            points_2d.append(point.astype(float))
            projection_mats.append(calibrations[camera_id].projection_matrix.astype(float))
            scores.append(float(pose.scores[joint_idx]))

        if len(points_2d) < min_views:
            continue

        point_3d = triangulate_n_view(points_2d, projection_mats)
        if point_3d is None:
            continue

        keypoints_3d[joint_idx] = point_3d
        triangulation_score[joint_idx] = float(np.mean(scores))
        used_cameras[joint_idx] = len(points_2d)
        reprojection_error[joint_idx] = mean_reprojection_error(point_3d, points_2d, projection_mats)

    return TriangulatedPose3D(
        frame_idx=frame_idx,
        keypoints_3d_world=keypoints_3d,
        triangulation_score=triangulation_score,
        reprojection_error=reprojection_error,
        used_cameras=used_cameras,
    )


def triangulate_n_view(points_2d: list[np.ndarray], projection_mats: list[np.ndarray]) -> np.ndarray | None:
    if len(points_2d) == 2:
        point_a = np.asarray(points_2d[0], dtype=float).reshape(2, 1)
        point_b = np.asarray(points_2d[1], dtype=float).reshape(2, 1)
        homogeneous = cv2.triangulatePoints(projection_mats[0], projection_mats[1], point_a, point_b)
        if abs(homogeneous[3, 0]) < 1e-12:
            return None
        return (homogeneous[:3, 0] / homogeneous[3, 0]).astype(float)

    rows = []
    for point, projection in zip(points_2d, projection_mats):
        x, y = point
        rows.append(x * projection[2, :] - projection[0, :])
        rows.append(y * projection[2, :] - projection[1, :])
    design = np.asarray(rows, dtype=float)
    _, _, vt = np.linalg.svd(design)
    homogeneous = vt[-1]
    if abs(homogeneous[3]) < 1e-12:
        return None
    return (homogeneous[:3] / homogeneous[3]).astype(float)


def mean_reprojection_error(
    point_3d: np.ndarray,
    points_2d: list[np.ndarray],
    projection_mats: list[np.ndarray],
) -> float:
    homogeneous = np.array([point_3d[0], point_3d[1], point_3d[2], 1.0], dtype=float)
    errors = []
    for observed, projection in zip(points_2d, projection_mats):
        projected = projection @ homogeneous
        if abs(projected[2]) < 1e-12:
            continue
        xy = projected[:2] / projected[2]
        errors.append(float(np.linalg.norm(xy - observed)))
    return float(np.mean(errors)) if errors else float("nan")


def stack_triangulated(poses: list[TriangulatedPose3D]) -> dict[str, np.ndarray]:
    return {
        "keypoints_3d_world": np.stack([pose.keypoints_3d_world for pose in poses], axis=0),
        "triangulation_score": np.stack([pose.triangulation_score for pose in poses], axis=0),
        "reprojection_error": np.stack([pose.reprojection_error for pose in poses], axis=0),
        "used_cameras": np.stack([pose.used_cameras for pose in poses], axis=0),
    }
