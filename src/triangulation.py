from __future__ import annotations

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
        pixel_points_2d: list[np.ndarray] = []
        normalized_points_2d: list[np.ndarray] = []
        pixel_projection_mats: list[np.ndarray] = []
        normalized_projection_mats: list[np.ndarray] = []
        scores: list[float] = []

        for camera_id, pose in poses_by_camera.items():
            if camera_id not in calibrations or not pose.valid_mask[joint_idx]:
                continue
            point = pose.keypoints_xy[joint_idx]
            if not np.all(np.isfinite(point)):
                continue
            calibration = calibrations[camera_id]
            pixel_point = undistort_point(point, calibration, pixel_coordinates=True)
            normalized_point = undistort_point(point, calibration, pixel_coordinates=False)
            pixel_points_2d.append(pixel_point.astype(float))
            normalized_points_2d.append(normalized_point.astype(float))
            pixel_projection_mats.append(calibration.projection_matrix.astype(float))
            normalized_projection_mats.append(normalized_projection_matrix(calibration))
            scores.append(float(pose.scores[joint_idx]))

        if len(normalized_points_2d) < min_views:
            continue

        result = triangulate_n_view(normalized_points_2d, normalized_projection_mats)
        if result is None:
            continue
        point_3d, conditioning_score = result

        keypoints_3d[joint_idx] = point_3d
        used_cameras[joint_idx] = len(normalized_points_2d)
        reprojection_error[joint_idx] = mean_reprojection_error(point_3d, pixel_points_2d, pixel_projection_mats)
        triangulation_score[joint_idx] = triangulation_quality_score(
            scores=scores,
            reprojection_error_px=reprojection_error[joint_idx],
            used_views=len(normalized_points_2d),
            conditioning_score=conditioning_score,
        )

    return TriangulatedPose3D(
        frame_idx=frame_idx,
        keypoints_3d_world=keypoints_3d,
        triangulation_score=triangulation_score,
        reprojection_error=reprojection_error,
        used_cameras=used_cameras,
    )

def undistort_point(point: np.ndarray, calibration: CameraCalibration, pixel_coordinates: bool = True) -> np.ndarray:
    distortion = np.asarray(calibration.distortion_coefficients, dtype=float).reshape(-1)
    intrinsic = calibration.intrinsic_matrix.astype(float)
    if distortion.size == 0 or not np.any(np.abs(distortion) > 1e-12):
        if pixel_coordinates:
            return np.asarray(point, dtype=float)
        homogeneous = np.array([point[0], point[1], 1.0], dtype=float)
        normalized = np.linalg.solve(intrinsic, homogeneous)
        if abs(normalized[2]) < 1e-12:
            return np.array([np.nan, np.nan], dtype=float)
        return (normalized[:2] / normalized[2]).astype(float)
    import cv2

    src = np.asarray(point, dtype=float).reshape(1, 1, 2)
    if pixel_coordinates:
        corrected = cv2.undistortPoints(src, intrinsic, distortion, P=intrinsic)
        return corrected.reshape(2)
    corrected = cv2.undistortPoints(src, intrinsic, distortion, P=None)
    return corrected.reshape(2)


def normalized_projection_matrix(calibration: CameraCalibration) -> np.ndarray:
    intrinsic = calibration.intrinsic_matrix.astype(float)
    projection = calibration.projection_matrix.astype(float)
    return np.linalg.solve(intrinsic, projection)


def triangulate_n_view(points_2d: list[np.ndarray], projection_mats: list[np.ndarray]) -> tuple[np.ndarray, float] | None:
    rows = []
    for point, projection in zip(points_2d, projection_mats):
        x, y = point
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        rows.append(x * projection[2, :] - projection[0, :])
        rows.append(y * projection[2, :] - projection[1, :])
    if len(rows) < 4:
        return None
    design = np.asarray(rows, dtype=float)
    row_norms = np.linalg.norm(design, axis=1)
    valid_rows = row_norms > 1e-12
    if np.count_nonzero(valid_rows) < 4:
        return None
    design = design[valid_rows] / row_norms[valid_rows, None]
    try:
        _, singular_values, vt = np.linalg.svd(design)
    except np.linalg.LinAlgError:
        return None
    homogeneous = vt[-1]
    if abs(homogeneous[3]) < 1e-12:
        return None
    conditioning_score = _conditioning_score(singular_values)
    return (homogeneous[:3] / homogeneous[3]).astype(float), conditioning_score


def triangulation_quality_score(
    scores: list[float],
    reprojection_error_px: float,
    used_views: int,
    conditioning_score: float,
    reprojection_tolerance_px: float = 12.0,
) -> float:
    finite_scores = np.asarray([score for score in scores if np.isfinite(score)], dtype=float)
    confidence = float(np.mean(np.clip(finite_scores, 0.0, 1.0))) if finite_scores.size else 0.0
    if np.isfinite(reprojection_error_px):
        reprojection_quality = float(np.exp(-max(reprojection_error_px, 0.0) / max(reprojection_tolerance_px, 1e-6)))
    else:
        reprojection_quality = 0.0
    view_quality = min(max((used_views - 1) / 2.0, 0.0), 1.0)
    return float(np.clip(confidence * reprojection_quality * view_quality * conditioning_score, 0.0, 1.0))


def _conditioning_score(singular_values: np.ndarray) -> float:
    if singular_values.size < 2 or singular_values[-2] <= 1e-12:
        return 0.0
    residual_ratio = float(np.clip(singular_values[-1] / singular_values[-2], 0.0, 1.0))
    return float(1.0 - residual_ratio)

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
    if not poses:
        return {
            "keypoints_3d_world": np.empty((0, COCO_WHOLEBODY_KEYPOINTS, 3), dtype=float),
            "triangulation_score": np.empty((0, COCO_WHOLEBODY_KEYPOINTS), dtype=float),
            "reprojection_error": np.empty((0, COCO_WHOLEBODY_KEYPOINTS), dtype=float),
            "used_cameras": np.empty((0, COCO_WHOLEBODY_KEYPOINTS), dtype=int),
        }
    return {
        "keypoints_3d_world": np.stack([pose.keypoints_3d_world for pose in poses], axis=0),
        "triangulation_score": np.stack([pose.triangulation_score for pose in poses], axis=0),
        "reprojection_error": np.stack([pose.reprojection_error for pose in poses], axis=0),
        "used_cameras": np.stack([pose.used_cameras for pose in poses], axis=0),
    }

