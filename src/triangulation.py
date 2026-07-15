from __future__ import annotations

import numpy as np
from itertools import combinations

from .data_structures import COCO_WHOLEBODY_KEYPOINTS, CameraCalibration, PersonPose2D, TriangulatedPose3D

def triangulate_frame(
    frame_idx: int,
    poses_by_camera: dict[str, PersonPose2D],
    calibrations: dict[str, CameraCalibration],
    min_views: int = 2,
    max_reprojection_error_px: float = 25.0,
    max_hypotheses: int = 16,
) -> TriangulatedPose3D:
    if min_views < 2:
        raise ValueError("min_views must be at least 2")
    if max_reprojection_error_px <= 0:
        raise ValueError("max_reprojection_error_px must be positive")
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

        result = robust_triangulate_n_view(
            normalized_points_2d=normalized_points_2d,
            normalized_projection_mats=normalized_projection_mats,
            pixel_points_2d=pixel_points_2d,
            pixel_projection_mats=pixel_projection_mats,
            scores=scores,
            min_views=min_views,
            max_reprojection_error_px=max_reprojection_error_px,
            max_hypotheses=max_hypotheses,
        )
        if result is None:
            continue
        point_3d, conditioning_score, inlier_indices, inlier_error = result

        keypoints_3d[joint_idx] = point_3d
        used_cameras[joint_idx] = len(inlier_indices)
        reprojection_error[joint_idx] = inlier_error
        triangulation_score[joint_idx] = triangulation_quality_score(
            scores=[scores[index] for index in inlier_indices],
            reprojection_error_px=reprojection_error[joint_idx],
            used_views=len(inlier_indices),
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


def triangulate_n_view(
    points_2d: list[np.ndarray],
    projection_mats: list[np.ndarray],
    weights: list[float] | None = None,
) -> tuple[np.ndarray, float] | None:
    if len(points_2d) != len(projection_mats):
        raise ValueError("points_2d and projection_mats must have equal length")
    if weights is not None and len(weights) != len(points_2d):
        raise ValueError("weights and points_2d must have equal length")
    rows = []
    for index, (point, projection) in enumerate(zip(points_2d, projection_mats, strict=True)):
        x, y = point
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        weight = float(np.sqrt(max(weights[index], 1e-6))) if weights is not None else 1.0
        rows.append((x * projection[2, :] - projection[0, :]) * weight)
        rows.append((y * projection[2, :] - projection[1, :]) * weight)
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


def robust_triangulate_n_view(
    normalized_points_2d: list[np.ndarray],
    normalized_projection_mats: list[np.ndarray],
    pixel_points_2d: list[np.ndarray],
    pixel_projection_mats: list[np.ndarray],
    scores: list[float],
    min_views: int,
    max_reprojection_error_px: float,
    max_hypotheses: int = 16,
) -> tuple[np.ndarray, float, list[int], float] | None:
    count = len(normalized_points_2d)
    lengths = {count, len(normalized_projection_mats), len(pixel_points_2d), len(pixel_projection_mats), len(scores)}
    if len(lengths) != 1:
        raise ValueError("All triangulation observation lists must have equal length")
    if count < min_views:
        return None
    if max_hypotheses < 1:
        raise ValueError("max_hypotheses must be positive")
    required_inliers = min_views if count <= 3 else max(min_views, count // 2 + 1)

    pair_candidates = list(combinations(range(count), 2))
    pair_candidates.sort(key=lambda pair: scores[pair[0]] * scores[pair[1]], reverse=True)
    if len(pair_candidates) > max_hypotheses:
        pair_candidates = pair_candidates[:max_hypotheses]
    hypotheses: list[tuple[int, float, float, list[int], np.ndarray]] = []
    for pair in pair_candidates:
        result = triangulate_n_view(
            [normalized_points_2d[index] for index in pair],
            [normalized_projection_mats[index] for index in pair],
            [scores[index] for index in pair],
        )
        if result is None:
            continue
        point, _ = result
        errors = reprojection_errors(point, pixel_points_2d, pixel_projection_mats)
        positive_depth = positive_depth_mask(point, normalized_projection_mats)
        inliers = [
            index for index, error in enumerate(errors)
            if positive_depth[index] and np.isfinite(error) and error <= max_reprojection_error_px
        ]
        if len(inliers) < required_inliers:
            continue
        median_error = float(np.median(errors[inliers]))
        confidence = float(np.mean(np.clip(np.asarray(scores)[inliers], 0.0, 1.0)))
        hypotheses.append((len(inliers), -median_error, confidence, inliers, point))
    if not hypotheses:
        return None
    _, _, _, inliers, initial_point = max(hypotheses, key=lambda item: item[:3])

    linear_result = triangulate_n_view(
        [normalized_points_2d[index] for index in inliers],
        [normalized_projection_mats[index] for index in inliers],
        [scores[index] for index in inliers],
    )
    if linear_result is None:
        return None
    conditioning = linear_result[1]
    point = refine_point_nonlinear(
        initial_point,
        [pixel_points_2d[index] for index in inliers],
        [pixel_projection_mats[index] for index in inliers],
        max_reprojection_error_px,
    )
    errors = reprojection_errors(point, pixel_points_2d, pixel_projection_mats)
    positive_depth = positive_depth_mask(point, normalized_projection_mats)
    refined_inliers = [
        index for index, error in enumerate(errors)
        if positive_depth[index] and np.isfinite(error) and error <= max_reprojection_error_px
    ]
    if len(refined_inliers) < required_inliers:
        return None
    if refined_inliers != inliers:
        point = refine_point_nonlinear(
            point,
            [pixel_points_2d[index] for index in refined_inliers],
            [pixel_projection_mats[index] for index in refined_inliers],
            max_reprojection_error_px,
        )
        errors = reprojection_errors(point, pixel_points_2d, pixel_projection_mats)
    mean_error = float(np.mean(errors[refined_inliers]))
    if not np.isfinite(mean_error) or mean_error > max_reprojection_error_px:
        return None
    if maximum_triangulation_angle_deg(
        point, [normalized_projection_mats[index] for index in refined_inliers]
    ) < 1.0:
        return None
    return point, conditioning, refined_inliers, mean_error


def refine_point_nonlinear(
    initial_point: np.ndarray,
    points_2d: list[np.ndarray],
    projection_mats: list[np.ndarray],
    reprojection_scale_px: float,
) -> np.ndarray:
    if len(points_2d) != len(projection_mats):
        raise ValueError("points_2d and projection_mats must have equal length")
    try:
        from scipy.optimize import least_squares
    except ModuleNotFoundError:
        return np.asarray(initial_point, dtype=float)

    def residuals(point: np.ndarray) -> np.ndarray:
        homogeneous = np.r_[point, 1.0]
        values: list[float] = []
        for observed, projection in zip(points_2d, projection_mats, strict=True):
            projected = projection @ homogeneous
            if abs(projected[2]) <= 1e-12:
                values.extend([reprojection_scale_px * 10.0, reprojection_scale_px * 10.0])
            else:
                values.extend((projected[:2] / projected[2] - observed).tolist())
        return np.asarray(values, dtype=float)

    optimized = least_squares(
        residuals,
        np.asarray(initial_point, dtype=float),
        method="trf",
        loss="soft_l1",
        f_scale=max(float(reprojection_scale_px) / 3.0, 1.0),
        max_nfev=60,
    )
    return optimized.x.astype(float) if optimized.success and np.all(np.isfinite(optimized.x)) else np.asarray(initial_point, dtype=float)


def reprojection_errors(
    point_3d: np.ndarray,
    points_2d: list[np.ndarray],
    projection_mats: list[np.ndarray],
) -> np.ndarray:
    if len(points_2d) != len(projection_mats):
        raise ValueError("points_2d and projection_mats must have equal length")
    homogeneous = np.r_[np.asarray(point_3d, dtype=float), 1.0]
    errors = np.full(len(points_2d), np.inf, dtype=float)
    for index, (observed, projection) in enumerate(zip(points_2d, projection_mats, strict=True)):
        projected = np.asarray(projection, dtype=float) @ homogeneous
        if abs(projected[2]) > 1e-12:
            errors[index] = float(np.linalg.norm(projected[:2] / projected[2] - observed))
    return errors


def positive_depth_mask(point_3d: np.ndarray, normalized_projection_mats: list[np.ndarray]) -> np.ndarray:
    homogeneous = np.r_[np.asarray(point_3d, dtype=float), 1.0]
    return np.asarray([(projection @ homogeneous)[2] > 1e-9 for projection in normalized_projection_mats], dtype=bool)


def maximum_triangulation_angle_deg(point_3d: np.ndarray, projection_mats: list[np.ndarray]) -> float:
    rays: list[np.ndarray] = []
    for projection in projection_mats:
        rotation = np.asarray(projection[:, :3], dtype=float)
        translation = np.asarray(projection[:, 3], dtype=float)
        try:
            center = -np.linalg.solve(rotation, translation)
        except np.linalg.LinAlgError:
            continue
        ray = np.asarray(point_3d, dtype=float) - center
        norm = np.linalg.norm(ray)
        if norm > 1e-12 and np.isfinite(norm):
            rays.append(ray / norm)
    maximum = 0.0
    for first, second in combinations(rays, 2):
        angle = float(np.degrees(np.arccos(np.clip(np.dot(first, second), -1.0, 1.0))))
        maximum = max(maximum, angle)
    return maximum


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
    errors = reprojection_errors(point_3d, points_2d, projection_mats)
    finite = errors[np.isfinite(errors)]
    return float(np.mean(finite)) if finite.size else float("nan")

def stack_triangulated(poses: list[TriangulatedPose3D]) -> dict[str, np.ndarray]:
    if not poses:
        return {
            "frame_idx": np.empty((0,), dtype=int),
            "keypoints_3d_world": np.empty((0, COCO_WHOLEBODY_KEYPOINTS, 3), dtype=float),
            "triangulation_score": np.empty((0, COCO_WHOLEBODY_KEYPOINTS), dtype=float),
            "reprojection_error": np.empty((0, COCO_WHOLEBODY_KEYPOINTS), dtype=float),
            "used_cameras": np.empty((0, COCO_WHOLEBODY_KEYPOINTS), dtype=int),
        }
    return {
        "frame_idx": np.asarray([pose.frame_idx for pose in poses], dtype=int),
        "keypoints_3d_world": np.stack([pose.keypoints_3d_world for pose in poses], axis=0),
        "triangulation_score": np.stack([pose.triangulation_score for pose in poses], axis=0),
        "reprojection_error": np.stack([pose.reprojection_error for pose in poses], axis=0),
        "used_cameras": np.stack([pose.used_cameras for pose in poses], axis=0),
    }
