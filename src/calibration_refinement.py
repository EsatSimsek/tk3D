"""Two-view epipolar consistency check and relative-rotation refinement.

The refinement re-estimates only the orientation of one camera about its own
centre from high-confidence 2D keypoint correspondences of a recording. Camera
centres, baseline length, intrinsics and the reference camera (and therefore the
world axes) are kept. It fits on one half of the frames and reports the other
half, so the improvement is not only measured on the data it was fitted to.

This is internal two-view consistency, not independent 3D accuracy: the same
athlete keypoints define the correction. The result must keep its own
calibration mode and provenance and never replace the source calibration file.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any

import cv2
import numpy as np
from scipy.optimize import least_squares

from src.data_structures import CameraCalibration

REFINED_CALIBRATION_MODE = "zed_fusion_multiview_keypoint_refined"
REFINEMENT_ALGORITHM = "tk3d_two_view_relative_rotation_refinement_v1"


def _rotation(calibration: CameraCalibration) -> np.ndarray:
    rotation, _ = cv2.Rodrigues(np.asarray(calibration.rotation_vector, dtype=float).reshape(3, 1))
    return rotation


def fundamental_matrix(first: CameraCalibration, second: CameraCalibration) -> np.ndarray:
    r1, r2 = _rotation(first), _rotation(second)
    t1 = np.asarray(first.translation_vector, dtype=float).reshape(3)
    t2 = np.asarray(second.translation_vector, dtype=float).reshape(3)
    relative = r2 @ r1.T
    translation = t2 - relative @ t1
    skew = np.array(
        [
            [0.0, -translation[2], translation[1]],
            [translation[2], 0.0, -translation[0]],
            [-translation[1], translation[0], 0.0],
        ]
    )
    k1 = np.asarray(first.intrinsic_matrix, dtype=float)
    k2 = np.asarray(second.intrinsic_matrix, dtype=float)
    return np.linalg.inv(k2).T @ skew @ relative @ np.linalg.inv(k1)


def signed_epipolar_distances(
    first: CameraCalibration,
    second: CameraCalibration,
    points_first: np.ndarray,
    points_second: np.ndarray,
) -> np.ndarray:
    """Signed pixel distance of each second-camera point to its epipolar line.

    Zero-distortion calibrations only; raw pixels are used as-is.
    """
    for calibration in (first, second):
        if np.any(np.abs(np.asarray(calibration.distortion_coefficients, dtype=float)) > 1e-12):
            raise ValueError("epipolar check expects zero-distortion (rectified) calibrations")
    a = np.asarray(points_first, dtype=float).reshape(-1, 2)
    b = np.asarray(points_second, dtype=float).reshape(-1, 2)
    if a.shape != b.shape:
        raise ValueError("point arrays must have equal shape")
    if a.size == 0:
        return np.empty(0, dtype=float)
    lines = np.c_[a, np.ones(len(a))] @ fundamental_matrix(first, second).T
    norm = np.hypot(lines[:, 0], lines[:, 1])
    return np.sum(lines * np.c_[b, np.ones(len(b))], axis=1) / np.where(norm > 1e-12, norm, np.nan)


def epipolar_summary(distances: np.ndarray) -> dict[str, Any]:
    values = np.asarray(distances, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0, "signed_median_px": None, "abs_median_px": None, "abs_p95_px": None, "same_sign_fraction": None}
    return {
        "count": int(values.size),
        "signed_median_px": float(np.median(values)),
        "abs_median_px": float(np.median(np.abs(values))),
        "abs_p95_px": float(np.percentile(np.abs(values), 95.0)),
        "same_sign_fraction": float(max(np.mean(values > 0), np.mean(values < 0))),
    }


def rotate_about_centre(calibration: CameraCalibration, rotation_vector: np.ndarray) -> CameraCalibration:
    """Rotate a camera by a small world-frame-independent rotation about its own centre."""
    delta, _ = cv2.Rodrigues(np.asarray(rotation_vector, dtype=float).reshape(3, 1))
    rotation = _rotation(calibration)
    translation = np.asarray(calibration.translation_vector, dtype=float).reshape(3)
    centre = -rotation.T @ translation
    new_rotation = delta @ rotation
    new_translation = -new_rotation @ centre
    new_rvec, _ = cv2.Rodrigues(new_rotation)
    intrinsic = np.asarray(calibration.intrinsic_matrix, dtype=float)
    return replace(
        calibration,
        rotation_vector=new_rvec.reshape(3),
        translation_vector=new_translation,
        projection_matrix=intrinsic @ np.hstack([new_rotation, new_translation.reshape(3, 1)]),
        reprojection_error_px=None,
    )


def refine_relative_rotation(
    reference: CameraCalibration,
    moving: CameraCalibration,
    points_reference: np.ndarray,
    points_moving: np.ndarray,
    train_mask: np.ndarray,
) -> tuple[CameraCalibration, dict[str, Any]]:
    """Fit a small rotation of ``moving`` on the train points; report held-out points."""
    a = np.asarray(points_reference, dtype=float)
    b = np.asarray(points_moving, dtype=float)
    train = np.asarray(train_mask, dtype=bool)
    if train.shape != (len(a),) or not train.any() or train.all():
        raise ValueError("train_mask must select a non-empty proper subset of correspondences")
    result = least_squares(
        lambda params: signed_epipolar_distances(reference, rotate_about_centre(moving, params), a[train], b[train]),
        np.zeros(3),
        loss="soft_l1",
        f_scale=2.0,
    )
    refined = rotate_about_centre(moving, result.x)
    held_out = ~train
    report = {
        "algorithm": REFINEMENT_ALGORITHM,
        "reference_camera_id": reference.camera_id,
        "refined_camera_id": moving.camera_id,
        "rotation_about_camera_centre_rad": [float(value) for value in result.x],
        "rotation_about_camera_centre_deg": float(np.degrees(np.linalg.norm(result.x))),
        "solver_success": bool(result.success),
        "train_correspondence_count": int(train.sum()),
        "held_out_correspondence_count": int(held_out.sum()),
        "held_out_before": epipolar_summary(signed_epipolar_distances(reference, moving, a[held_out], b[held_out])),
        "held_out_after": epipolar_summary(signed_epipolar_distances(reference, refined, a[held_out], b[held_out])),
        "kept_unchanged": ["intrinsics", "camera_centres", "baseline_length", "reference_camera_pose", "world_axes"],
        "independent_accuracy_claim": False,
    }
    return refined, report


def calibration_consistency_report(
    poses_2d_by_frame: dict[int, dict[str, Any]],
    calibrations: dict[str, CameraCalibration],
    *,
    min_keypoint_score: float,
    max_abs_signed_median_px: float,
    min_correspondences: int,
) -> dict[str, Any]:
    """Per camera pair, the signed epipolar bias of high-confidence 2D keypoints.

    Points are undistorted to pixel coordinates first, so the check also applies
    to calibrations with lens distortion. A pair fails when the absolute signed
    median exceeds the limit; a pair with too few correspondences is reported as
    not evaluated and does not fail the run.
    """
    from src.triangulation import undistort_point

    camera_ids = sorted(calibrations)
    rectified = {
        camera_id: replace(calibration, distortion_coefficients=np.zeros(5))
        for camera_id, calibration in calibrations.items()
    }
    pairs = []
    for index, first in enumerate(camera_ids):
        for second in camera_ids[index + 1:]:
            a_points: list[np.ndarray] = []
            b_points: list[np.ndarray] = []
            for poses in poses_2d_by_frame.values():
                pa, pb = poses.get(first), poses.get(second)
                if pa is None or pb is None:
                    continue
                usable = (
                    np.asarray(pa.valid_mask, bool) & np.asarray(pb.valid_mask, bool)
                    & (np.asarray(pa.scores, float) >= min_keypoint_score)
                    & (np.asarray(pb.scores, float) >= min_keypoint_score)
                    & np.all(np.isfinite(pa.keypoints_xy), axis=1) & np.all(np.isfinite(pb.keypoints_xy), axis=1)
                )
                for joint in np.flatnonzero(usable):
                    a_points.append(undistort_point(pa.keypoints_xy[joint], calibrations[first], pixel_coordinates=True))
                    b_points.append(undistort_point(pb.keypoints_xy[joint], calibrations[second], pixel_coordinates=True))
            summary = epipolar_summary(
                signed_epipolar_distances(rectified[first], rectified[second], np.asarray(a_points), np.asarray(b_points))
                if a_points else np.empty(0)
            )
            evaluated = summary["count"] >= min_correspondences
            passed = (not evaluated) or abs(summary["signed_median_px"]) <= max_abs_signed_median_px
            pairs.append({"cameras": [first, second], "evaluated": evaluated, "passed": passed, **summary})
    return {
        "algorithm": "tk3d_two_view_signed_epipolar_bias_v1",
        "min_keypoint_score": min_keypoint_score,
        "max_abs_signed_epipolar_median_px": max_abs_signed_median_px,
        "min_correspondences": min_correspondences,
        "passed": all(pair["passed"] for pair in pairs),
        "pairs": pairs,
        "interpretation": (
            "A same-signed epipolar bias means the calibration no longer matches the recorded camera geometry. "
            "Passing is internal consistency, not independent 3D accuracy."
        ),
    }
