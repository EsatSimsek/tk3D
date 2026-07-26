from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .data_structures import (
    COCO_BODY_JOINT_NAMES,
    CameraCalibration,
    PersonPose2D,
)
from .triangulation import (
    normalized_projection_matrix,
    robust_triangulate_n_view,
    triangulation_quality_score,
    undistort_point,
)


PROVENANCE_ORIGINAL = np.uint8(0)
PROVENANCE_IMAGE_GUIDED = np.uint8(1)
PROVENANCE_CROSSVIEW_PROJECTED = np.uint8(2)


@dataclass(frozen=True, slots=True)
class CrossView2DFeedbackConfig:
    enabled: bool = True
    min_supporting_views: int = 4
    min_observation_score: float = 0.30
    trigger_error_px: float = 24.0
    search_radius_px: float = 42.0
    max_candidate_error_px: float = 20.0
    min_geometric_improvement_px: float = 12.0
    max_error_ratio: float = 0.50
    min_image_score: float = 0.30
    min_peak_score_ratio: float = 0.08
    max_hypotheses: int = 16
    max_reprojection_error_px: float = 25.0
    projected_fallback_for_visualization: bool = True


@dataclass(slots=True)
class CrossViewFeedbackPlan:
    frame_indices: np.ndarray
    camera_ids: tuple[str, ...]
    priors_xy: dict[str, np.ndarray]
    target_mask: dict[str, np.ndarray]
    initial_error_px: dict[str, np.ndarray]
    supporting_views: dict[str, np.ndarray]
    prior_score: dict[str, np.ndarray]
    report: dict[str, Any]

    @property
    def target_count(self) -> int:
        return int(sum(np.count_nonzero(mask) for mask in self.target_mask.values()))


@dataclass(frozen=True, slots=True)
class GuidedCandidateDecision:
    accepted: bool
    reason: str
    initial_error_px: float
    candidate_error_px: float | None
    improvement_px: float | None
    guided_score: float
    unconstrained_score: float


def feedback_config_from_mapping(raw: dict[str, Any] | None) -> CrossView2DFeedbackConfig:
    values = raw or {}
    if not isinstance(values, dict):
        raise ValueError("crossview_2d_feedback must be a mapping")
    config = CrossView2DFeedbackConfig(
        enabled=bool(values.get("enabled", True)),
        min_supporting_views=int(values.get("min_supporting_views", 4)),
        min_observation_score=float(values.get("min_observation_score", 0.30)),
        trigger_error_px=float(values.get("trigger_error_px", 24.0)),
        search_radius_px=float(values.get("search_radius_px", 42.0)),
        max_candidate_error_px=float(values.get("max_candidate_error_px", 20.0)),
        min_geometric_improvement_px=float(values.get("min_geometric_improvement_px", 12.0)),
        max_error_ratio=float(values.get("max_error_ratio", 0.50)),
        min_image_score=float(values.get("min_image_score", 0.30)),
        min_peak_score_ratio=float(values.get("min_peak_score_ratio", 0.08)),
        max_hypotheses=int(values.get("max_hypotheses", 16)),
        max_reprojection_error_px=float(values.get("max_reprojection_error_px", 25.0)),
        projected_fallback_for_visualization=bool(
            values.get("projected_fallback_for_visualization", True)
        ),
    )
    _validate_config(config)
    return config


def build_feedback_plan(
    frame_indices: np.ndarray,
    poses_2d_by_frame: dict[int, dict[str, PersonPose2D]],
    triangulated_source_3d: np.ndarray,
    calibrations: dict[str, CameraCalibration],
    config: CrossView2DFeedbackConfig,
) -> CrossViewFeedbackPlan:
    frame_ids = np.asarray(frame_indices, dtype=int)
    points = np.asarray(triangulated_source_3d, dtype=float)
    if points.ndim != 3 or points.shape[0] != frame_ids.size or points.shape[2] != 3:
        raise ValueError("triangulated_source_3d must have shape [frames, joints, 3]")
    body_count = min(len(COCO_BODY_JOINT_NAMES), points.shape[1])
    camera_ids = tuple(sorted(calibrations))
    shape = (frame_ids.size, body_count)
    priors = {camera_id: np.full((*shape, 2), np.nan, dtype=float) for camera_id in camera_ids}
    targets = {camera_id: np.zeros(shape, dtype=bool) for camera_id in camera_ids}
    initial_errors = {camera_id: np.full(shape, np.nan, dtype=float) for camera_id in camera_ids}
    supporting_views = {camera_id: np.zeros(shape, dtype=np.int16) for camera_id in camera_ids}
    prior_scores = {camera_id: np.zeros(shape, dtype=float) for camera_id in camera_ids}
    consensus_errors = {camera_id: np.full(shape, np.nan, dtype=float) for camera_id in camera_ids}
    observed_xy = {
        camera_id: np.full((*shape, 2), np.nan, dtype=float)
        for camera_id in camera_ids
    }
    consensus_xy = {
        camera_id: np.full((*shape, 2), np.nan, dtype=float)
        for camera_id in camera_ids
    }

    for sample_idx, frame_idx in enumerate(frame_ids):
        frame_poses = poses_2d_by_frame.get(int(frame_idx), {})
        for camera_id in camera_ids:
            pose = frame_poses.get(camera_id)
            calibration = calibrations[camera_id]
            for joint_idx in range(body_count):
                observed_valid = _pose_joint_is_usable(
                    pose,
                    joint_idx,
                    config.min_observation_score,
                )
                consensus_point = points[sample_idx, joint_idx]
                if observed_valid and np.all(np.isfinite(consensus_point)):
                    projected = project_source_point_distorted(consensus_point, calibration)
                    if np.all(np.isfinite(projected)):
                        observed_xy[camera_id][sample_idx, joint_idx] = (
                            pose.keypoints_xy[joint_idx]
                        )
                        consensus_xy[camera_id][sample_idx, joint_idx] = projected
                        consensus_errors[camera_id][sample_idx, joint_idx] = float(
                            np.linalg.norm(pose.keypoints_xy[joint_idx] - projected)
                        )
                consensus_error = consensus_errors[camera_id][sample_idx, joint_idx]
                consensus_available = np.all(np.isfinite(consensus_point))
                should_check = (
                    not observed_valid
                    or not consensus_available
                    or (np.isfinite(consensus_error) and consensus_error >= config.trigger_error_px)
                )
                if not config.enabled or not should_check:
                    continue
                leave_one_out = _triangulate_joint_excluding_camera(
                    frame_poses,
                    calibrations,
                    excluded_camera_id=camera_id,
                    joint_idx=joint_idx,
                    config=config,
                )
                if leave_one_out is None:
                    continue
                prior_point, support_count, prior_score = leave_one_out
                prior_xy = project_source_point_distorted(prior_point, calibration)
                if not np.all(np.isfinite(prior_xy)):
                    continue
                error = (
                    float(np.linalg.norm(pose.keypoints_xy[joint_idx] - prior_xy))
                    if observed_valid
                    else float("inf")
                )
                if observed_valid and error < config.trigger_error_px:
                    continue
                priors[camera_id][sample_idx, joint_idx] = prior_xy
                targets[camera_id][sample_idx, joint_idx] = True
                initial_errors[camera_id][sample_idx, joint_idx] = error
                supporting_views[camera_id][sample_idx, joint_idx] = support_count
                prior_scores[camera_id][sample_idx, joint_idx] = prior_score

    report = {
        "algorithm": "tk3d_leave_one_camera_out_heatmap_feedback_v1",
        "enabled": config.enabled,
        "independence_rule": (
            "Each prior is triangulated without the camera being corrected. "
            "Projected-only points are excluded from 3D triangulation."
        ),
        "body_joint_count": body_count,
        "target_count": int(sum(np.count_nonzero(value) for value in targets.values())),
        "config": {
            "min_supporting_views": config.min_supporting_views,
            "min_observation_score": config.min_observation_score,
            "trigger_error_px": config.trigger_error_px,
            "search_radius_px": config.search_radius_px,
            "max_candidate_error_px": config.max_candidate_error_px,
            "min_geometric_improvement_px": config.min_geometric_improvement_px,
            "max_error_ratio": config.max_error_ratio,
            "min_image_score": config.min_image_score,
            "min_peak_score_ratio": config.min_peak_score_ratio,
            "projected_fallback_for_visualization": config.projected_fallback_for_visualization,
        },
        "cameras": {
            camera_id: _camera_diagnostic(
                consensus_errors[camera_id],
                targets[camera_id],
                initial_errors[camera_id],
                config.trigger_error_px,
                observed_xy[camera_id],
                consensus_xy[camera_id],
            )
            for camera_id in camera_ids
        },
    }
    return CrossViewFeedbackPlan(
        frame_indices=frame_ids,
        camera_ids=camera_ids,
        priors_xy=priors,
        target_mask=targets,
        initial_error_px=initial_errors,
        supporting_views=supporting_views,
        prior_score=prior_scores,
        report=report,
    )


def decide_guided_candidate(
    observed_xy: np.ndarray,
    prior_xy: np.ndarray,
    guided_xy: np.ndarray,
    guided_score: float,
    unconstrained_score: float,
    config: CrossView2DFeedbackConfig,
) -> GuidedCandidateDecision:
    observed = np.asarray(observed_xy, dtype=float).reshape(2)
    prior = np.asarray(prior_xy, dtype=float).reshape(2)
    guided = np.asarray(guided_xy, dtype=float).reshape(2)
    initial_error = (
        float(np.linalg.norm(observed - prior))
        if np.all(np.isfinite(observed)) and np.all(np.isfinite(prior))
        else float("inf")
    )
    candidate_error = (
        float(np.linalg.norm(guided - prior))
        if np.all(np.isfinite(guided)) and np.all(np.isfinite(prior))
        else None
    )
    improvement = (
        initial_error - candidate_error
        if candidate_error is not None and np.isfinite(initial_error)
        else None
    )
    score = float(guided_score) if np.isfinite(guided_score) else 0.0
    global_score = float(unconstrained_score) if np.isfinite(unconstrained_score) else 0.0
    reason = "accepted"
    accepted = True
    if candidate_error is None:
        accepted, reason = False, "nonfinite_candidate"
    elif score < config.min_image_score:
        accepted, reason = False, "insufficient_image_score"
    elif global_score > 0.0 and score < global_score * config.min_peak_score_ratio:
        accepted, reason = False, "weak_secondary_heatmap_peak"
    elif candidate_error > config.max_candidate_error_px:
        accepted, reason = False, "candidate_too_far_from_crossview_prior"
    elif np.isfinite(initial_error):
        if improvement is None or improvement < config.min_geometric_improvement_px:
            accepted, reason = False, "insufficient_geometric_improvement"
        elif candidate_error > initial_error * config.max_error_ratio:
            accepted, reason = False, "insufficient_relative_improvement"
    return GuidedCandidateDecision(
        accepted=accepted,
        reason=reason,
        initial_error_px=initial_error,
        candidate_error_px=candidate_error,
        improvement_px=improvement,
        guided_score=score,
        unconstrained_score=global_score,
    )


def copy_pose(pose: PersonPose2D) -> PersonPose2D:
    return PersonPose2D(
        camera_id=pose.camera_id,
        frame_idx=pose.frame_idx,
        keypoints_xy=pose.keypoints_xy.copy(),
        scores=pose.scores.copy(),
        valid_mask=pose.valid_mask.copy(),
        person_id=pose.person_id,
    )


def project_source_point_distorted(
    point_3d_source: np.ndarray,
    calibration: CameraCalibration,
) -> np.ndarray:
    point = np.asarray(point_3d_source, dtype=float).reshape(3)
    if not np.all(np.isfinite(point)):
        return np.full(2, np.nan, dtype=float)
    rotation, _ = cv2.Rodrigues(
        np.asarray(calibration.rotation_vector, dtype=float).reshape(3, 1)
    )
    camera_point = rotation @ point + np.asarray(
        calibration.translation_vector,
        dtype=float,
    ).reshape(3)
    if not np.all(np.isfinite(camera_point)) or camera_point[2] <= 1e-9:
        return np.full(2, np.nan, dtype=float)
    projected, _ = cv2.projectPoints(
        point.reshape(1, 1, 3),
        np.asarray(calibration.rotation_vector, dtype=float).reshape(3, 1),
        np.asarray(calibration.translation_vector, dtype=float).reshape(3, 1),
        np.asarray(calibration.intrinsic_matrix, dtype=float),
        np.asarray(calibration.distortion_coefficients, dtype=float).reshape(-1),
    )
    return projected.reshape(2).astype(float)


def finalize_feedback_report(
    plan: CrossViewFeedbackPlan,
    provenance_by_camera: dict[str, np.ndarray],
    decisions: list[dict[str, Any]],
) -> dict[str, Any]:
    report = dict(plan.report)
    accepted = sum(item.get("accepted") is True for item in decisions)
    rejected = sum(item.get("accepted") is False for item in decisions)
    projected = sum(item.get("visualization_fallback") is True for item in decisions)
    reason_counts: dict[str, int] = {}
    for item in decisions:
        reason = str(item.get("reason", "unknown"))
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    report["image_guided_accepted_count"] = int(accepted)
    report["geometry_outlier_rejected_count"] = int(rejected)
    report["geometry_changed_count"] = int(accepted + rejected)
    report["crossview_projected_visualization_count"] = int(projected)
    report["decision_reason_counts"] = reason_counts
    report["provenance_codes"] = {
        str(int(PROVENANCE_ORIGINAL)): "original_observation",
        str(int(PROVENANCE_IMAGE_GUIDED)): "image_guided_heatmap_peak",
        str(int(PROVENANCE_CROSSVIEW_PROJECTED)): (
            "leave_one_camera_out_crossview_projection_visualization_only"
        ),
    }
    camera_reports = {key: dict(value) for key, value in report["cameras"].items()}
    for camera_id, provenance in provenance_by_camera.items():
        camera_reports[camera_id]["image_guided_accepted_count"] = int(
            np.count_nonzero(provenance == PROVENANCE_IMAGE_GUIDED)
        )
        camera_reports[camera_id]["crossview_projected_visualization_count"] = int(
            np.count_nonzero(provenance == PROVENANCE_CROSSVIEW_PROJECTED)
        )
    report["cameras"] = camera_reports
    report["decisions"] = decisions
    return report


def _triangulate_joint_excluding_camera(
    poses_by_camera: dict[str, PersonPose2D],
    calibrations: dict[str, CameraCalibration],
    excluded_camera_id: str,
    joint_idx: int,
    config: CrossView2DFeedbackConfig,
) -> tuple[np.ndarray, int, float] | None:
    pixel_points: list[np.ndarray] = []
    normalized_points: list[np.ndarray] = []
    pixel_projections: list[np.ndarray] = []
    normalized_projections: list[np.ndarray] = []
    scores: list[float] = []
    for camera_id, pose in poses_by_camera.items():
        if camera_id == excluded_camera_id or camera_id not in calibrations:
            continue
        if not _pose_joint_is_usable(pose, joint_idx, config.min_observation_score):
            continue
        calibration = calibrations[camera_id]
        raw_point = pose.keypoints_xy[joint_idx]
        pixel_point = undistort_point(raw_point, calibration, pixel_coordinates=True)
        normalized_point = undistort_point(raw_point, calibration, pixel_coordinates=False)
        if not np.all(np.isfinite(pixel_point)) or not np.all(np.isfinite(normalized_point)):
            continue
        pixel_points.append(pixel_point)
        normalized_points.append(normalized_point)
        pixel_projections.append(np.asarray(calibration.projection_matrix, dtype=float))
        normalized_projections.append(normalized_projection_matrix(calibration))
        scores.append(float(np.clip(pose.scores[joint_idx], 0.0, 1.0)))
    if len(scores) < config.min_supporting_views:
        return None
    result = robust_triangulate_n_view(
        normalized_points_2d=normalized_points,
        normalized_projection_mats=normalized_projections,
        pixel_points_2d=pixel_points,
        pixel_projection_mats=pixel_projections,
        scores=scores,
        min_views=config.min_supporting_views,
        max_reprojection_error_px=config.max_reprojection_error_px,
        max_hypotheses=config.max_hypotheses,
    )
    if result is None:
        return None
    point, conditioning, inliers, error = result
    quality = triangulation_quality_score(
        [scores[index] for index in inliers],
        error,
        len(inliers),
        conditioning,
    )
    return point, len(inliers), quality


def _pose_joint_is_usable(
    pose: PersonPose2D | None,
    joint_idx: int,
    min_score: float,
) -> bool:
    if pose is None or joint_idx >= pose.keypoints_xy.shape[0]:
        return False
    return bool(
        pose.valid_mask[joint_idx]
        and np.all(np.isfinite(pose.keypoints_xy[joint_idx]))
        and np.isfinite(pose.scores[joint_idx])
        and pose.scores[joint_idx] >= min_score
    )


def _camera_diagnostic(
    consensus_error_px: np.ndarray,
    target_mask: np.ndarray,
    leave_one_out_error_px: np.ndarray,
    trigger_error_px: float,
    observed_xy: np.ndarray,
    projected_xy: np.ndarray,
) -> dict[str, Any]:
    body_count = consensus_error_px.shape[1]
    joint_medians: dict[str, float | None] = {}
    bad_joints: list[str] = []
    for joint_idx in range(body_count):
        values = consensus_error_px[:, joint_idx]
        finite = values[np.isfinite(values)]
        median = float(np.median(finite)) if finite.size else None
        name = COCO_BODY_JOINT_NAMES[joint_idx]
        joint_medians[name] = median
        if median is not None and median >= trigger_error_px:
            bad_joints.append(name)
    arm_names = {
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
    }
    bad_arm_count = len(arm_names.intersection(bad_joints))
    bad_nonarm_count = len(set(bad_joints).difference(arm_names))
    target_count = int(np.count_nonzero(target_mask))
    hypotheses = _camera_hypothesis_metrics(observed_xy, projected_xy)
    direct_median = hypotheses["direct_error_px"]["median"]
    swapped_median = hypotheses["left_right_swapped_error_px"]["median"]
    translation_median = hypotheses["translation_corrected_error_px"]["median"]
    affine_median = hypotheses["partial_affine_corrected_error_px"]["median"]
    direct_p95 = hypotheses["direct_error_px"]["p95"]
    affine_p95 = hypotheses["partial_affine_corrected_error_px"]["p95"]
    affine_plausible = hypotheses["partial_affine_plausible"]
    best_shift = hypotheses["best_temporal_shift"]
    zero_shift_error = hypotheses["zero_temporal_shift_error_px"]
    if target_count == 0 and not bad_joints:
        classification = "healthy"
        evidence = "no leave-one-camera-out correction targets"
    elif target_count == 0:
        classification = "insufficient_crossview_support"
        evidence = (
            "large residuals exist, but too few independent views are available "
            "for a correction prior"
        )
    elif (
        direct_median is not None
        and swapped_median is not None
        and swapped_median < 0.70 * direct_median
    ):
        classification = "left_right_identity_mismatch"
        evidence = "swapping left/right labels explains most of the multi-view residual"
    elif (
        direct_median is not None
        and best_shift is not None
        and zero_shift_error is not None
        and best_shift["shift_frames"] != 0
        and best_shift["median_error_px"] < 0.75 * zero_shift_error
    ):
        classification = "synchronization_offset"
        evidence = "a non-zero temporal shift substantially reduces torso and leg residuals"
    elif (
        direct_median is not None
        and affine_median is not None
        and direct_p95 is not None
        and affine_p95 is not None
        and affine_plausible
        and affine_median < 0.60 * direct_median
        and affine_p95 < 0.75 * direct_p95
    ):
        classification = "calibration_alignment_bias"
        evidence = "a single robust image-plane alignment explains most residuals"
    elif bad_arm_count >= 2 and bad_nonarm_count <= 3:
        classification = "self_occlusion_or_2d_joint_detection"
        evidence = "large residuals are localized mainly to shoulders, elbows, or wrists"
    elif (
        len(bad_joints) >= max(8, body_count // 2)
        and direct_median is not None
        and translation_median is not None
        and translation_median > 0.75 * direct_median
    ):
        classification = "broad_2d_pose_failure_or_occlusion"
        evidence = (
            "most joints disagree, while left/right swap, time shift, translation, "
            "and affine calibration hypotheses do not explain the error"
        )
    else:
        classification = "localized_2d_joint_detection"
        evidence = "large residuals affect a limited joint subset"
    finite_consensus = consensus_error_px[np.isfinite(consensus_error_px)]
    finite_leave_one_out = leave_one_out_error_px[
        target_mask & np.isfinite(leave_one_out_error_px)
    ]
    return {
        "classification": classification,
        "evidence": evidence,
        "target_count": target_count,
        "target_ratio": float(target_count / target_mask.size) if target_mask.size else 0.0,
        "bad_joint_names": bad_joints,
        "joint_median_consensus_error_px": joint_medians,
        "consensus_error_px": _distribution(finite_consensus),
        "leave_one_out_target_error_px": _distribution(finite_leave_one_out),
        "diagnostic_hypotheses": hypotheses,
    }


def _camera_hypothesis_metrics(
    observed_xy: np.ndarray,
    projected_xy: np.ndarray,
) -> dict[str, Any]:
    observed = np.asarray(observed_xy, dtype=float)
    projected = np.asarray(projected_xy, dtype=float)
    valid = np.all(np.isfinite(observed), axis=-1) & np.all(
        np.isfinite(projected),
        axis=-1,
    )
    direct_error = np.linalg.norm(observed - projected, axis=-1)
    swap = np.asarray(
        [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15],
        dtype=int,
    )
    swapped_valid = np.all(np.isfinite(observed[:, swap]), axis=-1) & np.all(
        np.isfinite(projected),
        axis=-1,
    )
    swapped_error = np.linalg.norm(observed[:, swap] - projected, axis=-1)
    median_translation = (
        np.median((observed - projected)[valid], axis=0)
        if np.any(valid)
        else np.zeros(2, dtype=float)
    )
    translation_error = np.linalg.norm(
        observed - (projected + median_translation),
        axis=-1,
    )
    affine_error = np.empty(0, dtype=float)
    affine_values = None
    affine_scale = None
    affine_rotation_deg = None
    affine_translation_norm_px = None
    affine_plausible = False
    if np.count_nonzero(valid) >= 6:
        affine, _ = cv2.estimateAffinePartial2D(
            projected[valid].astype(np.float32),
            observed[valid].astype(np.float32),
            method=cv2.RANSAC,
            ransacReprojThreshold=15.0,
            maxIters=5000,
        )
        if affine is not None:
            aligned = cv2.transform(
                projected.reshape(-1, 1, 2).astype(np.float32),
                affine,
            ).reshape(projected.shape)
            aligned_error = np.linalg.norm(observed - aligned, axis=-1)
            affine_error = aligned_error[valid & np.isfinite(aligned_error)]
            affine_values = affine.tolist()
            affine_scale = float(np.hypot(affine[0, 0], affine[1, 0]))
            affine_rotation_deg = float(
                np.degrees(np.arctan2(affine[1, 0], affine[0, 0]))
            )
            affine_translation_norm_px = float(np.linalg.norm(affine[:, 2]))
            affine_plausible = bool(
                0.85 <= affine_scale <= 1.15
                and abs(affine_rotation_deg) <= 5.0
                and affine_translation_norm_px <= 150.0
            )
    shift_scan = _temporal_shift_metrics(observed, projected)
    best_shift = (
        min(shift_scan, key=lambda item: item["median_error_px"])
        if shift_scan
        else None
    )
    zero_shift = next(
        (item["median_error_px"] for item in shift_scan if item["shift_frames"] == 0),
        None,
    )
    return {
        "direct_error_px": _distribution(direct_error[valid]),
        "left_right_swapped_error_px": _distribution(
            swapped_error[swapped_valid]
        ),
        "median_translation_xy_px": median_translation.tolist(),
        "translation_corrected_error_px": _distribution(
            translation_error[valid]
        ),
        "partial_affine_xy": affine_values,
        "partial_affine_scale": affine_scale,
        "partial_affine_rotation_deg": affine_rotation_deg,
        "partial_affine_translation_norm_px": affine_translation_norm_px,
        "partial_affine_plausible": affine_plausible,
        "partial_affine_corrected_error_px": _distribution(affine_error),
        "temporal_shift_scan": shift_scan,
        "best_temporal_shift": best_shift,
        "zero_temporal_shift_error_px": zero_shift,
    }


def _temporal_shift_metrics(
    observed_xy: np.ndarray,
    projected_xy: np.ndarray,
) -> list[dict[str, float | int]]:
    frame_count = observed_xy.shape[0]
    max_shift = min(6, max(frame_count // 4, 0))
    joint_indices = [5, 6, 11, 12, 13, 14, 15, 16]
    output = []
    for shift in range(-max_shift, max_shift + 1):
        if shift >= 0:
            observed = observed_xy[shift:]
            projected = projected_xy[: frame_count - shift]
        else:
            observed = observed_xy[: frame_count + shift]
            projected = projected_xy[-shift:]
        valid = np.all(np.isfinite(observed), axis=-1) & np.all(
            np.isfinite(projected),
            axis=-1,
        )
        error = np.linalg.norm(
            observed[:, joint_indices] - projected[:, joint_indices],
            axis=-1,
        )
        values = error[valid[:, joint_indices]]
        if values.size:
            output.append(
                {
                    "shift_frames": shift,
                    "median_error_px": float(np.median(values)),
                    "p95_error_px": float(np.percentile(values, 95.0)),
                }
            )
    return output


def _distribution(values: np.ndarray) -> dict[str, float | int | None]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p95": float(np.percentile(finite, 95.0)),
        "max": float(np.max(finite)),
    }


def _validate_config(config: CrossView2DFeedbackConfig) -> None:
    if config.min_supporting_views < 2:
        raise ValueError("crossview_2d_feedback.min_supporting_views must be at least 2")
    if config.max_hypotheses < 1:
        raise ValueError("crossview_2d_feedback.max_hypotheses must be positive")
    for name, value in (
        ("trigger_error_px", config.trigger_error_px),
        ("search_radius_px", config.search_radius_px),
        ("max_candidate_error_px", config.max_candidate_error_px),
        ("min_geometric_improvement_px", config.min_geometric_improvement_px),
        ("max_reprojection_error_px", config.max_reprojection_error_px),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"crossview_2d_feedback.{name} must be finite and positive")
    for name, value in (
        ("min_observation_score", config.min_observation_score),
        ("max_error_ratio", config.max_error_ratio),
        ("min_image_score", config.min_image_score),
        ("min_peak_score_ratio", config.min_peak_score_ratio),
    ):
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"crossview_2d_feedback.{name} must be between 0 and 1")
