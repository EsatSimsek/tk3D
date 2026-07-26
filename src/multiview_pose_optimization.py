from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from .data_structures import (
    COCO_BODY_JOINTS,
    CameraCalibration,
    PersonPose2D,
)
from .triangulation import undistort_point


BODY_BONES: tuple[tuple[int, int], ...] = tuple(
    (COCO_BODY_JOINTS[first], COCO_BODY_JOINTS[second])
    for first, second in (
        ("left_shoulder", "right_shoulder"),
        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"),
        ("left_shoulder", "left_hip"),
        ("right_shoulder", "right_hip"),
        ("left_hip", "right_hip"),
        ("left_hip", "left_knee"),
        ("left_knee", "left_ankle"),
        ("right_hip", "right_knee"),
        ("right_knee", "right_ankle"),
    )
)

JOINT_ANGLE_LIMITS: tuple[tuple[int, int, int, float, float], ...] = tuple(
    (
        COCO_BODY_JOINTS[parent],
        COCO_BODY_JOINTS[joint],
        COCO_BODY_JOINTS[child],
        lower,
        upper,
    )
    for parent, joint, child, lower, upper in (
        ("left_shoulder", "left_elbow", "left_wrist", 5.0, 179.0),
        ("right_shoulder", "right_elbow", "right_wrist", 5.0, 179.0),
        ("left_hip", "left_knee", "left_ankle", 5.0, 179.0),
        ("right_hip", "right_knee", "right_ankle", 5.0, 179.0),
    )
)

PROVENANCE_UNAVAILABLE = np.uint8(0)
PROVENANCE_OBSERVED = np.uint8(1)
PROVENANCE_TEMPORALLY_RECOVERED = np.uint8(2)


@dataclass(frozen=True, slots=True)
class GlobalPoseOptimizationConfig:
    enabled: bool = True
    min_observation_score: float = 0.30
    max_gap_frames: int = 12
    minimum_bone_samples: int = 12
    outer_iterations: int = 3
    max_solver_evaluations: int = 35
    reprojection_scale_px: float = 8.0
    bone_scale_m: float = 0.025
    acceleration_scale_mps2: float = 25.0
    jerk_scale_mps3: float = 500.0
    angle_scale_deg: float = 10.0
    anchor_scale_m: float = 0.15
    reprojection_weight: float = 2.0
    bone_weight: float = 1.0
    acceleration_weight: float = 0.06
    jerk_weight: float = 0.01
    joint_limit_weight: float = 0.50
    anchor_weight: float = 0.03
    camera_weight_floor: float = 0.20
    camera_error_scale_px: float = 12.0
    camera_weight_update_alpha: float = 0.50
    temporal_speed_reference_mps: float = 4.0
    temporal_weight_floor: float = 0.15
    max_median_reprojection_degradation_ratio: float = 1.05
    max_p95_reprojection_degradation_ratio: float = 1.10
    max_median_reprojection_increase_px: float = 1.0
    max_p95_reprojection_increase_px: float = 2.0
    max_p95_acceleration_degradation_ratio: float = 1.10
    max_p95_acceleration_increase_mps2: float = 5.0
    max_p95_correction_m: float = 0.30


@dataclass(slots=True)
class GlobalPoseOptimizationResult:
    keypoints_3d: np.ndarray
    valid_mask: np.ndarray
    provenance: np.ndarray
    body_reprojection_error_px: np.ndarray
    camera_weights: dict[str, float]
    report: dict[str, Any]
    applied: bool


@dataclass(slots=True)
class _Observations:
    frame: np.ndarray
    joint: np.ndarray
    camera: np.ndarray
    xy: np.ndarray
    projection: np.ndarray
    base_weight: np.ndarray
    camera_ids: tuple[str, ...]


@dataclass(slots=True)
class _Problem:
    seed: np.ndarray
    observed: np.ndarray
    active: np.ndarray
    variable_start: np.ndarray
    observations: _Observations
    bone_frame: np.ndarray
    bone_first: np.ndarray
    bone_second: np.ndarray
    bone_length: np.ndarray
    acceleration_frame: np.ndarray
    acceleration_joint: np.ndarray
    acceleration_weight: np.ndarray
    jerk_frame: np.ndarray
    jerk_joint: np.ndarray
    jerk_weight: np.ndarray
    angle_frame: np.ndarray
    angle_parent: np.ndarray
    angle_joint: np.ndarray
    angle_child: np.ndarray
    angle_lower: np.ndarray
    angle_upper: np.ndarray
    anchor_frame: np.ndarray
    anchor_joint: np.ndarray
    anchor_weight: np.ndarray
    timestamps: np.ndarray
    analysis_to_source: np.ndarray
    config: GlobalPoseOptimizationConfig

    def unpack(self, values: np.ndarray) -> np.ndarray:
        points = self.seed.copy()
        starts = self.variable_start[self.active]
        points[self.active] = values[starts[:, None] + np.arange(3)]
        return points

    def residuals(self, values: np.ndarray, camera_weights: np.ndarray) -> np.ndarray:
        points = self.unpack(values)
        config = self.config
        residual_blocks: list[np.ndarray] = []

        if self.observations.frame.size:
            observed_points = points[self.observations.frame, self.observations.joint]
            projected = _project_analysis_points(
                observed_points,
                self.observations.projection,
                self.analysis_to_source,
            )
            weights = np.sqrt(
                np.maximum(
                    config.reprojection_weight
                    * self.observations.base_weight
                    * camera_weights[self.observations.camera],
                    1e-8,
                )
            )
            normalized_reprojection = (
                (projected - self.observations.xy)
                / config.reprojection_scale_px
                * weights[:, None]
            )
            residual_blocks.append(_soft_l1_residual(normalized_reprojection).reshape(-1))

        if self.bone_frame.size:
            difference = (
                points[self.bone_frame, self.bone_first]
                - points[self.bone_frame, self.bone_second]
            )
            length = np.linalg.norm(difference, axis=1)
            residual_blocks.append(
                np.sqrt(config.bone_weight)
                * (length - self.bone_length)
                / config.bone_scale_m
            )

        if self.acceleration_frame.size:
            frame = self.acceleration_frame
            joint = self.acceleration_joint
            dt_previous = self.timestamps[frame] - self.timestamps[frame - 1]
            dt_next = self.timestamps[frame + 1] - self.timestamps[frame]
            velocity_previous = (
                points[frame, joint] - points[frame - 1, joint]
            ) / dt_previous[:, None]
            velocity_next = (
                points[frame + 1, joint] - points[frame, joint]
            ) / dt_next[:, None]
            acceleration = 2.0 * (velocity_next - velocity_previous) / (
                dt_previous + dt_next
            )[:, None]
            residual_blocks.append(
                (
                    np.sqrt(config.acceleration_weight * self.acceleration_weight)[:, None]
                    * acceleration
                    / config.acceleration_scale_mps2
                ).reshape(-1)
            )

        if self.jerk_frame.size:
            frame = self.jerk_frame
            joint = self.jerk_joint
            acceleration_first = _three_point_acceleration(
                points[frame - 1, joint],
                points[frame, joint],
                points[frame + 1, joint],
                self.timestamps[frame - 1],
                self.timestamps[frame],
                self.timestamps[frame + 1],
            )
            acceleration_second = _three_point_acceleration(
                points[frame, joint],
                points[frame + 1, joint],
                points[frame + 2, joint],
                self.timestamps[frame],
                self.timestamps[frame + 1],
                self.timestamps[frame + 2],
            )
            span = np.maximum(self.timestamps[frame + 1] - self.timestamps[frame], 1e-6)
            jerk = (acceleration_second - acceleration_first) / span[:, None]
            residual_blocks.append(
                (
                    np.sqrt(config.jerk_weight * self.jerk_weight)[:, None]
                    * jerk
                    / config.jerk_scale_mps3
                ).reshape(-1)
            )

        if self.angle_frame.size:
            frame = self.angle_frame
            parent = points[frame, self.angle_parent]
            center = points[frame, self.angle_joint]
            child = points[frame, self.angle_child]
            angles = _joint_angles_deg(parent, center, child)
            lower_error = np.maximum(self.angle_lower - angles, 0.0)
            upper_error = np.maximum(angles - self.angle_upper, 0.0)
            residual_blocks.append(
                (
                    np.sqrt(config.joint_limit_weight)
                    * np.column_stack([lower_error, upper_error])
                    / config.angle_scale_deg
                ).reshape(-1)
            )

        if self.anchor_frame.size:
            correction = (
                points[self.anchor_frame, self.anchor_joint]
                - self.seed[self.anchor_frame, self.anchor_joint]
            )
            residual_blocks.append(
                (
                    np.sqrt(config.anchor_weight * self.anchor_weight)[:, None]
                    * correction
                    / config.anchor_scale_m
                ).reshape(-1)
            )

        if not residual_blocks:
            return np.empty(0, dtype=float)
        residuals = np.concatenate(residual_blocks)
        return np.nan_to_num(residuals, nan=1e3, posinf=1e3, neginf=-1e3)

    def jacobian_sparsity(self):
        observation_count = self.observations.frame.size
        bone_count = self.bone_frame.size
        acceleration_count = self.acceleration_frame.size
        jerk_count = self.jerk_frame.size
        angle_count = self.angle_frame.size
        anchor_count = self.anchor_frame.size
        row_count = (
            observation_count * 2
            + bone_count
            + acceleration_count * 3
            + jerk_count * 3
            + angle_count * 2
            + anchor_count * 3
        )
        matrix = lil_matrix((row_count, int(np.count_nonzero(self.active)) * 3), dtype=np.int8)
        row = 0
        row = _mark_dependencies(
            matrix,
            row,
            self.variable_start,
            self.observations.frame,
            (self.observations.joint,),
            residuals_per_item=2,
        )
        row = _mark_dependencies(
            matrix,
            row,
            self.variable_start,
            self.bone_frame,
            (self.bone_first, self.bone_second),
            residuals_per_item=1,
        )
        row = _mark_temporal_dependencies(
            matrix,
            row,
            self.variable_start,
            self.acceleration_frame,
            self.acceleration_joint,
            offsets=(-1, 0, 1),
            residuals_per_item=3,
        )
        row = _mark_temporal_dependencies(
            matrix,
            row,
            self.variable_start,
            self.jerk_frame,
            self.jerk_joint,
            offsets=(-1, 0, 1, 2),
            residuals_per_item=3,
        )
        row = _mark_dependencies(
            matrix,
            row,
            self.variable_start,
            self.angle_frame,
            (self.angle_parent, self.angle_joint, self.angle_child),
            residuals_per_item=2,
        )
        _mark_dependencies(
            matrix,
            row,
            self.variable_start,
            self.anchor_frame,
            (self.anchor_joint,),
            residuals_per_item=3,
        )
        return matrix.tocsr()


def optimization_config_from_mapping(mapping: dict[str, Any] | None) -> GlobalPoseOptimizationConfig:
    raw = mapping or {}
    weights = raw.get("weights", {})
    return GlobalPoseOptimizationConfig(
        enabled=bool(raw.get("enabled", True)),
        min_observation_score=float(raw.get("min_observation_score", 0.30)),
        max_gap_frames=int(raw.get("max_gap_frames", 12)),
        minimum_bone_samples=int(raw.get("minimum_bone_samples", 12)),
        outer_iterations=int(raw.get("outer_iterations", 3)),
        max_solver_evaluations=int(raw.get("max_solver_evaluations", 35)),
        reprojection_scale_px=float(raw.get("reprojection_scale_px", 8.0)),
        bone_scale_m=float(raw.get("bone_scale_m", 0.025)),
        acceleration_scale_mps2=float(raw.get("acceleration_scale_mps2", 25.0)),
        jerk_scale_mps3=float(raw.get("jerk_scale_mps3", 500.0)),
        angle_scale_deg=float(raw.get("angle_scale_deg", 10.0)),
        anchor_scale_m=float(raw.get("anchor_scale_m", 0.15)),
        reprojection_weight=float(weights.get("reprojection", 2.0)),
        bone_weight=float(weights.get("bone", 1.0)),
        acceleration_weight=float(weights.get("acceleration", 0.06)),
        jerk_weight=float(weights.get("jerk", 0.01)),
        joint_limit_weight=float(weights.get("joint_limits", 0.50)),
        anchor_weight=float(weights.get("anchor", 0.03)),
        camera_weight_floor=float(raw.get("camera_weight_floor", 0.20)),
        camera_error_scale_px=float(raw.get("camera_error_scale_px", 12.0)),
        camera_weight_update_alpha=float(raw.get("camera_weight_update_alpha", 0.50)),
        temporal_speed_reference_mps=float(raw.get("temporal_speed_reference_mps", 4.0)),
        temporal_weight_floor=float(raw.get("temporal_weight_floor", 0.15)),
        max_median_reprojection_degradation_ratio=float(
            raw.get("max_median_reprojection_degradation_ratio", 1.05)
        ),
        max_p95_reprojection_degradation_ratio=float(
            raw.get("max_p95_reprojection_degradation_ratio", 1.10)
        ),
        max_median_reprojection_increase_px=float(
            raw.get("max_median_reprojection_increase_px", 1.0)
        ),
        max_p95_reprojection_increase_px=float(
            raw.get("max_p95_reprojection_increase_px", 2.0)
        ),
        max_p95_acceleration_degradation_ratio=float(
            raw.get("max_p95_acceleration_degradation_ratio", 1.10)
        ),
        max_p95_acceleration_increase_mps2=float(
            raw.get("max_p95_acceleration_increase_mps2", 5.0)
        ),
        max_p95_correction_m=float(raw.get("max_p95_correction_m", 0.30)),
    )


def optimize_body_sequence(
    initial_keypoints_3d: np.ndarray,
    base_valid_mask: np.ndarray,
    confidence: np.ndarray,
    frame_indices: np.ndarray,
    timestamps_sec: np.ndarray,
    poses_2d_by_frame: dict[int, dict[str, PersonPose2D]],
    calibrations: dict[str, CameraCalibration],
    analysis_to_source: np.ndarray,
    config: GlobalPoseOptimizationConfig,
) -> GlobalPoseOptimizationResult:
    points = np.asarray(initial_keypoints_3d, dtype=float)
    valid = np.asarray(base_valid_mask, dtype=bool)
    scores = np.asarray(confidence, dtype=float)
    frame_ids = np.asarray(frame_indices, dtype=int)
    timestamps = np.asarray(timestamps_sec, dtype=float)
    transform = np.asarray(analysis_to_source, dtype=float)
    _validate_inputs(points, valid, scores, frame_ids, timestamps, transform, config)

    body_count = min(17, points.shape[1])
    output = points.copy()
    output_valid = valid.copy()
    provenance = np.full(valid.shape, PROVENANCE_UNAVAILABLE, dtype=np.uint8)
    provenance[valid] = PROVENANCE_OBSERVED
    empty_error = np.full((points.shape[0], body_count), np.nan, dtype=float)
    if not config.enabled:
        return _disabled_result(output, output_valid, provenance, empty_error, "disabled_by_config")
    if points.shape[0] < 4:
        return _disabled_result(output, output_valid, provenance, empty_error, "fewer_than_four_frames")

    body_seed, body_observed, body_active, body_provenance = _seed_short_gaps(
        points[:, :body_count],
        valid[:, :body_count],
        config.max_gap_frames,
    )
    observations = _collect_observations(
        frame_ids,
        poses_2d_by_frame,
        calibrations,
        body_active,
        body_count,
        config.min_observation_score,
    )
    if observations.frame.size < max(12, int(np.count_nonzero(body_active))):
        return _disabled_result(
            output,
            output_valid,
            provenance,
            empty_error,
            "insufficient_multiview_observations",
        )

    reference_lengths = _reference_bone_lengths(
        body_seed,
        body_observed,
        scores[:, :body_count],
        config.minimum_bone_samples,
    )
    problem = _build_problem(
        body_seed,
        body_observed,
        body_active,
        scores[:, :body_count],
        timestamps,
        observations,
        reference_lengths,
        transform,
        config,
    )
    if not np.any(problem.active):
        return _disabled_result(output, output_valid, provenance, empty_error, "no_active_body_points")

    starts = problem.variable_start[problem.active]
    x = np.empty(int(np.count_nonzero(problem.active)) * 3, dtype=float)
    x[starts[:, None] + np.arange(3)] = problem.seed[problem.active]
    sparsity = problem.jacobian_sparsity()
    camera_weights = _initial_camera_weights(problem, x)
    iteration_reports: list[dict[str, Any]] = []
    solver_success = False
    solver_message = ""
    total_evaluations = 0

    try:
        for iteration in range(config.outer_iterations):
            result = least_squares(
                lambda values: problem.residuals(values, camera_weights),
                x,
                jac_sparsity=sparsity,
                method="trf",
                tr_solver="lsmr",
                loss="linear",
                x_scale="jac",
                max_nfev=config.max_solver_evaluations,
            )
            x = result.x
            total_evaluations += int(result.nfev)
            solver_success = bool(result.success or np.isfinite(result.cost))
            solver_message = str(result.message)
            updated_weights, camera_errors = _update_camera_weights(problem, x, camera_weights)
            iteration_reports.append(
                {
                    "iteration": iteration + 1,
                    "cost": float(result.cost),
                    "optimality": float(result.optimality),
                    "function_evaluations": int(result.nfev),
                    "camera_median_reprojection_error_px": camera_errors,
                    "camera_weights": {
                        camera_id: float(updated_weights[index])
                        for index, camera_id in enumerate(observations.camera_ids)
                    },
                }
            )
            if np.max(np.abs(updated_weights - camera_weights)) < 0.01:
                camera_weights = updated_weights
                break
            camera_weights = updated_weights
    except (FloatingPointError, ValueError, np.linalg.LinAlgError) as exc:
        return _disabled_result(
            output,
            output_valid,
            provenance,
            empty_error,
            f"solver_error:{type(exc).__name__}",
            extra={"message": str(exc)},
        )

    optimized_body = problem.unpack(x)
    before_errors, before_by_observation = _observation_errors(problem, problem.seed)
    after_errors, after_by_observation = _observation_errors(problem, optimized_body)
    correction = np.linalg.norm(optimized_body[body_observed] - body_seed[body_observed], axis=1)
    before_metrics = _quality_metrics(problem.seed, body_observed, before_errors, reference_lengths, timestamps)
    after_metrics = _quality_metrics(optimized_body, body_active, after_errors, reference_lengths, timestamps)
    gate = _acceptance_gate(before_metrics, after_metrics, correction, solver_success, config)

    report = {
        "algorithm": "tk3d_global_body17_bundle_adjustment_v1",
        "enabled": True,
        "applied": bool(gate["passed"]),
        "fallback_used": not bool(gate["passed"]),
        "fallback_reason": None if gate["passed"] else gate["reason"],
        "body_joint_count": body_count,
        "active_body_point_count": int(np.count_nonzero(body_active)),
        "observed_body_point_count": int(np.count_nonzero(body_observed)),
        "temporally_recovered_body_point_count": int(
            np.count_nonzero(body_provenance == PROVENANCE_TEMPORALLY_RECOVERED)
        ),
        "observation_count": int(observations.frame.size),
        "solver_success": solver_success,
        "solver_message": solver_message,
        "solver_function_evaluations": total_evaluations,
        "outer_iterations_completed": len(iteration_reports),
        "camera_weights": {
            camera_id: float(camera_weights[index])
            for index, camera_id in enumerate(observations.camera_ids)
        },
        "reference_bone_lengths_m": {
            f"{first}-{second}": float(length)
            for (first, second), length in reference_lengths.items()
        },
        "before": before_metrics,
        "after": after_metrics,
        "correction_m": _distribution(correction),
        "acceptance_gate": gate,
        "iterations": iteration_reports,
        "provenance_codes": {
            "0": "unavailable",
            "1": "observed",
            "2": "temporally_recovered_short_gap",
        },
    }
    if not gate["passed"]:
        return GlobalPoseOptimizationResult(
            output,
            output_valid,
            provenance,
            _mean_error_by_body_point(problem, before_by_observation),
            report["camera_weights"],
            report,
            False,
        )

    output[:, :body_count] = optimized_body
    output_valid[:, :body_count] = body_active
    provenance[:, :body_count] = body_provenance
    body_errors = _mean_error_by_body_point(problem, after_by_observation)
    return GlobalPoseOptimizationResult(
        output,
        output_valid,
        provenance,
        body_errors,
        report["camera_weights"],
        report,
        True,
    )


def _validate_inputs(
    points: np.ndarray,
    valid: np.ndarray,
    scores: np.ndarray,
    frame_ids: np.ndarray,
    timestamps: np.ndarray,
    transform: np.ndarray,
    config: GlobalPoseOptimizationConfig,
) -> None:
    if points.ndim != 3 or points.shape[-1] != 3 or points.shape[1] < 17:
        raise ValueError(f"initial_keypoints_3d must have shape [frames, >=17, 3], got {points.shape}")
    if valid.shape != points.shape[:2] or scores.shape != points.shape[:2]:
        raise ValueError("base_valid_mask and confidence must match keypoint frame/joint dimensions")
    if frame_ids.shape != (points.shape[0],) or timestamps.shape != (points.shape[0],):
        raise ValueError("frame_indices and timestamps_sec must match the frame count")
    if np.any(np.diff(frame_ids) <= 0) or np.any(np.diff(timestamps) <= 0.0):
        raise ValueError("frame indices and timestamps must be strictly increasing")
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("analysis_to_source must be a finite 4x4 matrix")
    positive = (
        config.max_gap_frames,
        config.minimum_bone_samples,
        config.outer_iterations,
        config.max_solver_evaluations,
    )
    if any(value < 1 for value in positive):
        raise ValueError("global optimization integer limits must be positive")


def _disabled_result(
    points: np.ndarray,
    valid: np.ndarray,
    provenance: np.ndarray,
    body_error: np.ndarray,
    reason: str,
    extra: dict[str, Any] | None = None,
) -> GlobalPoseOptimizationResult:
    report = {
        "algorithm": "tk3d_global_body17_bundle_adjustment_v1",
        "enabled": reason != "disabled_by_config",
        "applied": False,
        "fallback_used": True,
        "fallback_reason": reason,
    }
    if extra:
        report.update(extra)
    return GlobalPoseOptimizationResult(points, valid, provenance, body_error, {}, report, False)


def _seed_short_gaps(
    points: np.ndarray,
    valid_mask: np.ndarray,
    max_gap_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    seed = np.asarray(points, dtype=float).copy()
    observed = np.asarray(valid_mask, dtype=bool) & np.all(np.isfinite(seed), axis=-1)
    active = observed.copy()
    provenance = np.full(observed.shape, PROVENANCE_UNAVAILABLE, dtype=np.uint8)
    provenance[observed] = PROVENANCE_OBSERVED
    for joint in range(seed.shape[1]):
        valid_indices = np.flatnonzero(observed[:, joint])
        for first, second in zip(valid_indices[:-1], valid_indices[1:], strict=True):
            gap = int(second - first - 1)
            if gap < 1 or gap > max_gap_frames:
                continue
            for frame in range(first + 1, second):
                alpha = (frame - first) / (second - first)
                seed[frame, joint] = (1.0 - alpha) * seed[first, joint] + alpha * seed[second, joint]
                active[frame, joint] = True
                provenance[frame, joint] = PROVENANCE_TEMPORALLY_RECOVERED
    return seed, observed, active, provenance


def _collect_observations(
    frame_indices: np.ndarray,
    poses_2d_by_frame: dict[int, dict[str, PersonPose2D]],
    calibrations: dict[str, CameraCalibration],
    active: np.ndarray,
    body_count: int,
    min_score: float,
) -> _Observations:
    camera_ids = tuple(sorted(calibrations))
    camera_lookup = {camera_id: index for index, camera_id in enumerate(camera_ids)}
    frames: list[int] = []
    joints: list[int] = []
    cameras: list[int] = []
    xy: list[np.ndarray] = []
    projections: list[np.ndarray] = []
    weights: list[float] = []
    for array_frame, source_frame in enumerate(frame_indices):
        poses = poses_2d_by_frame.get(int(source_frame), {})
        for camera_id, pose in poses.items():
            calibration = calibrations.get(camera_id)
            if calibration is None:
                continue
            camera_index = camera_lookup[camera_id]
            for joint in range(body_count):
                score = float(pose.scores[joint])
                if (
                    not active[array_frame, joint]
                    or not pose.valid_mask[joint]
                    or not np.isfinite(score)
                    or score < min_score
                ):
                    continue
                point = undistort_point(pose.keypoints_xy[joint], calibration, pixel_coordinates=True)
                if not np.all(np.isfinite(point)):
                    continue
                frames.append(array_frame)
                joints.append(joint)
                cameras.append(camera_index)
                xy.append(point)
                projections.append(np.asarray(calibration.projection_matrix, dtype=float))
                weights.append(float(np.clip(score, 0.0, 1.0)) ** 2)
    return _Observations(
        frame=np.asarray(frames, dtype=int),
        joint=np.asarray(joints, dtype=int),
        camera=np.asarray(cameras, dtype=int),
        xy=np.asarray(xy, dtype=float).reshape(-1, 2),
        projection=np.asarray(projections, dtype=float).reshape(-1, 3, 4),
        base_weight=np.asarray(weights, dtype=float),
        camera_ids=camera_ids,
    )


def _reference_bone_lengths(
    points: np.ndarray,
    observed: np.ndarray,
    confidence: np.ndarray,
    minimum_samples: int,
) -> dict[tuple[int, int], float]:
    references: dict[tuple[int, int], float] = {}
    for first, second in BODY_BONES:
        usable = observed[:, first] & observed[:, second]
        lengths = np.linalg.norm(points[:, first] - points[:, second], axis=1)
        weights = np.minimum(confidence[:, first], confidence[:, second])
        mask = usable & np.isfinite(lengths) & np.isfinite(weights) & (weights > 0.0)
        if np.count_nonzero(mask) < minimum_samples:
            continue
        references[(first, second)] = _weighted_median(lengths[mask], weights[mask])
    return references


def _build_problem(
    seed: np.ndarray,
    observed: np.ndarray,
    active: np.ndarray,
    confidence: np.ndarray,
    timestamps: np.ndarray,
    observations: _Observations,
    reference_lengths: dict[tuple[int, int], float],
    analysis_to_source: np.ndarray,
    config: GlobalPoseOptimizationConfig,
) -> _Problem:
    variable_start = np.full(active.shape, -1, dtype=int)
    variable_start[active] = np.arange(np.count_nonzero(active), dtype=int) * 3

    bone_frame: list[int] = []
    bone_first: list[int] = []
    bone_second: list[int] = []
    bone_length: list[float] = []
    for (first, second), reference in reference_lengths.items():
        frames = np.flatnonzero(active[:, first] & active[:, second])
        bone_frame.extend(frames.tolist())
        bone_first.extend([first] * frames.size)
        bone_second.extend([second] * frames.size)
        bone_length.extend([reference] * frames.size)

    acceleration_frame: list[int] = []
    acceleration_joint: list[int] = []
    acceleration_weight: list[float] = []
    jerk_frame: list[int] = []
    jerk_joint: list[int] = []
    jerk_weight: list[float] = []
    for joint in range(active.shape[1]):
        for frame in range(1, active.shape[0] - 1):
            if not np.all(active[frame - 1 : frame + 2, joint]):
                continue
            acceleration_frame.append(frame)
            acceleration_joint.append(joint)
            speed = _local_speed(seed[:, joint], timestamps, frame)
            acceleration_weight.append(_motion_weight(speed, config))
        for frame in range(1, active.shape[0] - 2):
            if not np.all(active[frame - 1 : frame + 3, joint]):
                continue
            jerk_frame.append(frame)
            jerk_joint.append(joint)
            speed = max(
                _local_speed(seed[:, joint], timestamps, frame),
                _local_speed(seed[:, joint], timestamps, frame + 1),
            )
            jerk_weight.append(_motion_weight(speed, config))

    angle_frame: list[int] = []
    angle_parent: list[int] = []
    angle_joint: list[int] = []
    angle_child: list[int] = []
    angle_lower: list[float] = []
    angle_upper: list[float] = []
    for parent, joint, child, lower, upper in JOINT_ANGLE_LIMITS:
        frames = np.flatnonzero(active[:, parent] & active[:, joint] & active[:, child])
        angle_frame.extend(frames.tolist())
        angle_parent.extend([parent] * frames.size)
        angle_joint.extend([joint] * frames.size)
        angle_child.extend([child] * frames.size)
        angle_lower.extend([lower] * frames.size)
        angle_upper.extend([upper] * frames.size)

    anchor_frame, anchor_joint = np.nonzero(observed)
    anchor_confidence = np.clip(confidence[anchor_frame, anchor_joint], 0.05, 1.0)
    return _Problem(
        seed=seed,
        observed=observed,
        active=active,
        variable_start=variable_start,
        observations=observations,
        bone_frame=np.asarray(bone_frame, dtype=int),
        bone_first=np.asarray(bone_first, dtype=int),
        bone_second=np.asarray(bone_second, dtype=int),
        bone_length=np.asarray(bone_length, dtype=float),
        acceleration_frame=np.asarray(acceleration_frame, dtype=int),
        acceleration_joint=np.asarray(acceleration_joint, dtype=int),
        acceleration_weight=np.asarray(acceleration_weight, dtype=float),
        jerk_frame=np.asarray(jerk_frame, dtype=int),
        jerk_joint=np.asarray(jerk_joint, dtype=int),
        jerk_weight=np.asarray(jerk_weight, dtype=float),
        angle_frame=np.asarray(angle_frame, dtype=int),
        angle_parent=np.asarray(angle_parent, dtype=int),
        angle_joint=np.asarray(angle_joint, dtype=int),
        angle_child=np.asarray(angle_child, dtype=int),
        angle_lower=np.asarray(angle_lower, dtype=float),
        angle_upper=np.asarray(angle_upper, dtype=float),
        anchor_frame=anchor_frame.astype(int),
        anchor_joint=anchor_joint.astype(int),
        anchor_weight=anchor_confidence,
        timestamps=timestamps,
        analysis_to_source=analysis_to_source,
        config=config,
    )


def _project_analysis_points(
    points: np.ndarray,
    projections: np.ndarray,
    analysis_to_source: np.ndarray,
) -> np.ndarray:
    homogeneous = np.column_stack([points, np.ones(points.shape[0], dtype=float)])
    source_homogeneous = homogeneous @ analysis_to_source.T
    safe_w = np.where(
        np.abs(source_homogeneous[:, 3]) > 1e-12,
        source_homogeneous[:, 3],
        np.copysign(1e-12, source_homogeneous[:, 3] + 1e-12),
    )
    source = source_homogeneous[:, :3] / safe_w[:, None]
    projected_homogeneous = np.einsum(
        "nij,nj->ni",
        projections,
        np.column_stack([source, np.ones(source.shape[0], dtype=float)]),
    )
    depth = projected_homogeneous[:, 2]
    safe_depth = np.where(np.abs(depth) > 1e-8, depth, np.copysign(1e-8, depth + 1e-8))
    return projected_homogeneous[:, :2] / safe_depth[:, None]


def _initial_camera_weights(problem: _Problem, values: np.ndarray) -> np.ndarray:
    count = len(problem.observations.camera_ids)
    if count == 0:
        return np.empty(0, dtype=float)
    weights = np.ones(count, dtype=float)
    updated, _ = _update_camera_weights(problem, values, weights, blend=False)
    return updated


def _update_camera_weights(
    problem: _Problem,
    values: np.ndarray,
    current: np.ndarray,
    *,
    blend: bool = True,
) -> tuple[np.ndarray, dict[str, float | None]]:
    points = problem.unpack(values)
    _, observation_errors = _observation_errors(problem, points)
    raw = np.ones_like(current)
    medians: dict[str, float | None] = {}
    for camera_index, camera_id in enumerate(problem.observations.camera_ids):
        errors = observation_errors[problem.observations.camera == camera_index]
        finite = errors[np.isfinite(errors)]
        median = float(np.median(finite)) if finite.size else None
        medians[camera_id] = median
        if median is None:
            raw[camera_index] = problem.config.camera_weight_floor
        else:
            raw[camera_index] = np.exp(-median / problem.config.camera_error_scale_px)
    maximum = float(np.max(raw)) if raw.size else 1.0
    if maximum > 1e-12:
        raw /= maximum
    raw = np.clip(raw, problem.config.camera_weight_floor, 1.0)
    if not blend:
        return raw, medians
    alpha = problem.config.camera_weight_update_alpha
    return np.clip((1.0 - alpha) * current + alpha * raw, problem.config.camera_weight_floor, 1.0), medians


def _observation_errors(problem: _Problem, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    observed_points = points[problem.observations.frame, problem.observations.joint]
    projected = _project_analysis_points(
        observed_points,
        problem.observations.projection,
        problem.analysis_to_source,
    )
    errors = np.linalg.norm(projected - problem.observations.xy, axis=1)
    return errors[np.isfinite(errors)], errors


def _mean_error_by_body_point(problem: _Problem, observation_errors: np.ndarray) -> np.ndarray:
    output = np.full(problem.active.shape, np.nan, dtype=float)
    sums = np.zeros(problem.active.shape, dtype=float)
    counts = np.zeros(problem.active.shape, dtype=int)
    finite = np.isfinite(observation_errors)
    np.add.at(
        sums,
        (problem.observations.frame[finite], problem.observations.joint[finite]),
        observation_errors[finite],
    )
    np.add.at(
        counts,
        (problem.observations.frame[finite], problem.observations.joint[finite]),
        1,
    )
    np.divide(sums, counts, out=output, where=counts > 0)
    return output


def _quality_metrics(
    points: np.ndarray,
    valid: np.ndarray,
    reprojection_errors: np.ndarray,
    references: dict[tuple[int, int], float],
    timestamps: np.ndarray,
) -> dict[str, Any]:
    bone_deviations: list[float] = []
    bone_cvs: list[float] = []
    for (first, second), reference in references.items():
        mask = valid[:, first] & valid[:, second]
        lengths = np.linalg.norm(points[:, first] - points[:, second], axis=1)
        usable = lengths[mask & np.isfinite(lengths)]
        if usable.size:
            bone_deviations.extend(np.abs(usable - reference).tolist())
        if usable.size >= 2 and float(np.mean(usable)) > 1e-9:
            bone_cvs.append(float(np.std(usable) / np.mean(usable) * 100.0))
    accelerations: list[float] = []
    for joint in range(points.shape[1]):
        for frame in range(1, points.shape[0] - 1):
            if not np.all(valid[frame - 1 : frame + 2, joint]):
                continue
            acceleration = _three_point_acceleration(
                points[frame - 1, joint][None, :],
                points[frame, joint][None, :],
                points[frame + 1, joint][None, :],
                timestamps[frame - 1 : frame],
                timestamps[frame : frame + 1],
                timestamps[frame + 1 : frame + 2],
            )[0]
            accelerations.append(float(np.linalg.norm(acceleration)))
    joint_limit_excess: list[float] = []
    for parent, joint, child, lower, upper in JOINT_ANGLE_LIMITS:
        mask = valid[:, parent] & valid[:, joint] & valid[:, child]
        if not np.any(mask):
            continue
        angles = _joint_angles_deg(points[mask, parent], points[mask, joint], points[mask, child])
        excess = np.maximum(lower - angles, 0.0) + np.maximum(angles - upper, 0.0)
        joint_limit_excess.extend(excess.tolist())
    limit_values = np.asarray(joint_limit_excess, dtype=float)
    return {
        "reprojection_error_px": _distribution(reprojection_errors),
        "bone_absolute_deviation_m": _distribution(np.asarray(bone_deviations, dtype=float)),
        "mean_bone_length_cv_percent": float(np.mean(bone_cvs)) if bone_cvs else None,
        "acceleration_mps2": _distribution(np.asarray(accelerations, dtype=float)),
        "joint_limit_violation_count": int(np.count_nonzero(limit_values > 1e-6)),
        "joint_limit_max_excess_deg": (
            float(np.max(limit_values)) if limit_values.size else None
        ),
        "valid_body_ratio": float(np.mean(valid)) if valid.size else 0.0,
    }


def _acceptance_gate(
    before: dict[str, Any],
    after: dict[str, Any],
    correction: np.ndarray,
    solver_success: bool,
    config: GlobalPoseOptimizationConfig,
) -> dict[str, Any]:
    failures: list[str] = []
    before_median = before["reprojection_error_px"]["median"]
    after_median = after["reprojection_error_px"]["median"]
    before_p95 = before["reprojection_error_px"]["p95"]
    after_p95 = after["reprojection_error_px"]["p95"]
    before_acceleration_p95 = before["acceleration_mps2"]["p95"]
    after_acceleration_p95 = after["acceleration_mps2"]["p95"]
    correction_p95 = _distribution(correction)["p95"]
    if not solver_success:
        failures.append("solver_failed")
    if before_median is not None and after_median is not None:
        limit = max(
            before_median * config.max_median_reprojection_degradation_ratio,
            before_median + config.max_median_reprojection_increase_px,
        )
        if after_median > limit:
            failures.append("median_reprojection_degraded")
    if before_p95 is not None and after_p95 is not None:
        limit = max(
            before_p95 * config.max_p95_reprojection_degradation_ratio,
            before_p95 + config.max_p95_reprojection_increase_px,
        )
        if after_p95 > limit:
            failures.append("p95_reprojection_degraded")
    if correction_p95 is not None and correction_p95 > config.max_p95_correction_m:
        failures.append("body_correction_too_large")
    if before_acceleration_p95 is not None and after_acceleration_p95 is not None:
        limit = max(
            before_acceleration_p95 * config.max_p95_acceleration_degradation_ratio,
            before_acceleration_p95 + config.max_p95_acceleration_increase_mps2,
        )
        if after_acceleration_p95 > limit:
            failures.append("p95_acceleration_degraded")
    after_bone = after["mean_bone_length_cv_percent"]
    before_bone = before["mean_bone_length_cv_percent"]
    if before_bone is not None and after_bone is not None and after_bone > before_bone + 0.25:
        failures.append("bone_stability_degraded")
    return {
        "passed": not failures,
        "reason": None if not failures else ",".join(failures),
        "checks": {
            "solver_success": solver_success,
            "median_reprojection_not_degraded": "median_reprojection_degraded" not in failures,
            "p95_reprojection_not_degraded": "p95_reprojection_degraded" not in failures,
            "p95_acceleration_not_degraded": "p95_acceleration_degraded" not in failures,
            "p95_correction_within_limit": "body_correction_too_large" not in failures,
            "bone_stability_not_degraded": "bone_stability_degraded" not in failures,
        },
        "limits": {
            "max_median_reprojection_degradation_ratio": config.max_median_reprojection_degradation_ratio,
            "max_p95_reprojection_degradation_ratio": config.max_p95_reprojection_degradation_ratio,
            "max_median_reprojection_increase_px": config.max_median_reprojection_increase_px,
            "max_p95_reprojection_increase_px": config.max_p95_reprojection_increase_px,
            "max_p95_acceleration_degradation_ratio": config.max_p95_acceleration_degradation_ratio,
            "max_p95_acceleration_increase_mps2": config.max_p95_acceleration_increase_mps2,
            "max_p95_correction_m": config.max_p95_correction_m,
        },
    }


def _mark_dependencies(
    matrix,
    row: int,
    variable_start: np.ndarray,
    frames: np.ndarray,
    joints: tuple[np.ndarray, ...],
    *,
    residuals_per_item: int,
) -> int:
    for item, frame in enumerate(frames):
        columns: list[int] = []
        for joint_values in joints:
            start = int(variable_start[int(frame), int(joint_values[item])])
            if start >= 0:
                columns.extend([start, start + 1, start + 2])
        matrix[row : row + residuals_per_item, columns] = 1
        row += residuals_per_item
    return row


def _mark_temporal_dependencies(
    matrix,
    row: int,
    variable_start: np.ndarray,
    frames: np.ndarray,
    joints: np.ndarray,
    *,
    offsets: tuple[int, ...],
    residuals_per_item: int,
) -> int:
    for item, frame in enumerate(frames):
        columns: list[int] = []
        joint = int(joints[item])
        for offset in offsets:
            start = int(variable_start[int(frame) + offset, joint])
            if start >= 0:
                columns.extend([start, start + 1, start + 2])
        matrix[row : row + residuals_per_item, columns] = 1
        row += residuals_per_item
    return row


def _three_point_acceleration(
    previous: np.ndarray,
    current: np.ndarray,
    following: np.ndarray,
    previous_time: np.ndarray,
    current_time: np.ndarray,
    following_time: np.ndarray,
) -> np.ndarray:
    dt_previous = np.maximum(current_time - previous_time, 1e-6)
    dt_next = np.maximum(following_time - current_time, 1e-6)
    velocity_previous = (current - previous) / dt_previous[:, None]
    velocity_next = (following - current) / dt_next[:, None]
    return 2.0 * (velocity_next - velocity_previous) / (dt_previous + dt_next)[:, None]


def _joint_angles_deg(parent: np.ndarray, center: np.ndarray, child: np.ndarray) -> np.ndarray:
    first = parent - center
    second = child - center
    denominator = np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
    cosine = np.divide(
        np.sum(first * second, axis=1),
        denominator,
        out=np.ones(parent.shape[0], dtype=float),
        where=denominator > 1e-12,
    )
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def _local_speed(points: np.ndarray, timestamps: np.ndarray, frame: int) -> float:
    first = max(frame - 1, 0)
    second = min(frame + 1, points.shape[0] - 1)
    duration = float(timestamps[second] - timestamps[first])
    if duration <= 0.0:
        return 0.0
    return float(np.linalg.norm(points[second] - points[first]) / duration)


def _motion_weight(speed: float, config: GlobalPoseOptimizationConfig) -> float:
    reference = config.temporal_speed_reference_mps
    return float(np.clip(reference / (reference + max(speed, 0.0)), config.temporal_weight_floor, 1.0))


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    cutoff = 0.5 * float(cumulative[-1])
    return float(sorted_values[min(int(np.searchsorted(cumulative, cutoff, side="left")), sorted_values.size - 1)])


def _soft_l1_residual(values: np.ndarray) -> np.ndarray:
    raw = np.asarray(values, dtype=float)
    magnitude = np.sqrt(2.0 * (np.sqrt(1.0 + raw * raw) - 1.0))
    return np.copysign(magnitude, raw)


def _distribution(values: np.ndarray) -> dict[str, float | int | None]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p95": float(np.percentile(finite, 95)),
        "max": float(np.max(finite)),
    }
