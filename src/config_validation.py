from __future__ import annotations

from typing import Any


def validate_model_config(config: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise ValueError("model_config must be a mapping")
    pose = _mapping(config, "pose2d")
    for key in ("model_name", "backend", "config_path", "checkpoint_path"):
        if not pose.get(key):
            raise ValueError(f"pose2d.{key} is required")
    input_size = pose.get("input_size")
    if (
        not isinstance(input_size, list)
        or len(input_size) != 2
        or any(int(value) <= 0 or int(value) % 16 for value in input_size)
    ):
        raise ValueError("pose2d.input_size must contain two positive multiples of 16")
    if int(pose.get("keypoint_count", 0)) != 133:
        raise ValueError("pose2d.keypoint_count must be 133 for COCO-WholeBody")
    adapter_path = pose.get("adapter_checkpoint_path")
    if adapter_path is not None and not str(adapter_path).strip():
        raise ValueError("pose2d.adapter_checkpoint_path must be a non-empty path when provided")
    if "allow_unapproved_adapter" in pose and not isinstance(pose["allow_unapproved_adapter"], bool):
        raise ValueError("pose2d.allow_unapproved_adapter must be boolean")
    threshold = float(pose.get("score_threshold", 0.30))
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("pose2d.score_threshold must be between 0 and 1")
    for option in ("flip_test", "temporal_filter_enabled", "temporal_stabilize_left_right"):
        if option in pose and not isinstance(pose[option], bool):
            raise ValueError(f"pose2d.{option} must be boolean")
    offline = pose.get("offline_stabilization", {})
    if not isinstance(offline, dict):
        raise ValueError("pose2d.offline_stabilization must be a mapping")
    if "enabled" in offline and not isinstance(offline["enabled"], bool):
        raise ValueError("pose2d.offline_stabilization.enabled must be boolean")
    offline_window = int(offline.get("window_size", 9))
    if offline_window < 1 or offline_window % 2 == 0:
        raise ValueError("pose2d.offline_stabilization.window_size must be a positive odd integer")
    offline_order = int(offline.get("polynomial_order", 2))
    if offline_order < 1 or (offline_window > 1 and offline_order >= offline_window):
        raise ValueError("pose2d.offline_stabilization.polynomial_order must be positive and smaller than window_size")
    if float(offline.get("min_outlier_distance_px", 6.0)) < 0.0:
        raise ValueError("pose2d.offline_stabilization.min_outlier_distance_px must be non-negative")

    detector = config.get("person_detector", {})
    if not isinstance(detector, dict):
        raise ValueError("person_detector must be a mapping")
    if "enabled" in detector and not isinstance(detector["enabled"], bool):
        raise ValueError("person_detector.enabled must be boolean")
    if detector.get("enabled", False):
        if detector.get("backend", "rfdetr") != "rfdetr":
            raise ValueError("person_detector.backend must be rfdetr")
        if detector.get("model_variant", "small") not in {"nano", "small", "medium", "large"}:
            raise ValueError("person_detector.model_variant must be nano, small, medium, or large")
        for key, default in (
            ("threshold", 0.25),
            ("target_confidence_threshold", 0.65),
            ("track_activation_threshold", 0.25),
            ("minimum_matching_threshold", 0.80),
        ):
            value = float(detector.get(key, default))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"person_detector.{key} must be between 0 and 1")
        if not 0.0 <= float(detector.get("bbox_padding", 0.18)) <= 1.0:
            raise ValueError("person_detector.bbox_padding must be between 0 and 1")
        stationary_alpha = float(detector.get("bbox_stationary_alpha", 0.35))
        if not 0.0 < stationary_alpha <= 1.0:
            raise ValueError("person_detector.bbox_stationary_alpha must be between 0 and 1")
        if float(detector.get("bbox_motion_scale_ratio", 0.12)) <= 0.0:
            raise ValueError("person_detector.bbox_motion_scale_ratio must be positive")
        for key, default in (("lost_track_buffer", 30), ("reacquire_after_frames", 12)):
            if int(detector.get(key, default)) < 1:
                raise ValueError(f"person_detector.{key} must be positive")
        if "optimize_for_inference" in detector and not isinstance(detector["optimize_for_inference"], bool):
            raise ValueError("person_detector.optimize_for_inference must be boolean")

    triangulation = _mapping(config, "triangulation")
    if int(triangulation.get("min_views", 2)) < 2:
        raise ValueError("triangulation.min_views must be at least 2")
    keypoint_score = float(triangulation.get("min_keypoint_score", 0.30))
    if not 0.0 <= keypoint_score <= 1.0:
        raise ValueError("triangulation.min_keypoint_score must be between 0 and 1")
    if float(triangulation.get("max_reprojection_error_px", 25.0)) <= 0.0:
        raise ValueError("triangulation.max_reprojection_error_px must be positive")
    quality = float(triangulation.get("min_triangulation_score", 0.20))
    if not 0.0 <= quality <= 1.0:
        raise ValueError("triangulation.min_triangulation_score must be between 0 and 1")
    if int(triangulation.get("max_hypotheses", 16)) < 1:
        raise ValueError("triangulation.max_hypotheses must be positive")

    global_optimization = config.get("global_optimization", {})
    if not isinstance(global_optimization, dict):
        raise ValueError("global_optimization must be a mapping")
    if "enabled" in global_optimization and not isinstance(global_optimization["enabled"], bool):
        raise ValueError("global_optimization.enabled must be boolean")
    optimization_score = float(global_optimization.get("min_observation_score", 0.30))
    if not 0.0 <= optimization_score <= 1.0:
        raise ValueError("global_optimization.min_observation_score must be between 0 and 1")
    for key, default in (
        ("max_gap_frames", 12),
        ("minimum_bone_samples", 12),
        ("outer_iterations", 3),
        ("max_solver_evaluations", 35),
    ):
        if int(global_optimization.get(key, default)) < 1:
            raise ValueError(f"global_optimization.{key} must be positive")
    for key, default in (
        ("reprojection_scale_px", 8.0),
        ("bone_scale_m", 0.025),
        ("acceleration_scale_mps2", 25.0),
        ("jerk_scale_mps3", 500.0),
        ("angle_scale_deg", 10.0),
        ("anchor_scale_m", 0.15),
        ("camera_error_scale_px", 12.0),
        ("temporal_speed_reference_mps", 4.0),
        ("max_median_reprojection_increase_px", 1.0),
        ("max_p95_reprojection_increase_px", 2.0),
        ("max_p95_acceleration_increase_mps2", 5.0),
        ("max_p95_correction_m", 0.30),
    ):
        if float(global_optimization.get(key, default)) <= 0.0:
            raise ValueError(f"global_optimization.{key} must be positive")
    for key, default in (
        ("camera_weight_floor", 0.20),
        ("camera_weight_update_alpha", 0.50),
        ("temporal_weight_floor", 0.15),
    ):
        value = float(global_optimization.get(key, default))
        if not 0.0 < value <= 1.0:
            raise ValueError(f"global_optimization.{key} must be greater than 0 and at most 1")
    for key, default in (
        ("max_median_reprojection_degradation_ratio", 1.05),
        ("max_p95_reprojection_degradation_ratio", 1.10),
        ("max_p95_acceleration_degradation_ratio", 1.10),
    ):
        if float(global_optimization.get(key, default)) < 1.0:
            raise ValueError(f"global_optimization.{key} must be at least 1")
    optimization_weights = global_optimization.get("weights", {})
    if not isinstance(optimization_weights, dict):
        raise ValueError("global_optimization.weights must be a mapping")
    for key, default in (
        ("reprojection", 2.0),
        ("bone", 1.0),
        ("acceleration", 0.06),
        ("jerk", 0.01),
        ("joint_limits", 0.50),
        ("anchor", 0.03),
    ):
        if float(optimization_weights.get(key, default)) < 0.0:
            raise ValueError(f"global_optimization.weights.{key} must be non-negative")
    if float(optimization_weights.get("reprojection", 1.0)) <= 0.0:
        raise ValueError("global_optimization.weights.reprojection must be positive")

    smoothing = _mapping(config, "smoothing")
    if smoothing.get("method") not in {"moving_average", "robust_savgol"}:
        raise ValueError("smoothing.method must be moving_average or robust_savgol")
    window = int(smoothing.get("window_size", 5))
    if window < 1 or window % 2 == 0:
        raise ValueError("smoothing.window_size must be a positive odd integer")
    polynomial_order = int(smoothing.get("polynomial_order", 2))
    if polynomial_order < 1 or (window > 1 and polynomial_order >= window):
        raise ValueError("smoothing.polynomial_order must be positive and smaller than window_size")
    if float(smoothing.get("min_outlier_distance_m", 0.04)) < 0.0:
        raise ValueError("smoothing.min_outlier_distance_m must be non-negative")

    reliability = config.get("reliability", {})
    if not isinstance(reliability, dict):
        raise ValueError("reliability must be a mapping")
    if float(reliability.get("max_bone_relative_deviation", 0.25)) <= 0.0:
        raise ValueError("reliability.max_bone_relative_deviation must be positive")
    if float(reliability.get("max_bone_absolute_deviation_m", 0.08)) <= 0.0:
        raise ValueError("reliability.max_bone_absolute_deviation_m must be positive")
    if float(reliability.get("min_temporal_residual_m", 0.08)) <= 0.0:
        raise ValueError("reliability.min_temporal_residual_m must be positive")
    if float(reliability.get("max_temporal_acceleration_mps2", 70.0)) <= 0.0:
        raise ValueError("reliability.max_temporal_acceleration_mps2 must be positive")
    if int(reliability.get("minimum_bone_samples", 5)) < 1:
        raise ValueError("reliability.minimum_bone_samples must be positive")
    minimum_ratio = float(reliability.get("min_output_valid_body_ratio", 0.90))
    if not 0.0 <= minimum_ratio <= 1.0:
        raise ValueError("reliability.min_output_valid_body_ratio must be between 0 and 1")
    return config


def validate_calibration_config(config: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise ValueError("calibration_config must be a mapping")
    checker = _mapping(config, "checkerboard")
    pattern = checker.get("pattern_size")
    if not isinstance(pattern, list) or len(pattern) != 2 or any(int(value) < 2 for value in pattern):
        raise ValueError("checkerboard.pattern_size must contain two integers >= 2")
    if float(checker.get("square_size_m", 0.0)) <= 0.0:
        raise ValueError("checkerboard.square_size_m must be positive")
    if int(checker.get("min_valid_frames", 0)) < 3:
        raise ValueError("checkerboard.min_valid_frames must be at least 3")
    if int(checker.get("frame_stride", 0)) < 1:
        raise ValueError("checkerboard.frame_stride must be positive")
    if int(checker.get("min_common_frames", 3)) < 3:
        raise ValueError("checkerboard.min_common_frames must be at least 3")
    if "sync_tolerance_sec" in checker and float(checker["sync_tolerance_sec"]) <= 0.0:
        raise ValueError("checkerboard.sync_tolerance_sec must be positive")
    calibration = _mapping(config, "calibration")
    if float(calibration.get("reprojection_error_warn_px", 0.0)) <= 0.0:
        raise ValueError("calibration.reprojection_error_warn_px must be positive")
    return config


def _mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return value
