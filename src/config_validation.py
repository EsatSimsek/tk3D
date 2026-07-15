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
    if "allow_unapproved_adapter" in pose and not isinstance(
        pose["allow_unapproved_adapter"], bool
    ):
        raise ValueError("pose2d.allow_unapproved_adapter must be boolean")
    threshold = float(pose.get("score_threshold", 0.30))
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("pose2d.score_threshold must be between 0 and 1")

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

    smoothing = _mapping(config, "smoothing")
    if smoothing.get("method") != "moving_average":
        raise ValueError("Only smoothing.method=moving_average is currently supported")
    window = int(smoothing.get("window_size", 5))
    if window < 1 or window % 2 == 0:
        raise ValueError("smoothing.window_size must be a positive odd integer")

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
