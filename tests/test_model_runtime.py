from __future__ import annotations

import yaml
import pytest
import torch

from src.config_validation import validate_calibration_config, validate_model_config
from src.model_runtime import check_model_runtime


def test_model_runtime_reports_missing_files_and_backend(tmp_path) -> None:
    status = check_model_runtime(
        {
            "backend": "definitely_missing_backend_for_tk3d_tests",
            "model_name": "ViTPose-Huge-WholeBody",
            "config_path": "models/vitpose_missing.py",
            "checkpoint_path": "weights/vitpose_missing.pth",
        },
        tmp_path,
    )

    assert status.ready is False
    assert status.backend_available is False
    assert status.config_exists is False
    assert status.checkpoint_exists is False
    assert status.checkpoint_valid is False
    assert status.checkpoint_compatible is False
    assert "config file is missing" in status.message


def test_default_config_disables_optional_single_view_rtmw3d() -> None:
    with open("config/model_config.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    assert config["pose3d_single_view"]["enabled"] is False


def test_model_config_rejects_non_wholebody_keypoint_count() -> None:
    with open("config/model_config.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    config["pose2d"]["keypoint_count"] = 17

    with pytest.raises(ValueError, match="133"):
        validate_model_config(config)


def test_unapproved_adapter_is_not_runtime_ready(tmp_path) -> None:
    adapter = tmp_path / "adapter.pth"
    torch.save(
        {
            "heatmap_offsets_xy": torch.zeros((133, 2)),
            "metadata": {"production_approved": False},
        },
        adapter,
    )

    status = check_model_runtime(
        {
            "backend": "yaml",
            "model_name": "ViTPose-Huge-WholeBody",
            "config_path": "missing.py",
            "checkpoint_path": "missing.pth",
            "adapter_checkpoint_path": str(adapter),
            "keypoint_count": 133,
        },
        tmp_path,
    )

    assert status.adapter_checkpoint_exists is True
    assert status.adapter_checkpoint_compatible is True
    assert status.adapter_checkpoint_approved is False
    assert "has not passed held-out 3D approval" in status.message


def test_diagnostic_flag_allows_benchmarking_unapproved_adapter(tmp_path) -> None:
    adapter = tmp_path / "adapter.pth"
    torch.save(
        {
            "heatmap_offsets_xy": torch.zeros((133, 2)),
            "metadata": {"production_approved": False},
        },
        adapter,
    )

    status = check_model_runtime(
        {
            "backend": "yaml",
            "model_name": "ViTPose-Huge-WholeBody",
            "config_path": "missing.py",
            "checkpoint_path": "missing.pth",
            "adapter_checkpoint_path": str(adapter),
            "allow_unapproved_adapter": True,
            "keypoint_count": 133,
        },
        tmp_path,
    )

    assert "has not passed held-out 3D approval" not in status.message


def test_model_config_rejects_non_boolean_adapter_override() -> None:
    with open("config/model_config.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    config["pose2d"]["allow_unapproved_adapter"] = "yes"

    with pytest.raises(ValueError, match="must be boolean"):
        validate_model_config(config)


def test_model_config_rejects_invalid_person_detector_threshold() -> None:
    with open("config/model_config.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    config["person_detector"]["threshold"] = 1.5

    with pytest.raises(ValueError, match="person_detector.threshold"):
        validate_model_config(config)


def test_model_config_rejects_unsafe_global_optimization_limits() -> None:
    with open("config/model_config.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    config["global_optimization"]["max_p95_correction_m"] = 0.0

    with pytest.raises(ValueError, match="global_optimization.max_p95_correction_m"):
        validate_model_config(config)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf"), "nan", "inf", None, True])
@pytest.mark.parametrize(
    "field",
    [
        "pose2d.offline_stabilization.min_outlier_distance_px",
        "person_detector.bbox_motion_scale_ratio",
        "triangulation.max_reprojection_error_px",
        "zed_depth_fusion.max_depth_m",
        "zed_depth_fusion.max_final_median_reprojection_ratio",
        "crossview_2d_feedback.search_radius_px",
        "global_optimization.max_p95_correction_m",
        "global_optimization.max_median_reprojection_degradation_ratio",
        "global_optimization.weights.bone",
        "smoothing.min_outlier_distance_m",
        "reliability.max_temporal_acceleration_mps2",
    ],
)
def test_model_config_rejects_invalid_numeric_safety_settings(field, invalid) -> None:
    with open("config/model_config.yaml", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    section = config
    *parents, key = field.split(".")
    for parent in parents:
        section = section.setdefault(parent, {})
    section[key] = invalid

    with pytest.raises(ValueError, match=field):
        validate_model_config(config)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), "nan", None, True])
@pytest.mark.parametrize(
    "field", ["checkerboard.square_size_m", "checkerboard.sync_tolerance_sec", "calibration.reprojection_error_warn_px"]
)
def test_calibration_config_rejects_invalid_numeric_settings(field, invalid) -> None:
    config = {
        "checkerboard": {"pattern_size": [9, 6], "square_size_m": 0.025, "min_valid_frames": 5, "frame_stride": 1},
        "calibration": {"reprojection_error_warn_px": 2.0},
    }
    section, key = field.split(".")
    config[section][key] = invalid

    with pytest.raises(ValueError, match=field):
        validate_calibration_config(config)
