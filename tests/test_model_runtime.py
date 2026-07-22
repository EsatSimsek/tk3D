from __future__ import annotations

import yaml
import pytest
import torch

from src.config_validation import validate_model_config
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
