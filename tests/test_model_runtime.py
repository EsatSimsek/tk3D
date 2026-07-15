from __future__ import annotations

import yaml
import pytest

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
