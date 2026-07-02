from __future__ import annotations

from src.model_runtime import check_model_runtime


def test_model_runtime_reports_missing_files_and_backend(tmp_path) -> None:
    status = check_model_runtime(
        {
            "backend": "definitely_missing_backend_for_tk3d_tests",
            "model_name": "RTMW-x-l",
            "config_path": "models/missing.py",
            "checkpoint_path": "weights/missing.pth",
        },
        tmp_path,
    )

    assert status.ready is False
    assert status.backend_available is False
    assert status.config_exists is False
    assert status.checkpoint_exists is False
    assert "config file is missing" in status.message
