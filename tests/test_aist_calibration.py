from __future__ import annotations

import json
from pathlib import Path

from src.aist_calibration import find_aist_camera_setting, load_aist_camera_calibrations


def test_load_aist_camera_calibrations(tmp_path: Path) -> None:
    cameras_dir = tmp_path / "cameras"
    cameras_dir.mkdir()
    (cameras_dir / "mapping.txt").write_text("seq_cAll setting1\n", encoding="utf-8")
    camera_payload = [
        {
            "name": "c01",
            "size": [1920, 1080],
            "matrix": [[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]],
            "distortions": [0.1, 0.0, 0.0, 0.0, 0.0],
            "rotation": [0.0, 0.0, 0.0],
            "translation": [1.0, 2.0, 3.0],
        },
        {
            "name": "c02",
            "size": [1920, 1080],
            "matrix": [[1100.0, 0.0, 960.0], [0.0, 1100.0, 540.0], [0.0, 0.0, 1.0]],
            "distortions": [0.0, 0.0, 0.0, 0.0, 0.0],
            "rotation": [0.0, 0.0, 0.1],
            "translation": [4.0, 5.0, 6.0],
        },
    ]
    (cameras_dir / "setting1.json").write_text(json.dumps(camera_payload), encoding="utf-8")

    assert find_aist_camera_setting("seq_cAll", cameras_dir) == "setting1"
    calibrations = load_aist_camera_calibrations("seq_cAll", cameras_dir, camera_ids=["c02"])

    assert len(calibrations) == 1
    assert calibrations[0].camera_id == "c02"
    assert calibrations[0].image_size == (1920, 1080)
    assert calibrations[0].projection_matrix.shape == (3, 4)
    assert calibrations[0].translation_vector.tolist() == [4.0, 5.0, 6.0]
