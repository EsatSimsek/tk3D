from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_run_multiview_3d_dry_run_outputs_expected_files(tmp_path) -> None:
    root = Path(__file__).resolve().parents[1]
    output_root = tmp_path / "outputs"
    command = [
        sys.executable,
        str(root / "scripts" / "run_multiview_3d.py"),
        "--session",
        str(root / "data" / "session_001" / "session.yaml"),
        "--output-root",
        str(output_root),
        "--dry-run",
        "--dry-run-frames",
        "4",
    ]

    completed = subprocess.run(command, cwd=root, check=True, capture_output=True, text=True)

    assert "keypoints_3d_world shape: (4, 133, 3)" in completed.stdout
    assert "mean reprojection error px:" in completed.stdout
    session_json = output_root / "session_001" / "json" / "session_3d.json"
    assert session_json.exists()
    payload = json.loads(session_json.read_text(encoding="utf-8"))
    assert payload["shape"]["keypoints_3d_world"] == [4, 133, 3]
    assert (output_root / "session_001" / "csv" / "keypoints_2d_flat.csv").exists()
    assert (output_root / "session_001" / "csv" / "validation_joints.csv").exists()
    assert (output_root / "session_001" / "calibration" / "cameras.json").exists()
    skeleton_video = output_root / "session_001" / "videos" / "skeleton_3d_world.mp4"
    assert skeleton_video.exists()
    assert skeleton_video.stat().st_size > 0

    quality = json.loads((output_root / "session_001" / "json" / "quality_summary.json").read_text(encoding="utf-8"))
    assert quality["frame_count"] == 4
    assert quality["mean_reprojection_error_px"] < 1e-6

    manifest = json.loads((output_root / "session_001" / "json" / "artifact_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert not manifest["missing_outputs"]
