from __future__ import annotations

import numpy as np

from src.pose3d_html_viewer import write_pose3d_html_viewer


def test_pose3d_html_viewer_serializes_invalid_joints_as_json_null(tmp_path) -> None:
    keypoints = np.full((2, 133, 3), np.nan, dtype=float)
    keypoints[:, 5] = [0.0, 0.0, 1.0]
    keypoints[:, 7] = [0.2, 0.0, 0.8]
    output_path = tmp_path / "viewer.html"

    write_pose3d_html_viewer(keypoints, output_path, fps=30.0, title="test")

    html = output_path.read_text(encoding="utf-8")
    assert "TK3D" in html
    assert "null" in html
    assert "NaN" not in html
    assert "OrbitControls" in html
