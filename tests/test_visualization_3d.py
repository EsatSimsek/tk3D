from __future__ import annotations

import numpy as np

from src.visualization_3d import save_reprojection_timeline


def test_save_reprojection_timeline_handles_all_nan(tmp_path) -> None:
    output_path = tmp_path / "timeline.png"
    errors = np.full((3, 5), np.nan, dtype=float)

    save_reprojection_timeline(errors, output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
