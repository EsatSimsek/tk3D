from __future__ import annotations

import numpy as np

from src.visualization_3d import (
    COCO_HAND_EDGES,
    COCO_WHOLEBODY_EDGES,
    save_reprojection_timeline,
)


def test_save_reprojection_timeline_handles_all_nan(tmp_path) -> None:
    output_path = tmp_path / "timeline.png"
    errors = np.full((3, 5), np.nan, dtype=float)

    save_reprojection_timeline(errors, output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_wholebody_topology_contains_both_hands_and_fingers() -> None:
    assert (9, 91) in COCO_HAND_EDGES
    assert (10, 112) in COCO_HAND_EDGES
    assert (91, 92) in COCO_WHOLEBODY_EDGES
    assert (111, 110) in COCO_WHOLEBODY_EDGES or (110, 111) in COCO_WHOLEBODY_EDGES
    assert (112, 113) in COCO_WHOLEBODY_EDGES
    assert (131, 132) in COCO_WHOLEBODY_EDGES
