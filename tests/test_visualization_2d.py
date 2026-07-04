from __future__ import annotations

import numpy as np

from src.data_structures import PersonPose2D
from src.visualization_2d import draw_pose2d


def test_draw_pose2d_draws_body_edges() -> None:
    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    keypoints = np.full((133, 2), np.nan, dtype=float)
    scores = np.zeros(133, dtype=float)
    valid = np.zeros(133, dtype=bool)
    keypoints[5] = [20, 20]
    keypoints[6] = [60, 20]
    valid[[5, 6]] = True
    scores[[5, 6]] = 1.0
    pose = PersonPose2D("c01", 0, keypoints, scores, valid)

    output = draw_pose2d(frame, pose)

    assert output.sum() > 0
    assert output[20, 40].sum() > 0
