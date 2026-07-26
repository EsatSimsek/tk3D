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


def test_draw_pose2d_draws_hand_detail_by_default() -> None:
    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    keypoints = np.full((133, 2), np.nan, dtype=float)
    scores = np.zeros(133, dtype=float)
    valid = np.zeros(133, dtype=bool)
    keypoints[91] = [20, 40]
    keypoints[92] = [40, 40]
    scores[[91, 92]] = 1.0
    valid[[91, 92]] = True
    pose = PersonPose2D("c01", 0, keypoints, scores, valid)

    output = draw_pose2d(frame, pose)

    assert output.sum() > 0
    assert output[40, 30].sum() > 0


def test_draw_pose2d_can_hide_hand_detail() -> None:
    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    keypoints = np.full((133, 2), np.nan, dtype=float)
    scores = np.zeros(133, dtype=float)
    valid = np.zeros(133, dtype=bool)
    keypoints[100] = [40, 40]
    scores[100] = 1.0
    valid[100] = True
    pose = PersonPose2D("c01", 0, keypoints, scores, valid)

    assert draw_pose2d(frame, pose, draw_hands=False).sum() == 0


def test_draw_pose2d_marks_crossview_recovered_points_in_orange() -> None:
    frame = np.zeros((80, 240, 3), dtype=np.uint8)
    keypoints = np.full((133, 2), np.nan, dtype=float)
    scores = np.zeros(133, dtype=float)
    valid = np.zeros(133, dtype=bool)
    keypoints[5] = [100, 60]
    scores[5] = 0.8
    valid[5] = True
    provenance = np.zeros(133, dtype=np.uint8)
    provenance[5] = 2
    pose = PersonPose2D("c05", 0, keypoints, scores, valid)

    output = draw_pose2d(frame, pose, provenance=provenance)

    np.testing.assert_array_equal(output[60, 100], np.array([0, 165, 255], dtype=np.uint8))
