from __future__ import annotations

import numpy as np

from src.data_structures import PersonPose2D
from src.pose2d_estimator import _motion_person_bbox, _motion_requires_reacquisition, _update_tracked_bbox
from src.pose2d_sequence import interpolate_pose2d, pose2d_at_frame
from src.pose_temporal import TemporalPose2DConfig, TemporalPose2DFilter


def _pose(frame_idx: int, noise_x: float = 0.0, score: float = 1.0) -> PersonPose2D:
    xy = np.full((133, 2), np.nan, dtype=float)
    scores = np.zeros(133, dtype=float)
    valid = np.zeros(133, dtype=bool)
    base = {
        5: (100.0, 100.0), 6: (200.0, 100.0),
        7: (80.0, 150.0), 8: (220.0, 150.0),
        9: (60.0, 200.0), 10: (240.0, 200.0),
        11: (120.0, 220.0), 12: (180.0, 220.0),
        13: (120.0, 300.0), 14: (180.0, 300.0),
        15: (120.0, 380.0), 16: (180.0, 380.0),
    }
    for joint_idx, point in base.items():
        xy[joint_idx] = [point[0] + noise_x, point[1]]
        scores[joint_idx] = score
        valid[joint_idx] = True
    return PersonPose2D("c01", frame_idx, xy, scores, valid)


def test_temporal_filter_reduces_stationary_jitter() -> None:
    pose_filter = TemporalPose2DFilter()
    raw = []
    filtered = []
    for frame_idx in range(20):
        noise = 3.0 if frame_idx % 2 else -3.0
        raw.append(100.0 + noise)
        filtered.append(pose_filter.filter(_pose(frame_idx, noise_x=noise)).keypoints_xy[5, 0])
    assert np.std(filtered[5:]) < 0.55 * np.std(raw[5:])


def test_temporal_filter_rejects_low_confidence_large_jump() -> None:
    pose_filter = TemporalPose2DFilter(TemporalPose2DConfig(max_jump_ratio=0.20))
    pose_filter.filter(_pose(0))
    jumped = _pose(1, noise_x=200.0, score=0.4)
    result = pose_filter.filter(jumped)
    assert not result.valid_mask[5]
    assert result.scores[5] == 0.0


def test_temporal_filter_follows_motion_between_sparse_samples() -> None:
    pose_filter = TemporalPose2DFilter()
    pose_filter.filter(_pose(0))
    result = pose_filter.filter(_pose(20, noise_x=100.0))
    assert result.keypoints_xy[5, 0] > 185.0


def test_temporal_filter_corrects_left_right_identity_flip() -> None:
    pose_filter = TemporalPose2DFilter(
        TemporalPose2DConfig(stabilize_left_right=True, stationary_alpha=1.0, motion_alpha=1.0)
    )
    first = _pose(0)
    flipped = _pose(1)
    flipped.keypoints_xy[[5, 6]] = flipped.keypoints_xy[[6, 5]]

    pose_filter.filter(first)
    filtered = pose_filter.filter(flipped)

    np.testing.assert_allclose(filtered.keypoints_xy[5], [100.0, 100.0])
    np.testing.assert_allclose(filtered.keypoints_xy[6], [200.0, 100.0])


def test_tracked_bbox_changes_scale_slowly() -> None:
    previous = np.asarray([100.0, 50.0, 300.0, 350.0])
    candidate = np.asarray([130.0, 100.0, 230.0, 250.0])
    updated = _update_tracked_bbox(previous, candidate, image_width=640, image_height=480)
    previous_size = previous[2:] - previous[:2]
    updated_size = updated[2:] - updated[:2]
    assert np.all(updated_size > 0.95 * previous_size)


def test_tracked_bbox_catches_up_after_sparse_frame_gap() -> None:
    previous = np.asarray([100.0, 50.0, 300.0, 350.0])
    candidate = np.asarray([300.0, 50.0, 500.0, 350.0])
    one_frame = _update_tracked_bbox(previous, candidate, 640, 480, frame_delta=1)
    sparse = _update_tracked_bbox(previous, candidate, 640, 480, frame_delta=20)
    candidate_center = (candidate[:2] + candidate[2:]) / 2.0
    one_frame_center = (one_frame[:2] + one_frame[2:]) / 2.0
    sparse_center = (sparse[:2] + sparse[2:]) / 2.0
    assert np.linalg.norm(sparse_center - candidate_center) < np.linalg.norm(one_frame_center - candidate_center)
    np.testing.assert_allclose(sparse_center, candidate_center, atol=0.1)


def test_motion_bbox_reacquires_moving_person_instead_of_static_decoy() -> None:
    previous = np.full((384, 512, 3), 220, dtype=np.uint8)
    current = previous.copy()
    previous[80:360, 190:245] = 25  # Static human-shaped decoy.
    current[80:360, 190:245] = 25
    previous[85:365, 340:400] = (30, 210, 240)
    current[75:355, 375:435] = (30, 210, 240)

    motion_bbox = _motion_person_bbox(previous, current)
    assert motion_bbox is not None
    motion_center_x = float((motion_bbox[0] + motion_bbox[2]) / 2.0)
    assert motion_center_x > 330.0
    assert motion_bbox[3] - motion_bbox[1] > 300.0
    decoy_bbox = np.asarray([180.0, 70.0, 255.0, 370.0])
    assert _motion_requires_reacquisition(decoy_bbox, motion_bbox)


def test_pose_interpolation_uses_real_in_between_position() -> None:
    first = _pose(0)
    second = _pose(10, noise_x=100.0)
    middle = interpolate_pose2d(first, second, 5)
    selected = pose2d_at_frame([first, second], 5)
    np.testing.assert_allclose(middle.keypoints_xy[5], [150.0, 100.0])
    np.testing.assert_allclose(selected.keypoints_xy[5], middle.keypoints_xy[5])
