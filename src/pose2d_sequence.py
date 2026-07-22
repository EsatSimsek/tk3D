from __future__ import annotations

from bisect import bisect_right

import numpy as np

from .data_structures import PersonPose2D


def interpolate_pose2d(first: PersonPose2D, second: PersonPose2D, frame_idx: int) -> PersonPose2D:
    if first.camera_id != second.camera_id:
        raise ValueError("Cannot interpolate poses from different cameras")
    if second.frame_idx < first.frame_idx:
        raise ValueError("Pose interpolation requires increasing frame indices")
    if second.frame_idx == first.frame_idx:
        weight = 0.0
    else:
        weight = float(np.clip((frame_idx - first.frame_idx) / (second.frame_idx - first.frame_idx), 0.0, 1.0))
    both_valid = np.asarray(first.valid_mask, dtype=bool) & np.asarray(second.valid_mask, dtype=bool)
    xy = np.full_like(first.keypoints_xy, np.nan, dtype=float)
    xy[both_valid] = (1.0 - weight) * first.keypoints_xy[both_valid] + weight * second.keypoints_xy[both_valid]
    scores = np.zeros_like(first.scores, dtype=float)
    scores[both_valid] = (1.0 - weight) * first.scores[both_valid] + weight * second.scores[both_valid]

    if weight <= 1e-12:
        xy = np.asarray(first.keypoints_xy, dtype=float).copy()
        scores = np.asarray(first.scores, dtype=float).copy()
        both_valid = np.asarray(first.valid_mask, dtype=bool).copy()
    elif weight >= 1.0 - 1e-12:
        xy = np.asarray(second.keypoints_xy, dtype=float).copy()
        scores = np.asarray(second.scores, dtype=float).copy()
        both_valid = np.asarray(second.valid_mask, dtype=bool).copy()
    return PersonPose2D(
        camera_id=first.camera_id,
        frame_idx=int(frame_idx),
        keypoints_xy=xy,
        scores=scores,
        valid_mask=both_valid,
        person_id=first.person_id,
    )


def pose2d_at_frame(sampled_poses: list[PersonPose2D], frame_idx: int) -> PersonPose2D:
    if not sampled_poses:
        raise ValueError("sampled_poses cannot be empty")
    ordered = sampled_poses
    indices = [pose.frame_idx for pose in ordered]
    position = bisect_right(indices, int(frame_idx))
    if position <= 0:
        return interpolate_pose2d(ordered[0], ordered[0], frame_idx)
    if position >= len(ordered):
        return interpolate_pose2d(ordered[-1], ordered[-1], frame_idx)
    return interpolate_pose2d(ordered[position - 1], ordered[position], frame_idx)
