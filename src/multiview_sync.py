from __future__ import annotations

from collections import defaultdict

from .data_structures import PersonPose2D


def group_poses_by_global_frame(poses: list[PersonPose2D], frame_offsets: dict[str, int]) -> dict[int, list[PersonPose2D]]:
    grouped: dict[int, list[PersonPose2D]] = defaultdict(list)
    for pose in poses:
        global_idx = pose.frame_idx + int(frame_offsets.get(pose.camera_id, 0))
        grouped[global_idx].append(pose)
    return dict(sorted(grouped.items()))


def common_frame_count(frame_counts: dict[str, int], frame_offsets: dict[str, int]) -> int:
    if not frame_counts:
        return 0
    starts = [int(frame_offsets.get(camera_id, 0)) for camera_id in frame_counts]
    ends = [int(frame_offsets.get(camera_id, 0)) + count for camera_id, count in frame_counts.items()]
    return max(0, min(ends) - max(starts))


def global_frame_range(frame_counts: dict[str, int], frame_offsets: dict[str, int]) -> range:
    if not frame_counts:
        return range(0)
    starts = [int(frame_offsets.get(camera_id, 0)) for camera_id in frame_counts]
    ends = [int(frame_offsets.get(camera_id, 0)) + count for camera_id, count in frame_counts.items()]
    return range(max(starts), min(ends))


def local_frame_for_global(camera_id: str, global_frame_idx: int, frame_offsets: dict[str, int]) -> int:
    return int(global_frame_idx) - int(frame_offsets.get(camera_id, 0))
