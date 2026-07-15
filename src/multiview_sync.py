from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math

from .data_structures import PersonPose2D


@dataclass(frozen=True, slots=True)
class SynchronizedFrame:
    global_frame_idx: int
    timestamp_sec: float
    local_frame_indices: dict[str, int]


def synchronized_frame_map(
    frame_counts: dict[str, int],
    fps_by_camera: dict[str, float],
    frame_offsets: dict[str, int] | None = None,
    time_offsets_sec: dict[str, float] | None = None,
    target_fps: float | None = None,
) -> list[SynchronizedFrame]:
    """Map a common timestamp timeline to local camera frames.

    Positive offsets mean that local frame zero occurs later on the common
    timeline. Frame offsets are converted using each camera's own FPS, which
    prevents drift when camera rates are close but not identical.
    """
    if not frame_counts:
        return []
    camera_ids = list(frame_counts)
    missing_fps = [camera_id for camera_id in camera_ids if fps_by_camera.get(camera_id, 0.0) <= 0.0]
    if missing_fps:
        raise ValueError(f"Valid FPS is required for every camera: {missing_fps}")
    offsets = frame_offsets or {}
    seconds = time_offsets_sec or {}
    rates = [float(fps_by_camera[camera_id]) for camera_id in camera_ids]
    timeline_fps = float(target_fps or min(rates))
    if timeline_fps <= 0.0:
        raise ValueError("target_fps must be positive")

    starts: list[float] = []
    ends: list[float] = []
    total_offsets: dict[str, float] = {}
    for camera_id in camera_ids:
        fps = float(fps_by_camera[camera_id])
        offset_sec = float(seconds.get(camera_id, 0.0)) + int(offsets.get(camera_id, 0)) / fps
        total_offsets[camera_id] = offset_sec
        count = max(int(frame_counts[camera_id]), 0)
        if count == 0:
            return []
        starts.append(offset_sec)
        ends.append(offset_sec + (count - 1) / fps)

    first_global = math.ceil(max(starts) * timeline_fps - 1e-9)
    last_global = math.floor(min(ends) * timeline_fps + 1e-9)
    if last_global < first_global:
        return []

    result: list[SynchronizedFrame] = []
    for global_idx in range(first_global, last_global + 1):
        timestamp = global_idx / timeline_fps
        local_indices = {
            camera_id: int(round((timestamp - total_offsets[camera_id]) * float(fps_by_camera[camera_id])))
            for camera_id in camera_ids
        }
        if all(0 <= local_indices[camera_id] < int(frame_counts[camera_id]) for camera_id in camera_ids):
            result.append(SynchronizedFrame(global_idx, timestamp, local_indices))
    return result


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
