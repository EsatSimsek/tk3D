from __future__ import annotations

import numpy as np

from scripts.run_vitpose_multiview_3d import (
    _effective_smoothing_window,
    _repeat_array_for_video,
    _repeat_count,
    _target_sample_count,
)
from src.multiview_sync import global_frame_range, local_frame_for_global, synchronized_frame_map


def test_stride_sampling_preserves_output_frame_count() -> None:
    source_frames = 719
    stride = 10
    sample_count = _target_sample_count(source_frames, max_frames=None, stride=stride)
    repeats = [_repeat_count(frame_idx, source_frames, stride) for frame_idx in range(0, source_frames, stride)]

    assert sample_count == 72
    assert len(repeats) == sample_count
    assert sum(repeats) == source_frames
    assert repeats[-1] == 9


def test_sparse_sampling_disables_temporal_smoothing_by_default() -> None:
    assert _effective_smoothing_window(configured_window=5, stride=20, override=None) == 1
    assert _effective_smoothing_window(configured_window=5, stride=1, override=None) == 5
    assert _effective_smoothing_window(configured_window=5, stride=20, override=3) == 3


def test_repeated_arrays_match_video_timeline_length() -> None:
    sampled = np.arange(3 * 2 * 1).reshape(3, 2, 1)
    repeated = _repeat_array_for_video(sampled, repeats=[2, 2, 1])

    assert repeated.shape == (5, 2, 1)
    np.testing.assert_array_equal(repeated[0], sampled[0])
    np.testing.assert_array_equal(repeated[1], sampled[0])
    np.testing.assert_array_equal(repeated[2], sampled[1])
    np.testing.assert_array_equal(repeated[4], sampled[2])


def test_frame_offsets_define_common_global_timeline() -> None:
    frame_counts = {"cam_a": 5, "cam_b": 6}
    offsets = {"cam_a": 0, "cam_b": 2}

    synced = list(global_frame_range(frame_counts, offsets))

    assert synced == [2, 3, 4]
    assert local_frame_for_global("cam_a", 2, offsets) == 2
    assert local_frame_for_global("cam_b", 2, offsets) == 0


def test_timestamp_sync_handles_different_camera_fps_without_drift() -> None:
    frames = synchronized_frame_map(
        frame_counts={"cam_a": 301, "cam_b": 300},
        fps_by_camera={"cam_a": 30.0, "cam_b": 29.97},
        target_fps=30.0,
    )

    assert frames
    last = frames[-1]
    time_a = last.local_frame_indices["cam_a"] / 30.0
    time_b = last.local_frame_indices["cam_b"] / 29.97
    assert abs(time_a - time_b) <= 1.0 / 29.97
