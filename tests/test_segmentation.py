from __future__ import annotations

import itertools

import numpy as np
import pytest

from src.data_structures import COCO_BODY_JOINT_INDICES
from src.poomsae_scoring import ScoringContractError
from src.poomsae_scoring.segmentation import (
    body_motion_signal,
    detect_movement_segments,
    find_hold_frames,
)


def _synthetic_recording(
    *,
    movement_frames: int = 60,
    hold_frames: int = 40,
    movement_count: int = 4,
    lead_in_frames: int = 90,
) -> np.ndarray:
    """A recording shaped like a Poomsae: a still lead-in, then move/hold repeated.

    The lead-in is the athlete waiting to begin -- the stillest stretch in the whole
    recording, which is exactly the trap a depth-based hold search falls into.
    """
    body = [index for index in COCO_BODY_JOINT_INDICES if index < 133]
    frames: list[np.ndarray] = []
    position = 0.0

    def emit(count: int, step: float) -> None:
        nonlocal position
        for _ in range(count):
            position += step
            frame = np.zeros((133, 3), dtype=float)
            for joint in body:
                frame[joint] = (position, position * 0.5, 1.0)
            frames.append(frame)

    emit(lead_in_frames, 0.0)  # perfectly still: the ready stance
    for _ in range(movement_count):
        emit(movement_frames, 0.02)  # the movement
        emit(hold_frames, 0.0005)    # the fixation: nearly, but not quite, still
    return np.asarray(frames)


def test_motion_signal_is_low_during_holds_and_high_during_movement() -> None:
    keypoints = _synthetic_recording()

    signal = body_motion_signal(keypoints, fps=60.0)

    assert signal.shape == (keypoints.shape[0],)
    # Frame 120 sits inside the first movement, frame 170 inside the first hold.
    assert signal[120] > signal[170]


def test_find_hold_frames_returns_one_hold_per_expected_movement() -> None:
    keypoints = _synthetic_recording(movement_count=4)
    signal = body_motion_signal(keypoints, fps=60.0)

    holds = find_hold_frames(signal, expected_movement_count=4)

    assert len(holds) == 4
    assert holds == sorted(holds)
    assert all(0 <= frame < signal.size for frame in holds)


def test_find_hold_frames_skips_the_ready_stance_before_the_first_movement() -> None:
    """The ready stance is the stillest moment; picking by depth would choose it."""
    lead_in = 90
    keypoints = _synthetic_recording(movement_count=3, lead_in_frames=lead_in)
    signal = body_motion_signal(keypoints, fps=60.0)

    holds = find_hold_frames(signal, expected_movement_count=3)

    assert len(holds) == 3
    assert all(frame > lead_in for frame in holds), (
        f"a hold was placed inside the lead-in: {holds}"
    )


def test_find_hold_frames_sees_a_hold_at_the_very_end_of_the_recording() -> None:
    """The last movement's fixation sits near the end; an edge-skipping search loses it."""
    keypoints = _synthetic_recording(movement_count=2, hold_frames=25)
    signal = body_motion_signal(keypoints, fps=60.0)

    holds = find_hold_frames(signal, expected_movement_count=2)

    assert len(holds) == 2
    assert holds[-1] > signal.size - 40, (
        f"final hold {holds[-1]} is not near the recording end {signal.size}"
    )


def test_detect_movement_segments_proposes_one_span_per_movement() -> None:
    keypoints = _synthetic_recording(movement_count=4)

    report = detect_movement_segments(keypoints, fps=60.0, expected_movement_count=4)

    assert report["status"] == "hold_based_segment_proposal"
    assert report["proposed_segment_count"] == 4
    segments = report["segments"]
    assert [item["segment_id"] for item in segments] == [0, 1, 2, 3]

    for item in segments:
        assert item["start_frame"] <= item["fixation_frame"] <= item["end_frame"]
    # Spans must be ordered and must not overlap: the timeline contract forbids it.
    for earlier, later in itertools.pairwise(segments):
        assert earlier["end_frame"] <= later["start_frame"]


def test_detect_movement_segments_never_claims_to_replace_hand_labelling() -> None:
    keypoints = _synthetic_recording(movement_count=2)

    report = detect_movement_segments(keypoints, fps=60.0, expected_movement_count=2)

    assert report["label_source"] == "automatic_proposal_requires_human_confirmation"
    assert "human confirmation" in report["accuracy_note"]


def test_segmentation_rejects_impossible_inputs() -> None:
    keypoints = _synthetic_recording(movement_count=2)
    signal = body_motion_signal(keypoints, fps=60.0)

    with pytest.raises(ScoringContractError, match="fps must be positive"):
        body_motion_signal(keypoints, fps=0.0)
    with pytest.raises(ScoringContractError, match="expected_movement_count"):
        find_hold_frames(signal, expected_movement_count=0)
    with pytest.raises(ScoringContractError, match="non-empty"):
        find_hold_frames(np.array([]), expected_movement_count=1)
