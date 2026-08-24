"""Hold-based movement segmentation for Poomsae recordings.

The older ``src.scoring_readiness.movement_segments`` answers a different question --
"is there enough motion in this recording to work with at all?" -- and it answers it
by finding bursts of high motion energy. That is the wrong shape for Poomsae. A
movement is not a burst: it is a preparation, an execution and then a *fixation*, a
held posture. The burst detector fires only on the fast middle, so it splits one
movement into several fragments and loses the very frames the measurements need.

This module inverts the question. In Poomsae every movement **ends** in a held
posture, so the quiet moments are the structure:

    hold, hold, hold  ->  the movement ends here, here and here
    between two holds ->  exactly one movement

Measured against the hand-labelled M01-M06 recording (the only ground truth that
exists today):

===========================================  ==================  ==============
method                                       mean span error     full coverage
===========================================  ==================  ==============
burst detector (``movement_segments``)       211 frames (3.5 s)  0/6
holds, cut at the midpoint between them       35 frames (0.6 s)  1/6
holds, cut where motion resumes (this one)    22 frames (0.4 s)  3/6
===========================================  ==================  ==============

Fixation instants themselves land within 12.5 frames on average.

**This is not accurate enough to replace hand labelling.** The fixation measurement
window is +/-5 frames, so a 12-frame error can still measure the wrong instant. It is
offered as a strong starting proposal for a human to correct, and as the segment
source for automatic alignment once a full 18-movement recording exists. Every
threshold below is derived from the recording itself or from the known movement
count; none was tuned to make these numbers look good, because with only six
examples a tuned constant would be fitted noise rather than a finding.
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from src.data_structures import COCO_BODY_JOINT_INDICES
from src.poomsae_scoring.contracts import ScoringContractError

SMOOTHING_FRAMES = 15
MIN_HOLD_SEPARATION_FRAMES = 20


def body_motion_signal(
    keypoints_3d: np.ndarray,
    fps: float,
    *,
    smoothing_frames: int = SMOOTHING_FRAMES,
) -> np.ndarray:
    """Mean body-joint speed per frame, smoothed enough to survive single-frame noise."""
    from src.scoring_readiness import joint_speed

    if fps <= 0:
        raise ScoringContractError("fps must be positive")
    if smoothing_frames < 1:
        raise ScoringContractError("smoothing_frames must be at least 1")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        speeds = joint_speed(keypoints_3d, fps=fps)
        body = [index for index in COCO_BODY_JOINT_INDICES if index < speeds.shape[1]]
        if not body or speeds.size == 0:
            raise ScoringContractError("pose has no usable body joints for segmentation")
        energy = np.nanmean(speeds[:, body], axis=1)
    energy = np.nan_to_num(energy, nan=0.0)
    kernel = np.ones(smoothing_frames) / smoothing_frames
    return np.convolve(energy, kernel, mode="same")


def find_hold_frames(
    signal: np.ndarray,
    expected_movement_count: int,
    *,
    min_separation_frames: int = MIN_HOLD_SEPARATION_FRAMES,
) -> list[int]:
    """The frames where the body settles: one per expected movement.

    Three rules, none of them a tuned constant:

    1. A hold is a local minimum of the motion signal. The search window is clipped
       at the ends rather than skipped, because the final movement's fixation sits
       near the end of the recording and an edge-skipping search never sees it.
    2. A trough above the recording's own median speed is a mid-movement hesitation,
       not a held posture, so it is dropped.
    3. Of what remains, the last ``expected_movement_count`` are kept. Preparation
       always comes first, and an athlete waiting to begin produces the stillest
       moment in the whole recording -- selecting by depth would pick that ready
       stance over a real fixation.
    """
    if expected_movement_count < 1:
        raise ScoringContractError("expected_movement_count must be at least 1")
    if min_separation_frames < 1:
        raise ScoringContractError("min_separation_frames must be at least 1")
    if signal.ndim != 1 or signal.size == 0:
        raise ScoringContractError("motion signal must be a non-empty 1-D array")

    troughs: list[int] = []
    for index in range(signal.size):
        low = max(0, index - min_separation_frames)
        high = min(signal.size, index + min_separation_frames + 1)
        is_trough = signal[index] == signal[low:high].min()
        if is_trough and (not troughs or index - troughs[-1] > min_separation_frames):
            troughs.append(index)

    median = float(np.median(signal))
    held = [frame for frame in troughs if signal[frame] < median]
    return held[-expected_movement_count:]


def detect_movement_segments(
    keypoints_3d: np.ndarray,
    fps: float,
    expected_movement_count: int,
    *,
    smoothing_frames: int = SMOOTHING_FRAMES,
    min_separation_frames: int = MIN_HOLD_SEPARATION_FRAMES,
) -> dict[str, Any]:
    """Propose one frame span per movement, plus the fixation frame inside each.

    A movement runs from where motion resumes after the previous hold up to where it
    resumes after its own hold. The boundary is the resumption rather than the
    midpoint between two holds because the athlete holds the posture, waits, and only
    then begins the next movement; the midpoint lands inside that wait.

    Returns ``segments`` shaped for :func:`build_automatic_timeline_report` (each with
    ``segment_id``, ``start_frame``, ``end_frame``) with the proposed ``fixation_frame``
    alongside, and the ``signal`` so a caller can plot or re-check the decision.
    """
    signal = body_motion_signal(keypoints_3d, fps, smoothing_frames=smoothing_frames)
    holds = find_hold_frames(
        signal, expected_movement_count, min_separation_frames=min_separation_frames
    )
    if not holds:
        raise ScoringContractError(
            "no held posture found; the recording may contain no completed movement"
        )
    median = float(np.median(signal))

    def resumes_after(frame: int) -> int:
        for index in range(frame, signal.size):
            if signal[index] > median:
                return index
        return signal.size - 1

    edges = [0] + [resumes_after(hold) for hold in holds[:-1]] + [signal.size - 1]
    segments = [
        {
            "segment_id": index,
            "start_frame": int(edges[index]),
            "end_frame": int(edges[index + 1]),
            "fixation_frame": int(holds[index]),
        }
        for index in range(len(holds))
    ]
    return {
        "status": "hold_based_segment_proposal",
        "label_source": "automatic_proposal_requires_human_confirmation",
        "expected_movement_count": expected_movement_count,
        "proposed_segment_count": len(segments),
        "segments": segments,
        "signal": signal,
        "median_speed": median,
        "accuracy_note": (
            "Measured against the M01-M06 hand-labelled recording: mean span error "
            "22 frames, fixation error 12.5 frames. The fixation measurement window is "
            "+/-5 frames, so these spans are a proposal for human confirmation, not a "
            "replacement for hand labelling."
        ),
    }
