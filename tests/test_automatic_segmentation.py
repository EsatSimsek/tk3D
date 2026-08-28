from __future__ import annotations

import numpy as np
import pytest

from src.poomsae_scoring.automatic_segmentation import (
    compare_segments_to_reference,
    detect_automatic_segments,
)
from src.poomsae_scoring.contracts import ScoringContractError


def _motion_sequence() -> np.ndarray:
    frame_count = 280
    increments = np.zeros(frame_count, dtype=float)
    # A short pre-performance disturbance followed by three real movements.
    for start, end, displacement in (
        (5, 20, 0.10),
        (80, 105, 0.25),
        (140, 165, 0.20),
        (210, 235, 0.30),
    ):
        increments[start : end + 1] = displacement / (end - start + 1)
    position = np.cumsum(increments)
    points = np.zeros((frame_count, 133, 3), dtype=float)
    points[:, :17, 0] = position[:, None]
    return points


def _movements(count: int) -> list[dict]:
    return [
        {
            "movement_id": f"M{index + 1:02d}",
            "phases": ["preparation", "execution", "fixation"],
        }
        for index in range(count)
    ]


def test_detector_discards_preperformance_motion_and_builds_ordered_phases() -> None:
    result = detect_automatic_segments(
        _motion_sequence(),
        fps=30.0,
        movements=_movements(3),
    )

    assert len(result["candidate_episodes"]) == 4
    assert result["selection"]["selected_candidate_episode_ids"] == [1, 2, 3]
    assert [segment["movement_id"] for segment in result["segments"]] == ["M01", "M02", "M03"]
    assert result["segments"][0]["start_frame"] == 79
    assert result["segments"][-1]["end_frame"] == 279
    for segment in result["segments"]:
        anchors = segment["anchors"]
        assert anchors["preparation"] <= anchors["execution"] <= anchors["fixation"]
        assert segment["label_status"] == "automatic_diagnostic"
        assert 0.0 <= segment["confidence"] <= 0.95


def test_detector_fails_closed_without_usable_motion() -> None:
    points = np.full((30, 133, 3), np.nan, dtype=float)

    with pytest.raises(ScoringContractError, match="no finite positive motion signal"):
        detect_automatic_segments(points, fps=30.0, movements=_movements(1))


def test_reference_comparison_reports_frame_and_second_errors() -> None:
    proposed = [
        {
            "movement_id": "M01",
            "start_frame": 12,
            "end_frame": 61,
            "anchors": {"preparation": 12, "execution": 42, "fixation": 55},
        }
    ]
    reference = [
        {
            "movement_id": "M01",
            "start_frame": 10,
            "end_frame": 60,
            "anchors": {"preparation": 10, "execution": 40, "fixation": 50},
        }
    ]

    comparison = compare_segments_to_reference(proposed, reference, fps=50.0)

    assert comparison["summary"]["start_boundary_mae_frames"] == 2.0
    assert comparison["summary"]["end_boundary_mae_frames"] == 1.0
    assert comparison["summary"]["phase_anchor_mae_frames"] == 3.0
    assert comparison["summary"]["phase_anchor_max_error_frames"] == 5
    fixation = comparison["movements"][0]["phases"]["fixation"]
    assert fixation["delta_frames"] == 5
    assert fixation["absolute_error_sec"] == 0.1
