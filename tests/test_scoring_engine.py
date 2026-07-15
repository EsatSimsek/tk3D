from __future__ import annotations

import numpy as np

from src.scoring_engine import build_provisional_score
from src.scoring_readiness import biomechanics_timeseries


def _body_frame(offset: float = 0.0, lean: float = 0.0) -> np.ndarray:
    frame = np.full((133, 3), np.nan, dtype=float)
    frame[5] = [-0.5 + offset, lean, 2.0]
    frame[6] = [0.5 + offset, lean, 2.0]
    frame[7] = [-1.0 + offset, lean / 2.0, 1.5]
    frame[8] = [1.0 + offset, lean / 2.0, 1.5]
    frame[9] = [-1.5 + offset, 0.0, 1.0]
    frame[10] = [1.5 + offset, 0.0, 1.0]
    frame[11] = [-0.35 + offset, 0.0, 1.0]
    frame[12] = [0.35 + offset, 0.0, 1.0]
    frame[13] = [-0.35 + offset, 0.0, 0.5]
    frame[14] = [0.35 + offset, 0.0, 0.5]
    frame[15] = [-0.35 + offset, 0.0, 0.0]
    frame[16] = [0.35 + offset, 0.0, 0.0]
    return frame


def _thresholds() -> dict[str, float]:
    return {
        "trunk_lean_warn_deg": 10.0,
        "knee_angle_front_stance_min_deg": 130.0,
        "balance_min_score": 0.70,
    }


def test_provisional_scoring_builds_explainable_frame_and_step_scores() -> None:
    points = np.stack([_body_frame(index * 0.02) for index in range(6)])
    biomechanics = biomechanics_timeseries(points, fps=30.0)
    quality = [{"ready_for_scoring": True} for _ in range(6)]
    segments = [{"label": "motion_candidate", "start_frame": 1, "end_frame": 4}]

    result = build_provisional_score(points, biomechanics, quality, segments, _thresholds())

    assert result["status"] == "provisional_not_official"
    assert result["overall_score"] is not None
    assert result["overall_score"] > 95.0
    assert len(result["frame_scores"]) == 6
    assert result["step_scores"][0]["score"] > 95.0
    assert result["step_scores"][0]["status"] == "needs_reference_label"
    assert result["errors"] == []


def test_provisional_scoring_never_scores_unreliable_frames_and_reports_lean() -> None:
    points = np.stack([_body_frame(lean=0.4) for _ in range(3)])
    biomechanics = biomechanics_timeseries(points, fps=30.0)
    quality = [{"ready_for_scoring": False} for _ in range(3)]
    segments = [{"label": "pending_motion", "start_frame": 0, "end_frame": 2}]

    result = build_provisional_score(points, biomechanics, quality, segments, _thresholds())

    assert result["overall_score"] is None
    assert all(row["score"] == 0.0 for row in result["frame_scores"])
    codes = {row["code"] for row in result["errors"]}
    assert "unreliable_3d_frame" in codes
    assert "excessive_torso_lean" in codes
