from __future__ import annotations

import numpy as np

from src.biomechanics_3d import center_of_mass_proxy
from src.scoring_readiness import biomechanics_timeseries, build_scoring_readiness, movement_segments, torso_lean_deg


def _body_frame(offset: float = 0.0) -> np.ndarray:
    frame = np.full((133, 3), np.nan, dtype=float)
    frame[5] = [-0.5 + offset, 0.0, 2.0]
    frame[6] = [0.5 + offset, 0.0, 2.0]
    frame[7] = [-1.0 + offset, 0.0, 1.5]
    frame[8] = [1.0 + offset, 0.0, 1.5]
    frame[9] = [-1.5 + offset, 0.0, 1.0]
    frame[10] = [1.5 + offset, 0.0, 1.0]
    frame[11] = [-0.35 + offset, 0.0, 1.0]
    frame[12] = [0.35 + offset, 0.0, 1.0]
    frame[13] = [-0.35 + offset, 0.0, 0.5]
    frame[14] = [0.35 + offset, 0.0, 0.5]
    frame[15] = [-0.35 + offset, 0.0, 0.0]
    frame[16] = [0.35 + offset, 0.0, 0.0]
    return frame


def test_biomechanics_timeseries_outputs_body_angles() -> None:
    keypoints = np.stack([_body_frame(), _body_frame(0.1)])

    rows = biomechanics_timeseries(keypoints, fps=30.0)

    assert len(rows) == 2
    assert rows[0]["left_knee_deg"] == 180.0
    assert rows[0]["right_knee_deg"] == 180.0
    assert "torso_lean_deg" in rows[0]
    assert rows[1]["left_ankle_speed"] > 0


def test_scoring_readiness_marks_good_body_frames_ready() -> None:
    keypoints = np.stack([_body_frame(i * 0.05) for i in range(6)])
    shape = keypoints.shape[:2]
    reprojection = np.full(shape, 3.0)
    used_cameras = np.full(shape, 3.0)
    score = np.full(shape, 0.9)

    result = build_scoring_readiness(keypoints, score, reprojection, used_cameras, fps=30.0)

    assert result.report["scoring_ready_frame_ratio"] == 1.0
    assert result.report["reliable_body17_joint_count"] >= 12
    assert result.frame_quality_rows[0]["ready_for_scoring"] is True


def test_movement_segments_returns_candidates() -> None:
    keypoints = np.stack([_body_frame(0.0), _body_frame(0.0), _body_frame(0.5), _body_frame(1.0), _body_frame(1.5), _body_frame(1.5)])

    rows = movement_segments(keypoints, fps=30.0, min_segment_frames=2)

    assert rows[0]["label"] in {"motion_candidate", "pending_motion"}
    assert "status" in rows[0]


def test_center_of_mass_proxy_handles_all_nan_selection() -> None:
    keypoints = np.full((133, 3), np.nan, dtype=float)

    center = center_of_mass_proxy(keypoints, [5, 6, 11, 12])

    assert np.isnan(center).all()


def test_torso_lean_preserves_forward_backward_direction() -> None:
    frame = _body_frame()
    frame[5:7, 1] = 0.3
    forward = torso_lean_deg(frame)
    frame[5:7, 1] = -0.3
    backward = torso_lean_deg(frame)

    assert forward > 0
    assert backward < 0


def test_center_of_mass_proxy_uses_anatomical_weights() -> None:
    keypoints = _body_frame()
    weighted = center_of_mass_proxy(keypoints, [5, 6, 11, 12])
    unweighted = np.mean(keypoints[[5, 6, 11, 12]], axis=0)

    assert weighted[2] != unweighted[2]
