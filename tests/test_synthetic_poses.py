import numpy as np

from src.biomechanics_3d import angle_deg
from src.coordinate_system import ANALYSIS_COORDINATE_SYSTEM
from src.scoring_metrics import (
    left_knee_ankle_alignment,
    pelvis_height,
    right_knee_ankle_alignment,
    stance_length_forward,
    stance_width_lateral,
    torso_lean_from_vertical_deg,
)
from src.synthetic_poses import build_frame, build_leg, build_sequence, build_torso_frame


def test_scoring_axis_contract_is_right_forward_up():
    assert ANALYSIS_COORDINATE_SYSTEM["axes"] == {"x": "right", "y": "forward", "z": "up"}


def test_build_leg_produces_requested_knee_angle():
    for requested in [180, 150, 130, 90, 45]:
        hip, knee, ankle = build_leg(requested)
        measured = angle_deg(hip, knee, ankle)
        assert abs(measured - requested) < 0.01


def test_straight_leg_ankle_position():
    hip, knee, ankle = build_leg(180)
    np.testing.assert_allclose(ankle, [0.0, 0.0, 0.1], atol=1e-6)


def test_right_angle_leg_ankle_position():
    hip, knee, ankle = build_leg(90)
    np.testing.assert_allclose(ankle, [0.0, -0.4, 0.5], atol=1e-6)


def test_frame_has_correct_shape_and_filled_joints():
    frame = build_frame(130)
    assert frame.shape == (133, 3)
    assert np.all(np.isfinite(frame[[11, 13, 15]]))
    assert np.all(np.isnan(frame[9]))


def test_sequence_shape_and_endpoints():
    seq = build_sequence(180, 90, 60)
    assert seq.shape == (60, 133, 3)
    np.testing.assert_allclose(seq[0, 15], [0.1, 0.0, 0.1], atol=1e-6)
    np.testing.assert_allclose(seq[-1, 15], [0.1, -0.4, 0.5], atol=1e-6)


def test_upright_torso_has_zero_lean():
    frame = build_torso_frame(0)
    measured = torso_lean_from_vertical_deg(frame)
    assert abs(measured) < 0.01


def test_torso_lean_matches_requested_angle():
    for requested in [12, 30, 45, 60]:
        frame = build_torso_frame(requested)
        measured = torso_lean_from_vertical_deg(frame)
        assert abs(measured - requested) < 0.01


def test_lateral_torso_offset_does_not_count_as_forward_lean():
    frame = build_torso_frame(0)
    frame[[5, 6], 0] += 0.25
    assert abs(torso_lean_from_vertical_deg(frame)) < 0.01


def test_both_legs_stay_separated_across_angles():
    for angle in [180, 130, 90, 45]:
        frame = build_frame(angle)
        assert abs(frame[15][0] - 0.1) < 1e-6    # left_ankle x stays +0.1
        assert abs(frame[16][0] - (-0.1)) < 1e-6  # right_ankle x stays -0.1


def test_stance_width_is_ankle_x_distance():
    frame = build_frame(180)
    assert abs(stance_width_lateral(frame) - 0.2) < 1e-6


def test_stance_length_is_ankle_y_distance():
    frame = np.full((133, 3), np.nan)
    frame[15] = [0.1, 0.3, 0.2]  # left_ankle: forward at y=0.3
    frame[16] = [-0.1, 0.0, 0.7]  # different height must not affect stance length
    assert abs(stance_length_forward(frame) - 0.3) < 1e-6


def test_stance_returns_nan_when_ankle_missing():
    frame = build_frame(180)
    frame[16] = [np.nan, np.nan, np.nan]   # right_ankle missing
    assert np.isnan(stance_width_lateral(frame))
    assert np.isnan(stance_length_forward(frame))


def test_knee_ankle_alignment_zero_when_aligned():
    frame = build_frame(180)
    assert abs(left_knee_ankle_alignment(frame)) < 1e-6
    assert abs(right_knee_ankle_alignment(frame)) < 1e-6

def test_knee_ankle_alignment_measures_offset():
    frame = np.full((133, 3), np.nan)
    frame[13] = [0.15, 0.5, 0.0]   # left_knee: x=0.15
    frame[15] = [0.0, 0.1, 0.0]    # left_ankle: x=0.0
    assert abs(left_knee_ankle_alignment(frame) - 0.15) < 1e-6

def test_knee_ankle_alignment_nan_when_missing():
    frame = build_frame(180)
    frame[13] = [np.nan, np.nan, np.nan]   # left_knee missing
    assert np.isnan(left_knee_ankle_alignment(frame))

def test_pelvis_height_reads_up_axis():
    frame = build_frame(180)
    assert abs(pelvis_height(frame) - 0.9) < 1e-6

def test_pelvis_height_nan_when_hip_missing():
    frame = build_frame(180)
    frame[11] = [np.nan, np.nan, np.nan]   # left_hip missing
    assert np.isnan(pelvis_height(frame))
