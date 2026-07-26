import numpy as np

from src.biomechanics_3d import angle_deg
from src.synthetic_poses import build_frame, build_leg, build_sequence
from src.synthetic_poses import build_torso_frame
from src.scoring_metrics import torso_lean_from_vertical_deg

def test_build_leg_produces_requested_knee_angle():
    for requested in [180, 150, 130, 90, 45]:
        hip, knee, ankle = build_leg(requested)
        measured = angle_deg(hip, knee, ankle)
        assert abs(measured - requested) < 0.01


def test_straight_leg_ankle_position():
    hip, knee, ankle = build_leg(180)
    np.testing.assert_allclose(ankle, [0.0, 0.1, 0.0], atol=1e-6)


def test_right_angle_leg_ankle_position():
    hip, knee, ankle = build_leg(90)
    np.testing.assert_allclose(ankle, [0.0, 0.5, -0.4], atol=1e-6)


def test_frame_has_correct_shape_and_filled_joints():
    frame = build_frame(130)
    assert frame.shape == (133, 3)
    assert np.all(np.isfinite(frame[[11, 13, 15]]))
    assert np.all(np.isnan(frame[9]))


def test_sequence_shape_and_endpoints():
    seq = build_sequence(180, 90, 60)
    assert seq.shape == (60, 133, 3)
    np.testing.assert_allclose(seq[0, 15], [0.0, 0.1, 0.0], atol=1e-6)
    np.testing.assert_allclose(seq[-1, 15], [0.0, 0.5, -0.4], atol=1e-6)

def test_upright_torso_has_zero_lean():
    frame = build_torso_frame(0)
    measured = torso_lean_from_vertical_deg(frame)
    assert abs(measured) < 0.01


def test_torso_lean_matches_requested_angle():
    for requested in [12, 30, 45, 60]:
        frame = build_torso_frame(requested)
        measured = torso_lean_from_vertical_deg(frame)
        assert abs(measured - requested) < 0.01