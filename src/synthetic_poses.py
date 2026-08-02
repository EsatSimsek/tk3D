import numpy as np

from src.coordinate_system import ANALYSIS_FORWARD_AXIS, ANALYSIS_RIGHT_AXIS, ANALYSIS_UP_AXIS


def build_leg(knee_angle_deg, hip_x=0.0):
    """Build a leg in TK3D coordinates (x right, y forward, z up)."""
    hip = np.zeros(3, dtype=float)
    hip[ANALYSIS_RIGHT_AXIS] = hip_x
    hip[ANALYSIS_UP_AXIS] = 0.9
    thigh_length = 0.4
    shank_length = 0.4
    knee = hip.copy()
    knee[ANALYSIS_UP_AXIS] -= thigh_length
    flexion_deg = 180 - knee_angle_deg
    flexion_rad = np.radians(flexion_deg)
    down = shank_length * np.cos(flexion_rad)
    back = shank_length * np.sin(flexion_rad)
    ankle = knee.copy()
    ankle[ANALYSIS_FORWARD_AXIS] -= back
    ankle[ANALYSIS_UP_AXIS] -= down
    return hip, knee, ankle


def build_frame(left_knee_deg, right_knee_deg=None, hip_half_width=0.1):
    """Place both legs into a full 133-joint frame. Unknown joints stay NaN."""
    if right_knee_deg is None:
        right_knee_deg = left_knee_deg
    frame = np.full((133, 3), np.nan)
    left_hip, left_knee, left_ankle = build_leg(left_knee_deg, hip_x=hip_half_width)
    right_hip, right_knee, right_ankle = build_leg(right_knee_deg, hip_x=-hip_half_width)
    frame[11] = left_hip      # left_hip
    frame[13] = left_knee     # left_knee
    frame[15] = left_ankle    # left_ankle
    frame[12] = right_hip     # right_hip
    frame[14] = right_knee    # right_knee
    frame[16] = right_ankle   # right_ankle
    return frame


def build_sequence(start_angle, end_angle, frame_count):
    """Build a sequence where the knee angle moves from start to end."""
    frames = []
    for i in range(frame_count):
        t = i / (frame_count - 1)
        angle = start_angle + (end_angle - start_angle) * t
        frames.append(build_frame(angle))
    return np.stack(frames)


def build_torso_frame(lean_deg):
    """Build a TK3D frame with the torso leaning forward by lean_deg."""
    frame = np.full((133, 3), np.nan)

    torso_length = 0.5
    hip_center_z = 0.9

    lean_rad = np.radians(lean_deg)
    up = torso_length * np.cos(lean_rad)
    forward = torso_length * np.sin(lean_rad)

    # Hips at the same height, small gap left/right
    frame[11] = [0.1, 0.0, hip_center_z]  # left_hip
    frame[12] = [-0.1, 0.0, hip_center_z]  # right_hip

    # Shoulders: torso center goes up and forward
    shoulder_center_y = forward
    shoulder_center_z = hip_center_z + up
    frame[5] = [0.1, shoulder_center_y, shoulder_center_z]  # left_shoulder
    frame[6] = [-0.1, shoulder_center_y, shoulder_center_z]  # right_shoulder

    return frame

