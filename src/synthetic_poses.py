import numpy as np


def build_leg(knee_angle_deg):
    """Build a single left leg with the requested knee angle."""
    hip = np.array([0.0, 0.9, 0.0])
    thigh_length = 0.4
    shank_length = 0.4

    knee = np.array([0.0, hip[1] - thigh_length, 0.0])

    flexion_deg = 180 - knee_angle_deg
    flexion_rad = np.radians(flexion_deg)

    down = shank_length * np.cos(flexion_rad)
    back = shank_length * np.sin(flexion_rad)

    ankle = np.array([0.0, knee[1] - down, knee[2] - back])

    return hip, knee, ankle


def build_frame(knee_angle_deg):
    """Place one leg into a full 133-joint frame. Unknown joints stay NaN."""
    frame = np.full((133, 3), np.nan)

    hip, knee, ankle = build_leg(knee_angle_deg)

    frame[11] = hip     # left_hip
    frame[13] = knee    # left_knee
    frame[15] = ankle   # left_ankle

    return frame


def build_sequence(start_angle, end_angle, frame_count):
    """Build a sequence where the knee angle moves from start to end."""
    frames = []
    for i in range(frame_count):
        t = i / (frame_count - 1)
        angle = start_angle + (end_angle - start_angle) * t
        frames.append(build_frame(angle))
    return np.stack(frames)