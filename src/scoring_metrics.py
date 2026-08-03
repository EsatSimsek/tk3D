import numpy as np

from src.coordinate_system import ANALYSIS_FORWARD_AXIS, ANALYSIS_RIGHT_AXIS, ANALYSIS_UP_AXIS
from src.data_structures import COCO_BODY_JOINTS


def _mean_point(frame, joint_names):
    """Average the given joints. Returns NaN if any is missing."""
    idxs = [COCO_BODY_JOINTS[name] for name in joint_names]
    pts = frame[idxs]
    if not np.all(np.isfinite(pts)):
        return np.full(3, np.nan)
    return pts.mean(axis=0)

def torso_lean_from_vertical_deg(frame):
    """Angle between the torso and the vertical axis, in the forward-back plane.

    0 degrees means perfectly upright. Larger means more lean.
    """
    shoulder = _mean_point(frame, ["left_shoulder", "right_shoulder"])
    hip = _mean_point(frame, ["left_hip", "right_hip"])
    if not np.all(np.isfinite(shoulder)) or not np.all(np.isfinite(hip)):
        return float("nan")

    torso = shoulder - hip
    vertical = torso[ANALYSIS_UP_AXIS]
    forward = torso[ANALYSIS_FORWARD_AXIS]

    return float(np.degrees(np.arctan2(abs(forward), abs(vertical))))


def stance_width_lateral(frame):
    """Left-right distance between the two ankles (x axis)."""
    left_ankle = frame[COCO_BODY_JOINTS["left_ankle"]]
    right_ankle = frame[COCO_BODY_JOINTS["right_ankle"]]
    if not np.all(np.isfinite(left_ankle)) or not np.all(np.isfinite(right_ankle)):
        return float("nan")
    return float(abs(left_ankle[ANALYSIS_RIGHT_AXIS] - right_ankle[ANALYSIS_RIGHT_AXIS]))


def stance_length_forward(frame):
    """Front-back distance between the two ankles (y axis)."""
    left_ankle = frame[COCO_BODY_JOINTS["left_ankle"]]
    right_ankle = frame[COCO_BODY_JOINTS["right_ankle"]]
    if not np.all(np.isfinite(left_ankle)) or not np.all(np.isfinite(right_ankle)):
        return float("nan")
    return float(abs(left_ankle[ANALYSIS_FORWARD_AXIS] - right_ankle[ANALYSIS_FORWARD_AXIS]))

def left_knee_ankle_alignment(frame):
    """How far the left knee sits sideways from the left ankle (x axis). 0 = aligned."""
    knee = frame[COCO_BODY_JOINTS["left_knee"]]
    ankle = frame[COCO_BODY_JOINTS["left_ankle"]]
    if not np.all(np.isfinite(knee)) or not np.all(np.isfinite(ankle)):
        return float("nan")
    return float(abs(knee[ANALYSIS_RIGHT_AXIS] - ankle[ANALYSIS_RIGHT_AXIS]))

def right_knee_ankle_alignment(frame):
    """How far the right knee sits sideways from the right ankle (x axis). 0 = aligned."""
    knee = frame[COCO_BODY_JOINTS["right_knee"]]
    ankle = frame[COCO_BODY_JOINTS["right_ankle"]]
    if not np.all(np.isfinite(knee)) or not np.all(np.isfinite(ankle)):
        return float("nan")
    return float(abs(knee[ANALYSIS_RIGHT_AXIS] - ankle[ANALYSIS_RIGHT_AXIS]))