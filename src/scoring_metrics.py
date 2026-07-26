import numpy as np

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
    vertical = torso[1]      # y component (up)
    forward = torso[2]       # z component (forward-back)

    return float(np.degrees(np.arctan2(abs(forward), abs(vertical))))