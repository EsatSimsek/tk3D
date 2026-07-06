from __future__ import annotations

import numpy as np

from .data_structures import COCO_BODY_JOINTS
from .smoothing_3d import moving_average_nan

BODY_SEGMENT_MASS_WEIGHTS: dict[str, float] = {
    "head": 0.081,
    "torso": 0.497,
    "left_upper_arm": 0.028,
    "right_upper_arm": 0.028,
    "left_forearm_hand": 0.022,
    "right_forearm_hand": 0.022,
    "left_thigh": 0.100,
    "right_thigh": 0.100,
    "left_shank_foot": 0.061,
    "right_shank_foot": 0.061,
}
JOINT_MASS_WEIGHTS: dict[int, float] = {
    COCO_BODY_JOINTS["nose"]: BODY_SEGMENT_MASS_WEIGHTS["head"] / 5.0,
    COCO_BODY_JOINTS["left_eye"]: BODY_SEGMENT_MASS_WEIGHTS["head"] / 5.0,
    COCO_BODY_JOINTS["right_eye"]: BODY_SEGMENT_MASS_WEIGHTS["head"] / 5.0,
    COCO_BODY_JOINTS["left_ear"]: BODY_SEGMENT_MASS_WEIGHTS["head"] / 5.0,
    COCO_BODY_JOINTS["right_ear"]: BODY_SEGMENT_MASS_WEIGHTS["head"] / 5.0,
    COCO_BODY_JOINTS["left_shoulder"]: BODY_SEGMENT_MASS_WEIGHTS["torso"] / 4.0 + BODY_SEGMENT_MASS_WEIGHTS["left_upper_arm"] / 2.0,
    COCO_BODY_JOINTS["right_shoulder"]: BODY_SEGMENT_MASS_WEIGHTS["torso"] / 4.0 + BODY_SEGMENT_MASS_WEIGHTS["right_upper_arm"] / 2.0,
    COCO_BODY_JOINTS["left_elbow"]: BODY_SEGMENT_MASS_WEIGHTS["left_upper_arm"] / 2.0 + BODY_SEGMENT_MASS_WEIGHTS["left_forearm_hand"] / 2.0,
    COCO_BODY_JOINTS["right_elbow"]: BODY_SEGMENT_MASS_WEIGHTS["right_upper_arm"] / 2.0 + BODY_SEGMENT_MASS_WEIGHTS["right_forearm_hand"] / 2.0,
    COCO_BODY_JOINTS["left_wrist"]: BODY_SEGMENT_MASS_WEIGHTS["left_forearm_hand"] / 2.0,
    COCO_BODY_JOINTS["right_wrist"]: BODY_SEGMENT_MASS_WEIGHTS["right_forearm_hand"] / 2.0,
    COCO_BODY_JOINTS["left_hip"]: BODY_SEGMENT_MASS_WEIGHTS["torso"] / 4.0 + BODY_SEGMENT_MASS_WEIGHTS["left_thigh"] / 2.0,
    COCO_BODY_JOINTS["right_hip"]: BODY_SEGMENT_MASS_WEIGHTS["torso"] / 4.0 + BODY_SEGMENT_MASS_WEIGHTS["right_thigh"] / 2.0,
    COCO_BODY_JOINTS["left_knee"]: BODY_SEGMENT_MASS_WEIGHTS["left_thigh"] / 2.0 + BODY_SEGMENT_MASS_WEIGHTS["left_shank_foot"] / 2.0,
    COCO_BODY_JOINTS["right_knee"]: BODY_SEGMENT_MASS_WEIGHTS["right_thigh"] / 2.0 + BODY_SEGMENT_MASS_WEIGHTS["right_shank_foot"] / 2.0,
    COCO_BODY_JOINTS["left_ankle"]: BODY_SEGMENT_MASS_WEIGHTS["left_shank_foot"] / 2.0,
    COCO_BODY_JOINTS["right_ankle"]: BODY_SEGMENT_MASS_WEIGHTS["right_shank_foot"] / 2.0,
}


def angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom == 0 or not np.isfinite(denom):
        return float("nan")
    cosine = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def segment_length(a: np.ndarray, b: np.ndarray) -> float:
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return float("nan")
    return float(np.linalg.norm(a - b))


def joint_speed(keypoints_3d: np.ndarray, fps: float, smoothing_window: int = 5) -> np.ndarray:
    keypoints_3d = np.asarray(keypoints_3d, dtype=float)
    if keypoints_3d.shape[0] == 0:
        return np.empty(keypoints_3d.shape[:2], dtype=float)
    speed = np.full(keypoints_3d.shape[:2], np.nan, dtype=float)
    if keypoints_3d.shape[0] == 1:
        return speed
    if smoothing_window > 1 and keypoints_3d.shape[0] >= smoothing_window:
        filtered = moving_average_nan(keypoints_3d, window_size=smoothing_window)
    else:
        filtered = keypoints_3d
    valid_pairs = np.all(np.isfinite(filtered[1:]), axis=-1) & np.all(np.isfinite(filtered[:-1]), axis=-1)
    diffs = np.linalg.norm(np.diff(filtered, axis=0), axis=-1) * max(float(fps), 0.0)
    speed[1:] = np.where(valid_pairs, diffs, np.nan)
    return speed


def center_of_mass_proxy(keypoints_3d: np.ndarray, joint_indices: list[int] | None = None) -> np.ndarray:
    keypoints_3d = np.asarray(keypoints_3d, dtype=float)
    if joint_indices is not None:
        selected = keypoints_3d[joint_indices]
        if selected.size == 0 or not np.any(np.isfinite(selected)):
            return np.array([np.nan, np.nan, np.nan], dtype=float)
        weights = np.asarray([JOINT_MASS_WEIGHTS.get(int(index), 1.0) for index in joint_indices], dtype=float)
        return _weighted_average(selected, weights)

    segments = _body_segment_centers(keypoints_3d)
    if not segments:
        return np.array([np.nan, np.nan, np.nan], dtype=float)
    centers = np.asarray([center for center, _ in segments], dtype=float)
    weights = np.asarray([weight for _, weight in segments], dtype=float)
    return _weighted_average(centers, weights)


def _body_segment_centers(keypoints_3d: np.ndarray) -> list[tuple[np.ndarray, float]]:
    idx = COCO_BODY_JOINTS
    specs: list[tuple[str, tuple[int, ...]]] = [
        ("head", (idx["nose"], idx["left_eye"], idx["right_eye"], idx["left_ear"], idx["right_ear"])),
        ("torso", (idx["left_shoulder"], idx["right_shoulder"], idx["left_hip"], idx["right_hip"])),
        ("left_upper_arm", (idx["left_shoulder"], idx["left_elbow"])),
        ("right_upper_arm", (idx["right_shoulder"], idx["right_elbow"])),
        ("left_forearm_hand", (idx["left_elbow"], idx["left_wrist"])),
        ("right_forearm_hand", (idx["right_elbow"], idx["right_wrist"])),
        ("left_thigh", (idx["left_hip"], idx["left_knee"])),
        ("right_thigh", (idx["right_hip"], idx["right_knee"])),
        ("left_shank_foot", (idx["left_knee"], idx["left_ankle"])),
        ("right_shank_foot", (idx["right_knee"], idx["right_ankle"])),
    ]
    centers: list[tuple[np.ndarray, float]] = []
    for name, indices in specs:
        points = keypoints_3d[list(indices)]
        finite_rows = np.all(np.isfinite(points), axis=1)
        if not np.any(finite_rows):
            continue
        centers.append((np.mean(points[finite_rows], axis=0), BODY_SEGMENT_MASS_WEIGHTS[name]))
    return centers


def _weighted_average(points: np.ndarray, weights: np.ndarray) -> np.ndarray:
    finite_rows = np.all(np.isfinite(points), axis=1) & np.isfinite(weights) & (weights > 0)
    if not np.any(finite_rows):
        return np.array([np.nan, np.nan, np.nan], dtype=float)
    safe_points = points[finite_rows]
    safe_weights = weights[finite_rows]
    weight_sum = float(np.sum(safe_weights))
    if weight_sum <= 0.0:
        return np.array([np.nan, np.nan, np.nan], dtype=float)
    return np.average(safe_points, axis=0, weights=safe_weights).astype(float)
