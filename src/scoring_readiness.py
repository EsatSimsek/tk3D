from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .biomechanics_3d import angle_deg, segment_length
from .validation_3d import _safe_nanmean

COCO_BODY_JOINTS: dict[str, int] = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}

ANGLE_SPECS: dict[str, tuple[str, str, str]] = {
    "left_elbow_deg": ("left_shoulder", "left_elbow", "left_wrist"),
    "right_elbow_deg": ("right_shoulder", "right_elbow", "right_wrist"),
    "left_shoulder_deg": ("left_elbow", "left_shoulder", "left_hip"),
    "right_shoulder_deg": ("right_elbow", "right_shoulder", "right_hip"),
    "left_hip_deg": ("left_shoulder", "left_hip", "left_knee"),
    "right_hip_deg": ("right_shoulder", "right_hip", "right_knee"),
    "left_knee_deg": ("left_hip", "left_knee", "left_ankle"),
    "right_knee_deg": ("right_hip", "right_knee", "right_ankle"),
}

BONE_SPECS: dict[str, tuple[str, str]] = {
    "shoulder_width": ("left_shoulder", "right_shoulder"),
    "hip_width": ("left_hip", "right_hip"),
    "left_upper_arm": ("left_shoulder", "left_elbow"),
    "right_upper_arm": ("right_shoulder", "right_elbow"),
    "left_forearm": ("left_elbow", "left_wrist"),
    "right_forearm": ("right_elbow", "right_wrist"),
    "left_thigh": ("left_hip", "left_knee"),
    "right_thigh": ("right_hip", "right_knee"),
    "left_shin": ("left_knee", "left_ankle"),
    "right_shin": ("right_knee", "right_ankle"),
}


@dataclass(slots=True)
class ReadinessResult:
    frame_quality_rows: list[dict[str, Any]]
    joint_quality_rows: list[dict[str, Any]]
    biomechanics_rows: list[dict[str, Any]]
    segment_rows: list[dict[str, Any]]
    report: dict[str, Any]


def build_scoring_readiness(
    keypoints_3d: np.ndarray,
    triangulation_score: np.ndarray | None = None,
    reprojection_error: np.ndarray | None = None,
    used_cameras: np.ndarray | None = None,
    fps: float = 30.0,
    max_reprojection_error_px: float = 25.0,
) -> ReadinessResult:
    keypoints_3d = np.asarray(keypoints_3d, dtype=float)
    frame_quality_rows = frame_quality(keypoints_3d, triangulation_score, reprojection_error, used_cameras, max_reprojection_error_px)
    joint_quality_rows = joint_quality(keypoints_3d, triangulation_score, reprojection_error, used_cameras, max_reprojection_error_px)
    biomechanics_rows = biomechanics_timeseries(keypoints_3d, fps=fps)
    segment_rows = movement_segments(keypoints_3d, fps=fps)
    report = readiness_report(frame_quality_rows, joint_quality_rows, biomechanics_rows, segment_rows)
    return ReadinessResult(frame_quality_rows, joint_quality_rows, biomechanics_rows, segment_rows, report)


def frame_quality(
    keypoints_3d: np.ndarray,
    triangulation_score: np.ndarray | None,
    reprojection_error: np.ndarray | None,
    used_cameras: np.ndarray | None,
    max_reprojection_error_px: float,
) -> list[dict[str, Any]]:
    valid_xyz = np.all(np.isfinite(keypoints_3d), axis=-1)
    rows: list[dict[str, Any]] = []
    for frame_idx in range(keypoints_3d.shape[0]):
        row = {
            "frame_idx": frame_idx,
            "valid_joint_ratio": float(np.mean(valid_xyz[frame_idx])) if valid_xyz.shape[1] else 0.0,
            "valid_body17_ratio": _body17_ratio(valid_xyz[frame_idx]),
            "mean_triangulation_score": _row_nanmean(triangulation_score, frame_idx, positive_only=True),
            "mean_reprojection_error_px": _row_nanmean(reprojection_error, frame_idx),
            "mean_used_cameras": _row_nanmean(used_cameras, frame_idx, positive_only=True),
        }
        row["ready_for_scoring"] = bool(
            row["valid_body17_ratio"] >= 0.70
            and (not np.isfinite(row["mean_reprojection_error_px"]) or row["mean_reprojection_error_px"] <= max_reprojection_error_px)
            and (not np.isfinite(row["mean_used_cameras"]) or row["mean_used_cameras"] >= 2.0)
        )
        rows.append(row)
    return rows


def joint_quality(
    keypoints_3d: np.ndarray,
    triangulation_score: np.ndarray | None,
    reprojection_error: np.ndarray | None,
    used_cameras: np.ndarray | None,
    max_reprojection_error_px: float,
) -> list[dict[str, Any]]:
    valid_xyz = np.all(np.isfinite(keypoints_3d), axis=-1)
    rows = []
    for joint_idx in range(keypoints_3d.shape[1]):
        row = {
            "joint_idx": joint_idx,
            "joint_name": _joint_name(joint_idx),
            "valid_ratio": float(np.mean(valid_xyz[:, joint_idx])) if keypoints_3d.shape[0] else 0.0,
            "mean_triangulation_score": _col_nanmean(triangulation_score, joint_idx, positive_only=True),
            "mean_reprojection_error_px": _col_nanmean(reprojection_error, joint_idx),
            "mean_used_cameras": _col_nanmean(used_cameras, joint_idx, positive_only=True),
        }
        row["ready_for_scoring"] = bool(
            row["valid_ratio"] >= 0.70
            and (not np.isfinite(row["mean_reprojection_error_px"]) or row["mean_reprojection_error_px"] <= max_reprojection_error_px)
        )
        rows.append(row)
    return rows


def biomechanics_timeseries(keypoints_3d: np.ndarray, fps: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    velocities = joint_speed(keypoints_3d, fps=fps)
    for frame_idx, frame in enumerate(keypoints_3d):
        row: dict[str, Any] = {"frame_idx": frame_idx, "timestamp_sec": frame_idx / fps if fps > 0 else 0.0}
        for metric_name, (a_name, b_name, c_name) in ANGLE_SPECS.items():
            row[metric_name] = angle_deg(frame[_idx(a_name)], frame[_idx(b_name)], frame[_idx(c_name)])
        for metric_name, (a_name, b_name) in BONE_SPECS.items():
            row[f"{metric_name}_len"] = segment_length(frame[_idx(a_name)], frame[_idx(b_name)])
        row["torso_lean_deg"] = torso_lean_deg(frame)
        row["left_wrist_speed"] = _joint_speed_value(velocities, frame_idx, _idx("left_wrist"))
        row["right_wrist_speed"] = _joint_speed_value(velocities, frame_idx, _idx("right_wrist"))
        row["left_ankle_speed"] = _joint_speed_value(velocities, frame_idx, _idx("left_ankle"))
        row["right_ankle_speed"] = _joint_speed_value(velocities, frame_idx, _idx("right_ankle"))
        row["body_motion_energy"] = _safe_nanmean_1d(velocities[frame_idx]) if frame_idx < velocities.shape[0] else float("nan")
        rows.append(row)
    return rows


def movement_segments(keypoints_3d: np.ndarray, fps: float, min_segment_frames: int = 3) -> list[dict[str, Any]]:
    speeds = joint_speed(keypoints_3d, fps=fps)
    energy = _nanmean_axis1(speeds[:, :17]) if speeds.size else np.array([])
    if energy.size == 0 or not np.any(np.isfinite(energy)):
        return [{"segment_id": 0, "label": "pending_motion", "start_frame": 0, "end_frame": 0, "status": "not_enough_motion_data"}]
    threshold = float(np.nanpercentile(energy, 60))
    active = np.isfinite(energy) & (energy >= threshold) & (energy > 0)
    rows: list[dict[str, Any]] = []
    start: int | None = None
    for frame_idx, is_active in enumerate(active.tolist() + [False]):
        if is_active and start is None:
            start = frame_idx
        elif not is_active and start is not None:
            end = frame_idx - 1
            if end - start + 1 >= min_segment_frames:
                rows.append(
                    {
                        "segment_id": len(rows),
                        "label": "motion_candidate",
                        "start_frame": start,
                        "end_frame": end,
                        "start_time_sec": start / fps if fps > 0 else 0.0,
                        "end_time_sec": end / fps if fps > 0 else 0.0,
                        "mean_motion_energy": float(np.nanmean(energy[start : end + 1])),
                        "status": "needs_poomsae_label",
                    }
                )
            start = None
    if not rows:
        rows.append({"segment_id": 0, "label": "pending_motion", "start_frame": 0, "end_frame": int(energy.size - 1), "status": "no_stable_segment_found"})
    return rows


def readiness_report(
    frame_rows: list[dict[str, Any]],
    joint_rows: list[dict[str, Any]],
    biomechanics_rows: list[dict[str, Any]],
    segment_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    ready_frames = [row for row in frame_rows if row["ready_for_scoring"]]
    ready_joints = [row for row in joint_rows[:17] if row["ready_for_scoring"]]
    warnings: list[str] = []
    if frame_rows and len(ready_frames) / len(frame_rows) < 0.70:
        warnings.append("too_few_scoring_ready_frames")
    if len(ready_joints) < 12:
        warnings.append("too_few_reliable_body17_joints")
    if not segment_rows or segment_rows[0].get("status") in {"not_enough_motion_data", "no_stable_segment_found"}:
        warnings.append("movement_segments_need_review")
    return {
        "status": "ready_for_scoring_infrastructure" if not warnings else "needs_review_before_scoring",
        "frame_count": len(frame_rows),
        "scoring_ready_frame_ratio": len(ready_frames) / len(frame_rows) if frame_rows else 0.0,
        "reliable_body17_joint_count": len(ready_joints),
        "biomechanics_metric_count": len(biomechanics_rows[0]) - 2 if biomechanics_rows else 0,
        "movement_segment_count": len(segment_rows),
        "warnings": warnings,
        "next_step": "Use real poomsae videos to label movement segments before numeric scoring." if not warnings else "Inspect quality CSVs, smoothing output, and motion segments before enabling scoring.",
    }


def joint_speed(keypoints_3d: np.ndarray, fps: float) -> np.ndarray:
    if keypoints_3d.shape[0] == 0:
        return np.empty((0, keypoints_3d.shape[1]), dtype=float)
    speed = np.full(keypoints_3d.shape[:2], np.nan, dtype=float)
    if keypoints_3d.shape[0] == 1:
        return speed
    diffs = np.linalg.norm(np.diff(keypoints_3d, axis=0), axis=-1) * max(fps, 0.0)
    speed[1:] = diffs
    return speed


def torso_lean_deg(frame: np.ndarray) -> float:
    shoulder_center = _mean_points(frame, ["left_shoulder", "right_shoulder"])
    hip_center = _mean_points(frame, ["left_hip", "right_hip"])
    if not np.all(np.isfinite(shoulder_center)) or not np.all(np.isfinite(hip_center)):
        return float("nan")
    torso = shoulder_center - hip_center
    denom = np.linalg.norm(torso)
    if denom == 0 or not np.isfinite(denom):
        return float("nan")
    vertical = np.array([0.0, 0.0, 1.0], dtype=float)
    cosine = np.clip(np.dot(torso, vertical) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(abs(cosine))))




def _safe_nanmean_1d(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))

def _nanmean_axis1(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    counts = np.sum(finite, axis=1)
    sums = np.nansum(values, axis=1)
    return np.divide(sums, counts, out=np.full(values.shape[0], np.nan, dtype=float), where=counts > 0)

def _body17_ratio(valid_mask: np.ndarray) -> float:
    return float(np.mean(valid_mask[:17])) if valid_mask.size >= 17 else 0.0


def _idx(name: str) -> int:
    return COCO_BODY_JOINTS[name]


def _joint_name(joint_idx: int) -> str:
    for name, idx in COCO_BODY_JOINTS.items():
        if idx == joint_idx:
            return name
    return f"wholebody_{joint_idx}"


def _mean_points(frame: np.ndarray, names: list[str]) -> np.ndarray:
    points = np.asarray([frame[_idx(name)] for name in names], dtype=float)
    if not np.any(np.isfinite(points)):
        return np.array([np.nan, np.nan, np.nan], dtype=float)
    return np.nanmean(points, axis=0)


def _row_nanmean(values: np.ndarray | None, row_idx: int, positive_only: bool = False) -> float:
    if values is None:
        return float("nan")
    row = np.asarray(values[row_idx], dtype=float)
    if positive_only:
        row = row[row > 0]
    return _safe_nanmean_1d(row)


def _col_nanmean(values: np.ndarray | None, col_idx: int, positive_only: bool = False) -> float:
    if values is None:
        return float("nan")
    col = np.asarray(values[:, col_idx], dtype=float)
    if positive_only:
        col = col[col > 0]
    return _safe_nanmean_1d(col)


def _joint_speed_value(speed: np.ndarray, frame_idx: int, joint_idx: int) -> float:
    if frame_idx >= speed.shape[0] or joint_idx >= speed.shape[1]:
        return float("nan")
    return float(speed[frame_idx, joint_idx])


