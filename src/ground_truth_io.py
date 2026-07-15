from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .coordinate_system import ANALYSIS_COORDINATE_SYSTEM
from .data_structures import COCO_BODY_JOINT_NAMES


@dataclass(slots=True)
class PoseSequence:
    points_m: np.ndarray
    joint_names: list[str]
    frame_indices: np.ndarray | None
    timestamps_sec: np.ndarray | None
    fps: float | None
    metadata: dict[str, Any]


@dataclass(slots=True)
class MatchedPoseSequences:
    predicted_m: np.ndarray
    ground_truth_m: np.ndarray
    joint_names: list[str]
    match_rows: list[dict[str, Any]]
    fps: float


def load_pose_sequence_json(
    path: str | Path,
    key: str,
    require_joint_names: bool = False,
) -> PoseSequence:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    coordinate_system = payload.get("coordinate_system")
    if coordinate_system != ANALYSIS_COORDINATE_SYSTEM:
        raise ValueError(
            f"{source} must declare the TK3D analysis coordinate system. "
            "Convert source coordinates to meters with x right, y forward, z up first."
        )
    if key not in payload:
        raise ValueError(f"{source} does not contain {key}")
    points = np.asarray(payload[key], dtype=float)
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError(f"{key} must have shape [frames, joints, 3], got {points.shape}")

    names = payload.get("joint_names")
    if names is None and not require_joint_names and points.shape[1] >= len(COCO_BODY_JOINT_NAMES):
        names = list(COCO_BODY_JOINT_NAMES) + [
            f"wholebody_{index}" for index in range(len(COCO_BODY_JOINT_NAMES), points.shape[1])
        ]
    if not isinstance(names, list) or len(names) != points.shape[1]:
        raise ValueError("joint_names must be a list matching the pose joint dimension")
    if len(set(str(name) for name in names)) != len(names):
        raise ValueError("joint_names must be unique")

    frame_indices = _optional_vector(payload, "frame_indices", points.shape[0], dtype=int)
    timestamps = _optional_vector(payload, "timestamps_sec", points.shape[0], dtype=float)
    fps_value = payload.get("fps", payload.get("sample_fps"))
    fps = None if fps_value is None else float(fps_value)
    if fps is not None and (not np.isfinite(fps) or fps <= 0):
        raise ValueError("fps/sample_fps must be positive")
    return PoseSequence(points, [str(name) for name in names], frame_indices, timestamps, fps, payload)


def match_pose_sequences(
    predicted: PoseSequence,
    ground_truth: PoseSequence,
    joint_map: dict[str, str] | None = None,
    max_time_delta_sec: float | None = None,
) -> MatchedPoseSequences:
    predicted_joint_indices, truth_joint_indices, common_names = _joint_indices(
        predicted.joint_names, ground_truth.joint_names, joint_map
    )
    predicted_frame_indices, truth_frame_indices, rows = _frame_matches(
        predicted, ground_truth, max_time_delta_sec
    )
    predicted_points = predicted.points_m[np.asarray(predicted_frame_indices)][:, predicted_joint_indices]
    truth_points = ground_truth.points_m[np.asarray(truth_frame_indices)][:, truth_joint_indices]
    fps = predicted.fps or ground_truth.fps
    if fps is None:
        fps = _infer_fps(predicted.timestamps_sec)
    if fps is None:
        raise ValueError("An evaluation fps is required in either sequence or inferable from timestamps")
    return MatchedPoseSequences(predicted_points, truth_points, common_names, rows, float(fps))


def load_joint_map(path: str | Path | None) -> dict[str, str] | None:
    if path is None:
        return None
    import yaml

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    mapping = payload.get("joint_map", payload)
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("Joint-map YAML must contain a non-empty joint_map mapping")
    return {str(predicted): str(truth) for predicted, truth in mapping.items()}


def _joint_indices(
    predicted_names: list[str],
    truth_names: list[str],
    joint_map: dict[str, str] | None,
) -> tuple[list[int], list[int], list[str]]:
    predicted_lookup = {name: index for index, name in enumerate(predicted_names)}
    truth_lookup = {name: index for index, name in enumerate(truth_names)}
    mapping = joint_map or {name: name for name in predicted_names if name in truth_lookup}
    unknown_predicted = [name for name in mapping if name not in predicted_lookup]
    unknown_truth = [name for name in mapping.values() if name not in truth_lookup]
    if unknown_predicted or unknown_truth:
        raise ValueError(
            f"Joint map contains unknown names; prediction={unknown_predicted}, ground_truth={unknown_truth}"
        )
    if len(mapping) < 3:
        raise ValueError("At least three mapped joints are required for ground-truth validation")
    common_names = list(mapping.keys())
    return (
        [predicted_lookup[name] for name in common_names],
        [truth_lookup[mapping[name]] for name in common_names],
        common_names,
    )


def _frame_matches(
    predicted: PoseSequence,
    truth: PoseSequence,
    max_time_delta_sec: float | None,
) -> tuple[list[int], list[int], list[dict[str, Any]]]:
    if predicted.timestamps_sec is not None and truth.timestamps_sec is not None:
        tolerance = max_time_delta_sec
        if tolerance is None:
            truth_fps = truth.fps or _infer_fps(truth.timestamps_sec)
            tolerance = 0.51 / truth_fps if truth_fps else 0.01
        if tolerance < 0:
            raise ValueError("max_time_delta_sec must be non-negative")
        return _timestamp_matches(predicted.timestamps_sec, truth.timestamps_sec, float(tolerance))

    if predicted.frame_indices is not None and truth.frame_indices is not None:
        truth_lookup = {int(frame): index for index, frame in enumerate(truth.frame_indices)}
        predicted_indices: list[int] = []
        truth_indices: list[int] = []
        rows: list[dict[str, Any]] = []
        for predicted_index, frame in enumerate(predicted.frame_indices):
            truth_index = truth_lookup.get(int(frame))
            if truth_index is not None:
                predicted_indices.append(predicted_index)
                truth_indices.append(truth_index)
                rows.append(
                    {
                        "evaluation_frame_idx": len(rows),
                        "predicted_array_idx": predicted_index,
                        "ground_truth_array_idx": truth_index,
                        "source_frame_idx": int(frame),
                    }
                )
        if not rows:
            raise ValueError("No shared frame_indices between prediction and ground truth")
        return predicted_indices, truth_indices, rows

    if predicted.points_m.shape[0] != truth.points_m.shape[0]:
        raise ValueError("Sequences need timestamps, frame_indices, or equal frame counts for synchronization")
    count = predicted.points_m.shape[0]
    rows = [
        {
            "evaluation_frame_idx": index,
            "predicted_array_idx": index,
            "ground_truth_array_idx": index,
        }
        for index in range(count)
    ]
    indices = list(range(count))
    return indices, indices, rows


def _timestamp_matches(
    predicted_times: np.ndarray,
    truth_times: np.ndarray,
    tolerance: float,
) -> tuple[list[int], list[int], list[dict[str, Any]]]:
    if np.any(np.diff(truth_times) < 0):
        raise ValueError("Ground-truth timestamps must be sorted")
    predicted_indices: list[int] = []
    truth_indices: list[int] = []
    rows: list[dict[str, Any]] = []
    used_truth: set[int] = set()
    for predicted_index, timestamp in enumerate(predicted_times):
        insertion = int(np.searchsorted(truth_times, timestamp))
        candidates = [index for index in (insertion - 1, insertion) if 0 <= index < truth_times.size]
        if not candidates:
            continue
        truth_index = min(candidates, key=lambda index: abs(float(truth_times[index] - timestamp)))
        delta = float(truth_times[truth_index] - timestamp)
        if abs(delta) > tolerance or truth_index in used_truth:
            continue
        used_truth.add(truth_index)
        predicted_indices.append(predicted_index)
        truth_indices.append(truth_index)
        rows.append(
            {
                "evaluation_frame_idx": len(rows),
                "predicted_array_idx": predicted_index,
                "ground_truth_array_idx": truth_index,
                "predicted_timestamp_sec": float(timestamp),
                "ground_truth_timestamp_sec": float(truth_times[truth_index]),
                "time_delta_ms": 1000.0 * delta,
            }
        )
    if not rows:
        raise ValueError(f"No timestamp pairs matched within {tolerance:.6f} seconds")
    return predicted_indices, truth_indices, rows


def _optional_vector(
    payload: dict[str, Any],
    key: str,
    expected_length: int,
    dtype: type[int] | type[float],
) -> np.ndarray | None:
    if key not in payload:
        return None
    values = np.asarray(payload[key], dtype=dtype)
    if values.shape != (expected_length,) or not np.all(np.isfinite(values)):
        raise ValueError(f"{key} must be a finite vector with length {expected_length}")
    return values


def _infer_fps(timestamps: np.ndarray | None) -> float | None:
    if timestamps is None or timestamps.size < 2:
        return None
    differences = np.diff(timestamps)
    differences = differences[np.isfinite(differences) & (differences > 0)]
    return float(1.0 / np.median(differences)) if differences.size else None
