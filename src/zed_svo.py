from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .data_structures import CameraCalibration


@dataclass(frozen=True, slots=True)
class ZedSvoMetadata:
    path: Path
    serial_number: int
    camera_model: str
    firmware_version: int
    fps: float
    image_size: tuple[int, int]
    timestamps_ns: np.ndarray
    intrinsic_matrix: np.ndarray
    distortion_coefficients: np.ndarray
    imu_rotation: np.ndarray | None


def common_timestamp_timeline(
    timestamp_sequences: list[np.ndarray],
    fps: float,
) -> np.ndarray:
    if len(timestamp_sequences) < 2:
        raise ValueError("At least two timestamp sequences are required")
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError("fps must be finite and positive")
    normalized: list[np.ndarray] = []
    for sequence in timestamp_sequences:
        values = np.asarray(sequence, dtype=np.int64)
        if values.ndim != 1 or values.size < 2:
            raise ValueError("Every timestamp sequence must contain at least two values")
        if np.any(np.diff(values) <= 0):
            raise ValueError("Timestamps must be strictly increasing")
        normalized.append(values)
    start_ns = max(int(values[0]) for values in normalized)
    end_ns = min(int(values[-1]) for values in normalized)
    if end_ns <= start_ns:
        raise ValueError("Camera recordings have no overlapping timestamp interval")
    period_ns = 1_000_000_000.0 / float(fps)
    frame_count = int(np.floor((end_ns - start_ns) / period_ns + 1e-9)) + 1
    timeline = start_ns + np.rint(np.arange(frame_count, dtype=float) * period_ns).astype(np.int64)
    return timeline[timeline <= end_ns]


def nearest_timestamp_indices(source_timestamps_ns: np.ndarray, target_timestamps_ns: np.ndarray) -> np.ndarray:
    source = np.asarray(source_timestamps_ns, dtype=np.int64)
    target = np.asarray(target_timestamps_ns, dtype=np.int64)
    if source.ndim != 1 or source.size == 0 or np.any(np.diff(source) <= 0):
        raise ValueError("source timestamps must be a non-empty, strictly increasing vector")
    if target.ndim != 1 or target.size == 0 or np.any(np.diff(target) <= 0):
        raise ValueError("target timestamps must be a non-empty, strictly increasing vector")
    after = np.searchsorted(source, target, side="left")
    after = np.clip(after, 0, source.size - 1)
    before = np.maximum(after - 1, 0)
    before_error = np.abs(target - source[before])
    after_error = np.abs(source[after] - target)
    return np.where(before_error <= after_error, before, after).astype(np.int64)


def timestamp_mapping_report(
    source_timestamps_ns: np.ndarray,
    target_timestamps_ns: np.ndarray,
    source_indices: np.ndarray,
    fps: float,
) -> dict[str, Any]:
    source = np.asarray(source_timestamps_ns, dtype=np.int64)
    target = np.asarray(target_timestamps_ns, dtype=np.int64)
    indices = np.asarray(source_indices, dtype=np.int64)
    if target.shape != indices.shape:
        raise ValueError("target timestamps and source indices must have the same shape")
    if np.any(indices < 0) or np.any(indices >= source.size):
        raise ValueError("source index is outside the timestamp sequence")
    residual_ms = (source[indices] - target) / 1_000_000.0
    counts = np.bincount(indices, minlength=source.size)
    period_ms = 1000.0 / float(fps)
    return {
        "source_frame_count": int(source.size),
        "output_frame_count": int(target.size),
        "source_frames_unused": int(np.sum(counts == 0)),
        "source_frames_reused": int(np.sum(counts > 1)),
        "output_frames_from_reused_source": int(np.sum(np.maximum(counts - 1, 0))),
        "max_abs_timestamp_residual_ms": float(np.max(np.abs(residual_ms))),
        "median_abs_timestamp_residual_ms": float(np.median(np.abs(residual_ms))),
        "p95_abs_timestamp_residual_ms": float(np.percentile(np.abs(residual_ms), 95.0)),
        "nominal_frame_period_ms": period_ms,
        "large_residual_output_frames": int(np.sum(np.abs(residual_ms) > 0.75 * period_ms)),
        "mapping": [
            {
                "output_frame_idx": int(output_idx),
                "target_timestamp_ns": int(target[output_idx]),
                "source_frame_idx": int(source_idx),
                "source_timestamp_ns": int(source[source_idx]),
                "timestamp_residual_ms": float(residual_ms[output_idx]),
                "source_reused": bool(counts[source_idx] > 1),
            }
            for output_idx, source_idx in enumerate(indices)
        ],
    }


def absolute_camera_pose(
    fusion_camera_to_world: np.ndarray,
    imu_rotation: np.ndarray | None,
    override_gravity: bool,
) -> np.ndarray:
    pose = _finite_matrix(fusion_camera_to_world, (4, 4), "fusion_camera_to_world")
    if override_gravity:
        return pose.copy()
    if imu_rotation is None:
        raise ValueError("Fusion calibration requires an IMU rotation when override_gravity is false")
    rotation = _finite_matrix(imu_rotation, (3, 3), "imu_rotation")
    imu_transform = np.eye(4, dtype=float)
    imu_transform[:3, :3] = rotation
    return pose @ imu_transform


def calibration_from_camera_pose(
    camera_id: str,
    image_size: tuple[int, int],
    intrinsic_matrix: np.ndarray,
    distortion_coefficients: np.ndarray,
    camera_to_world: np.ndarray,
) -> CameraCalibration:
    intrinsic = _finite_matrix(intrinsic_matrix, (3, 3), "intrinsic_matrix")
    camera_pose = _finite_matrix(camera_to_world, (4, 4), "camera_to_world")
    world_to_camera_analysis = np.linalg.inv(camera_pose)
    analysis_camera_to_opencv = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    world_to_camera = analysis_camera_to_opencv @ world_to_camera_analysis
    rotation = world_to_camera[:3, :3]
    translation = world_to_camera[:3, 3]
    rotation_vector, _ = cv2.Rodrigues(rotation)
    projection = intrinsic @ world_to_camera[:3]
    return CameraCalibration(
        camera_id=camera_id,
        image_size=(int(image_size[0]), int(image_size[1])),
        intrinsic_matrix=intrinsic,
        distortion_coefficients=np.asarray(distortion_coefficients, dtype=float).reshape(-1),
        rotation_vector=rotation_vector.reshape(3),
        translation_vector=translation.reshape(3),
        projection_matrix=projection,
        reprojection_error_px=None,
    )


def _finite_matrix(value: np.ndarray, shape: tuple[int, int], name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != shape or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be finite with shape {shape}")
    return matrix
