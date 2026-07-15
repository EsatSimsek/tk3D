from __future__ import annotations

from typing import Any

import numpy as np


ANALYSIS_COORDINATE_SYSTEM: dict[str, Any] = {
    "name": "tk3d_analysis",
    "unit": "meter",
    "axes": {"x": "right", "y": "forward", "z": "up"},
    "handedness": "right",
}


def opencv_reference_to_analysis() -> np.ndarray:
    """OpenCV camera coordinates (x right, y down, z forward) -> TK3D."""
    return np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def aist_world_to_analysis() -> np.ndarray:
    """AIST++ centimeters (x right, y up, z forward) -> TK3D meters."""
    scale = 0.01
    return np.asarray(
        [
            [scale, 0.0, 0.0, 0.0],
            [0.0, 0.0, scale, 0.0],
            [0.0, scale, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=float)
    matrix = np.asarray(transform, dtype=float)
    if values.shape[-1] != 3:
        raise ValueError(f"Expected points ending in XYZ, got {values.shape}")
    if matrix.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 transform, got {matrix.shape}")
    output = np.full_like(values, np.nan, dtype=float)
    valid = np.all(np.isfinite(values), axis=-1)
    if not np.any(valid):
        return output
    homogeneous = np.concatenate([values[valid], np.ones((int(np.sum(valid)), 1), dtype=float)], axis=1)
    converted = (matrix @ homogeneous.T).T
    nonzero_w = np.abs(converted[:, 3]) > 1e-12
    converted[nonzero_w, :3] /= converted[nonzero_w, 3, None]
    output[valid] = converted[:, :3]
    return output


def calibration_metadata(
    calibration_mode: str,
    source_coordinate_system: dict[str, Any],
    source_to_analysis: np.ndarray,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "calibration_mode": calibration_mode,
        "source_coordinate_system": source_coordinate_system,
        "analysis_coordinate_system": ANALYSIS_COORDINATE_SYSTEM,
        "source_to_analysis": np.asarray(source_to_analysis, dtype=float).tolist(),
    }


def require_source_to_analysis(metadata: dict[str, Any]) -> np.ndarray:
    raw = metadata.get("source_to_analysis")
    if raw is None:
        raise ValueError(
            "Calibration metadata does not define source_to_analysis. "
            "Re-run camera calibration/import with the current TK3D version."
        )
    transform = np.asarray(raw, dtype=float)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("Calibration source_to_analysis must be a finite 4x4 matrix")
    return transform
