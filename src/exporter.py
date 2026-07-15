from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def export_session_json(payload: dict[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(_json_ready(payload), file, indent=2)


def export_keypoints3d_csv(
    keypoints_3d_world: np.ndarray,
    output_path: str | Path,
    frame_indices: np.ndarray | None = None,
    timestamps_sec: np.ndarray | None = None,
) -> None:
    frame_count = keypoints_3d_world.shape[0]
    indices = np.arange(frame_count, dtype=int) if frame_indices is None else np.asarray(frame_indices, dtype=int)
    timestamps = None if timestamps_sec is None else np.asarray(timestamps_sec, dtype=float)
    if indices.shape != (frame_count,):
        raise ValueError(f"frame_indices must have shape {(frame_count,)}, got {indices.shape}")
    if timestamps is not None and timestamps.shape != (frame_count,):
        raise ValueError(f"timestamps_sec must have shape {(frame_count,)}, got {timestamps.shape}")
    rows = []
    for array_idx in range(frame_count):
        for joint_idx in range(keypoints_3d_world.shape[1]):
            x, y, z = keypoints_3d_world[array_idx, joint_idx]
            row = {
                "frame_idx": int(indices[array_idx]),
                "joint_idx": joint_idx,
                "x_m": x,
                "y_m": y,
                "z_m": z,
            }
            if timestamps is not None:
                row["timestamp_sec"] = float(timestamps[array_idx])
            rows.append(row)
    _write_csv(rows, output_path)


def export_keypoints2d_csv(poses_2d_by_frame: dict[int, dict[str, Any]], output_path: str | Path) -> None:
    rows = []
    for frame_idx, poses_by_camera in poses_2d_by_frame.items():
        for camera_id, pose in poses_by_camera.items():
            for joint_idx in range(pose.keypoints_xy.shape[0]):
                x, y = pose.keypoints_xy[joint_idx]
                rows.append(
                    {
                        "frame_idx": frame_idx,
                        "camera_id": camera_id,
                        "joint_idx": joint_idx,
                        "x": x,
                        "y": y,
                        "score": pose.scores[joint_idx],
                        "valid": bool(pose.valid_mask[joint_idx]),
                    }
                )
    _write_csv(rows, output_path)


def export_quality_csv(
    triangulation_score: np.ndarray,
    reprojection_error: np.ndarray,
    used_cameras: np.ndarray,
    output_path: str | Path,
) -> None:
    rows = []
    for frame_idx in range(triangulation_score.shape[0]):
        for joint_idx in range(triangulation_score.shape[1]):
            rows.append(
                {
                    "frame_idx": frame_idx,
                    "joint_idx": joint_idx,
                    "triangulation_score": triangulation_score[frame_idx, joint_idx],
                    "reprojection_error_px": reprojection_error[frame_idx, joint_idx],
                    "used_cameras": used_cameras[frame_idx, joint_idx],
                }
            )
    _write_csv(rows, output_path)


def export_validation_csv(frame_valid_ratio: np.ndarray, output_path: str | Path) -> None:
    rows = [
        {"frame_idx": frame_idx, "valid_ratio": float(value)}
        for frame_idx, value in enumerate(frame_valid_ratio)
    ]
    _write_csv(rows, output_path)


def export_joint_validation_csv(joint_valid_ratio: np.ndarray, output_path: str | Path) -> None:
    rows = [
        {"joint_idx": joint_idx, "valid_ratio": float(value)}
        for joint_idx, value in enumerate(joint_valid_ratio)
    ]
    _write_csv(rows, output_path)


def export_placeholder_steps_csv(output_path: str | Path) -> None:
    _write_csv(
        [
            {
                "step_id": None,
                "step_name": "pending_phase_detection",
                "status": "not_computed",
            }
        ],
        output_path,
    )


def export_excel(summary: dict[str, Any], csv_paths: dict[str, Path], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame([summary]).to_excel(writer, sheet_name="summary", index=False)
        for sheet_name, csv_path in csv_paths.items():
            if csv_path.exists():
                pd.read_csv(csv_path).to_excel(writer, sheet_name=sheet_name[:31], index=False)


def _write_csv(rows: list[dict[str, Any]], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(_csv_ready(rows)).to_csv(path, index=False, na_rep="")


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _json_ready(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


def _csv_ready(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: _csv_cell_ready(value) for key, value in row.items()} for row in rows]


def _csv_cell_ready(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
