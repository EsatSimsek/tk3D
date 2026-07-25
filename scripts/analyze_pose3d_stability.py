from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_structures import COCO_BODY_JOINTS
from src.pose3d_stability import (
    ANGLE_SPECS,
    joint_angles_degrees,
    pose3d_stability_metrics,
)
from src.smoothing_3d import robust_savgol_pose


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze raw-versus-stabilized 3D joint and angle stability.",
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--candidate-windows",
        default="1,5,7,9,11",
        help="Comma-separated odd Savitzky-Golay windows to compare.",
    )
    parser.add_argument("--polynomial-order", type=int, default=2)
    parser.add_argument("--min-outlier-distance-m", type=float, default=0.04)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = (ROOT / run_dir).resolve()
    raw_path = run_dir / "csv" / "vitpose_keypoints_3d_world_unsmoothed_flat.csv"
    stabilized_path = run_dir / "csv" / "vitpose_keypoints_3d_world_flat.csv"
    raw, frame_indices = _read_keypoints3d_csv(raw_path)
    stabilized, stabilized_indices = _read_keypoints3d_csv(stabilized_path)
    if not np.array_equal(frame_indices, stabilized_indices):
        raise SystemExit("Raw and stabilized 3D CSV frame indices differ")

    valid = np.all(np.isfinite(raw), axis=-1)
    candidates: dict[str, dict] = {}
    for window in _parse_windows(args.candidate_windows):
        candidate = (
            raw.copy()
            if window == 1
            else robust_savgol_pose(
                raw,
                window_size=window,
                polynomial_order=args.polynomial_order,
                valid_mask=valid,
                min_outlier_distance_m=args.min_outlier_distance_m,
            )
        )
        candidates[str(window)] = pose3d_stability_metrics(raw, candidate)

    configured_metrics = pose3d_stability_metrics(raw, stabilized)
    figures_dir = run_dir / "figures"
    json_dir = run_dir / "json"
    figures_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    angle_plot_path = figures_dir / "pose3d_joint_angle_stability.png"
    adjustment_plot_path = figures_dir / "pose3d_adjustment_heatmap.png"
    _plot_angles(raw, stabilized, frame_indices, angle_plot_path)
    _plot_adjustments(raw, stabilized, frame_indices, adjustment_plot_path)

    report = {
        "raw_csv": str(raw_path),
        "stabilized_csv": str(stabilized_path),
        "configured_output": configured_metrics,
        "candidate_windows": candidates,
        "outputs": {
            "joint_angle_stability": str(angle_plot_path),
            "adjustment_heatmap": str(adjustment_plot_path),
        },
    }
    report_path = json_dir / "pose3d_stability_diagnostics.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"saved: {report_path}")
    print(json.dumps(candidates, ensure_ascii=False, indent=2))


def _read_keypoints3d_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise SystemExit(f"3D CSV not found: {path}")
    table = pd.read_csv(path)
    required = {"frame_idx", "joint_idx", "x_m", "y_m", "z_m"}
    missing = required - set(table.columns)
    if missing:
        raise SystemExit(f"{path} is missing columns: {sorted(missing)}")
    frame_indices = np.sort(table["frame_idx"].unique().astype(int))
    joint_count = int(table["joint_idx"].max()) + 1
    frame_lookup = {int(frame_idx): array_idx for array_idx, frame_idx in enumerate(frame_indices)}
    keypoints = np.full(
        (frame_indices.size, joint_count, 3),
        np.nan,
        dtype=float,
    )
    for row in table.itertuples(index=False):
        keypoints[frame_lookup[int(row.frame_idx)], int(row.joint_idx)] = (
            float(row.x_m),
            float(row.y_m),
            float(row.z_m),
        )
    return keypoints, frame_indices


def _parse_windows(raw: str) -> list[int]:
    windows = sorted({int(value.strip()) for value in raw.split(",")})
    if any(window < 1 or window % 2 == 0 for window in windows):
        raise SystemExit("Every candidate window must be a positive odd integer")
    return windows


def _plot_angles(
    raw: np.ndarray,
    stabilized: np.ndarray,
    frame_indices: np.ndarray,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(
        len(ANGLE_SPECS),
        1,
        figsize=(14, 2.6 * len(ANGLE_SPECS)),
        sharex=True,
        constrained_layout=True,
    )
    for axis, (name, joint_names) in zip(
        axes,
        ANGLE_SPECS.items(),
        strict=True,
    ):
        first, center, last = (COCO_BODY_JOINTS[joint_name] for joint_name in joint_names)
        raw_angle = joint_angles_degrees(
            raw[:, first],
            raw[:, center],
            raw[:, last],
        )
        stable_angle = joint_angles_degrees(
            stabilized[:, first],
            stabilized[:, center],
            stabilized[:, last],
        )
        valid = np.isfinite(raw_angle) & np.isfinite(stable_angle)
        axis.plot(
            frame_indices[valid],
            raw_angle[valid],
            color="#d62728",
            alpha=0.65,
            label="unsmoothed",
        )
        axis.plot(
            frame_indices[valid],
            stable_angle[valid],
            color="#2ca02c",
            linewidth=1.8,
            label="stabilized",
        )
        axis.set_title(name)
        axis.set_ylabel("degree")
        axis.grid(alpha=0.2)
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("Source frame")
    figure.savefig(output_path, dpi=170)
    plt.close(figure)


def _plot_adjustments(
    raw: np.ndarray,
    stabilized: np.ndarray,
    frame_indices: np.ndarray,
    output_path: Path,
) -> None:
    body_count = min(17, raw.shape[1])
    valid = np.all(np.isfinite(raw[:, :body_count]), axis=-1) & np.all(
        np.isfinite(stabilized[:, :body_count]),
        axis=-1,
    )
    adjustment_mm = 1000.0 * np.linalg.norm(
        stabilized[:, :body_count] - raw[:, :body_count],
        axis=-1,
    )
    adjustment_mm[~valid] = np.nan
    inverse_names = {index: name for name, index in COCO_BODY_JOINTS.items()}
    names = [inverse_names.get(joint_idx, f"joint_{joint_idx}") for joint_idx in range(body_count)]
    figure, axis = plt.subplots(figsize=(14, 6), constrained_layout=True)
    image = axis.imshow(
        adjustment_mm.T,
        aspect="auto",
        interpolation="nearest",
        cmap="magma",
    )
    axis.set_yticks(range(body_count), names)
    ticks = np.linspace(
        0,
        max(frame_indices.size - 1, 0),
        min(8, frame_indices.size),
        dtype=int,
    )
    if ticks.size:
        axis.set_xticks(ticks, frame_indices[ticks])
    axis.set_xlabel("Source frame")
    axis.set_title("3D unsmoothed → stabilized adjustment (mm)")
    figure.colorbar(image, ax=axis, label="Adjustment (mm)")
    figure.savefig(output_path, dpi=170)
    plt.close(figure)


if __name__ == "__main__":
    main()
