from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

from src.data_structures import COCO_BODY_JOINTS, PersonPose2D
from src.pose2d_sequence import pose2d_at_frame
from src.pose2d_stabilization import pose2d_stability_metrics
from src.video_io import load_session
from src.visualization_2d import draw_pose2d


ANGLE_SPECS = {
    "left_elbow": (5, 7, 9),
    "right_elbow": (6, 8, 10),
    "left_knee": (11, 13, 15),
    "right_knee": (12, 14, 16),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Render detailed raw-versus-stabilized 2D pose diagnostics.")
    parser.add_argument("--session", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--camera", required=True)
    parser.add_argument("--comparison-fps", type=float, default=None)
    args = parser.parse_args()

    session = load_session(args.session)
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = (ROOT / run_dir).resolve()
    camera = next(
        (item for item in session.cameras if item.camera_id == args.camera),
        None,
    )
    if camera is None:
        raise SystemExit(f"Camera not found in session: {args.camera}")

    raw_path, stable_path = _resolve_csv_paths(run_dir, args.camera)
    raw_poses = _read_pose_csv(raw_path, args.camera)
    stable_poses = _read_pose_csv(stable_path, args.camera)
    if len(raw_poses) != len(stable_poses):
        raise SystemExit("Raw and stabilized CSV files contain different frame counts")

    figures_dir = run_dir / "figures"
    videos_dir = run_dir / "videos"
    json_dir = run_dir / "json"
    for path in (figures_dir, videos_dir, json_dir):
        path.mkdir(parents=True, exist_ok=True)

    raw_xy, stable_xy, valid, frame_indices = _stack_comparable(
        raw_poses,
        stable_poses,
    )
    names = _body_joint_names()
    adjustment = np.linalg.norm(stable_xy - raw_xy, axis=-1)
    adjustment[~valid] = np.nan
    per_joint_adjustment = {names[joint_idx]: _safe_median(adjustment[:, joint_idx]) for joint_idx in range(len(names))}
    worst_joints = sorted(
        range(len(names)),
        key=lambda joint_idx: (
            per_joint_adjustment[names[joint_idx]] if per_joint_adjustment[names[joint_idx]] is not None else -1.0
        ),
        reverse=True,
    )[:4]

    heatmap_path = figures_dir / f"{args.camera}_pose_adjustment_heatmap.png"
    trajectory_path = figures_dir / f"{args.camera}_worst_joint_trajectories.png"
    angle_path = figures_dir / f"{args.camera}_joint_angle_stability.png"
    contact_sheet_path = figures_dir / f"{args.camera}_stabilized_contact_sheet.png"
    comparison_video_path = videos_dir / f"{args.camera}_raw_vs_stabilized_pose.mp4"

    _plot_adjustment_heatmap(adjustment, frame_indices, names, heatmap_path)
    _plot_worst_trajectories(
        raw_xy,
        stable_xy,
        valid,
        frame_indices,
        names,
        worst_joints,
        trajectory_path,
    )
    angle_metrics = _plot_angle_stability(
        raw_xy,
        stable_xy,
        valid,
        frame_indices,
        angle_path,
    )
    overlay_path = videos_dir / f"{args.camera}_vitpose_2d_overlay.mp4"
    if overlay_path.exists():
        _write_contact_sheet(overlay_path, contact_sheet_path)
    _write_comparison_video(
        camera.video_path,
        raw_poses,
        stable_poses,
        comparison_video_path,
        output_fps=args.comparison_fps,
    )

    report = {
        **pose2d_stability_metrics(raw_poses, stable_poses),
        "camera_id": args.camera,
        "raw_csv": str(raw_path),
        "stabilized_csv": str(stable_path),
        "per_joint_median_adjustment_px": per_joint_adjustment,
        "worst_adjusted_joints": [names[index] for index in worst_joints],
        "angle_high_frequency_metrics": angle_metrics,
        "outputs": {
            "adjustment_heatmap": str(heatmap_path),
            "worst_joint_trajectories": str(trajectory_path),
            "joint_angle_stability": str(angle_path),
            "contact_sheet": (str(contact_sheet_path) if contact_sheet_path.exists() else None),
            "comparison_video": str(comparison_video_path),
        },
    }
    report_path = json_dir / f"{args.camera}_pose_stability_diagnostics.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"saved: {report_path}")
    print(f"saved: {comparison_video_path}")


def _resolve_csv_paths(run_dir: Path, camera_id: str) -> tuple[Path, Path]:
    candidates = (
        (
            run_dir / "csv" / f"{camera_id}_vitpose_keypoints_2d_raw_flat.csv",
            run_dir / "csv" / f"{camera_id}_vitpose_keypoints_2d_stabilized_flat.csv",
        ),
        (
            run_dir / "csv" / "vitpose_keypoints_2d_raw_flat.csv",
            run_dir / "csv" / "vitpose_keypoints_2d_flat.csv",
        ),
    )
    for raw_path, stable_path in candidates:
        if raw_path.exists() and stable_path.exists():
            return raw_path, stable_path
    raise SystemExit(f"Raw/stabilized 2D CSV files not found under: {run_dir / 'csv'}")


def _read_pose_csv(path: Path, camera_id: str) -> list[PersonPose2D]:
    table = pd.read_csv(path)
    required = {"frame_idx", "camera_id", "joint_idx", "x", "y", "score", "valid"}
    missing = required - set(table.columns)
    if missing:
        raise SystemExit(f"{path} is missing columns: {sorted(missing)}")
    table = table[table["camera_id"] == camera_id].copy()
    if table.empty:
        raise SystemExit(f"{path} contains no rows for camera {camera_id}")
    joint_count = int(table["joint_idx"].max()) + 1
    poses: list[PersonPose2D] = []
    for frame_idx, frame in table.groupby("frame_idx", sort=True):
        xy = np.full((joint_count, 2), np.nan, dtype=float)
        scores = np.zeros(joint_count, dtype=float)
        valid = np.zeros(joint_count, dtype=bool)
        joint_indices = frame["joint_idx"].to_numpy(dtype=int)
        xy[joint_indices, 0] = frame["x"].to_numpy(dtype=float)
        xy[joint_indices, 1] = frame["y"].to_numpy(dtype=float)
        scores[joint_indices] = frame["score"].to_numpy(dtype=float)
        valid[joint_indices] = frame["valid"].astype(bool).to_numpy()
        person_id = int(frame["person_id"].iloc[0]) if "person_id" in frame.columns else 0
        poses.append(
            PersonPose2D(
                camera_id=camera_id,
                frame_idx=int(frame_idx),
                keypoints_xy=xy,
                scores=scores,
                valid_mask=valid,
                person_id=person_id,
            )
        )
    return poses


def _stack_comparable(
    raw_poses: list[PersonPose2D],
    stable_poses: list[PersonPose2D],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw_xy = np.stack([pose.keypoints_xy[:17] for pose in raw_poses]).astype(float)
    stable_xy = np.stack([pose.keypoints_xy[:17] for pose in stable_poses]).astype(float)
    valid = (
        np.stack([pose.valid_mask[:17] for pose in raw_poses])
        & np.stack([pose.valid_mask[:17] for pose in stable_poses])
        & np.all(np.isfinite(raw_xy), axis=-1)
        & np.all(np.isfinite(stable_xy), axis=-1)
    )
    frame_indices = np.asarray([pose.frame_idx for pose in raw_poses], dtype=int)
    return raw_xy, stable_xy, valid, frame_indices


def _body_joint_names() -> list[str]:
    inverse = {index: name for name, index in COCO_BODY_JOINTS.items()}
    return [inverse.get(index, f"joint_{index}") for index in range(17)]


def _plot_adjustment_heatmap(
    adjustment: np.ndarray,
    frame_indices: np.ndarray,
    joint_names: list[str],
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(13, 6), constrained_layout=True)
    image = axis.imshow(
        adjustment.T,
        aspect="auto",
        interpolation="nearest",
        cmap="magma",
    )
    axis.set_yticks(range(len(joint_names)), joint_names)
    ticks = np.linspace(0, max(len(frame_indices) - 1, 0), min(8, len(frame_indices)), dtype=int)
    if ticks.size:
        axis.set_xticks(ticks, frame_indices[ticks])
    axis.set_xlabel("Source frame")
    axis.set_title("Raw → stabilized body-joint adjustment (px)")
    figure.colorbar(image, ax=axis, label="Adjustment (px)")
    figure.savefig(output_path, dpi=170)
    plt.close(figure)


def _plot_worst_trajectories(
    raw_xy: np.ndarray,
    stable_xy: np.ndarray,
    valid: np.ndarray,
    frame_indices: np.ndarray,
    joint_names: list[str],
    worst_joints: list[int],
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(
        len(worst_joints),
        1,
        figsize=(13, 2.8 * len(worst_joints)),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    for axis, joint_idx in zip(axes, worst_joints, strict=True):
        mask = valid[:, joint_idx]
        axis.plot(
            frame_indices[mask],
            raw_xy[mask, joint_idx, 0],
            color="#d62728",
            alpha=0.60,
            label="raw x",
        )
        axis.plot(
            frame_indices[mask],
            stable_xy[mask, joint_idx, 0],
            color="#2ca02c",
            linewidth=2.0,
            label="stabilized x",
        )
        axis.plot(
            frame_indices[mask],
            raw_xy[mask, joint_idx, 1],
            color="#ff7f0e",
            alpha=0.45,
            label="raw y",
        )
        axis.plot(
            frame_indices[mask],
            stable_xy[mask, joint_idx, 1],
            color="#1f77b4",
            linewidth=2.0,
            label="stabilized y",
        )
        axis.set_ylabel("pixel")
        axis.set_title(joint_names[joint_idx])
        axis.grid(alpha=0.2)
    axes[0].legend(ncol=4, loc="upper right")
    axes[-1].set_xlabel("Source frame")
    figure.savefig(output_path, dpi=170)
    plt.close(figure)


def _plot_angle_stability(
    raw_xy: np.ndarray,
    stable_xy: np.ndarray,
    valid: np.ndarray,
    frame_indices: np.ndarray,
    output_path: Path,
) -> dict[str, dict[str, float | None]]:
    figure, axes = plt.subplots(
        len(ANGLE_SPECS),
        1,
        figsize=(13, 2.7 * len(ANGLE_SPECS)),
        sharex=True,
        constrained_layout=True,
    )
    metrics: dict[str, dict[str, float | None]] = {}
    for axis, (name, (first, center, last)) in zip(
        axes,
        ANGLE_SPECS.items(),
        strict=True,
    ):
        angle_valid = valid[:, first] & valid[:, center] & valid[:, last]
        raw_angles = _angles_2d(
            raw_xy[:, first],
            raw_xy[:, center],
            raw_xy[:, last],
        )
        stable_angles = _angles_2d(
            stable_xy[:, first],
            stable_xy[:, center],
            stable_xy[:, last],
        )
        angle_valid &= np.isfinite(raw_angles) & np.isfinite(stable_angles)
        axis.plot(
            frame_indices[angle_valid],
            raw_angles[angle_valid],
            color="#d62728",
            alpha=0.65,
            label="raw",
        )
        axis.plot(
            frame_indices[angle_valid],
            stable_angles[angle_valid],
            color="#2ca02c",
            linewidth=2.0,
            label="stabilized",
        )
        axis.set_ylabel("degree")
        axis.set_title(name)
        axis.grid(alpha=0.2)
        raw_hf = _angle_high_frequency(raw_angles[angle_valid])
        stable_hf = _angle_high_frequency(stable_angles[angle_valid])
        reduction = None
        if raw_hf is not None and raw_hf > 1e-12 and stable_hf is not None:
            reduction = 100.0 * (raw_hf - stable_hf) / raw_hf
        metrics[name] = {
            "raw_high_frequency_median_deg": raw_hf,
            "stabilized_high_frequency_median_deg": stable_hf,
            "reduction_percent": reduction,
        }
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("Source frame")
    figure.savefig(output_path, dpi=170)
    plt.close(figure)
    return metrics


def _write_contact_sheet(video_path: Path, output_path: Path) -> None:
    capture = cv2.VideoCapture(str(video_path))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if not capture.isOpened() or frame_count < 1:
        capture.release()
        return
    selected = np.linspace(0, frame_count - 1, min(12, frame_count), dtype=int)
    frames = []
    for frame_idx in selected:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = capture.read()
        if ok:
            frames.append((int(frame_idx), cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    capture.release()
    if not frames:
        return
    columns = 4
    rows = int(np.ceil(len(frames) / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(16, 4 * rows),
        constrained_layout=True,
    )
    axes = np.asarray(axes).reshape(-1)
    for axis in axes:
        axis.axis("off")
    for axis, (frame_idx, frame) in zip(axes, frames, strict=False):
        axis.imshow(frame)
        axis.set_title(f"frame {frame_idx}")
    figure.savefig(output_path, dpi=140)
    plt.close(figure)


def _write_comparison_video(
    source_video: Path,
    raw_poses: list[PersonPose2D],
    stable_poses: list[PersonPose2D],
    output_path: Path,
    output_fps: float | None,
) -> None:
    capture = cv2.VideoCapture(str(source_video))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open source video: {source_video}")
    fps = max(float(output_fps or capture.get(cv2.CAP_PROP_FPS) or 30.0), 1.0)
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    panel_width = max(source_width // 2, 1)
    panel_height = max(source_height // 2, 1)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (panel_width * 2, panel_height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open comparison video writer: {output_path}")
    first_frame = raw_poses[0].frame_idx
    last_frame = raw_poses[-1].frame_idx
    capture.set(cv2.CAP_PROP_POS_FRAMES, first_frame)
    try:
        for frame_idx in range(first_frame, last_frame + 1):
            ok, frame = capture.read()
            if not ok:
                break
            raw_panel = draw_pose2d(
                frame,
                pose2d_at_frame(raw_poses, frame_idx),
                color=(40, 40, 255),
                edge_color=(0, 130, 255),
            )
            stable_panel = draw_pose2d(
                frame,
                pose2d_at_frame(stable_poses, frame_idx),
                color=(30, 230, 30),
                edge_color=(255, 180, 40),
            )
            raw_panel = cv2.resize(raw_panel, (panel_width, panel_height))
            stable_panel = cv2.resize(stable_panel, (panel_width, panel_height))
            cv2.putText(
                raw_panel,
                "RAW / CAUSAL",
                (24, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (20, 20, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                stable_panel,
                "ZERO-PHASE STABILIZED",
                (24, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (20, 210, 20),
                2,
                cv2.LINE_AA,
            )
            writer.write(np.hstack([raw_panel, stable_panel]))
    finally:
        capture.release()
        writer.release()


def _angles_2d(
    first: np.ndarray,
    center: np.ndarray,
    last: np.ndarray,
) -> np.ndarray:
    first_vector = first - center
    last_vector = last - center
    denominator = np.linalg.norm(first_vector, axis=-1) * np.linalg.norm(
        last_vector,
        axis=-1,
    )
    cosine = np.divide(
        np.sum(first_vector * last_vector, axis=-1),
        denominator,
        out=np.full(denominator.shape, np.nan, dtype=float),
        where=denominator > 1e-12,
    )
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def _angle_high_frequency(values: np.ndarray) -> float | None:
    sequence = np.asarray(values, dtype=float)
    if sequence.size < 5:
        return None
    window = min(11, sequence.size if sequence.size % 2 else sequence.size - 1)
    if window < 5:
        return None
    trend = savgol_filter(
        sequence,
        window_length=window,
        polyorder=min(2, window - 1),
        mode="interp",
    )
    return float(np.median(np.abs(sequence - trend)))


def _safe_median(values: np.ndarray) -> float | None:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.median(finite)) if finite.size else None


if __name__ == "__main__":
    main()
