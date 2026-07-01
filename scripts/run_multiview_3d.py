from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_structures import COCO_WHOLEBODY_KEYPOINTS
from src.exporter import export_excel, export_keypoints3d_csv, export_quality_csv, export_session_json, export_validation_csv
from src.smoothing_3d import moving_average_nan
from src.validation_3d import validate_triangulation
from src.video_io import ensure_output_tree, load_session
from src.visualization_2d import write_placeholder_overlay_video
from src.visualization_3d import save_heatmap, save_reprojection_timeline, write_placeholder_3d_video


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TK3D multi-view 3D skeleton pipeline.")
    parser.add_argument("--session", required=True, help="Path to session.yaml")
    parser.add_argument("--model-config", default="config/model_config.yaml", help="Model config path")
    parser.add_argument("--output-root", default="outputs", help="Output root directory")
    parser.add_argument("--dry-run", action="store_true", help="Create expected outputs without model/video inference")
    parser.add_argument("--dry-run-frames", type=int, default=30)
    args = parser.parse_args()

    session = load_session(args.session)
    with (ROOT / args.model_config).open("r", encoding="utf-8") as file:
        model_config = yaml.safe_load(file)

    output_paths = ensure_output_tree(ROOT / args.output_root, session.session_id)

    if args.dry_run:
        result = build_dry_run_result(args.dry_run_frames)
    else:
        raise NotImplementedError(
            "Live RTMW inference will be enabled after model files and calibration videos are available. "
            "Use --dry-run to validate the project outputs now."
        )

    keypoints_3d_world = moving_average_nan(
        result["keypoints_3d_world"],
        window_size=int(model_config.get("smoothing", {}).get("window_size", 5)),
    )
    validation = validate_triangulation(
        keypoints_3d_world=keypoints_3d_world,
        reprojection_error=result["reprojection_error"],
        max_reprojection_error_px=float(model_config["triangulation"]["max_reprojection_error_px"]),
    )

    for camera in session.cameras:
        write_placeholder_overlay_video(output_paths["videos"] / f"{camera.camera_id}_2d_overlay.mp4")
    write_placeholder_3d_video(output_paths["videos"] / "skeleton_3d_world.mp4")

    save_reprojection_timeline(
        result["reprojection_error"],
        output_paths["figures"] / "reprojection_error_timeline.png",
    )
    valid_heatmap = np.all(np.isfinite(keypoints_3d_world), axis=-1).astype(float)
    save_heatmap(valid_heatmap, output_paths["figures"] / "keypoint_validity_heatmap.png", "Keypoint Validity")
    save_heatmap(result["used_cameras"], output_paths["figures"] / "camera_usage_heatmap.png", "Camera Usage")

    keypoints3d_csv = output_paths["csv"] / "keypoints_3d_world_flat.csv"
    quality_csv = output_paths["csv"] / "triangulation_quality.csv"
    validation_frames_csv = output_paths["csv"] / "validation_frames.csv"
    export_keypoints3d_csv(keypoints_3d_world, keypoints3d_csv)
    export_quality_csv(result["triangulation_score"], result["reprojection_error"], result["used_cameras"], quality_csv)
    export_validation_csv(validation.frame_valid_ratio, validation_frames_csv)

    create_empty_required_csvs(output_paths["csv"])

    payload = {
        "session_id": session.session_id,
        "task_name": session.task_name,
        "pose_dimension": "3d",
        "shape": {"keypoints_3d_world": list(keypoints_3d_world.shape)},
        "keypoints_3d_world": keypoints_3d_world,
        "triangulation_score": result["triangulation_score"],
        "reprojection_error": result["reprojection_error"],
        "used_cameras": result["used_cameras"],
        "validation": validation,
    }
    export_session_json(payload, output_paths["json"] / "session_3d.json")

    summary = {
        "session_id": session.session_id,
        "task_name": session.task_name,
        "frame_count": keypoints_3d_world.shape[0],
        "keypoint_count": keypoints_3d_world.shape[1],
        "mean_frame_valid_ratio": float(np.nanmean(validation.frame_valid_ratio)),
        "mean_reprojection_error_px": float(np.nanmean(result["reprojection_error"])),
    }
    export_excel(
        summary,
        {
            "keypoints_3d": keypoints3d_csv,
            "quality": quality_csv,
            "validation": validation_frames_csv,
        },
        output_paths["root"] / "session_3d_analysis.xlsx",
    )

    print(f"saved outputs under: {output_paths['root']}")
    print(f"keypoints_3d_world shape: {keypoints_3d_world.shape}")


def build_dry_run_result(frame_count: int) -> dict[str, np.ndarray]:
    keypoints = np.full((frame_count, COCO_WHOLEBODY_KEYPOINTS, 3), np.nan, dtype=float)
    score = np.zeros((frame_count, COCO_WHOLEBODY_KEYPOINTS), dtype=float)
    reprojection = np.full((frame_count, COCO_WHOLEBODY_KEYPOINTS), np.nan, dtype=float)
    used_cameras = np.zeros((frame_count, COCO_WHOLEBODY_KEYPOINTS), dtype=int)

    core_joint_count = 17
    for frame_idx in range(frame_count):
        phase = frame_idx / max(frame_count - 1, 1)
        for joint_idx in range(core_joint_count):
            keypoints[frame_idx, joint_idx] = [
                0.03 * joint_idx,
                0.02 * np.sin(phase * np.pi * 2 + joint_idx * 0.1),
                1.0 + 0.01 * frame_idx,
            ]
            score[frame_idx, joint_idx] = 0.95
            reprojection[frame_idx, joint_idx] = 2.0
            used_cameras[frame_idx, joint_idx] = 3
    return {
        "keypoints_3d_world": keypoints,
        "triangulation_score": score,
        "reprojection_error": reprojection,
        "used_cameras": used_cameras,
    }


def create_empty_required_csvs(csv_dir: Path) -> None:
    for filename in [
        "keypoints_2d_flat.csv",
        "validation_joints.csv",
        "validation_steps.csv",
    ]:
        path = csv_dir / filename
        if not path.exists():
            path.write_text("status\npending\n", encoding="utf-8")


if __name__ == "__main__":
    main()
