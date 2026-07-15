from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.artifacts import save_artifact_manifest
from src.camera_calibration import save_calibrations
from src.coordinate_system import calibration_metadata, opencv_reference_to_analysis
from src.config_validation import validate_model_config
from src.exporter import (
    export_excel,
    export_joint_validation_csv,
    export_keypoints2d_csv,
    export_keypoints3d_csv,
    export_placeholder_steps_csv,
    export_quality_csv,
    export_session_json,
    export_validation_csv,
)
from src.model_runtime import check_model_runtime, save_model_runtime_report
from src.preflight import has_errors, run_preflight, save_preflight_report
from src.run_outputs import create_run_output_tree
from src.smoothing_3d import moving_average_nan
from src.synthetic_data import build_synthetic_triangulation_result
from src.validation_3d import quality_summary, validate_triangulation
from src.video_io import load_session
from src.video_probe import probe_session_videos, video_probe_summary
from src.visualization_2d import write_placeholder_overlay_video
from src.visualization_3d import save_heatmap, save_reprojection_timeline, write_3d_skeleton_video


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TK3D multi-view 3D skeleton pipeline.")
    parser.add_argument("--session", required=True, help="Path to session.yaml")
    parser.add_argument("--model-config", default="config/model_config.yaml", help="Model config path")
    parser.add_argument("--output-root", default="outputs", help="Output root directory")
    parser.add_argument("--dry-run", action="store_true", help="Create expected outputs without model/video inference")
    parser.add_argument("--dry-run-frames", type=int, default=30)
    parser.add_argument("--run-id", default=None, help="Optional unique dry-run output identifier")
    parser.add_argument(
        "--strict-preflight",
        action="store_true",
        help="Fail dry-run when configured videos or model files are missing.",
    )
    args = parser.parse_args()

    session = load_session(args.session)
    with (ROOT / args.model_config).open("r", encoding="utf-8") as file:
        model_config = validate_model_config(yaml.safe_load(file))

    run_id, output_paths = create_run_output_tree(ROOT / args.output_root, session.session_id, args.run_id)
    issues = run_preflight(
        session=session,
        model_config=model_config,
        videos_required=not args.dry_run or args.strict_preflight,
        calibration_videos_required=not args.dry_run or args.strict_preflight,
        model_files_required=not args.dry_run or args.strict_preflight,
    )
    save_preflight_report(issues, output_paths["json"] / "preflight_report.json")
    with (output_paths["json"] / "video_probe_report.json").open("w", encoding="utf-8") as file:
        json.dump(video_probe_summary(probe_session_videos(session)), file, indent=2)
    save_model_runtime_report(
        {
            "pose2d": check_model_runtime(model_config.get("pose2d", {}), ROOT),
            "pose3d_single_view": check_model_runtime(model_config.get("pose3d_single_view", {}), ROOT),
        },
        output_paths["json"] / "model_runtime_report.json",
    )

    if args.dry_run:
        result = build_synthetic_triangulation_result(args.dry_run_frames)
    else:
        if has_errors(issues):
            raise SystemExit(
                "Preflight failed. Inspect outputs/<session_id>/json/preflight_report.json before running live mode."
            )
        raise SystemExit(
            "Live ViTPose inference is handled by scripts/run_vitpose_multiview_3d.py. "
            "Use this script with --dry-run for synthetic pipeline validation."
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
    summary = quality_summary(
        keypoints_3d_world=keypoints_3d_world,
        triangulation_score=result["triangulation_score"],
        reprojection_error=result["reprojection_error"],
        used_cameras=result["used_cameras"],
        validation=validation,
    )

    for camera in session.cameras:
        write_placeholder_overlay_video(output_paths["videos"] / f"{camera.camera_id}_2d_overlay.mp4")
    write_3d_skeleton_video(keypoints_3d_world, output_paths["videos"] / "skeleton_3d_world.mp4")
    if "calibrations" in result:
        save_calibrations(
            list(result["calibrations"].values()),
            output_paths["calibration"] / "cameras.json",
            metadata=calibration_metadata(
                "synthetic_test",
                {"name": "synthetic_opencv", "unit": "meter", "axes": {"x": "right", "y": "down", "z": "forward"}},
                opencv_reference_to_analysis(),
            ),
        )
    if "synthetic_ground_truth_3d_world" in result:
        summary["synthetic_mean_3d_error_m"] = _mean_3d_error(
            keypoints_3d_world,
            result["synthetic_ground_truth_3d_world"],
        )

    save_reprojection_timeline(
        result["reprojection_error"],
        output_paths["figures"] / "reprojection_error_timeline.png",
    )
    valid_heatmap = np.all(np.isfinite(keypoints_3d_world), axis=-1).astype(float)
    save_heatmap(valid_heatmap, output_paths["figures"] / "keypoint_validity_heatmap.png", "Keypoint Validity")
    save_heatmap(result["used_cameras"], output_paths["figures"] / "camera_usage_heatmap.png", "Camera Usage")

    keypoints3d_csv = output_paths["csv"] / "keypoints_3d_world_flat.csv"
    keypoints2d_csv = output_paths["csv"] / "keypoints_2d_flat.csv"
    quality_csv = output_paths["csv"] / "triangulation_quality.csv"
    validation_frames_csv = output_paths["csv"] / "validation_frames.csv"
    validation_joints_csv = output_paths["csv"] / "validation_joints.csv"
    validation_steps_csv = output_paths["csv"] / "validation_steps.csv"
    export_keypoints2d_csv(result.get("poses_2d_by_frame", {}), keypoints2d_csv)
    export_keypoints3d_csv(keypoints_3d_world, keypoints3d_csv)
    export_quality_csv(result["triangulation_score"], result["reprojection_error"], result["used_cameras"], quality_csv)
    export_validation_csv(validation.frame_valid_ratio, validation_frames_csv)
    export_joint_validation_csv(validation.joint_valid_ratio, validation_joints_csv)
    export_placeholder_steps_csv(validation_steps_csv)

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
    export_session_json(summary, output_paths["json"] / "quality_summary.json")

    excel_summary = {
        "session_id": session.session_id,
        "task_name": session.task_name,
        "frame_count": keypoints_3d_world.shape[0],
        "keypoint_count": keypoints_3d_world.shape[1],
        "mean_frame_valid_ratio": summary["mean_frame_valid_ratio"],
        "mean_reprojection_error_px": summary["mean_reprojection_error_px"],
    }
    export_excel(
        excel_summary,
        {
            "keypoints_2d": keypoints2d_csv,
            "keypoints_3d": keypoints3d_csv,
            "quality": quality_csv,
            "validation": validation_frames_csv,
        },
        output_paths["root"] / "session_3d_analysis.xlsx",
    )
    save_artifact_manifest(output_paths["root"], output_paths["json"] / "artifact_manifest.json")

    print(f"saved outputs under: {output_paths['root']}")
    print(f"run id: {run_id}")
    print(f"keypoints_3d_world shape: {keypoints_3d_world.shape}")
    print(f"mean reprojection error px: {summary['mean_reprojection_error_px']:.6f}")
    if "synthetic_mean_3d_error_m" in summary:
        print(f"synthetic mean 3d error m: {summary['synthetic_mean_3d_error_m']:.9f}")


def _mean_3d_error(predicted: np.ndarray, target: np.ndarray) -> float:
    valid = np.all(np.isfinite(predicted), axis=-1) & np.all(np.isfinite(target), axis=-1)
    if not np.any(valid):
        return float("nan")
    return float(np.mean(np.linalg.norm(predicted[valid] - target[valid], axis=-1)))


if __name__ == "__main__":
    main()
