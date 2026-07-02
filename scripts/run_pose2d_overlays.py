from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pose2d_estimator import Pose2DConfig, RTMW2DEstimator
from src.video_io import ensure_output_tree, iter_video_frames, load_session
from src.visualization_2d import draw_pose2d


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RTMW 2D wholebody pose overlays for TK3D videos.")
    parser.add_argument("--session", required=True, help="Path to session.yaml")
    parser.add_argument("--model-config", default="config/model_config.yaml", help="Model config path")
    parser.add_argument("--output-root", default="outputs", help="Output root directory")
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--camera", default=None, help="Optional single camera_id")
    args = parser.parse_args()

    session = load_session(args.session)
    with (ROOT / args.model_config).open("r", encoding="utf-8") as file:
        model_config = yaml.safe_load(file)
    pose2d_cfg = model_config["pose2d"]
    estimator = RTMW2DEstimator(
        Pose2DConfig(
            model_name=pose2d_cfg["model_name"],
            config_path=(ROOT / pose2d_cfg["config_path"]).resolve(),
            checkpoint_path=(ROOT / pose2d_cfg["checkpoint_path"]).resolve(),
            device=pose2d_cfg.get("device", "cuda:0"),
            score_threshold=float(pose2d_cfg.get("score_threshold", 0.30)),
        ),
        dry_run=False,
    )

    output_paths = ensure_output_tree(ROOT / args.output_root, session.session_id)
    cameras = [camera for camera in session.cameras if args.camera is None or camera.camera_id == args.camera]
    if not cameras:
        raise SystemExit(f"No cameras matched: {args.camera}")

    for camera in cameras:
        output_path = output_paths["videos"] / f"{camera.camera_id}_rtmw_2d_overlay.mp4"
        write_overlay(camera.video_path, output_path, camera.camera_id, estimator, args.max_frames, args.stride)
        print(f"saved: {output_path}")


def write_overlay(
    video_path: Path,
    output_path: Path,
    camera_id: str,
    estimator: RTMW2DEstimator,
    max_frames: int,
    stride: int,
) -> None:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps / max(stride, 1), (width, height))
    try:
        written = 0
        for frame_idx, _, frame in iter_video_frames(video_path, stride=stride):
            pose = estimator.predict(frame, camera_id=camera_id, frame_idx=frame_idx)
            overlay = draw_pose2d(frame, pose)
            writer.write(overlay)
            written += 1
            if written >= max_frames:
                break
    finally:
        writer.release()


if __name__ == "__main__":
    main()
