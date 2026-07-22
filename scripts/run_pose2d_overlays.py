from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pose2d_estimator import Pose2DConfig, ViTPose2DEstimator
from src.person_tracking import person_detector_config_from_mapping
from src.pose2d_sequence import pose2d_at_frame
from src.progress import ProgressBar, print_step
from src.config_validation import validate_model_config
from src.run_outputs import create_run_output_tree
from src.video_io import iter_video_frames, load_session
from src.visualization_2d import draw_pose2d


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ViTPose-Huge whole-body 2D pose overlays for TK3D videos.")
    parser.add_argument("--session", required=True, help="Path to session.yaml")
    parser.add_argument("--model-config", default="config/model_config.yaml", help="Model config path")
    parser.add_argument("--output-root", default="outputs", help="Output root directory")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum sampled inference frames. Omit for full video duration.")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--camera", default=None, help="Optional single camera_id")
    parser.add_argument("--progress-every", type=int, default=1, help="Update progress every N written frames.")
    parser.add_argument("--output-fps", type=float, default=None, help="Playback FPS. Defaults to the source video FPS.")
    parser.add_argument("--run-id", default=None, help="Optional unique output identifier")
    args = parser.parse_args()

    print("=" * 72, flush=True)
    print("TK3D VITPOSE 2D OVERLAYS", flush=True)
    print("=" * 72, flush=True)
    print_step(1, 4, "Loading session and model config")
    session = load_session(args.session)
    with (ROOT / args.model_config).open("r", encoding="utf-8") as file:
        model_config = validate_model_config(yaml.safe_load(file))
    pose2d_cfg = model_config["pose2d"]
    print_step(2, 4, f"Loading ViTPose model: {pose2d_cfg['model_name']}")
    load_start = time.perf_counter()
    estimator = ViTPose2DEstimator(
        Pose2DConfig(
            model_name=pose2d_cfg["model_name"],
            config_path=(ROOT / pose2d_cfg["config_path"]).resolve(),
            checkpoint_path=(ROOT / pose2d_cfg["checkpoint_path"]).resolve(),
            device=pose2d_cfg.get("device", "cuda:0"),
            score_threshold=float(pose2d_cfg.get("score_threshold", 0.30)),
            input_size=tuple(int(value) for value in pose2d_cfg.get("input_size", [256, 192])),
            flip_test=bool(pose2d_cfg.get("flip_test", True)),
            temporal_filter_enabled=bool(pose2d_cfg.get("temporal_filter_enabled", True)),
            temporal_stabilize_left_right=bool(
                pose2d_cfg.get("temporal_stabilize_left_right", True)
            ),
            person_detector=person_detector_config_from_mapping(
                model_config.get("person_detector"),
                frame_rate=session.fps / max(args.stride, 1),
            ),
        ),
        dry_run=False,
    )
    print(f"      model loaded in {time.perf_counter() - load_start:.1f}s", flush=True)

    print_step(3, 4, "Preparing output videos")
    run_id, output_paths = create_run_output_tree(ROOT / args.output_root, session.session_id, args.run_id)
    cameras = [camera for camera in session.cameras if args.camera is None or camera.camera_id == args.camera]
    if not cameras:
        raise SystemExit(f"No cameras matched: {args.camera}")

    print_step(4, 4, "Running ViTPose inference and writing overlays")
    for camera in cameras:
        output_path = output_paths["videos"] / f"{camera.camera_id}_vitpose_2d_overlay.mp4"
        write_overlay(
            camera.video_path,
            output_path,
            camera.camera_id,
            estimator,
            args.max_frames,
            args.stride,
            args.progress_every,
            args.output_fps,
        )
        print(f"saved: {output_path}")
    print(f"run id: {run_id}")


def write_overlay(
    video_path: Path,
    output_path: Path,
    camera_id: str,
    estimator: ViTPose2DEstimator,
    max_frames: int | None,
    stride: int,
    progress_every: int,
    output_fps: float | None,
) -> None:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    target = _target_frame_count(capture, max_frames, stride)
    capture.release()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    playback_fps = max(float(output_fps or fps), 1.0)
    inference_progress = ProgressBar(f"{camera_id} inference", target)
    sampled_poses = []
    written = 0
    for frame_idx, _, frame in iter_video_frames(video_path, stride=stride):
        sampled_poses.append(estimator.predict(frame, camera_id=camera_id, frame_idx=frame_idx))
        written += 1
        if written == 1 or written >= target or written % max(progress_every, 1) == 0:
            inference_progress.print(written, extra=f"src frame {frame_idx}")
        if max_frames is not None and written >= max_frames:
            break
    inference_progress.done()
    if not sampled_poses:
        raise RuntimeError(f"No frames could be inferred from: {video_path}")

    output_frame_count = min(
        source_frames,
        sampled_poses[-1].frame_idx + _repeat_count(sampled_poses[-1].frame_idx, source_frames, stride),
    )
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), playback_fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open overlay video writer: {output_path}")
    capture = cv2.VideoCapture(str(video_path))
    render_progress = ProgressBar(f"{camera_id} render", output_frame_count)
    try:
        for frame_idx in range(output_frame_count):
            ok, frame = capture.read()
            if not ok:
                break
            overlay = draw_pose2d(frame, pose2d_at_frame(sampled_poses, frame_idx))
            writer.write(overlay)
            if frame_idx == 0 or frame_idx + 1 >= output_frame_count or (frame_idx + 1) % max(progress_every, 1) == 0:
                render_progress.print(frame_idx + 1, extra=f"src frame {frame_idx}")
    finally:
        capture.release()
        writer.release()
        render_progress.done()


def _target_frame_count(capture: cv2.VideoCapture, max_frames: int | None, stride: int) -> int:
    source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if source_frames <= 0:
        return max(max_frames or 0, 0)
    sampled = (source_frames + max(stride, 1) - 1) // max(stride, 1)
    if max_frames is None:
        return sampled
    return min(max(max_frames, 0), sampled)


def _repeat_count(frame_idx: int, source_frames: int, stride: int) -> int:
    step = max(stride, 1)
    if source_frames <= 0:
        return step
    return max(min(step, source_frames - frame_idx), 1)


if __name__ == "__main__":
    main()
