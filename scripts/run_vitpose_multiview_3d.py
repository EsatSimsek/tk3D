from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.camera_calibration import load_calibrations
from src.data_structures import CameraCalibration
from src.exporter import export_keypoints3d_csv, export_session_json
from src.pose2d_estimator import Pose2DConfig, ViTPose2DEstimator
from src.progress import ProgressBar
from src.smoothing_3d import moving_average_nan
from src.triangulation import stack_triangulated, triangulate_frame
from src.video_io import ensure_output_tree, load_session
from src.visualization_2d import draw_pose2d
from src.visualization_3d import write_3d_skeleton_video


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ViTPose-Huge 2D + multi-view 3D test pipeline.")
    parser.add_argument("--session", required=True)
    parser.add_argument("--model-config", default="config/model_config.yaml")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum sampled inference frames. Omit for full video duration.")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-cameras", type=int, default=None, help="Optional limit for faster tests; default uses all session cameras.")
    parser.add_argument("--progress-every", type=int, default=1, help="Print progress every N written frames.")
    parser.add_argument("--output-fps", type=float, default=None, help="Playback FPS. Defaults to the source video FPS.")
    args = parser.parse_args()

    session = load_session(args.session)
    if len(session.cameras) < 2:
        raise SystemExit("Need at least two cameras.")
    cameras = session.cameras[: args.max_cameras] if args.max_cameras else session.cameras
    if len(cameras) < 2:
        raise SystemExit("Need at least two selected cameras. Increase --max-cameras or update the session.")
    output_paths = ensure_output_tree(ROOT / args.output_root, session.session_id)

    with (ROOT / args.model_config).open("r", encoding="utf-8") as file:
        model_config = yaml.safe_load(file)
    pose2d_config = model_config["pose2d"]
    print("=" * 72, flush=True)
    print("TK3D VITPOSE MULTI-VIEW 3D", flush=True)
    print("=" * 72, flush=True)
    print(f"[1/4] Loading ViTPose model", flush=True)
    print(f"      model : {pose2d_config['model_name']}", flush=True)
    print(f"      device: {pose2d_config.get('device', 'cuda:0')}", flush=True)
    estimator = ViTPose2DEstimator(
        Pose2DConfig(
            model_name=pose2d_config["model_name"],
            config_path=(ROOT / pose2d_config["config_path"]).resolve(),
            checkpoint_path=(ROOT / pose2d_config["checkpoint_path"]).resolve(),
            device=pose2d_config.get("device", "cuda:0"),
            score_threshold=float(pose2d_config.get("score_threshold", 0.30)),
        )
    )

    calibrations_path = output_paths["calibration"] / "cameras.json"
    if calibrations_path.exists():
        calibrations = load_calibrations(calibrations_path)
        if all(camera.camera_id in calibrations for camera in cameras):
            calibration_mode = "loaded"
        else:
            cameras = cameras[:2]
            calibrations = build_pair_test_calibrations(cameras[0].camera_id, cameras[1].camera_id)
            calibration_mode = "approximate_test_calibration"
            print("WARNING: calibration/cameras.json does not match selected cameras. Using the first two cameras with approximate test calibration.")
    else:
        cameras = cameras[:2]
        calibrations = build_pair_test_calibrations(cameras[0].camera_id, cameras[1].camera_id)
        calibration_mode = "approximate_test_calibration"
        print("WARNING: calibration/cameras.json not found. Using the first two cameras with approximate test calibration, not metric 3D.")

    captures = [cv2.VideoCapture(str(camera.video_path)) for camera in cameras]
    if not all(capture.isOpened() for capture in captures):
        raise SystemExit("Could not open all selected videos.")

    fps = captures[0].get(cv2.CAP_PROP_FPS) or 30.0
    print(
        "[2/4] Preparing videos and calibration\n"
        f"      cameras         : {len(cameras)}\n"
        f"      target frames   : {args.max_frames or 'full video'}\n"
        f"      stride          : {max(args.stride, 1)}\n"
        f"      calibration     : {calibration_mode}\n"
        "[3/4] Running 2D pose + 3D triangulation",
        flush=True,
    )
    overlay_writers = []
    for camera, capture in zip(cameras, captures):
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output_path = output_paths["videos"] / f"{camera.camera_id}_vitpose_2d_overlay.mp4"
        overlay_writers.append(cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), max(float(args.output_fps or fps), 1.0), (width, height)))

    triangulated = []
    source_frame_count = _min_source_frame_count(captures)
    target_frames = _target_sample_count(source_frame_count, args.max_frames, args.stride)
    output_repeats: list[int] = []
    progress = ProgressBar("2D + 3D", target_frames)
    try:
        frame_idx = 0
        written = 0
        while args.max_frames is None or written < args.max_frames:
            frames = []
            ok = True
            for capture in captures:
                ret, frame = capture.read()
                ok = ok and ret
                frames.append(frame)
            if not ok:
                break
            if frame_idx % args.stride != 0:
                frame_idx += 1
                continue

            poses_by_camera = {}
            repeat_count = _repeat_count(frame_idx, source_frame_count, args.stride)
            for camera, frame, writer in zip(cameras, frames, overlay_writers):
                pose = estimator.predict(frame, camera.camera_id, frame_idx)
                poses_by_camera[camera.camera_id] = pose
                overlay = draw_pose2d(frame, pose)
                for _ in range(repeat_count):
                    writer.write(overlay)

            triangulated.append(
                triangulate_frame(
                    frame_idx=frame_idx,
                    poses_by_camera=poses_by_camera,
                    calibrations=calibrations,
                    min_views=2,
                )
            )
            output_repeats.append(repeat_count)
            written += 1
            if written == 1 or written == target_frames or written % max(args.progress_every, 1) == 0:
                progress.print(written, extra=f"src frame {frame_idx}")
            frame_idx += 1
    finally:
        for capture in captures:
            capture.release()
        for writer in overlay_writers:
            writer.release()
        if triangulated:
            progress.done()
            print("[4/4] Saving 3D outputs", flush=True)

    arrays = stack_triangulated(triangulated)
    arrays["keypoints_3d_world"] = moving_average_nan(arrays["keypoints_3d_world"], window_size=5)
    export_keypoints3d_csv(arrays["keypoints_3d_world"], output_paths["csv"] / "vitpose_keypoints_3d_world_flat.csv")
    export_session_json(
        {
            "session_id": session.session_id,
            "source": "vitpose_multiview",
            "calibration_mode": calibration_mode,
            "shape": {"keypoints_3d_world": list(arrays["keypoints_3d_world"].shape)},
            "keypoints_3d_world": arrays["keypoints_3d_world"],
            "triangulation_score": arrays["triangulation_score"],
            "reprojection_error": arrays["reprojection_error"],
            "used_cameras": arrays["used_cameras"],
        },
        output_paths["json"] / "vitpose_session_3d.json",
    )
    video_keypoints_3d = _repeat_keypoints_for_video(arrays["keypoints_3d_world"], output_repeats)
    write_3d_skeleton_video(
        video_keypoints_3d,
        output_paths["videos"] / "vitpose_skeleton_3d_world.mp4",
        fps=max(float(args.output_fps or fps), 1.0),
    )

    print(f"saved: {output_paths['videos'] / 'vitpose_skeleton_3d_world.mp4'}")
    print(f"keypoints_3d_world shape: {arrays['keypoints_3d_world'].shape}")
    print(f"calibration_mode: {calibration_mode}")

def build_pair_test_calibrations(camera_a: str, camera_b: str) -> dict[str, CameraCalibration]:
    intrinsic = np.array([[1200.0, 0.0, 960.0], [0.0, 1200.0, 540.0], [0.0, 0.0, 1.0]], dtype=float)
    return {
        camera_a: _calibration(camera_a, intrinsic, np.eye(3), np.array([0.0, 0.0, 0.0])),
        camera_b: _calibration(camera_b, intrinsic, _rotation_y(np.deg2rad(18.0)), np.array([0.75, 0.0, 0.04])),
    }


def _calibration(camera_id: str, intrinsic: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> CameraCalibration:
    import cv2

    projection = intrinsic @ np.hstack([rotation, translation.reshape(3, 1)])
    rvec, _ = cv2.Rodrigues(rotation)
    return CameraCalibration(
        camera_id=camera_id,
        image_size=(1920, 1080),
        intrinsic_matrix=intrinsic,
        distortion_coefficients=np.zeros(5),
        rotation_vector=rvec.reshape(-1),
        translation_vector=translation,
        projection_matrix=projection,
        reprojection_error_px=None,
    )


def _rotation_y(angle_rad: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(angle_rad), 0.0, np.sin(angle_rad)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle_rad), 0.0, np.cos(angle_rad)],
        ],
        dtype=float,
    )


def _min_source_frame_count(captures: list[cv2.VideoCapture]) -> int:
    counts = [int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0) for capture in captures]
    counts = [count for count in counts if count > 0]
    return min(counts) if counts else 0


def _target_sample_count(source_frames: int, max_frames: int | None, stride: int) -> int:
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


def _repeat_keypoints_for_video(keypoints_3d: np.ndarray, repeats: list[int]) -> np.ndarray:
    if keypoints_3d.size == 0 or not repeats:
        return keypoints_3d
    safe_repeats = np.asarray(repeats[: keypoints_3d.shape[0]], dtype=int)
    safe_repeats = np.maximum(safe_repeats, 1)
    return np.repeat(keypoints_3d[: safe_repeats.shape[0]], safe_repeats, axis=0)


if __name__ == "__main__":
    main()

