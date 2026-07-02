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
from src.pose2d_estimator import Pose2DConfig, RTMW2DEstimator
from src.smoothing_3d import moving_average_nan
from src.triangulation import stack_triangulated, triangulate_frame
from src.video_io import ensure_output_tree, load_session
from src.visualization_2d import draw_pose2d
from src.visualization_3d import write_3d_skeleton_video


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RTMW 2D + multi-view 3D test pipeline.")
    parser.add_argument("--session", required=True)
    parser.add_argument("--model-config", default="config/model_config.yaml")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--max-frames", type=int, default=30)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-cameras", type=int, default=None, help="Optional limit for faster tests; default uses all session cameras.")
    args = parser.parse_args()

    session = load_session(args.session)
    if len(session.cameras) < 2:
        raise SystemExit("Need at least two cameras.")
    cameras = session.cameras[: args.max_cameras] if args.max_cameras else session.cameras
    output_paths = ensure_output_tree(ROOT / args.output_root, session.session_id)

    with (ROOT / args.model_config).open("r", encoding="utf-8") as file:
        model_config = yaml.safe_load(file)
    pose2d_config = model_config["pose2d"]
    estimator = RTMW2DEstimator(
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
            calibrations = build_pair_test_calibrations(cameras[0].camera_id, cameras[1].camera_id)
            calibration_mode = "approximate_test_calibration"
            print("WARNING: calibration/cameras.json does not match session cameras. Using approximate test calibration.")
    else:
        calibrations = build_pair_test_calibrations(cameras[0].camera_id, cameras[1].camera_id)
        calibration_mode = "approximate_test_calibration"
        print("WARNING: calibration/cameras.json not found. Using approximate test calibration, not metric 3D.")

    captures = [cv2.VideoCapture(str(camera.video_path)) for camera in cameras]
    if not all(capture.isOpened() for capture in captures):
        raise SystemExit("Could not open all selected videos.")

    fps = captures[0].get(cv2.CAP_PROP_FPS) or 30.0
    overlay_writers = []
    for camera, capture in zip(cameras, captures):
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output_path = output_paths["videos"] / f"{camera.camera_id}_rtmw_2d_overlay.mp4"
        overlay_writers.append(cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps / max(args.stride, 1), (width, height)))

    triangulated = []
    try:
        frame_idx = 0
        written = 0
        while written < args.max_frames:
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
            for camera, frame, writer in zip(cameras, frames, overlay_writers):
                pose = estimator.predict(frame, camera.camera_id, frame_idx)
                poses_by_camera[camera.camera_id] = pose
                writer.write(draw_pose2d(frame, pose))

            triangulated.append(
                triangulate_frame(
                    frame_idx=frame_idx,
                    poses_by_camera=poses_by_camera,
                    calibrations=calibrations,
                    min_views=2,
                )
            )
            written += 1
            frame_idx += 1
    finally:
        for capture in captures:
            capture.release()
        for writer in overlay_writers:
            writer.release()

    arrays = stack_triangulated(triangulated)
    arrays["keypoints_3d_world"] = moving_average_nan(arrays["keypoints_3d_world"], window_size=5)
    export_keypoints3d_csv(arrays["keypoints_3d_world"], output_paths["csv"] / "rtmw_keypoints_3d_world_flat.csv")
    export_session_json(
        {
            "session_id": session.session_id,
            "source": "rtmw_multiview",
            "calibration_mode": calibration_mode,
            "shape": {"keypoints_3d_world": list(arrays["keypoints_3d_world"].shape)},
            "keypoints_3d_world": arrays["keypoints_3d_world"],
            "triangulation_score": arrays["triangulation_score"],
            "reprojection_error": arrays["reprojection_error"],
            "used_cameras": arrays["used_cameras"],
        },
        output_paths["json"] / "rtmw_session_3d.json",
    )
    write_3d_skeleton_video(arrays["keypoints_3d_world"], output_paths["videos"] / "rtmw_skeleton_3d_world.mp4", fps=fps / max(args.stride, 1))

    print(f"saved: {output_paths['videos'] / 'rtmw_skeleton_3d_world.mp4'}")
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


if __name__ == "__main__":
    main()

