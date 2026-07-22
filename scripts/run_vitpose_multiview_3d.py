from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.camera_calibration import load_calibration_bundle
from src.coordinate_system import (
    ANALYSIS_COORDINATE_SYSTEM,
    opencv_reference_to_analysis,
    require_source_to_analysis,
    transform_points,
)
from src.config_validation import validate_model_config
from src.data_structures import CameraCalibration, PersonPose2D
from src.exporter import export_keypoints2d_csv, export_keypoints3d_csv, export_session_json
from src.multiview_sync import SynchronizedFrame, synchronized_frame_map
from src.pose2d_estimator import Pose2DConfig, ViTPose2DEstimator
from src.person_tracking import person_detector_config_from_mapping
from src.pose2d_sequence import pose2d_at_frame
from src.pose_reliability import filter_unreliable_pose
from src.progress import ProgressBar
from src.run_outputs import create_run_output_tree, mark_run_complete
from src.smoothing_3d import moving_average_pose
from src.triangulation import stack_triangulated, triangulate_frame
from src.video_io import load_session
from src.visualization_2d import draw_pose2d
from src.visualization_3d import write_3d_skeleton_video


PRODUCTION_CALIBRATION_MODES = {
    "multiview_common_reference",
    "aist_official_multiview",
    "mads_official_multiview",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ViTPose-Huge 2D + multi-view 3D test pipeline.")
    parser.add_argument("--session", required=True)
    parser.add_argument("--model-config", default="config/model_config.yaml")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum sampled inference frames. Omit for full video duration.")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=None,
        help=(
            "Optional odd smoothing window. By default the configured window is used for stride=1, "
            "while sparse stride runs use window=1 to avoid blending distant moments."
        ),
    )
    parser.add_argument("--max-cameras", type=int, default=None, help="Optional limit for faster tests; default uses all session cameras.")
    parser.add_argument("--progress-every", type=int, default=1, help="Print progress every N written frames.")
    parser.add_argument("--output-fps", type=float, default=None, help="Playback FPS. Defaults to the source video FPS.")
    parser.add_argument("--run-id", default=None, help="Unique output run identifier; defaults to a UTC timestamp.")
    parser.add_argument(
        "--allow-approximate-calibration",
        action="store_true",
        help="Explicitly allow non-metric two-camera preview calibration. Never use for scoring.",
    )
    parser.add_argument(
        "--allow-low-quality-output",
        action="store_true",
        help="Keep diagnostic files when quality gates fail; the run is not promoted as latest.",
    )
    args = parser.parse_args()

    session = load_session(args.session)
    if len(session.cameras) < 2:
        raise SystemExit("Need at least two cameras.")
    cameras = session.cameras[: args.max_cameras] if args.max_cameras else session.cameras
    if len(cameras) < 2:
        raise SystemExit("Need at least two selected cameras. Increase --max-cameras or update the session.")
    output_root = (ROOT / args.output_root).resolve()

    with (ROOT / args.model_config).open("r", encoding="utf-8") as file:
        model_config = validate_model_config(yaml.safe_load(file))
    pose2d_config = model_config["pose2d"]
    smoothing_window = _effective_smoothing_window(
        configured_window=int(model_config.get("smoothing", {}).get("window_size", 5)),
        stride=args.stride,
        override=args.smoothing_window,
    )
    calibrations_path = output_root / session.session_id / "calibration" / "cameras.json"
    if calibrations_path.exists():
        bundle = load_calibration_bundle(calibrations_path)
        calibrations = bundle.calibrations
        if all(camera.camera_id in calibrations for camera in cameras):
            calibration_mode = str(bundle.metadata.get("calibration_mode", "legacy_unknown"))
            if calibration_mode not in PRODUCTION_CALIBRATION_MODES:
                raise SystemExit(
                    f"Calibration mode is not production-ready: {calibration_mode}. "
                    "Re-run calibration/import with the current TK3D version."
                )
            source_to_analysis = require_source_to_analysis(bundle.metadata)
        else:
            if not args.allow_approximate_calibration:
                missing = [camera.camera_id for camera in cameras if camera.camera_id not in calibrations]
                raise SystemExit(f"Calibration is missing selected cameras: {missing}")
            cameras = cameras[:2]
            calibrations = build_pair_test_calibrations(cameras[0].camera_id, cameras[1].camera_id)
            calibration_mode = "approximate_test_calibration"
            source_to_analysis = opencv_reference_to_analysis()
    else:
        if not args.allow_approximate_calibration:
            raise SystemExit(
                f"Production calibration not found: {calibrations_path}. "
                "Run calibrate_cameras.py or import_aist_cameras.py first."
            )
        cameras = cameras[:2]
        calibrations = build_pair_test_calibrations(cameras[0].camera_id, cameras[1].camera_id)
        calibration_mode = "approximate_test_calibration"
        source_to_analysis = opencv_reference_to_analysis()

    run_id, output_paths = create_run_output_tree(output_root, session.session_id, args.run_id)
    production_ready_calibration = calibration_mode in PRODUCTION_CALIBRATION_MODES

    print("=" * 72, flush=True)
    print("TK3D VITPOSE MULTI-VIEW 3D", flush=True)
    print("=" * 72, flush=True)
    print("[1/4] Loading ViTPose model", flush=True)
    print(f"      model : {pose2d_config['model_name']}", flush=True)
    print(f"      device: {pose2d_config.get('device', 'cuda:0')}", flush=True)
    estimator = ViTPose2DEstimator(
        Pose2DConfig(
            model_name=pose2d_config["model_name"],
            config_path=(ROOT / pose2d_config["config_path"]).resolve(),
            checkpoint_path=(ROOT / pose2d_config["checkpoint_path"]).resolve(),
            adapter_checkpoint_path=(
                (ROOT / pose2d_config["adapter_checkpoint_path"]).resolve()
                if pose2d_config.get("adapter_checkpoint_path")
                else None
            ),
            allow_unapproved_adapter=bool(pose2d_config.get("allow_unapproved_adapter", False)),
            device=pose2d_config.get("device", "cuda:0"),
            score_threshold=float(pose2d_config.get("score_threshold", 0.30)),
            input_size=tuple(int(value) for value in pose2d_config.get("input_size", [256, 192])),
            flip_test=bool(pose2d_config.get("flip_test", True)),
            temporal_filter_enabled=bool(pose2d_config.get("temporal_filter_enabled", True)),
            temporal_stabilize_left_right=bool(
                pose2d_config.get("temporal_stabilize_left_right", True)
            ),
            person_detector=person_detector_config_from_mapping(
                model_config.get("person_detector"),
                frame_rate=session.fps / max(args.stride, 1),
            ),
        )
    )

    captures = [cv2.VideoCapture(str(camera.video_path)) for camera in cameras]
    if not all(capture.isOpened() for capture in captures):
        raise SystemExit("Could not open all selected videos.")

    fps_by_camera = {
        camera.camera_id: float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        for camera, capture in zip(cameras, captures, strict=True)
    }
    if any(value <= 0 for value in fps_by_camera.values()):
        raise SystemExit(f"Every video must report a valid FPS: {fps_by_camera}")
    fps = float(session.fps or min(fps_by_camera.values()))
    for camera, capture in zip(cameras, captures, strict=True):
        actual_size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        expected_size = tuple(calibrations[camera.camera_id].image_size)
        if actual_size != expected_size and calibration_mode != "approximate_test_calibration":
            raise SystemExit(
                f"{camera.camera_id}: video size {actual_size} does not match calibration {expected_size}"
            )
    print(
        "[2/4] Preparing videos and calibration\n"
        f"      cameras         : {len(cameras)}\n"
        f"      target frames   : {args.max_frames or 'full video'}\n"
        f"      stride          : {max(args.stride, 1)}\n"
        f"      smoothing window: {smoothing_window}\n"
        f"      calibration     : {calibration_mode}\n"
        "[3/4] Running 2D pose + 3D triangulation",
        flush=True,
    )
    overlay_paths: dict[str, Path] = {}
    for camera, capture in zip(cameras, captures):
        output_path = output_paths["videos"] / f"{camera.camera_id}_vitpose_2d_overlay.mp4"
        overlay_paths[camera.camera_id] = output_path

    triangulated = []
    poses_2d_by_frame: dict[int, dict[str, object]] = {}
    frame_counts = {
        camera.camera_id: int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        for camera, capture in zip(cameras, captures, strict=True)
    }
    frame_offsets = {camera.camera_id: int(camera.frame_offset) for camera in cameras}
    synced_frames = synchronized_frame_map(
        frame_counts=frame_counts,
        fps_by_camera=fps_by_camera,
        frame_offsets=frame_offsets,
        time_offsets_sec={camera.camera_id: camera.time_offset_sec for camera in cameras},
        target_fps=fps,
    )
    if not synced_frames:
        raise SystemExit("Selected camera videos have no overlapping synchronized timeline")
    source_frame_count = len(synced_frames)
    target_frames = _target_sample_count(source_frame_count, args.max_frames, args.stride)
    output_repeats: list[int] = []
    sampled_poses_by_camera = {camera.camera_id: [] for camera in cameras}
    processed_overlap_count = 0
    progress = ProgressBar("2D + 3D", target_frames)
    try:
        written = 0
        next_local_frame_by_camera = {camera.camera_id: 0 for camera in cameras}
        for overlap_idx, sync_frame in enumerate(synced_frames):
            global_frame_idx = sync_frame.global_frame_idx
            if args.max_frames is not None and written >= args.max_frames:
                break
            if overlap_idx % args.stride != 0:
                continue

            poses_by_camera = {}
            repeat_count = _repeat_count(overlap_idx, source_frame_count, args.stride)
            frames: list[np.ndarray] = []
            for camera, capture in zip(cameras, captures):
                local_frame_idx = sync_frame.local_frame_indices[camera.camera_id]
                frame = _read_frame_sequential(
                    capture=capture,
                    camera_id=camera.camera_id,
                    target_frame_idx=local_frame_idx,
                    next_frame_by_camera=next_local_frame_by_camera,
                )
                if frame is None:
                    frames = []
                    break
                frames.append(frame)
            if not frames:
                break

            camera_ids = [camera.camera_id for camera in cameras]
            local_frame_indices = [
                sync_frame.local_frame_indices[camera.camera_id]
                for camera in cameras
            ]
            poses = estimator.predict_many(frames, camera_ids, local_frame_indices)
            for camera, pose in zip(cameras, poses, strict=True):
                poses_by_camera[camera.camera_id] = pose
                sampled_poses_by_camera[camera.camera_id].append(pose)
            poses_2d_by_frame[global_frame_idx] = dict(poses_by_camera)

            triangulated.append(
                triangulate_frame(
                    frame_idx=global_frame_idx,
                    poses_by_camera=poses_by_camera,
                    calibrations=calibrations,
                    min_views=2,
                    max_reprojection_error_px=float(model_config["triangulation"].get("max_reprojection_error_px", 25.0)),
                    max_hypotheses=int(model_config["triangulation"].get("max_hypotheses", 16)),
                )
            )
            output_repeats.append(repeat_count)
            processed_overlap_count = min(overlap_idx + repeat_count, source_frame_count)
            written += 1
            if written == 1 or written == target_frames or written % max(args.progress_every, 1) == 0:
                progress.print(written, extra=f"global frame {global_frame_idx}")
    finally:
        for capture in captures:
            capture.release()
        if triangulated:
            progress.done()
            print("[4/4] Saving 3D outputs", flush=True)

    if triangulated:
        render_frames = synced_frames[:processed_overlap_count]
        for camera in cameras:
            _write_synced_pose_overlay(
                video_path=camera.video_path,
                output_path=overlay_paths[camera.camera_id],
                camera_id=camera.camera_id,
                sampled_poses=sampled_poses_by_camera[camera.camera_id],
                synchronized_frames=render_frames,
                output_fps=max(float(args.output_fps or fps), 1.0),
                progress_every=max(args.progress_every, 1),
            )

    arrays = stack_triangulated(triangulated)
    arrays["keypoints_3d_world"] = transform_points(arrays["keypoints_3d_world"], source_to_analysis)
    max_error = float(model_config["triangulation"].get("max_reprojection_error_px", 25.0))
    min_quality = float(model_config["triangulation"].get("min_triangulation_score", 0.20))
    accepted = (
        np.isfinite(arrays["reprojection_error"])
        & (arrays["reprojection_error"] <= max_error)
        & (arrays["triangulation_score"] >= min_quality)
        & (arrays["used_cameras"] >= int(model_config["triangulation"].get("min_views", 2)))
    )
    sampled_timestamps = np.asarray(
        [sync_frame.timestamp_sec for index, sync_frame in enumerate(synced_frames) if index % max(args.stride, 1) == 0][
            : arrays["frame_idx"].shape[0]
        ],
        dtype=float,
    )
    reliability_config = model_config.get("reliability", {})
    reliability = filter_unreliable_pose(
        arrays["keypoints_3d_world"],
        accepted,
        sampled_timestamps,
        confidence=arrays["triangulation_score"],
        max_bone_relative_deviation=float(
            reliability_config.get("max_bone_relative_deviation", 0.25)
        ),
        max_bone_absolute_deviation_m=float(
            reliability_config.get("max_bone_absolute_deviation_m", 0.08)
        ),
        min_temporal_residual_m=float(reliability_config.get("min_temporal_residual_m", 0.08)),
        max_temporal_acceleration_mps2=float(
            reliability_config.get("max_temporal_acceleration_mps2", 70.0)
        ),
        minimum_bone_samples=int(reliability_config.get("minimum_bone_samples", 5)),
    )
    arrays["keypoints_3d_world"] = moving_average_pose(
        reliability.keypoints_3d,
        window_size=smoothing_window,
        valid_mask=reliability.valid_mask,
    )
    arrays["keypoints_3d_world"] = np.where(
        reliability.valid_mask[..., None], arrays["keypoints_3d_world"], np.nan
    )
    export_session_json(
        reliability.summary,
        output_paths["json"] / "pose_reliability_report.json",
    )
    video_arrays = _repeat_arrays_for_video(arrays, output_repeats)
    export_keypoints3d_csv(
        arrays["keypoints_3d_world"],
        output_paths["csv"] / "vitpose_keypoints_3d_world_flat.csv",
        frame_indices=arrays["frame_idx"],
        timestamps_sec=sampled_timestamps,
    )
    export_keypoints2d_csv(poses_2d_by_frame, output_paths["csv"] / "vitpose_keypoints_2d_flat.csv")
    export_session_json(
        {
            "session_id": session.session_id,
            "run_id": run_id,
            "source": "vitpose_multiview",
            "calibration_mode": calibration_mode,
            "production_ready_calibration": production_ready_calibration,
            "coordinate_system": ANALYSIS_COORDINATE_SYSTEM,
            "frame_indices": arrays["frame_idx"],
            "timestamps_sec": sampled_timestamps,
            "sample_fps": fps / max(args.stride, 1),
            "smoothing_applied": smoothing_window > 1,
            "smoothing_window": smoothing_window,
            "inference_stride": max(args.stride, 1),
            "inference_sample_count": int(arrays["keypoints_3d_world"].shape[0]),
            "output_frame_count": int(video_arrays["keypoints_3d_world"].shape[0]),
            "shape": {"keypoints_3d_world": list(arrays["keypoints_3d_world"].shape)},
            "keypoints_3d_world": arrays["keypoints_3d_world"],
            "triangulation_score": arrays["triangulation_score"],
            "reprojection_error": arrays["reprojection_error"],
            "used_cameras": arrays["used_cameras"],
            "reliability_valid_mask": reliability.valid_mask,
            "reliability_rejection_reasons": reliability.rejection_reasons,
            "reliability_summary": reliability.summary,
        },
        output_paths["json"] / "vitpose_session_3d.json",
    )
    body_count = min(17, arrays["keypoints_3d_world"].shape[1])
    body_valid = np.all(np.isfinite(arrays["keypoints_3d_world"][:, :body_count]), axis=-1)
    mean_body_valid_ratio = float(np.mean(body_valid)) if body_valid.size else 0.0
    finite_errors = arrays["reprojection_error"][np.isfinite(arrays["reprojection_error"])]
    mean_reprojection_error = float(np.mean(finite_errors)) if finite_errors.size else None
    minimum_body_valid_ratio = float(reliability_config.get("min_output_valid_body_ratio", 0.90))
    quality_passed = bool(
        production_ready_calibration
        and mean_body_valid_ratio >= minimum_body_valid_ratio
        and mean_reprojection_error is not None
        and mean_reprojection_error <= max_error
    )
    export_session_json(
        {
            "session_id": session.session_id,
            "run_id": run_id,
            "status": "passed" if quality_passed else "failed",
            "quality_scope": "internal_geometry_only",
            "ground_truth_accuracy_evaluated": False,
            "scoring_ready": False,
            "scoring_readiness_reason": (
                "A separate ground-truth validation must pass before this run can support scoring."
            ),
            "production_ready_calibration": production_ready_calibration,
            "mean_body17_valid_ratio": mean_body_valid_ratio,
            "mean_reprojection_error_px": mean_reprojection_error,
            "max_reprojection_error_px": max_error,
            "reliability_filter": reliability.summary,
            "minimum_required_body17_valid_ratio": minimum_body_valid_ratio,
        },
        output_paths["json"] / "run_quality_report.json",
    )
    write_3d_skeleton_video(
        video_arrays["keypoints_3d_world"],
        output_paths["videos"] / "vitpose_skeleton_3d_world.mp4",
        fps=max(float(args.output_fps or fps), 1.0),
    )
    if quality_passed:
        mark_run_complete(output_root, session.session_id, run_id, output_paths["root"])

    print(f"saved: {output_paths['videos'] / 'vitpose_skeleton_3d_world.mp4'}")
    print(f"keypoints_3d_world shape: {video_arrays['keypoints_3d_world'].shape}")
    print(f"inference_sample_count: {arrays['keypoints_3d_world'].shape[0]}")
    print(f"calibration_mode: {calibration_mode}")
    print(f"run_id: {run_id}")
    print(f"internal_geometry_quality_status: {'passed' if quality_passed else 'failed'}")
    print("ground_truth_accuracy_status: not_evaluated_in_this_command")
    print("scoring_ready: false")
    if not quality_passed and not args.allow_low_quality_output:
        raise SystemExit(
            "3D output failed production quality gates. Diagnostic files were kept, "
            "but this run was not promoted as latest. Inspect run_quality_report.json."
        )

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


def _read_frame_sequential(
    capture: cv2.VideoCapture,
    camera_id: str,
    target_frame_idx: int,
    next_frame_by_camera: dict[str, int],
) -> np.ndarray | None:
    if target_frame_idx < 0:
        return None
    next_frame_idx = int(next_frame_by_camera.get(camera_id, 0))
    if target_frame_idx < next_frame_idx:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(target_frame_idx))
        next_frame_idx = target_frame_idx
    frame = None
    while next_frame_idx <= target_frame_idx:
        ok, frame = capture.read()
        if not ok:
            return None
        next_frame_idx += 1
    next_frame_by_camera[camera_id] = next_frame_idx
    return frame


def _write_synced_pose_overlay(
    video_path: Path,
    output_path: Path,
    camera_id: str,
    sampled_poses: list[PersonPose2D],
    synchronized_frames: list[SynchronizedFrame],
    output_fps: float,
    progress_every: int,
) -> None:
    if not sampled_poses or not synchronized_frames:
        return
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video for overlay rendering: {video_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open overlay video writer: {output_path}")
    next_frame = {camera_id: 0}
    progress = ProgressBar(f"{camera_id} overlay render", len(synchronized_frames))
    try:
        for output_idx, sync_frame in enumerate(synchronized_frames):
            local_idx = int(sync_frame.local_frame_indices[camera_id])
            frame = _read_frame_sequential(capture, camera_id, local_idx, next_frame)
            if frame is None:
                break
            pose = pose2d_at_frame(sampled_poses, local_idx)
            writer.write(draw_pose2d(frame, pose))
            if output_idx == 0 or output_idx + 1 >= len(synchronized_frames) or (output_idx + 1) % progress_every == 0:
                progress.print(output_idx + 1, extra=f"src frame {local_idx}")
    finally:
        capture.release()
        writer.release()
        progress.done()


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


def _effective_smoothing_window(configured_window: int, stride: int, override: int | None) -> int:
    window = int(override) if override is not None else (1 if int(stride) > 1 else int(configured_window))
    if window < 1 or window % 2 == 0:
        raise SystemExit("--smoothing-window must be a positive odd integer")
    return window


def _repeat_arrays_for_video(arrays: dict[str, np.ndarray], repeats: list[int]) -> dict[str, np.ndarray]:
    return {
        key: (
            _repeat_array_for_video(value, repeats)
            if key == "used_cameras"
            else _interpolate_array_for_video(value, repeats)
        )
        if key in {"keypoints_3d_world", "triangulation_score", "reprojection_error", "used_cameras"}
        else value
        for key, value in arrays.items()
    }


def _repeat_array_for_video(values: np.ndarray, repeats: list[int]) -> np.ndarray:
    if values.size == 0 or not repeats:
        return values
    safe_repeats = np.asarray(repeats[: values.shape[0]], dtype=int)
    safe_repeats = np.maximum(safe_repeats, 1)
    return np.repeat(values[: safe_repeats.shape[0]], safe_repeats, axis=0)


def _interpolate_array_for_video(values: np.ndarray, repeats: list[int]) -> np.ndarray:
    if values.size == 0 or not repeats:
        return values
    sample_count = min(values.shape[0], len(repeats))
    output: list[np.ndarray] = []
    for sample_idx in range(sample_count):
        count = max(int(repeats[sample_idx]), 1)
        current = np.asarray(values[sample_idx])
        if sample_idx + 1 >= sample_count:
            output.extend(current.copy() for _ in range(count))
            continue
        following = np.asarray(values[sample_idx + 1])
        for offset in range(count):
            weight = offset / count
            output.append((1.0 - weight) * current + weight * following)
    return np.stack(output, axis=0)


if __name__ == "__main__":
    main()
