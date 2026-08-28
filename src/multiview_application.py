from __future__ import annotations

import json
import time
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np
import yaml

from src.artifact_contracts import (
    MAIN_3D_SCHEMA_VERSION,
    RUN_QUALITY_SCHEMA_VERSION,
    load_main_3d_artifact,
    validate_artifact_manifest_binding,
    validate_run_quality_artifact,
)
from src.camera_calibration import PRODUCTION_CALIBRATION_MODES, load_calibration_bundle, save_calibrations
from src.coordinate_system import (
    ANALYSIS_COORDINATE_SYSTEM,
    opencv_reference_to_analysis,
    require_source_to_analysis,
    transform_points,
)
from src.config_validation import validate_model_config
from src.data_structures import (
    COCO_BODY_JOINT_NAMES,
    COCO_WHOLEBODY_KEYPOINTS,
    CameraCalibration,
    PersonPose2D,
)
from src.crossview_pose2d_feedback import (
    PROVENANCE_CROSSVIEW_PROJECTED,
    PROVENANCE_IMAGE_GUIDED,
    CrossView2DFeedbackConfig,
    CrossViewFeedbackPlan,
    build_feedback_plan,
    copy_pose,
    decide_guided_candidate,
    feedback_config_from_mapping,
    finalize_feedback_report,
)
from src.exporter import (
    export_depth_fusion_observations_csv,
    export_keypoints2d_csv,
    export_keypoints3d_csv,
    export_pose2d_feedback_provenance_csv,
    export_pose3d_provenance_csv,
    export_session_json,
)
from src.multiview_pose_optimization import (
    PROVENANCE_OBSERVED,
    PROVENANCE_TEMPORALLY_RECOVERED,
    optimization_config_from_mapping,
    optimize_body_sequence,
)
from src.multiview_sync import SynchronizedFrame, synchronized_frame_map
from src.pose2d_estimator import Pose2DConfig, ViTPose2DEstimator
from src.person_tracking import person_detector_config_from_mapping
from src.performance import PerformanceCollector, write_performance_report
from src.pose2d_sequence import pose2d_at_frame
from src.pose2d_stabilization import (
    Pose2DStabilizationConfig,
    pose2d_stability_metrics,
    stabilize_pose2d_sequence,
)
from src.pose3d_stability import pose3d_stability_metrics
from src.pose3d_html_viewer import write_pose3d_html_viewer
from src.pose_reliability import filter_unreliable_pose
from src.progress import ProgressBar
from src.quality_status import external_accuracy_not_evaluated, internal_sensor_consistency_status
from src.run_outputs import create_run_output_tree, mark_run_complete, mark_run_completed, mark_run_running
from src.run_manifest import (
    build_run_manifest,
    model_provenance,
    sha256_file,
    snapshot_file,
    utc_now,
    write_run_manifest,
)
from src.smoothing_3d import smooth_pose_sequence
from src.triangulation import stack_triangulated, triangulate_frame
from src.video_io import load_session
from src.visualization_2d import draw_pose2d
from src.visualization_3d import write_3d_skeleton_video
from src.zed_depth_fusion import (
    depth_fusion_config_from_mapping,
    final_depth_fusion_acceptance_gate,
    fuse_zed_depth_sequence,
    load_depth_sources,
)


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class MultiviewRunOptions:
    session: str | Path
    model_config: str | Path = "config/model_config.yaml"
    output_root: str | Path = "outputs"
    max_frames: int | None = None
    stride: int = 1
    smoothing_window: int | None = None
    max_cameras: int | None = None
    progress_every: int = 1
    output_fps: float | None = None
    run_id: str | None = None
    allow_approximate_calibration: bool = False
    allow_low_quality_output: bool = False
    promote_latest: bool = True
    invocation_argv: tuple[str, ...] = ()
    profile_performance: bool = False
    benchmark_window_start: int = 140
    benchmark_window_end: int = 259
    profiler_baseline_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class MultiviewRunResult:
    session_id: str
    run_id: str
    run_root: Path
    main_3d_path: Path
    quality_path: Path
    manifest_path: Path
    quality_passed: bool


class MultiviewQualityError(RuntimeError):
    def __init__(self, message: str, result: MultiviewRunResult):
        super().__init__(message)
        self.result = result


def run_multiview_pose(options: MultiviewRunOptions) -> MultiviewRunResult:
    profiler = (
        PerformanceCollector(
            benchmark_window=(options.benchmark_window_start, options.benchmark_window_end),
        )
        if options.profile_performance
        else None
    )
    started_at = utc_now()
    args = options
    if args.stride < 1:
        raise ValueError("stride must be a positive integer")
    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError("max_frames must be positive when provided")

    session = load_session(args.session)
    if len(session.cameras) < 2:
        raise SystemExit("Need at least two cameras.")
    cameras = session.cameras[: args.max_cameras] if args.max_cameras else session.cameras
    if len(cameras) < 2:
        raise SystemExit("Need at least two selected cameras. Increase --max-cameras or update the session.")
    output_root = (ROOT / args.output_root).resolve()

    session_path = Path(args.session).resolve()
    model_config_path = (ROOT / args.model_config).resolve()
    with model_config_path.open("r", encoding="utf-8") as file:
        model_config = validate_model_config(yaml.safe_load(file))
    pose2d_config = model_config["pose2d"]
    min_views = int(model_config["triangulation"].get("min_views", 2))
    min_keypoint_score = float(model_config["triangulation"].get("min_keypoint_score", 0.30))
    global_optimization_config = optimization_config_from_mapping(model_config.get("global_optimization"))
    depth_fusion_config = depth_fusion_config_from_mapping(model_config.get("zed_depth_fusion"))
    crossview_feedback_config = feedback_config_from_mapping(
        model_config.get("crossview_2d_feedback")
    )
    if args.stride != 1 and global_optimization_config.enabled:
        global_optimization_config = replace(global_optimization_config, enabled=False)
    smoothing_config = model_config.get("smoothing", {})
    smoothing_method = str(smoothing_config.get("method", "robust_savgol"))
    smoothing_window = _effective_smoothing_window(
        configured_window=int(smoothing_config.get("window_size", 11)),
        stride=args.stride,
        override=args.smoothing_window,
    )
    smoothing_polynomial_order = int(smoothing_config.get("polynomial_order", 2))
    smoothing_min_outlier_distance_m = float(smoothing_config.get("min_outlier_distance_m", 0.04))
    offline_2d_config = pose2d_config.get("offline_stabilization", {})
    pose2d_stabilization_config = Pose2DStabilizationConfig(
        enabled=bool(offline_2d_config.get("enabled", True)) and args.stride == 1,
        window_size=int(offline_2d_config.get("window_size", 9)),
        polynomial_order=int(offline_2d_config.get("polynomial_order", 2)),
        min_outlier_distance_px=float(offline_2d_config.get("min_outlier_distance_px", 6.0)),
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

    if min_views > len(cameras):
        raise SystemExit(
            f"triangulation.min_views={min_views} requires at least {min_views} selected cameras; got {len(cameras)}"
        )
    run_id, output_paths = create_run_output_tree(output_root, session.session_id, args.run_id)
    mark_run_running(output_paths["root"], session.session_id, run_id)
    production_ready_calibration = calibration_mode in PRODUCTION_CALIBRATION_MODES
    config_dir = output_paths["root"] / "config"
    pose_model_config_path = (ROOT / pose2d_config["config_path"]).resolve()
    config_snapshots = {
        "session": snapshot_file(session_path, config_dir / "session.yaml"),
        "model_config": snapshot_file(model_config_path, config_dir / "model_config.yaml"),
        "pose_model_config": snapshot_file(
            pose_model_config_path,
            config_dir / "pose_model_config.py",
        ),
    }
    calibration_snapshot_path = output_paths["calibration"] / "cameras.json"
    if calibrations_path.is_file() and calibration_mode != "approximate_test_calibration":
        calibration_provenance = snapshot_file(calibrations_path, calibration_snapshot_path)
    else:
        if calibration_snapshot_path.exists():
            raise FileExistsError(f"Calibration snapshot already exists: {calibration_snapshot_path}")
        save_calibrations(
            list(calibrations.values()),
            calibration_snapshot_path,
            metadata={
                "calibration_mode": calibration_mode,
                "analysis_coordinate_system": ANALYSIS_COORDINATE_SYSTEM,
                "source_to_analysis": source_to_analysis.tolist(),
            },
        )
        calibration_provenance = {
            "source_path": None,
            "snapshot_path": str(calibration_snapshot_path.resolve()),
            "sha256": sha256_file(calibration_snapshot_path),
            "size_bytes": calibration_snapshot_path.stat().st_size,
        }
    calibration_schema_version = json.loads(
        calibration_snapshot_path.read_text(encoding="utf-8")
    ).get("schema_version")
    if isinstance(calibration_schema_version, bool) or not isinstance(calibration_schema_version, int):
        raise ValueError("Calibration snapshot must declare an integer schema_version")
    calibration_provenance["schema_version"] = calibration_schema_version
    pose_checkpoint_path = (ROOT / pose2d_config["checkpoint_path"]).resolve()
    detector_variant = str(model_config.get("person_detector", {}).get("model_variant", "unknown"))
    detector_filename = {
        "small": "rf-detr-small.pth",
        "medium": "rf-detr-medium.pth",
        "base": "rf-detr-base.pth",
        "large": "rf-detr-large.pth",
    }.get(detector_variant)
    detector_checkpoint_path = (
        Path.home() / ".roboflow" / "models" / detector_filename
        if detector_filename is not None
        else None
    )
    model_records = [
        model_provenance(
            str(pose2d_config["model_name"]),
            pose_checkpoint_path,
            pose_model_config_path,
        ),
        model_provenance(
            f"RF-DETR-{detector_variant}",
            detector_checkpoint_path,
            None,
        ),
    ]
    if pose2d_config.get("adapter_checkpoint_path"):
        model_records.append(
            model_provenance(
                "ViTPose-adapter",
                (ROOT / pose2d_config["adapter_checkpoint_path"]).resolve(),
                None,
            )
        )

    captures = [cv2.VideoCapture(str(camera.video_path)) for camera in cameras]
    if not all(capture.isOpened() for capture in captures):
        for capture in captures:
            capture.release()
        raise SystemExit("Could not open all selected videos.")

    fps_by_camera = {
        camera.camera_id: float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        for camera, capture in zip(cameras, captures, strict=True)
    }
    if any(not np.isfinite(value) or value <= 0 for value in fps_by_camera.values()):
        for capture in captures:
            capture.release()
        raise SystemExit(f"Every video must report a valid FPS: {fps_by_camera}")
    fps = _effective_timeline_fps(session.fps, fps_by_camera.values())
    for camera, capture in zip(cameras, captures, strict=True):
        actual_size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        expected_size = tuple(calibrations[camera.camera_id].image_size)
        if actual_size != expected_size and calibration_mode != "approximate_test_calibration":
            for opened_capture in captures:
                opened_capture.release()
            raise SystemExit(f"{camera.camera_id}: video size {actual_size} does not match calibration {expected_size}")
    print("=" * 72, flush=True)
    print("TK3D VITPOSE MULTI-VIEW 3D", flush=True)
    print("=" * 72, flush=True)
    print("[1/5] Loading ViTPose model", flush=True)
    print(f"      model : {pose2d_config['model_name']}", flush=True)
    print(f"      device: {pose2d_config.get('device', 'cuda:0')}", flush=True)
    if profiler is not None:
        profiler.record(
            "preflight_provenance",
            time.perf_counter() - profiler.started_perf_counter,
        )
    try:
        estimator_config = Pose2DConfig(
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
                temporal_stabilize_left_right=bool(pose2d_config.get("temporal_stabilize_left_right", True)),
                person_detector=person_detector_config_from_mapping(
                    model_config.get("person_detector"),
                    frame_rate=fps / args.stride,
                ),
            )
        if profiler is None:
            estimator = ViTPose2DEstimator(estimator_config)
        else:
            with profiler.activate():
                estimator = ViTPose2DEstimator(estimator_config)
            profiler.reset_cuda_peak_memory()
    except Exception:
        for capture in captures:
            capture.release()
        raise
    print(
        "[2/5] Preparing videos and calibration\n"
        f"      cameras         : {len(cameras)}\n"
        f"      target frames   : {args.max_frames or 'full video'}\n"
        f"      stride          : {max(args.stride, 1)}\n"
        f"      cross-view 2D   : {crossview_feedback_config.enabled}\n"
        f"      ZED depth fusion: {depth_fusion_config.enabled}\n"
        f"      global body opt : {global_optimization_config.enabled}\n"
        f"      smoothing       : {smoothing_method}, window {smoothing_window}\n"
        f"      calibration     : {calibration_mode}\n"
        "[3/5] Running 2D pose inference",
        flush=True,
    )
    overlay_paths: dict[str, Path] = {}
    for camera, capture in zip(cameras, captures):
        output_path = output_paths["videos"] / f"{camera.camera_id}_vitpose_2d_overlay.mp4"
        overlay_paths[camera.camera_id] = output_path

    triangulated = []
    poses_2d_by_frame: dict[int, dict[str, PersonPose2D]] = {}
    raw_poses_2d_by_frame: dict[int, dict[str, PersonPose2D]] = {}
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
        for capture in captures:
            capture.release()
        raise SystemExit("Selected camera videos have no overlapping synchronized timeline")
    source_frame_count = len(synced_frames)
    target_frames = _target_sample_count(source_frame_count, args.max_frames, args.stride)
    output_repeats: list[int] = []
    raw_sampled_poses_by_camera = {camera.camera_id: [] for camera in cameras}
    sampled_global_frame_indices: list[int] = []
    sampled_sync_frames: list[SynchronizedFrame] = []
    processed_overlap_count = 0
    progress = ProgressBar("2D pose", target_frames)
    try:
        written = 0
        next_local_frame_by_camera = {camera.camera_id: 0 for camera in cameras}
        for overlap_idx, sync_frame in enumerate(synced_frames):
            global_frame_idx = sync_frame.global_frame_idx
            if args.max_frames is not None and written >= args.max_frames:
                break
            if overlap_idx % args.stride != 0:
                continue

            repeat_count = _repeat_count(overlap_idx, source_frame_count, args.stride)
            frames: list[np.ndarray] = []
            decode_started = time.perf_counter() if profiler is not None else 0.0
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
            if profiler is not None:
                profiler.record(
                    "rgb_decode_synchronization",
                    time.perf_counter() - decode_started,
                    frame_index=global_frame_idx,
                )
            if not frames:
                break

            camera_ids = [camera.camera_id for camera in cameras]
            local_frame_indices = [sync_frame.local_frame_indices[camera.camera_id] for camera in cameras]
            if profiler is None:
                poses = estimator.predict_many(frames, camera_ids, local_frame_indices)
            else:
                with profiler.activate(frame_index=global_frame_idx, sample_index=written):
                    poses = estimator.predict_many(frames, camera_ids, local_frame_indices)
            for camera, pose in zip(cameras, poses, strict=True):
                raw_sampled_poses_by_camera[camera.camera_id].append(pose)
            sampled_global_frame_indices.append(global_frame_idx)
            sampled_sync_frames.append(sync_frame)
            output_repeats.append(repeat_count)
            processed_overlap_count = min(overlap_idx + repeat_count, source_frame_count)
            written += 1
            if written == 1 or written == target_frames or written % max(args.progress_every, 1) == 0:
                progress.print(written, extra=f"global frame {global_frame_idx}")
    finally:
        for capture in captures:
            capture.release()
        if sampled_global_frame_indices:
            progress.done()
            print("[4/5] Stabilizing 2D trajectories and triangulating 3D", flush=True)

    offline_2d_started = time.perf_counter() if profiler is not None else 0.0
    sampled_poses_by_camera = {
        camera.camera_id: stabilize_pose2d_sequence(
            raw_sampled_poses_by_camera[camera.camera_id],
            config=pose2d_stabilization_config,
        )
        for camera in cameras
    }
    if profiler is not None:
        profiler.record(
            "offline_2d_stabilization",
            time.perf_counter() - offline_2d_started,
        )
    pose2d_stability_report = {
        camera.camera_id: pose2d_stability_metrics(
            raw_sampled_poses_by_camera[camera.camera_id],
            sampled_poses_by_camera[camera.camera_id],
        )
        for camera in cameras
    }
    for sample_idx, global_frame_idx in enumerate(sampled_global_frame_indices):
        raw_by_camera = {
            camera.camera_id: raw_sampled_poses_by_camera[camera.camera_id][sample_idx] for camera in cameras
        }
        poses_by_camera = {
            camera.camera_id: sampled_poses_by_camera[camera.camera_id][sample_idx] for camera in cameras
        }
        raw_poses_2d_by_frame[global_frame_idx] = raw_by_camera
        poses_2d_by_frame[global_frame_idx] = poses_by_camera
        triangulation_started = time.perf_counter() if profiler is not None else 0.0
        triangulated.append(
            triangulate_frame(
                frame_idx=global_frame_idx,
                poses_by_camera=poses_by_camera,
                calibrations=calibrations,
                min_views=min_views,
                min_keypoint_score=min_keypoint_score,
                max_reprojection_error_px=float(model_config["triangulation"].get("max_reprojection_error_px", 25.0)),
                max_hypotheses=int(model_config["triangulation"].get("max_hypotheses", 16)),
            )
        )
        if profiler is not None:
            profiler.record(
                "triangulation",
                time.perf_counter() - triangulation_started,
                frame_index=global_frame_idx,
            )

    prefeedback_poses_2d_by_frame = poses_2d_by_frame
    initial_triangulated = triangulated
    initial_arrays_source = stack_triangulated(initial_triangulated)
    selected_calibrations = {
        camera.camera_id: calibrations[camera.camera_id]
        for camera in cameras
    }
    crossview_started = time.perf_counter() if profiler is not None else 0.0
    feedback_plan = build_feedback_plan(
        initial_arrays_source["frame_idx"],
        prefeedback_poses_2d_by_frame,
        initial_arrays_source["keypoints_3d_world"],
        selected_calibrations,
        crossview_feedback_config,
    )
    (
        feedback_geometry_by_camera,
        feedback_display_by_camera,
        feedback_provenance_by_camera,
        crossview_feedback_report,
    ) = _run_crossview_2d_feedback(
        estimator=estimator,
        cameras=cameras,
        sampled_sync_frames=sampled_sync_frames,
        sampled_global_frame_indices=sampled_global_frame_indices,
        baseline_by_camera=sampled_poses_by_camera,
        plan=feedback_plan,
        config=crossview_feedback_config,
        stabilization_config=pose2d_stabilization_config,
        progress_every=max(args.progress_every, 1),
    )
    if profiler is not None:
        profiler.record(
            "crossview_guided_second_pass",
            time.perf_counter() - crossview_started,
            tags={
                "enabled_in_config": crossview_feedback_config.enabled,
                "target_count": int(crossview_feedback_report.get("target_count", 0)),
                "runtime_work": int(crossview_feedback_report.get("target_count", 0)) > 0,
            },
        )
    geometry_poses_2d_by_frame = _poses_by_global_frame(
        sampled_global_frame_indices,
        cameras,
        feedback_geometry_by_camera,
    )
    display_poses_2d_by_frame = _poses_by_global_frame(
        sampled_global_frame_indices,
        cameras,
        feedback_display_by_camera,
    )
    geometry_change_count = int(
        crossview_feedback_report.get("geometry_changed_count", 0)
    )
    feedback_gate = {
        "passed": True,
        "reason": "no_image_guided_candidates",
    }
    if geometry_change_count:
        feedback_triangulated = _triangulate_sampled_poses(
            sampled_global_frame_indices,
            geometry_poses_2d_by_frame,
            selected_calibrations,
            min_views=min_views,
            min_keypoint_score=min_keypoint_score,
            max_reprojection_error_px=float(
                model_config["triangulation"].get("max_reprojection_error_px", 25.0)
            ),
            max_hypotheses=int(model_config["triangulation"].get("max_hypotheses", 16)),
        )
        feedback_gate = _feedback_triangulation_gate(
            initial_triangulated,
            feedback_triangulated,
        )
        if feedback_gate["passed"]:
            triangulated = feedback_triangulated
            sampled_poses_by_camera = feedback_geometry_by_camera
            poses_2d_by_frame = geometry_poses_2d_by_frame
        else:
            triangulated = initial_triangulated
            poses_2d_by_frame = prefeedback_poses_2d_by_frame
    else:
        triangulated = initial_triangulated
        poses_2d_by_frame = prefeedback_poses_2d_by_frame
    crossview_feedback_report["geometry_feedback_gate"] = feedback_gate
    crossview_feedback_report["geometry_feedback_applied"] = bool(
        geometry_change_count and feedback_gate["passed"]
    )

    overlay_started = time.perf_counter() if profiler is not None else 0.0
    if triangulated:
        render_frames = synced_frames[:processed_overlap_count]
        for camera in cameras:
            _write_synced_pose_overlay(
                video_path=camera.video_path,
                output_path=overlay_paths[camera.camera_id],
                camera_id=camera.camera_id,
                sampled_poses=feedback_display_by_camera[camera.camera_id],
                sampled_provenance=feedback_provenance_by_camera[camera.camera_id],
                synchronized_frames=render_frames,
                output_fps=max(float(args.output_fps or fps), 1.0),
                progress_every=max(args.progress_every, 1),
            )
    if profiler is not None:
        profiler.record("overlay_rendering", time.perf_counter() - overlay_started)

    arrays = stack_triangulated(triangulated)
    arrays["keypoints_3d_world"] = transform_points(arrays["keypoints_3d_world"], source_to_analysis)
    triangulated_3d_analysis = arrays["keypoints_3d_world"].copy()
    max_error = float(model_config["triangulation"].get("max_reprojection_error_px", 25.0))
    min_quality = float(model_config["triangulation"].get("min_triangulation_score", 0.20))
    accepted = (
        np.isfinite(arrays["reprojection_error"])
        & (arrays["reprojection_error"] <= max_error)
        & (arrays["triangulation_score"] >= min_quality)
        & (arrays["used_cameras"] >= min_views)
    )
    print("      reading confidence-gated ZED stereo depth", flush=True)
    depth_started = time.perf_counter() if profiler is not None else 0.0
    if profiler is None:
        depth_fusion = fuse_zed_depth_sequence(
            arrays["keypoints_3d_world"],
            accepted,
            arrays["frame_idx"],
            poses_2d_by_frame,
            selected_calibrations,
            source_to_analysis,
            args.session,
            depth_fusion_config,
            body_joint_count=17,
        )
    else:
        with profiler.activate():
            depth_fusion = fuse_zed_depth_sequence(
                arrays["keypoints_3d_world"],
                accepted,
                arrays["frame_idx"],
                poses_2d_by_frame,
                selected_calibrations,
                source_to_analysis,
                args.session,
                depth_fusion_config,
                body_joint_count=17,
            )
        profiler.record("zed_depth_fusion", time.perf_counter() - depth_started)
    arrays["keypoints_3d_world"] = depth_fusion.keypoints_3d_analysis
    depth_fused_3d_analysis = arrays["keypoints_3d_world"].copy()
    sampled_timestamps = np.asarray(
        [
            sync_frame.timestamp_sec
            for index, sync_frame in enumerate(synced_frames)
            if index % max(args.stride, 1) == 0
        ][: arrays["frame_idx"].shape[0]],
        dtype=float,
    )
    rgb_reference_optimization = None
    optimizers_started = time.perf_counter() if profiler is not None else 0.0
    if depth_fusion.report.get("applied"):
        print("      optimizing RGB reference BODY-17 sequence", flush=True)
        rgb_optimizer_started = time.perf_counter() if profiler is not None else 0.0
        rgb_reference_optimization = optimize_body_sequence(
            triangulated_3d_analysis,
            accepted,
            arrays["triangulation_score"],
            arrays["frame_idx"],
            sampled_timestamps,
            poses_2d_by_frame,
            selected_calibrations,
            np.linalg.inv(source_to_analysis),
            global_optimization_config,
        )
        if profiler is not None:
            profiler.record(
                "rgb_reference_optimizer",
                time.perf_counter() - rgb_optimizer_started,
                parent="global_optimizers_depth_gate",
            )
    print("      optimizing depth-candidate BODY-17 sequence", flush=True)
    depth_optimizer_started = time.perf_counter() if profiler is not None else 0.0
    depth_candidate_optimization = optimize_body_sequence(
        arrays["keypoints_3d_world"],
        accepted,
        arrays["triangulation_score"],
        arrays["frame_idx"],
        sampled_timestamps,
        poses_2d_by_frame,
        selected_calibrations,
        np.linalg.inv(source_to_analysis),
        global_optimization_config,
    )
    if profiler is not None:
        profiler.record(
            "depth_candidate_optimizer",
            time.perf_counter() - depth_optimizer_started,
            parent="global_optimizers_depth_gate",
        )
    global_optimization = depth_candidate_optimization
    if rgb_reference_optimization is not None:
        depth_gate_started = time.perf_counter() if profiler is not None else 0.0
        final_depth_gate = final_depth_fusion_acceptance_gate(
            rgb_reference_optimization.report,
            depth_candidate_optimization.report,
            depth_fusion_config,
        )
        depth_fusion.report["final_acceptance_gate"] = final_depth_gate
        depth_fusion.report["final_output_used"] = bool(final_depth_gate["passed"])
        if not final_depth_gate["passed"]:
            global_optimization = rgb_reference_optimization
        if profiler is not None:
            profiler.record(
                "final_depth_acceptance_gate",
                time.perf_counter() - depth_gate_started,
                parent="global_optimizers_depth_gate",
            )
    else:
        depth_fusion.report["final_acceptance_gate"] = {
            "passed": False,
            "fallback_to_rgb_reference": False,
            "reason": "depth_fusion_not_applied",
        }
        depth_fusion.report["final_output_used"] = False
    if profiler is not None:
        profiler.record(
            "global_optimizers_depth_gate",
            time.perf_counter() - optimizers_started,
        )
    arrays["keypoints_3d_world"] = global_optimization.keypoints_3d
    body_count = min(17, arrays["keypoints_3d_world"].shape[1])
    accepted[:, :body_count] = global_optimization.valid_mask[:, :body_count]
    optimized_body_error = global_optimization.body_reprojection_error_px
    arrays["reprojection_error"][:, :body_count] = np.where(
        np.isfinite(optimized_body_error),
        optimized_body_error,
        arrays["reprojection_error"][:, :body_count],
    )
    reliability_started = time.perf_counter() if profiler is not None else 0.0
    reliability_config = model_config.get("reliability", {})
    reliability = filter_unreliable_pose(
        arrays["keypoints_3d_world"],
        accepted,
        sampled_timestamps,
        confidence=arrays["triangulation_score"],
        max_bone_relative_deviation=float(reliability_config.get("max_bone_relative_deviation", 0.25)),
        max_bone_absolute_deviation_m=float(reliability_config.get("max_bone_absolute_deviation_m", 0.08)),
        min_temporal_residual_m=float(reliability_config.get("min_temporal_residual_m", 0.08)),
        max_temporal_acceleration_mps2=float(reliability_config.get("max_temporal_acceleration_mps2", 70.0)),
        minimum_bone_samples=int(reliability_config.get("minimum_bone_samples", 5)),
    )
    print("[5/5] Saving stabilized 2D and 3D outputs", flush=True)
    reliable_unsmoothed_3d = np.where(
        reliability.valid_mask[..., None],
        reliability.keypoints_3d,
        np.nan,
    )
    arrays["keypoints_3d_world"] = smooth_pose_sequence(
        reliable_unsmoothed_3d,
        method=smoothing_method,
        window_size=smoothing_window,
        valid_mask=reliability.valid_mask,
        polynomial_order=smoothing_polynomial_order,
        min_outlier_distance_m=smoothing_min_outlier_distance_m,
    )
    if global_optimization.applied:
        arrays["keypoints_3d_world"][:, :body_count] = reliable_unsmoothed_3d[:, :body_count]
    arrays["keypoints_3d_world"] = np.where(reliability.valid_mask[..., None], arrays["keypoints_3d_world"], np.nan)
    pose3d_stability_report = pose3d_stability_metrics(
        reliable_unsmoothed_3d,
        arrays["keypoints_3d_world"],
    )
    if profiler is not None:
        profiler.record(
            "reliability_smoothing_quality",
            time.perf_counter() - reliability_started,
        )
    serialization_started = time.perf_counter() if profiler is not None else 0.0
    export_session_json(
        reliability.summary,
        output_paths["json"] / "pose_reliability_report.json",
    )
    export_session_json(
        global_optimization.report,
        output_paths["json"] / "global_pose_optimization_report.json",
    )
    if rgb_reference_optimization is not None:
        export_session_json(
            rgb_reference_optimization.report,
            output_paths["json"] / "rgb_reference_global_pose_optimization_report.json",
        )
    export_session_json(
        depth_fusion.report,
        output_paths["json"] / "zed_depth_fusion_report.json",
    )
    video_arrays = _repeat_arrays_for_video(arrays, output_repeats)
    export_keypoints3d_csv(
        arrays["keypoints_3d_world"],
        output_paths["csv"] / "vitpose_keypoints_3d_world_flat.csv",
        frame_indices=arrays["frame_idx"],
        timestamps_sec=sampled_timestamps,
    )
    export_keypoints3d_csv(
        reliable_unsmoothed_3d,
        output_paths["csv"] / "vitpose_keypoints_3d_world_unsmoothed_flat.csv",
        frame_indices=arrays["frame_idx"],
        timestamps_sec=sampled_timestamps,
    )
    export_keypoints3d_csv(
        triangulated_3d_analysis,
        output_paths["csv"] / "vitpose_keypoints_3d_world_triangulated_flat.csv",
        frame_indices=arrays["frame_idx"],
        timestamps_sec=sampled_timestamps,
    )
    export_keypoints3d_csv(
        depth_fused_3d_analysis,
        output_paths["csv"] / "vitpose_keypoints_3d_world_depth_fused_flat.csv",
        frame_indices=arrays["frame_idx"],
        timestamps_sec=sampled_timestamps,
    )
    export_keypoints3d_csv(
        global_optimization.keypoints_3d,
        output_paths["csv"] / "vitpose_keypoints_3d_world_global_optimized_flat.csv",
        frame_indices=arrays["frame_idx"],
        timestamps_sec=sampled_timestamps,
    )
    export_pose3d_provenance_csv(
        global_optimization.provenance,
        output_paths["csv"] / "vitpose_keypoints_3d_provenance.csv",
        frame_indices=arrays["frame_idx"],
        timestamps_sec=sampled_timestamps,
    )
    export_depth_fusion_observations_csv(
        depth_fusion.observation_rows,
        output_paths["csv"] / "zed_depth_fusion_observations.csv",
    )
    export_keypoints2d_csv(
        display_poses_2d_by_frame,
        output_paths["csv"] / "vitpose_keypoints_2d_flat.csv",
    )
    export_keypoints2d_csv(
        poses_2d_by_frame,
        output_paths["csv"] / "vitpose_keypoints_2d_geometry_flat.csv",
    )
    export_keypoints2d_csv(
        prefeedback_poses_2d_by_frame,
        output_paths["csv"] / "vitpose_keypoints_2d_prefeedback_flat.csv",
    )
    export_keypoints2d_csv(
        raw_poses_2d_by_frame,
        output_paths["csv"] / "vitpose_keypoints_2d_raw_flat.csv",
    )
    export_pose2d_feedback_provenance_csv(
        feedback_provenance_by_camera,
        output_paths["csv"] / "vitpose_keypoints_2d_feedback_provenance.csv",
        frame_indices=arrays["frame_idx"],
    )
    export_session_json(
        crossview_feedback_report,
        output_paths["json"] / "crossview_2d_feedback_report.json",
    )
    export_session_json(
        {
            "algorithm": crossview_feedback_report["algorithm"],
            "cameras": crossview_feedback_report["cameras"],
        },
        output_paths["json"] / "camera_health_report.json",
    )
    export_session_json(
        {
            "algorithm": "zero_phase_robust_savgol",
            "enabled": pose2d_stabilization_config.enabled,
            "window_size": pose2d_stabilization_config.window_size,
            "polynomial_order": pose2d_stabilization_config.polynomial_order,
            "min_outlier_distance_px": (pose2d_stabilization_config.min_outlier_distance_px),
            "cameras": pose2d_stability_report,
        },
        output_paths["json"] / "pose2d_stability_report.json",
    )
    export_session_json(
        {
            "algorithm": smoothing_method,
            "window_size": smoothing_window,
            "polynomial_order": smoothing_polynomial_order,
            "min_outlier_distance_m": smoothing_min_outlier_distance_m,
            **pose3d_stability_report,
        },
        output_paths["json"] / "pose3d_stability_report.json",
    )
    main_3d_payload = {
            "schema_version": MAIN_3D_SCHEMA_VERSION,
            "session_id": session.session_id,
            "run_id": run_id,
            "provenance": {
                "run_manifest": "run_manifest.json",
                "calibration_snapshot": str(calibration_snapshot_path.resolve()),
                "calibration_sha256": calibration_provenance["sha256"],
                "model_config_sha256": config_snapshots["model_config"]["sha256"],
            },
            "source": (
                "vitpose_multiview_zed_depth_auxiliary"
                if depth_fusion.report.get("applied")
                else "vitpose_multiview"
            ),
            "calibration_mode": calibration_mode,
            "production_ready_calibration": production_ready_calibration,
            "external_accuracy": external_accuracy_not_evaluated(),
            "coordinate_system": ANALYSIS_COORDINATE_SYSTEM,
            "frame_indices": arrays["frame_idx"],
            "timestamps_sec": sampled_timestamps,
            "sample_fps": fps / max(args.stride, 1),
            "smoothing_applied": smoothing_window > 1,
            "smoothing_method": smoothing_method,
            "smoothing_window": smoothing_window,
            "smoothing_polynomial_order": smoothing_polynomial_order,
            "smoothing_min_outlier_distance_m": (smoothing_min_outlier_distance_m),
            "pose2d_offline_stabilization": {
                "enabled": pose2d_stabilization_config.enabled,
                "window_size": pose2d_stabilization_config.window_size,
                "polynomial_order": pose2d_stabilization_config.polynomial_order,
                "min_outlier_distance_px": (pose2d_stabilization_config.min_outlier_distance_px),
            },
            "crossview_2d_feedback": crossview_feedback_report,
            "global_body17_optimization": global_optimization.report,
            "zed_depth_fusion": depth_fusion.report,
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
            "optimization_provenance": global_optimization.provenance,
            "reliability_summary": reliability.summary,
            "pose3d_stability": pose3d_stability_report,
        }
    main_3d_path = output_paths["json"] / "vitpose_session_3d.json"
    export_session_json(main_3d_payload, main_3d_path)
    load_main_3d_artifact(main_3d_path)
    body_valid = np.all(np.isfinite(arrays["keypoints_3d_world"][:, :body_count]), axis=-1)
    mean_body_valid_ratio = float(np.mean(body_valid)) if body_valid.size else 0.0
    body_provenance = global_optimization.provenance[:, :body_count]
    observed_body_ratio = float(np.mean(body_provenance == PROVENANCE_OBSERVED)) if body_provenance.size else 0.0
    recovered_body_ratio = (
        float(np.mean(body_provenance == PROVENANCE_TEMPORALLY_RECOVERED))
        if body_provenance.size
        else 0.0
    )
    finite_errors = arrays["reprojection_error"][np.isfinite(arrays["reprojection_error"])]
    mean_reprojection_error = float(np.mean(finite_errors)) if finite_errors.size else None
    minimum_body_valid_ratio = float(reliability_config.get("min_output_valid_body_ratio", 0.90))
    quality_passed = bool(
        production_ready_calibration
        and mean_body_valid_ratio >= minimum_body_valid_ratio
        and mean_reprojection_error is not None
        and mean_reprojection_error <= max_error
    )
    internal_sensor_consistency = internal_sensor_consistency_status(
        internal_geometry_passed=quality_passed,
        zed_depth_fusion_report=depth_fusion.report,
    )
    provisional_scoring_ready = internal_sensor_consistency["status"] == "passed"
    quality_payload = {
            "schema_version": RUN_QUALITY_SCHEMA_VERSION,
            "session_id": session.session_id,
            "run_id": run_id,
            "provenance": main_3d_payload["provenance"],
            "status": "passed" if quality_passed else "failed",
            "quality_scope": "internal_geometry_only",
            "ground_truth_accuracy_evaluated": False,
            "external_accuracy": external_accuracy_not_evaluated(),
            "internal_sensor_consistency": internal_sensor_consistency,
            "provisional_scoring_ready": provisional_scoring_ready,
            "provisional_scoring_mode": "internal_quality_gated_not_official",
            "external_accuracy_required_for_provisional_scoring": False,
            "scoring_ready": False,
            "official_scoring_ready": False,
            "scoring_readiness_reason": (
                "Internal provisional scoring may proceed when provisional_scoring_ready is true. "
                "Independent external 3D accuracy was not evaluated and no historical benchmark "
                "was inherited; official scoring remains unavailable."
            ),
            "production_ready_calibration": production_ready_calibration,
            "mean_body17_valid_ratio": mean_body_valid_ratio,
            "observed_body17_ratio": observed_body_ratio,
            "temporally_recovered_body17_ratio": recovered_body_ratio,
            "mean_reprojection_error_px": mean_reprojection_error,
            "max_reprojection_error_px": max_error,
            "reliability_filter": reliability.summary,
            "global_body17_optimization": {
                "applied": global_optimization.applied,
                "fallback_used": global_optimization.report.get("fallback_used", True),
                "fallback_reason": global_optimization.report.get("fallback_reason"),
                "camera_weights": global_optimization.camera_weights,
                "acceptance_gate": global_optimization.report.get("acceptance_gate"),
            },
            "crossview_2d_feedback": {
                "target_count": crossview_feedback_report.get("target_count", 0),
                "image_guided_accepted_count": crossview_feedback_report.get(
                    "image_guided_accepted_count",
                    0,
                ),
                "geometry_outlier_rejected_count": crossview_feedback_report.get(
                    "geometry_outlier_rejected_count",
                    0,
                ),
                "crossview_projected_visualization_count": crossview_feedback_report.get(
                    "crossview_projected_visualization_count",
                    0,
                ),
                "geometry_feedback_applied": crossview_feedback_report.get(
                    "geometry_feedback_applied",
                    False,
                ),
                "geometry_feedback_gate": crossview_feedback_report.get(
                    "geometry_feedback_gate"
                ),
            },
            "zed_depth_fusion": depth_fusion.report,
            "minimum_required_body17_valid_ratio": minimum_body_valid_ratio,
        }
    quality_path = output_paths["json"] / "run_quality_report.json"
    export_session_json(quality_payload, quality_path)
    validate_run_quality_artifact(json.loads(quality_path.read_text(encoding="utf-8")))
    write_3d_skeleton_video(
        video_arrays["keypoints_3d_world"],
        output_paths["videos"] / "vitpose_skeleton_3d_world.mp4",
        fps=max(float(args.output_fps or fps), 1.0),
    )
    viewer_path = write_pose3d_html_viewer(
        arrays["keypoints_3d_world"],
        output_paths["root"] / "viewer" / "pose3d_viewer.html",
        fps=fps / max(args.stride, 1),
        title=run_id,
    )
    depth_sources = load_depth_sources(session_path)
    input_paths = [session_path, *(camera.video_path for camera in cameras)]
    input_paths.extend(source.svo_path for source in depth_sources)
    input_paths.extend(source.timestamp_mapping_report for source in depth_sources)
    manifest = build_run_manifest(
        workspace_root=ROOT,
        session_id=session.session_id,
        run_id=run_id,
        started_at=started_at,
        completed_at=utc_now(),
        calibration=calibration_provenance,
        configs=config_snapshots,
        models=model_records,
        inputs=input_paths,
        argv=options.invocation_argv or ("tk3d-multiview",),
    )
    manifest_path = write_run_manifest(
        manifest,
        output_paths["json"] / "run_manifest.json",
    )
    serialized_main_3d, _ = load_main_3d_artifact(main_3d_path)
    validate_artifact_manifest_binding(serialized_main_3d, manifest)
    serialized_quality = json.loads(quality_path.read_text(encoding="utf-8"))
    validate_artifact_manifest_binding(serialized_quality, manifest)
    if profiler is not None:
        profiler.record(
            "artifact_serialization",
            time.perf_counter() - serialization_started,
        )
        core_stages = (
            "rgb_decode_synchronization",
            "rfdetr_bytetrack",
            "vitpose_causal_2d",
            "offline_2d_stabilization",
            "triangulation",
            "crossview_guided_second_pass",
            "zed_depth_fusion",
            "global_optimizers_depth_gate",
            "reliability_smoothing_quality",
        )
        steady_stages = (
            "rgb_decode_synchronization",
            "rfdetr_bytetrack",
            "vitpose_causal_2d",
            "triangulation",
        )
        core_seconds = profiler.total_wall(*core_stages)
        steady_seconds = profiler.total_wall(*steady_stages, window_only=True)
        steady_frame_count = profiler.call_count(
            "rgb_decode_synchronization",
            window_only=True,
        )
        report = profiler.build_report(
            workflow="tk3d_current_active_multiview_performance_v1",
            run_identity={"session_id": session.session_id, "run_id": run_id},
            input_identity=manifest["inputs"],
            environment_identity=manifest["environment"],
            config_identity=manifest["configs"],
            model_identity=manifest["models"],
            calibration_identity=manifest["calibration"],
            processed_frame_count=int(arrays["keypoints_3d_world"].shape[0]),
            runtime_summary={
                "startup_model_load_seconds": profiler.total_wall(
                    "preflight_provenance",
                    "vitpose_initialization",
                    "rfdetr_initialization",
                ),
                "core_processing_seconds": core_seconds,
                "core_effective_fps": (
                    None if core_seconds <= 0.0 else int(arrays["frame_idx"].shape[0]) / core_seconds
                ),
                "export_render_seconds": profiler.total_wall(
                    "overlay_rendering",
                    "artifact_serialization",
                ),
                "steady_state_frame_count": steady_frame_count,
                "steady_state_inference_geometry_seconds": steady_seconds,
                "steady_state_inference_geometry_fps": (
                    None if steady_seconds <= 0.0 else steady_frame_count / steady_seconds
                ),
                "first_detector_call_seconds": profiler.first_wall("rfdetr_bytetrack"),
                "first_vitpose_call_seconds": profiler.first_wall("vitpose_causal_2d"),
                "steady_state_detector_calls": profiler.call_count(
                    "rfdetr_bytetrack",
                    window_only=True,
                ),
                "steady_state_detector_total_seconds": profiler.total_wall(
                    "rfdetr_bytetrack",
                    window_only=True,
                ),
                "steady_state_vitpose_calls": profiler.call_count(
                    "vitpose_causal_2d",
                    window_only=True,
                ),
                "steady_state_vitpose_total_seconds": profiler.total_wall(
                    "vitpose_causal_2d",
                    window_only=True,
                ),
                "crossview": {
                    "enabled_in_config": crossview_feedback_config.enabled,
                    "target_count": int(crossview_feedback_report.get("target_count", 0)),
                    "runtime_work": int(crossview_feedback_report.get("target_count", 0)) > 0,
                },
                "wholebody_keypoint_count": int(arrays["keypoints_3d_world"].shape[1]),
                "body17_depth_fusion_applied": bool(depth_fusion.report.get("applied")),
                "body17_optimizer_applied": bool(global_optimization.applied),
                "optimizer": {
                    "solver_function_evaluations": global_optimization.report.get(
                        "solver_function_evaluations"
                    ),
                    "outer_iterations_completed": global_optimization.report.get(
                        "outer_iterations_completed"
                    ),
                    "solver_success": global_optimization.report.get("solver_success"),
                    "fallback_used": global_optimization.report.get("fallback_used"),
                    "fallback_reason": global_optimization.report.get("fallback_reason"),
                },
            },
            limitations=[
                "RF-DETR predict is an opaque synchronized wall-clock boundary; no internal GPU event is claimed.",
                "Steady-state FPS covers per-frame RGB decode, detector/tracker, ViTPose and triangulation; "
                "sequence-level depth, optimization and exports are reported separately.",
                "ZED SDK allocations are not included in PyTorch peak VRAM counters.",
            ],
            profiler_baseline_seconds=options.profiler_baseline_seconds,
        )
        write_performance_report(
            output_paths["json"] / "performance_report.json",
            report,
        )
    if quality_passed:
        if options.promote_latest:
            mark_run_complete(output_root, session.session_id, run_id, output_paths["root"])
        else:
            mark_run_completed(output_paths["root"], session.session_id, run_id)

    print(f"saved: {output_paths['videos'] / 'vitpose_skeleton_3d_world.mp4'}")
    print(f"saved: {viewer_path}")
    print(f"saved: {manifest_path}")
    print(f"keypoints_3d_world shape: {video_arrays['keypoints_3d_world'].shape}")
    print(f"inference_sample_count: {arrays['keypoints_3d_world'].shape[0]}")
    print(f"calibration_mode: {calibration_mode}")
    print(
        "crossview_2d_feedback: "
        f"{crossview_feedback_report.get('image_guided_accepted_count', 0)} image-guided, "
        f"{crossview_feedback_report.get('crossview_projected_visualization_count', 0)} "
        "visualization-only"
    )
    print(f"global_body17_optimization: {'applied' if global_optimization.applied else 'fallback'}")
    print(
        "zed_depth_fusion: "
        f"{depth_fusion.report.get('status')} "
        f"({depth_fusion.report.get('fused_body_point_count', 0)} body points, "
        f"final used={depth_fusion.report.get('final_output_used', False)})"
    )
    if not global_optimization.applied:
        print(f"global_body17_fallback_reason: {global_optimization.report.get('fallback_reason')}")
    print(f"run_id: {run_id}")
    print(f"internal_geometry_quality_status: {'passed' if quality_passed else 'failed'}")
    print("ground_truth_accuracy_status: not_evaluated_in_this_command")
    print(f"provisional_scoring_ready: {str(provisional_scoring_ready).lower()}")
    print("official_scoring_ready: false")
    result = MultiviewRunResult(
        session_id=session.session_id,
        run_id=run_id,
        run_root=output_paths["root"],
        main_3d_path=main_3d_path,
        quality_path=quality_path,
        manifest_path=manifest_path,
        quality_passed=quality_passed,
    )
    if not quality_passed and not args.allow_low_quality_output:
        raise MultiviewQualityError(
            (
                "3D output failed production quality gates. Diagnostic files were kept, "
                "but this run was not promoted as latest. Inspect run_quality_report.json."
            ),
            result,
        )
    return result


def _run_crossview_2d_feedback(
    estimator: ViTPose2DEstimator,
    cameras: list,
    sampled_sync_frames: list[SynchronizedFrame],
    sampled_global_frame_indices: list[int],
    baseline_by_camera: dict[str, list[PersonPose2D]],
    plan: CrossViewFeedbackPlan,
    config: CrossView2DFeedbackConfig,
    stabilization_config: Pose2DStabilizationConfig,
    progress_every: int,
) -> tuple[
    dict[str, list[PersonPose2D]],
    dict[str, list[PersonPose2D]],
    dict[str, np.ndarray],
    dict,
]:
    geometry = {
        camera.camera_id: [copy_pose(pose) for pose in baseline_by_camera[camera.camera_id]]
        for camera in cameras
    }
    provenance = {
        camera.camera_id: np.zeros(
            (len(sampled_global_frame_indices), COCO_WHOLEBODY_KEYPOINTS),
            dtype=np.uint8,
        )
        for camera in cameras
    }
    decisions: list[dict] = []
    target_frame_count = sum(
        int(np.count_nonzero(np.any(plan.target_mask[camera.camera_id], axis=1)))
        for camera in cameras
    )
    if not config.enabled or target_frame_count == 0:
        display = {
            camera_id: [copy_pose(pose) for pose in poses]
            for camera_id, poses in geometry.items()
        }
        return (
            geometry,
            display,
            provenance,
            finalize_feedback_report(plan, provenance, decisions),
        )

    print(
        f"      cross-view 2D feedback: {plan.target_count} joint targets "
        f"across {target_frame_count} camera-frames",
        flush=True,
    )
    progress = ProgressBar("2D feedback", target_frame_count)
    completed = 0
    for camera in cameras:
        camera_id = camera.camera_id
        target_frames = np.flatnonzero(np.any(plan.target_mask[camera_id], axis=1))
        if target_frames.size == 0:
            continue
        capture = cv2.VideoCapture(str(camera.video_path))
        if not capture.isOpened():
            for sample_idx in target_frames:
                _record_unavailable_feedback_targets(
                    decisions,
                    plan,
                    camera_id,
                    int(sample_idx),
                    sampled_global_frame_indices,
                    sampled_sync_frames,
                    "video_reopen_failed",
                    config.projected_fallback_for_visualization,
                )
            continue
        next_frame = {camera_id: 0}
        try:
            for sample_idx_value in target_frames:
                sample_idx = int(sample_idx_value)
                local_frame_idx = int(
                    sampled_sync_frames[sample_idx].local_frame_indices[camera_id]
                )
                frame = _read_frame_sequential(
                    capture,
                    camera_id,
                    local_frame_idx,
                    next_frame,
                )
                if frame is None:
                    _record_unavailable_feedback_targets(
                        decisions,
                        plan,
                        camera_id,
                        sample_idx,
                        sampled_global_frame_indices,
                        sampled_sync_frames,
                        "frame_read_failed",
                        config.projected_fallback_for_visualization,
                    )
                    completed += 1
                    continue
                prior_points = np.full(
                    (COCO_WHOLEBODY_KEYPOINTS, 2),
                    np.nan,
                    dtype=float,
                )
                prior_valid = np.zeros(COCO_WHOLEBODY_KEYPOINTS, dtype=bool)
                body_count = plan.priors_xy[camera_id].shape[1]
                prior_points[:body_count] = plan.priors_xy[camera_id][sample_idx]
                prior_valid[:body_count] = plan.target_mask[camera_id][sample_idx]
                baseline_pose = baseline_by_camera[camera_id][sample_idx]
                guided = estimator.predict_guided(
                    frame,
                    camera_id,
                    local_frame_idx,
                    baseline_pose,
                    prior_points,
                    prior_valid,
                    config.search_radius_px,
                )
                for joint_idx_value in np.flatnonzero(prior_valid):
                    joint_idx = int(joint_idx_value)
                    decision = decide_guided_candidate(
                        baseline_pose.keypoints_xy[joint_idx],
                        prior_points[joint_idx],
                        guided.guided_pose.keypoints_xy[joint_idx],
                        float(guided.guided_pose.scores[joint_idx]),
                        float(guided.unconstrained_pose.scores[joint_idx]),
                        config,
                    )
                    item = {
                        "sample_index": sample_idx,
                        "global_frame_idx": int(sampled_global_frame_indices[sample_idx]),
                        "local_frame_idx": local_frame_idx,
                        "camera_id": camera_id,
                        "joint_idx": joint_idx,
                        "joint_name": (
                            COCO_BODY_JOINT_NAMES[joint_idx]
                            if joint_idx < len(COCO_BODY_JOINT_NAMES)
                            else f"joint_{joint_idx}"
                        ),
                        "accepted": decision.accepted,
                        "reason": decision.reason,
                        "initial_error_px": decision.initial_error_px,
                        "candidate_error_px": decision.candidate_error_px,
                        "improvement_px": decision.improvement_px,
                        "guided_score": decision.guided_score,
                        "unconstrained_score": decision.unconstrained_score,
                        "supporting_views": int(
                            plan.supporting_views[camera_id][sample_idx, joint_idx]
                        ),
                        "visualization_fallback": bool(
                            not decision.accepted
                            and config.projected_fallback_for_visualization
                        ),
                    }
                    decisions.append(item)
                    if not decision.accepted:
                        continue
                    corrected = geometry[camera_id][sample_idx]
                    corrected.keypoints_xy[joint_idx] = (
                        guided.guided_pose.keypoints_xy[joint_idx]
                    )
                    corrected.scores[joint_idx] = guided.guided_pose.scores[joint_idx]
                    corrected.valid_mask[joint_idx] = True
                    provenance[camera_id][sample_idx, joint_idx] = (
                        PROVENANCE_IMAGE_GUIDED
                    )
                completed += 1
                if (
                    completed == 1
                    or completed == target_frame_count
                    or completed % max(progress_every, 1) == 0
                ):
                    progress.print(completed, extra=f"{camera_id} frame {local_frame_idx}")
        finally:
            capture.release()
    progress.done()

    for camera in cameras:
        camera_id = camera.camera_id
        body_count = plan.target_mask[camera_id].shape[1]
        rejected = (
            plan.target_mask[camera_id]
            & (provenance[camera_id][:, :body_count] != PROVENANCE_IMAGE_GUIDED)
        )
        for sample_idx, joint_idx in zip(*np.nonzero(rejected), strict=True):
            geometry[camera_id][sample_idx].valid_mask[joint_idx] = False

    if plan.target_count:
        geometry = {
            camera.camera_id: stabilize_pose2d_sequence(
                geometry[camera.camera_id],
                config=stabilization_config,
            )
            for camera in cameras
        }
    display = {
        camera_id: [copy_pose(pose) for pose in poses]
        for camera_id, poses in geometry.items()
    }
    if config.projected_fallback_for_visualization:
        for camera in cameras:
            camera_id = camera.camera_id
            fallback = (
                plan.target_mask[camera_id]
                & (provenance[camera_id][:, : plan.target_mask[camera_id].shape[1]] == 0)
            )
            for sample_idx, joint_idx in zip(*np.nonzero(fallback), strict=True):
                prior = plan.priors_xy[camera_id][sample_idx, joint_idx]
                if not np.all(np.isfinite(prior)):
                    continue
                pose = display[camera_id][sample_idx]
                pose.keypoints_xy[joint_idx] = prior
                pose.scores[joint_idx] = float(
                    np.clip(plan.prior_score[camera_id][sample_idx, joint_idx], 0.0, 1.0)
                )
                pose.valid_mask[joint_idx] = True
                provenance[camera_id][sample_idx, joint_idx] = (
                    PROVENANCE_CROSSVIEW_PROJECTED
                )
    else:
        for camera in cameras:
            camera_id = camera.camera_id
            body_count = plan.target_mask[camera_id].shape[1]
            rejected = (
                plan.target_mask[camera_id]
                & (provenance[camera_id][:, :body_count] != PROVENANCE_IMAGE_GUIDED)
            )
            for sample_idx, joint_idx in zip(*np.nonzero(rejected), strict=True):
                original = baseline_by_camera[camera_id][sample_idx]
                pose = display[camera_id][sample_idx]
                pose.keypoints_xy[joint_idx] = original.keypoints_xy[joint_idx]
                pose.scores[joint_idx] = original.scores[joint_idx]
                pose.valid_mask[joint_idx] = original.valid_mask[joint_idx]
    return (
        geometry,
        display,
        provenance,
        finalize_feedback_report(plan, provenance, decisions),
    )


def _record_unavailable_feedback_targets(
    decisions: list[dict],
    plan: CrossViewFeedbackPlan,
    camera_id: str,
    sample_idx: int,
    sampled_global_frame_indices: list[int],
    sampled_sync_frames: list[SynchronizedFrame],
    reason: str,
    visualization_fallback: bool,
) -> None:
    local_frame_idx = int(
        sampled_sync_frames[sample_idx].local_frame_indices[camera_id]
    )
    for joint_idx_value in np.flatnonzero(plan.target_mask[camera_id][sample_idx]):
        joint_idx = int(joint_idx_value)
        decisions.append(
            {
                "sample_index": sample_idx,
                "global_frame_idx": int(sampled_global_frame_indices[sample_idx]),
                "local_frame_idx": local_frame_idx,
                "camera_id": camera_id,
                "joint_idx": joint_idx,
                "joint_name": (
                    COCO_BODY_JOINT_NAMES[joint_idx]
                    if joint_idx < len(COCO_BODY_JOINT_NAMES)
                    else f"joint_{joint_idx}"
                ),
                "accepted": False,
                "reason": reason,
                "initial_error_px": float(
                    plan.initial_error_px[camera_id][sample_idx, joint_idx]
                ),
                "candidate_error_px": None,
                "improvement_px": None,
                "guided_score": 0.0,
                "unconstrained_score": 0.0,
                "supporting_views": int(
                    plan.supporting_views[camera_id][sample_idx, joint_idx]
                ),
                "visualization_fallback": visualization_fallback,
            }
        )


def _poses_by_global_frame(
    global_frame_indices: list[int],
    cameras: list,
    poses_by_camera: dict[str, list[PersonPose2D]],
) -> dict[int, dict[str, PersonPose2D]]:
    return {
        int(global_frame_idx): {
            camera.camera_id: poses_by_camera[camera.camera_id][sample_idx]
            for camera in cameras
        }
        for sample_idx, global_frame_idx in enumerate(global_frame_indices)
    }


def _triangulate_sampled_poses(
    global_frame_indices: list[int],
    poses_by_frame: dict[int, dict[str, PersonPose2D]],
    calibrations: dict[str, CameraCalibration],
    min_views: int,
    min_keypoint_score: float,
    max_reprojection_error_px: float,
    max_hypotheses: int,
) -> list:
    return [
        triangulate_frame(
            frame_idx=int(frame_idx),
            poses_by_camera=poses_by_frame[int(frame_idx)],
            calibrations=calibrations,
            min_views=min_views,
            min_keypoint_score=min_keypoint_score,
            max_reprojection_error_px=max_reprojection_error_px,
            max_hypotheses=max_hypotheses,
        )
        for frame_idx in global_frame_indices
    ]


def _feedback_triangulation_gate(before: list, after: list) -> dict:
    before_arrays = stack_triangulated(before)
    after_arrays = stack_triangulated(after)
    body_count = min(17, before_arrays["keypoints_3d_world"].shape[1])
    before_valid = np.all(
        np.isfinite(before_arrays["keypoints_3d_world"][:, :body_count]),
        axis=-1,
    )
    after_valid = np.all(
        np.isfinite(after_arrays["keypoints_3d_world"][:, :body_count]),
        axis=-1,
    )
    before_errors = before_arrays["reprojection_error"][:, :body_count]
    after_errors = after_arrays["reprojection_error"][:, :body_count]
    before_errors = before_errors[np.isfinite(before_errors)]
    after_errors = after_errors[np.isfinite(after_errors)]
    before_median = float(np.median(before_errors)) if before_errors.size else None
    after_median = float(np.median(after_errors)) if after_errors.size else None
    before_p95 = float(np.percentile(before_errors, 95.0)) if before_errors.size else None
    after_p95 = float(np.percentile(after_errors, 95.0)) if after_errors.size else None
    before_ratio = float(np.mean(before_valid)) if before_valid.size else 0.0
    after_ratio = float(np.mean(after_valid)) if after_valid.size else 0.0
    checks = {
        "body_valid_ratio_not_degraded": after_ratio + 0.01 >= before_ratio,
        "median_reprojection_not_degraded": (
            before_median is None
            or after_median is None
            or after_median <= max(before_median * 1.05, before_median + 1.0)
        ),
        "p95_reprojection_not_degraded": (
            before_p95 is None
            or after_p95 is None
            or after_p95 <= max(before_p95 * 1.10, before_p95 + 2.0)
        ),
    }
    passed = all(checks.values())
    return {
        "passed": passed,
        "reason": None if passed else "triangulation_quality_degraded",
        "checks": checks,
        "before": {
            "body_valid_ratio": before_ratio,
            "median_reprojection_error_px": before_median,
            "p95_reprojection_error_px": before_p95,
        },
        "after": {
            "body_valid_ratio": after_ratio,
            "median_reprojection_error_px": after_median,
            "p95_reprojection_error_px": after_p95,
        },
    }


def build_pair_test_calibrations(camera_a: str, camera_b: str) -> dict[str, CameraCalibration]:
    intrinsic = np.array([[1200.0, 0.0, 960.0], [0.0, 1200.0, 540.0], [0.0, 0.0, 1.0]], dtype=float)
    return {
        camera_a: _calibration(camera_a, intrinsic, np.eye(3), np.array([0.0, 0.0, 0.0])),
        camera_b: _calibration(camera_b, intrinsic, _rotation_y(np.deg2rad(18.0)), np.array([0.75, 0.0, 0.04])),
    }


def _calibration(
    camera_id: str, intrinsic: np.ndarray, rotation: np.ndarray, translation: np.ndarray
) -> CameraCalibration:
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
    sampled_provenance: np.ndarray | None,
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
            provenance = _provenance_at_frame(
                sampled_poses,
                sampled_provenance,
                local_idx,
            )
            writer.write(draw_pose2d(frame, pose, provenance=provenance))
            if output_idx == 0 or output_idx + 1 >= len(synchronized_frames) or (output_idx + 1) % progress_every == 0:
                progress.print(output_idx + 1, extra=f"src frame {local_idx}")
    finally:
        capture.release()
        writer.release()
        progress.done()


def _provenance_at_frame(
    sampled_poses: list[PersonPose2D],
    sampled_provenance: np.ndarray | None,
    local_frame_idx: int,
) -> np.ndarray | None:
    if sampled_provenance is None or not sampled_poses:
        return None
    values = np.asarray(sampled_provenance, dtype=np.uint8)
    if values.ndim != 2 or values.shape[0] != len(sampled_poses):
        raise ValueError("sampled_provenance must match sampled_poses")
    frame_indices = np.asarray([pose.frame_idx for pose in sampled_poses], dtype=int)
    insertion = int(np.searchsorted(frame_indices, int(local_frame_idx), side="left"))
    if insertion <= 0:
        selected = 0
    elif insertion >= frame_indices.size:
        selected = frame_indices.size - 1
    else:
        before = insertion - 1
        selected = (
            before
            if abs(local_frame_idx - frame_indices[before])
            <= abs(frame_indices[insertion] - local_frame_idx)
            else insertion
        )
    return values[selected]


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


def _effective_timeline_fps(
    declared_fps: float | None,
    camera_fps_values: Iterable[float],
) -> float:
    """Choose a common FPS that never duplicates frames from the slowest camera."""
    actual_rates = np.asarray(list(camera_fps_values), dtype=float)
    if actual_rates.size == 0 or np.any(~np.isfinite(actual_rates)) or np.any(actual_rates <= 0.0):
        raise ValueError("Every camera FPS must be finite and positive")
    slowest_camera_fps = float(np.min(actual_rates))
    if declared_fps is None:
        return slowest_camera_fps
    requested = float(declared_fps)
    if not np.isfinite(requested) or requested <= 0.0:
        raise ValueError("Session FPS must be finite and positive")
    return min(requested, slowest_camera_fps)


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
