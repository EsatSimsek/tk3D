from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from .coordinate_system import transform_points
from .data_structures import CameraCalibration, PersonPose2D
from .performance import performance_profiling_active, profile_stage, record_profile_stage


@dataclass(frozen=True, slots=True)
class ZedDepthFusionConfig:
    enabled: bool = True
    depth_mode: str = "neural"
    confidence_threshold: float = 50.0
    min_pose_score: float = 0.30
    patch_radius_px: int = 4
    min_patch_samples: int = 4
    min_depth_m: float = 0.40
    max_depth_m: float = 10.0
    surface_gate_m: float = 0.35
    surface_gate_ratio: float = 0.10
    minimum_offset_samples: int = 24
    max_centered_residual_m: float = 0.20
    baseline_weight: float = 1.0
    depth_weight: float = 0.35
    max_correction_m: float = 0.06
    max_median_reprojection_increase_px: float = 0.75
    max_camera_reprojection_increase_px: float = 2.0
    min_depth_residual_improvement_m: float = 0.001
    max_final_median_reprojection_ratio: float = 1.02
    max_final_median_reprojection_increase_px: float = 0.25
    max_final_p95_reprojection_ratio: float = 1.03
    max_final_p95_reprojection_increase_px: float = 0.50
    max_final_p95_acceleration_ratio: float = 1.05
    max_final_p95_acceleration_increase_mps2: float = 1.0
    max_final_bone_cv_increase_percent: float = 0.10


@dataclass(frozen=True, slots=True)
class DepthSource:
    camera_id: str
    svo_path: Path
    timestamp_mapping_report: Path
    prepared_frame_offset: int
    imu_gravity_rotation_applied: bool


@dataclass(frozen=True, slots=True)
class DepthPatchSample:
    depth_m: float
    confidence: float
    pixel_count: int


@dataclass(slots=True)
class ZedDepthFusionResult:
    keypoints_3d_analysis: np.ndarray
    report: dict[str, Any]
    observation_rows: list[dict[str, Any]]


def depth_fusion_config_from_mapping(raw: dict[str, Any] | None) -> ZedDepthFusionConfig:
    values = raw or {}
    return ZedDepthFusionConfig(
        enabled=bool(values.get("enabled", True)),
        depth_mode=str(values.get("depth_mode", "neural")).lower(),
        confidence_threshold=float(values.get("confidence_threshold", 50.0)),
        min_pose_score=float(values.get("min_pose_score", 0.30)),
        patch_radius_px=int(values.get("patch_radius_px", 4)),
        min_patch_samples=int(values.get("min_patch_samples", 4)),
        min_depth_m=float(values.get("min_depth_m", 0.40)),
        max_depth_m=float(values.get("max_depth_m", 10.0)),
        surface_gate_m=float(values.get("surface_gate_m", 0.35)),
        surface_gate_ratio=float(values.get("surface_gate_ratio", 0.10)),
        minimum_offset_samples=int(values.get("minimum_offset_samples", 24)),
        max_centered_residual_m=float(values.get("max_centered_residual_m", 0.20)),
        baseline_weight=float(values.get("baseline_weight", 1.0)),
        depth_weight=float(values.get("depth_weight", 0.35)),
        max_correction_m=float(values.get("max_correction_m", 0.06)),
        max_median_reprojection_increase_px=float(
            values.get("max_median_reprojection_increase_px", 0.75)
        ),
        max_camera_reprojection_increase_px=float(
            values.get("max_camera_reprojection_increase_px", 2.0)
        ),
        min_depth_residual_improvement_m=float(
            values.get("min_depth_residual_improvement_m", 0.001)
        ),
        max_final_median_reprojection_ratio=float(
            values.get("max_final_median_reprojection_ratio", 1.02)
        ),
        max_final_median_reprojection_increase_px=float(
            values.get("max_final_median_reprojection_increase_px", 0.25)
        ),
        max_final_p95_reprojection_ratio=float(
            values.get("max_final_p95_reprojection_ratio", 1.03)
        ),
        max_final_p95_reprojection_increase_px=float(
            values.get("max_final_p95_reprojection_increase_px", 0.50)
        ),
        max_final_p95_acceleration_ratio=float(
            values.get("max_final_p95_acceleration_ratio", 1.05)
        ),
        max_final_p95_acceleration_increase_mps2=float(
            values.get("max_final_p95_acceleration_increase_mps2", 1.0)
        ),
        max_final_bone_cv_increase_percent=float(
            values.get("max_final_bone_cv_increase_percent", 0.10)
        ),
    )


def final_depth_fusion_acceptance_gate(
    rgb_reference_report: dict[str, Any],
    depth_candidate_report: dict[str, Any],
    config: ZedDepthFusionConfig,
) -> dict[str, Any]:
    rgb = rgb_reference_report.get("after", {})
    depth = depth_candidate_report.get("after", {})
    failures: list[str] = []
    if depth_candidate_report.get("fallback_used", True) and not rgb_reference_report.get(
        "fallback_used", True
    ):
        failures.append("depth_candidate_optimizer_fallback")
    _check_upper_bound(
        "median_reprojection_degraded",
        _nested_metric(rgb, "reprojection_error_px", "median"),
        _nested_metric(depth, "reprojection_error_px", "median"),
        config.max_final_median_reprojection_ratio,
        config.max_final_median_reprojection_increase_px,
        failures,
    )
    _check_upper_bound(
        "p95_reprojection_degraded",
        _nested_metric(rgb, "reprojection_error_px", "p95"),
        _nested_metric(depth, "reprojection_error_px", "p95"),
        config.max_final_p95_reprojection_ratio,
        config.max_final_p95_reprojection_increase_px,
        failures,
    )
    _check_upper_bound(
        "p95_acceleration_degraded",
        _nested_metric(rgb, "acceleration_mps2", "p95"),
        _nested_metric(depth, "acceleration_mps2", "p95"),
        config.max_final_p95_acceleration_ratio,
        config.max_final_p95_acceleration_increase_mps2,
        failures,
    )
    rgb_bone = _finite_float(rgb.get("mean_bone_length_cv_percent"))
    depth_bone = _finite_float(depth.get("mean_bone_length_cv_percent"))
    if (
        rgb_bone is not None
        and depth_bone is not None
        and depth_bone > rgb_bone + config.max_final_bone_cv_increase_percent
    ):
        failures.append("bone_stability_degraded")
    return {
        "passed": not failures,
        "fallback_to_rgb_reference": bool(failures),
        "failures": failures,
        "checks": {
            "optimizer_not_degraded": not any("optimizer_fallback" in item for item in failures),
            "median_reprojection_not_degraded": not any(
                item.startswith("median_reprojection_degraded") for item in failures
            ),
            "p95_reprojection_not_degraded": not any(
                item.startswith("p95_reprojection_degraded") for item in failures
            ),
            "p95_acceleration_not_degraded": not any(
                item.startswith("p95_acceleration_degraded") for item in failures
            ),
            "bone_stability_not_degraded": "bone_stability_degraded" not in failures,
        },
        "rgb_reference_after": rgb,
        "depth_candidate_after": depth,
        "limits": {
            "max_final_median_reprojection_ratio": config.max_final_median_reprojection_ratio,
            "max_final_median_reprojection_increase_px": (
                config.max_final_median_reprojection_increase_px
            ),
            "max_final_p95_reprojection_ratio": config.max_final_p95_reprojection_ratio,
            "max_final_p95_reprojection_increase_px": config.max_final_p95_reprojection_increase_px,
            "max_final_p95_acceleration_ratio": config.max_final_p95_acceleration_ratio,
            "max_final_p95_acceleration_increase_mps2": (
                config.max_final_p95_acceleration_increase_mps2
            ),
            "max_final_bone_cv_increase_percent": config.max_final_bone_cv_increase_percent,
        },
    }


def load_depth_sources(session_path: str | Path) -> list[DepthSource]:
    path = Path(session_path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return []
    zed = raw.get("zed")
    if not isinstance(zed, dict):
        return []
    raw_sources = zed.get("depth_sources")
    if not isinstance(raw_sources, list):
        return []
    sources: list[DepthSource] = []
    for item in raw_sources:
        if not isinstance(item, dict):
            continue
        camera_id = item.get("camera_id")
        svo_path = _resolved_path(item.get("svo_path"), path.parent)
        mapping_path = _resolved_path(item.get("timestamp_mapping_report"), path.parent)
        if not isinstance(camera_id, str) or svo_path is None or mapping_path is None:
            continue
        sources.append(
            DepthSource(
                camera_id=camera_id,
                svo_path=svo_path,
                timestamp_mapping_report=mapping_path,
                prepared_frame_offset=int(item.get("prepared_frame_offset", 0)),
                imu_gravity_rotation_applied=bool(item.get("imu_gravity_rotation_applied", False)),
            )
        )
    return sources


def robust_depth_patch_sample(
    depth: np.ndarray,
    confidence: np.ndarray,
    xy: np.ndarray,
    expected_depth_m: float,
    config: ZedDepthFusionConfig,
) -> DepthPatchSample | None:
    if not np.all(np.isfinite(xy)) or not np.isfinite(expected_depth_m) or expected_depth_m <= 0.0:
        return None
    values = np.asarray(depth, dtype=float)
    quality = np.asarray(confidence, dtype=float)
    if values.shape != quality.shape or values.ndim != 2:
        raise ValueError("depth and confidence must be equally shaped 2D arrays")
    center_x, center_y = np.rint(np.asarray(xy, dtype=float)).astype(int)
    radius = config.patch_radius_px
    x0, x1 = max(center_x - radius, 0), min(center_x + radius + 1, values.shape[1])
    y0, y1 = max(center_y - radius, 0), min(center_y + radius + 1, values.shape[0])
    if x0 >= x1 or y0 >= y1:
        return None
    patch_depth = values[y0:y1, x0:x1].reshape(-1)
    patch_confidence = quality[y0:y1, x0:x1].reshape(-1)
    gate = max(config.surface_gate_m, config.surface_gate_ratio * expected_depth_m)
    valid = (
        np.isfinite(patch_depth)
        & np.isfinite(patch_confidence)
        & (patch_depth >= config.min_depth_m)
        & (patch_depth <= config.max_depth_m)
        & (patch_confidence <= config.confidence_threshold)
        & (np.abs(patch_depth - expected_depth_m) <= gate)
    )
    if int(np.count_nonzero(valid)) < config.min_patch_samples:
        return None
    usable_depth = patch_depth[valid]
    median = float(np.median(usable_depth))
    absolute_deviation = np.abs(usable_depth - median)
    mad = float(np.median(absolute_deviation))
    if mad > 1e-9:
        robust = absolute_deviation <= max(3.0 * 1.4826 * mad, 0.015)
        usable_depth = usable_depth[robust]
        usable_confidence = patch_confidence[valid][robust]
    else:
        usable_confidence = patch_confidence[valid]
    if usable_depth.size < config.min_patch_samples:
        return None
    return DepthPatchSample(
        depth_m=float(np.median(usable_depth)),
        confidence=float(np.median(usable_confidence)),
        pixel_count=int(usable_depth.size),
    )


def fuse_depth_constraints(
    baseline_point: np.ndarray,
    observations: list[tuple[CameraCalibration, float, float]],
    config: ZedDepthFusionConfig,
) -> np.ndarray:
    point = np.asarray(baseline_point, dtype=float).reshape(3)
    if not np.all(np.isfinite(point)) or not observations:
        return point.copy()
    matrix = config.baseline_weight * np.eye(3, dtype=float)
    target = config.baseline_weight * point.copy()
    for calibration, depth_m, sensor_weight in observations:
        rotation, _ = cv2.Rodrigues(np.asarray(calibration.rotation_vector, dtype=float))
        optical_axis = rotation[2]
        translation_z = float(np.asarray(calibration.translation_vector, dtype=float)[2])
        weight = config.depth_weight * max(float(sensor_weight), 0.0)
        matrix += weight * np.outer(optical_axis, optical_axis)
        target += weight * optical_axis * (float(depth_m) - translation_z)
    return np.linalg.solve(matrix, target)


def fuse_zed_depth_sequence(
    keypoints_3d_analysis: np.ndarray,
    accepted_mask: np.ndarray,
    frame_indices: np.ndarray,
    poses_2d_by_frame: dict[int, dict[str, PersonPose2D]],
    calibrations: dict[str, CameraCalibration],
    source_to_analysis: np.ndarray,
    session_path: str | Path,
    config: ZedDepthFusionConfig,
    body_joint_count: int = 17,
) -> ZedDepthFusionResult:
    baseline_analysis = np.asarray(keypoints_3d_analysis, dtype=float)
    output = baseline_analysis.copy()
    empty_report = _empty_report(config)
    if not config.enabled:
        empty_report["status"] = "disabled_by_config"
        return ZedDepthFusionResult(output, empty_report, [])
    sources = [source for source in load_depth_sources(session_path) if source.camera_id in calibrations]
    if not sources:
        empty_report["status"] = "unavailable_no_zed_depth_sources"
        return ZedDepthFusionResult(output, empty_report, [])

    points_source = transform_points(baseline_analysis, np.linalg.inv(source_to_analysis))
    accepted = np.asarray(accepted_mask, dtype=bool)
    frames = np.asarray(frame_indices, dtype=int)
    joint_count = min(body_joint_count, points_source.shape[1])
    raw: dict[tuple[int, int], list[dict[str, Any]]] = {}
    camera_reports: dict[str, dict[str, Any]] = {}
    for source in sources:
        with profile_stage(
            "zed_camera_depth_extraction",
            parent="zed_depth_fusion",
            tags={"camera_id": source.camera_id},
        ):
            camera_rows, camera_report = _extract_camera_depth(
                source,
                frames,
                points_source,
                accepted,
                poses_2d_by_frame,
                calibrations[source.camera_id],
                config,
                joint_count,
            )
        camera_reports[source.camera_id] = camera_report
        for row in camera_rows:
            raw.setdefault((int(row["sample_idx"]), int(row["joint_idx"])), []).append(row)

    with profile_stage("zed_surface_offset_estimation", parent="zed_depth_fusion"):
        offsets = _surface_offsets(raw, config.minimum_offset_samples)
    observation_rows: list[dict[str, Any]] = []
    accepted_corrections: list[float] = []
    residuals_before: list[float] = []
    residuals_after: list[float] = []
    modified = np.zeros((points_source.shape[0], joint_count), dtype=bool)
    candidate_started = time.perf_counter() if performance_profiling_active() else 0.0
    for (sample_idx, joint_idx), rows in raw.items():
        usable: list[dict[str, Any]] = []
        constraints: list[tuple[CameraCalibration, float, float]] = []
        baseline_point = points_source[sample_idx, joint_idx]
        for row in rows:
            offset = offsets.get((str(row["camera_id"]), joint_idx))
            if offset is None:
                row["decision"] = "rejected_insufficient_surface_offset_samples"
                observation_rows.append(row)
                continue
            corrected_depth = float(row["surface_depth_m"]) - offset
            centered_residual = corrected_depth - float(row["predicted_joint_depth_m"])
            row["surface_to_joint_offset_m"] = offset
            row["corrected_joint_depth_m"] = corrected_depth
            row["centered_residual_before_m"] = centered_residual
            if abs(centered_residual) > config.max_centered_residual_m:
                row["decision"] = "rejected_centered_depth_residual"
                observation_rows.append(row)
                continue
            confidence_weight = np.clip(1.0 - float(row["confidence"]) / 100.0, 0.05, 1.0)
            pose_weight = np.clip(float(row["pose_score"]), config.min_pose_score, 1.0)
            constraints.append(
                (
                    calibrations[str(row["camera_id"])],
                    corrected_depth,
                    float(confidence_weight * pose_weight),
                )
            )
            usable.append(row)
        if not usable:
            continue
        candidate = fuse_depth_constraints(baseline_point, constraints, config)
        correction = float(np.linalg.norm(candidate - baseline_point))
        reprojection_before = _joint_reprojection_errors(
            baseline_point, joint_idx, int(frames[sample_idx]), poses_2d_by_frame, calibrations
        )
        reprojection_after = _joint_reprojection_errors(
            candidate, joint_idx, int(frames[sample_idx]), poses_2d_by_frame, calibrations
        )
        before_depth = []
        after_depth = []
        for row in usable:
            calibration = calibrations[str(row["camera_id"])]
            before_depth.append(abs(_camera_depth(baseline_point, calibration) - row["corrected_joint_depth_m"]))
            after_depth.append(abs(_camera_depth(candidate, calibration) - row["corrected_joint_depth_m"]))
        before_median = float(np.median(before_depth))
        after_median = float(np.median(after_depth))
        decision = _fusion_decision(
            correction,
            reprojection_before,
            reprojection_after,
            before_median,
            after_median,
            config,
        )
        if decision == "accepted_depth_fusion":
            points_source[sample_idx, joint_idx] = candidate
            modified[sample_idx, joint_idx] = True
            accepted_corrections.append(correction)
            residuals_before.append(before_median)
            residuals_after.append(after_median)
        for row in usable:
            calibration = calibrations[str(row["camera_id"])]
            row["candidate_correction_m"] = correction
            row["centered_residual_after_m"] = (
                _camera_depth(candidate, calibration) - float(row["corrected_joint_depth_m"])
            )
            row["reprojection_before_px"] = _finite_median(reprojection_before)
            row["reprojection_after_px"] = _finite_median(reprojection_after)
            row["decision"] = decision
            observation_rows.append(row)

    if performance_profiling_active():
        record_profile_stage(
            "zed_candidate_construction_fusion",
            time.perf_counter() - candidate_started,
            parent="zed_depth_fusion",
        )

    output = transform_points(points_source, source_to_analysis)
    accepted_count = int(np.count_nonzero(modified))
    report = {
        "algorithm": "tk3d_zed_stereo_depth_auxiliary_fusion_v1",
        "status": "applied" if accepted_count else "completed_no_accepted_corrections",
        "applied": accepted_count > 0,
        "depth_is_auxiliary_evidence": True,
        "rgb_multiview_triangulation_remains_primary": True,
        "surface_to_joint_offset_scope": "estimated_per_run_per_camera_per_joint_not_global",
        "framewise_imu_pose_correction_applied": False,
        "imu_usage": "camera gravity/orientation calibration only",
        "configuration": _config_payload(config),
        "camera_count": len(sources),
        "cameras": camera_reports,
        "raw_depth_sample_count": int(sum(len(rows) for rows in raw.values())),
        "surface_offset_count": len(offsets),
        "fused_body_point_count": accepted_count,
        "candidate_body_point_count": len(raw),
        "fused_candidate_ratio": accepted_count / len(raw) if raw else 0.0,
        "correction_m": _distribution(accepted_corrections),
        "centered_depth_residual_before_m": _distribution(residuals_before),
        "centered_depth_residual_after_m": _distribution(residuals_after),
        "decision_counts": _decision_counts(observation_rows),
    }
    return ZedDepthFusionResult(output, report, observation_rows)


def _extract_camera_depth(
    source: DepthSource,
    frame_indices: np.ndarray,
    points_source: np.ndarray,
    accepted: np.ndarray,
    poses_2d_by_frame: dict[int, dict[str, PersonPose2D]],
    calibration: CameraCalibration,
    config: ZedDepthFusionConfig,
    joint_count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not source.svo_path.is_file() or not source.timestamp_mapping_report.is_file():
        return [], {
            "status": "missing_source",
            "svo_path": str(source.svo_path),
            "timestamp_mapping_report": str(source.timestamp_mapping_report),
        }
    mapping_payload = json.loads(source.timestamp_mapping_report.read_text(encoding="utf-8"))
    mapping = {
        int(item["output_frame_idx"]): int(item["source_frame_idx"])
        for item in mapping_payload.get("mapping", [])
    }
    try:
        import pyzed.sl as sl
    except ModuleNotFoundError as exc:
        raise RuntimeError("pyzed is required for configured ZED depth fusion") from exc
    init = sl.InitParameters()
    init.set_from_svo_file(str(source.svo_path))
    init.svo_real_time_mode = False
    mode_name = config.depth_mode.upper()
    if not hasattr(sl.DEPTH_MODE, mode_name):
        raise ValueError(f"Unsupported ZED depth mode: {config.depth_mode}")
    init.depth_mode = getattr(sl.DEPTH_MODE, mode_name)
    init.coordinate_units = sl.UNIT.METER
    camera = sl.Camera()
    with profile_stage(
        "zed_svo_open",
        parent="zed_camera_depth_extraction",
        tags={"camera_id": source.camera_id},
    ):
        status = camera.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Could not open ZED depth source {source.svo_path}: {status}")
    runtime = sl.RuntimeParameters()
    runtime.confidence_threshold = int(round(config.confidence_threshold))
    depth_mat = sl.Mat()
    confidence_mat = sl.Mat()
    rows: list[dict[str, Any]] = []
    requested = 0
    retrieved = 0
    depth_output_frames = 0
    last_source_idx: int | None = None
    depth: np.ndarray | None = None
    confidence: np.ndarray | None = None
    try:
        for sample_idx, frame_idx in enumerate(frame_indices):
            prepared_idx = int(frame_idx) + source.prepared_frame_offset
            source_idx = mapping.get(prepared_idx)
            if source_idx is None:
                continue
            requested += 1
            if source_idx != last_source_idx:
                if last_source_idx is None or source_idx != last_source_idx + 1:
                    with profile_stage(
                        "zed_mapping_seek",
                        parent="zed_camera_depth_extraction",
                        frame_index=int(frame_idx),
                    ):
                        camera.set_svo_position(source_idx)
                with profile_stage(
                    "zed_grab",
                    parent="zed_camera_depth_extraction",
                    frame_index=int(frame_idx),
                ):
                    grab_status = camera.grab(runtime)
                if grab_status != sl.ERROR_CODE.SUCCESS:
                    continue
                with profile_stage(
                    "zed_depth_retrieval",
                    parent="zed_camera_depth_extraction",
                    frame_index=int(frame_idx),
                ):
                    depth_status = camera.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
                if depth_status != sl.ERROR_CODE.SUCCESS:
                    continue
                with profile_stage(
                    "zed_confidence_retrieval",
                    parent="zed_camera_depth_extraction",
                    frame_index=int(frame_idx),
                ):
                    confidence_status = camera.retrieve_measure(
                        confidence_mat,
                        sl.MEASURE.CONFIDENCE,
                    )
                if confidence_status != sl.ERROR_CODE.SUCCESS:
                    continue
                retrieved += 1
                depth = np.asarray(depth_mat.get_data())
                confidence = np.asarray(confidence_mat.get_data())
                last_source_idx = source_idx
            if depth is None or confidence is None:
                continue
            depth_output_frames += 1
            pose = poses_2d_by_frame.get(int(frame_idx), {}).get(source.camera_id)
            if pose is None:
                continue
            with profile_stage(
                "zed_patch_extraction",
                parent="zed_camera_depth_extraction",
                frame_index=int(frame_idx),
            ):
                for joint_idx in range(joint_count):
                    if not accepted[sample_idx, joint_idx] or not pose.valid_mask[joint_idx]:
                        continue
                    pose_score = float(pose.scores[joint_idx])
                    if pose_score < config.min_pose_score:
                        continue
                    predicted = _camera_depth(points_source[sample_idx, joint_idx], calibration)
                    sample = robust_depth_patch_sample(
                        depth,
                        confidence,
                        pose.keypoints_xy[joint_idx],
                        predicted,
                        config,
                    )
                    if sample is None:
                        continue
                    rows.append(
                        {
                            "sample_idx": sample_idx,
                            "frame_idx": int(frame_idx),
                            "prepared_frame_idx": prepared_idx,
                            "source_frame_idx": source_idx,
                            "camera_id": source.camera_id,
                            "joint_idx": joint_idx,
                            "pose_score": pose_score,
                            "surface_depth_m": sample.depth_m,
                            "predicted_joint_depth_m": predicted,
                            "raw_surface_residual_m": sample.depth_m - predicted,
                            "confidence": sample.confidence,
                            "patch_pixel_count": sample.pixel_count,
                            "imu_gravity_rotation_applied": source.imu_gravity_rotation_applied,
                        }
                    )
    finally:
        camera.close()
    return rows, {
        "status": "complete",
        "svo_path": str(source.svo_path),
        "timestamp_mapping_report": str(source.timestamp_mapping_report),
        "prepared_frame_offset": source.prepared_frame_offset,
        "requested_frame_count": requested,
        "depth_output_frame_count": depth_output_frames,
        "unique_source_depth_frame_count": retrieved,
        "source_depth_frame_reuse_count": max(depth_output_frames - retrieved, 0),
        "depth_sample_count": len(rows),
        "imu_gravity_rotation_applied": source.imu_gravity_rotation_applied,
    }


def _surface_offsets(
    observations: dict[tuple[int, int], list[dict[str, Any]]],
    minimum_samples: int,
) -> dict[tuple[str, int], float]:
    grouped: dict[tuple[str, int], list[float]] = {}
    for rows in observations.values():
        for row in rows:
            key = (str(row["camera_id"]), int(row["joint_idx"]))
            grouped.setdefault(key, []).append(float(row["raw_surface_residual_m"]))
    return {
        key: float(np.median(values))
        for key, values in grouped.items()
        if len(values) >= minimum_samples
    }


def _fusion_decision(
    correction_m: float,
    reprojection_before: np.ndarray,
    reprojection_after: np.ndarray,
    depth_before_m: float,
    depth_after_m: float,
    config: ZedDepthFusionConfig,
) -> str:
    if not np.isfinite(correction_m) or correction_m > config.max_correction_m:
        return "rejected_correction_limit"
    if depth_before_m - depth_after_m < config.min_depth_residual_improvement_m:
        return "rejected_no_depth_improvement"
    finite = np.isfinite(reprojection_before) & np.isfinite(reprojection_after)
    if np.any(finite):
        increases = reprojection_after[finite] - reprojection_before[finite]
        if float(np.median(increases)) > config.max_median_reprojection_increase_px:
            return "rejected_median_reprojection_degradation"
        if float(np.max(increases)) > config.max_camera_reprojection_increase_px:
            return "rejected_camera_reprojection_degradation"
    return "accepted_depth_fusion"


def _joint_reprojection_errors(
    point: np.ndarray,
    joint_idx: int,
    frame_idx: int,
    poses_2d_by_frame: dict[int, dict[str, PersonPose2D]],
    calibrations: dict[str, CameraCalibration],
) -> np.ndarray:
    homogeneous = np.append(np.asarray(point, dtype=float), 1.0)
    errors: list[float] = []
    for camera_id, pose in poses_2d_by_frame.get(frame_idx, {}).items():
        calibration = calibrations.get(camera_id)
        if calibration is None or not pose.valid_mask[joint_idx]:
            continue
        projected = np.asarray(calibration.projection_matrix, dtype=float) @ homogeneous
        if not np.isfinite(projected[2]) or abs(projected[2]) <= 1e-9:
            continue
        xy = projected[:2] / projected[2]
        errors.append(float(np.linalg.norm(xy - pose.keypoints_xy[joint_idx])))
    return np.asarray(errors, dtype=float)


def _camera_depth(point: np.ndarray, calibration: CameraCalibration) -> float:
    rotation, _ = cv2.Rodrigues(np.asarray(calibration.rotation_vector, dtype=float))
    translation = np.asarray(calibration.translation_vector, dtype=float)
    return float((rotation @ np.asarray(point, dtype=float) + translation)[2])


def _resolved_path(value: Any, root: Path) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _finite_median(values: np.ndarray) -> float | None:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.median(finite)) if finite.size else None


def _nested_metric(payload: dict[str, Any], group: str, name: str) -> float | None:
    values = payload.get(group)
    return _finite_float(values.get(name)) if isinstance(values, dict) else None


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _check_upper_bound(
    failure_name: str,
    reference: float | None,
    candidate: float | None,
    ratio: float,
    increase: float,
    failures: list[str],
) -> None:
    if reference is None or candidate is None:
        failures.append(f"{failure_name}_metric_missing")
        return
    if candidate > max(reference * ratio, reference + increase):
        failures.append(failure_name)


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return {"count": 0, "median": None, "p95": None, "max": None}
    return {
        "count": int(finite.size),
        "median": float(np.median(finite)),
        "p95": float(np.percentile(finite, 95)),
        "max": float(np.max(finite)),
    }


def _decision_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        decision = str(row.get("decision", "unknown"))
        counts[decision] = counts.get(decision, 0) + 1
    return counts


def _config_payload(config: ZedDepthFusionConfig) -> dict[str, Any]:
    return {name: getattr(config, name) for name in config.__dataclass_fields__}


def _empty_report(config: ZedDepthFusionConfig) -> dict[str, Any]:
    return {
        "algorithm": "tk3d_zed_stereo_depth_auxiliary_fusion_v1",
        "status": "not_run",
        "applied": False,
        "depth_is_auxiliary_evidence": True,
        "rgb_multiview_triangulation_remains_primary": True,
        "framewise_imu_pose_correction_applied": False,
        "configuration": _config_payload(config),
        "camera_count": 0,
        "raw_depth_sample_count": 0,
        "fused_body_point_count": 0,
    }
