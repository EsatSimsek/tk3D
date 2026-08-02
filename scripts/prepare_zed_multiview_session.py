from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.camera_calibration import save_calibrations
from src.coordinate_system import ANALYSIS_COORDINATE_SYSTEM, calibration_metadata
from src.zed_svo import (
    ZedSvoMetadata,
    absolute_camera_pose,
    calibration_from_camera_pose,
    common_timestamp_timeline,
    nearest_timestamp_indices,
    timestamp_mapping_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare timestamp-synchronized ZED SVO2 files for the TK3D multi-view pipeline."
    )
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--fusion-config", required=True, help="ZED Fusion/ZED360 JSON calibration")
    parser.add_argument("--svo", nargs="+", required=True, help="Two or more SVO/SVO2 recordings")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--fps", type=float, default=None, help="Output FPS; defaults to the slowest recording")
    parser.add_argument(
        "--codec",
        choices=("ffv1", "mp4v"),
        default="ffv1",
        help="FFV1 is lossless and recommended for highest-quality inference.",
    )
    parser.add_argument("--skip-sha256", action="store_true")
    args = parser.parse_args()

    if len(args.svo) < 2:
        parser.error("At least two --svo recordings are required")
    output_root = (ROOT / args.output_root).resolve()
    session_root = output_root / args.session_id
    source_root = session_root / "source"
    calibration_root = session_root / "calibration"
    guarded_paths = [source_root, calibration_root / "cameras.json"]
    if any(path.exists() for path in guarded_paths):
        raise SystemExit(
            f"Refusing to overwrite an existing prepared session: {session_root}. Use a new --session-id."
        )

    sl = _load_zed_sdk()
    svo_paths = [Path(value).resolve() for value in args.svo]
    missing = [str(path) for path in svo_paths if not path.is_file()]
    if missing:
        raise SystemExit(f"SVO files not found: {missing}")
    fusion_path = Path(args.fusion_config).resolve()
    if not fusion_path.is_file():
        raise SystemExit(f"Fusion configuration not found: {fusion_path}")

    metadata = [_scan_svo(path, sl) for path in svo_paths]
    serials = [item.serial_number for item in metadata]
    if len(serials) != len(set(serials)):
        raise SystemExit(f"Every SVO must have a unique camera serial number: {serials}")
    output_fps = float(args.fps or min(item.fps for item in metadata))
    if output_fps > min(item.fps for item in metadata) + 1e-6:
        raise SystemExit("Output FPS cannot exceed the slowest source recording")
    timeline = common_timestamp_timeline([item.timestamps_ns for item in metadata], output_fps)
    mappings = {
        item.serial_number: nearest_timestamp_indices(item.timestamps_ns, timeline)
        for item in metadata
    }

    fusion_configs = sl.read_fusion_configuration_file(
        str(fusion_path),
        sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP,
        sl.UNIT.METER,
    )
    fusion_by_serial = {int(config.serial_number): config for config in fusion_configs}
    missing_fusion = sorted(set(serials) - set(fusion_by_serial))
    if missing_fusion:
        raise SystemExit(f"Fusion calibration is missing SVO camera serials: {missing_fusion}")

    source_root.mkdir(parents=True, exist_ok=False)
    videos_root = source_root / "videos"
    videos_root.mkdir()
    reports_root = source_root / "reports"
    reports_root.mkdir()
    calibrations = []
    camera_entries = []
    camera_reports = []
    depth_sources = []
    for item in metadata:
        camera_id = f"zed_{item.serial_number}"
        suffix = ".avi" if args.codec == "ffv1" else ".mp4"
        video_path = videos_root / f"{camera_id}{suffix}"
        _write_synchronized_video(
            metadata=item,
            source_indices=mappings[item.serial_number],
            output_path=video_path,
            output_fps=output_fps,
            codec=args.codec,
            sl=sl,
        )
        fusion = fusion_by_serial[item.serial_number]
        camera_to_world = absolute_camera_pose(
            np.asarray(fusion.pose.m, dtype=float),
            item.imu_rotation,
            bool(fusion.override_gravity),
        )
        calibrations.append(
            calibration_from_camera_pose(
                camera_id=camera_id,
                image_size=item.image_size,
                intrinsic_matrix=item.intrinsic_matrix,
                distortion_coefficients=item.distortion_coefficients,
                camera_to_world=camera_to_world,
            )
        )
        mapping_report = timestamp_mapping_report(
            item.timestamps_ns,
            timeline,
            mappings[item.serial_number],
            output_fps,
        )
        mapping_path = reports_root / f"{camera_id}_timestamp_mapping.json"
        _write_json(mapping_path, mapping_report)
        camera_entries.append(
            {
                "camera_id": camera_id,
                "video_path": video_path.relative_to(source_root).as_posix(),
                "calibration_video_path": None,
            }
        )
        camera_reports.append(
            {
                "camera_id": camera_id,
                "serial_number": item.serial_number,
                "source_svo": str(item.path),
                "source_sha256": None if args.skip_sha256 else _sha256(item.path),
                "camera_model": item.camera_model,
                "firmware_version": item.firmware_version,
                "source_fps": item.fps,
                "source_image_size": list(item.image_size),
                "source_frame_count": int(item.timestamps_ns.size),
                "first_timestamp_ns": int(item.timestamps_ns[0]),
                "last_timestamp_ns": int(item.timestamps_ns[-1]),
                "fusion_override_gravity": bool(fusion.override_gravity),
                "imu_gravity_rotation_applied": not bool(fusion.override_gravity),
                "camera_to_world": camera_to_world.tolist(),
                "timestamp_mapping_report": str(mapping_path.resolve()),
                "timestamp_summary": {key: value for key, value in mapping_report.items() if key != "mapping"},
                "prepared_video": str(video_path.resolve()),
            }
        )
        depth_sources.append(
            {
                "camera_id": camera_id,
                "svo_path": str(item.path),
                "timestamp_mapping_report": str(mapping_path.resolve()),
                "prepared_frame_offset": 0,
                "imu_gravity_rotation_applied": not bool(fusion.override_gravity),
            }
        )

    calibration_metadata_payload = calibration_metadata(
        calibration_mode="zed_fusion_multiview",
        source_coordinate_system=ANALYSIS_COORDINATE_SYSTEM,
        source_to_analysis=np.eye(4, dtype=float),
    )
    calibration_metadata_payload.update(
        {
            "source": "ZED Fusion configuration",
            "source_path": str(fusion_path),
            "source_sha256": None if args.skip_sha256 else _sha256(fusion_path),
            "camera_pose_convention": (
                "camera_to_world imported in TK3D axes, inverted, then camera axes converted "
                "from x-right/y-forward/z-up to OpenCV x-right/y-down/z-forward"
            ),
            "zed_coordinate_system": "RIGHT_HANDED_Z_UP",
            "zed_unit": "METER",
        }
    )
    calibration_path = calibration_root / "cameras.json"
    save_calibrations(calibrations, calibration_path, metadata=calibration_metadata_payload)

    session_payload = {
        "session_id": args.session_id,
        "task_name": args.task_name,
        "fps": output_fps,
        "cameras": camera_entries,
        "sync": {
            "method": "zed_hardware_timestamp_resampled",
            "offsets": {entry["camera_id"]: 0 for entry in camera_entries},
            "offsets_sec": {entry["camera_id"]: 0.0 for entry in camera_entries},
        },
        "zed": {
            "fusion_config": str(fusion_path),
            "timestamp_timeline_start_ns": int(timeline[0]),
            "timestamp_timeline_end_ns": int(timeline[-1]),
            "prepared_frame_count": int(timeline.size),
            "video_codec": args.codec,
            "depth_sources": depth_sources,
            "imu_usage": "gravity/orientation calibration; not framewise athlete motion correction",
        },
    }
    session_path = source_root / "session.yaml"
    _write_yaml(session_path, session_payload)
    report = {
        "schema_version": 1,
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "session_id": args.session_id,
        "task_name": args.task_name,
        "camera_count": len(metadata),
        "output_fps": output_fps,
        "output_frame_count": int(timeline.size),
        "output_duration_sec": float(timeline.size / output_fps),
        "timestamp_span_sec": float((timeline[-1] - timeline[0]) / 1_000_000_000.0),
        "video_codec": args.codec,
        "lossless_video": args.codec == "ffv1",
        "fusion_config": str(fusion_path),
        "calibration_path": str(calibration_path.resolve()),
        "session_path": str(session_path.resolve()),
        "cameras": camera_reports,
    }
    report_path = reports_root / "zed_ingest_report.json"
    _write_json(report_path, report)
    print(f"session: {session_path}")
    print(f"calibration: {calibration_path}")
    print(f"report: {report_path}")
    print(f"cameras: {len(metadata)}")
    print(f"frames: {timeline.size} @ {output_fps:g} FPS")


def _load_zed_sdk() -> Any:
    try:
        import pyzed.sl as sl
    except ModuleNotFoundError as exc:
        raise SystemExit("pyzed is required; install the matching ZED SDK Python API") from exc
    return sl


def _scan_svo(path: Path, sl: Any) -> ZedSvoMetadata:
    init = sl.InitParameters()
    init.set_from_svo_file(str(path))
    init.svo_real_time_mode = False
    init.depth_mode = sl.DEPTH_MODE.NONE
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP
    init.coordinate_units = sl.UNIT.METER
    camera = sl.Camera()
    status = camera.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        raise SystemExit(f"Could not open {path}: {status}")
    timestamps: list[int] = []
    imu_rotation: np.ndarray | None = None
    try:
        info = camera.get_camera_information()
        configuration = info.camera_configuration
        left = configuration.calibration_parameters.left_cam
        while camera.grab() == sl.ERROR_CODE.SUCCESS:
            timestamps.append(camera.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds())
            if imu_rotation is None:
                sensors = sl.SensorsData()
                if camera.get_sensors_data(sensors, sl.TIME_REFERENCE.IMAGE) == sl.ERROR_CODE.SUCCESS:
                    imu = sensors.get_imu_data()
                    if imu.is_available:
                        imu_rotation = np.asarray(imu.get_pose().m, dtype=float)[:3, :3].copy()
    finally:
        camera.close()
    if len(timestamps) < 2:
        raise SystemExit(f"SVO contains fewer than two readable frames: {path}")
    intrinsic = np.asarray(
        [[left.fx, 0.0, left.cx], [0.0, left.fy, left.cy], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    return ZedSvoMetadata(
        path=path,
        serial_number=int(info.serial_number),
        camera_model=str(info.camera_model),
        firmware_version=int(configuration.firmware_version),
        fps=float(configuration.fps),
        image_size=(int(configuration.resolution.width), int(configuration.resolution.height)),
        timestamps_ns=np.asarray(timestamps, dtype=np.int64),
        intrinsic_matrix=intrinsic,
        distortion_coefficients=np.zeros(5, dtype=float),
        imu_rotation=imu_rotation,
    )


def _write_synchronized_video(
    metadata: ZedSvoMetadata,
    source_indices: np.ndarray,
    output_path: Path,
    output_fps: float,
    codec: str,
    sl: Any,
) -> None:
    fourcc = cv2.VideoWriter_fourcc(*("FFV1" if codec == "ffv1" else "mp4v"))
    writer = cv2.VideoWriter(str(output_path), fourcc, output_fps, metadata.image_size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output_path}")
    desired = Counter(int(value) for value in np.asarray(source_indices, dtype=np.int64))
    init = sl.InitParameters()
    init.set_from_svo_file(str(metadata.path))
    init.svo_real_time_mode = False
    init.depth_mode = sl.DEPTH_MODE.NONE
    camera = sl.Camera()
    status = camera.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        writer.release()
        raise RuntimeError(f"Could not reopen {metadata.path}: {status}")
    image = sl.Mat()
    written = 0
    try:
        for source_index in range(metadata.timestamps_ns.size):
            if camera.grab() != sl.ERROR_CODE.SUCCESS:
                raise RuntimeError(f"Unexpected grab failure at source frame {source_index}: {metadata.path}")
            repeat = desired.get(source_index, 0)
            if repeat == 0:
                continue
            retrieve_status = camera.retrieve_image(image, sl.VIEW.LEFT)
            if retrieve_status != sl.ERROR_CODE.SUCCESS:
                raise RuntimeError(f"Could not retrieve source frame {source_index}: {retrieve_status}")
            frame = np.asarray(image.get_data())
            if frame.ndim != 3 or frame.shape[2] < 3:
                raise RuntimeError(f"Unexpected ZED frame shape at {source_index}: {frame.shape}")
            bgr = np.ascontiguousarray(frame[:, :, :3])
            for _ in range(repeat):
                writer.write(bgr)
                written += 1
    finally:
        camera.close()
        writer.release()
    if written != len(source_indices):
        raise RuntimeError(f"Prepared video has {written} frames; expected {len(source_indices)}: {output_path}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    temporary.replace(path)


if __name__ == "__main__":
    main()
