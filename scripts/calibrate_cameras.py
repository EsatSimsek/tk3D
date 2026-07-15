from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.camera_calibration import calibrate_multiview_cameras, calibrate_single_camera, calibration_report, save_calibrations
from src.coordinate_system import calibration_metadata, opencv_reference_to_analysis
from src.config_validation import validate_calibration_config
from src.video_io import ensure_output_tree, load_session


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate TK3D cameras from checkerboard videos.")
    parser.add_argument("--session", required=True, help="Path to session.yaml")
    parser.add_argument("--config", default="config/calibration_config.yaml", help="Calibration config path")
    parser.add_argument("--output-root", default="outputs", help="Output root directory")
    parser.add_argument(
        "--allow-intrinsics-only-fallback",
        action="store_true",
        help="Save per-camera intrinsics to intrinsics_only.json when common extrinsics fail. Never marks them production-ready.",
    )
    args = parser.parse_args()

    session = load_session(args.session)
    with (ROOT / args.config).open("r", encoding="utf-8") as file:
        config = validate_calibration_config(yaml.safe_load(file))

    checker = config["checkerboard"]
    pattern_size = tuple(checker["pattern_size"])
    square_size_m = float(checker["square_size_m"])
    frame_stride = int(checker["frame_stride"])
    min_valid_frames = int(checker["min_valid_frames"])
    calibration_flags = _opencv_calibration_flags(config.get("calibration", {}).get("flags", {}))
    reference_camera_id = checker.get("reference_camera_id") or config.get("extrinsics", {}).get("world_origin_camera")

    output_paths = ensure_output_tree(ROOT / args.output_root, session.session_id)
    calibrations = []
    errors = []
    calibration_mode = "multiview_common_reference"
    try:
        camera_videos = {
            camera.camera_id: camera.calibration_video_path
            for camera in session.cameras
            if camera.calibration_video_path is not None
        }
        missing_calibration = [camera.camera_id for camera in session.cameras if camera.calibration_video_path is None]
        if missing_calibration:
            raise FileNotFoundError(f"calibration_video_path is missing for: {', '.join(missing_calibration)}")
        calibrations = calibrate_multiview_cameras(
            camera_videos=camera_videos,
            frame_offsets={camera.camera_id: camera.frame_offset for camera in session.cameras},
            pattern_size=pattern_size,
            square_size_m=square_size_m,
            frame_stride=frame_stride,
            min_valid_frames=min_valid_frames,
            min_common_frames=int(checker.get("min_common_frames", 3)),
            reference_camera_id=reference_camera_id,
            calibration_flags=calibration_flags,
            time_offsets_sec={camera.camera_id: camera.time_offset_sec for camera in session.cameras},
            sync_tolerance_sec=checker.get("sync_tolerance_sec"),
        )
    except Exception as exc:
        calibration_mode = "intrinsics_only_fallback"
        errors.append(
            {
                "camera_id": None,
                "error": f"multiview common-reference calibration failed: {exc}",
            }
        )
        if not args.allow_intrinsics_only_fallback:
            report_path = output_paths["calibration"] / "calibration_report.json"
            with report_path.open("w", encoding="utf-8") as file:
                json.dump({"calibration_mode": "failed", "errors": errors, "camera_count": 0}, file, indent=2)
            raise SystemExit(
                "Common-reference calibration failed; no cameras.json was written. "
                f"Inspect {report_path}. Use --allow-intrinsics-only-fallback only for intrinsic diagnostics."
            )
        for camera in session.cameras:
            try:
                if camera.calibration_video_path is None:
                    raise FileNotFoundError("calibration_video_path is missing")
                calibration = calibrate_single_camera(
                    camera_id=camera.camera_id,
                    video_path=camera.calibration_video_path,
                    pattern_size=pattern_size,
                    square_size_m=square_size_m,
                    frame_stride=frame_stride,
                    min_valid_frames=min_valid_frames,
                    calibration_flags=calibration_flags,
                )
                calibrations.append(calibration)
            except Exception as camera_exc:
                errors.append({"camera_id": camera.camera_id, "error": str(camera_exc)})
        if len(calibrations) > 1:
            errors.append(
                {
                    "camera_id": None,
                    "error": (
                        "single-camera fallback calibrations do not share a guaranteed world coordinate frame; "
                        "capture synchronized checkerboard detections across cameras for metric multi-view 3D."
                    ),
                }
            )

    cameras_path = output_paths["calibration"] / (
        "cameras.json" if calibration_mode == "multiview_common_reference" else "intrinsics_only.json"
    )
    report_path = output_paths["calibration"] / "calibration_report.json"
    metadata = calibration_metadata(
        calibration_mode=calibration_mode,
        source_coordinate_system={
            "name": "reference_camera_opencv",
            "unit": "meter",
            "axes": {"x": "right", "y": "down", "z": "forward"},
        },
        source_to_analysis=opencv_reference_to_analysis(),
    )
    save_calibrations(calibrations, cameras_path, metadata=metadata)
    report = calibration_report(calibrations)
    report["calibration_mode"] = calibration_mode
    report["production_ready"] = calibration_mode == "multiview_common_reference"
    report["reprojection_error_warn_px"] = float(config.get("calibration", {}).get("reprojection_error_warn_px", 1.5))
    report["warnings"] = [
        f"{calibration.camera_id}: intrinsic RMS {calibration.reprojection_error_px:.3f}px exceeds warning threshold"
        for calibration in calibrations
        if calibration.reprojection_error_px is not None
        and calibration.reprojection_error_px > report["reprojection_error_warn_px"]
    ]
    report["errors"] = errors
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print(f"saved: {cameras_path}")
    print(f"saved: {report_path}")
    if errors:
        print("calibration completed with errors; inspect calibration_report.json")


def _opencv_calibration_flags(raw: dict) -> int:
    flags = 0
    if bool(raw.get("fix_aspect_ratio", False)):
        flags |= cv2.CALIB_FIX_ASPECT_RATIO
    if bool(raw.get("zero_tangent_dist", False)):
        flags |= cv2.CALIB_ZERO_TANGENT_DIST
    return flags


if __name__ == "__main__":
    main()
