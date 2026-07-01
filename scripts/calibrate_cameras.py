from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.camera_calibration import calibrate_single_camera, calibration_report, save_calibrations
from src.video_io import ensure_output_tree, load_session


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate TK3D cameras from checkerboard videos.")
    parser.add_argument("--session", required=True, help="Path to session.yaml")
    parser.add_argument("--config", default="config/calibration_config.yaml", help="Calibration config path")
    parser.add_argument("--output-root", default="outputs", help="Output root directory")
    args = parser.parse_args()

    session = load_session(args.session)
    with (ROOT / args.config).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    checker = config["checkerboard"]
    pattern_size = tuple(checker["pattern_size"])
    square_size_m = float(checker["square_size_m"])
    frame_stride = int(checker["frame_stride"])
    min_valid_frames = int(checker["min_valid_frames"])

    output_paths = ensure_output_tree(ROOT / args.output_root, session.session_id)
    calibrations = []
    errors = []
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
            )
            calibrations.append(calibration)
        except Exception as exc:
            errors.append({"camera_id": camera.camera_id, "error": str(exc)})

    cameras_path = output_paths["calibration"] / "cameras.json"
    report_path = output_paths["calibration"] / "calibration_report.json"
    save_calibrations(calibrations, cameras_path)
    report = calibration_report(calibrations)
    report["errors"] = errors
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print(f"saved: {cameras_path}")
    print(f"saved: {report_path}")
    if errors:
        print("calibration completed with errors; inspect calibration_report.json")


if __name__ == "__main__":
    main()
