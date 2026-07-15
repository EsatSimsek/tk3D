from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.aist_calibration import find_aist_camera_setting, load_aist_camera_calibrations
from src.camera_calibration import save_calibrations
from src.coordinate_system import aist_world_to_analysis, calibration_metadata
from src.video_io import ensure_output_tree, load_session


def main() -> None:
    parser = argparse.ArgumentParser(description="Import AIST++ camera parameters into TK3D cameras.json.")
    parser.add_argument("--session", required=True, help="Path to AIST session YAML")
    parser.add_argument("--output-root", default="outputs", help="Output root directory")
    parser.add_argument("--annotations-dir", default=None, help="Override AIST++ annotations directory")
    args = parser.parse_args()

    session_path = Path(args.session).resolve()
    session = load_session(session_path)
    with session_path.open("r", encoding="utf-8") as file:
        raw_session = yaml.safe_load(file)

    aist_cfg = raw_session.get("aist", {})
    sequence = aist_cfg.get("sequence") or session.task_name
    annotations_dir = Path(args.annotations_dir) if args.annotations_dir else session.root_dir / aist_cfg.get("annotations_dir", "annotations")
    cameras_dir = annotations_dir / "cameras"
    camera_ids = [camera.camera_id for camera in session.cameras]

    setting = find_aist_camera_setting(sequence, cameras_dir)
    calibrations = load_aist_camera_calibrations(sequence, cameras_dir, camera_ids=camera_ids)
    output_paths = ensure_output_tree(ROOT / args.output_root, session.session_id)
    output_path = output_paths["calibration"] / "cameras.json"
    metadata = calibration_metadata(
        calibration_mode="aist_official_multiview",
        source_coordinate_system={
            "name": "aist_world",
            "unit": "centimeter",
            "axes": {"x": "right", "y": "up", "z": "forward"},
        },
        source_to_analysis=aist_world_to_analysis(),
    )
    save_calibrations(calibrations, output_path, metadata=metadata)

    report = {
        "source": "AIST++ cameras",
        "sequence": sequence,
        "setting": setting,
        "camera_count": len(calibrations),
        "camera_ids": [camera.camera_id for camera in calibrations],
        "output_path": str(output_path),
    }
    report_path = output_paths["json"] / "aist_camera_import_report.json"
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print(f"saved: {output_path}")
    print(f"setting: {setting}")
    print(f"camera_count: {len(calibrations)}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
