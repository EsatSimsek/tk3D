from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preflight import preflight_summary, run_preflight, save_preflight_report
from src.config_validation import validate_model_config
from src.video_io import ensure_output_tree, load_session


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TK3D project readiness checks.")
    parser.add_argument("--session", required=True, help="Path to session.yaml")
    parser.add_argument("--model-config", default="config/model_config.yaml", help="Model config path")
    parser.add_argument("--output-root", default="outputs", help="Output root directory")
    parser.add_argument("--require-videos", action="store_true", help="Report missing videos as errors")
    parser.add_argument("--require-calibration-videos", action="store_true", help="Report missing calibration videos as errors")
    parser.add_argument("--require-model-files", action="store_true", help="Report missing model files as errors")
    args = parser.parse_args()

    session = load_session(args.session)
    with (ROOT / args.model_config).open("r", encoding="utf-8") as file:
        model_config = validate_model_config(yaml.safe_load(file))

    issues = run_preflight(
        session=session,
        model_config=model_config,
        videos_required=args.require_videos,
        calibration_videos_required=args.require_calibration_videos,
        model_files_required=args.require_model_files,
    )
    output_paths = ensure_output_tree(ROOT / args.output_root, session.session_id)
    report_path = output_paths["json"] / "preflight_report.json"
    save_preflight_report(issues, report_path)

    summary = preflight_summary(issues)
    print(f"preflight status: {summary['status']}")
    print(f"errors: {summary['error_count']}")
    print(f"warnings: {summary['warning_count']}")
    print(f"saved: {report_path}")


if __name__ == "__main__":
    main()
