from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model_runtime import check_model_runtime, save_model_runtime_report
from src.config_validation import validate_model_config
from src.video_io import ensure_output_tree, load_session


def main() -> None:
    parser = argparse.ArgumentParser(description="Check TK3D ViTPose model runtime readiness.")
    parser.add_argument("--session", required=True, help="Path to session.yaml")
    parser.add_argument("--model-config", default="config/model_config.yaml", help="Model config path")
    parser.add_argument("--output-root", default="outputs", help="Output root directory")
    args = parser.parse_args()

    session = load_session(args.session)
    with (ROOT / args.model_config).open("r", encoding="utf-8") as file:
        model_config = validate_model_config(yaml.safe_load(file))

    statuses = {"pose2d": check_model_runtime(model_config.get("pose2d", {}), ROOT)}
    pose3d_config = model_config.get("pose3d_single_view", {})
    if pose3d_config.get("enabled", True):
        statuses["pose3d_single_view"] = check_model_runtime(pose3d_config, ROOT)
    output_paths = ensure_output_tree(ROOT / args.output_root, session.session_id)
    output_path = output_paths["json"] / "model_runtime_report.json"
    save_model_runtime_report(statuses, output_path)

    for name, status in statuses.items():
        print(f"{name}: {'ready' if status.ready else 'not_ready'} - {status.message}")
    if not pose3d_config.get("enabled", True):
        print("pose3d_single_view: skipped - optional helper is disabled")
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
