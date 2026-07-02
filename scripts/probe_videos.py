from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.video_io import ensure_output_tree, load_session
from src.video_probe import probe_session_videos, video_probe_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect configured TK3D camera videos.")
    parser.add_argument("--session", required=True, help="Path to session.yaml")
    parser.add_argument("--output-root", default="outputs", help="Output root directory")
    args = parser.parse_args()

    session = load_session(args.session)
    probes = probe_session_videos(session)
    summary = video_probe_summary(probes)
    output_paths = ensure_output_tree(ROOT / args.output_root, session.session_id)
    output_path = output_paths["json"] / "video_probe_report.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(f"camera_count: {summary['camera_count']}")
    print(f"opened_count: {summary['opened_count']}")
    print(f"all_opened: {summary['all_opened']}")
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
