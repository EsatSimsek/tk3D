from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.video_io import load_session


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a TK3D session file.")
    parser.add_argument("--session", required=True, help="Path to session.yaml")
    args = parser.parse_args()

    session = load_session(args.session)
    print(f"session_id: {session.session_id}")
    print(f"task_name: {session.task_name}")
    print(f"camera_count: {len(session.cameras)}")
    for camera in session.cameras:
        print(f"- {camera.camera_id}")
        print(f"  video: {camera.video_path}")
        print(f"  calibration: {camera.calibration_video_path}")
        print(f"  frame_offset: {camera.frame_offset}")


if __name__ == "__main__":
    main()
