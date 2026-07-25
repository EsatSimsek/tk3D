from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pose3d_html_viewer import write_pose3d_html_viewer


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a 3D pose JSON run as an interactive HTML viewer.")
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    input_path = Path(args.input_json).resolve()
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    keypoints = np.asarray(payload["keypoints_3d_world"], dtype=float)
    output_path = (
        Path(args.output).resolve() if args.output else input_path.parent.parent / "viewer" / "pose3d_viewer.html"
    )
    result = write_pose3d_html_viewer(
        keypoints,
        output_path,
        fps=float(payload.get("sample_fps") or 30.0),
        title=str(payload.get("run_id") or "3D pose"),
    )
    print(f"saved: {result}")


if __name__ == "__main__":
    main()
