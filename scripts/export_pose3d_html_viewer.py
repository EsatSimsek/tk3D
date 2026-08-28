from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

from src.pose3d_html_viewer import write_pose3d_html_viewer
from src.artifact_contracts import load_run_bound_main_3d_artifact


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a 3D pose JSON run as an interactive HTML viewer.")
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    input_path = Path(args.input_json).resolve()
    payload, compatibility = load_run_bound_main_3d_artifact(input_path)
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
    print(f"artifact_compatibility: {compatibility.value}")


if __name__ == "__main__":
    main()
