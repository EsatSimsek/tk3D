"""Refine one camera's orientation from a recording's own 2D keypoints.

Writes a NEW calibration file with its own calibration mode and provenance; the
source calibration is never modified. Fits on even frames and reports odd frames.
The result is internal two-view consistency, not independent 3D accuracy.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.artifact_io import sha256_file, write_json_exclusive  # noqa: E402
from src.calibration_refinement import REFINED_CALIBRATION_MODE, refine_relative_rotation  # noqa: E402
from src.camera_calibration import load_calibration_bundle  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", required=True, type=Path, help="Source cameras.json (never modified).")
    parser.add_argument("--keypoints-2d", required=True, type=Path, help="Exported vitpose_keypoints_2d_flat.csv of the recording.")
    parser.add_argument("--reference-camera", required=True, help="Camera whose pose (and the world axes) is kept.")
    parser.add_argument("--min-score", type=float, default=0.8, help="Minimum 2D keypoint score in both cameras.")
    parser.add_argument("--output", required=True, type=Path, help="New calibration JSON; refused if it exists.")
    args = parser.parse_args()

    source = args.calibration.resolve()
    output = args.output.resolve()
    if output == source or output.exists():
        raise SystemExit(f"Refusing to overwrite: {output}")
    bundle = load_calibration_bundle(source)
    if len(bundle.calibrations) != 2:
        raise SystemExit("Two-view refinement needs exactly two cameras")
    if args.reference_camera not in bundle.calibrations:
        raise SystemExit(f"Unknown reference camera: {args.reference_camera}")
    moving_id = next(camera_id for camera_id in bundle.calibrations if camera_id != args.reference_camera)

    frame = pd.read_csv(args.keypoints_2d)
    first = frame[frame.camera_id == args.reference_camera].set_index(["frame_idx", "joint_idx"])
    second = frame[frame.camera_id == moving_id].set_index(["frame_idx", "joint_idx"])
    joined = first.join(second, lsuffix="_a", rsuffix="_b", how="inner")
    keep = (
        (joined.score_a >= args.min_score)
        & (joined.score_b >= args.min_score)
        & joined.valid_a.astype(bool)
        & joined.valid_b.astype(bool)
    )
    joined = joined[keep].dropna(subset=["x_a", "y_a", "x_b", "y_b"])
    if len(joined) < 100:
        raise SystemExit(f"Too few high-confidence correspondences: {len(joined)}")
    frames = joined.index.get_level_values("frame_idx").to_numpy()
    refined, report = refine_relative_rotation(
        bundle.calibrations[args.reference_camera],
        bundle.calibrations[moving_id],
        joined[["x_a", "y_a"]].to_numpy(float),
        joined[["x_b", "y_b"]].to_numpy(float),
        frames % 2 == 0,
    )
    report.update(
        {
            "split": "fit_on_even_frames_report_odd_frames",
            "min_keypoint_score": args.min_score,
            "keypoints_2d": {"path": str(args.keypoints_2d.resolve()), "sha256": sha256_file(args.keypoints_2d)},
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    raw = json.loads(source.read_text(encoding="utf-8"))
    cameras = []
    for item in raw["cameras"]:
        calibration = refined if item["camera_id"] == moving_id else bundle.calibrations[item["camera_id"]]
        cameras.append(calibration.to_json_dict())
    metadata = dict(raw.get("metadata", {}))
    metadata.update(
        {
            "calibration_mode": REFINED_CALIBRATION_MODE,
            "refined_from": {
                "path": str(source),
                "sha256": sha256_file(source),
                "calibration_mode": raw.get("metadata", {}).get("calibration_mode"),
            },
            "keypoint_refinement": report,
            "interpretation": (
                "One camera's orientation was re-estimated from this recording's own high-confidence keypoints "
                "(held-out frames reported). Internal two-view consistency only; not independent 3D accuracy."
            ),
        }
    )
    metadata.pop("supersedes", None)
    write_json_exclusive(output, {**{key: value for key, value in raw.items() if key not in {"metadata", "cameras"}}, "metadata": metadata, "cameras": cameras})
    before, after = report["held_out_before"], report["held_out_after"]
    print(f"refined camera: {moving_id}  rotation {report['rotation_about_camera_centre_deg']:.3f} deg")
    print(f"held-out signed epipolar median {before['signed_median_px']:.2f} -> {after['signed_median_px']:.2f} px; "
          f"|d| median {before['abs_median_px']:.2f} -> {after['abs_median_px']:.2f} px")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
