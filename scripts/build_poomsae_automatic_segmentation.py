from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from urllib.parse import quote

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.artifact_io import sha256_file  # noqa: E402

from src.artifact_contracts import load_run_bound_main_3d_artifact  # noqa: E402
from src.poomsae_scoring import (  # noqa: E402
    build_automatic_segmentation_diagnostics,
    load_movement_timeline,
    load_poomsae_spec,
)
from src.poomsae_scoring.fixation_review import build_fixation_review_html  # noqa: E402
from src.poomsae_scoring.fixation import build_fixation_review_binding  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build scoreless pose-driven movement and phase boundary diagnostics."
    )
    parser.add_argument("--pose", required=True)
    parser.add_argument("--poomsae-spec", required=True)
    parser.add_argument("--reference-timeline", help="Optional comparison labels; never used to tune fixation times.")
    parser.add_argument("--output-html", help="Optional standalone fixation review, also works without labels.")
    parser.add_argument("--camera", action="append", default=[], metavar="NAME=VIDEO", help="Browser-compatible MP4/WebM for the review; repeat per camera.")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    pose_path = _resolve(args.pose)
    spec_path = _resolve(args.poomsae_spec)
    timeline_path = _resolve(args.reference_timeline) if args.reference_timeline else None
    output_json = _resolve(args.output_json)
    output_csv = _resolve(args.output_csv)
    output_html = _resolve(args.output_html) if args.output_html else None
    videos = {}
    for camera in args.camera:
        if "=" not in camera:
            raise SystemExit("--camera must be NAME=VIDEO")
        name, path = camera.split("=", 1)
        if not name.strip() or name in videos:
            raise SystemExit("camera names must be nonempty and unique")
        videos[name] = _resolve(path)
    for path in (pose_path, spec_path, *([timeline_path] if timeline_path else []), *videos.values()):
        if not path.is_file():
            raise SystemExit(f"Input file is missing: {path}")
    outputs = [output_json, output_csv, *([output_html] if output_html else [])]
    if len(set(outputs)) != len(outputs):
        raise SystemExit("Output paths must be distinct")
    for path in outputs:
        if path.exists():
            raise SystemExit(f"Output already exists; refusing to overwrite: {path}")

    pose_payload, pose_compatibility = load_run_bound_main_3d_artifact(pose_path)
    spec = load_poomsae_spec(spec_path)
    timeline = load_movement_timeline(timeline_path, spec) if timeline_path else None
    pose_sha256 = _sha256(pose_path)
    bound_sha256 = timeline.get("source_binding", {}).get("pose_file_sha256") if timeline else None
    if bound_sha256 and bound_sha256 != pose_sha256:
        raise SystemExit("Reference timeline pose SHA-256 does not match the supplied pose file.")

    report, rows = build_automatic_segmentation_diagnostics(pose_payload, spec, timeline)
    for name, path in videos.items():
        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened() or int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT))) != report["frame_count"] or not np.isclose(capture.get(cv2.CAP_PROP_FPS), report["fps"], rtol=1e-3):
                raise SystemExit(f"Camera {name} must have the same frame count and FPS as the pose.")
        finally:
            capture.release()
    report["bindings"] = {
        "pose": {
            "path": str(pose_path),
            "sha256": pose_sha256,
            "artifact_compatibility": pose_compatibility.value,
        },
        "poomsae_spec": {"path": str(spec_path), "sha256": _sha256(spec_path)},
        "reference_timeline": {"path": str(timeline_path), "sha256": _sha256(timeline_path)} if timeline_path else None,
        "videos": {name: {"path": str(path), "sha256": _sha256(path)} for name, path in videos.items()},
    }
    report["review_binding"] = build_fixation_review_binding(report)
    page = None
    if output_html is not None:
        relative_videos = {name: quote(Path(os.path.relpath(path, output_html.parent)).as_posix(), safe="/") for name, path in videos.items()}
        page = build_fixation_review_html(report, relative_videos, report["review_binding"])
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("x", encoding="utf-8", newline="") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    with output_csv.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    print(output_json)
    print(output_csv)
    if output_html is not None:
        output_html.parent.mkdir(parents=True, exist_ok=True)
        with output_html.open("x", encoding="utf-8") as stream:
            stream.write(page)
        print(output_html)


def _resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


_sha256 = sha256_file


if __name__ == "__main__":
    main()
