from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from src.artifact_io import sha256_file  # noqa: E402

from src.artifact_contracts import load_run_bound_main_3d_artifact  # noqa: E402
from src.poomsae_scoring import (  # noqa: E402
    build_automatic_segmentation_diagnostics,
    load_movement_timeline,
    load_poomsae_spec,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build scoreless pose-driven movement and phase boundary diagnostics."
    )
    parser.add_argument("--pose", required=True)
    parser.add_argument("--poomsae-spec", required=True)
    parser.add_argument("--reference-timeline", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    pose_path = _resolve(args.pose)
    spec_path = _resolve(args.poomsae_spec)
    timeline_path = _resolve(args.reference_timeline)
    output_json = _resolve(args.output_json)
    output_csv = _resolve(args.output_csv)
    for path in (pose_path, spec_path, timeline_path):
        if not path.is_file():
            raise SystemExit(f"Input file is missing: {path}")
    for path in (output_json, output_csv):
        if path.exists():
            raise SystemExit(f"Output already exists; refusing to overwrite: {path}")

    pose_payload, pose_compatibility = load_run_bound_main_3d_artifact(pose_path)
    spec = load_poomsae_spec(spec_path)
    timeline = load_movement_timeline(timeline_path, spec)
    pose_sha256 = _sha256(pose_path)
    bound_sha256 = timeline.get("source_binding", {}).get("pose_file_sha256")
    if bound_sha256 and bound_sha256 != pose_sha256:
        raise SystemExit("Reference timeline pose SHA-256 does not match the supplied pose file.")

    report, rows = build_automatic_segmentation_diagnostics(pose_payload, spec, timeline)
    report["bindings"] = {
        "pose": {
            "path": str(pose_path),
            "sha256": pose_sha256,
            "artifact_compatibility": pose_compatibility.value,
        },
        "poomsae_spec": {"path": str(spec_path), "sha256": _sha256(spec_path)},
        "reference_timeline": {"path": str(timeline_path), "sha256": _sha256(timeline_path)},
    }
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


def _resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


_sha256 = sha256_file


if __name__ == "__main__":
    main()
