from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.artifact_contracts import load_run_bound_main_3d_artifact  # noqa: E402
from src.artifact_io import sha256_file  # noqa: E402
from src.poomsae_scoring import (  # noqa: E402
    build_technical_accuracy_diagnostics,
    load_movement_timeline,
    load_poomsae_spec,
    load_technical_accuracy_profile,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run comprehensive score-neutral Taegeuk 1 technical-accuracy diagnostics."
    )
    parser.add_argument("--pose", required=True)
    parser.add_argument("--poomsae-spec", required=True)
    parser.add_argument("--timeline", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--wholebody-diagnostics", required=True)
    parser.add_argument("--direction-reference")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--coverage-csv", required=True)
    parser.add_argument("--landmark-coverage-csv", required=True)
    args = parser.parse_args()

    inputs = {
        "pose": _resolve(args.pose),
        "poomsae_spec": _resolve(args.poomsae_spec),
        "movement_timeline": _resolve(args.timeline),
        "profile": _resolve(args.profile),
        "wholebody_diagnostics": _resolve(args.wholebody_diagnostics),
    }
    if args.direction_reference:
        inputs["direction_reference"] = _resolve(args.direction_reference)
    for label, path in inputs.items():
        if not path.is_file():
            raise SystemExit(f"Input file is missing ({label}): {path}")
    output_json = _resolve(args.output_json)
    coverage_csv = _resolve(args.coverage_csv)
    landmark_coverage_csv = _resolve(args.landmark_coverage_csv)
    for target in (output_json, coverage_csv, landmark_coverage_csv):
        if target.exists():
            raise SystemExit(f"Output already exists; refusing to overwrite: {target}")

    pose, compatibility = load_run_bound_main_3d_artifact(inputs["pose"])
    spec = load_poomsae_spec(inputs["poomsae_spec"])
    timeline = load_movement_timeline(inputs["movement_timeline"], spec)
    profile = load_technical_accuracy_profile(inputs["profile"])
    wholebody = _read_json(inputs["wholebody_diagnostics"])
    direction = _read_json(inputs["direction_reference"]) if "direction_reference" in inputs else None
    report = build_technical_accuracy_diagnostics(
        pose,
        spec,
        timeline,
        profile,
        wholebody,
        direction_reference=direction,
    )
    report["bindings"] = {
        label: {"path": str(path), "sha256": sha256_file(path)} for label, path in inputs.items()
    }
    report["bindings"]["pose"]["artifact_compatibility"] = compatibility.value

    output_json.parent.mkdir(parents=True, exist_ok=True)
    coverage_csv.parent.mkdir(parents=True, exist_ok=True)
    landmark_coverage_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("x", encoding="utf-8", newline="") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    rows = report["coverage_matrix"]
    with coverage_csv.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    landmark_rows = report["landmark_inventory"]
    with landmark_coverage_csv.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(landmark_rows[0]))
        writer.writeheader()
        writer.writerows(landmark_rows)
    print(output_json)
    print(coverage_csv)
    print(landmark_coverage_csv)


def _resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"JSON root must be an object: {path}")
    return payload


if __name__ == "__main__":
    main()
