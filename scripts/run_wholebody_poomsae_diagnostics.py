from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from src.artifact_io import sha256_file  # noqa: E402

from src.artifact_contracts import load_run_bound_main_3d_artifact  # noqa: E402
from src.poomsae_scoring import (  # noqa: E402
    build_wholebody_diagnostics,
    load_movement_timeline,
    load_poomsae_spec,
    load_wholebody_diagnostic_profile,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run no-score COCO-WholeBody-133 Poomsae diagnostics.")
    parser.add_argument("--pose", required=True)
    parser.add_argument("--poomsae-spec", required=True)
    parser.add_argument("--timeline", required=True)
    parser.add_argument("--diagnostic-profile", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    paths = {
        "pose": _resolve(args.pose),
        "poomsae_spec": _resolve(args.poomsae_spec),
        "movement_timeline": _resolve(args.timeline),
        "diagnostic_profile": _resolve(args.diagnostic_profile),
    }
    for label, path in paths.items():
        if not path.is_file():
            raise SystemExit(f"Input file is missing ({label}): {path}")
    output_json, output_csv = _resolve(args.output_json), _resolve(args.output_csv)
    for target in (output_json, output_csv):
        if target.exists():
            raise SystemExit(f"Output already exists; refusing to overwrite: {target}")

    pose, pose_compatibility = load_run_bound_main_3d_artifact(paths["pose"])
    spec = load_poomsae_spec(paths["poomsae_spec"])
    timeline = load_movement_timeline(paths["movement_timeline"], spec)
    profile = load_wholebody_diagnostic_profile(paths["diagnostic_profile"])
    report = build_wholebody_diagnostics(pose, spec, timeline, profile)
    report["bindings"] = {
        label: {"path": str(path), "sha256": _sha256(path)} for label, path in paths.items()
    }
    report["bindings"]["pose"]["artifact_compatibility"] = pose_compatibility.value
    rows = [metric for movement in report["movements"] for metric in movement["metrics"]]

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("x", encoding="utf-8", newline="") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    with output_csv.open("x", encoding="utf-8-sig", newline="") as stream:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
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
