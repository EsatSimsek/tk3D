from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from src.artifact_io import sha256_file  # noqa: E402

from src.poomsae_scoring import (  # noqa: E402
    build_movement_evidence,
    build_partial_engineering_trial,
    load_engineering_profile,
    load_movement_timeline,
    load_poomsae_spec,
    load_rule_pack,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a development-only partial Poomsae Accuracy trial with non-official engineering tolerances."
    )
    parser.add_argument("--pose", required=True)
    parser.add_argument("--rule-pack", required=True)
    parser.add_argument("--poomsae-spec", required=True)
    parser.add_argument("--timeline", required=True)
    parser.add_argument("--engineering-profile", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    paths = {
        "pose": _resolve(args.pose),
        "rule_pack": _resolve(args.rule_pack),
        "poomsae_spec": _resolve(args.poomsae_spec),
        "movement_timeline": _resolve(args.timeline),
        "engineering_profile": _resolve(args.engineering_profile),
    }
    for label, path in paths.items():
        if not path.is_file():
            raise SystemExit(f"Input file is missing ({label}): {path}")
    output_json = _resolve(args.output_json)
    output_csv = _resolve(args.output_csv)
    for target in (output_json, output_csv):
        if target.exists():
            raise SystemExit(f"Output already exists; refusing to overwrite: {target}")

    pose_payload = json.loads(paths["pose"].read_text(encoding="utf-8"))
    rule_pack = load_rule_pack(paths["rule_pack"])
    spec = load_poomsae_spec(paths["poomsae_spec"])
    timeline = load_movement_timeline(paths["movement_timeline"], spec)
    profile = load_engineering_profile(paths["engineering_profile"])
    evidence, anchor_rows = build_movement_evidence(pose_payload, spec, timeline)
    report = build_partial_engineering_trial(
        rule_pack,
        spec,
        timeline,
        profile,
        evidence,
        anchor_rows,
    )
    report["bindings"] = {
        label: {"path": str(path), "sha256": _sha256(path)} for label, path in paths.items()
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("x", encoding="utf-8", newline="") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    rows = report["measurements"]
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
