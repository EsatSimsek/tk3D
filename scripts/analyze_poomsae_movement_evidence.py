from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.poomsae_scoring import (  # noqa: E402
    build_movement_evidence,
    load_movement_timeline,
    load_poomsae_spec,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create no-deduction movement/phase measurement evidence.")
    parser.add_argument("--pose", required=True)
    parser.add_argument("--poomsae-spec", required=True)
    parser.add_argument("--timeline", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    pose_path = _resolve(args.pose)
    spec_path = _resolve(args.poomsae_spec)
    timeline_path = _resolve(args.timeline)
    output_json = _resolve(args.output_json)
    output_csv = _resolve(args.output_csv)
    for target in (output_json, output_csv):
        if target.exists():
            raise SystemExit(f"Output already exists; refusing to overwrite: {target}")

    pose_payload = json.loads(pose_path.read_text(encoding="utf-8"))
    spec = load_poomsae_spec(spec_path)
    timeline = load_movement_timeline(timeline_path, spec)
    report, rows = build_movement_evidence(pose_payload, spec, timeline)
    report["bindings"] = {
        "pose": {"path": str(pose_path), "sha256": _sha256(pose_path)},
        "poomsae_spec": {"path": str(spec_path), "sha256": _sha256(spec_path)},
        "movement_timeline": {"path": str(timeline_path), "sha256": _sha256(timeline_path)},
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
