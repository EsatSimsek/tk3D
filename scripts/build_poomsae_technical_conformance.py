from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from src.poomsae_scoring import (  # noqa: E402
    build_technical_conformance,
    load_movement_timeline,
    load_poomsae_spec,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build movement-level technical conformance diagnostics without scoring."
    )
    parser.add_argument("--wholebody-diagnostics", required=True)
    parser.add_argument("--categorical-diagnostics", required=True)
    parser.add_argument("--poomsae-spec", required=True)
    parser.add_argument("--timeline", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    spec = load_poomsae_spec(_resolve(args.poomsae_spec))
    timeline = load_movement_timeline(_resolve(args.timeline), spec)
    report = build_technical_conformance(
        _read_json(_resolve(args.wholebody_diagnostics)),
        _read_json(_resolve(args.categorical_diagnostics)),
        spec,
        timeline,
    )
    output = _resolve(args.output)
    if output.exists():
        raise SystemExit(f"Output already exists; refusing to overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(output)


def _read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"JSON root must be an object: {path}")
    return payload


def _resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


if __name__ == "__main__":
    main()
