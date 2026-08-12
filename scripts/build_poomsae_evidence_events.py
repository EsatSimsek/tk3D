from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.poomsae_scoring import (  # noqa: E402
    build_decision_evidence_events,
    load_movement_timeline,
    load_poomsae_spec,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build camera-renderable evidence events from Accuracy decisions.")
    parser.add_argument("--accuracy-decisions", required=True)
    parser.add_argument("--poomsae-spec", required=True)
    parser.add_argument("--timeline", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    paths = {
        "accuracy_decisions": _resolve(args.accuracy_decisions),
        "poomsae_spec": _resolve(args.poomsae_spec),
        "movement_timeline": _resolve(args.timeline),
    }
    for label, path in paths.items():
        if not path.is_file():
            raise SystemExit(f"Input file is missing ({label}): {path}")
    output = _resolve(args.output)
    if output.exists():
        raise SystemExit(f"Output already exists; refusing to overwrite: {output}")

    decisions = json.loads(paths["accuracy_decisions"].read_text(encoding="utf-8"))
    spec = load_poomsae_spec(paths["poomsae_spec"])
    timeline = load_movement_timeline(paths["movement_timeline"], spec)
    report = build_decision_evidence_events(decisions, spec, timeline)
    report["bindings"] = {
        label: {"path": str(path), "sha256": _sha256(path)} for label, path in paths.items()
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(output)


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
