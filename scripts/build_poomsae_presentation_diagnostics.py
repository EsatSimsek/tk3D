from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.poomsae_scoring import (  # noqa: E402
    build_presentation_diagnostics,
    load_movement_timeline,
    load_poomsae_spec,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate WholeBody measurements and timeline tempo into a presentation diagnostic "
            "panel. This never produces a WT Presentation score: the report carries total_score=null "
            "and score_claim_allowed=false until judge calibration exists."
        )
    )
    parser.add_argument("--wholebody-diagnostics", required=True)
    parser.add_argument("--poomsae-spec", required=True)
    parser.add_argument("--timeline", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    paths = {
        "wholebody_diagnostics": _resolve(args.wholebody_diagnostics),
        "poomsae_spec": _resolve(args.poomsae_spec),
        "movement_timeline": _resolve(args.timeline),
    }
    for label, path in paths.items():
        if not path.is_file():
            raise SystemExit(f"Input file is missing ({label}): {path}")
    output = _resolve(args.output_json)
    if output.exists():
        raise SystemExit(f"Output already exists; refusing to overwrite: {output}")

    diagnostics = _read_json(paths["wholebody_diagnostics"])
    spec = load_poomsae_spec(paths["poomsae_spec"])
    timeline = load_movement_timeline(paths["movement_timeline"], spec)
    report = build_presentation_diagnostics(diagnostics, spec, timeline)

    # Fail closed rather than writing a report that silently claims a score.
    if report.get("total_score") is not None or report["safety_contract"]["score_claim_allowed"]:
        raise SystemExit("Presentation report violated its no-score contract; refusing to write it.")

    report["bindings"] = {
        label: {"path": str(path), "sha256": _sha256(path)} for label, path in paths.items()
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(output)


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


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
