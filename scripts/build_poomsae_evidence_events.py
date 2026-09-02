from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.artifact_io import sha256_file  # noqa: E402

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
    parser.add_argument("--wholebody-diagnostics")
    parser.add_argument("--technical-accuracy-diagnostics")
    parser.add_argument(
        "--alignment-anomalies",
        help=(
            "Alignment anomaly report from the automatic timeline draft. Only meaningful "
            "for an automatically labelled timeline; a hand-labelled one has no alignment step."
        ),
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    paths = {
        "accuracy_decisions": _resolve(args.accuracy_decisions),
        "poomsae_spec": _resolve(args.poomsae_spec),
        "movement_timeline": _resolve(args.timeline),
    }
    if args.wholebody_diagnostics:
        paths["wholebody_diagnostics"] = _resolve(args.wholebody_diagnostics)
    if args.technical_accuracy_diagnostics:
        paths["technical_accuracy_diagnostics"] = _resolve(args.technical_accuracy_diagnostics)
    if args.alignment_anomalies:
        paths["alignment_anomalies"] = _resolve(args.alignment_anomalies)
    for label, path in paths.items():
        if not path.is_file():
            raise SystemExit(f"Input file is missing ({label}): {path}")
    output = _resolve(args.output)
    if output.exists():
        raise SystemExit(f"Output already exists; refusing to overwrite: {output}")

    decisions = json.loads(paths["accuracy_decisions"].read_text(encoding="utf-8"))
    spec = load_poomsae_spec(paths["poomsae_spec"])
    timeline = load_movement_timeline(paths["movement_timeline"], spec)
    diagnostics = None
    if "wholebody_diagnostics" in paths:
        diagnostics = json.loads(paths["wholebody_diagnostics"].read_text(encoding="utf-8"))
    technical_accuracy = None
    if "technical_accuracy_diagnostics" in paths:
        technical_accuracy = json.loads(
            paths["technical_accuracy_diagnostics"].read_text(encoding="utf-8")
        )
    alignment_anomalies = None
    if "alignment_anomalies" in paths:
        payload = json.loads(paths["alignment_anomalies"].read_text(encoding="utf-8"))
        if payload.get("timeline_id") != timeline["timeline_id"]:
            raise SystemExit("alignment anomalies were produced for a different timeline")
        alignment_anomalies = payload.get("alignment_anomalies")
        if not isinstance(alignment_anomalies, list):
            raise SystemExit("alignment anomalies report has no alignment_anomalies list")
    report = build_decision_evidence_events(
        decisions,
        spec,
        timeline,
        diagnostics,
        alignment_anomalies=alignment_anomalies,
        technical_accuracy_diagnostics=technical_accuracy,
    )
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


_sha256 = sha256_file


if __name__ == "__main__":
    main()
