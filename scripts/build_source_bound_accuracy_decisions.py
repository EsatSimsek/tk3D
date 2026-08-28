from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from src.artifact_io import sha256_file  # noqa: E402

from src.poomsae_scoring import (  # noqa: E402
    build_source_bound_accuracy_decisions,
    load_movement_timeline,
    load_poomsae_spec,
    load_source_bound_accuracy_profile,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build uncertainty-gated, source-bound observed-scope Accuracy decisions."
    )
    parser.add_argument("--wholebody-diagnostics", required=True)
    parser.add_argument("--poomsae-spec", required=True)
    parser.add_argument("--timeline", required=True)
    parser.add_argument("--accuracy-profile", required=True)
    parser.add_argument("--observations")
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    paths = {
        "wholebody_diagnostics": _resolve(args.wholebody_diagnostics),
        "poomsae_spec": _resolve(args.poomsae_spec),
        "movement_timeline": _resolve(args.timeline),
        "accuracy_profile": _resolve(args.accuracy_profile),
    }
    if args.observations:
        paths["observations"] = _resolve(args.observations)
    for label, path in paths.items():
        if not path.is_file():
            raise SystemExit(f"Input file is missing ({label}): {path}")
    output = _resolve(args.output_json)
    if output.exists():
        raise SystemExit(f"Output already exists; refusing to overwrite: {output}")

    diagnostics = _read_json(paths["wholebody_diagnostics"])
    spec = load_poomsae_spec(paths["poomsae_spec"])
    timeline = load_movement_timeline(paths["movement_timeline"], spec)
    profile = load_source_bound_accuracy_profile(paths["accuracy_profile"])
    observations_payload = _read_json(paths["observations"]) if "observations" in paths else []
    if isinstance(observations_payload, dict):
        if observations_payload.get("status") != "categorical_diagnostics_only":
            raise SystemExit("Observation report status is unsupported.")
        if observations_payload.get("movement_timeline_id") != timeline["timeline_id"]:
            raise SystemExit("Observation report does not match the MovementTimeline.")
        observations = observations_payload.get("observations")
    else:
        observations = observations_payload
    if not isinstance(observations, list):
        raise SystemExit("Categorical observations must be a list.")
    report = build_source_bound_accuracy_decisions(
        diagnostics,
        spec,
        timeline,
        profile,
        observations,
    )
    report["bindings"] = {label: {"path": str(path), "sha256": _sha256(path)} for label, path in paths.items()}
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


_sha256 = sha256_file


if __name__ == "__main__":
    main()
