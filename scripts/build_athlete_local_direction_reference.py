from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.artifact_contracts import load_run_bound_main_3d_artifact  # noqa: E402
from src.artifact_io import sha256_file  # noqa: E402
from src.poomsae_scoring import (  # noqa: E402
    derive_athlete_local_direction_reference,
    load_movement_timeline,
    load_poomsae_spec,
    load_technical_accuracy_profile,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Derive the session-bound athlete-local direction reference from the opening "
            "ready stance. Diagnostic reference only; never a production calibration."
        )
    )
    parser.add_argument("--pose", required=True)
    parser.add_argument("--poomsae-spec", required=True)
    parser.add_argument("--timeline", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--anchor-movement", default="M01")
    parser.add_argument("--anchor-phase", default="preparation")
    parser.add_argument("--output-json", required=True)
    parser.add_argument(
        "--output-reference-json",
        help="Optional path for the bare reference consumed by the diagnostics stage.",
    )
    args = parser.parse_args()

    inputs = {
        "pose": _resolve(args.pose),
        "poomsae_spec": _resolve(args.poomsae_spec),
        "movement_timeline": _resolve(args.timeline),
        "profile": _resolve(args.profile),
    }
    for label, path in inputs.items():
        if not path.is_file():
            raise SystemExit(f"Input file is missing ({label}): {path}")
    output_json = _resolve(args.output_json)
    reference_json = _resolve(args.output_reference_json) if args.output_reference_json else None
    for target in (output_json, reference_json):
        if target is not None and target.exists():
            raise SystemExit(f"Output already exists; refusing to overwrite: {target}")

    pose, compatibility = load_run_bound_main_3d_artifact(inputs["pose"])
    spec = load_poomsae_spec(inputs["poomsae_spec"])
    timeline = load_movement_timeline(inputs["movement_timeline"], spec)
    profile = load_technical_accuracy_profile(inputs["profile"])

    envelope = derive_athlete_local_direction_reference(
        pose,
        spec,
        timeline,
        profile,
        anchor_movement_id=args.anchor_movement,
        anchor_phase=args.anchor_phase,
    )
    envelope["bindings"] = {
        label: {"path": str(path), "sha256": sha256_file(path)} for label, path in inputs.items()
    }
    envelope["bindings"]["pose"]["artifact_compatibility"] = compatibility.value

    _write_json(output_json, envelope)
    print(output_json)
    if reference_json is not None and envelope["status"] == "derived":
        _write_json(reference_json, envelope["reference"])
        print(reference_json)
    else:
        print(f"direction reference not derived: {envelope['reason']}")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


if __name__ == "__main__":
    main()
