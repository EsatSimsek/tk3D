from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.poomsae_scoring import (  # noqa: E402
    derive_categorical_observations,
    load_movement_timeline,
    load_poomsae_spec,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Derive the categorical observations a labelled MovementTimeline can prove on its own. "
            "Only pauses of at least the given length are emitted; wrong action, wrong stance, kihap "
            "and gaze events still require a human observation payload."
        )
    )
    parser.add_argument("--poomsae-spec", required=True)
    parser.add_argument("--timeline", required=True)
    parser.add_argument(
        "--pause-threshold-sec",
        type=float,
        default=3.0,
        help="Gap length that counts as a pause (WT Article 16-1.2.3 uses 3 seconds).",
    )
    parser.add_argument(
        "--minimum-confidence",
        type=float,
        default=0.80,
        help="Confidence floor for an emitted observation; the segment value is used when higher.",
    )
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    if args.pause_threshold_sec <= 0:
        raise SystemExit("--pause-threshold-sec must be positive")
    if not 0.0 < args.minimum_confidence <= 1.0:
        raise SystemExit("--minimum-confidence must be greater than 0 and at most 1")

    paths = {
        "poomsae_spec": _resolve(args.poomsae_spec),
        "movement_timeline": _resolve(args.timeline),
    }
    for label, path in paths.items():
        if not path.is_file():
            raise SystemExit(f"Input file is missing ({label}): {path}")
    output = _resolve(args.output_json)
    if output.exists():
        raise SystemExit(f"Output already exists; refusing to overwrite: {output}")

    spec = load_poomsae_spec(paths["poomsae_spec"])
    timeline = load_movement_timeline(paths["movement_timeline"], spec)
    observations = derive_categorical_observations(
        spec,
        timeline,
        pause_threshold_sec=args.pause_threshold_sec,
        minimum_confidence=args.minimum_confidence,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="") as stream:
        json.dump(observations, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")

    manifest = output.with_name(output.stem + "_manifest.json")
    if not manifest.exists():
        payload = {
            "schema_version": 1,
            "status": "automatic_categorical_observations",
            "derivation": "timeline_gap_duration_measurement_only",
            "pause_threshold_sec": args.pause_threshold_sec,
            "minimum_confidence": args.minimum_confidence,
            "observation_count": len(observations),
            "event_kinds": sorted({item["event_kind"] for item in observations}),
            "not_derived_event_kinds": [
                "wrong_action",
                "wrong_stance",
                "missing_kihap",
                "wrong_kihap_timing",
                "gaze_not_following_direction",
                "restart",
            ],
            "not_derived_reason": (
                "These events cannot be proven by timeline geometry alone and still require a human "
                "observation payload confirmed by manual_video_review."
            ),
            "bindings": {
                label: {"path": str(path), "sha256": _sha256(path)} for label, path in paths.items()
            },
        }
        with manifest.open("x", encoding="utf-8", newline="") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
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
