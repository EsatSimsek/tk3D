from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from src.artifact_io import sha256_file  # noqa: E402

from src.poomsae_scoring import (  # noqa: E402
    build_categorical_diagnostics,
    load_movement_timeline,
    load_poomsae_spec,
    load_wholebody_diagnostic_profile,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build non-scoring pause, wrong-action and wrong-stance diagnostics."
    )
    parser.add_argument("--wholebody-diagnostics", required=True)
    parser.add_argument("--poomsae-spec", required=True)
    parser.add_argument("--timeline", required=True)
    parser.add_argument("--diagnostic-profile", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    paths = {
        "wholebody_diagnostics": _resolve(args.wholebody_diagnostics),
        "poomsae_spec": _resolve(args.poomsae_spec),
        "movement_timeline": _resolve(args.timeline),
        "diagnostic_profile": _resolve(args.diagnostic_profile),
    }
    for label, path in paths.items():
        if not path.is_file():
            raise SystemExit(f"Input file is missing ({label}): {path}")
    output = _resolve(args.output)
    if output.exists():
        raise SystemExit(f"Output already exists; refusing to overwrite: {output}")

    diagnostics = _read_json(paths["wholebody_diagnostics"])
    spec = load_poomsae_spec(paths["poomsae_spec"])
    timeline = load_movement_timeline(paths["movement_timeline"], spec)
    profile = load_wholebody_diagnostic_profile(paths["diagnostic_profile"])
    report = build_categorical_diagnostics(diagnostics, spec, timeline, profile)
    report["bindings"] = {
        label: {"path": str(path), "sha256": _sha256(path)}
        for label, path in paths.items()
    }
    _write_json(output, report)
    print(output)


def _read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"JSON root must be an object: {path}")
    return payload


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


_sha256 = sha256_file


if __name__ == "__main__":
    main()
