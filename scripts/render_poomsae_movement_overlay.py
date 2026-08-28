from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from src.artifact_io import sha256_file  # noqa: E402

from src.poomsae_scoring import (  # noqa: E402
    load_movement_timeline,
    load_poomsae_spec,
    render_movement_overlay,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render source-bound Poomsae movement and phase labels on a video.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--poomsae-spec", required=True)
    parser.add_argument("--timeline", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    video_path = _resolve(args.video)
    spec_path = _resolve(args.poomsae_spec)
    timeline_path = _resolve(args.timeline)
    output_path = _resolve(args.output)
    manifest_path = _resolve(args.manifest)
    if manifest_path.exists():
        raise SystemExit(f"Manifest already exists; refusing to overwrite: {manifest_path}")

    spec = load_poomsae_spec(spec_path)
    timeline = load_movement_timeline(timeline_path, spec)
    report = render_movement_overlay(video_path, output_path, spec, timeline)
    report["bindings"] = {
        "source_video_sha256": _sha256(video_path),
        "poomsae_spec": {"path": str(spec_path), "sha256": _sha256(spec_path)},
        "movement_timeline": {"path": str(timeline_path), "sha256": _sha256(timeline_path)},
        "output_video_sha256": _sha256(output_path),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with manifest_path.open("x", encoding="utf-8", newline="") as stream:
        stream.write(encoded)
    print(manifest_path)


def _resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


_sha256 = sha256_file


if __name__ == "__main__":
    main()
