from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from src.artifact_io import sha256_file  # noqa: E402

from src.poomsae_scoring.error_video import (  # noqa: E402
    render_decision_evidence_video,
    write_render_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render synchronized cameras with source-bound error evidence.")
    parser.add_argument("--camera", action="append", required=True, metavar="CAMERA_ID=VIDEO")
    parser.add_argument("--keypoints-2d", required=True)
    parser.add_argument("--evidence-events", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--deduction-freeze-seconds",
        type=float,
        default=3.0,
        help="Hold each unique confirmed-deduction anchor for this many seconds (default: 3).",
    )
    args = parser.parse_args()

    cameras = dict(_parse_camera(value) for value in args.camera)
    if len(cameras) != len(args.camera):
        raise SystemExit("Camera ids must be unique.")
    events_path = _resolve(args.evidence_events)
    if not events_path.is_file():
        raise SystemExit(f"Evidence event report is missing: {events_path}")
    events = json.loads(events_path.read_text(encoding="utf-8"))
    report = render_decision_evidence_video(
        camera_videos=cameras,
        keypoints_2d_csv=_resolve(args.keypoints_2d),
        evidence_events=events,
        output_path=_resolve(args.output),
        deduction_freeze_seconds=args.deduction_freeze_seconds,
    )
    report["evidence_events"] = {"path": str(events_path), "sha256": _sha256(events_path)}
    manifest = write_render_manifest(_resolve(args.manifest), report)
    print(report["output_video"]["path"])
    print(manifest)


def _parse_camera(value: str) -> tuple[str, Path]:
    camera_id, separator, raw_path = value.partition("=")
    if not separator or not camera_id.strip() or not raw_path.strip():
        raise SystemExit("--camera must use CAMERA_ID=VIDEO format.")
    return camera_id.strip(), _resolve(raw_path.strip())


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


_sha256 = sha256_file


if __name__ == "__main__":
    main()
