from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.poomsae_scoring import build_review_html, load_movement_timeline, load_poomsae_spec  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a synchronized two-camera Poomsae review page.")
    parser.add_argument("--poomsae-spec", required=True)
    parser.add_argument("--timeline", required=True)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--readiness", required=True)
    parser.add_argument("--engineering-trial")
    parser.add_argument("--wholebody-diagnostics")
    parser.add_argument("--video-a", required=True)
    parser.add_argument("--video-a-label", default="Kamera A")
    parser.add_argument("--video-b", required=True)
    parser.add_argument("--video-b-label", default="Kamera B")
    parser.add_argument(
        "--video-extra",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Add another synchronized camera; may be repeated.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    inputs = {
        "poomsae_spec": _resolve(args.poomsae_spec),
        "movement_timeline": _resolve(args.timeline),
        "evidence_report": _resolve(args.evidence),
        "readiness_report": _resolve(args.readiness),
        "video_a": _resolve(args.video_a),
        "video_b": _resolve(args.video_b),
    }
    if args.engineering_trial:
        inputs["engineering_trial_report"] = _resolve(args.engineering_trial)
    if args.wholebody_diagnostics:
        inputs["wholebody_diagnostics_report"] = _resolve(args.wholebody_diagnostics)
    extra_videos: list[tuple[str, Path]] = []
    for index, value in enumerate(args.video_extra, start=1):
        label, path = _parse_extra_video(value)
        extra_videos.append((label, path))
        inputs[f"video_extra_{index:02d}"] = path
    for label, path in inputs.items():
        if not path.is_file():
            raise SystemExit(f"Input file is missing ({label}): {path}")
    output = _resolve(args.output)
    manifest = _resolve(args.manifest)
    for target in (output, manifest):
        if target.exists():
            raise SystemExit(f"Output already exists; refusing to overwrite: {target}")

    spec = load_poomsae_spec(inputs["poomsae_spec"])
    timeline = load_movement_timeline(inputs["movement_timeline"], spec)
    evidence = _read_json(inputs["evidence_report"])
    readiness = _read_json(inputs["readiness_report"])
    engineering_trial = (
        _read_json(inputs["engineering_trial_report"]) if "engineering_trial_report" in inputs else None
    )
    wholebody_diagnostics = (
        _read_json(inputs["wholebody_diagnostics_report"])
        if "wholebody_diagnostics_report" in inputs
        else None
    )
    videos = {
        args.video_a_label: _relative_url(inputs["video_a"], output.parent),
        args.video_b_label: _relative_url(inputs["video_b"], output.parent),
    }
    if len(videos) != 2:
        raise SystemExit("Video labels must be unique.")
    for label, path in extra_videos:
        if label in videos:
            raise SystemExit(f"Video label is repeated: {label}")
        videos[label] = _relative_url(path, output.parent)
    rendered = build_review_html(
        spec,
        timeline,
        evidence,
        readiness,
        videos,
        engineering_trial,
        wholebody_diagnostics,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="") as stream:
        stream.write(rendered)
    manifest_payload = {
        "schema_version": 1,
        "artifact_type": "poomsae_synchronized_review",
        "output": {"path": str(output), "sha256": _sha256(output)},
        "inputs": {label: {"path": str(path), "sha256": _sha256(path)} for label, path in inputs.items()},
        "scoring_status": evidence.get("scoring_status"),
        "accuracy_score": evidence.get("accuracy_score"),
        "partial_engineering_trial_score": (
            None if engineering_trial is None else engineering_trial.get("partial_engineering_trial_score")
        ),
        "wholebody_review_candidate_count": (
            None
            if wholebody_diagnostics is None
            else wholebody_diagnostics.get("summary", {}).get("review_candidate_count")
        ),
    }
    with manifest.open("x", encoding="utf-8", newline="") as stream:
        json.dump(manifest_payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(output)
    print(manifest)


def _read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"JSON root must be an object: {path}")
    return payload


def _relative_url(path: Path, parent: Path) -> str:
    return Path(os.path.relpath(path, parent)).as_posix()


def _parse_extra_video(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label.strip() or not raw_path.strip():
        raise SystemExit("--video-extra must use LABEL=PATH format.")
    return label.strip(), _resolve(raw_path.strip())


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
