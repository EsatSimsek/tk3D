from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.poomsae_scoring import (  # noqa: E402
    load_movement_timeline,
    load_poomsae_spec,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract one reference pose per movement from a hand-labelled recording. "
            "The result is the template library that automatic timeline alignment matches "
            "detected segments against. It covers only the movements the source recording "
            "actually contains, and that limit is written into the output."
        )
    )
    parser.add_argument("--pose", required=True, help="vitpose_session_3d.json of the labelled run")
    parser.add_argument("--poomsae-spec", required=True)
    parser.add_argument("--timeline", required=True, help="hand-labelled MovementTimeline")
    parser.add_argument(
        "--window-radius-frames",
        type=int,
        default=5,
        help="Frames either side of the fixation anchor to average over (matches the diagnostics window).",
    )
    parser.add_argument(
        "--min-valid-samples",
        type=int,
        default=3,
        help="Minimum valid frames a joint needs in the window before it enters the template.",
    )
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    if args.window_radius_frames < 0:
        raise SystemExit("--window-radius-frames cannot be negative")
    if args.min_valid_samples < 1:
        raise SystemExit("--min-valid-samples must be at least 1")

    paths = {
        "pose": _resolve(args.pose),
        "poomsae_spec": _resolve(args.poomsae_spec),
        "movement_timeline": _resolve(args.timeline),
    }
    for label, path in paths.items():
        if not path.is_file():
            raise SystemExit(f"Input file is missing ({label}): {path}")
    output = _resolve(args.output_json)
    if output.exists():
        raise SystemExit(f"Output already exists; refusing to overwrite: {output}")

    pose = json.loads(paths["pose"].read_text(encoding="utf-8"))
    spec = load_poomsae_spec(paths["poomsae_spec"])
    timeline = load_movement_timeline(paths["movement_timeline"], spec)

    if timeline["label_source"] != "manual":
        raise SystemExit(
            "Templates must come from a hand-labelled timeline; deriving them from an "
            "automatic one would let the alignment validate itself."
        )

    keypoints = np.asarray(pose.get("keypoints_3d_world"), dtype=float)
    if keypoints.ndim != 3 or keypoints.shape[1:] != (133, 3):
        raise SystemExit("pose keypoints_3d_world must have shape [frames, 133, 3]")
    valid = pose.get("reliability_valid_mask")
    valid_mask = (
        np.asarray(valid, dtype=bool)
        if valid is not None
        else np.ones(keypoints.shape[:2], dtype=bool)
    )
    if valid_mask.shape != keypoints.shape[:2]:
        raise SystemExit("reliability_valid_mask shape does not match keypoints_3d_world")
    if keypoints.shape[0] != int(timeline["frame_count"]):
        raise SystemExit(
            f"pose has {keypoints.shape[0]} frames but the timeline declares "
            f"{timeline['frame_count']}; they do not describe the same recording"
        )

    templates: list[dict[str, object]] = []
    for segment in timeline["segments"]:
        anchor = segment["anchors"].get("fixation")
        if anchor is None:
            raise SystemExit(
                f"{segment['movement_id']} has no fixation anchor; the template is the "
                "finished posture, so a fixation anchor is required"
            )
        start = max(int(segment["start_frame"]), int(anchor) - args.window_radius_frames)
        end = min(int(segment["end_frame"]), int(anchor) + args.window_radius_frames)
        window = keypoints[start : end + 1]
        window_valid = valid_mask[start : end + 1]
        counts = window_valid.sum(axis=0)
        # A joint that was rarely seen must stay NaN rather than average a couple of
        # noisy samples; pose_distance already ignores NaN joints.
        usable = counts >= args.min_valid_samples
        masked = np.where(window_valid[:, :, None], window, np.nan)
        with warnings.catch_warnings():
            # A joint that was never valid in the window averages an all-NaN slice.
            # That is the intended outcome, not a problem: it is masked to NaN below.
            warnings.simplefilter("ignore", RuntimeWarning)
            mean_pose = np.nanmean(masked, axis=0)
        mean_pose[~usable] = np.nan
        templates.append(
            {
                "movement_id": segment["movement_id"],
                "sequence_index": segment["sequence_index"],
                "anchor_frame": int(anchor),
                "window": {"start_frame": start, "end_frame": end},
                "valid_joint_count": int(usable.sum()),
                "total_joint_count": int(usable.size),
                "label_status": segment["label_status"],
                "mean_pose": [
                    [None if not np.isfinite(value) else float(value) for value in joint]
                    for joint in mean_pose
                ],
            }
        )

    covered = [item["movement_id"] for item in templates]
    missing = [
        movement["movement_id"]
        for movement in spec["movements"]
        if movement["movement_id"] not in set(covered)
    ]
    payload = {
        "schema_version": 1,
        "status": "reference_pose_templates",
        "poomsae_id": spec["poomsae_id"],
        "poomsae_version": spec["version"],
        "derived_from": {
            "timeline_id": timeline["timeline_id"],
            "label_source": timeline["label_source"],
            "session_id": timeline["source_binding"]["session_id"],
            "run_id": timeline["source_binding"]["run_id"],
            "recording_scope": timeline["coverage"]["recording_scope"],
        },
        "extraction": {
            "anchor": "fixation",
            "window_radius_frames": args.window_radius_frames,
            "min_valid_samples": args.min_valid_samples,
            "reduction": "per_joint_nanmean_over_window",
            "missing_joint_representation": "null",
        },
        "coverage": {
            "covered_movement_ids": covered,
            "missing_movement_ids": missing,
            "template_count": len(templates),
            "expected_movement_count": len(spec["movements"]),
        },
        "limitations": [
            (
                "Single athlete, single recording: these templates describe how one person "
                "performed the form once, not a validated reference standard."
            ),
            (
                "Movements listed in missing_movement_ids have no template at all; automatic "
                "alignment cannot match them and will report them as unmatched."
            ),
            (
                "Templates are geometry for matching only. They carry no tolerance and can "
                "never by themselves justify a deduction."
            ),
        ],
        "bindings": {
            label: {"path": _binding_path(path), "sha256": _sha256(path)}
            for label, path in paths.items()
        },
        "binding_note": (
            "The SHA-256 is the authoritative binding; the path is a convenience and is "
            "reduced to a file name when the input lives outside the repository."
        ),
        "templates": templates,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(output)
    print(f"templates: {len(templates)}/{len(spec['movements'])} movements covered")
    if missing:
        print(f"no template for: {', '.join(missing)}")


def _resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _binding_path(path: Path) -> str:
    """Repo-relative when the input is in the repository, otherwise just the file name.

    This file is meant to be committed, and the pose artifact normally lives outside
    the working tree, so writing its absolute path would publish somebody's home
    directory. The hash below identifies the input either way.
    """
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
