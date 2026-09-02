from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.artifact_contracts import load_run_bound_main_3d_artifact  # noqa: E402
from src.artifact_io import sha256_file  # noqa: E402
from src.poomsae_scoring import (  # noqa: E402
    ScoringContractError,
    build_automatic_timeline_report,
    detect_automatic_segments,
    load_poomsae_spec,
)

TEMPLATE_STATUS = "reference_pose_templates"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Propose a movement timeline for an unlabelled recording by matching detected "
            "segments against the reference pose templates. The result is a draft for a "
            "person to correct; it is never fed straight into scoring."
        )
    )
    parser.add_argument("--pose", required=True)
    parser.add_argument("--poomsae-spec", required=True)
    parser.add_argument("--templates", required=True, help="reference pose template library")
    parser.add_argument("--timeline-id", required=True)
    parser.add_argument(
        "--expect-movements",
        default="all",
        help=(
            "Which movements the recording is expected to contain: 'all' for the whole "
            "form, or a comma-separated prefix such as M01,M02,M03."
        ),
    )
    parser.add_argument(
        "--window-radius-frames",
        type=int,
        default=5,
        help="Frames either side of the fixation anchor averaged into the segment pose.",
    )
    parser.add_argument(
        "--min-valid-samples",
        type=int,
        default=3,
        help="Minimum valid frames a joint needs in the window before it enters the pose.",
    )
    parser.add_argument("--output-timeline", required=True, help="draft MovementTimeline YAML")
    parser.add_argument("--output-anomalies", required=True, help="alignment anomalies JSON")
    args = parser.parse_args()

    if args.window_radius_frames < 0:
        raise SystemExit("--window-radius-frames cannot be negative")
    if args.min_valid_samples < 1:
        raise SystemExit("--min-valid-samples must be at least 1")

    inputs = {
        "pose": _resolve(args.pose),
        "poomsae_spec": _resolve(args.poomsae_spec),
        "templates": _resolve(args.templates),
    }
    for label, path in inputs.items():
        if not path.is_file():
            raise SystemExit(f"Input file is missing ({label}): {path}")
    outputs = (_resolve(args.output_timeline), _resolve(args.output_anomalies))
    for target in outputs:
        if target.exists():
            raise SystemExit(f"Output already exists; refusing to overwrite: {target}")

    pose, compatibility = load_run_bound_main_3d_artifact(inputs["pose"])
    spec = load_poomsae_spec(inputs["poomsae_spec"])
    templates = _load_templates(inputs["templates"], spec)

    keypoints = np.asarray(pose.get("keypoints_3d_world"), dtype=float)
    if keypoints.ndim != 3 or keypoints.shape[1:] != (133, 3):
        raise SystemExit("pose keypoints_3d_world must have shape [frames, 133, 3]")
    frame_count = int(keypoints.shape[0])
    valid = pose.get("reliability_valid_mask")
    valid_mask = (
        np.asarray(valid, dtype=bool) if valid is not None else np.ones(keypoints.shape[:2], dtype=bool)
    )
    if valid_mask.shape != keypoints.shape[:2]:
        raise SystemExit("reliability_valid_mask shape does not match keypoints_3d_world")
    fps = float(pose.get("sample_fps") or 0.0)
    if not np.isfinite(fps) or fps <= 0:
        raise SystemExit("pose must carry a positive sample_fps")

    movements = _expected_movements(spec, args.expect_movements)
    detection = detect_automatic_segments(keypoints, fps=fps, movements=movements)
    detected = detection["segments"]
    if not detected:
        raise SystemExit("the detector found no movement episode in this recording")

    segments = [
        {
            "segment_id": index,
            "start_frame": int(segment["start_frame"]),
            "end_frame": int(segment["end_frame"]),
            "anchors": {name: int(frame) for name, frame in segment["anchors"].items()},
            "mean_pose": _window_mean_pose(
                keypoints,
                valid_mask,
                anchor=int(segment["anchors"]["fixation"]),
                start=int(segment["start_frame"]),
                end=int(segment["end_frame"]),
                radius=args.window_radius_frames,
                min_valid_samples=args.min_valid_samples,
            ),
        }
        for index, segment in enumerate(detected)
    ]

    try:
        report = build_automatic_timeline_report(
            segments,
            _expected_poses(spec, templates),
            spec,
            frame_count=frame_count,
            fps=fps,
            source_binding={
                "session_id": pose.get("session_id"),
                "run_id": pose.get("run_id"),
                "pose_file": inputs["pose"].name,
                "pose_file_sha256": sha256_file(inputs["pose"]),
            },
            timeline_id=args.timeline_id,
        )
    except ScoringContractError as error:
        if "observed prefix" not in str(error):
            raise
        raise SystemExit(
            "The matched movements do not start at the beginning of the form, and the "
            "timeline format cannot describe that: it can only mark a run of movements "
            "from the first one onwards as observed.\n"
            f"The detector proposed {len(segments)} segment(s) for "
            f"{len(movements)} expected movement(s). Either an opening movement was "
            "missed, or the recording does not begin at the start of the form.\n"
            "Check that the recording starts at the first movement. If it does and a "
            "movement was still missed, this recording needs a hand-written timeline."
        ) from error
    timeline = report["timeline"]
    anomalies = report["alignment_anomalies"]

    outputs[0].parent.mkdir(parents=True, exist_ok=True)
    outputs[0].write_text(
        yaml.safe_dump(_yaml_safe(timeline), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    outputs[1].parent.mkdir(parents=True, exist_ok=True)
    with outputs[1].open("x", encoding="utf-8", newline="") as stream:
        json.dump(
            {
                "schema_version": 1,
                "status": "automatic_timeline_alignment_anomalies",
                "scoring_status": "review_candidates_only_no_deduction",
                "timeline_id": timeline["timeline_id"],
                "detector": {
                    "selection": detection["selection"],
                    "thresholds": detection["thresholds"],
                    "valid_signal_ratio": detection["valid_signal_ratio"],
                },
                "expected_movement_ids": [movement["movement_id"] for movement in movements],
                "detected_segment_count": len(segments),
                "template_coverage": templates["coverage"],
                "alignment_anomalies": anomalies,
                "interpretation": (
                    "Every entry says the segment-to-movement match was not clean, never "
                    "that the athlete made a mistake. This draft requires human review "
                    "before any measurement is run against it."
                ),
                "bindings": {
                    label: {"path": _binding_path(path), "sha256": sha256_file(path)}
                    for label, path in inputs.items()
                },
            },
            stream,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        stream.write("\n")
    for target in outputs:
        print(target)
    print(f"pose artifact compatibility: {compatibility.value}")

    matched = timeline["coverage"]["observed_movement_ids"]
    missing = timeline["coverage"]["missing_movement_ids"]
    uncertain = [
        segment["movement_id"] for segment in timeline["segments"] if segment["label_status"] == "ambiguous"
    ]
    print(f"matched {len(matched)}/{len(movements)}: {', '.join(matched) if matched else '-'}")
    if missing:
        print(f"not matched: {', '.join(missing)}")
    print(f"low confidence, check these first: {', '.join(uncertain) if uncertain else '-'}")
    print(f"alignment anomalies: {len(anomalies)}")
    print("This timeline is a proposal. Review and correct it before scoring uses it.")


def _load_templates(path: Path, spec: dict) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("status") != TEMPLATE_STATUS:
        raise SystemExit(f"not a reference pose template library: {path}")
    if payload.get("poomsae_id") != spec["poomsae_id"]:
        raise SystemExit("template library was built for a different Poomsae")
    if payload.get("derived_from", {}).get("label_source") != "manual":
        raise SystemExit(
            "templates must come from a hand-labelled recording; matching against "
            "automatically derived templates would let the alignment confirm itself"
        )
    return payload


def _expected_movements(spec: dict, requested: str) -> list[dict]:
    movements = spec["movements"]
    if requested.strip().lower() == "all":
        return movements
    wanted = [item.strip() for item in requested.split(",") if item.strip()]
    known = {movement["movement_id"]: movement for movement in movements}
    unknown = [item for item in wanted if item not in known]
    if unknown:
        raise SystemExit(f"unknown movement ids: {', '.join(unknown)}")
    expected_prefix = [movement["movement_id"] for movement in movements[: len(wanted)]]
    if wanted != expected_prefix:
        raise SystemExit(
            "--expect-movements must be a prefix of the form in spec order, "
            f"for example {','.join(expected_prefix)}"
        )
    return [known[item] for item in wanted]


def _expected_poses(spec: dict, templates: dict) -> list[np.ndarray]:
    """One reference pose per spec movement; movements without a template stay all-NaN.

    An all-NaN pose has no joint in common with any segment, so the matcher scores it
    as infinitely far away and reports the movement as unmatched instead of guessing.
    """
    by_movement = {item["movement_id"]: item for item in templates["templates"]}
    poses: list[np.ndarray] = []
    for movement in spec["movements"]:
        template = by_movement.get(movement["movement_id"])
        if template is None:
            poses.append(np.full((133, 3), np.nan))
            continue
        pose = np.asarray(
            [[np.nan if value is None else float(value) for value in joint] for joint in template["mean_pose"]],
            dtype=float,
        )
        if pose.shape != (133, 3):
            raise SystemExit(f"template {movement['movement_id']} is not shaped [133, 3]")
        poses.append(pose)
    return poses


def _window_mean_pose(
    keypoints: np.ndarray,
    valid_mask: np.ndarray,
    *,
    anchor: int,
    start: int,
    end: int,
    radius: int,
    min_valid_samples: int,
) -> np.ndarray:
    """Average the held posture around the fixation anchor, the same way templates are built."""
    low = max(start, anchor - radius)
    high = min(end, anchor + radius)
    window = keypoints[low : high + 1]
    window_valid = valid_mask[low : high + 1]
    usable = window_valid.sum(axis=0) >= min_valid_samples
    masked = np.where(window_valid[:, :, None], window, np.nan)
    with warnings.catch_warnings():
        # A joint never seen in the window averages an all-NaN slice; that is the
        # intended outcome and it is masked back to NaN immediately below.
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_pose = np.nanmean(masked, axis=0)
    mean_pose[~usable] = np.nan
    return mean_pose


def _yaml_safe(value):
    if isinstance(value, dict):
        return {key: _yaml_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_yaml_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _binding_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


def _resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


if __name__ == "__main__":
    main()
