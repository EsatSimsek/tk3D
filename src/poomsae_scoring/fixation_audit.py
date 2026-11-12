"""Explain source rejection and multicomponent hold failures without changing detection."""
from __future__ import annotations

from collections import Counter
import csv
from pathlib import Path

import numpy as np

from .contracts import ScoringContractError
from .fixation import FixationParameters, prepare_fixation_signals

DEPENDENCIES = {
    "left_hand": (9, 11, 12), "right_hand": (10, 11, 12), "feet": (15, 16),
    "pelvis": (11, 12), "torso": (5, 6), "shoulder_rotation": (5, 6), "pelvis_rotation": (11, 12),
}
JOINT_NAMES = {5: "left_shoulder", 6: "right_shoulder", 9: "left_wrist", 10: "right_wrist",
               11: "left_hip", 12: "right_hip", 15: "left_ankle", 16: "right_ankle"}


def _ranges(mask: np.ndarray, offset=0) -> list[list[int]]:
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return [[int(a + offset), int(b + offset - 1)] for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True)]


def _longest_duration(mask: np.ndarray, times: np.ndarray) -> float:
    runs = _ranges(mask)
    return max((float(times[end] - times[start]) for start, end in runs), default=0.0)


def _distribution(values: np.ndarray) -> dict:
    finite = values[np.isfinite(values)]
    return {"count": int(finite.size), "median": float(np.median(finite)) if finite.size else None,
            "p95": float(np.percentile(finite, 95)) if finite.size else None}


def build_fixation_audit(pose: dict, report: dict) -> dict:
    points = np.asarray(pose["keypoints_3d_world"], dtype=float)
    if points.shape != (report["frame_count"], 133, 3) or pose["sample_fps"] != report["fps"]:
        raise ScoringContractError("audit pose time base/shape mismatch")
    parameters = FixationParameters(**report["detector"]["fixation"]["parameters"])
    signals = prepare_fixation_signals(points, report["fps"], quality=pose, parameters=parameters)
    masks = {"nonfinite_position": ~np.all(np.isfinite(points), axis=-1)}
    for key, label in (("reliability_valid_mask", "upstream_reliability_rejected"),
                       ("used_cameras", "insufficient_cameras"), ("reprojection_error", "reprojection_rejected")):
        if pose.get(key) is None:
            masks["missing_" + key] = np.ones(points.shape[:2], dtype=bool)
            continue
        values = np.asarray(pose[key], dtype=float)
        if key == "reliability_valid_mask":
            masks[label] = values != 1
        elif key == "used_cameras":
            masks[label] = ~np.isfinite(values) | (values < parameters.minimum_cameras)
        else:
            masks[label] = ~np.isfinite(values) | (values < 0) | (values > parameters.maximum_reprojection_px)
    reasons = pose.get("reliability_rejection_reasons")
    reasons = np.asarray(reasons, dtype=int) if reasons is not None else None
    if reasons is not None and reasons.shape != points.shape[:2]:
        raise ScoringContractError("audit rejection reasons shape mismatch")
    rejected = np.any(np.stack(list(masks.values())), axis=0)
    result = {
        "schema_version": 1, "status": "diagnostic_not_accuracy_validation",
        "frame_count": len(points), "fps": report["fps"], "blocking_reason": signals["blocking_reason"],
        "minimum_hold_sec": parameters.minimum_hold_sec,
        "body17_valid_ratio": float((~rejected[:, :17]).mean()),
        "all_components_valid_ratio": signals["all_components_valid_ratio"],
        "reprojection_px": _distribution(np.asarray(pose.get("reprojection_error", []), float)[:, :17] if pose.get("reprojection_error") is not None else np.array([])),
        "camera_count": dict(sorted(Counter(float(value) for value in np.asarray(pose.get("used_cameras", []), float)[:, :17].ravel() if np.isfinite(value)).items())) if pose.get("used_cameras") is not None else {},
        "source_stability": pose.get("pose3d_stability"),
        "interpretation": "A rejected upstream reliability flag remains rejected even when camera count and reprojection pass. Derivative support includes adjacent positions and the smoothing window. Low motion is not evidence of correct technique.",
        "candidate_count": len(report["fixation_candidates"]), "candidates": [],
    }
    for event in report["fixation_candidates"]:
        start = event["completion_candidate_frame"]
        start = event["peak_frame"] if start is None else start
        end = event["end_frame"]
        times = signals["times"][start:end + 1]
        channel_masks, low_masks = [], []
        row = {"candidate_episode_id": event["candidate_episode_id"], "status": event["status"],
               "window": [start, end], "observation_duration_sec": float(times[-1] - times[0]),
               "minimum_hold_sec": parameters.minimum_hold_sec, "components": {}, "joints": {}}
        for name, ids in DEPENDENCIES.items():
            values = signals["channels"][name][start:end + 1]
            known = np.isfinite(values)
            threshold = event.get("components", {}).get(name, {}).get("entry_threshold")
            low = known & (values <= threshold) if threshold is not None else np.zeros(len(values), bool)
            channel_masks.append(known)
            low_masks.append(low)
            row["components"][name] = {
                "dependency_joints": list(ids), "valid_ratio": float(known.mean()),
                "longest_supported_sec": _longest_duration(known, times),
                "longest_low_motion_sec": _longest_duration(low, times),
                "entry_threshold": threshold, "observed_speed": _distribution(values),
            }
        known = np.all(channel_masks, axis=0)
        low = np.all(low_masks, axis=0)
        row.update(all_components_valid_ratio=float(known.mean()),
                   longest_common_supported_sec=_longest_duration(known, times),
                   longest_common_low_motion_sec=_longest_duration(low, times))
        for joint, name in JOINT_NAMES.items():
            failures = {label: int(mask[start:end + 1, joint].sum()) for label, mask in masks.items()}
            context_start = max(0, start - signals["signal_window_frames"])
            if not np.any(rejected[context_start:end + 1, joint]):
                continue
            # Include preceding positions used by the velocity/smoothing calculation.
            item = {"joint_index": joint, "rejected_frame_counts": failures,
                    "rejected_ranges": _ranges(rejected[context_start:end + 1, joint], context_start),
                    "context_start_frame": context_start}
            if pose.get("triangulation_score") is not None:
                scores = np.asarray(pose["triangulation_score"], dtype=float)
                if scores.shape != points.shape[:2]:
                    raise ScoringContractError("audit triangulation score shape mismatch")
                item["source_triangulation_score"] = _distribution(scores[start:end + 1, joint])
            if reasons is not None:
                item["upstream_reason_bits"] = {name: int(((reasons[start:end + 1, joint] & bit) != 0).sum())
                                                for name, bit in (("base_quality", 1), ("bone_length", 2), ("temporal", 4))}
            row["joints"][name] = item
        result["candidates"].append(row)
    return result


def attach_camera_audit(audit: dict, path: Path, camera_ids: list[str]) -> None:
    """Describe exported 2D measurements; never treat projection/fallback as new proof."""
    observations: dict[tuple, list] = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"frame_idx", "camera_id", "joint_idx", "x", "y", "score", "valid"}.issubset(reader.fieldnames or []):
            raise ScoringContractError("2D audit CSV is missing required fields")
        for row in reader:
            joint = int(row["joint_idx"])
            if joint not in JOINT_NAMES or row["camera_id"] not in camera_ids:
                continue
            frame = int(row["frame_idx"])
            if not 0 <= frame < audit["frame_count"]:
                raise ScoringContractError("2D audit CSV frame outside pose time base")
            valid = row["valid"].strip().lower()
            if valid not in {"true", "false", "1", "0"}:
                raise ScoringContractError("2D audit CSV valid flag is invalid")
            observation = {"finite": bool(np.all(np.isfinite([float(row["x"]), float(row["y"])]))),
                           "score": float(row["score"]), "valid": valid in {"true", "1"}}
            observations.setdefault((frame, row["camera_id"], joint), []).append(observation)
    audit["camera_2d_interpretation"] = "Exported 2D scores and flags are descriptive only. CSV lacks per-joint image/projection provenance; this audit does not certify independent image evidence. Ambiguous multiple-person rows remain ambiguous."
    for candidate in audit["candidates"]:
        start, end = candidate["window"]
        cameras = {}
        for camera in camera_ids:
            joints = {}
            for joint, name in JOINT_NAMES.items():
                groups = [observations.get((frame, camera, joint), []) for frame in range(start, end + 1)]
                unique = [items[0] for items in groups if len(items) == 1]
                joints[name] = {
                    "expected_frames": len(groups), "unique_frames": len(unique),
                    "missing_frames": sum(not group for group in groups),
                    "ambiguous_frames": sum(len(group) > 1 for group in groups),
                    "valid_finite_frames": sum(row["valid"] and row["finite"] for row in unique),
                    "score": _distribution(np.asarray([row["score"] for row in unique])),
                }
            cameras[camera] = joints
        candidate["camera_2d"] = cameras
