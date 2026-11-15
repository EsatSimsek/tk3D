"""Quality-gated hold evidence; no manual labels or scoring thresholds are inputs.

The engineering parameters below describe a motion detector, not correct sport
technique. A low-motion interval is an automatic review proposal, never a human
approval. Completion candidates are retained even when a later correction settles.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any

import numpy as np

from src.poomsae_scoring.contracts import ScoringContractError


@dataclass(frozen=True)
class FixationParameters:
    signal_window_sec: float = 0.05
    minimum_hold_sec: float = 0.10
    minimum_speed_body_lengths_sec: float = 0.08
    maximum_speed_body_lengths_sec: float = 0.35
    peak_speed_fraction: float = 0.25
    orientation_speed_deg_sec: float = 25.0
    exit_threshold_multiplier: float = 1.5
    maximum_reprojection_px: float = 15.0
    minimum_cameras: int = 2
    # Technique completion: the later of the two hands whose speed first falls to this
    # fraction of its own episode peak. Engineering default; 0.25-0.50 gave the same frames within 4 on the
    # one labelled recording, so the value is not tuned to those labels.
    completion_peak_fraction: float = 0.35

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
                raise ScoringContractError(f"fixation parameter {name} must be finite and positive")
        if self.minimum_cameras != int(self.minimum_cameras) or self.minimum_cameras < 2:
            raise ScoringContractError("fixation minimum_cameras must be an integer >= 2")
        if self.minimum_speed_body_lengths_sec > self.maximum_speed_body_lengths_sec:
            raise ScoringContractError("fixation minimum speed exceeds maximum speed")
        if self.peak_speed_fraction >= 1 or self.exit_threshold_multiplier <= 1:
            raise ScoringContractError("fixation fraction must be < 1 and exit multiplier > 1")
        if self.completion_peak_fraction >= 1:
            raise ScoringContractError("fixation completion_peak_fraction must be < 1")


def prepare_fixation_signals(
    points: np.ndarray, fps: float, *, quality: dict[str, Any] | None = None,
    parameters: FixationParameters | None = None,
) -> dict[str, Any]:
    """Use supported adjacent samples only; never interpolate a visibility gap."""
    params = parameters or FixationParameters()
    quality = quality or {}
    count = len(points)
    times = np.asarray(quality.get("timestamps_sec", np.arange(count) / fps), dtype=float)
    if times.shape != (count,) or not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise ScoringContractError("fixation timestamps must be finite and strictly increasing")
    # The legacy movement proposal uses FPS. Reject inconsistent time bases.
    if count > 1 and not np.allclose(np.diff(times), 1 / fps, rtol=0.05, atol=1e-6):
        raise ScoringContractError("fixation timestamps do not match sample_fps")
    stride = quality.get("inference_stride", 1)
    if isinstance(stride, bool) or not isinstance(stride, (int, float)) or not np.isfinite(stride) or stride < 1 or int(stride) != stride:
        raise ScoringContractError("fixation inference_stride must be a positive integer")
    valid = np.all(np.isfinite(points), axis=-1)
    required = ("reliability_valid_mask", "used_cameras", "reprojection_error")
    missing = [name for name in required if quality.get(name) is None]
    for name in required:
        if quality.get(name) is None:
            continue
        values = np.asarray(quality[name])
        if values.shape != points.shape[:2]:
            raise ScoringContractError(f"fixation {name} shape does not match pose")
        if name == "reliability_valid_mask":
            if values.dtype.kind != "b":
                raise ScoringContractError("fixation reliability mask must contain booleans")
            valid &= values
        else:
            values = np.asarray(values, dtype=float)
            valid &= np.isfinite(values)
            valid &= (values >= params.minimum_cameras) if name == "used_cameras" else (
                (values >= 0) & (values <= params.maximum_reprojection_px)
            )
    masked = np.where(valid[:, :, None], points, np.nan)
    window = max(1, int(round(fps * params.signal_window_sec)))
    raw_speed = _linear_speed(masked, times)
    speed = _supported_median(raw_speed, window)
    # Leg length is translation/rotation invariant and independent of technique angles.
    lengths = np.stack([
        np.linalg.norm(masked[:, hip] - masked[:, knee], axis=1)
        + np.linalg.norm(masked[:, knee] - masked[:, ankle], axis=1)
        for hip, knee, ankle in ((11, 13, 15), (12, 14, 16))
    ], axis=1)
    finite_lengths = lengths[np.isfinite(lengths) & (lengths > 0.05)]
    scale = float(np.median(finite_lengths)) if finite_lengths.size else None
    divisor = scale if scale is not None else np.nan
    pelvis = (masked[:, 11] + masked[:, 12]) / 2
    channels, raw_channels, hand_observations = {}, {}, {}
    for name, joint in (("left_hand", 9), ("right_hand", 10)):
        raw_relative = _linear_speed(masked[:, joint] - pelvis, times)
        relative = _supported_median(raw_relative, window)
        channels[name] = np.maximum(speed[:, joint], relative) / divisor
        raw_channels[name] = np.maximum(raw_speed[:, joint], raw_relative) / divisor
        hand_observations[name] = {"absolute": speed[:, joint] / divisor, "pelvis_relative": relative / divisor}
    for name, indices in (("feet", [15, 16]), ("pelvis", [11, 12]), ("torso", [5, 6])):
        channels[name] = np.max(speed[:, indices], axis=1) / divisor
        raw_channels[name] = np.max(raw_speed[:, indices], axis=1) / divisor
    for name, left, right in (("shoulder_rotation", 5, 6), ("pelvis_rotation", 11, 12)):
        line = masked[:, right] - masked[:, left]
        norm = np.linalg.norm(line, axis=1)
        unit = np.divide(line, norm[:, None], out=np.full_like(line, np.nan), where=norm[:, None] > 1e-6)
        angular = np.full(count, np.nan)
        if count > 1:
            angular[1:] = np.degrees(np.arccos(np.clip(np.sum(unit[1:] * unit[:-1], axis=1), -1, 1))) / np.diff(times)
        channels[name] = _supported_median(angular, window)
        raw_channels[name] = angular
    # Movement segmentation uses its own BODY-17 signal, separate from the hold channels.
    energy = segmentation_motion_energy(speed)
    # Unsupported sampling cannot establish holds from repeated output frames.
    blocking = "unsupported_inference_stride" if stride != 1 else (
        "missing_quality_metadata" if missing else "invalid_body_scale" if scale is None else None
    )
    return {
        "channels": channels, "raw_channels": raw_channels, "hand_observations": hand_observations,
        "segmentation_motion_energy": energy, "times": times,
        "parameters": asdict(params), "signal_window_frames": window,
        "body_scale_m": scale, "blocking_reason": blocking,
        "missing_quality_fields": missing,
        "body17_quality_valid_ratio": float(valid[:, :17].mean()),
        "all_components_valid_ratio": float(np.all(np.isfinite(np.stack(list(channels.values()))), axis=0).mean()),
        "method": "quality_gated_multicomponent_v1", "inference_stride": int(stride),
    }


SEGMENTATION_ENERGY_JOINTS = tuple(range(17))
# Same 3/4 support ratio as the former 9-of-12 limb rule, scaled to BODY-17.
SEGMENTATION_MINIMUM_FINITE_JOINTS = 13
SEGMENTATION_ENERGY_SIGNAL = "quality_gated_mean_body17_joint_speed_mps"


def segmentation_motion_energy(speed: np.ndarray) -> np.ndarray:
    """Mean quality-gated BODY-17 (joints 0..16) speed for movement-episode detection.

    The head joints are kept: they move during transitions between movements,
    and excluding them splits one movement into several low-energy valleys.
    Unsupported (NaN) joint speeds never contribute motion.
    """
    body = np.asarray(speed, dtype=float)[:, list(SEGMENTATION_ENERGY_JOINTS)]
    supported = np.isfinite(body).sum(axis=1)
    return np.divide(np.nansum(body, axis=1), supported, out=np.full(len(body), np.nan),
                     where=supported >= SEGMENTATION_MINIMUM_FINITE_JOINTS)


def evaluate_fixation_candidates(
    signals: dict[str, Any], episodes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Evaluate every episode, without knowing expected count, IDs or label times."""
    channels, times = signals["channels"], signals["times"]
    params = FixationParameters(**signals["parameters"])
    result = []
    for index, episode in enumerate(episodes):
        start, peak, active_end = (int(episode[key]) for key in ("start_frame", "peak_frame", "end_frame"))
        end = int(episodes[index + 1]["start_frame"]) - 1 if index + 1 < len(episodes) else len(times) - 1
        completion = active_end + 1 if active_end < end else None
        row: dict[str, Any] = {
            "candidate_episode_id": int(episode["candidate_episode_id"]),
            "start_frame": start, "end_frame": end, "peak_frame": peak,
            "completion_candidate_frame": completion,
            "completion_method": "aggregate_motion_exit_not_technique_ground_truth",
            "fixation_frame": None, "hold_end_frame": None, "hold_duration_sec": None,
            "confirmation_frame": None, "timing_support_frames": None,
            "status": "not_observed", "reason": None, "components": {},
            "later_motion_frame": None, "requires_human_review": True,
            "confidence_probability": None, "score_effect": False,
            "technique_completion": technique_completion(signals, start, active_end, end, params),
        }
        if signals["blocking_reason"]:
            row.update(status="insufficient_evidence", reason=signals["blocking_reason"])
            result.append(row)
            continue
        if completion is None:
            row.update(status="recording_truncated" if end == len(times) - 1 else "not_observed",
                       reason="no_observed_motion_exit")
            result.append(row)
            continue
        masks, known_masks = [], []
        for name, values in channels.items():
            moving = values[start:active_end + 1]
            moving = moving[np.isfinite(moving)]
            peak_speed = float(np.percentile(moving, 90)) if moving.size else None
            threshold = params.orientation_speed_deg_sec if name.endswith("rotation") else (
                max(params.minimum_speed_body_lengths_sec,
                    min(params.maximum_speed_body_lengths_sec, (peak_speed or 0) * params.peak_speed_fraction))
            )
            known = np.isfinite(values)
            low = known & (values <= threshold)
            runs = _sustained_runs(low, times, peak, end, params.minimum_hold_sec)
            first = runs[0][0] if runs else None
            row["components"][name] = {
                "settled_frame": first, "entry_threshold": threshold,
                "unit": "deg/s" if name.endswith("rotation") else "leg_lengths/s",
                "peak_p90": peak_speed, "valid_ratio": float(known[peak:end + 1].mean()),
            }
            masks.append(low)
            known_masks.append(known)
        joint_known = np.all(known_masks, axis=0)
        joint_low = np.all(masks, axis=0)
        holds = _sustained_runs(joint_low, times, completion, end, params.minimum_hold_sec)
        if holds:
            anchor, hold_end, confirmation = holds[0]
            later = np.zeros(len(times), dtype=bool)
            for name, values in channels.items():
                later |= np.isfinite(values) & (values > row["components"][name]["entry_threshold"] * params.exit_threshold_multiplier)
            later_runs = _sustained_runs(later & joint_known, times, hold_end + 1, end, params.signal_window_sec)
            row.update(
                status="hold_observed", reason="supported_multicomponent_low_motion",
                fixation_frame=anchor, hold_end_frame=hold_end,
                confirmation_frame=confirmation, hold_duration_sec=round(float(times[hold_end] - times[anchor]), 6),
                timing_support_frames=[max(start, anchor - signals["signal_window_frames"]), anchor],
                later_motion_frame=later_runs[0][0] if later_runs else None,
            )
        elif not _sustained_runs(joint_known, times, completion, end, params.minimum_hold_sec):
            enough_time = times[end] - times[completion] + 1e-9 >= params.minimum_hold_sec
            row.update(status="insufficient_evidence" if enough_time else "recording_truncated" if end == len(times) - 1 else "not_observed",
                       reason="missing_component_evidence" if enough_time else "insufficient_observation_duration")
        else:
            settled = [component["settled_frame"] for component in row["components"].values()]
            row.update(status="components_disagree" if any(frame is not None for frame in settled) else "not_observed",
                       reason="no_common_sustained_hold")
        result.append(row)
    return result


def technique_completion(
    signals: dict[str, Any], start: int, active_end: int, end: int, params: FixationParameters,
) -> dict[str, Any]:
    """When the technique ends: both hands have decelerated to a fraction of their peak.

    This is the moment the rules read the final pose. It is separate from a hold: an
    athlete who flows into the next movement still completes the block or punch, but
    shows no common multi-component hold. Each hand's own episode peak sets its
    threshold, and the technique ends when the later of the two hands has slowed
    down, so the striking/blocking hand and the reaction hand both count. No movement
    label, side or timeline is used. The fastest hand alone is not enough: the
    chambering or reaction arm is often faster than the blocking arm.
    """
    row: dict[str, Any] = {
        "frame": None, "last_hand": None, "hands": {}, "unit": "leg_lengths/s",
        "peak_fraction": params.completion_peak_fraction,
        "method": "both_hands_decelerated_below_peak_fraction",
        "status": "not_observed", "reason": None,
        "requires_human_review": True, "score_effect": False,
    }
    if signals["blocking_reason"]:
        row.update(status="insufficient_evidence", reason=signals["blocking_reason"])
        return row
    truncated = end == len(signals["times"]) - 1
    for hand, observations in signals["hand_observations"].items():
        speed = observations["absolute"]
        moving = speed[start:active_end + 1]
        if not np.any(np.isfinite(moving)):
            row["hands"][hand] = {"frame": None, "status": "insufficient_evidence"}
            continue
        peak = start + int(np.nanargmax(moving))
        threshold = float(speed[peak]) * params.completion_peak_fraction
        frame = next((index for index in range(peak, end + 1) if np.isfinite(speed[index]) and speed[index] <= threshold), None)
        row["hands"][hand] = {
            "frame": None if frame is None else int(frame), "peak_frame": peak,
            "peak_speed": float(speed[peak]), "threshold_speed": threshold,
            "status": "observed" if frame is not None else "recording_truncated" if truncated else "not_observed",
        }
    statuses = [hand["status"] for hand in row["hands"].values()]
    if "insufficient_evidence" in statuses:
        row.update(status="insufficient_evidence", reason="hand_speed_not_supported")
    elif all(status == "observed" for status in statuses):
        last_hand = max(row["hands"], key=lambda hand: row["hands"][hand]["frame"])
        row.update(frame=row["hands"][last_hand]["frame"], last_hand=last_hand,
                   status="observed", reason="both_hands_decelerated")
    else:
        row.update(status="recording_truncated" if truncated else "not_observed",
                   reason="a_hand_did_not_decelerate_below_threshold")
    return row


def observe_hold(signals: dict[str, Any], start_frame: int, completion_frame: int, end_frame: int) -> dict[str, Any]:
    """Run the same multi-component hold detector on one labelled segment.

    Motion is taken from ``start_frame`` up to the technique completion (the fixation
    anchor); the hold is searched from the anchor to ``end_frame`` only, so a hold of
    the next movement can never be borrowed. Used by the technical rules, which read
    the reviewed timeline rather than the automatic episodes.
    """
    count = len(signals["times"])
    empty = {"status": "not_observed", "onset_frame": None, "end_frame": None,
             "duration_sec": None, "reason": None}
    if not 0 <= int(start_frame) < int(completion_frame) <= int(end_frame) < count:
        return {**empty, "reason": "invalid_hold_search_window"}
    episodes = [{"candidate_episode_id": 0, "start_frame": int(start_frame), "peak_frame": int(start_frame),
                 "end_frame": int(completion_frame) - 1}]
    if int(end_frame) + 1 < count:
        # A boundary episode ends the search at the segment end.
        boundary = int(end_frame) + 1
        episodes.append({"candidate_episode_id": 1, "start_frame": boundary, "peak_frame": boundary, "end_frame": boundary})
    row = evaluate_fixation_candidates(signals, episodes)[0]
    return {"status": row["status"], "onset_frame": row["fixation_frame"], "end_frame": row["hold_end_frame"],
            "duration_sec": row["hold_duration_sec"], "reason": row["reason"]}


def build_fixation_timing_diagnostics(signals: dict[str, Any], events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Check the first proposed hold without speed smoothing; never substitute a later correction.

    The source pose may already be filtered. This is an additional consistency
    check, not independent image evidence or a replacement fixation label.
    """
    times = signals["times"]
    duration = signals["parameters"]["minimum_hold_sec"]
    rows = []
    for event in events:
        start = event["completion_candidate_frame"]
        start = event["peak_frame"] if start is None else start
        end = event["end_frame"]
        row = {"candidate_episode_id": event["candidate_episode_id"], "window": [start, end],
               "hands": {}, "missing_components": [], "speed_unsmoothed_onset_frame": None,
               "confirmation_frame": None, "shift_frames": None,
               "status": "no_supported_hold", "replaces_automatic_proposal": False}
        for name, observations in signals["hand_observations"].items():
            row["hands"][name] = {
                key + "_valid_ratio": float(np.isfinite(values[start:end + 1]).mean())
                for key, values in observations.items()
            }
        for name, values in signals["channels"].items():
            if not _sustained_runs(np.isfinite(values), times, start, end, duration):
                row["missing_components"].append(name)
        if signals["blocking_reason"]:
            row["status"] = "blocked_quality_metadata"
        elif event["fixation_frame"] is not None:
            # Restrict to the original hold, so a later correction cannot replace it.
            low = np.ones(len(times), dtype=bool)
            for name, values in signals["raw_channels"].items():
                low &= np.isfinite(values) & (values <= event["components"][name]["entry_threshold"])
            runs = _sustained_runs(low, times, event["fixation_frame"], event["hold_end_frame"], duration)
            row["status"] = "not_confirmed_without_speed_smoothing"
            if runs:
                onset, _, confirmation = runs[0]
                row.update(status="supported_without_speed_smoothing", speed_unsmoothed_onset_frame=onset,
                           confirmation_frame=confirmation, shift_frames=onset - event["fixation_frame"])
        rows.append(row)
    return rows


def _linear_speed(points: np.ndarray, times: np.ndarray) -> np.ndarray:
    result = np.full(points.shape[:-1], np.nan)
    if len(times) > 1:
        dt = np.diff(times).reshape((-1,) + (1,) * (result.ndim - 1))
        result[1:] = np.linalg.norm(np.diff(points, axis=0), axis=-1) / dt
    return result


def _supported_median(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing window; every input must be supported, including its endpoints."""
    result = np.full_like(values, np.nan)
    for index in range(window - 1, len(values)):
        block = values[index - window + 1:index + 1]
        known = np.all(np.isfinite(block), axis=0)
        result[index] = np.where(known, np.median(block, axis=0), np.nan)
    return result


def _sustained_runs(mask: np.ndarray, times: np.ndarray, start: int, end: int, seconds: float) -> list[tuple[int, int, int]]:
    runs = []
    onset = None
    for frame in range(start, end + 2):
        low = frame <= end and bool(mask[frame])
        if low and onset is None:
            onset = frame
        if not low and onset is not None:
            last = frame - 1
            if times[last] - times[onset] + 1e-9 >= seconds:
                confirmation = int(np.searchsorted(times, times[onset] + seconds - 1e-9))
                runs.append((onset, last, confirmation))
            onset = None
    return runs


def validate_fixation_review_export(payload: dict, report: dict, binding: dict) -> dict:
    """A review correction is an annotation, not an approved scoring timeline."""
    if "review_binding" in report and report["review_binding"] != binding:
        raise ScoringContractError("fixation report binding mismatch")
    if not isinstance(payload, dict) or payload.get("schema_version") != 1 or payload.get("kind") != "fixation_review" or payload.get("binding") != binding:
        raise ScoringContractError("fixation review binding mismatch")
    candidates = {str(row["candidate_episode_id"]): row for row in report["fixation_candidates"]}
    decisions = payload.get("decisions")
    if not isinstance(decisions, dict) or not set(decisions).issubset(candidates):
        raise ScoringContractError("unknown fixation review candidate")
    for key, item in decisions.items():
        if not isinstance(item, dict) or item.get("decision") not in {"accepted", "corrected", "no_hold", "uncertain"}:
            raise ScoringContractError("invalid fixation review decision")
        if not all(isinstance(item.get(name), str) and item[name].strip() for name in ("reviewer", "reviewed_at")):
            raise ScoringContractError("fixation review needs reviewer and timestamp")
        if item.get("movement_id") is not None and item["movement_id"] not in report.get("selected_scope_movement_ids", []):
            raise ScoringContractError("fixation review movement is outside selected scope")
        frame = item.get("frame")
        if item["decision"] in {"accepted", "corrected"}:
            if isinstance(frame, bool) or not isinstance(frame, int) or not 0 <= frame < report["frame_count"]:
                raise ScoringContractError("invalid fixation review frame")
            if item["decision"] == "accepted" and frame != candidates[key]["fixation_frame"]:
                raise ScoringContractError("accepted fixation must match automatic proposal")
        elif frame is not None:
            raise ScoringContractError("uncertain/no_hold fixation frame must be null")
    return payload


def build_fixation_review_binding(report: dict) -> dict:
    """Bind annotations to the source-backed proposal payload before adding this field."""
    if "review_binding" in report or not isinstance(report.get("bindings"), dict):
        raise ScoringContractError("fixation review needs unbound source-backed report")
    digest = hashlib.sha256(json.dumps(report, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()
    return {"review_id": digest, "proposal_sha256": digest, "source_binding": report["bindings"]}
