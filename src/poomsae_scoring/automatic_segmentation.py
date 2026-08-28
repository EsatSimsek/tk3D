"""Pose-driven movement and phase boundary diagnostics.

The detector in this module is deliberately diagnostic-only.  It proposes
movement windows from the BODY-17 motion signal, maps them to the already
selected recording scope in sequence order, and compares the proposal with a
source-bound reference timeline.  It never replaces a confirmed timeline and
never authorizes a score or an automatic deduction.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from src.data_structures import COCO_BODY_JOINT_INDICES
from src.poomsae_scoring.contracts import (
    ScoringContractError,
    validate_movement_timeline,
    validate_poomsae_spec,
)
from src.scoring_readiness import adaptive_motion_threshold, joint_speed


AUTOMATIC_SEGMENTATION_STATUS = "automatic_segmentation_diagnostic_only"


def build_automatic_segmentation_diagnostics(
    pose_payload: dict[str, Any],
    poomsae_spec: dict[str, Any],
    reference_timeline: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build scoreless automatic movement/phase proposals and frame rows."""
    spec = validate_poomsae_spec(poomsae_spec)
    timeline = validate_movement_timeline(reference_timeline, spec)
    points = np.asarray(pose_payload.get("keypoints_3d_world"), dtype=float)
    if points.ndim != 3 or points.shape[1:] != (133, 3) or points.shape[0] == 0:
        raise ScoringContractError("automatic segmentation requires keypoints_3d_world[t,133,3]")
    if points.shape[0] != int(timeline["frame_count"]):
        raise ScoringContractError("pose frame count does not match the reference timeline")
    fps = float(pose_payload.get("sample_fps") or timeline["fps"])
    if not np.isfinite(fps) or fps <= 0 or not np.isclose(fps, float(timeline["fps"]), atol=1e-9):
        raise ScoringContractError("pose FPS does not match the reference timeline")

    observed_ids = list(timeline["coverage"]["observed_movement_ids"])
    movement_by_id = {movement["movement_id"]: movement for movement in spec["movements"]}
    movement_definitions = [movement_by_id[movement_id] for movement_id in observed_ids]
    detection = detect_automatic_segments(points, fps=fps, movements=movement_definitions)
    comparison = compare_segments_to_reference(detection["segments"], timeline["segments"], fps=fps)
    frame_rows = _frame_rows(
        detection["motion_energy"],
        detection["smoothed_motion_energy"],
        detection["segments"],
        detection["thresholds"],
        fps,
    )
    report = {
        "schema_version": 1,
        "status": AUTOMATIC_SEGMENTATION_STATUS,
        "poomsae": {"poomsae_id": spec["poomsae_id"], "version": spec["version"]},
        "movement_timeline_id": timeline["timeline_id"],
        "selected_scope_movement_ids": observed_ids,
        "detector": {
            "signal": "mean_body17_joint_speed_mps",
            "smoothing": "centered_nanmedian",
            "selection": detection["selection"],
            "parameters": detection["parameters"],
            "thresholds": detection["thresholds"],
            "valid_signal_ratio": detection["valid_signal_ratio"],
        },
        "summary": {
            "expected_movement_count": len(observed_ids),
            "detected_candidate_count": len(detection["candidate_episodes"]),
            "selected_movement_count": len(detection["segments"]),
            "movement_count_match": len(detection["segments"]) == len(observed_ids),
            "start_boundary_mae_frames": comparison["summary"]["start_boundary_mae_frames"],
            "end_boundary_mae_frames": comparison["summary"]["end_boundary_mae_frames"],
            "phase_anchor_mae_frames": comparison["summary"]["phase_anchor_mae_frames"],
            "phase_anchor_max_error_frames": comparison["summary"]["phase_anchor_max_error_frames"],
        },
        "candidate_episodes": detection["candidate_episodes"],
        "segments": detection["segments"],
        "reference_comparison": comparison,
        "safety_contract": {
            "diagnostic_only": True,
            "confirmed_timeline_replacement_allowed": False,
            "score_claim_allowed": False,
            "automatic_deduction_allowed": False,
            "boundary_detection_uses_reference_frames": False,
            "reference_timeline_used_only_for_scope_ids_and_post_detection_comparison": True,
        },
        "interpretation": (
            "Hareket ve faz sınırları yalnız 3B BODY-17 hareket sinyalinden önerilir. "
            "Onaylı timeline değiştirilmez; karşılaştırma sonuçları puan veya kesinti değildir."
        ),
    }
    return report, frame_rows


def detect_automatic_segments(
    keypoints_3d: np.ndarray,
    *,
    fps: float,
    movements: list[dict[str, Any]],
) -> dict[str, Any]:
    """Detect a contiguous movement sequence without using reference boundaries."""
    points = np.asarray(keypoints_3d, dtype=float)
    if points.ndim != 3 or points.shape[1:] != (133, 3) or points.shape[0] == 0:
        raise ScoringContractError("keypoints_3d must have shape [t,133,3]")
    if not np.isfinite(fps) or fps <= 0:
        raise ScoringContractError("fps must be finite and positive")
    if not movements:
        raise ScoringContractError("movements must be a non-empty list")
    for movement in movements:
        if not isinstance(movement.get("movement_id"), str) or not movement["movement_id"]:
            raise ScoringContractError("each movement requires movement_id")
        phases = movement.get("phases")
        if not isinstance(phases, list) or not phases:
            raise ScoringContractError("each movement requires an ordered phase list")

    energy = _body_motion_energy(points, fps)
    smoothing_frames = _odd_frames(fps * 0.083, minimum=3)
    smoothed = _rolling_nanmedian(energy, smoothing_frames)
    high_threshold = _finite_motion_threshold(smoothed)
    onset_threshold = high_threshold * 0.5
    min_active_frames = max(3, int(round(fps * 0.05)))
    merge_gap_frames = max(1, int(round(fps * 0.15)))
    active = np.isfinite(smoothed) & (smoothed >= onset_threshold)
    episodes = _merge_runs(
        _contiguous_runs(active, min_active_frames),
        max_gap_frames=merge_gap_frames,
    )
    episode_rows = [_episode_row(index, start, end, smoothed, fps) for index, (start, end) in enumerate(episodes)]
    selected_indices, selection = _select_episode_sequence(episode_rows, len(movements), fps)
    selected = [episode_rows[index] for index in selected_indices]

    segments: list[dict[str, Any]] = []
    high_active = np.isfinite(smoothed) & (smoothed >= high_threshold)
    valid_signal_ratio = float(np.mean(np.isfinite(smoothed)))
    for index, (movement, episode) in enumerate(zip(movements, selected, strict=False)):
        start = int(episode["start_frame"])
        end = int(selected[index + 1]["start_frame"] - 1) if index + 1 < len(selected) else points.shape[0] - 1
        fixation = min(max(int(episode["end_frame"]) + 1, start), end)
        high_runs = [
            run
            for run in _contiguous_runs(high_active, min_active_frames)
            if run[0] <= int(episode["end_frame"]) and run[1] >= start
        ]
        if high_runs:
            last_high_end = min(high_runs[-1][1], fixation)
        else:
            window = smoothed[start : int(episode["end_frame"]) + 1]
            last_high_end = start + int(np.nanargmax(window)) if np.any(np.isfinite(window)) else start
        execution = int(round(last_high_end + 0.4 * max(0, fixation - last_high_end)))
        execution = min(max(execution, start), fixation)
        anchors, anchor_methods = _phase_anchors(
            movement["phases"],
            start_frame=start,
            execution_frame=execution,
            fixation_frame=fixation,
        )
        peak = float(episode["peak_motion_energy_mps"])
        strength = min(max(peak / high_threshold - 1.0, 0.0), 1.0) if high_threshold > 0 else 0.0
        selection_bonus = 1.0 if selection["status"] == "exact_cluster_match" else 0.5
        confidence = round(min(0.95, 0.55 + 0.15 * strength + 0.15 * valid_signal_ratio + 0.10 * selection_bonus), 4)
        segments.append(
            {
                "sequence_index": index + 1,
                "movement_id": movement["movement_id"],
                "start_frame": start,
                "end_frame": end,
                "start_time_sec": round(start / fps, 6),
                "end_time_sec": round(end / fps, 6),
                "anchors": anchors,
                "anchor_methods": anchor_methods,
                "confidence": confidence,
                "label_status": "automatic_diagnostic",
                "motion_episode": {
                    "candidate_episode_id": episode["candidate_episode_id"],
                    "active_start_frame": int(episode["start_frame"]),
                    "active_end_frame": int(episode["end_frame"]),
                    "last_high_motion_frame": int(last_high_end),
                    "peak_motion_energy_mps": peak,
                },
            }
        )

    return {
        "motion_energy": energy,
        "smoothed_motion_energy": smoothed,
        "thresholds": {
            "high_motion_mps": round(float(high_threshold), 9),
            "movement_onset_mps": round(float(onset_threshold), 9),
        },
        "parameters": {
            "smoothing_frames": smoothing_frames,
            "minimum_active_frames": min_active_frames,
            "merge_gap_frames": merge_gap_frames,
        },
        "valid_signal_ratio": round(valid_signal_ratio, 6),
        "candidate_episodes": episode_rows,
        "selection": selection,
        "segments": segments,
    }


def compare_segments_to_reference(
    proposed_segments: list[dict[str, Any]],
    reference_segments: list[dict[str, Any]],
    *,
    fps: float,
) -> dict[str, Any]:
    """Compare already-detected proposals to a reference; never alter proposals."""
    proposed_by_id = {segment["movement_id"]: segment for segment in proposed_segments}
    rows: list[dict[str, Any]] = []
    start_errors: list[int] = []
    end_errors: list[int] = []
    anchor_errors: list[int] = []
    for reference in reference_segments:
        movement_id = reference["movement_id"]
        proposed = proposed_by_id.get(movement_id)
        if proposed is None:
            rows.append({"movement_id": movement_id, "status": "not_detected"})
            continue
        start_delta = int(proposed["start_frame"]) - int(reference["start_frame"])
        end_delta = int(proposed["end_frame"]) - int(reference["end_frame"])
        start_errors.append(abs(start_delta))
        end_errors.append(abs(end_delta))
        phase_rows: dict[str, Any] = {}
        for phase_id, reference_frame in reference.get("anchors", {}).items():
            proposed_frame = proposed.get("anchors", {}).get(phase_id)
            if proposed_frame is None:
                phase_rows[phase_id] = {"status": "not_detected"}
                continue
            delta = int(proposed_frame) - int(reference_frame)
            anchor_errors.append(abs(delta))
            phase_rows[phase_id] = {
                "status": "compared",
                "proposed_frame": int(proposed_frame),
                "reference_frame": int(reference_frame),
                "delta_frames": delta,
                "absolute_error_frames": abs(delta),
                "absolute_error_sec": round(abs(delta) / fps, 6),
            }
        rows.append(
            {
                "movement_id": movement_id,
                "status": "compared",
                "start": _boundary_comparison(proposed["start_frame"], reference["start_frame"], fps),
                "end": _boundary_comparison(proposed["end_frame"], reference["end_frame"], fps),
                "phases": phase_rows,
            }
        )
    return {
        "status": "same_pose_reference_comparison",
        "summary": {
            "reference_movement_count": len(reference_segments),
            "compared_movement_count": sum(1 for row in rows if row["status"] == "compared"),
            "start_boundary_mae_frames": _mean_or_none(start_errors),
            "end_boundary_mae_frames": _mean_or_none(end_errors),
            "phase_anchor_mae_frames": _mean_or_none(anchor_errors),
            "phase_anchor_max_error_frames": max(anchor_errors) if anchor_errors else None,
            "phase_anchor_mae_sec": (
                None if not anchor_errors else round(float(np.mean(anchor_errors)) / fps, 6)
            ),
        },
        "movements": rows,
    }


def _body_motion_energy(points: np.ndarray, fps: float) -> np.ndarray:
    speeds = joint_speed(points, fps=fps)
    indices = [index for index in COCO_BODY_JOINT_INDICES if index < speeds.shape[1]]
    body = speeds[:, indices]
    finite = np.isfinite(body)
    counts = np.sum(finite, axis=1)
    return np.divide(
        np.nansum(body, axis=1),
        counts,
        out=np.full(points.shape[0], np.nan, dtype=float),
        where=counts > 0,
    )


def _rolling_nanmedian(values: np.ndarray, window: int) -> np.ndarray:
    radius = window // 2
    padded = np.pad(np.asarray(values, dtype=float), (radius, radius), constant_values=np.nan)
    output = np.full(values.shape, np.nan, dtype=float)
    for index in range(values.size):
        window_values = padded[index : index + window]
        finite = window_values[np.isfinite(window_values)]
        if finite.size:
            output[index] = float(np.median(finite))
    return output


def _finite_motion_threshold(values: np.ndarray) -> float:
    threshold = float(adaptive_motion_threshold(values))
    finite_positive = values[np.isfinite(values) & (values > 1e-12)]
    if finite_positive.size == 0:
        raise ScoringContractError("automatic segmentation has no finite positive motion signal")
    if not np.isfinite(threshold) or threshold <= 1e-12:
        threshold = float(np.percentile(finite_positive, 35))
    return threshold


def _contiguous_runs(mask: np.ndarray, minimum_frames: int) -> list[tuple[int, int]]:
    padded = np.concatenate(([False], np.asarray(mask, dtype=bool), [False])).astype(np.int8)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    return [
        (int(start), int(end))
        for start, end in zip(starts, ends, strict=True)
        if end - start + 1 >= minimum_frames
    ]


def _merge_runs(runs: list[tuple[int, int]], *, max_gap_frames: int) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in runs:
        if merged and start - merged[-1][1] - 1 <= max_gap_frames:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def _episode_row(index: int, start: int, end: int, energy: np.ndarray, fps: float) -> dict[str, Any]:
    window = energy[start : end + 1]
    peak_offset = int(np.nanargmax(window))
    return {
        "candidate_episode_id": index,
        "start_frame": start,
        "end_frame": end,
        "start_time_sec": round(start / fps, 6),
        "end_time_sec": round(end / fps, 6),
        "duration_sec": round((end - start + 1) / fps, 6),
        "peak_frame": start + peak_offset,
        "peak_motion_energy_mps": round(float(window[peak_offset]), 9),
        "mean_motion_energy_mps": round(float(np.mean(window[np.isfinite(window)])), 9),
    }


def _select_episode_sequence(
    episodes: list[dict[str, Any]],
    expected_count: int,
    fps: float,
) -> tuple[list[int], dict[str, Any]]:
    if not episodes:
        return [], {"status": "no_motion_episode", "selected_candidate_episode_ids": []}
    if len(episodes) <= expected_count:
        status = "exact_episode_count" if len(episodes) == expected_count else "insufficient_episode_count"
        ids = list(range(len(episodes)))
        return ids, {"status": status, "selected_candidate_episode_ids": ids}

    gaps = [
        int(episodes[index]["start_frame"]) - int(episodes[index - 1]["end_frame"]) - 1
        for index in range(1, len(episodes))
    ]
    gap_array = np.asarray(gaps, dtype=float)
    median_gap = float(np.median(gap_array))
    mad_gap = float(np.median(np.abs(gap_array - median_gap)))
    split_gap = max(int(round(fps * 0.75)), int(round(median_gap + 2.0 * 1.4826 * mad_gap)))
    clusters: list[list[int]] = [[0]]
    for index, gap in enumerate(gaps, start=1):
        if gap > split_gap:
            clusters.append([index])
        else:
            clusters[-1].append(index)
    exact = [cluster for cluster in clusters if len(cluster) == expected_count]
    if exact:
        selected = max(
            exact,
            key=lambda cluster: sum(float(episodes[index]["peak_motion_energy_mps"]) for index in cluster),
        )
        return selected, {
            "status": "exact_cluster_match",
            "split_gap_frames": split_gap,
            "cluster_sizes": [len(cluster) for cluster in clusters],
            "selected_candidate_episode_ids": selected,
        }

    windows = [list(range(start, start + expected_count)) for start in range(len(episodes) - expected_count + 1)]
    selected = min(windows, key=lambda window: _window_gap_dispersion(window, episodes))
    return selected, {
        "status": "minimum_gap_dispersion_fallback",
        "split_gap_frames": split_gap,
        "cluster_sizes": [len(cluster) for cluster in clusters],
        "selected_candidate_episode_ids": selected,
    }


def _window_gap_dispersion(window: list[int], episodes: list[dict[str, Any]]) -> float:
    gaps = [
        int(episodes[right]["start_frame"]) - int(episodes[left]["end_frame"]) - 1
        for left, right in zip(window[:-1], window[1:], strict=True)
    ]
    if not gaps:
        return 0.0
    median = float(np.median(gaps))
    return float(np.median(np.abs(np.asarray(gaps, dtype=float) - median)))


def _phase_anchors(
    phases: list[str],
    *,
    start_frame: int,
    execution_frame: int,
    fixation_frame: int,
) -> tuple[dict[str, int], dict[str, str]]:
    anchors: dict[str, int] = {}
    methods: dict[str, str] = {}
    execution_index = phases.index("execution") if "execution" in phases else max(0, len(phases) - 2)
    fixation_index = phases.index("fixation") if "fixation" in phases else len(phases) - 1
    for index, phase in enumerate(phases):
        if phase == "preparation" or index == 0:
            value, method = start_frame, "movement_onset"
        elif phase == "execution":
            value, method = execution_frame, "last_high_motion_deceleration"
        elif phase == "fixation":
            value, method = fixation_frame, "sustained_low_motion_entry"
        elif index < execution_index:
            fraction = index / max(execution_index, 1)
            value = int(round(start_frame + fraction * (execution_frame - start_frame)))
            method = "ordered_interpolation_before_execution"
        else:
            fraction = (index - execution_index) / max(fixation_index - execution_index, 1)
            value = int(round(execution_frame + fraction * (fixation_frame - execution_frame)))
            method = "ordered_interpolation_after_execution"
        anchors[phase] = int(min(max(value, start_frame), fixation_frame))
        methods[phase] = method
    return anchors, methods


def _frame_rows(
    energy: np.ndarray,
    smoothed: np.ndarray,
    segments: list[dict[str, Any]],
    thresholds: dict[str, float],
    fps: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for frame_index in range(energy.size):
        segment = next(
            (
                item
                for item in segments
                if int(item["start_frame"]) <= frame_index <= int(item["end_frame"])
            ),
            None,
        )
        rows.append(
            {
                "frame_index": frame_index,
                "timestamp_sec": round(frame_index / fps, 6),
                "body_motion_energy_mps": _finite_or_none(energy[frame_index]),
                "smoothed_motion_energy_mps": _finite_or_none(smoothed[frame_index]),
                "above_onset_threshold": bool(
                    np.isfinite(smoothed[frame_index])
                    and smoothed[frame_index] >= thresholds["movement_onset_mps"]
                ),
                "above_high_motion_threshold": bool(
                    np.isfinite(smoothed[frame_index])
                    and smoothed[frame_index] >= thresholds["high_motion_mps"]
                ),
                "movement_id": None if segment is None else segment["movement_id"],
                "is_fixation_anchor": bool(
                    segment is not None and frame_index == segment["anchors"].get("fixation")
                ),
            }
        )
    return rows


def _boundary_comparison(proposed: Any, reference: Any, fps: float) -> dict[str, Any]:
    proposed_frame = int(proposed)
    reference_frame = int(reference)
    delta = proposed_frame - reference_frame
    return {
        "proposed_frame": proposed_frame,
        "reference_frame": reference_frame,
        "delta_frames": delta,
        "absolute_error_frames": abs(delta),
        "absolute_error_sec": round(abs(delta) / fps, 6),
    }


def _mean_or_none(values: list[int]) -> float | None:
    return None if not values else round(float(np.mean(values)), 6)


def _finite_or_none(value: float) -> float | None:
    return round(float(value), 9) if np.isfinite(value) else None


def _odd_frames(value: float, *, minimum: int) -> int:
    frames = max(minimum, int(round(value)))
    return frames if frames % 2 else frames + 1
