"""Fixed-input engineering sensitivity study; never selects detector defaults."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, replace
from itertools import product

import numpy as np

from .contracts import ScoringContractError
from .fixation import FixationParameters, evaluate_fixation_candidates, prepare_fixation_signals


def build_fixation_sensitivity(pose: dict, report: dict) -> dict:
    points = np.asarray(pose["keypoints_3d_world"], float)
    if points.shape != (report["frame_count"], 133, 3) or pose["sample_fps"] != report["fps"]:
        raise ScoringContractError("sensitivity pose time base/shape mismatch")
    base = FixationParameters(**report["detector"]["fixation"]["parameters"])
    signals = prepare_fixation_signals(points, report["fps"], quality=pose, parameters=base)
    episodes = report["candidate_episodes"]
    baseline = evaluate_fixation_candidates(signals, episodes)
    if baseline != report["fixation_candidates"]:
        raise ScoringContractError("sensitivity cannot reproduce source report; compare source versions before experimenting")
    variants, excluded = [], []
    for hold_factor, speed_factor in product((.75, 1.0, 1.25), (.8, 1.0, 1.2)):
        try:
            params = replace(base, minimum_hold_sec=base.minimum_hold_sec * hold_factor,
                             minimum_speed_body_lengths_sec=base.minimum_speed_body_lengths_sec * speed_factor,
                             maximum_speed_body_lengths_sec=base.maximum_speed_body_lengths_sec * speed_factor,
                             peak_speed_fraction=base.peak_speed_fraction * speed_factor,
                             orientation_speed_deg_sec=base.orientation_speed_deg_sec * speed_factor)
        except ScoringContractError as error:
            excluded.append({"hold_duration_factor": hold_factor, "motion_threshold_factor": speed_factor, "reason": str(error)})
            continue
        candidates = evaluate_fixation_candidates({**signals, "parameters": asdict(params)}, episodes)
        variants.append({"hold_duration_factor": hold_factor, "motion_threshold_factor": speed_factor,
                         "parameters": asdict(params), "observed_hold_count": sum(c["fixation_frame"] is not None for c in candidates),
                         "candidates": [{"candidate_episode_id": c["candidate_episode_id"], "status": c["status"],
                                         "fixation_frame": c["fixation_frame"], "hold_end_frame": c["hold_end_frame"]} for c in candidates]})
    summaries = []
    for index, event in enumerate(baseline):
        rows = [v["candidates"][index] for v in variants]
        frames = [row["fixation_frame"] for row in rows if row["fixation_frame"] is not None]
        summaries.append({"candidate_episode_id": event["candidate_episode_id"], "baseline_frame": event["fixation_frame"],
                          "supported_configuration_count": len(frames), "configuration_count": len(variants),
                          "status_counts": dict(Counter(row["status"] for row in rows)),
                          "observed_frame_range": [min(frames), max(frames)] if frames else None,
                          "changed_configuration_count": sum(row["fixation_frame"] != event["fixation_frame"] for row in rows)})
    return {
        "schema_version": 1, "status": "single_recording_parameter_sensitivity_not_accuracy",
        "frame_count": len(points), "fps": report["fps"], "baseline_reproduced": True,
        "fixed_motion_episode_count": len(episodes), "configuration_count": len(variants),
        "requested_configuration_count": 9, "excluded_configurations": excluded,
        "manual_labels_used": False, "quality_gate_relaxed": False, "automatic_parameter_selection": False,
        "grid_support_fraction_is_probability": False, "frame_range_is_confidence_interval": False,
        "interpretation": "Motion episodes, pose samples, quality masks, camera evidence and smoothing are held fixed. Only hold duration and motion decision thresholds vary. This grid measures parameter sensitivity; it cannot establish which configuration is correct. No defaults are changed.",
        "candidates": summaries, "variants": variants,
    }
