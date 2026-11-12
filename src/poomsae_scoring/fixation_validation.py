"""Source-bound independent annotations and one-to-one interval event evaluation."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from itertools import combinations
import math
from typing import Any

import numpy as np

from .contracts import ScoringContractError

PROTOCOL = "fixation-independent-v1"
EVENT_KINDS = ("completion", "hold_onset", "hold_end", "no_hold", "uncertain")
PREDICTION_FIELDS = {"completion": "completion_candidate_frame", "hold_onset": "fixation_frame", "hold_end": "hold_end_frame"}


def recording_binding(report: dict) -> dict:
    """Paths, candidate times and detector parameters never define recording identity."""
    sources = report["bindings"]
    if not _frame(report["frame_count"] - 1, report["frame_count"]) or isinstance(report["frame_count"], bool) or not isinstance(report["fps"], (int, float)) or isinstance(report["fps"], bool) or not math.isfinite(report["fps"]) or report["fps"] <= 0:
        raise ScoringContractError("invalid recording time base")
    identity = {
        "protocol": PROTOCOL, "frame_count": report["frame_count"], "fps": float(report["fps"]),
        "video_sha256": {name: value["sha256"] for name, value in sources["videos"].items()},
    }
    if not identity["video_sha256"]:
        raise ScoringContractError("independent annotation requires bound camera videos")
    if any(not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value) for value in identity["video_sha256"].values()):
        raise ScoringContractError("annotation video hashes must be SHA-256 hex strings")
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return {**identity, "recording_id": digest}


def annotation_template(binding: dict) -> dict:
    return {
        "schema_version": 2, "kind": "fixation_independent_annotation", "binding": deepcopy(binding),
        "reviewer": "", "annotation_round": "tur-1", "athlete_id": "", "session_id": "", "reviewed_at": "",
        "blinded_to_predictions": False, "coverage_complete": False,
        "coverage": {"start_frame": 0, "end_frame": binding["frame_count"] - 1}, "events": [],
    }


def _frame(value: Any, count: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 0 <= value < count


def validate_annotation(annotation: dict, binding: dict) -> dict:
    fields = set(annotation_template(binding))
    if isinstance(annotation, dict) and annotation.get("schema_version") == 1:
        fields.remove("annotation_round")
    if not isinstance(annotation, dict) or set(annotation) != fields:
        raise ScoringContractError("independent annotation fields are invalid")
    if type(annotation["schema_version"]) is not int or annotation["schema_version"] not in {1, 2} or annotation["kind"] != "fixation_independent_annotation" or annotation["binding"] != binding:
        raise ScoringContractError("independent annotation recording/protocol mismatch")
    for key in ("reviewer", "athlete_id", "session_id", "reviewed_at"):
        if not isinstance(annotation[key], str) or not annotation[key].strip():
            raise ScoringContractError(f"annotation requires {key}")
    if annotation["schema_version"] == 2 and (not isinstance(annotation["annotation_round"], str) or not annotation["annotation_round"].strip()):
        raise ScoringContractError("annotation requires a distinct round code")
    for key in ("coverage_complete", "blinded_to_predictions"):
        if not isinstance(annotation[key], bool):
            raise ScoringContractError(f"annotation {key} must be boolean")
    coverage = annotation["coverage"]
    count = binding["frame_count"]
    if not isinstance(coverage, dict) or set(coverage) != {"start_frame", "end_frame"} or not all(_frame(coverage[k], count) for k in coverage) or coverage["start_frame"] > coverage["end_frame"]:
        raise ScoringContractError("invalid reviewed coverage")
    if not isinstance(annotation["events"], list):
        raise ScoringContractError("annotation events must be a list")
    ids = set()
    events = annotation["events"]
    for event in events:
        if not isinstance(event, dict) or set(event) != {"event_id", "kind", "start_frame", "end_frame", "note"}:
            raise ScoringContractError("invalid independent event fields")
        if not isinstance(event["event_id"], str) or not event["event_id"].strip() or event["event_id"] in ids:
            raise ScoringContractError("event IDs must be nonempty and unique")
        ids.add(event["event_id"])
        if event["kind"] not in EVENT_KINDS or not isinstance(event["note"], str):
            raise ScoringContractError("invalid independent event kind/note")
        if not all(_frame(event[k], count) for k in ("start_frame", "end_frame")) or not coverage["start_frame"] <= event["start_frame"] <= event["end_frame"] <= coverage["end_frame"]:
            raise ScoringContractError("event outside reviewed coverage")
    for first, second in combinations(events, 2):
        overlap = max(first["start_frame"], second["start_frame"]) <= min(first["end_frame"], second["end_frame"])
        if not overlap:
            continue
        kinds = {first["kind"], second["kind"]}
        if first["kind"] == second["kind"] or ("uncertain" in kinds) or ("no_hold" in kinds and bool(kinds & {"hold_onset", "hold_end"})):
            raise ScoringContractError("overlapping duplicate or contradictory annotation intervals")
    return annotation


def _annotation_round(annotation: dict) -> str:
    return annotation.get("annotation_round", "legacy-1")


def match_events(predictions: list[dict], references: list[dict], tolerance: int) -> dict:
    """Maximum-cardinality ordered matching, then minimum distance to reference interval.

    Each prediction can match one reference only. Duplicate predictions remain false
    positives. Reference intervals must not overlap within a semantic event kind.
    """
    if isinstance(tolerance, bool) or not isinstance(tolerance, int) or tolerance < 0:
        raise ScoringContractError("event tolerance must be a nonnegative integer")
    def bounds(row):
        return (row["frame"], row["frame"]) if "frame" in row else (row["start_frame"], row["end_frame"])

    pred = sorted(predictions, key=lambda row: bounds(row)[0])
    ref = sorted(references, key=lambda row: row["start_frame"])
    n, m = len(pred), len(ref)
    scores = [[(0, 0) for _ in range(m + 1)] for _ in range(n + 1)]
    previous = {}
    for i in range(n + 1):
        for j in range(m + 1):
            if i == j == 0:
                continue
            options = []
            if i:
                options.append((scores[i - 1][j], (i - 1, j, False)))
            if j:
                options.append((scores[i][j - 1], (i, j - 1, False)))
            if i and j:
                left, right = bounds(pred[i - 1])
                interval = ref[j - 1]
                distance = max(interval["start_frame"] - right, left - interval["end_frame"], 0)
                if distance <= tolerance:
                    hits, neg_error = scores[i - 1][j - 1]
                    options.append(((hits + 1, neg_error - distance), (i - 1, j - 1, True)))
            scores[i][j], previous[i, j] = max(options, key=lambda item: item[0])
    pairs, matched_p, matched_r = [], set(), set()
    i, j = n, m
    while i or j:
        pi, pj, matched = previous[i, j]
        if matched:
            left, right = bounds(pred[i - 1])
            interval = ref[j - 1]
            delta = left - interval["end_frame"] if left > interval["end_frame"] else right - interval["start_frame"] if right < interval["start_frame"] else 0
            pairs.append({"prediction_id": pred[i - 1]["event_id"], "reference_id": interval["event_id"], "delta_to_interval_frames": delta})
            matched_p.add(i - 1)
            matched_r.add(j - 1)
        i, j = pi, pj
    tp, fp, fn = len(pairs), n - len(pairs), m - len(pairs)
    errors = [abs(row["delta_to_interval_frames"]) for row in pairs]
    return {
        "true_positives": tp, "false_positives": fp, "false_negatives": fn,
        "precision": tp / n if n else None, "recall": tp / m if m else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else None,
        "timing_mae_frames": float(np.mean(errors)) if errors else None,
        "timing_median_frames": float(np.median(errors)) if errors else None,
        "timing_p95_frames": float(np.percentile(errors, 95)) if errors else None,
        "matches": list(reversed(pairs)),
        "unmatched_predictions": [row["event_id"] for k, row in enumerate(pred) if k not in matched_p],
        "unmatched_references": [row["event_id"] for k, row in enumerate(ref) if k not in matched_r],
    }


def _evaluation_scope(annotation: dict, other: dict | None = None) -> tuple[int, int, set[int]]:
    participants = [annotation] if other is None else [annotation, other]
    start = max(a["coverage"]["start_frame"] for a in participants)
    end = min(a["coverage"]["end_frame"] for a in participants)
    excluded = {frame for a in participants for e in a["events"] if e["kind"] == "uncertain"
                for frame in range(max(start, e["start_frame"]), min(end, e["end_frame"]) + 1)}
    return start, end, excluded


def _eligible(event: dict, start: int, end: int, excluded: set[int]) -> bool:
    return start <= event["start_frame"] <= event["end_frame"] <= end and not any(f in excluded for f in range(event["start_frame"], event["end_frame"] + 1))


def _guarded_exclusion(excluded: set[int], start: int, end: int, tolerance: int) -> set[int]:
    """Unknown events can match a prediction up to tolerance away from their region."""
    guarded = set()
    left, right = None, None
    for frame in sorted(excluded):
        first, last = max(start, frame - tolerance), min(end, frame + tolerance)
        if left is None:
            left, right = first, last
        elif first <= right + 1:
            right = max(right, last)
        else:
            guarded.update(range(left, right + 1))
            left, right = first, last
    if left is not None:
        guarded.update(range(left, right + 1))
    return guarded


def build_fixation_benchmark(report: dict, annotations: list[dict], tolerances_ms=(50.0, 100.0, 200.0)) -> dict:
    binding = recording_binding(report)
    ids = set()
    for candidate in report["fixation_candidates"]:
        identifier = candidate["candidate_episode_id"]
        if not isinstance(identifier, int) or isinstance(identifier, bool) or identifier in ids:
            raise ScoringContractError("benchmark candidate IDs must be unique integers")
        ids.add(identifier)
        if not _frame(candidate["peak_frame"], report["frame_count"]):
            raise ScoringContractError("benchmark peak frame outside recording")
        for field in PREDICTION_FIELDS.values():
            if candidate[field] is not None and not _frame(candidate[field], report["frame_count"]):
                raise ScoringContractError("benchmark prediction frame outside recording")
        if (candidate["status"] == "hold_observed") != (candidate["fixation_frame"] is not None):
            raise ScoringContractError("benchmark hold status and observation disagree")
        if (candidate["fixation_frame"] is None) != (candidate["hold_end_frame"] is None) or (candidate["fixation_frame"] is not None and candidate["hold_end_frame"] < candidate["fixation_frame"]):
            raise ScoringContractError("benchmark hold interval is invalid")
    if not tolerances_ms or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0 for x in tolerances_ms):
        raise ScoringContractError("benchmark tolerances must be finite nonnegative milliseconds")
    tolerances = sorted({int(math.floor(ms * report["fps"] / 1000 + 1e-9)) for ms in tolerances_ms})
    for a in annotations:
        validate_annotation(a, binding)
    reviewer_ids = {a["reviewer"].strip().casefold() for a in annotations}
    if len({(a["reviewer"].strip().casefold(), _annotation_round(a).strip().casefold()) for a in annotations}) != len(annotations):
        raise ScoringContractError("reviewer and annotation round must form a unique pair")
    if len({(a["athlete_id"], a["session_id"]) for a in annotations}) > 1:
        raise ScoringContractError("reviewers disagree on athlete/session identity")
    result = {
        "schema_version": 1, "recording_binding": binding, "status": "awaiting_independent_annotations",
        "reviewer_count": len(reviewer_ids), "annotation_count": len(annotations),
        "aggregation_unit": "annotation_round", "per_reviewer": [], "reviewer_agreement": [],
        "incomplete_reviewers": [a["reviewer"] for a in annotations if not a["coverage_complete"]],
        "nonblind_reviewers": [a["reviewer"] for a in annotations if not a["blinded_to_predictions"]],
        "deployment_gate": {"automatic_deployment_allowed": False, "external_accuracy_verified": False,
                            "reason": "requires_adjudicated_multisession_held_out_validation", "reviewer_identity_authenticated": False},
        "interpretation": "Timing error is distance to an annotation interval, not its centre. Completion and hold events are evaluated separately. Uncertain intervals are expanded by each evaluation tolerance; excluded reference/prediction/frame counts are reported. Self-declared blind reviews are not authenticated ground truth.",
    }
    complete = [a for a in annotations if a["coverage_complete"]]
    if complete:
        result["status"] = "single_recording_reference_comparison"
    for a in complete:
        start, end, excluded = _evaluation_scope(a)
        evaluable = max(0, end - start + 1 - len(excluded))
        row = {"reviewer": a["reviewer"], "annotation_round": _annotation_round(a), "blinded_to_predictions": a["blinded_to_predictions"],
               "athlete_id": a["athlete_id"], "session_id": a["session_id"], "coverage": a["coverage"],
               "excluded_uncertain_frames": len(excluded), "evaluable_frames": evaluable, "metrics": {}}
        for kind, field in PREDICTION_FIELDS.items():
            all_predictions = [{"event_id": str(c["candidate_episode_id"]), "frame": c[field]}
                               for c in report["fixation_candidates"] if c[field] is not None]
            values = []
            for tolerance in tolerances:
                guard = _guarded_exclusion(excluded, start, end, tolerance)
                metric_frames = max(0, end - start + 1 - len(guard))
                predictions = [p for p in all_predictions if start <= p["frame"] <= end and p["frame"] not in guard]
                references = [e for e in a["events"] if e["kind"] == kind and _eligible(e, start, end, guard)]
                widths = [e["end_frame"] - e["start_frame"] for e in references]
                metric = match_events(predictions, references, tolerance)
                metric.update(tolerance_frames=tolerance, tolerance_ms=tolerance * 1000 / report["fps"],
                              ignored_predictions=len(all_predictions) - len(predictions),
                              ignored_reference_events=sum(e["kind"] == kind for e in a["events"]) - len(references),
                              evaluable_frames=metric_frames, excluded_uncertainty_guard_frames=len(guard),
                              reference_interval_width_median_frames=float(np.median(widths)) if widths else None,
                              reference_interval_width_p95_frames=float(np.percentile(widths, 95)) if widths else None,
                              false_positives_per_minute=metric["false_positives"] * 60 * report["fps"] / metric_frames if metric_frames else None,
                              timing_mae_ms=metric["timing_mae_frames"] * 1000 / report["fps"] if metric["timing_mae_frames"] is not None else None)
                values.append(metric)
            row["metrics"][kind] = values
        row["unresolved_candidate_count_in_scope"] = sum(start <= c["peak_frame"] <= end and c["fixation_frame"] is None for c in report["fixation_candidates"])
        result["per_reviewer"].append(row)
    # Compare both interval bounds conservatively: overlapping intervals cost zero;
    # disjoint intervals measure the nearest possible disagreement, not consensus.
    for first, second in combinations(complete, 2):
        comparison_kind = "intra_rater_repeatability" if first["reviewer"].strip().casefold() == second["reviewer"].strip().casefold() else "inter_rater_agreement"
        if first["coverage"] != second["coverage"]:
            result["reviewer_agreement"].append({"reviewers": [first["reviewer"], second["reviewer"]],
                                                "annotation_rounds": [_annotation_round(first), _annotation_round(second)],
                                                "comparison_kind": comparison_kind,
                                                "status": "different_review_scopes", "metrics": {}})
            continue
        start, end, excluded = _evaluation_scope(first, second)
        pair = {"reviewers": [first["reviewer"], second["reviewer"]], "coverage": [start, end],
                "annotation_rounds": [_annotation_round(first), _annotation_round(second)],
                "comparison_kind": comparison_kind,
                "status": "descriptive_interval_agreement",
                "evaluable_frames": max(0, end - start + 1 - len(excluded)), "metrics": {},
                "interpretation": "Minimum interval-to-interval distance describes possible agreement. Wide intervals reduce timing resolution; this is not adjudicated consensus."}
        for kind in PREDICTION_FIELDS:
            pair["metrics"][kind] = []
            for tolerance in tolerances:
                guard = _guarded_exclusion(excluded, start, end, tolerance)
                predictions = [{"event_id": e["event_id"], "start_frame": e["start_frame"], "end_frame": e["end_frame"]}
                               for e in first["events"] if e["kind"] == kind and _eligible(e, start, end, guard)]
                references = [e for e in second["events"] if e["kind"] == kind and _eligible(e, start, end, guard)]
                pair["metrics"][kind].append({"tolerance_frames": tolerance, **match_events(predictions, references, tolerance),
                                             "evaluable_frames": max(0, end - start + 1 - len(guard)),
                                             "ignored_first_reviewer_events": sum(e["kind"] == kind for e in first["events"]) - len(predictions),
                                             "ignored_second_reviewer_events": sum(e["kind"] == kind for e in second["events"]) - len(references)})
        result["reviewer_agreement"].append(pair)
    return result


def validate_dataset_split(manifest: dict) -> dict:
    """Detect recording, athlete and session leakage before interpreting test metrics."""
    if not isinstance(manifest, dict) or set(manifest) != {"schema_version", "recordings"} or manifest["schema_version"] != 1 or not isinstance(manifest["recordings"], list):
        raise ScoringContractError("invalid fixation dataset split manifest")
    recordings, athletes, sessions = set(), {}, {}
    counts = {"development": 0, "validation": 0, "test": 0}
    for row in manifest["recordings"]:
        if not isinstance(row, dict) or set(row) != {"recording_id", "athlete_id", "session_id", "split"} or any(not isinstance(v, str) or not v.strip() for v in row.values()) or row["split"] not in counts:
            raise ScoringContractError("invalid dataset split recording")
        if row["recording_id"] in recordings:
            raise ScoringContractError("recording occurs more than once in dataset split")
        recordings.add(row["recording_id"])
        for group, key in ((athletes, "athlete_id"), (sessions, "session_id")):
            identifier = row[key].strip().casefold()
            if identifier in group and group[identifier] != row["split"]:
                raise ScoringContractError(f"dataset split leakage: {key}")
            group[identifier] = row["split"]
        counts[row["split"]] += 1
    return {"status": "declared_groups_separated" if all(counts.values()) else "incomplete_dataset_split",
            "recording_counts": counts, "athlete_count": len(athletes), "session_count": len(sessions),
            "interpretation": "Group IDs are user declarations. Split separation alone is not accuracy validation or deployment approval."}
