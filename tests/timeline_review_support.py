"""Synthetic review records for arithmetic/contract tests, never real annotations."""
from copy import deepcopy

from src.poomsae_scoring.timeline_review import timeline_review_digest


def synthetic_prefix_timeline(spec: dict, count: int) -> dict:
    movements = spec["movements"][:count]
    return {
        "schema_version": 2, "timeline_id": "synthetic-review-test-prefix",
        "poomsae_id": spec["poomsae_id"], "poomsae_version": spec["version"],
        "status": "draft", "label_source": "manual", "frame_index_space": "sample_index",
        "frame_count": count * 12, "fps": 60.0,
        "source_binding": {"session_id": "synthetic", "run_id": "synthetic",
                           "pose_file": "outputs/synthetic/pose.json", "pose_file_sha256": "a" * 64},
        "coverage": {"recording_scope": "partial_sequence",
                     "observed_movement_ids": [m["movement_id"] for m in movements],
                     "missing_movement_ids": [m["movement_id"] for m in spec["movements"][count:]],
                     "source_end_reason": "synthetic test prefix"},
        "segments": [
            {"sequence_index": i + 1, "movement_id": m["movement_id"],
             "start_frame": i * 12, "end_frame": i * 12 + 11,
             "anchors": {phase: i * 12 + j + 1 for j, phase in enumerate(m["phases"])},
             "confidence": 1.0, "label_status": "confirmed"}
            for i, m in enumerate(movements)
        ],
    }


def reviewed_timeline(timeline: dict) -> dict:
    payload = deepcopy(timeline)
    if payload["source_binding"]["pose_file_sha256"] is None:
        payload["source_binding"]["pose_file_sha256"] = "a" * 64
    payload["review"] = {
        "schema_version": 1,
        "reviewer": {"name": "SYNTHETIC TEST FIXTURE", "role": "contract test, not a real referee"},
        "reviewed_at": "2026-09-21T10:00:00+03:00",
        "method": "video_review",
        "evidence": {"reference": "synthetic-test-only", "sha256": "b" * 64},
        "timeline_sha256": timeline_review_digest(payload),
        "scope": [
            {"movement_id": segment["movement_id"], "boundaries_reviewed": True,
             "phases": list(segment["anchors"])}
            for segment in payload["segments"]
        ],
    }
    return payload
