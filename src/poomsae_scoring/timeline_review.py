"""Content and scope checks for recorded timeline video reviews.

``verified`` means this record is bound to the current timeline and requested
scope. It does not authenticate a person, verify external evidence-file bytes,
or establish independent biomechanical accuracy.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any

from src.poomsae_scoring.contracts import ScoringContractError, validate_timeline_review_record


def timeline_review_digest(timeline: dict[str, Any]) -> str:
    """Hash all timeline content except its review, without modifying input.

    Normalize the two numeric fields normalized by the timeline loader so a
    YAML integer FPS or confidence has the same meaning before and after load.
    """
    if not isinstance(timeline, dict):
        raise ScoringContractError("timeline review digest requires a mapping")
    content = deepcopy({key: value for key, value in timeline.items() if key != "review"})
    if type(content.get("fps")) in (int, float):
        content["fps"] = float(content["fps"])
    for segment in content.get("segments", []):
        if isinstance(segment, dict) and type(segment.get("confidence")) in (int, float):
            segment["confidence"] = float(segment["confidence"])
    try:
        encoded = json.dumps(content, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise ScoringContractError("timeline review digest requires finite JSON content") from exc
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def timeline_review_status(
    timeline: dict[str, Any], movement_id: str | None = None, phase_id: str | None = None,
) -> dict[str, Any]:
    """Fail closed unless a recorded video review covers this exact content.

    A phase query covers that anchor only. A movement query also requires its
    boundaries and every existing anchor. A timeline query requires every
    segment, including boundaries and anchors. The input is never modified.
    """
    def result(reason: str, *, verified: bool = False) -> dict[str, Any]:
        return {
            "status": "verified" if verified else "unverified",
            "reason": reason,
            "movement_id": movement_id,
            "phase_id": phase_id,
            "verification_scope": "recorded_video_review_content_and_scope",
            "reviewer_identity_authenticated": False,
            "independent_accuracy_verified": False,
        }

    if not isinstance(timeline, dict):
        return result("invalid_timeline")
    if phase_id is not None and movement_id is None:
        return result("phase_query_requires_movement")
    if "review" not in timeline:
        return result("missing_review_record")
    segments = timeline.get("segments")
    if not isinstance(segments, list) or not segments:
        return result("missing_timeline_segments")
    try:
        review = validate_timeline_review_record(timeline["review"], timeline)
        digest = timeline_review_digest(timeline)
    except (ScoringContractError, KeyError, TypeError, ValueError):
        return result("invalid_review_record")
    if review["timeline_sha256"].lower() != digest:
        return result("review_timeline_digest_mismatch")
    source = timeline.get("source_binding")
    if not isinstance(source, dict) or not isinstance(source.get("pose_file_sha256"), str) or len(source["pose_file_sha256"]) != 64:
        return result("missing_source_pose_digest")
    if timeline.get("label_source") not in {"manual", "manual_reviewed_automatic"}:
        return result("timeline_label_source_not_reviewed")
    selected = [segment for segment in segments if movement_id is None or segment["movement_id"] == movement_id]
    if not selected:
        return result("movement_not_present_in_timeline")
    scope = {item["movement_id"]: item for item in review["scope"]}
    for segment in selected:
        if segment.get("label_status") != "confirmed":
            return result("timeline_segment_not_confirmed")
        anchors = segment.get("anchors")
        if not isinstance(anchors, dict) or not anchors:
            return result("missing_phase_anchors")
        item = scope.get(segment["movement_id"])
        if item is None:
            return result("movement_not_in_review_scope")
        if phase_id is not None:
            if phase_id not in anchors:
                return result("phase_not_present_in_timeline")
            if phase_id not in item["phases"]:
                return result("phase_not_in_review_scope")
        else:
            if not item["boundaries_reviewed"]:
                return result("movement_boundaries_not_reviewed")
            if not set(anchors).issubset(item["phases"]):
                return result("phase_not_in_review_scope")
    return result("content_bound_video_review_covers_requested_scope", verified=True)
