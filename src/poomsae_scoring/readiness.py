from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from src.poomsae_scoring.contracts import (
    validate_movement_timeline,
    validate_poomsae_spec,
    validate_rule_pack,
)


def assess_accuracy_readiness(
    rule_pack: dict[str, Any],
    poomsae_spec: dict[str, Any],
    movement_timeline: dict[str, Any],
    *,
    workspace_root: str | Path,
    wholebody_diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Explain whether an immutable pose artifact may enter Accuracy scoring."""
    pack = validate_rule_pack(rule_pack)
    spec = validate_poomsae_spec(poomsae_spec)
    timeline = validate_movement_timeline(movement_timeline, spec)
    root = Path(workspace_root).resolve()
    blockers: list[dict[str, str]] = []

    def block(code: str, message: str) -> None:
        blockers.append({"code": code, "message": message})

    if pack["status"] != "active":
        block("rule_pack_not_active", "RulePack must be active.")
    if spec["rule_pack_id"] != pack["rule_pack_id"]:
        block("rule_pack_mismatch", "PoomsaeSpec references a different RulePack.")
    if spec["status"] != "active":
        block("poomsae_spec_not_active", "PoomsaeSpec must be active before deductions can be applied.")
    if spec["sequence_status"] != "active":
        block("movement_sequence_not_active", "The source-transcribed movement sequence is not active.")
    if spec["blocked_reasons"]:
        block("poomsae_spec_has_source_gaps", "PoomsaeSpec still records unresolved source or tolerance gaps.")
    if timeline["status"] != "complete":
        block("movement_timeline_not_complete", "MovementTimeline must be complete.")
    if timeline["coverage"]["recording_scope"] == "partial_sequence":
        block(
            "partial_source_recording",
            "The source recording contains only a declared prefix of the Poomsae sequence.",
        )
    elif len(timeline["segments"]) != len(spec["movements"]):
        block("movement_count_mismatch", "MovementTimeline does not contain every PoomsaeSpec movement.")
    if any(segment["label_status"] != "confirmed" for segment in timeline["segments"]):
        block("unconfirmed_timeline_labels", "Every movement interval must be confirmed.")
    if any(not segment["anchors"] for segment in timeline["segments"]):
        block("missing_phase_anchors", "Every movement interval needs at least one phase anchor.")

    pose_binding = _verify_pose_binding(timeline["source_binding"], root)
    if pose_binding["status"] != "verified":
        block(pose_binding["status"], pose_binding["message"])

    diagnostic_binding = _verify_wholebody_diagnostics(
        wholebody_diagnostics,
        timeline,
        expected_pose_sha256=timeline["source_binding"]["pose_file_sha256"],
    )
    if wholebody_diagnostics is not None and diagnostic_binding["status"] != "verified":
        block(diagnostic_binding["status"], diagnostic_binding["message"])

    ready = not blockers
    return {
        "schema_version": 1,
        "status": "ready" if ready else "blocked",
        "rule_scoring_ready": ready,
        "judge_calibrated_ready": False,
        "official_scoring_ready": False,
        "rule_pack": {"rule_pack_id": pack["rule_pack_id"], "version": pack["version"]},
        "poomsae_spec": {
            "poomsae_id": spec["poomsae_id"],
            "version": spec["version"],
            "status": spec["status"],
            "sequence_status": spec["sequence_status"],
        },
        "movement_timeline": {
            "timeline_id": timeline["timeline_id"],
            "status": timeline["status"],
            "movement_count": len(timeline["segments"]),
            "expected_movement_count": len(spec["movements"]),
            "recording_scope": timeline["coverage"]["recording_scope"],
            "missing_movement_ids": list(timeline["coverage"]["missing_movement_ids"]),
            "source_end_reason": timeline["coverage"]["source_end_reason"],
        },
        "pose_binding": pose_binding,
        "wholebody_diagnostic_binding": diagnostic_binding,
        "blockers": blockers,
        "interpretation": (
            "Ready means the source, schema, sequence and pose-binding gates passed for provisional "
            "rule scoring. It does not mean judge-calibrated or official scoring."
        ),
    }


def _verify_wholebody_diagnostics(
    report: dict[str, Any] | None,
    timeline: dict[str, Any],
    *,
    expected_pose_sha256: str | None,
) -> dict[str, Any]:
    if report is None:
        return {
            "status": "not_provided",
            "message": "WholeBody diagnostics were not supplied to this readiness check.",
        }
    if not isinstance(report, dict):
        return {"status": "wholebody_diagnostics_invalid", "message": "WholeBody diagnostics must be a mapping."}
    if report.get("movement_timeline_id") != timeline["timeline_id"]:
        return {
            "status": "wholebody_timeline_mismatch",
            "message": "WholeBody diagnostics reference a different MovementTimeline.",
        }
    contract = report.get("keypoint_contract", {})
    if contract.get("keypoint_count") != 133 or set(contract.get("groups_used", [])) != {
        "body17",
        "feet",
        "face",
        "left_hand",
        "right_hand",
    }:
        return {
            "status": "wholebody_contract_mismatch",
            "message": "WholeBody diagnostics do not prove use of the required 133-point groups.",
        }
    report_pose_sha256 = report.get("bindings", {}).get("pose", {}).get("sha256")
    if expected_pose_sha256 is None or report_pose_sha256 != expected_pose_sha256:
        return {
            "status": "wholebody_pose_binding_mismatch",
            "message": "WholeBody diagnostics do not bind to the exact pose artifact used by the timeline.",
        }
    if report.get("numeric_score_enabled") is not False or report.get("accuracy_score") is not None:
        return {
            "status": "wholebody_fail_closed_contract_broken",
            "message": "WholeBody diagnostics unexpectedly enabled or emitted a numeric score.",
        }
    coverage = report.get("coverage", {})
    if coverage.get("coverage_gate_passed") is not True:
        return {
            "status": "wholebody_metric_coverage_below_minimum",
            "message": "WholeBody metric coverage does not pass its declared minimum quality gate.",
            "measurement_coverage_ratio": coverage.get("measurement_coverage_ratio"),
            "minimum_required_ratio": coverage.get("minimum_required_ratio"),
        }
    return {
        "status": "verified",
        "message": "WholeBody diagnostics match the timeline, pose hash, 133-point contract and coverage gate.",
        "measurement_coverage_ratio": coverage.get("measurement_coverage_ratio"),
        "minimum_required_ratio": coverage.get("minimum_required_ratio"),
    }


def _verify_pose_binding(source_binding: dict[str, Any], root: Path) -> dict[str, Any]:
    relative_or_absolute = Path(source_binding["pose_file"])
    pose_path = relative_or_absolute if relative_or_absolute.is_absolute() else root / relative_or_absolute
    pose_path = pose_path.resolve()
    try:
        pose_path.relative_to(root)
    except ValueError:
        return {
            "status": "pose_file_outside_workspace",
            "message": "The bound pose file resolves outside the workspace.",
            "pose_file": str(pose_path),
            "expected_sha256": source_binding["pose_file_sha256"],
            "actual_sha256": None,
        }
    if not pose_path.is_file():
        return {
            "status": "pose_file_missing",
            "message": "The bound pose file does not exist.",
            "pose_file": str(pose_path),
            "expected_sha256": source_binding["pose_file_sha256"],
            "actual_sha256": None,
        }
    expected_sha256 = source_binding["pose_file_sha256"]
    actual_sha256 = _sha256(pose_path)
    if expected_sha256 is None or actual_sha256 != expected_sha256.lower():
        return {
            "status": "pose_sha256_mismatch",
            "message": "The bound pose file SHA-256 does not match the MovementTimeline.",
            "pose_file": str(pose_path),
            "expected_sha256": expected_sha256,
            "actual_sha256": actual_sha256,
        }
    return {
        "status": "verified",
        "message": "The bound pose file exists inside the workspace and its SHA-256 matches.",
        "pose_file": str(pose_path),
        "expected_sha256": expected_sha256,
        "actual_sha256": actual_sha256,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
