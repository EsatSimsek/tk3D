from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from src.artifact_contracts import validate_run_quality_artifact
from src.artifact_io import sha256_file


AUTHORIZATION_ALGORITHM = "tk3d_ground_truth_bound_scoring_authorization_v1"
PASSED_VALIDATION_STATUS = "passed_for_scoring_validation"


def file_fingerprint(path: str | Path, *, label: str | None = None) -> dict[str, object]:
    source = Path(path).resolve()
    return {
        "path": label if label is not None else str(source),
        "size_bytes": source.stat().st_size,
        "sha256": sha256_file(source),
    }


def build_scoring_authorization(
    *,
    prediction_path: str | Path,
    ground_truth_path: str | Path,
    validation_config_path: str | Path,
    validation_report_path: str | Path,
    validation_manifest_path: str | Path,
    frame_matches_path: str | Path,
    joint_map_path: str | Path | None = None,
) -> dict[str, Any]:
    prediction = Path(prediction_path).resolve()
    ground_truth = Path(ground_truth_path).resolve()
    validation_config = Path(validation_config_path).resolve()
    validation_report = Path(validation_report_path).resolve()
    validation_manifest = Path(validation_manifest_path).resolve()
    frame_matches = Path(frame_matches_path).resolve()

    prediction_payload = _read_json(prediction)
    truth_payload = _read_json(ground_truth)
    report_payload = _read_json(validation_report)
    manifest_payload = _read_json(validation_manifest)
    config_payload = yaml.safe_load(validation_config.read_text(encoding="utf-8")) or {}
    profile = config_payload.get("scoring_authorization")
    reasons: list[str] = []

    if not isinstance(profile, dict):
        profile = {}
        reasons.append("authorization_profile_missing")
    profile_id = str(profile.get("profile_id") or "")
    if profile.get("enabled") is not True:
        reasons.append("authorization_profile_disabled")
    if not profile_id:
        reasons.append("authorization_profile_id_missing")
    if profile.get("scoring_mode") != "provisional_not_official":
        reasons.append("unsupported_scoring_mode")

    validation = report_payload.get("validation")
    if not isinstance(validation, dict):
        validation = {}
        reasons.append("validation_payload_missing")
    quality_gates = validation.get("quality_gates")
    gates_passed = isinstance(quality_gates, dict) and bool(quality_gates) and all(
        value is True for value in quality_gates.values()
    )
    ground_truth_validated = (
        validation.get("status") == PASSED_VALIDATION_STATUS and gates_passed
    )
    if not ground_truth_validated:
        reasons.append("ground_truth_quality_gate_failed")

    mapped_joint_names = report_payload.get("mapped_joint_names")
    if not isinstance(mapped_joint_names, list):
        mapped_joint_names = []
        reasons.append("mapped_joint_names_missing")
    minimum_mapped_joints = int(profile.get("minimum_mapped_joints", 3))
    if len(mapped_joint_names) < minimum_mapped_joints:
        reasons.append("too_few_mapped_joints")

    matched_frame_count = int(report_payload.get("matched_frame_count", 0))
    evaluation_frame_count = int(validation.get("evaluation_frame_count", 0))
    if matched_frame_count <= 0 or matched_frame_count != evaluation_frame_count:
        reasons.append("matched_timeline_count_mismatch")

    report_prediction = report_payload.get("prediction")
    report_ground_truth = report_payload.get("ground_truth")
    if not _same_path(report_prediction, prediction):
        reasons.append("validation_report_prediction_mismatch")
    if not _same_path(report_ground_truth, ground_truth):
        reasons.append("validation_report_ground_truth_mismatch")

    allowed_datasets = profile.get("allowed_ground_truth_datasets", [])
    dataset_name = str(truth_payload.get("dataset") or "")
    if not isinstance(allowed_datasets, list) or not allowed_datasets:
        reasons.append("allowed_ground_truth_datasets_missing")
    elif dataset_name not in {str(value) for value in allowed_datasets}:
        reasons.append("ground_truth_dataset_not_allowed")

    bindings: dict[str, dict[str, object]] = {
        "prediction": file_fingerprint(prediction),
        "ground_truth": file_fingerprint(ground_truth),
        "validation_config": file_fingerprint(validation_config),
        "validation_report": file_fingerprint(validation_report),
        "validation_manifest": file_fingerprint(validation_manifest),
        "frame_matches": file_fingerprint(frame_matches),
    }
    if joint_map_path is not None:
        bindings["joint_map"] = file_fingerprint(joint_map_path)

    manifest_ok, manifest_reasons = _validate_manifest_bindings(
        manifest_payload,
        bindings,
        validation_report_name=validation_report.name,
        frame_matches_name=frame_matches.name,
    )
    if not manifest_ok:
        reasons.extend(manifest_reasons)

    require_internal_geometry = profile.get("require_internal_geometry_pass") is not False
    run_quality_path = prediction.parent / "run_quality_report.json"
    internal_geometry_ready = not require_internal_geometry
    if require_internal_geometry:
        if not run_quality_path.exists():
            reasons.append("run_quality_report_missing")
        else:
            bindings["run_quality_report"] = file_fingerprint(run_quality_path)
            run_quality = _read_json(run_quality_path)
            internal_geometry_ready = (
                run_quality.get("status") == "passed"
                and run_quality.get("quality_scope") == "internal_geometry_only"
                and run_quality.get("production_ready_calibration") is True
                and run_quality.get("session_id") == prediction_payload.get("session_id")
                and run_quality.get("run_id") == prediction_payload.get("run_id")
            )
            if not internal_geometry_ready:
                reasons.append("internal_geometry_quality_gate_failed")

    artifact_integrity_valid = manifest_ok and not any(
        reason.endswith("_mismatch") or reason.endswith("_missing")
        for reason in reasons
        if reason.startswith(("validation_", "manifest_", "matched_"))
    )
    scoring_infrastructure_ready = (
        not reasons
        and artifact_integrity_valid
        and internal_geometry_ready
        and ground_truth_validated
    )
    poomsae_rules_validated = profile.get("poomsae_rules_validated") is True
    unique_reasons = list(dict.fromkeys(reasons))
    return {
        "schema_version": 1,
        "authorization_algorithm": AUTHORIZATION_ALGORITHM,
        "decision": (
            "authorized_for_provisional_scoring"
            if scoring_infrastructure_ready
            else "denied"
        ),
        "scoring_ready": scoring_infrastructure_ready,
        "official_scoring_ready": scoring_infrastructure_ready and poomsae_rules_validated,
        "statuses": {
            "internal_geometry_ready": internal_geometry_ready,
            "ground_truth_validated": ground_truth_validated,
            "artifact_integrity_valid": artifact_integrity_valid,
            "scoring_infrastructure_ready": scoring_infrastructure_ready,
            "poomsae_rules_validated": poomsae_rules_validated,
        },
        "profile": {
            "profile_id": profile_id,
            "scoring_mode": profile.get("scoring_mode"),
            "minimum_mapped_joints": minimum_mapped_joints,
            "require_internal_geometry_pass": require_internal_geometry,
            "allowed_ground_truth_datasets": allowed_datasets,
        },
        "run": {
            "session_id": prediction_payload.get("session_id"),
            "run_id": prediction_payload.get("run_id"),
            "ground_truth_dataset": dataset_name,
            "ground_truth_sequence_id": truth_payload.get("sequence_id"),
        },
        "validation_summary": {
            "status": validation.get("status"),
            "matched_frame_count": matched_frame_count,
            "mapped_joint_count": len(mapped_joint_names),
            "mapped_joint_names": mapped_joint_names,
            "quality_gates": quality_gates,
            "failed_gates": validation.get("failed_gates", []),
            "mpjpe_mm": validation.get("mpjpe_mm"),
            "p95_error_mm": validation.get("p95_error_mm"),
            "angle_mae_deg": validation.get("angle_mae_deg"),
        },
        "bindings": bindings,
        "denial_reasons": unique_reasons,
        "interpretation": {
            "scoring_ready": (
                "The bound 3D prediction passed the configured external ground-truth "
                "and internal geometry gates for provisional scoring infrastructure."
            ),
            "official_scoring_ready": (
                "Additionally requires a separately validated poomsae rules and judge-label profile."
            ),
        },
    }


def verify_scoring_authorization(
    prediction_path: str | Path,
    authorization_path: str | Path,
) -> dict[str, Any]:
    prediction = Path(prediction_path).resolve()
    authorization_file = Path(authorization_path).resolve()
    if not authorization_file.exists():
        raise ValueError(f"Scoring authorization file not found: {authorization_file}")
    authorization = _read_json(authorization_file)
    if authorization.get("authorization_algorithm") != AUTHORIZATION_ALGORITHM:
        raise ValueError("Unsupported scoring authorization algorithm")
    if authorization.get("scoring_ready") is not True:
        reasons = authorization.get("denial_reasons") or ["authorization_denied"]
        raise ValueError(f"Scoring authorization is denied: {', '.join(str(value) for value in reasons)}")
    statuses = authorization.get("statuses")
    required_statuses = (
        "internal_geometry_ready",
        "ground_truth_validated",
        "artifact_integrity_valid",
        "scoring_infrastructure_ready",
    )
    if not isinstance(statuses, dict) or not all(statuses.get(name) is True for name in required_statuses):
        raise ValueError("Scoring authorization status fields are incomplete or false")

    bindings = authorization.get("bindings")
    if not isinstance(bindings, dict):
        raise ValueError("Scoring authorization bindings are missing")
    required_bindings = {
        "prediction",
        "ground_truth",
        "validation_config",
        "validation_report",
        "validation_manifest",
        "frame_matches",
    }
    missing = sorted(required_bindings - bindings.keys())
    if missing:
        raise ValueError(f"Scoring authorization bindings are incomplete: {', '.join(missing)}")
    _verify_binding(bindings["prediction"], expected_path=prediction)
    for name in sorted(bindings.keys() - {"prediction"}):
        _verify_binding(bindings[name])

    report_path = Path(str(bindings["validation_report"]["path"])).resolve()
    report = _read_json(report_path)
    validation = report.get("validation")
    if not isinstance(validation, dict) or validation.get("status") != PASSED_VALIDATION_STATUS:
        raise ValueError("The bound ground-truth validation report no longer passes")
    gates = validation.get("quality_gates")
    if not isinstance(gates, dict) or not gates or not all(value is True for value in gates.values()):
        raise ValueError("The bound ground-truth quality gates are incomplete or failed")
    if not _same_path(report.get("prediction"), prediction):
        raise ValueError("The bound validation report targets a different prediction")

    prediction_payload = _read_json(prediction)
    run = authorization.get("run")
    if not isinstance(run, dict):
        raise ValueError("Scoring authorization run identity is missing")
    if (
        run.get("session_id") != prediction_payload.get("session_id")
        or run.get("run_id") != prediction_payload.get("run_id")
    ):
        raise ValueError("Scoring authorization run identity does not match the prediction")
    if "run_quality_report" in bindings:
        quality = _read_json(Path(str(bindings["run_quality_report"]["path"])))
        validate_run_quality_artifact(quality)
        if (
            quality.get("status") != "passed"
            or quality.get("quality_scope") != "internal_geometry_only"
            or quality.get("production_ready_calibration") is not True
        ):
            raise ValueError("The bound internal geometry report no longer passes")
    return authorization


def _validate_manifest_bindings(
    manifest: dict[str, Any],
    bindings: dict[str, dict[str, object]],
    *,
    validation_report_name: str,
    frame_matches_name: str,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    inputs = manifest.get("inputs")
    outputs = manifest.get("outputs")
    if not isinstance(inputs, list) or not isinstance(outputs, list):
        return False, ["validation_manifest_invalid"]

    input_by_path = {
        str(Path(str(item.get("path"))).resolve()): item
        for item in inputs
        if isinstance(item, dict) and item.get("path")
    }
    for name in ("prediction", "ground_truth", "validation_config"):
        binding = bindings[name]
        manifest_item = input_by_path.get(str(Path(str(binding["path"])).resolve()))
        if not _fingerprints_equal(binding, manifest_item):
            reasons.append(f"manifest_{name}_mismatch")

    output_by_name = {
        str(item.get("path")): item
        for item in outputs
        if isinstance(item, dict) and item.get("path")
    }
    if not _fingerprints_equal(bindings["validation_report"], output_by_name.get(validation_report_name)):
        reasons.append("manifest_validation_report_mismatch")
    if not _fingerprints_equal(bindings["frame_matches"], output_by_name.get(frame_matches_name)):
        reasons.append("manifest_frame_matches_mismatch")
    return not reasons, reasons


def _verify_binding(binding: Any, expected_path: Path | None = None) -> None:
    if not isinstance(binding, dict) or not {"path", "size_bytes", "sha256"} <= binding.keys():
        raise ValueError("Scoring authorization contains an invalid artifact binding")
    path = Path(str(binding["path"])).resolve()
    if expected_path is not None and path != expected_path:
        raise ValueError("Scoring authorization prediction path does not match the requested input")
    if not path.exists():
        raise ValueError(f"A bound scoring artifact is missing: {path}")
    current = file_fingerprint(path)
    if not _fingerprints_equal(binding, current):
        raise ValueError(f"A bound scoring artifact changed after validation: {path}")


def _fingerprints_equal(first: Any, second: Any) -> bool:
    return (
        isinstance(first, dict)
        and isinstance(second, dict)
        and int(first.get("size_bytes", -1)) == int(second.get("size_bytes", -2))
        and str(first.get("sha256", "")) == str(second.get("sha256", "__missing__"))
    )


def _same_path(value: Any, expected: Path) -> bool:
    if not isinstance(value, str) or not value:
        return False
    return Path(value).resolve() == expected


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload
