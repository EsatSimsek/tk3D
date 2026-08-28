from __future__ import annotations

import json
import math
from enum import StrEnum
from pathlib import Path
from typing import Any

from src.coordinate_system import ANALYSIS_COORDINATE_SYSTEM


MAIN_3D_SCHEMA_VERSION = 1
RUN_QUALITY_SCHEMA_VERSION = 1


class ArtifactCompatibility(StrEnum):
    CURRENT = "CURRENT"
    LEGACY_SUPPORTED = "LEGACY_SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"


class ArtifactContractError(ValueError):
    def __init__(self, message: str, compatibility: ArtifactCompatibility = ArtifactCompatibility.UNSUPPORTED):
        super().__init__(message)
        self.compatibility = compatibility


def load_main_3d_artifact(path: str | Path) -> tuple[dict[str, Any], ArtifactCompatibility]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactContractError(f"Cannot read main 3D artifact {source}: {exc}") from exc
    compatibility = validate_main_3d_artifact(payload)
    return payload, compatibility


def load_run_bound_main_3d_artifact(path: str | Path) -> tuple[dict[str, Any], ArtifactCompatibility]:
    source = Path(path).resolve()
    payload, compatibility = load_main_3d_artifact(source)
    if compatibility is ArtifactCompatibility.LEGACY_SUPPORTED:
        return payload, compatibility
    from src.artifact_io import load_json_object
    from src.run_manifest import validate_run_manifest

    manifest_reference = payload["provenance"]["run_manifest"]
    manifest_path = (source.parent / manifest_reference).resolve()
    manifest = validate_run_manifest(load_json_object(manifest_path))
    validate_artifact_manifest_binding(payload, manifest)
    return payload, compatibility


def validate_main_3d_artifact(payload: Any) -> ArtifactCompatibility:
    data = _mapping(payload, "main 3D artifact")
    version = _schema_compatibility(data, MAIN_3D_SCHEMA_VERSION, "main 3D artifact")
    _reject_nonfinite(data)
    _require_analysis_coordinates(data)
    points = data.get("keypoints_3d_world")
    if not isinstance(points, list):
        raise ArtifactContractError("keypoints_3d_world must be a JSON array")
    frame_count = len(points)
    for frame_index, frame in enumerate(points):
        if not isinstance(frame, list) or len(frame) != 133:
            raise ArtifactContractError(
                f"keypoints_3d_world[{frame_index}] must contain 133 keypoints"
            )
        for joint in frame:
            if not isinstance(joint, list) or len(joint) != 3:
                raise ArtifactContractError("Every keypoint must contain exactly x, y, z")
            if any(value is not None and not _is_finite_number(value) for value in joint):
                raise ArtifactContractError("Keypoint coordinates must be finite numbers or null")
    if version is ArtifactCompatibility.CURRENT:
        _require_nonempty_string(data, "session_id")
        _require_nonempty_string(data, "run_id")
        shape = data.get("shape", {}).get("keypoints_3d_world") if isinstance(data.get("shape"), dict) else None
        if shape != [frame_count, 133, 3]:
            raise ArtifactContractError(
                f"shape.keypoints_3d_world must equal [{frame_count}, 133, 3]"
            )
        _require_vector_length(data, "frame_indices", frame_count)
        _require_vector_length(data, "timestamps_sec", frame_count)
        _require_provenance(data)
        sample_fps = data.get("sample_fps")
        if not _is_finite_number(sample_fps) or float(sample_fps) <= 0:
            raise ArtifactContractError("sample_fps must be a positive finite number")
    return version


def validate_run_quality_artifact(payload: Any) -> ArtifactCompatibility:
    data = _mapping(payload, "run quality artifact")
    version = _schema_compatibility(data, RUN_QUALITY_SCHEMA_VERSION, "run quality artifact")
    _reject_nonfinite(data)
    if version is ArtifactCompatibility.CURRENT:
        _require_nonempty_string(data, "session_id")
        _require_nonempty_string(data, "run_id")
        if data.get("status") not in {"passed", "failed"}:
            raise ArtifactContractError("run quality status must be passed or failed")
        for field in ("ground_truth_accuracy_evaluated", "scoring_ready", "official_scoring_ready"):
            if not isinstance(data.get(field), bool):
                raise ArtifactContractError(f"{field} must be boolean")
        if data["official_scoring_ready"] and not data["scoring_ready"]:
            raise ArtifactContractError("official_scoring_ready cannot be true when scoring_ready is false")
        if data["ground_truth_accuracy_evaluated"] is False and data.get("official_score") is not None:
            raise ArtifactContractError("An official score cannot exist when external accuracy was not evaluated")
        _require_provenance(data)
    return version


def validate_artifact_manifest_binding(artifact: dict[str, Any], manifest: dict[str, Any]) -> None:
    if not isinstance(manifest, dict):
        raise ArtifactContractError("run manifest root must be a mapping")
    for field in ("session_id", "run_id"):
        if artifact.get(field) != manifest.get(field):
            raise ArtifactContractError(f"Artifact {field} does not match the run manifest")
    provenance = artifact.get("provenance")
    calibration = manifest.get("calibration")
    if not isinstance(provenance, dict) or not isinstance(calibration, dict):
        raise ArtifactContractError("Artifact/manifest provenance is incomplete")
    if provenance.get("calibration_sha256") != calibration.get("sha256"):
        raise ArtifactContractError("Artifact calibration checksum does not match the run manifest")
    artifact_snapshot = Path(str(provenance.get("calibration_snapshot", ""))).resolve()
    manifest_snapshot = Path(str(calibration.get("snapshot_path", ""))).resolve()
    if artifact_snapshot != manifest_snapshot:
        raise ArtifactContractError("Artifact calibration snapshot does not match the run manifest")


def _schema_compatibility(
    data: dict[str, Any], supported: int, label: str
) -> ArtifactCompatibility:
    if "schema_version" not in data:
        return ArtifactCompatibility.LEGACY_SUPPORTED
    version = data["schema_version"]
    if isinstance(version, bool) or not isinstance(version, int):
        raise ArtifactContractError(f"{label} schema_version must be an integer")
    if version != supported:
        raise ArtifactContractError(
            f"Unsupported {label} schema_version {version}; supported version is {supported}"
        )
    return ArtifactCompatibility.CURRENT


def _mapping(payload: Any, label: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ArtifactContractError(f"{label} root must be a mapping")
    return payload


def _require_analysis_coordinates(data: dict[str, Any]) -> None:
    if data.get("coordinate_system") != ANALYSIS_COORDINATE_SYSTEM:
        raise ArtifactContractError(
            "Main 3D artifacts must use TK3D analysis coordinates: meters, x right, y forward, z up"
        )


def _require_nonempty_string(data: dict[str, Any], field: str) -> None:
    if not isinstance(data.get(field), str) or not data[field].strip():
        raise ArtifactContractError(f"{field} must be a non-empty string")


def _require_vector_length(data: dict[str, Any], field: str, length: int) -> None:
    values = data.get(field)
    if not isinstance(values, list) or len(values) != length:
        raise ArtifactContractError(f"{field} must contain {length} values")


def _require_provenance(data: dict[str, Any]) -> None:
    provenance = data.get("provenance")
    if not isinstance(provenance, dict):
        raise ArtifactContractError("Current artifacts must declare provenance")
    for field in ("run_manifest", "calibration_snapshot", "calibration_sha256", "model_config_sha256"):
        if not isinstance(provenance.get(field), str) or not provenance[field]:
            raise ArtifactContractError(f"provenance.{field} must be a non-empty string")


def _is_finite_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


def _reject_nonfinite(value: Any, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ArtifactContractError(f"Non-finite JSON number at {path}")
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_nonfinite(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_nonfinite(item, f"{path}[{index}]")
