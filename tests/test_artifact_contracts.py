from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from src.artifact_contracts import (
    MAIN_3D_SCHEMA_VERSION,
    ArtifactCompatibility,
    ArtifactContractError,
    load_main_3d_artifact,
    load_run_bound_main_3d_artifact,
    validate_artifact_manifest_binding,
    validate_main_3d_artifact,
    validate_run_quality_artifact,
)
FIXTURES = Path(__file__).parent / "fixtures" / "contracts"


def _current_pose() -> dict:
    return deepcopy(json.loads((FIXTURES / "current_main_3d.json").read_text(encoding="utf-8")))


def test_current_main_3d_schema_is_accepted_and_round_trips(tmp_path) -> None:
    path = tmp_path / "vitpose_session_3d.json"
    payload = _current_pose()
    path.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")

    loaded, compatibility = load_main_3d_artifact(path)

    assert loaded == payload
    assert compatibility is ArtifactCompatibility.CURRENT


def test_unknown_future_main_3d_schema_is_rejected() -> None:
    payload = _current_pose()
    payload["schema_version"] = MAIN_3D_SCHEMA_VERSION + 1

    with pytest.raises(ArtifactContractError, match="Unsupported") as error:
        validate_main_3d_artifact(payload)

    assert error.value.compatibility is ArtifactCompatibility.UNSUPPORTED


def test_malformed_and_nonfinite_main_3d_artifacts_are_rejected() -> None:
    malformed = _current_pose()
    malformed["keypoints_3d_world"][0].pop()
    with pytest.raises(ArtifactContractError, match="133 keypoints"):
        validate_main_3d_artifact(malformed)

    nonfinite = _current_pose()
    nonfinite["keypoints_3d_world"][0][0][0] = float("nan")
    with pytest.raises(ArtifactContractError, match="Non-finite"):
        validate_main_3d_artifact(nonfinite)


def test_unversioned_legacy_main_3d_artifact_is_explicitly_supported() -> None:
    legacy = _current_pose()
    legacy.pop("schema_version")

    compatibility = validate_main_3d_artifact(legacy)

    assert compatibility is ArtifactCompatibility.LEGACY_SUPPORTED
    assert "schema_version" not in legacy


def test_current_run_bound_reader_requires_manifest_but_legacy_does_not(tmp_path) -> None:
    current_path = tmp_path / "current.json"
    current_path.write_text(json.dumps(_current_pose(), allow_nan=False), encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        load_run_bound_main_3d_artifact(current_path)

    legacy = _current_pose()
    legacy.pop("schema_version")
    legacy_path = tmp_path / "legacy.json"
    legacy_path.write_text(json.dumps(legacy, allow_nan=False), encoding="utf-8")
    _, compatibility = load_run_bound_main_3d_artifact(legacy_path)
    assert compatibility is ArtifactCompatibility.LEGACY_SUPPORTED


def test_quality_contract_rejects_official_score_without_external_evaluation() -> None:
    report = {
        "schema_version": 1,
        "session_id": "session",
        "run_id": "run-1",
        "provenance": _current_pose()["provenance"],
        "status": "passed",
        "ground_truth_accuracy_evaluated": False,
        "scoring_ready": False,
        "official_scoring_ready": False,
        "official_score": 9.5,
    }

    with pytest.raises(ArtifactContractError, match="official score"):
        validate_run_quality_artifact(report)


@pytest.mark.parametrize(
    ("fixture_name", "expected"),
    [
        ("current_quality.json", ArtifactCompatibility.CURRENT),
        ("legacy_quality.json", ArtifactCompatibility.LEGACY_SUPPORTED),
    ],
)
def test_quality_compatibility_matrix_accepts_supported_fixtures(
    fixture_name: str, expected: ArtifactCompatibility
) -> None:
    report = json.loads((FIXTURES / fixture_name).read_text(encoding="utf-8"))

    assert validate_run_quality_artifact(report) is expected


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda report: report.update(schema_version=2), "Unsupported"),
        (lambda report: report.pop("run_id"), "run_id"),
        (lambda report: report.update(mean_reprojection_error_px=float("inf")), "Non-finite"),
        (lambda report: report.update(status="unknown"), "status"),
    ],
)
def test_quality_compatibility_matrix_rejects_unsupported_or_malformed(mutation, message: str) -> None:
    report = json.loads((FIXTURES / "current_quality.json").read_text(encoding="utf-8"))
    mutation(report)

    with pytest.raises(ArtifactContractError, match=message):
        validate_run_quality_artifact(report)


def test_artifact_run_and_calibration_must_match_manifest() -> None:
    artifact = _current_pose()
    manifest = {
        "session_id": artifact["session_id"],
        "run_id": artifact["run_id"],
        "calibration": {
            "sha256": "a" * 64,
            "snapshot_path": artifact["provenance"]["calibration_snapshot"],
        },
    }
    validate_artifact_manifest_binding(artifact, manifest)

    manifest["run_id"] = "different-run"
    with pytest.raises(ArtifactContractError, match="run_id"):
        validate_artifact_manifest_binding(artifact, manifest)
