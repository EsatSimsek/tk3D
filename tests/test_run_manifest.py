from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.run_manifest import (
    build_run_manifest,
    environment_provenance,
    git_provenance,
    input_file_provenance,
    model_provenance,
    sha256_file,
    snapshot_file,
    validate_run_manifest,
    write_run_manifest,
)


def _unavailable_git(*args, **kwargs):
    raise FileNotFoundError("git unavailable")


def _manifest(tmp_path: Path, calibration: dict, configs: dict, models: list[dict]) -> dict:
    input_path = tmp_path / "video.mp4"
    input_path.write_bytes(b"video")
    return build_run_manifest(
        workspace_root=tmp_path,
        session_id="session",
        run_id="run-1",
        started_at="2026-08-25T00:00:00+00:00",
        completed_at="2026-08-25T00:00:01+00:00",
        calibration=calibration,
        configs=configs or {"test_config": calibration},
        models=models or [model_provenance("unavailable-test-model", None, None)],
        inputs=[input_path, input_path],
        argv=["run.py", "--run-id", "run-1"],
        command_runner=_unavailable_git,
    )


def test_run_manifest_records_snapshots_models_environment_and_writes_once(tmp_path) -> None:
    calibration_source = tmp_path / "calibration.json"
    calibration_source.write_text('{"schema_version":2,"cameras":[{"camera_id":"a"}]}', encoding="utf-8")
    calibration = snapshot_file(calibration_source, tmp_path / "run" / "calibration" / "cameras.json")
    calibration["schema_version"] = 2
    config_source = tmp_path / "model.yaml"
    config_source.write_text("pose2d: {}\n", encoding="utf-8")
    config = snapshot_file(config_source, tmp_path / "run" / "config" / "model.yaml")
    model_path = tmp_path / "model.pth"
    model_path.write_bytes(b"small-model-fixture")
    models = [model_provenance("ViTPose-Huge-WholeBody", model_path, config_source)]

    manifest = _manifest(tmp_path, calibration, {"model_config": config}, models)
    output = write_run_manifest(manifest, tmp_path / "run" / "json" / "run_manifest.json")

    assert json.loads(output.read_text(encoding="utf-8"))["schema_version"] == 1
    assert manifest["code"] == {"available": False, "sha": None, "branch": None, "dirty": None}
    assert manifest["environment"]["python"]["version"]
    assert manifest["models"][0]["artifact"]["sha256"] == sha256_file(model_path)
    assert len(manifest["inputs"]) == 1
    with pytest.raises(FileExistsError):
        write_run_manifest(manifest, output)


def test_calibration_snapshot_remains_bound_when_session_calibration_changes(tmp_path) -> None:
    source = tmp_path / "session" / "calibration.json"
    source.parent.mkdir()
    source.write_text('{"calibration":"X"}', encoding="utf-8")
    run_a = snapshot_file(source, tmp_path / "run-a" / "calibration.json")
    source.write_text('{"calibration":"Y"}', encoding="utf-8")
    run_b = snapshot_file(source, tmp_path / "run-b" / "calibration.json")

    assert run_a["sha256"] != run_b["sha256"]
    assert json.loads(Path(run_a["snapshot_path"]).read_text(encoding="utf-8"))["calibration"] == "X"
    assert json.loads(Path(run_b["snapshot_path"]).read_text(encoding="utf-8"))["calibration"] == "Y"


def test_manifest_rejects_calibration_checksum_snapshot_mismatch(tmp_path) -> None:
    source = tmp_path / "calibration.json"
    source.write_text('{"calibration":"X"}', encoding="utf-8")
    calibration = snapshot_file(source, tmp_path / "run" / "calibration.json")
    calibration["schema_version"] = 2
    manifest = _manifest(tmp_path, calibration, {}, [])
    Path(calibration["snapshot_path"]).write_text('{"calibration":"changed"}', encoding="utf-8")

    with pytest.raises(ValueError, match="checksum"):
        validate_run_manifest(manifest)


def test_git_provenance_is_graceful_when_git_is_unavailable(tmp_path) -> None:
    assert git_provenance(tmp_path, command_runner=_unavailable_git)["available"] is False


def test_git_provenance_records_dirty_checkout_without_machine_values(tmp_path) -> None:
    def runner(command, **kwargs):
        arguments = command[1:]
        values = {
            ("rev-parse", "HEAD"): "abc123\n",
            ("branch", "--show-current"): "main\n",
            ("status", "--porcelain"): " M src/example.py\n",
        }
        return type("Completed", (), {"stdout": values[tuple(arguments)]})()

    assert git_provenance(tmp_path, command_runner=runner) == {
        "available": True,
        "sha": "abc123",
        "branch": "main",
        "dirty": True,
    }


def test_environment_provenance_can_record_cuda_unavailable_deterministically() -> None:
    environment = environment_provenance(torch_module=None)

    assert environment["python"]["version"]
    assert environment["os"]["system"]
    assert environment["torch"]["available"] is False
    assert environment["torch"]["cuda_available"] is False
    assert environment["torch"]["gpu_names"] == []


def test_large_input_checksum_policy_is_explicit_without_hashing_fixture(tmp_path) -> None:
    input_path = tmp_path / "large.svo2"
    input_path.write_bytes(b"large-enough-for-test")

    record = input_file_provenance(input_path, large_input_threshold_bytes=1)

    assert record["checksum"] == {
        "algorithm": "sha256",
        "status": "omitted_large_file",
        "threshold_bytes": 1,
    }
