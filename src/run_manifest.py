from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from src.artifact_io import sha256_file, write_json_exclusive


RUN_MANIFEST_SCHEMA_VERSION = 1
DEFAULT_LARGE_INPUT_THRESHOLD_BYTES = 512 * 1024 * 1024
_AUTO_TORCH = object()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def snapshot_file(source: str | Path, destination: str | Path) -> dict[str, Any]:
    source_path = Path(source).resolve()
    destination_path = Path(destination).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Snapshot source does not exist: {source_path}")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with source_path.open("rb") as reader, destination_path.open("xb") as writer:
        for chunk in iter(lambda: reader.read(1024 * 1024), b""):
            writer.write(chunk)
    source_hash = sha256_file(source_path)
    snapshot_hash = sha256_file(destination_path)
    if source_hash != snapshot_hash:
        raise RuntimeError(f"Snapshot checksum mismatch: {source_path} -> {destination_path}")
    return {
        "source_path": str(source_path),
        "snapshot_path": str(destination_path),
        "sha256": source_hash,
        "size_bytes": source_path.stat().st_size,
    }


def build_run_manifest(
    *,
    workspace_root: str | Path,
    session_id: str,
    run_id: str,
    started_at: str,
    completed_at: str,
    calibration: dict[str, Any],
    configs: dict[str, dict[str, Any]],
    models: list[dict[str, Any]],
    inputs: Sequence[str | Path],
    argv: Sequence[str] | None = None,
    large_input_threshold_bytes: int = DEFAULT_LARGE_INPUT_THRESHOLD_BYTES,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    root = Path(workspace_root).resolve()
    manifest = {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "session_id": session_id,
        "run_id": run_id,
        "code": git_provenance(root, command_runner=command_runner),
        "invocation": {
            "executable": str(Path(sys.executable).resolve()),
            "argv": list(sys.argv if argv is None else argv),
            "started_at": started_at,
            "completed_at": completed_at,
        },
        "environment": environment_provenance(),
        "models": models,
        "calibration": calibration,
        "configs": configs,
        "inputs": [
            input_file_provenance(path, large_input_threshold_bytes=large_input_threshold_bytes)
            for path in _unique_paths(inputs)
        ],
        "large_file_checksum_policy": {
            "algorithm": "sha256",
            "hash_files_at_or_below_bytes": large_input_threshold_bytes,
            "larger_inputs": "metadata_only",
            "model_artifacts": "always_sha256_once_per_run",
            "snapshots": "always_sha256",
        },
    }
    validate_run_manifest(manifest)
    return manifest


def write_run_manifest(manifest: dict[str, Any], path: str | Path) -> Path:
    validate_run_manifest(manifest)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    return write_json_exclusive(destination, manifest, sort_keys=True)


def validate_run_manifest(manifest: Any) -> dict[str, Any]:
    if not isinstance(manifest, dict):
        raise ValueError("run manifest root must be a mapping")
    if manifest.get("schema_version") != RUN_MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"run manifest schema_version must be {RUN_MANIFEST_SCHEMA_VERSION}")
    for field in ("session_id", "run_id"):
        if not isinstance(manifest.get(field), str) or not manifest[field]:
            raise ValueError(f"run manifest {field} must be a non-empty string")
    invocation = manifest.get("invocation")
    if not isinstance(invocation, dict) or not all(
        isinstance(invocation.get(field), str) and invocation[field]
        for field in ("executable", "started_at", "completed_at")
    ):
        raise ValueError("run manifest invocation is incomplete")
    calibration = manifest.get("calibration")
    if not isinstance(calibration, dict):
        raise ValueError("run manifest calibration must be a mapping")
    if isinstance(calibration.get("schema_version"), bool) or not isinstance(
        calibration.get("schema_version"), int
    ):
        raise ValueError("run manifest calibration schema_version must be an integer")
    _validate_snapshot_record(calibration, "calibration")
    configs = manifest.get("configs")
    if not isinstance(configs, dict) or not configs:
        raise ValueError("run manifest configs must be a non-empty mapping")
    for name, record in configs.items():
        if not isinstance(record, dict):
            raise ValueError(f"run manifest config {name} must be a mapping")
        _validate_snapshot_record(record, f"config {name}")
    models = manifest.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("run manifest models must be a non-empty list")
    if any(not isinstance(model, dict) or not isinstance(model.get("logical_name"), str) for model in models):
        raise ValueError("run manifest model records require logical_name")
    if not isinstance(manifest.get("inputs"), list) or not manifest["inputs"]:
        raise ValueError("run manifest inputs must be a non-empty list")
    if not isinstance(manifest.get("code"), dict) or not isinstance(manifest.get("environment"), dict):
        raise ValueError("run manifest code/environment provenance is incomplete")
    json.dumps(manifest, allow_nan=False)
    return manifest


def model_provenance(logical_name: str, artifact_path: str | Path | None, config_path: str | Path | None) -> dict[str, Any]:
    return {
        "logical_name": logical_name,
        "artifact": _always_hashed_file(artifact_path),
        "config": _always_hashed_file(config_path),
    }


def input_file_provenance(
    path: str | Path,
    *,
    large_input_threshold_bytes: int = DEFAULT_LARGE_INPUT_THRESHOLD_BYTES,
) -> dict[str, Any]:
    source = Path(path).resolve()
    record: dict[str, Any] = {"path": str(source), "exists": source.is_file()}
    if not source.is_file():
        record["checksum"] = {"algorithm": "sha256", "status": "unavailable"}
        return record
    stat = source.stat()
    record.update({"size_bytes": stat.st_size, "modified_time_ns": stat.st_mtime_ns})
    if stat.st_size <= large_input_threshold_bytes:
        record["checksum"] = {"algorithm": "sha256", "status": "computed", "value": sha256_file(source)}
    else:
        record["checksum"] = {
            "algorithm": "sha256",
            "status": "omitted_large_file",
            "threshold_bytes": large_input_threshold_bytes,
        }
    return record


def git_provenance(
    root: str | Path,
    *,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    workspace = Path(root).resolve()
    try:
        sha = _git_value(command_runner, workspace, "rev-parse", "HEAD")
        branch = _git_value(command_runner, workspace, "branch", "--show-current") or None
        status = _git_value(command_runner, workspace, "status", "--porcelain")
    except (OSError, subprocess.SubprocessError):
        return {"available": False, "sha": None, "branch": None, "dirty": None}
    return {"available": True, "sha": sha, "branch": branch, "dirty": bool(status)}


def environment_provenance(*, torch_module: Any = _AUTO_TORCH) -> dict[str, Any]:
    packages = {}
    for distribution in ("numpy", "opencv-python", "scipy", "pandas", "torch", "torchvision", "rfdetr", "supervision", "timm"):
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[distribution] = None
    torch_info: dict[str, Any] = {"available": False, "version": packages["torch"], "cuda_available": False, "cuda_version": None, "gpu_names": []}
    if torch_module is _AUTO_TORCH:
        try:
            import torch
        except (ImportError, OSError):
            torch_module = None
        else:
            torch_module = torch
    if torch_module is not None:
        torch_info["available"] = True
        torch_info["cuda_version"] = torch_module.version.cuda
        try:
            torch_info["cuda_available"] = bool(torch_module.cuda.is_available())
            if torch_info["cuda_available"]:
                torch_info["gpu_names"] = [
                    torch_module.cuda.get_device_name(index) for index in range(torch_module.cuda.device_count())
                ]
        except (RuntimeError, AssertionError):
            torch_info["cuda_available"] = False
    return {
        "python": {"version": platform.python_version(), "implementation": platform.python_implementation()},
        "os": {"system": platform.system(), "release": platform.release(), "version": platform.version(), "machine": platform.machine()},
        "packages": packages,
        "torch": torch_info,
    }


def _git_value(command_runner: Callable[..., subprocess.CompletedProcess[str]], root: Path, *args: str) -> str:
    completed = command_runner(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True, encoding="utf-8"
    )
    return completed.stdout.strip()


def _validate_snapshot_record(record: dict[str, Any], label: str) -> None:
    snapshot_path = Path(str(record.get("snapshot_path", "")))
    expected_hash = record.get("sha256")
    if not snapshot_path.is_file() or not isinstance(expected_hash, str):
        raise ValueError(f"run manifest {label} snapshot/hash is incomplete")
    if sha256_file(snapshot_path) != expected_hash:
        raise ValueError(f"run manifest {label} checksum does not match its snapshot")


def _always_hashed_file(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {"path": None, "exists": False, "sha256": None, "size_bytes": None}
    source = Path(path).resolve()
    if not source.is_file():
        return {"path": str(source), "exists": False, "sha256": None, "size_bytes": None}
    return {"path": str(source), "exists": True, "sha256": sha256_file(source), "size_bytes": source.stat().st_size}


def _unique_paths(paths: Sequence[str | Path]) -> list[Path]:
    unique: dict[str, Path] = {}
    for path in paths:
        resolved = Path(path).resolve()
        unique.setdefault(str(resolved).casefold(), resolved)
    return list(unique.values())
