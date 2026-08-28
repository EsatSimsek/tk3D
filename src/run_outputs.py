from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

from src.artifact_io import load_json_object, write_json_atomic, write_json_exclusive


_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")


def create_run_output_tree(
    output_root: str | Path,
    session_id: str,
    run_id: str | None = None,
) -> tuple[str, dict[str, Path]]:
    _validate_component(session_id, "session_id")
    identifier = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    _validate_component(identifier, "run_id")
    root = Path(output_root).resolve() / session_id / "runs" / identifier
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Run output already exists and will not be overwritten: {root}")
    paths = {
        "root": root,
        "videos": root / "videos",
        "figures": root / "figures",
        "csv": root / "csv",
        "json": root / "json",
        "calibration": root / "calibration",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    initialize_run_state(root, session_id, identifier)
    return identifier, paths


def initialize_run_state(run_root: str | Path, session_id: str, run_id: str) -> Path:
    _validate_component(session_id, "session_id")
    _validate_component(run_id, "run_id")
    root = Path(run_root).resolve()
    return write_json_exclusive(
        root / "run_state.json",
        _run_state_payload(session_id, run_id, "preparing"),
    )


def mark_run_running(run_root: str | Path, session_id: str, run_id: str) -> Path:
    return _update_run_state(run_root, session_id, run_id, "running")


def mark_run_failed(run_root: str | Path, session_id: str, run_id: str, error: str) -> Path:
    if not error.strip():
        raise ValueError("failed run error must be non-empty")
    return _update_run_state(run_root, session_id, run_id, "failed", error=error)


def mark_run_completed(run_root: str | Path, session_id: str, run_id: str) -> Path:
    return _update_run_state(run_root, session_id, run_id, "completed")


def mark_run_complete(output_root: str | Path, session_id: str, run_id: str, run_root: Path) -> Path:
    _validate_component(session_id, "session_id")
    _validate_component(run_id, "run_id")
    expected_root = (Path(output_root).resolve() / session_id / "runs" / run_id).resolve()
    if Path(run_root).resolve() != expected_root:
        raise ValueError(f"run_root does not match session_id/run_id: {run_root}")
    mark_run_completed(run_root, session_id, run_id)
    marker = Path(output_root).resolve() / session_id / "latest_run.json"
    write_json_atomic(
        marker,
        {"run_id": run_id, "run_root": str(run_root.resolve()), "status": "complete"},
    )
    return marker


def resolve_latest_run(output_root: str | Path, session_id: str) -> Path:
    _validate_component(session_id, "session_id")
    marker = Path(output_root).resolve() / session_id / "latest_run.json"
    if not marker.exists():
        raise FileNotFoundError(f"No completed run marker found: {marker}")
    raw = load_json_object(marker)
    root = Path(raw["run_root"]).resolve()
    expected_parent = (Path(output_root).resolve() / session_id / "runs").resolve()
    if expected_parent not in root.parents or raw.get("status") != "complete" or not root.exists():
        raise ValueError(f"Invalid latest run marker: {marker}")
    state_path = root / "run_state.json"
    if state_path.is_file() and load_json_object(state_path).get("status") != "completed":
        raise ValueError(f"Latest run is not completed according to its lifecycle state: {state_path}")
    return root


def _update_run_state(
    run_root: str | Path,
    session_id: str,
    run_id: str,
    status: str,
    *,
    error: str | None = None,
) -> Path:
    _validate_component(session_id, "session_id")
    _validate_component(run_id, "run_id")
    root = Path(run_root).resolve()
    state_path = root / "run_state.json"
    if not state_path.is_file():
        raise FileNotFoundError(f"Run lifecycle state is missing: {state_path}")
    current = load_json_object(state_path)
    if current.get("session_id") != session_id or current.get("run_id") != run_id:
        raise ValueError(f"Run lifecycle identity mismatch: {state_path}")
    return write_json_atomic(state_path, _run_state_payload(session_id, run_id, status, error=error))


def _run_state_payload(session_id: str, run_id: str, status: str, *, error: str | None = None) -> dict:
    if status not in {"preparing", "running", "completed", "failed"}:
        raise ValueError(f"Unsupported run lifecycle status: {status}")
    payload = {
        "schema_version": 1,
        "session_id": session_id,
        "run_id": run_id,
        "status": status,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if error is not None:
        payload["error"] = error
    return payload


def _validate_component(value: str, label: str) -> None:
    if not _SAFE_RUN_ID.fullmatch(value):
        raise ValueError(f"{label} must contain only letters, numbers, dot, underscore, or hyphen")
