from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path


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
    return identifier, paths


def mark_run_complete(output_root: str | Path, session_id: str, run_id: str, run_root: Path) -> Path:
    _validate_component(session_id, "session_id")
    _validate_component(run_id, "run_id")
    expected_root = (Path(output_root).resolve() / session_id / "runs" / run_id).resolve()
    if Path(run_root).resolve() != expected_root:
        raise ValueError(f"run_root does not match session_id/run_id: {run_root}")
    marker = Path(output_root).resolve() / session_id / "latest_run.json"
    marker.parent.mkdir(parents=True, exist_ok=True)
    temporary = marker.with_suffix(".tmp")
    temporary.write_text(
        json.dumps({"run_id": run_id, "run_root": str(run_root.resolve()), "status": "complete"}, indent=2),
        encoding="utf-8",
    )
    temporary.replace(marker)
    return marker


def resolve_latest_run(output_root: str | Path, session_id: str) -> Path:
    _validate_component(session_id, "session_id")
    marker = Path(output_root).resolve() / session_id / "latest_run.json"
    if not marker.exists():
        raise FileNotFoundError(f"No completed run marker found: {marker}")
    raw = json.loads(marker.read_text(encoding="utf-8"))
    root = Path(raw["run_root"]).resolve()
    expected_parent = (Path(output_root).resolve() / session_id / "runs").resolve()
    if expected_parent not in root.parents or raw.get("status") != "complete" or not root.exists():
        raise ValueError(f"Invalid latest run marker: {marker}")
    return root


def _validate_component(value: str, label: str) -> None:
    if not _SAFE_RUN_ID.fullmatch(value):
        raise ValueError(f"{label} must contain only letters, numbers, dot, underscore, or hyphen")
