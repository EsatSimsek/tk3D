from __future__ import annotations

import importlib.util
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ModelRuntimeStatus:
    backend: str
    model_name: str
    config_path: str | None
    checkpoint_path: str | None
    config_exists: bool
    checkpoint_exists: bool
    backend_available: bool
    ready: bool
    message: str


class ModelRuntimeError(RuntimeError):
    pass


def check_model_runtime(section: dict[str, Any], project_root: str | Path) -> ModelRuntimeStatus:
    root = Path(project_root)
    backend = str(section.get("backend", "mmpose"))
    model_name = str(section.get("model_name", "unknown"))
    config_path = _resolve_optional_path(section.get("config_path"), root)
    checkpoint_path = _resolve_optional_path(section.get("checkpoint_path"), root)
    backend_available = is_backend_available(backend)
    config_exists = bool(config_path and config_path.exists())
    checkpoint_exists = bool(checkpoint_path and checkpoint_path.exists())

    missing = []
    if not backend_available:
        missing.append(f"backend '{backend}' is not importable")
    if not config_exists:
        missing.append("config file is missing")
    if not checkpoint_exists:
        missing.append("checkpoint file is missing")

    return ModelRuntimeStatus(
        backend=backend,
        model_name=model_name,
        config_path=str(config_path) if config_path else None,
        checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
        config_exists=config_exists,
        checkpoint_exists=checkpoint_exists,
        backend_available=backend_available,
        ready=not missing,
        message="ready" if not missing else "; ".join(missing),
    )


def is_backend_available(backend: str) -> bool:
    if backend == "mmpose":
        return importlib.util.find_spec("mmpose") is not None
    return importlib.util.find_spec(backend) is not None


def require_ready(status: ModelRuntimeStatus) -> None:
    if not status.ready:
        raise ModelRuntimeError(f"{status.model_name} runtime is not ready: {status.message}")


def save_model_runtime_report(statuses: dict[str, ModelRuntimeStatus], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump({name: asdict(status) for name, status in statuses.items()}, file, indent=2)


def _resolve_optional_path(raw_path: Any, root: Path) -> Path | None:
    if not raw_path:
        return None
    path = Path(str(raw_path))
    return path if path.is_absolute() else root / path
