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
    checkpoint_valid: bool
    checkpoint_compatible: bool
    adapter_checkpoint_path: str | None
    adapter_checkpoint_exists: bool
    adapter_checkpoint_compatible: bool
    adapter_checkpoint_approved: bool
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
    adapter_checkpoint_path = _resolve_optional_path(section.get("adapter_checkpoint_path"), root)
    backend_available = is_backend_available(backend)
    config_exists = bool(config_path and config_path.exists())
    checkpoint_exists = bool(checkpoint_path and checkpoint_path.exists())
    checkpoint_valid = bool(checkpoint_path and _is_probable_checkpoint(checkpoint_path))
    checkpoint_compatible = _is_checkpoint_compatible(section, checkpoint_path) if checkpoint_valid else False
    adapter_checkpoint_exists = bool(adapter_checkpoint_path and adapter_checkpoint_path.exists())
    if adapter_checkpoint_exists:
        adapter_checkpoint_compatible, adapter_checkpoint_approved = _inspect_adapter_checkpoint(
            section, adapter_checkpoint_path
        )
    else:
        adapter_checkpoint_compatible = adapter_checkpoint_path is None
        adapter_checkpoint_approved = adapter_checkpoint_path is None
    allow_unapproved_adapter = section.get("allow_unapproved_adapter") is True

    missing = []
    if not backend_available:
        missing.append(f"backend '{backend}' is not importable")
    if not config_exists:
        missing.append("config file is missing")
    if not checkpoint_exists:
        missing.append("checkpoint file is missing")
    elif not checkpoint_valid:
        missing.append("checkpoint file does not look like a valid model checkpoint")
    elif not checkpoint_compatible:
        missing.append("checkpoint file is not compatible with the configured model")
    if adapter_checkpoint_path is not None and not adapter_checkpoint_exists:
        missing.append("adapter checkpoint file is missing")
    elif not adapter_checkpoint_compatible:
        missing.append("adapter checkpoint is not compatible with the configured model")
    elif not adapter_checkpoint_approved and not allow_unapproved_adapter:
        missing.append("adapter checkpoint has not passed held-out 3D approval")

    return ModelRuntimeStatus(
        backend=backend,
        model_name=model_name,
        config_path=str(config_path) if config_path else None,
        checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
        config_exists=config_exists,
        checkpoint_exists=checkpoint_exists,
        checkpoint_valid=checkpoint_valid,
        checkpoint_compatible=checkpoint_compatible,
        adapter_checkpoint_path=str(adapter_checkpoint_path) if adapter_checkpoint_path else None,
        adapter_checkpoint_exists=adapter_checkpoint_exists,
        adapter_checkpoint_compatible=adapter_checkpoint_compatible,
        adapter_checkpoint_approved=adapter_checkpoint_approved,
        backend_available=backend_available,
        ready=not missing,
        message="ready" if not missing else "; ".join(missing),
    )


def is_backend_available(backend: str) -> bool:
    if backend == "tk3d_vitpose_plus":
        return importlib.util.find_spec("torch") is not None and importlib.util.find_spec("timm") is not None
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


def _is_probable_checkpoint(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    if path.stat().st_size < 1_000_000:
        return False
    try:
        header = path.read_bytes()[:256].lstrip().lower()
    except OSError:
        return False
    return not (header.startswith(b"<!doctype html") or header.startswith(b"<html"))


def _is_checkpoint_compatible(section: dict[str, Any], path: Path | None) -> bool:
    model_name = str(section.get("model_name", "")).lower()
    if "vitpose" not in model_name:
        return True
    if path is None:
        return False
    expected_keypoints = int(section.get("keypoint_count", 133))
    try:
        import torch

        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            checkpoint = torch.load(path, map_location="cpu")
    except Exception:
        return False
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else {}
    head = state_dict.get("associate_keypoint_heads.4.final_layer.weight")
    if head is None:
        head = state_dict.get("head.final_layer.weight")
    if head is None:
        return False
    return bool(getattr(head, "shape", [None])[0] == expected_keypoints)


def _inspect_adapter_checkpoint(
    section: dict[str, Any], path: Path | None
) -> tuple[bool, bool]:
    if path is None:
        return True, True
    expected_keypoints = int(section.get("keypoint_count", 133))
    try:
        import torch

        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            checkpoint = torch.load(path, map_location="cpu")
    except Exception:
        return False, False
    state_dict = checkpoint.get("head_state_dict", checkpoint) if isinstance(checkpoint, dict) else {}
    weight = state_dict.get("final_layer.weight")
    head_compatible = bool(getattr(weight, "shape", [None])[0] == expected_keypoints)
    offsets = checkpoint.get("heatmap_offsets_xy") if isinstance(checkpoint, dict) else None
    offset_compatible = bool(getattr(offsets, "shape", None) == (expected_keypoints, 2))
    metadata = checkpoint.get("metadata", {}) if isinstance(checkpoint, dict) else {}
    approved = isinstance(metadata, dict) and metadata.get("production_approved") is True
    return head_compatible or offset_compatible, approved
