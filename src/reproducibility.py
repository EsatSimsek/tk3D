from __future__ import annotations

import importlib
import importlib.util
import sys
import tempfile
import tomllib
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml

from src.artifact_io import sha256_file
from src.camera_calibration import PRODUCTION_CALIBRATION_MODES, load_calibration_bundle
from src.config_validation import validate_model_config
from src.run_manifest import environment_provenance


class ReproducibilityStatus(StrEnum):
    READY = "READY"
    PARTIALLY_READY = "PARTIALLY_READY"
    NOT_READY = "NOT_READY"


class RequirementCategory(StrEnum):
    INCLUDED_IN_REPOSITORY = "INCLUDED_IN_REPOSITORY"
    INSTALLABLE_DEPENDENCY = "INSTALLABLE_DEPENDENCY"
    EXTERNAL_MODEL = "EXTERNAL_MODEL"
    EXTERNAL_DATA = "EXTERNAL_DATA"
    EXTERNAL_CALIBRATION = "EXTERNAL_CALIBRATION"
    MACHINE_SPECIFIC = "MACHINE_SPECIFIC"
    OPTIONAL = "OPTIONAL"


@dataclass(frozen=True, slots=True)
class ReproducibilityCheck:
    name: str
    category: RequirementCategory
    scope: str
    status: str
    message: str
    path: str | None = None


@dataclass(frozen=True, slots=True)
class ReproducibilityReport:
    schema_version: int
    status: ReproducibilityStatus
    tier1_ready: bool
    current_active_ready: bool
    workflow: str
    environment: dict[str, Any]
    checks: tuple[ReproducibilityCheck, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        for check in payload["checks"]:
            check["category"] = check["category"].value
        return payload


def check_reproducibility(
    project_root: str | Path,
    *,
    model_config_path: str | Path = "config/model_config.yaml",
    active_profile_path: str | Path = "config/scoring/profiles/poomsae1_trimmed.yaml",
    output_root: str | Path = "outputs",
    user_home: str | Path | None = None,
) -> ReproducibilityReport:
    root = Path(project_root).resolve()
    home = Path(user_home).resolve() if user_home is not None else Path.home()
    checks: list[ReproducibilityCheck] = []

    _check_python(checks)
    _check_imports(checks)
    _check_console_entry_points(checks, root)
    _check_writable_output(checks, _resolve(root, output_root))

    model_path = _resolve(root, model_config_path)
    model_config = _load_model_config(checks, model_path)
    profile_path = _resolve(root, active_profile_path)
    profile = _load_profile(checks, profile_path)
    if model_config is not None:
        _check_model_assets(checks, root, home, model_config)
    if profile is not None:
        _check_active_profile_assets(checks, root, profile)

    tier1_ready = not any(check.scope == "TIER1" and check.status == "FAIL" for check in checks)
    current_active_ready = tier1_ready and not any(
        check.scope == "CURRENT_ACTIVE" and check.status != "PASS" for check in checks
    )
    if not tier1_ready:
        status = ReproducibilityStatus.NOT_READY
    elif not current_active_ready:
        status = ReproducibilityStatus.PARTIALLY_READY
    else:
        status = ReproducibilityStatus.READY
    return ReproducibilityReport(
        schema_version=1,
        status=status,
        tier1_ready=tier1_ready,
        current_active_ready=current_active_ready,
        workflow="ZED multiview RGBD -> poomsae1_trimmed analysis",
        environment=environment_provenance(),
        checks=tuple(checks),
    )


def _check_python(checks: list[ReproducibilityCheck]) -> None:
    supported = sys.version_info >= (3, 11)
    checks.append(
        ReproducibilityCheck(
            name="python_version",
            category=RequirementCategory.INSTALLABLE_DEPENDENCY,
            scope="TIER1",
            status="PASS" if supported else "FAIL",
            message=(
                f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}; "
                "TK3D requires Python >=3.11"
            ),
        )
    )


def _check_imports(checks: list[ReproducibilityCheck]) -> None:
    modules = (
        "src.artifact_contracts",
        "src.multiview_application",
        "src.poomsae_scoring.application",
    )
    failures = []
    for module in modules:
        try:
            importlib.import_module(module)
        except (ImportError, OSError) as exc:
            failures.append(f"{module}: {exc}")
    checks.append(
        ReproducibilityCheck(
            name="application_imports",
            category=RequirementCategory.INSTALLABLE_DEPENDENCY,
            scope="TIER1",
            status="FAIL" if failures else "PASS",
            message="; ".join(failures) if failures else "Core contracts and both application APIs import successfully",
        )
    )


def _check_console_entry_points(checks: list[ReproducibilityCheck], root: Path) -> None:
    pyproject_path = root / "pyproject.toml"
    expected = {
        "tk3d-check": "scripts.check_reproducibility:main",
        "tk3d-multiview": "scripts.run_vitpose_multiview_3d:main",
        "tk3d-poomsae": "scripts.run_poomsae_scoring:main",
    }
    try:
        payload = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
        declared = payload["project"]["scripts"]
        valid = all(declared.get(name) == target for name, target in expected.items())
    except (OSError, KeyError, tomllib.TOMLDecodeError) as exc:
        valid = False
        message = f"Cannot validate pyproject console entry points: {exc}"
    else:
        message = "Console entry points are declared for environment, multiview and poomsae commands"
        if not valid:
            message = f"Expected console entry points are missing or changed: {expected}"
    checks.append(
        ReproducibilityCheck(
            name="console_entry_points",
            category=RequirementCategory.INCLUDED_IN_REPOSITORY,
            scope="TIER1",
            status="PASS" if valid else "FAIL",
            message=message,
            path=str(pyproject_path),
        )
    )


def _check_writable_output(checks: list[ReproducibilityCheck], output_root: Path) -> None:
    try:
        output_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="tk3d-readiness-", dir=output_root):
            pass
    except OSError as exc:
        status = "FAIL"
        message = f"Output root is not writable: {exc}"
    else:
        status = "PASS"
        message = "Output root is writable"
    checks.append(
        ReproducibilityCheck(
            name="output_root",
            category=RequirementCategory.MACHINE_SPECIFIC,
            scope="TIER1",
            status=status,
            message=message,
            path=str(output_root),
        )
    )


def _load_model_config(checks: list[ReproducibilityCheck], path: Path) -> dict[str, Any] | None:
    try:
        payload = validate_model_config(yaml.safe_load(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, yaml.YAMLError) as exc:
        status = "FAIL"
        message = f"Model config cannot be loaded and validated: {exc}"
        payload = None
    else:
        status = "PASS"
        message = "Active model configuration is valid"
    checks.append(
        ReproducibilityCheck(
            name="model_config",
            category=RequirementCategory.INCLUDED_IN_REPOSITORY,
            scope="TIER1",
            status=status,
            message=message,
            path=str(path),
        )
    )
    return payload


def _load_profile(checks: list[ReproducibilityCheck], path: Path) -> dict[str, Any] | None:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("profile_id") != "poomsae1_trimmed":
            raise ValueError("Active profile must be the poomsae1_trimmed mapping")
    except (OSError, ValueError, yaml.YAMLError) as exc:
        status = "FAIL"
        message = f"Active Poomsae profile cannot be loaded: {exc}"
        payload = None
    else:
        status = "PASS"
        message = "CURRENT_ACTIVE poomsae1_trimmed profile is available"
    checks.append(
        ReproducibilityCheck(
            name="active_profile",
            category=RequirementCategory.INCLUDED_IN_REPOSITORY,
            scope="TIER1",
            status=status,
            message=message,
            path=str(path),
        )
    )
    return payload


def _check_model_assets(
    checks: list[ReproducibilityCheck], root: Path, home: Path, config: dict[str, Any]
) -> None:
    pose = config["pose2d"]
    _path_check(
        checks,
        name="vitpose_model_config",
        category=RequirementCategory.INCLUDED_IN_REPOSITORY,
        scope="TIER1",
        path=_resolve(root, pose["config_path"]),
        label=f"{pose['model_name']} model config",
    )
    _path_check(
        checks,
        name="vitpose_checkpoint",
        category=RequirementCategory.EXTERNAL_MODEL,
        scope="CURRENT_ACTIVE",
        path=_resolve(root, pose["checkpoint_path"]),
        label=f"{pose['model_name']} checkpoint selected by pose2d.checkpoint_path",
    )
    detector = config.get("person_detector", {})
    variant = str(detector.get("model_variant", "small"))
    filename = {
        "nano": "rf-detr-nano.pth",
        "small": "rf-detr-small.pth",
        "medium": "rf-detr-medium.pth",
        "base": "rf-detr-base.pth",
        "large": "rf-detr-large.pth",
    }.get(variant, f"rf-detr-{variant}.pth")
    _path_check(
        checks,
        name="person_detector_checkpoint",
        category=RequirementCategory.EXTERNAL_MODEL,
        scope="CURRENT_ACTIVE",
        path=home / ".roboflow" / "models" / filename,
        label=f"RF-DETR {variant} checkpoint selected by person_detector.model_variant",
    )
    for module in ("torch", "timm", "rfdetr", "supervision"):
        available = importlib.util.find_spec(module) is not None
        checks.append(
            ReproducibilityCheck(
                name=f"dependency_{module}",
                category=RequirementCategory.INSTALLABLE_DEPENDENCY,
                scope="CURRENT_ACTIVE",
                status="PASS" if available else "WARN",
                message=f"{module} is importable" if available else f"{module} is required for active inference",
            )
        )
    cuda = environment_provenance()["torch"]["cuda_available"]
    checks.append(
        ReproducibilityCheck(
            name="cuda_runtime",
            category=RequirementCategory.MACHINE_SPECIFIC,
            scope="CURRENT_ACTIVE",
            status="PASS" if cuda else "WARN",
            message="CUDA GPU is available" if cuda else "CUDA GPU is required by active model_config device=cuda:0",
        )
    )


def _check_active_profile_assets(
    checks: list[ReproducibilityCheck], root: Path, profile: dict[str, Any]
) -> None:
    for key in ("rule_pack", "poomsae_spec", "movement_timeline", "diagnostic_profile", "accuracy_profile"):
        _yaml_mapping_check(
            checks,
            name=f"profile_{key}",
            category=RequirementCategory.INCLUDED_IN_REPOSITORY,
            scope="TIER1",
            path=_resolve(root, profile[key]),
            label=f"Profile {key}",
        )
    session_path = _resolve(root, profile["session"])
    reference_pose = _resolve(root, profile["reference_pose"])
    _path_check(
        checks,
        name="active_session",
        category=RequirementCategory.EXTERNAL_DATA,
        scope="CURRENT_ACTIVE",
        path=session_path,
        label="Active ZED session.yaml",
    )
    _path_check(
        checks,
        name="active_reference_pose",
        category=RequirementCategory.EXTERNAL_DATA,
        scope="CURRENT_ACTIVE",
        path=reference_pose,
        label="Run-bound active WholeBody-133 reference pose",
    )
    for video in profile.get("videos", []):
        _path_check(
            checks,
            name=f"active_video_{video.get('camera_id', 'unknown')}",
            category=RequirementCategory.EXTERNAL_DATA,
            scope="CURRENT_ACTIVE",
            path=_resolve(root, video.get("path", "")),
            label=f"Active camera video {video.get('camera_id', 'unknown')}",
        )
    session_root = session_path.parent.parent
    calibration_path = session_root / "calibration" / "cameras.json"
    _check_calibration(checks, calibration_path)
    _check_profile_bindings(checks, root, profile)
    if session_path.is_file():
        _check_zed_sources(checks, session_path)


def _check_calibration(checks: list[ReproducibilityCheck], path: Path) -> None:
    if not path.is_file():
        status = "WARN"
        message = "Production calibration is missing; active multiview inference must fail closed"
    else:
        try:
            bundle = load_calibration_bundle(path)
            mode = str(bundle.metadata.get("calibration_mode", "legacy_unknown"))
            if mode not in PRODUCTION_CALIBRATION_MODES:
                raise ValueError(f"non-production calibration_mode={mode}")
        except (OSError, ValueError, KeyError) as exc:
            status = "WARN"
            message = f"Calibration is present but invalid for production: {exc}"
        else:
            status = "PASS"
            message = f"Production calibration is available ({mode})"
    checks.append(
        ReproducibilityCheck(
            name="active_calibration",
            category=RequirementCategory.EXTERNAL_CALIBRATION,
            scope="CURRENT_ACTIVE",
            status=status,
            message=message,
            path=str(path),
        )
    )


def _check_profile_bindings(
    checks: list[ReproducibilityCheck], root: Path, profile: dict[str, Any]
) -> None:
    binding_paths = {
        "session_sha256": _resolve(root, profile["session"]),
        "reference_pose_sha256": _resolve(root, profile["reference_pose"]),
        "movement_timeline_sha256": _resolve(root, profile["movement_timeline"]),
    }
    for name, path in binding_paths.items():
        expected = profile.get("bindings", {}).get(name)
        if not path.is_file():
            status = "WARN"
            message = f"Cannot verify {name}; bound asset is absent"
        else:
            actual = sha256_file(path)
            status = "PASS" if actual == expected else "WARN"
            message = f"{name} matches" if status == "PASS" else f"{name} mismatch: expected {expected}, got {actual}"
        checks.append(
            ReproducibilityCheck(
                name=f"binding_{name}",
                category=(
                    RequirementCategory.INCLUDED_IN_REPOSITORY
                    if name == "movement_timeline_sha256"
                    else RequirementCategory.EXTERNAL_DATA
                ),
                scope="CURRENT_ACTIVE",
                status=status,
                message=message,
                path=str(path),
            )
        )


def _check_zed_sources(checks: list[ReproducibilityCheck], session_path: Path) -> None:
    try:
        session = yaml.safe_load(session_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return
    sources = session.get("zed", {}).get("depth_sources", []) if isinstance(session, dict) else []
    for source in sources:
        camera_id = source.get("camera_id", "unknown")
        for field in ("svo_path", "timestamp_mapping_report"):
            raw = source.get(field)
            path = Path(str(raw)) if raw else Path("missing")
            if not path.is_absolute():
                path = (session_path.parent / path).resolve()
            _path_check(
                checks,
                name=f"zed_{field}_{camera_id}",
                category=(
                    RequirementCategory.MACHINE_SPECIFIC if field == "svo_path" else RequirementCategory.EXTERNAL_DATA
                ),
                scope="CURRENT_ACTIVE",
                path=path,
                label=f"ZED {camera_id} {field}",
            )
    available = importlib.util.find_spec("pyzed") is not None
    checks.append(
        ReproducibilityCheck(
            name="zed_sdk_python",
            category=RequirementCategory.MACHINE_SPECIFIC,
            scope="CURRENT_ACTIVE",
            status="PASS" if available else "WARN",
            message="pyzed is importable" if available else "ZED SDK Python bindings are required for stereo depth",
        )
    )


def _path_check(
    checks: list[ReproducibilityCheck],
    *,
    name: str,
    category: RequirementCategory,
    scope: str,
    path: Path,
    label: str,
) -> None:
    exists = path.is_file()
    checks.append(
        ReproducibilityCheck(
            name=name,
            category=category,
            scope=scope,
            status="PASS" if exists else ("FAIL" if scope == "TIER1" else "WARN"),
            message=f"{label} is available" if exists else f"{label} is missing; expected at {path}",
            path=str(path),
        )
    )


def _yaml_mapping_check(
    checks: list[ReproducibilityCheck],
    *,
    name: str,
    category: RequirementCategory,
    scope: str,
    path: Path,
    label: str,
) -> None:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("top-level YAML value must be a mapping")
    except (OSError, ValueError, yaml.YAMLError) as exc:
        status = "FAIL" if scope == "TIER1" else "WARN"
        message = f"{label} cannot be loaded as a YAML mapping: {exc}"
    else:
        status = "PASS"
        message = f"{label} is available and parses as a YAML mapping"
    checks.append(
        ReproducibilityCheck(
            name=name,
            category=category,
            scope=scope,
            status=status,
            message=message,
            path=str(path),
        )
    )


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()
