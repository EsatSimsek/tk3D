from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .data_structures import Session


@dataclass(slots=True)
class PreflightIssue:
    severity: str
    code: str
    message: str
    camera_id: str | None = None
    path: str | None = None


def run_preflight(
    session: Session,
    model_config: dict[str, Any],
    videos_required: bool,
    calibration_videos_required: bool,
    model_files_required: bool,
) -> list[PreflightIssue]:
    issues: list[PreflightIssue] = []
    camera_ids = [camera.camera_id for camera in session.cameras]

    if len(session.cameras) < 2:
        issues.append(
            PreflightIssue(
                severity="error",
                code="not_enough_cameras",
                message="Multi-view triangulation requires at least 2 cameras.",
            )
        )
    if len(camera_ids) != len(set(camera_ids)):
        issues.append(
            PreflightIssue(
                severity="error",
                code="duplicate_camera_id",
                message="Camera IDs must be unique within a session.",
            )
        )

    for camera in session.cameras:
        _check_path(
            issues,
            exists=camera.video_path.exists(),
            required=videos_required,
            code="video_missing",
            message="Camera video is missing.",
            camera_id=camera.camera_id,
            path=camera.video_path,
        )
        if camera.calibration_video_path is None:
            issues.append(
                PreflightIssue(
                    severity="error" if calibration_videos_required else "warning",
                    code="calibration_video_not_configured",
                    message="Calibration video path is not configured.",
                    camera_id=camera.camera_id,
                )
            )
        else:
            _check_path(
                issues,
                exists=camera.calibration_video_path.exists(),
                required=calibration_videos_required,
                code="calibration_video_missing",
                message="Calibration video is missing.",
                camera_id=camera.camera_id,
                path=camera.calibration_video_path,
            )

    for section_name, path_keys in {
        "pose2d": ["config_path", "checkpoint_path"],
        "pose3d_single_view": ["config_path", "checkpoint_path"],
    }.items():
        section = model_config.get(section_name, {})
        if section_name == "pose3d_single_view" and not section.get("enabled", True):
            continue
        for key in path_keys:
            raw_path = section.get(key)
            if not raw_path:
                issues.append(
                    PreflightIssue(
                        severity="error" if model_files_required else "warning",
                        code=f"{section_name}_{key}_missing",
                        message=f"{section_name}.{key} is not configured.",
                    )
                )
                continue
            path = Path(raw_path)
            if not path.is_absolute():
                path = session.root_dir.parents[1] / path
            _check_path(
                issues,
                exists=path.exists(),
                required=model_files_required,
                code=f"{section_name}_{key}_not_found",
                message=f"{section_name}.{key} does not exist.",
                path=path,
            )

    return issues


def preflight_summary(issues: list[PreflightIssue]) -> dict[str, Any]:
    error_count = sum(1 for issue in issues if issue.severity == "error")
    warning_count = sum(1 for issue in issues if issue.severity == "warning")
    return {
        "status": "failed" if error_count else "passed",
        "error_count": error_count,
        "warning_count": warning_count,
        "issues": [asdict(issue) for issue in issues],
    }


def save_preflight_report(issues: list[PreflightIssue], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(preflight_summary(issues), file, indent=2)


def has_errors(issues: list[PreflightIssue]) -> bool:
    return any(issue.severity == "error" for issue in issues)


def _check_path(
    issues: list[PreflightIssue],
    exists: bool,
    required: bool,
    code: str,
    message: str,
    camera_id: str | None = None,
    path: Path | None = None,
) -> None:
    if exists:
        return
    issues.append(
        PreflightIssue(
            severity="error" if required else "warning",
            code=code,
            message=message,
            camera_id=camera_id,
            path=str(path) if path else None,
        )
    )
