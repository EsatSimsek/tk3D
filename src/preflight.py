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
        _check_media_path(
            issues,
            path=camera.video_path,
            required=videos_required,
            code="video_missing",
            unreadable_code="video_unreadable",
            message="Camera video is missing.",
            unreadable_message="Camera video exists but cannot be opened.",
            camera_id=camera.camera_id,
            probe_video=True,
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
            _check_media_path(
                issues,
                path=camera.calibration_video_path,
                required=calibration_videos_required,
                code="calibration_video_missing",
                unreadable_code="calibration_video_unreadable",
                message="Calibration video is missing.",
                unreadable_message="Calibration video exists but cannot be opened.",
                camera_id=camera.camera_id,
                probe_video=True,
            )

    project_root = _find_project_root(session.root_dir)
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
                path = project_root / path
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


def _check_media_path(
    issues: list[PreflightIssue],
    path: Path,
    required: bool,
    code: str,
    unreadable_code: str,
    message: str,
    unreadable_message: str,
    camera_id: str | None = None,
    probe_video: bool = False,
) -> None:
    if not path.exists():
        _check_path(
            issues,
            exists=False,
            required=required,
            code=code,
            message=message,
            camera_id=camera_id,
            path=path,
        )
        return
    if probe_video and not _video_is_openable(path):
        issues.append(
            PreflightIssue(
                severity="error" if required else "warning",
                code=unreadable_code,
                message=unreadable_message,
                camera_id=camera_id,
                path=str(path),
            )
        )


def _video_is_openable(path: Path) -> bool:
    try:
        import cv2
    except ModuleNotFoundError:
        return False
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return False
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        return frame_count != 0 and width > 0 and height > 0
    finally:
        capture.release()


def _find_project_root(start: Path) -> Path:
    """Walk up from *start* until a directory containing ``pyproject.toml`` is found."""
    current = start.resolve()
    for parent in (current, *current.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    # Fallback: assume session dir is two levels below project root (data/<name>/).
    return start.parents[1] if len(start.parents) > 1 else start
