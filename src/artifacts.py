from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_OUTPUTS = [
    "videos/cam_front_2d_overlay.mp4",
    "videos/cam_back_2d_overlay.mp4",
    "videos/cam_side_2d_overlay.mp4",
    "videos/skeleton_3d_world.mp4",
    "figures/reprojection_error_timeline.png",
    "figures/keypoint_validity_heatmap.png",
    "figures/camera_usage_heatmap.png",
    "csv/keypoints_2d_flat.csv",
    "csv/keypoints_3d_world_flat.csv",
    "csv/triangulation_quality.csv",
    "csv/validation_frames.csv",
    "csv/validation_joints.csv",
    "csv/validation_steps.csv",
    "json/session_3d.json",
    "json/preflight_report.json",
    "json/video_probe_report.json",
    "json/model_runtime_report.json",
    "json/quality_summary.json",
    "session_3d_analysis.xlsx",
]


def build_artifact_manifest(output_root: str | Path) -> dict[str, Any]:
    root = Path(output_root)
    expected = []
    for relative_path in EXPECTED_OUTPUTS:
        path = root / relative_path
        expected.append(
            {
                "relative_path": relative_path,
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else 0,
            }
        )

    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative_path = path.relative_to(root).as_posix()
        if relative_path == "json/artifact_manifest.json":
            continue
        files.append(
            {
                "relative_path": relative_path,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )

    missing = [item["relative_path"] for item in expected if not item["exists"]]
    return {
        "status": "complete" if not missing else "incomplete",
        "expected_outputs": expected,
        "missing_outputs": missing,
        "files": files,
    }


def save_artifact_manifest(output_root: str | Path, output_path: str | Path) -> dict[str, Any]:
    manifest = build_artifact_manifest(output_root)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
    return manifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
