from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.camera_calibration import save_calibrations
from src.coordinate_system import (
    ANALYSIS_COORDINATE_SYSTEM,
    calibration_metadata,
    mads_world_to_analysis,
    transform_points,
)
from src.exporter import export_session_json
from src.mads_dataset import (
    MADS_DEPTH_JOINT_NAMES,
    MADS_MULTIVIEW_JOINT_NAMES,
    MadsSequence,
    discover_mads_sequences,
    load_mads_camera_calibration,
    load_mads_ground_truth,
    probe_mads_video,
    resolve_mads_roots,
)
from src.video_io import ensure_output_tree


MADS_SOURCE_COORDINATE_SYSTEM = {
    "name": "mads_world",
    "unit": "millimeter",
    "axes": {"x": "horizontal", "y": "up", "z": "backward"},
    "handedness": "right",
}

MADS_EDGE_NAMES: tuple[tuple[str, str], ...] = (
    ("neck", "pelvis"),
    ("pelvis", "left_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("left_ankle", "left_foot"),
    ("pelvis", "right_hip"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("right_ankle", "right_foot"),
    ("neck", "left_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("left_wrist", "left_hand"),
    ("neck", "right_shoulder"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("right_wrist", "right_hand"),
    ("neck", "head"),
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Index extracted MADS data and prepare TK3D sessions, calibration, and metric ground truth."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--actions", nargs="+", default=["Kata"])
    parser.add_argument("--local-data-dir", type=Path, default=ROOT / "data" / "mads_test" / "local")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--hash-files", action="store_true")
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()

    roots = resolve_mads_roots(args.dataset_root)
    sequences = discover_mads_sequences(roots)
    selected_actions = {action.lower() for action in args.actions}
    selected = [sequence for sequence in sequences if sequence.action.lower() in selected_actions]
    missing_actions = selected_actions.difference({sequence.action.lower() for sequence in selected})
    if missing_actions:
        raise SystemExit(f"MADS actions not found: {sorted(missing_actions)}")

    local_root = args.local_data_dir.resolve()
    output_root = args.output_root.resolve()
    session_root = local_root / "sessions"
    gt_root = local_root / "ground_truth"
    session_root.mkdir(parents=True, exist_ok=True)
    gt_root.mkdir(parents=True, exist_ok=True)
    (output_root / "mads_setup" / "previews").mkdir(parents=True, exist_ok=True)

    sequence_reports = [_inspect_sequence(sequence, args.hash_files and sequence in selected) for sequence in sequences]
    generated: list[dict[str, Any]] = []
    preview_reports: list[dict[str, Any]] = []

    for sequence in selected:
        gt_path = _export_ground_truth(sequence, gt_root)
        item: dict[str, Any] = {
            "modality": sequence.modality,
            "action": sequence.action,
            "sequence": sequence.sequence,
            "ground_truth_json": str(gt_path),
        }
        if sequence.modality == "multiview":
            session_path, calibration_path, calibrations = _prepare_multiview_session(
                sequence, session_root, output_root, roots.dataset_root, gt_path
            )
            item.update(
                {
                    "session_path": str(session_path),
                    "calibration_path": str(calibration_path),
                }
            )
            if args.preview:
                preview_reports.extend(
                    _write_projection_previews(
                        sequence,
                        calibrations,
                        output_root / "mads_setup" / "previews",
                    )
                )
        generated.append(item)

    warnings = _dataset_warnings(sequence_reports)
    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "MADS",
        "dataset_root": str(roots.dataset_root),
        "multiview_data_root": str(roots.multiview_data),
        "depth_data_root": str(roots.depth_data) if roots.depth_data else None,
        "selected_actions": sorted(args.actions),
        "hashes_enabled": bool(args.hash_files),
        "sequence_count": len(sequences),
        "multiview_sequence_count": sum(item.modality == "multiview" for item in sequences),
        "depth_sequence_count": sum(item.modality == "depth" for item in sequences),
        "sequences": sequence_reports,
        "generated": generated,
        "projection_previews": preview_reports,
        "warnings": warnings,
    }
    local_manifest = local_root / "mads_manifest.json"
    setup_report = output_root / "mads_setup" / "mads_setup_report.json"
    export_session_json(manifest, local_manifest)
    export_session_json(manifest, setup_report)

    print(f"MADS sequences indexed: {len(sequences)}")
    print(f"Selected assets prepared: {len(generated)}")
    print(f"Warnings: {len(warnings)}")
    print(f"Local manifest: {local_manifest}")
    print(f"Setup report: {setup_report}")


def _inspect_sequence(sequence: MadsSequence, hash_files: bool) -> dict[str, Any]:
    points = load_mads_ground_truth(sequence.ground_truth_path)
    if points.shape[1] not in {15, 19}:
        raise ValueError(
            f"Unsupported MADS joint count for {sequence.action}/{sequence.sequence}: {points.shape[1]}"
        )
    videos = {camera: probe_mads_video(path) for camera, path in sorted(sequence.videos.items())}
    for camera, path in sequence.videos.items():
        if hash_files:
            videos[camera]["sha256"] = _sha256(path)
    if hash_files:
        gt_sha256 = _sha256(sequence.ground_truth_path)
    else:
        gt_sha256 = None
    valid_frames = np.all(np.isfinite(points), axis=(1, 2))
    auxiliary = []
    for path in sequence.auxiliary_paths:
        item: dict[str, Any] = {"path": str(path), "size_bytes": path.stat().st_size}
        if hash_files:
            item["sha256"] = _sha256(path)
        auxiliary.append(item)
    return {
        "modality": sequence.modality,
        "action": sequence.action,
        "sequence": sequence.sequence,
        "videos": videos,
        "ground_truth": {
            "path": str(sequence.ground_truth_path),
            "size_bytes": sequence.ground_truth_path.stat().st_size,
            "sha256": gt_sha256,
            "frame_count": int(points.shape[0]),
            "joint_count": int(points.shape[1]),
            "fully_valid_frame_count": int(np.sum(valid_frames)),
            "invalid_frame_count": int(points.shape[0] - np.sum(valid_frames)),
        },
        "auxiliary": auxiliary,
    }


def _export_ground_truth(sequence: MadsSequence, output_root: Path) -> Path:
    points_mm = load_mads_ground_truth(sequence.ground_truth_path)
    joint_names = MADS_MULTIVIEW_JOINT_NAMES if points_mm.shape[1] == 15 else MADS_DEPTH_JOINT_NAMES
    if points_mm.shape[1] != len(joint_names):
        raise ValueError(
            f"Unsupported MADS joint count for {sequence.action}/{sequence.sequence}: {points_mm.shape[1]}"
        )
    points_m = transform_points(points_mm, mads_world_to_analysis())
    video_probe = probe_mads_video(next(iter(sequence.videos.values())))
    fps = float(video_probe["fps"])
    frame_count = points_m.shape[0]
    if any(probe_mads_video(path)["frame_count"] != frame_count for path in sequence.videos.values()):
        raise ValueError(f"Video/ground-truth frame mismatch for {sequence.action}/{sequence.sequence}")
    output_dir = output_root / sequence.modality
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{sequence.action}_{sequence.sequence}.json"
    payload = {
        "schema_version": 1,
        "dataset": "MADS",
        "modality": sequence.modality,
        "action": sequence.action,
        "sequence_id": sequence.sequence,
        "source": str(sequence.ground_truth_path),
        "coordinate_system": ANALYSIS_COORDINATE_SYSTEM,
        "source_coordinate_system": MADS_SOURCE_COORDINATE_SYSTEM,
        "source_to_analysis": mads_world_to_analysis(),
        "fps": fps,
        "frame_alignment": "one_ground_truth_pose_per_video_frame",
        "frame_indices": np.arange(frame_count, dtype=int),
        "timestamps_sec": np.arange(frame_count, dtype=float) / fps,
        "joint_names": list(joint_names),
        "keypoints_3d_ground_truth": points_m,
    }
    export_session_json(payload, output_path)
    return output_path.resolve()


def _prepare_multiview_session(
    sequence: MadsSequence,
    session_root: Path,
    output_root: Path,
    dataset_root: Path,
    ground_truth_path: Path,
) -> tuple[Path, Path, dict[str, Any]]:
    probes = {camera: probe_mads_video(path) for camera, path in sequence.videos.items()}
    if set(probes) != {"0", "1", "2"}:
        raise ValueError(f"Expected MADS cameras 0,1,2 for {sequence.action}/{sequence.sequence}")
    frame_counts = {int(probe["frame_count"]) for probe in probes.values()}
    fps_values = {round(float(probe["fps"]), 6) for probe in probes.values()}
    image_sizes = {tuple(probe["image_size"]) for probe in probes.values()}
    if len(frame_counts) != 1 or len(fps_values) != 1 or len(image_sizes) != 1:
        raise ValueError(f"MADS camera streams disagree for {sequence.action}/{sequence.sequence}")
    fps = next(iter(fps_values))
    image_size = next(iter(image_sizes))
    session_id = f"mads_{sequence.action}_{sequence.sequence}".lower()
    action_root = sequence.ground_truth_path.parent
    calibrations = {
        f"C{camera}": load_mads_camera_calibration(
            action_root / f"Calib_Cam{camera}.mat",
            camera_id=f"C{camera}",
            image_size=image_size,
        )
        for camera in ("0", "1", "2")
    }
    output_paths = ensure_output_tree(output_root, session_id)
    calibration_path = output_paths["calibration"] / "cameras.json"
    metadata = calibration_metadata(
        calibration_mode="mads_official_multiview",
        source_coordinate_system=MADS_SOURCE_COORDINATE_SYSTEM,
        source_to_analysis=mads_world_to_analysis(),
    )
    metadata.update(
        {
            "dataset": "MADS",
            "action": sequence.action,
            "sequence": sequence.sequence,
            "dataset_root": str(dataset_root),
        }
    )
    save_calibrations(list(calibrations.values()), calibration_path, metadata=metadata)

    session = {
        "session_id": session_id,
        "task_name": f"mads_{sequence.action}_{sequence.sequence}".lower(),
        "fps": fps,
        "cameras": [
            {"camera_id": f"C{camera}", "video_path": str(sequence.videos[camera])}
            for camera in ("0", "1", "2")
        ],
        "sync": {"method": "frame_index", "offsets": {f"C{camera}": 0 for camera in ("0", "1", "2")}},
        "mads": {
            "dataset_root": str(dataset_root),
            "action": sequence.action,
            "sequence": sequence.sequence,
            "ground_truth_path": str(ground_truth_path),
            "frame_alignment": "one_ground_truth_pose_per_video_frame",
        },
    }
    session_path = session_root / f"{session_id}.yaml"
    session_path.write_text(yaml.safe_dump(session, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return session_path.resolve(), calibration_path.resolve(), calibrations


def _write_projection_previews(
    sequence: MadsSequence,
    calibrations: dict[str, Any],
    output_root: Path,
) -> list[dict[str, Any]]:
    points = load_mads_ground_truth(sequence.ground_truth_path)
    joint_names = MADS_MULTIVIEW_JOINT_NAMES if points.shape[1] == 15 else MADS_DEPTH_JOINT_NAMES
    joint_index = {name: index for index, name in enumerate(joint_names)}
    edges = [
        (joint_index[first], joint_index[second])
        for first, second in MADS_EDGE_NAMES
        if first in joint_index and second in joint_index
    ]
    valid_frames = np.flatnonzero(np.all(np.isfinite(points), axis=(1, 2)))
    if valid_frames.size == 0:
        return []
    frame_idx = int(valid_frames[0])
    pose = points[frame_idx]
    reports: list[dict[str, Any]] = []
    for camera_number, video_path in sorted(sequence.videos.items()):
        camera_id = f"C{camera_number}"
        calibration = calibrations[camera_id]
        capture = cv2.VideoCapture(str(video_path))
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, image = capture.read()
        capture.release()
        if not ok:
            raise ValueError(f"Could not read preview frame {frame_idx} from {video_path}")
        projected, _ = cv2.projectPoints(
            pose,
            calibration.rotation_vector,
            calibration.translation_vector,
            calibration.intrinsic_matrix,
            calibration.distortion_coefficients,
        )
        xy = projected.reshape(-1, 2)
        height, width = image.shape[:2]
        in_frame = (
            np.isfinite(xy).all(axis=1)
            & (xy[:, 0] >= 0)
            & (xy[:, 0] < width)
            & (xy[:, 1] >= 0)
            & (xy[:, 1] < height)
        )
        for first, second in edges:
            if in_frame[first] and in_frame[second]:
                cv2.line(image, tuple(np.rint(xy[first]).astype(int)), tuple(np.rint(xy[second]).astype(int)), (0, 255, 0), 2)
        for joint_idx, point in enumerate(xy):
            if in_frame[joint_idx]:
                cv2.circle(image, tuple(np.rint(point).astype(int)), 4, (0, 0, 255), -1)
        output_path = output_root / f"{sequence.action}_{sequence.sequence}_{camera_id}_gt_overlay.png"
        if not cv2.imwrite(str(output_path), image):
            raise OSError(f"Could not write MADS projection preview: {output_path}")
        reports.append(
            {
                "action": sequence.action,
                "sequence": sequence.sequence,
                "camera_id": camera_id,
                "frame_idx": frame_idx,
                "in_frame_joint_ratio": float(np.mean(in_frame)),
                "output_path": str(output_path.resolve()),
            }
        )
    return reports


def _dataset_warnings(sequence_reports: list[dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    multiview_fps = {
        round(float(video["fps"]), 6)
        for sequence in sequence_reports
        if sequence["modality"] == "multiview"
        for video in sequence["videos"].values()
    }
    if multiview_fps != {15.0}:
        warnings.append(
            "The extracted multi-view AVI headers report FPS values different from the publication's 15 fps; "
            "TK3D uses frame-index alignment and records the actual header FPS."
        )
    return warnings


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
