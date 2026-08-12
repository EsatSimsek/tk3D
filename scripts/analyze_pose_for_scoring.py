from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.exporter import export_keypoints3d_csv, export_session_json
from src.coordinate_system import ANALYSIS_COORDINATE_SYSTEM
from src.run_outputs import resolve_latest_run
from src.quality_status import external_accuracy_not_evaluated
from src.scoring_readiness import build_scoring_readiness
from src.smoothing_3d import smooth_pose_sequence
from src.validation_3d import quality_summary, validate_triangulation
from src.video_io import load_session


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create scoring-readiness quality, smoothing, biomechanics and segment reports from 3D pose output."
    )
    parser.add_argument("--session", required=True)
    parser.add_argument(
        "--input-json", default=None, help="Defaults to outputs/<session_id>/json/vitpose_session_3d.json"
    )
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--smoothing-window", type=int, default=11)
    parser.add_argument(
        "--smoothing-method",
        choices=("robust_savgol", "moving_average"),
        default="robust_savgol",
    )
    parser.add_argument("--smoothing-polynomial-order", type=int, default=2)
    parser.add_argument("--smoothing-min-outlier-distance-m", type=float, default=0.04)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--max-reprojection-error-px", type=float, default=25.0)
    parser.add_argument("--min-triangulation-score", type=float, default=0.20)
    parser.add_argument("--allow-legacy-coordinate-system", action="store_true")
    args = parser.parse_args()

    session = load_session(args.session)
    output_root = (ROOT / args.output_root).resolve()
    if args.input_json:
        input_path = Path(args.input_json)
    else:
        input_path = resolve_latest_run(output_root, session.session_id) / "json" / "vitpose_session_3d.json"
    if not input_path.is_absolute():
        input_path = (ROOT / input_path).resolve()
    if not input_path.exists():
        raise SystemExit(f"3D input JSON bulunamadı: {input_path}")

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    external_accuracy = payload.get("external_accuracy")
    if not isinstance(external_accuracy, dict):
        external_accuracy = external_accuracy_not_evaluated()
    coordinate_system = payload.get("coordinate_system")
    if coordinate_system != ANALYSIS_COORDINATE_SYSTEM and not args.allow_legacy_coordinate_system:
        raise SystemExit(
            "Input 3D coordinates do not declare the TK3D meter/x-right/y-forward/z-up contract. "
            "Re-run the 3D pipeline or explicitly pass --allow-legacy-coordinate-system for diagnostics only."
        )
    run_root = input_path.parent.parent
    output_paths = {
        "root": run_root,
        "json": run_root / "json",
        "csv": run_root / "csv",
    }
    for path in output_paths.values():
        path.mkdir(parents=True, exist_ok=True)
    keypoints = _array(payload, "keypoints_3d_world", ndim=3)
    if keypoints.shape[0] == 0:
        raise SystemExit("keypoints_3d_world en az bir kare içermeli")
    triangulation_score = _optional_array(payload, "triangulation_score")
    reprojection_error = _optional_array(payload, "reprojection_error")
    used_cameras = _optional_array(payload, "used_cameras")
    fps = float(args.fps if args.fps is not None else payload.get("sample_fps") or session.fps or 30.0)
    frame_indices = np.asarray(payload.get("frame_indices", np.arange(keypoints.shape[0])), dtype=int)
    timestamps = np.asarray(payload.get("timestamps_sec", frame_indices / fps), dtype=float)
    if frame_indices.shape != (keypoints.shape[0],) or timestamps.shape != (keypoints.shape[0],):
        raise SystemExit("frame_indices ve timestamps_sec, keypoints_3d_world kare sayısıyla birebir eşleşmeli")
    _require_quality_shape(triangulation_score, keypoints.shape[:2], "triangulation_score")
    _require_quality_shape(reprojection_error, keypoints.shape[:2], "reprojection_error")
    _require_quality_shape(used_cameras, keypoints.shape[:2], "used_cameras")

    accepted = np.all(np.isfinite(keypoints), axis=-1)
    if triangulation_score is not None:
        accepted &= triangulation_score >= args.min_triangulation_score
    if reprojection_error is not None:
        accepted &= np.isfinite(reprojection_error) & (reprojection_error <= args.max_reprojection_error_px)
    if used_cameras is not None:
        accepted &= used_cameras >= 2
    smoothed = (
        keypoints.copy()
        if payload.get("smoothing_applied")
        else smooth_pose_sequence(
            keypoints,
            method=args.smoothing_method,
            window_size=args.smoothing_window,
            valid_mask=accepted,
            polynomial_order=args.smoothing_polynomial_order,
            min_outlier_distance_m=args.smoothing_min_outlier_distance_m,
        )
    )
    validation = validate_triangulation(
        smoothed,
        reprojection_error if reprojection_error is not None else np.full(smoothed.shape[:2], np.nan),
        args.max_reprojection_error_px,
    )
    summary = quality_summary(
        smoothed,
        triangulation_score if triangulation_score is not None else np.full(smoothed.shape[:2], np.nan),
        reprojection_error if reprojection_error is not None else np.full(smoothed.shape[:2], np.nan),
        used_cameras if used_cameras is not None else np.full(smoothed.shape[:2], np.nan),
        validation,
    )
    readiness = build_scoring_readiness(
        smoothed,
        triangulation_score=triangulation_score,
        reprojection_error=reprojection_error,
        used_cameras=used_cameras,
        fps=fps,
        max_reprojection_error_px=args.max_reprojection_error_px,
        min_triangulation_score=args.min_triangulation_score,
    )
    _attach_readiness_timeline(readiness, frame_indices, timestamps)

    smoothed_json = output_paths["json"] / "vitpose_session_3d_smoothed.json"
    readiness_json = output_paths["json"] / "scoring_readiness_report.json"
    smoothed_csv = output_paths["csv"] / "vitpose_keypoints_3d_world_smoothed_flat.csv"
    frame_quality_csv = output_paths["csv"] / "pose_quality_frames.csv"
    joint_quality_csv = output_paths["csv"] / "pose_quality_joints.csv"
    biomechanics_csv = output_paths["csv"] / "biomechanics_timeseries.csv"
    segments_csv = output_paths["csv"] / "movement_segments.csv"
    excel_path = output_paths["root"] / "scoring_readiness_analysis.xlsx"

    export_session_json(
        {
            "session_id": session.session_id,
            "source": str(input_path),
            "smoothing_window": args.smoothing_window,
            "smoothing_method": (
                payload.get("smoothing_method") if payload.get("smoothing_applied") else args.smoothing_method
            ),
            "fps": fps,
            "coordinate_system": ANALYSIS_COORDINATE_SYSTEM,
            "frame_indices": frame_indices,
            "timestamps_sec": timestamps,
            "shape": {"keypoints_3d_world_smoothed": list(smoothed.shape)},
            "keypoints_3d_world_smoothed": smoothed,
        },
        smoothed_json,
    )
    export_keypoints3d_csv(smoothed, smoothed_csv, frame_indices=frame_indices, timestamps_sec=timestamps)
    _write_csv(readiness.frame_quality_rows, frame_quality_csv)
    _write_csv(readiness.joint_quality_rows, joint_quality_csv)
    _write_csv(readiness.biomechanics_rows, biomechanics_csv)
    _write_csv(readiness.segment_rows, segments_csv)
    report = {
        "session_id": session.session_id,
        "source": str(input_path),
        "fps": fps,
        "smoothing_window": args.smoothing_window,
        "smoothing_method": (
            payload.get("smoothing_method") if payload.get("smoothing_applied") else args.smoothing_method
        ),
        "external_accuracy": external_accuracy,
        "quality_summary": summary,
        "scoring_readiness": readiness.report,
        "score_generated": False,
        "next_stage": "source_bound_rulepack_accuracy",
        "outputs": {
            "smoothed_json": str(smoothed_json),
            "smoothed_csv": str(smoothed_csv),
            "frame_quality_csv": str(frame_quality_csv),
            "joint_quality_csv": str(joint_quality_csv),
            "biomechanics_csv": str(biomechanics_csv),
            "segments_csv": str(segments_csv),
            "excel": str(excel_path),
        },
    }
    export_session_json(report, readiness_json)
    _write_excel(
        report,
        {
            "frame_quality": frame_quality_csv,
            "joint_quality": joint_quality_csv,
            "biomechanics": biomechanics_csv,
            "segments": segments_csv,
        },
        excel_path,
    )

    print(f"saved: {readiness_json}")
    print(f"smoothed: {smoothed_json}")
    print(f"biomechanics: {biomechanics_csv}")
    print(f"segments: {segments_csv}")
    print("score generated: false (use the source-bound RulePack accuracy pipeline)")
    print(f"excel: {excel_path}")
    print(f"status: {readiness.report['status']}")


def _array(payload: dict[str, Any], key: str, ndim: int) -> np.ndarray:
    if key not in payload:
        raise SystemExit(f"Input JSON içinde zorunlu alan yok: {key}")
    value = np.asarray(payload[key], dtype=float)
    if value.ndim != ndim:
        raise SystemExit(f"{key} beklenen boyut {ndim}, gelen shape {value.shape}")
    return value


def _optional_array(payload: dict[str, Any], key: str) -> np.ndarray | None:
    if key not in payload:
        return None
    return np.asarray(payload[key], dtype=float)


def _require_quality_shape(value: np.ndarray | None, expected: tuple[int, int], name: str) -> None:
    if value is not None and value.shape != expected:
        raise SystemExit(f"{name} beklenen shape {expected}, gelen {value.shape}")


def _attach_readiness_timeline(readiness: Any, frame_indices: np.ndarray, timestamps: np.ndarray) -> None:
    for rows in (readiness.frame_quality_rows, readiness.biomechanics_rows):
        for row in rows:
            sample_idx = int(row["frame_idx"])
            row["source_frame_idx"] = int(frame_indices[sample_idx])
            row["timestamp_sec"] = float(timestamps[sample_idx])
    for segment in readiness.segment_rows:
        start = int(np.clip(int(segment.get("start_frame", 0)), 0, len(frame_indices) - 1))
        end = int(np.clip(int(segment.get("end_frame", start)), start, len(frame_indices) - 1))
        segment["source_start_frame_idx"] = int(frame_indices[start])
        segment["source_end_frame_idx"] = int(frame_indices[end])
        segment["start_time_sec"] = float(timestamps[start])
        segment["end_time_sec"] = float(timestamps[end])


def _write_csv(rows: list[dict[str, Any]], output_path: Path, columns: list[str] | None = None) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(output_path, index=False)


def _write_excel(report: dict[str, Any], csv_paths: dict[str, Path], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        pd.DataFrame([_flatten_report(report)]).to_excel(writer, sheet_name="summary", index=False)
        for sheet_name, csv_path in csv_paths.items():
            if csv_path.exists():
                pd.read_csv(csv_path).to_excel(writer, sheet_name=sheet_name[:31], index=False)


def _flatten_report(report: dict[str, Any]) -> dict[str, Any]:
    readiness = report.get("scoring_readiness", {})
    quality = report.get("quality_summary", {})
    return {
        "session_id": report.get("session_id"),
        "status": readiness.get("status"),
        "frame_count": readiness.get("frame_count"),
        "scoring_ready_frame_ratio": readiness.get("scoring_ready_frame_ratio"),
        "reliable_body17_joint_count": readiness.get("reliable_body17_joint_count"),
        "movement_segment_count": readiness.get("movement_segment_count"),
        "mean_reprojection_error_px": quality.get("mean_reprojection_error_px"),
        "mean_used_cameras": quality.get("mean_used_cameras"),
        "score_generated": report.get("score_generated", False),
        "next_stage": report.get("next_stage"),
        "warnings": ";".join(readiness.get("warnings", [])),
        "next_step": readiness.get("next_step"),
    }


if __name__ == "__main__":
    main()
