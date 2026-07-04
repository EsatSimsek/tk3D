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
from src.scoring_readiness import build_scoring_readiness
from src.smoothing_3d import moving_average_nan
from src.validation_3d import quality_summary, validate_triangulation
from src.video_io import ensure_output_tree, load_session


def main() -> None:
    parser = argparse.ArgumentParser(description="Create scoring-readiness quality, smoothing, biomechanics and segment reports from 3D pose output.")
    parser.add_argument("--session", required=True)
    parser.add_argument("--input-json", default=None, help="Defaults to outputs/<session_id>/json/rtmw_session_3d.json")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--smoothing-window", type=int, default=5)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--max-reprojection-error-px", type=float, default=25.0)
    args = parser.parse_args()

    session = load_session(args.session)
    output_paths = ensure_output_tree(ROOT / args.output_root, session.session_id)
    input_path = Path(args.input_json) if args.input_json else output_paths["json"] / "rtmw_session_3d.json"
    if not input_path.is_absolute():
        input_path = (ROOT / input_path).resolve()
    if not input_path.exists():
        raise SystemExit(f"3D input JSON bulunamadı: {input_path}")

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    keypoints = _array(payload, "keypoints_3d_world", ndim=3)
    triangulation_score = _optional_array(payload, "triangulation_score")
    reprojection_error = _optional_array(payload, "reprojection_error")
    used_cameras = _optional_array(payload, "used_cameras")
    fps = float(args.fps if args.fps is not None else session.fps or 30.0)

    smoothed = moving_average_nan(keypoints, window_size=args.smoothing_window)
    validation = validate_triangulation(smoothed, reprojection_error if reprojection_error is not None else np.full(smoothed.shape[:2], np.nan), args.max_reprojection_error_px)
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
    )

    smoothed_json = output_paths["json"] / "rtmw_session_3d_smoothed.json"
    readiness_json = output_paths["json"] / "scoring_readiness_report.json"
    smoothed_csv = output_paths["csv"] / "rtmw_keypoints_3d_world_smoothed_flat.csv"
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
            "fps": fps,
            "shape": {"keypoints_3d_world_smoothed": list(smoothed.shape)},
            "keypoints_3d_world_smoothed": smoothed,
        },
        smoothed_json,
    )
    export_keypoints3d_csv(smoothed, smoothed_csv)
    _write_csv(readiness.frame_quality_rows, frame_quality_csv)
    _write_csv(readiness.joint_quality_rows, joint_quality_csv)
    _write_csv(readiness.biomechanics_rows, biomechanics_csv)
    _write_csv(readiness.segment_rows, segments_csv)
    report = {
        "session_id": session.session_id,
        "source": str(input_path),
        "fps": fps,
        "smoothing_window": args.smoothing_window,
        "quality_summary": summary,
        "scoring_readiness": readiness.report,
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
    _write_excel(report, {
        "frame_quality": frame_quality_csv,
        "joint_quality": joint_quality_csv,
        "biomechanics": biomechanics_csv,
        "segments": segments_csv,
    }, excel_path)

    print(f"saved: {readiness_json}")
    print(f"smoothed: {smoothed_json}")
    print(f"biomechanics: {biomechanics_csv}")
    print(f"segments: {segments_csv}")
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


def _write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)


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
        "warnings": ";".join(readiness.get("warnings", [])),
        "next_step": readiness.get("next_step"),
    }


if __name__ == "__main__":
    main()
