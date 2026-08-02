from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.exporter import export_session_json
from src.ground_truth_io import load_joint_map, load_pose_sequence_json, match_pose_sequences
from src.ground_truth_validation import evaluate_ground_truth_3d
from src.scoring_authorization import build_scoring_authorization, file_fingerprint


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare TK3D 3D predictions with synchronized metric ground truth."
    )
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--prediction-key", default="keypoints_3d_world")
    parser.add_argument("--ground-truth-key", default="keypoints_3d_ground_truth")
    parser.add_argument("--joint-map", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "ground_truth_validation.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-time-delta-sec", type=float)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument(
        "--allow-failed-quality-gate",
        action="store_true",
        help="Return success for diagnostic report generation even when accuracy gates fail.",
    )
    args = parser.parse_args()

    predicted = load_pose_sequence_json(args.prediction, args.prediction_key)
    truth = load_pose_sequence_json(args.ground_truth, args.ground_truth_key, require_joint_names=True)
    matched = match_pose_sequences(
        predicted,
        truth,
        joint_map=load_joint_map(args.joint_map),
        max_time_delta_sec=args.max_time_delta_sec,
    )
    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    thresholds = config.get("thresholds", config)
    result = evaluate_ground_truth_3d(
        matched.predicted_m,
        matched.ground_truth_m,
        matched.joint_names,
        fps=matched.fps,
        thresholds=thresholds,
        bootstrap_samples=args.bootstrap_samples,
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 2,
        "validation_algorithm_version": 2,
        "prediction": str(args.prediction.resolve()),
        "ground_truth": str(args.ground_truth.resolve()),
        "validation_config": str(args.config.resolve()),
        "joint_map": str(args.joint_map.resolve()) if args.joint_map else None,
        "prediction_identity": {
            "session_id": predicted.metadata.get("session_id"),
            "run_id": predicted.metadata.get("run_id"),
        },
        "ground_truth_identity": {
            "dataset": truth.metadata.get("dataset"),
            "sequence_id": truth.metadata.get("sequence_id"),
        },
        "matched_frame_count": len(matched.match_rows),
        "mapped_joint_names": matched.joint_names,
        "evaluation_fps": matched.fps,
        "validation": result.report,
    }
    report_path = output_dir / "ground_truth_validation_report.json"
    frame_errors_path = output_dir / "ground_truth_frame_errors.csv"
    joint_errors_path = output_dir / "ground_truth_joint_errors.csv"
    angle_errors_path = output_dir / "ground_truth_angle_errors.csv"
    frame_matches_path = output_dir / "ground_truth_frame_matches.csv"
    manifest_path = output_dir / "validation_manifest.json"
    authorization_path = output_dir / "scoring_authorization.json"
    export_session_json(report, report_path)
    _write_rows(result.frame_rows, frame_errors_path)
    _write_rows(result.joint_rows, joint_errors_path)
    _write_rows(result.angle_rows, angle_errors_path)
    _write_rows(matched.match_rows, frame_matches_path)
    _write_validation_manifest(
        manifest_path,
        [args.prediction, args.ground_truth, args.config] + ([args.joint_map] if args.joint_map else []),
        [
            report_path,
            frame_errors_path,
            joint_errors_path,
            angle_errors_path,
            frame_matches_path,
        ],
        output_dir=output_dir,
    )
    authorization = build_scoring_authorization(
        prediction_path=args.prediction,
        ground_truth_path=args.ground_truth,
        validation_config_path=args.config,
        validation_report_path=report_path,
        validation_manifest_path=manifest_path,
        frame_matches_path=frame_matches_path,
        joint_map_path=args.joint_map,
    )
    export_session_json(authorization, authorization_path)

    print(f"status: {result.report['status']}")
    print(f"scoring_ready: {str(authorization['scoring_ready']).lower()}")
    print(f"matched frames: {len(matched.match_rows)}")
    print(f"mapped joints: {len(matched.joint_names)}")
    print(f"MPJPE mm: {result.report['mpjpe_mm']:.3f}")
    print(f"P95 error mm: {result.report['p95_error_mm']:.3f}")
    print(f"report: {report_path}")
    print(f"authorization: {authorization_path}")
    if result.report["status"] != "passed_for_scoring_validation" and not args.allow_failed_quality_gate:
        raise SystemExit(2)


def _write_rows(rows: list[dict[str, object]], path: Path) -> None:
    pd.DataFrame(rows).to_csv(path, index=False, na_rep="")


def _write_validation_manifest(
    manifest_path: Path,
    inputs: list[Path],
    outputs: list[Path],
    *,
    output_dir: Path,
) -> None:
    manifest = {
        "schema_version": 1,
        "inputs": [file_fingerprint(path.resolve()) for path in inputs],
        "outputs": [
            file_fingerprint(path.resolve(), label=path.resolve().relative_to(output_dir).as_posix())
            for path in sorted(outputs)
        ],
    }
    export_session_json(manifest, manifest_path)


if __name__ == "__main__":
    main()
