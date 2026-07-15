from __future__ import annotations

import argparse
import hashlib
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
        "schema_version": 1,
        "validation_algorithm_version": 1,
        "prediction": str(args.prediction.resolve()),
        "ground_truth": str(args.ground_truth.resolve()),
        "matched_frame_count": len(matched.match_rows),
        "mapped_joint_names": matched.joint_names,
        "evaluation_fps": matched.fps,
        "validation": result.report,
    }
    export_session_json(report, output_dir / "ground_truth_validation_report.json")
    _write_rows(result.frame_rows, output_dir / "ground_truth_frame_errors.csv")
    _write_rows(result.joint_rows, output_dir / "ground_truth_joint_errors.csv")
    _write_rows(result.angle_rows, output_dir / "ground_truth_angle_errors.csv")
    _write_rows(matched.match_rows, output_dir / "ground_truth_frame_matches.csv")
    _write_validation_manifest(
        output_dir,
        [args.prediction, args.ground_truth, args.config] + ([args.joint_map] if args.joint_map else []),
    )

    print(f"status: {result.report['status']}")
    print(f"matched frames: {len(matched.match_rows)}")
    print(f"mapped joints: {len(matched.joint_names)}")
    print(f"MPJPE mm: {result.report['mpjpe_mm']:.3f}")
    print(f"P95 error mm: {result.report['p95_error_mm']:.3f}")
    print(f"report: {output_dir / 'ground_truth_validation_report.json'}")
    if result.report["status"] != "passed_for_scoring_validation" and not args.allow_failed_quality_gate:
        raise SystemExit(2)


def _write_rows(rows: list[dict[str, object]], path: Path) -> None:
    pd.DataFrame(rows).to_csv(path, index=False, na_rep="")


def _write_validation_manifest(output_dir: Path, inputs: list[Path]) -> None:
    output_paths = sorted(path for path in output_dir.iterdir() if path.is_file())
    manifest = {
        "schema_version": 1,
        "inputs": [_fingerprint(path.resolve()) for path in inputs],
        "outputs": [_fingerprint(path.resolve(), relative_to=output_dir) for path in output_paths],
    }
    export_session_json(manifest, output_dir / "validation_manifest.json")


def _fingerprint(path: Path, relative_to: Path | None = None) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    label = path.relative_to(relative_to).as_posix() if relative_to is not None else str(path)
    return {"path": label, "size_bytes": path.stat().st_size, "sha256": digest.hexdigest()}


if __name__ == "__main__":
    main()
