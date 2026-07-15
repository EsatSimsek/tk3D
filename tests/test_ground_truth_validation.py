from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.coordinate_system import ANALYSIS_COORDINATE_SYSTEM
from src.data_structures import COCO_BODY_JOINT_NAMES
from src.ground_truth_io import PoseSequence, load_pose_sequence_json, match_pose_sequences
from src.ground_truth_validation import evaluate_ground_truth_3d


def _body_sequence(frame_count: int = 20) -> np.ndarray:
    frame = np.full((17, 3), np.nan, dtype=float)
    frame[0] = [0.0, 0.0, 1.8]
    frame[5] = [-0.2, 0.0, 1.5]
    frame[6] = [0.2, 0.0, 1.5]
    frame[7] = [-0.45, 0.0, 1.25]
    frame[8] = [0.45, 0.0, 1.25]
    frame[9] = [-0.65, 0.0, 1.0]
    frame[10] = [0.65, 0.0, 1.0]
    frame[11] = [-0.15, 0.0, 1.0]
    frame[12] = [0.15, 0.0, 1.0]
    frame[13] = [-0.15, 0.0, 0.5]
    frame[14] = [0.15, 0.0, 0.5]
    frame[15] = [-0.15, 0.0, 0.0]
    frame[16] = [0.15, 0.0, 0.0]
    return np.stack([frame + [0.01 * index, 0.0, 0.0] for index in range(frame_count)])


def _permissive_thresholds() -> dict[str, float]:
    return {
        "min_evaluated_frames": 1,
        "min_valid_joint_ratio": 0.70,
        "max_mpjpe_mm": 1.0,
        "max_p95_error_mm": 1.0,
        "max_root_relative_mpjpe_mm": 1.0,
        "min_pck_100mm": 1.0,
        "max_angle_mae_deg": 0.01,
        "max_velocity_mae_mps": 0.01,
        "max_acceleration_mae_mps2": 0.01,
        "max_bone_length_cv_percent": 0.01,
    }


def test_exact_ground_truth_passes_metric_quality_gates() -> None:
    truth = _body_sequence()

    result = evaluate_ground_truth_3d(
        truth.copy(),
        truth,
        COCO_BODY_JOINT_NAMES,
        fps=30.0,
        thresholds=_permissive_thresholds(),
        bootstrap_samples=100,
    )

    assert result.report["status"] == "passed_for_scoring_validation"
    assert result.report["mpjpe_mm"] == 0.0
    assert result.report["root_relative_mpjpe_mm"] == 0.0
    assert result.report["pa_mpjpe_mm"] < 1e-9
    assert result.report["pck_50mm"] == 1.0
    assert len(result.joint_rows) == 17
    assert len(result.angle_rows) == 8


def test_global_translation_is_visible_but_diagnostic_alignments_remove_it() -> None:
    truth = _body_sequence()
    predicted = truth + np.asarray([0.10, -0.20, 0.30])

    result = evaluate_ground_truth_3d(
        predicted,
        truth,
        COCO_BODY_JOINT_NAMES,
        fps=30.0,
        thresholds=_permissive_thresholds(),
        bootstrap_samples=0,
    )

    assert result.report["status"] == "failed_ground_truth_quality_gate"
    assert result.report["mpjpe_mm"] == pytest.approx(374.1657, rel=1e-4)
    assert result.report["root_relative_mpjpe_mm"] < 1e-9
    assert result.report["pa_mpjpe_mm"] < 1e-9
    assert "mpjpe" in result.report["failed_gates"]


def test_timestamp_matching_supports_video_and_mocap_at_different_fps() -> None:
    predicted_points = _body_sequence(frame_count=3)
    truth_points = np.repeat(predicted_points, 4, axis=0)
    predicted = PoseSequence(
        predicted_points,
        list(COCO_BODY_JOINT_NAMES),
        None,
        np.asarray([0.0, 1 / 15, 2 / 15]),
        15.0,
        {},
    )
    truth = PoseSequence(
        truth_points,
        list(COCO_BODY_JOINT_NAMES),
        None,
        np.arange(12) / 60.0,
        60.0,
        {},
    )

    matched = match_pose_sequences(predicted, truth)

    assert matched.predicted_m.shape == (3, 17, 3)
    assert [row["ground_truth_array_idx"] for row in matched.match_rows] == [0, 4, 8]
    assert all(abs(row["time_delta_ms"]) < 1e-9 for row in matched.match_rows)


def test_pose_loader_rejects_undeclared_coordinate_system(tmp_path) -> None:
    source = tmp_path / "pose.json"
    source.write_text(
        json.dumps(
            {
                "coordinate_system": {"name": "unknown"},
                "joint_names": list(COCO_BODY_JOINT_NAMES),
                "keypoints_3d_ground_truth": _body_sequence(frame_count=1).tolist(),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="TK3D analysis coordinate system"):
        load_pose_sequence_json(source, "keypoints_3d_ground_truth", require_joint_names=True)


def test_pose_loader_accepts_explicit_metric_analysis_coordinates(tmp_path) -> None:
    source = tmp_path / "pose.json"
    source.write_text(
        json.dumps(
            {
                "coordinate_system": ANALYSIS_COORDINATE_SYSTEM,
                "joint_names": list(COCO_BODY_JOINT_NAMES),
                "fps": 30.0,
                "keypoints_3d_ground_truth": _body_sequence(frame_count=2).tolist(),
            }
        ),
        encoding="utf-8",
    )

    loaded = load_pose_sequence_json(source, "keypoints_3d_ground_truth", require_joint_names=True)

    assert loaded.points_m.shape == (2, 17, 3)
    assert loaded.fps == 30.0


def test_ground_truth_cli_writes_auditable_reports(tmp_path) -> None:
    root = Path(__file__).resolve().parents[1]
    points = _body_sequence(frame_count=20)
    common = {
        "coordinate_system": ANALYSIS_COORDINATE_SYSTEM,
        "joint_names": list(COCO_BODY_JOINT_NAMES),
        "timestamps_sec": (np.arange(20) / 30.0).tolist(),
        "fps": 30.0,
    }
    prediction_path = tmp_path / "prediction.json"
    truth_path = tmp_path / "truth.json"
    config_path = tmp_path / "validation.yaml"
    output_dir = tmp_path / "report"
    prediction_path.write_text(
        json.dumps({**common, "keypoints_3d_world": points.tolist()}), encoding="utf-8"
    )
    truth_path.write_text(
        json.dumps({**common, "keypoints_3d_ground_truth": points.tolist()}), encoding="utf-8"
    )
    config_path.write_text(
        "thresholds:\n"
        "  min_evaluated_frames: 1\n"
        "  min_valid_joint_ratio: 0.7\n"
        "  max_mpjpe_mm: 1\n"
        "  max_p95_error_mm: 1\n"
        "  max_root_relative_mpjpe_mm: 1\n"
        "  min_pck_100mm: 1\n"
        "  max_angle_mae_deg: 0.01\n"
        "  max_velocity_mae_mps: 0.01\n"
        "  max_acceleration_mae_mps2: 0.01\n"
        "  max_bone_length_cv_percent: 0.01\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "evaluate_ground_truth_3d.py"),
            "--prediction",
            str(prediction_path),
            "--ground-truth",
            str(truth_path),
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--bootstrap-samples",
            "20",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads((output_dir / "ground_truth_validation_report.json").read_text(encoding="utf-8"))
    assert "status: passed_for_scoring_validation" in completed.stdout
    assert report["validation"]["status"] == "passed_for_scoring_validation"
    assert (output_dir / "ground_truth_joint_errors.csv").exists()
    assert (output_dir / "ground_truth_frame_matches.csv").exists()
    manifest = json.loads((output_dir / "validation_manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["inputs"]) == 3
    assert {item["path"] for item in manifest["outputs"]} >= {
        "ground_truth_validation_report.json",
        "ground_truth_joint_errors.csv",
    }


def test_ground_truth_cli_returns_failure_when_quality_gate_fails(tmp_path) -> None:
    root = Path(__file__).resolve().parents[1]
    points = _body_sequence(frame_count=3)
    common = {
        "coordinate_system": ANALYSIS_COORDINATE_SYSTEM,
        "joint_names": list(COCO_BODY_JOINT_NAMES),
        "timestamps_sec": (np.arange(3) / 30.0).tolist(),
        "fps": 30.0,
    }
    prediction_path = tmp_path / "prediction.json"
    truth_path = tmp_path / "truth.json"
    output_dir = tmp_path / "failed_report"
    prediction_path.write_text(
        json.dumps({**common, "keypoints_3d_world": (points + 1.0).tolist()}), encoding="utf-8"
    )
    truth_path.write_text(
        json.dumps({**common, "keypoints_3d_ground_truth": points.tolist()}), encoding="utf-8"
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "evaluate_ground_truth_3d.py"),
            "--prediction",
            str(prediction_path),
            "--ground-truth",
            str(truth_path),
            "--output-dir",
            str(output_dir),
            "--bootstrap-samples",
            "20",
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    report = json.loads((output_dir / "ground_truth_validation_report.json").read_text(encoding="utf-8"))
    assert report["validation"]["status"] == "failed_ground_truth_quality_gate"
