from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.analyze_pose_for_scoring import _load_scoring_config, _require_scoring_authorization
from src.scoring_engine import build_provisional_score
from src.scoring_readiness import biomechanics_timeseries


def _body_frame(offset: float = 0.0, lean: float = 0.0) -> np.ndarray:
    frame = np.full((133, 3), np.nan, dtype=float)
    frame[5] = [-0.5 + offset, lean, 2.0]
    frame[6] = [0.5 + offset, lean, 2.0]
    frame[7] = [-1.0 + offset, lean / 2.0, 1.5]
    frame[8] = [1.0 + offset, lean / 2.0, 1.5]
    frame[9] = [-1.5 + offset, 0.0, 1.0]
    frame[10] = [1.5 + offset, 0.0, 1.0]
    frame[11] = [-0.35 + offset, 0.0, 1.0]
    frame[12] = [0.35 + offset, 0.0, 1.0]
    frame[13] = [-0.35 + offset, 0.0, 0.5]
    frame[14] = [0.35 + offset, 0.0, 0.5]
    frame[15] = [-0.35 + offset, 0.0, 0.0]
    frame[16] = [0.35 + offset, 0.0, 0.0]
    return frame


def _thresholds() -> dict[str, float]:
    return {
        "trunk_lean_warn_deg": 10.0,
        "knee_angle_front_stance_min_deg": 130.0,
        "balance_min_score": 0.70,
    }


def test_provisional_scoring_builds_explainable_frame_and_step_scores() -> None:
    points = np.stack([_body_frame(index * 0.02) for index in range(6)])
    biomechanics = biomechanics_timeseries(points, fps=30.0)
    quality = [{"ready_for_scoring": True} for _ in range(6)]
    segments = [{"label": "motion_candidate", "start_frame": 1, "end_frame": 4}]

    result = build_provisional_score(points, biomechanics, quality, segments, _thresholds())

    assert result["status"] == "provisional_not_official"
    assert result["overall_score"] is not None
    assert result["overall_score"] > 95.0
    assert len(result["frame_scores"]) == 6
    assert result["step_scores"][0]["score"] > 95.0
    assert result["step_scores"][0]["status"] == "needs_reference_label"
    assert result["errors"] == []


def test_provisional_scoring_never_scores_unreliable_frames_and_reports_lean() -> None:
    points = np.stack([_body_frame(lean=0.4) for _ in range(3)])
    biomechanics = biomechanics_timeseries(points, fps=30.0)
    quality = [{"ready_for_scoring": False} for _ in range(3)]
    segments = [{"label": "pending_motion", "start_frame": 0, "end_frame": 2}]

    result = build_provisional_score(points, biomechanics, quality, segments, _thresholds())

    assert result["overall_score"] is None
    assert all(row["score"] == 0.0 for row in result["frame_scores"])
    codes = {row["code"] for row in result["errors"]}
    assert "unreliable_3d_frame" in codes
    assert "excessive_torso_lean" in codes


def test_scoring_authorization_is_fail_closed() -> None:
    with pytest.raises(SystemExit, match="input path"):
        _require_scoring_authorization({"scoring_ready": False}, allow_unvalidated=False)
    with pytest.raises(SystemExit, match="input path"):
        _require_scoring_authorization({"scoring_ready": True}, allow_unvalidated=False)

    override = _require_scoring_authorization({"scoring_ready": True}, allow_unvalidated=True)
    assert override["scoring_ready"] is False
    assert override["provisional_scoring_ready"] is True
    assert override["embedded_scoring_ready_ignored"] is True


def test_run_local_internal_quality_authorizes_provisional_scoring_without_ground_truth(
    tmp_path: Path,
) -> None:
    run_json = tmp_path / "json"
    run_json.mkdir()
    prediction_path = run_json / "vitpose_session_3d.json"
    payload = {"session_id": "zed_run", "run_id": "rgbd_001", "scoring_ready": False}
    prediction_path.write_text(json.dumps(payload), encoding="utf-8")
    (run_json / "run_quality_report.json").write_text(
        json.dumps(
            {
                "session_id": "zed_run",
                "run_id": "rgbd_001",
                "status": "passed",
                "production_ready_calibration": True,
                "internal_sensor_consistency": {"status": "passed"},
                "external_accuracy": {
                    "status": "not_evaluated_for_this_run",
                    "historical_benchmark_inherited": False,
                },
            }
        ),
        encoding="utf-8",
    )

    authorization = _require_scoring_authorization(
        payload,
        allow_unvalidated=False,
        input_path=prediction_path,
        authorization_path=tmp_path / "missing_external_authorization.json",
    )

    assert authorization["provisional_scoring_ready"] is True
    assert authorization["official_scoring_ready"] is False
    assert authorization["external_ground_truth_required"] is False
    assert authorization["historical_benchmark_inherited"] is False


def test_failed_internal_quality_blocks_provisional_scoring(tmp_path: Path) -> None:
    run_json = tmp_path / "json"
    run_json.mkdir()
    prediction_path = run_json / "vitpose_session_3d.json"
    payload = {"session_id": "zed_run", "run_id": "rgbd_001"}
    prediction_path.write_text(json.dumps(payload), encoding="utf-8")
    (run_json / "run_quality_report.json").write_text(
        json.dumps(
            {
                "session_id": "zed_run",
                "run_id": "rgbd_001",
                "status": "failed",
                "production_ready_calibration": True,
                "internal_sensor_consistency": {"status": "failed"},
                "external_accuracy": {"historical_benchmark_inherited": False},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="internal quality checks"):
        _require_scoring_authorization(
            payload,
            allow_unvalidated=False,
            input_path=prediction_path,
            authorization_path=tmp_path / "missing_external_authorization.json",
        )


def test_project_scoring_config_has_valid_nonduplicated_wt_rules() -> None:
    root = Path(__file__).resolve().parents[1]

    config = _load_scoring_config(str(root / "config" / "scoring_config.yaml"))

    assert config["wt_rules"]["total"] == 10.0
    assert set(config["wt_rules"]["deductions"]) == {
        "minor",
        "major",
        "restart",
        "time_exceeded",
        "boundary_crossing",
    }


def test_scoring_config_rejects_duplicate_yaml_keys(tmp_path) -> None:
    config_path = tmp_path / "duplicate.yaml"
    config_path.write_text(
        "scoring: {}\nscoring: {}\nthresholds: {}\nwt_rules: {}\n",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="duplicate key"):
        _load_scoring_config(str(config_path))
