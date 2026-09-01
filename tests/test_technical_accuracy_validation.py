from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from src.poomsae_scoring import ScoringContractError, load_movement_timeline, load_poomsae_spec
from src.poomsae_scoring.technical_accuracy import load_technical_accuracy_profile
from src.poomsae_scoring.technical_accuracy_validation import (
    build_rule_accuracy_validation,
    load_rule_accuracy_validation_profile,
    validate_rule_accuracy_validation_profile,
)
from scripts import run_technical_accuracy_rule_validation as validation_cli


ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = ROOT / "config" / "scoring" / "engineering" / "taegeuk_1_wholebody_diagnostics_v3.yaml"
SPEC_PATH = ROOT / "config" / "scoring" / "poomsae" / "taegeuk_1_jang_v0_draft.yaml"
TIMELINE_PATH = ROOT / "config" / "scoring" / "timelines" / "poomsae1_zed2i_rgbd_rerun_20260802_draft.yaml"
VALIDATION_PATH = ROOT / "config" / "scoring" / "validation" / "taegeuk_1_rule_accuracy_validation_v1.yaml"


@pytest.fixture(scope="module")
def validation_report() -> dict:
    profile = load_technical_accuracy_profile(PROFILE_PATH)
    spec = load_poomsae_spec(SPEC_PATH)
    timeline = load_movement_timeline(TIMELINE_PATH, spec)
    validation = load_rule_accuracy_validation_profile(VALIDATION_PATH)
    return build_rule_accuracy_validation(profile, spec, timeline, validation)


def test_validation_profile_is_strict_and_bound_to_wholebody_133() -> None:
    profile = load_rule_accuracy_validation_profile(VALIDATION_PATH)
    assert profile["required_keypoint_count"] == 133
    assert len(profile["geometry_scenarios"]) == 12

    unknown = deepcopy(profile)
    unknown["unknown"] = True
    with pytest.raises(ScoringContractError, match="keys must be exactly"):
        validate_rule_accuracy_validation_profile(unknown)

    duplicate = deepcopy(profile)
    duplicate["geometry_scenarios"][1]["scenario_id"] = duplicate["geometry_scenarios"][0]["scenario_id"]
    with pytest.raises(ScoringContractError, match="unique"):
        validate_rule_accuracy_validation_profile(duplicate)

    unsupported = deepcopy(profile)
    unsupported["geometry_scenarios"][0]["fixture_mutation"] = "invented"
    with pytest.raises(ScoringContractError, match="unsupported fixture mutation"):
        validate_rule_accuracy_validation_profile(unsupported)

    missing_scenario = deepcopy(profile)
    missing_scenario["geometry_scenarios"].pop()
    with pytest.raises(ScoringContractError, match="inventory must be exact"):
        validate_rule_accuracy_validation_profile(missing_scenario)

    weakened = deepcopy(profile)
    weakened["geometry_scenarios"][1]["target_metrics"].pop()
    with pytest.raises(ScoringContractError, match="scenario contract mismatch"):
        validate_rule_accuracy_validation_profile(weakened)


def test_all_174_rules_have_explicit_validation_depth(validation_report: dict) -> None:
    summary = validation_report["summary"]
    inventory = validation_report["rule_validation_inventory"]

    assert validation_report["status"] == "passed"
    assert summary["rule_count"] == 174
    assert summary["rule_inventory_passed_count"] == 174
    assert summary["active_rule_count"] == 33
    assert summary["rule_state_counts"] == {
        "active_diagnostic": 33,
        "measurement_only": 116,
        "blocked_missing_reference": 17,
        "not_observable_with_current_pipeline": 8,
    }
    assert len({row["rule_id"] for row in inventory}) == 174
    assert all(row["passed"] for row in inventory)
    assert {
        row["validation_depth"] for row in inventory
    } == {
        "threshold_classification_and_fail_closed",
        "evaluator_contract_and_score_neutrality",
        "reference_binding_and_fail_closed",
        "pipeline_limit_and_score_neutrality",
    }


def test_all_133_landmarks_have_explicit_rule_coverage(validation_report: dict) -> None:
    landmarks = validation_report["landmark_validation_inventory"]
    assert validation_report["summary"]["landmark_count"] == 133
    assert validation_report["summary"]["landmark_passed_count"] == 133
    assert [row["landmark_index"] for row in landmarks] == list(range(133))
    assert all(row["declared_rule_count"] > 0 and row["passed"] for row in landmarks)


def test_every_active_rule_gets_all_boundary_and_nonfinite_cases(validation_report: dict) -> None:
    rows = validation_report["classification_cases"]
    expected_cases = {
        "pass",
        "boundary",
        "opposite_boundary",
        "fail",
        "opposite_fail",
        "missing",
        "nan",
        "positive_infinity",
        "negative_infinity",
        "wrong_type",
    }
    by_metric: dict[str, set[str]] = {}
    for row in rows:
        by_metric.setdefault(row["metric_id"], set()).add(row["case_id"])

    assert len(by_metric) == 33
    assert len(rows) == 330
    assert all(cases == expected_cases for cases in by_metric.values())
    assert all(row["passed"] for row in rows)
    assert all(
        row["actual_evaluation"] == "unmeasurable"
        for row in rows
        if row["case_id"] in {"missing", "nan", "positive_infinity", "negative_infinity", "wrong_type"}
    )
    assert any(row["case_id"] == "opposite_fail" and row["applicable"] for row in rows)
    encoded = json.dumps(validation_report, allow_nan=False)
    assert "NaN" not in encoded and "Infinity" not in encoded


def test_geometry_scenarios_cover_fail_closed_sensitivity_and_symmetry(validation_report: dict) -> None:
    scenarios = {row["scenario_id"]: row for row in validation_report["geometry_scenarios"]}
    assert len(scenarios) == 12
    assert all(row["passed"] for row in scenarios.values())
    assert scenarios["body17_contract_rejected"]["details"]["raised"] == "ScoringContractError"
    assert scenarios["left_right_mirror"]["details"]["mirror_involution"] is True
    assert scenarios["left_right_mirror"]["details"]["nonzero_baseline_metrics"] is True
    assert all(scenarios["left_right_mirror"]["details"]["mirror_involution_by_array"].values())
    assert scenarios["valid_direction_binding"]["details"]["evaluated_rule_count"] == 17
    for scenario_id in (
        "missing_face_evidence",
        "missing_hand_evidence",
        "missing_foot_evidence",
        "insufficient_camera_evidence",
        "excessive_reprojection_error",
        "degenerate_face_geometry",
    ):
        observations = scenarios[scenario_id]["details"]["observations"]
        assert all(
            item["baseline_value"] is not None
            and item["state"] == "unmeasurable"
            and item["decision_status"] == "no_candidate"
            for item in observations.values()
        )


def test_validation_report_never_claims_scoring_or_external_accuracy(validation_report: dict) -> None:
    assert validation_report["validation_scope"] == "synthetic_engineering_validation_only"
    assert validation_report["scientific_accuracy_claim_allowed"] is False
    assert validation_report["judge_calibration_claim_allowed"] is False
    assert validation_report["score_effect"] is None
    assert validation_report["deduction_effect"] is None
    assert validation_report["limitations"]
    assert validation_report["readiness"] == {
        "synthetic_contract_validation_ready": True,
        "external_rule_accuracy_ready": False,
        "judge_calibrated_ready": False,
        "production_threshold_ready": False,
        "official_scoring_ready": False,
        "blockers": [
            "no_rule_level_expert_labelled_positive_negative_dataset",
            "no_blinded_judge_agreement_study",
            "no_external_precision_recall_or_false_positive_analysis",
            "temporary_thresholds_not_calibrated_on_independent_data",
        ],
    }


def test_cli_writes_atomic_hashed_artifact_set_and_refuses_reuse(tmp_path: Path, monkeypatch) -> None:
    output_dir = tmp_path / "validation-run"
    argv = [
        "run_technical_accuracy_rule_validation.py",
        "--validation-profile", str(VALIDATION_PATH),
        "--technical-profile", str(PROFILE_PATH),
        "--poomsae-spec", str(SPEC_PATH),
        "--timeline", str(TIMELINE_PATH),
        "--output-json", str(output_dir / "report.json"),
        "--rule-inventory-csv", str(output_dir / "rules.csv"),
        "--landmark-inventory-csv", str(output_dir / "landmarks.csv"),
        "--classification-csv", str(output_dir / "classifications.csv"),
        "--scenario-csv", str(output_dir / "scenarios.csv"),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    assert validation_cli.main() == 0
    report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    manifest = json.loads((output_dir / "validation_manifest.json").read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert len(report["input_files"]) == 4
    assert len(report["implementation_files"]) == 4
    assert len(manifest["artifacts"]) == 5
    assert all(len(row["sha256"]) == 64 and row["size_bytes"] > 0 for row in manifest["artifacts"])
    assert not list(tmp_path.glob(".*.partial-*"))

    with pytest.raises(FileExistsError, match="run directory already exists"):
        validation_cli.main()
