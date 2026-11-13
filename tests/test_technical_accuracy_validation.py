from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from src.poomsae_scoring import ScoringContractError, load_movement_timeline, load_poomsae_spec
from src.poomsae_scoring.technical_accuracy import load_technical_accuracy_profile
from src.poomsae_scoring.technical_accuracy_metrics import METRIC_REGISTRY
from src.poomsae_scoring.technical_accuracy_validation import (
    HEAD_DIRECTION_TEST_YAWS,
    SCENARIO_CONTRACTS,
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
def profile() -> dict:
    return load_technical_accuracy_profile(PROFILE_PATH)


@pytest.fixture(scope="module")
def validation_report(profile: dict) -> dict:
    spec = load_poomsae_spec(SPEC_PATH)
    timeline = load_movement_timeline(TIMELINE_PATH, spec)
    validation = load_rule_accuracy_validation_profile(VALIDATION_PATH)
    return build_rule_accuracy_validation(profile, spec, timeline, validation)


def test_validation_profile_is_strict_and_bound_to_wholebody_133() -> None:
    profile = load_rule_accuracy_validation_profile(VALIDATION_PATH)
    assert profile["required_keypoint_count"] == 133
    assert {row["scenario_id"] for row in profile["geometry_scenarios"]} == set(SCENARIO_CONTRACTS)

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


def test_every_rule_has_explicit_validation_depth(validation_report: dict, profile: dict) -> None:
    summary = validation_report["summary"]
    inventory = validation_report["rule_validation_inventory"]
    rule_count = len(profile["resolved_rules"])

    assert validation_report["status"] == "passed"
    assert summary["rule_count"] == rule_count == len(METRIC_REGISTRY)
    assert summary["rule_inventory_passed_count"] == rule_count
    assert summary["active_rule_count"] == len(profile["active_rules"])
    assert summary["rule_state_counts"] == {
        "active_diagnostic": len(profile["active_rules"]),
        "measurement_only": len(profile["measurement_only_reasons"]),
        "blocked_missing_reference": len(profile["direction_bound_rules"]),
        "not_observable_with_current_pipeline": len(profile["not_observable_rules"]),
    }
    assert len({row["rule_id"] for row in inventory}) == rule_count
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
    assert all(row["passed"] for row in landmarks)
    # Points no evaluator reads are reported as such, never claimed by a rule.
    undeclared = {row["landmark_index"] for row in landmarks if row["declared_rule_count"] == 0}
    assert all(
        row["usage_status"] == "not_declared_by_any_rule" and row["blocking_or_scope_reason"]
        for row in landmarks
        if row["landmark_index"] in undeclared
    )
    jaw_and_lips = {23 + index for index in (*range(0, 8), *range(9, 17), *range(48, 68))}
    # Face nose points are no longer read: head facing uses the body nose and ears (AD-050).
    face_nose = {23 + index for index in range(27, 36)}
    assert undeclared == jaw_and_lips | face_nose


def test_every_active_rule_gets_all_boundary_and_nonfinite_cases(validation_report: dict, profile: dict) -> None:
    rows = validation_report["classification_cases"]
    expected_cases = set(load_rule_accuracy_validation_profile(VALIDATION_PATH)["classification_cases"])
    by_metric: dict[str, set[str]] = {}
    for row in rows:
        by_metric.setdefault(row["metric_id"], set()).add(row["case_id"])

    # Every screening rule (active, or reference-bound with a configured screen) is
    # classified once per threshold context; stance/technique ranges add contexts.
    screening = [
        rule
        for rule in profile["resolved_rules"]
        if rule["status"] in {"active_diagnostic", "blocked_missing_reference"}
        and (rule["threshold"] is not None or rule["expected_boolean"] is not None or rule["contextual_thresholds"])
    ]
    assert set(profile["active_rules"]) <= set(by_metric)
    assert set(by_metric) == {rule["metric_id"] for rule in screening}
    assert len(rows) == len(expected_cases) * sum(len(rule["contextual_thresholds"] or {"default": None}) for rule in screening)
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


def test_geometry_scenarios_cover_fail_closed_sensitivity_and_symmetry(validation_report: dict, profile: dict) -> None:
    scenarios = {row["scenario_id"]: row for row in validation_report["geometry_scenarios"]}
    configured = load_rule_accuracy_validation_profile(VALIDATION_PATH)["geometry_scenarios"]
    assert len(scenarios) == len(configured) + len(HEAD_DIRECTION_TEST_YAWS)
    assert all(row["passed"] for row in scenarios.values())
    assert scenarios["body17_contract_rejected"]["details"]["raised"] == "ScoringContractError"
    assert scenarios["left_right_mirror"]["details"]["mirror_involution"] is True
    assert scenarios["left_right_mirror"]["details"]["nonzero_baseline_metrics"] is True
    assert all(scenarios["left_right_mirror"]["details"]["mirror_involution_by_array"].values())
    direction_screens = [
        rule
        for rule in profile["resolved_rules"]
        if rule["status"] == "blocked_missing_reference"
        and (rule["threshold"] is not None or rule["expected_boolean"] is not None)
    ]
    assert scenarios["valid_direction_binding"]["details"]["evaluated_rule_count"] == len(direction_screens)
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
    assert len(report["implementation_files"]) == 5
    assert "decision_evidence" in report["implementation_files"]
    assert len(manifest["artifacts"]) == 5
    assert all(len(row["sha256"]) == 64 and row["size_bytes"] > 0 for row in manifest["artifacts"])
    assert not list(tmp_path.glob(".*.partial-*"))

    with pytest.raises(FileExistsError, match="run directory already exists"):
        validation_cli.main()
