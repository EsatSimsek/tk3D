from __future__ import annotations

import json
from pathlib import Path

from src.camera_calibration import load_calibration_bundle
from src.reproducibility import ReproducibilityStatus, check_reproducibility


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures" / "contracts"


def test_clean_checkout_without_research_assets_is_partially_ready(tmp_path) -> None:
    _write_minimal_checkout(tmp_path)

    report = check_reproducibility(
        tmp_path,
        output_root=tmp_path / "verification-output",
        user_home=tmp_path / "empty-home",
    )

    assert report.status is ReproducibilityStatus.PARTIALLY_READY
    assert report.tier1_ready is True
    assert report.current_active_ready is False
    checks = {check.name: check for check in report.checks}
    assert checks["vitpose_checkpoint"].status == "WARN"
    assert checks["active_session"].status == "WARN"
    assert checks["active_calibration"].status == "WARN"
    assert checks["model_config"].status == "PASS"
    json.dumps(report.to_dict(), allow_nan=False)


def test_missing_repository_contract_is_not_ready(tmp_path) -> None:
    _write_minimal_checkout(tmp_path)
    (tmp_path / "config" / "model_config.yaml").unlink()

    report = check_reproducibility(tmp_path, output_root=tmp_path / "output", user_home=tmp_path / "home")

    assert report.status is ReproducibilityStatus.NOT_READY
    assert report.tier1_ready is False


def test_malformed_included_profile_contract_is_not_ready(tmp_path) -> None:
    _write_minimal_checkout(tmp_path)
    (tmp_path / "config" / "scoring" / "poomsae.yaml").write_text("- not\n- a\n- mapping\n", encoding="utf-8")

    report = check_reproducibility(tmp_path, output_root=tmp_path / "output", user_home=tmp_path / "home")

    checks = {check.name: check for check in report.checks}
    assert report.status is ReproducibilityStatus.NOT_READY
    assert checks["profile_poomsae_spec"].status == "FAIL"


def test_versioned_calibration_fixture_is_valid() -> None:
    bundle = load_calibration_bundle(FIXTURES / "calibration_v2.json")

    assert bundle.metadata["calibration_mode"] == "zed_fusion_multiview"
    assert tuple(bundle.calibrations) == ("fixture-camera",)


def _write_minimal_checkout(root: Path) -> None:
    (root / "config" / "scoring" / "profiles").mkdir(parents=True)
    (root / "config" / "mmpose_configs").mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        """
[project]
name = "tk3d"
version = "0.1.0"
[project.scripts]
tk3d-check = "scripts.check_reproducibility:main"
tk3d-multiview = "scripts.run_vitpose_multiview_3d:main"
tk3d-poomsae = "scripts.run_poomsae_scoring:main"
""".strip(),
        encoding="utf-8",
    )
    model_config = (REPOSITORY_ROOT / "config" / "model_config.yaml").read_text(encoding="utf-8")
    (root / "config" / "model_config.yaml").write_text(model_config, encoding="utf-8")
    pose_config = root / "config" / "mmpose_configs" / "wholebody_2d_keypoint" / "vitpose" / "coco-wholebody"
    pose_config.mkdir(parents=True)
    (pose_config / "td-hm_ViTPose-huge_8xb64-210e_coco-wholebody-256x192.py").write_text(
        "# deterministic fixture\n", encoding="utf-8"
    )
    included_paths = {
        "rule_pack": "config/scoring/rules.yaml",
        "poomsae_spec": "config/scoring/poomsae.yaml",
        "movement_timeline": "config/scoring/timeline.yaml",
        "diagnostic_profile": "config/scoring/diagnostics.yaml",
        "accuracy_profile": "config/scoring/accuracy.yaml",
    }
    for value in included_paths.values():
        path = root / value
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("schema_version: 1\n", encoding="utf-8")
    profile = {
        "schema_version": 1,
        "profile_id": "poomsae1_trimmed",
        "session": "outputs/active/source/session.yaml",
        "reference_pose": "outputs/active/runs/reference/json/vitpose_session_3d.json",
        **included_paths,
        "bindings": {
            "session_sha256": "a" * 64,
            "reference_pose_sha256": "b" * 64,
            "movement_timeline_sha256": "c" * 64,
        },
        "videos": [
            {"camera_id": "zed-a", "path": "outputs/active/source/videos/a.avi"},
            {"camera_id": "zed-b", "path": "outputs/active/source/videos/b.avi"},
        ],
    }
    import yaml

    (root / "config" / "scoring" / "profiles" / "poomsae1_trimmed.yaml").write_text(
        yaml.safe_dump(profile, sort_keys=False), encoding="utf-8"
    )
