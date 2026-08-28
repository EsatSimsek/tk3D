from __future__ import annotations

import json
from pathlib import Path

import pytest

from src import multiview_application
from src.poomsae_scoring import application as poomsae_application


def test_multiview_application_rejects_invalid_sampling_before_io() -> None:
    options = multiview_application.MultiviewRunOptions(
        session="missing.yaml",
        stride=0,
    )

    with pytest.raises(ValueError, match="stride"):
        multiview_application.run_multiview_pose(options)


def test_poomsae_application_returns_explicit_result(monkeypatch, tmp_path: Path) -> None:
    summary_path = tmp_path / "run" / "json" / "poomsae_scoring_summary.json"
    summary_path.parent.mkdir(parents=True)
    summary = {"run": {"root": str(tmp_path / "run")}, "status": "diagnostics_only_no_score"}
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    monkeypatch.setattr(poomsae_application, "run_workflow", lambda **kwargs: summary_path)

    result = poomsae_application.run_poomsae_analysis(profile_value="fixture")

    assert result.summary_path == summary_path
    assert result.run_root == tmp_path / "run"
    assert result.summary == summary
