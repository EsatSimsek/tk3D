from __future__ import annotations

import pytest

from src.run_outputs import create_run_output_tree, mark_run_complete, resolve_latest_run


def test_run_outputs_are_isolated_and_not_overwritten(tmp_path) -> None:
    run_id, paths = create_run_output_tree(tmp_path, "session", "run-1")
    (paths["json"] / "result.json").write_text("{}", encoding="utf-8")
    mark_run_complete(tmp_path, "session", run_id, paths["root"])

    assert resolve_latest_run(tmp_path, "session") == paths["root"]
    with pytest.raises(FileExistsError):
        create_run_output_tree(tmp_path, "session", "run-1")


def test_run_outputs_reject_path_traversal_and_mismatched_completion_root(tmp_path) -> None:
    with pytest.raises(ValueError):
        create_run_output_tree(tmp_path, "../outside", "run-1")

    _, paths = create_run_output_tree(tmp_path, "session", "run-1")
    with pytest.raises(ValueError):
        mark_run_complete(tmp_path, "session", "run-1", paths["root"].parent)
