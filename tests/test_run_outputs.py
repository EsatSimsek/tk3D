from __future__ import annotations

import pytest

from src.artifact_io import load_json_object
from src.run_outputs import (
    create_run_output_tree,
    mark_run_complete,
    mark_run_completed,
    mark_run_failed,
    mark_run_running,
    resolve_latest_run,
)


def test_run_outputs_are_isolated_and_not_overwritten(tmp_path) -> None:
    run_id, paths = create_run_output_tree(tmp_path, "session", "run-1")
    assert load_json_object(paths["root"] / "run_state.json")["status"] == "preparing"
    (paths["json"] / "result.json").write_text("{}", encoding="utf-8")
    mark_run_running(paths["root"], "session", run_id)
    mark_run_complete(tmp_path, "session", run_id, paths["root"])

    assert resolve_latest_run(tmp_path, "session") == paths["root"]
    assert load_json_object(paths["root"] / "run_state.json")["status"] == "completed"
    with pytest.raises(FileExistsError):
        create_run_output_tree(tmp_path, "session", "run-1")


def test_run_outputs_reject_path_traversal_and_mismatched_completion_root(tmp_path) -> None:
    with pytest.raises(ValueError):
        create_run_output_tree(tmp_path, "../outside", "run-1")

    _, paths = create_run_output_tree(tmp_path, "session", "run-1")
    with pytest.raises(ValueError):
        mark_run_complete(tmp_path, "session", "run-1", paths["root"].parent)


def test_failed_or_incomplete_run_never_replaces_latest_success(tmp_path) -> None:
    first_id, first = create_run_output_tree(tmp_path, "session", "success")
    mark_run_complete(tmp_path, "session", first_id, first["root"])

    failed_id, failed = create_run_output_tree(tmp_path, "session", "failed")
    mark_run_running(failed["root"], "session", failed_id)
    mark_run_failed(failed["root"], "session", failed_id, "synthetic processing failure")

    assert resolve_latest_run(tmp_path, "session") == first["root"]
    state = load_json_object(failed["root"] / "run_state.json")
    assert state["status"] == "failed"
    assert state["error"] == "synthetic processing failure"

    _, incomplete = create_run_output_tree(tmp_path, "session", "incomplete")
    mark_run_running(incomplete["root"], "session", "incomplete")
    assert resolve_latest_run(tmp_path, "session") == first["root"]


def test_deferred_multistage_completion_promotes_only_after_final_stage(tmp_path) -> None:
    first_id, first = create_run_output_tree(tmp_path, "session", "previous-success")
    mark_run_complete(tmp_path, "session", first_id, first["root"])

    combined_id, combined = create_run_output_tree(tmp_path, "session", "combined")
    mark_run_running(combined["root"], "session", combined_id)
    mark_run_completed(combined["root"], "session", combined_id)

    assert resolve_latest_run(tmp_path, "session") == first["root"]

    mark_run_running(combined["root"], "session", combined_id)
    assert resolve_latest_run(tmp_path, "session") == first["root"]

    mark_run_complete(tmp_path, "session", combined_id, combined["root"])
    assert resolve_latest_run(tmp_path, "session") == combined["root"]
