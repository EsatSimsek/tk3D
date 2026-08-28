from __future__ import annotations

import json

import pytest

from src import performance


def _report(collector: performance.PerformanceCollector, **kwargs):
    return collector.build_report(
        workflow="test_workflow",
        run_identity={"run_id": "test"},
        input_identity=[],
        environment_identity={},
        config_identity={},
        model_identity=[],
        calibration_identity={},
        processed_frame_count=2,
        **kwargs,
    )


def test_collector_keeps_parent_and_child_timings_separate(monkeypatch) -> None:
    monkeypatch.setattr(performance, "_optional_torch", lambda: None)
    collector = performance.PerformanceCollector(benchmark_window=(140, 259))
    collector.record("pose", 2.0, frame_index=140)
    collector.record("forward", 1.5, parent="pose", frame_index=140)
    collector.record("pose", 4.0, frame_index=260)

    report = _report(collector)

    top_level = report["stage_timings"]["top_level_parent_stages"]
    children = report["stage_timings"]["child_breakdown"]
    window = report["stage_timings"]["benchmark_window_top_level"]
    assert top_level[0]["stage"] == "pose"
    assert top_level[0]["total_wall_seconds"] == pytest.approx(6.0)
    assert children[0]["stage"] == "forward"
    assert children[0]["parent"] == "pose"
    assert window[0]["calls"] == 1
    assert window[0]["total_wall_seconds"] == pytest.approx(2.0)


def test_context_helpers_are_noop_without_active_collector() -> None:
    with performance.profile_stage("unused"):
        value = 3
    with performance.cuda_event_stage("unused_gpu", parent="unused"):
        value += 1
    assert value == 4


def test_report_records_matched_profiler_overhead(monkeypatch) -> None:
    monkeypatch.setattr(performance, "_optional_torch", lambda: None)
    collector = performance.PerformanceCollector()
    collector.started_perf_counter -= 10.0

    report = _report(collector, profiler_baseline_seconds=8.0)

    overhead = report["profiler_overhead_estimate"]
    assert overhead["status"] == "estimated_from_matched_profile_disabled_wall_time"
    assert overhead["seconds"] == pytest.approx(2.0, abs=0.1)
    assert overhead["percent"] == pytest.approx(25.0, abs=1.0)


def test_performance_report_round_trip(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(performance, "_optional_torch", lambda: None)
    collector = performance.PerformanceCollector()
    collector.record("stage", 0.25)
    report = _report(collector)
    path = tmp_path / "performance_report.json"

    performance.write_performance_report(path, report)

    loaded = json.loads(path.read_text(encoding="utf-8"))
    performance.validate_performance_report(loaded)
    assert loaded["schema_version"] == performance.PERFORMANCE_REPORT_SCHEMA_VERSION


def test_performance_report_validation_rejects_missing_fields(monkeypatch) -> None:
    monkeypatch.setattr(performance, "_optional_torch", lambda: None)
    report = _report(performance.PerformanceCollector())
    del report["workflow"]

    with pytest.raises(ValueError, match="Performance report keys"):
        performance.validate_performance_report(report)
