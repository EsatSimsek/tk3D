from __future__ import annotations

import contextlib
import contextvars
import math
import platform
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from src.artifact_io import write_json_exclusive


PERFORMANCE_REPORT_SCHEMA_VERSION = 1

_ACTIVE_COLLECTOR: contextvars.ContextVar[PerformanceCollector | None] = contextvars.ContextVar(
    "tk3d_performance_collector",
    default=None,
)
_ACTIVE_TAGS: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "tk3d_performance_tags",
    default={},
)


def performance_profiling_active() -> bool:
    return _ACTIVE_COLLECTOR.get() is not None


@dataclass(slots=True)
class _Sample:
    stage: str
    parent: str | None
    wall_seconds: float
    frame_index: int | None
    tags: dict[str, Any] = field(default_factory=dict)
    gpu_seconds: float | None = None
    gpu_timing_status: str = "not_requested"
    cuda_start: Any | None = None
    cuda_end: Any | None = None


class PerformanceCollector:
    """Small optional hierarchical wall/CUDA timing collector."""

    def __init__(self, *, benchmark_window: tuple[int, int] | None = None) -> None:
        if benchmark_window is not None:
            start, end = benchmark_window
            if start < 0 or end < start:
                raise ValueError("benchmark_window must be an inclusive non-negative range")
        self.benchmark_window = benchmark_window
        self.started_perf_counter = time.perf_counter()
        self.samples: list[_Sample] = []
        self.notes: list[str] = []
        self._cuda_peak_reset = False

    @contextlib.contextmanager
    def activate(self, **tags: Any) -> Iterator[None]:
        collector_token = _ACTIVE_COLLECTOR.set(self)
        tags_token = _ACTIVE_TAGS.set({**_ACTIVE_TAGS.get(), **tags})
        try:
            yield
        finally:
            _ACTIVE_TAGS.reset(tags_token)
            _ACTIVE_COLLECTOR.reset(collector_token)

    @contextlib.contextmanager
    def stage(
        self,
        name: str,
        *,
        parent: str | None = None,
        frame_index: int | None = None,
        tags: dict[str, Any] | None = None,
    ) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self.record(
                name,
                time.perf_counter() - started,
                parent=parent,
                frame_index=frame_index,
                tags=tags,
            )

    def record(
        self,
        name: str,
        wall_seconds: float,
        *,
        parent: str | None = None,
        frame_index: int | None = None,
        tags: dict[str, Any] | None = None,
    ) -> None:
        if not math.isfinite(wall_seconds) or wall_seconds < 0.0:
            raise ValueError("wall_seconds must be finite and non-negative")
        self.samples.append(
            _Sample(
                stage=name,
                parent=parent,
                wall_seconds=float(wall_seconds),
                frame_index=frame_index,
                tags=dict(tags or {}),
            )
        )

    def reset_cuda_peak_memory(self) -> None:
        torch = _optional_torch()
        if torch is None or not torch.cuda.is_available():
            return
        torch.cuda.reset_peak_memory_stats()
        self._cuda_peak_reset = True

    def total_wall(self, *stage_names: str, window_only: bool = False) -> float:
        names = set(stage_names)
        return float(
            sum(
                sample.wall_seconds
                for sample in self.samples
                if sample.stage in names and (not window_only or self._in_window(sample))
            )
        )

    def first_wall(self, stage_name: str) -> float | None:
        return next(
            (sample.wall_seconds for sample in self.samples if sample.stage == stage_name),
            None,
        )

    def call_count(self, stage_name: str, *, window_only: bool = False) -> int:
        return sum(
            1
            for sample in self.samples
            if sample.stage == stage_name and (not window_only or self._in_window(sample))
        )

    def build_report(
        self,
        *,
        workflow: str,
        run_identity: dict[str, Any],
        input_identity: Any,
        environment_identity: dict[str, Any],
        config_identity: Any,
        model_identity: Any,
        calibration_identity: Any,
        processed_frame_count: int,
        runtime_summary: dict[str, Any] | None = None,
        limitations: list[str] | None = None,
        profiler_baseline_seconds: float | None = None,
    ) -> dict[str, Any]:
        self._resolve_cuda_events()
        total_runtime = time.perf_counter() - self.started_perf_counter
        top_level = _summaries(self.samples, parent=None)
        child_breakdown = _child_summaries(self.samples)
        window_samples = [sample for sample in self.samples if self._in_window(sample)]
        window_top_level = _summaries(window_samples, parent=None)
        torch_identity, gpu_memory = _torch_environment_and_memory(self._cuda_peak_reset)
        environment = dict(environment_identity)
        if torch_identity:
            environment["torch_runtime"] = torch_identity
        if profiler_baseline_seconds is not None and profiler_baseline_seconds <= 0.0:
            raise ValueError("profiler_baseline_seconds must be positive when provided")
        overhead_seconds = (
            None if profiler_baseline_seconds is None else total_runtime - profiler_baseline_seconds
        )
        overhead_percent = (
            None
            if profiler_baseline_seconds is None
            else 100.0 * overhead_seconds / profiler_baseline_seconds
        )
        report = {
            "schema_version": PERFORMANCE_REPORT_SCHEMA_VERSION,
            "workflow": workflow,
            "run_identity": run_identity,
            "input_identity": input_identity,
            "environment_identity": environment,
            "config_identity": config_identity,
            "model_identity": model_identity,
            "calibration_identity": calibration_identity,
            "processed_frame_count": int(processed_frame_count),
            "benchmark_window": (
                None
                if self.benchmark_window is None
                else {
                    "start_frame_inclusive": self.benchmark_window[0],
                    "end_frame_inclusive": self.benchmark_window[1],
                    "label": f"STEADY_STATE_{self.benchmark_window[0]}_{self.benchmark_window[1]}",
                }
            ),
            "total_runtime_seconds": float(total_runtime),
            "runtime_summary": dict(runtime_summary or {}),
            "timing_semantics": {
                "top_level": "Parent/top-level stages; do not add child timings to these totals.",
                "children": "Nested diagnostic breakdown already contained by its declared parent.",
                "wall_clock": "time.perf_counter",
                "gpu": "PyTorch CUDA Events where stage boundaries expose PyTorch GPU work.",
            },
            "stage_timings": {
                "top_level_parent_stages": top_level,
                "child_breakdown": child_breakdown,
                "benchmark_window_top_level": window_top_level,
            },
            "gpu_memory": gpu_memory,
            "system_resources": {
                "process_ram_peak_bytes": None,
                "coarse_cpu_utilization_percent": None,
                "status": "unavailable_without_an_existing_lightweight_process_monitor",
            },
            "profiler_overhead_estimate": {
                "status": (
                    "requires_matched_profile_disabled_run"
                    if profiler_baseline_seconds is None
                    else "estimated_from_matched_profile_disabled_wall_time"
                ),
                "baseline_seconds": profiler_baseline_seconds,
                "seconds": overhead_seconds,
                "percent": overhead_percent,
            },
            "limitations": [*self.notes, *(limitations or [])],
        }
        validate_performance_report(report)
        return report

    def _in_window(self, sample: _Sample) -> bool:
        if self.benchmark_window is None or sample.frame_index is None:
            return False
        return self.benchmark_window[0] <= sample.frame_index <= self.benchmark_window[1]

    def _resolve_cuda_events(self) -> None:
        pending = [sample for sample in self.samples if sample.cuda_end is not None]
        if not pending:
            return
        torch = _optional_torch()
        if torch is None or not torch.cuda.is_available():
            for sample in pending:
                sample.gpu_timing_status = "cuda_unavailable"
            return
        torch.cuda.synchronize()
        for sample in pending:
            try:
                sample.gpu_seconds = float(sample.cuda_start.elapsed_time(sample.cuda_end)) / 1000.0
                sample.gpu_timing_status = "measured_cuda_event"
            except RuntimeError:
                sample.gpu_timing_status = "cuda_event_resolution_failed"
            finally:
                sample.cuda_start = None
                sample.cuda_end = None


@contextlib.contextmanager
def profile_stage(
    name: str,
    *,
    parent: str | None = None,
    frame_index: int | None = None,
    tags: dict[str, Any] | None = None,
) -> Iterator[None]:
    collector = _ACTIVE_COLLECTOR.get()
    if collector is None:
        yield
        return
    inherited = _ACTIVE_TAGS.get()
    effective_frame = frame_index if frame_index is not None else inherited.get("frame_index")
    effective_tags = {**inherited, **(tags or {})}
    effective_tags.pop("frame_index", None)
    with collector.stage(
        name,
        parent=parent,
        frame_index=effective_frame,
        tags=effective_tags,
    ):
        yield


def record_profile_stage(
    name: str,
    wall_seconds: float,
    *,
    parent: str | None = None,
    frame_index: int | None = None,
    tags: dict[str, Any] | None = None,
) -> None:
    collector = _ACTIVE_COLLECTOR.get()
    if collector is None:
        return
    inherited = _ACTIVE_TAGS.get()
    effective_frame = frame_index if frame_index is not None else inherited.get("frame_index")
    effective_tags = {**inherited, **(tags or {})}
    effective_tags.pop("frame_index", None)
    collector.record(
        name,
        wall_seconds,
        parent=parent,
        frame_index=effective_frame,
        tags=effective_tags,
    )


@contextlib.contextmanager
def cuda_event_stage(
    name: str,
    *,
    parent: str,
) -> Iterator[None]:
    collector = _ACTIVE_COLLECTOR.get()
    torch = _optional_torch()
    if collector is None or torch is None or not torch.cuda.is_available():
        yield
        return
    inherited = _ACTIVE_TAGS.get()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    wall_started = time.perf_counter()
    start_event.record()
    try:
        yield
    finally:
        end_event.record()
        collector.samples.append(
            _Sample(
                stage=name,
                parent=parent,
                wall_seconds=time.perf_counter() - wall_started,
                frame_index=inherited.get("frame_index"),
                tags={key: value for key, value in inherited.items() if key != "frame_index"},
                gpu_timing_status="pending_cuda_event",
                cuda_start=start_event,
                cuda_end=end_event,
            )
        )


@contextlib.contextmanager
def synchronized_cuda_wall_stage(
    name: str,
    *,
    parent: str | None = None,
    synchronize_before: bool = True,
) -> Iterator[None]:
    collector = _ACTIVE_COLLECTOR.get()
    if collector is None:
        yield
        return
    started = time.perf_counter()
    torch = _optional_torch()
    if synchronize_before and torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()
    try:
        yield
    finally:
        if torch is not None and torch.cuda.is_available():
            torch.cuda.synchronize()
        inherited = _ACTIVE_TAGS.get()
        collector.record(
            name,
            time.perf_counter() - started,
            parent=parent,
            frame_index=inherited.get("frame_index"),
            tags={
                **{key: value for key, value in inherited.items() if key != "frame_index"},
                "timing_boundary": "cuda_synchronized_wall",
            },
        )


def write_performance_report(path: str | Path, report: dict[str, Any]) -> Path:
    validate_performance_report(report)
    destination = Path(path)
    write_json_exclusive(destination, report)
    return destination


def validate_performance_report(report: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "workflow",
        "run_identity",
        "input_identity",
        "environment_identity",
        "config_identity",
        "model_identity",
        "calibration_identity",
        "processed_frame_count",
        "benchmark_window",
        "total_runtime_seconds",
        "runtime_summary",
        "timing_semantics",
        "stage_timings",
        "gpu_memory",
        "system_resources",
        "profiler_overhead_estimate",
        "limitations",
    }
    if set(report) != required:
        raise ValueError(
            "Performance report keys are invalid: "
            f"missing={sorted(required - set(report))}, unexpected={sorted(set(report) - required)}"
        )
    if report["schema_version"] != PERFORMANCE_REPORT_SCHEMA_VERSION:
        raise ValueError("Unsupported performance report schema_version")
    if not isinstance(report["workflow"], str) or not report["workflow"]:
        raise ValueError("Performance report workflow must be a non-empty string")
    if isinstance(report["processed_frame_count"], bool) or not isinstance(
        report["processed_frame_count"], int
    ):
        raise ValueError("processed_frame_count must be an integer")
    if report["processed_frame_count"] < 0:
        raise ValueError("processed_frame_count must be non-negative")
    total_runtime = report["total_runtime_seconds"]
    if not isinstance(total_runtime, (int, float)) or not math.isfinite(total_runtime) or total_runtime < 0:
        raise ValueError("total_runtime_seconds must be finite and non-negative")
    timings = report["stage_timings"]
    if not isinstance(timings, dict) or set(timings) != {
        "top_level_parent_stages",
        "child_breakdown",
        "benchmark_window_top_level",
    }:
        raise ValueError("stage_timings has an invalid structure")
    for group in timings.values():
        if not isinstance(group, list):
            raise ValueError("Every stage timing group must be a list")
        for item in group:
            _validate_stage_summary(item)


def _validate_stage_summary(item: Any) -> None:
    required = {
        "stage",
        "parent",
        "calls",
        "total_wall_seconds",
        "mean_wall_seconds",
        "median_wall_seconds",
        "p95_wall_seconds",
        "min_wall_seconds",
        "max_wall_seconds",
        "total_gpu_seconds",
        "gpu_timed_calls",
    }
    if not isinstance(item, dict) or set(item) != required:
        raise ValueError("Invalid stage timing summary")
    if not isinstance(item["calls"], int) or item["calls"] < 1:
        raise ValueError("Stage timing calls must be a positive integer")


def _summaries(samples: list[_Sample], *, parent: str | None) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str | None], list[_Sample]] = {}
    for sample in samples:
        if sample.parent == parent:
            grouped.setdefault((sample.stage, sample.parent), []).append(sample)
    summaries = [_summarize(stage_samples) for stage_samples in grouped.values()]
    return sorted(summaries, key=lambda item: item["total_wall_seconds"], reverse=True)


def _child_summaries(samples: list[_Sample]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[_Sample]] = {}
    for sample in samples:
        if sample.parent is not None:
            grouped.setdefault((sample.stage, sample.parent), []).append(sample)
    summaries = [_summarize(stage_samples) for stage_samples in grouped.values()]
    return sorted(summaries, key=lambda item: (str(item["parent"]), -item["total_wall_seconds"]))


def _summarize(samples: list[_Sample]) -> dict[str, Any]:
    wall = np.asarray([sample.wall_seconds for sample in samples], dtype=float)
    gpu = [sample.gpu_seconds for sample in samples if sample.gpu_seconds is not None]
    return {
        "stage": samples[0].stage,
        "parent": samples[0].parent,
        "calls": len(samples),
        "total_wall_seconds": float(np.sum(wall)),
        "mean_wall_seconds": float(np.mean(wall)),
        "median_wall_seconds": float(statistics.median(wall.tolist())),
        "p95_wall_seconds": float(np.percentile(wall, 95)),
        "min_wall_seconds": float(np.min(wall)),
        "max_wall_seconds": float(np.max(wall)),
        "total_gpu_seconds": None if not gpu else float(sum(gpu)),
        "gpu_timed_calls": len(gpu),
    }


def _optional_torch() -> Any | None:
    try:
        import torch
    except ImportError:
        return None
    return torch


def _torch_environment_and_memory(peak_reset: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    torch = _optional_torch()
    if torch is None:
        return {}, {"status": "torch_unavailable", "peak_allocated_mib": None, "peak_reserved_mib": None}
    identity = {
        "torch_version": str(torch.__version__),
        "cuda_version": getattr(torch.version, "cuda", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_names": (
            [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
            if torch.cuda.is_available()
            else []
        ),
    }
    if not torch.cuda.is_available():
        return identity, {"status": "cuda_unavailable", "peak_allocated_mib": None, "peak_reserved_mib": None}
    divisor = 1024.0 * 1024.0
    return identity, {
        "status": "pytorch_peak_only",
        "peak_stats_reset_at_benchmark_start": peak_reset,
        "peak_allocated_mib": float(torch.cuda.max_memory_allocated()) / divisor,
        "peak_reserved_mib": float(torch.cuda.max_memory_reserved()) / divisor,
        "limitation": "PyTorch counters exclude memory allocated directly by the ZED SDK.",
    }


def basic_environment_identity() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
