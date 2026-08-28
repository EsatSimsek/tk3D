from __future__ import annotations

import argparse
import sys

from src import multiview_application as _application

MultiviewQualityError = _application.MultiviewQualityError
MultiviewRunOptions = _application.MultiviewRunOptions
MultiviewRunResult = _application.MultiviewRunResult
run_multiview_pose = _application.run_multiview_pose


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the canonical ViTPose-Huge WholeBody multi-view 3D reconstruction pipeline."
    )
    parser.add_argument(
        "--session",
        required=True,
        help="Session YAML defining cameras, synchronization and optional ZED depth sources.",
    )
    parser.add_argument(
        "--model-config",
        default="config/model_config.yaml",
        help="Model and reconstruction config (default: config/model_config.yaml).",
    )
    parser.add_argument(
        "--output-root",
        default="outputs",
        help="Root for immutable session run directories (default: outputs).",
    )
    parser.add_argument(
        "--max-frames", type=int, default=None, help="Maximum sampled inference frames. Omit for full video duration."
    )
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=None,
        help=(
            "Optional odd smoothing window. By default the configured window is used for stride=1, "
            "while sparse stride runs use window=1 to avoid blending distant moments."
        ),
    )
    parser.add_argument(
        "--max-cameras",
        type=int,
        default=None,
        help="Optional limit for faster tests; default uses all session cameras.",
    )
    parser.add_argument("--progress-every", type=int, default=1, help="Print progress every N written frames.")
    parser.add_argument(
        "--output-fps", type=float, default=None, help="Playback FPS. Defaults to the source video FPS."
    )
    parser.add_argument("--run-id", default=None, help="Unique output run identifier; defaults to a UTC timestamp.")
    parser.add_argument(
        "--allow-approximate-calibration",
        action="store_true",
        help="Explicitly allow non-metric two-camera preview calibration. Never use for scoring.",
    )
    parser.add_argument(
        "--allow-low-quality-output",
        action="store_true",
        help="Keep diagnostic files when quality gates fail; the run is not promoted as latest.",
    )
    parser.add_argument("--defer-latest", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--profile-performance",
        action="store_true",
        help="Write a separate performance_report.json; scientific artifacts are unchanged.",
    )
    parser.add_argument(
        "--benchmark-window-start",
        type=int,
        default=140,
        help="Inclusive sampled-frame index where steady-state profiling begins (default: 140).",
    )
    parser.add_argument(
        "--benchmark-window-end",
        type=int,
        default=259,
        help="Inclusive sampled-frame index where steady-state profiling ends (default: 259).",
    )
    parser.add_argument(
        "--profiler-baseline-seconds",
        type=float,
        default=None,
        help="Matched profiling-disabled wall time used only for overhead estimation.",
    )
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    if args.stride < 1:
        parser.error("--stride must be a positive integer")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be positive when provided")
    if args.benchmark_window_start < 0 or args.benchmark_window_end < args.benchmark_window_start:
        parser.error("benchmark window must be an inclusive non-negative range")
    if args.profiler_baseline_seconds is not None and args.profiler_baseline_seconds <= 0.0:
        parser.error("--profiler-baseline-seconds must be positive")
    options = MultiviewRunOptions(
        session=args.session,
        model_config=args.model_config,
        output_root=args.output_root,
        max_frames=args.max_frames,
        stride=args.stride,
        smoothing_window=args.smoothing_window,
        max_cameras=args.max_cameras,
        progress_every=args.progress_every,
        output_fps=args.output_fps,
        run_id=args.run_id,
        allow_approximate_calibration=args.allow_approximate_calibration,
        allow_low_quality_output=args.allow_low_quality_output,
        promote_latest=not args.defer_latest,
        invocation_argv=tuple(sys.argv),
        profile_performance=args.profile_performance,
        benchmark_window_start=args.benchmark_window_start,
        benchmark_window_end=args.benchmark_window_end,
        profiler_baseline_seconds=args.profiler_baseline_seconds,
    )
    try:
        run_multiview_pose(options)
    except MultiviewQualityError as exc:
        parser.exit(1, f"{exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
