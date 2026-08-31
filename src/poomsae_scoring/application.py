from __future__ import annotations

import re
import shutil
import subprocess
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.artifact_io import load_json_object, sha256_file, write_json_exclusive
from src.performance import PerformanceCollector, basic_environment_identity, write_performance_report
from src.poomsae_scoring.contracts import load_movement_timeline, load_poomsae_spec, load_yaml_mapping
from src.run_outputs import initialize_run_state, mark_run_complete, mark_run_completed, mark_run_running
from src.video_io import load_session


ROOT = Path(__file__).resolve().parents[2]
_sha256 = sha256_file


_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")
_PROFILE_KEYS = {
    "schema_version",
    "profile_id",
    "session",
    "output_root",
    "reference_pose",
    "model_config",
    "rule_pack",
    "poomsae_spec",
    "movement_timeline",
    "diagnostic_profile",
    "accuracy_diagnostic_profile",
    "accuracy_profile",
    "bindings",
    "processing",
    "videos",
}
_PATH_KEYS = {
    "session",
    "output_root",
    "reference_pose",
    "model_config",
    "rule_pack",
    "poomsae_spec",
    "movement_timeline",
    "diagnostic_profile",
    "accuracy_diagnostic_profile",
    "accuracy_profile",
}
_POOMSAE_PRESENTATION_SCRIPTS = {
    "scripts/build_browser_review_videos.py",
    "scripts/render_poomsae_error_video.py",
    "scripts/create_poomsae_review_report.py",
    "scripts/build_poomsae_run_history.py",
}


class WorkflowError(RuntimeError):
    """Raised when the one-command scoring workflow must fail closed."""


@dataclass(frozen=True, slots=True)
class PoomsaeAnalysisResult:
    summary_path: Path
    run_root: Path
    summary: dict[str, Any]


def run_poomsae_analysis(
    *,
    profile_value: str,
    process_video: bool = False,
    requested_run_id: str | None = None,
    profile_performance: bool = False,
) -> PoomsaeAnalysisResult:
    summary_path = run_workflow(
        profile_value=profile_value,
        process_video=process_video,
        requested_run_id=requested_run_id,
        profile_performance=profile_performance,
    )
    summary = _read_json(summary_path)
    return PoomsaeAnalysisResult(
        summary_path=summary_path,
        run_root=Path(summary["run"]["root"]),
        summary=summary,
    )


def run_workflow(
    *,
    profile_value: str,
    process_video: bool,
    requested_run_id: str | None,
    profile_performance: bool = False,
) -> Path:
    profiler = PerformanceCollector() if profile_performance else None
    run_stage = _run_stage if profiler is None else partial(_run_stage, profiler=profiler)
    profile_path = _resolve_profile(profile_value)
    profile = _load_profile(profile_path)
    paths = {key: _resolve_path(profile[key]) for key in _PATH_KEYS}
    _verify_profile_bindings(profile, paths)
    session = load_session(paths["session"])
    reference_pose = _read_json(paths["reference_pose"])
    if reference_pose.get("session_id") != session.session_id:
        raise WorkflowError("Reference pose session_id does not match the profile session.")

    run_id = requested_run_id or _default_run_id(process_video)
    if not _SAFE_RUN_ID.fullmatch(run_id):
        raise WorkflowError("run_id may contain only letters, numbers, dot, underscore, or hyphen (max 80).")
    run_root = paths["output_root"] / session.session_id / "runs" / run_id
    if run_root.exists():
        raise WorkflowError(f"Output run already exists; refusing to overwrite: {run_root}")

    if process_video:
        _verify_process_inputs(paths["session"], session.session_id, paths["output_root"])
        run_stage(
            "Model ve çalışma ortamı kontrolü",
            "scripts/check_models.py",
            "--session",
            paths["session"],
            "--model-config",
            paths["model_config"],
            "--output-root",
            paths["output_root"],
        )
        processing = profile["processing"]
        run_stage(
            "ViTPose-Huge WholeBody + çok-kamera RGBD 3B işleme",
            "scripts/run_vitpose_multiview_3d.py",
            "--session",
            paths["session"],
            "--model-config",
            paths["model_config"],
            "--output-root",
            paths["output_root"],
            "--stride",
            processing["stride"],
            "--smoothing-window",
            processing["smoothing_window"],
            "--progress-every",
            processing["progress_every"],
            "--run-id",
            run_id,
            "--defer-latest",
        )
        pose_path = run_root / "json" / "vitpose_session_3d.json"
        _require_file(pose_path, "processed 3D pose")
        run_stage(
            "Skorsuz kalite, biomekanik ve segment hazırlığı",
            "scripts/analyze_pose_for_scoring.py",
            "--session",
            paths["session"],
            "--input-json",
            pose_path,
            "--smoothing-window",
            processing["smoothing_window"],
        )
    else:
        pose_path = paths["reference_pose"]
        for directory in ("json", "csv", "config", "review"):
            (run_root / directory).mkdir(parents=True, exist_ok=False)
        initialize_run_state(run_root, session.session_id, run_id)
    mark_run_running(run_root, session.session_id, run_id)

    config_paths = _snapshot_configuration(
        run_root=run_root,
        profile_path=profile_path,
        paths=paths,
        pose_path=pose_path,
        session_id=session.session_id,
        run_id=run_id,
        process_video=process_video,
    )
    if profiler is not None:
        binding_elapsed = time.perf_counter() - profiler.started_perf_counter
        profiler.record(
            "poomsae_binding_snapshot",
            binding_elapsed,
            parent="poomsae_analysis_decisions",
        )
        profiler.record("poomsae_analysis_decisions", binding_elapsed)
    outputs = _output_paths(run_root)
    videos = _resolve_videos(profile["videos"])
    browser_video_args: list[str | Path] = []
    for video in videos:
        browser_video_args.extend(("--camera", f"{video['camera_id']}={video['path']}"))
    run_stage(
        "Tarayıcı uyumlu, zaman çizelgesi korunan kamera videoları",
        "scripts/build_browser_review_videos.py",
        *browser_video_args,
        "--output-dir",
        run_root / "videos" / "browser",
        "--manifest",
        outputs["browser_review_video_manifest"],
    )
    browser_videos = [
        {
            **video,
            "path": (run_root / "videos" / "browser" / f"{video['camera_id']}_browser.mp4").resolve(),
        }
        for video in videos
    ]
    keypoints_2d_csv = pose_path.parent.parent / "csv" / "vitpose_keypoints_2d_flat.csv"
    _require_file(keypoints_2d_csv, "camera-observed WholeBody-133 2D evidence")

    run_stage(
        "Otomatik hareket ve faz sınırı teşhisi (puan değil)",
        "scripts/build_poomsae_automatic_segmentation.py",
        "--pose",
        pose_path,
        "--poomsae-spec",
        config_paths["poomsae_spec"],
        "--reference-timeline",
        config_paths["movement_timeline"],
        "--output-json",
        outputs["automatic_segmentation"],
        "--output-csv",
        outputs["automatic_segmentation_signal"],
    )
    run_stage(
        "WholeBody-133 Poomsae ölçümleri",
        "scripts/run_wholebody_poomsae_diagnostics.py",
        "--pose",
        pose_path,
        "--poomsae-spec",
        config_paths["poomsae_spec"],
        "--timeline",
        config_paths["movement_timeline"],
        "--diagnostic-profile",
        config_paths["diagnostic_profile"],
        "--output-json",
        outputs["wholebody_diagnostics"],
        "--output-csv",
        outputs["wholebody_metrics"],
    )
    run_stage(
        "Kapsamlı Taegeuk 1 teknik-doğruluk teşhisleri (puan değil)",
        "scripts/run_technical_accuracy_diagnostics.py",
        "--pose",
        pose_path,
        "--poomsae-spec",
        config_paths["poomsae_spec"],
        "--timeline",
        config_paths["movement_timeline"],
        "--profile",
        config_paths["accuracy_diagnostic_profile"],
        "--wholebody-diagnostics",
        outputs["wholebody_diagnostics"],
        "--output-json",
        outputs["technical_accuracy_diagnostics"],
        "--coverage-csv",
        outputs["technical_accuracy_coverage"],
        "--landmark-coverage-csv",
        outputs["technical_accuracy_landmark_coverage"],
    )
    run_stage(
        "Hareket ve faz kanıtları",
        "scripts/analyze_poomsae_movement_evidence.py",
        "--pose",
        pose_path,
        "--poomsae-spec",
        config_paths["poomsae_spec"],
        "--timeline",
        config_paths["movement_timeline"],
        "--output-json",
        outputs["movement_evidence"],
        "--output-csv",
        outputs["movement_evidence_csv"],
    )
    run_stage(
        "Duraklama, yanlış hareket ve yanlış duruş teşhisleri",
        "scripts/run_categorical_poomsae_diagnostics.py",
        "--wholebody-diagnostics",
        outputs["wholebody_diagnostics"],
        "--poomsae-spec",
        config_paths["poomsae_spec"],
        "--timeline",
        config_paths["movement_timeline"],
        "--diagnostic-profile",
        config_paths["diagnostic_profile"],
        "--output",
        outputs["categorical_diagnostics"],
    )
    run_stage(
        "Hareket bazlı teknik uygunluk ve güven füzyonu (puan değil)",
        "scripts/build_poomsae_technical_conformance.py",
        "--wholebody-diagnostics",
        outputs["wholebody_diagnostics"],
        "--categorical-diagnostics",
        outputs["categorical_diagnostics"],
        "--poomsae-spec",
        config_paths["poomsae_spec"],
        "--timeline",
        config_paths["movement_timeline"],
        "--output",
        outputs["technical_conformance"],
    )
    run_stage(
        "Presentation teşhisleri (puan değil)",
        "scripts/build_poomsae_presentation_diagnostics.py",
        "--wholebody-diagnostics",
        outputs["wholebody_diagnostics"],
        "--poomsae-spec",
        config_paths["poomsae_spec"],
        "--timeline",
        config_paths["movement_timeline"],
        "--output",
        outputs["presentation_diagnostics"],
    )
    run_stage(
        "RulePack puanlama hazırlık kapısı",
        "scripts/assess_poomsae_scoring_readiness.py",
        "--rule-pack",
        config_paths["rule_pack"],
        "--poomsae-spec",
        config_paths["poomsae_spec"],
        "--timeline",
        config_paths["movement_timeline"],
        "--wholebody-diagnostics",
        outputs["wholebody_diagnostics"],
        "--output",
        outputs["rule_scoring_readiness"],
    )
    run_stage(
        "Kaynak-bağlı Accuracy kararları",
        "scripts/build_source_bound_accuracy_decisions.py",
        "--wholebody-diagnostics",
        outputs["wholebody_diagnostics"],
        "--poomsae-spec",
        config_paths["poomsae_spec"],
        "--timeline",
        config_paths["movement_timeline"],
        "--accuracy-profile",
        config_paths["accuracy_profile"],
        "--observations",
        outputs["categorical_diagnostics"],
        "--output-json",
        outputs["accuracy_decisions"],
    )
    run_stage(
        "Kararları görsel kanıt olaylarına dönüştürme",
        "scripts/build_poomsae_evidence_events.py",
        "--accuracy-decisions",
        outputs["accuracy_decisions"],
        "--poomsae-spec",
        config_paths["poomsae_spec"],
        "--timeline",
        config_paths["movement_timeline"],
        "--wholebody-diagnostics",
        outputs["wholebody_diagnostics"],
        "--technical-accuracy-diagnostics",
        outputs["technical_accuracy_diagnostics"],
        "--output",
        outputs["decision_evidence_events"],
    )
    error_video_args: list[str | Path] = []
    for video in videos:
        error_video_args.extend(("--camera", f"{video['camera_id']}={video['path']}"))
    error_video_args.extend(
        (
            "--keypoints-2d",
            keypoints_2d_csv,
            "--evidence-events",
            outputs["decision_evidence_events"],
            "--output",
            outputs["annotated_error_video"],
            "--manifest",
            outputs["annotated_error_video_manifest"],
        )
    )
    run_stage(
        "İki kameralı işaretli hata videosu",
        "scripts/render_poomsae_error_video.py",
        *error_video_args,
    )
    review_args: list[str | Path] = [
        "--poomsae-spec",
        config_paths["poomsae_spec"],
        "--timeline",
        config_paths["movement_timeline"],
        "--evidence",
        outputs["movement_evidence"],
        "--readiness",
        outputs["rule_scoring_readiness"],
        "--wholebody-diagnostics",
        outputs["wholebody_diagnostics"],
        "--technical-accuracy-diagnostics",
        outputs["technical_accuracy_diagnostics"],
        "--accuracy-decisions",
        outputs["accuracy_decisions"],
        "--decision-evidence-events",
        outputs["decision_evidence_events"],
        "--categorical-diagnostics",
        outputs["categorical_diagnostics"],
        "--technical-conformance",
        outputs["technical_conformance"],
        "--presentation-diagnostics",
        outputs["presentation_diagnostics"],
        "--automatic-segmentation",
        outputs["automatic_segmentation"],
        "--run-history-url",
        "run_history.html",
        "--video-a",
        browser_videos[0]["path"],
        "--video-a-label",
        videos[0]["label"],
        "--video-b",
        browser_videos[1]["path"],
        "--video-b-label",
        videos[1]["label"],
    ]
    for video in browser_videos[2:]:
        review_args.extend(("--video-extra", f"{video['label']}={video['path']}"))
    review_args.extend(("--output", outputs["review_html"], "--manifest", outputs["review_manifest"]))
    run_stage("Senkron kamera inceleme ekranı", "scripts/create_poomsae_review_report.py", *review_args)

    summary_started = time.perf_counter() if profiler is not None else 0.0
    summary = _build_summary(
        profile=profile,
        profile_path=profile_path,
        process_video=process_video,
        run_id=run_id,
        run_root=run_root,
        pose_path=pose_path,
        config_paths=config_paths,
        outputs=outputs,
    )
    _write_json_exclusive(outputs["summary"], summary)
    if profiler is not None:
        summary_elapsed = time.perf_counter() - summary_started
        profiler.record("poomsae_presentation_export", summary_elapsed)
        profiler.record(
            "poomsae_summary_serialization",
            summary_elapsed,
            parent="poomsae_presentation_export",
        )
    run_stage(
        "Koşu geçmişi ve regresyon teşhisi",
        "scripts/build_poomsae_run_history.py",
        "--runs-root",
        run_root.parent,
        "--current-summary",
        outputs["summary"],
        "--output-json",
        outputs["run_history_json"],
        "--output-html",
        outputs["run_history_html"],
    )
    if profiler is not None:
        pose_payload = _read_json(pose_path)
        source_manifest_path = pose_path.parent / "run_manifest.json"
        source_manifest = _read_json(source_manifest_path) if source_manifest_path.is_file() else {}
        config_identity = {
            key: {"path": str(path), "sha256": _sha256(path)}
            for key, path in config_paths.items()
        }
        report = profiler.build_report(
            workflow="tk3d_current_active_poomsae_performance_v1",
            run_identity={"session_id": session.session_id, "run_id": run_id},
            input_identity={
                "reference_pose": {"path": str(pose_path), "sha256": _sha256(pose_path)},
                "source_run_inputs": source_manifest.get("inputs", []),
            },
            environment_identity=source_manifest.get(
                "environment",
                basic_environment_identity(),
            ),
            config_identity=config_identity,
            model_identity=source_manifest.get("models", []),
            calibration_identity=source_manifest.get("calibration"),
            processed_frame_count=int(pose_payload.get("inference_sample_count", 0)),
            runtime_summary={
                "mode": "score_verified_pose" if not process_video else "process_video_then_score",
                "analysis_decision_seconds": profiler.total_wall("poomsae_analysis_decisions"),
                "presentation_export_seconds": profiler.total_wall("poomsae_presentation_export"),
                "wholebody_keypoint_count": 133,
                "reference_pose_frame_count": int(pose_payload.get("inference_sample_count", 0)),
            },
            limitations=[
                "Subprocess stages use wall-clock timing and intentionally include interpreter startup.",
                "Video/report presentation is separated from analytical and decision computation.",
            ],
        )
        write_performance_report(run_root / "json" / "performance_report.json", report)
    if process_video:
        mark_run_complete(paths["output_root"], session.session_id, run_id, run_root)
    else:
        mark_run_completed(run_root, session.session_id, run_id)
    return outputs["summary"]


def _load_profile(path: Path) -> dict[str, Any]:
    _require_file(path, "workflow profile")
    payload = load_yaml_mapping(path, label="workflow profile")
    if set(payload) != _PROFILE_KEYS:
        raise WorkflowError(
            "Workflow profile keys are invalid; "
            f"missing={sorted(_PROFILE_KEYS - set(payload))}, unexpected={sorted(set(payload) - _PROFILE_KEYS)}"
        )
    if payload["schema_version"] != 1 or payload["profile_id"] != path.stem:
        raise WorkflowError("Workflow profile schema_version/profile_id is invalid.")
    for key in _PATH_KEYS:
        if not isinstance(payload[key], str) or not payload[key].strip():
            raise WorkflowError(f"Workflow profile {key} must be a non-empty path.")
    bindings = payload["bindings"]
    expected_bindings = {"session_sha256", "reference_pose_sha256", "movement_timeline_sha256"}
    if not isinstance(bindings, dict) or set(bindings) != expected_bindings:
        raise WorkflowError("Workflow profile bindings are incomplete or unexpected.")
    if any(not _is_sha256(value) for value in bindings.values()):
        raise WorkflowError("Workflow profile bindings must be SHA-256 digests.")
    processing = payload["processing"]
    if not isinstance(processing, dict) or set(processing) != {"stride", "smoothing_window", "progress_every"}:
        raise WorkflowError("Workflow profile processing settings are invalid.")
    if processing["stride"] != 1:
        raise WorkflowError("Scoring processing profile must use stride=1.")
    if not isinstance(processing["smoothing_window"], int) or processing["smoothing_window"] < 1:
        raise WorkflowError("smoothing_window must be a positive integer.")
    if processing["smoothing_window"] % 2 == 0:
        raise WorkflowError("smoothing_window must be odd.")
    if not isinstance(processing["progress_every"], int) or processing["progress_every"] < 1:
        raise WorkflowError("progress_every must be a positive integer.")
    videos = payload["videos"]
    if not isinstance(videos, list) or len(videos) < 2:
        raise WorkflowError("Workflow profile requires at least two camera videos.")
    labels: set[str] = set()
    for video in videos:
        if not isinstance(video, dict) or set(video) != {"camera_id", "label", "path"}:
            raise WorkflowError("Each profile video requires exactly camera_id, label and path.")
        if not all(
            isinstance(video[key], str) and video[key].strip()
            for key in ("camera_id", "label", "path")
        ):
            raise WorkflowError("Profile video camera_id/label/path must be non-empty strings.")
        if video["label"] in labels:
            raise WorkflowError(f"Profile video label is repeated: {video['label']}")
        labels.add(video["label"])
    return payload


def _verify_profile_bindings(profile: dict[str, Any], paths: dict[str, Path]) -> None:
    for key in _PATH_KEYS - {"output_root"}:
        _require_file(paths[key], key)
    checks = {
        "session_sha256": paths["session"],
        "reference_pose_sha256": paths["reference_pose"],
        "movement_timeline_sha256": paths["movement_timeline"],
    }
    for binding, path in checks.items():
        actual = _sha256(path)
        expected = profile["bindings"][binding]
        if actual != expected:
            raise WorkflowError(f"Profile binding mismatch for {binding}: expected {expected}, got {actual}")


def _snapshot_configuration(
    *,
    run_root: Path,
    profile_path: Path,
    paths: dict[str, Path],
    pose_path: Path,
    session_id: str,
    run_id: str,
    process_video: bool,
) -> dict[str, Path]:
    config_root = run_root / "config"
    config_root.mkdir(parents=True, exist_ok=True)
    snapshots = {
        "workflow_profile": config_root / "workflow_profile.yaml",
        "rule_pack": config_root / "rule_pack.yaml",
        "poomsae_spec": config_root / "poomsae_spec.yaml",
        "movement_timeline": config_root / "movement_timeline.yaml",
        "diagnostic_profile": config_root / "wholebody_diagnostic_profile.yaml",
        "accuracy_diagnostic_profile": config_root / "technical_accuracy_diagnostic_profile.yaml",
        "accuracy_profile": config_root / "accuracy_profile.yaml",
    }
    copies = {
        "workflow_profile": profile_path,
        "rule_pack": paths["rule_pack"],
        "poomsae_spec": paths["poomsae_spec"],
        "diagnostic_profile": paths["diagnostic_profile"],
        "accuracy_diagnostic_profile": paths["accuracy_diagnostic_profile"],
        "accuracy_profile": paths["accuracy_profile"],
    }
    for key, source in copies.items():
        shutil.copyfile(source, snapshots[key])

    if process_video:
        timeline = _transfer_timeline_binding(
            timeline_path=paths["movement_timeline"],
            reference_pose_path=paths["reference_pose"],
            new_pose_path=pose_path,
            session_id=session_id,
            run_id=run_id,
        )
        snapshots["movement_timeline"].write_text(
            yaml.safe_dump(timeline, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
    else:
        shutil.copyfile(paths["movement_timeline"], snapshots["movement_timeline"])

    spec = load_poomsae_spec(snapshots["poomsae_spec"])
    load_movement_timeline(snapshots["movement_timeline"], spec)
    return snapshots


def _transfer_timeline_binding(
    *,
    timeline_path: Path,
    reference_pose_path: Path,
    new_pose_path: Path,
    session_id: str,
    run_id: str,
) -> dict[str, Any]:
    timeline = load_yaml_mapping(timeline_path, label="MovementTimeline")
    reference = _read_json(reference_pose_path)
    new = _read_json(new_pose_path)
    source_binding = timeline.get("source_binding")
    if not isinstance(source_binding, dict):
        raise WorkflowError("Reference MovementTimeline has no source binding.")
    if source_binding.get("pose_file_sha256") != _sha256(reference_pose_path):
        raise WorkflowError("Reference MovementTimeline no longer matches its reference pose hash.")
    for label, payload in (("reference", reference), ("new", new)):
        if payload.get("session_id") != session_id:
            raise WorkflowError(f"{label} pose session_id does not match the scoring profile.")
        if payload.get("inference_stride") != 1:
            raise WorkflowError(f"{label} pose must use inference_stride=1 for timeline transfer.")
    reference_frames = np.asarray(reference.get("frame_indices"), dtype=int)
    new_frames = np.asarray(new.get("frame_indices"), dtype=int)
    reference_times = np.asarray(reference.get("timestamps_sec"), dtype=float)
    new_times = np.asarray(new.get("timestamps_sec"), dtype=float)
    if reference_frames.ndim != 1 or new_frames.ndim != 1 or not np.array_equal(reference_frames, new_frames):
        raise WorkflowError("New pose frame indices differ from the manually reviewed reference timeline.")
    if reference_times.shape != new_times.shape or not np.allclose(reference_times, new_times, rtol=0.0, atol=1e-9):
        raise WorkflowError("New pose timestamps differ from the manually reviewed reference timeline.")
    if len(new_frames) != timeline.get("frame_count"):
        raise WorkflowError("New pose frame count differs from the manually reviewed MovementTimeline.")
    if not np.isclose(float(new.get("sample_fps", 0.0)), float(timeline.get("fps", 0.0)), atol=1e-9):
        raise WorkflowError("New pose FPS differs from the manually reviewed MovementTimeline.")

    transferred = deepcopy(timeline)
    transferred["timeline_id"] = f"{timeline['timeline_id']}-{run_id}"
    transferred["source_binding"] = {
        "session_id": session_id,
        "run_id": run_id,
        "pose_file": _portable_pose_path(new_pose_path),
        "pose_file_sha256": _sha256(new_pose_path),
    }
    return transferred


def _portable_pose_path(path: Path) -> str:
    """Prefer a repository-relative path, but preserve valid external paths."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _verify_process_inputs(session_path: Path, session_id: str, output_root: Path) -> None:
    raw = load_yaml_mapping(session_path, label="session")
    for camera in raw.get("cameras", []):
        video = Path(camera.get("video_path", ""))
        resolved = video if video.is_absolute() else session_path.parent / video
        _require_file(resolved.resolve(), f"camera video {camera.get('camera_id')}")
    for depth in raw.get("zed", {}).get("depth_sources", []):
        svo = Path(depth.get("svo_path", ""))
        resolved_svo = svo if svo.is_absolute() else session_path.parent / svo
        _require_file(resolved_svo.resolve(), f"ZED SVO2 {depth.get('camera_id')}")
    _require_file(output_root / session_id / "calibration" / "cameras.json", "production camera calibration")


def _build_summary(
    *,
    profile: dict[str, Any],
    profile_path: Path,
    process_video: bool,
    run_id: str,
    run_root: Path,
    pose_path: Path,
    config_paths: dict[str, Path],
    outputs: dict[str, Path],
) -> dict[str, Any]:
    decisions = _read_json(outputs["accuracy_decisions"])
    readiness = _read_json(outputs["rule_scoring_readiness"])
    diagnostics = _read_json(outputs["wholebody_diagnostics"])
    categorical = _read_json(outputs["categorical_diagnostics"])
    technical_conformance = _read_json(outputs["technical_conformance"])
    technical_accuracy = _read_json(outputs["technical_accuracy_diagnostics"])
    presentation = _read_json(outputs["presentation_diagnostics"])
    automatic_segmentation = _read_json(outputs["automatic_segmentation"])
    timeline = yaml.safe_load(config_paths["movement_timeline"].read_text(encoding="utf-8"))
    summary = decisions.get("summary", {})
    coverage = timeline["coverage"]
    expected_count = len(coverage["observed_movement_ids"]) + len(coverage["missing_movement_ids"])
    selected_ids = list(coverage["observed_movement_ids"])
    segment_ids = [segment.get("movement_id") for segment in timeline["segments"]]
    selected_scope_complete = segment_ids == selected_ids and all(
        segment.get("label_status") == "confirmed" for segment in timeline["segments"]
    )
    selected_scope_label = (
        selected_ids[0] if len(selected_ids) == 1 else f"{selected_ids[0]}-{selected_ids[-1]}"
    )
    selected_scope_id = "current_recording_" + "_".join(item.lower() for item in selected_ids)
    diagnostic_summary = diagnostics.get("summary", {})
    diagnostic_coverage = diagnostics.get("coverage", {})
    categorical_summary = categorical.get("summary", {})
    technical_summary = technical_conformance.get("summary", {})
    technical_accuracy_summary = technical_accuracy.get("summary", {})
    presentation_components = presentation.get("components", {})
    presentation_requested = sum(
        int(component.get("requested_metric_count", 0))
        for component in presentation_components.values()
    )
    presentation_measurable = sum(
        int(component.get("measurable_metric_count", 0))
        for component in presentation_components.values()
    )
    automatic_summary = automatic_segmentation.get("summary", {})
    accuracy_score = decisions.get("accuracy_score")
    partial = decisions.get("observed_scope_provisional_deduction_total")
    accuracy_evaluation_status = decisions.get("accuracy_evaluation_status")
    if partial is not None:
        status = "provisional_observed_scope_analysis_generated"
    else:
        status = "diagnostics_only_no_score"
    bindings = {
        "profile": {"path": str(profile_path), "sha256": _sha256(profile_path)},
        "pose": {"path": str(pose_path.resolve()), "sha256": _sha256(pose_path)},
    }
    bindings.update(
        {key: {"path": str(path.resolve()), "sha256": _sha256(path)} for key, path in config_paths.items()}
    )
    return {
        "schema_version": 1,
        "workflow": "tk3d_source_bound_poomsae_scoring_v1",
        "status": status,
        "mode": "process_video_then_score" if process_video else "score_verified_pose",
        "profile_id": profile["profile_id"],
        "run": {"run_id": run_id, "root": str(run_root.resolve())},
        "coverage": {
            "recording_scope": coverage["recording_scope"],
            "observed_movement_count": len(coverage["observed_movement_ids"]),
            "expected_movement_count": expected_count,
            "selected_scope_id": selected_scope_id,
            "selected_scope_label": selected_scope_label,
            "selected_scope_movement_ids": selected_ids,
            "selected_scope_observed_count": len(selected_ids),
            "selected_scope_expected_count": len(selected_ids),
            "selected_scope_complete": selected_scope_complete,
            "observed_movement_ids": coverage["observed_movement_ids"],
            "missing_movement_ids": coverage["missing_movement_ids"],
        },
        "results": {
            "result_kind": decisions.get("result_kind"),
            "accuracy_evaluation_status": accuracy_evaluation_status,
            "accuracy_score_unavailable_reason": decisions.get("accuracy_score_unavailable_reason"),
            "accuracy_score": accuracy_score,
            "official_score_status": decisions.get("official_score_status"),
            "official_score": decisions.get("official_score"),
            "provisional_deduction_status": decisions.get("provisional_deduction_status"),
            "observed_scope_provisional_deduction_total": partial,
            "confirmed_numeric_minor_count": int(summary.get("confirmed_numeric_minor_count", 0)),
            "not_measurable_count": int(summary.get("not_measurable_count", 0)),
            "boundary_uncertain_count": int(summary.get("boundary_uncertain_count", 0)),
            "automatic_segmentation_expected_count": int(
                automatic_summary.get("expected_movement_count", 0)
            ),
            "automatic_segmentation_selected_count": int(
                automatic_summary.get("selected_movement_count", 0)
            ),
            "automatic_segmentation_movement_count_match": bool(
                automatic_summary.get("movement_count_match", False)
            ),
            "automatic_segmentation_start_boundary_mae_frames": automatic_summary.get(
                "start_boundary_mae_frames"
            ),
            "automatic_segmentation_end_boundary_mae_frames": automatic_summary.get(
                "end_boundary_mae_frames"
            ),
            "automatic_segmentation_phase_anchor_mae_frames": automatic_summary.get(
                "phase_anchor_mae_frames"
            ),
            "applied_categorical_count": int(summary.get("applied_categorical_count", 0)),
            "wholebody_thresholded_metric_count": int(
                diagnostic_coverage.get("thresholded_metric_count", 0)
            ),
            "wholebody_measurable_metric_count": int(
                diagnostic_coverage.get("measurable_metric_count", 0)
            ),
            "diagnostic_review_candidate_count": int(
                diagnostic_summary.get("review_candidate_count", 0)
            ),
            "categorical_mismatch_candidate_count": int(
                categorical_summary.get("mismatch_candidate_count", 0)
            ),
            "categorical_pause_observation_count": int(
                categorical_summary.get("pause_observation_count", 0)
            ),
            "technical_conformance_movement_count": int(
                technical_summary.get("movement_count", 0)
            ),
            "technical_conformance_review_required_count": int(
                technical_summary.get("review_required_count", 0)
            ),
            "technical_conformance_mismatch_candidate_count": int(
                technical_summary.get("mismatch_candidate_count", 0)
            ),
            "technical_conformance_consistent_within_measured_scope_count": int(
                technical_summary.get("consistent_within_measured_scope_count", 0)
            ),
            "technical_conformance_expected_criterion_count": int(
                technical_summary.get("expected_criterion_count", 0)
            ),
            "technical_conformance_measurable_criterion_count": int(
                technical_summary.get("measurable_criterion_count", 0)
            ),
            "technical_conformance_threshold_evaluable_criterion_count": int(
                technical_summary.get("threshold_evaluable_criterion_count", 0)
            ),
            "technical_accuracy_rule_count": int(technical_accuracy_summary.get("rule_count", 0)),
            "technical_accuracy_active_diagnostic_rule_count": int(
                technical_accuracy_summary.get("active_diagnostic_rule_count", 0)
            ),
            "technical_accuracy_temporary_candidate_count": int(
                technical_accuracy_summary.get("temporary_candidate_count", 0)
            ),
            "technical_accuracy_score_effect_count": int(
                technical_accuracy_summary.get("score_effect_count", 0)
            ),
            "presentation_measurable_proxy_count": presentation_measurable,
            "presentation_requested_proxy_count": presentation_requested,
            "rule_scoring_ready": bool(readiness.get("rule_scoring_ready", False)),
            "judge_calibrated_ready": bool(readiness.get("judge_calibrated_ready", False)),
            "official_scoring_ready": bool(readiness.get("official_scoring_ready", False)),
        },
        "interpretation": (
            "This workflow emits diagnostics and provisional observed-scope deduction analysis. "
            "Eligibility for a separate full evaluation does not mean that evaluation ran. "
            "No Accuracy or official score is produced by this workflow."
        ),
        "outputs": {key: str(path.resolve()) for key, path in outputs.items() if key != "summary"},
        "bindings": bindings,
    }


def _output_paths(run_root: Path) -> dict[str, Path]:
    return {
        "automatic_segmentation": run_root / "json" / "automatic_segmentation_report.json",
        "automatic_segmentation_signal": run_root / "csv" / "automatic_segmentation_signal.csv",
        "wholebody_diagnostics": run_root / "json" / "wholebody_diagnostics_report.json",
        "wholebody_metrics": run_root / "csv" / "wholebody_metrics.csv",
        "technical_accuracy_diagnostics": run_root / "json" / "technical_accuracy_diagnostics_report.json",
        "technical_accuracy_coverage": run_root / "csv" / "technical_accuracy_coverage_matrix.csv",
        "technical_accuracy_landmark_coverage": run_root / "csv" / "technical_accuracy_landmark_coverage.csv",
        "movement_evidence": run_root / "json" / "movement_evidence_report.json",
        "movement_evidence_csv": run_root / "csv" / "movement_evidence.csv",
        "categorical_diagnostics": run_root / "json" / "categorical_diagnostics_report.json",
        "technical_conformance": run_root / "json" / "technical_conformance_report.json",
        "presentation_diagnostics": run_root / "json" / "presentation_diagnostics_report.json",
        "rule_scoring_readiness": run_root / "json" / "rule_scoring_readiness.json",
        "accuracy_decisions": run_root / "json" / "source_bound_accuracy_decisions.json",
        "decision_evidence_events": run_root / "json" / "decision_evidence_events.json",
        "annotated_error_video": run_root / "videos" / "poomsae_scoring_annotated.mp4",
        "annotated_error_video_manifest": run_root / "videos" / "poomsae_scoring_annotated_manifest.json",
        "browser_review_video_manifest": run_root / "videos" / "browser_review_video_manifest.json",
        "review_html": run_root / "review" / "poomsae_scoring_review.html",
        "review_manifest": run_root / "review" / "poomsae_scoring_review_manifest.json",
        "run_history_json": run_root / "json" / "run_history_report.json",
        "run_history_html": run_root / "review" / "run_history.html",
        "summary": run_root / "json" / "poomsae_scoring_summary.json",
    }


def _resolve_videos(raw_videos: list[dict[str, str]]) -> list[dict[str, str | Path]]:
    videos: list[dict[str, str | Path]] = []
    for video in raw_videos:
        path = _resolve_path(video["path"])
        _require_file(path, f"review video {video['label']}")
        videos.append({"camera_id": video["camera_id"], "label": video["label"], "path": path})
    return videos


def _run_stage(
    label: str,
    script: str,
    *args: str | int | Path,
    profiler: PerformanceCollector | None = None,
) -> None:
    command = [sys.executable, str((ROOT / script).resolve()), *(str(arg) for arg in args)]
    print(f"\n[{label}]", flush=True)
    started = time.perf_counter() if profiler is not None else 0.0
    try:
        subprocess.run(command, cwd=ROOT, check=True)
    except subprocess.CalledProcessError as exc:
        raise WorkflowError(f"{label} failed with exit code {exc.returncode}.") from exc
    finally:
        if profiler is not None:
            elapsed = time.perf_counter() - started
            parent = (
                "poomsae_presentation_export"
                if script in _POOMSAE_PRESENTATION_SCRIPTS
                else "poomsae_analysis_decisions"
            )
            profiler.record(parent, elapsed)
            profiler.record(
                f"poomsae_stage:{Path(script).stem}",
                elapsed,
                parent=parent,
                tags={"label": label, "script": script},
            )


def _resolve_profile(value: str) -> Path:
    candidate = Path(value)
    if candidate.suffix.lower() not in {".yaml", ".yml"} and candidate.parent == Path("."):
        candidate = Path("config") / "scoring" / "profiles" / f"{value}.yaml"
    return _resolve_path(candidate)


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _default_run_id(process_video: bool) -> str:
    mode = "full" if process_video else "score"
    return f"poomsae1-{mode}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"


def _read_json(path: Path) -> dict[str, Any]:
    return load_json_object(path)


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    write_json_exclusive(path, payload)


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise WorkflowError(f"Required file is missing ({label}): {path}")


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value) is not None


def _display(value: Any) -> str:
    return "null" if value is None else str(value)
