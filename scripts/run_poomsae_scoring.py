from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.poomsae_scoring import load_movement_timeline, load_poomsae_spec  # noqa: E402
from src.video_io import load_session  # noqa: E402


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
    "accuracy_profile",
}


class WorkflowError(RuntimeError):
    """Raised when the one-command scoring workflow must fail closed."""


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"duplicate key: {key}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete source-bound Poomsae scoring workflow with one command. "
            "By default it scores the profile's verified 3D pose; --process-video first "
            "re-runs ViTPose/RGBD on the exact bound session."
        )
    )
    parser.add_argument(
        "--profile",
        default="poomsae1_trimmed",
        help="Profile id under config/scoring/profiles or an explicit YAML path.",
    )
    parser.add_argument(
        "--process-video",
        action="store_true",
        help="Run full stride-1 ViTPose/RGBD processing before source-bound scoring.",
    )
    parser.add_argument("--run-id", help="Optional unique output run id.")
    args = parser.parse_args()

    try:
        summary_path = run_workflow(
            profile_value=args.profile,
            process_video=args.process_video,
            requested_run_id=args.run_id,
        )
    except (WorkflowError, OSError, ValueError, yaml.YAMLError, json.JSONDecodeError) as exc:
        raise SystemExit(f"TK3D scoring workflow failed: {exc}") from exc

    summary = _read_json(summary_path)
    results = summary["results"]
    coverage = summary["coverage"]
    print("\n=== TK3D PUANLAMA SONUCU ===")
    print(f"Durum: {summary['status']}")
    print(f"Hareket kapsamı: {coverage['observed_movement_count']}/{coverage['expected_movement_count']}")
    print(f"Tam Accuracy skoru: {_display(results['accuracy_score'])}")
    print(
        "Gözlenen kapsam provisional kesintisi: "
        f"{_display(results['observed_scope_provisional_deduction_total'])}"
    )
    print(f"Küçük hata sayısı: {results['confirmed_numeric_minor_count']}")
    print(f"Ölçülemeyen karar: {results['not_measurable_count']}")
    print(f"Sınır-belirsiz karar: {results['boundary_uncertain_count']}")
    print(f"Rule scoring ready: {str(results['rule_scoring_ready']).lower()}")
    print(f"Çıktı klasörü: {summary['run']['root']}")
    print(f"Ana özet: {summary_path}")
    print(f"İnceleme ekranı: {summary['outputs']['review_html']}")


def run_workflow(
    *,
    profile_value: str,
    process_video: bool,
    requested_run_id: str | None,
) -> Path:
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
        _run_stage(
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
        _run_stage(
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
        )
        pose_path = run_root / "json" / "vitpose_session_3d.json"
        _require_file(pose_path, "processed 3D pose")
        _run_stage(
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

    config_paths = _snapshot_configuration(
        run_root=run_root,
        profile_path=profile_path,
        paths=paths,
        pose_path=pose_path,
        session_id=session.session_id,
        run_id=run_id,
        process_video=process_video,
    )
    outputs = _output_paths(run_root)
    videos = _resolve_videos(profile["videos"])
    keypoints_2d_csv = pose_path.parent.parent / "csv" / "vitpose_keypoints_2d_flat.csv"
    _require_file(keypoints_2d_csv, "camera-observed WholeBody-133 2D evidence")

    _run_stage(
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
    _run_stage(
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
    _run_stage(
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
    _run_stage(
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
        "--output-json",
        outputs["accuracy_decisions"],
    )
    _run_stage(
        "Kararları görsel kanıt olaylarına dönüştürme",
        "scripts/build_poomsae_evidence_events.py",
        "--accuracy-decisions",
        outputs["accuracy_decisions"],
        "--poomsae-spec",
        config_paths["poomsae_spec"],
        "--timeline",
        config_paths["movement_timeline"],
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
    _run_stage(
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
        "--accuracy-decisions",
        outputs["accuracy_decisions"],
        "--decision-evidence-events",
        outputs["decision_evidence_events"],
        "--video-a",
        videos[0]["path"],
        "--video-a-label",
        videos[0]["label"],
        "--video-b",
        videos[1]["path"],
        "--video-b-label",
        videos[1]["label"],
    ]
    for video in videos[2:]:
        review_args.extend(("--video-extra", f"{video['label']}={video['path']}"))
    review_args.extend(("--video-extra", f"Isaretli hata kaniti={outputs['annotated_error_video']}"))
    review_args.extend(("--output", outputs["review_html"], "--manifest", outputs["review_manifest"]))
    _run_stage("Senkron kamera inceleme ekranı", "scripts/create_poomsae_review_report.py", *review_args)

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
    return outputs["summary"]


def _load_profile(path: Path) -> dict[str, Any]:
    _require_file(path, "workflow profile")
    payload = yaml.load(path.read_text(encoding="utf-8-sig"), Loader=_UniqueKeyLoader)
    if not isinstance(payload, dict):
        raise WorkflowError("Workflow profile root must be a mapping.")
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
        "accuracy_profile": config_root / "accuracy_profile.yaml",
    }
    copies = {
        "workflow_profile": profile_path,
        "rule_pack": paths["rule_pack"],
        "poomsae_spec": paths["poomsae_spec"],
        "diagnostic_profile": paths["diagnostic_profile"],
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
    timeline = yaml.load(timeline_path.read_text(encoding="utf-8-sig"), Loader=_UniqueKeyLoader)
    if not isinstance(timeline, dict):
        raise WorkflowError("Reference MovementTimeline must be a mapping.")
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
        "pose_file": new_pose_path.resolve().relative_to(ROOT).as_posix(),
        "pose_file_sha256": _sha256(new_pose_path),
    }
    return transferred


def _verify_process_inputs(session_path: Path, session_id: str, output_root: Path) -> None:
    raw = yaml.load(session_path.read_text(encoding="utf-8-sig"), Loader=_UniqueKeyLoader)
    if not isinstance(raw, dict):
        raise WorkflowError("Session YAML root must be a mapping.")
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
    timeline = yaml.safe_load(config_paths["movement_timeline"].read_text(encoding="utf-8"))
    summary = decisions.get("summary", {})
    coverage = timeline["coverage"]
    expected_count = len(coverage["observed_movement_ids"]) + len(coverage["missing_movement_ids"])
    accuracy_score = decisions.get("accuracy_score")
    partial = decisions.get("observed_scope_provisional_deduction_total")
    if accuracy_score is not None:
        status = "accuracy_score_generated"
    elif partial is not None:
        status = "partial_sequence_decisions_generated"
    else:
        status = "no_score_insufficient_evidence"
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
            "observed_movement_ids": coverage["observed_movement_ids"],
            "missing_movement_ids": coverage["missing_movement_ids"],
        },
        "results": {
            "accuracy_score": accuracy_score,
            "observed_scope_provisional_deduction_total": partial,
            "confirmed_numeric_minor_count": int(summary.get("confirmed_numeric_minor_count", 0)),
            "not_measurable_count": int(summary.get("not_measurable_count", 0)),
            "boundary_uncertain_count": int(summary.get("boundary_uncertain_count", 0)),
            "applied_categorical_count": int(summary.get("applied_categorical_count", 0)),
            "rule_scoring_ready": bool(readiness.get("rule_scoring_ready", False)),
            "judge_calibrated_ready": bool(readiness.get("judge_calibrated_ready", False)),
            "official_scoring_ready": bool(readiness.get("official_scoring_ready", False)),
        },
        "interpretation": (
            "A null accuracy_score is intentional for a partial recording or insufficient evidence. "
            "The observed-scope deduction is provisional and is not a full 4-point Accuracy score."
        ),
        "outputs": {key: str(path.resolve()) for key, path in outputs.items() if key != "summary"},
        "bindings": bindings,
    }


def _output_paths(run_root: Path) -> dict[str, Path]:
    return {
        "wholebody_diagnostics": run_root / "json" / "wholebody_diagnostics_report.json",
        "wholebody_metrics": run_root / "csv" / "wholebody_metrics.csv",
        "movement_evidence": run_root / "json" / "movement_evidence_report.json",
        "movement_evidence_csv": run_root / "csv" / "movement_evidence.csv",
        "rule_scoring_readiness": run_root / "json" / "rule_scoring_readiness.json",
        "accuracy_decisions": run_root / "json" / "source_bound_accuracy_decisions.json",
        "decision_evidence_events": run_root / "json" / "decision_evidence_events.json",
        "annotated_error_video": run_root / "videos" / "poomsae_scoring_annotated.mp4",
        "annotated_error_video_manifest": run_root / "videos" / "poomsae_scoring_annotated_manifest.json",
        "review_html": run_root / "review" / "poomsae_scoring_review.html",
        "review_manifest": run_root / "review" / "poomsae_scoring_review_manifest.json",
        "summary": run_root / "json" / "poomsae_scoring_summary.json",
    }


def _resolve_videos(raw_videos: list[dict[str, str]]) -> list[dict[str, str | Path]]:
    videos: list[dict[str, str | Path]] = []
    for video in raw_videos:
        path = _resolve_path(video["path"])
        _require_file(path, f"review video {video['label']}")
        videos.append({"camera_id": video["camera_id"], "label": video["label"], "path": path})
    return videos


def _run_stage(label: str, script: str, *args: str | int | Path) -> None:
    command = [sys.executable, str((ROOT / script).resolve()), *(str(arg) for arg in args)]
    print(f"\n[{label}]", flush=True)
    try:
        subprocess.run(command, cwd=ROOT, check=True)
    except subprocess.CalledProcessError as exc:
        raise WorkflowError(f"{label} failed with exit code {exc.returncode}.") from exc


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
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise WorkflowError(f"JSON root must be an object: {path}")
    return payload


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise WorkflowError(f"Required file is missing ({label}): {path}")


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value) is not None


def _display(value: Any) -> str:
    return "null" if value is None else str(value)


if __name__ == "__main__":
    main()
