from __future__ import annotations

import unicodedata
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.poomsae_scoring.contracts import validate_movement_timeline, validate_poomsae_spec


def overlay_state_for_frame(
    poomsae_spec: dict[str, Any],
    movement_timeline: dict[str, Any],
    frame_index: int,
) -> dict[str, Any] | None:
    """Return the source-bound movement/phase label for one sample frame."""
    spec = validate_poomsae_spec(poomsae_spec)
    timeline = validate_movement_timeline(movement_timeline, spec)
    if not isinstance(frame_index, int) or isinstance(frame_index, bool):
        raise ValueError("frame_index must be an integer")
    if not 0 <= frame_index < timeline["frame_count"]:
        raise ValueError("frame_index is outside MovementTimeline")
    return _overlay_state_from_validated(spec, timeline, frame_index)


def _overlay_state_from_validated(
    spec: dict[str, Any],
    timeline: dict[str, Any],
    frame_index: int,
) -> dict[str, Any] | None:
    movements = {movement["movement_id"]: movement for movement in spec["movements"]}
    for segment in timeline["segments"]:
        if not segment["start_frame"] <= frame_index <= segment["end_frame"]:
            continue
        movement = movements[segment["movement_id"]]
        phase_id = _phase_for_frame(movement["phases"], segment["anchors"], frame_index)
        return {
            "movement_id": movement["movement_id"],
            "sequence_index": movement["sequence_index"],
            "expected_movement_count": len(spec["movements"]),
            "display_name": movement["display_name"],
            "phase_id": phase_id,
            "label_status": segment["label_status"],
            "confidence": segment["confidence"],
        }
    return None


def render_movement_overlay(
    video_path: str | Path,
    output_path: str | Path,
    poomsae_spec: dict[str, Any],
    movement_timeline: dict[str, Any],
) -> dict[str, Any]:
    """Render an immutable review video without changing the source timeline."""
    spec = validate_poomsae_spec(poomsae_spec)
    timeline = validate_movement_timeline(movement_timeline, spec)
    source = Path(video_path).resolve()
    target = Path(output_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"video not found: {source}")
    if target.exists():
        raise FileExistsError(f"refusing to overwrite existing overlay: {target}")

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {source}")
    source_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if source_frame_count != timeline["frame_count"]:
        capture.release()
        raise ValueError("video frame count does not match MovementTimeline")
    if not np.isclose(source_fps, timeline["fps"], atol=1e-3):
        capture.release()
        raise ValueError("video FPS does not match MovementTimeline")

    target.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), source_fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"cannot create overlay video: {target}")

    labeled_frame_count = 0
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            state = _overlay_state_from_validated(spec, timeline, frame_index)
            if state is not None:
                labeled_frame_count += 1
            _draw_overlay(frame, state, frame_index, source_fps, len(timeline["segments"]))
            writer.write(frame)
            frame_index += 1
    finally:
        capture.release()
        writer.release()

    if frame_index != source_frame_count:
        raise RuntimeError("overlay render ended before the declared source frame count")
    return {
        "status": "draft_label_review",
        "source_video": str(source),
        "output_video": str(target),
        "frame_count": frame_index,
        "fps": source_fps,
        "labeled_frame_count": labeled_frame_count,
        "unlabeled_frame_count": frame_index - labeled_frame_count,
        "labeled_movement_count": len(timeline["segments"]),
        "expected_movement_count": len(spec["movements"]),
    }


def _phase_for_frame(phases: list[str], anchors: dict[str, int], frame_index: int) -> str:
    ordered = [(phase_id, anchors[phase_id]) for phase_id in phases if phase_id in anchors]
    if not ordered:
        return "unlabeled_phase"
    phase_id = ordered[0][0]
    for candidate_phase, anchor_frame in ordered:
        if frame_index < anchor_frame:
            break
        phase_id = candidate_phase
    return phase_id


def _draw_overlay(
    frame: np.ndarray,
    state: dict[str, Any] | None,
    frame_index: int,
    fps: float,
    labeled_movement_count: int,
) -> None:
    width = frame.shape[1]
    panel_height = 104
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, panel_height), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0.0, frame)
    if state is None:
        title = "Taegeuk 1 Jang - etiketlenmemis bolum"
        detail = "Hareket/faz kesintisi uygulanmaz"
        color = (170, 170, 170)
    else:
        title = (
            f"{state['movement_id']} / {state['expected_movement_count']}  "
            f"{_ascii(state['display_name'])}"
        )
        detail = (
            f"Faz: {state['phase_id']}  |  Etiket: {state['label_status']}  "
            f"|  Guven: {state['confidence']:.2f}"
        )
        color = (0, 210, 255) if state["label_status"] == "provisional" else (70, 220, 70)
    cv2.putText(frame, title, (24, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.82, color, 2, cv2.LINE_AA)
    cv2.putText(frame, detail, (24, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 245, 245), 1, cv2.LINE_AA)
    timing = f"frame {frame_index}  |  {frame_index / fps:.3f} s  |  {labeled_movement_count} hareket etiketli"
    cv2.putText(frame, timing, (24, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 210, 210), 1, cv2.LINE_AA)


def _ascii(value: str) -> str:
    replacements = str.maketrans({"ı": "i", "İ": "I", "ş": "s", "Ş": "S", "ğ": "g", "Ğ": "G"})
    normalized = unicodedata.normalize("NFKD", value.translate(replacements))
    return normalized.encode("ascii", "ignore").decode("ascii")
