from __future__ import annotations

import csv
import json
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.artifact_io import sha256_file
from src.poomsae_scoring.video_cards import draw_evidence_panel, draw_pause_banner

_sha256 = sha256_file


_COLORS = {
    "red": (60, 70, 245),
    "amber": (40, 190, 255),
    "gray": (165, 165, 165),
    "green": (80, 220, 90),
    "blue": (245, 170, 50),
}
_STATUS_PRIORITY = {
    "confirmed_source_bound_minor": 0,
    "boundary_uncertain": 1,
    "not_measurable": 2,
    "diagnostic_review_candidate": 3,
    "within_source_range": 4,
    "not_applicable": 5,
}


def render_decision_evidence_video(
    *,
    camera_videos: dict[str, str | Path],
    keypoints_2d_csv: str | Path,
    evidence_events: dict[str, Any],
    output_path: str | Path,
    deduction_freeze_seconds: float = 3.0,
) -> dict[str, Any]:
    """Render synchronized camera panes with source-bound decision evidence."""
    if len(camera_videos) < 2:
        raise ValueError("decision evidence video requires at least two cameras")
    if evidence_events.get("status") != "decision_evidence_events":
        raise ValueError("invalid decision evidence report")
    events = evidence_events.get("events")
    if not isinstance(events, list):
        raise ValueError("decision evidence events must be a list")
    frame_count = int(evidence_events["frame_count"])
    expected_fps = float(evidence_events["fps"])
    if not np.isfinite(deduction_freeze_seconds) or deduction_freeze_seconds < 0.0:
        raise ValueError("deduction_freeze_seconds must be finite and nonnegative")
    target = Path(output_path).resolve()
    if target.exists():
        raise FileExistsError(f"refusing to overwrite decision evidence video: {target}")

    captures: list[tuple[str, Path, cv2.VideoCapture]] = []
    source_details: dict[str, dict[str, Any]] = {}
    try:
        for camera_id, raw_path in camera_videos.items():
            path = Path(raw_path).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"camera video not found ({camera_id}): {path}")
            capture = cv2.VideoCapture(str(path))
            if not capture.isOpened():
                raise RuntimeError(f"cannot open camera video ({camera_id}): {path}")
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            if count != frame_count:
                capture.release()
                raise ValueError(f"{camera_id} frame count {count} does not match evidence {frame_count}")
            if not np.isclose(fps, expected_fps, atol=1e-3):
                capture.release()
                raise ValueError(f"{camera_id} FPS {fps} does not match evidence {expected_fps}")
            captures.append((camera_id, path, capture))
            source_details[camera_id] = {
                "path": str(path),
                "sha256": _sha256(path),
                "frame_count": count,
                "fps": fps,
            }

        events_by_frame, active_frames, joint_indices = _index_events(events, frame_count)
        freeze_anchor_frames = _deduction_freeze_anchor_frames(events)
        freeze_frames_per_pause = int(round(expected_fps * deduction_freeze_seconds))
        observations = _load_observed_2d(
            Path(keypoints_2d_csv).resolve(),
            camera_ids=set(camera_videos),
            active_frames=active_frames,
            joint_indices=joint_indices,
        )
        pane_width, pane_height, panel_height = 960, 540, 540
        output_size = (pane_width * len(captures), pane_height + panel_height)
        target.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(target),
            cv2.VideoWriter_fourcc(*"mp4v"),
            expected_fps,
            output_size,
        )
        if not writer.isOpened():
            raise RuntimeError(f"cannot create decision evidence video: {target}")
        rendered_frames = 0
        source_frames_rendered = 0
        evidence_frames = 0
        try:
            for frame_index in range(frame_count):
                active = events_by_frame.get(frame_index, [])
                if active:
                    evidence_frames += 1
                panes = []
                for camera_id, _, capture in captures:
                    ok, frame = capture.read()
                    if not ok:
                        raise RuntimeError(f"camera video ended early: {camera_id} at frame {frame_index}")
                    original_height, original_width = frame.shape[:2]
                    pane = cv2.resize(frame, (pane_width, pane_height), interpolation=cv2.INTER_AREA)
                    scale = (pane_width / original_width, pane_height / original_height)
                    _draw_camera_pane(
                        pane,
                        camera_id=camera_id,
                        frame_index=frame_index,
                        active_events=active,
                        observations=observations,
                        scale=scale,
                    )
                    panes.append(pane)
                canvas = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
                canvas[:pane_height] = np.hstack(panes)
                draw_evidence_panel(canvas, active, frame_index, expected_fps, pane_height)
                writer.write(canvas)
                rendered_frames += 1
                source_frames_rendered += 1
                if frame_index in freeze_anchor_frames and freeze_frames_per_pause > 0:
                    for freeze_index in range(freeze_frames_per_pause):
                        frozen = canvas.copy()
                        seconds_left = (freeze_frames_per_pause - freeze_index) / expected_fps
                        draw_pause_banner(frozen, seconds_left)
                        writer.write(frozen)
                        rendered_frames += 1
        finally:
            writer.release()
    finally:
        for _, _, capture in captures:
            capture.release()

    if source_frames_rendered != frame_count:
        raise RuntimeError("decision evidence render did not preserve the complete timeline")
    inserted_freeze_frames = len(freeze_anchor_frames) * freeze_frames_per_pause
    if rendered_frames != frame_count + inserted_freeze_frames:
        raise RuntimeError("decision evidence freeze-frame accounting is inconsistent")
    return {
        "schema_version": 1,
        "status": "decision_evidence_video_rendered",
        "source_videos": source_details,
        "keypoints_2d_csv": {
            "path": str(Path(keypoints_2d_csv).resolve()),
            "sha256": _sha256(Path(keypoints_2d_csv).resolve()),
        },
        "output_video": {"path": str(target), "sha256": _sha256(target)},
        "source_frame_count": source_frames_rendered,
        "frame_count": rendered_frames,
        "fps": expected_fps,
        "source_duration_sec": source_frames_rendered / expected_fps,
        "duration_sec": rendered_frames / expected_fps,
        "freeze_pause_count": len(freeze_anchor_frames),
        "freeze_seconds_per_pause": deduction_freeze_seconds,
        "inserted_freeze_frame_count": inserted_freeze_frames,
        "freeze_anchor_source_frames": sorted(freeze_anchor_frames),
        "source_timeline_preserved": True,
        "output_timeline_extended_for_readability": bool(inserted_freeze_frames),
        "evidence_event_count": len(events),
        "evidence_frame_count": evidence_frames,
        "measurement_space": evidence_events["measurement_space"],
        "camera_overlay_space": evidence_events["camera_overlay_space"],
        "interpretation": evidence_events["camera_overlay_warning"],
    }


def _deduction_freeze_anchor_frames(events: list[dict[str, Any]]) -> set[int]:
    anchors: set[int] = set()
    for event in events:
        points = event.get("deduction_points")
        if event.get("decision_status") != "confirmed_source_bound_minor" or points is None:
            continue
        if float(points) <= 0.0:
            continue
        anchors.add(int(event["evidence_window"]["anchor_frame"]))
    return anchors


def _index_events(
    events: list[dict[str, Any]],
    frame_count: int,
) -> tuple[dict[int, list[dict[str, Any]]], set[int], set[int]]:
    indexed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    joints: set[int] = set()
    for event in events:
        window = event.get("evidence_window")
        if not window:
            # An alignment anomaly about a movement that never got a segment has no
            # frame window. Drawing it at a guessed frame would assert a location the
            # data does not support, so it is left out of the video and reported in
            # the HTML review instead.
            continue
        start, end = int(window["start_frame"]), int(window["end_frame"])
        if not 0 <= start <= end < frame_count:
            raise ValueError(f"evidence event is outside the video timeline: {event.get('event_id')}")
        geometry = event.get("visual_geometry") or {}
        joints.update(int(value) for value in geometry.get("joint_indices", []))
        for frame_index in range(start, end + 1):
            indexed[frame_index].append(event)
    for frame_events in indexed.values():
        frame_events.sort(key=lambda item: (_STATUS_PRIORITY.get(item["decision_status"], 99), item["event_id"]))
    return dict(indexed), set(indexed), joints


def _load_observed_2d(
    path: Path,
    *,
    camera_ids: set[str],
    active_frames: set[int],
    joint_indices: set[int],
) -> dict[tuple[str, int, int], tuple[float, float, float]]:
    if not path.is_file():
        raise FileNotFoundError(f"2D keypoint evidence CSV not found: {path}")
    observations: dict[tuple[str, int, int], tuple[float, float, float]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"frame_idx", "camera_id", "joint_idx", "x", "y", "score", "valid"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("2D keypoint CSV does not match the TK3D flat contract")
        for row in reader:
            frame_index = int(row["frame_idx"])
            joint_index = int(row["joint_idx"])
            camera_id = row["camera_id"]
            if frame_index not in active_frames or joint_index not in joint_indices or camera_id not in camera_ids:
                continue
            if row["valid"].strip().lower() not in {"true", "1"}:
                continue
            x, y, score = float(row["x"]), float(row["y"]), float(row["score"])
            if np.isfinite([x, y, score]).all():
                observations[(camera_id, frame_index, joint_index)] = (x, y, score)
    return observations


def _draw_camera_pane(
    pane: np.ndarray,
    *,
    camera_id: str,
    frame_index: int,
    active_events: list[dict[str, Any]],
    observations: dict[tuple[str, int, int], tuple[float, float, float]],
    scale: tuple[float, float],
) -> None:
    cv2.rectangle(pane, (0, 0), (pane.shape[1], 38), (61, 55, 33), -1)
    cv2.putText(pane, camera_id, (16, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.67, (235, 245, 250), 2, cv2.LINE_AA)
    for event_number, event in enumerate(active_events[:3], start=1):
        color = _COLORS.get(event["display_color"], _COLORS["gray"])
        _draw_geometry(
            pane,
            event,
            camera_id,
            frame_index,
            observations,
            scale,
            color,
            event_number,
        )


def _draw_geometry(
    pane: np.ndarray,
    event: dict[str, Any],
    camera_id: str,
    frame_index: int,
    observations: dict[tuple[str, int, int], tuple[float, float, float]],
    scale: tuple[float, float],
    color: tuple[int, int, int],
    event_number: int,
) -> None:
    visual = event.get("visual_geometry") or {}
    indices = [int(value) for value in visual.get("joint_indices", [])]
    points = {
        index: _screen_point(observations.get((camera_id, frame_index, index)), scale)
        for index in indices
    }
    kind = visual.get("kind")
    if kind == "joint_angle" and len(indices) == 3:
        a, b, c = (points[index] for index in indices)
        _polyline(pane, [a, b, c], color)
        if b is not None:
            cv2.circle(pane, b, 13, color, 3, cv2.LINE_AA)
            _label(pane, b, f"[{event_number}] DIRSEK", color, offset=(14, -16))
    elif kind == "foot_direction_angle" and len(indices) == 5:
        front_ankle, back_ankle, heel, big_toe, small_toe = (points[index] for index in indices)
        toe_center = _mean_points(big_toe, small_toe)
        _draw_foot_direction_guide(
            pane,
            heel=heel,
            toe_center=toe_center,
            color=color,
            event_number=event_number,
        )
        for point in (front_ankle, back_ankle, heel, big_toe, small_toe):
            _point(pane, point, color)
    elif kind == "fist_to_thigh_distance" and len(indices) == 6:
        hand = _mean_points(*(points[index] for index in indices[:4]))
        hip, knee = points[indices[4]], points[indices[5]]
        _line(pane, hip, knee, color, 4)
        closest = _closest_point_2d(hand, hip, knee)
        _line(pane, hand, closest, color, 3)
        _point(pane, hand, color, radius=10)
        _point(pane, closest, color)
        _label(pane, hand, f"[{event_number}] YUMRUK", color, offset=(12, -14))
        _label(pane, closest, "UYLUK", (235, 235, 235), offset=(12, 18))
    elif kind == "hand_shape" and len(indices) == 10:
        wrist = points[indices[0]]
        for mcp_index, tip_index in ((2, 3), (4, 5), (6, 7), (8, 9)):
            _line(pane, points[indices[mcp_index]], points[indices[tip_index]], color, 2)
        for point in points.values():
            _point(pane, point, color)
        _label(pane, wrist, f"[{event_number}] EL/YUMRUK", color, offset=(12, -14))
    elif kind == "head_torso_direction" and len(indices) == 14:
        left_eye = _mean_points(*(points[index] for index in indices[:6]))
        right_eye = _mean_points(*(points[index] for index in indices[6:12]))
        left_shoulder, right_shoulder = points[indices[12]], points[indices[13]]
        _line(pane, left_eye, right_eye, color, 4)
        _line(pane, left_shoulder, right_shoulder, (235, 235, 235), 3)
        _point(pane, left_eye, color)
        _point(pane, right_eye, color)
        label_anchor = None if left_eye is None else (left_eye[0], max(left_eye[1], 50))
        _label(pane, label_anchor, f"[{event_number}] BAS/YUZ YONU", color, offset=(12, 20))
    else:
        for point in points.values():
            _point(pane, point, color)


def _screen_point(
    observation: tuple[float, float, float] | None,
    scale: tuple[float, float],
) -> tuple[int, int] | None:
    if observation is None:
        return None
    return int(round(observation[0] * scale[0])), int(round(observation[1] * scale[1]))


def _draw_foot_direction_guide(
    pane: np.ndarray,
    *,
    heel: tuple[int, int] | None,
    toe_center: tuple[int, int] | None,
    color: tuple[int, int, int],
    event_number: int,
) -> None:
    _line(pane, heel, toe_center, color, 5)
    if heel is None or toe_center is None:
        return
    _label(pane, heel, f"[{event_number}] 2B AYAK IZI", color, offset=(-8, -20))


def _label(
    image: np.ndarray,
    point: tuple[int, int] | None,
    text: str,
    color: tuple[int, int, int],
    *,
    offset: tuple[int, int] = (0, 0),
) -> None:
    if point is None:
        return
    origin = (max(4, point[0] + offset[0]), max(18, point[1] + offset[1]))
    cv2.putText(image, _ascii(text), origin, cv2.FONT_HERSHEY_SIMPLEX, 0.42, (4, 8, 12), 3, cv2.LINE_AA)
    cv2.putText(image, _ascii(text), origin, cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)


def _line(
    image: np.ndarray,
    first: tuple[int, int] | None,
    second: tuple[int, int] | None,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    if first is not None and second is not None:
        cv2.line(image, first, second, color, thickness, cv2.LINE_AA)


def _polyline(image: np.ndarray, points: list[tuple[int, int] | None], color: tuple[int, int, int]) -> None:
    for first, second in zip(points, points[1:]):
        _line(image, first, second, color, 4)
    for point in points:
        _point(image, point, color)


def _point(
    image: np.ndarray,
    point: tuple[int, int] | None,
    color: tuple[int, int, int],
    radius: int = 7,
) -> None:
    if point is not None:
        cv2.circle(image, point, radius, color, -1, cv2.LINE_AA)
        cv2.circle(image, point, radius + 3, (245, 245, 245), 1, cv2.LINE_AA)


def _mean_points(*points: tuple[int, int] | None) -> tuple[int, int] | None:
    valid = [point for point in points if point is not None]
    if not valid:
        return None
    return int(round(sum(point[0] for point in valid) / len(valid))), int(round(sum(point[1] for point in valid) / len(valid)))


def _closest_point_2d(
    point: tuple[int, int] | None,
    start: tuple[int, int] | None,
    end: tuple[int, int] | None,
) -> tuple[int, int] | None:
    if point is None or start is None or end is None:
        return None
    p, a, b = (np.asarray(value, dtype=float) for value in (point, start, end))
    direction = b - a
    denominator = float(np.dot(direction, direction))
    if denominator <= 1e-9:
        return start
    ratio = float(np.clip(np.dot(p - a, direction) / denominator, 0.0, 1.0))
    result = a + ratio * direction
    return int(round(result[0])), int(round(result[1]))


def _limits(operator: Any, limits: Any, unit: Any) -> str:
    if not isinstance(limits, list) or not limits:
        return "tanimli degil"
    suffix = f" {unit}" if unit else ""
    if operator == "max":
        return f"<= {_number(limits[0])}{suffix}"
    if len(limits) == 2:
        return f"{_number(limits[0])}-{_number(limits[1])}{suffix}"
    return str(limits)


def _number(value: Any) -> str:
    return "null" if value is None else f"{float(value):.2f}"


def _ascii(value: Any) -> str:
    replacements = str.maketrans({"ı": "i", "İ": "I", "ş": "s", "Ş": "S", "ğ": "g", "Ğ": "G", "ö": "o", "Ö": "O", "ü": "u", "Ü": "U", "ç": "c", "Ç": "C"})
    normalized = unicodedata.normalize("NFKD", str(value).translate(replacements))
    return normalized.encode("ascii", "ignore").decode("ascii")


def write_render_manifest(path: str | Path, payload: dict[str, Any]) -> Path:
    target = Path(path).resolve()
    if target.exists():
        raise FileExistsError(f"refusing to overwrite evidence video manifest: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return target
