from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

from src.poomsae_scoring import (
    build_decision_evidence_events,
    load_movement_timeline,
    load_poomsae_spec,
)
from src.poomsae_scoring.error_video import render_decision_evidence_video


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "config" / "scoring" / "poomsae" / "taegeuk_1_jang_v0_draft.yaml"
TIMELINE_PATH = ROOT / "config" / "scoring" / "timelines" / "poomsae1_zed2i_rgbd_rerun_20260802_draft.yaml"


def test_decision_events_preserve_3d_measurement_and_camera_visual_trace_contract() -> None:
    spec = load_poomsae_spec(SPEC_PATH)
    timeline = load_movement_timeline(TIMELINE_PATH, spec)
    decisions = {
        "status": "source_bound_accuracy_decisions",
        "timeline_id": timeline["timeline_id"],
        "numeric_decisions": [
            {
                "movement_id": "M01",
                "rule_id": "HIST-FOOT-30",
                "metric_id": "back_foot_yaw_to_stance_direction_deg",
                "deduction_kind": "minor",
                "deduction_points": 0.1,
                "application_status": "applied",
                "source": {"source_id": "historical", "pages": [5]},
                "description": "Back foot maximum 30 degrees.",
                "sample_count": 11,
                "measurement_evidence": {
                    "scope": "fixation_window",
                    "start_frame": 213,
                    "anchor_frame": 218,
                    "end_frame": 223,
                },
                "decision_status": "confirmed_source_bound_minor",
                "measured_value": 39.35,
                "effective_uncertainty_95": 5.0,
                "measurement_interval_95": [34.35, 44.35],
                "rule_operator": "max",
                "rule_limits": [30.0],
                "boundary_guard": 1.0,
                "reason": "outside",
            }
        ],
        "categorical_decisions": [],
    }

    report = build_decision_evidence_events(decisions, spec, timeline)

    event = report["events"][0]
    assert report["measurement_space"] == "tk3d_world_3d"
    assert report["camera_overlay_space"] == "observed_vitpose_2d_visual_trace_only"
    assert event["decision_status"] == "confirmed_source_bound_minor"
    assert event["measurement"]["value"] == 39.35
    assert event["visual_geometry"]["kind"] == "foot_direction_angle"
    assert len(event["visual_geometry"]["joint_indices"]) == 5
    assert event["user_explanation"] == {
        "title": "ARKA AYAK ACISI",
        "expected": "Olmasi gereken: en fazla 30.00 derece.",
        "measured": "39.35 derece",
        "interval": "%95 olasi aralik: 34.35 - 44.35 derece",
        "comparison": "Fark: ust sinirdan 9.35 derece fazla; %95'e gore en az 4.35 fazla.",
        "result": "Sonuc: belirsizlik araligi da sinir disinda; kucuk hata adayi (-0.1).",
        "correction": "Arka ayagi durus yonune yaklastir; ayak acisini izin verilen sinira indir.",
        "source_note": "historical, sayfa 5.",
    }


def test_annotated_video_preserves_timeline_and_draws_observed_joint_trace(tmp_path: Path) -> None:
    fps, frame_count = 2.0, 3
    cameras = {
        "camera_a": tmp_path / "camera_a.avi",
        "camera_b": tmp_path / "camera_b.avi",
    }
    for index, path in enumerate(cameras.values()):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (320, 180))
        assert writer.isOpened()
        for frame_index in range(frame_count):
            frame = np.full((180, 320, 3), 25 + 20 * index + frame_index, dtype=np.uint8)
            writer.write(frame)
        writer.release()

    keypoints = tmp_path / "keypoints.csv"
    with keypoints.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["frame_idx", "camera_id", "person_id", "joint_idx", "x", "y", "score", "valid"],
        )
        writer.writeheader()
        for camera_id in cameras:
            for joint_index, point in zip((5, 7, 9), ((100, 50), (130, 90), (180, 80)), strict=True):
                writer.writerow(
                    {
                        "frame_idx": 1,
                        "camera_id": camera_id,
                        "person_id": 1,
                        "joint_idx": joint_index,
                        "x": point[0],
                        "y": point[1],
                        "score": 0.95,
                        "valid": True,
                    }
                )
    events = {
        "status": "decision_evidence_events",
        "frame_count": frame_count,
        "fps": fps,
        "measurement_space": "tk3d_world_3d",
        "camera_overlay_space": "observed_vitpose_2d_visual_trace_only",
        "camera_overlay_warning": "3D decision; 2D visual trace.",
        "events": [
            {
                "event_id": "E1",
                "movement_id": "M01",
                "movement_name": "Movement one",
                "metric_id": "executing_elbow_deg",
                "description": "Elbow source range.",
                "decision_status": "confirmed_source_bound_minor",
                "display_label": "Deduction candidate",
                "display_color": "red",
                "deduction_points": 0.1,
                "measurement": {
                    "value": 130.0,
                    "unit": "deg",
                    "interval_95": [125.0, 135.0],
                    "rule_operator": "range",
                    "rule_limits": [90.0, 120.0],
                },
                "evidence_window": {"start_frame": 1, "anchor_frame": 1, "end_frame": 1},
                "visual_geometry": {
                    "kind": "joint_angle",
                    "joint_indices": [5, 7, 9],
                    "vertex_joint_index": 7,
                    "unit": "deg",
                },
            }
        ],
    }
    output = tmp_path / "annotated.mp4"

    report = render_decision_evidence_video(
        camera_videos=cameras,
        keypoints_2d_csv=keypoints,
        evidence_events=events,
        output_path=output,
    )

    assert output.is_file() and output.stat().st_size > 0
    assert report["frame_count"] == frame_count
    assert report["evidence_frame_count"] == 1
    capture = cv2.VideoCapture(str(output))
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == frame_count
    capture.release()
