from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from scripts.run_poomsae_scoring import WorkflowError, _load_profile, _transfer_timeline_binding


ROOT = Path(__file__).resolve().parents[1]


def _write_pose(path: Path, *, timestamps: list[float], run_id: str) -> None:
    path.write_text(
        json.dumps(
            {
                "session_id": "poomsae_test",
                "run_id": run_id,
                "inference_stride": 1,
                "sample_fps": 2.0,
                "frame_indices": [0, 1, 2],
                "timestamps_sec": timestamps,
            }
        ),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_timeline(path: Path, reference_pose: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 2,
                "timeline_id": "timeline-test",
                "frame_count": 3,
                "fps": 2.0,
                "source_binding": {
                    "session_id": "poomsae_test",
                    "run_id": "reference",
                    "pose_file": str(reference_pose),
                    "pose_file_sha256": _sha256(reference_pose),
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_project_one_command_profile_is_valid() -> None:
    profile = _load_profile(ROOT / "config" / "scoring" / "profiles" / "poomsae1_trimmed.yaml")

    assert profile["profile_id"] == "poomsae1_trimmed"
    assert profile["processing"]["stride"] == 1
    assert len(profile["videos"]) == 2
    assert {video["camera_id"] for video in profile["videos"]} == {"zed_35151067", "zed_37137479"}


def test_timeline_transfer_requires_identical_video_time_axis(tmp_path: Path) -> None:
    reference = tmp_path / "reference.json"
    new = tmp_path / "new.json"
    timeline = tmp_path / "timeline.yaml"
    _write_pose(reference, timestamps=[0.0, 0.5, 1.0], run_id="reference")
    _write_pose(new, timestamps=[0.0, 0.5, 1.0], run_id="new-run")
    _write_timeline(timeline, reference)

    transferred = _transfer_timeline_binding(
        timeline_path=timeline,
        reference_pose_path=reference,
        new_pose_path=new,
        session_id="poomsae_test",
        run_id="new-run",
    )

    assert transferred["source_binding"]["run_id"] == "new-run"
    assert transferred["source_binding"]["pose_file_sha256"] == _sha256(new)
    assert transferred["timeline_id"].endswith("new-run")


def test_timeline_transfer_fails_closed_when_timestamps_change(tmp_path: Path) -> None:
    reference = tmp_path / "reference.json"
    new = tmp_path / "new.json"
    timeline = tmp_path / "timeline.yaml"
    _write_pose(reference, timestamps=[0.0, 0.5, 1.0], run_id="reference")
    _write_pose(new, timestamps=[0.0, 0.6, 1.0], run_id="new-run")
    _write_timeline(timeline, reference)

    with pytest.raises(WorkflowError, match="timestamps differ"):
        _transfer_timeline_binding(
            timeline_path=timeline,
            reference_pose_path=reference,
            new_pose_path=new,
            session_id="poomsae_test",
            run_id="new-run",
        )
