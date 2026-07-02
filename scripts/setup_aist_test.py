from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


DEFAULT_SEQUENCE = "gBR_sBM_cAll_d04_mBR0_ch01"
DOWNLOADER_URL = "https://raw.githubusercontent.com/google/aistplusplus_api/main/downloader.py"


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare local AIST/AIST++ test folders for TK3D.")
    parser.add_argument("--root", default="data/aist_test", help="AIST test root folder")
    parser.add_argument("--sequence", default=DEFAULT_SEQUENCE, help="AIST++ sequence name with cAll")
    parser.add_argument("--cameras", nargs="+", default=["c01", "c02"], help="Camera IDs to use")
    parser.add_argument("--session-out", default="data/aist_test/session.yaml", help="Generated TK3D session yaml")
    args = parser.parse_args()

    root = Path(args.root)
    videos_dir = root / "videos"
    annotations_dir = root / "annotations"
    videos_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    expected_videos = [video_name_for_camera(args.sequence, camera_id) + ".mp4" for camera_id in args.cameras]
    manifest = {
        "sequence": args.sequence,
        "cameras": args.cameras,
        "downloader_url": DOWNLOADER_URL,
        "folders": {
            "videos": str(videos_dir),
            "annotations": str(annotations_dir),
        },
        "expected_videos": expected_videos,
        "annotation_targets": [
            "annotations/cameras/",
            "annotations/keypoints3d/",
            "annotations/keypoints2d/ optional",
        ],
    }
    (root / "aist_test_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_session_yaml(Path(args.session_out), root, args.sequence, args.cameras)

    print(f"prepared: {root}")
    print(f"manifest: {root / 'aist_test_manifest.json'}")
    print(f"session: {args.session_out}")
    print("expected videos:")
    for video in expected_videos:
        print(f"  {videos_dir / video}")
    print(f"downloader.py URL: {DOWNLOADER_URL}")


def video_name_for_camera(sequence: str, camera_id: str) -> str:
    if "_cAll_" in sequence:
        return sequence.replace("_cAll_", f"_{camera_id}_")
    parts = sequence.split("_")
    return "_".join(camera_id if part.startswith("c") and len(part) == 3 else part for part in parts)


def write_session_yaml(path: Path, root: Path, sequence: str, cameras: list[str]) -> None:
    session = {
        "session_id": "aist_test",
        "task_name": sequence,
        "fps": 60.0,
        "cameras": [
            {
                "camera_id": camera_id,
                "video_path": f"videos/{video_name_for_camera(sequence, camera_id)}.mp4",
                "calibration_video_path": None,
            }
            for camera_id in cameras
        ],
        "sync": {
            "method": "frame_index",
            "offsets": {camera_id: 0 for camera_id in cameras},
        },
        "aist": {
            "sequence": sequence,
            "annotations_dir": "annotations",
            "camera_source": "AIST++ cameras/*.json + mapping.txt",
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(session, sort_keys=False), encoding="utf-8")


if __name__ == "__main__":
    main()
