from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.video_io import ensure_output_tree, load_session


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a side-by-side preview video for two TK3D camera views.")
    parser.add_argument("--session", required=True, help="Path to a session yaml with at least 2 cameras")
    parser.add_argument("--output-root", default="outputs", help="Output root directory")
    parser.add_argument("--max-frames", type=int, default=240, help="Maximum frames to write")
    parser.add_argument("--width", type=int, default=960, help="Width of each camera tile")
    args = parser.parse_args()

    session = load_session(args.session)
    if len(session.cameras) < 2:
        raise SystemExit("Need at least 2 cameras for pair preview.")

    camera_a, camera_b = session.cameras[:2]
    output_paths = ensure_output_tree(ROOT / args.output_root, session.session_id)
    output_path = output_paths["videos"] / f"{camera_a.camera_id}_{camera_b.camera_id}_pair_preview.mp4"
    write_pair_preview(
        camera_a.video_path,
        camera_b.video_path,
        output_path,
        camera_a.camera_id,
        camera_b.camera_id,
        max_frames=args.max_frames,
        tile_width=args.width,
    )
    print(f"saved: {output_path}")


def write_pair_preview(
    video_a: Path,
    video_b: Path,
    output_path: Path,
    label_a: str,
    label_b: str,
    max_frames: int,
    tile_width: int,
) -> None:
    cap_a = cv2.VideoCapture(str(video_a))
    cap_b = cv2.VideoCapture(str(video_b))
    if not cap_a.isOpened():
        raise FileNotFoundError(f"Could not open {video_a}")
    if not cap_b.isOpened():
        raise FileNotFoundError(f"Could not open {video_b}")

    fps = cap_a.get(cv2.CAP_PROP_FPS) or cap_b.get(cv2.CAP_PROP_FPS) or 30.0
    height_a = int(cap_a.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    width_a = int(cap_a.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    tile_height = int(round(tile_width * height_a / width_a))
    output_size = (tile_width * 2, tile_height)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, output_size)

    try:
        for frame_idx in range(max_frames):
            ok_a, frame_a = cap_a.read()
            ok_b, frame_b = cap_b.read()
            if not ok_a or not ok_b:
                break
            tile_a = cv2.resize(frame_a, (tile_width, tile_height))
            tile_b = cv2.resize(frame_b, (tile_width, tile_height))
            _draw_label(tile_a, label_a, frame_idx)
            _draw_label(tile_b, label_b, frame_idx)
            writer.write(np.hstack([tile_a, tile_b]))
    finally:
        cap_a.release()
        cap_b.release()
        writer.release()


def _draw_label(frame: np.ndarray, label: str, frame_idx: int) -> None:
    cv2.rectangle(frame, (0, 0), (260, 44), (0, 0, 0), -1)
    cv2.putText(
        frame,
        f"{label} frame {frame_idx}",
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


if __name__ == "__main__":
    main()
