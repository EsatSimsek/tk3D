from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import cv2
import imageio_ffmpeg

from src.artifact_io import sha256_file

_sha256 = sha256_file


_SAFE_CAMERA_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")


def build_browser_review_videos(
    camera_specs: list[tuple[str, Path]],
    output_dir: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    """Transcode lossless/source review videos to browser-compatible H.264 MP4."""
    if len(camera_specs) < 2:
        raise ValueError("at least two camera videos are required")
    if manifest_path.exists():
        raise FileExistsError(f"Output already exists; refusing to overwrite: {manifest_path}")
    if len({camera_id for camera_id, _ in camera_specs}) != len(camera_specs):
        raise ValueError("camera ids must be unique")
    output_dir.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, Any]] = []
    for camera_id, raw_input in camera_specs:
        if not _SAFE_CAMERA_ID.fullmatch(camera_id):
            raise ValueError(f"invalid camera id: {camera_id}")
        input_path = raw_input.resolve()
        if not input_path.is_file():
            raise FileNotFoundError(f"camera video is missing: {input_path}")
        output_path = (output_dir / f"{camera_id}_browser.mp4").resolve()
        if output_path.exists():
            raise FileExistsError(f"Output already exists; refusing to overwrite: {output_path}")
        entries.append(_transcode_one(camera_id, input_path, output_path))

    manifest = {
        "schema_version": 1,
        "artifact_type": "browser_review_video_set",
        "codec": "h264",
        "container": "mp4",
        "timeline_contract": {
            "frame_count_preserved": True,
            "fps_preserved": True,
            "dimensions_preserved": True,
            "frames_dropped": False,
        },
        "videos": entries,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("x", encoding="utf-8", newline="") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return manifest


def _transcode_one(camera_id: str, input_path: Path, output_path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open camera video: {input_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    declared_frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    if declared_frames <= 0 or fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError(f"camera video metadata is invalid: {input_path}")

    temporary = output_path.with_name(f"{output_path.stem}.tmp{output_path.suffix}")
    if temporary.exists():
        raise FileExistsError(f"Temporary output already exists: {temporary}")
    capture.release()
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg_exe,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-i",
        str(input_path),
        "-map",
        "0:v:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(temporary),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"H.264 browser transcode failed for {camera_id}: {result.stderr.strip()}"
        )

    verification = cv2.VideoCapture(str(temporary))
    if not verification.isOpened():
        raise RuntimeError(f"could not reopen browser review video: {temporary}")
    output_frames = int(round(verification.get(cv2.CAP_PROP_FRAME_COUNT)))
    output_fps = float(verification.get(cv2.CAP_PROP_FPS))
    output_width = int(round(verification.get(cv2.CAP_PROP_FRAME_WIDTH)))
    output_height = int(round(verification.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    verification.release()
    if output_frames != declared_frames:
        raise RuntimeError(
            f"browser review verification frame count changed for {camera_id}: "
            f"input={declared_frames}, output={output_frames}"
        )
    if abs(output_fps - fps) > 1e-3:
        raise RuntimeError(
            f"browser review FPS changed for {camera_id}: input={fps}, output={output_fps}"
        )
    if (output_width, output_height) != (width, height):
        raise RuntimeError(
            f"browser review dimensions changed for {camera_id}: "
            f"input={width}x{height}, output={output_width}x{output_height}"
        )
    temporary.replace(output_path)
    return {
        "camera_id": camera_id,
        "input": {
            "path": str(input_path),
            "sha256": _sha256(input_path),
            "frame_count": declared_frames,
            "fps": fps,
            "width": width,
            "height": height,
            "duration_sec": declared_frames / fps,
        },
        "output": {
            "path": str(output_path),
            "sha256": _sha256(output_path),
            "frame_count": output_frames,
            "fps": output_fps,
            "width": output_width,
            "height": output_height,
            "duration_sec": output_frames / output_fps,
            "mime_type": "video/mp4",
        },
    }


def _parse_camera(value: str) -> tuple[str, Path]:
    camera_id, separator, raw_path = value.partition("=")
    if not separator or not camera_id.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("camera must use CAMERA_ID=PATH format")
    return camera_id.strip(), Path(raw_path.strip())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create exact-timeline H.264/MP4 camera proxies for the HTML review screen."
    )
    parser.add_argument("--camera", action="append", required=True, type=_parse_camera)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()

    manifest = build_browser_review_videos(
        [(camera_id, path.resolve()) for camera_id, path in args.camera],
        args.output_dir.resolve(),
        args.manifest.resolve(),
    )
    for item in manifest["videos"]:
        print(item["output"]["path"])
    print(args.manifest.resolve())


if __name__ == "__main__":
    main()
