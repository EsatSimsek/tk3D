from __future__ import annotations

import argparse
import base64
import html
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.artifact_io import sha256_file  # noqa: E402
from src.poomsae_scoring import load_movement_timeline, load_poomsae_spec  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a single review page showing the frames around each movement's anchor, so a "
            "proposed timeline can be checked by eye instead of by scrubbing the video."
        )
    )
    parser.add_argument("--timeline", required=True, help="the timeline to review, usually a draft")
    parser.add_argument("--poomsae-spec", required=True)
    parser.add_argument(
        "--camera",
        action="append",
        required=True,
        metavar="CAMERA_ID=VIDEO",
        help="repeatable; one strip per camera is rendered for every movement",
    )
    parser.add_argument("--anchor", default="fixation", help="which phase anchor to review")
    parser.add_argument(
        "--radius-frames",
        type=int,
        default=15,
        help=(
            "How far either side of the proposed anchor to look. The default covers the worst "
            "error measured against the hand-labelled recording, which was thirteen frames."
        ),
    )
    parser.add_argument(
        "--step-frames",
        type=int,
        default=5,
        help="Frame spacing inside that range; smaller means more thumbnails and a tighter read.",
    )
    parser.add_argument(
        "--thumbnail-width",
        type=int,
        default=180,
        help="Kept small so the whole strip fits on one screen without sideways scrolling.",
    )
    parser.add_argument("--output-html", required=True)
    args = parser.parse_args()

    if args.radius_frames < 1:
        raise SystemExit("--radius-frames must be at least 1")
    if args.step_frames < 1:
        raise SystemExit("--step-frames must be at least 1")
    if args.thumbnail_width < 80:
        raise SystemExit("--thumbnail-width must be at least 80")

    timeline_path = _resolve(args.timeline)
    spec_path = _resolve(args.poomsae_spec)
    for label, path in (("timeline", timeline_path), ("poomsae_spec", spec_path)):
        if not path.is_file():
            raise SystemExit(f"Input file is missing ({label}): {path}")
    output = _resolve(args.output_html)
    if output.exists():
        raise SystemExit(f"Output already exists; refusing to overwrite: {output}")

    cameras = [_parse_camera(value) for value in args.camera]
    spec = load_poomsae_spec(spec_path)
    timeline = load_movement_timeline(timeline_path, spec)

    offsets = list(range(-args.radius_frames, args.radius_frames + 1, args.step_frames))
    if 0 not in offsets:
        offsets = sorted(set(offsets) | {0})

    rows: list[dict] = []
    for segment in timeline["segments"]:
        anchor = segment["anchors"].get(args.anchor)
        if anchor is None:
            continue
        frames = [
            frame
            for frame in (int(anchor) + offset for offset in offsets)
            if 0 <= frame < timeline["frame_count"]
        ]
        rows.append({"segment": segment, "anchor": int(anchor), "frames": frames})
    if not rows:
        raise SystemExit(f"no segment in this timeline carries a {args.anchor!r} anchor")

    wanted = {frame for row in rows for frame in row["frames"]}
    shots: dict[str, dict[int, str]] = {}
    for camera_id, video_path in cameras:
        if not video_path.is_file():
            raise SystemExit(f"video not found ({camera_id}): {video_path}")
        shots[camera_id] = _grab_frames(video_path, wanted, timeline, args.thumbnail_width)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        _render_html(
            rows=rows,
            cameras=[camera_id for camera_id, _ in cameras],
            shots=shots,
            timeline=timeline,
            anchor_name=args.anchor,
            bindings={
                "timeline": {"path": _binding_path(timeline_path), "sha256": sha256_file(timeline_path)},
                "poomsae_spec": {"path": _binding_path(spec_path), "sha256": sha256_file(spec_path)},
            },
        ),
        encoding="utf-8",
    )
    print(output)
    print(f"{len(rows)} movement(s), {len(offsets)} frame(s) each, {len(cameras)} camera(s)")
    print("Open the page, find the frame where the posture is actually held, and correct the timeline.")


def _grab_frames(video_path: Path, wanted: set[int], timeline: dict, width: int) -> dict[int, str]:
    """Read the video once and keep only the frames the sheet needs.

    Seeking by frame is unreliable on some codecs, so the video is walked from the start
    and the wanted frames are kept as they go past. A recording of a few hundred frames
    reads in well under a second.
    """
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise SystemExit(f"cannot open video: {video_path}")
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count != int(timeline["frame_count"]):
            raise SystemExit(
                f"{video_path.name} has {frame_count} frames but the timeline declares "
                f"{timeline['frame_count']}; they do not describe the same recording"
            )
        images: dict[int, str] = {}
        index = 0
        last = max(wanted)
        while index <= last:
            ok, frame = capture.read()
            if not ok:
                break
            if index in wanted:
                images[index] = _encode(frame, width)
            index += 1
    finally:
        capture.release()
    missing = sorted(wanted - set(images))
    if missing:
        raise SystemExit(f"{video_path.name} ended before frame {missing[0]}")
    return images


def _encode(frame, width: int) -> str:
    height = max(1, round(frame.shape[0] * width / frame.shape[1]))
    small = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    ok, buffer = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    if not ok:
        raise SystemExit("could not encode a frame as JPEG")
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def _render_html(*, rows, cameras, shots, timeline, anchor_name, bindings) -> str:
    fps = float(timeline["fps"])
    parts = [
        "<!DOCTYPE html>",
        '<html lang="tr"><head><meta charset="utf-8">',
        f"<title>{html.escape(timeline['timeline_id'])} — çapa incelemesi</title>",
        "<style>",
        "body{font-family:system-ui,Segoe UI,sans-serif;margin:24px;background:#fafafa;color:#111}",
        "h1{font-size:18px;margin:0 0 4px}",
        "p.note{color:#444;max-width:70ch;line-height:1.5}",
        "section{margin:28px 0;padding:12px;background:#fff;border:1px solid #ddd;border-radius:6px}",
        "h2{font-size:15px;margin:0 0 2px}",
        "div.meta{color:#666;font-size:12px;margin-bottom:10px}",
        "div.strip{display:flex;gap:8px;overflow-x:auto;padding-bottom:6px}",
        "figure{margin:0;text-align:center;flex:0 0 auto}",
        "figure img{display:block;border:2px solid transparent;border-radius:4px}",
        "figure.proposed img{border-color:#c0392b}",
        "figcaption{font-size:12px;color:#555;margin-top:3px}",
        "figure.proposed figcaption{color:#c0392b;font-weight:600}",
        "table.bind{border-collapse:collapse;font-size:11px;color:#666;margin-top:28px}",
        "table.bind td{border:1px solid #ddd;padding:3px 6px}",
        "</style></head><body>",
        f"<h1>{html.escape(timeline['timeline_id'])} — “{html.escape(anchor_name)}” çapa incelemesi</h1>",
        "<p class='note'>Her satır bir hareketin çapasının etrafındaki kareleri gösterir. "
        "Kırmızı çerçeveli kare taslağın önerdiğidir. Duruşun gerçekten tutulduğu kare "
        "başka biriyse, o karenin numarasını zaman çizelgesine yaz. "
        "Bu sayfa bir öneridir; hiçbir kesinti veya puan iddiası taşımaz.</p>",
        f"<p class='note'>Etiket kaynağı: <b>{html.escape(str(timeline['label_source']))}</b> · "
        f"{len(rows)} hareket · {fps:g} kare/saniye</p>",
    ]
    for row in rows:
        segment = row["segment"]
        seconds = row["anchor"] / fps if fps else 0.0
        parts.append("<section>")
        parts.append(
            f"<h2>{html.escape(segment['movement_id'])} — önerilen çapa {row['anchor']}. kare "
            f"({seconds:.2f} sn)</h2>"
        )
        parts.append(
            f"<div class='meta'>aralık {segment['start_frame']}–{segment['end_frame']} · "
            f"durum {html.escape(str(segment['label_status']))} · "
            f"güven {segment['confidence']}</div>"
        )
        for camera_id in cameras:
            parts.append(f"<div class='meta'>{html.escape(camera_id)}</div><div class='strip'>")
            for frame in row["frames"]:
                offset = frame - row["anchor"]
                css = "proposed" if offset == 0 else ""
                label = f"{frame}" if offset == 0 else f"{frame} ({offset:+d})"
                parts.append(
                    f"<figure class='{css}'>"
                    f"<img src='data:image/jpeg;base64,{shots[camera_id][frame]}' alt='{frame}'>"
                    f"<figcaption>{label}</figcaption></figure>"
                )
            parts.append("</div>")
        parts.append("</section>")
    parts.append("<table class='bind'>")
    for label, item in bindings.items():
        parts.append(
            f"<tr><td>{html.escape(label)}</td><td>{html.escape(item['path'])}</td>"
            f"<td>{html.escape(item['sha256'])}</td></tr>"
        )
    parts.append("</table></body></html>")
    return "\n".join(parts) + "\n"


def _parse_camera(value: str) -> tuple[str, Path]:
    camera_id, separator, raw_path = value.partition("=")
    if not separator or not camera_id.strip() or not raw_path.strip():
        raise SystemExit(f"--camera must look like CAMERA_ID=VIDEO, got: {value}")
    return camera_id.strip(), _resolve(raw_path.strip())


def _binding_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


def _resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


if __name__ == "__main__":
    main()
