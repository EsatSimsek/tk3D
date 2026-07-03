from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.smpl_mesh import find_smpl_model_file, load_aist_smpl_motion, selected_frame_indices, split_smpl_pose
from src.video_io import ensure_output_tree, load_session


def main() -> None:
    parser = argparse.ArgumentParser(description="Render AIST++ SMPL motion as a real 3D human mesh video.")
    parser.add_argument("--session", default="data/aist_test/session_all.yaml")
    parser.add_argument("--annotations-dir", default=None)
    parser.add_argument("--smpl-dir", default="models/smpl")
    parser.add_argument("--gender", default="MALE", choices=["MALE", "FEMALE", "NEUTRAL"])
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    session_path = (ROOT / args.session).resolve()
    session = load_session(session_path)
    with session_path.open("r", encoding="utf-8") as file:
        raw_session = yaml.safe_load(file)
    sequence = raw_session.get("aist", {}).get("sequence") or session.task_name
    annotations_dir = Path(args.annotations_dir) if args.annotations_dir else session.root_dir / raw_session.get("aist", {}).get("annotations_dir", "annotations")
    motion_path = annotations_dir / "motions" / f"{sequence}.pkl"

    try:
        smpl_model_file = find_smpl_model_file(ROOT / args.smpl_dir, args.gender)
    except FileNotFoundError as exc:
        raise SystemExit(
            "SMPL model dosyası eksik. Gerçek mesh için SMPL lisanslı dosyasını indirip şu klasöre koymalısın:\n"
            f"  {ROOT / args.smpl_dir}\n"
            "Beklenen dosya örneği:\n"
            f"  {ROOT / args.smpl_dir / f'SMPL_{args.gender.upper()}.pkl'}\n"
            "AIST++ motion dosyası hazır olsa bile bu lisanslı body model olmadan mesh üretilemez."
        ) from exc

    motion = load_aist_smpl_motion(motion_path)
    indices = selected_frame_indices(motion.poses.shape[0], args.max_frames, args.stride)
    if not indices:
        raise SystemExit("No SMPL frames selected.")

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    smpl = _load_smpl(ROOT / args.smpl_dir, args.gender, batch_size=len(indices), device=device)
    vertices, faces = _motion_to_vertices(smpl, motion, indices, device=device)

    output_paths = ensure_output_tree(ROOT / args.output_root, session.session_id)
    video_path = output_paths["videos"] / "aist_smpl_mesh.mp4"
    obj_path = output_paths["figures"] / "aist_smpl_mesh_frame0.obj"
    report_path = output_paths["json"] / "aist_smpl_mesh_report.json"

    render_mesh_video(vertices, faces, video_path, fps=args.fps, size=(1280, 720))
    export_obj(vertices[0], faces, obj_path)
    report = {
        "source": "AIST++ SMPL motion",
        "sequence": sequence,
        "motion_path": str(motion_path),
        "smpl_model_file": str(smpl_model_file),
        "frames_rendered": len(indices),
        "stride": args.stride,
        "video_path": str(video_path),
        "obj_path": str(obj_path),
    }
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print(f"saved: {video_path}")
    print(f"obj: {obj_path}")
    print(f"frames_rendered: {len(indices)}")
    print(f"report: {report_path}")


def _load_smpl(smpl_dir: Path, gender: str, batch_size: int, device: torch.device):
    _patch_legacy_smpl_runtime()
    from smplx import SMPL

    return SMPL(model_path=str(smpl_dir), gender=gender.upper(), batch_size=batch_size).to(device)


def _patch_legacy_smpl_runtime() -> None:
    import inspect

    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec  # type: ignore[attr-defined]
    for alias, value in {
        "bool": bool,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "str": str,
        "unicode": str,
    }.items():
        if alias not in np.__dict__:
            setattr(np, alias, value)


def _motion_to_vertices(smpl, motion, indices: list[int], device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    global_orient_np, body_pose_np = split_smpl_pose(motion.poses[indices])
    scale = float(motion.scaling[0]) if motion.scaling.size else 1.0
    global_orient = torch.from_numpy(global_orient_np).float().to(device)
    body_pose = torch.from_numpy(body_pose_np).float().to(device)
    transl = torch.from_numpy(motion.translation[indices]).float().to(device)
    with torch.no_grad():
        output = smpl(
            global_orient=global_orient,
            body_pose=body_pose,
            transl=transl,
            return_verts=True,
        )
    vertices = output.vertices.detach().cpu().numpy()
    vertices = vertices * scale
    return vertices, np.asarray(smpl.faces, dtype=np.int32)


def render_mesh_video(vertices: np.ndarray, faces: np.ndarray, path: Path, fps: float, size: tuple[int, int]) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    display_vertices = _center_vertices_per_frame(vertices)
    render_faces = faces[::2] if faces.shape[0] > 7000 else faces
    limits = _mesh_axis_limits(display_vertices)
    frames = []
    width, height = size
    dpi = 100
    for frame_idx, frame_vertices in enumerate(display_vertices):
        fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor("#fbfbfb")
        fig.patch.set_facecolor("#ffffff")
        mesh = Poly3DCollection(frame_vertices[render_faces], alpha=0.96, antialiased=True)
        mesh.set_facecolor("#8fb3ff")
        mesh.set_edgecolor("#1f2d4d")
        mesh.set_linewidth(0.025)
        ax.add_collection3d(mesh)
        ax.set_xlim(*limits["x"])
        ax.set_ylim(*limits["y"])
        ax.set_zlim(*limits["z"])
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=8, azim=-82)
        ax.set_axis_off()
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        frames.append(_zoom_frame(cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR), zoom=1.85))
        plt.close(fig)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()


def _zoom_frame(frame: np.ndarray, zoom: float) -> np.ndarray:
    if zoom <= 1.0:
        return frame
    height, width = frame.shape[:2]
    crop_width = max(1, int(width / zoom))
    crop_height = max(1, int(height / zoom))
    x0 = max(0, (width - crop_width) // 2)
    y0 = max(0, (height - crop_height) // 2)
    crop = frame[y0 : y0 + crop_height, x0 : x0 + crop_width]
    return cv2.resize(crop, (width, height), interpolation=cv2.INTER_CUBIC)

def _center_vertices_per_frame(vertices: np.ndarray) -> np.ndarray:
    centers = np.nanmedian(vertices, axis=1, keepdims=True)
    return vertices - centers

def _mesh_axis_limits(vertices: np.ndarray) -> dict[str, tuple[float, float]]:
    finite = vertices[np.all(np.isfinite(vertices), axis=-1)]
    mins = np.nanpercentile(finite, 1, axis=0)
    maxs = np.nanpercentile(finite, 99, axis=0)
    centers = (mins + maxs) / 2.0
    span = float(np.max(np.maximum(maxs - mins, 0.5)) * 1.15)
    return {
        "x": (float(centers[0] - span / 2), float(centers[0] + span / 2)),
        "y": (float(centers[1] - span / 2), float(centers[1] + span / 2)),
        "z": (float(centers[2] - span / 2), float(centers[2] + span / 2)),
    }


def export_obj(vertices: np.ndarray, faces: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for vertex in vertices:
            file.write(f"v {vertex[0]:.8f} {vertex[1]:.8f} {vertex[2]:.8f}\n")
        for face in faces + 1:
            file.write(f"f {face[0]} {face[1]} {face[2]}\n")


if __name__ == "__main__":
    main()







