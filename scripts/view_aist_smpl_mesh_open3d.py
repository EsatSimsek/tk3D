from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.render_aist_smpl_mesh import _center_vertices_per_frame, _load_smpl, _motion_to_vertices
from src.smpl_mesh import find_smpl_model_file, load_aist_smpl_motion, selected_frame_indices
from src.video_io import load_session


def main() -> None:
    parser = argparse.ArgumentParser(description="Play AIST++ SMPL motion in an interactive Open3D 3D window.")
    parser.add_argument("--session", default="data/aist_test/session_all.yaml")
    parser.add_argument("--annotations-dir", default=None)
    parser.add_argument("--smpl-dir", default="models/smpl")
    parser.add_argument("--gender", default="MALE", choices=["MALE", "FEMALE", "NEUTRAL"])
    parser.add_argument("--max-frames", type=int, default=240)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--world", action="store_true", help="Use original world coordinates instead of centering the body for viewing.")
    args = parser.parse_args()

    session_path = (ROOT / args.session).resolve()
    session = load_session(session_path)
    with session_path.open("r", encoding="utf-8") as file:
        raw_session = yaml.safe_load(file)
    sequence = raw_session.get("aist", {}).get("sequence") or session.task_name
    annotations_dir = Path(args.annotations_dir) if args.annotations_dir else session.root_dir / raw_session.get("aist", {}).get("annotations_dir", "annotations")
    motion_path = annotations_dir / "motions" / f"{sequence}.pkl"

    find_smpl_model_file(ROOT / args.smpl_dir, args.gender)
    motion = load_aist_smpl_motion(motion_path)
    indices = selected_frame_indices(motion.poses.shape[0], args.max_frames, args.stride)
    if not indices:
        raise SystemExit("No SMPL frames selected.")

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    smpl = _load_smpl(ROOT / args.smpl_dir, args.gender, batch_size=len(indices), device=device)
    vertices, faces = _motion_to_vertices(smpl, motion, indices, device=device)
    if not args.world:
        vertices = _center_vertices_per_frame(vertices)

    play_open3d(vertices, faces, fps=args.fps)


def play_open3d(vertices: np.ndarray, faces: np.ndarray, fps: float) -> None:
    import open3d as o3d

    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(vertices[0].astype(np.float64)),
        triangles=o3d.utility.Vector3iVector(faces.astype(np.int32)),
    )
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.45, 0.65, 1.0])

    frame_interval = 1.0 / max(float(fps), 1.0)
    state = {"frame": 0, "paused": False, "last": time.perf_counter()}

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="TK3D SMPL Animated Viewer", width=1280, height=720)
    vis.add_geometry(mesh)

    render = vis.get_render_option()
    render.background_color = np.asarray([1.0, 1.0, 1.0])
    render.mesh_show_back_face = True
    render.light_on = True

    bbox = mesh.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    extent = max(float(np.max(bbox.get_extent())), 1.0)
    view = vis.get_view_control()
    view.set_lookat(center)
    view.set_front([0.0, -0.65, 0.35])
    view.set_up([0.0, 0.0, 1.0])
    view.set_zoom(0.75 if extent < 250 else 0.55)

    def apply_frame(frame: int) -> None:
        state["frame"] = frame % len(vertices)
        mesh.vertices = o3d.utility.Vector3dVector(vertices[state["frame"]].astype(np.float64))
        mesh.compute_vertex_normals()
        vis.update_geometry(mesh)

    def toggle_pause(_: object) -> bool:
        state["paused"] = not state["paused"]
        print("paused" if state["paused"] else "playing")
        return False

    def next_frame(_: object) -> bool:
        state["paused"] = True
        apply_frame(state["frame"] + 1)
        return False

    def previous_frame(_: object) -> bool:
        state["paused"] = True
        apply_frame(state["frame"] - 1)
        return False

    def reset_view(_: object) -> bool:
        apply_frame(0)
        state["paused"] = True
        return False

    def close(_: object) -> bool:
        vis.close()
        return False

    def animate(_: object) -> bool:
        if state["paused"]:
            return False
        now = time.perf_counter()
        if now - state["last"] >= frame_interval:
            state["last"] = now
            apply_frame(state["frame"] + 1)
        return False

    vis.register_key_callback(32, toggle_pause)  # Space
    vis.register_key_callback(ord("N"), next_frame)
    vis.register_key_callback(ord("B"), previous_frame)
    vis.register_key_callback(ord("R"), reset_view)
    vis.register_key_callback(ord("Q"), close)
    vis.register_animation_callback(animate)

    print("Open3D viewer acildi.")
    print("Mouse: dondur / yakinlastir / kaydir")
    print("Space: durdur-devam, N: sonraki kare, B: onceki kare, R: basa al, Q: kapat")
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
