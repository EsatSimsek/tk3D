from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.render_aist_smpl_mesh import _center_vertices_per_frame, _load_smpl, _motion_to_vertices
from src.progress import ProgressBar, print_step
from src.smpl_mesh import find_smpl_model_file, load_aist_smpl_motion, selected_frame_indices
from src.video_io import ensure_output_tree, load_session


def main() -> None:
    parser = argparse.ArgumentParser(description="Export AIST++ SMPL motion to an interactive Three.js HTML viewer.")
    parser.add_argument("--session", default="data/aist_test/session_all.yaml")
    parser.add_argument("--annotations-dir", default=None)
    parser.add_argument("--smpl-dir", default="models/smpl")
    parser.add_argument("--gender", default="MALE", choices=["MALE", "FEMALE", "NEUTRAL"])
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--max-frames", type=int, default=240)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--world", action="store_true", help="Use original world coordinates instead of centering the body for viewing.")
    args = parser.parse_args()

    print("=" * 72, flush=True)
    print("TK3D SMPL THREE.JS VIEWER EXPORT", flush=True)
    print("=" * 72, flush=True)
    print_step(1, 5, "Loading session and SMPL inputs")
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

    print_step(2, 5, f"Loading SMPL model and preparing {len(indices)} frames")
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    smpl = _load_smpl(ROOT / args.smpl_dir, args.gender, batch_size=len(indices), device=device)
    print_step(3, 5, "Converting SMPL motion to mesh vertices")
    vertices, faces = _motion_to_vertices(smpl, motion, indices, device=device)
    if not args.world:
        vertices = _center_vertices_per_frame(vertices)
    print_step(4, 5, "Normalizing mesh for browser viewing")
    vertices = _normalize_for_viewer(vertices)

    output_paths = ensure_output_tree(ROOT / args.output_root, session.session_id)
    viewer_dir = output_paths["root"] / "viewer"
    viewer_dir.mkdir(parents=True, exist_ok=True)
    html_path = viewer_dir / "aist_smpl_viewer.html"

    print_step(5, 5, "Writing interactive HTML viewer")
    html_path.write_text(build_html(vertices, faces, fps=args.fps, sequence=sequence), encoding="utf-8")
    print(f"saved: {html_path}")
    print(f"frames: {vertices.shape[0]}")
    print(f"vertices: {vertices.shape[1]}")
    print(f"faces: {faces.shape[0]}")


def _normalize_for_viewer(vertices: np.ndarray) -> np.ndarray:
    finite = vertices[np.all(np.isfinite(vertices), axis=-1)]
    mins = np.nanpercentile(finite, 1, axis=0)
    maxs = np.nanpercentile(finite, 99, axis=0)
    scale = float(np.max(np.maximum(maxs - mins, 1e-6)))
    normalized = vertices / scale * 2.1
    normalized[:, :, 0] -= float(np.nanmedian(normalized[:, :, 0]))
    normalized[:, :, 2] -= float(np.nanmedian(normalized[:, :, 2]))
    normalized[:, :, 1] -= float(np.nanmin(normalized[:, :, 1]))
    return normalized.astype(np.float32)


def _round_nested(values: np.ndarray, decimals: int = 4) -> list[list[float]]:
    flattened = values.reshape(values.shape[0], -1)
    progress = ProgressBar("encode frames", flattened.shape[0])
    rows = []
    for frame_idx, frame in enumerate(flattened):
        rows.append(np.round(frame, decimals=decimals).tolist())
        progress.print(frame_idx + 1)
    progress.done()
    return rows


def build_html(vertices: np.ndarray, faces: np.ndarray, fps: float, sequence: str) -> str:
    payload = {
        "sequence": sequence,
        "fps": fps,
        "vertexCount": int(vertices.shape[1]),
        "frames": _round_nested(vertices),
        "faces": faces.astype(np.int32).reshape(-1).tolist(),
    }
    data_json = json.dumps(payload, separators=(",", ":"))
    return f"""<!doctype html>
<html lang=\"tr\">
<head>
<meta charset=\"utf-8\" />
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
<title>TK3D SMPL Viewer</title>
<style>
  html, body {{ margin: 0; width: 100%; height: 100%; overflow: hidden; background: #f6f7f9; color: #172033; font-family: Arial, Helvetica, sans-serif; }}
  #stage {{ position: fixed; inset: 0; }}
  .topbar {{ position: fixed; left: 16px; right: 16px; bottom: 16px; min-height: 48px; display: flex; align-items: center; gap: 10px; padding: 10px 12px; background: rgba(255,255,255,0.88); border: 1px solid rgba(120,132,155,0.35); border-radius: 8px; box-shadow: 0 10px 28px rgba(20,30,50,0.14); backdrop-filter: blur(8px); }}
  button {{ width: 38px; height: 34px; border: 1px solid #9aa7bd; border-radius: 7px; background: #ffffff; color: #1d2a44; font-size: 16px; cursor: pointer; }}
  button.active {{ background: #21385f; color: white; border-color: #21385f; }}
  input[type=range] {{ flex: 1; min-width: 120px; accent-color: #21385f; }}
  .readout {{ min-width: 88px; font-size: 13px; color: #2b3448; text-align: center; }}
  .speed {{ width: 86px; }}
  .badge {{ position: fixed; left: 18px; top: 14px; padding: 8px 10px; background: rgba(255,255,255,0.84); border: 1px solid rgba(120,132,155,0.28); border-radius: 8px; color: #26354f; font-size: 13px; }}
</style>
</head>
<body>
<div id=\"stage\"></div>
<div class=\"badge\">TK3D · {sequence}</div>
<div class=\"topbar\">
  <button id=\"play\" title=\"Play/Pause\">▶</button>
  <button id=\"prev\" title=\"Previous frame\">‹</button>
  <input id=\"frame\" type=\"range\" min=\"0\" max=\"{vertices.shape[0] - 1}\" value=\"0\" />
  <button id=\"next\" title=\"Next frame\">›</button>
  <span id=\"readout\" class=\"readout\">1 / {vertices.shape[0]}</span>
  <input id=\"speed\" class=\"speed\" type=\"range\" min=\"0.25\" max=\"2\" step=\"0.25\" value=\"1\" title=\"Speed\" />
  <button id=\"wire\" title=\"Wireframe\">▦</button>
  <button id=\"reset\" title=\"Reset camera\">⌂</button>
</div>
<script id=\"tk3d-data\" type=\"application/json\">{data_json}</script>
<script type=\"importmap\">
{{
  "imports": {{
    "three": "https://unpkg.com/three@0.164.1/build/three.module.js",
    "three/addons/": "https://unpkg.com/three@0.164.1/examples/jsm/"
  }}
}}
</script>
<script type=\"module\">
import * as THREE from 'three';
import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';

const data = JSON.parse(document.getElementById('tk3d-data').textContent);
const stage = document.getElementById('stage');
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xf6f7f9);

const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.01, 100);
camera.position.set(0.15, 1.35, 4.4);

const renderer = new THREE.WebGLRenderer({{ antialias: true }});
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
stage.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.target.set(0, 1.05, 0);

scene.add(new THREE.HemisphereLight(0xffffff, 0xb5bfd2, 2.5));
const key = new THREE.DirectionalLight(0xffffff, 2.0);
key.position.set(2.0, 4.0, 3.0);
scene.add(key);
const fill = new THREE.DirectionalLight(0x9db7ff, 1.2);
fill.position.set(-3.0, 2.0, 2.5);
scene.add(fill);

const grid = new THREE.GridHelper(3.2, 16, 0xcbd2df, 0xe4e8ef);
grid.position.y = 0;
scene.add(grid);

const geometry = new THREE.BufferGeometry();
const positions = new Float32Array(data.frames[0]);
geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
geometry.setIndex(data.faces);
geometry.computeVertexNormals();

const material = new THREE.MeshStandardMaterial({{
  color: 0x75a4ff,
  roughness: 0.52,
  metalness: 0.04,
  side: THREE.DoubleSide,
}});
const mesh = new THREE.Mesh(geometry, material);
scene.add(mesh);

const playButton = document.getElementById('play');
const prevButton = document.getElementById('prev');
const nextButton = document.getElementById('next');
const frameSlider = document.getElementById('frame');
const speedSlider = document.getElementById('speed');
const readout = document.getElementById('readout');
const wireButton = document.getElementById('wire');
const resetButton = document.getElementById('reset');

let frame = 0;
let playing = true;
let lastStep = performance.now();

function setFrame(nextFrame) {{
  frame = (nextFrame + data.frames.length) % data.frames.length;
  geometry.attributes.position.array.set(data.frames[frame]);
  geometry.attributes.position.needsUpdate = true;
  geometry.computeVertexNormals();
  frameSlider.value = frame;
  readout.textContent = `${{frame + 1}} / ${{data.frames.length}}`;
}}

function setPlaying(value) {{
  playing = value;
  playButton.textContent = playing ? 'Ⅱ' : '▶';
  playButton.classList.toggle('active', playing);
}}

playButton.addEventListener('click', () => setPlaying(!playing));
prevButton.addEventListener('click', () => {{ setPlaying(false); setFrame(frame - 1); }});
nextButton.addEventListener('click', () => {{ setPlaying(false); setFrame(frame + 1); }});
frameSlider.addEventListener('input', () => {{ setPlaying(false); setFrame(Number(frameSlider.value)); }});
wireButton.addEventListener('click', () => {{ material.wireframe = !material.wireframe; wireButton.classList.toggle('active', material.wireframe); }});
resetButton.addEventListener('click', () => {{ camera.position.set(0.15, 1.35, 4.4); controls.target.set(0, 1.05, 0); controls.update(); }});

window.addEventListener('keydown', (event) => {{
  if (event.code === 'Space') {{ event.preventDefault(); setPlaying(!playing); }}
  if (event.key === 'ArrowRight') {{ setPlaying(false); setFrame(frame + 1); }}
  if (event.key === 'ArrowLeft') {{ setPlaying(false); setFrame(frame - 1); }}
}});

window.addEventListener('resize', () => {{
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}});

function animate(now) {{
  requestAnimationFrame(animate);
  const interval = 1000 / (data.fps * Number(speedSlider.value));
  if (playing && now - lastStep >= interval) {{
    lastStep = now;
    setFrame(frame + 1);
  }}
  controls.update();
  renderer.render(scene, camera);
}}

setPlaying(true);
setFrame(0);
animate(performance.now());
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()





