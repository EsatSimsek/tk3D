from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .visualization_3d import COCO_WHOLEBODY_EDGES


def write_pose3d_html_viewer(
    keypoints_3d: np.ndarray,
    output_path: str | Path,
    *,
    fps: float,
    title: str,
) -> Path:
    """Write a standalone, interactive Three.js viewer for a 3D pose sequence."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = _normalize_for_viewer(keypoints_3d)
    payload = {
        "title": title,
        "fps": float(fps),
        "jointCount": int(normalized.shape[1]),
        "edges": [list(edge) for edge in COCO_WHOLEBODY_EDGES if max(edge) < normalized.shape[1]],
        "frames": _json_frames(normalized),
    }
    path.write_text(_build_html(payload), encoding="utf-8")
    return path


def _normalize_for_viewer(keypoints_3d: np.ndarray) -> np.ndarray:
    values = np.asarray(keypoints_3d, dtype=float).copy()
    if values.ndim != 3 or values.shape[-1] != 3:
        raise ValueError(f"Expected [frames, joints, 3], got {values.shape}")
    finite = values[np.all(np.isfinite(values), axis=-1)]
    if finite.size == 0:
        return values
    low = np.percentile(finite, 1.0, axis=0)
    high = np.percentile(finite, 99.0, axis=0)
    scale = float(np.max(np.maximum(high - low, 1e-6)))
    values /= scale
    values[..., 0] -= float(np.nanmedian(values[..., 0]))
    values[..., 1] -= float(np.nanmedian(values[..., 1]))
    values[..., 2] -= float(np.nanmin(values[..., 2]))
    # TK3D uses x-right/y-forward/z-up; Three.js uses x-right/y-up/z-toward-viewer.
    return values[..., [0, 2, 1]] * np.asarray([1.0, 1.0, -1.0])


def _json_frames(values: np.ndarray) -> list[list[list[float | None]]]:
    rounded = np.round(values, decimals=4)
    rows: list[list[list[float | None]]] = []
    for frame in rounded:
        rows.append([[float(value) if np.isfinite(value) else None for value in point] for point in frame])
    return rows


def _build_html(payload: dict[str, object]) -> str:
    data = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    frame_count = len(payload["frames"])
    return f"""<!doctype html>
<html lang="tr">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>TK3D Pose Viewer</title>
<style>
html,body,#stage{{margin:0;width:100%;height:100%;overflow:hidden;background:#f4f6fa;font-family:Arial,sans-serif}}
.badge,.controls{{position:fixed;background:rgba(255,255,255,.9);border:1px solid #c9d0dd;border-radius:9px;padding:10px 12px;box-shadow:0 5px 20px #15213a22}}
.badge{{top:16px;left:16px;color:#17223a}} .controls{{left:16px;right:16px;bottom:16px;display:flex;align-items:center;gap:10px}}
button{{width:38px;height:34px;border:1px solid #9aa7bd;border-radius:7px;background:white;cursor:pointer;font-size:16px}} input[type=range]{{flex:1}} #readout{{min-width:100px;text-align:center;color:#263550}}
</style>
</head>
<body>
<div id="stage"></div><div class="badge" id="title"></div>
<div class="controls"><button id="play">▶</button><button id="prev">‹</button><input id="frame" type="range" min="0" max="{max(frame_count - 1, 0)}" value="0" /><button id="next">›</button><span id="readout"></span><button id="reset">⌂</button></div>
<script id="pose-data" type="application/json">{data}</script>
<script type="importmap">{{"imports":{{"three":"https://unpkg.com/three@0.164.1/build/three.module.js","three/addons/":"https://unpkg.com/three@0.164.1/examples/jsm/"}}}}</script>
<script type="module">
import * as THREE from 'three'; import {{OrbitControls}} from 'three/addons/controls/OrbitControls.js';
const data=JSON.parse(document.getElementById('pose-data').textContent), stage=document.getElementById('stage');
document.getElementById('title').textContent='TK3D · '+data.title;
const scene=new THREE.Scene(), camera=new THREE.PerspectiveCamera(45,innerWidth/innerHeight,.01,100); camera.position.set(1.8,1.5,3.4);
const renderer=new THREE.WebGLRenderer({{antialias:true}}); renderer.setPixelRatio(Math.min(devicePixelRatio,2)); renderer.setSize(innerWidth,innerHeight); stage.appendChild(renderer.domElement);
const controls=new OrbitControls(camera,renderer.domElement); controls.target.set(0,1,0); controls.enableDamping=true;
scene.add(new THREE.HemisphereLight(0xffffff,0xaab6ca,2.4)); const key=new THREE.DirectionalLight(0xffffff,2); key.position.set(3,4,2); scene.add(key);
const grid=new THREE.GridHelper(3.6,18,0xc6cedd,0xe7ebf2); scene.add(grid);
const jointGeometry=new THREE.SphereGeometry(.025,8,6), jointMaterial=new THREE.MeshStandardMaterial({{color:0x00d94f}}), boneMaterial=new THREE.LineBasicMaterial({{color:0x2498ed}});
const joints=Array.from({{length:data.jointCount}},()=>{{const m=new THREE.Mesh(jointGeometry,jointMaterial);scene.add(m);return m;}});
const bones=data.edges.map(()=>{{const g=new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(),new THREE.Vector3()]);const l=new THREE.Line(g,boneMaterial);scene.add(l);return l;}});
let frame=0,playing=true,last=performance.now(); const slider=document.getElementById('frame'),readout=document.getElementById('readout'),play=document.getElementById('play');
function valid(p){{return p&&p.every(v=>Number.isFinite(v));}} function setFrame(index){{frame=(index+data.frames.length)%data.frames.length;const points=data.frames[frame];joints.forEach((mesh,i)=>{{const p=points[i];mesh.visible=valid(p);if(mesh.visible)mesh.position.set(...p);}});bones.forEach((line,i)=>{{const [a,b]=data.edges[i],pa=points[a],pb=points[b];line.visible=valid(pa)&&valid(pb);if(line.visible)line.geometry.setFromPoints([new THREE.Vector3(...pa),new THREE.Vector3(...pb)]);}});slider.value=frame;readout.textContent=`${{frame+1}} / ${{data.frames.length}}`;}}
function toggle(){{playing=!playing;play.textContent=playing?'Ⅱ':'▶';}} play.onclick=toggle; document.getElementById('prev').onclick=()=>{{playing=false;setFrame(frame-1);}};document.getElementById('next').onclick=()=>{{playing=false;setFrame(frame+1);}};slider.oninput=()=>{{playing=false;setFrame(Number(slider.value));}};document.getElementById('reset').onclick=()=>{{camera.position.set(1.8,1.5,3.4);controls.target.set(0,1,0);controls.update();}};
addEventListener('resize',()=>{{camera.aspect=innerWidth/innerHeight;camera.updateProjectionMatrix();renderer.setSize(innerWidth,innerHeight);}}); addEventListener('keydown',e=>{{if(e.code==='Space'){{e.preventDefault();toggle();}}}});
function animate(now){{requestAnimationFrame(animate);if(playing&&now-last>=1000/data.fps){{last=now;setFrame(frame+1);}}controls.update();renderer.render(scene,camera);}} setFrame(0);animate(performance.now());
</script></body></html>"""
