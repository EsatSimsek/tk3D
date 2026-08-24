"""Vücut hızını çizip elle etiketlenmiş hareket sınırlarını üstüne işaretler.

Amaç: segment tespitinin neyle boğuştuğunu gözle görmek. Geçici inceleme
script'idir, depoya commit edilmez.

Çalıştırma (repo kökünden):
    .venv312\\Scripts\\python tmp_enerji_grafigi.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data_structures import COCO_BODY_JOINT_INDICES
from src.poomsae_scoring import load_movement_timeline, load_poomsae_spec
from src.scoring_readiness import joint_speed

POSE = ROOT / "outputs/poomsae_1_zed2i_20260731_trimmed/runs/poomsae1-zed2i-rgbd-gated-ultra-rerun-20260802/json/vitpose_session_3d.json"
SPEC = ROOT / "config/scoring/poomsae/taegeuk_1_jang_v0_draft.yaml"
TIMELINE = ROOT / "config/scoring/timelines/poomsae1_zed2i_rgbd_rerun_20260802_draft.yaml"
OUT = ROOT / "tmp_enerji_grafigi.png"

INK = "#1f2328"
MUTED = "#8b949e"
LINE = "#2f6fb0"      # vücut hızı
BOUND = "#c2410c"     # gerçek sınır


def main() -> None:
    if not POSE.is_file():
        raise SystemExit(f"Poz dosyası bulunamadı:\n  {POSE}\nEsat'ın gönderdiği dosyayı bu yola koy.")

    spec = load_poomsae_spec(SPEC)
    timeline = load_movement_timeline(TIMELINE, spec)
    pose = json.loads(POSE.read_text(encoding="utf-8"))
    keypoints = np.asarray(pose["keypoints_3d_world"], dtype=float)
    fps = float(timeline["fps"])

    # Kare başına vücut hızı: gövde eklemlerinin ortalama hızı.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        speeds = joint_speed(keypoints, fps=fps)
        body = [index for index in COCO_BODY_JOINT_INDICES if index < speeds.shape[1]]
        energy = np.nanmean(speeds[:, body], axis=1)
    energy = np.nan_to_num(energy, nan=0.0)

    # Yumuşatma: tek karelik gürültüyü sil, şeklini bozma.
    window = 15
    smooth = np.convolve(energy, np.ones(window) / window, mode="same")

    boundaries = [segment["start_frame"] for segment in timeline["segments"]]
    boundaries.append(timeline["segments"][-1]["end_frame"])
    labels = [segment["movement_id"] for segment in timeline["segments"]]

    figure, axes = plt.subplots(figsize=(13, 4.5))
    axes.plot(smooth, color=LINE, linewidth=2.0, zorder=3)

    for index, frame in enumerate(boundaries):
        axes.axvline(frame, color=BOUND, linewidth=1.5, linestyle="--", alpha=0.85, zorder=2)
        if index < len(labels):
            middle = (frame + boundaries[index + 1]) / 2
            axes.text(
                middle, smooth.max() * 1.06, labels[index],
                ha="center", va="bottom", fontsize=11, color=BOUND, fontweight="bold",
            )

    axes.set_title(
        "Vücut hızı ve elle etiketlenmiş hareket sınırları\n"
        "Kesikli çizgiler gerçek sınırlar — hızın orada düşmesi beklenirdi",
        fontsize=12, color=INK, loc="left", pad=16,
    )
    axes.set_xlabel("kare", color=MUTED)
    axes.set_ylabel("vücut hızı", color=MUTED)
    axes.set_xlim(0, len(smooth) - 1)
    axes.set_ylim(0, smooth.max() * 1.18)
    axes.grid(axis="y", color=MUTED, alpha=0.18, linewidth=0.8)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(MUTED)
    axes.tick_params(colors=MUTED)

    figure.tight_layout()
    figure.savefig(OUT, dpi=140, facecolor="white")
    print(f"grafik: {OUT}")
    print()
    print("sınır | o karedeki hız | o anki genel seviye")
    median = float(np.median(smooth))
    for frame in boundaries:
        value = float(smooth[frame])
        durum = "DÜŞÜK" if value < median * 0.75 else ("orta" if value < median else "YÜKSEK")
        print(f"{frame:5d} | {value:14.3f} | {durum}")
    print()
    print(f"kaydın genel hız ortancası: {median:.3f}")


if __name__ == "__main__":
    main()
