"""Hız çukurları neye yakın: hareket sınırlarına mı, bitiş duruşlarına mı?

Geçici inceleme script'i, depoya commit edilmez.

Çalıştırma (repo kökünden):
    .venv312\\Scripts\\python tmp_cukur_olc.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data_structures import COCO_BODY_JOINT_INDICES
from src.poomsae_scoring import load_movement_timeline, load_poomsae_spec
from src.scoring_readiness import joint_speed

POSE = ROOT / "outputs/poomsae_1_zed2i_20260731_trimmed/runs/poomsae1-zed2i-rgbd-gated-ultra-rerun-20260802/json/vitpose_session_3d.json"
SPEC = ROOT / "config/scoring/poomsae/taegeuk_1_jang_v0_draft.yaml"
TIMELINE = ROOT / "config/scoring/timelines/poomsae1_zed2i_rgbd_rerun_20260802_draft.yaml"


def find_dips(signal: np.ndarray, min_gap: int) -> list[int]:
    """Yerel dipler: kendi komşuluğunun en düşüğü olan kareler."""
    dips: list[int] = []
    for index in range(min_gap, len(signal) - min_gap):
        window = signal[index - min_gap : index + min_gap + 1]
        if signal[index] == window.min():
            if not dips or index - dips[-1] > min_gap:
                dips.append(index)
    return dips


def main() -> None:
    if not POSE.is_file():
        raise SystemExit(f"Poz dosyası bulunamadı:\n  {POSE}")

    spec = load_poomsae_spec(SPEC)
    timeline = load_movement_timeline(TIMELINE, spec)
    pose = json.loads(POSE.read_text(encoding="utf-8"))
    keypoints = np.asarray(pose["keypoints_3d_world"], dtype=float)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        speeds = joint_speed(keypoints, fps=float(timeline["fps"]))
        body = [index for index in COCO_BODY_JOINT_INDICES if index < speeds.shape[1]]
        energy = np.nan_to_num(np.nanmean(speeds[:, body], axis=1), nan=0.0)
    smooth = np.convolve(energy, np.ones(15) / 15, mode="same")

    segments = timeline["segments"]
    boundaries = [item["start_frame"] for item in segments] + [segments[-1]["end_frame"]]
    fixations = [item["anchors"]["fixation"] for item in segments]

    dips = find_dips(smooth, min_gap=20)
    print(f"bulunan çukur sayısı: {len(dips)}")
    print(f"gerçek sınır sayısı : {len(boundaries)}")
    print(f"bitiş duruşu sayısı : {len(fixations)}")
    print()
    print("çukur | en yakın sınır (uzaklık) | en yakın bitiş duruşu (uzaklık)")
    to_boundary, to_fixation = [], []
    for dip in dips:
        nearest_b = min(boundaries, key=lambda value: abs(value - dip))
        nearest_f = min(fixations, key=lambda value: abs(value - dip))
        db, df = abs(nearest_b - dip), abs(nearest_f - dip)
        to_boundary.append(db)
        to_fixation.append(df)
        print(f"{dip:5d} | {nearest_b:5d} ({db:3d} kare)        | {nearest_f:5d} ({df:3d} kare)")

    print()
    print(f"çukurların sınırlara ortalama uzaklığı        : {np.mean(to_boundary):.1f} kare")
    print(f"çukurların bitiş duruşlarına ortalama uzaklığı: {np.mean(to_fixation):.1f} kare")
    print()

    # Ters yön: her gerçek olay için en yakın çukur ne kadar uzakta?
    b_err = [min(abs(dip - value) for dip in dips) for value in boundaries]
    f_err = [min(abs(dip - value) for dip in dips) for value in fixations]
    print(f"her sınır için en yakın çukur        : {b_err}  ortalama {np.mean(b_err):.1f}")
    print(f"her bitiş duruşu için en yakın çukur : {f_err}  ortalama {np.mean(f_err):.1f}")
    print()

    # Kayma sistematik mi? İşaretli farka bak.
    signed = [min(boundaries, key=lambda v: abs(v - dip)) - dip for dip in dips]
    print(f"işaretli fark (sınır - çukur): {signed}")
    print(f"ortalama kayma: {np.mean(signed):+.1f} kare  (pozitif = çukur sınırdan önce)")


if __name__ == "__main__":
    main()
