"""Yeni yöntem: yavaşlama = bitiş duruşu, iki yavaşlama arası = bir hareket.

Eski yöntemle (hızlı anları ara) yan yana ölçer.
Geçici inceleme script'i, depoya commit edilmez.

Çalıştırma (repo kökünden):
    .venv312\\Scripts\\python tmp_yeni_yontem.py
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
from src.scoring_readiness import joint_speed, movement_segments

POSE = ROOT / "outputs/poomsae_1_zed2i_20260731_trimmed/runs/poomsae1-zed2i-rgbd-gated-ultra-rerun-20260802/json/vitpose_session_3d.json"
SPEC = ROOT / "config/scoring/poomsae/taegeuk_1_jang_v0_draft.yaml"
TIMELINE = ROOT / "config/scoring/timelines/poomsae1_zed2i_rgbd_rerun_20260802_draft.yaml"


def body_speed(keypoints: np.ndarray, fps: float) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        speeds = joint_speed(keypoints, fps=fps)
        body = [index for index in COCO_BODY_JOINT_INDICES if index < speeds.shape[1]]
        energy = np.nanmean(speeds[:, body], axis=1)
    energy = np.nan_to_num(energy, nan=0.0)
    return np.convolve(energy, np.ones(15) / 15, mode="same")


def find_holds(signal: np.ndarray, expected_count: int, min_gap: int = 20) -> list[int]:
    """Duruş anları: hızın yerel olarak dibe vurduğu kareler.

    Üç kural, üçü de uydurma sayı içermiyor:

    1. Kenarları da tara. Pencereyi dışlamak yerine kırp -- son hareketin bitiş
       duruşu kaydın sonuna yakın olduğu için, kenarları atlayan bir arama onu
       hiç göremez.
    2. Sığ dipleri at. Kaydın genel hız ortancasının üstünde kalan bir "dip"
       gerçek bir duruş değil, hareketin ortasındaki bir duraksamadır.
    3. Son `expected_count` tanesini al. Hazırlık her zaman başta olur; sporcu
       yerine geçip beklerken en derin duruşu yapar ama bu bir hareketin bitişi
       değildir.
    """
    candidates: list[int] = []
    for index in range(len(signal)):
        low, high = max(0, index - min_gap), min(len(signal), index + min_gap + 1)
        if signal[index] == signal[low:high].min():
            if not candidates or index - candidates[-1] > min_gap:
                candidates.append(index)

    median = float(np.median(signal))
    deep = [frame for frame in candidates if signal[frame] < median]
    return deep[-expected_count:]


def score(name: str, found: list[tuple[int, int]], truth: list[tuple[int, int]]) -> None:
    """Her gerçek hareket için: bulunan parça onu içeriyor mu, kaç kare sapmış?"""
    print(f"\n--- {name} ---")
    contained, errors = 0, []
    for index, (t0, t1) in enumerate(truth):
        if index >= len(found):
            print(f"  hareket {index+1}: parça yok")
            continue
        f0, f1 = found[index]
        covers = f0 <= t0 and f1 >= t1
        overlap = max(0, min(f1, t1) - max(f0, t0) + 1)
        ratio = overlap / (t1 - t0 + 1)
        contained += covers
        errors += [abs(f0 - t0), abs(f1 - t1)]
        print(
            f"  hareket {index+1}: bulunan {f0:3d}-{f1:3d} | gerçek {t0:3d}-{t1:3d} "
            f"| örtüşme %{ratio*100:5.1f} | {'tam kapsıyor' if covers else 'eksik kapsıyor'}"
        )
    if errors:
        print(f"  ortalama sınır hatası: {np.mean(errors):.1f} kare ({np.mean(errors)/60:.2f} sn)")
        print(f"  tam kapsanan hareket : {contained}/{len(truth)}")


def main() -> None:
    if not POSE.is_file():
        raise SystemExit(f"Poz dosyası bulunamadı:\n  {POSE}")

    spec = load_poomsae_spec(SPEC)
    timeline = load_movement_timeline(TIMELINE, spec)
    pose = json.loads(POSE.read_text(encoding="utf-8"))
    keypoints = np.asarray(pose["keypoints_3d_world"], dtype=float)
    fps = float(timeline["fps"])

    truth = [(item["start_frame"], item["end_frame"]) for item in timeline["segments"]]
    true_fixations = [item["anchors"]["fixation"] for item in timeline["segments"]]
    expected = len(truth)

    # ESKİ YÖNTEM: hızlı anları ara
    old = movement_segments(keypoints, fps=fps)
    old_spans = [(item["start_frame"], item["end_frame"]) for item in old if "start_frame" in item]
    print(f"ESKİ yöntem {len(old_spans)} parça buldu ({expected} hareket için)")
    score("ESKİ: hızlı anları ara", old_spans[:expected], truth)

    # YENİ YÖNTEM: duruşları bul, aralarını hareket say
    signal = body_speed(keypoints, fps)
    holds = find_holds(signal, expected_count=expected)
    print(f"\n\nYENİ yöntem {len(holds)} duruş buldu: {holds}")
    print(f"gerçek bitiş duruşları        : {true_fixations}")
    hold_error = [abs(h - f) for h, f in zip(holds, true_fixations)]
    print(f"duruş hatası: {hold_error}  ortalama {np.mean(hold_error):.1f} kare")

    # Sınır nerede? İki duruşun ortası değil -- sporcu duruşu tuttuktan sonra bir
    # süre bekliyor, sonra yeni harekete başlıyor. Sınır o başlama anı: hızın
    # ortancanın üstüne yeniden çıktığı ilk kare. Uydurma sabit yok, kaydın kendi
    # temposundan geliyor.
    median = float(np.median(signal))

    def rise_after(frame: int) -> int:
        for index in range(frame, len(signal)):
            if signal[index] > median:
                return index
        return len(signal) - 1

    mid_edges = [0] + [(holds[i] + holds[i + 1]) // 2 for i in range(len(holds) - 1)] + [len(signal) - 1]
    score(
        "YENİ-A: duruşları bul, ortadan kes",
        [(mid_edges[i], mid_edges[i + 1]) for i in range(len(mid_edges) - 1)],
        truth,
    )

    rise_edges = [0] + [rise_after(hold) for hold in holds[:-1]] + [len(signal) - 1]
    print(f"\nbulunan sınırlar: {rise_edges[1:-1]}")
    print(f"gerçek sınırlar : {[item[0] for item in truth[1:]]}")
    score(
        "YENİ-B: duruşları bul, hareketin yeniden başladığı yerden kes",
        [(rise_edges[i], rise_edges[i + 1]) for i in range(len(rise_edges) - 1)],
        truth,
    )


if __name__ == "__main__":
    main()
