"""Create an immutable source audit, blind annotation page and benchmark workspace."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path
import sys
from urllib.parse import quote
import uuid

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.artifact_contracts import load_run_bound_main_3d_artifact  # noqa: E402
from src.artifact_io import sha256_file  # noqa: E402
from src.poomsae_scoring.contracts import ScoringContractError  # noqa: E402
from src.poomsae_scoring.fixation import build_fixation_review_binding  # noqa: E402
from src.poomsae_scoring.fixation_annotation import build_independent_annotation_html  # noqa: E402
from src.poomsae_scoring.fixation_audit import attach_camera_audit, build_fixation_audit  # noqa: E402
from src.poomsae_scoring.fixation_experiments import build_fixation_sensitivity  # noqa: E402
from src.poomsae_scoring.fixation_validation import annotation_template, build_fixation_benchmark, recording_binding, validate_dataset_split  # noqa: E402
from src.poomsae_scoring.report_style import REPORT_CSS  # noqa: E402


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _verified_report(path: Path) -> dict:
    report = _read(path)
    core = {key: value for key, value in report.items() if key != "review_binding"}
    if report.get("review_binding") != build_fixation_review_binding(core):
        raise ScoringContractError(f"Report proposal binding mismatch: {path}")
    return report


def _quality_html(audit: dict, benchmarks: list[dict], sensitivity: dict) -> str:
    def display(value):
        return "—" if value is None else f"{value:.3f}"

    rows = []
    for candidate in audit["candidates"]:
        issues = []
        for name, joint in candidate["joints"].items():
            causes = ", ".join(f"{key}: {value}" for key, value in joint["rejected_frame_counts"].items() if value)
            issues.append(f"{name} ({causes})")
        camera_rows = []
        for camera, joints in candidate.get("camera_2d", {}).items():
            for joint, evidence in joints.items():
                if joint not in candidate["joints"]:
                    continue
                source_score = candidate["joints"][joint].get("source_triangulation_score", {}).get("median")
                camera_rows.append(f'<tr><td>{html.escape(camera)}</td><td>{joint}</td><td>{evidence["valid_finite_frames"]}/{evidence["expected_frames"]}</td><td>{display(evidence["score"]["median"])}</td><td>{display(source_score)}</td><td>{evidence["missing_frames"]}/{evidence["ambiguous_frames"]}</td></tr>')
        camera_detail = ('<details><summary>Reddedilen eklemlerin kamera ölçümleri</summary><p>2B dışa aktarımın görüntü/izdüşüm kökeni bu CSV ile doğrulanmaz. Yüksek 2B güven puanı tek başına güvenilir 3B ölçüm değildir.</p><table><thead><tr><th>Kamera</th><th>Eklem</th><th>Geçerli 2B kare</th><th>2B medyan güven</th><th>3B medyan kaynak kalite</th><th>Eksik/çoklu</th></tr></thead><tbody>' + ''.join(camera_rows) + '</tbody></table></details>') if camera_rows else ''
        rows.append(f'<tr><td>{candidate["candidate_episode_id"] + 1}</td><td>{candidate["window"]}</td><td>{html.escape(candidate["status"])}</td><td>{candidate["longest_common_supported_sec"]:.3f} sn</td><td>{candidate["longest_common_low_motion_sec"]:.3f} sn</td><td>{html.escape("; ".join(issues) or "Kaynak ret yok; hareket ve bileşen sürelerini inceleyin.")}{camera_detail}</td></tr>')
    tables = []
    for variant in benchmarks:
        benchmark = variant["benchmark"]
        scores = []
        for reviewer in benchmark["per_reviewer"]:
            for kind, metrics in reviewer["metrics"].items():
                for metric in metrics:
                    label = f'{reviewer["reviewer"]} / {reviewer["annotation_round"]}'
                    scores.append(f'<tr><td>{html.escape(label)}</td><td>{kind}</td><td>{metric["tolerance_ms"]:.1f} ms</td><td>{display(metric["precision"])}</td><td>{display(metric["recall"])}</td><td>{display(metric["f1"])}</td><td>{display(metric["timing_mae_ms"])} ms</td><td>{metric["false_positives"]}/{metric["false_negatives"]}</td></tr>')
        body = ''.join(scores) or '<tr><td colspan="8">Tamamlanmış bağımsız işaretleme bekleniyor; doğruluk metriği hesaplanmadı.</td></tr>'
        tables.append(f'<h3>{html.escape(variant["report_path"])}</h3><p>{html.escape(benchmark["status"])}</p><div class="metric-table-wrap"><table><thead><tr><th>İnceleyen</th><th>Olay</th><th>Tolerans</th><th>Precision</th><th>Recall</th><th>F1</th><th>Aralığa MAE</th><th>FP/FN</th></tr></thead><tbody>{body}</tbody></table></div>')
        for pair in benchmark["reviewer_agreement"]:
            title = "Kişisel tekrar tutarlılığı" if pair["comparison_kind"] == "intra_rater_repeatability" else "İnceleyenler arası uyum"
            pair_label = " ↔ ".join(f"{reviewer} / {round_id}" for reviewer, round_id in zip(pair["reviewers"], pair["annotation_rounds"], strict=True))
            pair_rows = ''.join(f'<tr><td>{kind}</td><td>{metric["tolerance_frames"]} kare</td><td>{display(metric["f1"])}</td><td>{display(metric["timing_mae_frames"])}</td><td>{metric["false_positives"]}/{metric["false_negatives"]}</td></tr>' for kind, metrics in pair["metrics"].items() for metric in metrics)
            pair_body = pair_rows or '<tr><td colspan="5">İncelenen kapsamlar aynı olmadığı için uyum hesaplanmadı.</td></tr>'
            tables.append(f'<h4>{title} · {html.escape(pair_label)}</h4><p>{html.escape(pair["status"])}</p><table><thead><tr><th>Olay</th><th>Tolerans</th><th>F1</th><th>Aralık uzaklığı/kare</th><th>Eşleşmeyen olaylar</th></tr></thead><tbody>{pair_body}</tbody></table>')
    sensitivity_rows = ''.join(f'<tr><td>{row["candidate_episode_id"] + 1}</td><td>{row["baseline_frame"] if row["baseline_frame"] is not None else "—"}</td><td>{row["supported_configuration_count"]}/{row["configuration_count"]}</td><td>{row["observed_frame_range"] or "—"}</td><td>{row["changed_configuration_count"]}</td></tr>' for row in sensitivity["candidates"])
    return f'''<!doctype html><html lang="tr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>TK3D — Sabitleme kalite incelemesi</title><style>{REPORT_CSS}td,th{{padding:.6rem}}</style></head><body><main>
<header><h1>Sabitleme kalite incelemesi</h1><p>{audit['frame_count']} kare · {audit['fps']} FPS</p><a href="blind_annotation.html">Bağımsız işaretleme ekranını aç</a></header>
<section class="section"><h2>Kaynak kanıtı</h2><p>BODY-17 geçerli oranı %{audit['body17_valid_ratio'] * 100:.1f}; bütün hareket kanallarının birlikte gözlenebilirliği %{audit['all_components_valid_ratio'] * 100:.1f}.</p><p>Kalite maskesinin ret kararı, kamera sayısı ve reprojection geçse de korunur. Tablodaki süreler hareketten çıkış adayından sonraki gözleme aittir. Sabit duruş için en az {audit['minimum_hold_sec']:.3f} sn ortak kanıt gerekir. Ayrıntılı eklem/kare ret aralıkları <a href="fixation_audit.json">JSON raporundadır</a>.</p>
<div class="metric-table-wrap"><table><thead><tr><th>Aday</th><th>Kare aralığı</th><th>Durum</th><th>En uzun ortak gözlem</th><th>En uzun ortak düşük hareket</th><th>Kaynak ret nedenleri / kare sayısı</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div></section>
<section class="section"><h2>Eşik duyarlılığı</h2><p>Aynı pose, hareket kümeleri, filtre ve kalite maskesi üzerinde {sensitivity['configuration_count']} deney yapıldı. Bekleme süresi %75/%100/%125; hareket eşikleri %80/%100/%120 oranında denendi. Destek veren deney sayısı doğruluk olasılığı değildir; kare aralığı güven aralığı değildir. Varsayılanlar değiştirilmedi.</p><table><thead><tr><th>Aday</th><th>V1 karesi</th><th>Kararlı duruş veren deney</th><th>Deneylerdeki kare aralığı</th><th>V1 kararından farklı deney</th></tr></thead><tbody>{sensitivity_rows}</tbody></table><p><a href="fixation_sensitivity.json">Deneylerin parametreleri ve tüm sonuçları</a></p></section>
<section class="section"><h2>Değerlendirme</h2><p>Tamamlanma, sabitleme başlangıcı ve bitişi ayrı olaylardır. Aynı öneri iki referansa eşleştirilemez. Kare aralıkları insan belirsizliğidir; aralığa uzaklık merkez kare hatası değildir. Belirsiz bölgeler ve eksik incelemeler ayrı raporlanır. Aynı kişinin turları kişisel tekrar tutarlılığını ölçer. Tek kayıt sonucu genellenmiş doğruluk veya otomatik puanlama onayı oluşturmaz.</p>{''.join(tables)}<p><a href="fixation_benchmark.json">Tam değerlendirme ve tur/inceleyen uyuşmazlıkları</a></p></section></main></body></html>'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path, help="Source-bound automatic segmentation JSON")
    parser.add_argument("--compare-report", action="append", default=[], type=Path, help="Another detector on the exact same recording")
    parser.add_argument("--annotation", action="append", default=[], type=Path, help="Annotation JSON; repeat for each reviewer/round")
    parser.add_argument("--keypoints-2d", type=Path, help="Optional exported 2D CSV for per-camera failure audit")
    parser.add_argument("--split-plan", type=Path, help="Optional cohort split manifest; rejects athlete/session leakage")
    parser.add_argument("--tolerance-ms", nargs="+", type=float, default=[50, 100, 200], help="Descriptive tolerances, not expert-validated acceptance limits")
    parser.add_argument("--output-dir", type=Path, help="New outputs/SESSION/runs/RUN_ID directory; never overwritten")
    args = parser.parse_args()
    paths = [path.resolve() for path in [args.report, *args.compare_report]]
    reports = [_verified_report(path) for path in paths]
    primary = reports[0]
    binding = recording_binding(primary)
    if any(recording_binding(report) != binding for report in reports):
        raise ScoringContractError("Comparison reports must use exactly the same recording/time base")
    pose_info = primary["bindings"]["pose"]
    pose_path = Path(pose_info["path"])
    if sha256_file(pose_path) != pose_info["sha256"]:
        raise ScoringContractError("Audit pose SHA-256 mismatch")
    pose, _ = load_run_bound_main_3d_artifact(pose_path)
    video_paths = {}
    for name, info in primary["bindings"]["videos"].items():
        path = Path(info["path"])
        if sha256_file(path) != info["sha256"]:
            raise ScoringContractError(f"Video SHA-256 mismatch: {name}")
        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened() or int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT))) != binding["frame_count"] or not np.isclose(capture.get(cv2.CAP_PROP_FPS), binding["fps"], rtol=1e-3):
                raise ScoringContractError(f"Video time base mismatch: {name}")
        finally:
            capture.release()
        video_paths[name] = path
    annotations = [_read(path.resolve()) for path in args.annotation]
    split_audit = {"status": "not_supplied"}
    if args.split_plan:
        manifest = _read(args.split_plan)
        split_audit = validate_dataset_split(manifest)
        current = next((row for row in manifest["recordings"] if row["recording_id"] == binding["recording_id"]), None)
        if current is None or any(a["athlete_id"] != current["athlete_id"] or a["session_id"] != current["session_id"] for a in annotations):
            raise ScoringContractError("annotation/current recording not in declared dataset split")
        split_audit["current_recording"] = current
    audit = build_fixation_audit(pose, primary)
    sensitivity = build_fixation_sensitivity(pose, primary)
    csv_path = args.keypoints_2d or pose_path.parent.parent / "csv" / "vitpose_keypoints_2d_flat.csv"
    if args.keypoints_2d or csv_path.is_file():
        csv_path = csv_path.resolve()
        attach_camera_audit(audit, csv_path, list(video_paths))
        audit["exported_2d_source"] = {"path": str(csv_path), "sha256": sha256_file(csv_path)}
    benchmarks = [{"report_path": str(path), "report_sha256": sha256_file(path),
                   "benchmark": build_fixation_benchmark(report, annotations, args.tolerance_ms)}
                  for path, report in zip(paths, reports, strict=True)]
    for variant in benchmarks:
        variant["benchmark"]["dataset_split"] = split_audit
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    output = args.output_dir or ROOT / "outputs" / pose["session_id"] / "runs" / f"fixation-quality-{stamp}-{uuid.uuid4().hex[:8]}"
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    state = {"schema_version": 1, "run_id": output.name, "status": "running"}
    _write(output / "run_state.json", state)
    try:
        _write(output / "fixation_audit.json", audit)
        _write(output / "fixation_sensitivity.json", sensitivity)
        _write(output / "fixation_benchmark.json", {"schema_version": 1, "variants": benchmarks})
        _write(output / "annotation_template.json", annotation_template(binding))
        _write(output / "source_report.json", primary)
        _write(output / "provenance.json", {"reports": [{"path": str(path), "sha256": sha256_file(path)} for path in paths],
                                           "annotations": [{"path": str(path.resolve()), "sha256": sha256_file(path)} for path in args.annotation],
                                           "recording_binding": binding, "pose": pose_info,
                                           "split_plan": {"path": str(args.split_plan.resolve()), "sha256": sha256_file(args.split_plan)} if args.split_plan else None})
        relative = {name: quote(Path(os.path.relpath(path, output)).as_posix(), safe="/") for name, path in video_paths.items()}
        (output / "blind_annotation.html").write_text(build_independent_annotation_html(binding, relative), encoding="utf-8")
        (output / "quality_report.html").write_text(_quality_html(audit, benchmarks, sensitivity), encoding="utf-8")
        state["status"] = "completed"
    except Exception:
        state["status"] = "failed"
        raise
    finally:
        (output / "run_state.json").write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    print(output / "quality_report.html")
    print(output / "blind_annotation.html")
    print(output / "fixation_benchmark.json")


if __name__ == "__main__":
    main()
