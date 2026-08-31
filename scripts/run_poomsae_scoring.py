from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from src.poomsae_scoring import application as _application

WorkflowError = _application.WorkflowError
run_workflow = _application.run_workflow
run_poomsae_analysis = _application.run_poomsae_analysis
_display = _application._display
_read_json = _application._read_json

__all__ = ["WorkflowError", "main", "run_workflow"]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run Poomsae analysis, diagnostics and provisional source-bound decisions. "
            "By default it analyzes the profile's verified 3D pose; --process-video first "
            "re-runs ViTPose/RGBD on the exact bound session. This is not official scoring."
        )
    )
    parser.add_argument(
        "--profile",
        default="poomsae1_trimmed",
        help="Profile id under config/scoring/profiles or an explicit YAML path (default: poomsae1_trimmed).",
    )
    parser.add_argument(
        "--process-video",
        action="store_true",
        help="Run full stride-1 ViTPose/RGBD processing before the source-bound analysis.",
    )
    parser.add_argument("--run-id", help="Optional unique output run id.")
    parser.add_argument(
        "--profile-performance",
        action="store_true",
        help="Write a separate performance_report.json without changing scoring artifacts.",
    )
    args = parser.parse_args()
    try:
        result = run_poomsae_analysis(
            profile_value=args.profile,
            process_video=args.process_video,
            requested_run_id=args.run_id,
            profile_performance=args.profile_performance,
        )
    except (WorkflowError, OSError, ValueError, yaml.YAMLError, json.JSONDecodeError) as exc:
        parser.exit(1, f"TK3D Poomsae analysis workflow failed: {exc}\n")
    _print_summary(result.summary, result.summary_path)
    return 0


def _print_summary(summary: dict[str, Any], summary_path: Path) -> None:
    results = summary["results"]
    coverage = summary["coverage"]
    print("\n=== TK3D POOMSAE ANALİZ SONUCU ===")
    print(f"Durum: {summary['status']}")
    print(
        f"Aktif {coverage['selected_scope_label']} çalışma kapsamı: "
        f"{coverage['selected_scope_observed_count']}/{coverage['selected_scope_expected_count']}"
    )
    print(
        "Tam Poomsae kapsam bilgisi: "
        f"{coverage['observed_movement_count']}/{coverage['expected_movement_count']} "
        "(bu çalışma aşamasında hedef değil)"
    )
    print(f"Tam Accuracy skoru: {_display(results['accuracy_score'])}")
    print(
        "Gözlenen kapsam provisional kesintisi: "
        f"{_display(results['observed_scope_provisional_deduction_total'])}"
    )
    print(f"Küçük hata sayısı: {results['confirmed_numeric_minor_count']}")
    print(f"Ölçülemeyen karar: {results['not_measurable_count']}")
    print(f"Sınır-belirsiz karar: {results['boundary_uncertain_count']}")
    print(
        "Otomatik hareket/faz önerisi: "
        f"{results['automatic_segmentation_selected_count']}/"
        f"{results['automatic_segmentation_expected_count']} · "
        f"faz MAE {results['automatic_segmentation_phase_anchor_mae_frames']} kare"
    )
    print(
        "WholeBody ölçüm kapsamı: "
        f"{results['wholebody_measurable_metric_count']}/{results['wholebody_thresholded_metric_count']}"
    )
    print(f"WholeBody teşhis adayı (puan yok): {results['diagnostic_review_candidate_count']}")
    print(
        "Kapsamlı teknik-doğruluk envanteri/adayı (puan yok): "
        f"{results['technical_accuracy_rule_count']} kural / "
        f"{results['technical_accuracy_temporary_candidate_count']} aday / "
        f"{results['technical_accuracy_score_effect_count']} skor etkisi"
    )
    print(
        "Otomatik duraklama gözlemi (puan yetkisi yok): "
        f"{results['categorical_pause_observation_count']}"
    )
    print(f"Yanlış hareket/duruş inceleme adayı (puan yok): {results['categorical_mismatch_candidate_count']}")
    print(
        "Teknik uygunluk hareket incelemesi (puan yok): "
        f"{results['technical_conformance_review_required_count']}/"
        f"{results['technical_conformance_movement_count']}"
    )
    print(
        "Presentation ölçülebilir proxy: "
        f"{results['presentation_measurable_proxy_count']}/"
        f"{results['presentation_requested_proxy_count']} (puan iddiası kapalı)"
    )
    print(f"Rule scoring ready: {str(results['rule_scoring_ready']).lower()}")
    print(f"Çıktı klasörü: {summary['run']['root']}")
    print(f"Ana özet: {summary_path}")
    print(f"İnceleme ekranı: {summary['outputs']['review_html']}")
    print(f"Koşu geçmişi: {summary['outputs']['run_history_html']}")


if __name__ == "__main__":
    raise SystemExit(main())
