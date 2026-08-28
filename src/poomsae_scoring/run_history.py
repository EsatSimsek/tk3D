"""Comparable run history and fail-closed regression diagnostics."""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from math import isfinite
from typing import Any

from src.poomsae_scoring.contracts import ScoringContractError


_METRICS = (
    ("wholebody_measurement_coverage", "WholeBody ölçüm kapsamı"),
    ("technical_criterion_coverage", "Teknik ölçüt kapsamı"),
    ("not_measurable_count", "Ölçülemeyen Accuracy kararı"),
    ("boundary_uncertain_count", "Sınır-belirsiz Accuracy kararı"),
    ("diagnostic_review_candidate_count", "WholeBody inceleme adayı"),
    ("categorical_mismatch_candidate_count", "Hareket/duruş uyuşmazlığı adayı"),
    ("technical_conformance_review_required_count", "Teknik inceleme gereken hareket"),
    ("observed_scope_provisional_deduction_total", "Gözlenen kapsam provisional kesintisi"),
)


def build_run_history(
    current_summary: dict[str, Any],
    prior_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build history from workflow summaries without inventing a quality score."""
    current = _entry(current_summary, is_current=True)
    prior_entries = []
    skipped = 0
    for record in prior_records:
        payload = record.get("summary")
        if not isinstance(payload, dict):
            skipped += 1
            continue
        try:
            entry = _entry(payload, is_current=False)
        except ScoringContractError:
            skipped += 1
            continue
        entry["summary_path"] = record.get("summary_path")
        entry["modified_at_utc"] = record.get("modified_at_utc")
        prior_entries.append(entry)

    compatible = [item for item in prior_entries if _same_comparison_scope(current, item)]
    compatible.sort(key=lambda item: (str(item.get("modified_at_utc") or ""), item["run_id"]))
    baseline = compatible[-1] if compatible else None
    comparison = _compare(current, baseline) if baseline is not None else None
    all_entries = [current, *reversed(compatible)]
    return {
        "schema_version": 1,
        "status": "run_history_diagnostic_only",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "current_run_id": current["run_id"],
        "comparison_contract": {
            "same_profile_required": True,
            "same_selected_scope_required": True,
            "same_workflow_mode_required": True,
            "same_pose_required_for_regression_alerts": True,
            "candidate_count_change_is_not_quality_change": True,
            "automatic_quality_ranking_allowed": False,
        },
        "summary": {
            "discovered_prior_count": len(prior_records),
            "compatible_prior_count": len(compatible),
            "incompatible_prior_count": len(prior_entries) - len(compatible),
            "skipped_prior_count": skipped,
            "baseline_run_id": None if baseline is None else baseline["run_id"],
            "alert_count": 0 if comparison is None else len(comparison["alerts"]),
        },
        "current": current,
        "baseline": baseline,
        "comparison": comparison,
        "entries": all_entries,
        "interpretation": (
            "Bu rapor koşular arasında resmî doğruluk veya performans sıralaması yapmaz. "
            "Yalnız aynı profil, kapsam ve çalışma modundaki çıktıları yan yana getirir; "
            "regresyon uyarısı için ayrıca aynı pose SHA-256 bağını ister."
        ),
    }


def build_run_history_html(report: dict[str, Any]) -> str:
    if report.get("status") != "run_history_diagnostic_only":
        raise ScoringContractError("run history report status is invalid")
    current = report["current"]
    baseline = report.get("baseline")
    comparison = report.get("comparison")
    alerts = [] if comparison is None else comparison.get("alerts", [])
    alert_rows = "".join(
        f'<li><code>{_escape(item["code"])}</code><span>{_escape(item["message"])}</span></li>'
        for item in alerts
    ) or "<li><span>Aynı pose bağlı karşılaştırmada regresyon uyarısı yok.</span></li>"
    metric_rows = "".join(
        _metric_row(metric_id, label, current, baseline, comparison)
        for metric_id, label in _METRICS
    )
    history_rows = "".join(_history_row(item) for item in report.get("entries", []))
    baseline_label = "Yok" if baseline is None else baseline["run_id"]
    comparison_kind = (
        "Karşılaştırma yok"
        if comparison is None
        else {
            "same_pose_regression": "Aynı pose regresyon kontrolü",
            "different_pose_context_only": "Farklı pose · yalnız bağlam",
        }.get(comparison["comparison_kind"], comparison["comparison_kind"])
    )
    return f'''<!doctype html>
<html lang="tr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TK3D Koşu Geçmişi</title><style>
:root{{color-scheme:dark;--bg:#071019;--panel:#10202d;--line:#294151;--text:#edf7ff;--muted:#9eb3c2;--cyan:#46d7e8;--green:#64e5a5;--amber:#ffca6a;--red:#ff7d7d}}
*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(circle at 80% 0,#14354a 0,transparent 35%),var(--bg);color:var(--text);font-family:Inter,Segoe UI,system-ui,sans-serif}}
main{{width:min(1280px,95vw);margin:auto;padding:30px 0 60px}}h1{{font-size:clamp(28px,4vw,46px);margin:5px 0}}h2{{margin:0 0 14px}}p{{color:var(--muted);line-height:1.55}}
.eyebrow{{color:var(--cyan);font-weight:800;letter-spacing:.12em;text-transform:uppercase;font-size:12px}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:20px 0}}
.card,.section{{border:1px solid var(--line);background:linear-gradient(145deg,#142838,#0c1924);border-radius:16px;padding:18px}}.card span{{color:var(--muted);font-size:12px}}.card b{{display:block;font-size:22px;margin-top:6px}}
.section{{margin-bottom:18px}}table{{width:100%;border-collapse:collapse}}th,td{{text-align:left;padding:10px;border-bottom:1px solid #213643}}th{{color:var(--cyan)}}td{{color:var(--muted)}}code{{color:var(--amber)}}ul{{list-style:none;padding:0}}li{{display:flex;gap:10px;padding:9px 0;border-bottom:1px solid #213643;color:var(--muted)}}
.alert{{border-color:#885e31;background:#2b2113}}.delta-pos{{color:var(--green)}}.delta-neg{{color:var(--red)}}.delta-neutral{{color:var(--muted)}}
@media(max-width:760px){{.grid{{grid-template-columns:1fr}}.table-wrap{{overflow:auto}}}}
</style></head><body><main><a href="poomsae_scoring_review.html">← İnceleme ekranına dön</a><div class="eyebrow" style="margin-top:18px">TK3D · regresyon teşhisi</div><h1>Koşu geçmişi ve karşılaştırma</h1>
<p>{_escape(report.get("interpretation"))}</p><div class="grid">
<div class="card"><span>Güncel koşu</span><b>{_escape(current["run_id"])}</b></div>
<div class="card"><span>Karşılaştırılan koşu</span><b>{_escape(baseline_label)}</b></div>
<div class="card"><span>Karşılaştırma türü</span><b>{_escape(comparison_kind)}</b></div></div>
<section class="section {'alert' if alerts else ''}"><h2>Regresyon uyarıları</h2><ul>{alert_rows}</ul></section>
<section class="section"><h2>Güncel ve önceki ölçümler</h2><div class="table-wrap"><table><thead><tr><th>Ölçüm</th><th>Güncel</th><th>Önceki</th><th>Fark</th></tr></thead><tbody>{metric_rows}</tbody></table></div></section>
<section class="section"><h2>Uyumlu koşu geçmişi</h2><div class="table-wrap"><table><thead><tr><th>Koşu</th><th>WholeBody kapsamı</th><th>Teknik kapsam</th><th>İnceleme gereken hareket</th><th>Rule ready</th></tr></thead><tbody>{history_rows}</tbody></table></div></section>
</main></body></html>'''


def _entry(summary: dict[str, Any], *, is_current: bool) -> dict[str, Any]:
    if summary.get("workflow") != "tk3d_source_bound_poomsae_scoring_v1":
        raise ScoringContractError("run history requires a TK3D scoring workflow summary")
    run = summary.get("run") or {}
    coverage = summary.get("coverage") or {}
    results = summary.get("results") or {}
    bindings = summary.get("bindings") or {}
    run_id = run.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ScoringContractError("workflow summary run_id is missing")
    metrics = {
        "wholebody_measurement_coverage": _ratio(
            results.get("wholebody_measurable_metric_count"),
            results.get("wholebody_thresholded_metric_count"),
        ),
        "technical_criterion_coverage": _ratio(
            results.get("technical_conformance_measurable_criterion_count"),
            results.get("technical_conformance_expected_criterion_count"),
        ),
        "not_measurable_count": _number(results.get("not_measurable_count")),
        "boundary_uncertain_count": _number(results.get("boundary_uncertain_count")),
        "diagnostic_review_candidate_count": _number(
            results.get("diagnostic_review_candidate_count")
        ),
        "categorical_mismatch_candidate_count": _number(
            results.get("categorical_mismatch_candidate_count")
        ),
        "technical_conformance_review_required_count": _number(
            results.get("technical_conformance_review_required_count")
        ),
        "observed_scope_provisional_deduction_total": _number(
            results.get("observed_scope_provisional_deduction_total")
        ),
    }
    return {
        "run_id": run_id,
        "run_root": run.get("root"),
        "is_current": is_current,
        "profile_id": summary.get("profile_id"),
        "mode": summary.get("mode"),
        "selected_scope_id": coverage.get("selected_scope_id"),
        "pose_sha256": (bindings.get("pose") or {}).get("sha256"),
        "status": summary.get("status"),
        "rule_scoring_ready": bool(results.get("rule_scoring_ready", False)),
        "metrics": metrics,
    }


def _compare(current: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    same_pose = current.get("pose_sha256") == baseline.get("pose_sha256")
    deltas = {
        metric_id: _delta(current["metrics"].get(metric_id), baseline["metrics"].get(metric_id))
        for metric_id, _ in _METRICS
    }
    alerts: list[dict[str, str]] = []
    if same_pose:
        wholebody_delta = deltas["wholebody_measurement_coverage"]
        technical_delta = deltas["technical_criterion_coverage"]
        missing_delta = deltas["not_measurable_count"]
        if wholebody_delta is not None and wholebody_delta < -1e-9:
            alerts.append(
                {
                    "code": "wholebody_measurement_coverage_decreased",
                    "message": "Aynı pose için WholeBody ölçüm kapsamı azaldı.",
                }
            )
        if technical_delta is not None and technical_delta < -1e-9:
            alerts.append(
                {
                    "code": "technical_criterion_coverage_decreased",
                    "message": "Aynı pose için teknik ölçüt kapsamı azaldı.",
                }
            )
        if missing_delta is not None and missing_delta > 1e-9:
            alerts.append(
                {
                    "code": "not_measurable_count_increased",
                    "message": "Aynı pose için ölçülemeyen Accuracy kararı arttı.",
                }
            )
        if baseline["rule_scoring_ready"] and not current["rule_scoring_ready"]:
            alerts.append(
                {
                    "code": "rule_scoring_readiness_regressed",
                    "message": "Rule scoring readiness true değerinden false değerine geriledi.",
                }
            )
    return {
        "baseline_run_id": baseline["run_id"],
        "comparison_kind": "same_pose_regression" if same_pose else "different_pose_context_only",
        "same_pose_sha256": same_pose,
        "metric_deltas": deltas,
        "alerts": alerts,
    }


def _same_comparison_scope(current: dict[str, Any], candidate: dict[str, Any]) -> bool:
    return all(
        current.get(key) == candidate.get(key)
        for key in ("profile_id", "selected_scope_id", "mode")
    )


def _metric_row(
    metric_id: str,
    label: str,
    current: dict[str, Any],
    baseline: dict[str, Any] | None,
    comparison: dict[str, Any] | None,
) -> str:
    current_value = current["metrics"].get(metric_id)
    baseline_value = None if baseline is None else baseline["metrics"].get(metric_id)
    delta = None if comparison is None else comparison["metric_deltas"].get(metric_id)
    delta_class = "delta-neutral"
    is_ratio = metric_id.endswith("coverage")
    return (
        f"<tr><td>{_escape(label)}</td><td>{_format_value(current_value, is_ratio)}</td>"
        f"<td>{_format_value(baseline_value, is_ratio)}</td>"
        f'<td class="{delta_class}">{_format_delta(delta, is_ratio)}</td></tr>'
    )


def _history_row(entry: dict[str, Any]) -> str:
    metrics = entry["metrics"]
    return (
        f"<tr><td><code>{_escape(entry['run_id'])}</code></td>"
        f"<td>{_format_value(metrics.get('wholebody_measurement_coverage'), True)}</td>"
        f"<td>{_format_value(metrics.get('technical_criterion_coverage'), True)}</td>"
        f"<td>{_format_value(metrics.get('technical_conformance_review_required_count'), False)}</td>"
        f"<td>{str(entry['rule_scoring_ready']).lower()}</td></tr>"
    )


def _ratio(numerator: Any, denominator: Any) -> float | None:
    top = _number(numerator)
    bottom = _number(denominator)
    if top is None or bottom is None or bottom <= 0:
        return None
    return top / bottom


def _number(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not isfinite(number):
        raise ScoringContractError("run history metric must be finite or null")
    return number


def _delta(current: float | None, baseline: float | None) -> float | None:
    if current is None or baseline is None:
        return None
    return current - baseline


def _format_value(value: Any, is_ratio: bool) -> str:
    if value is None:
        return "—"
    return f"%{float(value) * 100:.1f}" if is_ratio else f"{float(value):.2f}"


def _format_delta(value: Any, is_ratio: bool) -> str:
    if value is None:
        return "—"
    prefix = "+" if float(value) > 0 else ""
    return f"{prefix}{float(value) * 100:.1f} puan" if is_ratio else f"{prefix}{float(value):.2f}"


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def dump_report_json(report: dict[str, Any]) -> str:
    """Serialize with the same strict non-finite contract as workflow artifacts."""
    return json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
