from __future__ import annotations

import html
import json
from typing import Any

from src.poomsae_scoring.contracts import (
    ScoringContractError,
    validate_movement_timeline,
    validate_poomsae_spec,
)


def build_review_html(
    poomsae_spec: dict[str, Any],
    movement_timeline: dict[str, Any],
    evidence_report: dict[str, Any],
    readiness_report: dict[str, Any],
    video_sources: dict[str, str],
    engineering_trial_report: dict[str, Any] | None = None,
    wholebody_diagnostics_report: dict[str, Any] | None = None,
    accuracy_decisions_report: dict[str, Any] | None = None,
    decision_evidence_report: dict[str, Any] | None = None,
) -> str:
    """Build a self-contained, synchronized two-camera review page."""
    spec = validate_poomsae_spec(poomsae_spec)
    timeline = validate_movement_timeline(movement_timeline, spec)
    if len(video_sources) < 2 or any(not label.strip() or not source.strip() for label, source in video_sources.items()):
        raise ScoringContractError("review report requires at least two labeled video sources")
    if evidence_report.get("timeline", {}).get("timeline_id") != timeline["timeline_id"]:
        raise ScoringContractError("evidence report must reference the same MovementTimeline")
    readiness_timeline = readiness_report.get("movement_timeline", {})
    if readiness_timeline.get("timeline_id") != timeline["timeline_id"]:
        raise ScoringContractError("readiness report must reference the same MovementTimeline")
    if engineering_trial_report is not None and engineering_trial_report.get("movement_timeline_id") != timeline["timeline_id"]:
        raise ScoringContractError("engineering trial must reference the same MovementTimeline")
    if wholebody_diagnostics_report is not None and wholebody_diagnostics_report.get("movement_timeline_id") != timeline["timeline_id"]:
        raise ScoringContractError("WholeBody diagnostics must reference the same MovementTimeline")
    if accuracy_decisions_report is not None and accuracy_decisions_report.get("timeline_id") != timeline["timeline_id"]:
        raise ScoringContractError("Accuracy decisions must reference the same MovementTimeline")
    if decision_evidence_report is not None and decision_evidence_report.get("timeline_id") != timeline["timeline_id"]:
        raise ScoringContractError("decision evidence must reference the same MovementTimeline")

    movement_by_id = {movement["movement_id"]: movement for movement in spec["movements"]}
    evidence_by_id = {segment["movement_id"]: segment for segment in evidence_report.get("segments", [])}
    observed = timeline["coverage"]["observed_movement_ids"]
    missing = timeline["coverage"]["missing_movement_ids"]
    observability = evidence_report.get("observability", {})
    anchor_count = int(observability.get("anchor_count", 0))
    observed_ratio = float(observability.get("observed_anchor_ratio", 0.0))

    video_html = "".join(
        f'''<article class="video-card">
          <div class="video-label"><span class="camera-dot"></span>{_escape(label)}</div>
          <video controls preload="metadata" playsinline src="{_escape(source)}"></video>
        </article>'''
        for label, source in video_sources.items()
    )
    movement_cards = "".join(
        _movement_card(segment, movement_by_id[segment["movement_id"]], evidence_by_id.get(segment["movement_id"]), timeline["fps"])
        for segment in timeline["segments"]
    )
    missing_cards = "".join(
        f'<span class="missing-chip">{_escape(movement_id)} · {_escape(movement_by_id[movement_id]["display_name"])}</span>'
        for movement_id in missing
    )
    blockers = readiness_report.get("blockers", [])
    blocker_html = "".join(
        f'<li><code>{_escape(item.get("code", "unknown"))}</code><span>{_escape(item.get("message", ""))}</span></li>'
        for item in blockers
    ) or "<li><span>Aktif engel raporlanmadı.</span></li>"
    sources = "".join(
        f'<a href="{_escape(source["url"])}" target="_blank" rel="noreferrer">{_escape(source["title"])}</a>'
        for source in spec["source_documents"]
    )
    page_data = {
        "timeline_id": timeline["timeline_id"],
        "fps": timeline["fps"],
        "segments": [
            {
                "movement_id": segment["movement_id"],
                "start": segment["start_frame"] / timeline["fps"],
                "end": segment["end_frame"] / timeline["fps"],
            }
            for segment in timeline["segments"]
        ],
    }
    safe_data = json.dumps(page_data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    source_end_reason = timeline["coverage"]["source_end_reason"] or "Tam performans kaydı"
    trial_stat, trial_section = _engineering_trial_html(engineering_trial_report)
    wholebody_stat, wholebody_section = _wholebody_diagnostics_html(wholebody_diagnostics_report)
    decision_stat, decision_section = _decision_evidence_html(
        accuracy_decisions_report,
        decision_evidence_report,
    )
    partial_total = (
        None
        if accuracy_decisions_report is None
        else accuracy_decisions_report.get("observed_scope_provisional_deduction_total")
    )
    decision_notice = (
        "Kaynak-bağlı Accuracy kararı sağlanmadı."
        if accuracy_decisions_report is None
        else (
            f"Gözlenen kapsam için {_number(partial_total, '')} provisional kesinti üretildi; "
            "kayıt kısmi olduğu için tam Accuracy skoru hesaplanmadı."
        )
    )

    return f'''<!doctype html>
<html lang="tr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Taegeuk 1 Kısa Kayıt İncelemesi</title>
  <style>
    :root {{ color-scheme: dark; --bg:#071019; --panel:#101d29; --panel2:#142534; --line:#294151;
      --text:#edf7ff; --muted:#9eb3c2; --cyan:#46d7e8; --green:#64e5a5; --amber:#ffca6a; --red:#ff7d7d; }}
    * {{ box-sizing:border-box; }} body {{ margin:0; background:radial-gradient(circle at 80% 0,#14354a 0,transparent 35%),var(--bg);
      color:var(--text); font-family:Inter,Segoe UI,system-ui,sans-serif; }}
    main {{ width:min(1480px,96vw); margin:0 auto; padding:30px 0 60px; }}
    header {{ display:flex; justify-content:space-between; gap:20px; align-items:flex-start; margin-bottom:20px; }}
    h1 {{ margin:5px 0 7px; font-size:clamp(26px,3vw,44px); letter-spacing:-.04em; }}
    h2 {{ font-size:20px; margin:0 0 14px; }} p {{ color:var(--muted); margin:0; line-height:1.55; }}
    .eyebrow {{ color:var(--cyan); font-weight:750; letter-spacing:.13em; text-transform:uppercase; font-size:12px; }}
    .pill {{ border:1px solid #9c762c; color:var(--amber); background:#2d2414; border-radius:999px; padding:9px 13px; white-space:nowrap; font-weight:700; }}
    .notice {{ display:flex; gap:13px; border:1px solid #855e28; background:linear-gradient(90deg,#2b2113,#1a1d1e); padding:16px 18px; border-radius:14px; margin-bottom:18px; }}
    .notice strong {{ color:var(--amber); display:block; margin-bottom:3px; }}
    .stats {{ display:grid; grid-template-columns:repeat(6,1fr); gap:12px; margin-bottom:18px; }}
    .stat,.section {{ border:1px solid var(--line); background:linear-gradient(145deg,rgba(20,37,52,.94),rgba(12,25,36,.94)); border-radius:16px; }}
    .stat {{ padding:16px; }} .stat b {{ display:block; font-size:25px; margin-top:5px; }} .stat span {{ color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.09em; }}
    .section {{ padding:18px; margin-bottom:18px; }} .videos {{ display:grid; grid-template-columns:repeat({len(video_sources)},minmax(0,1fr)); gap:12px; }}
    .video-card {{ border-radius:12px; overflow:hidden; background:#03070a; border:1px solid var(--line); }} video {{ width:100%; display:block; aspect-ratio:16/9; background:#000; }}
    .video-label {{ padding:10px 12px; color:#d9edf8; font-size:13px; font-weight:700; }} .camera-dot {{ display:inline-block; width:8px; height:8px; background:var(--green); border-radius:50%; margin-right:8px; box-shadow:0 0 10px var(--green); }}
    .toolbar {{ display:flex; align-items:center; gap:10px; margin-top:13px; flex-wrap:wrap; }} button {{ color:var(--text); background:#183247; border:1px solid #37627b; border-radius:9px; padding:9px 13px; cursor:pointer; font-weight:700; }} button:hover {{ background:#21445e; }}
    #clock {{ color:var(--cyan); font-variant-numeric:tabular-nums; font-weight:750; }}
    .movement-grid {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:10px; }}
    .movement {{ text-align:left; padding:14px; border:1px solid var(--line); border-radius:12px; background:#0b1822; cursor:pointer; transition:.15s ease; }}
    .movement:hover,.movement.active {{ border-color:var(--cyan); transform:translateY(-1px); background:#102839; }}
    .movement-top {{ display:flex; justify-content:space-between; gap:8px; }} .movement-id {{ color:var(--cyan); font-weight:800; }} .movement-name {{ font-weight:700; margin:6px 0 10px; }}
    .metrics {{ display:grid; grid-template-columns:1fr 1fr; gap:5px 10px; color:var(--muted); font-size:12px; }} .metrics b {{ color:#d8e9f2; font-weight:650; }}
    .anchors {{ display:flex; flex-wrap:wrap; gap:5px; margin-top:10px; }} .anchor {{ padding:5px 7px; font-size:11px; border-color:#355466; background:#10222f; }}
    .two-col {{ display:grid; grid-template-columns:1.1fr .9fr; gap:18px; }} .missing-list {{ display:flex; flex-wrap:wrap; gap:7px; }}
    .missing-chip {{ color:#c6d3db; background:#17222b; border:1px dashed #4c5d68; border-radius:999px; padding:7px 10px; font-size:12px; }}
    ul {{ padding:0; list-style:none; margin:0; }} li {{ display:flex; gap:10px; align-items:flex-start; padding:8px 0; border-bottom:1px solid #213643; color:var(--muted); }} li:last-child {{ border-bottom:0; }} code {{ color:var(--amber); }}
    .candidate-row {{ display:grid; grid-template-columns:minmax(120px,.35fr) 1fr auto; align-items:center; }} .candidate-row button {{ padding:6px 9px; white-space:nowrap; }}
    .metric-table-wrap {{ overflow:auto; margin-top:16px; }} .metric-table {{ width:100%; border-collapse:collapse; font-size:13px; }}
    .metric-table th,.metric-table td {{ padding:9px 10px; border-bottom:1px solid #213643; text-align:left; white-space:nowrap; }}
    .metric-table th {{ color:var(--cyan); background:#0a1721; position:sticky; top:0; }} .metric-table td {{ color:var(--muted); }}
    .metric-table .review {{ color:var(--amber); font-weight:700; }} .metric-table .ok {{ color:var(--green); }} .metric-table .missing {{ color:#9aa8b2; }}
    .decision-row {{ display:grid; grid-template-columns:150px minmax(0,1fr) auto; gap:12px; align-items:center; }}
    .decision-row.red {{ border-left:4px solid var(--red); padding-left:10px; }} .decision-row.amber {{ border-left:4px solid var(--amber); padding-left:10px; }}
    .decision-row.gray {{ border-left:4px solid #8998a3; padding-left:10px; }} .decision-row.green {{ border-left:4px solid var(--green); padding-left:10px; }}
    .review-actions {{ display:flex; gap:5px; flex-wrap:wrap; }} .review-actions button.selected {{ outline:2px solid var(--cyan); background:#285575; }}
    .sources {{ display:flex; flex-wrap:wrap; gap:8px; margin-top:12px; }} a {{ color:var(--cyan); }}
    .wb-movement-block {{ margin-bottom:14px; border:1px solid var(--line); border-radius:12px; background:#0b1822; padding:14px; }}
    .wb-movement-hdr {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:10px; cursor:pointer; user-select:none; }}
    .wb-movement-hdr:hover {{ opacity:.85; }}
    .wb-mid {{ color:var(--cyan); font-weight:800; font-size:15px; }} .wb-mname {{ font-weight:700; font-size:13px; color:var(--text); margin-left:8px; }}
    .wb-badge {{ font-size:11px; padding:3px 8px; border-radius:999px; font-weight:700; }}
    .wb-badge-ok {{ background:#123526; color:var(--green); border:1px solid #2a6b48; }}
    .wb-badge-warn {{ background:#2d2414; color:var(--amber); border:1px solid #9c762c; }}
    .wb-metrics-grid {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(300px,1fr)); gap:6px; }}
    .wb-m {{ display:grid; grid-template-columns:22px 1fr auto; gap:2px 8px; align-items:center; padding:7px 10px; border-radius:8px; font-size:12px; border:1px solid var(--line); }}
    .wb-m-name {{ color:var(--muted); font-size:11px; letter-spacing:.02em; }}
    .wb-m-val {{ font-weight:700; color:var(--text); font-size:13px; font-variant-numeric:tabular-nums; }}
    .wb-m-range {{ color:var(--muted); font-size:10px; grid-column:2/4; margin-top:-1px; }}
    .wb-ok {{ border-left:3px solid var(--green); }} .wb-ok .wb-dot {{ color:var(--green); }}
    .wb-cand {{ border-left:3px solid var(--red); background:#1a1018; }} .wb-cand .wb-dot {{ color:var(--red); }}
    .wb-nm {{ border-left:3px solid #5a6a75; opacity:.6; }} .wb-nm .wb-dot {{ color:#5a6a75; }}
    .wb-diag {{ border-left:3px solid var(--cyan); opacity:.75; }} .wb-diag .wb-dot {{ color:var(--cyan); }}
    .wb-collapse {{ display:none; }} .wb-movement-block.open .wb-collapse {{ display:grid; }}
    footer {{ color:#6f8795; font-size:12px; text-align:center; margin-top:20px; }}
    @media(max-width:950px) {{ .stats {{ grid-template-columns:1fr 1fr; }} .videos,.two-col {{ grid-template-columns:1fr; }} .movement-grid {{ grid-template-columns:1fr 1fr; }} }}
    @media(max-width:580px) {{ header {{ display:block; }} .pill {{ display:inline-block; margin-top:12px; }} .movement-grid,.stats {{ grid-template-columns:1fr; }} .candidate-row {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body><main>
  <header><div><div class="eyebrow">TK3D · Ölçüm kanıtı</div><h1>Taegeuk 1 Kısa Kayıt İncelemesi</h1>
    <p>{len(video_sources)} kamera, aynı zaman çizelgesi ve hareket/faz ölçümleri tek ekranda.</p></div>
    <div class="pill">Kısmi kayıt · {len(observed)}/{len(spec["movements"])}</div></header>
  <section class="notice"><div>⚠</div><div><strong>Bu çıktı tam veya resmî puan değildir.</strong><p>{_escape(decision_notice)} {_escape(source_end_reason)}</p></div></section>
  <section class="stats">
    <div class="stat"><span>Kayıttaki hareket</span><b>{len(observed)} / {len(spec["movements"])}</b></div>
    <div class="stat"><span>Ölçülen faz ankrajı</span><b>{anchor_count}</b></div>
    <div class="stat"><span>Tam gözlenen ankraj</span><b>%{observed_ratio * 100:.1f}</b></div>
    {decision_stat}
    {wholebody_stat}
    {trial_stat}
  </section>
  <section class="section"><h2>Senkron kamera incelemesi</h2><div class="videos">{video_html}</div>
    <div class="toolbar"><button type="button" id="sync-play">▶ İkisini oynat</button><button type="button" id="sync-pause">Ⅱ Duraklat</button><button type="button" id="sync-zero">↺ Başa dön</button><span id="clock">00:00.000</span><p>Bir videoda sarınca diğeri aynı zamana gelir.</p></div>
  </section>
  <section class="section"><h2>Kayıtta bulunan hareketler</h2><div class="movement-grid">{movement_cards}</div></section>
  {trial_section}
  {wholebody_section}
  {decision_section}
  <div class="two-col">
    <section class="section"><h2>Kayıtta bulunmayan hareketler</h2><p style="margin-bottom:12px">Bunlar etiketleme hatası değildir; kaynak video M06 sonrasında devam etmiyor.</p><div class="missing-list">{missing_cards}</div></section>
    <section class="section"><h2>Puanlamayı kapalı tutan kapılar</h2><ul>{blocker_html}</ul></section>
  </div>
  <section class="section"><h2>Kaynaklar ve yorum sınırı</h2><p>{_escape(evidence_report.get("interpretation", ""))}</p><div class="sources">{sources}</div></section>
  <footer>Timeline: {_escape(timeline["timeline_id"])} · Pose SHA-256 doğrulaması: {_escape(readiness_report.get("pose_binding", {}).get("status", "unknown"))}</footer>
</main>
<script id="review-data" type="application/json">{safe_data}</script>
<script>
(() => {{
  const data = JSON.parse(document.getElementById('review-data').textContent);
  const videos = [...document.querySelectorAll('video')];
  const cards = [...document.querySelectorAll('.movement')];
  const reviewKey = `tk3d-review-${{data.timeline_id}}`;
  const reviewSelections = JSON.parse(localStorage.getItem(reviewKey) || '{{}}');
  let propagating = false;
  const seekAll = time => {{ videos.forEach(v => {{ v.currentTime = Math.max(0, time); }}); update(time); }};
  const playAll = () => Promise.allSettled(videos.map(v => v.play()));
  const pauseAll = () => videos.forEach(v => v.pause());
  const update = time => {{
    const mins = Math.floor(time / 60); const secs = time - mins * 60;
    document.getElementById('clock').textContent = `${{String(mins).padStart(2,'0')}}:${{secs.toFixed(3).padStart(6,'0')}}`;
    const active = data.segments.find(s => time >= s.start && time <= s.end);
    cards.forEach(c => c.classList.toggle('active', Boolean(active && c.dataset.movement === active.movement_id)));
  }};
  videos.forEach(source => {{
    source.addEventListener('seeking', () => {{ if (!propagating) {{ propagating=true; videos.filter(v=>v!==source).forEach(v=>v.currentTime=source.currentTime); propagating=false; }} update(source.currentTime); }});
    source.addEventListener('play', () => {{ if (!propagating) {{ propagating=true; videos.filter(v=>v!==source).forEach(v=>v.play()); propagating=false; }} }});
    source.addEventListener('pause', () => {{ if (!propagating) {{ propagating=true; videos.filter(v=>v!==source).forEach(v=>v.pause()); propagating=false; }} }});
    source.addEventListener('timeupdate', () => {{ if (source === videos[0]) {{ videos.slice(1).forEach(v => {{ if (Math.abs(v.currentTime-source.currentTime)>.12) v.currentTime=source.currentTime; }}); update(source.currentTime); }} }});
  }});
  document.getElementById('sync-play').onclick = playAll;
  document.getElementById('sync-pause').onclick = pauseAll;
  document.getElementById('sync-zero').onclick = () => {{ pauseAll(); seekAll(0); }};
  document.querySelectorAll('[data-seek]').forEach(node => node.addEventListener('click', event => {{ event.stopPropagation(); seekAll(Number(node.dataset.seek)); }}));
  cards.forEach(card => card.addEventListener('click', () => seekAll(Number(card.dataset.start))));
  document.querySelectorAll('[data-review-event]').forEach(button => {{
    const eventId = button.dataset.reviewEvent;
    if (reviewSelections[eventId] === button.dataset.reviewValue) button.classList.add('selected');
    button.addEventListener('click', event => {{
      event.stopPropagation(); reviewSelections[eventId] = button.dataset.reviewValue;
      localStorage.setItem(reviewKey, JSON.stringify(reviewSelections));
      document.querySelectorAll(`[data-review-event="${{eventId}}"]`).forEach(item => item.classList.toggle('selected', item === button));
    }});
  }});
  const exportButton = document.getElementById('export-review');
  if (exportButton) exportButton.onclick = () => {{
    const payload = {{schema_version:1, timeline_id:data.timeline_id, created_at:new Date().toISOString(), reviews:reviewSelections}};
    const link = document.createElement('a'); link.href = URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)], {{type:'application/json'}}));
    link.download = `tk3d-review-${{data.timeline_id}}.json`; link.click(); URL.revokeObjectURL(link.href);
  }};
  update(0);
}})();
</script></body></html>
'''


def _movement_card(
    segment: dict[str, Any],
    movement: dict[str, Any],
    evidence: dict[str, Any] | None,
    fps: float,
) -> str:
    evidence = evidence or {}
    start_sec = segment["start_frame"] / fps
    end_sec = segment["end_frame"] / fps
    anchors = "".join(
        f'<button type="button" class="anchor" data-seek="{frame / fps:.6f}">{_escape(phase)} · {frame / fps:.2f}s</button>'
        for phase, frame in segment["anchors"].items()
    )
    valid_ratio = evidence.get("body17_valid_ratio")
    reprojection = evidence.get("median_reprojection_error_px")
    cameras = evidence.get("median_used_cameras")
    return f'''<article class="movement" data-movement="{_escape(segment["movement_id"])}" data-start="{start_sec:.6f}">
      <div class="movement-top"><span class="movement-id">{_escape(segment["movement_id"])}</span><span>{start_sec:.2f}–{end_sec:.2f}s</span></div>
      <div class="movement-name">{_escape(movement["display_name"])}</div>
      <div class="metrics"><span>Etiket güveni <b>%{segment["confidence"] * 100:.0f}</b></span><span>BODY-17 <b>{_percent(valid_ratio)}</b></span>
        <span>Reprojection <b>{_number(reprojection, " px")}</b></span><span>Kamera <b>{_number(cameras, "")}</b></span></div>
      <div class="anchors">{anchors}</div>
    </article>'''


def _engineering_trial_html(report: dict[str, Any] | None) -> tuple[str, str]:
    if report is None:
        return '<div class="stat"><span>Mühendislik denemesi</span><b>Yok</b></div>', ""
    score = report.get("partial_engineering_trial_score")
    reference = report.get("reference_accuracy_score")
    deductions = report.get("trial_deductions", [])
    deduction_items = "".join(
        f'''<li><code>{_escape(item.get("movement_id"))} · {_escape(item.get("family"))}</code>
          <span>{_escape(item.get("metric_id"))}: {_number(item.get("measured_value"), "")} · deneme -{_number(item.get("deduction_points"), "")}</span></li>'''
        for item in deductions
    ) or "<li><span>Kalite kapılarını geçen mühendislik hata adayı seçilmedi.</span></li>"
    summary = report.get("summary", {})
    coverage = float(report.get("measurement_coverage_ratio", 0.0))
    disclaimer = report.get("engineering_profile", {}).get("disclaimer", "")
    if score is None:
        stat = '<div class="stat"><span>BODY-17 denemesi</span><b>Devre dışı</b></div>'
        section = '''<section class="section"><h2>BODY-17 sayısal denemesi iptal edildi</h2>
          <div class="notice"><div>⛔</div><div><strong>Bu motor artık puan üretmiyor.</strong>
          <p>WholeBody teknik hatalarını kaçırdığı ve hata bulamamayı doğruluk gibi gösterdiği için geçersizleştirildi.</p></div></div></section>'''
        return stat, section
    stat = (
        '<div class="stat"><span>M01-M06 mühendislik denemesi</span>'
        f'<b>{_number(score, "")} / {_number(reference, "")} · %{coverage * 100:.0f} kapsam</b></div>'
    )
    section = f'''<section class="section"><h2>M01-M06 provisional mühendislik denemesi</h2>
      <div class="notice"><div>⚙</div><div><strong>Bu değer WT Accuracy puanı değildir.</strong>
      <p>{_escape(disclaimer)} Tam Taegeuk 1 performansıyla karşılaştırılamaz.</p></div></div>
      <p style="margin-bottom:12px">{int(summary.get("measurable_count", 0))}/{int(summary.get("measurement_count", 0))} ölçülebilir · {int(summary.get("candidate_count", 0))} aday ·
      {int(summary.get("not_measurable_count", 0))} ölçülemez · aile/faz tekrarları birleştirilmiştir.</p>
      <ul>{deduction_items}</ul></section>'''
    return stat, section


def _wholebody_diagnostics_html(report: dict[str, Any] | None) -> tuple[str, str]:
    if report is None:
        return '<div class="stat"><span>WholeBody-133</span><b>Yok</b></div>', ""
    summary = report.get("summary", {})
    coverage = report.get("coverage", {})
    candidates = report.get("candidate_events", [])
    movements = report.get("movements", [])

    # --- Existing candidate list (kept for quick-filter) ---
    candidate_items = "".join(
        f'''<li class="candidate-row"><code>{_escape(item.get("movement_id"))} · {_escape(item.get("family"))}</code>
          <span>{_escape(item.get("metric_id"))} → {_escape(item.get("criterion_id", "eşlenmemiş"))}: {_number(item.get("value"), "")} · eşik farkı {_number(item.get("threshold_margin"), "")} · yalnız inceleme adayı</span>
          <button type="button" data-seek="{float(item.get("measurement_evidence", {}).get("anchor_time_sec", 0.0)):.6f}">Videoda aç</button></li>'''
        for item in candidates
    ) or "<li><span>WholeBody inceleme adayı bulunmadı.</span></li>"

    # --- NEW: per-movement all-metrics blocks ---
    movement_blocks = "".join(_wholebody_movement_block(movement) for movement in movements)

    coverage_ratio = float(coverage.get("measurement_coverage_ratio", 0.0))
    candidate_count = int(summary.get("review_candidate_count", 0))
    within_count = int(summary.get("within_screening_range_count", 0))
    metric_count = int(summary.get("metric_count", 0))
    measurement_table = _wholebody_measurement_table(report)
    stat = (
        '<div class="stat"><span>WholeBody-133 ölçüm</span>'
        f'<b>{metric_count} · {candidate_count} aday · %{coverage_ratio * 100:.0f}</b></div>'
    )
    section = f'''<section class="section"><h2>WholeBody-133 hata inceleme adayları ve tüm ölçümler</h2>
      <div class="notice"><div>🔎</div><div><strong>Skor yok; her aday videoda doğrulanmalı.</strong>
      <p>Dirsek, bilek, omuz-kalça rotasyonu, baş yönü, hikite, yumruk, eşzaman, fixation, ağırlık aktarımı, trajectory
      ve daha fazlası ölçülür. Yeşil = eşik içi, kırmızı = aday, gri = ölçülemedi, mavi = yalnız tanılı.</p></div></div>
      <p style="margin-bottom:12px">{int(coverage.get("measurable_metric_count", 0))}/{int(coverage.get("thresholded_metric_count", 0))} ölçülebilir ·
      {within_count} eşik içi · {candidate_count} aday ·
      kapsama kapısı: {"geçti" if coverage.get("coverage_gate_passed") else "başarısız"} · Accuracy: hesaplanmadı.
      Hareket başlığına tıkla → metrikleri aç/kapa.</p>
      {movement_blocks}
      <h2 style="margin-top:18px">Eşik aşan inceleme adayları</h2>
      <ul>{candidate_items}</ul>
      <h2 style="margin-top:20px">El, yumruk ve baş/yüz ölçüm matrisi</h2>
      <p>Yumruk kapalılığı 21 noktalı el geometrisinden hesaplanır. Baş/yüz yönü 68 yüz noktası ve omuz hattını kullanır; göz küresi takibi değildir.</p>
      {measurement_table}</section>'''
    return stat, section


def _wholebody_movement_block(movement: dict[str, Any]) -> str:
    """Render a collapsible block for one movement showing ALL its metrics."""
    metrics = movement.get("metrics", [])
    if not metrics:
        return ""
    candidate_count = sum(1 for m in metrics if m.get("screening_status") == "review_candidate")
    not_measurable_count = sum(1 for m in metrics if m.get("screening_status") == "not_measurable")
    pass_count = sum(1 for m in metrics if m.get("screening_status") == "within_screening_range")
    if candidate_count > 0:
        badge_class = "wb-badge-warn"
        badge_text = f"{candidate_count} aday"
    elif not_measurable_count > 0:
        badge_class = "wb-badge-warn"
        badge_text = f"{not_measurable_count} ölçülemedi"
    else:
        badge_class = "wb-badge-ok"
        badge_text = f"{pass_count} geçti"
    metric_cells = "".join(_wholebody_metric_cell(m) for m in metrics)
    anchor_time = 0.0
    evidence = metrics[0].get("measurement_evidence") if metrics else None
    if evidence:
        anchor_time = float(evidence.get("anchor_time_sec", 0.0))
    movement_id = _escape(movement.get("movement_id", ""))
    display_name = _escape(movement.get("display_name", ""))
    return f'''<article class="wb-movement-block" onclick="this.classList.toggle('open')">
      <div class="wb-movement-hdr">
        <div><span class="wb-mid">{movement_id}</span><span class="wb-mname">{display_name}</span></div>
        <div style="display:flex;gap:6px;align-items:center">
          <span class="wb-badge {badge_class}">{badge_text}</span>
          <button type="button" class="anchor" data-seek="{anchor_time:.6f}" onclick="event.stopPropagation()" style="font-size:11px;padding:4px 8px">Videoda aç</button>
        </div>
      </div>
      <div class="wb-collapse wb-metrics-grid">{metric_cells}</div>
    </article>'''


_CRITERION_LABELS: dict[str, str] = {
    "balance.torso_vertical": "Gövde dikliği",
    "stance.foot_direction": "Arka ayak yönü",
    "rotation.shoulder_hip": "Omuz-kalça rotasyonu",
    "gaze.direction": "Baş/bakış yönü",
    "technique.side": "Doğru taraf dominansı",
    "technique.hikite.position": "Hikite (geri çekiş)",
    "technique.hand.wrist_alignment": "Bilek-önkol hizası",
    "technique.hand.fist": "Yumruk kapanışı",
    "timing.hand_foot.simultaneity": "El-ayak eşzamanlılığı",
    "technique.fixation.stability": "Fixation kararlılığı",
    "stance.weight_transfer": "Ağırlık aktarımı",
    "technique.trajectory": "Hareket yolu verimliliği",
    "presentation.kinematics": "Tepe hızı",
    "stance.ap_seogi.span": "Ap seogi açıklığı",
    "stance.ap_seogi.front_knee": "Ap seogi ön diz",
    "stance.ap_gubi.span": "Apkubi açıklığı",
    "stance.ap_gubi.front_knee": "Apkubi ön diz",
    "technique.momtong_jireugi.height": "Yumruk bilek yüksekliği",
    "technique.momtong_jireugi.elbow": "Yumruk dirsek açısı",
    "technique.arae_makki.height": "Arae makki bilek yüksekliği",
    "technique.arae_makki.elbow": "Arae makki dirsek açısı",
    "technique.ap_chagi.knee_extension": "Ap chagi diz açılımı",
    "technique.ap_chagi.height": "Ap chagi yüksekliği",
    "technique.ap_chagi.rechamber": "Ap chagi geri çekme",
    "technique.ap_chagi.support_foot_pivot": "Destek ayak pivotu",
    "timing.kick_landing_punch.sequence": "Tekme-yumruk sırası",
}


def _wholebody_metric_cell(metric: dict[str, Any]) -> str:
    """Render a single metric cell inside the per-movement grid."""
    status = metric.get("screening_status", "")
    value = metric.get("value")
    unit = metric.get("unit", "")
    criterion_id = metric.get("criterion_id", "")
    screening_rule = metric.get("screening_rule")

    if status == "within_screening_range":
        css_class = "wb-ok"
        dot = "&#10003;"
    elif status == "review_candidate":
        css_class = "wb-cand"
        dot = "&#9888;"
    elif status == "not_measurable":
        css_class = "wb-nm"
        dot = "&mdash;"
    else:
        css_class = "wb-diag"
        dot = "&#8505;"

    value_text = "&mdash;" if value is None else f"{float(value):.2f}"
    label = _CRITERION_LABELS.get(criterion_id, criterion_id)

    if screening_rule is None:
        range_text = "yalnız tanılı"
    elif screening_rule.get("operator") == "max":
        range_text = f"eşik ≤ {float(screening_rule['value']):.1f} {_escape(unit)}"
    elif screening_rule.get("operator") == "min":
        range_text = f"eşik ≥ {float(screening_rule['value']):.1f} {_escape(unit)}"
    elif screening_rule.get("operator") == "range":
        limits = screening_rule["value"]
        range_text = f"aralık {float(limits[0]):.1f}–{float(limits[1]):.1f} {_escape(unit)}"
    else:
        range_text = ""

    return f'''<div class="wb-m {css_class}">
      <span class="wb-dot">{dot}</span>
      <span class="wb-m-name">{_escape(label)}</span>
      <span class="wb-m-val">{value_text} {_escape(unit)}</span>
      <span class="wb-m-range">{_escape(range_text)}</span>
    </div>'''
def _wholebody_measurement_table(report: dict[str, Any]) -> str:
    selected = {
        "fist_closure_ratio": "Yumruk kapalılığı",
        "wrist_forearm_alignment_deg": "Bilek–ön kol hizası",
        "head_torso_yaw_mismatch_deg": "Baş/yüz–gövde yön farkı",
        "executing_wrist_height_torso_ratio": "Teknik el yüksekliği",
        "executing_elbow_deg": "Uygulayan dirsek açısı",
        "reaction_hand_hip_distance_ratio": "Hikite–kalça mesafesi",
    }
    status_labels = {
        "within_screening_range": ("Tarama içinde", "ok"),
        "review_candidate": ("İnceleme adayı — puan yok", "review"),
        "not_measurable": ("Ölçülemedi", "missing"),
        "measured_diagnostic_only": ("Ölçüldü — eşik yok", "ok"),
    }
    rows: list[str] = []
    for movement in report.get("movements", []):
        movement_id = _escape(movement.get("movement_id"))
        for metric in movement.get("metrics", []):
            metric_id = metric.get("metric_id")
            if metric_id not in selected:
                continue
            status = str(metric.get("screening_status"))
            status_text, status_class = status_labels.get(status, (status, "missing"))
            if status == "not_measurable":
                evidence = metric.get("measurement_evidence") or {}
                sample_counts = evidence.get("required_joint_sample_counts") or {}
                missing = evidence.get("missing_required_joints") or []
                window_size = int(evidence.get("end_frame", 0)) - int(evidence.get("start_frame", 0)) + 1
                if missing:
                    details = ", ".join(
                        f"{label} {int(sample_counts.get(label, 0))}/{window_size}" for label in missing
                    )
                    status_text = f"Ölçülemedi · {details}"
            value = _number(metric.get("value"), f" {metric.get('unit', '')}".rstrip())
            rows.append(
                f"<tr><td><code>{movement_id}</code></td><td>{_escape(selected[metric_id])}</td>"
                f"<td>{_escape(value)}</td><td class=\"{status_class}\">{_escape(status_text)}</td></tr>"
            )
    body = "".join(rows) or '<tr><td colspan="4">Seçili WholeBody ölçümü bulunmadı.</td></tr>'
    return (
        '<div class="metric-table-wrap"><table class="metric-table"><thead><tr>'
        '<th>Hareket</th><th>Ölçüm</th><th>3B değer</th><th>Durum</th>'
        f"</tr></thead><tbody>{body}</tbody></table></div>"
    )


def _decision_evidence_html(
    decisions: dict[str, Any] | None,
    evidence: dict[str, Any] | None,
) -> tuple[str, str]:
    if decisions is None or evidence is None:
        return '<div class="stat"><span>Accuracy</span><b>Hesaplanmadı</b></div>', ""
    events = evidence.get("events", [])
    if not isinstance(events, list):
        raise ScoringContractError("decision evidence events must be a list")
    summary = evidence.get("summary", {})
    partial_total = decisions.get("observed_scope_provisional_deduction_total")
    rows = "".join(_decision_event_row(event) for event in events)
    if not rows:
        rows = "<li><span>Görselleştirilebilir kaynak-bağlı karar bulunmadı.</span></li>"
    stat = (
        '<div class="stat"><span>Kaynak-bağlı küçük hata</span>'
        f'<b>{int(summary.get("confirmed_deduction_candidate_count", 0))} · -{_number(partial_total, "")}</b></div>'
    )
    section = f'''<section class="section"><h2>Kaynak-bağlı hata kanıtları</h2>
      <div class="notice"><div>🎯</div><div><strong>Ölçüm 3B, kamera çizimi yalnız görsel izdir.</strong>
      <p>{_escape(evidence.get("camera_overlay_warning", ""))} Kırmızı kesinti adayı, sarı sınır-belirsiz,
      gri ölçülemedi, yeşil kaynak aralığı içinde anlamına gelir.</p></div></div>
      <p style="margin-bottom:12px">{len(events)} karar · {int(summary.get("confirmed_deduction_candidate_count", 0))} küçük hata ·
      {int(summary.get("boundary_uncertain_count", 0))} sınır-belirsiz · {int(summary.get("not_measurable_count", 0))} ölçülemedi.
      Kararı videoda inceleyip kendi kontrolünü kaydedebilirsin.</p>
      <div class="toolbar" style="margin-bottom:10px"><button type="button" id="export-review">İnceleme kararlarını JSON indir</button></div>
      <ul>{rows}</ul></section>'''
    return stat, section


def _decision_event_row(event: dict[str, Any]) -> str:
    measurement = event.get("measurement") or {}
    window = event.get("evidence_window") or {}
    value = _number(measurement.get("value"), f" {measurement.get('unit', '')}".rstrip())
    interval = measurement.get("interval_95")
    interval_text = "—" if not interval else f"{_number(interval[0], '')}–{_number(interval[1], '')}"
    limits = measurement.get("rule_limits")
    if measurement.get("rule_operator") == "max" and isinstance(limits, list) and limits:
        limit_text = f"≤ {_number(limits[0], '')}"
    elif isinstance(limits, list) and len(limits) == 2:
        limit_text = f"{_number(limits[0], '')}–{_number(limits[1], '')}"
    else:
        limit_text = "—"
    deduction = event.get("deduction_points")
    deduction_text = "kesinti yok" if deduction is None else f"-{_number(deduction, '')}"
    event_id = _escape(event.get("event_id"))
    seek = float(window.get("anchor_time_sec", 0.0))
    source = event.get("source") or {}
    source_id = source.get("source_id") or source.get("source_ref") or "kaynak belirtilmedi"
    return f'''<li class="decision-row {_escape(event.get("display_color", "gray"))}">
      <code>{_escape(event.get("movement_id") or "PERF")} · {_escape(event.get("display_label"))}</code>
      <span><b>{_escape(event.get("metric_id") or event.get("event_kind"))}</b> · 3B {_escape(value)} · %95 {_escape(interval_text)} ·
      kaynak sınırı {_escape(limit_text)} · <b>{_escape(deduction_text)}</b><br>
      {_escape(event.get("description"))} · kaynak: {_escape(source_id)}</span>
      <div><button type="button" data-seek="{seek:.6f}">Videoda aç</button>
      <div class="review-actions" style="margin-top:6px">
        <button type="button" data-review-event="{event_id}" data-review-value="confirmed">Doğru</button>
        <button type="button" data-review-event="{event_id}" data-review-value="rejected">Yanlış</button>
        <button type="button" data-review-event="{event_id}" data-review-value="uncertain">Belirsiz</button>
      </div></div></li>'''


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _percent(value: Any) -> str:
    return "—" if value is None else f"%{float(value) * 100:.1f}"


def _number(value: Any, suffix: str) -> str:
    return "—" if value is None else f"{float(value):.2f}{suffix}"
