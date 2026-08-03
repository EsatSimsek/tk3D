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
    .sources {{ display:flex; flex-wrap:wrap; gap:8px; margin-top:12px; }} a {{ color:var(--cyan); }}
    footer {{ color:#6f8795; font-size:12px; text-align:center; margin-top:20px; }}
    @media(max-width:950px) {{ .stats {{ grid-template-columns:1fr 1fr; }} .videos,.two-col {{ grid-template-columns:1fr; }} .movement-grid {{ grid-template-columns:1fr 1fr; }} }}
    @media(max-width:580px) {{ header {{ display:block; }} .pill {{ display:inline-block; margin-top:12px; }} .movement-grid,.stats {{ grid-template-columns:1fr; }} .candidate-row {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body><main>
  <header><div><div class="eyebrow">TK3D · Ölçüm kanıtı</div><h1>Taegeuk 1 Kısa Kayıt İncelemesi</h1>
    <p>{len(video_sources)} kamera, aynı zaman çizelgesi ve hareket/faz ölçümleri tek ekranda.</p></div>
    <div class="pill">Kısmi kayıt · {len(observed)}/{len(spec["movements"])}</div></header>
  <section class="notice"><div>⚠</div><div><strong>Bu çıktı puan değildir.</strong><p>Accuracy hesaplanmadı ve kesinti üretilmedi. {_escape(source_end_reason)}</p></div></section>
  <section class="stats">
    <div class="stat"><span>Kayıttaki hareket</span><b>{len(observed)} / {len(spec["movements"])}</b></div>
    <div class="stat"><span>Ölçülen faz ankrajı</span><b>{anchor_count}</b></div>
    <div class="stat"><span>Tam gözlenen ankraj</span><b>%{observed_ratio * 100:.1f}</b></div>
    <div class="stat"><span>Accuracy</span><b>Hesaplanmadı</b></div>
    {trial_stat}
    {wholebody_stat}
  </section>
  <section class="section"><h2>Senkron kamera incelemesi</h2><div class="videos">{video_html}</div>
    <div class="toolbar"><button type="button" id="sync-play">▶ İkisini oynat</button><button type="button" id="sync-pause">Ⅱ Duraklat</button><button type="button" id="sync-zero">↺ Başa dön</button><span id="clock">00:00.000</span><p>Bir videoda sarınca diğeri aynı zamana gelir.</p></div>
  </section>
  <section class="section"><h2>Kayıtta bulunan hareketler</h2><div class="movement-grid">{movement_cards}</div></section>
  {trial_section}
  {wholebody_section}
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
    candidate_items = "".join(
        f'''<li class="candidate-row"><code>{_escape(item.get("movement_id"))} · {_escape(item.get("family"))}</code>
          <span>{_escape(item.get("metric_id"))} → {_escape(item.get("criterion_id", "eşlenmemiş"))}: {_number(item.get("value"), "")} · eşik farkı {_number(item.get("threshold_margin"), "")} · yalnız inceleme adayı</span>
          <button type="button" data-seek="{float(item.get("measurement_evidence", {}).get("anchor_time_sec", 0.0)):.6f}">Videoda aç</button></li>'''
        for item in candidates
    ) or "<li><span>WholeBody inceleme adayı bulunmadı.</span></li>"
    coverage_ratio = float(coverage.get("measurement_coverage_ratio", 0.0))
    candidate_count = int(summary.get("review_candidate_count", 0))
    stat = (
        '<div class="stat"><span>WholeBody-133 aday</span>'
        f'<b>{candidate_count} · %{coverage_ratio * 100:.0f} kapsam</b></div>'
    )
    section = f'''<section class="section"><h2>WholeBody-133 hata inceleme adayları</h2>
      <div class="notice"><div>🔎</div><div><strong>Skor yok; her aday videoda doğrulanmalı.</strong>
      <p>El, ayak, yüz, gövde, trajectory, hikite, rotasyon, eşzaman ve fixation ölçülür. Eşik içi olmak teknik doğruluk değildir.</p></div></div>
      <p style="margin-bottom:12px">{int(coverage.get("measurable_metric_count", 0))}/{int(coverage.get("thresholded_metric_count", 0))} ölçülebilir ·
      kapsama kapısı: {"geçti" if coverage.get("coverage_gate_passed") else "başarısız"} · Accuracy: hesaplanmadı. “Videoda aç” aynı kanıt zamanına bütün kameraları taşır.</p>
      <ul>{candidate_items}</ul></section>'''
    return stat, section


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _percent(value: Any) -> str:
    return "—" if value is None else f"%{float(value) * 100:.1f}"


def _number(value: Any, suffix: str) -> str:
    return "—" if value is None else f"{float(value):.2f}{suffix}"
