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
    categorical_diagnostics_report: dict[str, Any] | None = None,
    presentation_diagnostics_report: dict[str, Any] | None = None,
    technical_conformance_report: dict[str, Any] | None = None,
    run_history_url: str | None = None,
    automatic_segmentation_report: dict[str, Any] | None = None,
    technical_accuracy_diagnostics_report: dict[str, Any] | None = None,
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
    if categorical_diagnostics_report is not None and categorical_diagnostics_report.get("movement_timeline_id") != timeline["timeline_id"]:
        raise ScoringContractError("categorical diagnostics must reference the same MovementTimeline")
    if presentation_diagnostics_report is not None and presentation_diagnostics_report.get("timeline_id") != timeline["timeline_id"]:
        raise ScoringContractError("Presentation diagnostics must reference the same MovementTimeline")
    if technical_conformance_report is not None and technical_conformance_report.get("movement_timeline_id") != timeline["timeline_id"]:
        raise ScoringContractError("technical conformance must reference the same MovementTimeline")
    if technical_accuracy_diagnostics_report is not None:
        if technical_accuracy_diagnostics_report.get("status") != "technical_accuracy_diagnostics_only":
            raise ScoringContractError("technical accuracy diagnostics status is invalid")
        if technical_accuracy_diagnostics_report.get("movement_timeline_id") != timeline["timeline_id"]:
            raise ScoringContractError("technical accuracy diagnostics must reference the same MovementTimeline")
    if automatic_segmentation_report is not None:
        if automatic_segmentation_report.get("status") != "automatic_segmentation_diagnostic_only":
            raise ScoringContractError("automatic segmentation report status is invalid")
        if automatic_segmentation_report.get("movement_timeline_id") != timeline["timeline_id"]:
            raise ScoringContractError("automatic segmentation must reference the same MovementTimeline")

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
          <video controls preload="auto" playsinline data-video-index="{index}">
            <source src="{_escape(source)}" type="{_video_mime_type(source)}">
            Tarayıcınız bu video biçimini desteklemiyor.
          </video>
          <div class="video-health" data-video-health="{index}" aria-live="polite">Video hazırlanıyor…</div>
        </article>'''
        for index, (label, source) in enumerate(video_sources.items())
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
    categorical_stat, categorical_section = _categorical_diagnostics_html(
        categorical_diagnostics_report,
        timeline["fps"],
    )
    technical_stat, technical_section = _technical_conformance_html(
        technical_conformance_report,
        timeline["fps"],
    )
    accuracy_diagnostic_stat, accuracy_diagnostic_section = _technical_accuracy_html(
        technical_accuracy_diagnostics_report
    )
    presentation_stat, presentation_section = _presentation_diagnostics_html(
        presentation_diagnostics_report
    )
    automatic_stat, automatic_section = _automatic_segmentation_html(
        automatic_segmentation_report,
        timeline["fps"],
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
    history_link = (
        ""
        if not run_history_url
        else f'<a class="history-link" href="{_escape(run_history_url)}">Koşu geçmişi ve regresyon raporunu aç →</a>'
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
    .stats {{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-bottom:18px; }}
    .stat,.section {{ border:1px solid var(--line); background:linear-gradient(145deg,rgba(20,37,52,.94),rgba(12,25,36,.94)); border-radius:16px; }}
    .stat {{ padding:16px; }} .stat b {{ display:block; font-size:25px; margin-top:5px; }} .stat span {{ color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.09em; }}
    .section {{ padding:18px; margin-bottom:18px; }} .videos {{ display:grid; grid-template-columns:repeat({len(video_sources)},minmax(0,1fr)); gap:12px; }}
    .video-card {{ border-radius:12px; overflow:hidden; background:#03070a; border:1px solid var(--line); }} video {{ width:100%; display:block; aspect-ratio:16/9; background:#000; }}
    .video-label {{ padding:10px 12px; color:#d9edf8; font-size:13px; font-weight:700; }} .camera-dot {{ display:inline-block; width:8px; height:8px; background:var(--green); border-radius:50%; margin-right:8px; box-shadow:0 0 10px var(--green); }}
    .video-health {{ padding:7px 12px; color:var(--muted); font-size:12px; border-top:1px solid var(--line); }}
    .video-health[data-state="ready"],.video-health[data-state="playing"] {{ color:var(--green); }}
    .video-health[data-state="loading"] {{ color:var(--amber); }} .video-health[data-state="error"] {{ color:var(--red); font-weight:700; }}
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
    .history-link {{ display:inline-block; margin:0 0 18px; padding:11px 14px; border:1px solid #37627b; border-radius:10px; background:#102839; font-weight:750; text-decoration:none; }}
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
    .diag-row {{ display:grid; grid-template-columns:150px 150px minmax(0,1fr) auto; gap:10px; align-items:center; }}
    .diag-consistent {{ color:var(--green); }} .diag-mismatch {{ color:var(--red); font-weight:750; }}
    .diag-ambiguous {{ color:var(--amber); }} .proxy-grid {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:10px; }}
    .proxy-card {{ border:1px solid var(--line); border-radius:12px; background:#0b1822; padding:14px; }}
    .proxy-card h3 {{ margin:0 0 10px; color:var(--cyan); }} .proxy-card li {{ display:block; }}
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
  {history_link}
  <section class="stats">
    <div class="stat"><span>Kayıttaki hareket</span><b>{len(observed)} / {len(spec["movements"])}</b></div>
    <div class="stat"><span>Ölçülen faz ankrajı</span><b>{anchor_count}</b></div>
    <div class="stat"><span>Tam gözlenen ankraj</span><b>%{observed_ratio * 100:.1f}</b></div>
    {decision_stat}
    {wholebody_stat}
    {categorical_stat}
    {technical_stat}
    {accuracy_diagnostic_stat}
    {presentation_stat}
    {automatic_stat}
    {trial_stat}
  </section>
  <section class="section"><h2>Senkron kamera incelemesi</h2><div class="videos">{video_html}</div>
    <div class="toolbar"><button type="button" id="sync-play">▶ {len(video_sources)} kamerayı oynat</button><button type="button" id="sync-pause">Ⅱ Duraklat</button><button type="button" id="sync-zero">↺ Başa dön</button><span id="clock">00:00.000</span><span id="sync-status" aria-live="polite">Videolar hazırlanıyor…</span><p>Bir videoda sarınca diğerleri aynı zamana gelir.</p></div>
  </section>
  <section class="section"><h2>Kayıtta bulunan hareketler</h2><div class="movement-grid">{movement_cards}</div></section>
  {automatic_section}
  {trial_section}
  {wholebody_section}
  {categorical_section}
  {technical_section}
  {accuracy_diagnostic_section}
  {presentation_section}
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
  let reviewSelections = {{}};
  try {{
    const stored = JSON.parse(localStorage.getItem(reviewKey) || '{{}}');
    if (stored && typeof stored === 'object' && !Array.isArray(stored)) reviewSelections = stored;
  }} catch (error) {{ localStorage.removeItem(reviewKey); }}
  let syncingSeek = false;
  let seekTimer = null;
  let desiredSeekTime = null;
  let syncingTransport = false;
  let transportTimer = null;
  const reviewVideoObjectUrls = [];
  const holdSeekSync = (delay=8000) => {{
    syncingSeek = true;
    if (seekTimer) clearTimeout(seekTimer);
    seekTimer = setTimeout(() => {{ syncingSeek = false; desiredSeekTime = null; seekTimer = null; }}, delay);
  }};
  const holdTransportSync = () => {{
    syncingTransport = true;
    if (transportTimer) clearTimeout(transportTimer);
    transportTimer = setTimeout(() => {{ syncingTransport = false; transportTimer = null; }}, 150);
  }};
  const syncStatus = document.getElementById('sync-status');
  const healthNodes = [...document.querySelectorAll('[data-video-health]')];
  const setHealth = (index, state, message) => {{
    const node = healthNodes[index]; if (!node) return;
    node.dataset.state = state; node.textContent = message;
  }};
  const boundedTime = (video, time) => Number.isFinite(video.duration) ? Math.min(Math.max(0, time), Math.max(0, video.duration - .001)) : Math.max(0, time);
  const timeAvailable = (video, time) => {{
    if (time <= .03) return true;
    const target = boundedTime(video, time);
    const ranges = video.seekable;
    for (let index=0; index<ranges.length; index += 1) {{
      if (target >= ranges.start(index)-.03 && target <= ranges.end(index)+.03) return true;
    }}
    return false;
  }};
  const seekOne = (video, time, index) => {{
    const apply = () => {{
      if (!timeAvailable(video, time)) {{
        video.preload = 'auto';
        setHealth(index, 'loading', 'Hedef kareler yükleniyor…');
        syncStatus.textContent = 'Kanıt zamanı yükleniyor; hedef korunuyor…';
        return;
      }}
      try {{ video.currentTime = boundedTime(video, time); }} catch (error) {{ setHealth(index, 'error', 'Bu zamana gidilemedi.'); }}
    }};
    if (video.readyState === 0) {{ video.addEventListener('loadedmetadata', apply, {{once:true}}); video.load(); }} else apply();
  }};
  const seekAll = time => {{ desiredSeekTime = time; holdSeekSync(); videos.forEach((video,index) => seekOne(video, time, index)); update(time); }};
  const confirmDesiredSeek = () => {{
    if (!syncingSeek || desiredSeekTime === null) return;
    videos.forEach((video,index) => {{
      if (!video.seeking && Math.abs(video.currentTime-boundedTime(video,desiredSeekTime))>.03) seekOne(video,desiredSeekTime,index);
    }});
    if (videos.every(video => !video.seeking && Math.abs(video.currentTime-boundedTime(video,desiredSeekTime))<=.03)) holdSeekSync(80);
  }};
  const prepareSeekableVideo = async (video, index) => {{
    if (!['http:', 'https:'].includes(location.protocol) || video.dataset.seekPreparation) return;
    const source = video.currentSrc || video.querySelector('source')?.src;
    if (!source) return;
    video.dataset.seekPreparation = 'loading';
    setHealth(index, 'loading', 'Atlama için video hazırlanıyor…');
    try {{
      const response = await fetch(source, {{cache:'force-cache'}});
      if (!response.ok) throw new Error(`HTTP ${{response.status}}`);
      const objectUrl = URL.createObjectURL(await response.blob());
      reviewVideoObjectUrls.push(objectUrl);
      video.dataset.seekPreparation = 'blob';
      video.src = objectUrl;
      video.load();
    }} catch (error) {{
      video.dataset.seekPreparation = 'native';
      setHealth(index, 'loading', 'Sunucu atlama desteği bekleniyor…');
    }}
  }};
  const playAll = async () => {{
    holdTransportSync();
    const results = await Promise.allSettled(videos.map(v => v.play()));
    const failed = results.filter(item => item.status === 'rejected').length;
    syncStatus.textContent = failed ? `${{failed}} kamera oynatılamadı; video durumunu kontrol edin.` : 'Senkron oynatılıyor.';
  }};
  const pauseAll = () => {{ holdTransportSync(); videos.forEach(v => v.pause()); }};
  const update = time => {{
    const mins = Math.floor(time / 60); const secs = time - mins * 60;
    document.getElementById('clock').textContent = `${{String(mins).padStart(2,'0')}}:${{secs.toFixed(3).padStart(6,'0')}}`;
    const active = data.segments.find(s => time >= s.start && time <= s.end);
    cards.forEach(c => c.classList.toggle('active', Boolean(active && c.dataset.movement === active.movement_id)));
  }};
  videos.forEach((source,index) => {{
    source.addEventListener('loadedmetadata', () => setHealth(index, 'ready', `Hazır · ${{source.duration.toFixed(2)}} sn`));
    source.addEventListener('canplay', () => setHealth(index, 'ready', `Hazır · ${{source.duration.toFixed(2)}} sn`));
    source.addEventListener('canplaythrough', () => {{ setHealth(index, 'ready', `Tam hazır · ${{source.duration.toFixed(2)}} sn`); confirmDesiredSeek(); }});
    source.addEventListener('progress', confirmDesiredSeek);
    source.addEventListener('playing', () => setHealth(index, 'playing', 'Oynatılıyor'));
    source.addEventListener('waiting', () => setHealth(index, 'loading', 'Kareler yükleniyor…'));
    source.addEventListener('stalled', () => setHealth(index, 'loading', 'Video yüklemesi bekliyor…'));
    source.addEventListener('error', () => {{
      const code = source.error ? source.error.code : 'bilinmiyor';
      setHealth(index, 'error', `Video açılamadı · hata ${{code}}`);
      syncStatus.textContent = 'En az bir kamera videosu açılamadı.';
    }});
    source.addEventListener('seeking', () => {{ if (!syncingSeek) {{ desiredSeekTime = source.currentTime; holdSeekSync(); videos.filter(v=>v!==source).forEach(v=>seekOne(v,desiredSeekTime,videos.indexOf(v))); update(desiredSeekTime); }} }});
    source.addEventListener('seeked', confirmDesiredSeek);
    source.addEventListener('play', () => {{ if (!syncingTransport) {{ holdTransportSync(); Promise.allSettled(videos.filter(v=>v!==source).map(v=>v.play())); }} }});
    source.addEventListener('pause', () => {{ if (!syncingTransport) {{ holdTransportSync(); videos.filter(v=>v!==source).forEach(v=>v.pause()); }} }});
    source.addEventListener('timeupdate', () => {{ if (source === videos[0]) {{ if (syncingSeek) confirmDesiredSeek(); else if (!source.seeking) videos.slice(1).forEach((v,index) => {{ if (Math.abs(v.currentTime-source.currentTime)>.12) {{ desiredSeekTime = source.currentTime; holdSeekSync(); seekOne(v,desiredSeekTime,index+1); }} }}); update(syncingSeek && desiredSeekTime !== null ? desiredSeekTime : source.currentTime); }} }});
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
      updateReviewStatus('İnceleme kaydedildi.');
    }});
  }});
  const reviewStatus = document.getElementById('review-status');
  const updateReviewStatus = message => {{
    if (!reviewStatus) return;
    const count = Object.keys(reviewSelections).length;
    reviewStatus.textContent = `${{message || 'Kayıtlı inceleme'}} · ${{count}} karar`;
  }};
  const exportButton = document.getElementById('export-review');
  if (exportButton) exportButton.onclick = () => {{
    const payload = {{schema_version:1, timeline_id:data.timeline_id, created_at:new Date().toISOString(), reviews:reviewSelections}};
    const link = document.createElement('a'); link.href = URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)], {{type:'application/json'}}));
    link.download = `tk3d-review-${{data.timeline_id}}.json`; document.body.appendChild(link); link.click(); link.remove();
    setTimeout(() => URL.revokeObjectURL(link.href), 1000); updateReviewStatus('JSON indirildi');
  }};
  const clearReview = document.getElementById('clear-review');
  if (clearReview) clearReview.onclick = () => {{
    Object.keys(reviewSelections).forEach(key => delete reviewSelections[key]); localStorage.removeItem(reviewKey);
    document.querySelectorAll('[data-review-event].selected').forEach(item => item.classList.remove('selected'));
    updateReviewStatus('İncelemeler temizlendi');
  }};
  const metricFilter = document.getElementById('metric-filter');
  if (metricFilter) metricFilter.addEventListener('input', () => {{
    const query = metricFilter.value.trim().toLocaleLowerCase('tr');
    const section = document.getElementById('wholebody-diagnostics');
    let visible = 0;
    section.querySelectorAll('.wb-movement-block').forEach(block => {{
      let blockMatches = 0;
      block.querySelectorAll('.wb-m').forEach(cell => {{
        const searchable = (cell.dataset.metricSearch || cell.textContent).toLocaleLowerCase('tr');
        const matches = !query || searchable.includes(query);
        cell.hidden = !matches; if (matches) {{ blockMatches += 1; visible += 1; }}
      }});
      block.hidden = Boolean(query) && blockMatches === 0;
      if (query && blockMatches > 0) block.classList.add('open');
    }});
    document.getElementById('metric-filter-status').textContent = query ? `${{visible}} eşleşen ölçüm` : 'Tüm ölçümler gösteriliyor';
  }});
  videos.forEach((video,index) => {{
    if (video.readyState >= 1) setHealth(index, 'ready', `Hazır · ${{video.duration.toFixed(2)}} sn`); else video.load();
    prepareSeekableVideo(video, index);
  }});
  window.addEventListener('beforeunload', () => reviewVideoObjectUrls.forEach(url => URL.revokeObjectURL(url)));
  updateReviewStatus();
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
    section = f'''<section class="section" id="wholebody-diagnostics"><h2>WholeBody-133 hata inceleme adayları ve tüm ölçümler</h2>
      <div class="notice"><div>🔎</div><div><strong>Skor yok; her aday videoda doğrulanmalı.</strong>
      <p>Dirsek, bilek, omuz-kalça rotasyonu, baş yönü, hikite, yumruk, eşzaman, fixation, ağırlık aktarımı, trajectory
      ve daha fazlası ölçülür. Yeşil = eşik içi, kırmızı = aday, gri = ölçülemedi, mavi = yalnız tanılı.</p></div></div>
      <p style="margin-bottom:12px">{int(coverage.get("measurable_metric_count", 0))}/{int(coverage.get("thresholded_metric_count", 0))} ölçülebilir ·
      {within_count} eşik içi · {candidate_count} aday ·
      kapsama kapısı: {"geçti" if coverage.get("coverage_gate_passed") else "başarısız"} · Accuracy: hesaplanmadı.
      Hareket başlığına tıkla → metrikleri aç/kapa.</p>
      <div class="toolbar" style="margin-bottom:12px"><input id="metric-filter" type="search" placeholder="Metrik ara…" aria-label="Metrik ara" style="min-width:260px;padding:9px 11px;border-radius:9px;border:1px solid #37627b;background:#0b1822;color:var(--text)"><span id="metric-filter-status" aria-live="polite">Tüm ölçümler gösteriliyor</span></div>
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
    evidence = metric.get("measurement_evidence") or {}

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

    sample_count = metric.get("sample_count")
    uncertainty = metric.get("uncertainty_95")
    scope = evidence.get("scope", "kanıt penceresi yok")
    detail_parts = [range_text, f"pencere: {scope}"]
    if sample_count is not None:
        detail_parts.append(f"örnek: {int(sample_count)}")
    if uncertainty is not None:
        detail_parts.append(f"%95 ±{float(uncertainty):.2f}")
    detail_text = " · ".join(part for part in detail_parts if part)
    seek = float(evidence.get("anchor_time_sec", 0.0))

    search_text = f"{label} {metric.get('metric_id', '')} {criterion_id}"
    return f'''<div class="wb-m {css_class}" data-metric-search="{_escape(search_text)}">
      <span class="wb-dot">{dot}</span>
      <span class="wb-m-name">{_escape(label)}</span>
      <span class="wb-m-val">{value_text} {_escape(unit)}</span>
      <span class="wb-m-range">{_escape(detail_text)} · <button type="button" class="anchor" data-seek="{seek:.6f}" style="padding:2px 6px">kanıta git</button></span>
    </div>'''


def _categorical_diagnostics_html(
    report: dict[str, Any] | None,
    fps: float,
) -> tuple[str, str]:
    if report is None:
        return '<div class="stat"><span>Hareket/duruş teşhisi</span><b>Yok</b></div>', ""
    if report.get("status") != "categorical_diagnostics_only":
        raise ScoringContractError("categorical diagnostics report status is invalid")
    summary = report.get("summary", {})
    checks = report.get("checks", [])
    if not isinstance(checks, list):
        raise ScoringContractError("categorical diagnostics checks must be a list")
    rows = "".join(_categorical_check_row(item, fps) for item in checks)
    mismatch_count = int(summary.get("mismatch_candidate_count", 0))
    stat = (
        '<div class="stat"><span>Yanlış hareket/duruş</span>'
        f'<b>{mismatch_count} aday · puan yok</b></div>'
    )
    section = f'''<section class="section"><h2>Yanlış hareket ve yanlış duruş teşhisi</h2>
      <div class="notice"><div>🧭</div><div><strong>Bu satırlar otomatik WT kesintisi değildir.</strong>
      <p>{_escape(report.get("interpretation", ""))}</p></div></div>
      <p style="margin-bottom:12px">{int(summary.get("check_count", 0))} kontrol · {mismatch_count} uyuşmazlık adayı ·
      {int(summary.get("consistent_count", 0))} beklenen profile uyumlu · {int(summary.get("ambiguous_count", 0))} belirsiz ·
      {int(summary.get("not_measurable_count", 0))} ölçülemedi · {int(summary.get("unsupported_count", 0))} desteklenmiyor.</p>
      <ul>{rows}</ul></section>'''
    return stat, section


def _categorical_check_row(check: dict[str, Any], fps: float) -> str:
    status = check.get("status", "unknown")
    status_labels = {
        "consistent": ("Beklenenle uyumlu", "diag-consistent"),
        "mismatch_candidate": ("Uyuşmazlık adayı", "diag-mismatch"),
        "ambiguous": ("Belirsiz", "diag-ambiguous"),
        "not_measurable": ("Ölçülemedi", ""),
        "unsupported": ("Desteklenmiyor", ""),
    }
    label, css_class = status_labels.get(status, (status, ""))
    evidence = check.get("evidence") or []
    details = "; ".join(
        f"{item.get('metric_id')}={_number(item.get('value'), '')} "
        f"(beklenen {_range_text(item.get('expected_range'))}, alternatif {_range_text(item.get('alternate_range'))})"
        for item in evidence
    ) or check.get("reason", "kanıt yok")
    seek = float(check.get("anchor_frame", 0)) / fps
    return f'''<li class="diag-row"><code>{_escape(check.get("movement_id"))} · {_escape(check.get("event_kind"))}</code>
      <span class="{css_class}">{_escape(label)}</span>
      <span>beklenen: {_escape(check.get("expected_label"))} · alternatif: {_escape(check.get("alternate_label"))}<br>{_escape(details)}</span>
      <button type="button" data-seek="{seek:.6f}">Kanıta git</button></li>'''


def _technical_conformance_html(
    report: dict[str, Any] | None,
    fps: float,
) -> tuple[str, str]:
    if report is None:
        return '<div class="stat"><span>Teknik uygunluk</span><b>Yok</b></div>', ""
    if report.get("status") != "technical_conformance_diagnostic_only":
        raise ScoringContractError("technical conformance report status is invalid")
    movements = report.get("movements")
    if not isinstance(movements, list):
        raise ScoringContractError("technical conformance movements must be a list")
    summary = report.get("summary", {})
    review_count = int(summary.get("review_required_count", 0))
    movement_count = int(summary.get("movement_count", 0))
    cards = "".join(_technical_movement_card(item, fps) for item in movements)
    stat = (
        '<div class="stat"><span>Teknik uygunluk</span>'
        f'<b>{review_count}/{movement_count} inceleme · puan yok</b></div>'
    )
    section = f'''<section class="section"><h2>M01–M06 hareket bazlı teknik uygunluk</h2>
      <div class="notice"><div>🧩</div><div><strong>Kimlik, teknik geometri ve kanıt kalitesi birlikte gösterilir.</strong>
      <p>{_escape(report.get("interpretation", ""))}</p></div></div>
      <p style="margin-bottom:12px">{movement_count} hareket · {int(summary.get("mismatch_candidate_count", 0))} hareket/duruş uyuşmazlığı ·
      {int(summary.get("review_candidate_count", 0))} teknik inceleme adayı · {int(summary.get("ambiguous_count", 0))} belirsiz ·
      {int(summary.get("consistent_within_measured_scope_count", 0))} ölçülen kapsamda uyumlu ·
      {int(summary.get("measurable_criterion_count", 0))}/{int(summary.get("expected_criterion_count", 0))} ölçüt ölçülebilir.</p>
      {cards}</section>'''
    return stat, section


def _technical_movement_card(movement: dict[str, Any], fps: float) -> str:
    status = movement.get("conformance_status", "unknown")
    status_labels = {
        "mismatch_candidate": ("Hareket/duruş uyuşmazlığı adayı", "wb-badge-warn"),
        "review_candidate": ("Teknik inceleme adayı", "wb-badge-warn"),
        "ambiguous": ("Belirsiz", "wb-badge-warn"),
        "consistent_within_measured_scope": ("Ölçülen kapsamda uyumlu", "wb-badge-ok"),
        "not_measurable": ("Ölçülemedi", "wb-badge-warn"),
    }
    status_label, badge_class = status_labels.get(status, (status, "wb-badge-warn"))
    coverage = movement.get("criterion_coverage") or {}
    confidence = float(movement.get("fused_evidence_confidence", 0.0))
    aspects = "".join(
        f'<span class="wb-badge {_technical_badge_class(item.get("status"))}">'
        f'{_escape(_technical_aspect_label(item.get("aspect_id")))}: '
        f'{_escape(_technical_status_label(item.get("status")))}</span>'
        for item in movement.get("aspects", [])
    )
    identities = "".join(
        f'''<li><code>{_escape(item.get("event_kind"))}</code><span>{_escape(_technical_status_label(item.get("status")))} ·
        beklenen {_escape(item.get("expected_label"))} · alternatif {_escape(item.get("alternate_label"))} ·
        birleşik güven %{float(item.get("fused_confidence", 0.0)) * 100:.0f}</span></li>'''
        for item in movement.get("identity_checks", [])
    ) or "<li><span>Hareket/duruş kimlik kontrolü yok.</span></li>"
    criteria = "".join(_technical_criterion_row(item) for item in movement.get("criteria", []))
    anchor_frame = int(movement.get("anchor_frame", 0))
    return f'''<article class="wb-movement-block" onclick="this.classList.toggle('open')">
      <div class="wb-movement-hdr">
        <div><span class="wb-mid">{_escape(movement.get("movement_id"))}</span><span class="wb-mname">{_escape(movement.get("display_name"))}</span></div>
        <div style="display:flex;gap:6px;align-items:center"><span class="wb-badge {badge_class}">{_escape(status_label)}</span>
        <button type="button" class="anchor" data-seek="{anchor_frame / fps:.6f}" onclick="event.stopPropagation()">Kanıta git</button></div>
      </div>
      <p>Birleşik kanıt güveni %{confidence * 100:.0f} · ölçülebilir {int(coverage.get("measurable_count", 0))}/{int(coverage.get("expected_count", 0))} ·
      eşikle değerlendirilebilir {int(coverage.get("threshold_evaluable_count", 0))} · {_escape(_technical_reason_label(movement.get("reason")))}</p>
      <div style="display:flex;gap:6px;flex-wrap:wrap;margin-top:10px">{aspects}</div>
      <div class="wb-collapse" style="margin-top:12px"><h3>Hareket ve duruş kimliği</h3><ul>{identities}</ul>
      <h3>Beklenen teknik ölçütler</h3><div class="wb-metrics-grid">{criteria}</div></div>
    </article>'''


def _technical_criterion_row(criterion: dict[str, Any]) -> str:
    status = criterion.get("status")
    confidence = criterion.get("evidence_confidence")
    confidence_text = "—" if confidence is None else f"%{float(confidence) * 100:.0f}"
    metrics = criterion.get("metrics") or []
    metric_text = "; ".join(
        f"{_escape(item.get('metric_id'))}={_number(item.get('value'), '')} {_escape(item.get('unit') or '')}"
        for item in metrics
    ) or "kanıt yok"
    return f'''<div class="wb-m {_technical_cell_class(status)}">
      <span class="wb-dot">{_technical_status_icon(status)}</span>
      <span class="wb-m-name">{_escape(_CRITERION_LABELS.get(criterion.get("criterion_id"), criterion.get("criterion_id")))}</span>
      <span class="wb-m-val">{_escape(_technical_status_label(status))}</span>
      <span class="wb-m-range">kanıt güveni {confidence_text} · {metric_text}</span>
    </div>'''


def _technical_status_label(status: Any) -> str:
    return {
        "consistent": "beklenenle uyumlu",
        "mismatch_candidate": "uyuşmazlık adayı",
        "review_candidate": "inceleme adayı",
        "ambiguous": "belirsiz",
        "boundary_uncertain": "sınır belirsiz",
        "within_screening_range": "eşik içinde",
        "consistent_within_measured_scope": "ölçülen kapsamda uyumlu",
        "measured_diagnostic_only": "ölçüldü, eşik yok",
        "not_measurable": "ölçülemedi",
        "unsupported": "desteklenmiyor",
    }.get(str(status), str(status))


def _technical_aspect_label(aspect_id: Any) -> str:
    return {
        "posture_and_stance": "Duruş ve postür",
        "technique_execution": "Teknik uygulama",
        "timing_and_control": "Zamanlama ve kontrol",
    }.get(str(aspect_id), str(aspect_id))


def _technical_reason_label(reason: Any) -> str:
    return {
        "inferred_technique_or_stance_identity_mismatch": "Hareket veya duruş kimliği uyuşmazlığı inceleme gerektiriyor.",
        "one_or_more_measured_criteria_outside_screening_range": "En az bir teknik ölçüt tarama aralığının dışında.",
        "identity_or_numeric_boundary_is_ambiguous": "Kimlik kanıtı veya sayısal sınır belirsiz.",
        "no_thresholded_criterion_was_evaluable": "Eşikle değerlendirilebilir teknik ölçüt yok.",
        "no_conflict_found_in_evaluable_screening_criteria": "Ölçülebilen tarama ölçütlerinde çelişki bulunmadı.",
        "movement_timeline_label_is_not_confirmed": "Hareket zaman çizelgesi etiketi doğrulanmış değil.",
    }.get(str(reason), str(reason))


def _technical_badge_class(status: Any) -> str:
    return "wb-badge-ok" if status in {"within_screening_range", "consistent"} else "wb-badge-warn"


def _technical_cell_class(status: Any) -> str:
    if status == "within_screening_range":
        return "wb-ok"
    if status == "review_candidate":
        return "wb-cand"
    if status in {"boundary_uncertain", "not_measurable"}:
        return "wb-nm"
    return "wb-diag"


def _technical_status_icon(status: Any) -> str:
    if status == "within_screening_range":
        return "&#10003;"
    if status == "review_candidate":
        return "&#9888;"
    if status == "boundary_uncertain":
        return "?"
    if status == "not_measurable":
        return "&mdash;"
    return "&#8505;"


def _technical_accuracy_html(report: dict[str, Any] | None) -> tuple[str, str]:
    if report is None:
        return "", ""
    summary = report.get("summary", {})
    candidates = int(summary.get("temporary_candidate_count", 0))
    rule_count = int(summary.get("rule_count", 0))
    stat = (
        '<div class="stat"><span>Teknik doğruluk envanteri</span>'
        f'<b>{rule_count}</b><small>{candidates} puansız aday</small></div>'
    )
    movement_rows: list[str] = []
    for movement in report.get("movements", []):
        item = movement.get("summary", {})
        movement_rows.append(
            "<tr>"
            f'<td>{_escape(movement.get("movement_id"))}</td>'
            f'<td>{_escape(movement.get("movement_label"))}</td>'
            f'<td>{int(item.get("applicable_rule_count", 0))}</td>'
            f'<td>{int(item.get("measured_rule_count", 0))}</td>'
            f'<td>{int(item.get("in_range_rule_count", 0))}</td>'
            f'<td>{int(item.get("temporary_candidate_count", 0))}</td>'
            f'<td>{int(item.get("blocked_count", 0))}</td>'
            f'<td>{int(item.get("unmeasurable_count", 0))}</td>'
            "</tr>"
        )
    table = "".join(movement_rows)
    section = f'''<section class="section" id="technical-accuracy-diagnostics">
      <h2>Kapsamlı teknik doğruluk teşhisleri · puan yok</h2>
      <p>Geçici mühendislik eşikleri yalnız inceleme adayı üretir. Baş yönelimi gerçek göz bakışı değildir; M07–M18 mevcut videoda ölçülmüş sayılmaz.</p>
      <div class="metric-table-wrap"><table class="metric-table"><thead><tr>
        <th>Hareket</th><th>Kontrat</th><th>Uygulanır</th><th>Ölçüldü</th><th>Aralıkta</th><th>Puansız aday</th><th>Bloke</th><th>Ölçülemez</th>
      </tr></thead><tbody>{table}</tbody></table></div>
    </section>'''
    return stat, section


def _presentation_diagnostics_html(report: dict[str, Any] | None) -> tuple[str, str]:
    if report is None:
        return '<div class="stat"><span>Presentation proxy</span><b>Yok</b></div>', ""
    if report.get("status") != "presentation_diagnostic_only":
        raise ScoringContractError("Presentation diagnostics report status is invalid")
    components = report.get("components", {})
    requested = sum(int(item.get("requested_metric_count", 0)) for item in components.values())
    measurable = sum(int(item.get("measurable_metric_count", 0)) for item in components.values())
    cards = "".join(
        _presentation_component_card(component_id, component)
        for component_id, component in components.items()
    )
    stat = (
        '<div class="stat"><span>Presentation proxy</span>'
        f'<b>{measurable}/{requested} · skor yok</b></div>'
    )
    section = f'''<section class="section"><h2>Presentation kinematik göstergeleri</h2>
      <div class="notice"><div>⚡</div><div><strong>Hakem kalibrasyonu yok; toplam puan null.</strong>
      <p>{_escape(report.get("interpretation", ""))}</p></div></div>
      <div class="proxy-grid">{cards}</div></section>'''
    return stat, section


def _automatic_segmentation_html(
    report: dict[str, Any] | None,
    fps: float,
) -> tuple[str, str]:
    if report is None:
        return '<div class="stat"><span>Otomatik segment/faz</span><b>Yok</b></div>', ""
    summary = report.get("summary", {})
    comparison = report.get("reference_comparison", {})
    movement_rows = comparison.get("movements", [])
    if not isinstance(movement_rows, list):
        raise ScoringContractError("automatic segmentation comparison movements must be a list")
    proposals = {
        segment.get("movement_id"): segment
        for segment in report.get("segments", [])
        if isinstance(segment, dict)
    }
    rows = "".join(
        _automatic_segmentation_row(item, proposals.get(item.get("movement_id")), fps)
        for item in movement_rows
    ) or '<tr><td colspan="7">Karşılaştırılabilir otomatik segment bulunmadı.</td></tr>'
    selected = int(summary.get("selected_movement_count", 0))
    expected = int(summary.get("expected_movement_count", 0))
    anchor_mae = summary.get("phase_anchor_mae_frames")
    stat = (
        '<div class="stat"><span>Otomatik segment/faz</span>'
        f'<b>{selected}/{expected} · {_number(anchor_mae, " kare")}</b></div>'
    )
    section = f'''<section class="section" id="automatic-segmentation"><h2>Otomatik hareket ve faz sınırı doğrulaması</h2>
      <div class="notice"><div>⏱</div><div><strong>Onaylı timeline değiştirilmedi; bu bölüm puan üretmez.</strong>
      <p>{_escape(report.get("interpretation", ""))}</p></div></div>
      <p style="margin-bottom:12px">{int(summary.get("detected_candidate_count", 0))} hareket kümesi bulundu; {selected}/{expected} hareket seçildi ·
      başlangıç MAE {_number(summary.get("start_boundary_mae_frames"), " kare")} · bitiş MAE {_number(summary.get("end_boundary_mae_frames"), " kare")} ·
      faz ankrajı MAE {_number(anchor_mae, " kare")} ({_number(comparison.get("summary", {}).get("phase_anchor_mae_sec"), " sn")}) ·
      en büyük faz hatası {_number(summary.get("phase_anchor_max_error_frames"), " kare")}.</p>
      <div class="metric-table-wrap"><table class="metric-table"><thead><tr><th>Hareket</th><th>Oto başlangıç</th><th>Referans başlangıç</th>
      <th>Başlangıç farkı</th><th>Oto fixation</th><th>Fixation farkı</th><th>Kanıt</th></tr></thead><tbody>{rows}</tbody></table></div>
    </section>'''
    return stat, section


def _automatic_segmentation_row(
    comparison: dict[str, Any],
    proposal: dict[str, Any] | None,
    fps: float,
) -> str:
    movement_id = comparison.get("movement_id")
    if proposal is None or comparison.get("status") != "compared":
        return f'<tr><td><code>{_escape(movement_id)}</code></td><td colspan="6">Algılanmadı</td></tr>'
    start = comparison.get("start", {})
    fixation = comparison.get("phases", {}).get("fixation", {})
    fixation_frame = proposal.get("anchors", {}).get("fixation")
    seek = 0.0 if fixation_frame is None else float(fixation_frame) / fps
    fixation_delta = fixation.get("delta_frames")
    fixation_text = "—" if fixation_delta is None else f"{int(fixation_delta):+d} kare"
    fixation_frame_text = "—" if fixation_frame is None else str(int(fixation_frame))
    return f'''<tr><td><code>{_escape(movement_id)}</code></td>
      <td>{int(start.get("proposed_frame", 0))}</td><td>{int(start.get("reference_frame", 0))}</td>
      <td>{int(start.get("delta_frames", 0)):+d} kare</td>
      <td>{fixation_frame_text}</td><td>{fixation_text}</td>
      <td><button type="button" data-seek="{seek:.6f}">Oto fixation</button></td></tr>'''


def _presentation_component_card(component_id: str, component: dict[str, Any]) -> str:
    metric_rows = "".join(
        f'''<li><code>{_escape(metric_id)}</code><span>medyan {_number(metric.get("median"), "")} {_escape(metric.get("unit") or "")} ·
        IQR {_number(metric.get("interquartile_range"), "")} · n={int(metric.get("sample_count", 0))}</span></li>'''
        for metric_id, metric in component.get("metrics", {}).items()
    ) or "<li>Ölçüm yok.</li>"
    return f'''<article class="proxy-card"><h3>{_escape(component_id)}</h3><ul>{metric_rows}</ul></article>'''


def _range_text(value: Any) -> str:
    if not isinstance(value, list) or len(value) != 2:
        return "—"
    return f"{_number(value[0], '')}–{_number(value[1], '')}"
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
      <div class="toolbar" style="margin-bottom:10px"><button type="button" id="export-review">İnceleme kararlarını JSON indir</button><button type="button" id="clear-review">Kayıtlı incelemeleri temizle</button><span id="review-status" aria-live="polite">Kayıtlı inceleme · 0 karar</span></div>
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


def _video_mime_type(source: str) -> str:
    path = source.partition("?")[0].lower()
    if path.endswith(".mp4"):
        return "video/mp4"
    if path.endswith(".webm"):
        return "video/webm"
    return "video/x-msvideo"


def _percent(value: Any) -> str:
    return "—" if value is None else f"%{float(value) * 100:.1f}"


def _number(value: Any, suffix: str) -> str:
    return "—" if value is None else f"{float(value):.2f}{suffix}"
