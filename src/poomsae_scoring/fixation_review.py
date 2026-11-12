"""Self-contained review of automatic fixation proposals, independent of scoring."""
from __future__ import annotations

import html
import json
from typing import Any

from src.poomsae_scoring.report_style import REPORT_CSS


def fixation_review_section(report: dict[str, Any] | None) -> str:
    if report is None or "fixation_candidates" not in report:
        return ""
    labels = {"hold_observed": "Kararlı aralık gözlendi — öneri",
              "components_disagree": "Bileşenler birlikte sabitlenmedi",
              "insufficient_evidence": "Güvenilir 3B hareket kanıtı yetersiz",
              "recording_truncated": "Kayıt sonu / gözlem süresi yetersiz",
              "not_observed": "Sabitleme gözlenmedi"}
    names = {"left_hand": "Sol el", "right_hand": "Sağ el", "feet": "Ayaklar",
             "pelvis": "Pelvis", "torso": "Gövde", "shoulder_rotation": "Omuz dönüşü",
             "pelvis_rotation": "Pelvis dönüşü"}
    movement_options = '<option value="">Hareket bilinmiyor / hazırlık</option>' + ''.join(
        f'<option value="{html.escape(str(movement_id), quote=True)}">{html.escape(str(movement_id))}</option>'
        for movement_id in report.get("selected_scope_movement_ids", [])
    )
    rows = []
    diagnostics = {item["candidate_episode_id"]: item for item in report.get("fixation_timing_diagnostics", [])}
    for event in report["fixation_candidates"]:
        key = str(int(event["candidate_episode_id"]))
        frame = event["fixation_frame"]
        suggested = frame if frame is not None else event["completion_candidate_frame"]
        seek = suggested if suggested is not None else event["peak_frame"]
        components = "".join(
            f'<li>{names.get(name, html.escape(name))}: {_frame_link(value["settled_frame"], report["fps"])} · '
            f'gözlenebilirlik %{value["valid_ratio"] * 100:.0f}</li>'
            for name, value in event["components"].items()
        )
        later = "" if event["later_motion_frame"] is None else f'<p>Sonradan hareket: {int(event["later_motion_frame"])}. kare. İlk aday korundu.</p>'
        value = "" if suggested is None else str(int(suggested))
        hold = "yok / belirsiz" if frame is None else f'{_frame_link(frame, report["fps"])} – {event["hold_end_frame"]}. kare ({event["hold_duration_sec"]:.3f} sn)'
        completion = _frame_link(event["completion_candidate_frame"], report["fps"])
        detail = _timing_diagnostics_html(diagnostics.get(event["candidate_episode_id"]), report["fps"], names)
        rows.append(f'''<article class="proxy-card" id="fixation-candidate-{key}" data-fixation-card="{key}">
          <h3>Aday {int(key) + 1} · {event['start_frame']}–{event['end_frame']}. kare</h3>
          <p><strong>{labels[event['status']]}</strong></p>
          <p>Otomatik hareketten çıkış adayı: {completion} (sabitleme değildir)</p>
          <p>Otomatik ortak kararlı aralık: {hold}</p>{later}{detail}
          <button type="button" data-seek="{max(0, seek / report['fps'] - .5):.6f}">Öncesini videoda aç</button>
          <details><summary>Bileşen kanıtını göster</summary><ul>{components or '<li>Yeterli kanıt yok.</li>'}</ul>
          <p>Bileşen kareleri ilk düşük hareket aralıklarıdır; tekniğin doğru yapıldığını göstermez.</p></details>
          <label>İnceleme karesi <input type="number" data-fixation-frame="{key}" min="0" max="{report['frame_count'] - 1}" step="1" value="{value}"></label>
          <label>Hareket kimliği (isteğe bağlı) <select data-fixation-movement="{key}">{movement_options}</select></label>
          <button type="button" data-fixation-seek="{key}">Kareye git</button>
          <button type="button" data-fixation-capture="{key}">Videodaki kareyi al</button>
          <button type="button" data-fixation-step="{key}" data-step="-1">−1 kare</button>
          <button type="button" data-fixation-step="{key}" data-step="1">+1 kare</button>
          <label>Not <input type="text" data-fixation-note="{key}" maxlength="2000"></label>
          <div class="sync-controls">
            <button type="button" data-fixation-decision="{key}" data-decision="accepted" {'disabled' if frame is None else ''}>Otomatik sabitlemeyi kabul et</button>
            <button type="button" data-fixation-decision="{key}" data-decision="corrected">Seçtiğim kareyi kaydet</button>
            <button type="button" data-fixation-decision="{key}" data-decision="no_hold">Sabitleme yok</button>
            <button type="button" data-fixation-decision="{key}" data-decision="uncertain">Belirsiz</button>
          </div><p data-fixation-saved="{key}" role="status">Henüz incelenmedi.</p></article>''')
    detector = report.get("detector", {}).get("fixation", {})
    empty = "Yeterli hareket adayı bulunamadı. Bu sonuç sıfır hata veya doğru teknik anlamına gelmez."
    if detector.get("blocking_reason"):
        empty += " Sabitleme kanıtı üretilemedi: " + html.escape(detector["blocking_reason"])
    return f'''<section class="section" id="fixation-review"><h2>Otomatik sabitleme incelemesi</h2>
      <div class="notice"><div>⏱</div><div><strong>Elle işaretler kesin doğru kabul edilmez.</strong>
      <p>Adaylar elle etiketlerden bağımsızdır. Yavaşlama adayı, gözlenen sabitleme ve görüntü yetersizliği ayrı gösterilir.
      Bunlar bütün kaydın hareket kümeleridir; hazırlık hareketleri de bulunabilir. Hareket kimliği otomatik doğrulanmış değildir.
      İlk aday değişmez; incelemen ayrı kaydedilir ve puanlama onayı oluşturmaz.</p></div></div>
      <label>İnceleyen adı/kodu <input id="fixation-reviewer" type="text" autocomplete="off"></label>
      <div class="sync-controls"><button id="export-fixation-review" type="button">Sabitleme incelemesini JSON indir</button>
      <label>İnceleme JSON yükle <input id="import-fixation-review" type="file" accept="application/json,.json"></label></div>
      <p id="fixation-review-status" role="status">Otomatik öneriler insan incelemesi değildir.</p>
      <div class="proxy-grid">{''.join(rows) or f'<p>{empty}</p>'}</div></section>'''


def _frame_link(frame: int | None, fps: float) -> str:
    if frame is None:
        return "belirlenemedi"
    return f'<button type="button" data-seek="{frame / fps:.6f}">{frame / fps:.2f} sn · {frame}. kare</button>'


def _timing_diagnostics_html(detail: dict | None, fps: float, names: dict) -> str:
    if detail is None:
        return ""
    messages = {
        "no_supported_hold": "Ortak sabitleme belirlenemediği için başlangıç hassaslaştırılmadı.",
        "blocked_quality_metadata": "Kaynak kalite bilgisi ek zaman kontrolüne izin vermiyor.",
        "not_confirmed_without_speed_smoothing": "İlk önerilen aralık, hız yumuşatması olmadan süre kontrolünü geçmedi; zaman belirsiz.",
    }
    status = detail["status"]
    if status == "supported_without_speed_smoothing":
        message = (f'Hız yumuşatması olmadan desteklenen başlangıç: {_frame_link(detail["speed_unsmoothed_onset_frame"], fps)} '
                   f'(ilk öneriye göre +{detail["shift_frames"]} kare).')
    else:
        message = messages[status]
    missing = ", ".join(names.get(name, html.escape(name)) for name in detail["missing_components"])
    hands = "".join(
        f'<li>{names[name]}: kendi hızının ölçülebildiği oran %{values["absolute_valid_ratio"] * 100:.0f}; '
        f'kalçaya göre hızın ölçülebildiği oran %{values["pelvis_relative_valid_ratio"] * 100:.0f}.</li>'
        for name, values in detail["hands"].items()
    )
    return (f'<p>{message}</p>' + (f'<p>Sürekli ölçüm eksik: {missing}.</p>' if missing else "")
            + f'<details><summary>Ek zaman ve el kontrolü</summary><ul>{hands}</ul>'
            '<p>Oranlar hareketin sabitliğini değil ölçümün mevcut oluşunu gösterir. Kalça ölçümü eksikse '
            'elin kendi hızı tek başına ortak sabitlemeyi doğrulamaz. Ek kontrol kaynak 3B poza dayanır; '
            'bağımsız görüntü doğrulaması değildir. İlk otomatik öneri korunur.</p></details>')


FIXATION_REVIEW_JS = r"""
  const fixationReport = data.fixation_report;
  if (fixationReport && document.getElementById('fixation-review')) {
    const fixationBinding = data.fixation_review_binding;
    const proposals = new Map(fixationReport.fixation_candidates.map(row => [String(row.candidate_episode_id), row]));
    const fixationKey = `tk3d-fixation-v1-${fixationBinding.review_id}`;
    const status = document.getElementById('fixation-review-status');
    let decisions = {};
    const frameInput = id => document.querySelector(`[data-fixation-frame="${id}"]`);
    const noteInput = id => document.querySelector(`[data-fixation-note="${id}"]`);
    const movementInput = id => document.querySelector(`[data-fixation-movement="${id}"]`);
    const movementIds = new Set(fixationReport.selected_scope_movement_ids);
    const validFrame = frame => Number.isInteger(frame) && frame >= 0 && frame < fixationReport.frame_count;
    const readFrame = id => frameInput(id).value.trim() === '' ? null : Number(frameInput(id).value);
    const validDecisions = entries => entries && typeof entries === 'object' && !Array.isArray(entries) &&
      Object.entries(entries).every(([id, item]) => proposals.has(id) && item &&
        ['accepted','corrected','no_hold','uncertain'].includes(item.decision) &&
        typeof item.reviewer === 'string' && item.reviewer.trim() &&
        typeof item.reviewed_at === 'string' && item.reviewed_at.trim() &&
        (item.movement_id === null || movementIds.has(item.movement_id)) &&
        (['accepted','corrected'].includes(item.decision) ? validFrame(item.frame) : item.frame === null) &&
        (item.decision !== 'accepted' || item.frame === proposals.get(id).fixation_frame));
    const paint = () => {
      const labels = {accepted:'Öneri kabul edildi', corrected:'Seçilen kare kaydedildi', no_hold:'Sabitleme yok', uncertain:'Belirsiz'};
      proposals.forEach((proposal,id) => {
        const item = decisions[id];
        document.querySelector(`[data-fixation-saved="${id}"]`).textContent = item ?
          `${labels[item.decision]}${item.frame === null ? '' : ` · kare ${item.frame}`} · ${item.reviewer}` : 'Henüz incelenmedi.';
        if (item) { if (item.frame !== null) frameInput(id).value = item.frame; noteInput(id).value = item.note || ''; movementInput(id).value = item.movement_id || ''; }
      });
    };
    const persist = () => {
      paint();
      try { localStorage.setItem(fixationKey, JSON.stringify(decisions)); status.textContent = `${Object.keys(decisions).length} inceleme bu tarayıcıya kaydedildi. Kalıcı kopya için JSON indirin.`; }
      catch (error) { status.textContent = 'Tarayıcı kaydı kullanılamıyor; JSON indirerek saklayın.'; }
    };
    try { const stored = JSON.parse(localStorage.getItem(fixationKey) || '{}'); if (validDecisions(stored)) decisions = stored; } catch (error) { /* No prior annotation. */ }
    paint();
    document.querySelectorAll('[data-fixation-decision]').forEach(button => button.onclick = () => {
      const id = button.dataset.fixationDecision, decision = button.dataset.decision;
      const reviewer = document.getElementById('fixation-reviewer').value.trim();
      if (!reviewer) { status.textContent = 'Önce inceleyen adını/kodunu girin.'; return; }
      const frame = decision === 'accepted' ? proposals.get(id).fixation_frame : decision === 'corrected' ? readFrame(id) : null;
      if (['accepted','corrected'].includes(decision) && !validFrame(frame)) { status.textContent = 'Kaynak kayıt içinde geçerli bir kare seçin.'; return; }
      decisions[id] = {decision, frame, movement_id:movementInput(id).value || null, reviewer, reviewed_at:new Date().toISOString(), note:noteInput(id).value};
      persist();
    });
    document.querySelectorAll('[data-fixation-seek]').forEach(button => button.onclick = () => {
      const frame = readFrame(button.dataset.fixationSeek); if (validFrame(frame)) { pauseAll(); seekAll(frame / data.fps); }
    });
    document.querySelectorAll('[data-fixation-capture]').forEach(button => button.onclick = () => {
      if (videos.length && validFrame(Math.round(videos[0].currentTime * data.fps))) frameInput(button.dataset.fixationCapture).value = Math.round(videos[0].currentTime * data.fps);
    });
    document.querySelectorAll('[data-fixation-step]').forEach(button => button.onclick = () => {
      const id = button.dataset.fixationStep, current = readFrame(id);
      if (current === null) { status.textContent = 'Önce bir kare girin veya videodaki kareyi alın.'; return; }
      const frame = current + Number(button.dataset.step);
      if (validFrame(frame)) { frameInput(id).value = frame; pauseAll(); seekAll(frame / data.fps); }
    });
    document.getElementById('export-fixation-review').onclick = () => {
      const payload = {schema_version:1, kind:'fixation_review', binding:fixationBinding, decisions};
      const link = document.createElement('a'); const url = URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)], {type:'application/json'}));
      link.href = url; link.download = `tk3d-fixation-${fixationBinding.review_id}.json`; link.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000); status.textContent = 'İnceleme JSON indirildi; otomatik öneriler ve puanlama değiştirilmedi.';
    };
    document.getElementById('import-fixation-review').onchange = async event => {
      try {
        if (!event.target.files.length) return;
        const payload = JSON.parse(await event.target.files[0].text());
        const canonical = value => JSON.stringify(value, (key,item) => item && typeof item === 'object' && !Array.isArray(item) ? Object.fromEntries(Object.entries(item).sort(([a],[b]) => a.localeCompare(b))) : item);
        if (payload.schema_version !== 1 || payload.kind !== 'fixation_review' || canonical(payload.binding) !== canonical(fixationBinding) || !validDecisions(payload.decisions)) throw new Error('Bu kayıt ve öneri sürümüyle eşleşmeyen veya geçersiz inceleme.');
        decisions = payload.decisions; persist();
      } catch (error) { status.textContent = `Yüklenemedi: ${error.message}`; }
      finally { event.target.value = ''; }
    };
  }
"""


def build_fixation_review_html(report: dict, videos: dict[str, str], binding: dict) -> str:
    """Standalone view works without a manually labelled movement timeline."""
    data = {"fixation_report": report, "fixation_review_binding": binding, "fps": report["fps"]}
    encoded = json.dumps(data, ensure_ascii=False, allow_nan=False).replace("</", "<\\/")
    camera_html = ''.join(f'<article><h3>{html.escape(name)}</h3><video controls preload="metadata" src="{html.escape(url, quote=True)}" style="width:100%"></video></article>' for name, url in videos.items())
    return f'''<!doctype html><html lang="tr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    <title>TK3D — Otomatik sabitleme incelemesi</title><style>{REPORT_CSS}</style></head><body><main>
    <header><h1>Otomatik sabitleme incelemesi</h1></header><section class="section"><div class="videos">{camera_html or '<p>Kamera videosu eklenmedi; kare numaraları görüntü doğrulaması değildir.</p>'}</div>
    <div class="sync-controls"><button id="fixation-play">Tüm kameraları oynat</button><button id="fixation-pause">Duraklat</button></div>
    <p id="video-status" role="status"></p></section>{fixation_review_section(report)}</main>
    <script id="review-data" type="application/json">{encoded}</script><script>(() => {{
    const data = JSON.parse(document.getElementById('review-data').textContent);
    const videos = [...document.querySelectorAll('video')];
    const pauseAll = () => videos.forEach(video => video.pause());
    let syncingUntil = 0;
    const seekAll = time => {{ syncingUntil = performance.now() + 500; videos.forEach(video => {{
      const seek = () => {{ video.currentTime = Math.max(0, Math.min(time, Number.isFinite(video.duration) ? video.duration : time)); }};
      if (video.readyState >= 1) seek(); else video.addEventListener('loadedmetadata', seek, {{once:true}});
    }}); }};
    videos.forEach(video => {{
      video.addEventListener('seeking', () => {{ if (performance.now() > syncingUntil) seekAll(video.currentTime); }});
      video.addEventListener('error', () => {{ document.getElementById('video-status').textContent = 'Kamera videosu açılamadı. MP4/WebM dosyasını kontrol edin.'; }});
    }});
    if (videos.length) videos[0].addEventListener('timeupdate', () => {{
      if (!videos[0].paused && performance.now() > syncingUntil) videos.slice(1).forEach(video => {{
        if (Math.abs(video.currentTime - videos[0].currentTime) > .1) {{ syncingUntil = performance.now() + 500; video.currentTime = videos[0].currentTime; }}
      }});
    }});
    document.querySelectorAll('[data-seek]').forEach(button => button.onclick = () => {{ pauseAll(); seekAll(Number(button.dataset.seek)); }});
    document.getElementById('fixation-pause').onclick = pauseAll;
    document.getElementById('fixation-play').onclick = async () => {{
      const results = await Promise.allSettled(videos.map(video => video.play()));
      document.getElementById('video-status').textContent = results.some(result => result.status === 'rejected') ? 'En az bir kamera oynatılamadı; tarayıcı uyumlu MP4 kullanın.' : 'Kameralar oynatılıyor.';
    }};
    {FIXATION_REVIEW_JS}
    }})();</script></body></html>'''
