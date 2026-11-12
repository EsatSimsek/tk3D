"""Independent full-recording annotation UI; no proposals or legacy labels are inputs."""
from __future__ import annotations

import html
import json

from .fixation_validation import annotation_template
from .report_style import REPORT_CSS


def build_independent_annotation_html(binding: dict, videos: dict[str, str]) -> str:
    data = json.dumps({"template": annotation_template(binding)}, ensure_ascii=False, allow_nan=False).replace("</", "<\\/")
    cameras = "".join(f'<article><h3>{html.escape(name)}</h3><video controls preload="metadata" src="{html.escape(url, quote=True)}" style="width:100%"></video></article>' for name, url in videos.items())
    options = '<option value="completion">Tekniğin tamamlanması</option><option value="hold_onset">Kararlı duruş başlangıcı</option><option value="hold_end">Kararlı duruş bitişi</option><option value="no_hold">Sabitleme gözlenmeyen aralık</option><option value="uncertain">Görüntü / karar belirsiz aralık</option>'
    return f'''<!doctype html><html lang="tr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TK3D — Bağımsız sabitleme işaretleme</title><style>{REPORT_CSS}
input,select{{max-width:100%;padding:.5rem}} .fields{{display:flex;gap:1rem;flex-wrap:wrap}} label{{display:block}} table{{width:100%}}</style></head><body><main>
<header><h1>Bağımsız sabitleme işaretleme</h1><p>Kaydın tamamını iki kamerada inceleyin. Kareler 0'dan başlar. Emin olmadığınız anı tek kareye zorlamadan bir aralıkla belirtin.</p></header>
<section class="section"><h2>İşaretleme tanımları</h2><p><b>Tamamlanma:</b> tekniğin hedef son konuma ulaştığı an. <b>Kararlı duruş başlangıcı:</b> el, ayak ve gövdenin yerleştiği gözlenen aralığın başlangıcı. <b>Bitiş:</b> bu duruşun bozulduğu veya sonraki hareketin başladığı an.</p>
<p>Bunlar inceleme protokolünün çalışma tanımlarıdır; resmî puanlama ölçütü değildir. Teknik tamamlanıp sabit duruş oluşmayabilir. Görünmeyen vücut bölümü için belirsiz aralık kullanın. Aynı olayı iki kez eklemeyin.</p>
<div class="fields"><label>İnceleyen kodu <input id="reviewer" autocomplete="off"></label><label>İşaretleme turu <input id="annotation_round" value="tur-1" autocomplete="off"></label><label>Sporcu kodu <input id="athlete_id" autocomplete="off"></label><label>Oturum kodu <input id="session_id" autocomplete="off"></label></div>
<p>Tek inceleyen varsa aynı kaydı farklı zamanlarda tur-1 ve tur-2 olarak işaretleyebilirsiniz. Yeni turu önceki işaretleri açmadan boş başlatın; inceleyen, sporcu ve oturum kodları aynı kalsın. Bu karşılaştırma kişisel tekrar tutarlılığını ölçer.</p>
<label><input id="blinded_to_predictions" type="checkbox">Bu işaretleme sırasında otomatik önerileri veya önceki işaretleri referans almadım.</label><p>Bu beyan kimlik veya bağımsız uzman doğrulaması değildir. Önceki işaretleri kullandıysanız kutuyu boş bırakın.</p></section>
<section class="section"><div class="videos">{cameras}</div><div class="sync-controls"><button id="play">Kameraları oynat</button><button id="pause">Duraklat</button><button id="back">−1 kare</button><button id="next">+1 kare</button></div>
<label>Kareye git <input id="seek-frame" type="number" min="0" max="{binding['frame_count'] - 1}" value="0"><button id="seek">Git</button></label>
<input id="scrub" type="range" min="0" max="{binding['frame_count'] - 1}" value="0" style="width:100%"><p id="video-status" role="status">Görüntülenen kare bilgisi bekleniyor.</p></section>
<section class="section"><h2>Olay ekle veya düzenle</h2><div class="fields"><label>Olay <select id="event-kind">{options}</select></label><label>İlk olası kare <input id="event-start" type="number" min="0" max="{binding['frame_count'] - 1}" value="0"><button id="capture-start">Görüntülenen kareyi al</button></label>
<label>Son olası kare <input id="event-end" type="number" min="0" max="{binding['frame_count'] - 1}" value="0"><button id="capture-end">Görüntülenen kareyi al</button></label><label>Not <input id="event-note" maxlength="2000"></label></div><button id="save-event">Olayı kaydet</button><button id="cancel-edit">Düzenlemeyi bırak</button>
<div class="metric-table-wrap"><table><thead><tr><th>Olay</th><th>Kare aralığı</th><th>Not</th><th>İşlem</th></tr></thead><tbody id="events"></tbody></table></div></section>
<section class="section"><h2>İnceleme kapsamı ve kayıt</h2><div class="fields"><label>İncelenen ilk kare <input id="coverage-start" type="number" value="0" min="0" max="{binding['frame_count'] - 1}"></label><label>İncelenen son kare <input id="coverage-end" type="number" value="{binding['frame_count'] - 1}" min="0" max="{binding['frame_count'] - 1}"></label></div>
<label><input id="coverage_complete" type="checkbox">Yukarıdaki aralığın tamamını inceledim; olay olmayan bölümleri de kontrol ettim.</label>
<p>Bu kutu işaretlenene kadar kayıt taslak sayılır ve başarı metriklerine girmez. Belirsiz işaretlenen kareler değerlendirmeden çıkarılır ve miktarı raporlanır.</p>
<div class="sync-controls"><button id="export">İşaretleme JSON indir</button><button id="load-local">Bu inceleyen ve turun kaydını yükle</button><button id="new-review">Yeni boş inceleme / tur başlat</button><label>JSON yükle <input id="import" type="file" accept="application/json,.json"></label></div><p id="status" role="status">Kalıcı kopya için JSON indirin. İnceleme puanlama onayı oluşturmaz.</p></section>
</main><script id="annotation-data" type="application/json">{data}</script><script>{ANNOTATION_JS}</script></body></html>'''


ANNOTATION_JS = r"""(() => {
  const template = JSON.parse(document.getElementById('annotation-data').textContent).template;
  const binding = template.binding, fps = binding.fps, count = binding.frame_count;
  const $ = id => document.getElementById(id);
  const videos = [...document.querySelectorAll('video')];
  const kinds = {completion:'Tekniğin tamamlanması',hold_onset:'Kararlı duruş başlangıcı',hold_end:'Kararlı duruş bitişi',no_hold:'Sabitleme yok',uncertain:'Belirsiz'};
  let events = [], editing = null, ownerReviewer = null, serial = 0, presentedFrame = null, syncingUntil = 0;
  const frame = value => Number.isInteger(value) && value >= 0 && value < count;
  const canonical = value => JSON.stringify(value, (key,item) => item && typeof item === 'object' && !Array.isArray(item) ? Object.fromEntries(Object.entries(item).sort(([a],[b])=>a.localeCompare(b))) : item);
  const readInteger = id => $(id).value.trim() === '' ? null : Number($(id).value);
  const owner = (reviewer,round) => JSON.stringify([reviewer.trim(),round.trim()]);
  function payload() {
    if(events.length && ownerReviewer && owner($('reviewer').value,$('annotation_round').value)!==ownerReviewer) throw Error('İnceleyen veya tur değişti. Önce kendi kodlarınıza dönün veya yeni boş inceleme başlatın.');
    const data = {...template, binding, events, reviewed_at:new Date().toISOString(),
      coverage:{start_frame:readInteger('coverage-start'),end_frame:readInteger('coverage-end')}};
    ['reviewer','annotation_round','athlete_id','session_id'].forEach(key=>data[key]=$(key).value.trim());
    ['coverage_complete','blinded_to_predictions'].forEach(key=>data[key]=$(key).checked);
    return data;
  }
  function validate(data) {
    if (!data || canonical(Object.keys(data).sort()) !== canonical(Object.keys(template).sort()) || data.schema_version !== template.schema_version || data.kind !== template.kind || canonical(data.binding) !== canonical(binding)) throw Error('Kayıt veya protokol eşleşmiyor.');
    ['reviewer','annotation_round','athlete_id','session_id','reviewed_at'].forEach(key=>{if(typeof data[key] !== 'string' || !data[key].trim()) throw Error('İnceleyen, tur, sporcu ve oturum kodlarını doldurun.');});
    if (typeof data.coverage_complete !== 'boolean' || typeof data.blinded_to_predictions !== 'boolean') throw Error('Geçersiz inceleme beyanı.');
    const c=data.coverage;
    if (!c || canonical(Object.keys(c).sort()) !== canonical(['end_frame','start_frame']) || !frame(c.start_frame) || !frame(c.end_frame) || c.start_frame>c.end_frame || !Array.isArray(data.events)) throw Error('Geçersiz inceleme kapsamı.');
    const ids=new Set();
    for (const e of data.events) {
      if (!e || canonical(Object.keys(e).sort()) !== canonical(['end_frame','event_id','kind','note','start_frame']) || typeof e.event_id!=='string' || !e.event_id.trim() || ids.has(e.event_id) || !Object.hasOwn(kinds,e.kind) || typeof e.note!=='string' || !frame(e.start_frame) || !frame(e.end_frame) || e.start_frame<c.start_frame || e.end_frame>c.end_frame || e.start_frame>e.end_frame) throw Error('Geçersiz veya kapsam dışı olay.');
      ids.add(e.event_id);
    }
    for(let i=0;i<data.events.length;i++) for(let j=i+1;j<data.events.length;j++) {
      const a=data.events[i],b=data.events[j], k=new Set([a.kind,b.kind]);
      if (Math.max(a.start_frame,b.start_frame)<=Math.min(a.end_frame,b.end_frame) && (a.kind===b.kind || k.has('uncertain') || (k.has('no_hold') && (k.has('hold_onset') || k.has('hold_end'))))) throw Error('Üst üste gelen aynı veya çelişen olaylar var.');
    }
    return data;
  }
  function paint() {
    $('events').replaceChildren();
    [...events].sort((a,b)=>a.start_frame-b.start_frame).forEach(e=>{
      const row=document.createElement('tr');
      [kinds[e.kind],`${e.start_frame}–${e.end_frame}`,e.note].forEach(value=>{const cell=document.createElement('td');cell.textContent=value;row.appendChild(cell);});
      const cell=document.createElement('td');
      for(const [label,action] of [['Göster','show'],['Düzenle','edit'],['Sil','delete']]) { const button=document.createElement('button');button.textContent=label;button.dataset.action=action;button.dataset.eventId=e.event_id;cell.appendChild(button); }
      row.appendChild(cell);$('events').appendChild(row);
    });
  }
  const key = (reviewer,round) => `tk3d-independent-v2-${binding.recording_id}-${encodeURIComponent(owner(reviewer,round))}`;
  function persist() {
    paint();
    try { const data=validate(payload());localStorage.setItem(key(data.reviewer,data.annotation_round),JSON.stringify(data));$('status').textContent=`${events.length} olay bu tur için tarayıcıya kaydedildi. Kalıcı kopya için JSON indirin.`; }
    catch(error) { $('status').textContent=error.message+' Kalıcı kopya için JSON kullanın.'; }
  }
  function restore(data) {
    if(data && data.schema_version===1 && !Object.hasOwn(data,'annotation_round')) data={...data,schema_version:2,annotation_round:'legacy-1'};
    validate(data);
    ['reviewer','annotation_round','athlete_id','session_id'].forEach(k=>$(k).value=data[k]);
    ['coverage_complete','blinded_to_predictions'].forEach(k=>$(k).checked=data[k]);
    $('coverage-start').value=data.coverage.start_frame;$('coverage-end').value=data.coverage.end_frame;
    events=data.events;ownerReviewer=owner(data.reviewer,data.annotation_round);editing=null;paint();$('status').textContent='İşaretleme turu yüklendi.';
  }
  const pause=()=>videos.forEach(video=>video.pause());
  const seek = value => {
    if(!frame(value)) return;
    pause();presentedFrame=null;syncingUntil=performance.now()+500;$('seek-frame').value=value;$('scrub').value=value;
    videos.forEach(video=>{const apply=()=>{video.currentTime=(value+.5)/fps;};if(video.readyState>=1)apply();else video.addEventListener('loadedmetadata',apply,{once:true});});
  };
  if(videos.length && videos[0].requestVideoFrameCallback) {
    const onFrame=(now,metadata)=>{presentedFrame=Math.round(metadata.mediaTime*fps);$('video-status').textContent=`Görüntülenen kaynak kare: ${presentedFrame}`;$('seek-frame').value=presentedFrame;$('scrub').value=presentedFrame;videos[0].requestVideoFrameCallback(onFrame);};
    videos[0].requestVideoFrameCallback(onFrame);
  } else $('video-status').textContent='Tarayıcı çözümlenen kare bilgisini sağlamıyor. Kareleri sayısal olarak girin; görüntüden kare alma kullanılamaz.';
  videos.forEach(video=>{
    video.addEventListener('error',()=>{$('video-status').textContent='Kamera videosu açılamadı.';});
    video.addEventListener('seeking',()=>{if(performance.now()>syncingUntil)seek(Math.min(count-1,Math.floor(video.currentTime*fps)));});
  });
  if(videos.length) videos[0].addEventListener('timeupdate',()=>{if(!videos[0].paused && performance.now()>syncingUntil) videos.slice(1).forEach(video=>{if(Math.abs(video.currentTime-videos[0].currentTime)>.1){syncingUntil=performance.now()+500;video.currentTime=videos[0].currentTime;}});});
  $('seek').onclick=()=>seek(readInteger('seek-frame'));$('scrub').oninput=()=>seek(readInteger('scrub'));
  $('back').onclick=()=>seek(Math.max(0,readInteger('seek-frame')-1));$('next').onclick=()=>seek(Math.min(count-1,readInteger('seek-frame')+1));
  $('pause').onclick=pause;$('play').onclick=async()=>{const results=await Promise.allSettled(videos.map(v=>v.play()));if(results.some(r=>r.status==='rejected'))$('video-status').textContent='En az bir kamera oynatılamadı.';};
  ['start','end'].forEach(which=>$('capture-'+which).onclick=()=>{if(frame(presentedFrame))$('event-'+which).value=presentedFrame;else $('status').textContent='Önce videonun çözümlenen karesini bekleyin veya kareyi sayısal olarak girin.';});
  $('save-event').onclick=()=>{
    const event={event_id:editing || `E${Date.now()}-${serial++}`,kind:$('event-kind').value,start_frame:readInteger('event-start'),end_frame:readInteger('event-end'),note:$('event-note').value};
    try{const data=payload();data.events=[...events.filter(e=>e.event_id!==event.event_id),event];validate(data);events=data.events;ownerReviewer=owner(data.reviewer,data.annotation_round);editing=null;$('event-note').value='';persist();}catch(error){$('status').textContent=error.message;}
  };
  $('cancel-edit').onclick=()=>{editing=null;$('event-note').value='';$('status').textContent='Yeni olay eklenebilir.';};
  $('events').onclick=event=>{
    const id=event.target.dataset.eventId, action=event.target.dataset.action, item=events.find(e=>e.event_id===id);if(!item)return;
    if(action==='delete'){events=events.filter(e=>e.event_id!==id);if(editing===id)editing=null;persist();}
    if(action==='show')seek(item.start_frame);
    if(action==='edit'){editing=id;$('event-kind').value=item.kind;$('event-start').value=item.start_frame;$('event-end').value=item.end_frame;$('event-note').value=item.note;$('status').textContent='Olay düzenleniyor.';}
  };
  $('export').onclick=()=>{
    try{const data=validate(payload());const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));const link=document.createElement('a');link.href=url;link.download=`tk3d-independent-${binding.recording_id}-${encodeURIComponent(data.reviewer)}-${encodeURIComponent(data.annotation_round)}.json`;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);persist();$('status').textContent=data.coverage_complete?'İnceleme JSON indirildi.':'Taslak JSON indirildi; tamamlanma beyanı henüz yok.';}catch(error){$('status').textContent=error.message;}
  };
  $('load-local').onclick=()=>{try{const reviewer=$('reviewer').value.trim(),round=$('annotation_round').value.trim();if(!reviewer||!round)throw Error('İnceleyen ve tur kodunu girin.');const raw=localStorage.getItem(key(reviewer,round)) || (round==='legacy-1'?localStorage.getItem(`tk3d-independent-v1-${binding.recording_id}-${encodeURIComponent(reviewer)}`):null);if(!raw)throw Error('Bu inceleyen ve tur için kayıt bulunamadı.');restore(JSON.parse(raw));}catch(error){$('status').textContent=error.message;}};
  $('new-review').onclick=()=>{pause();events=[];editing=null;ownerReviewer=null;['reviewer','annotation_round','athlete_id','session_id','event-note'].forEach(id=>$(id).value='');['coverage_complete','blinded_to_predictions'].forEach(id=>$(id).checked=false);$('coverage-start').value=0;$('coverage-end').value=count-1;paint();seek(0);$('status').textContent='Yeni boş inceleme başladı. Önceki turların tarayıcı kayıtları korunuyor.';};
  $('import').onchange=async event=>{try{if(event.target.files.length)restore(JSON.parse(await event.target.files[0].text()));}catch(error){$('status').textContent=error.message;}finally{event.target.value='';}};
  ['reviewer','annotation_round','athlete_id','session_id','coverage-start','coverage-end','coverage_complete','blinded_to_predictions'].forEach(id=>$(id).addEventListener('change',persist));
  paint();
})();"""
