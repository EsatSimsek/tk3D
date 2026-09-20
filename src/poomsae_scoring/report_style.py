"""Shared, embedded stylesheet for portable evidence and history reports."""

REPORT_CSS = """
:root{color-scheme:light;--bg:#f3f4f0;--panel:#fff;--panel2:#f6f8f6;--line:#dce3df;
 --text:#21373d;--muted:#63757a;--cyan:#226b64;--green:#287158;--amber:#946318;--red:#b5483f}
*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:24px}
body{margin:0;background:var(--bg);color:var(--text);font:14px/1.55 'Segoe UI',system-ui,sans-serif}
main{width:min(1440px,94vw);margin:auto;padding:38px 0 60px}
header{display:flex;justify-content:space-between;align-items:center;gap:24px;margin-bottom:24px}
h1{font-size:clamp(28px,3.2vw,44px);line-height:1.16;letter-spacing:-.035em;margin:10px 0 12px;font-weight:650}
h2{font-size:20px;letter-spacing:-.015em;margin:0 0 16px;font-weight:650}h3{font-size:16px}
p{margin:0;color:var(--muted);line-height:1.6}a{color:var(--cyan);text-underline-offset:4px}
button,input,select{font:inherit}button,input:not([type=file]),select{border:1px solid #bdcdc5;border-radius:6px;padding:9px 12px;background:white;color:var(--text)}
button{cursor:pointer;font-weight:600;transition:background .15s,border-color .15s}button:hover{background:#edf4ef;border-color:var(--cyan)}
button:focus-visible,a:focus-visible,input:focus-visible,select:focus-visible,summary:focus-visible{outline:3px solid #5c958b;outline-offset:3px}
button:disabled{opacity:.5;cursor:wait}input,select{max-width:100%;min-width:0}input[type=search]{width:min(380px,100%)}
input[type=file]{font-size:12px;width:240px}input::file-selector-button{background:var(--panel2);border:1px solid var(--line);border-radius:4px;padding:6px;margin-right:8px}
.eyebrow{color:var(--cyan);font-size:11px;font-weight:750;letter-spacing:.15em;text-transform:uppercase}
.pill{font-size:12px;font-weight:650;padding:7px 12px;border-radius:5px;background:#e9eeea;border:1px solid #d4ddd6;white-space:nowrap}
.notice{display:flex;gap:12px;padding:13px 16px;background:#f9f4e9;border-left:3px solid #c29851;border-radius:0 6px 6px 0;margin:0 0 20px}
.notice strong{color:#775622;display:block;font-size:13px}.notice p{font-size:12px;margin-top:3px;color:#766951}
.stats,.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:0;margin:0 0 24px;border:1px solid var(--line);border-radius:8px;background:white;overflow:hidden}
.stat,.card{padding:17px 22px;min-width:0;border-right:1px solid var(--line)}.stat:last-child,.card:last-child{border-right:0}
.stat span,.card span{color:var(--muted);font-size:12px}.stat b,.card b{display:block;font-size:25px;line-height:1.3;margin-top:6px;font-weight:650;font-variant-numeric:tabular-nums;overflow-wrap:anywhere}
.grid{grid-template-columns:repeat(3,minmax(0,1fr));margin-top:22px}.card b{font-size:18px}
.section{padding:24px;background:white;border:1px solid var(--line);border-radius:8px;margin-bottom:20px;min-width:0}
.report-nav{display:flex;gap:22px;flex-wrap:wrap;border-bottom:1px solid var(--line);margin:0 0 22px;padding:0 0 12px}
.report-nav a{font-weight:600;text-decoration:none;font-size:13px}.history-link{font-size:12px}
.videos{display:grid;grid-template-columns:repeat(var(--camera-count,2),minmax(0,1fr));gap:16px}
.video-card{border:1px solid var(--line);border-radius:7px;overflow:hidden;background:#1c3036;min-width:0}
.video-label{padding:10px 14px;font-size:12px;letter-spacing:.03em;color:#edf2f0;font-weight:600}
.camera-dot{display:inline-block;width:6px;height:6px;border-radius:50%;background:#93bfb0;margin-right:9px}
video{display:block;width:100%;aspect-ratio:16/9;background:#101e22}.video-health{padding:7px 14px;font-size:11px;color:#c2d1ca}
.video-health[data-state=error]{color:#ffb7ae}.video-health[data-state=loading]{color:#f3d08a}
.toolbar{display:flex;flex-wrap:wrap;align-items:center;gap:9px;margin-top:16px}.toolbar p{font-size:12px}.toolbar label{font-size:12px;color:var(--muted)}
#sync-play{background:var(--cyan);border-color:var(--cyan);color:white}#sync-play:hover{background:#1b5954}
#clock{font-variant-numeric:tabular-nums;color:var(--text);font-weight:650;margin-left:auto}#sync-status,#review-status{font-size:12px;color:var(--muted)}
.movement-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}
.movement{padding:16px;border:1px solid var(--line);border-radius:7px;background:#fbfcfa;text-align:left;min-width:0}
.movement:hover,.movement.active{background:#eef5f0;border-color:#6e9a88;box-shadow:inset 3px 0 var(--cyan)}
.movement-top{display:flex;justify-content:space-between;gap:8px;font-size:12px}.movement-id{color:var(--cyan);font-weight:750}
.movement-name{font-weight:650;font-size:14px;margin:8px 0 12px}.metrics{display:grid;grid-template-columns:1fr 1fr;gap:5px 10px;color:var(--muted);font-size:11px}.metrics b{color:var(--text)}
.anchors{display:flex;gap:5px;flex-wrap:wrap;margin-top:12px}.anchor{font-size:11px;padding:4px 7px;background:white;border-color:var(--line)}
.report-detail{border:1px solid var(--line);border-radius:8px;background:white;margin-bottom:12px;min-width:0}
.report-detail>summary{cursor:pointer;padding:18px 22px;font-size:15px;font-weight:650;list-style:none;display:flex;align-items:center;gap:14px}
.report-detail>summary::-webkit-details-marker{display:none}.report-detail>summary:before{content:'+';color:var(--cyan);font-weight:400;font-size:22px;width:15px}
.report-detail[open]>summary:before{content:'−'}.report-detail>summary small{margin-left:auto;color:var(--muted);font-size:12px;font-weight:400}
.report-detail>.section{border:0;border-top:1px solid var(--line);margin:0;border-radius:0 0 8px 8px}
.report-detail>.stats{margin:0 20px 20px;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));border:0;gap:8px}
.report-detail .stat{border:1px solid var(--line);border-radius:6px}.report-detail .stat b{font-size:20px}
ul{list-style:none;padding:0;margin:0}li{padding:12px 0;display:flex;align-items:flex-start;gap:12px;border-bottom:1px solid var(--line);color:var(--muted);overflow-wrap:anywhere}li:last-child{border-bottom:0}
code{font:12px/1.5 Consolas,monospace;color:var(--cyan);overflow-wrap:anywhere}
.candidate-row{display:grid;grid-template-columns:minmax(130px,.35fr) 1fr auto;align-items:center}
.decision-row{display:grid;grid-template-columns:145px minmax(0,1fr) 220px;gap:20px;padding:20px 16px;margin-top:8px;background:#fafbf9;border-radius:5px;align-items:center;border-left:3px solid var(--line)}
.decision-row.red{border-left-color:var(--red)}.decision-row.amber{border-left-color:var(--amber)}.decision-row.green{border-left-color:var(--green)}.decision-row.blue{border-left-color:#346e91}.decision-row.gray{border-left-color:#89989d}
.decision-row b{color:var(--text)}.decision-title{font-size:15px;display:block;margin-bottom:6px}.decision-meta{display:block;font-size:12px;margin-top:7px}
.review-actions{display:flex;gap:5px;flex-wrap:wrap}.review-actions button{font-size:12px;padding:6px 8px}.review-actions button.selected{border-color:var(--cyan);background:#e1eee7;box-shadow:inset 0 -2px var(--cyan)}
.metric-table-wrap,.table-wrap{overflow:auto;margin-top:16px}table{width:100%;border-collapse:collapse;font-size:12px}
th,td{padding:11px 12px;text-align:left;border-bottom:1px solid var(--line)}th{background:var(--panel2);color:var(--text);font-weight:650;white-space:nowrap}td{color:var(--muted)}
.metric-table th{position:sticky;top:0}.metric-table td{white-space:nowrap}.metric-table .review{color:var(--amber);font-weight:650}.metric-table .ok,.delta-pos{color:var(--green)}.metric-table .missing,.delta-neutral{color:var(--muted)}.delta-neg{color:var(--red)}
.two-col{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-top:24px}.missing-list,.sources{display:flex;gap:8px;flex-wrap:wrap}.missing-chip{padding:5px 8px;background:var(--panel2);border:1px solid var(--line);border-radius:4px;font-size:11px;color:var(--muted)}
.sources{margin-top:14px}.sources a{font-size:12px}.wb-movement-block{padding:16px;border:1px solid var(--line);border-radius:6px;margin-bottom:12px;background:#fafbf9}
.wb-movement-hdr{display:flex;justify-content:space-between;align-items:center;gap:10px;margin-bottom:10px;cursor:pointer}.wb-mid{font-weight:750;color:var(--cyan)}.wb-mname{font-weight:600;font-size:13px;margin-left:8px}
.wb-badge{font-size:11px;padding:3px 8px;border-radius:4px}.wb-badge-ok{color:var(--green);background:#e7f1eb}.wb-badge-warn{color:var(--amber);background:#f7edda}
.wb-metrics-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:8px}.wb-m{display:grid;grid-template-columns:20px minmax(0,1fr) auto;gap:3px 8px;padding:10px;border:1px solid var(--line);border-radius:5px;background:white}
.wb-m-name{color:var(--muted);font-size:11px;overflow-wrap:anywhere}.wb-m-val{font-size:13px;font-weight:650;font-variant-numeric:tabular-nums}.wb-m-range{grid-column:2/4;font-size:10px;color:var(--muted)}
.wb-ok{border-left:3px solid var(--green)}.wb-cand{border-left:3px solid var(--red)}.wb-nm{border-left:3px solid #89989d}.wb-diag{border-left:3px solid #346e91}.wb-dot{color:var(--muted)}
.wb-collapse{display:none}.wb-movement-block.open .wb-collapse{display:grid}
.diag-row{display:grid;grid-template-columns:140px 140px minmax(0,1fr) auto;gap:12px;align-items:center}.diag-consistent{color:var(--green)}.diag-mismatch{color:var(--red);font-weight:650}.diag-ambiguous{color:var(--amber)}
.proxy-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.proxy-card{border:1px solid var(--line);border-radius:6px;padding:16px;background:#fafbf9}.proxy-card h3{margin:0 0 10px;color:var(--cyan)}.proxy-card li{display:block}
.alert{border-left:3px solid #c29851}footer{font-size:11px;color:var(--muted);margin-top:28px;overflow-wrap:anywhere}
@media(max-width:1000px){.movement-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.decision-row{grid-template-columns:120px minmax(0,1fr)}.decision-row>div:last-child{grid-column:2}.diag-row{grid-template-columns:1fr 1fr}.two-col{grid-template-columns:1fr}.proxy-grid{grid-template-columns:1fr}}
@media(max-width:700px){main{width:94vw;padding-top:22px}header{align-items:flex-start;flex-direction:column;gap:0}.section{padding:16px}.stats{grid-template-columns:1fr 1fr}.stat{border-bottom:1px solid var(--line);padding:14px}.stat b{font-size:21px}.videos,.movement-grid,.grid{grid-template-columns:1fr}.decision-row,.candidate-row,.diag-row{grid-template-columns:1fr}.decision-row>div:last-child{grid-column:auto}.wb-metrics-grid{grid-template-columns:minmax(0,1fr)}.wb-movement-hdr{flex-wrap:wrap}.report-detail>summary small{display:none}.report-nav{gap:15px}.toolbar{align-items:flex-start}#clock{margin-left:0}.card{border-right:0;border-bottom:1px solid var(--line)}.metrics{font-size:11px}}
@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}button{transition:none}}
@media print{body{background:white}main{width:100%}.report-nav,.toolbar,button{display:none}.section,.report-detail{break-inside:avoid}video{max-height:200px}details:not([open])>summary{color:#555}}
"""
