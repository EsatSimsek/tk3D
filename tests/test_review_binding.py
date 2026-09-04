from copy import deepcopy
import json
from pathlib import Path
import re
import shutil
import subprocess

import pytest

from src.poomsae_scoring import ScoringContractError, load_movement_timeline, load_poomsae_spec
from src.poomsae_scoring.review_report import build_review_binding, build_review_html, validate_review_export


ROOT = Path(__file__).resolve().parents[1]


def _inputs():
    spec = load_poomsae_spec(ROOT / "config/scoring/poomsae/taegeuk_1_jang_v0_draft.yaml")
    timeline = load_movement_timeline(ROOT / "config/scoring/timelines/poomsae1_zed2i_rgbd_rerun_20260802_draft.yaml", spec)
    return spec, timeline


def test_review_identity_changes_with_run_pose_rules_and_measurements():
    _, timeline = _inputs()
    reports = {"events": [{"event_id": "E1", "value": 8}], "rules": {"expected_boolean": False}}
    binding = build_review_binding(timeline, reports, "analysis-a")
    assert binding == build_review_binding(deepcopy(timeline), deepcopy(reports), "analysis-a")
    assert binding != build_review_binding(timeline, reports, "analysis-b")
    changed_timeline = deepcopy(timeline)
    changed_timeline["source_binding"]["pose_file_sha256"] = "a" * 64
    assert binding != build_review_binding(changed_timeline, reports, "analysis-a")
    for changed in ({**reports, "rules": {"expected_boolean": True}}, {**reports, "events": [{"event_id": "E1", "value": 9}]}):
        assert binding != build_review_binding(timeline, changed, "analysis-a")
    changed_timeline = deepcopy(timeline)
    changed_timeline["segments"][0]["anchors"]["fixation"] += 1
    assert binding != build_review_binding(changed_timeline, reports, "analysis-a")


@pytest.mark.parametrize("mutation", [None, "legacy", "foreign", "unknown_event", "missing_reviewer", "bad_decision"])
def test_review_import_is_bound_and_requires_attributed_labels(mutation):
    _, timeline = _inputs()
    binding = build_review_binding(timeline, {}, "analysis-a")
    payload = {"schema_version": 2, "binding": deepcopy(binding), "reviews": {
        "E1": {"decision": "confirmed", "reviewer": "judge-a", "reviewed_at": "2026-09-04T12:00:00Z"},
    }}
    if mutation == "legacy":
        payload["schema_version"] = 1
    elif mutation == "foreign":
        payload["binding"]["analysis_run_id"] = "analysis-b"
    elif mutation == "unknown_event":
        payload["reviews"]["E2"] = payload["reviews"].pop("E1")
    elif mutation == "missing_reviewer":
        payload["reviews"]["E1"]["reviewer"] = " "
    elif mutation == "bad_decision":
        payload["reviews"]["E1"]["decision"] = "deduct"
    if mutation is None:
        assert validate_review_export(payload, binding, {"E1"}) == payload
    else:
        with pytest.raises(ScoringContractError):
            validate_review_export(payload, binding, {"E1"})


def test_review_html_embeds_content_binding_and_valid_javascript(tmp_path):
    spec, timeline = _inputs()
    rendered = build_review_html(
        spec, timeline, {"timeline": {"timeline_id": timeline["timeline_id"]}},
        {"movement_timeline": {"timeline_id": timeline["timeline_id"]}},
        {"a": "a.mp4", "b": "b.mp4"}, analysis_run_id="analysis-a",
    )
    data = json.loads(re.search(r'<script id="review-data" type="application/json">(.*?)</script>', rendered, re.S)[1])
    assert data["review_binding"]["analysis_run_id"] == "analysis-a"
    assert "tk3d-review-v2-${data.review_binding.review_id}" in rendered
    assert "schema_version:2, binding:data.review_binding" in rendered
    assert "validReviews(payload.reviews)" in rendered
    assert "Tarayıcı kaydı kullanılamıyor" in rendered
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is optional; binding assertions above still ran")
    script = re.search(r'<script>\s*(.*?)</script>', rendered, re.S)[1]
    path = tmp_path / "review.js"
    path.write_text(script, encoding="utf-8")
    checked = subprocess.run([node, "--check", str(path)], capture_output=True, text=True)
    assert checked.returncode == 0, checked.stderr
    # Run the actual review handlers against a minimal DOM/storage, including
    # disabled storage and foreign imports. Video playback is not simulated.
    initialization = script[script.index("const reviewKey"):script.index("let syncingSeek")]
    handlers = script[script.index("document.querySelectorAll('[data-review-event]').forEach"):script.index("const metricFilter")]
    setup = r"""
const assert = require('node:assert/strict');
const store = new Map();
const localStorage = {getItem:key=>store.get(key), setItem:(key,value)=>store.set(key,value), removeItem:key=>store.delete(key)};
const elements = Object.fromEntries(['reviewer-name','review-status','export-review','import-review','clear-review'].map(key=>[key,{value:''}]));
const buttons = ['confirmed','rejected','uncertain'].map(decision=>({
  dataset:{reviewEvent:'E1',reviewValue:decision},
  classList:{add(){},remove(){},toggle(){}},
  addEventListener(event,handler){this.handler=handler;}
}));
let exported = null;
const URL = {createObjectURL(blob){exported=blob;return 'blob:test';},revokeObjectURL(){}};
const setTimeout = ()=>{};
const document = {
  querySelectorAll:()=>buttons,
  getElementById:key=>elements[key],
  createElement:()=>({click(){},remove(){}}),
  body:{appendChild(){}}
};
"""
    checks = r"""
buttons[0].handler({stopPropagation(){}});
assert.equal(Object.keys(reviewSelections).length,0);
elements['reviewer-name'].value='judge-a';
buttons[0].handler({stopPropagation(){}});
assert.equal(reviewSelections.E1.decision,'confirmed');
assert.equal(reviewSelections.E1.reviewer,'judge-a');
assert.ok(reviewSelections.E1.reviewed_at);
assert.equal(JSON.parse(store.get(reviewKey)).E1.decision,'confirmed');
elements['export-review'].onclick();
const payload=JSON.parse(await exported.text());
assert.equal(payload.schema_version,2);
assert.deepEqual(payload.binding,data.review_binding);
assert.equal(payload.reviews.E1.reviewer,'judge-a');
for (const invalid of [
  {...payload,schema_version:1},
  {...payload,binding:{...payload.binding,analysis_run_id:'foreign'}},
  {...payload,reviews:{E2:payload.reviews.E1}}
]) {
  elements['import-review'].files=[{text:async()=>JSON.stringify(invalid)}];
  await elements['import-review'].onchange();
  assert.equal(reviewSelections.E1.decision,'confirmed');
  assert.match(elements['review-status'].textContent,/İçe aktarılamadı/);
}
payload.reviews.E1.decision='uncertain';
elements['import-review'].files=[{text:async()=>JSON.stringify(payload)}];
await elements['import-review'].onchange();
assert.equal(reviewSelections.E1.decision,'uncertain');
localStorage.setItem=()=>{throw Error('storage disabled');};
buttons[1].handler({stopPropagation(){}});
assert.equal(reviewSelections.E1.decision,'rejected');
assert.match(elements['review-status'].textContent,/JSON indirin/);
elements['clear-review'].onclick();
assert.equal(Object.keys(reviewSelections).length,0);
assert.equal(store.has(reviewKey),false);
"""
    executable = setup + "\n(async()=>{\nconst data=" + json.dumps(data) + ";\n" + initialization + handlers + checks + "\n})().catch(error=>{console.error(error);process.exitCode=1;});"
    result = subprocess.run([node], input=executable, capture_output=True, text=True, encoding="utf-8")
    assert result.returncode == 0, result.stderr
