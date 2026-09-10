"""Hakemin cevaplaması gereken soruları profilden çıkarır.

Teşhis profili hangi sayının geliştirme sırasında yazıldığını ve hangisinin hakem
imzası taşıdığını zaten biliyor. Bu komut birinci grubu bir soru listesine
çevirir: her açık değer için ne ölçüldüğü, bugünkü geçici değer ve cevap
geldiğinde profile yazılacak blok. Sayfa hiçbir şeyi puanlamaz ve kanonik
puanlama akışının parçası değildir.
"""

from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.artifact_io import sha256_file  # noqa: E402
from src.poomsae_scoring import load_technical_accuracy_profile  # noqa: E402

FAMILY_LABELS = {
    "head_orientation": "Baş yönelimi",
    "torso_pelvis": "Gövde ve kalça",
    "stance_feet": "Duruş ve ayaklar",
    "lower_body": "Alt gövde",
    "kick": "Tekme",
    "upper_body": "Üst gövde",
    "wrist_hand": "Bilek ve el",
    "phase_structure": "Faz yapısı",
    "transition": "Geçiş",
    "fixation": "Duruşun tutulması",
    "sequence": "Sıra bütünlüğü",
    "pipeline_limits": "Ölçüm sınırları",
}

OPERATOR_TEXT = {
    "max": "şunu geçmemeli",
    "abs_max": "iki yönde de şunu geçmemeli",
    "min": "en az şu kadar olmalı",
    "range": "şu aralıkta kalmalı",
}

BLOCKED_REASON = {
    "blocked_missing_reference": "Sporcunun baktığı yön bilinmeden ölçülemiyor.",
    "measurement_only": "Altındaki duruş aralığı da doğrulanmamış; iki sayı birden gerekiyor.",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Hakem soru listesini teşhis profilinden üretir. İmzasız her eşiği, "
            "duruş aralıklarını ve tanımlanmamış teknik hedeflerini listeler."
        )
    )
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output-html", required=True)
    args = parser.parse_args()

    profile_path = _resolve(args.profile)
    if not profile_path.is_file():
        raise SystemExit(f"Profil dosyası yok: {profile_path}")
    output = _resolve(args.output_html)
    if output.exists():
        raise SystemExit(f"Çıktı zaten var, üzerine yazılmayacak: {output}")

    profile = load_technical_accuracy_profile(profile_path)
    questions = collect_open_questions(profile)
    total = (
        len(questions["priority"])
        + len(questions["deferred"])
        + len(questions["stance_ranges"])
        + len(questions["technique_targets"])
    )
    if total == 0:
        raise SystemExit("Açık soru yok: bütün değerler hakem imzası taşıyor.")

    page = render_questionnaire(
        questions,
        profile_id=profile["profile_id"],
        profile_path=profile_path,
        profile_sha256=sha256_file(profile_path),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="") as stream:
        stream.write(page)
    print(output)
    print(
        f"oncelikli esik: {len(questions['priority'])}; "
        f"durus araligi: {len(questions['stance_ranges'])}; "
        f"ikinci tur esik: {len(questions['deferred'])}; "
        f"teknik hedefi: {len(questions['technique_targets'])}"
    )


def collect_open_questions(profile: dict) -> dict[str, list[dict]]:
    """Açık kalan değerleri, cevabın ne zaman işe yarayacağına göre ayırır.

    Öncelikli grup: kural bugün çalışıyor, tek eksik doğru sayı. Cevap gelir
    gelmez o kural iş yapar. İkinci grup: sayı gelse bile başka bir eksik
    olduğu için kural hemen açılmaz.
    """
    by_metric = {rule["metric_id"]: rule for rule in profile["resolved_rules"]}
    priority: list[dict] = []
    deferred: list[dict] = []
    for metric_id, threshold in sorted(profile["thresholds"].items()):
        if "judge_source" in threshold:
            continue
        rule = by_metric.get(metric_id, {})
        status = rule.get("status", "unknown")
        row = {
            "metric_id": metric_id,
            "family": rule.get("rule_family", "unassigned"),
            "status": status,
            "operator": threshold["operator"],
            "value": threshold["value"],
            "unit": threshold["unit"],
            "uncertainty_band": threshold["uncertainty_band"],
            "blocking_reason": BLOCKED_REASON.get(status),
        }
        (priority if status == "active_diagnostic" else deferred).append(row)

    stance_ranges: list[dict] = []
    for stance, contract in sorted(profile["stance_contracts"].items()):
        for field, value in sorted(contract.items()):
            if isinstance(value, list):
                stance_ranges.append(
                    {"stance": stance, "field": field, "low": value[0], "high": value[1]}
                )

    technique_targets: list[dict] = []
    for technique, contract in sorted(profile["technique_contracts"].items()):
        for field, value in sorted(contract.items()):
            if isinstance(value, str) and value.endswith("_unavailable"):
                technique_targets.append(
                    {"technique": technique, "field": field, "state": value}
                )

    return {
        "priority": priority,
        "deferred": deferred,
        "stance_ranges": stance_ranges,
        "technique_targets": technique_targets,
    }


def _threshold_row(item: dict, *, with_reason: bool) -> str:
    family = FAMILY_LABELS.get(item["family"], item["family"].replace("_", " "))
    operator = OPERATOR_TEXT.get(item["operator"], item["operator"])
    current = item["value"]
    current_text = f"{current[0]} – {current[1]}" if isinstance(current, list) else f"{current}"
    last = (
        f"<td class='state'>{html.escape(item['blocking_reason'] or '')}</td>"
        if with_reason
        else ""
    )
    return (
        "<tr>"
        f"<td class='metric'>{html.escape(item['metric_id'])}"
        f"<span>{html.escape(family)}</span></td>"
        f"<td>{html.escape(operator)}</td>"
        f"<td class='value'>{html.escape(current_text)} {html.escape(item['unit'])}"
        f"<span>&plusmn;{item['uncertainty_band']}</span></td>"
        f"{last}"
        "<td class='answer'></td>"
        "</tr>"
    )


def render_questionnaire(
    questions: dict[str, list[dict]],
    *,
    profile_id: str,
    profile_path: Path,
    profile_sha256: str,
) -> str:
    priority_rows = "".join(_threshold_row(item, with_reason=False) for item in questions["priority"])
    deferred_rows = "".join(_threshold_row(item, with_reason=True) for item in questions["deferred"])
    stance_rows = "".join(
        "<tr>"
        f"<td class='metric'>{html.escape(item['stance'])}<span>{html.escape(item['field'])}</span></td>"
        f"<td class='value'>{item['low']} – {item['high']}</td>"
        "<td class='answer'></td>"
        "</tr>"
        for item in questions["stance_ranges"]
    )
    technique_rows = "".join(
        "<tr>"
        f"<td class='metric'>{html.escape(item['technique'])}<span>{html.escape(item['field'])}</span></td>"
        f"<td class='value'>{html.escape(item['state'])}</td>"
        "<td class='answer'></td>"
        "</tr>"
        for item in questions["technique_targets"]
    )
    first_count = len(questions["priority"]) + len(questions["stance_ranges"])
    second_count = len(questions["deferred"]) + len(questions["technique_targets"])

    return f"""<!DOCTYPE html>
<html lang="tr"><head><meta charset="utf-8">
<title>Hakem soru listesi &mdash; {html.escape(profile_id)}</title>
<style>
 body {{ font: 14px/1.55 system-ui, sans-serif; margin: 2.5rem auto; max-width: 62rem; color: #1a1a1a; }}
 h1 {{ font-size: 1.45rem; margin-bottom: .25rem; }}
 h2 {{ font-size: 1.12rem; margin-top: 2.6rem; border-top: 2px solid #333; padding-top: .8rem; }}
 h3 {{ font-size: .98rem; margin-top: 1.6rem; }}
 p.lead {{ color: #444; }}
 table {{ border-collapse: collapse; width: 100%; margin-top: .75rem; }}
 th, td {{ border: 1px solid #ccc; padding: .5rem .6rem; vertical-align: top; text-align: left; }}
 th {{ background: #f2f2f2; font-weight: 600; }}
 td.metric {{ font-family: ui-monospace, monospace; font-size: .8rem; width: 28%; }}
 td.metric span {{ display: block; font-family: system-ui, sans-serif; color: #666; font-size: .78rem; }}
 td.value {{ white-space: nowrap; }}
 td.value span {{ display: block; color: #666; font-size: .78rem; }}
 td.state {{ color: #555; font-size: .84rem; }}
 td.answer {{ width: 16%; background: #fffdf3; }}
 pre {{ background: #f6f6f6; padding: .8rem; overflow-x: auto; font-size: .8rem; }}
 .note {{ border-left: 3px solid #999; padding-left: .9rem; color: #444; }}
</style></head><body>
<h1>Hakem soru listesi</h1>
<p class="lead">Profil <code>{html.escape(profile_id)}</code>. Aşağıdaki değerlerin
tamamı geliştirme sırasında yazıldı ve bugün hiçbiri puana etki etmiyor. Bir değer,
ancak yetkili bir hakem verdikten ve imzası kayda geçtikten sonra puana etki eder.</p>

<p class="note">Bu sayfa soru sorar. Hiçbir performans hakkında iddia taşımaz, hiçbir
kesinti ve puan içermez, puanlama akışının parçası değildir.</p>

<h2>Bölüm 1 &mdash; öncelikli ({first_count} soru)</h2>
<p>Bu ölçümler bugün çalışıyor. Tek eksikleri doğru sayı. Cevap geldiği anda ilgili
kural iş yapmaya başlar.</p>

<h3>1.1 Ekran eşikleri ({len(questions['priority'])} soru)</h3>
<table><thead><tr>
<th>Ölçüm</th><th>Kural nasıl okuyor</th><th>Bugünkü geçici değer</th><th>Hakemin değeri</th>
</tr></thead><tbody>{priority_rows}</tbody></table>

<h3>1.2 Duruş aralıkları ({len(questions['stance_ranges'])} soru)</h3>
<p>Bu aralıklar bir duruşun doğru sayılıp sayılmadığına karar veriyor. Altındaki aralık
onaylanmadan üstüne konan tolerans anlamsız kalır.</p>
<table><thead><tr>
<th>Duruş</th><th>Bugünkü geçici aralık</th><th>Hakemin aralığı</th>
</tr></thead><tbody>{stance_rows}</tbody></table>

<h2>Bölüm 2 &mdash; vakit kalırsa ({second_count} soru)</h2>
<p>Bu ölçümlerde sayı tek eksik değil. Hakem değeri verse bile kural hemen açılmaz,
çünkü başka bir bilgi de eksik. Yine de cevap kayda geçerse eksik kapandığında hazır olur.</p>

<h3>2.1 Başka bir eksiği olan eşikler ({len(questions['deferred'])} soru)</h3>
<table><thead><tr>
<th>Ölçüm</th><th>Kural nasıl okuyor</th><th>Bugünkü geçici değer</th><th>Sayı dışındaki eksik</th><th>Hakemin değeri</th>
</tr></thead><tbody>{deferred_rows}</tbody></table>

<h3>2.2 Hiç tanımlanmamış teknik hedefleri ({len(questions['technique_targets'])} soru)</h3>
<table><thead><tr>
<th>Teknik</th><th>Bugünkü durum</th><th>Hakemin değeri</th>
</tr></thead><tbody>{technique_rows}</tbody></table>

<h2>Cevap nasıl kaydedilir</h2>
<p>Cevaplanan her eşik profile bir imza bloğu kazanır ve adı
<code>judge_validated_rules</code> listesine eklenir. İkisi de zorunludur; biri
tek başına reddedilir.</p>
<pre>thresholds:
  &lt;ölçümün adı&gt;:
    operator: max
    value: &lt;hakemin verdiği sayı&gt;
    uncertainty_band: &lt;belirsizlik payı&gt;
    unit: deg
    judge_source:
      origin: judge_supplied_validated_threshold
      judge_name: &lt;hakemin adı&gt;
      judge_credential: &lt;yetki belgesi veya derece&gt;
      decision_date: '&lt;YYYY-AA-GG&gt;'
      approval_reference: &lt;onay kaydı&gt;
      score_effect: deduction_candidate
      deduction_points: &lt;puan&gt;

judge_validated_rules:
  - &lt;ölçümün adı&gt;</pre>

<p class="note">Kaynak profil: <code>{html.escape(str(profile_path))}</code><br>
SHA-256 <code>{html.escape(profile_sha256)}</code></p>
</body></html>
"""


def _resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


if __name__ == "__main__":
    main()
