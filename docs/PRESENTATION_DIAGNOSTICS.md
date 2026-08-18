# Presentation Teşhis Katmanı

Son güncelleme: **18 Ağustos 2026**

Bu belge `src/poomsae_scoring/presentation.py` modülünü açıklar. Modül WT
Presentation puanı **üretmez**. Ne ürettiği, neden puan üretmediği ve ileride
puan üretebilmesi için nelerin gerektiği aşağıdadır.

## 1. Neden ayrı bir katman ve neden puansız

WT Poomsae Competition Rules (30 Eylül 2024) Presentation'ı `6,0` puanlık üç
bütünsel başlığa böler: `speed and power`, `rhythm & tempo` ve
`expression of energy`, her biri `2,0`. Kural metni bunları hakem takdirine
bırakır; Accuracy'deki gibi sayılabilir olay listesi vermez.

Buradan iki sonuç çıkar:

1. **Her sallantıya `-0,1` yazmak yanlış mimaridir.** Presentation kesinti
   toplamı değil, bütünsel bir değerlendirmedir. Accuracy motorunun olay/kesinti
   modeli buraya kopyalanamaz.
2. **Sensör hakem yerine geçemez.** Kamera hız, salınım ve zamanlama ölçer;
   “canlılık” veya “özgüven” gibi hakem yargısını ölçmez. Kinematik hız/ivme
   birer **vekildir (proxy)**, kuvvet ölçümü değildir; sporcunun gövdesinde IMU
   veya kuvvet platformu yoktur.

Bu yüzden katman, bir puan katmanı değil **kalibrasyon altyapısıdır**: ileride
hakem etiketi toplanırsa aynı ölçümler `0-6` aralığına kalibre edilebilsin diye
gereken sayısal zemini, provenance'ı bozmadan hazırlar.

## 2. Sözleşme

`build_presentation_diagnostics()` çıktısı her koşulda şu alanları taşır:

| Alan | Değer | Anlamı |
|---|---|---|
| `status` | `presentation_diagnostic_only` | Rapor türü; skor raporu değildir |
| `total_score` | `null` | Toplam puan hiçbir koşulda üretilmez |
| `judge_calibrated` | `false` | Hakem etiketiyle kalibrasyon yapılmamıştır |
| `not_judge_validated` | `true` | Hakem eşdeğerliği doğrulanmamıştır |
| `safety_contract.score_claim_allowed` | `false` | Alt katmanlar puan iddia edemez |
| `safety_contract.judge_calibration_required_for_score` | `true` | Puan için ön koşul |
| `safety_contract.partial_recording_can_produce_score` | `false` | Kısmi kayıt puan üretemez |
| `safety_contract.kinematic_proxy_is_not_force_measurement` | `true` | Hız/ivme kuvvet değildir |

Girdi kapıları fail-closed'dur:

- `poomsae_spec` `validate_poomsae_spec()` ile doğrulanır;
- `movement_timeline` aynı spec'e karşı `validate_movement_timeline()` ile
  doğrulanır;
- `wholebody_diagnostics["status"]` değeri `wholebody_diagnostics_only`
  değilse `ScoringContractError` fırlatılır — yani rastgele bir sözlük
  presentation raporuna dönüştürülemez;
- `wholebody_diagnostics["movements"]` liste değilse yine hata fırlatılır.

Modül **yeni tolerans değeri veya yeni dış kaynak eklemez**. Hâlihazırda
üretilmiş WholeBody metriklerini toplar; buna ek olarak yalnız doğrulanmış
zaman çizelgesinden iki süre ölçüsü türetir (§3.2). Hiçbir eşik veya sınır
tanımlamaz.

## 3. Bileşenler ve metrikler

### 3.1 `speed_and_power`

| Metrik | Kaynak |
|---|---|
| `executing_wrist_peak_speed_body_scale_per_sec` | WholeBody teşhis raporu |

Uygulayan bilek zirve hızı, vücut ölçeğine normalize edilmiş `birim/saniye`
cinsindendir. Kişiler arası uzuv boyu farkını dışarıda bırakmak için ham metre
yerine vücut-ölçekli değer kullanılır.

### 3.2 `rhythm_and_tempo`

| Metrik | Nasıl hesaplanır |
|---|---|
| `movement_duration_sec` | `(end_frame - start_frame + 1) / fps`, her segment için |
| `transition_gap_sec` | Ardışık iki segment arasındaki boş kare sayısı `/ fps` |

Bu iki metrik WholeBody raporundan değil, doğrudan **zaman çizelgesinden**
türetilir. Bu yüzden manuel etiketli veya otomatik türetilmiş her timeline ile
çalışır. Negatif boşluk `0`'a kırpılır; tek segmentli kayıtta
`transition_gap_sec` örneksiz (`sample_count=0`) kalır.

### 3.3 `expression_of_energy`

| Metrik | Ne gösterir |
|---|---|
| `fixation_wrist_jitter_ratio` | Bitiş pozunda kontrollü duruş / titreme |
| `torso_lean_p95_deg` | Gövde eğimi uç değeri |
| `head_torso_yaw_mismatch_deg` | Baş ile gövde yönü uyumsuzluğu |
| `shoulder_hip_twist_deg` | Omuz-kalça burulması |

Baş yönü metriği **göz takibi değildir**; bakış yönünün tam karşılığı olarak
okunamaz.

## 4. Özet istatistikleri (`_summarize`)

Her metrik için tek bir sayı değil, kalibratörün kendi indirgemesini seçmesine
izin veren bir özet döner:

| Alan | Not |
|---|---|
| `sample_count` | Kaç geçerli örnek bulundu |
| `median` | Dayanıklı merkez ölçüsü (ortalama değil) |
| `min`, `max` | Uç değerler |
| `interquartile_range` | Yayılım; **yalnız `sample_count >= 4` ise** hesaplanır, aksi halde `null` |
| `unit` | WholeBody metriklerinde ilk geçerli örnekten okunur; timeline metriklerinde sabittir (`sec`) |

Örneksiz metrikte `sample_count=0` olur ve `median`, `min`, `max`,
`interquartile_range` `null` döner — `0.0` gibi yanıltıcı bir varsayılan
yazılmaz. `unit` alanı bu durumda da dolu kalabilir (timeline metriklerinde
her zaman doludur).

Süzme `_summarize` içinde değil, bir üst katmanda yapılır:
`_aggregate_metric_component()` WholeBody metriklerini gezerken `None` ve
sayıya çevrilemeyen değerleri atlar. `_summarize()` kendisi zaten süzülmüş bir
`list[float]` bekler; doğrudan yeniden kullanılacaksa bu varsayım
unutulmamalıdır.

Bileşen düzeyinde ayrıca `measurable_metric_count` / `requested_metric_count`
taşınır; böylece “kaç metrik istendi, kaçı gerçekten ölçüldü” raporda görünür.

## 5. Çağrı biçimi

```python
from src.poomsae_scoring import build_presentation_diagnostics

report = build_presentation_diagnostics(
    wholebody_diagnostics,   # status="wholebody_diagnostics_only" olan rapor
    poomsae_spec,            # doğrulanmış PoomsaeSpec
    movement_timeline,       # manuel veya otomatik MovementTimeline
)
assert report["total_score"] is None  # sözleşme
```

Girdi zaman çizelgesi `build_automatic_movement_timeline()` çıktısı da olabilir;
katman `label_source` değerine bakmaz, yalnız doğrulanmış olmasını ister.

## 6. İleride puana kalibre etmek için gerekenler

Sıralama önemlidir; hiçbiri atlanamaz:

1. **Hakem etiketi.** Aynı kayıtlar için birden çok yetkili hakemin verdiği
   Presentation alt puanları. Tek hakem yeterli değildir; hakemler arası
   uyum (inter-rater agreement) ölçülmelidir.
2. **Yeterli örneklem.** Sporcu, seviye ve çekim koşulu çeşitliliği. Tek
   sporcunun tek kaydından kalibrasyon yapılamaz.
3. **Kalibrasyon modeli ve ayrık doğrulama kümesi.** Modelin eğitildiği kayıtlar
   ile doğrulandığı kayıtlar ayrı olmalıdır.
4. **Ayrı bir kalibrasyon profili.** Kaynak-bağlı Accuracy profilinde olduğu
   gibi sürümlü, hash bağlı ve statüsü açıkça işaretli.
5. **Sözleşme bayraklarının açılması.** Ancak yukarıdakiler sağlandıktan sonra
   `judge_calibrated` ve `score_claim_allowed` değiştirilebilir.

Bu adımlar tamamlanmadan raporun herhangi bir alanı `0-6` puan olarak sunulamaz.

`safety_contract` alanlarının hepsi **bildirimdir**, çalışma anında
değerlendirilen bir dal değildir: modül `recording_scope` değerine bakıp karar
değiştirmez, çünkü zaten hiçbir koşulda puan üretmez. Örneğin
`partial_recording_can_produce_score=false`, ileride yazılacak kalibrasyon
katmanına “kısmi kayıttan puan türetme” yasağını taşır; bugünkü davranışı
değiştirmez.

## 7. İlgili belgeler

- [`TAEGEUK1_ERROR_TAXONOMY.md`](TAEGEUK1_ERROR_TAXONOMY.md) — bölüm 2.3,
  Presentation'ın neden tekil kesinti olmadığı;
- [`SCORING_SOURCE_REGISTER.md`](SCORING_SOURCE_REGISTER.md) — WT `4,0 / 6,0`
  bütçesinin kaynak doğrulaması;
- `../PROJECT_STATUS.md` — güncel puanlama zinciri ve koşu durumu.
