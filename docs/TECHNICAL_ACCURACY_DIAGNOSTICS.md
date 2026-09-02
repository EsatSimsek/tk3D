# Taegeuk 1 Kapsamlı Teknik Doğruluk Teşhisleri

Son güncelleme: **31 Ağustos 2026**

Bu belge, `technical_accuracy_diagnostic` kategorisindeki v3 Taegeuk 1
doğruluk katmanını tanımlar. Katman, mevcut WholeBody-133 ölçüm mimarisini ve
kanonik `tk3d-poomsae` uygulamasını genişletir; ikinci bir puanlama motoru veya
uygulama giriş noktası oluşturmaz.

## Mimari

Akış dört ayrı katmandır:

1. Ölçüm: sonlu, birimli ve kalite kapılı gövde/yüz/ayak/el geometrisi.
2. Hareket doğruluk kontratı: M01–M18 için teknik, duruş, taraf, yön, faz,
   hazırlık, icra ve fixation beklentileri.
3. Geçici değerlendirme: yalnız yapılandırılmış, doğrulanmamış mühendislik
   eşikleriyle inceleme adayı.
4. Kanıt/raporlama: JSON, CSV kapsam matrisi, HTML ve mavi video olayı; skor
   motoruna geri besleme yok.

Ana dosyalar:

- `config/scoring/engineering/taegeuk_1_wholebody_diagnostics_v3.yaml`:
  174 kurallık katalog, bütün eşikler, duruş ve teknik kontratları;
- `src/poomsae_scoring/technical_accuracy.py`: katı şema, yön bağı, kontrat
  çözümü, ölçüm/değerlendirme ve kapsam matrisi;
- `src/poomsae_scoring/technical_accuracy_metrics.py`: baş/torso/alt gövde,
  ayak, kol, ayrıntılı el, faz, geçiş, fixation ve tekme ölçüm kayıtları;
- `scripts/run_technical_accuracy_diagnostics.py`: kanonik uygulama içindeki
  ince CLI adaptörü;
- `scripts/build_athlete_local_direction_reference.py`: oturuma bağlı yön
  referansını açılış duruşundan türeten üretici aşama.
- `src/poomsae_scoring/technical_accuracy_validation.py`: 174 kuralın kapsam,
  eşik ve sentetik geometri doğrulama çekirdeği;
- `scripts/run_technical_accuracy_rule_validation.py`: üzerine yazmayan,
  makine-okunur JSON/CSV validation artifact'leri üreten CLI.

Tarihsel v2 profil değiştirilmemiştir ve mevcut WholeBody/source-bound akışının
girdisi olmaya devam eder. V3, `accuracy_diagnostic_profile` olarak ayrı
bağlanır. Source-bound karar sahipliği
`config/scoring/accuracy/taegeuk_1_source_bound_v1.yaml` ve
`source_bound_accuracy.py` içinde kalır.

## Kural aileleri ve durumları

Katalog şu 11 ölçülebilir aile ile açık pipeline sınırlarını kapsar:

- baş yönelimi;
- torso/omuz/pelvis;
- duruş ve ayaklar;
- diz/bacak/alt gövde;
- Taegeuk 1'de gerçekten bulunan ap-chagi;
- omuz/kol/dirsek;
- bilek/el/yumruk;
- hazırlık/icra/fixation faz yapısı;
- yön değişimi ve geçiş;
- geometrik fixation kararlılığı;
- sıra ve hareket bütünlüğü;
- ayrıca göz bakışı, gerçek ağırlık dağılımı, kuvvet ve basınç gibi pipeline
  sınırları.

Profil düzeyinde 174 kuralın durumu:

| Durum | Sayı | Anlam |
| --- | ---: | --- |
| `active_diagnostic` | 33 | Ölçüm ve geçici eşik değerlendiricisi var |
| `measurement_only` | 116 | Ölçüm var; güvenli karar yetkisi veya sayısal hedef yok |
| `blocked_missing_reference` | 17 | Sporcu-yerel mutlak yön bağı olmadan kapanır |
| `not_observable_with_current_pipeline` | 8 | Mevcut pose/video kanıtından iddia edilemez |

Çalışma zamanında yetersiz eklem, kamera, reprojection, örnek veya dejenere
geometri ayrıca `unmeasurable` üretir. Hareket video timeline'ında yoksa kural
tanımı kaybolmaz; M07–M18 satırları
`blocked_missing_reference / movement_not_present_in_timeline` olarak kalır.
Pipeline sınırı olan 8 kural dışında kalan 166 kuralın tamamı
`measurement_evaluator_status=implemented` taşır. Eksik sayısal teknik hedef
`blocked_missing_reference`; var olan evaluator içindeki yetersiz gerçek
landmark kanıtı ise `unmeasurable` olur. `evaluator_not_implemented` gözlenen
M01–M06 kapsamında sıfırdır.

## Hareket kontratları

Aktif PoomsaeSpec'teki her M01–M18 hareketi için tekniğe ve duruşa göre ortak
semantik kontratlar çözülür. Kontratlar aktif kol, reaction/chamber kolu, ön ve
arka bacak, hareket eden ayak, destek/pivot ayağı, hazırlık/icra/fixation,
hedef bölge/yükseklik/yan/derinlik, dirsek-bilek ilişkisi, duruş uzunluğu ve
genişliği, diz aralığı, izin verilen geçiş hareketi ve kanıt kapılarını taşır.

Teknik tablolar yalnız aktif spec'te bulunan `arae_makki`,
`momtong_jireugi`, `momtong_an_makki`, `eolgul_makki`, `ap_chagi` ve pose ile
ölçülemeyen `kihap` olayını kapsar. M14/M16 tekme-yumruk bileşik kontratıdır;
kihap ses kanıtı pose'dan uydurulmaz.

## Sporcu-yerel yön bağı

`initial_left/right/forward/backward` etiketleri dünya ekseni sayılmaz.
Opsiyonel yön referansı şu bağları taşır:

- session kimliği ve reference pose SHA-256;
- gravity/up ve yataya izdüşürülmüş initial-forward vektörü;
- manuel veya türetilmiş session-bound provenance;
- yalnız `validated_diagnostic_reference` kalite durumu.

Doğrulayıcı sonluluk, norm, yataylık, diklik, dejenerasyon ve tanımlı el
yönlülüğünü kontrol eder. Bağ yoksa gövdeye göre bağıl ölçümler çalışır;
mutlak baş/torso/pelvis/duruş yönü `null` ve
`missing_athlete_local_direction_binding` olur. Dünya ekseni veya sporcunun
gözlenen yönü “beklenen doğru” diye tahmin edilmez. Manuel bağ üretim
kalibrasyonu değildir.

### Referansın üretimi

Bağ her run'da kendi kaydından türetilir; sabit bir dosya olarak commit'lenmez.
`scripts/build_athlete_local_direction_reference.py` açılış hareketinin
(`M01`, varsayılan olarak `preparation` çapası) anchor penceresinde torso-forward
geometrisinin medyanını alır, yerçekimi ekseninde yataya izdüşürür ve sonucu
timeline'ın `source_binding` session kimliği ile pose SHA-256'sına bağlar. Ölçülen
şey sporcunun o kayıtta baktığı yöndür; dünya ekseni ya da hakem referansı değildir
ve `basis_source=derived_session_bound`, `production_calibration_claim=false`
kalır.

Kanonik run önce bu aşamayı koşar, sonra teşhis aşamasına referansı geçirir.
Aşama her koşulda `json/athlete_local_direction_reference_status.json` yazar:
türetilebildiyse `status=derived` ve referans, türetilemediyse `status=not_derived`
ve profilin skip kodlarından biri (`movement_not_present_in_timeline`,
`movement_contract_incomplete`, `insufficient_valid_samples`,
`degenerate_body_axis`). Türetilemediğinde referans dosyası hiç yazılmaz, teşhis
aşamasına `--direction-reference` geçilmez ve 17 yön kuralı bugünkü gibi
`blocked_missing_reference` kalır. Run bu yüzden düşmez; boşluk sessizce
kapanmaz, kayda geçer.

`build_technical_accuracy_diagnostics` gelen referansı timeline'ın session
kimliği ve pose hash'iyle karşılaştırır. Başka bir oturumdan gelen referans
kabul edilmez; yön kuralları yanlış geometriyle açılamaz.

## Baş yönelimi göz bakışı değildir

Yüz göz kümeleri, burun/yüz geometrisi, omuz hattı ve torso-up kullanılarak
`head_orientation_proxy` ölçülür. Yaw, pitch ve roll ayrı tutulur. Pipeline;
pupil, göz küresi, görsel dikkat, rakip farkındalığı veya gerçek gaze fixation
ölçtüğünü iddia etmez. İşaret/ön yön belirsizse imzalı karar kapanır; yararlı
imzasız ölçüm `measurement_only` kalabilir.

## Geometrik denge sınırı

Fixation metrikleri baş/torso/pelvis/el/ayak dağılımı, yükseklik değişimi ve
sonradan düzeltmeyi ölçer. Pelvis ortası yalnız pelvis/body-centre proxy'sidir.
Merkez kütle, basınç merkezi, ayak basıncı, gerçek ağırlık dağılımı,
ground-reaction force, darbe gücü veya kas gerilimi ölçülmez.

## Eşik ve karar politikası

Bütün sayısal değerler v3 YAML içindeki `thresholds` alanındadır; evaluator
fonksiyonlarına gömülü eşik yoktur. Her eşik birim, operatör, belirsizlik bandı
ve ortak provenance politikasını taşır. Başlıca istek-bağlı geçici değerler:

- baş hedef/torso yaw `25°`, roll `15°`, pitch `20°`, baş fixation/drift `10°`;
- torso hedef `25°`, pelvis hedef `30°`, stance hedef `25°`, torso-pelvis
  bağıl yaw `20°`;
- torso lean `15°`, lateral bend/omuz roll/pelvis roll `12°`;
- torso/pelvis fixation ve drift `8°`;
- ayak slip/stance dispersion `0,04` bacak uzunlığı, landing hatası `0,15`;
- elbow flare `0,20`, kol düzlemi `0,15`, el/kol fixation `0,05`, geç düzeltme
  `0,08` kol uzunluğu;
- wrist-forearm `20°`, fist yön `25°`;
- bileşen settle farkı `0,20 s`.

Ap-seogi ve ap-gubi duruş/diz aralıkları YAML kontratında ayrı tutulur. Bunlar
mevcut sporcunun videosundan ayarlanmadı. Tarihsel kaynak-bağlı arka ayak,
arae-makki, momtong-an-makki ve eolgul geometrileri v3 tarafından yeniden
kesintiye çevrilmez; karar sahipliği source-bound katmanındadır.

Her geçici adayın değişmez alanları:

```text
decision_status = review_candidate_not_deduction
score_effect = null
deduction_points = null
numeric_score_enabled = false
deduction_enabled = false
rule_eligibility = blocked_unvalidated_screening_threshold
provenance = self_authored_temporary_accuracy_rule
```

## Çıktılar ve güncel kanıt kapsamı

Her kanonik run şunları ekler:

- `json/athlete_local_direction_reference_status.json` (ve türetilebildiyse
  `json/athlete_local_direction_reference.json`);
- `json/technical_accuracy_diagnostics_report.json`;
- `csv/technical_accuracy_coverage_matrix.csv`;
- `csv/technical_accuracy_landmark_coverage.csv`.

JSON tam kural envanterini, M01–M18 kontratlarını, kural-hareket matrisi ve
M01–M06 ölçüm sonuçlarını taşır. Ayrıca 0–132 arasındaki 133 landmarkın
tamamını; adını/bölgesini, hangi kurallar tarafından gerekli ilan edildiğini,
aktif eşikli bir kurala bağlanıp bağlanmadığını ve aktif değilse kapsam
gerekçesini ayrı `landmark_inventory` alanında taşır. Bu envanter her
landmarkın bağımsız bir teknik hata olduğu anlamına gelmez. Ayrıntılı el ve
ayak geometrisi kalite kapısından geçerse ölçülür. Aktif eşikli kurallar 51
landmarkı doğrudan gerekli ilan eder: 12 body, 23 yüz, 6 ayak ve iki elde
toplam 10 palm/wrist noktası. Kalan parmak ve yüz detayları ölçüm/observability
sözleşmesindedir; sırf 133 noktayı tüketmiş görünmek için bağımsız hata
üretilmez.

Her ölçüm en az 3 geçerli örneğe ek olarak profilin varsayılan `%75` zorunlu
landmark-grup kapsamını geçmelidir. Bu kapı hareket/pencere kanıtına uygulanır;
birkaç tesadüfi geçerli el veya ayak örneği aday üretmeye yetmez.

HTML ayrı “Kapsamlı teknik doğruluk
teşhisleri · puan yok” bölümünü gösterir. Eşik dışı v3 olayları
`decision_evidence_events.json` üzerinden mavi diagnostic video olaylarına
dönüşür; görselleştirme yeniden değerlendirilmez.

Sayısal eşiği olmayan boolean adaylar sayısal `null` limit gibi işlenmez.
EvidenceEvent içinde `rule_operator=bool_true`, `expected_boolean=true`,
`rule_limits=[]` ve gerçek boolean ölçüm taşırlar. `false`, yalnız puansız
inceleme adayıdır; `float(null)` dönüşümü veya sahte `0/1` sayısal eşik yoktur.
Eşiksiz fakat boolean olmayan bir aday fail-closed sözleşme hatasıdır.
Sayısal `range` eşikleri iki elemanlı limit dizisi, `abs_max` ise mutlak-değer
operatörü olarak korunur; sunum adaptörü listeyi veya `null` değeri skaler
`float` gibi yorumlayamaz.

Aktif kayıt yalnız M01–M06'dır. M07–M18 kontrat/şema/sentetik test kapsamıdır;
gerçek video kanıtı değildir. Dış 3B ground truth ve hakem kalibrasyonu
bulunmadığından eşiklerin precision/recall veya resmî hakem uyumu bilinmez.

## Kural doğrulama düzeneği

`config/scoring/validation/taegeuk_1_rule_accuracy_validation_v1.yaml`, runtime
puanlamasından ayrı sentetik mühendislik doğrulamasını tanımlar. Her aktif
kural çift yönlü sınır/fail, eksik, yanlış-tip ve NaN/±sonsuzluk vakalarından geçer. WholeBody-133
fixture üzerinde yüz, el, ayak, kamera ve reprojection kanıt kaybı; dejenere
geometri; kontrollü fixation drift'i; sağ-sol ayna; BODY-17 sözleşme reddi ve
session-bound yön davranışı çalıştırılır. 133 landmarkın tamamı ayrı coverage
satırı taşır; eksik kanıt hedefleri önce ölçülebilir baz çizgiye karşı
karşılaştırılır. Artifact seti input/implementation SHA-256 bağları ve çıktı
manifesti taşır. Ayrıntılı protokol ve komut:
[`TECHNICAL_ACCURACY_RULE_VALIDATION.md`](TECHNICAL_ACCURACY_RULE_VALIDATION.md).

Harness başarısı yalnız yazılım sözleşmesinin doğrulandığını gösterir. Gerçek
teknik hata doğruluğu için kural başına uzman etiketli video, kör hakem
karşılaştırması ve precision/recall analizi hâlâ gereklidir.
