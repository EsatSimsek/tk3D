# TK3D Güncel Proje Durumu

Son doğrulama tarihi: **2 Eylül 2026**

Dal: **`main`**

HEAD: **bu belgenin bulunduğu teslim commit'i (`git rev-parse HEAD`)**

Çalışma ağacı: **teslim commit'i dışında beklenen repository değişikliği yok;
yerel `outputs/` validation run'ları Git dışıdır**

Bu dosya yalnız güncel ve doğrulanmış durumu özetler. Final Polish öncesindeki
905 satırlık faz/pilot günlüğü
[`docs/history/PROJECT_STATUS_PRE_FINAL_POLISH.md`](docs/history/PROJECT_STATUS_PRE_FINAL_POLISH.md)
altında değiştirilmeden korunur. Sistem bağlamı için
[`docs/PROJECT_CONTEXT.md`](docs/PROJECT_CONTEXT.md), kurulum ve komutlar için
[`README.md`](README.md) kullanılmalıdır.

## 1. Güncel doğrulanmış durum

TK3D'nin kanonik araştırma hattı, iki ZED 2i kameradan alınan senkron AVI RGB
ve eşlenmiş SVO2 depth kayıtlarını WholeBody-133 3B poza, ardından
kaynak-bağlı Poomsae analiz/diagnostic/review artifact'lerine dönüştürür.

Mevcut mühendislik kapsamı şunları içerir:

- kurulabilir paket ve üç kanonik console entry point;
- kalibrasyonlu multiview uygulama API'si ve ince CLI adaptörü;
- run manifesti, immutable run dizini, lifecycle ve latest-success güvenliği;
- şemalı 3B/kalite artifact sözleşmeleri ve fail-closed hash bağları;
- Poomsae uygulama API'si, WholeBody-133 teşhisleri, provisional kaynak-bağlı
  kararlar, readiness, video/HTML inceleme ve run history;
- clean-checkout Tier 1 ve yerel CURRENT_ACTIVE Tier 2 reproducibility kontrolü;
- davranışı değiştirmeyen opsiyonel performans enstrümantasyonu.

Sistem **resmî Poomsae puanlamasına veya bağımsız olarak doğrulanmış bilimsel
doğruluk iddiasına hazır değildir**. Bu sınır kullanıcı çıktılarında fail-closed
durumlarla açıkça korunur.

31 Ağustos 2026'da kanonik Poomsae uygulamasına ayrı, puansız v3 teknik
doğruluk katmanı eklendi. Aktif PoomsaeSpec'in M01–M18 hareketlerinin tamamı
kontrata çözülür; 174 kurallık envanter 33 `active_diagnostic`, 116
`measurement_only`, 17 `blocked_missing_reference` ve 8
`not_observable_with_current_pipeline` kural taşır. Aktif kayıt kanıtı yine
yalnız M01–M06'dır. Geçici adaylar source-bound karar, Accuracy skoru,
Presentation veya readiness'i değiştiremez.

Landmark kapsam envanteri 0–132 arasındaki `17 body + 6 foot + 68 face + 42
hand = 133` noktanın tamamını listeler. Tamamı en az bir kural sözleşmesine
bağlıdır; 51 landmark aktif eşikli değerlendiricilerce doğrudan gerekli ilan
edilir: 12 body, 23 yüz, 6 ayak ve iki elde toplam 10 palm/wrist noktası. Kalan
parmak/yüz detayları ölçüm veya observability sözleşmesindedir. Bu ayrım yeni
`csv/technical_accuracy_landmark_coverage.csv` artifact'inde açıkça raporlanır.

## 2. CURRENT_ACTIVE workflow

Aktif profil: `config/scoring/profiles/poomsae1_trimmed.yaml`

```text
iki ZED 2i
  -> senkron AVI RGB + SVO2 depth/timestamp mapping
  -> RF-DETR Small + ByteTrack
  -> ViTPose-Huge WholeBody 2B (133 nokta, flip test)
  -> causal filtre + offline 2B stabilizasyon
  -> robust multiview triangulation
  -> güven kapılı BODY-17 ZED depth fusion
  -> BODY-17 global sekans optimizasyonu/fallback
  -> anatomik-zamansal güvenilirlik + 3B stabilizasyon
  -> keypoints_3d_world[t, 133, 3]
  -> Poomsae analiz, diagnostic, provisional karar ve inceleme çıktıları
```

Ana 3B koordinat sözleşmesi metre, `x=sağ`, `y=ileri`, `z=yukarı`dır.
BODY-17 işlemleri 133 noktalı exportu küçültmez. İki kamera, en az dört başka
view isteyen cross-view guided ikinci geçiş için doğal olarak `ZERO_WORK`
üretebilir; bu durumda görüntü kanıtı uydurulmaz.

## 3. Doğrulanmış yerel ortam

| Bileşen | Doğrulanmış değer |
| --- | --- |
| İşletim sistemi | Windows 11 `10.0.26200`, AMD64 |
| Python | CPython `3.12.13` |
| NumPy | `2.5.1` |
| SciPy | `1.18.0` |
| OpenCV | `5.0.0.93` |
| PyTorch | `2.13.0+cu130` |
| TorchVision | `0.28.0+cu130` |
| CUDA runtime | `13.0` |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| RF-DETR | `1.7.0` |
| Supervision | `0.27.0` |
| timm | `1.0.28` |

Paket metadata'sı Python `>=3.11` ister. CI Windows/Python 3.11 ve CPU Torch
hedefler; gerçek CUDA/ZED inference yalnız yukarıdaki yerel Python 3.12
araştırma ortamında doğrulanmıştır. Ayrıntı:
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## 4. Güncel test ve kalite kapısı

2 Eylül 2026 boolean technical-accuracy EvidenceEvent ve lifecycle düzeltmesi:

- `python -m ruff check src scripts tests`: geçti;
- güncel `origin/main` üzerine rebase sonrası tam pytest:
  **`329 passed in 50.87s`**;
- decision-evidence + runner odaklı testler: **`29 passed in 15.91s`**;
- `git diff --check`: temiz;
- gerçek `benim-denemem-17` artifact'lerinden ayrı validation çıktısında
  **70** event, **50** teknik aday ve sorun çıkaran **6/6** boolean event
  üretildi; boolean event'lerin tamamı puansız kaldı;
- yeni uçtan uca `benim-denemem-17-fix-r1` run'ı tamamlandı; 70 event,
  işaretli video (**40.429.676 byte**), HTML review, run history ve ana özet
  üretildi;
- yeni run `run_state=completed`; eski yarım `benim-denemem-17` run'ı hata
  açıklamasıyla `failed` olarak kapatıldı; önceki uygun `latest_run.json`
  işaretçisi değiştirilmedi.

1 Eylül 2026 teknik-doğruluk kural validation inceleme ve güçlendirme kapısı:

- `python -m ruff check src scripts tests`: geçti;
- tam pytest: **`316 passed in 48.16s`**;
- validation + teknik-doğruluk odaklı testler: **`24 passed in 20.21s`**;
- `git diff --check`: temiz;
- sentetik validation CLI: **`passed`**;
- kural envanteri **`174/174`**, landmark kapsamı **`133/133`**,
  aktif-kural çift-yönlü sınır/eksik/non-finite/yanlış-tip vakaları
  **`330/330`**, WholeBody-133 geometri senaryoları **`12/12`** geçti;
- JSON `allow_nan=false` ile parse edildi; ham non-finite sayı yazılmadı;
- eksik-kanıt hedeflerinin baz çizgide önce ölçülebilir olduğu, sıfır olmayan
  ayna metriklerinin değişmezliği, dört pose/kalite dizisinin ayna involution'ı
  ve geçerli bağla yön kurallarının **`17/17`** değerlendirildiği doğrulandı;
- beş ana artifact'in SHA-256/boyut manifesti tekrar hesaplanarak doğrulandı;
  CLI staging dizininden atomik yayın yapıyor ve run dizini tekrarını reddediyor;
- rapor readiness'i `synthetic_contract_validation_ready=true` fakat dış
  doğruluk, hakem kalibrasyonu, üretim eşiği ve resmî puan hazırlığını `false`
  tutuyor;
- doğrulama çıktısı
  `outputs/validation/runs/taegeuk1-rule-validation-20260901-r6/` altında
  benzersiz run olarak üretildi.

Bu düzeneğin başarısı yazılım sözleşmesi ve fail-closed davranışı doğrular.
Uzman etiketli gerçek video, kör hakem karşılaştırması ve precision/recall
olmadan biomekanik, resmî veya ground-truth doğruluk iddiası yapılamaz.

31 Ağustos 2026 kapsamlı teknik-doğruluk teslim kapısı:

- `python -m ruff check src scripts tests`: geçti;
- tam pytest: **`314 passed in 35.88s`**;
- `git diff --check`: temiz;
- v3 odaklı testler: **`15 passed`** (tam koşuya dahil);
- gerçek 741-kare Poomsae analizi multiview inference yeniden çalıştırılmadan
  mevcut doğrulanmış reference pose ile tamamlandı;
- gerçek run'ın 16 JSON'u parse edildi; JSON/CSV'de `NaN`/`inf` bulunmadı;
- işaretli video karesi ve HTML'deki ayrı “puan yok” teknik doğruluk bölümü
  görsel olarak incelendi.

28 Ağustos 2026 Final Polish teslim kapısı:

- `python -m ruff check src scripts tests`: geçti;
- tam pytest: **`273 passed in 20.34s`**;
- `python -m pip check`: `No broken requirements found`;
- `git diff --check`: temiz;
- `tk3d-check`: `READY`;
- Tier 1 clean/lightweight: `READY`;
- CURRENT_ACTIVE local research: `READY`;
- `tk3d-check --help`, `tk3d-multiview --help` ve `tk3d-poomsae --help`:
  geçti.

Bu Final Polish geçişi belge ve üç CLI adaptöründeki kullanıcı metniyle
sınırlıdır. Bilimsel/runtime davranışı değiştirilmediği için 260/741-kare gerçek
inference yeniden çalıştırılmadı.

## 5. Son tam CURRENT_ACTIVE multiview referansı

Tam aktif bilimsel referans run:

`outputs/poomsae_1_zed2i_20260731_trimmed/runs/poomsae1-zed2i-rgbd-gated-ultra-rerun-20260802/`

- iki ZED 2i, `741/741` kare, 60 FPS, stride 1;
- ViTPose-Huge WholeBody ve 133 noktalı çıktı;
- `7.669` BODY-17 depth fusion noktası kullanıldı;
- geçerli BODY-17 oranı `%97,8884`;
- ortalama reprojection `5,210 px`;
- depth adayı final RGB-vs-depth kalite kapısını geçti;
- global optimizer kabul edildi, fallback kullanılmadı;
- iç geometri ve sensör tutarlılığı geçti;
- JSON/CSV'de `NaN`/`inf` sızıntısı yok;
- dış 3B ground truth bu run için değerlendirilmedi;
- `provisional_scoring_ready:true`, `official_scoring_ready:false`.

Bu run'ın tarihsel generic `71,7178/100` çıktısı artık kaldırılmış kaynaksız
legacy motora aittir; güncel sistem bunu yeniden üretmez ve resmî skor olarak
kullanmaz.

Phase C uygulama smoke'u ayrı validation root'ta aynı session'ın ilk 30
karesiyle çalıştı. İlk 26 kare korunmuş tam run ile sayısal olarak aynıydı;
son dört karedeki fark 9-kare offline filtrenin beklenen sağ-sınır etkisidir.
Ana session'ın `latest_run.json` işaretçisi smoke için değiştirilmedi.

## 6. Son CURRENT_ACTIVE Poomsae sonucu

Son yerel teknik-doğruluk regresyon run'ı:

`outputs/poomsae_1_zed2i_20260731_trimmed/runs/taegeuk1-comprehensive-active-metrics-20260831-r2/`

- durum: `provisional_observed_scope_analysis_generated`;
- bağlı 3B pose: 741 kare, WholeBody-133;
- kayıt kapsamı: partial sequence; M01–M06 seçili kapsamı `6/6`, tam Poomsae
  kapsamı `6/18`;
- otomatik segmentasyon `6/6`; faz anchor MAE `6,041667` kare;
- WholeBody ölçüm kapsamı `67/87`;
- puansız WholeBody inceleme adayı `10`;
- technical-conformance inceleme gereken hareket `6/6`;
- gözlenen-kapsam provisional kesinti toplamı `0,4` ve `4` kaynak-bağlı küçük
  karar; tam Accuracy değildir;
- v3 envanter `174` kural ve `18/18` hareket kontratı üretti;
- pipeline sınırındaki 8 kural dışında `166/166` ölçüm evaluator yolu
  uygulanmış, gözlenen kapsamda `evaluator_not_implemented=0` kalmıştır;
- gerçek M01–M06 toplamında `589` kural ölçüldü, `153` aktif kural
  değerlendirildi, `109` aralık-içi, `37` puansız aday ve `7` sınır-belirsiz
  sonuç oluştu;
- gözlenen M01–M06'da `436` measurement-only, yön ve eksik sayısal teknik
  hedef nedeniyle `114` bloke, `239` ölçülemez ve `48` pipeline'da
  gözlenemez kural-hareket sonucu raporlandı; bunların `85`i `%75` zorunlu
  landmark-grup kalite kapısında güvenle kapandı;
- M07–M18 için video ölçümü yapılmadı; bütün uygulanabilir satırlar
  `movement_not_present_in_timeline` olarak bloke kaldı;
- tam Accuracy `null`, resmî skor `null`;
- `rule_scoring_ready:false`;
- `judge_calibrated_ready:false`;
- `official_scoring_ready:false`.

Önceki karşılaştırılabilir `taegeuk1-accuracy-v3-20260831-r2` run'ına göre
source-bound numeric/categorical/applied karar dizileri ve karar özeti eşittir:
`4` küçük karar, `0,4` gözlenen-kapsam toplamı, `3` ölçülemez ve `1`
sınır-belirsiz karar değişmedi. Presentation bileşenleri eşit ve iki run'da da
Presentation toplamı `null` kaldı. V3'ün `37` adayı score/deduction etkisi
olmayan mavi evidence olaylarıdır.

## 7. Phase D performans özeti

Phase D yalnız ölçüm/enstrümantasyon ekledi; model, threshold, örnekleme,
geometri, depth fusion, optimizer veya Poomsae karar davranışını optimize
etmedi.

Üç eş 260-kare CURRENT_ACTIVE profilli koşunun ortalaması:

| Ölçüm | Ortalama |
| --- | ---: |
| Internal toplam | `282,778 s` |
| Core işleme | `218,503 s` |
| Steady-state 140–259 inference+geometry | `1,833 FPS` |
| ZED depth fusion | `70,688 s` |
| ViTPose + causal 2B | `61,254 s` |
| Triangulation | `52,057 s` |
| Artifact serialization | `38,546 s` |
| RF-DETR + ByteTrack | `25,174 s` |
| RGB/depth optimizer toplamı | `7,328 s` |

Torch peak allocated/reserved sırasıyla `3642,120 MiB` ve `3854 MiB` idi;
ZED SDK belleği ve process-level RAM/CPU buna dahil değildir. Profiler-disabled
tek koşuya karşı `%4,04` kaba fark, tekrar CV'si `%9,26` olduğu için overhead
kanıtı olarak yorumlanmaz.

741-kare Poomsae profili `56,673 s` sürdü: analiz/karar `12,821 s`,
presentation/export `43,597 s`. En büyük sunum maliyetleri hata videosu
`25,036 s` ve browser video üretimi `15,863 s` oldu.

Üç profilli multiview koşunun 12 CSV'si profillemesiz eş koşuyla byte-eşitti;
ana 3B bilimsel alanlar ve normalize kalite raporları da eşitti.

## 8. Bilimsel doğrulama durumu

CURRENT_ACTIVE ZED workflow'u için şu anda bağımsız 2B annotation, bağımsız 3B
mocap/ölçüm ground truth veya uzman/hakem karar ground truth'u yoktur.

Bu nedenle ölçülebilenler:

- iç geometri ve reprojection dağılımları;
- kamera kanıtı ve BODY-17 geçerlilik;
- RGB-vs-SVO2 iç sensör tutarlılığı;
- depth/optimizer acceptance-fallback durumu;
- temporal, açı ve kemik kararlılığı;
- tek kayıt içindeki M01–M06 manuel timeline'a göre segment sınır/anchor hatası;
- artifact, provenance ve davranış regresyonu.

Henüz ölçülemeyen/iddia edilemeyenler:

- mutlak CURRENT_ACTIVE 3B doğruluğu;
- genellenebilir 2B keypoint doğruluğu;
- el/yüz/ayak teknik teşhis doğruluğu;
- tam Taegeuk 1 segmentasyon doğruluğu;
- uzman/hakem hata uyumu ve resmî puan doğruluğu.

AIST `CURRENT_VALIDATION` ikincil smoke'tur. MADS F2
`HISTORICAL_BENCHMARK` sonucu yalnız bağlı RGB-only koşusuna aittir ve ZED RGBD
doğruluğuna devredilmez.

## 9. Readiness durumu

| Durum | Güncel değer | Anlam |
| --- | --- | --- |
| Tier 1 clean/lightweight | `READY` | Paket/config/import/fixture ve hafif sözleşmeler hazır |
| CURRENT_ACTIVE yerel varlıklar | `READY` | Bu makinede bağlı model, video, SVO2, calibration ve pose mevcut |
| İç multiview kalite | `passed` | Son tam aktif referansta iç geometri/sensör kapıları geçti |
| `provisional_scoring_ready` | `true` | Kaynak-bağlı provisional analize veri hazırlığı var |
| `rule_scoring_ready` | `false` | Tam ve doğrulanmış kural kanıtı hazır değil |
| `judge_calibrated_ready` | `false` | Uzman/hakem kalibrasyon verisi yok |
| `official_scoring_ready` | `false` | Resmî puan önkoşulları sağlanmadı |

`tk3d-check` Final Polish son doğrulamasında da `READY` döndürdü.

## 10. Bilinen sınırlamalar

1. Aktif session, video/SVO2/calibration/reference pose ve checkpoint'ler Git
   dışı yerel araştırma varlıklarıdır; clean checkout tek başına Tier 2 çalışmaz.
2. Session YAML içinde makineye özgü SVO2/timestamp yolları vardır.
3. Bağımsız CURRENT_ACTIVE ground truth yoktur.
4. Manuel doğrulanmış Poomsae timeline yalnız M01–M06 kapsamındadır.
5. El, yüz ve ayak noktaları 133 exportta korunur; BODY-17 ile aynı depth fusion
   veya global optimizer doğrulamasına sahip değildir.
6. İki kameralı aktif kayıtta dört destekleyici view isteyen cross-view guided
   ikinci geçiş çalışmaz.
7. Uzman/hakem etiketi ve judge calibration yoktur.
8. Repository'de açık bir public license dosyası yoktur; bu durum dış dağıtım
   öncesinde açıklığa kavuşturulmalıdır.
9. CI yapılandırması yerel olarak doğrulandı, fakat bu kirli çalışma ağacının
   GitHub-hosted CI sonucu henüz yoktur.

## 11. Opsiyonel gelecek çalışmaları

- M07–M18 için manuel/uzman doğrulanmış hareket ve faz etiketleri;
- farklı sporcu, seviye, kıyafet, kamera düzeni ve oturumlarla değerlendirme;
- imkân olduğunda senkron bağımsız mocap/ölçüm ground truth;
- uzman/hakem annotation ve opsiyonel judge-calibration çalışması;
- yüz/el/ayak için ayrı dış doğrulama ve güvenli multiview optimizasyon;
- üç veya daha fazla ZED saha düzeninin ayrı pilot ve stres doğrulaması;
- public dağıtım hedeflenirse license, örnek veri ve indirilebilir asset akışı.

Bu maddeler mevcut dürüst araştırma kapsamının çalışması için zorunlu yeni fazlar
değildir.

## 12. Tarihsel dokümantasyon

- Final Polish öncesi tam durum günlüğü:
  [`docs/history/PROJECT_STATUS_PRE_FINAL_POLISH.md`](docs/history/PROJECT_STATUS_PRE_FINAL_POLISH.md)
- Final Polish öncesi tam README:
  [`docs/history/README_PRE_FINAL_POLISH.md`](docs/history/README_PRE_FINAL_POLISH.md)
- Erken oturum özetleri: `sohbet1.md`–`sohbet6.md`
- Erken mimari deep dive: `tk3d_architecture_deep_dive.md`
- Tarihsel ZED yol haritası:
  [`docs/ZED2I_OFFLINE_MULTICAMERA_PLAN.md`](docs/ZED2I_OFFLINE_MULTICAMERA_PLAN.md)

Tarihsel belgeler güncel komut, readiness veya workflow gerçeğinin önüne
geçmez. Çelişkide güncel kod/config, `AGENTS.md` ve bu dosya esas alınır.
