# TK3D Güncel Teknik Bağlam

Bu belge TK3D'nin **güncel teknik hafızasıdır**. Mühendislerin ve yeni
AI/Codex oturumlarının sistemi tarihsel geliştirme günlüğünü okumadan doğru
sınırlar içinde anlamasını amaçlar. Değişken test, benchmark ve run sonuçları
burada tutulmaz; onlar için [`../PROJECT_STATUS.md`](../PROJECT_STATUS.md)
kullanılır. Final Polish öncesi ayrıntılı tarihsel README
[`history/README_PRE_FINAL_POLISH.md`](history/README_PRE_FINAL_POLISH.md)
altında aynen korunur.

## 1. Amaç ve ürün sınırı

TK3D'nin amacı kalibrasyonlu çok-kameralı videodan güvenilir 3B insan pozu
üretmek ve bu kanıtı tekvando poomsae teknik analizine taşımaktır. Repository
iki ana uygulama sınırı içerir:

1. Senkronize kameralardan WholeBody-133 2B/3B rekonstrüksiyon, kalite ve
   provenance üretimi.
2. Bağlı 3B artifact, video kanıtı, hareket zaman çizelgesi ve kural
   kaynaklarından Poomsae teşhis/karar/inceleme artifact'leri üretimi.

Bu bir araştırma ve mühendislik sistemidir. Güncel iç kalite kapıları resmî
puan doğruluğunu, dış 3B doğruluğu veya hakem eşdeğerliğini kanıtlamaz.

## 2. Kanonik CURRENT_ACTIVE workflow

Güncel aktif geliştirme örneği `poomsae1_trimmed` profilidir:

```text
iki ZED 2i kayıt seti
  -> senkron AVI RGB kareleri
  -> RF-DETR Small kişi tespiti
  -> ByteTrack kamera-içi kimlik sürekliliği
  -> stabilize/adaptif kişi crop'u
  -> ViTPose-Huge WholeBody 2B + flip test
  -> causal temporal filtre + offline sıfır-fazlı 2B stabilizasyon
  -> robust çok-kameralı triangulation
  -> varsa güvenli cross-view image-guided ikinci geçiş
  -> ZED SVO2 depth ile BODY-17 yardımcı fusion
  -> BODY-17 global sekans optimizasyonu
  -> anatomik/zamansal güvenilirlik değerlendirmesi
  -> 3B robust Savitzky–Golay stabilizasyonu
  -> WholeBody-133 3B + kalite + provenance artifact'leri
  -> kaynak-bağlı Poomsae analiz ve insan inceleme akışı
```

Aktif workflow'un birincil davranış/regresyon kontrolü bu hat üzerinde
yapılmalıdır. AIST veya tarihsel MADS sonucu bunun yerine kullanılamaz.

## 3. İki-ZED giriş düzeni

Aktif session:

`outputs/poomsae_1_zed2i_20260731_trimmed/source/session.yaml`

Session iki fiziksel kamerayı tanımlar:

- `zed_35151067`
- `zed_37137479`

Aktif kayıt 60 FPS, kullanıcı tarafından seçilmiş 741 karelik trimdir. Session
YAML, kamera kimliklerini, AVI yollarını, senkron/trim bilgisini ve her kamera
için SVO2/timestamp eşlemesini bağlar. Bu `outputs/` varlıkları Git ile
dağıtılmaz ve makineye özgü yollar içerebilir.

### AVI RGB rolü

AVI dosyaları görüntü kanıtının kanonik RGB kaynağıdır. Decode edilen kareler;
kişi tespiti, tracking, crop, ViTPose heatmap'i, 2B gözlem, triangulation ve
inceleme videosu için kullanılır. Frame index ve timestamp kimliği kaynak
zaman çizelgesiyle korunur.

### SVO2 depth rolü

SVO2 dosyaları aynı ZED kayıtlarının stereo depth kaynağıdır. Session'daki
timestamp mapping ve prepared frame offset, AVI kareleri ile SVO2 depth
örneklerini eşler. SVO2, RGB 2B görüntü kanıtının yerine geçmez; triangulation
sonrası yalnız yardımcı, kapılı BODY-17 depth fusion sağlar. IMU kullanımı
gravity/orientation calibration içindir; kare-bazlı sporcu hareket düzeltmesi
değildir.

## 4. 2B kişi ve pose katmanı

### RF-DETR Small + ByteTrack

Aktif kişi dedektörü RF-DETR Small'dır. Kişi sınıfı bbox'ları üretir; ByteTrack
kamera içinde hedef kimliğini sürdürür, kısa kayıpları yönetir ve yeniden
edinime yardımcı olur. Bbox padding, hareket duyarlı crop ve stabilizasyon,
ViTPose girdisinin kişiyi güvenli biçimde kapsamasını sağlar.

Bu katman 3B eklem üretmez ve tek başına poomsae kararı vermez. Dedektör/tracker
çıktısı yalnız sonraki görüntü-temelli WholeBody tahminini sınırlar.

### ViTPose-Huge WholeBody

Aktif 2B model:

- ad: `ViTPose-Huge-WholeBody`
- backend: `tk3d_vitpose_plus`
- giriş: `256x192`
- çıktı: 133 COCO-WholeBody heatmap/keypoint
- flip test: açık
- cihaz: aktif yerel profilde `cuda:0`

2B tahminler güven ve hareket duyarlı causal filtre ile işlenir; tam sekans
mevcut olduğunda ayrıca sıfır-fazlı offline stabilizasyon uygulanır. Bu
stabilizasyon kare atamaz, video süresini kısaltamaz veya 133 noktayı azaltamaz.

## 5. WholeBody-133 sözleşmesi

Kanonik indeks yerleşimi:

| İndeks | Grup | Adet |
| --- | --- | ---: |
| `0:17` | COCO body | 17 |
| `17:23` | ayak (iki ayakta big toe, small toe, heel) | 6 |
| `23:91` | yüz | 68 |
| `91:112` | sol el | 21 |
| `112:133` | sağ el | 21 |

Her el; wrist ile thumb/index/middle/ring/pinky eklem zincirlerini içerir.
Güncel ana 3B dizi sözleşmesi:

```text
keypoints_3d_world[t, 133, 3]
```

Eksik koordinatlar bellekte non-finite olabilir, fakat JSON'a `null`, CSV'ye
boş hücre olarak yazılmalıdır. Downstream artifact'lerde `NaN`/`inf` metni
yasaktır.

## 6. Koordinat sistemi

Kanonik analiz koordinatı `src/coordinate_system.py` ile tanımlanır:

- birim: metre
- `x`: sağ
- `y`: ileri
- `z`: yukarı
- handedness: right

Kalibrasyon importer'ları kaynak koordinatını bu sisteme açık 4x4 dönüşümle
taşır. Scoring ve sentetik fixture'lar aynı eksen sabitlerini kullanmalıdır.
Kaynak-veri setine özgü eksen veya ölçek varsayımı genel runtime'a sessizce
taşınamaz.

## 7. Çok-kameralı rekonstrüksiyon

### Triangulation

Triangulation, farklı kameralardaki bağımsız 2B görüntü gözlemlerinden ilk 3B
geometriyi üretir. Güven eşiği, minimum view sayısı, pozitif depth,
triangulation açısı ve reprojection hatasıyla robust hipotez seçimi yapar.

Her nokta için en az şu kanıtlar korunur:

- `triangulation_score`
- `reprojection_error`
- `used_cameras`
- frame/timestamp kimliği

Ham triangulation sonraki fusion veya optimizasyon tarafından silinmemelidir.

### Cross-view 2B feedback

Hedef kameraya rehberlik edilirken o kamera 3B öncülün triangulation hesabından
çıkarılmalıdır. 3B izdüşüm yalnız heatmap arama öncülüdür; yeni bağımsız görüntü
kanıtı sayılamaz. Aday ancak gerçek heatmap kanıtı ve geometrik kalite birlikte
iyileştiğinde kabul edilir. Yalnız görselleştirme fallback'i ayrı provenance
taşır.

Aktif model config cross-view özelliğini açık tutsa da en az dört destekleyici
view ister. İki kameralı CURRENT_ACTIVE kayıtta bu aşama normal olarak
`ZERO_WORK` olabilir; bu bir çalışma hatası değildir ve sahte destek üretilmez.

### ZED depth fusion

ZED depth fusion yalnız BODY-17 üzerinde çalışır. SVO2 patch depth; ZED güveni,
yüzey uzaklığı, geçerli aralık, kamera eşlemesi ve residual kapılarıyla
değerlendirilir. Saf RGB triangulation referans dalı aynı koşuda korunur.
Depth sonucu ancak depth residual ve final reprojection/temporal/bone kalite
kapıları kabul ederse kullanılır; aksi halde RGB tabanlı güvenli sonuç seçilir.

Depth fusion yüz, el veya ayak için doğrulanmış bir 3B optimizer değildir ve
133 noktalı veri sözleşmesini değiştirmez.

### Güvenilirlik, optimizer ve smoothing

Global optimizer yalnız BODY-17 için tüm senkron sekansı birlikte çözer;
reprojection, kamera ağırlığı, kemik uzunluğu, dirsek/diz limitleri, ivme, jerk
ve kısa kapanma bilgisini kullanır. Solver sonucu; reprojection, ivme, kemik
kararlılığı ve düzeltme büyüklüğü kapılarını geçemezse ham/güvenli sonuca döner.
Kısa temporal recovery doğrudan gözlemden ayrı provenance taşır; uzun kanıtsız
boşluklar doldurulmaz.

Seçilen optimizer/fallback sonucundan sonra anatomik ve zamansal güvenilirlik
katmanı kemik sapması, temporal residual ve ivme kanıtından
`reliability_valid_mask`, rejection reason ve özet üretir. Güvenilir pose'a
robust sıfır-fazlı smoothing uygulanır; global optimizer kabul edildiyse onun
BODY-17 sonucu ayrıca smoothing ile değiştirilmeden korunur. Smoothing kalite
kanıtının yerine geçmez; kare sayısını veya eklem sözleşmesini değiştiremez.

## 8. Ana 3B artifact sözleşmesi

Kanonik dosya:

`outputs/<session_id>/runs/<run_id>/json/vitpose_session_3d.json`

Güncel schema v1 en az şu alanları bağlar:

- `session_id`, `run_id`, `schema_version`
- `coordinate_system`
- `frame_indices`, `timestamps_sec`, `sample_fps`
- `shape.keypoints_3d_world = [t, 133, 3]`
- `keypoints_3d_world`
- `triangulation_score`, `reprojection_error`, `used_cameras`
- `reliability_valid_mask`, rejection reasons ve summary
- optimizer/depth/cross-view raporları ve provenance
- calibration snapshot/hash, model config hash ve `run_manifest.json` bağı

`run_quality_report.json` iç geometri, BODY-17 geçerlilik, reprojection,
depth/optimizer acceptance-fallback ve readiness durumlarını ayrı kaydeder.
Artifact ve manifest kimlik/hash bağları uyuşmazsa güncel tüketici fail-closed
olur. Şemasız tarihsel artifact'ler yalnız açık `LEGACY_SUPPORTED` yolu ile
okunabilir.

## 9. Poomsae uygulama akışı

`src/poomsae_scoring/application.py`, aktif profil ve bağlı artifact'leri tek
run altında orkestre eder. Varsayılan mod profildeki doğrulanmış 3B pose'u
kullanır; `--process-video` aynı bağlı session için stride-1 multiview 3B'yi
önce yeniden üretir.

Yüksek düzey akış:

1. Profil, session, pose, RulePack, PoomsaeSpec, timeline ve diagnostic/accuracy
   config sözleşmelerini ve SHA-256 bağlarını doğrula.
2. Run dizisini ve immutable config snapshot'larını oluştur.
3. Gerekirse bağlı videodan yeni multiview 3B üret ve timeline'ın kare/zaman
   kimliğini güvenli biçimde yeni pose'a aktar.
4. Otomatik hareket/faz segmentasyon önerileri üret; onaylı timeline'ı sessizce
   değiştirme.
5. BODY-17 hareket kanıtı ile WholeBody yüz/ayak/el ölçümlerini çıkar.
6. Kategorik, teknik uygunluk, presentation ve kaynak-bağlı Accuracy karar
   artifact'lerini oluştur.
7. Readiness'i fail-closed hesapla.
8. Kanıt olayları, işaretli video, HTML review, run history ve ana summary yaz.

### WholeBody ve el kullanımı

BODY-17; hareket enerjisi, duruş, açı, merkez ve temel segmentasyon kanıtının
omurgasıdır. Ayak noktaları stance/foot temas-yön teşhislerini; el zincirleri
fist/finger/wrist biçimi gibi teknik ölçümleri; yüz grubu baş/yönelimle ilgili
uygun proxy'leri destekler. Bu gruplar ancak tanımlı kalite/coverage kapıları
altında ölçülebilir sayılır.

WholeBody diagnostic profile `diagnostic_only_unvalidated` durumundadır.
Threshold dışı bir ölçüm yalnız insan inceleme adayıdır; otomatik WT kesintisi,
doğruluk etiketi veya numeric Accuracy skoru değildir.

Kapsamlı v3 teknik-doğruluk katmanı bunun üzerinde M01–M18 hareket kontratı,
174 kurallık envanter ve her kural-hareket için measured/evaluated/state
matrisi üretir. Aktif kayıt yalnız M01–M06 kanıtıdır; M07–M18 config/sentetik
kapsamdır. Sporcu-yerel yön bağı yoksa mutlak yön kuralları fail-closed kalır.
V3 katmanı Accuracy veya Presentation skorunu değiştiremez.

## 10. Readiness ve scoring semantiği

Kritik durumlar birbirine eşit değildir:

- `provisional_scoring_ready`: iç sensör/geometri kalitesi kaynak-bağlı
  provisional analize izin veriyor; resmî doğruluk iddiası yok.
- `rule_scoring_ready`: gerekli timeline, coverage, WholeBody ve karar
  sözleşmelerinin bağlı analiz için geçip geçmediği.
- `judge_calibrated_ready`: uzman/hakem kalibrasyon kanıtının varlığı.
- `official_scoring_ready`: dış doğruluk ve resmî puan önkoşullarının bütünü.

Güncel sistemde tam Accuracy ve resmî skor çoğunlukla `null` kalır;
`judge_calibrated_ready` ve `official_scoring_ready` kapalıdır. Gözlenen kapsam
üzerindeki provisional karar toplamı tam poomsae puanı değildir.

Readiness, bağlı WholeBody-133 diagnostic raporu yoksa, hash/timeline bağı
uyuşmuyorsa, gereken gruplar kanıtlanmıyorsa veya coverage kapısı geçmiyorsa
fail-closed kalır.

## 11. Bilimsel sınırlamalar

- CURRENT_ACTIVE ZED kaydı için bağımsız mocap/3B ground truth yoktur.
- Bağımsız 2B annotation ground truth bulunmadığından keypoint 2B doğruluğu
  yalnız iç tutarlılıkla kanıtlanamaz.
- M01–M06 için sınırlı elle doğrulanmış zaman çizelgesi vardır; tam poomsae
  segmentasyon doğruluğu sonucu değildir.
- Uzman/hakem hata ve puan etiketi yoktur; diagnostic adayların precision,
  recall veya hakem uyumu ölçülemez.
- İki kamera ve self-consistency ölçüleri mutlak 3B doğruluk sağlamaz.
- AIST/MADS sonuçları farklı domain, kamera, calibration ve sensör koşullarına
  bağlıdır; güncel ZED RGBD doğruluğu olarak aktarılamaz.
- El/yüz/ayak noktaları 133 çıktıda korunur, ancak BODY-17 ile aynı depth fusion
  veya global çok-kamera optimizer kapsamına sahip değildir.

Bu nedenle “üretime hazır”, “puanlamaya hazır” veya “ground-truth doğrulanmış”
ifadeleri yalnız yeni ve uygun dış kanıt sağlandığında kullanılmalıdır.

## 12. Konsol giriş noktaları

`pyproject.toml` içindeki kanonik komutlar:

| Komut | Uygulama sınırı |
| --- | --- |
| `tk3d-check` | model yüklemeden Tier 1 ve CURRENT_ACTIVE dış varlık hazırlığı |
| `tk3d-multiview` | kalibrasyonlu WholeBody-133 çok-kamera 3B üretimi |
| `tk3d-poomsae` | kaynak-bağlı Poomsae analiz/diagnostic/review workflow'u |

Doğrudan `scripts/run_vitpose_multiview_3d.py` ve
`scripts/run_poomsae_scoring.py` çağrıları geriye uyumluluk adaptörleridir;
yeni kullanıcı komutlarında console entry point tercih edilir.

## 13. Aktif config ve profil bağları

Ana aktif yollar:

- `config/model_config.yaml`
- `config/mmpose_configs/_base_/default_runtime.py`
- `config/mmpose_configs/wholebody_2d_keypoint/vitpose/coco-wholebody/td-hm_ViTPose-huge_8xb64-210e_coco-wholebody-256x192.py`
- `config/scoring/profiles/poomsae1_trimmed.yaml`
- `config/scoring/rules/wt_recognized_2024-09-30.yaml`
- `config/scoring/poomsae/taegeuk_1_jang_v0_draft.yaml`
- `config/scoring/timelines/poomsae1_zed2i_rgbd_rerun_20260802_draft.yaml`
- `config/scoring/engineering/taegeuk_1_wholebody_diagnostics_v2.yaml`
- `config/scoring/engineering/taegeuk_1_wholebody_diagnostics_v3.yaml`
- `config/scoring/accuracy/taegeuk_1_source_bound_v1.yaml`

Profil; session, reference pose ve timeline SHA-256 bağlarını taşır. Aktif
config, profil, session ve output yollarını yalnız kozmetik düzenleme amacıyla
taşımak veya yeniden biçimlendirmek reproducibility bağlarını bozabilir.

## 14. Artifact, provenance ve run lifecycle

- Her run benzersiz `outputs/<session_id>/runs/<run_id>/` dizisine yazılır.
- Var olan run'ın üzerine yazılmaz; exclusive write uygulanır.
- Run lifecycle: `preparing -> running -> completed` veya `failed`.
- Failed/incomplete run `latest_run.json` işaretçisini değiştiremez.
- Birleşik Poomsae akışında ara multiview başarı tek başına latest promotion
  yapmaz; üst düzey workflow tamamlanmalıdır.
- Config ve calibration snapshot'ları, model/input hash'leri, kod ve environment
  kimliği `run_manifest.json` ile korunur.
- Ana 3B ve kalite artifact'leri session/run/calibration/manifest bağı taşır.
- Ham 2B, ham triangulation ve önce/sonra provenance kullanıcı açıkça istemeden
  temizlenmez.
- `outputs/`, kayıtlar, checkpoint'ler ve yerel ortam Git'e eklenmez.

## 15. Önemli fail-closed davranışlar

- Üretim calibration eksik, approximate veya kamera setiyle uyumsuzsa normal
  aktif inference durur.
- Yeni/bozuk schema, non-finite serialized değer, eksik alan veya manifest/hash
  uyuşmazlığı kabul edilmez.
- Depth fusion son kalite kapısını geçmezse saf RGB sonucu korunur.
- Global optimizer kaliteyi kötüleştirirse ham/güvenli triangulation'a döner.
- Cross-view image evidence yoksa projected prior triangulation kanıtı olmaz.
- Quality gate başarısız run `--allow-low-quality-output` ile diagnostic olarak
  korunabilir, fakat latest olarak ilerletilmez.
- WholeBody/readiness bağı eksikse numeric scoring etkinleşmez.
- Dış doğruluk yoksa iç kalite resmî scoring readiness'e yükseltilmez.

## 16. Workflow ve veri sınıfları

| Sınıf | Repository'deki anlamı | Örnek |
| --- | --- | --- |
| `CURRENT_ACTIVE` | Ürün geliştirme ve birincil regresyon hattı | iki-ZED RGBD + `poomsae1_trimmed` |
| `CURRENT_VALIDATION` | Korunmuş ikincil uyumluluk/geometri smoke'u | AIST/AIST++ multiview |
| `HISTORICAL_BENCHMARK` | Sonucu yalnız kendi koşuluna bağlı eski benchmark | MADS Kata F2 RGB-only + mocap |
| `LEGACY` | Kanonik olmayan eski/sentetik uyumluluk yolu | `run_multiview_3d.py --dry-run` |

`CURRENT_VALIDATION`, `CURRENT_ACTIVE` yerine davranış dondurma kanıtı değildir.
`HISTORICAL_BENCHMARK` sonucu güncel modele taşınmaz. `LEGACY` bileşenler dış
kullanım/deprecation denetimi olmadan silinmez.

## 17. Gelecek refactor'ların koruması gereken invariant'lar

1. Ana çıktı `[t, 133, 3]`, metre ve `x=sağ, y=ileri, z=yukarı` kalır.
2. BODY-17 işlem 133 noktayı küçültmez; yüz, ayak ve eller exportta korunur.
3. Frame/timestamp kimliği ve kaynak video süresi korunur; stride yalnız
   inference örneklemesidir.
4. Hedef kamera cross-view öncül triangulation'ından çıkarılır; projection
   görüntü kanıtı olarak yeniden kullanılamaz.
5. Ham triangulation ve ham 2B provenance korunur.
6. Depth fusion yardımcı ve BODY-17 kapsamlıdır; RGB referans dalı ile kalite
   kapısına tabidir.
7. Global optimizer ham sonucu silmez; kötüleşmede fallback zorunludur.
8. Yaklaşık/sentetik calibration üretim calibration'ı gibi işaretlenmez.
9. Missing JSON=`null`, CSV=boş; serialized `NaN`/`inf` yasaktır.
10. Run dizinleri immutable/benzersizdir; yalnız tamamlanan uygun run latest
    olabilir.
11. Artifact-manifest-calibration/config bağları fail-closed doğrulanır.
12. İç kalite, ground-truth veya resmî skor doğruluğu olarak sunulmaz.
13. WholeBody diagnostic adayları otomatik kesinti değildir; score effect
    ancak doğrulanmış kaynak ve readiness izin verirse oluşabilir.
14. Veri setine özgü eşik, offset veya koordinat varsayımı genel varsayılana
    taşınmaz.
15. Regresyon karşılaştırmaları aynı video bölümü, kamera seti, stride, model ve
    config ile yapılır; smoke ile tam koşu eşdeğer sayılmaz.

## 18. Odaklı belgeler

- Güncel durum ve doğrulanmış ölçümler:
  [`../PROJECT_STATUS.md`](../PROJECT_STATUS.md)
- Reproducibility ve ortam katmanları:
  [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md)
- Mimari kararlar: [`ARCHITECTURE_DECISIONS.md`](ARCHITECTURE_DECISIONS.md)
- Mühendislik iş akışı: [`ENGINEERING_WORKFLOW.md`](ENGINEERING_WORKFLOW.md)
- Dataset/workflow sınıfları: [`DATASET_NOTES.md`](DATASET_NOTES.md)
- Legacy/compatibility sınırı: [`LEGACY_COMPONENTS.md`](LEGACY_COMPONENTS.md)
- Poomsae kural ve hareket modeli:
  [`POOMSAE1_HAREKET_VE_KURAL_SISTEMI.md`](POOMSAE1_HAREKET_VE_KURAL_SISTEMI.md)
- Otomatik segmentasyon: [`AUTOMATIC_SEGMENTATION.md`](AUTOMATIC_SEGMENTATION.md)
- Kategorik teşhis: [`CATEGORICAL_DIAGNOSTICS.md`](CATEGORICAL_DIAGNOSTICS.md)
- Teknik uygunluk: [`TECHNICAL_CONFORMANCE.md`](TECHNICAL_CONFORMANCE.md)
- Presentation teşhisleri: [`PRESENTATION_DIAGNOSTICS.md`](PRESENTATION_DIAGNOSTICS.md)
- Run geçmişi sözleşmesi: [`RUN_HISTORY.md`](RUN_HISTORY.md)
- Hata taksonomisi: [`TAEGEUK1_ERROR_TAXONOMY.md`](TAEGEUK1_ERROR_TAXONOMY.md)
- Kaynak kaydı: [`SCORING_SOURCE_REGISTER.md`](SCORING_SOURCE_REGISTER.md)
- Korunmuş Final-Polish öncesi README:
  [`history/README_PRE_FINAL_POLISH.md`](history/README_PRE_FINAL_POLISH.md)
