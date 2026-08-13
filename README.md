# TK3D

TK3D'nin asıl amacı, tekvando poomsae videolarını teknik olarak analiz edip puanlayabilen bir 3D poomsae scoring sistemi geliştirmektir.

Bu repository şu anda nihai puanlama motoruna giden ara katmanı kurar: çok kameralı poomsae videolarından kalibrasyonlu 3D insan pozu/iskeleti üretmek, bu çıktıyı kalite kontrolünden geçirmek, hareket segmentlerine hazırlamak ve puanlama algoritmasının kullanacağı veri sözleşmesini oluşturmak.

Yeni bir AI/kodlama oturumu için önce [AGENTS.md](AGENTS.md) ve
[PROJECT_STATUS.md](PROJECT_STATUS.md) okunmalıdır. Uygulama/test sırası
[mühendislik iş akışında](docs/ENGINEERING_WORKFLOW.md), korunması gereken
kararlar [mimari karar kaydında](docs/ARCHITECTURE_DECISIONS.md), veri setine
özel ayrıntılar ise [veri seti notlarında](docs/DATASET_NOTES.md) tutulur.

## AI Aracı İçin Hızlı Bağlam

Bu projeyi okuyan bir AI aracı şunu varsaymalıdır:

- Nihai ürün, poomsae performansını otomatik veya yarı otomatik puanlayan bir analiz sistemidir.
- 3D iskelet üretimi projenin asıl amacı değil, puanlama için gerekli ara çıktıdır.
- Ana ara veri sözleşmesi `keypoints_3d_world[t, 133, 3]` formatındaki COCO-WholeBody tabanlı 3D dünya koordinatlarıdır.
- Çalışan zincir: video -> 2D pose -> sağlamlaştırılmış multi-view 3D pose -> kalite analizi -> biomekanik özellikler -> hareket/faz kanıtı -> kaynak-bağlı Accuracy kararları.
- Geçici skor resmî hakem puanı değildir. Sıradaki iş, gerçek poomsae
  kayıtlarında hareket/faz etiketlerini kaynak bağlı olarak tamamlamak ve açık
  mühendislik toleranslarını ayrı sürümde tanımlamaktır; hakem/uzman etiketi
  provisional kural motorunun önkoşulu değildir.
- AIST Dance/AIST++ verisi gerçek poomsae videosu gelmeden kamera, triangulation, ViTPose inference, SMPL mesh ve scoring-readiness akışını test etmek için kullanılıyor.
- MADS Karate/Tai-chi verisi, kalibre üç RGB kamera ve motion-capture ground truth kullanan tarihsel RGB-only dış benchmark'tır; F2 dizisi model uyarlamasından tamamen ayrı testtir ve sonuçları başka koşulara devredilmez.
- Üretim 2B hattı RF-DETR kişi tespiti, ByteTrack kimlik takibi, adaptif kişi-kutusu stabilizasyonu, ViTPose flip-test ve zamansal eklem filtresi kullanır.
- ZED SVO/SVO2 oturumlarında NEURAL stereo depth ve confidence, RGB çok-kamera triangulation'ı ana kanıt olarak koruyan güven kapılı yardımcı BODY-17 ölçümüdür; aynı koşudaki saf RGB referans dalı kötüleşirse depth adayı otomatik reddedilir.
- 27 Temmuz 2026 tarihli 300 örnekli MADS F2 RGB-only kör testinde iç geometri geçti, global MPJPE `90,409 mm` ile `50 mm` hedefi geçmedi. Bu tarihsel sonuç güncel ZED RGBD koşusunun doğruluğu değildir ve ona uygulanmaz.
- Kendi poomsae videolarında ortak dünya kalibrasyonu ve poomsae hareket/faz
  etiketleri gerekir. Çok kişili çekimlerde ayrıca kimlik eşleme gerekir;
  hakem/koç puanları yalnız gelecekte hakem kalibrasyonu istenirse opsiyoneldir.

Ana ara hedef veri:

```python
keypoints_3d_world[t, 133, 3]
```

Bu ilk sürüm, nihai puanlama sistemine temel olacak şu bileşenleri içerir:

- Checkerboard tabanlı kamera kalibrasyonu için giriş noktası
- ViTPose-Huge 2D wholebody tahmin sınıfı için entegrasyon arayüzü
- RF-DETR + ByteTrack ile kamera başına kalıcı sporcu kimliği ve adaptif bounding-box stabilizasyonu
- ViTPose flip-test, güven ağırlıklı zamansal filtre ve dönüşlerde anatomik sağ/sol kimlik koruması
- Kalibrasyonlu multi-view triangulation
- ZED stereo depth/confidence ile yüzey-eklem ofsetli yardımcı BODY-17 fusion ve saf RGB referansa güvenli fallback
- Kamera FPS'i ile saniye/frame offsetlerini dikkate alan ortak zaman çizelgesi senkronizasyonu
- Görüş aykırılıklarını eleyen sağlam triangulation, pozitif derinlik/açı kontrolleri ve robust reprojection optimizasyonu
- Sentetik 3 kamera dry-run verisi ile triangulation doğrulama
- 3D temporal smoothing
- 3D validation, kalite ölçümleri ve scoring-readiness analizi
- Kaynak, ölçülebilirlik ve belirsizlik kapılı RulePack Accuracy kararları
- JSON, CSV, Excel ve figür export iskeleti
- Pytest tabanlı çekirdek algoritma testleri
- Gelecekteki 3D poomsae scoring motoruna uygun veri yapıları

## Taegeuk 1 Accuracy puanlama altyapısı

### Tek komutla trimmed ZED kaydını puanlama

Doğrulanmış ultra RGBD 3B çıktısını bütün kaynak-bağlı aşamalardan geçirmek,
benzersiz bir run oluşturmak ve iki kameralı inceleme ekranını üretmek için:

```powershell
cd C:\Users\WWWW\Desktop\tk3d
.\.venv312\Scripts\python.exe scripts\run_poomsae_scoring.py --profile poomsae1_trimmed
```

Bu hızlı mod ViTPose/depth işlemesini tekrarlamaz. Timeline'a SHA-256 ile bağlı,
daha önce kalite kapılarından geçmiş ultra RGBD 3B pozu kullanır ve doğrudan
WholeBody-133 ölçüm, hareket/faz kanıtı, RulePack kontrolü, Accuracy kararları
ve HTML inceleme raporunu üretir.

Trimmed videoyu ViTPose-Huge-WholeBody, iki kamera ve ZED NEURAL depth ile
baştan `stride 1` işleyip ardından puanlamak için yalnız son komuta
`--process-video` eklenir:

```powershell
cd C:\Users\WWWW\Desktop\tk3d
.\.venv312\Scripts\python.exe scripts\run_poomsae_scoring.py --profile poomsae1_trimmed --process-video
```

İkinci mod, elle doğrulanmış timeline'ı yalnız session kimliği, pose hash'i,
741 kare, frame indeksleri, 60 FPS ve bütün timestamp'ler referans run ile
birebir uyuşursa yeni pose dosyasına bağlar; herhangi bir farkta puanlama
fail-closed durur. Yaklaşık kalibrasyon, düşük kaliteyi zorla kabul etme,
`stride > 1` veya eski kaynaksız provisional motor bu profil tarafından
kullanılmaz.

Her iki modda da benzersiz çıktı klasörü otomatik oluşturulur ve tam yolu
terminalin sonunda `Çıktı klasörü:` satırında gösterilir:

```text
C:\Users\WWWW\Desktop\tk3d\outputs\poomsae_1_zed2i_20260731_trimmed\runs\<run_id>\
├── json\
│   ├── poomsae_scoring_summary.json
│   ├── source_bound_accuracy_decisions.json
│   ├── decision_evidence_events.json
│   ├── rule_scoring_readiness.json
│   ├── wholebody_diagnostics_report.json
│   └── movement_evidence_report.json
├── csv\
│   ├── wholebody_metrics.csv
│   └── movement_evidence.csv
├── config\
│   ├── workflow_profile.yaml
│   ├── rule_pack.yaml
│   ├── poomsae_spec.yaml
│   ├── movement_timeline.yaml
│   ├── wholebody_diagnostic_profile.yaml
│   └── accuracy_profile.yaml
├── videos\
│   ├── poomsae_scoring_annotated.mp4
│   └── poomsae_scoring_annotated_manifest.json
└── review\
    ├── poomsae_scoring_review.html
    └── poomsae_scoring_review_manifest.json
```

`--process-video` modu bunlara ek olarak `vitpose_session_3d.json`, depth ve
kalite raporları, kamera overlay videoları, 3B iskelet videosu, HTML 3B viewer
ve `scoring_readiness_analysis.xlsx` üretir.

En hızlı kontrol sırası:

1. `json/poomsae_scoring_summary.json`: kapsam ve sonuç özeti.
2. `videos/poomsae_scoring_annotated.mp4`: iki kamerayı yan yana gösteren;
   karar anında ilgili eklemi, ölçümü, kaynak aralığını ve kesintiyi videoya
   çizen görsel kanıt çıktısı. Her benzersiz doğrulanmış kesinti anı okumak
   için 3 saniye dondurulur; bu nedenle video kaynak kayıttan daha uzundur.
   Mavi kartlar WholeBody mühendislik teşhisidir, puan kesmez ve kesinti
   dondurması oluşturmaz.
3. `json/source_bound_accuracy_decisions.json`: kaynak ve belirsizlik bağlı
   tek tek kararlar.
4. `review/poomsae_scoring_review.html`: zaman çizelgesi değişmeyen ham
   kameraları senkron izleme, `Videoda aç` ile kanıt anına gitme ve her karar için
   `Doğru / Yanlış / Belirsiz` inceleme kaydı oluşturma ekranı. İnceleme
   kararları tarayıcıdan JSON olarak indirilebilir; motorun ürettiği kaynak
   karar dosyası değiştirilmez. Aynı ekranda M01-M06 için yumruk kapalılığı,
   bilek–ön kol hizası, baş/yüz–gövde yön farkı, teknik el yüksekliği, dirsek
   açısı ve hikite mesafesi ölçüm matrisi bulunur.
5. `json/decision_evidence_events.json`: video ve HTML'nin kullandığı değişmez
   olay sözleşmesi; kare/zaman, ölçüm, `%95` aralığı, kaynak sınırı ve çizilecek
   eklem geometrisini taşır.
6. `csv/wholebody_metrics.csv`: bütün sayısal WholeBody-133 ölçümleri.

İşaretli videodaki eklem çizgileri kayıtlı ViTPose 2B noktalarının görsel
izidir. Kesinti kararı bu çizgiden yeniden hesaplanmaz; kalibre, çok kameralı
3B ölçüm ve onun belirsizlik aralığından gelir. Böylece kullanıcı hatanın
nerede görüldüğünü denetleyebilir, fakat görselleştirme ikinci ve gizli bir
puan motoruna dönüşmez.

Bir kanıt anında her aktif kural ayrı numaralı kutuda gösterilir. Kutuda
`KURAL`, `ÖLÇÜLEN (KALİBRE 3B)`, `%95 olası aralık`, `FARK`, `SONUÇ`,
`NASIL DÜZELTİLİR` ve `KAYNAK DURUMU` satırları bulunur. Kamera üstünde yalnız
kayıtlı 2B ayak izi gösterilir; 3B `30°` sınırı perspektifli kamera görüntüsüne
çizilmez. Bunun yerine alt kutudaki açıkça “üstten 3B şema—kamera görüntüsü
değildir” etiketli diyagramda beyaz kesik duruş yönü, yeşil kabul alanı ve
kırmızı ölçülen 3B açı gösterilir. Aynı karede iki karar varsa `[1]` ve `[2]`
işaretleri ilgili kutuyla eşleşir. Tarihsel resmî geometri kaynağı, güncel WT
ekiymiş gibi gösterilmez.

Dondurmalı açıklama videosu kaynak karelerin hiçbirini atmaz ve sıralarını
değiştirmez; yalnız kesinti anchor'larından sonra 3 saniyelik tekrar kareleri
ekler. Bu yüzden ham kameralarla aynı zaman eksenine sahip değildir ve HTML'nin
senkron video grubuna eklenmez. Eklenen süre ve kaynak anchor kareleri video
manifestinde `freeze_*` alanlarıyla kaydedilir.

Mevcut geliştirme kapsamı yalnız kayıtta bulunan M01-M06'dır. Terminal bu
kapsamı `Aktif M01-M06 çalışma kapsamı: 6/6` olarak, bütün Taegeuk 1 bağlamını
ise ayrıca `6/18` olarak gösterir. M07-M18 bu aşamanın geliştirme hedefi veya
M01-M06 analizinin engeli değildir. Beklenen kaynak-bağlı sonuç dört küçük
hata, gözlenen kapsam için `0,4` provisional kesinti ve `accuracy_score=null`
değeridir. `null`, M01-M06'nın işlenmediği anlamına gelmez; 4,0 puan bütün
Poomsae Accuracy bütçesi olduğu için kısa bölümden tam Poomsae puanı
uydurulmadığını belirtir. `official_scoring_ready=false` kalır.

Yeni kural motoru üç sürümlü ve fail-closed sözleşme kullanır:

- `RulePack`: yürürlükteki WT puan bütçesi ve kesinti miktarları;
- `PoomsaeSpec`: Kukkiwon kaynaklı Taegeuk 1 hareket, teknik ve faz sırası;
- `MovementTimeline`: belirli pose dosyasına SHA-256 ile bağlı video etiketleri
  ve kaydın tam/kısmi kapsam bildirimi.

Güncel WT RulePack aktiftir. Gerçek `741` karelik kısa ZED2i kaydındaki
M01-M06, iki kameralı inceleme sonrası çoklu faz anchor'larıyla doğrulandı.
M01-M06 geliştirme kapsamı tamamdır; bütün Poomsae readiness raporunun kapalı
kalması mevcut altı hareketin ölçüm ve hata incelemesini durdurmaz:

```powershell
.\.venv312\Scripts\python.exe scripts\assess_poomsae_scoring_readiness.py `
  --rule-pack config\scoring\rules\wt_recognized_2024-09-30.yaml `
  --poomsae-spec config\scoring\poomsae\taegeuk_1_jang_v0_draft.yaml `
  --timeline config\scoring\timelines\poomsae1_zed2i_rgbd_rerun_20260802_draft.yaml
```

Kaynaklar, erişim durumu ve ücretli seçenekler
[`docs/SCORING_SOURCE_REGISTER.md`](docs/SCORING_SOURCE_REGISTER.md), uygulama
yolu [`docs/PUANLAMA_PLANI.md`](docs/PUANLAMA_PLANI.md), kaynak otoritesiyle
ayrılmış bütün hata/ölçülebilirlik matrisi
[`docs/TAEGEUK1_ERROR_TAXONOMY.md`](docs/TAEGEUK1_ERROR_TAXONOMY.md), gerçek kayıt
etiketleri ise
[`docs/POOMSAE1_KISA_KAYIT_ETIKETLERI.md`](docs/POOMSAE1_KISA_KAYIT_ETIKETLERI.md)
içindedir.

Mevcut kısa kayıt için iki kamerayı birlikte oynatan, hareket kartlarından aynı
zamana atlayan, WholeBody-133 hata adaylarını ve el/yüz ölçüm matrisini gösteren
güncel inceleme ekranı:
[`poomsae_scoring_review.html`](outputs/poomsae_1_zed2i_20260731_trimmed/runs/poomsae1-m01m06-wholebody-evidence-v8-20260813/review/poomsae_scoring_review.html).
Rapor üreticisi en az iki kamera ister; gelecekteki üçüncü kamera tekrar
edilebilir `--video-extra "Etiket=dosya"` argümanıyla eklenebilir.

v2.4 motoru fixation ölçülerini tek kareden değil ±5 karelik sağlam faz
penceresinden çıkarır. Yüz, el ve ayak geometrisi vücut ölçeğine göre fiziksel
makullük kapısından geçmeden teknik aday olamaz. Her aday ölçüm ve kriter
kimliğini, kare/zaman kanıtını, eşik farkını ve bekleyen inceleme durumunu taşır;
inceleme ekranındaki `Videoda aç` düğmesi bütün kameraları kanıt anına götürür.
Yumruk kapalılığı 21 el noktasının avuç merkezine göre geometrisinden ölçülür.
Baş/yüz yönü 68 yüz noktası ve omuz hattını kullanır; göz küresi takibi gibi
sunulmaz. Eşik dışı mühendislik adayları mavi ve `score_effect=null` olarak
işaretli videoya eklenir; kaynak-bağlı kırmızı kesintilerle karıştırılmaz.
Her ölçülemez satır gerekli kritik noktaların adını ve kanıt penceresindeki
örnek sayısını (`right_wrist 0/11` gibi) gösterir. İzole eklem boşlukları yalnız
aynı fixation penceresindeki en az üç gerçek örnekten sağlam medyanla
tamamlanabilir; bütün pencere boyunca kayıp kritik nokta simetriden veya başka
bir eklemden uydurulmaz.
Bir olayın kesintiye girmesi ayrıca hareketin izin verdiği `criterion_id`, doğru
WT `rule_id/source_ref` ve ayrı kural doğrulama kaydı gerektirir.

v2.4 ile iki ayağın birbirine göre açısını kullanan hatalı yaklaşım kaldırıldı;
arka ayak, `arka ayak bileği -> ön ayak bileği` duruş doğrultusuna göre ölçülür.
Fixation penceresinden sağlam MAD tabanlı `%95` belirsizlik hesaplanır. Tarihsel
resmî 2014 kılavuzundaki açık geometriler ayrı
`taegeuk1-source-bound-accuracy-v1` profilindedir: ap-seogi/apkubi arka ayak
`30°`, arae-makki yumruk-uyluk uzaklığı `1–2 yumruk`, momtong-an-makki dirsek
`90–120°`. Bunlar güncel WT eki diye gösterilmez. Ölçüm aralığı sınırla
çakışırsa kesinti yoktur; sayısal geometri yalnız küçük hata `-0,1` üretebilir.
`-0,3` yalnız doğrudan gözlenen kategorik yanlış hareket/duruş, kihap,
en az üç saniyelik duraklama veya bakış yönü olayıyla uygulanabilir.

Eski BODY-17 v1 denemesi teknik ayrıntıları kaçırdığı için sayısal sonuç üretme
yetkisi kaldırılarak yalnız tarihsel teşhis olarak bırakıldı. Güncel v2 motoru
133 noktanın gövde, ayak, yüz ve iki el gruplarını; hareket boyunca trajectory,
eşzaman, fixation ve kalite kanıtıyla birlikte işler. Eşikler WT/Kukkiwon
toleransı değildir; adaylar ceza değildir ve tüm skor alanları `null` kalır:

PoomsaeSpec artık M01-M18'in tamamında ölçülebilir kriter taşır. M14/M16
`ap-chagi + momtong-jireugi` bileşikleri tek final poza indirgenmez;
`kick_apex`, `rechamber`, destek ayağı pivotu, `landing` ve
`punch_execution` ayrı teşhis metrikleridir. Güncel sayısal tolerans kaynağı
olmadığı için yeni tekme metrikleri yalnız `measured_diagnostic_only` üretir.

```powershell
.\.venv312\Scripts\python.exe scripts\run_wholebody_poomsae_diagnostics.py `
  --pose outputs\poomsae_1_zed2i_20260731_trimmed\runs\poomsae1-zed2i-rgbd-gated-ultra-rerun-20260802\json\vitpose_session_3d.json `
  --poomsae-spec config\scoring\poomsae\taegeuk_1_jang_v0_draft.yaml `
  --timeline config\scoring\timelines\poomsae1_zed2i_rgbd_rerun_20260802_draft.yaml `
  --diagnostic-profile config\scoring\engineering\taegeuk_1_wholebody_diagnostics_v2.yaml `
  --output-json outputs\<session_id>\runs\<yeni_run_id>\json\wholebody_diagnostics_report.json `
  --output-csv outputs\<session_id>\runs\<yeni_run_id>\csv\wholebody_metrics.csv

.\.venv312\Scripts\python.exe scripts\build_source_bound_accuracy_decisions.py `
  --wholebody-diagnostics outputs\<session_id>\runs\<yeni_run_id>\json\wholebody_diagnostics_report.json `
  --poomsae-spec config\scoring\poomsae\taegeuk_1_jang_v0_draft.yaml `
  --timeline config\scoring\timelines\poomsae1_zed2i_rgbd_rerun_20260802_draft.yaml `
  --accuracy-profile config\scoring\accuracy\taegeuk_1_source_bound_v1.yaml `
  --output-json outputs\<session_id>\runs\<yeni_run_id>\json\source_bound_accuracy_decisions.json
```

Gerçek kısa kayıt v2.4 koşusu
`outputs/poomsae_1_zed2i_20260731_trimmed/runs/poomsae1-source-bound-20260810-221705/`
altındadır. Dokuz kaynak-bağlı kararın dördü küçük hata olarak uygulandı, üçü
ölçülemedi, biri aralık içinde, biri sınır-belirsiz kaldı. Gözlenen kapsam
kesinti toplamı `0,4`; M07-M18 videoda olmadığı için `accuracy_score=null`
kalır.

RulePack `1.1.0`, WT Article 16.3.1 süre ihlali ve 16.3.2 sınır geçme
kesintilerini Accuracy'den ayrı `final_score_deductions` alanında taşır. İkisi
de final skordan `-0,3` metadata'sıdır. Sınır geçmenin tekrar frekansı kaynakta
açık olmadığı için otomatik runtime uygulaması kapalıdır.

Kullanılan bütün yerel PDF/TXT kaynakları
[`output/pdf/scoring_sources/`](output/pdf/scoring_sources/) altında; dosya
hash'leri, çevrim içi Kukkiwon video bağlantıları ve kaynak statüleri
[`docs/scoring_sources/README.md`](docs/scoring_sources/README.md) içindedir.

Yeni bir kural/teknik PDF geldiğinde önce
[`docs/scoring_sources/SOURCE_INTAKE_TEMPLATE.yaml`](docs/scoring_sources/SOURCE_INTAKE_TEMPLATE.yaml)
kopyalanıp doldurulur ve şu komut çalıştırılır:

```powershell
.\.venv312\Scripts\python.exe scripts\validate_scoring_source_intake.py `
  --manifest <doldurulan-kaynak-manifesti.yaml> `
  --output outputs\source-intake\<benzersiz-kaynak-raporu>.json
```

Araç dosya imzası ve SHA-256 bağını denetler, kaynak otoritesini kullanım
amacıyla karşılaştırır ve hiçbir belgeyi otomatik olarak aktif kurala
dönüştürmez. Tarihsel/ikincil kaynak güncel sayısal tolerans talep ederse kapı
fail-closed kapanır.

## Sıfırdan Kurulum ve Gerekli İndirmeler

Git deposu bilinçli olarak büyük video, veri seti, model ağırlığı ve lisanslı insan modeli içermez. Aynı testleri
yeniden üretmek için kodu klonladıktan sonra aşağıdaki varlıklar resmî kaynaklarından indirilmelidir.

### Hangi bileşen gerçekten gerekli?

| Bileşen | Ne için gerekli? | Durum |
| --- | --- | --- |
| Python 3.12 ve `requirements.txt` | Sentetik dry-run ve model gerektirmeyen çekirdek işlemler | Zorunlu başlangıç |
| `requirements-pose.txt` içindeki PyTorch, RF-DETR ve Supervision ortamı | Otomatik testlerin tamamı ve gerçek inference | Tam doğrulama için zorunlu |
| NVIDIA GPU, ViTPose kaynak kodu ve WholeBody ağırlığı | Gerçek videodan 2B/3B iskelet üretimi | Gerçek inference için zorunlu |
| MADS multi-view | Bağlı RGB-only profil için kalibre üç kamera ve mocap ground-truth benchmark'ı | Tarihsel RGB hattını doğrulamak için; ZED RGBD'ye doğrudan uygulanmaz |
| MADS depth | ZED depth-fusion hattının bağımsız RGB-D benchmark uyarlaması | Mevcut ZED inference için gereksiz; dış depth doğrulaması için opsiyonel |
| AIST videoları ve AIST++ kamera verisi | Eski dans smoke/regresyon testi | Opsiyonel |
| SMPL model dosyası | Gerçekçi insan mesh'i çizmek | Opsiyonel; iskelet ve puanlama onsuz çalışır |

Test edilen ana Windows ortamı Python `3.12.13`, PyTorch `2.13.0+cu130`, CUDA 13 uyumlu NVIDIA sürücüsü ve RTX
4060 Laptop GPU'dur. Testlerin tamamı için PyTorch paketi gerekir; GPU, ViTPose kaynak ağacı, checkpoint, MADS, AIST
veya SMPL gerekmez. ViTPose-Huge gerçek inference için NVIDIA GPU kuvvetle önerilir; CPU üzerinde çalıştırmak pratik
olmayacak kadar yavaştır. MADS'in tamamı yaklaşık
24 GB, ViTPose checkpoint'i yaklaşık 3,47 GiB olduğundan en az 50 GB boş alan ayırmak güvenlidir.

### 1. Depoyu klonla ve Python ortamını kur

Windows'ta aşağıdaki araçlar kurulu olmalıdır:

- Git for Windows: https://git-scm.com/download/win
- 64-bit Python 3.12: https://www.python.org/downloads/windows/
- MADS split ZIP'leri için güncel 7-Zip: https://www.7-zip.org/
- Gerçek inference yapılacaksa güncel NVIDIA sürücüsü: https://www.nvidia.com/Download/index.aspx
- Yalnız AIST API'nin bazı görselleştirmeleri için opsiyonel FFmpeg: https://ffmpeg.org/download.html

```powershell
git clone https://github.com/EsatSimsek/tk3D.git
cd .\tk3D

py -3.12 -m venv .venv312
.\.venv312\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

PowerShell sanal ortam aktivasyonunu engellerse yalnız açık terminal için:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv312\Scripts\Activate.ps1
```

Gerçek ViTPose inference için NVIDIA/CUDA paketlerini ayrıca kur:

```powershell
python -m pip install -r requirements-pose.txt
```

`requirements-pose.txt` bu projede test edilen CUDA 13.0 PyTorch wheel'ini kullanır. Farklı CUDA/PyTorch ortamında
paket sürümlerini körlemesine değiştirmek yerine önce PyTorch'un resmî kurulum seçicisine göre uyumlu wheel kurulmalı,
ardından `requirements.txt`, `timm` ve `torchvision` tamamlanmalıdır: https://pytorch.org/get-started/locally/

RF-DETR-Small ağırlığı ilk gerçek inference sırasında `C:\Users\<kullanıcı>\.roboflow\models\` altına otomatik
indirilir. Sonraki çalıştırmalar doğrulanmış yerel ağırlığı kullanır. İnternetsiz bir makinede çalıştırmadan önce bu
ilk indirme tamamlanmış olmalıdır.

### 2. Resmî ViTPose kaynak kodunu kur

TK3D, MMPose kurulumu yerine resmî ViTPose deposundan sınırlı bir yerel runtime kullanır. Yeniden üretilebilirlik için
bu projede test edilen commit sabitlenmiştir:

```powershell
New-Item -ItemType Directory -Force external | Out-Null
git clone https://github.com/ViTAE-Transformer/ViTPose.git external\vitpose
git -C external\vitpose checkout c050ed29112da7704797cc1a65af0234b525010d
```

Resmî depo ve lisans: https://github.com/ViTAE-Transformer/ViTPose

Beklenen klasör:

```text
external/vitpose/mmpose/
external/vitpose/configs/
```

### 3. ViTPose-Huge WholeBody ağırlığını indir ve doğrula

COCO-only 17 eklemli ViTPose-H ağırlığını kullanmayın. Gerekli dosya, resmî ViTPose WholeBody tablosundaki
`ViTPose++-H COCO+AIC+MPII+AP10K+APT36K+WholeBody 256x192` checkpoint'idir.

1. Resmî bağlantıyı tarayıcıda açın:
   https://1drv.ms/u/s!AimBgYV7JjTlgccoXv8rCUgVe7oD9Q?e=ZBw6gR
2. İndirilen gerçek `.pth` dosyasını aşağıdaki ada taşıyın:

```text
weights/vitpose_huge_wholebody_256x192.pth
```

3. Boyut ve SHA-256 değerini doğrulayın:

```powershell
New-Item -ItemType Directory -Force weights | Out-Null
Get-Item weights\vitpose_huge_wholebody_256x192.pth | Select-Object Name,Length
Get-FileHash weights\vitpose_huge_wholebody_256x192.pth -Algorithm SHA256
```

Beklenen değerler:

```text
Boyut  : 3723960207 byte
SHA-256: A714AE5F0B45F7A3F1A86624CF7382913454EE1D61A4AE5F06C40573D5B6A459
```

OneDrive birkaç KB/MB boyutunda HTML dosyası kaydettiyse bu checkpoint değildir; silip tarayıcıdan yeniden indirin.
`check_models.py` HTML veya uyumsuz model dosyasını reddeder.

### 4. MADS veri setini indir, çıkar ve hazırla

MADS, bağlı olduğu RGB-only koşu/profil için tarihsel dış doğruluk benchmark'ıdır. Güncel ZED RGBD sonucuna otomatik uygulanmaz; proje veri setini yeniden dağıtmaz, resmî sayfadan indirin:

- https://visal.cs.cityu.edu.hk/downloads/
- Sayfada `Human Pose Datasets -> MADS -> download here`
- Doğrudan MADS Google Drive klasörü: https://drive.google.com/drive/folders/0B0AquUC4V8cFU2otR3l3WWRUVVk?resourcekey=0-KC-rxBAHiIIpylFRCTESNQ

Aynı ortamı tamamen yeniden üretmek için `MADS_multiview` ve `MADS_depth` arşivlerini indirin. Yalnız mevcut F2 RGB
benchmark'ını çalıştırmak için multi-view parçaları yeterlidir. Split ZIP'in tüm `.z01`, `.z02`, ... ve `.zip`
parçaları aynı klasörde olmalıdır. Güncel 7-Zip ile ana `.zip` dosyasını açıp örneğin `C:\datasets\MADS` altına
çıkarın.

7-Zip varsayılan konumdaysa komut satırından çıkarma örneği:

```powershell
New-Item -ItemType Directory -Force C:\datasets\MADS\MADS_multiview | Out-Null
& "C:\Program Files\7-Zip\7z.exe" x `
  C:\Downloads\MADS\MADS_multiview.zip `
  "-oC:\datasets\MADS\MADS_multiview"

# Depth arşivi indirildiyse:
New-Item -ItemType Directory -Force C:\datasets\MADS\MADS_depth | Out-Null
& "C:\Program Files\7-Zip\7z.exe" x `
  C:\Downloads\MADS\MADS_depth.zip `
  "-oC:\datasets\MADS\MADS_depth"
```

Arşiv yolu farklıysa yalnız `C:\Downloads\MADS` bölümünü değiştirin. Çıkarma bittikten sonra aşağıdaki kesin klasör
düzenini kontrol etmeden kurulum scriptini çalıştırmayın.

TK3D'nin beklediği kesin klasör düzeni:

```text
C:\datasets\MADS\
├── MADS_multiview\MADS\multi_view_data\
│   ├── Kata\Kata_F2_C0.avi
│   ├── Kata\Kata_F2_GT.mat
│   ├── Kata\Calib_Cam0.mat
│   └── Taichi\...
└── MADS_depth\MADS\depth_data\       # opsiyonel
```

Veriyi indeksle, resmî kamera kalibrasyonlarını içe aktar, metric ground-truth JSON'larını ve önizlemeleri üret:

```powershell
python scripts\setup_mads_test.py `
  --dataset-root C:\datasets\MADS `
  --actions Kata `
  --hash-files `
  --preview
```

Başarılı kurulumdan sonra en az şu dosyalar bulunmalıdır:

```text
data/mads_test/local/sessions/mads_kata_f2.yaml
data/mads_test/local/ground_truth/multiview/Kata_F2.json
outputs/mads_kata_f2/calibration/cameras.json
outputs/mads_setup/mads_setup_report.json
```

`data/mads_test/local/` makineye özgü mutlak yollar içerir ve Git'e eklenmez. Ayrıntılı MADS açıklaması:
`docs/mads_ground_truth_setup.md`.

### 5. AIST dans smoke testi için opsiyonel indirme

AIST, MADS'in yerine geçen doğruluk benchmark'ı değildir. Yalnız eski çok-kamera akışını tekrar test etmek isteyenler
için gerekir. Önce AIST kullanım koşullarını okuyun: https://aistdancedb.ongaaccel.jp/terms_of_use/

İki kameralı küçük örnek için klasörleri hazırla ve resmî AIST video URL'lerinden yalnız gerekli iki videoyu indir:

```powershell
python scripts\setup_aist_test.py `
  --sequence gBR_sBM_cAll_d04_mBR0_ch01 `
  --cameras c01 c02

$aistBase = "https://aistdancedb.ongaaccel.jp/video_raw/10M"
Invoke-WebRequest `
  "$aistBase/gBR_sBM_c01_d04_mBR0_ch01.mp4" `
  -OutFile data\aist_test\videos\gBR_sBM_c01_d04_mBR0_ch01.mp4
Invoke-WebRequest `
  "$aistBase/gBR_sBM_c02_d04_mBR0_ch01.mp4" `
  -OutFile data\aist_test\videos\gBR_sBM_c02_d04_mBR0_ch01.mp4
```

AIST++ kamera kalibrasyonunu resmî GitHub release'inden indir ve çıkar:

```powershell
Invoke-WebRequest `
  "https://github.com/google/aistplusplus_dataset/releases/download/v1.0/cameras.zip" `
  -OutFile data\aist_test\cameras.zip
Expand-Archive `
  -Path data\aist_test\cameras.zip `
  -DestinationPath data\aist_test\annotations `
  -Force

python scripts\import_aist_cameras.py --session data\aist_test\session.yaml
```

Beklenen kamera dosyası `data/aist_test/annotations/cameras/mapping.txt` konumundadır. AIST++ motion, 2B/3B
annotation veya API kodu yalnız bu ek özellikler kullanılacaksa gerekir. Resmî annotation indirmeleri:
https://google.github.io/aistplusplus_dataset/download.html

Opsiyonel API kodunu bu projede test edilen commit ile kurmak için:

```powershell
git clone https://github.com/google/aistplusplus_api.git external\aistplusplus_api
git -C external\aistplusplus_api checkout 2dd7b3e946b794fd0081c98e2e2433545abf8b87
```

### 6. SMPL insan modeli yalnız mesh için opsiyoneldir

SMPL dosyası 3B iskelet, MADS ground-truth ölçümü veya puanlama altyapısı için gerekli değildir. Yalnız gerçekçi insan
yüzeyi/mesh render etmek isteyen kullanıcı kurmalıdır. Standart SMPL modeli lisans nedeniyle bu repo tarafından
indirilemez veya üçüncü kişilere dağıtılamaz.

1. https://smpl.is.tue.mpg.de/ adresinde hesap açın.
2. Lisansı okuyup kendi hesabınızla modeli indirin.
3. İndirdiğiniz dosyaları şu adlarla yerleştirin:

```text
models/smpl/SMPL_MALE.pkl
models/smpl/SMPL_FEMALE.pkl   # opsiyonel
```

4. Mesh paketlerini kurun:

```powershell
python -m pip install -r requirements-smpl.txt
```

Open3D kurulamazsa tarayıcı tabanlı Three.js export için yalnız temel mesh paketleri yeterlidir:

```powershell
python -m pip install smplx trimesh chumpy --no-build-isolation
```

Lisans ve mesh komutları: `docs/smpl_mesh_setup.md`.

### 7. Kurulumu doğrula

Önce hiçbir model veya video gerektirmeyen core testleri ve sentetik üç kamera dry-run'ını çalıştırın:

```powershell
python -m pytest -q
python scripts\run_multiview_3d.py `
  --session data\session_001\session.yaml `
  --dry-run
```

Codex/sandbox ortamında pytest temp izni sorun çıkarırsa:

```powershell
python -m pytest -q -p no:cacheprovider --basetemp outputs\pytest-tmp
```

Bu bölüm yazıldığında doğrulanan test sonucu:

```text
204 passed in 18.25s
```

`--dry-run`, sentetik dünya koordinatlarını üç kameraya projekte edip gerçek triangulation kodundan geçirir. Çıktısı:

```text
outputs/session_001/runs/<run_id>/videos/skeleton_3d_world.mp4
```

ViTPose ve MADS kurulduktan sonra gerçek varlık kontrolü:

```powershell
python scripts\check_models.py `
  --session data\mads_test\local\sessions\mads_kata_f2.yaml `
  --model-config config\model_config.yaml
```

Beklenen satır:

```text
pose2d: ready - ready
```

Tam 300 örnekli MADS benchmark ve geçici puanlama komutları aşağıdaki `Puanlama Altyapısı: Güvenli Geliştirme
Akışı` bölümündedir. Gerçek kullanıcı videosunda önce strict preflight çalıştırılmalıdır:

```powershell
python scripts\preflight_session.py `
  --session data\session_001\session.yaml `
  --require-videos `
  --require-calibration-videos `
  --require-model-files
```

### 8. Git'e konmayan yerel varlıklar

Aşağıdaki dosyalar `.gitignore` ile yerelde tutulur; `git push` bunları GitHub'a yüklemez veya bilgisayardan silmez:

```text
weights/*.pth
external/vitpose/
external/aistplusplus_api/
data/mads_test/local/
data/aist_test/videos/*.mp4
data/aist_test/annotations/
models/smpl/*.pkl
outputs/**
```

## Ground-truth 3B Doğrulama

Poomsae'ye yakın hareket alanı ve optik motion-capture referansı nedeniyle birincil dış doğrulama veri seti olarak
MADS (Martial Arts, Dancing and Sports) seçildi. AIST mevcut çok-kamera smoke testleri için korunur; gerçek 3B hata
ölçümünün ana benchmark'ı değildir.

Ground-truth karşılaştırma katmanı global/pelvis-relative/PA-MPJPE, PCK-3D, açı, hız, ivme ve kemik kararlılığı
raporlarını üretir. Girdi koordinatı ve birimi açıkça doğrulanmadan çalışmaz. Kurulum, veri sözleşmesi ve resmî indirme
durumu: `docs/mads_ground_truth_setup.md`.

Yerel MADS arşivini (çoklu-görüş ve depth) indeksleyip Kata oturumlarını hazırlamak:

```powershell
python scripts\setup_mads_test.py --dataset-root C:\Users\WWWW\Desktop\MADS --actions Kata --hash-files --preview
```

Kurulum 30 çoklu-görüş ve 30 depth diziyi indeksler; seçilen diziler için resmî kalibrasyon, yerel session ve metre
cinsinden 3B ground-truth üretir. Makineye özgü yolları içeren `data/mads_test/local/` Git'e eklenmez.

```powershell
python scripts\evaluate_ground_truth_3d.py --prediction <tk3d_3d.json> --ground-truth <metric_gt.json> --output-dir outputs\ground_truth_validation
```

Komut, ground-truth kalite kapısı başarısızsa CI/otomasyonun bunu başarı sanmaması için sıfırdan farklı kodla çıkar.
Yalnız tanısal başarısız raporu bilinçli biçimde kabul etmek için `--allow-failed-quality-gate` kullanılabilir.

MADS domain-adaptation altyapısı da eklidir. Donmuş ViTPose omurga özelliklerinden 2B heatmap head eğitimi ve robust
eklem offset kalibrasyonu yapılabilir; ancak üretilen adapter varsayılan olarak onaysızdır. Normal çalışma, ayrı 3B
testte onaylanmamış adapter'ı reddeder. Mevcut deneylerde 2B doğrulama kaybı iyileşmesine rağmen hiç görülmemiş F2
3B testi kötüleştiği için MADS adapter üretime alınmadı; kullanılan model hâlâ daha iyi sonuç veren temel
ViTPose-Huge modelidir. Ayrıntılar ve sayısal sonuçlar: `docs/mads_ground_truth_setup.md`.

## AIST Video Testi

AIST Dance Video DB videoları, poomsae videosu gelmeden çok kameralı görüntü akışını test etmek için kullanılabilir. AIST++ annotation dosyaları COCO 17 eklem formatındadır; bu nihai COCO-WholeBody 133 hedefini değiştirmez. Videolar bizim ViTPose-Huge WholeBody adapter yoluna girdiğinde hedef yine 133 eklemdir. AIST++ 17 eklem verisi sadece calibration, projection, triangulation ve hata ölçümü için opsiyonel doğrulama verisidir.

Yerel test klasörlerini hazırlamak:

```powershell
python scripts\setup_aist_test.py --sequence gBR_sBM_cAll_d04_mBR0_ch01 --cameras c01 c02
```

Bu komut şunları hazırlar:

- `data/aist_test/videos/`
- `data/aist_test/annotations/`
- `data/aist_test/session.yaml`
- `data/aist_test/aist_test_manifest.json`

Beklenen ilk video dosyaları:

```text
data/aist_test/videos/gBR_sBM_c01_d04_mBR0_ch01.mp4
data/aist_test/videos/gBR_sBM_c02_d04_mBR0_ch01.mp4
```

AIST++ API bu projede `external/aistplusplus_api` altına kurulur ve Git'e eklenmez. Büyük video ve annotation dosyaları da Git'e eklenmez.

## Kalibrasyon

### ZED SVO2 + ZED Fusion kalibrasyonu

ZED 2i kayıtları için `scripts\prepare_zed_multiview_session.py`, iki veya daha
fazla SVO/SVO2 dosyasını donanım zaman damgalarıyla ortak zaman çizelgesine
eşler, sol rektifiye görüntüyü çıkarır ve ZED Fusion/ZED360 kalibrasyonunu TK3D
projeksiyon sözleşmesine dönüştürür. En yüksek inference kalitesi için varsayılan
`FFV1` çıktısı kayıpsızdır. Komut ZED SDK ile aynı sürümdeki `pyzed` Python
modülünü gerektirir.

```powershell
.\.venv312\Scripts\python.exe scripts\prepare_zed_multiview_session.py `
  --session-id poomsae_take_001 `
  --task-name poomsae_1 `
  --fusion-config C:\take01\cal.json `
  --svo C:\take01\cam01.svo2 C:\take01\cam02.svo2

.\.venv312\Scripts\python.exe scripts\run_vitpose_multiview_3d.py `
  --session outputs\poomsae_take_001\source\session.yaml `
  --stride 1 `
  --run-id poomsae-take-001-ultra
```

Üçüncü kamera için aynı hazırlama komutunun `--svo` listesine üçüncü dosya
eklenir; Fusion JSON dosyasında bu kameranın aynı ortak dünya pozunun bulunması
zorunludur. Hazırlayıcı var olan oturumun üzerine yazmaz. SVO zaman boşlukları,
yeniden kullanılan/atlanmış kaynak kareler, SHA-256 kimlikleri ve senkron artık
hataları `outputs/<session_id>/source/reports/` altında korunur.

ZED Fusion pozları `x=sağ, y=ileri, z=yukarı` ve metre olarak alınır; kamera
ekseni ayrıca OpenCV'nin `x=sağ, y=aşağı, z=ileri` optik projeksiyon eksenine
dönüştürülür. Bu optik-baz dönüşümü olmadan düşük/yanıltıcı 3B geçerlilik
oluşabileceği için `zed_fusion_multiview` kalibrasyonu yalnız hazırlayıcı
tarafından üretilen şemayla kabul edilir.

```powershell
python scripts\calibrate_cameras.py --session data\session_001\session.yaml
```

Kalibrasyon scripti senkron checkerboard tespitleriyle tüm kameraları ortak bir referans kamera koordinat sistemine bağlayan `multiview_common_reference` modunu üretir. Bu mod için:

- Her kamerada checkerboard aynı fiziksel anda görünmelidir.
- `session.yaml` içindeki `sync.offsets` değerleri kalibrasyon frame eşlemesine uygulanır.
- `config/calibration_config.yaml` içinde opsiyonel `checkerboard.min_common_frames` ve referans kamera için `extrinsics.world_origin_camera` veya `checkerboard.reference_camera_id` kullanılabilir.

Ortak kare bulunamazsa komut güvenli biçimde durur ve üretim `cameras.json` dosyası yazmaz. Yalnızca intrinsic teşhisi gerekiyorsa açık `--allow-intrinsics-only-fallback` seçeneği `intrinsics_only.json` üretir; canlı 3B akış bu dosyayı kabul etmez.

Kalibrasyon çıktıları:

- `outputs/session_001/calibration/cameras.json`
- `outputs/session_001/calibration/calibration_report.json`

## Çok Kameralı 3D Pipeline

```powershell
python scripts\run_multiview_3d.py --session data\session_001\session.yaml --dry-run
python scripts\run_vitpose_multiview_3d.py --session data\session_001\session.yaml
```

ViTPose multi-view pipeline:

- `session.yaml` içindeki kamera `frame_offset` değerlerini global frame zaman çizelgesine uygular.
- RF-DETR ile kişileri bulur; ByteTrack ile kamera başına aynı sporcunun kimliğini korur ve kısa tespit kayıplarında son güvenilir kutuyu kullanır.
- Adaptif kutu filtresi sabit duruşta dedektör titreşimini bastırır, büyük gerçek harekette kutuyu gecikmesiz izler.
- ViTPose-Huge WholeBody üzerinde flip-test çalıştırır; güven ağırlıklı zamansal filtre küçük eklem sıçramalarını azaltır.
- Dönüşlerde tek karelik anatomik sağ/sol değişimlerini önceki poz ve hareket tahminiyle düzeltir.
- Stride kullanılan tanısal koşularda örnek pozları gerçek kaynak karelerine enterpole eder; çıktı videosu hızlandırılmış görünmez.
- Üretim kalibrasyonu yoksa veya ortak dünya extrinsic bilgisi doğrulanmamışsa güvenli biçimde durur. Yaklaşık iki-kamera kalibrasyonu yalnızca açık `--allow-approximate-calibration` seçeneğiyle diagnostik preview için kullanılabilir.
- Gerçek üretim çıktısı için `outputs/<session_id>/calibration/cameras.json` dosyasının ilgili kamera ID'leriyle uyumlu olması gerekir.
- Her canlı çalışma `outputs/<session_id>/runs/<run_id>/` altında izole edilir; yalnızca kalite kapısını geçen çalışma `latest_run.json` olarak işaretlenir.
- Canlı çalışmanın kalite raporu iç geometri ve sensör tutarlılığını ölçer. Bunlar geçerse
  `provisional_scoring_ready=true` olur; bu yalnız verinin kaynak-bağlı kural analizine girebileceğini belirtir ve
  kendi başına puan üretmez. `official_scoring_ready` ayrı kalır ve uzman kural/hakem doğrulaması olmadan açılmaz. Opsiyonel dış
  doğrulama komutu tahmin, ground truth ve kalite raporlarını SHA-256 ile aynı koşuya bağlar; başka koşunun metriği
  veya JSON içine elle yazılmış bir alan yetki kaynağı sayılmaz.

Sentetik dry-run çıktıları `outputs/session_001/runs/<run_id>/` altında, canlı ViTPose çıktıları da aynı izole çalışma yapısında tutulur. Beklenen dry-run dosyaları:

- `outputs/session_001/runs/<run_id>/json/session_3d.json`
- `outputs/session_001/runs/<run_id>/json/preflight_report.json`
- `outputs/session_001/runs/<run_id>/json/video_probe_report.json`
- `outputs/session_001/runs/<run_id>/json/model_runtime_report.json`
- `outputs/session_001/runs/<run_id>/json/quality_summary.json`
- `outputs/session_001/runs/<run_id>/json/artifact_manifest.json`
- `outputs/session_001/runs/<run_id>/csv/keypoints_3d_world_flat.csv`
- `outputs/session_001/runs/<run_id>/session_3d_analysis.xlsx`
- `outputs/session_001/runs/<run_id>/figures/reprojection_error_timeline.png`
- `outputs/session_001/runs/<run_id>/figures/keypoint_validity_heatmap.png`
- `outputs/session_001/runs/<run_id>/figures/camera_usage_heatmap.png`

## Veri Mimarisi

```text
Session
-> CameraView
-> Frame
-> PersonPose2D
-> PersonPose3D
-> TriangulatedPose3D
-> Phase
-> Step
-> Validation
-> Scoring
```

Nihai scoring hiyerarşisi `Episode -> Task -> Phase -> Step -> Metric -> Error -> Score` şeklinde düşünülür.

Kod, 3B poz ve kalite kapılarının üzerine açıklanabilir bir teknik ön skor üretir. Bu skor yalnızca altyapı/doğrulama içindir; onaylı poomsae adım şablonu ve hakem kriteri olmadan resmi puan olarak kullanılamaz.

## Güncel Durum

Hazır olanlar:

- Proje iskeleti ve Git ignore kuralları
- `keypoints_3d_world[t, 133, 3]` veri sözleşmesi
- Checkerboard tabanlı multiview ortak referans kalibrasyonu ve üretimde fail-closed davranış
- Farklı FPS, frame offset ve saniye offsetlerini destekleyen timestamp senkronizasyonu
- ViTPose-Huge WholeBody 2D runtime; opsiyonel RTMW3D yardımcı adapter'ı varsayılan olarak kapalı
- Batch/multi-camera 2D inference arayüzü
- RF-DETR-Small kişi dedektörü ve ByteTrack tabanlı kalıcı sporcu kimliği; kısa kayıpta yeniden edinme desteği
- Sabit duruş için adaptif bounding-box stabilizasyonu, ViTPose flip-test ve güven ağırlıklı zamansal eklem filtresi
- Arka/yan dönüşlerde anatomik sağ-sol eklem kimliğini koruyan sıçrama düzeltmesi
- BODY-17 odaklı 2B görselleştirme ve stride çıktılarında gerçek kare zaman çizelgesi enterpolasyonu
- Aykırı kamerayı eleyen, pozitif derinlik ve triangulation açısı kontrolü yapan robust multi-view triangulation
- Hedef kamerayı üçgenlemeden çıkaran, ViTPose ısı haritasını bağımsız çok-kamera öncülüyle ikinci kez arayan 2B geri besleme
- Görüntü kanıtı bulunmayan kamera-eklem aykırılarını 3B geometriden çıkarıp yalnız videoda turuncu, izlenebilir çok-kamera kurtarması olarak gösterme
- Robust reprojection optimizasyonu; reprojection error ve inlier kamera sayısına dayalı kalite skoru
- Sentetik 3 kamera dry-run pipeline
- JSON/CSV/Excel/PNG/MP4 output üretimi; CSV export eksik sayıları `NaN` stringi yerine boş hücre yazar
- Preflight raporu: eksik/açılamayan video, eksik kalibrasyon videosu, eksik model config/checkpoint kontrolü
- Video probe raporu: her kamera videosu için açılabilirlik, FPS, çözünürlük, frame count, duration
- Model runtime raporu: ViTPose-Huge WholeBody config/checkpoint hazır mı kontrolü
- AIST++ camera data importer: mapping.txt + setting_*.json dosyalarından gerçek 9 kamera intrinsic/extrinsic üretimi
- ViTPose-Huge gerçek inference ile AIST videolarından 133 eklemli 2D overlay ve kalibrasyonlu multi-view 3D çıktı
- MADS multi-view/depth indeksleme, resmî üç kamera kalibrasyonu ve metre cinsinden mocap ground-truth dönüşümü
- F2'yi eğitimden ayıran domain-adaptation altyapısı ve onaysız adapter'ı üretimde reddeden güvenlik kilidi
- Tarihsel MADS F2 RGB-only 300 örnekli kör test: `90,409 mm` MPJPE, `162,504 mm` P95, `13,426°` açı MAE ve `%95,89` geçerli eklem oranı; bu değerler yalnız o bağlı koşuyu tanımlar ve güncel ZED RGBD koşusuna devredilmez
- İç kalite geçtiğinde ground-truth beklemeden `provisional_scoring_ready=true` üreten; dış benchmark varsa onu
  yalnız aynı koşuya bağlayan ve başka koşunun metriğini otomatik reddeden puanlama yetkilendirmesi
- Artifact manifest: her run için beklenen çıktılar, dosya boyutları ve SHA-256 özetleri
- Quality summary: valid frame/joint oranı, triangulation score, reprojection error, kullanılan kamera sayısı
- Yönlü torso lean, smoothing sonrası hız, ağırlıklı center-of-mass proxy, adaptif hareket segmentasyonu
- Kalite kapılı geçici frame/step skoru ve puan kırılma nedenlerini listeleyen teknik hata raporu
- Triangulation, smoothing, validation, scoring-readiness ve pipeline testleri

Bekleyenler / sıradaki büyük işler:

- Kendi poomsae kameraları için senkron checkerboard calibration videoları ile ortak referans intrinsic/extrinsic üretimi
- Aynı görüntüde birden fazla aktif sporcu varsa forma/kimlik tabanlı uzun süreli kamera-arası eşleme
- Etiketli gerçek Taekwondo BODY-17 verisiyle yüksek çözünürlüklü pose modelinin yeniden eğitilmesi
- Poomsae phase/step detection: hareketleri poomsae adımlarına bölme
- Gerçek poomsae için phase/step adlarını ve başlangıç-bitiş karelerini onaylayacak etiket veri seti
- Denge, açı, hizalama, yükseklik, zamanlama ve simetri hedeflerinin hakem/koç tarafından onaylanması
- Geçici teknik ön skoru resmi kurallara bağlayan sürümlü referans şablonları ve uzman doğrulaması

## ViTPose Gerçek Video Testi

ViTPose gerçek video inference `.venv312` ortamında çalışır. Ayrıntılı Windows sürüm notu: `docs/vitpose_windows_setup.md`.

En güçlü tam test için proje klasöründe yalnız şu komutu çalıştır:

```powershell
.\scripts\run_full_aist_pose.ps1
```

Bu komut dokuz kamerayı, bütün geçerli ortak zaman çizelgesi karelerini (`stride 1`) ve ViTPose-Huge WholeBody
modelinin 133 noktasını kullanır.
Her çalıştırmada yerel tarih ve saat içeren yeni bir klasör oluşturur; eski sonuçların üzerine yazmaz:

```text
outputs/aist_test/runs/pose_YYYY-MM-DD_HH-mm-ss_fff/
```

Tam hat; RF-DETR kişi tespiti ve kimlik takibi, gerçek video FPS değerleriyle kamera senkronizasyonu, ileri-geri
çalışan sıfır-fazlı 2B yörünge stabilizasyonu, güven ve reprojection hatasına dayanıklı çok-kameralı 3B
triangulation uygular. İlk üçgenlemeden sonra her kamera ayrı ayrı dışarıda bırakılır; diğer kameralardan gelen
bağımsız 3B sonuç hedef görüntüye izdüşürülür. Yalnız büyük hata gösteren eklemler için ViTPose ısı haritası ikinci
kez aranır. Yeni nokta hem gerçek bir ısı-haritası tepesine sahip olmalı hem de bağımsız geometri hatasını belirlenen
sınırda azaltmalıdır. Bu iki koşulu geçemeyen ölçüm 3B hesaptan çıkarılır; görüntü videosunda ise turuncu renkle
`cross-view recovered` olarak gösterilir. Böylece izdüşürülen nokta yeni kamera ölçümü gibi tekrar üçgenlemeye
sokulmaz ve veri döngüsü oluşturmaz. Ardından BODY-17, bütün senkronize video boyunca tek bir global optimizasyonda çözülür:
dokuz kameranın yeniden izdüşüm hatası, kişiye özel sabit kemik uzunlukları, güvenli dirsek/diz limitleri,
harekete uyarlanan ivme ve jerk sürekliliği, kamera güvenleri ve kısa kapanmalar aynı çözümde kullanılır.
Gövdeyle birlikte yüz, ayak ve el/parmak noktaları veri dosyalarında korunur; el ve ayak bağlantıları 2B/3B
görselleştirmelerde çizilir. Stabilizasyon kare atarak veya eklem sayısını azaltarak yapılmaz.

Global optimizasyon yalnız tam çözünürlüklü `stride 1` çalışmalarında açılır. Ham üçgenleme hiçbir zaman silinmez.
Optimizasyon reprojection, kemik kararlılığı veya gövde düzeltme güvenlik sınırını aşarsa otomatik olarak ham
sonuca döner. Kısa kapanmadan tamamlanan noktalar doğrudan gözlenmiş noktalardan ayrı etiketlenir; uzun ve
kanıtsız boşluklar uydurulmaz.

Her yeni çalışma klasöründe başlıca şu sonuçlar oluşur:

- `videos/c01_vitpose_2d_overlay.mp4` ... `videos/c09_vitpose_2d_overlay.mp4`
- `videos/vitpose_skeleton_3d_world.mp4`
- `viewer/pose3d_viewer.html` — tarayıcıda döndürme, yakınlaştırma, oynatma ve kare seçme
- `csv/vitpose_keypoints_2d_raw_flat.csv` ve `csv/vitpose_keypoints_2d_flat.csv`
- `csv/vitpose_keypoints_2d_prefeedback_flat.csv` — geri besleme öncesi stabilize 2B
- `csv/vitpose_keypoints_2d_geometry_flat.csv` — yalnız gerçek görüntü kanıtı taşıyan 3B giriş gözlemleri
- `csv/vitpose_keypoints_2d_feedback_provenance.csv` — özgün, ısı-haritasıyla düzeltilmiş veya yalnız videoda çok-kameradan kurtarılmış
- `csv/vitpose_keypoints_3d_world_triangulated_flat.csv` — değiştirilmeyen ham 3B başlangıç
- `csv/vitpose_keypoints_3d_world_global_optimized_flat.csv` — global optimizasyon sonucu
- `csv/vitpose_keypoints_3d_provenance.csv` — gözlenmiş, kısa kapanmadan tamamlanmış veya kullanılamaz
- `csv/vitpose_keypoints_3d_world_unsmoothed_flat.csv` ve `csv/vitpose_keypoints_3d_world_flat.csv`
- `json/global_pose_optimization_report.json` — önce/sonra ölçümleri, kamera ağırlıkları ve güvenlik kapısı
- `json/crossview_2d_feedback_report.json` — her düzeltme kararı, kabul/red nedeni ve geometri geri dönüş kapısı
- `json/camera_health_report.json` — kamera/eklem hata dağılımı; sol-sağ, senkron, kalibrasyon ve 2B algılama hipotezleri
- `json/pose2d_stability_report.json`, `json/pose3d_stability_report.json` ve `json/run_quality_report.json`

HTML dosyasını çift tıklayarak açabilirsin; ayrıca bir web sunucusu çalıştırmak gerekmez. Son başarıyla tamamlanan
çalışmanın konumu `outputs/aist_test/latest_run.json` içinde tutulur.

Model ve tek kamera kontrollerini ayrı çalıştırmak veya daha hızlı önizleme almak için:

```powershell
cd C:\Users\WWWW\Desktop\tk3d
.\.venv312\Scripts\Activate.ps1
python scripts\check_models.py --session data\aist_test\session.yaml
python scripts\run_pose2d_overlays.py --session data\aist_test\session.yaml --camera c01 --stride 10
python scripts\run_vitpose_multiview_3d.py --session data\aist_test\session.yaml --stride 10 --run-id preview-01
```

Not: `--max-frames` sadece kısa preview üretmek için kullanılır. Tam video ile aynı süreli çıktı istiyorsan
`--max-frames` verme. `--stride` modelin kaç karede bir çalışacağını belirler; çıktı videosunun süresi korunur.
Kameralar arası zaman farkları `session.yaml` içindeki `sync.offsets` alanından okunur. Elle verdiğin `--run-id`
daha önce kullanıldıysa veri kaybını önlemek için işlem başlamaz; yeni bir ad kullan veya tarihli klasörü otomatik
üreten `run_full_aist_pose.ps1` komutunu çalıştır.

Var olan bir tam çalışmada belirli kameranın sol-sağ kimliği, görüntü hizası ve zaman kayması hipotezlerini ayrıca
incelemek için:

```powershell
python scripts\diagnose_camera_consistency.py `
  --run outputs\aist_test\runs\<run_id> `
  --camera c05 `
  --max-shift 350
```

AIST örnek dosyasında c05 videosunun dans bölümü diğer kameralardan 268 kare geç başlar. Geometri taraması
`+268`, görüntüden bağımsız el/gövde hareketi taraması `+269` verdiği için oturum sözleşmesine göre
`sync.offsets.c05: -268` kullanılır. Bu düzeltme c05 medyan uzlaşma hatasını kısa dokuz-kamera testinde
`45,57 px` değerinden `4,58 px` değerine, c05 düzeltme hedefi sayısını `300` değerinden `0` değerine indirdi.
Dokuz kameranın gerçek ortak bölgesi 451 karedir; sistem eşzamanlı olmayan 268 kareyi aynı anmış gibi
üçgenleştirmez. Başka veri setlerinde ofset otomatik yazılmaz; iki bağımsız sinyal uyuşmuyorsa tanı aracı
`ambiguous_periodic_motion` bildirir.

Ana çıktılar:

- `outputs/aist_test/runs/<run_id>/videos/c01_vitpose_2d_overlay.mp4` ... `c09_vitpose_2d_overlay.mp4`
- `outputs/aist_test/runs/<run_id>/videos/vitpose_skeleton_3d_world.mp4`
- `outputs/aist_test/runs/<run_id>/viewer/pose3d_viewer.html`
- `outputs/aist_test/runs/<run_id>/json/vitpose_session_3d.json`
- `outputs/aist_test/runs/<run_id>/json/global_pose_optimization_report.json`
- `outputs/aist_test/runs/<run_id>/csv/vitpose_keypoints_3d_world_flat.csv`

Not: AIST++ camera data indirildiğinde `scripts\import_aist_cameras.py` sekansın `mapping.txt` kaydını okuyup `outputs/aist_test/calibration/cameras.json` üretir. Bu dosya `aist_official_multiview` olarak işaretlenir ve gerçek AIST++ intrinsic/extrinsic değerlerini kullanır. Kendi poomsae kameraların için senkron checkerboard calibration gerekir.
## SMPL Mesh İnsan Modeli

Çubuk iskelet yerine gerçek insan yüzeyi/mesh görmek için SMPL aşaması kullanılır. Bunun için AIST++ motion annotation
dosyası ve kullanıcının kendi hesabıyla indirdiği lisanslı SMPL body model dosyası gerekir; ikisi de repoda tutulmaz.
Ayrıntılı kurulum: `docs/smpl_mesh_setup.md`.

SMPL model dosyasını koyduktan sonra:

```powershell
cd C:\Users\WWWW\Desktop\tk3d
.\.venv312\Scripts\Activate.ps1
python scripts\render_aist_smpl_mesh.py --session data\aist_test\session_all.yaml --smpl-dir models\smpl --gender MALE --max-frames 120 --stride 1
```

Beklenen mesh çıktıları:

- `outputs/aist_test/videos/aist_smpl_mesh.mp4`
- `outputs/aist_test/figures/aist_smpl_mesh_frame0.obj`
- `outputs/aist_test/json/aist_smpl_mesh_report.json`


Mouse ile döndürülebilen oynayan Open3D viewer opsiyoneldir. Python 3.13 ortamında Open3D paketi bulunmayabilir; bu durumda tarayıcıdaki Three.js viewer kullanılmalıdır.

```powershell
python scripts\view_aist_smpl_mesh_open3d.py --session data\aist_test\session_all.yaml --smpl-dir models\smpl --gender MALE --max-frames 240 --stride 1
```


Tarayicida acilan interaktif Three.js viewer uretmek icin:

```powershell
python scripts\export_aist_smpl_threejs_viewer.py --session data\aist_test\session_all.yaml --smpl-dir models\smpl --gender MALE --max-frames 240 --stride 1
```

Cikti: `outputs/aist_test/viewer/aist_smpl_viewer.html`

## Puanlama Altyapısı: Güvenli Geliştirme Akışı

Puanlama hazırlık altyapısı çalışır durumdadır. Güncel ZED RGBD koşusunda iç geometri ve RGB-vs-depth sensör tutarlılığı
geçtiği için `provisional_scoring_ready=true` olur; sistem mocap veya başka dış ground-truth beklemeden
kalite, biomekanik ve hareket-segmenti kanıtlarını üretir. Kaynaksız eski 0-100 provisional motor kaldırılmıştır;
puan kararları yalnız kaynak-bağlı RulePack hattında üretilebilir. Dış doğruluk bilgi amaçlı
`external_accuracy.status=not_evaluated_for_this_run` kalır ve resmî doğruluk iddiası oluşturmaz. ZED stereo depth,
confidence, kalibrasyon, zaman damgaları ve IMU sistemin kendi sensör kanıtıdır. Geçersiz eklem veya yetersiz kamera
görüşü puana katılmaz; `official_scoring_ready` uzman kural/hakem doğrulaması olmadığı için `false` kalır.

27 Temmuz 2026 tarihli tarihsel MADS F2 RGB-only kör testi üç resmî kamera, stride 2 ve 300 inference örneğiyle
`90,409 mm` MPJPE ölçmüştür; iç geometri geçmesine rağmen o koşunun `50 mm` dış doğruluk hedefi geçmemiştir.
P95 `162,504 mm`, açı MAE `13,426°`, hız MAE `0,400 m/s`, ivme MAE `5,947 m/s²` ve kemik uzunluğu CV `%6,490`
değerleri yalnız bu MADS koşusuna bağlı tarihsel referanstır. Güncel ZED RGBD çıktısına kopyalanmaz, onun başarısız
metriği sayılmaz ve yeni mimarinin dış doğruluğunu ölçmez.

### MADS F2 üzerinde baştan sona çalıştırma

Aşağıdaki PowerShell bloğu tek seferde kopyalanabilir. Her çalıştırmada benzersiz bir çalışma kimliği üretildiği için
eski sonuçların üzerine yazılmaz:

```powershell
cd C:\Users\WWWW\Desktop\tk3d
.\.venv312\Scripts\Activate.ps1

$runId = "mads-kata-f2-scoring-$(Get-Date -Format yyyyMMdd-HHmmss)"
$runRoot = "outputs\mads_kata_f2\runs\$runId"

python scripts\check_models.py `
  --session data\mads_test\local\sessions\mads_kata_f2.yaml `
  --model-config config\model_config.yaml

python scripts\run_vitpose_multiview_3d.py `
  --session data\mads_test\local\sessions\mads_kata_f2.yaml `
  --model-config config\model_config.yaml `
  --stride 2 `
  --max-frames 300 `
  --run-id $runId `
  --allow-low-quality-output

python scripts\evaluate_ground_truth_3d.py `
  --prediction "$runRoot\json\vitpose_session_3d.json" `
  --ground-truth data\mads_test\local\ground_truth\multiview\Kata_F2.json `
  --output-dir "$runRoot\ground_truth_validation" `
  --allow-failed-quality-gate

python scripts\analyze_pose_for_scoring.py `
  --session data\mads_test\local\sessions\mads_kata_f2.yaml `
  --input-json "$runRoot\json\vitpose_session_3d.json" `
  --smoothing-window 5

Write-Host "Sonuç klasörü: $runRoot"
```

`--allow-low-quality-output` ve `--allow-failed-quality-gate` yalnız geliştirme/teşhis içindir.
`analyze_pose_for_scoring.py` puan üretmez; kalite, biomekanik ve segment kanıtlarını hazırlar. Hızlı bir smoke test için
yalnız 3B üretim komutunda `--stride 20 --max-frames 30` kullanılabilir; 30
kare güvenilirlik kararı veya model onayı için yeterli değildir.

Ground-truth bulunursa `evaluate_ground_truth_3d.py` ayrıca koşuya bağlı dış doğruluk raporu üretir. Bulunmaması
hazırlık analizini durdurmaz. Kaynak-bağlı puanlama katmanı 3B tahmin, RulePack, PoomsaeSpec, MovementTimeline ve
mühendislik profilinin koşuyla uyumunu ayrı olarak doğrular. Tarihsel MADS raporu başka bir koşunun doğruluğu olarak
kabul edilmez. `official_scoring_ready`, uzman onaylı poomsae kural/hakem doğrulaması tamamlanana kadar ayrı olarak
`false` kalır.

### Çıktılar nerede?

Bütün sonuçlar `outputs/mads_kata_f2/runs/<run_id>/` altında aynı çalışmaya ait olacak şekilde tutulur.

| Öncelik | Dosya | Ne gösterir? |
| --- | --- | --- |
| 1 | `scoring_readiness_analysis.xlsx` | Kalite, biomekanik ve segment kanıtlarını tek dosyada gösterir; puan içermez. |
| 2 | `ground_truth_validation/ground_truth_validation_report.json` | MPJPE, P95, PCK, açı hatası ve ground-truth kalite kapısı sonucunu gösterir. |
| 3 | `ground_truth_validation/scoring_authorization.json` | Koşuya bağlı `scoring_ready` kararı, dosya özetleri ve ret nedenlerini gösterir. |
| 4 | `json/scoring_readiness_report.json` | Kaç kare ve eklemin değerlendirmeye uygun olduğunu gösterir. |
| 5 | `videos/vitpose_skeleton_3d_world.mp4` | Üretilen 3B iskeleti görsel olarak kontrol etmeyi sağlar. |

Diğer ayrıntılı çıktılar:

- `json/vitpose_session_3d.json`: filtrelenmiş ham 3B eklemler, güvenler ve kullanılan kamera sayıları
- `json/vitpose_session_3d_smoothed.json`: puanlama analizinde kullanılan yumuşatılmış 3B iskelet
- `json/run_quality_report.json`: kalibrasyon/reprojection/geçerlilik temelli iç geometri kontrolü
- `csv/pose_quality_frames.csv`: kare bazında kalite ve puanlamaya kabul durumu
- `csv/pose_quality_joints.csv`: eklem bazında kalite özeti
- `csv/biomechanics_timeseries.csv`: açı, hız, denge ve diğer biomekanik zaman serileri
- `csv/movement_segments.csv`: otomatik hareket segment adayları
- `ground_truth_validation/ground_truth_frame_errors.csv`: kare bazında gerçek 3B hata
- `ground_truth_validation/ground_truth_joint_errors.csv`: eklem bazında gerçek 3B hata

Eski `provisional_scoring_report.json` ve 0-100 generic teknik skor artık üretilmez. Accuracy kararları
`build_source_bound_accuracy_decisions.py` ile kaynak-bağlı RulePack/PoomsaeSpec hattında oluşturulur; gözlenemeyen
veya eksik kanıtta puan alanı `null` kalır.

### AIST üzerinde eski geliştirme akışı

AIST yalnız akış/smoke testi için korunur; MADS ground-truth doğruluk benchmark'ının yerini almaz:

```powershell
python scripts\analyze_pose_for_scoring.py `
  --session data\aist_test\session_all.yaml `
  --smoothing-window 5
```

Tek kamera 2D cubuk overlay gerekiyorsa zaten mevcut komut kullanilir:

```powershell
python scripts\run_pose2d_overlays.py --session data\aist_test\session_all.yaml --camera c01 --stride 1
```
