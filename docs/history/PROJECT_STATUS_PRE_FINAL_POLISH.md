# TK3D Güncel Proje Durumu

Son doğrulama tarihi: **27 Ağustos 2026**
Doğrulanan çalışma ağacı: **temel `49bf063` + Phase A/Phase B/Phase C/Phase D düzeltmeleri (henüz commitlenmedi)**
Dal: **`main`**

Bu dosya değişken proje durumunun tek kısa kaynağıdır. Yeni oturumlarda geçmiş
`sohbet*.md` dosyalarını baştan okumak yerine önce burası okunmalıdır.

## Professionalization Phase D ölçümü

- Phase D yalnız ölçüm/enstrümantasyon aşamasıdır; model, eşik, örnekleme,
  geometri, depth-fusion, optimizer veya puanlama davranışı optimize
  edilmemiştir. Opsiyonel `PerformanceCollector`, hiyerarşik duvar-zamanını,
  ViTPose PyTorch CUDA Event sürelerini, opaque RF-DETR çağrısının senkronize
  duvar-zamanını ve Torch peak VRAM değerlerini ayrı
  `json/performance_report.json` artifact'ine yazar.
- Birincil `CURRENT_ACTIVE` ölçümü iki gerçek ZED kamera, gerçek SVO depth,
  `zed_fusion_multiview`, stride 1 ve kare 0-259 ile yapıldı. Üç profilli
  tekrarın internal toplamı ortalama `282,778 s`; core işleme ortalaması
  `218,503 s`; steady-state 140-259 inference+geometry hızı ortalama
  `1,833 FPS` oldu. Tek profillemesiz eş koşunun dış duvar-zamanına göre kaba
  fark `%4,04` olsa da tekrar CV'si `%9,26` ve üçüncü koşudaki belirgin
  thermal/power yavaşlaması nedeniyle bu fark profiler overhead kanıtı değildir.
- Ölçülen en büyük üst düzey maliyetler: ZED depth-fusion `70,688 s`, ViTPose
  ve causal 2B `61,254 s`, triangulation `52,057 s`, artifact serialization
  `38,546 s`, RF-DETR + ByteTrack `25,174 s`. Cross-view açık olsa da hedef
  sayısı sıfırdı; guided ikinci pose geçişi bu aktif koşuda `ZERO_WORK` olarak
  sınıflandırıldı.
- ViTPose normal+flip CUDA Event toplamı koşu başına ortalama `49,496 s`;
  RF-DETR predict senkronize duvar-zamanı `23,143 s`; ZED patch extraction
  `32,493 s`; sequence-level RGB/depth optimizer toplamı `7,328 s` ölçüldü.
  Torch peak allocated/reserved değerleri sırasıyla `3642,120 MiB` ve
  `3854 MiB` idi. ZED SDK belleği ve process-level RAM/CPU bu değerlere dahil
  değildir.
- Ayrı 741-kare Poomsae profili `56,673 s` sürdü: analiz/karar `12,821 s`,
  presentation/export `43,597 s`. En büyük sunum maliyetleri işaretli hata
  videosu `25,036 s` ve tarayıcı video üretimi `15,863 s` oldu.
- Aynı uzunluktaki profillemesiz koşu ile üç profilli koşunun 12 CSV'si byte
  düzeyinde eşit; ana 3B bilimsel alanlar ve normalize kalite raporları eşit.
  Poomsae Phase C referansına karşı üç sayısal CSV ve technical-conformance
  JSON byte-eşit; karar, readiness, segmentasyon, WholeBody ve video manifest
  sözleşmeleri run yolu/provenance dışında semantik olarak eşittir.
- Phase D yerel teslim kapısı: Ruff temiz, `273 passed`, `pip check`, performans
  raporu şema doğrulaması ve `git diff --check` temiz. Commit/push yapılmadı.

## Professionalization Phase C doğrulaması

- Clean-checkout ile yerel araştırma makinesi arasındaki sözleşme
  `docs/REPRODUCIBILITY.md` içinde tanımlıdır. Tier 1; GPU, model ağırlığı,
  gerçek video/SVO, üretim calibration veya tarihsel `outputs/` gerektirmez.
  Tier 2, aktif ZED RGBD + `poomsae1_trimmed` araştırma doğrulamasıdır.
- `tk3d-check` modelleri yüklemeden paket/import, giriş noktası, config/YAML,
  output yazılabilirliği ve aktif dış varlıkları denetler. Bu makinede `READY`;
  dış araştırma varlıkları olmayan sentetik clean-checkout testinde Tier 1
  `READY`, genel durum `PARTIALLY_READY` olur.
- Küçük versioned fixture'lar current/legacy ana 3B ve kalite artifact'lerini
  ve sentetik production-mode calibration'ı korur. Future schema, malformed,
  eksik alan, non-finite sayı ve manifest-binding uyuşmazlıkları fail-closed
  test edilir; testler gerçek `outputs/` içeriği okumaz.
- Run lifecycle `preparing`, `running`, `completed`, `failed` durumlarını açıkça
  kaydeder. Failed/incomplete veya birleşik akışın ara aşaması önceki başarılı
  `latest_run.json` işaretçisini değiştiremez; geçmiş run üzerine yazılmaz.
- 26 Ağustos 2026 yerel teslim kapısı: editable kurulum ve üç console entry
  point çalıştı; readiness `READY`; Ruff temiz; `268 passed`; `pip check` ve
  `git diff --check` temiz. CI, Windows/Python 3.11 + CPU Torch ile aynı hafif
  kapıları çalıştıracak şekilde düzenlendi; GitHub-hosted CI bu çalışma ağacında
  henüz uzaktan çalıştırılmış sayılmaz.
- Phase C aktif ZED smoke'u, ayrı validation kökünde iki gerçek kamera, SVO
  depth, `zed_fusion_multiview`, stride 1 ve 30 kareyle tamamlandı. Phase B kısa
  run'ıyla 12 CSV byte düzeyinde aynı; kare/zaman kimliği, 5.01222182931652 px
  ortalama reprojection, depth acceptance, optimizer/fallback, quality ve
  readiness durumları değişmedi. Run lifecycle `completed` oldu; ana aktif
  session'ın `latest_run.json` işaretçisi değişmedi.
- `phasec-poomsae-smoke-20260826`, Phase B Poomsae smoke'uyla üç sayısal CSV ve
  technical-conformance JSON düzeyinde byte-eşittir. M01-M06 kapsamı ve karar
  sonuçları aynı; `rule_scoring_ready:false`, `judge_calibrated_ready:false`,
  `official_scoring_ready:false` korunmuştur.

## Professionalization Phase B doğrulaması

- Kanonik uygulama API'leri `src/multiview_application.py` ve
  `src/poomsae_scoring/application.py` içindedir. CLI dosyaları sırasıyla 86 ve
  105 satırlık argüman/çıkış kodu adaptörleridir.
- Editable kurulum `tk3d-multiview` ve `tk3d-poomsae` konsol komutlarını sağlar;
  doğrudan script çağrıları geriye uyumluluk için korunur.
- Artifact SHA-256, katı JSON okuma ve exclusive JSON yazma işlemleri
  `src/artifact_io.py` içinde merkezileştirilmiştir. `scripts/` altında çalışma
  zamanında `sys.path` değiştiren bootstrap kalmamıştır.
- Güncel şemalı ana 3B artifact tüketicileri, artifact ile run manifestinin
  `session_id`, `run_id`, calibration snapshot ve checksum bağlarını doğrular.
  Şemasız tarihsel artifact desteği açıkça `LEGACY_SUPPORTED` olarak ayrılır.
- Workflow/veri sınıfları ile aktif, uyumluluk ve tarihsel bileşen sınırları
  `docs/LEGACY_COMPONENTS.md` içinde kayıtlıdır; bu aşamada legacy dosya
  silinmemiştir.
- 26 Ağustos 2026 doğrulama kapısı: Ruff temiz, `255 passed`, `pip check`
  temiz. Birincil `CURRENT_ACTIVE` regresyonu, aynı profil ve aynı bağlı ZED 3B
  pose hash'ini kullanan `poomsae1-auto-segmentation-final-v3-20260823` ile
  `phaseb-poomsae-smoke-20260826` arasında yapıldı. Üç sayısal CSV ile technical
  conformance JSON'u byte düzeyinde aynı; diğer domain JSON farkları yalnız run
  yolları, doğal run-history alanları ve Phase A'nın açık fail-closed/legacy
  contract alanlarıdır. M01-M06 kapsamı 6/6 işlendi; tam Accuracy ve resmî skor
  yine `null`, `rule_scoring_ready:false`, `official_scoring_ready:false`.
- Aktif 3B uygulama API'si ayrıca ayrı
  `outputs/phaseb_current_active_validation/` kökünde gerçek iki ZED kamera,
  gerçek SVO depth kaynakları, `zed_fusion_multiview` calibration, stride 1 ve
  30 kareyle çalıştırıldı. `phaseb-current-active-zed-smoke-20260826` run'ında
  ham 2B ve stabilize/triangulated kare 0-25, korunmuş 741-kare aktif run'ın aynı
  bölümüyle sayısal olarak birebir aynıdır. Kısa run'ın son dört karesindeki fark
  9-karelik offline filtrenin bilinçli sağ-sınır etkisidir; farklı video bölümü
  sonucu gibi yorumlanmaz. Depth final gate geçti ve çıktı kullanıldı; global
  optimizer uygulandı, fallback kullanmadı; iç kalite geçti ve resmî readiness
  kapalı kaldı. Ana ZED session `latest_run.json` işaretçisi değiştirilmedi.
- AIST `CURRENT_VALIDATION` sınıfında ikincil smoke'tur. `phaseb-smoke-20260826`
  iç geometriyi geçti; Phase A AIST referansının dokuz JSON çıktısıyla run'a özgü
  kimlik/yol alanları dışında, on iki CSV çıktısıyla byte düzeyinde aynı sonuç
  verdi. Bu sonuç aktif ZED/poomsae regresyonunun yerine kullanılmaz.

## Projenin amacı ve mevcut sınırı

TK3D'nin hedefi çok kameralı videodan güvenilir 3B insan pozu üretip bunu
tekvando poomsae analiz ve puanlama sistemine girdi yapmaktır. Mevcut repository
güçlü bir görüntü işleme, 3B geometri, kalite kontrolü ve geçici teknik analiz
altyapısı içerir.

Sistem henüz resmî poomsae puanlamasına hazır değildir:

- Güncel ZED RGBD koşusuna bağlı bağımsız dış 3B ground-truth değerlendirmesi yoktur.
- Tarihsel MADS F2 RGB-only sonucu yalnız kendi koşusunu tanımlar; ZED RGBD
  koşusuna devredilmez ve güncel doğruluk metriği sayılmaz.
- Gerçek poomsae hareket/faz zaman çizelgesi henüz tamamlanmadı.
- Hakem/koç etiketi yoktur; bu, provisional kural motorunun önkoşulu değildir.
  Bu nedenle `judge_calibrated_ready` ve `official_scoring_ready` kapalı kalır.
- İç geometri raporunun geçmesi gerçek 3B doğruluğu tek başına kanıtlamaz.
- Normal çok-kameralı run iç kalite geçtiğinde `provisional_scoring_ready:true`
  yazar; bu değer yalnız kaynak-bağlı kural analizi için veri hazırlığını belirtir,
  kendi başına puan üretmez.
- `official_scoring_ready:false` kalır; dış doğruluk yokluğu provisional akışı
  durdurmaz fakat resmî doğruluk/hakem puanı iddiasına izin vermez.
- Phase A artifact sözleşmesi aktiftir: ana 3B ve run-quality JSON şemaları v1,
  run manifest şeması v1'dir. Her yeni run calibration/config snapshot'larını ve
  model/input/code/environment provenance bilgisini kendi immutable dizininde tutar.
- Accuracy readiness, bağlı WholeBody-133 diagnostics verilmezse artık fail-closed
  davranır. Kaynak-bağlı karar toplamları açıkça provisional observed-scope analizidir;
  tam değerlendirmeye uygunluk, değerlendirilmiş veya resmî skor anlamına gelmez.

## Aktif üretim hattı

```text
senkronize kamera videoları
  -> RF-DETR-Small kişi tespiti
  -> ByteTrack kamera içi kimlik takibi
  -> adaptif ve stabilize kişi crop'u
  -> ViTPose-Huge WholeBody (133 nokta, flip-test)
  -> güven/hareket duyarlı temporal filtre
  -> sıfır-fazlı offline 2B stabilizasyon
  -> robust çok-kameralı ilk triangulation
  -> leave-one-camera-out çok-kameralı 2B geri besleme
  -> güvenli yeniden triangulation
  -> ZED varsa confidence/yüzey-ofset kapılı yardımcı stereo depth fusion
  -> aynı koşuda saf RGB referans dalı ve depth-vs-RGB son kalite kapısı
  -> anatomik/zamansal 3B güvenilirlik kontrolü
  -> BODY-17 global çok-kameralı optimizasyon
  -> 3B stabilizasyon, export ve kalite raporları
```

Aktif yapılandırma `config/model_config.yaml` içindedir:

- 2B model: `ViTPose-Huge-WholeBody`
- Backend: `tk3d_vitpose_plus`
- Girdi: `256x192`
- Veri sözleşmesi: 133 COCO-WholeBody noktası
- Kişi dedektörü: RF-DETR Small
- Tek kamera RTMW3D helper: opsiyonel ve varsayılan kapalı
- Cross-view 2B feedback: açık
- ZED stereo depth fusion: kaynak tanımlı ZED session'larında açık; BODY-17 ile sınırlı
- Global BODY-17 optimization: açık

## Ana veri ve koordinat sözleşmeleri

- 3B çıktı şekli: `keypoints_3d_world[t, 133, 3]`
- Birim: metre
- Analiz eksenleri: `x=sağ`, `y=ileri`, `z=yukarı`
- Scoring eksen indeksleri `src/coordinate_system.py` içindeki kanonik
  sabitlerden alınır. Gövde eğimi `z=yukarı` ve `y=ileri`, stance length
  `y=ileri` kullanır; sentetik pose fixture'ları da aynı sözleşmededir.
- 133 nokta export ve görselleştirmede korunur.
- Cross-view geri besleme ve global optimizasyon şu anda yalnız BODY-17 üzerinde
  çalışır. El, yüz ve ayak için eşdeğer çapraz-kamera optimizasyonu doğrulanmış
  değildir.
- Kalibrasyon canlı üretimde fail-closed'dur. Kabul edilen üretim modları ortak
  çok-kamera referansı veya resmî AIST çok-kamera kalibrasyonudur.

## Kritik güvenlik mekanizmaları

### Cross-view 2B feedback

- Bir kamera düzeltilirken o kamera 3B öncülden çıkarılır.
- En az dört başka kamera bağımsız öncülü destekler.
- Öncül hedef görüntüde yalnız ViTPose heatmap aramasını sınırlar.
- Adayın gerçek heatmap skoru ve geometrik hatası birlikte iyileşmelidir.
- Kabul edilmeyen aykırı görüntü ölçümü 3B geometriden çıkarılır.
- Yalnız izdüşümle gösterilen nokta turuncu ve
  `visualization_only` provenance ile işaretlenir.
- Yeniden triangulation geçerli nokta, medyan ve P95 reprojection kapılarını
  geçmezse ilk geometriye dönülür.

### Global BODY-17 optimization

- Tüm senkron sekansı birlikte çözer.
- Reprojection, kamera güveni, kemik uzunluğu, dirsek/diz limitleri, ivme,
  jerk ve kısa kapanma bilgilerini kullanır.
- Ham triangulation ayrı dosyada korunur.
- Solver, reprojection, ivme, kemik kararlılığı veya düzeltme sınırı
  kötüleşirse ham sonuca otomatik dönüş yapılır.
- Kısa kapanmadan tamamlanan nokta doğrudan gözlenmiş noktadan ayrı provenance
  taşır; uzun kanıtsız boşluklar doldurulmaz.

## Güncel doğrulama

### İlk ZED 2i SVO2 pilotu

28 Temmuz 2026 tarihinde
`HD720_SN39504762_13-06-55.svo2` gerçek ZED SDK 5.4.1 ile işlendi:

- ZED 2i seri `39504762`, firmware `1523`;
- kalibre odak `3.823 mm`, yatay görüş `67.73°`; 4 mm lens varyantıyla uyumlu;
- `1280×720`, `60 FPS`, 666 başarıyla okunan kare, `11.134046 s`;
- IMU verisi `666/666` karede mevcut;
- üç adet yaklaşık `33 ms` zaman aralığı görüldü; uzun stres testi henüz yok;
- NEURAL depth geçerli piksel medyanı `%99.945`, güçlü güven medyanı `%95.905`;
- RF-DETR + ViTPose smoke: 67 örnek karede BODY-17 algılama `%100`;
- kişi-depth kapısıyla BODY-17 ve ayak depth eşleşmesi `%100`;
- güvenli yüz depth eşleşmesi `%90.89`, el depth eşleşmesi `%81.69`;
- naif eklem-piksel depth örneklemesinin siluet kenarlarında arka plana
  sıçradığı gösterildi; üretim entegrasyonu kişi maskesi ve çok-kamera
  residual kapısı olmadan yapılmamalıdır;
- doğru zaman çizelgeli sol RGB ve teşhis raporları
  `outputs/zed2i_pilot/runs/zed2i-svo-diagnostic-20260728-180605/`
  altındadır.

Bu pilot tek-kamera RGB-D teknik doğrulamasıdır. Ortak dünya koordinatlı
çok-kamera 3B veya puanlama yetkisi sağlamaz. Daha sonra eklenen çok-kamera ZED
ingest ve güven kapılı stereo depth-fusion yolu aşağıdaki doğrulamada açıklanır.

### İlk iki ZED 2i ortak-dünya poomsae koşusu

1 Ağustos 2026 tarihinde seri `35151067` ve `37137479` olan iki ZED 2i SVO2
kaydı ZED SDK 5.4.1, donanım zaman damgaları ve ZED Fusion kalibrasyonuyla
işlendi:

- her kaynakta `1189` kare, `1280×720`, `60 FPS`; ikisinde de iki adet yaklaşık
  `33 ms` zaman boşluğu;
- ortak kayıpsız FFV1 zaman çizelgesi `1190` kare ve `19.833333 s`;
- zaman eşleme P95 artık hatası sırasıyla `5.704 ms` ve `1.066 ms`; ikinci
  kamerada iki kare `16.041 ms` büyük artık taşıdı;
- ZED Fusion kamera pozları `RIGHT_HANDED_Z_UP/METER` olarak alındı, IMU
  yerçekimi dönüşü uygulandı ve kamera bazları OpenCV optik projeksiyonuna
  çevrildi;
- tam koşu `stride 1`, `1190` inference örneği, ViTPose-Huge flip-test,
  sıfır-fazlı 2B stabilizasyon ve global BODY-17 optimizasyon adayıyla çalıştı;
- iki kamera da sağlık raporunda `healthy`; en iyi temporal shift iki kamerada
  da `0` kare;
- BODY-17 gözlenen oranı `%77.7904`, güvenilir son oran `%72.2936`;
- ortalama reprojection `5.083 px`; kamera konsensüs P95 değerleri `9.701 px`
  ve `9.200 px`;
- ana hareket bölgesi olan kare `300–900` için güvenilir BODY-17 oranı
  `%90.1537`; ilk/son yakın-kadraj bölümleri toplam oranı düşürdü ve video
  kırpılmadı;
- 2B yüksek-frekans oranı kameralarda `%44.90` ve `%46.02`, 3B eklem oranı
  `%33.48`, açı oranı `%31.31` azaldı;
- global optimizasyon kemik ve ivme kararlılığını iyileştirse de reprojection
  P95'i `7.971 -> 11.223 px` kötüleştirdiği için güvenli ham triangulation'a
  dönüldü;
- iç geometri kapısı başarısız, ground-truth değerlendirilmedi ve
  `scoring_ready=false`; koşu `latest_run.json` olarak işaretlenmedi;
- `[1190, 133, 3]` JSON/CSV, iki 2B overlay, 3B video ve HTML izleyici üretildi;
  JSON/CSV içinde `NaN`/`inf` sızıntısı bulunmadı.

Koşu:
`outputs/poomsae_1_zed2i_20260731_ultra/runs/poomsae1-zed2i-ultra-full-20260801/`.
Hazırlama ve senkron provenance raporu aynı session'ın `source/reports/`
klasöründedir. Kalıcı `scripts/prepare_zed_multiview_session.py` giriş yolu
kamera sayısından bağımsızdır; üçüncü kamera aynı Fusion ortak dünyasıyla SVO
listesine eklenebilir. İki/üç kamera, en az dört bağımsız destek görüşü isteyen
cross-view 2B düzeltmesini otomatik olarak etkinleştirmez.

Kullanıcı isteğiyle yakın-kadraj başlangıç/bitiş ayrıca türetilmiş bir session'da
kırpıldı; parent tam koşu korunmuştur. Ortak kaynak kare `190–930`, çıktı `741`
kare/`12.35 s` ve `stride 1`'dir. Kırpılmış koşuda iç geometri geçti, BODY-17
gözlenen oran `%96.6024`, temporal recovery `%1.2860`, güvenilir son oran
`%97.8884`, ortalama reprojection `5.209 px` oldu. Global optimizasyon kabul
edildi; kemik CV `%8.788 -> %1.765`, ivme P95 `15.807 -> 11.941 m/s²` değişti.
Ground-truth değerlendirilmediği için `scoring_ready=false` kaldı. Koşu:
`outputs/poomsae_1_zed2i_20260731_trimmed/runs/poomsae1-zed2i-trimmed-ultra-20260801/`.
Bu oranlar daha kısa, kullanıcı-kırpılmış kapsamı ölçer; tam koşuyla model
iyileştirmesi iddiası için doğrudan karşılaştırılamaz.

### İki ZED 2i güven kapılı RGB + depth koşusu

2 Ağustos 2026 tarihinde aynı kırpılmış `741` kare, iki kamera, `60 FPS`,
`stride 1`, ViTPose-Huge ve yapılandırmayla yeniden çalıştırıldı. Her iki
kamerada `741/741` çıktı karesi NEURAL depth/confidence desteği aldı; donanım
zaman eşlemesi nedeniyle ikinci kamerada `740` benzersiz SVO depth karesi ve
bir yeniden kullanılan kaynak kare vardır.

- RGB çok-kamera triangulation ana 3B kanıt olarak korundu;
- deri/kıyafet yüzeyi ile eklem merkezi farkı koşuya özel kamera/eklem ofsetiyle
  ayrıldı; bu ofset genel yapılandırmaya taşınmadı;
- `23.014` güvenli depth örneğinden `7.669` BODY-17 noktası nokta kapısını geçti;
- medyan fusion düzeltmesi `9,97 mm`, P95 `23,34 mm`, maksimum `49,94 mm`;
- depth uyum medyanı `28,42 -> 19,99 mm`, P95 `67,22 -> 47,07 mm`;
- aynı koşuda ayrı saf RGB global optimizasyon dalı çözüldü; depth adayı son
  reprojection, ivme ve kemik kararlılığı kapılarının tamamını geçti ve seçildi;
- son ivme P95 `11,941 -> 11,777 m/s²`, eklem limiti ihlali `28 -> 27`;
- son reprojection P95 `8,833 -> 8,867 px` ve kemik CV `%1,765 -> %1,767`
  küçük ölçüde yükseldi fakat tanımlı güven sınırları içinde kaldı;
- geçerli BODY-17 oranı değişmedi: `%97,8884`; ortalama reprojection `5,210 px`;
- üç video da `741` kare, `60 FPS`, `12,35 s`; JSON/CSV'de `NaN`/`inf` yok;
- iç geometri ve ZED RGB-vs-depth iç sensör tutarlılığı geçti;
- bu koşuya bağlı dış ground-truth değerlendirilmedi, tarihsel benchmark
  devralınmadı; `external_accuracy=not_evaluated_for_this_run` bilgi amaçlı kaldı;
- dış ground-truth beklenmeden `provisional_scoring_ready=true`; o tarihte çalışan
  legacy generic motor `71,7178` değerli `provisional_not_official` skor üretti;
- hareket eşiğinin sürekli harekette aşırı yükselmesi düzeltildi; ikinci koşuda
  durum `ready_for_scoring_infrastructure`, hazır kare oranı `1,0` oldu;
- `official_scoring_ready=false`.

Son yeniden üretilen nihai koşu:
`outputs/poomsae_1_zed2i_20260731_trimmed/runs/poomsae1-zed2i-rgbd-gated-ultra-rerun-20260802/`.
Koşu 741/741 kare, stride 1 ve aynı ultra profille tamamlandı; üç video da
`60 FPS`, `12,35 s`, `1280x720` çıktı. İç geometri ve depth kapısı geçti,
`7.669` BODY-17 depth fusion noktası kullanıldı. Tarihsel legacy analiz
`71,7178/100`, `ready_for_scoring_infrastructure`, hazır kare oranı `1,0` ve
`15` hareket segment adayı üretti. Kaynaksız 0-100 legacy motor 12 Ağustos 2026
tarihinde koddan kaldırıldı; bu sayı yalnız tarihsel koşu kaydıdır ve yeniden
üretilmez. O koşuda toplam 13 JSON ve 20 CSV doğrulandı;
`NaN`/`inf` sızıntısı ve tarihsel MADS referansı bulunmadı.
IMU bu koşuda kamera yerçekimi/yönelim kalibrasyonunda kullanıldı; sporcu
pozu için kare bazlı IMU düzeltmesi uygulanmadı. ZED depth/confidence kendi
sensör zincirimizin iç kanıtıdır; dış ground-truth değildir. Depth'in gerçek 3B
doğruluğu artırdığı iddiası için bu koşuyla uyumlu bağımsız mocap/ölçüm referansı
hâlâ gerekir.

### Taegeuk 1 Accuracy kural motoru

18 Ağustos 2026 tarihli puanlama zinciri (v5). Her aşama fail-closed'dur ve
hiçbir aşama başka aşamanın çıktısını sahtelemez:

```text
video -> doğrulanmış 3B poz
  -> MovementTimeline
       (manuel etiket | build_automatic_movement_timeline())
  -> wholebody_diagnostics_report.json                (ölçüm, puansız)
  -> derive_categorical_observations()                (otomatik 3 sn duraklama)
  -> build_technical_conformance()                    (hareket bazlı füzyon, puansız)
  -> build_source_bound_accuracy_decisions()          (tarihsel provisional -0,1)
  -> build_presentation_diagnostics()                 (puan iddiası yok)
```

Zaman çizelgesi zincirin **girdisidir**, ara çıktısı değil: WholeBody teşhisi
segment ve anchor'ları okumak için doğrulanmış bir timeline ister.

**Bağlanma durumu.** Otomatik duraklama, yanlış hareket/duruş teşhisi, hareket
bazlı teknik uygunluk ve Presentation aşamaları `scripts/run_poomsae_scoring.py`
tek-komut akışına bağlıdır. Mevcut kısa kayıt için manuel ve hash-bağlı timeline korunur;
otomatik segment/onay katmanı kullanılmaz.

Zincire 15-18 Ağustos 2026 arasında eklenen dört yetenek:

- **Eolgul-makki sayısal kuralı.** `_eolgul_fist_to_forehead_ratio()` blok
  yumruğundan alın merkezine mesafeyi yumruk genişliği biriminde ölçer. Alın
  merkezi doğrudan ölçülen kaş çizgisidir (iBUG `17-26`), ekstrapolasyon
  yoktur. Kural `HIST-2014-EOLGUL-FIST-FOREHEAD-ONE`, aralık `0,5-1,5` yumruk,
  kaynak 2014 guideline sayfa `17`, statü tarihsel-provisional. M13 ve M15
  `technique.eolgul_makki.forehead_distance` kriterini taşır. Göz ayrımı ve
  yumruk genişliği kalite kapıları geçilmezse ölçüm yapılmaz;
- **Otomatik 3 saniye duraklama tespiti.** `derive_categorical_observations()`
  zaman çizelgesindeki yalnız iki etiketli segment arasında kalan boşlukları
  tarar, `>= 3` saniye olanları
  `pause_at_least_3_sec` kategorik gözlemine dönüştürür
  (`confirmation_method: duration_measurement`). Kayıt sonundaki sınırı belirsiz
  kuyruk boşluğu taranmaz. Otomatik timeline'dan gelen gözlem `inferred` kalır;
  yalnız manuel doğrulanmış komşu segmentler ve yeterli gerçek güven değeri
  `observed` olabilir. Güven değeri hiçbir zaman yapay olarak yükseltilmez.
  `wrong_action`/`wrong_stance` için yalnız timeline sırasına dayalı çıkarım
  yapılmaz. `build_categorical_diagnostics()` mevcut 3B ölçümlerin beklenen
  teknik/duruş profili yerine alternatif profile açıkça uyduğu durumları
  `kinematic_screening` ve `evidence_status=inferred` olarak raporlar. Bunlar
  inceleme adayıdır; doğrulanmış sınıflandırıcı veya doğrudan gözlem olmadığı
  için otomatik büyük kesinti oluşturmaz;
- **Otomatik MovementTimeline türetme.** `build_automatic_movement_timeline()`
  pose-DTW, otomatik atlama cezası ve belirsizlik bandını birleştirip
  `label_source=automatic` bir taslak timeline üretir; otomatik eşleşmeler
  `provisional` veya `ambiguous` kalır. DTW aynı segmenti birden çok harekete
  ya da aynı hareketi birden çok segmente eşleyebildiği için iki yönde de
  en düşük maliyetli bire-bir eşleşme seçilir. Kısmi kayıt sözleşmesinin ifade
  edemediği orta/prefix eksikleri manuel timeline incelemesine yönlendirilir.
  Fonksiyon
  segment **tespiti** yapmaz, dışarıdan pre-detected segment alır;
- **Presentation teşhis motoru.** `src/poomsae_scoring/presentation.py` üç
  bileşenli teşhis raporu üretir: `speed_and_power`, `rhythm_and_tempo`,
  `expression_of_energy`. Sözleşme gereği `total_score=null`,
  `judge_calibrated=false`, `score_claim_allowed=false`. Yeni tolerans değeri
  veya yeni dış kaynak eklemez; mevcut WholeBody metriklerini toplar ve zaman
  çizelgesinden süre/boşluk türetir. Ayrıntısı
  `docs/PRESENTATION_DIAGNOSTICS.md` içindedir.
- **M01-M06 teknik uygunluk motoru.**
  `src/poomsae_scoring/technical_conformance.py`, WholeBody ölçütlerini ve
  kategorik hareket/duruş kontrollerini hareket başına birleştirir. `%95`
  aralığı eşik üzerine biniyorsa kesin aday yerine `boundary_uncertain` üretir;
  timeline güveni, gerekli eklem örnek oranı ve grup kapsamının minimumunu
  `fused_evidence_confidence` olarak korur. Sonuçlar
  `mismatch_candidate`, `review_candidate`, `ambiguous`,
  `consistent_within_measured_scope` veya `not_measurable` olabilir.
  `score_claim_allowed=false` ve `automatic_deduction_allowed=false` değişmez;
  ayrıntı `docs/TECHNICAL_CONFORMANCE.md` içindedir.

18 Ağustos 2026 test kapsamı genişletmesinde bu özelliklerin edge case'leri ve
uçtan uca zinciri test edildi. Bu sırada gerçek bir sözleşme ihlali bulundu ve
düzeltildi: `build_automatic_movement_timeline()` bütün hareketler eşleştiğinde
`recording_scope=complete_performance` ile birlikte dolu bir
`source_end_reason` yazıyordu; sözleşme complete kayıtta bu alanın `null`
olmasını şart koştuğu için fonksiyon kendi çıktısında hata fırlatıyordu. Mevcut
testlerin tamamı kısmi kayıt senaryosu olduğu için bu yol hiç çalışmamıştı.
Alan artık complete kayıtta `null`, kısmi kayıtta
`auto_alignment_missing_movements_detected` değerini alır.

12 Ağustos 2026 tarihinde trimmed ZED kaydı için tek komutlu, profil tabanlı
çalıştırıcı eklendi. `scripts/run_poomsae_scoring.py --profile
poomsae1_trimmed` doğrulanmış 3B pozu WholeBody-133 teşhis, hareket/faz kanıtı,
RulePack readiness, kaynak-bağlı Accuracy kararı ve senkron iki-kamera HTML
incelemesinden geçirir. `--process-video` modu önce ViTPose/RGBD `stride 1`
çalıştırır ve manuel timeline'ı yalnız aynı session/frame/FPS/timestamp
sözleşmesi birebir doğrulanırsa yeni pose hash'ine bağlar. Eski kaynaksız
provisional motor çağrılmaz.

23 Ağustos 2026'da tek-komut zincirine üç yeni çıktı bağlandı:
`categorical_diagnostics_report.json`, `technical_conformance_report.json` ve
`presentation_diagnostics_report.json`.
İnceleme ekranı artık yanlış hareket/duruş kontrollerini beklenen/alternatif
ölçüm aralıklarıyla, Presentation proxy özetlerini ve her WholeBody metriğinin
kanıt penceresine doğrudan atlama düğmesini gösterir; metrik araması içerir.
Gerçek doğrulama koşusu
`outputs/poomsae_1_zed2i_20260731_trimmed/runs/poomsae1-integrated-final-20260823/`
altındadır: 12 kategorik kontrolden 1'i (M05 arae-makki yerine
momtong-jireugi profiline uyum) yalnız inceleme adayı çıktı, 8'i beklenen
profile uydu, 3'ü ölçülemedi. Aday kesintiye uygulanmadı. Presentation proxy
kapsamı 7/7, `total_score=null`; mevcut kaynak-bağlı toplam `0.4` değişmedi.

Aynı gün hareket bazlı teknik uygunluk motorunun nihai gerçek tek-komut koşusu
`outputs/poomsae_1_zed2i_20260731_trimmed/runs/poomsae1-technical-conformance-final-v2-20260823/`
altında üretildi. Gözlenen M01-M06 için PoomsaeSpec'teki `96` beklenen teknik
ölçütün `73` tanesi ölçülebildi, `67` tanesi eşikle değerlendirilebildi. M05
yanlış hareket kimliği adayı nedeniyle `mismatch_candidate`; diğer beş hareket
fixation, el-ayak zamanlaması, bakış veya ağırlık aktarımı adayları nedeniyle
`review_candidate` oldu. Bu altı hareketin inceleme gerektirmesi altı otomatik
kesinti anlamına gelmez: `automatic_deduction_allowed=false`, tam Accuracy
`null`, kategorik uygulanan kesinti sayısı `0` ve mevcut gözlenen-kapsam
provisional toplamı `0.4` olarak kaldı. Dokuz JSON dosyasının tamamı parse
edildi; `NaN`/`Infinity` sızıntısı bulunmadı. HTML iki videoyu senkron olarak
aynı kanıt anına taşıdı, altı teknik kart açıldı, yatay taşma ve tarayıcı konsol
hatası görülmedi.

23 Ağustos 2026 HTML kullanılabilirlik denetiminde kaynak kamera dosyalarının
FFV1/AVI olması nedeniyle tarayıcıların iki videoyu da açamadığı bulundu
(`readyState=0`, süre yok). `scripts/build_browser_review_videos.py`, sabitlenmiş
`imageio-ffmpeg==0.6.0` FFmpeg çalıştırıcısıyla her kamerayı H.264/yuv420p MP4'e
dönüştürür. Kaynak dosya korunur; çıktı tekrar açılıp kare sayısı, FPS ve
çözünürlük birebir doğrulanır. Gerçek iki kamera dönüşümü yaklaşık `15,15 s`
sürdü; her MP4 `741` kare, `60 FPS`, `1280x720`, `12,35 s` kaldı.

HTML senkronizasyonundaki ikinci hata da düzeltildi: iki videonun asenkron
`seeking` olayları birbirini geri tetikleyip `Kanıta git` hedefini tekrar sıfıra
çekiyordu. Olay kilidi artık programatik seek tamamlanana kadar korunur.
İnceleme ekranı ayrıca kamera başına hazır/yükleniyor/hata durumunu, kapsamı
WholeBody bölümüyle sınırlı metrik aramasını, arama sonucunu, kayıtlı karar
sayısını, güvenli JSON indirmeyi ve incelemeleri temizleme düğmesini gösterir.

Aynı geliştirmede `src/poomsae_scoring/run_history.py` eklendi. Tek-komut akışı
`run_history_report.json` ve `run_history.html` üretir; yalnız aynı profil,
kapsam ve moddaki koşuları karşılaştırır. Ölçüm kapsamı/readiness regresyon
uyarıları için ayrıca aynı pose SHA-256 zorunludur. Farklı veri koşulları kalite
sıralamasına dönüştürülmez; ayrıntı `docs/RUN_HISTORY.md` içindedir.

Bu hattın son gerçek doğrulaması
`outputs/poomsae_1_zed2i_20260731_trimmed/runs/poomsae1-html-history-final-v4-20260823/`
altındadır. İki tarayıcı videosunun her biri `741` kare, `60 FPS`, `1280x720`
ve `12,35 s`; manifest kare/FPS/çözünürlük sözleşmesinin korunduğunu doğrular.
Tarayıcı stres denetiminde hızlı oynat-duraklat sonrasında kanıt hedefi iki
kamerayı da `3,633333 s` konumuna sıfır kaymayla taşıdı. `fixation` araması altı
WholeBody sonucu döndürdü ve teknik kartları gizlemedi; kullanıcı inceleme
kararı yeniden yüklemede korundu, JSON indirildi ve temizleme çalıştı. Koşu
geçmişi 22 önceki özetin 11 uyumlusunu karşılaştırdı, aynı pose baseline'ına
karşı regresyon uyarısı üretmedi. V4 altındaki 13 JSON dosyasının tamamı parse
edildi, `NaN`/`Infinity` sızıntısı bulunmadı. Son kod kapısı Ruff, `235` pytest
ve `git diff --check` ile geçti.

23 Ağustos 2026'da analiz çekirdeğine otomatik hareket/faz segmentasyonu
eklendi. `src/poomsae_scoring/automatic_segmentation.py`, BODY-17 eklem
hızlarından uyarlamalı hareket enerjisi çıkarır; kısa aktif kümeleri birleştirir,
performans öncesi hareketi ayırır ve hazırlık/icra/sabitleme ankrajlarını önerir.
Onaylı timeline algılama sırasında sınır kaynağı olarak kullanılmaz ve hiçbir
otomatik öneri timeline, puan veya kesintiyi değiştirmez. Gerçek nihai koşu
`outputs/poomsae_1_zed2i_20260731_trimmed/runs/poomsae1-auto-segmentation-final-v3-20260823/`
altındadır. Yedi hareket kümesinden kayıt öncesi küme ayrıldı, M01-M06 için 6/6
hareket seçildi. Onaylı timeline'a karşı başlangıç MAE `11,33` kare, bitiş MAE
`10,33` kare, faz ankrajı MAE `6,04` kare (`~0,10 s`), en büyük faz hatası `19`
karedir. Bunlar tek kayıttaki iç doğrulama metrikleridir.

HTML, otomatik/referans sınır tablosunu ve her otomatik fixation'a senkron
atlama düğmesini gösterir. Range desteği vermeyen basit HTTP sunucularında ileri
atlamanın sıfıra dönmesi de kökten ele alındı: aynı-origin H.264 inceleme
videoları tarayıcıda seekable Blob olarak hazırlanır. Soğuk açılış denetiminde
M06 otomatik fixation düğmesi iki kamerayı `12,266667 s` konumuna sıfır kaymayla
taşıdı; iki video da `0–12,35 s` seekable, konsol temiz ve yatay taşma sıfırdı.
Algoritma ve kullanım ayrıntısı `docs/AUTOMATIC_SEGMENTATION.md` içindedir.

13 Ağustos 2026 tarihinde 3B açı şeması ve okunabilir dondurma katmanını da
içeren son gerçek tek-komut doğrulama koşusu:
`outputs/poomsae_1_zed2i_20260731_trimmed/runs/poomsae1-correct-3d-guide-freeze-v2-20260813/`.
Koşu 6/18 hareket için dört kaynak-bağlı küçük hata ve `0,4` gözlenen-kapsam
kesintisi üretti; kayıt kısmi olduğu için doğru biçimde `accuracy_score=null`,
`rule_scoring_ready=false` kaldı. Kaynak kararlar değişmez
`decision_evidence_events.json` olaylarına dönüştürüldü. İki kamera yan yana,
kaynak 741/741 kareyi ve 60 FPS'i koruyan
`poomsae_scoring_annotated.mp4` üretildi. Üç benzersiz doğrulanmış kesinti
anchor'ına 3'er saniyelik okuma duraklaması eklendi; çıktı 1281 kare ve
21,35 saniye oldu. Hata anında ilgili eklem geometrisi,
3B ölçüm, `%95` aralığı, kaynak sınırı ve kesinti ekranda gösterilir. HTML
inceleme ekranı aynı olaya atlama, `Doğru / Yanlış / Belirsiz` kullanıcı kaydı
ve bu kayıtları ayrı JSON olarak indirme işlevlerini içerir. 2B çizimler yalnız
görsel izdir; puan kararı yeniden hesaplanmaz.

İşaretli videoda aynı karedeki her karar ayrı `[1]`, `[2]` kutusuna ayrılır;
her kutu olması gerekeni, ölçülen kalibre 3B değeri, `%95` aralığı, sınırdan
farkı, kesinti gerekçesini, düzeltme önerisini ve kaynağın güncellik durumunu
açıkça yazar. Perspektifli kamera görüntüsüne 3B `30°` kabul çizgisi çizilmez;
kamera üzerinde yalnız gözlenen 2B ayak izi kalır. Kural, alt kutudaki üstten
3B şemada beyaz duruş yönü, yeşil kabul alanı ve kırmızı ölçülen açıyla
gösterilir. M04 dondurma karesi görsel olarak incelendi; şema etiketleri panel
içinde okunur ve geri sayım bandı görünürdür. Dondurmalı video uzatıldığı için
HTML senkron grubuna alınmaz; HTML yalnız aynı zaman eksenindeki iki ham
kamera ile çalışır.

13 Ağustos 2026 M01-M06 WholeBody kanıt genişletmesi:

- çalışma hedefi terminal ve özet JSON'da `current_recording_m01_m06`, `6/6`
  olarak ayrıldı; bütün Taegeuk 1 kapsamı provenance için ayrıca `6/18`
  tutulur. M07-M18 bu geliştirme aşamasının kapısı değildir;
- 21 noktalı el geometrisinden `fist_closure_ratio`, el bileği hizası ve 68
  yüz noktası/omuz hattından baş-yüz yön ölçümleri HTML'de hareket bazlı
  matriste görünür hale getirildi. Baş/yüz metriği göz küresi takibi diye
  etiketlenmez;
- eşik dışı WholeBody mühendislik ölçüleri aynı değişmez görsel olay zincirine
  mavi `diagnostic_review_candidate`, `deduction_points=null` olarak eklenir.
  Yalnız kaynak-bağlı doğrulanmış küçük hatalar kırmızı ve üç saniye dondurmalı
  kalır;
- gerçek başarı koşusu
  `outputs/poomsae_1_zed2i_20260731_trimmed/runs/poomsae1-m01m06-wholebody-evidence-v8-20260813/`
  altında `67/87` ölçülebilir eşikli metrik, `10` puansız teşhis adayı, `19`
  toplam görsel olay ve `4` kaynak-bağlı küçük hata üretmiştir. Teşhis
  olaylarının hiçbirinde puan yoktur; video yalnız `218,529,637` kaynak
  karelerinde üç kesinti dondurması içerir;
- tam kayıt için kapıyı sonsuza kadar kapatacak iki durum adı hatası düzeltildi:
  kod artık sözleşmedeki `complete_performance` ve `complete` değerlerini
  kullanır. Bu düzeltme mevcut kısmi kaydın `accuracy_score=null` sonucunu
  değiştirmez;
- fixation ölçümleri izole kritik-eklem boşluklarını yalnız aynı ±5 kare
  penceresinde yeterli gerçek örnek varsa sağlam biçimde tamamlar. Gerçek
  kayıttaki 20 ölçülemez eşikli metriğin 19'u bütün ilgili pencerede yetersiz
  kritik eklem kapsamına, biri fiziksel makullük/trajectory kapısına bağlıdır;
  değer uydurulmadığı için `67/87` kapsam bilinçli olarak değişmemiştir. Her
  metrik artık `required_joint_sample_counts`, `missing_required_joints` ve
  ayrıntılı `not_measurable_reason` taşır. Örnekler: M02 sol omuz `0/11`, M03
  sağ omuz/sağ bilek `0/11`, M05 sağ ayak bileği `0/11`.

2 Ağustos 2026 tarihinde ilk kaynak bağlı puanlama katmanı kuruldu:

- WT'nin canlı `CURRENT` listesinde 30 Eylül 2024 tarihli Poomsae Competition
  Rules & Interpretation güncel kural olarak doğrulandı;
- resmî PDF'nin SHA-256 değeri
  `3fc994363544ab2a1717d9d4d805b368e283a122d3f3c52fc444c70b8c206e24`;
- Recognized Poomsae Accuracy `4,0`, küçük hata `-0,1`, büyük hata `-0,3`
  ve yeniden başlatma `-0,6` aktif RulePack'e aktarıldı;
- Kukkiwon'un 2022 tam sıra gösterimi ve 2025 ayrıntılı resmî eğitiminden
  Taegeuk 1 için 18 hareketli, çok fazlı `PoomsaeSpec` taslağı çıkarıldı;
- RulePack, PoomsaeSpec ve MovementTimeline için benzersiz YAML anahtarlı,
  sürümlü ve fail-closed veri sözleşmeleri eklendi;
- MovementTimeline v2, gerçek bir kısmi kaynak kaydını etiket eksikliğinden
  ayırır: `partial_sequence`, gözlenen M01-M06 ve kaynakta bulunmayan M07-M18
  ayrı alanlarda taşınır;
- Accuracy motoru yalnız aktif spec, eksiksiz/çakışmasız zaman çizelgesi,
  gözlenmiş kanıt ve kuralca doğrulanmış olaylarda kesinti uygular; düşük güven,
  tekrar olay veya bilinmeyen faz puan kesmez;
- ZED2i pose dosyası gerçek RGBD koşusuna SHA-256 ile bağlandı ve doğrulandı:
  `a3098284bed5cf83bea7e0d7488fe52d82f97b7d06977219b4c8f5eeccaf5947`;
- iki kamera incelemesi kısa kaydın tam Taegeuk 1 olmadığını, yalnız M01-M06'yı
  içerdiğini gösterdi. Altı hareket, `24` çoklu faz anchor'ıyla etiketlendi;
  başlangıç/geçiş/fixation temas sayfalarının ikinci kamera kontrolü sonrası
  zaman etiketleri `confirmed` oldu; videoda olmayan M07-M18 üretilmedi;
- etiketli iki inceleme videosu `741` kare, `60 FPS`, `1280x720` olarak kaynak
  zaman çizelgesini korudu;
- iki kamera videosunu senkron oynatan, hareket/faz düğmeleriyle aynı zamana
  atlayan; gözlenebilirlik ölçülerini ve ayrı mühendislik denemesini gösteren
  tek dosyalık HTML inceleme ekranı üretildi;
- `24` anchor'ın `22` tanesi `observed`, `2` tanesi `partially_observed`, `0`
  tanesi `not_measurable`; bu yalnız pose gözlenebilirliğidir, teknik doğruluk
  veya kesinti değildir;
- kayıt kısmi olduğu ve teknik küçük/büyük hata toleransları
  kaynaklandırılmadığı için hazırlık raporu bilinçli olarak `6/18`, `blocked`,
  `rule_scoring_ready=false` üretir; ölçüm raporu
  `not_scored_partial_recording`, `accuracy_score=null`, `deductions=[]` kalır;
- M01-M06 için ilk BODY-17 engineering v1 denemesi teknik ayrıntıların büyük
  kısmını kullanmadığı ve ölçülemeyen alanları başlangıç puanında bıraktığı için
  **sayısal olarak geçersizleştirildi**. Motor artık
  `deprecated_body17_screening_no_score`, skor/kesinti alanlarında `null`/boş
  üretir;
- güncel `taegeuk1-wholebody-diagnostics-v2` profil `2.4.0`, 133 noktanın gövde, 6 ayak, 68
  yüz ve iki 21-noktalı el grubunu kullanır. Duruş, ayak yönü, gövde/baş
  rotasyonu, uygulayan taraf, hikite, el-bilek hizası/yumruk kapanması,
  hand-foot simultaneity, fixation jitter, ağırlık aktarımı ve bilek trajectory
  ve teknik dirsek açısı ölçülür. Fixation ölçüleri tek kare yerine ±5 karelik
  sağlam pencere, yüz/el/ayak ise vücut ölçekli geometri makullük kapısı kullanır;
- büyük hata WT'deki “yanlış hareket” semantiği nedeniyle yalnız sayısal pose
  sapmasından çıkarılmaz; otomatik major tespiti bilinçli olarak kapalıdır;
- gerçek kısa kayıt WholeBody koşusunda eşikli `96` metriğin `74` tanesi
  ölçülebildi, `22` tanesi yetersiz veya makul olmayan eklem kanıtı nedeniyle
  ölçülemedi; kapsam `%77,08` ile gerekli `%90` kapısını geçemedi. `13`
  video-inceleme adayı çıktı. Bunlar ceza değildir;
  `accuracy_score=null`, `deductions=[]` ve `numeric_score_enabled=false`
  kalır;
- bu blokaj depth, mocap veya hakem etiketi beklemekten kaynaklanmaz. Açık işler
  kamuya açık olmayan resmî scoring eki, M07-M18'i içeren yeni çekim ve tam
  Poomsae'nin hareket/faz doğrulamasıdır;
- WT kuralının atıf yaptığı ayrı güncel Poomsae Competition Scoring Guidelines
  eki kamuya açık `CURRENT` listede bulunamadı. Buna karşılık 3 Ağustos 2026
  araştırmasında 2014 WT kurallarına ekli resmî tarihsel 43 sayfalık sürüm
  doğrulandı ve yerel kaynak arşivine alındı. Masaüstündeki 35 sayfalık kopyanın
  ilk 35 sayfası tam sürümle metinsel olarak eşleşir; tarihsel teknik geometrisi
  aday metriklerde kullanılabilir fakat güncel resmî tolerans sayılamaz. 2022
  çevrim içi yarışmaya özgü Poomsae Deduction belgesi güncel ek yerine
  kullanılmadı;
- güncel WT olayları, tarihsel teknik geometri, Kukkiwon hareket semantiği,
  ikincil ulusal kılavuzlar, akademik yöntemler ve sensörle ölçülebilirlik
  `docs/TAEGEUK1_ERROR_TAXONOMY.md` içinde kaynak sınıflarıyla ayrıldı.

3 Ağustos 2026 tarihli source-bound Accuracy genişletmesi:

- iki ayağın birbirine göre açısını kullanan eski ayak metriği kaldırıldı;
  arka ayak yönü artık arka ayak bileğinden ön ayak bileğine duruş doğrultusuna
  göre hesaplanır;
- fixation ölçülerine `1.96 * 1.4826 * MAD` ile `%95` oynaklık belirsizliği ve
  örnek sayısı eklendi; profil tabanlarıyla birlikte karar aralığı oluşturulur;
- 2014 tarihsel resmî kılavuzdan ap-seogi/apkubi `30°`, arae-makki `1–2`
  yumruk ve momtong-an-makki `90–120°` kuralları hash/sayfa bağlı ayrı profile
  alındı; profil bunları güncel WT eki olarak işaretleyemez;
- `%95` aralığının tamamı sınır dışında değilse kesinti uygulanmaz. Sayısal
  geometri yalnız `minor=-0,1`; `major=-0,3` yalnız açık kategorik gözlem,
  restart ise `-0,6` üretebilir. Hareket+hata birimiyle tekilleştirme yapılır;
- güncel koruma bandıyla gerçek M01-M06 koşusunda 9 kaynak-bağlı kararın 4'ü
  küçük hata, 1'i aralık içi, 1'i sınır-belirsiz ve 3'ü ölçülemez çıktı.
  Gözlenen kapsam geçici kesinti toplamı `0,4` oldu;
  kategorik gözlem girilmediği için major yoktur. Kısmi kayıtta
  `accuracy_score=null` sözleşmesi korunmuştur;
- başarılı immutable koşu:
  `outputs/poomsae_1_zed2i_20260731_trimmed/runs/poomsae1-source-bound-20260810-221705/`.

10 Ağustos 2026 uzak dal entegrasyonu:

- `origin/main` üzerindeki `5394558` ve `9e2bd42` commitleri fast-forward ile
  alındı;
- WT Article 16 major hata örnekleri RulePack'te kaynak bağlı kategorik örnek
  olarak korundu;
- süre ihlali ve sınır geçme `accuracy` bütçesine değil, ayrı
  `final_score_deductions` alanına bağlandı; RulePack sürümü `1.1.0` oldu;
- uzak committe yinelenen kategorik örnek doğrulama döngüsü kaldırıldı ve yeni
  final kesinti alanı sıkı sözleşmeye eklendi. Entegrasyon öncesi 14 puanlama
  testi başarısızken düzeltme sonrası ilgili testler `35/35` geçti;
- sınır geçmenin tekrar frekansı WT metninde açık olmadığı için otomatik
  runtime uygulaması kapalı, `source_ambiguity` ile provisional tutuldu;
- bütün yerel PDF/TXT kaynakları `output/pdf/scoring_sources/` altında
  merkezileştirildi; Git'te tutulan hash ve kullanım indeksi
  `docs/scoring_sources/README.md` içindedir.

Güncel WholeBody teşhis ve senkron inceleme koşusu:
`outputs/poomsae_1_zed2i_20260731_trimmed/runs/poomsae1-scoring-expand-v2_3-20260803-123907/`.
PoomsaeSpec `0.6.0-draft` bütün M01-M18 hareketlerinde ölçülebilir kriter
taşır. M14/M16 tekme-yumruk bileşiklerinin beş yeni tekme/faz metriği eşiksiz
teşhis olarak tanımlıdır; kısa kayıt bu hareketleri içermediği için mevcut
M01-M06 metrik/kapsam/adet sonuçlarını değiştirmez.
- yeni `SourceIntake` kapısı kullanıcıdan gelen PDF'nin imza, SHA-256, kurum,
  tarih, otorite ve kullanım amacını doğrular; otomatik kural aktivasyonu daima
  kapalıdır. Tarihsel/ikincil bir kaynak güncel sayısal tolerans talep ederse
  fail-closed engel üretir.
Etiketli videolar ve temel evidence/readiness dosyaları immutable önceki koşuda
korunur; güncel manifest bunları SHA-256 ile bağlar.
Kaynak ve ücretli seçenek sicili `docs/SCORING_SOURCE_REGISTER.md` içindedir.

### Tam tek-kamera RGB-D kontrol koşusu

28 Temmuz 2026 tarihinde aynı SVO2'nin 666 karesinin tamamı, mevcut RF-DETR,
ViTPose, 2B stabilizasyon, güvenilirlik filtresi, 3B yumuşatma ve HTML görüntüleyici
modülleri yeniden kullanılarak izole bir çıktı koşusunda işlendi:

- kaynak zaman çizelgesi korundu: `666` kare, `60 FPS`, `11.134046 s`;
- güvenilir son BODY-17 oranı `%95.3983`;
- güvenilir son WHOLEBODY-133 oranı `%96.1245`;
- 2B video, kamera-koordinatlı 3B video, birleşik video, etkileşimli HTML,
  `[666, 133, 3]` JSON ve düz CSV üretildi;
- JSON/CSV çıktılarında `NaN`/`inf` sızıntısı görülmedi; eksikler `null`/boş
  hücre olarak korundu;
- koşu `outputs/zed2i_single_view/runs/zed2i-single-rgbd-20260728-183704/`
  altındadır ve ana `src/`/`config/` hattını değiştirmemiştir.

Bu çıktı sol kamera optik merkezine göre metre cinsinden
`x=sağ, y=ileri, z=yukarı` RGB-D önizlemesidir. Ortak dünya koordinatlı
çok-kamera triangulation sonucu değildir ve puanlama yetkisi taşımaz.

### Otomatik testler

25 Ağustos 2026 temizlik ve bütünleşik doğrulama turundan sonra:

```text
238 passed in 18.27s
```

Komut:

```powershell
.\.venv312\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp outputs\pytest-cleanup-final2-20260825
```

Önceki iki hata kapatıldı: readiness testi artık temiz checkout koşulunu
deterministik olarak sınar; timeline transferi depo dışındaki mutlak yolları
korur. Ek olarak otomatik timeline bire-birliği, otomatik gözlem güveni,
WholeBody→Accuracy/Presentation provenance bağları ve metrik kanıt pencereleri
fail-closed hale getirildi. Yanlışlıkla Git'e alınmış üç eski `.bak` dosyası kaldırıldı.

Gerçek pose üzerinde tek-komut smoke koşusu da tamamlandı:
`outputs/poomsae_1_zed2i_20260731_trimmed/runs/poomsae1-audit-20260823-v2/`.
Koşu 67/87 WholeBody ölçümü, 10 puansız teşhis adayı, 4 kaynak-bağlı küçük
kesinti ve `0.4` gözlenen-kapsam provisional toplamı üretti; tam Accuracy skoru
timeline 6/18 olduğu için yine `null`, rule scoring readiness `false` kaldı.
Smoke sırasında profil session hash'inin mevcut artefakt ve önceki başarılı
run snapshot'ıyla uyuşmadığı görüldü; yanlış `5408...` değeri doğrulanmış
`09ef...` bağ değerine geri alındı.

Kapsam ölçümü (`pytest --cov=src/poomsae_scoring`): `presentation.py` `%97`,
`sequence_alignment.py` `%97`, `wholebody_diagnostics.py` `%84`,
`source_bound_accuracy.py` `%80`.

Önceki sohbetlerdeki `28`, `31`, `47`, `73`, `91`, `128`, `140`, `144`, `147`
ve `151`, `173`, `176`, `201`, `204` sayıları kendi
tarihlerindeki sonuçlardır; güncel test sayısı değildir.

### Son korunan tam AIST koşusu

```text
outputs/aist_test/runs/pose_2026-07-26_15-07-28_797
```

- 9 kamera
- `stride 1`
- 451 ortak senkron kare
- 9 adet 2B overlay video
- 1 adet 3B iskelet video
- İç geometri durumu: `passed`
- Ground-truth değerlendirmesi: yapılmadı
- `scoring_ready`: `false`
- BODY-17 geçerli oranı: `1.0`
- Ortalama reprojection: `6.364 px`
- Cross-view hedefi: `814`
- Görüntü heatmap'iyle kabul edilen düzeltme: `51`
- Geometriden reddedilen aykırı gözlem: `763`
- Feedback medyan reprojection: `5.314 -> 5.275 px`
- Feedback P95 reprojection: `9.089 -> 8.953 px`
- Global BODY-17 optimization: uygulandı, fallback kullanılmadı

Tam koşudaki kamera sağlığı:

- `c01`, `c09`: `healthy`
- `c02`–`c08`: farklı yoğunluklarda `localized_2d_joint_detection`
- c05 kamera ağırlığı: yaklaşık `0.811`
- c05 medyan konsensüs hatası: yaklaşık `5.40 px`

c05 senkron sorunu çözülmüş olsa da tam koşuda bütün lokal 2B eklem
aykırılarının bittiği iddia edilmemelidir.

### Tarihsel RGB-only dış ground-truth benchmark — ZED RGBD'ye uygulanmaz

27 Temmuz 2026 tarihinde yeni 3B koduyla yeniden çalıştırılan MADS Kata F2
koşusu:

- 3 resmî kalibre kamera
- 300 inference örneği, stride 2
- Global MPJPE: `90.409 mm`
- P95: `162.504 mm`
- Açı MAE: `13.426°`
- Geçerli eklem oranı: `%95.89`
- Sonuç: ground-truth kalite kapısı başarısız
- Koşu:
  `outputs/mads_kata_f2/runs/mads-kata-f2-rescore-20260727-201327`
- Yetkilendirme: `decision=denied`, `scoring_ready=false`
- İç geometri: geçti; yetkilendirme dosya bütünlüğü: geçti
- Üretim modeli: temel ViTPose-Huge; deneysel MADS adapter'ları reddedildi

Bu metrikler yalnız
`mads-kata-f2-rescore-20260727-201327` koşusuna bağlıdır. Güncel iki ZED 2i
RGBD koşusu bu değeri okumaz, devralmaz veya kendi dış doğruluğu olarak
raporlamaz. ZED koşusunun dış doğruluk durumu başarısız bir MADS metriği değil,
`not_evaluated_for_this_run` değeridir.

Sonuç önceki ölçümle aynıdır. Nedeni MADS'in yalnız üç kameralı olması nedeniyle
en az dört destekleyici kamera isteyen cross-view düzeltmenin aday üretememesi
ve MADS session yapılandırmasında global BODY-17 optimizasyonunun kapalı
olmasıdır. Bu özellikler MADS'e güvenli ve ölçülebilir bir profil eklenmeden
iyileşme iddia edilemez.

### Puanlama yetkilendirmesi

- `evaluate_ground_truth_3d.py` tahmin, ground truth, doğrulama config'i, kare
  eşleşmeleri, validation manifesti ve iç-geometri raporunu SHA-256 ile bağlar.
- `analyze_pose_for_scoring.py` yalnız skorsuz kalite, biomekanik ve segment
  kanıtları üretir; puanlama yetkilendirmesi uygulamaz.
- Herhangi bir bağlı dosya değişirse yetki kapanır.
- `scoring_ready`, dış 3B doğruluğu geçen geçici puanlama altyapısı içindir.
  `official_scoring_ready`, gerçek poomsae kural ve hakem doğrulaması için ayrı
  tutulur ve şu anda `false` değerindedir.

## Çalıştırma

İkincil AIST çok-kamera validation testi:

```powershell
.\scripts\run_full_aist_pose.ps1
```

Bu komut benzersiz tarihli run klasörü oluşturur ve `stride 1` kullanır.

Hızlı preview:

```powershell
.\.venv312\Scripts\python.exe scripts\run_vitpose_multiview_3d.py `
  --session data\aist_test\session.yaml `
  --stride 10 `
  --run-id preview-<benzersiz-id>
```

`--max-frames` yalnız bilinçli kısa preview için kullanılmalıdır. `stride`,
çıktı video süresini değiştirmemelidir.

## Temel çıktı dosyaları

```text
outputs/<session_id>/runs/<run_id>/
  videos/<camera>_vitpose_2d_overlay.mp4
  videos/vitpose_skeleton_3d_world.mp4
  viewer/pose3d_viewer.html
  json/run_quality_report.json
  json/camera_health_report.json
  json/crossview_2d_feedback_report.json
  json/global_pose_optimization_report.json
  csv/vitpose_keypoints_2d_prefeedback_flat.csv
  csv/vitpose_keypoints_2d_geometry_flat.csv
  csv/vitpose_keypoints_2d_feedback_provenance.csv
  csv/vitpose_keypoints_3d_world_triangulated_flat.csv
  csv/vitpose_keypoints_3d_world_global_optimized_flat.csv
  csv/vitpose_keypoints_3d_provenance.csv
  csv/vitpose_keypoints_3d_world_flat.csv
```

## Bilinen eksikler ve doğru öncelik

1. Gerçek ZED2i videosunun 18 Taegeuk 1 hareket aralığını ve çoklu faz
   anchor'larını etiketleyip mevcut segment adaylarıyla eşlemek.
2. Gerçek Taekwondo BODY-17 görüntülerini etiketleyip yüksek çözünürlüklü
   alan-özel pose modeli eğitmek.
3. El, yüz ve ayak için güvenli cross-view düzeltme/optimizasyonu geliştirmek.
4. c02–c08 lokal 2B algılama aykırılarını kamera, eklem ve hareket fazı
   bazında azaltmak.
5. Gerçek poomsae kameralarında senkron checkerboard kalibrasyonu, drift ve
   uzun süreli stres testi yapmak.
6. Çok sporculu görüntülerde kamera-arası kimlik eşlemeyi doğrulamak.
7. Açık mühendislik toleranslarını kontrollü örneklerle sürümlemek; hakem/koç
   verisi ileride oluşursa yalnız opsiyonel judge-calibration katmanında
   kullanmak.
8. İmkân oluşursa ZED RGBD mimarisini bağımsız mocap/ölçüm ground truth üzerinde
   ayrıca benchmark etmek; bu opsiyonel dış doğrulamayı provisional akışın
   çalışma önkoşulu yapmamak.
9. Motion tabanlı otomatik segment tespiti eklemek.
   `build_automatic_movement_timeline()` şu an dışarıdan pre-detected segment
   alır; fixation/hareket enerjisinden segment sınırı çıkaran katman ayrı bir
   problemdir ve henüz yoktur.
10. M07-M18 için 2014 kılavuzdan ek sayısal geometri çıkarmak (örneğin
    momtong-jireugi solar-plexus hedef yüksekliği). Her yeni kural yine
    hash/sayfa bağlı ve `%95` belirsizlik kapılı olmalıdır.

## Tarihsel belgeler

- `sohbet1.md`–`sohbet6.md`: oturum özetleri, kalıcı gerçek kaynağı değildir.
- `tk3d_architecture_deep_dive.md`: erken mimariyi anlatır; MMPose, basit DLT
  ve eski scoring örnekleri güncel hattı temsil etmeyebilir.
- `README.md`: kurulum ve kullanıcı komutları için ana belgedir; değişken
  benchmark/test durumu için bu dosya tercih edilmelidir.
