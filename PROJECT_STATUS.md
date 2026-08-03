# TK3D Güncel Proje Durumu

Son doğrulama tarihi: **3 Ağustos 2026**
Doğrulanan çalışma ağacı: **temel `ad3f113` + puanlama altyapısı yerel değişiklikleri**
Dal: **`main`**

Bu dosya değişken proje durumunun tek kısa kaynağıdır. Yeni oturumlarda geçmiş
`sohbet*.md` dosyalarını baştan okumak yerine önce burası okunmalıdır.

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
  yazar ve dış ground-truth beklemeden `provisional_not_official` analiz çalışır.
- `official_scoring_ready:false` kalır; dış doğruluk yokluğu provisional akışı
  durdurmaz fakat resmî doğruluk/hakem puanı iddiasına izin vermez.

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
- dış ground-truth beklenmeden `provisional_scoring_ready=true`; normal analiz
  `71,7178` değerli `provisional_not_official` skor üretti;
- hareket eşiğinin sürekli harekette aşırı yükselmesi düzeltildi; ikinci koşuda
  durum `ready_for_scoring_infrastructure`, hazır kare oranı `1,0` oldu;
- `official_scoring_ready=false`.

Son yeniden üretilen nihai koşu:
`outputs/poomsae_1_zed2i_20260731_trimmed/runs/poomsae1-zed2i-rgbd-gated-ultra-rerun-20260802/`.
Koşu 741/741 kare, stride 1 ve aynı ultra profille tamamlandı; üç video da
`60 FPS`, `12,35 s`, `1280x720` çıktı. İç geometri ve depth kapısı geçti,
`7.669` BODY-17 depth fusion noktası kullanıldı. Normal provisional analiz
`71,7178/100`, `ready_for_scoring_infrastructure`, hazır kare oranı `1,0` ve
`15` hareket segment adayı üretti. Toplam 13 JSON ve 20 CSV doğrulandı;
`NaN`/`inf` sızıntısı ve tarihsel MADS referansı bulunmadı.
IMU bu koşuda kamera yerçekimi/yönelim kalibrasyonunda kullanıldı; sporcu
pozu için kare bazlı IMU düzeltmesi uygulanmadı. ZED depth/confidence kendi
sensör zincirimizin iç kanıtıdır; dış ground-truth değildir. Depth'in gerçek 3B
doğruluğu artırdığı iddiası için bu koşuyla uyumlu bağımsız mocap/ölçüm referansı
hâlâ gerekir.

### Taegeuk 1 Accuracy kural motoru

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
- gerçek M01-M06 koşusunda 9 kaynak-bağlı kararın 5'i küçük hata, 1'i aralık
  içi, 3'ü ölçülemez çıktı. Gözlenen kapsam geçici kesinti toplamı `0,5` oldu;
  kategorik gözlem girilmediği için major yoktur. Kısmi kayıtta
  `accuracy_score=null` sözleşmesi korunmuştur;
- başarılı immutable koşu:
  `outputs/poomsae_1_zed2i_20260731_trimmed/runs/poomsae1-source-bound-20260803-151214/`.

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

3 Ağustos 2026 tarihinde Taegeuk 1 Accuracy sözleşmeleri, kısa kayıt
hareket/faz overlay'i, puansız ölçüm kanıtı, geçersizleştirilmiş BODY-17 denemesi
ve WholeBody-133 teşhis motoru eklendikten sonra mevcut çalışma ağacında:

```text
183 passed in 36.29s
```

Komut:

```powershell
.\.venv312\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp outputs\pytest-scoring-engine-v2_1-full-20260803
```

Önceki sohbetlerdeki `28`, `31`, `47`, `73`, `91`, `128`, `140`, `144`, `147`
ve `151`, `173`, `176` sayıları kendi
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
- `analyze_pose_for_scoring.py` normal modda yalnız bu sidecar'ı doğrular;
  tahmin JSON'undaki bağımsız `scoring_ready` değerini güven kaynağı saymaz.
- Herhangi bir bağlı dosya değişirse yetki kapanır.
- `scoring_ready`, dış 3B doğruluğu geçen geçici puanlama altyapısı içindir.
  `official_scoring_ready`, gerçek poomsae kural ve hakem doğrulaması için ayrı
  tutulur ve şu anda `false` değerindedir.

## Çalıştırma

En güçlü yerel AIST testi:

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

## Tarihsel belgeler

- `sohbet1.md`–`sohbet6.md`: oturum özetleri, kalıcı gerçek kaynağı değildir.
- `tk3d_architecture_deep_dive.md`: erken mimariyi anlatır; MMPose, basit DLT
  ve eski scoring örnekleri güncel hattı temsil etmeyebilir.
- `README.md`: kurulum ve kullanıcı komutları için ana belgedir; değişken
  benchmark/test durumu için bu dosya tercih edilmelidir.
