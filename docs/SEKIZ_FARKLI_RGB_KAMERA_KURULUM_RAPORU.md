# TK3D'de Sekiz Farklı RGB Kamera Kullanım Raporu

Rapor tarihi: **24 Eylül 2026**

Revizyon: **2 — kod, komut ve ölçüm varsayımları yeniden denetlendi**

İncelenen dal/commit: **`main` / `2f95cb1`**

Hedef: ZED 2i yerine marka, model, lens veya çözünürlüğü birbirinden farklı
olabilen sekiz standart RGB kamerayla güvenilir, tekrarlanabilir ve TK3D'ye
uygun çok-kameralı kayıt üretmek.

Bu belge bir kurulum ve geliştirme planıdır. Kamera modelleri, salon ölçüleri,
bütçe ve hedef ölçüm hatası henüz verilmedi. Önerilen ayarlar başlangıç
denemesidir; sekiz gerçek kamerada doğrulanmış donanım listesi değildir.
Repository'deki dosya işleme hattı incelendi; sekiz cihazı yöneten hazır bir
canlı kayıt uygulaması bulunmuyor. Örnek YAML ve komutların önkoşulları aşağıda
açıkça belirtilmiştir.

## 1. Kısa sonuç

Bu değişim mümkündür, ancak yalnızca “sekiz videoyu aynı klasöre koymak” kadar
basit değildir. TK3D'nin RGB tabanlı 2B pose, robust triangulation, BODY-17
global optimizasyon, WholeBody-133 çıktı ve Poomsae analiz katmanları sekiz
kamera kullanabilecek yapıdadır. ZED stereo depth zorunlu değildir; ZED depth
kaynağı yoksa kod RGB-only sonuçla devam eder.

Bugünkü repository'de bu geçiş için dört esas konu vardır:

1. **Tüketici kameralarının gerçek çekim anları senkron değildir.** Sekiz
   cihazda “kayıt” düğmesine aynı anda basmak, sensör pozlamalarının aynı anda
   başladığı anlamına gelmez. Mevcut TK3D senkron modeli kamera başına sabit
   kare/saniye ofsetini destekler; saat drift'i, düşen kare ve zaman içinde
   değişen ofseti modellemez.
2. **Mevcut extrinsic kalibrasyon bütün kameraların aynı checkerboard'u aynı
   anda görmesini ister.** Çevresel/360 derece sekiz kamera düzeninde karşılıklı
   kameraların aynı düzlemsel tahtayı görmesi genellikle mümkün değildir.
   Bunun için bağlı-kamera grafiği + bundle adjustment veya bilinen geometrili
   çok-yüzlü bir kalibrasyon hedefi eklenmelidir.
3. **Dünya zemini ve dikey yön ayrıca belirlenmelidir.** Mevcut checkerboard
   komutu referans kameranın eksenlerini dönüştürür; kameranın aşağı eğimini
   veya yerden yüksekliğini ölçerek zemin koordinatına geçmez.
4. **Poomsae video sunumu hazırlanmış ortak timeline bekler.** 3B motorunun
   farklı başlangıç/FPS'leri eşleyebilmesi, inceleme videosunun ham dosyaları
   aynı şekilde eşlediği anlamına gelmez. Yeni kayıt için profil, hareket/faz
   etiketleri ve görüntü zaman eşlemesi ayrıca kurulmalıdır.

Bu nedenle önerilen yol şudur:

- Yerel kayıt yapabilen kameralar kendi kartına/dosyasına sabit ayarlarla
  kaydeder. Webcam veya yalnız HDMI/IP çıkışı olan modeller için bilgisayar
  ve uygun kayıt arayüzü ayrıca gerekir; her kamera tipinin SD kartı yoktur.
- Kayıt başlangıcı ile pozlama senkronu birbirinden ayrılır. Tüm kameralar
  erken başlatılır; ortak görülen optik senkron işareti başlangıçta ve sonda
  kaydedilir.
- İlk doğrulama 2, sonra 4, sonra aynı kaydın 8 kamera alt kümesiyle yapılır.
  Böylece sekiz kamerayı yeniden çekmeden sorun hangi aşamada oluşuyor görülür.
- Üretim benzeri 8-kamera koşusundan önce kalibrasyon yolu genişletilir.
  Bugünkü checkerboard komutu çevresel sekizli düzen için doğrudan yeterli
  kabul edilmemelidir.

Elde yerel kayıt yapabilen kameralar varsa ek donanımı sınırlı tutan başlangıç,
ortak LED işareti + kısa çekimler + offline hizalamadır. Bu yöntem hızlı
hareket için yeterli zaman doğruluğunu garanti etmez. Ölçüm hedefi milisaniye
düzeyinde pozlama uyumu gerektiriyorsa ortak hardware trigger veya PTP ile
gerçekten senkron pozlama sunan kameralar tercih edilmelidir. Global shutter
satır-zamanı sorununu ayrıca azaltır; tek başına kameraları senkronlamaz. Basler'ın resmi
belgeleri, çoklu çekimin PTP tabanlı periyodik sinyal veya hardware trigger ile
senkronlanmasını açıkça tarif eder: [Periodic Signal](https://docs.baslerweb.com/periodic-signal)
ve [Image Acquisition and I/O Control](https://docs.baslerweb.com/image-acquisition-and-io-control).

## 2. Mevcut TK3D sisteminde gerçekten ne var?

### 2.1 Yeniden kullanılabilen parçalar

| Parça | Sekiz farklı RGB kamera durumu | Kanıt / sınır |
| --- | --- | --- |
| Session YAML | Uygun | Kamera listesi sabit sayıya bağlı değil; kimlik, video yolu ve ofset okunuyor. |
| Video okuma | Uygun | Her kamera OpenCV `VideoCapture` ile ayrı açılıyor. |
| Farklı FPS | Kısmen uygun | Ortak FPS, istenen FPS ile en düşük kamera FPS'inin küçüğüdür; per-frame PTS veya bilinmeyen saat kayması okunmuyor. |
| Farklı çözünürlük | Uygun, koşullu | Her kameranın kalibrasyon çözünürlüğü kendi video çözünürlüğüyle birebir aynı olmalı. |
| RF-DETR + ByteTrack + ViTPose | Uygun | Kamera başına aynı model çalışıyor; marka bağımlılığı yok. |
| WholeBody-133 | Uygun | Ana çıktı `[T, 133, 3]`; BODY-17 işlemleri bu sözleşmeyi küçültmüyor. |
| Robust N-view triangulation | Uygun | İki veya daha fazla görüntüyle çalışıyor, aykırı kamera gözlemlerini reddedebiliyor. |
| Cross-view 2B ikinci geçiş | Sekiz kamerada çalışabilir | BODY-17 için hedef dışındaki en az dört geçerli view gerekir; kamera sayısı tek başına yetmez. |
| Global BODY-17 optimizer | Uygun | RGB reprojection, kemik ve zaman kısıtlarıyla çalışıyor; ZED'e mecbur değil. |
| ZED depth fusion | Opsiyonel | SVO/depth kaynağı yoksa sonuç `unavailable_no_zed_depth_sources` olur ve RGB verisi korunur. |
| Poomsae analizi | Yeni kayıtla yeniden bağlanmalı | Ölçüm motoru 133 noktalı artifact'i tüketir; aktif profil eski ZED session/pose/timeline hash'lerine bağlıdır. Video sunumu ayrıca eş zaman çizelgesi ister. |
| Run/provenance | Uygun | Her koşu immutable `outputs/<session>/runs/<run>/` dizinine yazılır. |

### 2.2 Sekiz kamerada özellikle bilinmesi gereken kod davranışları

- `src/multiview_sync.py` zaman hesabı
  `t_common = yerel_kare / kamera_FPS + frame_offset / kamera_FPS + time_offset_sec`
  şeklindedir. Bilinen sabit 59.94/60 FPS farkını dikkate alır; aynı yerel
  indeksi zorla eşlemez. Dosya başlığındaki FPS'in temsil etmediği gerçek saat
  kaymasını, VFR'yi veya aradaki kayıp kareyi ise çözmez.
- Session `fps: 60` olsa bile bir kamera 30 FPS ise kanonik motor ortak
  timeline'ı 30 FPS'e indirir. Hızlı kameranın bazı kaynak kareleri eşlemede
  kullanılmaz; ham dosyalar korunur. Yalnız bütün kameraların örtüşen bölümü
  işlenir. Kayıt başı/sonu farkından kaybedilen süre raporlanmalıdır.
- `sync.method` açıklayıcı metindir. `optical_pulse_fixed_offset` yazılması
  LED dedektörü çalıştırmaz; ofsetleri kullanıcı/hazırlama aracı ölçmelidir.
- `src/triangulation.py`, geçerli gözlem sayısı üçten fazlaysa çoğunluk inlier
  ister. Örneğin sekiz kameranın sekizinde de bir eklem tespit edilmişse en az
  beşinin geometrik olarak uyuşması gerekir. Bu güvenlidir fakat çok farklı
  görüntü kalitesi olan kameralarda valid oranı düşürebilir; eşik pilot verisi
  olmadan gevşetilmemelidir.
- ViTPose `predict_many` bugün gerçek batch inference değil, kamera karelerini
  sırayla işler. Sekiz kamera ilk geçişte iki kameraya göre dört kat kamera
  çağrısı getirir; toplam süre, RAM/VRAM ve ikinci geçiş maliyeti için bu oran
  benchmark değildir. Mevcut kullanım dosya üzerinden offline'dır.
- Aktif `config/model_config.yaml` içinde ZED depth açık olsa da kaynak yokken
  güvenli RGB fallback vardır. Yine de sekiz-RGB profilinde niyeti açık etmek
  için ayrı bir config'te `zed_depth_fusion.enabled: false` kullanılması daha
  temizdir; kanonik ZED config'i yerinde değiştirilmemelidir.
- Üretim koşusu yalnız `outputs/<session_id>/calibration/cameras.json` içinde
  bütün seçilen kameralar bulunan ve kabul edilen bir calibration mode taşıyan
  kalibrasyonu kullanır. Yaklaşık iki-kamera kalibrasyonu sekiz kamera için bir
  kaçış yolu değildir.
- Kalibrasyon JSON'undaki `production_ready` veya izin verilen mode etiketi,
  bağımsız doğruluk onayı değildir. Checkerboard komutu bu etiketi ortak
  çözüm elde etmesine göre verir; `1.5 px` RMS aşımı yalnız uyarıdır.
- Piksel eşikleri kamera başına normalize edilmiyor. 720p ve 4K'da aynı
  `25 px` sınırı farklı açısal hataya karşılık gelir. Bu yüzden karışık
  çözünürlükler ayrıca kamera bazında değerlendirilmelidir.

### 2.3 Bugünkü kalite kapısı neyi kanıtlar?

Aktif kodda ana iç geometri kapısı yaklaşık olarak şunları ister:

- üretim olarak kabul edilen kalibrasyon;
- ortalama BODY-17 geçerli oranı en az `0.90`;
- ortalama reprojection hatası `25 px` sınırını aşmamalı;
- artifact ve provenance bağları geçerli olmalı.

Bu kapının geçmesi, mutlak 3B doğruluğu kanıtlamaz. Sekiz farklı kamera için
ayrıca holdout kalibrasyon hedefi, bilinen uzunluk/3B wand veya mümkünse mocap
ile dış doğrulama gerekir.

`mean_body17_valid_ratio`, nihai BODY-17 koordinatlarının sonlu olmasını sayar;
geçici olarak tamamlanan noktalarla doğrudan gözlenen noktalar aynı şey
değildir. `observed_body17_ratio`, `temporally_recovered_body17_ratio`,
recovery/provenance, BODY-17 dışı
yüz/el/ayak geçerliliği, reprojection median/p95, kullanılan view sayısı,
temporal/angle jitter ve dış referans varsa MPJPE/P95 ayrı okunmalıdır. Bugünkü
tek bir `passed` değeri bu ölçütlerin tümünü bağımsız olarak doğrulamaz.

## 3. Kamera seçimi: “farklı kamera” ne kadar farklı olabilir?

Farklı marka/model kullanılabilir. Aşağıdaki özelliklerin ortaklaştırılması
gerekir:

| Özellik | Başlangıç tercihi / kontrol | Neden |
| --- | --- | --- |
| Kare hızı | Ortak sabit 50 veya 60 FPS; 30 FPS daha sınırlı pilot | FPS, ışık ve cihazların ortak moduna göre seçilir; 59.94 ile 60 ayırt edilir. |
| Pozlama | Manuel, sabit; hızlı harekette yaklaşık 1/500 s veya daha kısa hedef | Motion blur 2B eklem yerini bozar. Işık yetersizse 1/1000 s görüntüyü karartabilir. |
| Odak | Manuel/sabit | Autofocus intrinsics'i ve keskinliği çekim içinde değiştirebilir. |
| Beyaz ayarı | Sabit | Görsel tutarlılık ve model girdisi için. |
| Zoom | Sabit, kilitli | Zoom değişirse kalibrasyon geçersiz olur. |
| Stabilizasyon | Optik/elektronik/dijital kapalı | Kare kırpma ve sanal kamera hareketi projection modelini bozar. |
| HDR / otomatik yeniden kadraj | Kapalı | Kareler arası geometri ve pozlama davranışını değiştirebilir. |
| Shutter türü | Hızlı hareket için global shutter veya ölçülmüş kısa sensör tarama süresi | Kısa pozlama bulanıklığı azaltır; rolling-shutter satır eğilmesini ortadan kaldırmaz. |
| Kayıt biçimi | CFR, frame-accurate çözülebilen, mümkünse intraframe veya yüksek kaliteli codec | VFR ve agresif interframe sıkıştırma zaman/ayrıntı kaybı yaratır. |
| Güç | Sürekli adaptör veya doğrulanmış pil süresi | Bir kameranın durması bütün ortak timeline'ı kısaltır. |
| Termal davranış | En az gerçek çekim süresi + %50 stres testi | Telefon/action cam uzun 4K60 kayıtta kapanabilir. |

Farklı lens ve çözünürlük prensipte kullanılabilir. Ancak her kamera
kendi gerçek kayıt çözünürlüğü, lensi, zoom'u ve focus ayarıyla kalibre
edilmelidir. Kalibrasyondan sonra kamera/lens oynatılırsa extrinsics; zoom,
focus, çözünürlük, stabilizasyon veya kırpma değişirse intrinsics de yeniden
alınmalıdır. Fisheye/action-camera ultra-wide modu için ayrıca doğru lens modeli
gerekir: mevcut triangulation `cv2.undistortPoints` pinhole/distortion yolunu
kullanır. Fisheye katsayıları aynı alana doğrudan yazılamaz; önce doğrulanmış
rectification ve buna uygun yeni K/çözünürlük ya da açık fisheye desteği gerekir.
[OpenCV fisheye modeli](https://docs.opencv.org/4.x/db/d58/group__calib3d__fisheye.html).

Rolling shutter'da görüntünün farklı satırları farklı zamanlarda pozlanır.
1/1000 saniyeye geçmek bunu 1/1000 saniyelik tam-kare çekime dönüştürmez.
Sensörün tarama süresi ve ilgili video modu ayrı ölçülmelidir.
[Basler shutter açıklaması](https://docs.baslerweb.com/electronic-shutter-types).

Başlangıç profili olarak **1080p50/60**, cihaz ve ışık uyumuyla seçilebilir.
Şebeke/armatür kaynaklı titreşim varsa 50 Hz ortamında 50 FPS denenebilir;
1/500 gibi kısa pozlamada bu seçim tek başına banding'i çözmez. Işık testi
gerekir. 30 FPS, yavaş hareket/kurulum pilotu için değerlendirilebilir; hızlı
tekme doğruluğunu ayrıca sınırlar.

4K, uzak el/ayak ayrıntısını iyileştirebilir; disk, decode ve bellek yükünü
büyütür. ViTPose crop girdisi mevcut config'te sabit `256 × 192` olduğundan
4K'ya çıkış ana pose ağının işini doğrudan dört kat yapmaz ve parmak
doğruluğunu garanti etmez. İlk kararda sporcu ve el/ayakların gerçek piksel
boyutu ile model çıktısı birlikte incelenmelidir.

## 4. Fiziksel kamera yerleşimi

Sekiz kamerayı sadece eşit aralıkla koymak yeterli değildir. Amaç, her önemli
eklemin aynı anda en az iki, tercihen üç-dört kaliteli ve yeterli açı farkı olan
kamera tarafından görülmesidir.

Önerilen başlangıç düzeni:

```text
                     c01 (ön)
               c08               c02

          c07                           c03
                     SPORCU
          c06                           c04

                     c05 (arka)
```

- Kameraları sağlam tripodlara sabitleyin; orta kolon ve gevşek video kafası
  kullanmayın.
- Bütün kameralar çalışma hacminin tamamını, uzanmış el ve tekme ayağı dahil,
  kenarlarda en az yaklaşık %15–20 payla görmelidir.
- Tam karşılıklı çiftler tek başına yeterli değildir. Komşu kameraların görüş
  alanları merkezde geniş biçimde örtüşmelidir; bu hem pose hem kalibrasyon
  grafiği için gerekir.
- Hepsini aynı yüksekliğe koymak kolaydır fakat bazı eklem kapanmalarını ortak
  hale getirir. Pilot düzeninde dört kamera gövde/kalça yüksekliğine, kalanlar
  biraz daha yüksek/alçak konabilir; aşırı üstten veya alttan bakıştan kaçının.
- Görüş çizgisinde ayna, parlak zemin, titreşen LED aydınlatma ve güçlü arka
  ışık olmamalıdır.
- Zeminde tripod ayakları ve optik merkez yaklaşık konumları bantla
  işaretlenmelidir. Bir tripod oynarsa çekime devam etmek yerine extrinsic
  kalibrasyon yenilenmelidir.
- Kamera kimlikleri fiziksel etikete dönüşmelidir: `c01` etiketi hem tripodda,
  hem hafıza kartında, hem dosya adında bulunmalıdır.

Kesin mesafe lens görüş açısına ve salon boyutuna bağlıdır. Sabit “4 metre” gibi
tek bir sayı bütün farklı kameralar için doğru değildir. Mesafe, çalışma
hacminin tam görünmesi ve sporcunun görüntüde yeterli piksel kaplamasıyla
belirlenmelidir.

## 5. Senkronizasyon: aynı anda kayda başlamak ile aynı anı çekmek farklıdır

### 5.1 Üç senkron seviyesi

| Seviye | Yöntem | TK3D açısından durum |
| --- | --- | --- |
| A — Pozlama senkronu | Uyumlu ortak hardware trigger, sensör genlock'u veya PTP'ye bağlı planlı FrameStart | Gecikme/jitter ölçülür; cihazlar aynı fiziksel anda pozlar. |
| B — Zaman etiketleme | Timecode veya ortak clock'tan gelen görüntü timestamp'i | Kareyi adlandırır/eşlemeye yardım eder; sensörlerin aynı anda pozladığını tek başına kanıtlamaz. |
| C — Free-running tüketici kamera | Her cihaz ayrı clock, optik/audio slate ile offline hizalama | Pilot için mümkün; drift ve dropped-frame ölçülmeden üretim kalitesi denemez. |

“Sekiz kayıt düğmesine aynı anda basmak” sadece dosyaların yaklaşık aynı anda
başlamasını sağlar. Yazılım üzerinden arka arkaya `VideoCapture` açmak da
hardware sync değildir. Önemli olan, her karedeki **pozlama anının** ortak
timeline'daki karşılığıdır.

PTP'nin saatleri eşlemesi ile o saatten eşzamanlı `FrameStart` üretilmesi ayrı
adımlardır. Basler'ın periyodik tetikleme tarifinde ikisi birlikte kurulur.
[PTP](https://docs.baslerweb.com/precision-time-protocol) ve
[Periodic Signal](https://docs.baslerweb.com/periodic-signal).
Genlock adı taşıyan bir bağlantının yalnız monitör çıkışını mı yoksa sensörü
mü kilitlediği de modele göre kontrol edilmelidir; RED bu ayrımı sensör
senkron ayarıyla açıklar.
[RED Sensor Sync](https://docs.red.com/955-0192_v7.3/RANGER_GEMINI_HTML_v7.3/Content/5_Advanced_Menus/1_Settings/Setup/GPIO_Sync.htm).
Rastgele sekiz tüketici kameraya bir tetik kablosu takarak bu özellikler
eklenemez; bağlantı, elektrik seviyesi, exposure gecikmesi ve firmware desteği
her model için doğrulanır. USB/Ethernet/Wi-Fi ile “record” komutu göndermek
pozlama senkronu sağlamaz.

### 5.2 Farklı normal kameralar için önerilen çekim prosedürü

1. Bütün kameraları, ışıkları ve varsa senkron işaret cihazını açın; en az 5–10
   dakika ısınma verin.
2. Her kamerada çözünürlük, FPS, shutter, ISO/gain, white balance, focus, zoom,
   stabilizasyon ve codec ayarını kontrol listesine işleyin.
3. Kameraları sırayla kayda alın. Aynı saniyede başlamaları gerekmez; hepsi
   ortak işaretten önce kayıt yapıyor olmalıdır.
4. Salon merkezinde bütün kameralardan görülen elektronik LED işaretini
   çalıştırın. Tanınabilir bir dizi kullanın; örneğin başlangıç pilotunda
   100 ms açık / 200 ms kapalı / 100 ms açık / 200 ms kapalı / 300 ms açık /
   200 ms kapalı / 100 ms açık. Bu süreler başlangıç örneğidir; her kamerada
   görünürlük doğrulanır. Çok kısa tek flaş, kısa exposure aralarının dışında
   kalıp tamamen kaçabilir. Mümkünse aynı cihaz bir ses işareti de üretsin.
5. 5–10 saniye sonra sporcu çekime başlasın.
6. Her take kısa tutulmalı; ilk pilotta 1–2 dakika uygundur. Bu öneri kamera
   clock drift'ini sınırlamak içindir, performansı kırpmak için değildir.
7. Take sonunda aynı optik diziyi tekrar üretin. Başlangıç ve bitiş ofsetleri
   arasındaki fark olası clock drift'ini gösterir; iki işaret arasındaki kare
   kaybını ayırt etmek için timestamp aralıkları ve mümkünse ara kontrol
   işaretleri de incelenir.
8. Kameraları ancak son işaretten sonra durdurun.
9. Ham dosyaları değiştirmeden kopyalayın; checksum alın; çalışma kopyasını
   ayrı oluşturun.

Telefon ekranındaki beyaz kare, ekran yenileme ve tarama gecikmesi nedeniyle
hassas zaman referansı sayılmamalıdır. Tek kontrolcüye bağlı çevreye bakan LED
yüzeyleri veya her kameranın görebildiği diffüz bir ışık hedefi kullanılabilir.
Bütün LED'lerin aynı anda değiştiği doğrulanmalıdır. Hedef kadrajda net,
parlamayla taşmayan ve sporcunun kapatmadığı bir bölgede olmalıdır. Rolling
shutter'da LED satırı ile eklem satırının zamanı farklı olabilir; bu belirsizlik
ayrıca kaydedilir. Bu LED düzeni kameranın shutter'ını tetiklemez.

Ses, kaba hizalamada yardımcıdır. Mikrofon mesafesi, yankı, otomatik ses
işleme ve cihazın ses-görüntü gecikmesi sonucu etkiler. Yaklaşık 343 m/s ses
hızı varsayımıyla 3 metre yol farkı yaklaşık 8.7 ms'dir; bu hesap bile bir
alkışın kendiliğinden milisaniye doğruluğu sağlamadığını gösterir. İnce ayar
görüntüdeki bağımsız referansla doğrulanmalıdır.

### 5.3 Senkron kabul ölçütü

Önce **sabit gecikme**, **çekim içindeki kayma** ve **hizalama sonrası kalan
pozlama zamanı farkı** ayrı raporlanmalıdır. İlk rapordaki “yarım kare iyidir,
bir kare kabul edilir” yaklaşımı hızlı hareket için genel kabul ölçütü değildir.
İlk pilot kontrolleri:

- başlangıç LED dizisi bütün sekiz kamerada otomatik veya elle aynı sırada
  bulunmalı;
- hizalama sonrası hareket işaretleri ölçülen ofseti desteklemeli; tekrar eden
  Poomsae hareketleriyle yapılan korelasyon tek başına referans sayılmamalı;
- her kamera için baş/ara/son artık zaman hatası ve ölçüm belirsizliği ms
  cinsinden raporlanmalı;
- açıklanamayan düşen/tekrarlanan kare olmamalı;
- bütün hedef performans ortak kayıtta bulunmalı. Baş/son kayıt fazlalıkları
  provenance ile ayrı tutulabilir; performans içindeki kayıp, kırparak kalite
  başarısı gibi gösterilmemeli.

60 FPS'te bir kare 16.67 ms, yarım kare 8.33 ms'dir. Yalnız zaman farkından
doğan konum tutarsızlığı için kaba fizik hesabı `mesafe ≈ hız × zaman`:

| Varsayılan eklem hızı | 8.33 ms fark | 16.67 ms fark |
| --- | ---: | ---: |
| 1 m/s | 8.3 mm | 16.7 mm |
| 5 m/s | 41.7 mm | 83.3 mm |
| 10 m/s | 83.3 mm | 166.7 mm |

Bunlar sporcu üzerinde ölçülmüş hızlar veya doğrudan 3B hata tahmini değildir;
senkron ihtiyacını göstermek için varsayımsal hesaplamadır. Geometrik hata,
görüş açıları ve hareket yönüne de bağlıdır. Örneğin 5 m/s varsayımında zaman
uyuşmazlığına 10 mm bütçe ayırırsak `Δt ≤ 2 ms` gerekir; lens, 2B tespit ve
kalibrasyon hataları bu bütçeye ayrıca eklenir. Dolayısıyla kabul eşiği hedef
ölçüme göre belirlenmeli ve ölçülebilir olmalıdır.

60 FPS kayıtta ilk parlak kareyi elle bulmak genellikle kare düzeyinde
belirsizlik taşır; bundan otomatik olarak `<2 ms` doğruluk çıkmaz. Sabit ofset
kusursuz bilinse bile free-running kameraların exposure fazı farklı olabilir.
Mevcut kod en yakın **gerçek kareyi** seçer; keypoint'i tam ortak pozlama anına
subframe olarak taşımaz. Zaman interpolasyonu ayrı bir geliştirmedir ve hızlı
hareket/occlusion sırasında yine doğrulanmalıdır. Kısa take yalnız drift'i
sınırlar; başlangıç exposure faz farkını ve rolling shutter'ı çözmez.

Mevcut session YAML yalnız sabit `offsets` ve `offsets_sec` kabul eder. Take
sonunda drift varsa şu seçeneklerden biri gerekir:

- daha kısa take ve yeniden senkron;
- donanım/PTP senkrona geçiş;
- per-frame timestamp veya parça-bazlı zaman eşleme özelliğinin TK3D'ye
  eklenmesi.

FFmpeg'in resmi dokümanı `-r`/`fps_mode cfr` işleminin kare düşürüp
çoğaltabileceğini belirtir. Bu yüzden VFR kaydı körlemesine 60 FPS'e çevirip
“senkronlandı” demek doğru değildir. Ham timestamp ve dönüşüm haritası
korunmalıdır: [FFmpeg Documentation](https://ffmpeg.org/ffmpeg.html).

### 5.4 Ofset işareti ve iki ayrı kayıt

`c01` referans olsun. Aynı ışık olayı `c01` videosunda 2.000 s,
`c02` videosunda 3.000 s konumundaysa `c02.offsets_sec = -1.000` olur:
`3.000 + (-1.000) = 2.000`. Eşit 60 FPS'te aynı örnek `c01` kare 120 ve
`c02` kare 180'in eşlenmesidir. Kare ofseti kullanılırsa `-60` yazılır;
aynı düzeltme hem kare hem saniye alanına yazılıp iki kez uygulanmamalıdır.

Kalibrasyon çekimini durdurup sporcu kaydını yeniden başlatınca bu ofset
değişir. Bugünkü kalibrasyon komutu **aynı session'ın sync alanlarını
kalibrasyon videosuna da uygular**. Bu yüzden `session_calibration.yaml`
kalibrasyon olaylarından ölçülen ofsetleri, `session.yaml` performans
kaydının ofsetlerini taşımalıdır. İkisinin kamera kimlikleri ve fiziksel
kurulumu aynı olur. Alternatif olarak tek kesintisiz kayıtta kalibrasyon ve
performans bulunabilir; türetilmiş dosyaların başlangıç eşlemesi yine korunur.

## 6. Sekiz kamerayı nereye ve nasıl kaydetmeli?

Kayıt mimarisi, kamera listesinden sonra seçilir:

| Kamera türü | Kaydın ilk oluştuğu yer | İş akışı |
| --- | --- | --- |
| Telefon / action cam / kartlı fotoğraf-video kamera | Cihaz belleği veya SD kart | Sekizini önceden başlat; işaretle; sonradan orijinal dosyaları kablo/kart okuyucu ile kopyala. |
| USB webcam | Kayıt bilgisayarının diski | Her cihazı kararlı seri/port kimliğiyle açan recorder gerekir. TK3D'de hazır değil. |
| HDMI çıkışlı kamera | Capture card bağlı bilgisayar veya bağımsız recorder | Clean HDMI, gerçek çıkış FPS'i, capture kartının kare kaybı/latansı ve güç sınırı kontrol edilir. |
| IP/GigE kamera | PC diski, NVR veya cihaz kartı | PTP destekli endüstriyel model ile sıradan RTSP kamera aynı sınıf değildir; kodlama ve ağ gecikmesi çekim timestamp'i yerine geçmez. |

Tek bilgisayarda sekiz görüntüyü bir mozaik videoya kaydetmek analiz girdisi
olarak önerilmez: her kamera tam çözünürlükte ayrı dosya ve zaman kaydı
üretmelidir. Önizleme mozaiği buna ek olabilir. Kayıt sırasında ağır pose
inference çalıştırılmaması, veri yakalama yükünün ayrı ölçülmesini sağlar.

### 6.1 Önerilen klasör yapısı

Her fiziksel çekim için yeni bir `session_id` kullanılmalıdır:

```text
C:\Users\WWWW\Desktop\tk3d\data\rgb8_20260924_take01\
├── session.yaml
├── session_calibration.yaml
├── calibration_rgb8.yaml
├── capture_manifest.yaml
├── videos\
│   ├── c01.mp4
│   ├── c02.mp4
│   ├── ...
│   └── c08.mp4
├── calibration\
│   ├── c01_calib.mp4
│   ├── c02_calib.mp4
│   ├── ...
│   └── c08_calib.mp4
├── sync\
│   ├── sync_events.csv
│   └── sync_review.md
└── checksums\
    └── sha256.txt
```

Mevcut `.gitignore` `.mp4`, `.mov`, `.avi`, SVO/SVO2 ve `outputs/` dosyalarını
kapsar; `.mkv`, `.mxf`, `.mts`, ses, görüntü dizisi ve her yeni üretici biçimini
genel olarak kapsamaz. Ham medyayı repository dışında saklayın; `git status`
ile kapsamı denetleyin. Aşağıdaki D: yolu **disk varsa kullanılabilecek örnek**
yoldur; bu makinede mevcut olduğu doğrulanmadı:

```text
D:\TK3D_CAPTURES\rgb8_20260924_take01\raw\c01\...
D:\TK3D_CAPTURES\rgb8_20260924_take01\raw\c02\...
...
D:\TK3D_CAPTURES\rgb8_20260924_take01\raw\c08\...
```

`data/.../videos` altındaki dosyalar TK3D'nin seçilmiş çalışma kopyalarıdır;
ham master dosyaların yerine geçmez. Hiçbir zaman karttan gelen tek kopya
üzerinde trim/transcode yapılmamalıdır.

`capture_manifest.yaml`, `sync_events.csv` ve checksum listesi bu raporun
önerdiği saha kayıtlarıdır; mevcut `load_session` bunları otomatik okuyup
doğrulamaz. `session_calibration.yaml` yalnız kalibrasyon kliplerinin ofsetini
taşır. `session.yaml` inference için seçilen dosyaların ofsetini taşır.
Dosyaları hizalayarak yeni videolar üretildiyse yeni session/manifest ve
`ortak_kare -> kamera_kaynak_karesi, PTS, zaman_farkı` haritası saklanmalıdır.

### 6.2 Dosya adlandırma

Dosya adı en az şunları taşımalıdır:

```text
<session_id>_<camera_id>_<take_id>_<original-sequence>.<ext>
```

Örnek:

```text
rgb8_20260924_take01_c03_take01_0001.mp4
```

TK3D çalışma kopyası daha sonra sabit `c03.mp4` adına bağlanabilir; orijinal ad
ve SHA-256 `capture_manifest.yaml` içinde kalmalıdır.

### 6.3 Capture manifest alanları

Her kamera için şu alanlar kaydedilmelidir:

- `camera_id`, üretici, model, seri numarası;
- lens/focal length veya zoom konumu;
- çözünürlük, beyan edilen ve ölçülen FPS;
- shutter, aperture, ISO/gain, white balance, focus;
- stabilizasyon/HDR durumu;
- codec, pixel format, bitrate, audio var/yok;
- dosya adı, byte boyutu, SHA-256;
- recording start/stop yaklaşık zamanı;
- başlangıç ve bitiş sync pulse kareleri;
- pil/güç tipi, sıcaklık/termal kapanma gözlemi;
- tripod konumu/yüksekliği ve değişiklik notu.

İki ayrı diskte checksum ile doğrulanmış kopya elde edilmeden kartı yeniden
kullanmayın. Dosya kopyalama tamamlandı mesajı yerine dosya boyutu/hash ve
son kareye kadar decode kontrolü kullanın. Bölünmüş kamera kliplerinin sırası
ve bölünme noktalarındaki süreklilik korunmalıdır. Görüntüleri mesajlaşma
uygulamalarından sıkıştırılarak aktarmayın. Sporcu kayıtlarının erişim ve
saklama süresini oturum kaydında belirtin.

### 6.4 Depolama hesabı

Yaklaşık saatlik depolama, `sekiz kamera × bitrate × 3600 / 8` ile bulunur:

| Kamera başı bitrate | Sekiz kamera toplamı | Yaklaşık 1 saat |
| ---: | ---: | ---: |
| 25 Mbit/s | 200 Mbit/s | 90 GB |
| 50 Mbit/s | 400 Mbit/s | 180 GB |
| 100 Mbit/s | 800 Mbit/s | 360 GB |

Tablo ondalık GB kullanır ve yalnız sekiz video için örnek bitrate hesabıdır;
gerçek bitrate VBR'de değişebilir. 50 Mbit/s için kamera başına bir saat
22.5 GB, toplam 180 GB'dir. İkinci kopya, hazırlanmış videolar ve TK3D
çıktıları da yer ister; FFV1 boyutu görüntüye göre değiştiği için önceden sabit
sıkıştırma oranı varsayılmamalıdır.

1080p60 decode edilmiş BGR için `1920 × 1080 × 3 × 60 × 8` hesabı yaklaşık
2.99 GB/s ve 10.75 TB/saat verir. Bu **bellekteki/ham BGR** hacmidir; USB
kameranın mutlaka bu biçimi gönderdiği anlamına gelmez. YUY2, Bayer, MJPEG veya
H.264 taşıma hacimleri farklıdır; arayüz ve disk yükü gerçek formatla ölçülür.

Sekiz USB kameranın tek hub'da yeterli kapasite bulacağı varsayılmamalıdır.
Güçlü hub eklemek upstream veri kapasitesini artırmaz; birden fazla fiziksel
port da aynı host controller'ı paylaşabilir. Windows'un resmi USB
belgesi, isochronous cihazların controller bandwidth'ini ayırdığını ve sonraki
cihazların kaynak bulamayabileceğini açıklar: [USB Bandwidth Allocation](https://learn.microsoft.com/en-us/windows-hardware/drivers/usbcon/usb-bandwidth-allocation).
Merkezi USB kayıt kullanılacaksa fiziksel olarak ayrı host controller'lar,
aktif/güçlü hub'lar, codec ve disk yazma stres testi gerekir.

## 7. Örnek sekiz-kamera session YAML

Aşağıdaki dosya sabit-ofsetli mevcut TK3D şemasına uygundur. **Sıfır ofsetler
ölçüm değildir.** Dosyalar gerçekten ortak başlangıca hazırlanmadıysa veya
kayıtla doğrulanmadıysa sıfır bırakılmamalıdır. FPS burada 60 örneğidir;
50/59.94/30 veya `null` seçimi gerçek dosya bilgisine göre yapılır. Şablon bu
rapor içinde verilmiştir; gerçek sekizli session dosyası henüz oluşturulmadı.

```yaml
session_id: rgb8_20260924_take01
task_name: taegeuk_1_jang
fps: 60.0

cameras:
  - camera_id: c01
    video_path: videos/c01.mp4
    calibration_video_path: calibration/c01_calib.mp4
  - camera_id: c02
    video_path: videos/c02.mp4
    calibration_video_path: calibration/c02_calib.mp4
  - camera_id: c03
    video_path: videos/c03.mp4
    calibration_video_path: calibration/c03_calib.mp4
  - camera_id: c04
    video_path: videos/c04.mp4
    calibration_video_path: calibration/c04_calib.mp4
  - camera_id: c05
    video_path: videos/c05.mp4
    calibration_video_path: calibration/c05_calib.mp4
  - camera_id: c06
    video_path: videos/c06.mp4
    calibration_video_path: calibration/c06_calib.mp4
  - camera_id: c07
    video_path: videos/c07.mp4
    calibration_video_path: calibration/c07_calib.mp4
  - camera_id: c08
    video_path: videos/c08.mp4
    calibration_video_path: calibration/c08_calib.mp4

sync:
  method: optical_pulse_fixed_offset
  offsets:
    c01: 0
    c02: 0
    c03: 0
    c04: 0
    c05: 0
    c06: 0
    c07: 0
    c08: 0
  offsets_sec:
    c01: 0.0
    c02: 0.0
    c03: 0.0
    c04: 0.0
    c05: 0.0
    c06: 0.0
    c07: 0.0
    c08: 0.0
```

Pozitif ofsetin mevcut kodda anlamı şudur: ilgili kameranın yerel kare sıfırı
ortak timeline'da daha sonra oluşur. Ofset işareti elle tahmin edilmemeli;
senkron preview ve hareket korelasyonuyla doğrulanmalıdır.

## 8. Kalibrasyon, adım adım

Kalibrasyon iki ayrı problemi çözer:

- **Intrinsics:** Kameranın kendi focal length/pixel center/distortion modeli.
- **Extrinsics:** Kameranın ortak dünya koordinatındaki konumu ve yönü.

Intrinsics doğru, extrinsics yanlışsa sekiz güzel 2B görüntüden fiziksel olarak
yanlış 3B üretilebilir. Düşük görünen reprojection hatası tek başına bunu her
zaman yakalamaz.

### 8.1 Mevcut checkerboard ayarı

`config/calibration_config.yaml` bugün şunları kullanır:

- pattern: `9 × 6` iç köşe;
- kare boyu: `0.025 m`;
- kamera başına en az 12 geçerli detection;
- her 10 karede bir örnekleme;
- intrinsic RMS uyarı sınırı: `1.5 px`;
- dünya referansı: `cam_front` (sekizli dosyada `c01` yapılmalıdır).

Tahta sert, tamamen düz ve kare boyu kumpasla ölçülmüş olmalıdır. Katlanmış
kağıt veya dalgalı levha metrik ölçeği bozar. `9 × 6`, kare sayısı değil **iç
köşe** sayısıdır: normal checkerboard için `10 × 7` kare gerekir. 25 mm kare
ile desen yalnız `250 × 175 mm` olur; salonun öbür ucundan güvenilir köşe
algılama için küçük kalabilir. Desen boyu gerçek uzaklıkta deneyerek seçilmeli,
baskı %100 yapılmalı ve gerçek kare aralığı ölçülmelidir. Daha büyük basıp
config'te 0.025 bırakmak bütün 3B ölçeği yanlış yapar. OpenCV'nin resmi araçları
checkerboard yanında ChArUco üretimini de destekler: [Create calibration pattern](https://docs.opencv.org/4.8.0/da/d0d/tutorial_camera_calibration_pattern.html).
Yüksek hassasiyetli kalibrasyonda ChArUco'nun standart ArUco'dan daha uygun
olduğu OpenCV tarafından ayrıca belirtilir: [Detection of ChArUco Boards](https://docs.opencv.org/5.0/tutorials/objdetect/charuco_detection/charuco_detection.html).

### 8.2 Intrinsic çekim

Her kamera için, üretim videosuyla tamamen aynı çözünürlük/focus/zoom/stab
ayarında ayrı kalibrasyon videosu çekin:

1. Tahtayı görüntünün merkezi, dört köşesi ve kenarlarında gösterin.
2. Yakın, orta ve daha uzak mesafeler kullanın; tahta görüntünün anlamlı bir
   bölümünü kaplasın.
3. Yalnız cepheden değil, ölçülü eğik açılardan da çekin; çok keskin ve bulanık
   açılar kullanılmamalıdır.
4. Aynı pozun onlarca kopyası yerine görüntü alanını ve açıyı kapsayan çeşitli
   pozlar toplayın.
5. Motion blur olmaması için tahtayı her pozda kısa süre sabit tutun.
6. Kalibrasyon videosundaki çözünürlüğün üretim videosuyla birebir aynı
   olduğunu probe ile doğrulayın.

Başlangıçta kamera başına 20–40 **çeşitli, keskin** görünüm hedeflenebilir;
bu resmî minimum değildir. Videoda art arda 40 aynı poz, 40 farklı açı kadar
bilgi sağlamaz. Mevcut komut intrinsic için her kameranın kendi klibindeki
detection'ları kullanır; extrinsic için bunların ortak zamanlarını ayrıca
arar. Bağımsız zamanlarda çekilmiş sekiz intrinsic klibi tek başına ortak
extrinsic oluşturmaz.

### 8.3 Extrinsic için mevcut komutun sınırı

Bugünkü `calibrate_multiview_cameras` yolu, referans kameradaki her detection
zamanına diğer **bütün** kameraların detection'ını eşler ve yalnız sekizinde de
tahta bulunduğu ortak anları kullanır. Minimum ortak an varsayılan olarak
üçtür. Bu yaklaşım aynı tarafta duran kamera dizisi için çalışabilir; 360 derece
çevresel yerleşimde düz tahta karşılıklı kameralara aynı anda bakamaz.

`--allow-intrinsics-only-fallback` bunu çözmez. O seçenek yalnız teşhis için
`intrinsics_only.json` üretir ve ortak dünya koordinatı garanti etmediği için
üretim 3B'sinde kabul edilmez.

İkinci pratik sınır `frame_stride: 10` değeridir: her kamera yerel 0,10,20…
karelerini tarar. Başlangıçları 3 kare farklı iki 60 FPS videoda, ofset doğru
olsa bile seçilmiş detection zamanları 50 ms ayrık kalabilir. Varsayılan
eşleme toleransı ise 8.33 ms'dir. Bu durumda hiçbir ortak detection
eşleşmeyebilir. İlk kalibrasyon incelemesinde `frame_stride: 1` veya ortak
zamana göre seçilmiş kareler kullanılmalıdır. Toleransı büyütmek, hareketli
tahtanın farklı pozlarını birleştirebileceğinden çözüm sayılmaz. Mevcut kodda
köşe sıralamasının karşılıklı görüşlerde aynı fiziksel noktayı temsil ettiği
ayrıca görsel kontrol edilmelidir.

### 8.4 Sekiz çevresel kamera için önerilen extrinsic çözüm

İki uygulanabilir mimari vardır:

#### Seçenek A — Bilinen geometrili çok-yüzlü hedef

- Salon merkezine rijit, ölçülmüş, birden fazla yüzünde benzersiz ChArUco/AprilTag
  bulunan 3B hedef yerleştirilir.
- Her yüzün köşe koordinatı tek bir hedef/world koordinatında önceden ölçülür.
- Her kamera gördüğü yüzlerden kendi `world -> camera` pozunu çözer.
- Hedef sabitse bütün kameraların aynı yüzü görmesi gerekmez ve zaman senkronu
  extrinsic çözümünün ana şartı olmaktan çıkar.
- Sonuç bütün kameralar ve bütün gözlemlerle bundle adjustment'ta birlikte
  iyileştirilir.

Avantajı, 360 derece düzende bütün kameraları tek ölçülmüş hedef geometrisine
bağlayabilmesidir; sağlamlık ve tekrarlanabilirlik hedefin ölçümüne ve teste bağlıdır.
Dezavantajı, hedef geometrisinin hassas üretilip ölçülmesi ve TK3D'ye yeni
detector/importer eklenmesi gereğidir. Bir kutunun yüzlerine rastgele desen
yapıştırmak yeterli değildir: yüzlerin kalınlık, açı, boyut ve köşe konumları
tek 3B modelde bilinmelidir. Yüzler arası benzersiz marker kimlikleri gerekir.
Intrinsic önceden çözülmeli; mümkünse birden fazla sabit hedef pozu ve
holdout veri kullanılmalıdır.

#### Seçenek B — Komşu kamera çiftleri + pose graph + bundle adjustment

- `c01-c02`, `c02-c03`, ..., `c08-c01` komşu çiftlerinin ortak görüşünde
  ChArUco tahta gösterilir.
- Her eşleşme kamera-kamera dönüşümü üretir.
- Kamera grafiği bağlı olmalı ve `c08-c01` loop closure mutlaka bulunmalıdır.
- Bütün kamera pozları tek grafikte çözülüp bundle adjustment ile birlikte
  iyileştirilir; yalnız dönüşümleri zincirlemek drift biriktirir.

Tahta her ortak pozda 1–2 saniye sabit tutularak başlangıç pilotunda zaman
eşleme hassasiyeti azaltılabilir; farklı durakların aynı hedef pozu sanılmaması
gerekir. Yeterli görünürlük varsa komşu çiftler dışında ek kenarlar da toplanır.
ChArUco tek başına görünmeyen kameraları bağlamaz: ortak gözlem grafiğinin
tamamı bağlı olmalıdır. OpenCV'nin kısmi tahta gözlemlerini desteklemesi, bu
özelliğin TK3D'de mevcut olduğu anlamına gelmez.
[OpenCV ChArUco calibration](https://docs.opencv.org/4.x/da/d13/tutorial_aruco_calibration.html).

İlk prototipte hassas çok-yüzlü hedef üretmek zor ise B seçeneği daha erişilebilir
olabilir. Kalıcı laboratuvar kurulumu için A değerlendirilebilir. İkisi de
mevcut komuta yalnız config ekleyerek etkinleşmez; yazılım ve doğrulama işi
gerektirir.

OpenCV'nin resmi `multicalib` açıklaması da önce kameraları ayrı kalibre edip
sonra extrinsics'i bundle adjustment benzeri optimizasyonla iyileştiren bir
çok-kamera yaklaşımını tarif eder: [cv::multicalib](https://docs.opencv.org/3.4.2/d8/db8/namespacecv_1_1multicalib.html).

### 8.5 Kalibrasyon sonrası bağımsız doğrulama

Kalibrasyonun kendi kullandığı karelerdeki RMS yetmez. Ayrı bir validation
çekimi yapılmalıdır:

- hedef, kalibrasyonda kullanılmayan en az 10–20 konuma getirilmeli;
- kamera başına ve çift başına holdout reprojection dağılımı raporlanmalı;
- median, p95, maksimum, detection sayısı ve kamera grafiği residual'ı
  yazılmalı;
- bilinen iki/üç nokta arası fiziksel mesafe 3B'de kontrol edilmeli;
- zemindeki bilinen noktalarla ölçek ve eksen yönü doğrulanmalı;
- kamera merkezleri 3B çizimde salon düzeniyle görsel olarak karşılaştırılmalı;
- bir kamera çıkarıldığında sonucun aşırı sıçrayıp sıçramadığı test edilmeli.

İlk mühendislik hedefi olarak intrinsic RMS'in mevcut `1.5 px` uyarı sınırının
altında olması, holdout median reprojection'ın tercihen `1 px` civarı/altı ve
p95'in birkaç piksel içinde kalması amaçlanabilir. Bunlar resmî doğruluk
garantisi değil, pilot hedefidir; çözünürlük ve lens distorsiyonuna göre raporla
birlikte yorumlanmalıdır.

### 8.6 Dünya zemini, yön ve import sözleşmesi

Mevcut checkerboard çözümünde referans kamera için `R=I`, `t=0` alınır.
`opencv_reference_to_analysis()` yalnız eksenleri `x, z, -y` sırasına çevirir.
Referans kamera aşağı eğikse sonuçtaki `z` gerçek yerçekimi dikeyi değildir;
orijin de salon zemini yerine referans kameranın merkezidir. Zemine/yatay
düzleme bağlı Poomsae ölçümleri bundan etkilenebilir.

Bu yüzden zeminde metreyle ölçülmüş en az üç doğrusal olmayan noktayı ve
tercihen daha fazla holdout noktayı kullanarak zemin düzlemi, zemin orijini
ve pozitif yatay yön belirlenmelidir. Dikey eksen fiziksel çekülle veya başka
bağımsız referansla kontrol edilir; mevcut pose'tan tahmin edilip bağımsız
doğrulama sayılmaz. Dönüşüm bütün kamera/pose verilerinde tutarlı olmalıdır.
Yeni RGB kalibrasyon yoluna açık zemin hizalaması ve kontrol raporu gerekir;
bugünkü komutta buna özel CLI yoktur.

Bir dış kalibrasyon aracı kullanılacaksa hedef JSON'a aktarımda:

- `image_size` her kameranın gerçek `(width, height)` değeridir;
- `K` o görüntüdeki piksel intrinsics'idir; kullanılan distortion modeli
  TK3D'nin projeksiyon/undistortion yoluyla uyuşmalıdır;
- `R,t` dünya noktasını OpenCV kamera koordinatına götürür:
  `X_camera = R X_world + t`, `P = K [R | t]`;
- kamera merkezi `C = -Rᵀt`'dir; fiziksel konum doğrudan `t` değildir;
- dünya birimi ve `source_to_analysis` 4×4 dönüşümü belgelenir; metre ölçeği,
  handedness, sağ/ileri/yukarı yönler kontrol edilir;
- `schema_version`, `camera_id`, kalibrasyon kaynağı ve dosya hash'i korunur.

Mevcut whitelist yeni dış araç adına otomatik izin vermez. Sırf geçsin diye
`zed_fusion_multiview` veya doğrulanmamış bir production etiketi kullanılmaz;
importer, mode ve sayısal kontrol yolu ayrıca geliştirilir.

## 9. Bugünkü repository ile çalıştırılabilen kontroller

Aşağıdaki komutlar, örnek session dosyası oluşturulduktan ve videolar yerlerine
kopyalandıktan sonra kullanılabilir. Bunlar bu incelemede gerçek sekiz kamera
üzerinde çalıştırılmadı. `probe_videos` yalnız video başlığı/açılabilirliği,
preflight ise dosya/temel bağımlılık varlığını denetler. Hiçbiri LED senkronu,
VFR, sensör frame kaybı, bütün dosyanın decode edilebilmesi veya extrinsic
doğruluğunu kanıtlamaz. Preflight'ın process exit code'u başarısız raporda da
0 olabildiği için aşağıda JSON durumu ayrıca kontrol edilir.

Yardımcı hazırlama komutları ana inference run'ları gibi immutable değildir;
aynı rapor dosyasını yeniden yazabilir. İlk hazırlamada aşağıdaki blok mevcut
rapor üzerine yazmayı engeller. Yeniden denemede önceki raporu tarihli
hazırlama dizininde koruyup yeni çıktı hedefi/session sürümü kullanın.

```powershell
Set-Location 'C:\Users\WWWW\Desktop\tk3d'
$ErrorActionPreference = 'Stop'
$Python = 'C:\Users\WWWW\Desktop\tk3d\.venv312\Scripts\python.exe'
$Session = 'C:\Users\WWWW\Desktop\tk3d\data\rgb8_20260924_take01\session.yaml'
$ProbeReport = 'C:\Users\WWWW\Desktop\tk3d\outputs\rgb8_20260924_take01\json\video_probe_report.json'
$PreflightReport = 'C:\Users\WWWW\Desktop\tk3d\outputs\rgb8_20260924_take01\json\preflight_report.json'
if ((Test-Path -LiteralPath $ProbeReport) -or (Test-Path -LiteralPath $PreflightReport)) { throw 'Hazırlama raporu var; yeni çıktı hedefi kullanın.' }
& $Python 'C:\Users\WWWW\Desktop\tk3d\scripts\inspect_session.py' --session $Session
if ($LASTEXITCODE -ne 0) { throw 'Session okunamadı.' }
& $Python 'C:\Users\WWWW\Desktop\tk3d\scripts\probe_videos.py' --session $Session --output-root 'C:\Users\WWWW\Desktop\tk3d\outputs'
if ($LASTEXITCODE -ne 0) { throw 'Video probe çalışmadı.' }
if (-not (Get-Content -Raw -LiteralPath $ProbeReport | ConvertFrom-Json).all_opened) { throw 'Bütün videolar açılamadı.' }
& $Python 'C:\Users\WWWW\Desktop\tk3d\scripts\preflight_session.py' --session $Session --model-config 'config/model_config.yaml' --output-root 'C:\Users\WWWW\Desktop\tk3d\outputs' --require-videos --require-calibration-videos --require-model-files
if ($LASTEXITCODE -ne 0) { throw 'Preflight çalışmadı.' }
if ((Get-Content -Raw -LiteralPath $PreflightReport | ConvertFrom-Json).status -ne 'passed') { throw 'Preflight başarısız; raporu inceleyin.' }
```

Beklenen rapor yolları:

```text
C:\Users\WWWW\Desktop\tk3d\outputs\rgb8_20260924_take01\json\video_probe_report.json
C:\Users\WWWW\Desktop\tk3d\outputs\rgb8_20260924_take01\json\preflight_report.json
```

Mevcut checkerboard yönteminin fiziksel görüş şartı gerçekten sağlanıyorsa
aşağıdaki **ayrı** config'i
`C:\Users\WWWW\Desktop\tk3d\data\rgb8_20260924_take01\calibration_rgb8.yaml`
olarak hazırlayın. `square_size_m: 0.025` yalnız 25 mm ölçülmüş kare için
doğrudur; gerçek tahta ölçüsünü yazın. `min_valid_frames` ve `min_common_frames`
örnekteki sayı kapılarıdır; görüntü çeşitliliği/holdout kontrolünün yerini tutmaz.

```yaml
checkerboard:
  pattern_size: [9, 6]
  square_size_m: 0.025
  min_valid_frames: 20
  min_common_frames: 8
  frame_stride: 1
  reference_camera_id: c01
calibration:
  flags:
    fix_aspect_ratio: false
    zero_tangent_dist: false
  reprojection_error_warn_px: 1.5
extrinsics:
  world_origin_camera: c01
  solve_from_checkerboard: true
```

`session_calibration.yaml` dosyasını sekizinci bölümdeki şartlarla hazırlayın:
aynı session ID, kamera kimlikleri ve kalibrasyon video yolları; **kalibrasyon
klibinden ölçülen ofsetler**. Performans session'ını yalnız kopyalayıp ofsetleri
doğru varsaymayın. Varsayılan `config/calibration_config.yaml`, `cam_front`
aradığından `c01`–`c08` ile doğrudan kullanılamaz.

Kalibrasyon komutu mevcut `cameras.json` üzerine yazabilir; başarısız denemede
eski JSON da yerinde kalabilir. Bu nedenle aşağıdaki blok var olan sonuçları
korur. Çevresel sekizli düzen için yeni graph/hedef desteği geliştirilmeden
bu komutun görüş şartını sağlayacağını varsaymayın.

```powershell
Set-Location 'C:\Users\WWWW\Desktop\tk3d'
$ErrorActionPreference = 'Stop'
$CalibrationDir = 'C:\Users\WWWW\Desktop\tk3d\outputs\rgb8_20260924_take01\calibration'
foreach ($Name in @('cameras.json', 'intrinsics_only.json', 'calibration_report.json')) {
    if (Test-Path -LiteralPath (Join-Path $CalibrationDir $Name)) { throw 'Önceki kalibrasyon sonucu var; yeni session/sürüm kullanın.' }
}
& 'C:\Users\WWWW\Desktop\tk3d\.venv312\Scripts\python.exe' 'C:\Users\WWWW\Desktop\tk3d\scripts\calibrate_cameras.py' --session 'C:\Users\WWWW\Desktop\tk3d\data\rgb8_20260924_take01\session_calibration.yaml' --config 'C:\Users\WWWW\Desktop\tk3d\data\rgb8_20260924_take01\calibration_rgb8.yaml' --output-root 'C:\Users\WWWW\Desktop\tk3d\outputs'
if ($LASTEXITCODE -ne 0) { throw 'Kalibrasyon başarısız; eski/başka JSON kullanarak devam etmeyin.' }
Get-Content -Raw -LiteralPath (Join-Path $CalibrationDir 'calibration_report.json')
```

Başarılı ortak çözümün beklenen dosyaları (fiziksel kabul için holdout ve zemin
kontrolü ayrıca gerekir):

```text
C:\Users\WWWW\Desktop\tk3d\outputs\rgb8_20260924_take01\calibration\cameras.json
C:\Users\WWWW\Desktop\tk3d\outputs\rgb8_20260924_take01\calibration\calibration_report.json
```

Kalibrasyon, zemin hizası ve zaman hatası hedefe göre doğrulandıktan sonra
önce kısa smoke aşağıdaki gibi çalıştırılır. `--defer-latest`, mevcut parser'da
bulunan yardımcı seçenektir; bu kısa koşunun `latest_run.json` işaretçisini
ilerletmesini engeller. Tek başına `--allow-low-quality-output` başarılı kısa
run'ın latest olmasını engellemez; yalnız başarısız kalite sonucunu teşhis
için tutar. Run ID her çalıştırmada değişmelidir.

Bu örnek bugün var olan tam `model_config.yaml` ile çalışır; RGB8 session'ında
`zed.depth_sources` olmamalıdır. RGB-only dönüş bu durumda güvenle gerçekleşir.
ZED'i açıkça kapatan ayrı tam RGB8 config hazırlandığında `--model-config` ona
çevrilir; yalnız tek alan içeren bir YAML mevcut config'e otomatik eklenmez.

```powershell
Set-Location 'C:\Users\WWWW\Desktop\tk3d'
$RunId = 'rgb8-smoke-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff')
& 'C:\Users\WWWW\Desktop\tk3d\.venv312\Scripts\tk3d-multiview.exe' --session 'C:\Users\WWWW\Desktop\tk3d\data\rgb8_20260924_take01\session.yaml' --model-config 'C:\Users\WWWW\Desktop\tk3d\config\model_config.yaml' --output-root 'C:\Users\WWWW\Desktop\tk3d\outputs' --max-frames 120 --stride 1 --run-id $RunId --allow-low-quality-output --defer-latest
if ($LASTEXITCODE -ne 0) { throw 'Smoke çalışması hata verdi; run_state ve kalite raporunu inceleyin.' }
Get-Content -Raw -LiteralPath "C:\Users\WWWW\Desktop\tk3d\outputs\rgb8_20260924_take01\runs\$RunId\json\run_quality_report.json"
```

Kalite raporundaki `status` ve `run_state.json` ayrıca incelenir; düşük kaliteye
izin verilmiş koşuda process'in 0 dönmesi iç geometri kapısının geçtiğini
göstermez. Smoke ortak timeline'ın **ilk** 120 karesidir; 60 FPS'te yaklaşık
2 saniye eder. Kayıt başındaki LED/boş salon bölümünü test etmiş olabilir.
Hareket içeren bir aralık seçilecekse kaynak eşlemeli yeni pilot session
hazırlanmalıdır; bu CLI'da `--start-frame` yoktur.

Tam çekimin kalitesini temsil etmeyen bu smoke'tan sonra, hareketli bölüm ve
diğer saha kapıları da geçtiyse yeni `run_id` ile tam koşu:

```powershell
Set-Location 'C:\Users\WWWW\Desktop\tk3d'
$RunId = 'rgb8-full-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff')
& 'C:\Users\WWWW\Desktop\tk3d\.venv312\Scripts\tk3d-multiview.exe' --session 'C:\Users\WWWW\Desktop\tk3d\data\rgb8_20260924_take01\session.yaml' --model-config 'C:\Users\WWWW\Desktop\tk3d\config\model_config.yaml' --output-root 'C:\Users\WWWW\Desktop\tk3d\outputs' --stride 1 --run-id $RunId --profile-performance
if ($LASTEXITCODE -ne 0) { throw 'Tam koşu başarısız; kalite raporunu inceleyin.' }
Write-Output "C:\Users\WWWW\Desktop\tk3d\outputs\rgb8_20260924_take01\runs\$RunId"
```

Ana çıktı yolları (`<run_id>`, blokta yazdırılan `$RunId` değeridir):

```text
C:\Users\WWWW\Desktop\tk3d\outputs\rgb8_20260924_take01\runs\<run_id>\json\vitpose_session_3d.json
C:\Users\WWWW\Desktop\tk3d\outputs\rgb8_20260924_take01\runs\<run_id>\json\run_quality_report.json
C:\Users\WWWW\Desktop\tk3d\outputs\rgb8_20260924_take01\runs\<run_id>\json\performance_report.json
```

### 9.1 Poomsae incelemesine geçmeden önce

`poomsae1_trimmed` profili eski iki ZED videosuna, 741 karelik pose'a ve
M01–M06 timeline hash'lerine bağlıdır. Yeni sekizli kayıt için bu profil
komutunu olduğu gibi çalıştırmak yeni kaydı analiz etmez.

Yeni kayda özel profil, pose/quality/manifest bağları, kayıt kapsamı ve
hareket/faz zaman çizelgesi gerekir. Yeni kaydın etiketleri insanlarca
incelenmelidir; eski `manual/confirmed` alanları veya eski hareket kare
numaraları yeni performansa kopyalanmamalıdır. Geometrinin iyileşmesi bu onayı
sağlamaz ve resmî puanı açmaz.

Sunum kodunda ayrıca şu varsayımlar var:

- `render_decision_evidence_video`, video FPS/kare sayısını evidence ile
  karşılaştırıp uyuşmazlığı reddeder; frame 0'dan itibaren sırayla okur.
  Session'daki kamera ofsetlerini uygulamaz.
- HTML incelemesi bütün videolarda aynı `currentTime` değerine gider; kaynak
  kameralara ait zaman dönüşümünü bilmez. Tarayıcı playback eşitlemesi
  ölçüm düzeyinde capture senkronizasyonu değildir.
- `build_browser_review_videos.py` codec uyumlu kopya oluşturur, kamera
  ofsetlerini çözmez. “Browser proxy üretildi” ifadesi “senkronlandı” demek
  değildir.
- Sekiz kamera mevcut hata videosunda `960 × 8 = 7680` piksel genişliğinde
  tek satıra dizilir. Kullanışlı 2×4 düzen/aktif kamera seçimi ayrı sunum
  iyileştirmesidir; sekizli kullanıcı arayüzü doğrulanmış sayılmamalıdır.

Önerilen hazırlama sözleşmesi: ortak başlangıç, ortak FPS ve aynı kare sayısı
olan videolar; aynı timeline'a bağlı 2B CSV, 3B pose ve evidence; orijinal
dosyalara dönen kaynak kare/PTS haritası. Bu hazırlama performans bölümünü
eksiltmemeli. Hazırlanmış dosyalarla inference yeniden yürütülecekse yeni
session/run oluşturulur ve o dosyaların ofsetleri sıfır olur; önceden yapılmış
düzeltme ikinci kez uygulanmaz. Mevcut generic RGB için bunu otomatik yapan
doğrulanmış bir hazırlama komutu henüz yoktur.

## 10. Her take için saha kontrol listesi

### Çekimden önce

- [ ] Sekiz fiziksel kamera etiketi ve session kimliği doğru.
- [ ] Tripod işaretleri ve kamera yönleri değişmemiş.
- [ ] Lens temiz, focus/zoom/stabilizasyon kilitli.
- [ ] Bütün kameralar hedef FPS ve çözünürlükte.
- [ ] Shutter, ISO/gain ve white balance sabit.
- [ ] Hafıza kartlarında yeterli boşluk var.
- [ ] Sürekli güç veya yeterli pil var.
- [ ] Bütün kameralar çalışma hacmini tam görüyor.
- [ ] LED + buzzer senkron cihazı sekiz görüntüde de görünüyor/duyuluyor.
- [ ] Kalibrasyondan sonra hiçbir tripod/lens hareket etmedi.

### Take sırasında

- [ ] Bütün kameralar ortak işaretten önce kayıt yapıyor.
- [ ] Başlangıç pulse dizisi kaydedildi.
- [ ] Sporcu pulse'tan sonra başladı.
- [ ] Kimse tripodlara veya ışıklara dokunmadı.
- [ ] Bitiş pulse dizisi kaydedildi.
- [ ] Kameralar bitiş pulse'undan sonra durduruldu.

### Kopyalama sonrası

- [ ] Sekiz dosya açılıyor.
- [ ] Süre/FPS/frame-count raporu kaydedildi.
- [ ] Başlangıç ve bitiş pulse kareleri not edildi.
- [ ] Drift/dropped-frame kontrol edildi.
- [ ] SHA-256 üretildi ve manifest'e yazıldı.
- [ ] Ham master ile çalışma kopyası ayrıldı.
- [ ] `git status --short` büyük medyanın Git'e girmediğini gösteriyor.

## 11. Pilot kabul tablosu

| Alan | Asgari kabul | Başarısızsa yapılacak |
| --- | --- | --- |
| Dosya bütünlüğü | 8/8 dosya açılıyor, checksum var | Yeniden kopyala veya take'i reddet. |
| Timeline | Başlangıç ve bitiş pulse'u 8/8 görünür | Senkron kanıtı yoksa 3B üretim yapma. |
| Zaman hatası ve drift | Bölüm 5.3'teki hız/ölçüm bütçesine uygun ms sınırı; ölçüm belirsizliği dahil | Kısa take yalnız drift'i azaltır; exposure fazı için senkron donanım veya doğrulanmış zaman eşleme gerekir. |
| Frame kaybı | Açıklanamayan düşme/tekrar yok | Kaynağı düzelt; CFR transcode ile gizleme. |
| Intrinsics | Kamera çözünürlüğü eşleşiyor; RMS uyarı altında | Kamera bazlı kalibrasyonu yenile. |
| Extrinsics | Bağlı graph/çok-yüzlü hedef, holdout raporu var | `cameras.json` üretim olarak işaretlenmemeli. |
| Dünya referansı | Metre ölçeği, gerçek zemin ve dikey yön doğrulanmış | Eksen dönüşümünü zemin ölçümüyle çöz; kamera tilt'ini world z sanma. |
| Görüş | BODY-17 her anda en az 2, tercihen 3–4 iyi view | Yerleşim/ışık/lens düzenini değiştir. |
| İç 3B kalite | Kod kapısı geçer; BODY-17 valid ≥0.90 | Kamera bazlı sync/calibration/2B teşhisi yap. |
| Reprojection | Kod ortalaması ≤25 px; pilot hedefi çok daha düşük | 25 px'i başarı hedefi gibi kullanma; kök nedeni bul. |
| Süre | Kaynak ortak timeline korunur | Kare atma/kırpma ile kaliteyi yapay artırma. |
| Provenance | Session, calibration, model ve giriş hash'leri bağlı | Run'ı kanıt olarak kullanma. |

`used_cameras[t,j]` bir **kamera sayısıdır**, kamera kimlikleri listesi değildir.
Eklem/zaman dağılımı buradan incelenir; hangi kameranın dışlandığını kesin
raporlamak için ek inlier-kimliği kaydı gerekir. Kamera başına 2B geçerlilik,
ayrı reprojection incelemesi ve optimizer ağırlıkları yardımcıdır fakat birebir
inlier kaydının yerini tutmaz. Özellikle eller, ayaklar ve hızlı tekmelerde
BODY-17 ortalaması tek başına yeterli değildir.

## 12. Repository'de yapılması gereken geliştirmeler

### P0 — Kayıt ve senkron denetimi

1. `capture_manifest` şeması ve kamera ayar envanteri.
2. FFprobe ile stream/frame PTS ve codec, OpenCV ile decode kontrolü; cihaz
   frame counter'ı varsa onunla dropped-frame denetimi. Dosya PTS'i tek başına
   sensörün hiç kare kaybetmediğini kanıtlamaz.
3. LED/audio senkron pulse detector; kamera başına başlangıç ve bitiş ofseti.
4. Drift raporu ve “fixed offset yeterli/yetersiz” kapısı.
5. Pair/grid preview; mevcut `preview_camera_pair.py` ilk iki dosyayı aynı
   yerel indeksle okur, sync ofsetlerini uygulamaz. Ofset doğrulamak için
   mevcut hali yeterli değildir.

### P1 — Sekiz kamera kalibrasyonu

1. ChArUco veya çok-yüzlü hedef detector'ı.
2. Kamera gözlem grafiği ve bağlantılılık/loop-closure kontrolü.
3. Ortak world extrinsics + robust bundle adjustment.
4. Holdout reprojection ve bilinen mesafe raporu.
5. Yeni, açık calibration mode; ancak dış/holdout doğrulama sonrası production
   whitelist'e alınmalı.
6. Kamera merkezi/frustum 3B görselleştirmesi.
7. Gerçek zemin/yön dönüşümü ve lens-model/çözünürlük kontrolü.
8. Intrinsic ve extrinsic gözlemlerinin ayrı provenance'ı; calibration/performance
   sync ayrımı, immutable kalibrasyon sürümleri ve başarısız çözümde eski JSON
   kullanılmasını engelleyen doğrulama.

### P2 — Zaman modeli

1. Session şemasına per-frame timestamp map veya affine/piecewise time map.
2. Sabit ofsetin yetmediği koşuda fail-closed davranış.
3. Timeline resampling'in hangi kaynak kareyi seçtiğini machine-readable kayıt.
4. VFR ve dropped-frame kaynaklarını sessizce kabul etmeme.

### P3 — RGB8 çalışma profili

1. Kanonik ZED config'ini değiştirmeyen ayrı `model_config_rgb8.yaml`.
2. ZED depth'in açıkça kapalı olduğu sensor profile/provenance.
3. Cross-view ikinci geçişin 4/6/8 kamera karşılaştırması: dört kamerada hedef
   dışı yalnız üç kamera kaldığından mevcut ayarla işlem beklenmez; beş veya
   daha fazla toplam kamera gerekir. Aynı zaman aralığı/model/stride korunur;
   yalnız kontrollü kamera alt kümesi veya feedback ayarı değiştirilir.
4. `max_hypotheses`, çoğunluk-inlier davranışı ve kalite eşikleri için farklı
   sporcu/kıyafet/ışık kayıtlarıyla doğrulama.
5. 2/4/8 kamera hız ve GPU/RAM/disk benchmark'ı.
6. Yeni kayıt için Poomsae profil/timeline/hash bağları, hazırlanmış ortak
   video timeline'ı ve sekiz kamera sunumunun doğrulanması.

### P4 — Kayıt orkestrasyonu (yalnız merkezi capture seçilirse)

1. Kamera cihaz kimliğini kararlı `camera_id` ile eşleyen recorder.
2. Her akış için ayrı process/thread, bounded queue ve dropped-frame sayacı.
3. Per-frame host/device timestamp kaydı.
4. Disk throughput ve USB-controller topolojisi preflight'ı.
5. Tek hata halinde diğer yedi kaydı sessizce “başarılı” saymayan run durumu.

## 13. Önerilen uygulama sırası

1. **Envanter:** Kullanılacak sekiz kameranın gerçek model, kayıt modu,
   manuel kontrol ve timestamp yeteneklerini çıkarın.
2. **Masa testi:** Uyumlu kameralarda seçilen 1080p50/60 modu ile 10 dakikalık
   kayıt; termal kapanma, dosya bölme, VFR, ışık titremesi ve ses kontrolü.
3. **Sekizli bandwidth/depolama testi:** Gerçek hedef codec ile planlanan take
   süresinin en az 1.5 katı.
4. **Saha yerleşimi:** Sekiz kamera kurulup fiziksel kimlik ve çalışma hacmi
   doğrulanır.
5. **Senkron pilotu:** Baş/son LED dizisiyle 2 dakikalık kayıt, drift ölçümü.
6. **Kalibrasyon pilotu:** Önce intrinsics; sonra çok-yüzlü hedef veya bağlı
   ChArUco graph. Mevcut all-camera checkerboard ancak gerçekten görünüyorsa.
7. **Statik 3B kontrol:** Bilinen noktalar/uzunluklar triangulate edilir.
8. **Dinamik kısa kontrol:** Yavaş yürüyüş/kol hareketi; 2, 4 ve 8 kamera aynı
   zaman aralığında karşılaştırılır.
9. **Poomsae smoke:** Sporcunun hareket ettiği doğrulanmış ortak bölümden
   120 kare, stride 1; kamera kullanımı ve kalite raporu. İlk 120 kare boş
   hazırlıksa bu koşu hareket kalitesini doğrulamaz.
10. **Tam take:** Yeni run kimliği, stride 1; ancak önceki bütün kapılar
    geçtiyse.
11. **Dış doğrulama:** Mümkünse ölçülmüş 3B wand, mocap veya bağımsız referans.

## 14. Net karar

Sekiz farklı standart kamera TK3D'nin uzun vadeli yönüyle uyumludur ve yalnız
RGB ile 3B üretmek mümkündür. Hatta sekiz view, kapanma direnci ve cross-view
2B geri besleme açısından iki ZED RGB görüntüsünden daha fazla geometrik kanıt
sağlayabilir. Buna karşılık ZED'in cihaz içi timestamp/depth/kalibrasyon
avantajları kaybolur; bu yük kayıt, senkron ve kalibrasyon altyapısına taşınır.

Bugün güvenle söylenebilecek sınır şudur:

- **Dosya okuma, 2B pose, robust N-view triangulation, RGB global optimizer ve
  WholeBody-133 çıktı çok-kameralı yapıya sahiptir.** Sekiz sanal kamerayla
  geometri kontrolü yapılmıştır; sekiz gerçek cihazla uçtan uca doğrulama
  henüz yoktur.
- **Çevresel extrinsic kalibrasyon/zemin hizası, kalan zaman hatasının
  ölçülmesi, kayıt denetimi ve Poomsae sunumuna hazırlama tamamlanmalıdır.**
- Ortak kalibrasyon etiketi veya test başarısı tek başına gerçek kayıt
  doğruluğu veya resmî puan hazırlığı değildir.

İlk adım, mevcut/alınacak modellerin sabit FPS, manuel
exposure/focus/white balance, stabilizasyon kapatma, kayıt süresi ve senkron
yeteneklerini çıkarmak; hedef ölçüm hatasıyla eşleştirmek; ardından uygun
senkron işareti ve kalibrasyon yöntemini seçmektir. Kamera sayısı tek başına doğruluğu artırmaz;
aynı fiziksel an ve aynı ortak dünya koordinatı doğru kurulmuşsa artırır.

## 15. İkinci incelemenin kanıtları ve sınırı

Bu revizyonda uygulama/config değiştirilmedi; rapor ve belge bağlantıları
güncellendi. Kaynak taraması özellikle şu sözleşmeleri kapsadı:

| Kod | Denetlenen konu |
| --- | --- |
| `src/video_io.py`, `src/data_structures.py` | Session yolları, kimlikler, ofsetler; kayıt cihazı değil dosya girdisi |
| `src/multiview_sync.py`, `src/multiview_application.py` | FPS alt sınırı, overlap, en yakın kare, kalite ve run davranışı |
| `src/camera_calibration.py`, `scripts/calibrate_cameras.py` | Ortak detection şartı, stride/tolerans, referans adı, fallback ve yazma davranışı |
| `src/coordinate_system.py` | Eksen dönüşümü ile gerçek zemin hizası arasındaki fark |
| `src/triangulation.py`, `src/crossview_pose2d_feedback.py` | Lens modeli, çoğunluk inlier, BODY-17 leave-one-out desteği |
| `src/zed_depth_fusion.py`, `src/quality_status.py` | Depth olmadan RGB sonucunun korunması ve iç kalite sınırı |
| `src/pose2d_estimator.py`, `config/model_config.yaml` | Sıralı kamera inference'ı, sabit crop ve piksel eşikleri |
| `src/preflight.py`, `src/video_probe.py`, ilgili CLI'lar | Hazırlık denetimlerinin gerçekte neyi kontrol ettiği |
| `src/poomsae_scoring/application.py`, `error_video.py`, `review_report.py` | Kayıt/hash bağı, video FPS/kare sayısı ve playback varsayımları |
| `scripts/build_browser_review_videos.py`, `scripts/preview_camera_pair.py` | Proxy/preview işleminin senkron çözümü olmadığı |
| `PROJECT_STATUS.md`, `README.md`, aktif profil ve testler | Güncel kanıt ile yeni RGB8 planının ayrımı |

Gerçekten çalıştırılan doğrulamalar:

- İlgili yedi test dosyası: **30 passed in 1.32s**.
- Tam test paketi: **541 passed in 81.21s**. JUnit:
  `C:\Users\WWWW\Desktop\tk3d\outputs\pytest-rgb8-reaudit-full-20260924-01.xml`.
- `ruff check src scripts tests`: **All checks passed**.
- `pip check`: **No broken requirements found**.
- İki YAML bloğu parse edildi; kalibrasyon config'i mevcut validator'dan
  geçti, sekiz kamera kimliği ve `c01` referansı eşleşti. Dört PowerShell
  bloğu parser ile sözdizimi kontrolünden geçti; örnek kayıt komutları gerçek
  kameralar üzerinde çalıştırılmış sayılmaz.
- `tk3d-check --json`: **READY**. Bu sonuç mevcut **ZED** araştırma ortamı
  içindir; RGB8 sisteminin hazır olduğunu göstermez. Ölçülen Python sürümü
  `3.12.14`; eski durum tablosundaki `3.12.13` güncellendi.
- Model yüklemeyen sentetik ek kontrol: sekiz farklı K/çözünürlük/distortion
  ve çevresel kamera pozu; 133 noktanın geometrik yeniden kurulumu; bir
  kamera gözlemi bozulduğunda yedi inlier ile sonucu koruma; depth'siz RGB
  dönüşü; örnek YAML ve zaman ofseti işareti.
- 60 FPS, üç kare ofsetli yapay detection dizisinde `frame_stride=10`
  karşılığı **0 ortak eşleşme**, bütün karelerle **117 eşleşme** elde edildi.
  Bu gerçek checkerboard videosu testi değildir; eşleme kısıtının matematiksel
  yeniden üretimidir.

Sentetik doğrulama dosyaları:

```text
C:\Users\WWWW\Desktop\tk3d\outputs\rgb8_report_audit\runs\recheck-20260924-01\audit_report_claims.py
C:\Users\WWWW\Desktop\tk3d\outputs\rgb8_report_audit\runs\recheck-20260924-01\verification.json
```

Sekiz fiziksel kamera kaydı, yeni gerçek kalibrasyon, RGB8 gerçek model
inference'ı, sekizli arayüz görsel testi veya dış 3B doğruluk benchmark'ı bu
incelemede yapılmadı. Eski ZED/AIST/MADS test ve benchmark sonuçları RGB8
doğruluğu olarak aktarılmadı. “Projeye hâkimiyet” bu kaynak ve sözleşme
incelemesidir; bütün olası cihaz kombinasyonlarının doğrulandığı iddiası değildir.
