# TK3D — 10 ZED 2i Offline Çok-Kamera Yol Haritası

> **ARŞİVLENMİŞ TARİHSEL PLAN:** Bu belge erken ZED geliştirme planını ve
> pilot kararlarını korur. `CURRENT_ACTIVE` uygulama durumunu temsil etmez;
> güncel kanonik workflow iki ZED 2i ile AVI RGB + SVO2 depth kullanan
> multiview/Poomsae hattıdır. Güncel durum için `PROJECT_STATUS.md` ve
> `docs/PROJECT_CONTEXT.md` belgelerine bakın.

Durum: **tek-ZED SVO2 pilotu ölçüldü; üretim ZED çalışma zamanı henüz kaynak
kod hattına uygulanmadı**
Ana hedef: 10 adet ZED 2i ile gerçek zaman zorunluluğu olmadan, mümkün olan
en yüksek kaliteli 133 noktalı 3B poomsae pozu üretmek.

Bu belge mevcut `AGENTS.md`, `PROJECT_STATUS.md` ve
`docs/ARCHITECTURE_DECISIONS.md` kurallarını genişletir. Buradaki planın ana
ilkesi, ZED'in hazır iskeletini TK3D'nin yerine geçirmek değil; ZED RGB,
stereo-depth, IMU, zaman damgası ve kalibrasyonunu mevcut ViTPose-Huge
WholeBody hattına güvenli ölçüm olarak eklemektir.

## Mevcut pilot kararı — 27 Temmuz 2026

Kullanıcıdan alınan mevcut bilgi:

- Sabit ve ölçülmüş bir oda/aktif alan henüz belirlenmedi.
- ZED 2i lens tipi henüz bilinmiyor.
- Özel Ethernet switch bulunmuyor.
- Özel LED senkron paneli bulunmuyor.
- Kameralar kurulduktan sonra sabit kalacak.
- İlk pilot take yaklaşık 2 dakika olacak.
- İlk aşamada günde yalnız bir deneme yapılacak.
- Kullanıcı, diğer kayıt bilgisayarlarının da güçlü olduğunu belirtti; kesin
  model ve USB/disk topolojileri henüz envantere alınmadı.

Bu nedenle ilk pilot için karar:

```text
tek ZED 2i
  -> mevcut ana bilgisayarda yerel SVO2 kayıt
  -> yaklaşık 2 dakika
  -> offline RGB ve depth çıkarma
  -> ViTPose + depth-lift teknik doğrulaması
```

İlk pilotta 10 kamera, ZED360 Network Workflow veya ZED Fusion zorlanmaz.
Tek-kamera veri sözleşmesi doğrulandıktan sonra iki kamera, ardından dört ve
on kamera aşamasına geçilir.

### İlk SVO2 pilot sonucu — 28 Temmuz 2026

Kaynak:

```text
HD720_SN39504762_13-06-55.svo2
```

Dosya ve kamera SDK 5.4.1 ile gerçek olarak açılıp baştan sona tarandı:

| Alan | Ölçüm |
|---|---:|
| Kamera | ZED 2i, seri `39504762` |
| Firmware | `1523` |
| Kalibre odak / görüş | `3.823 mm`, yatay `67.73°`; 4 mm lens varyantıyla uyumlu |
| Kayıt modu | `1280×720`, `60 FPS` |
| SVO'nun raporladığı kare | `667` |
| Başarıyla grab edilen kare | `666`, konum `0–665` |
| Kamera zaman süresi | `11.134046 s` |
| SVO2 boyutu | `34,354,190 byte` |
| SHA-256 | `6410CCF95CA1A671B13ED6FD50CA3143EADDDFCE19F5AE489AD1EAC182469BE0` |
| IMU bulunan kare | `666/666` |

Zaman damgaları monoton ve SVO konumları kesintisizdi. Bununla birlikte 204,
364 ve 484 numaralı karelerden sonra yaklaşık `33 ms` aralık görüldü; bunlar
60 FPS zaman çizelgesinde yaklaşık birer eksik kareye karşılık gelir. Bu kısa
kayıt dosya bütünlüğü açısından açılabilir durumdadır fakat üretim için
`timestamp_gap_count=0` hedeflenmeli ve en az 15–30 dakikalık stres testi
yapılmalıdır.

SDK'nın örnek `ZED_SVO_Export` programı dosyayı açtı ve NEURAL depth üretti,
fakat çıkardığı AVI'yi `25 FPS` olarak etiketledi. Kaynak gerçekte `60 FPS`
olduğundan bu örnek AVI doğrudan TK3D zaman çizelgesine verilmemelidir.
Doğru `1280×720/60 FPS`, 666 karelik sol RGB MP4 ayrıca üretildi.

NEURAL depth, her 10 karede bir örneklenen 70 ölçümde:

- tüm görüntü geçerli depth medyanı: `%99.945`;
- güçlü güvenli depth medyanı: `%95.905`;
- orta ROI geçerli depth medyanı: `%99.912`;
- orta ROI güçlü güvenli depth medyanı: `%95.496`;
- görüntü medyan depth'i: `5.706 m`.

RF-DETR-Small + ViTPose-Huge WholeBody ile 67 karelik gerçek `stride 10`
smoke koşusunda BODY-17 algılama `67/67 × 17/17` oldu. Tek piksellik/naif
depth örnekleme siluet sınırındaki burun, kulak ve bilekleri bazen yaklaşık
`3.1 m` kişiden `7.5–8.1 m` arka plana sıçrattı; kol uzunluğunda `4 m` üstü
aykırılar oluştu. Bu nedenle ham "geçerli depth" oranı kalite kanıtı değildir.

Omuz ve kalçaların medyanını kişi-depth öncülü yapan, eklem çevresinde yalnız
güçlü ve anatomik banttaki bağımsız stereo-depth piksellerini seçen pilot
kapıyla:

- BODY-17 depth eşleşmesi: `%100`;
- ayak depth eşleşmesi: `%100`;
- yüz depth eşleşmesi: `%90.89`;
- el depth eşleşmesi: `%81.69`;
- kişi-depth bandı dışındaki aykırı: `0`;
- BODY-17 depth aralığı: `2.325–3.631 m`, medyan `3.134 m`;
- bacak/torso robust kemik değişkenliği yaklaşık `%1.9–3.7`;
- kol robust kemik değişkenliği yaklaşık `%6–9`.

Bu kişi-depth kapısı yalnız pilot kanıtıdır. `1.25 m` bandı veri setine özel
genel varsayılana taşınmayacak; çok-kamera reprojection, depth residual ve
dış doğrulamayla yeniden ayarlanacaktır. Yüz/el depth'i zorla doldurulmayacak,
bu noktalar ağırlıklı olarak bağımsız çok-kamera RGB triangulation'ından
gelecektir.

Pilot çıktıları:

```text
outputs/zed2i_pilot/runs/zed2i-svo-diagnostic-20260728-180605/
  videos/left_rgb_60fps.mp4
  videos/vitpose_depth_smoke_stride10.mp4
  json/svo_diagnostic_report.json
  json/pose_depth_smoke_report.json
  json/pose_depth_masked_comparison.json
  csv/frame_timestamps.csv
  csv/pose_depth_observations.csv
  csv/pose_depth_observations_person_masked.csv
```

Bu tek kayıtla sistemin SVO2 okuyabildiği, doğru sol RGB üretebildiği,
ViTPose çalıştırabildiği ve BODY-17 için kullanılabilir depth kanıtı
çıkarabildiği doğrulandı. Tek kamera sonucu ortak dünya koordinatlı,
çok-kamera doğrulanmış 3B veya puanlamaya hazır sonuç değildir.

Hedef kayıt dağılımı `5 güçlü bilgisayar × 2 ZED 2i` olarak seçilmiştir.
Bilgisayar gücü bu dağılımı makul kılar; ancak iki kameranın aynı USB
denetleyicisini paylaşması ve yerel diskin sürekli yazma kapasitesi ayrı stres
testi gerektirir. Testi geçemeyen bilgisayarda kamera sayısı bire düşürülür.

Özel LED paneli yerine bütün kameraların görebileceği bir telefon ekranı veya
telefon feneriyle benzersiz görsel zaman dizisi kullanılabilir. Telefon da
kullanılamıyorsa ağsız ve farklı bilgisayarlı çekimlerde hassas
senkronizasyon onaylanamaz; yalnız kaba hareket korelasyonu üretim kalitesi
için yeterli sayılmaz.

### Mevcut ana bilgisayar envanteri

27 Temmuz 2026 tarihinde yerel olarak okunan bilgiler:

| Alan | Değer |
|---|---|
| Bilgisayar | MONSTER TULPAR T7 V20.8 |
| İşlemci | Intel Core i7-12700H |
| RAM | 31.7 GB |
| NVIDIA GPU | RTX 4060 Laptop GPU |
| NVIDIA VRAM | 8188 MiB |
| İşletim sistemi | Windows 11 Pro |
| Fiziksel disk | CT1000P3PSSD8, yaklaşık 931.5 GB |
| C: boş alan | 28 Temmuz pilotu sonunda yaklaşık 47.0 GB |
| Görünen USB topolojisi | bir Intel USB 3.10 host controller/root hub |
| ZED SDK | `5.4.1`, Python API `5.4`, firmware `1523` |

Değerlendirme:

- Bu bilgisayar offline ViTPose ve ZED depth işlemesi için ana makine olmaya
  uygundur.
- 8 GB VRAM nedeniyle on kameranın depth ve ViTPose işlemleri aynı anda
  çalıştırılmamalı; kamera bazında veya kontrollü küçük batch'lerle offline
  işlenmelidir.
- Tek görünen USB host controller nedeniyle aynı anda çok sayıda ZED bu
  makineye bağlanmamalıdır.
- İlk test tek ZED ile yapılır. İki ZED ancak HD720/60 stres testi geçerse
  onaylanır.
- Mevcut boş disk alanıyla kayıt formatı ve gerçek SVO2 boyutu ölçülmeden
  çok-kameralı take planlanmaz. Harici SSD veya daha fazla boş alan gerekebilir.
- ZED SDK ve Python API kuruludur. ZED Diagnostics ile kamera/USB testi ve
  NEURAL_PLUS model optimizasyonu ayrıca tamamlanmalıdır.

## 1. Net mimari kararı

Gerçek zaman gerekli olmadığı için önerilen üretim yolu şudur:

```text
10 × ZED 2i
  -> kameralara yakın farklı bilgisayarlarda yerel SVO2 kayıt
  -> bütün kayıtların ana bilgisayara taşınması
  -> ortak fiziksel zaman çizelgesinin kurulması
  -> ZED sol RGB üzerinde RF-DETR + ViTPose-Huge WholeBody
  -> ZED stereo-depth ile 2B eklemlerin kamera-3B'ye kaldırılması
  -> bütün kameraların ortak dünya koordinatına dönüştürülmesi
  -> robust triangulation + depth residual'ları
  -> anatomik ve zamansal global optimizasyon
  -> 133 noktalı nihai TK3D çıktısı
```

Bu yapıda:

- ZED 2i yalnız kamera değildir; RGB, stereo-depth, IMU ve zaman damgası
  sağlayan ölçüm kaynağıdır.
- ViTPose fiziksel ZED 2i'nin içinde çalışmaz. ZED 2i üzerinde hesaplama
  donanımı yoktur. ViTPose kayıt bilgisayarında veya ana bilgisayarda çalışır.
- En yüksek kalite için ViTPose ana bilgisayarda kayıt sonrasında çalıştırılır.
- ZED360 ilk kamera yerleşimi ve kalibrasyon kontrolünde kullanılır.
- ZED Fusion, ZED BODY-38 sonucu için paralel karşılaştırma ve isteğe bağlı
  canlı önizleme sağlar.
- Nihai puanlama girdisini ZED BODY-38 değil, TK3D'nin 133 noktalı çözümü
  üretir.

## 2. Neden offline yol seçildi?

Offline yol aşağıdaki nedenlerle bu proje için daha uygundur:

- Her kayıt bilgisayarında ViTPose-Huge çalıştıracak güçlü GPU zorunluluğunu
  kaldırır.
- Kayıt sırasında model gecikmesi yüzünden kare kaybetme riskini azaltır.
- ZED depth daha yüksek kaliteli ayarla sonradan tekrar üretilebilir.
- Flip-test, sıfır-fazlı stabilizasyon ve tüm-sekans optimizasyonu
  kullanılabilir.
- Aynı ham SVO2 verisi üzerinde eski ve yeni algoritma adil biçimde
  karşılaştırılabilir.
- Hatalı bir model ayarı yüzünden çekimin tekrar yapılması gerekmez.
- Ham stereo görüntü, zaman damgası ve IMU korunur.

Canlı mod ileride ayrıca geliştirilebilir, fakat ilk üretim hedefi değildir.

## 3. ZED360, ZED Fusion ve TK3D'nin görev ayrımı

### ZED360

ZED360 görsel bir çok-kamera kurulum ve kalibrasyon aracıdır. Kamera
görüntülerindeki ZED gövde noktalarını ortak WORLD koordinatına hizalayarak
kamera pozlarını içeren bir JSON üretir.

TK3D'deki görevi:

- kamera seri numaralarını ve fiziksel yerleşimi kontrol etmek;
- komşu kamera örtüşmesini görsel olarak doğrulamak;
- ilk extrinsic tahminini üretmek;
- ZED'in birleşik gövdesini kurulum sırasında görmek.

Sınırı:

- Kalibrasyon ZED'in kendi body keypoint'lerine dayanır.
- Mutlak eklem/puanlama hassasiyeti için tek başına üretim kanıtı sayılmaz.
- Son üretim extrinsic'leri ChArUco/checkerboard gözlemleri ve ortak bundle
  adjustment ile iyileştirilmelidir.

### ZED Fusion

ZED Fusion, ZED SDK içindeki publish/subscribe tabanlı çok-kamera API'sidir.
ZED'in body tracking, object detection, spatial mapping ve ilgili modüllerini
yerel veya ağ üzerinden birleştirebilir.

TK3D'deki görevi:

- ZED BODY-38 için bağımsız bir karşılaştırma sonucu üretmek;
- isteğe bağlı düşük gecikmeli canlı önizleme sağlamak;
- kamera bağlantısı ve kalibrasyon sağlığını hızlı kontrol etmek.

Sınırı:

- Belgelenmiş Body Tracking biçimleri BODY-18, BODY-34 ve BODY-38'dir.
- ViTPose WholeBody 133 noktayı ZED Body Fusion'a özel iskelet olarak veren
  belgelenmiş bir API yoktur.
- ZED SDK özel nesne kutularını alabilir; bu, özel 133 noktalı iskelet
  girişine eşdeğer değildir.
- ZED Fusion çıktısı ground truth değildir.

### TK3D Fusion

Bu projede geliştirilecek asıl birleştirme katmanıdır:

- ViTPose 133 nokta;
- ZED eklem depth ölçümleri;
- çok-kamera 2B reprojection;
- kamera güvenilirliği;
- kemik uzunluğu ve eklem limitleri;
- hız, ivme ve jerk;
- kapanma ve provenance

aynı çözümde değerlendirilir.

## 4. Fiziksel bilgisayar senaryoları

### Senaryo A — Önerilen: 5 bilgisayar, bilgisayar başına 2 ZED

```text
PC-01 -> ZED C01 + C02
PC-02 -> ZED C03 + C04
PC-03 -> ZED C05 + C06
PC-04 -> ZED C07 + C08
PC-05 -> ZED C09 + C10
```

Ana TK3D bilgisayarı bu beş bilgisayardan biri olabilir. Kayıt bittikten sonra
SVO2 dosyaları bu makineye taşınır.

Bu senaryo ancak her bilgisayarın iki kamerayı hedef çözünürlük/FPS'de kare
kaybetmeden kaydettiği stres testiyle doğrulanır. İki fiziksel USB portunun
aynı USB kontrolcüsünü paylaşabileceği unutulmamalıdır.

### Senaryo B — En güvenli USB dağılımı: bilgisayar başına 1 ZED

Herhangi bir bilgisayar iki ZED ile yeşil/mor kare, kopma veya frame-drop
üretiyorsa o bilgisayara tek ZED bağlanır. Toplam bilgisayar sayısı artar,
fakat özel PCIe USB kartı gerekmez.

### Senaryo C — Karışık bilgisayarlar

Bilgisayarlar aynı güçte değilse sabit `2+2+2+2+2` dağılımı zorlanmaz.
Her bilgisayarın gerçek test sonucuna göre örneğin `2+2+2+1+1+1+1`
kullanılabilir.

ZED SDK'nın güncel PC gereksinimleri NVIDIA RTX sınıfı GPU, desteklenen
Windows/Ubuntu sürümü ve USB 3 bağlantısı içerir. NVIDIA GPU bulunmayan bir
bilgisayar, test edilmeden güvenilir ZED kayıt düğümü sayılmaz. UVC ile özel
ham kayıt yolu teknik olarak ayrıca araştırılabilir; ancak SVO2, kalibrasyon,
IMU ve SDK zaman damgası sözleşmesini kaybetme riski nedeniyle ana plan
değildir.

## 5. Ağ senaryoları

### Senaryo 1 — Önerilen: özel kablolu yerel ağ

```text
Eduroam Wi-Fi -> yalnız internet/ZED Hub
Özel Ethernet -> ZED360, Fusion, saat/sağlık ve kontrol
Yerel SSD     -> asıl SVO2 kayıt
```

Örnek özel adresleme:

```text
Ana bilgisayar  192.168.50.10
Kayıt PC-01     192.168.50.11
Kayıt PC-02     192.168.50.12
Kayıt PC-03     192.168.50.13
Kayıt PC-04     192.168.50.14
Kayıt PC-05     192.168.50.15
```

Asıl kayıt ağ üzerinden yapılmaz; her bilgisayar yerel diske yazar. Ağ
koparsa SVO2 kaydı korunur.

### Senaryo 2 — Ağsız çalışma

Bu senaryo mümkündür:

- Kayıt her bilgisayarda manuel olarak erken başlatılır.
- Ortak LED zaman kodu çekim başında ve sonunda gösterilir.
- SVO2 dosyaları harici SSD ile ana bilgisayara taşınır.
- Zaman ofseti ve saat drift'i LED dizisinden offline hesaplanır.
- ZED360 Network Workflow canlı kullanılamaz.
- Üretim extrinsic kalibrasyonu ChArUco/checkerboard ile TK3D tarafında
  yapılır.
- ZED Fusion canlı karşılaştırması zorunlu olmaz.

Bu yol dosya kalitesi açısından kabul edilebilir; daha fazla manuel işlem ve
daha güçlü otomatik doğrulama gerektirir.

### Senaryo 3 — Yalnız Eduroam

Üretim yolu olarak kabul edilmez:

- Eduroam kurum politikası cihazlar arası trafiği engelleyebilir.
- Wi-Fi jitter ve paket kaybı üretebilir.
- Wi-Fi üzerinden güvenilir donanımsal PTP beklenmemelidir.
- Kullanıcı cihazlarının aynı SSID'de görünmesi, birbirine doğrudan
  erişebildiği anlamına gelmez.

Eduroam internet için kullanılabilir; kamera iç ağı olarak kullanılmamalıdır.

## 6. Kamera yerleşiminin başlangıç planı

Kesin `x, y, z`, yaw, pitch ve roll değerleri oda ölçülmeden verilemez.
Aşağıdaki düzen başlangıç geometrisidir:

| Kamera | Yerleşim | Başlangıç yüksekliği | Rol |
|---|---|---:|---|
| C01 | ön orta | 1.3–1.5 m | ana ön görünüş |
| C02 | ön-sağ çapraz | 1.6–1.8 m | sağ kol/bacak ayrımı |
| C03 | sağ orta | 1.3–1.5 m | sağ profil |
| C04 | arka-sağ çapraz | 1.6–1.8 m | çapraz kapanma |
| C05 | arka orta | 1.3–1.5 m | ana arka görünüş |
| C06 | arka-sol çapraz | 1.6–1.8 m | çapraz kapanma |
| C07 | sol orta | 1.3–1.5 m | sol profil |
| C08 | ön-sol çapraz | 1.6–1.8 m | sol kol/bacak ayrımı |
| C09 | yüksek çapraz A | 2.5–3.0 m | ayak/kesişme/örtüşme |
| C10 | C09'un karşı çaprazı | 2.5–3.0 m | karşı yüksek görünüş |

Yüksek kameralar başlangıçta yaklaşık 20–35 derece aşağı bakar. Kesin açı,
sporcunun başı ve ayakları bütün performans alanında kadraj içinde kalacak
şekilde belirlenir.

Yerleşim kuralları:

- Sporcu alanın her noktasında mümkünse en az 4–6 kamerada görünmelidir.
- Komşu kameraların görüşleri ortak kalibrasyon hedefini aynı anda görmelidir.
- Tüm kameralar tek bir yatay halkada ve aynı yükseklikte olmamalıdır.
- Ayaklar, yüksek kameralar dahil, zeminle birlikte görülmelidir.
- Güçlü pencere/spot ışığı doğrudan kameraya bakmamalıdır.
- Tripod ayakları ve kamera yönü zeminde işaretlenmelidir.
- Kamera seri numarası fiziksel `C01–C10` kimliğiyle kalıcı eşleştirilmelidir.
- Kalibrasyon sonrasında kamera oynarsa üretim kalibrasyonu geçersiz sayılır.

8×8 metre civarı performans alanı için ilk deneme merkezden yaklaşık
4.5–6 metre uzaklık olabilir. Bu değer oda ve ZED 2i'nin 2.1 mm/4 mm lens
tipine göre yeniden hesaplanmalıdır.

## 7. Kayıt standardı

Başlangıç video modu:

```text
HD720
60 FPS
aynı çözünürlük ve FPS
yerel SVO2
depth kayıt sırasında zorunlu değil; stereo görüntüden offline üretilecek
```

60 FPS hızlı tekme ve dönüşlerde 30 FPS'e göre daha iyi zaman çözünürlüğü
sağlar. Kamera başına exposure/gain/white-balance ayarı önce test edilir.
Sabit kontrollü ışıkta kameralar arası görünüm farkını azaltmak için otomatik
ayarlar kilitlenebilir; kilitleme kararı histogram ve motion-blur testiyle
verilir.

Önerilen take dizini:

```text
takes/
  take_YYYY-MM-DD_HH-mm-ss_NNN/
    manifest.json
    sync/
      led_events.json
    svo/
      C01_<serial>.svo2
      C02_<serial>.svo2
      ...
      C10_<serial>.svo2
    calibration/
      zed360_initial.json
      cameras_production.json
      calibration_report.json
```

`manifest.json` en az şu bilgileri taşır:

- take ve session kimliği;
- kamera kimliği ve seri numarası;
- kayıt bilgisayarı kimliği;
- ZED SDK ve firmware sürümü;
- çözünürlük, hedef FPS ve gerçek FPS;
- ilk/son kamera zaman damgası;
- kare sayısı ve eksik kare aralıkları;
- lens/model bilgisi;
- exposure/gain/white-balance;
- SVO2 dosya boyutu ve SHA-256 özeti;
- kullanılan kalibrasyonun özeti;
- senkronizasyon yöntemi.

Ham SVO2, modeller ve take çıktıları Git'e eklenmez.

## 8. Senkronizasyon planı

ZED 2i USB kameralarında harici donanımsal trigger bulunmadığı için
senkronizasyon zaman damgası tabanlıdır. Farklı bilgisayarların aynı anda
"başlat" komutu alması tek başına yeterli değildir.

### Özel ağ varsa

- Ortak saat için mümkünse kablolu PTP kullanılır.
- Bütün kayıt düğümleri gelecekteki ortak bir zamanda başlatılır.
- PTP'ye rağmen görsel LED zaman kodu kaydedilir.
- USB kamera pozlama farkı offline olarak ölçülür.

### Ağ yoksa

- Bütün bilgisayarlarda kayıt, hareketten önce başlatılır.
- Bütün kameraların gördüğü LED panel şu tür benzersiz bir dizi gösterir:
  `kısa-kısa-uzun-kısa`.
- Aynı dizi çekim sonunda tekrarlanır.
- Başlangıç ve bitiş eşleşmesinden her kameranın ofseti ve doğrusal saat
  drift'i hesaplanır.
- Uzun take'lerde ara LED olayı eklenir.

Ortak zaman eşlemesi:

```text
t_common = a_camera * t_camera + b_camera
```

Burada `b_camera` başlangıç ofsetini, `a_camera` saat drift'ini temsil eder.
Hızlı harekette en yakın kareyi kaba şekilde seçmek yerine 2B/depth
gözlemleri ortak zamana kontrollü biçimde enterpole edilir.

Senkron kalite raporu olmadan run üretim kalitesinde kabul edilmez.

## 9. Kalibrasyon planı

Kalibrasyon üç aşamalıdır:

### Aşama 1 — ZED fabrika intrinsics

Her kameranın ZED SDK tarafından sağlanan sol/sağ intrinsic ve stereo
kalibrasyonu seri numarasıyla alınır. Bu değerler manuel olarak başka
kameraya kopyalanmaz.

### Aşama 2 — ZED360 ilk extrinsics

Özel ağ mevcutsa:

- bütün ZED publisher'ları açılır;
- tek kişi alanın tamamında yavaşça dolaşır;
- komşu kameralar aynı kişiyi ortak görür;
- ZED360 extrinsic JSON'u dışarı aktarılır.

Bu dosya `initial` provenance taşır; tek başına üretim onayı değildir.

### Aşama 3 — Üretim geometrik iyileştirme

- Büyük ve rijit ChArUco/checkerboard hedefi tüm hacimde gezdirilir.
- Hedef komşu kamera gruplarında aynı fiziksel anda görünür.
- Bütün kamera pozları tek global bundle adjustment içinde iyileştirilir.
- Zemin düzlemi, metre ölçeği ve TK3D eksenleri belirlenir.
- Kamera başına ve ortak reprojection dağılımı raporlanır.
- ZED360 başlangıcıyla geometrik çözüm arasındaki fark kaydedilir.

İlk mühendislik hedefi olarak kalibrasyon hedef noktalarında medyan
reprojection'ın yaklaşık 1 px altında, P95'in yaklaşık 2 px altında olması
istenebilir. Bu değerler pilot çekimde gerçek lens, mesafe ve çözünürlüğe göre
yeniden doğrulanacak hedeflerdir; ground-truth doğruluğu garantisi değildir.

Her çekim gününde kısa drift kontrolü yapılır. Eşik aşılırsa eski
kalibrasyonla üretim çalıştırılmaz.

## 10. ViTPose ve ZED depth entegrasyonu

Her kamera ve her senkron kare için:

1. ZED'in rectified sol RGB görüntüsü alınır.
2. RF-DETR + mevcut kişi takibi sporcuyu seçer.
3. ViTPose-Huge WholeBody 133 adet 2B keypoint ve güven üretir.
4. Her keypoint çevresinde küçük adaptif depth bölgesi incelenir.
5. Geçersiz, düşük güvenli ve önplan/arkaplan karışmış depth çıkarılır.
6. Sağlam depth medyanı, yayılımı ve örnek sayısı hesaplanır.
7. `(u, v, depth)` intrinsic ile kamera-3B noktasına çevrilir.
8. Extrinsic ile TK3D dünya koordinatına dönüştürülür.
9. Ölçüm, güven ve provenance ile global çözücüye verilir.

Tek depth pikseli doğrudan kullanılmaz. Silüet kenarlarında, ellerde,
ayaklarda, siyah/parlak kıyafette ve motion blur sırasında depth aykırıları
beklenir.

Ölçüm ağırlığı en az şunlara bağlıdır:

- ViTPose heatmap güveni;
- ZED depth confidence;
- yerel depth yayılımı;
- eklemin silüet kenarına uzaklığı;
- kameraya uzaklık;
- diğer kameralarla reprojection/depth uyumu;
- kamera sağlık ağırlığı;
- kapanma durumu.

Bir ZED'in ViTPose 2B noktası ve aynı ZED'in depth'i iki bağımsız kamera
kanıtı gibi sayılmaz. Bunlar aynı `RGB-D observation` içinde bağlı tutulur.

## 11. Global çözüm

Nihai çözücü aşağıdaki residual ve kısıtları birlikte değerlendirir:

- kamera başına 2B reprojection;
- kamera başına ZED depth/ray residual'ı;
- robust çok-kamera triangulation;
- kamera ve eklem güven ağırlıkları;
- sabit/kararlı kemik uzunlukları;
- insan eklem limitleri;
- hız, ivme ve jerk sürekliliği;
- kısa kapanmalar;
- zemin ve ayak teması;
- ölçüm provenance.

Güvenlik kuralları:

- Depth triangulation'ı zorla ezmez.
- Ham 2B, ham depth-lift, ham triangulation ve optimize sonuç ayrı korunur.
- Aykırı depth reddedilebilir fakat görüntüde bulunmuş gibi yeniden
  etiketlenemez.
- Görüntü/depth kanıtı olmayan uzun boşluk doldurulmaz.
- Global çözüm kalite kapısını geçmezse güvenli ham triangulation'a dönülür.
- 133 nokta sözleşmesi BODY-17 optimizasyonu nedeniyle küçültülmez.

## 12. Planlanan çıktı/provenance

Kamera-eklem-kare başına aşağıdaki alanların saklanması hedeflenir:

```text
take_id
camera_id
camera_serial
host_id
local_frame_idx
camera_timestamp_ns
common_timestamp_ns
joint_id
keypoint_xy
vitpose_confidence
depth_m
depth_confidence
depth_valid_sample_count
depth_spread_m
point3d_camera
point3d_world
observation_source
accepted
rejection_reason
calibration_hash
```

`observation_source` örnekleri:

- `vitpose_rgb`
- `zed_depth_lift`
- `multiview_triangulated`
- `zed_body_baseline`
- `global_optimized`
- `temporal_recovered`
- `visualization_only`

## 13. Uygulama aşamaları

### Faz 0 — Donanım envanteri

- 10 ZED'in model, seri numarası, lens ve firmware bilgisi;
- kullanılabilecek bilgisayarların CPU/GPU/RAM/USB/disk/OS bilgisi;
- oda ve performans alanı ölçüsü;
- kablo uzunlukları ve kamera sabitleme imkanları

toplanır.

Çıktı: kesin kamera-PC eşleme tablosu ve fiziksel yerleşim planı.

### Faz 1 — Tek ZED kayıt pilotu

- SVO2 kayıt;
- seri numarası;
- gerçek FPS ve kare zaman damgaları;
- IMU;
- offline sol RGB ve depth çıkarma;
- dosya bütünlüğü

doğrulanır.

### Faz 2 — Bilgisayar başına iki ZED stres testi

Her aday bilgisayarda hedef modda en az 15–30 dakika kayıt alınır. Şunlar
kontrol edilir:

- eksik/bozuk kare;
- yeşil veya mor kare;
- bağlantı kopması;
- gerçek FPS;
- timestamp aralıkları;
- disk yazma yetişmesi;
- sıcaklık ve GPU/CPU yükü.

Başarısız bilgisayara iki kamera zorlanmaz.

### Faz 3 — İki kameralı geometri ve depth pilotu

- ortak calibration;
- aynı eklem için iki RGB-D observation;
- depth-lift ile triangulation karşılaştırması;
- depth aykırı kapıları;
- provenance

uygulanır ve sentetik testlerle doğrulanır.

### Faz 4 — Dört kameralı ortak optimizasyon

Önce dört kamerada:

- senkronizasyon;
- ChArUco bundle adjustment;
- depth + reprojection global çözümü;
- ham/optimize karşılaştırması;
- rollback

doğrulanır. Dört kamera geçmeden on kameraya ölçeklenmez.

Not: mevcut cross-view 2B geri besleme hedef kamerayı öncülden çıkarır ve en
az dört başka destekleyici kamera ister. Bu nedenle dört kamera kalibrasyon,
triangulation, depth residual ve global optimizasyon pilotu için yeterlidir;
mevcut cross-view düzeltmenin gerçekten devreye girmesi için toplam en az beş
kamera gerekir. Bu güvenlik eşiği yalnız kamera sayısını düşürmek amacıyla
gevşetilmemelidir.

### Faz 5 — On kamera tam kayıt

- bütün kamera seri numaraları;
- ortak take manifesti;
- zaman hizalama;
- kalibrasyon drift kontrolü;
- 10 SVO2 dosya bütünlüğü;
- tam offline ViTPose;
- depth fusion;
- 3B çıktı ve videolar

üretilir.

### Faz 6 — Dış doğrulama

ZED depth veya ZED Fusion sonucu ground truth sayılmaz. Şunlardan biri gerekir:

- marker tabanlı mocap;
- ölçülmüş rijit hareketli kalibrasyon çubuğu;
- doğruluğu bilinen robotik/optik referans;
- bağımsız ve uygun poomsae 3B ground-truth sistemi.

Önce/sonra karşılaştırmasında aynı take kullanılır:

1. yalnız RGB triangulation;
2. RGB + ZED depth;
3. RGB + ZED depth + global optimizasyon;
4. paralel ZED BODY-38 Fusion.

MPJPE/açı hatası, valid oran, reprojection, temporal jitter, hız/ivme ve
kamera kanıtı birlikte raporlanır.

## 14. Faz kabul kriterleri

Bir sonraki faza ancak aşağıdakiler geçerse ilerlenir:

- bütün beklenen kamera dosyaları mevcut;
- kamera kimliği ile seri numarası eşleşiyor;
- kare sayısı ve gerçek FPS raporlu;
- açıklanamayan timestamp sıçraması yok;
- kalibrasyon fail-closed kontrolünden geçiyor;
- depth geçerlilik ve aykırı oranları raporlu;
- ham ve optimize sonuçlar ayrı;
- rollback testi mevcut;
- 133 nokta korunuyor;
- kullanılan veri ve model sürümü manifestte kayıtlı.

Görsel olarak daha sakin video tek başına başarı sayılmaz.

## 15. Kullanıcıdan karar/bilgi bekleyen noktalar

Kesin fiziksel plan ve uygulama sırası için aşağıdaki bilgiler gereklidir:

1. Kullanılabilecek güçlü bilgisayarların toplam sayısı en az beş mi?
2. Her bilgisayardaki iki ZED, HD720/60 stres testini geçebiliyor mu?
3. Çekim odasının uzunluğu, genişliği ve tavan yüksekliği nedir?
4. Tatami/aktif performans alanının ölçüsü nedir?
5. ZED 2i kameralar 2.1 mm lens mi, 4 mm lens mi?
6. Özel LED paneli yerine telefon ekranı/feneri bütün kameralarca görülebilir
   mi?

Şu kararlar verilmiştir:

- Şimdilik özel switch yoktur ve ilk pilot ağsızdır.
- Kameralar kurulduğunda sabit kalacaktır.
- İlk take yaklaşık 2 dakika ve günde bir denemedir.

Bu cevaplar alınmadan kesin kamera koordinatı, depolama kapasitesi veya
bilgisayar başına kamera sayısı varsayılmamalıdır.

## 16. Resmî kaynaklar

- [ZED çoklu kamera ve zaman senkronizasyonu](https://docs.stereolabs.com/docs/development/zed-sdk/modules/camera/multi-camera)
- [ZED Fusion API](https://docs.stereolabs.com/docs/development/zed-sdk/modules/fusion)
- [ZED360](https://docs.stereolabs.com/docs/development/zed-tools/zed-360)
- [ZED Body Tracking](https://docs.stereolabs.com/docs/development/zed-sdk/modules/body-tracking/using-the-api)
- [ZED özel nesne algılama](https://docs.stereolabs.com/docs/development/zed-sdk/modules/object-detection/custom-object-detection)
- [ZED yerel ağ akışı](https://docs.stereolabs.com/docs/development/zed-sdk/modules/camera/local-network-streaming)
- [ZED SDK sistem gereksinimleri](https://docs.stereolabs.com/docs/development/zed-sdk/specifications)
