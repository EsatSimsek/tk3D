# TK3D ZED 2i Çoklu Kamera Kurulum, Kayıt ve İşleme Rehberi

Durum: **saha kurulum rehberi — ilk tek-kamera SVO2 pilotu doğrulandı,
çok-kamera ZED çalışma zamanı henüz ana `src/` hattına eklenmedi**

Son güncelleme: **28 Temmuz 2026**

Bu belge 2–10 adet ZED 2i kamerayla, kameraların bir veya birden fazla
bilgisayara bağlı olduğu, gerçek zamanlı çalışmanın zorunlu olmadığı TK3D
puanlama sisteminin nasıl kurulacağını baştan sona açıklar.

Bu rehberin temel kararı şudur:

> ZED kameralar birbirine doğrudan bağlanmaz. Her ZED 2i USB 3 üzerinden bir
> kayıt bilgisayarına bağlanır ve kendi SVO2 dosyasını yerel diske kaydeder.
> Kayıtlar sonradan ortak zamana ve ortak dünya koordinatına getirilerek TK3D
> hattında birlikte işlenir.

Ethernet olmadan da kayıt yapılabilir. Ancak farklı bilgisayarlardaki USB ZED
2i kameralarında donanımsal ortak tetikleme bulunmadığı için bütün kameraların
gördüğü fiziksel bir ışık zaman işareti zorunlu hale gelir. Yalnız bilgisayar
saatlerine veya aynı anda elle düğmeye basmaya güvenilmez.

---

## 1. Hedef sistemin kısa özeti

Önerilen üretim hattı:

```text
2–10 × ZED 2i
  -> USB 3 ile yakın kayıt bilgisayarlarına bağlantı
  -> her kameranın yerel SSD'ye bağımsız SVO2 kaydı
  -> başlangıç/ara/bitiş görsel senkron işaretleri
  -> SVO2 dosyalarını ana işleme bilgisayarına kopyalama
  -> dosya bütünlüğü ve timestamp kontrolü
  -> ortak zaman çizelgesi
  -> ortak dünya kalibrasyonu
  -> sol RGB üzerinde RF-DETR + ViTPose-Huge WholeBody
  -> ZED stereo depth ve confidence ölçümleri
  -> robust çok-kamera triangulation
  -> depth/reprojection/anatomi/zaman kalite kapıları
  -> [T, 133, 3] metre cinsinden ortak dünya pozu
  -> dış doğrulama geçerse puanlama
```

ZED 2i'nin bu sistemde sağladıkları:

- kalibre edilmiş sol ve sağ stereo görüntü;
- sol RGB üzerinde ViTPose için görüntü kanıtı;
- sağ-sol stereo görüntüden sonradan üretilebilen depth;
- piksel başına depth confidence;
- kamera intrinsics ve kendi stereo baseline kalibrasyonu;
- kare timestamp'leri;
- IMU, sıcaklık ve diğer sensör verileri;
- SVO2 içinde ham kaydı tekrar işleme imkânı.

ZED'in sağlamadığı şeyler:

- farklı bilgisayarlardaki USB kameralar arasında donanımsal tetikleme;
- kameraların odadaki ortak konumunun otomatik ve hatasız bilgisi;
- ground-truth 3B eklem;
- tek başına resmî poomsae puanlama yetkisi.

---

## 2. Kaç kamera kullanılmalı?

| Kamera sayısı | Kullanım değeri | Puanlama açısından karar |
|---:|---|---|
| 2 | Temel triangulation ve senkronizasyon geliştirmesi | Kırılgan; resmî puanlama için yetersiz |
| 3 | İlk ortak dünya 3B denemesi | Örtüşmeye ve tek aykırı kameraya hassas |
| 4 | Orta seviye 3B ve saha pilotu | Faydalı; mevcut cross-view güvenlik hattı tam çalışmaz |
| 5 | Mevcut leave-one-camera-out tasarımının teknik alt sınırı | Kontrollü pilot için kabul edilebilir |
| 6 | Dört çevre + iki çapraz/yüksek görünüş | Önerilen başlangıç üretim düzeni |
| 7–8 | Daha iyi kapanma toleransı ve kamera yedeği | En dengeli öneri |
| 9–10 | Büyük alan, yoğun kapanma veya birden fazla kişi | En yüksek yedek; kurulum ve veri yükü artar |

TK3D bir hedef kamerayı düzeltirken o kamerayı 3B öncülden çıkarır ve en az
dört başka kameradan bağımsız görüntü kanıtı ister. Bu nedenle beş kamera
mevcut güvenlik tasarımının gerçek alt sınırıdır.

Pratik öneri:

- geliştirme sırası: `1 -> 2 -> 4 -> 6 -> gerekirse 8/10`;
- ilk güvenilir tek-sporcu saha sistemi: `6 kamera`;
- bütçe ve bilgisayar uygunsa nihai düzen: `8 kamera`;
- on kamera yalnız altı veya sekiz kameranın ölçülmüş eksiğini kapatıyorsa
  kullanılmalıdır.

Daha fazla kamera otomatik olarak daha doğru sonuç anlamına gelmez. Kötü
kalibre edilmiş veya senkronu bozuk bir kamera, sisteme faydadan çok zarar
verebilir. Her kamera güven ağırlığıyla kullanılmalı ve gerektiğinde otomatik
dışarı alınmalıdır.

---

## 3. Kameralar fiziksel olarak nasıl bağlanacak?

### 3.1 Temel bağlantı

Her ZED 2i:

```text
ZED 2i -> ZED'in uygun USB 3 kablosu -> kayıt bilgisayarı -> yerel SSD
```

Kameralar birbirine USB, HDMI veya başka bir görüntü kablosuyla bağlanmaz.
Bilgisayarlar da offline kayıt için birbirine bağlı olmak zorunda değildir.

Örnek altı kameralı dağılım:

```text
PC-01 -> C01 + C02
PC-02 -> C03 + C04
PC-03 -> C05 + C06
```

En güvenli fakat daha çok bilgisayar isteyen dağılım:

```text
PC-01 -> C01
PC-02 -> C02
...
PC-06 -> C06
```

### 3.2 Bir bilgisayara iki ZED bağlama

İki fiziksel USB port aynı USB host controller'ı paylaşabilir. Port sayısının
iki olması, iki kameranın bant genişliğinin gerçekten bağımsız olduğu anlamına
gelmez.

İki ZED aynı bilgisayarda kullanılmadan önce:

1. Her iki kamera `HD720/60 FPS` olarak aynı anda açılır.
2. En az 30 dakika eşzamanlı SVO2 kaydı yapılır.
3. Yeşil/mor kare, tearing, kamera kopması ve SDK hatası aranır.
4. Her iki dosyanın timestamp aralıkları taranır.
5. Disk yazma hızı ve boş alan kontrol edilir.
6. GPU donanım kodlayıcı oturum sınırı kontrol edilir.

Test geçmezse:

- kameraları farklı USB controller'lara dağıt;
- masaüstü bilgisayarda bağımsız kanallı PCIe USB 3 kartı kullan;
- bilgisayar başına kamera sayısını bire indir;
- aktif/kaliteli uzatma çözümü kullan;
- pasif, uzun ve markasız USB kabloyla sorunu maskeleme.

USB hub ancak üretici ve gerçek stres testiyle doğrulanır. Güçlü görünen bir
hub'ın bütün portları aynı veri yolunu paylaşabilir.

### 3.3 Ethernet yoksa

Ethernet olmaması offline kayıt için engel değildir:

- her bilgisayar yerel diske kaydeder;
- bilgisayarlar birbirinden bağımsız başlatılabilir;
- kayıtların ortak bölümü fiziksel ışık işaretiyle bulunur;
- dosyalar daha sonra harici SSD ile ana bilgisayara taşınır.

Ethernet olmadan kullanılamayan veya zorlaşan özellikler:

- merkezi canlı sağlık ekranı;
- tek düğmeyle bütün kayıt düğümlerini başlatma;
- kablolu PTP saat senkronizasyonu;
- dağıtık ZED360/Fusion Network Workflow;
- canlı çok-kamera önizleme.

Bunlar kaliteyi kolaylaştırır fakat offline işleme için zorunlu değildir.

### 3.4 Wi-Fi varsa

Kapalı bir yerel Wi-Fi ağı şu işler için kullanılabilir:

- “hazır/başlat/durdur” komutu;
- kayıt durumu ve disk alanı mesajı;
- dosya adlarının ortak `take_id` ile oluşturulması;
- kaba bilgisayar saat eşlemesi.

Wi-Fi aşağıdakiler için tek güven kaynağı değildir:

- hassas kare senkronizasyonu;
- ana SVO2 kaydını ağ üzerinden yazma;
- görüntü akışının kayıpsız olduğunun kanıtı.

Eduroam gibi kurum ağlarında cihazlar birbirini göremeyebilir. Wi-Fi jitter
ürettiği için görsel senkron işareti yine kullanılmalıdır.

---

## 4. Kamera yerleşimi

Kesin konum, oda ve aktif alan ölçüldükten sonra belirlenir. Başlangıç düzeni:

### Altı kamera

| Kamera | Konum | Yaklaşık yükseklik | Rol |
|---|---|---:|---|
| C01 | Ön-sol çapraz | 1.5–1.8 m | Ön ve sol taraf |
| C02 | Ön-sağ çapraz | 1.5–1.8 m | Ön ve sağ taraf |
| C03 | Sağ yan | 1.3–1.6 m | Sağ profil |
| C04 | Arka-sağ çapraz | 1.5–1.8 m | Arka kapanmalar |
| C05 | Arka-sol çapraz | 1.5–1.8 m | Arka kapanmalar |
| C06 | Sol yan veya yüksek çapraz | 2.2–2.8 m | Ayak/kol kesişmeleri |

### Sekiz kamera

Altı kameraya şunlar eklenir:

- karşı tarafta ikinci yüksek çapraz kamera;
- örtüşmenin en zayıf olduğu bölgeyi gören profil/ön kamera.

Yerleşim kuralları:

- sporcunun başı ve ayakları aktif alanın her noktasında kadrajda kalmalı;
- kritik eklemler mümkünse aynı anda en az dört kamerada görünmeli;
- komşu kameralar aynı kalibrasyon hedefini birlikte görebilmeli;
- bütün kameralar aynı yükseklikte tek halka oluşturmamalı;
- karşılıklı kameralar tam aynı doğru üzerinde olmak zorunda değil;
- güneş, pencere veya spot ışık doğrudan lense bakmamalı;
- parlak ve yansıtıcı zemin mümkünse mat kaplanmalı;
- tripod ayakları ve kamera yönü zeminde işaretlenmeli;
- güç ve USB kabloları sporcu alanına girmemeli;
- her kameranın gövdesine `C01`, `C02` gibi fiziksel etiket yapıştırılmalı;
- etiket ile ZED seri numarası manifestte eşleştirilmeli.

İlk kameramızın ölçülen odak değeri `3.823 mm` ve yatay görüşü `67.73°`;
4 mm ZED 2i varyantıyla uyumludur. Aynı kurulumdaki bütün kameraların lens
tipi ve görüş açısı tek tek envantere alınmalıdır.

---

## 5. Ağsız çok-bilgisayarlı senkronizasyon

### 5.1 Neden aynı anda düğmeye basmak yetmez?

Farklı bilgisayarlar:

- farklı sistem saatlerine;
- farklı USB gecikmelerine;
- farklı kamera başlangıç sürelerine;
- kayıt boyunca farklı saat drift'ine

sahip olabilir.

ZED 2i USB kameralarında harici donanımsal trigger girişi yoktur. ZED SDK
görüntü timestamp'i sağlar fakat farklı bilgisayarların saatleri fiziksel
olarak ortaklaştırılmadıysa sayıların aynı zaman tabanında olduğu
varsayılamaz.

`60 FPS` için bir kare `16.667 ms` sürer. Yalnız en yakın kareyi seçmek hızlı
tekme ve dönüşlerde ciddi 3B bozulmaya yol açabilir.

### 5.2 Gerekli senkron işareti

En iyi ağsız çözüm, bütün kameraların aynı anda gördüğü merkezi ve parlak bir
LED zaman işaretidir.

Önerilen donanım:

- büyük ve homojen yanan LED panel veya güçlü LED modül;
- sabit parlaklık;
- kameraların tamamından görünen yüksek bir konum;
- tekrar üretilebilir kısa/uzun ışık dizisi;
- mümkünse mikrodenetleyiciyle otomatik desen;
- güç kesintisinde deseni baştan başlatan basit kontrol.

Telefon feneri ilk pilotta kullanılabilir. Telefon ekranı, ekran yenilemesi
ve rolling-shutter bantları nedeniyle daha risklidir. Işığın elle açılma
gecikmesi önemli değildir; önemli olan aynı fiziksel ışık değişimini bütün
kameraların görmesidir.

### 5.3 Örnek ışık dizisi

Her olayda benzersiz bir desen göster:

```text
2 s kapalı
200 ms açık
200 ms kapalı
200 ms açık
200 ms kapalı
800 ms açık
500 ms kapalı
200 ms açık
2 s kapalı
```

Deseni:

- sporcu alana girmeden önce;
- uzun kayıtta her 30–60 saniyede;
- hareket bittikten sonra

tekrarla.

Tek bir başlangıç flaşı yalnız başlangıç ofsetini verir. Başlangıç ve bitiş
olayları birlikte kullanılırsa bilgisayar/kamera saat drift'i de ölçülebilir.
Ara olaylar uzun kayıtta doğrusal olmayan sapmaları yakalar.

### 5.4 Kayıtların ortak zamana alınması

Her kamera için:

```text
t_common = a_camera * t_camera + b_camera
```

- `b_camera`: başlangıç zaman ofseti;
- `a_camera`: kayıt boyunca saat drift'i.

İşleme sırasında:

1. Her SVO2'nin görüntü timestamp'leri çıkarılır.
2. LED'in açılış/kapanış kareleri otomatik veya elle işaretlenir.
3. Kameralar arası aynı LED olayları eşleştirilir.
4. Kamera başına `a` ve `b` çözülür.
5. 2B pozlar ortak zaman çizelgesine enterpole edilir.
6. Hızlı hareket korelasyonu yalnız ikincil ince ayar olarak kullanılır.

Sporcunun hareketini tek başına senkron referansı yapmak güvenli değildir;
ViTPose hatası ile zaman hatasını birbirine karıştırabilir.

### 5.5 Başlangıç kalite hedefleri

Bunlar TK3D için önerilen ilk mühendislik hedefleridir; dış doğrulama
sonuçlarıyla kesinleştirilecektir:

- bütün kameralarda aynı LED olaylarının eksiksiz bulunması;
- başlangıç, ara ve bitiş olayları arasında tutarlı drift modeli;
- ortak zaman sonrası p95 residual hedefi `<=5 ms`;
- açıklanamayan timestamp geri gidişi olmaması;
- kayıp karelerin ve yaklaşık fiziksel zamanlarının raporlanması.

Bu kapı geçmezse koşu tanısal olabilir fakat puanlama için yetkilendirilmez.

---

## 6. Kalibrasyon: iki farklı problem

“ZED zaten kalibre” ifadesi yalnız kameranın kendi sol-sağ stereo geometrisi
için doğrudur. Çok-kamera sisteminde iki ayrı kalibrasyon vardır.

### 6.1 Kamera içi fabrika kalibrasyonu

Her ZED seri numarasına özel olarak:

- `fx`, `fy`, `cx`, `cy`;
- distorsiyon modeli;
- sol-sağ dönüşümü;
- stereo baseline

ZED SDK'dan alınır.

Kurallar:

- bir kameranın kalibrasyon dosyası başka kameraya kopyalanmaz;
- kullanılan çözünürlüğün parametreleri alınır;
- rectified sol görüntü kullanılıyorsa rectified intrinsics kullanılır;
- SDK'nın çalışma başında döndürdüğü gerçek parametreler manifestte saklanır;
- kameralar ısındıktan sonra kalibrasyon ve kayıt yapılır;
- darbe alan veya fiziksel olarak zorlanan kamera yeniden kontrol edilir.

ZED'in başlangıç self-calibration özelliği sıcaklık ve mekanik değişimleri
düzeltebilir. Tekrarlanabilirlik için özelliği rastgele açıp kapatmak yerine
seçilen davranış bütün kamera ve oturumlarda sabitlenmeli, gerçek dönen
intrinsics kaydedilmelidir.

### 6.2 Kameralar arası extrinsic kalibrasyon

Bu kalibrasyon her kameranın ortak odadaki:

- `x, y, z` konumunu;
- yaw, pitch, roll yönünü;
- ortak metre ölçeğini;
- ortak zemin ve dünya eksenlerini

belirler.

Üretim için önerilen yöntem büyük, rijit ve ölçüsü bilinen bir ChArUco veya
AprilTag grid hedefidir.

#### Kalibrasyon hedefi

- mat ve bükülmeyen levha;
- baskı sonrası kare/tag boyutları fiziksel olarak ölçülmüş;
- mümkünse 1 metreye yakın büyük yüzey;
- kimliği ve ölçüleri bir JSON/YAML dosyasında kayıtlı;
- parlama yapmayan malzeme;
- köşeleri deforme olmayan rijit taşıyıcı.

Normal ofis kâğıdına basılmış küçük bir desen büyük hacim kalibrasyonu için
yeterli olmayabilir.

#### Offline kalibrasyon çekimi

1. Bütün kameraları son yerlerine sabitle.
2. Kameraları ve ışıkları 10–15 dakika ısıt.
3. Bütün kameralarda `HD720/60` kaydı başlat.
4. LED başlangıç senkron dizisini göster.
5. Kalibrasyon hedefini alanın merkezinde, kenarlarında ve köşelerinde gezdir.
6. Hedefi yalnız düz karşıdan değil, farklı eğim ve yönlerde göster.
7. Her komşu kamera çifti hedefi aynı anda birçok kez görsün.
8. Yakın, orta ve uzak derinlikler örneklensin.
9. Zemine yakın, göğüs yüksekliği ve baş üstü bölgeler kapsansın.
10. Kamera grafiği döngüler içersin; yalnız zincir biçiminde kalmasın.
11. LED bitiş senkron dizisini göster.
12. Kaydı durdur ve SVO2 dosyalarını doğrula.

Bir kamera diğer hiçbir kamerayla ortak hedef görmüyorsa ortak dünya
kalibrasyonuna güvenilir biçimde bağlanamaz.

#### Dünya koordinatı

TK3D sözleşmesi:

```text
x = sağ
y = ileri
z = yukarı
birim = metre
```

Önerilen dünya başlangıcı aktif alanın zemin merkezidir. Dünya yönü tatami
veya performans alanının ölçülmüş kenarlarına bağlanır. Yalnız ilk kameranın
optik merkezini dünya başlangıcı yapmak puanlama ve saha ölçümü için uygun
değildir.

#### Çözüm ve doğrulama

- bütün kameralar ortak bundle adjustment içinde çözülür;
- düşük kaliteli hedef tespitleri robust biçimde dışlanır;
- kamera başına medyan ve p95 reprojection raporlanır;
- çevrim kapanma hatası kontrol edilir;
- pozitif derinlik ve triangulation açısı kontrol edilir;
- ölçüsü bilinen bir çubuk/küp farklı bölgelerde yeniden yapılandırılır;
- zemin düzlemi ve bilinen mesafeler metre cinsinden doğrulanır;
- kalibrasyon dosyası kamera seri numaralarının SHA-256 özetiyle bağlanır.

Başlangıç hedefi:

- medyan reprojection `<1 px`;
- p95 reprojection `<2 px`;
- ölçülmüş rijit mesafe hatası uygulamanın kabul sınırı içinde;
- alanın hiçbir bölgesinde sistematik yön veya ölçek sapması olmaması.

Bu değerler dış 3B ground-truth doğruluğunun yerine geçmez.

### 6.3 ZED360'ın rolü

ZED360:

- kamera seri numaralarını görme;
- görüş örtüşmesini kontrol etme;
- ilk extrinsic tahmini;
- ZED body fusion önizlemesi

için faydalıdır.

Dağıtık bilgisayarlarda ZED360 Network Workflow için yerel ağ gerekir.
Ethernet yoksa offline ChArUco/AprilTag kalibrasyonu yapılabilir. ZED360'ın
body noktalarından oluşturduğu kalibrasyon üretim ground truth'u sayılmaz;
geometrik hedef ve ölçülü cisimle ayrıca doğrulanır.

### 6.4 Kalibrasyon ne zaman yenilenir?

Şunlardan biri olursa tam veya ilgili kamera için yeniden kalibrasyon gerekir:

- tripod veya kamera oynadı;
- kamera çıkarılıp yeniden takıldı;
- odanın/aktif alanın konumu değişti;
- kamera darbe aldı;
- lens önüne cam/koruyucu eklendi;
- çözünürlük/rectification sözleşmesi değişti;
- günlük doğrulama hedefi kalite kapısını geçmedi.

Kamera yalnız birkaç milimetre oynamış görünse bile doğrulama yapılmadan eski
kalibrasyon kullanılmaz.

---

## 7. Kayıt ayarları

Başlangıç standardı:

```text
çözünürlük: HD720 / 1280×720
FPS: 60
dosya: SVO2
kayıt: her bilgisayarın yerel SSD'sine
işleme: offline
```

Neden `HD720/60`:

- hızlı tekme ve dönüşlerde daha iyi zaman çözünürlüğü;
- ilk gerçek SVO2 pilotumuz bu modda başarıyla işlendi;
- çok-kamera USB ve disk yükü açısından 1080p'ye göre daha uygulanabilir.

### 7.1 Depth kayıt sırasında hesaplanmalı mı?

Hayır. SVO2 stereo görüntüyü ve sensör metadata'sını koruduğu için depth
sonradan farklı ZED depth modlarıyla yeniden üretilebilir. Kayıt bilgisayarının
asıl görevi kare kaybetmeden ham kaydı korumaktır.

Kayıt sırasında ağır ViTPose, Fusion ve NEURAL_PLUS çalıştırmak zorunlu
değildir. Bunlar offline ana bilgisayarda çalıştırılabilir.

### 7.2 SVO2 sıkıştırma

ZED SDK H.264, H.265 ve lossless seçenekleri destekler. Seçim gerçek stres
testiyle yapılmalıdır:

- kalibrasyon ve kısa kalite referansı: mümkünse lossless;
- uzun hareket kaydı: doğrulanmış H.265/H.264 veya disk uygunsa lossless;
- aynı bilgisayarda iki kamera: NVENC oturum sınırı ve GPU yükü ölçülmeli;
- lossy kayıt, lossless referansa göre 2B eklem doğruluğunu bozmuyorsa
  onaylanmalı.

“Dosya açılıyor” testi yeterli değildir. Kare kaybı, timestamp boşlukları ve
pose doğruluğu birlikte kontrol edilmelidir.

### 7.3 Pozlama, gain ve white balance

Kontrollü ışıkta:

- hızlı harekette motion blur üretmeyecek kısa pozlama hedeflenir;
- kameralar ısındıktan ve ışık sabitlendikten sonra ayarlar kilitlenebilir;
- bütün kameralara körlemesine aynı sayısal exposure verilmez;
- histogram, ten/kıyafet ayrımı ve blur ölçülür;
- 50 Hz şebeke aydınlatmasının flicker etkisi test edilir;
- otomatik ayar değişimleri kayıt ortasında görünümü değiştirmemeli.

Karanlık görüntüyü yalnız gain artırarak düzeltmek gürültü ve depth hatasını
artırabilir. Öncelik yeterli, homojen ve flicker üretmeyen aydınlatmadır.

---

## 8. Her çekimin standart kayıt sırası

### 8.1 Çekimden önce

1. Bütün kamera etiketlerini ve seri numaralarını kontrol et.
2. Tripod zemin işaretleriyle eşleşiyor mu kontrol et.
3. Lensleri uygun bezle temizle.
4. Güç, USB ve SSD bağlantılarını kontrol et.
5. Her bilgisayarda yeterli boş alan olduğundan emin ol.
6. ZED SDK/firmware ve kayıt uygulaması sürümünü kaydet.
7. Kameraları ve ışıkları 10–15 dakika ısıt.
8. Kadrajda baş, eller ve ayakların bütün aktif alanda kaldığını kontrol et.
9. Beş saniyelik hızlı hareketle blur ve exposure kontrolü yap.
10. O günün kısa kalibrasyon doğrulamasını yap.

### 8.2 Kaydı başlatma

Bilgisayarların aynı anda kayıt düğmesine basması gerekmez:

1. Ortak ve benzersiz bir `take_id` belirle.
2. Bütün bilgisayarlarda ilgili kameraların kaydını başlat.
3. Her operatör dosyanın gerçekten büyüdüğünü ve kare sayacının ilerlediğini
   doğrulasın.
4. Son bilgisayar da hazır olduktan sonra en az 5–10 saniye bekle.
5. Başlangıç LED zaman dizisini göster.
6. İki saniye sabit bekle.
7. Sporcu performansına başlasın.
8. Uzun kayıtta ara LED olaylarını göster.
9. Performans bitince iki saniye bekle.
10. Bitiş LED zaman dizisini göster.
11. En az 5 saniye daha kaydet.
12. Bütün bilgisayarlarda kayıtları güvenli biçimde durdur.

Bu sıra sayesinde dosyalar farklı anlarda başlamış olsa bile ortak fiziksel
bölüm sonradan bulunabilir.

### 8.3 Dosya adları

Önerilen yapı:

```text
takes/
  take_2026-07-28_001/
    manifest.json
    notes.md
    sync/
      sync_events.json
    calibration/
      cameras_world.json
      calibration_report.json
    svo/
      C01_SN39504762.svo2
      C02_SNxxxxxxxx.svo2
      C03_SNxxxxxxxx.svo2
```

Dosya adında en az:

- fiziksel kamera kimliği;
- gerçek ZED seri numarası;
- ortak `take_id`

bulunmalıdır.

SVO2 dosyası sonradan yeniden kodlanmamalı veya video düzenleyicide
kesilmemelidir. Ortak aralık manifestte tanımlanır; ham dosya korunur.

### 8.4 Manifestte tutulacak bilgiler

- session ve take kimliği;
- tarih, saat ve operatör;
- sporcu/deneme için anonim kimlik;
- kamera kimliği ve seri numarası;
- kayıt bilgisayarı kimliği;
- ZED model, firmware ve SDK sürümü;
- çözünürlük, hedef FPS ve gerçek FPS;
- sıkıştırma modu;
- exposure, gain ve white balance;
- ilk/son görüntü timestamp'i;
- başarıyla okunan kare sayısı;
- timestamp gap listesi;
- lens ve gerçek intrinsics özeti;
- SVO2 dosya boyutu ve SHA-256;
- kalibrasyon dosyasının kimliği ve SHA-256;
- senkronizasyon yöntemi;
- tripod/kamera hareketi veya çekim notu;
- ışık ve saha koşulları.

---

## 9. Kayıt bittikten sonra yapılacaklar

### 9.1 Dosyayı hemen doğrula

Her SVO2 için:

1. ZED SDK ile dosyanın açıldığını kontrol et.
2. İlk, orta ve son kareyi oku.
3. Bütün dosyayı hızlı veya tam taramayla `grab` et.
4. Raporlanan ve başarıyla okunan kare sayılarını kaydet.
5. FPS ve görüntü çözünürlüğünü doğrula.
6. Timestamp'lerin monotonluğunu kontrol et.
7. Normal kare aralığının 1.5 katını aşan boşlukları raporla.
8. Kamera seri numarası ile dosya adını karşılaştır.
9. IMU/sensör metadata'sının bulunup bulunmadığını kaydet.
10. Dosyanın SHA-256 özetini üret.

İlk gerçek pilotta SDK `667` kare raporladı fakat `666` kare başarıyla
okundu; üç yerde yaklaşık `33 ms` aralık görüldü. Bu nedenle yalnız dosya
özelliklerine bakmak yerine gerçek baştan sona tarama zorunludur.

### 9.2 Kopyalama

- ham kayıt önce kayıt bilgisayarında korunur;
- harici SSD ile ana bilgisayara kopyalanır;
- kopya sonrası SHA-256 yeniden hesaplanır;
- kaynak ve hedef özetleri eşleşmeden kaynak silinmez;
- mümkünse iki ayrı fiziksel diskte yedek tutulur;
- ham SVO2 Git'e eklenmez.

### 9.3 Ortak zaman

- LED olaylarını bütün kameralarda tespit et;
- kamera başına offset ve drift çöz;
- ortak fiziksel başlangıç/bitiş aralığını belirle;
- kayıp kareleri zaman çizelgesinde boş bırak;
- `stride > 1` kullanılsa bile gerçek video süresini koru;
- senkron kalite raporu üret.

### 9.4 Kalibrasyon doğrulaması

- kalibrasyon hedefinden ayrılmış doğrulama kareleri kullan;
- kamera başına reprojection dağılımını ölç;
- bilinen çubuk/mesafe sonuçlarını kontrol et;
- bir kamera hareket etmişse onu sessizce eski pozla kullanma;
- yeterli kamera kalıyorsa hatalı kamerayı koşudan çıkar;
- yeterli kamera kalmıyorsa puan üretme.

### 9.5 TK3D işleme sırası

1. Her kameranın rectified sol RGB görüntüsünü çıkar.
2. RF-DETR ile kişiyi bul ve kamera içinde kimliği takip et.
3. ViTPose-Huge WholeBody ile 133 adet 2B nokta üret.
4. Ham 2B ölçümleri ve skorlarını koru.
5. Sıfır-fazlı offline 2B stabilizasyon uygula.
6. ZED NEURAL depth ve confidence haritalarını üret.
7. Eklemlerin çevresinden kişi-maskeli güvenli depth örnekle.
8. Bütün 2B gözlemleri ortak zaman çizelgesine getir.
9. Ortak dünya kalibrasyonuyla robust triangulation yap.
10. Hedef kamerayı dışarıda bırakan cross-view kontrolü uygula.
11. ZED depth ile triangulation uzaklığını karşılaştır.
12. Kemik, eklem limiti, hız, ivme ve reprojection kapılarını uygula.
13. Ham triangulation'ı koruyarak global optimizasyon yap.
14. Optimizasyon kaliteyi kötüleştirirse ham sonuca dön.
15. `[T, 133, 3]` dünya koordinatlı JSON/CSV ve provenance üret.
16. Dış ground-truth ve puan yetkilendirme kapılarını çalıştır.

Depth veya 3B bulunamayan nokta:

- JSON'da `null`;
- CSV'de boş hücre

olmalıdır. `NaN` veya `inf` metni downstream çıktıya sızmamalıdır.

---

## 10. ZED depth nasıl kullanılmalı?

ZED depth değerlidir fakat tek ground-truth değildir.

Güvenli kullanım:

- ViTPose 2B eklemi görüntüden bulur;
- ZED depth eklemin kamera uzaklığına aday ölçüm verir;
- confidence haritası düşük kaliteli depth'i eler;
- kişi-depth öncülü arka plana sıçramayı azaltır;
- çok-kamera triangulation bağımsız geometri üretir;
- depth ile triangulation residual'ı kalite kanıtı olur.

Örnek:

```text
triangulation uzaklığı = 3.08 m
ZED depth             = 3.10 m
sonuç                 = güçlü uyum
```

```text
triangulation uzaklığı = 3.08 m
ZED depth             = 7.60 m
sonuç                 = muhtemel arka plan; depth reddedilir
```

İlk tam tek-kamera koşumuzda:

- BODY-17 güvenilir son 3B oranı `%95.3983`;
- WHOLEBODY-133 güvenilir son oran `%96.1245`;
- el depth oranı `%89.5369`;
- kişi medyan uzaklığı `3.0879 m`

oldu. Bu sonuç ZED depth'in güçlü olduğunu gösterir; yüz/el ve siluet
kenarlarında çok-kamera doğrulamasının hâlâ gerekli olduğunu da gösterir.

---

## 11. IMU ve diğer ZED verileri nasıl kullanılmalı?

Sabit kamera sisteminde IMU doğrudan eklem koordinatı üretmek için ana kaynak
değildir. Şu amaçlarla etkilidir:

- tripodun çekim sırasında titrediğini tespit etme;
- kameraya çarpıldığını veya yönünün değiştiğini belirleme;
- kalibrasyon geçerliliğini kapatma;
- ZED360 ilk pitch/roll tahminine yardım;
- görüntü ve sensör timestamp tutarlılığını kontrol etme.

Kullanılabilecek sağlık sinyalleri:

- IMU ani açı/hız değişimi;
- kamera sıcaklığı;
- USB bağlantı hataları;
- frame gap sayısı;
- exposure/gain değişimi;
- depth confidence dağılımı;
- kamera başına reprojection residual'ı.

Kamera sabitken IMU büyük hareket bildirirse o aralık otomatik işaretlenmeli ve
kalibrasyon yeniden doğrulanmalıdır.

---

## 12. Sık karşılaşılacak sorunlar

| Sorun | Belirti | Yapılacak |
|---|---|---|
| USB bant genişliği | Yeşil/mor kare, tearing, kopma | Farklı controller/PC, PCIe USB kartı |
| Disk yavaş/dolu | Kayıt durur veya timestamp boşlukları | Yerel SSD, önceden stres testi |
| NVENC sınırı | İkinci kayıt açılamaz | Sıkıştırmayı/topolojiyi değiştir |
| Saat drift'i | Başta iyi, sonda kötü triangulation | Başlangıç+ara+bitiş LED olayları |
| Kamera oynadı | Bölgesel reprojection artar | Kalibrasyonu geçersiz say ve yenile |
| Motion blur | ViTPose eklemleri kayar | Daha kısa pozlama ve daha güçlü ışık |
| Işık flicker'ı | Kareler arası parlaklık bantları | Flicker uyumlu aydınlatma/pozlama |
| Arka plan depth'i | El/bilek metrelerce sıçrar | Kişi maskesi, confidence, multiview residual |
| Örtüşme az | Kamera grafiği parçalanır | Yerleşimi değiştir, ortak hedef görünümü artır |
| Küçük kalibrasyon hedefi | Uzak bölgede hassasiyet düşer | Büyük, rijit ve ölçülü hedef |
| Parlak zemin | Ayak depth'i bozulur | Mat zemin/kaplama ve çapraz görünüş |
| Isınma | İlk dakikalarda kalibrasyon değişir | 10–15 dakika warm-up |
| Otomatik exposure | Kameralar farklı görünür | Kontrollü ışıkta doğrulanmış ayar kilidi |
| Kişi kimliği karışması | Kameralar farklı kişiyi eşler | Tek kişi pilotu, sonra cross-camera ID doğrulaması |
| Dosya adı karışması | Yanlış extrinsic kullanılır | Seri numarası ve SHA-256 fail-closed eşleme |
| SVO düzenleme | Timestamp/metadata kaybolur | Ham SVO2'yi değiştirme, mantıksal trim kullan |
| Ağ kopması | Merkezi önizleme gider | Asıl kaydı her zaman yerel SSD'ye yaz |
| Güç kesintisi | Eksik/bozuk dosya | UPS, kablo sabitleme, çekim sonrası tam tarama |

---

## 13. Aşamalı devreye alma planı

### Faz 1 — Tek kamera

Tamamlanan pilot:

- ZED 2i SVO2 açıldı;
- 666 kare/60 FPS işlendi;
- ViTPose 133 nokta üretildi;
- ZED NEURAL depth bağlandı;
- kamera-koordinatlı 3B video/JSON/CSV oluşturuldu.

Eksik:

- 15–30 dakikalık uzun kayıt stres testi;
- birden fazla ışık/hız/kıyafet testi;
- NEURAL ile NEURAL_PLUS kontrollü karşılaştırması.

### Faz 2 — İki kamera

- aynı bilgisayarda iki ZED USB/disk stres testi;
- farklı bilgisayarlarda ağsız LED senkron testi;
- ChArUco ile ortak dünya kalibrasyonu;
- ölçülü rijit çubukla ilk 3B hata ölçümü.

İki kamera puanlama için değil, altyapı doğrulaması içindir.

### Faz 3 — Dört kamera

- alanın dört yönünde örtüşme;
- senkron drift çözümü;
- robust triangulation ve kamera aykırı testi;
- kamera çıkarma/bozma testleri;
- uzun hareket kaydı.

### Faz 4 — Altı kamera

- mevcut cross-view güvenlik hattının tam kullanımı;
- yüksek/çapraz kameralarla el ve ayak kapanma testi;
- aynı hareketin tekrarlanabilirliği;
- bağımsız ground-truth veya ölçülü hareket referansı.

### Faz 5 — Sekiz/on kamera

- yalnız altı kameralı sistemin ölçülmüş açıklarını kapatmak;
- bir kamera kaybında güvenli devam testi;
- bilgisayar, disk ve veri taşıma kapasitesi testi;
- tam poomsae ve uzun süreli saha testi.

Her faz, önceki fazın kalite raporu geçmeden büyütülmez.

---

## 14. Puanlamayı açmadan önce zorunlu kapılar

Başlangıç mühendislik hedefleri:

- beklenen bütün kamera dosyaları mevcut;
- seri numarası ve kalibrasyon eşleşiyor;
- dosyaların SHA-256 bütünlüğü geçiyor;
- timestamp ve LED senkron raporu geçiyor;
- medyan reprojection `<1 px`, p95 `<2 px` hedefi;
- BODY-17 geçerli oranı `>%95`;
- kritik eklemde yeterli bağımsız kamera kanıtı;
- ham ve optimize 3B ayrı saklanıyor;
- optimizasyon rollback testi geçiyor;
- tek kamera çıkarıldığında sonuç kabul sınırında kalıyor;
- JSON/CSV sözleşmesi `[T, 133, 3]`, metre ve ortak dünya;
- dış 3B ground-truth değerlendirmesi geçiyor;
- skor tekrar edilebilirliği ve uzman/hakem değerlendirmesi geçiyor.

Sistem şu durumlarda puan üretmemelidir:

- kalibrasyon yaklaşık veya seri numarasıyla eşleşmiyor;
- senkron residual kalite kapısını geçmiyor;
- kamera sayısı veya bağımsız görüntü kanıtı yetersiz;
- kritik eklemler uzun süre kapanmış;
- kamera hareketi/IMU alarmı var;
- ölçüm yalnız ZED depth veya yalnız 3B izdüşümden geliyor;
- dış doğrulama yetkisi yok.

İç reprojection değerlerinin iyi olması ground-truth doğruluğunu tek başına
kanıtlamaz.

---

## 15. Satın alınması veya hazırlanması faydalı parçalar

Zorunlu/öncelikli:

- sağlam ve kilitlenebilir tripod veya duvar/tavan aparatı;
- her kamera ve kablo için fiziksel kimlik etiketi;
- yeterli yerel SSD kapasitesi;
- kaliteli ZED uyumlu USB 3 kabloları;
- kablo sabitleme ve güvenli güç dağıtımı;
- büyük, rijit ChArUco/AprilTag kalibrasyon levhası;
- merkezi LED senkron işareti;
- mezura/lazer mesafe ölçer ve su terazisi;
- harici veri taşıma SSD'si;
- mümkünse UPS.

Topolojiye göre:

- bağımsız kanallı PCIe USB 3 kartları;
- ek kayıt bilgisayarları;
- özel yerel switch ve kablolu ağ;
- PTP destekli ağ kartları;
- homojen, flicker üretmeyen saha aydınlatması.

Ucuz bir özel Ethernet switch merkezi kontrolü kolaylaştırır, ancak USB ZED
2i'lere donanımsal kamera trigger özelliği kazandırmaz. Görsel senkron işareti
yine doğrulama amacıyla tutulmalıdır.

---

## 16. Günlük saha kontrol listesi

### Kurulum

- [ ] Kamera sayısı ve `C01–C10` etiketleri doğru.
- [ ] Seri numarası eşlemeleri doğru.
- [ ] Tripod/zemin işaretleri eşleşiyor.
- [ ] Lensler temiz.
- [ ] USB ve güç kabloları sabit.
- [ ] Aktif alan tüm kritik kameralarda görünür.
- [ ] LED senkron işareti tüm kameralarda görünür.
- [ ] SSD boş alanı yeterli.
- [ ] Kamera ve ışık warm-up tamamlandı.

### Kalite

- [ ] Exposure, gain ve white balance kontrol edildi.
- [ ] Hızlı harekette blur kabul edilebilir.
- [ ] Günlük kalibrasyon doğrulaması geçti.
- [ ] Kamera hareketi/IMU alarmı yok.
- [ ] İki kameralı bilgisayarlarda USB sağlık testi geçti.

### Kayıt

- [ ] Ortak `take_id` belirlendi.
- [ ] Bütün kayıtların gerçekten ilerlediği görüldü.
- [ ] Başlangıç LED dizisi kaydedildi.
- [ ] Gerekli ara LED dizileri kaydedildi.
- [ ] Bitiş LED dizisi kaydedildi.
- [ ] Kayıtlar güvenli biçimde kapatıldı.

### Kayıt sonrası

- [ ] Her SVO2 baştan sona açıldı.
- [ ] Kare/FPS/çözünürlük/timestamp raporu üretildi.
- [ ] SHA-256 alındı.
- [ ] Ana bilgisayara kopya doğrulandı.
- [ ] Ham kaynak silinmedi.
- [ ] Senkron ve kalibrasyon raporları geçti.
- [ ] Koşu benzersiz `run_id` altında işlendi.

---

## 17. Mevcut TK3D durumu ve geliştirilmesi gereken parçalar

Bugün doğrulanmış olan:

- tek ZED 2i SVO2 okuma;
- gerçek 60 FPS zaman çizelgesi;
- RF-DETR + ViTPose-Huge WholeBody;
- ZED NEURAL depth ve confidence;
- kamera-koordinatlı `[666, 133, 3]` çıktı;
- güvenilirlik filtresi, smoothing ve 3B görselleştirme;
- mevcut çok-kamera RGB triangulation/optimization altyapısı.

Henüz kalıcı olarak geliştirilmesi gereken:

1. çoklu SVO2 take manifesti ve kamera seri numarası doğrulaması;
2. ağsız LED olay algılayıcı ve offset/drift çözücüsü;
3. offline ChArUco/AprilTag extrinsic kalibrasyon aracı;
4. SVO2 sol RGB/depth/confidence/IMU giriş adaptörü;
5. ZED depth residual'ını çok-kamera triangulation'a güvenli bağlama;
6. kamera hareketi ve kayıt sağlığı raporu;
7. çok-kamera gerçek ZED smoke/stres testleri;
8. bağımsız 3B ground-truth ve puanlama yetkilendirmesi.

Bu parçalar tamamlanmadan “kameraları bağladık, sistem puanlamaya hazır”
denmemelidir.

---

## 18. İlk gerçek çok-kamera denemesi için önerilen net plan

İlk adımda on kamerayı aynı anda kurma. Şu deneyi yap:

1. İki ZED 2i seç.
2. Mümkünse farklı bilgisayarlara bağla.
3. Her bilgisayarda yerel SSD'ye `HD720/60 SVO2` kaydet.
4. Ortak LED'i başta, ortada ve sonda göster.
5. Büyük ChArUco hedefini ortak görüşte ve alan boyunca gezdir.
6. Ölçüsü bilinen rijit bir çubuğu farklı yönlerde hareket ettir.
7. Kısa, hızlı bir tekvando hareketi kaydet.
8. Dosyaları SHA-256 ile ana bilgisayara taşı.
9. Timestamp offset/drift ve extrinsic kalibrasyonu çöz.
10. İki-kamera 3B ile ZED depth'i karşılaştır.
11. Sonuç raporu geçerse dört, ardından altı kameraya çık.

Bu iki-kamera deneyi şu sorulara ölçülü cevap verir:

- bilgisayarlar kare kaybediyor mu;
- LED ile zaman drift'i çözülebiliyor mu;
- kalibrasyon hedefi yeterince büyük mü;
- oda ve kamera açıları uygun mu;
- ZED depth ile çok-kamera geometri ne kadar uyuşuyor;
- gerçek depolama ihtiyacı ne kadar;
- kaç bilgisayara gerçekten ihtiyaç var.

---

## 19. Resmî kaynaklar

- [ZED çoklu kamera kurulumu ve USB zaman senkronizasyonu](https://docs.stereolabs.com/docs/development/zed-sdk/modules/camera/multi-camera)
- [ZED SVO/SVO2 kayıt sistemi](https://docs.stereolabs.com/docs/development/zed-sdk/modules/camera/recording)
- [ZED kamera fabrika kalibrasyonu](https://docs.stereolabs.com/docs/development/zed-sdk/modules/camera/camera-calibration)
- [ZED360 çok-kamera kalibrasyonu](https://docs.stereolabs.com/docs/development/zed-tools/zed-360)
- [ZED Fusion API](https://docs.stereolabs.com/docs/development/zed-sdk/modules/fusion)
- [ZED depth ayarları ve confidence filtreleme](https://docs.stereolabs.com/docs/development/zed-sdk/modules/depth-sensing/depth-settings)
- [ZED depth modları](https://docs.stereolabs.com/docs/development/zed-sdk/modules/depth-sensing/depth-modes)

Repository içindeki daha teknik mimari ayrıntılar:

- `docs/ZED2I_OFFLINE_MULTICAMERA_PLAN.md`
- `docs/ARCHITECTURE_DECISIONS.md`
- `PROJECT_STATUS.md`
