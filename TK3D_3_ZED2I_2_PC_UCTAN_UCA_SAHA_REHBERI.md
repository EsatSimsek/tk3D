# TK3D — 3 ZED 2i / 2 PC Uçtan Uca Saha Uygulama Rehberi

> **FUTURE FIELD PLAN:** Bu rehber gelecekteki 3-ZED / 2-PC saha düzeni
> içindir. Güncel `CURRENT_ACTIVE` üretim/araştırma workflow'u değildir;
> mevcut kanonik hat tek bilgisayarda işlenen iki ZED 2i kaydını kullanır.

Son güncelleme: **31 Temmuz 2026**

Hedef kullanıcı: Tek operatör
İşletim sistemi: **İki bilgisayarda Windows varsayılmıştır**
Kamera düzeni: **3 × ZED 2i, 2 bilgisayar**
Kayıt standardı: **HD720, 60 FPS, SVO2, yerel SSD, offline işleme**

Bu belgeyi baştan sona izleyerek donanımı hazırlayabilir, kameraları
yerleştirebilir, ilk testleri yapabilir, üç yerel SVO2 kaydı alabilir,
dosyaları doğrulayabilir ve TK3D işleme için güvenli bir take hazırlayabilirsin.

Bu belge iki çalışma biçimini açıkça ayırır:

1. **Bugün uygulanabilir manuel yöntem:** ZED SDK araçlarıyla iki
   bilgisayarda yerel kayıt.
2. **Geliştirilecek nihai yöntem:** PC-1'de tek tuşla üç kameranın
   hazırlanması, başlatılması, durdurulması ve doğrulanması.

> Önemli: Repository'de dağıtık tek-tuş kayıt uygulaması ve kalıcı çoklu ZED
> ingest/senkron adaptörü henüz yoktur. Bu nedenle bu rehberde “bugün
> yapılabilir” ve “uygulama geliştirildikten sonra” adımları ayrı gösterilir.

---

## 1. Son mimari karar

```text
PC-1 — ana kayıt ve işleme bilgisayarı
  ├─ C01 ZED 2i -> USB 3 -> PC-1 yerel SSD -> C01 SVO2
  └─ C02 ZED 2i -> USB 3 -> PC-1 yerel SSD -> C02 SVO2

PC-2 — kayıt düğümü
  └─ C03 ZED 2i -> USB 3 -> PC-2 yerel SSD -> C03 SVO2

PC-1 <---- telefon hotspot'u / özel Wi-Fi ----> PC-2
              yalnız kontrol ve durum

Çekim sonrası:
  C03 SVO2 -> doğrulanmış kopya -> PC-1
  C01 + C02 + C03 -> offline zaman hizalama
  -> kalibrasyon -> ViTPose-133 -> 3B TK3D
```

Ana ilkeler:

- Ana kayıtlar ağ üzerinden değil, kameranın bağlı olduğu yerel SSD'ye yazılır.
- PC-1 merkezî kontrol bilgisayarıdır.
- Nihai kullanımda operatör yalnız PC-1'de tek düğmeye basar.
- Ağ koparsa üç yerel kayıt mümkün olduğunca devam eder.
- Ham SVO2 dosyalarının üzerine yazılmaz veya video editöründe kesilmez.
- Ham dosyaların kare sayısı birkaç kare farklı olabilir.
- TK3D'ye verilen hizalı zaman çizelgesi aynı başlangıç, bitiş ve uzunluktadır.
- Kayıp kareler çoğaltılarak veya zaman çizelgesi kısaltılarak gizlenmez.
- ZED360 ve Fusion kullanılır; fakat tek doğruluk veya zaman kaynağı sayılmaz.

---

## 2. Mevcut sistemde doğrulanmış ve eksik olanlar

### Bu bilgisayarda doğrulanmış olanlar

- ZED SDK araçları kurulu:
  - `C:\Program Files (x86)\ZED SDK\tools\ZED Studio.exe`
  - `C:\Program Files (x86)\ZED SDK\tools\ZED360.exe`
  - `C:\Program Files (x86)\ZED SDK\tools\ZED Explorer.exe`
  - `C:\Program Files (x86)\ZED SDK\tools\ZED Diagnostic.exe`
  - `C:\Program Files (x86)\ZED SDK\tools\ZED Sensor Viewer.exe`
- ZED Python API mevcut sanal ortamda açılabiliyor.
- ZED SDK `5.4.1` ile gerçek ZED 2i SVO2 pilotu işlendi.
- `1280×720`, `60 FPS`, 666 karelik ilk SVO2 başarıyla okundu.
- IMU, sol RGB, NEURAL depth ve ViTPose WholeBody pilotu çalıştı.

### Henüz geliştirilmemiş olanlar

- PC-1/PC-2 dağıtık `ARM/START/STOP` kayıt servisi;
- PC-1'de tek tuşlu kayıt arayüzü;
- otomatik SVO2 tam tarama ve manifest üretimi;
- görsel senkron olay algılayıcısı;
- iki host arasındaki offset/drift çözücüsü;
- üç ZED SVO2'yi kalıcı TK3D girişine bağlayan adaptör;
- ZED360 ağ publisher yapılandırmasını otomatik kuran uygulama.

Bugün manuel yöntemle veri toplanabilir. “Tek tuş” kullanımı için önce bu
yazılım parçaları geliştirilmelidir.

---

## 3. Gerekli donanım ve hazırlık listesi

Zorunlu:

- 3 × ZED 2i;
- 2 × yeterli Windows bilgisayar;
- her kameranın uygun ve sağlam USB 3 kablosu;
- PC-1'de iki kamerayı aynı anda taşıyabilecek USB controller topolojisi;
- iki bilgisayarda yeterli yerel SSD alanı;
- 3 × sabit tripod veya sağlam kamera aparatı;
- kamera ve kablo etiketleri;
- telefon hotspot'u veya bilgisayardan açılan özel erişim noktası;
- dosya taşımak için yeterli kapasiteli harici SSD;
- büyük, rijit ChArUco/AprilTag kalibrasyon hedefi;
- ölçüsü bilinen rijit çubuk;
- mezura veya lazer mesafe ölçer;
- homojen ve flicker üretmeyen ışık.

Yararlı:

- UPS;
- PC-1 için bağımsız kanallı PCIe USB 3 kartı;
- kablo sabitleme aparatları;
- zemine kamera/tripod işareti koymak için sökülebilir bant;
- iki ucunda parlak renkli işaret bulunan yaklaşık 1 metre çubuk.

Kullanılmayacak veya güvenilmeyecek şeyler:

- ucuz, doğrulanmamış USB hub;
- uzun, pasif ve markasız USB uzatma;
- ham SVO2'yi doğrudan Wi-Fi ağ diskine yazma;
- yalnız eduroam komut zamanını hassas senkron sayma;
- yalnız aynı anda düğmeye basmaya güvenme.

---

## 4. Kamera kimliklendirme

Kameraların USB indeksleri kalıcı değildir. Her kamera seri numarasıyla
eşleştirilmelidir.

Fiziksel etiket tablosunu ilk kurulumda doldur:

| Kamera | Seri numarası | Bilgisayar | Fiziksel konum | USB portu |
|---|---|---|---|---|
| C01 | SDK otomatik okuyacak | PC-1 | ön-sol çapraz | doldur |
| C02 | SDK otomatik okuyacak | PC-1 | ön-sağ çapraz | doldur |
| C03 | SDK otomatik okuyacak | PC-2 | arka/yan çapraz | doldur |

Seri numaralarını görmek için her bilgisayarda ZED Explorer'ı açabilir veya
şu PowerShell komutunu kullanabilirsin:

```powershell
& "C:\Program Files (x86)\ZED SDK\tools\ZED Explorer.exe" --all
```

Yapılacaklar:

1. C01, C02 ve C03 etiketlerini kameraya yapıştır.
2. Aynı etiketi USB kablosunun iki ucuna koy.
3. Seri numarasını tabloya yaz.
4. Tripod konumunu ve baktığı yönü zeminde işaretle.
5. Kamera taşınırsa seri numarası değişmese bile extrinsic kalibrasyonu
   geçersiz say.

---

## 5. İki bilgisayarı hazırlama

Her iki bilgisayarda:

1. Aynı ana ZED SDK sürümünü kur.
2. NVIDIA sürücüsünün ZED SDK ile uyumlu olduğunu doğrula.
3. Windows güncellemesini çekim sırasında otomatik yeniden başlamayacak şekilde
   planla.
4. Bilgisayarı prize bağla.
5. Uyku ve hazırda bekletmeyi çekim süresince kapat.
6. Hedef yerel diskte yeterli boş alan bırak.
7. ZED araçlarının açıldığını kontrol et.
8. Windows Güvenlik Duvarında yalnız kullanılacak özel ağ ve uygulamalara izin
   ver.
9. İki bilgisayarda tarih, saat ve saat diliminin doğru olduğunu kontrol et.
10. İki bilgisayarın saat dilimi `Europe/Istanbul`/Türkiye saati olsa da
    senkronun yalnız görünen saate dayanmadığını unutma.

Bu bilgisayarda ZED Python API kontrolü:

```powershell
Set-Location -LiteralPath "C:\Users\WWWW\Desktop\tk3d"
& "C:\Users\WWWW\Desktop\tk3d\.venv312\Scripts\python.exe" -c "import pyzed.sl as sl; print('ZED Python API hazır:', sl.__name__)"
```

İkinci bilgisayarda repository veya aynı sanal ortam yoksa bu Python komutu
zorunlu değildir. ZED Studio/Explorer ile manuel kayıt yapılabilir. Tek tuşlu
uygulama geliştirildiğinde PC-2'ye onun paketlenmiş worker kurulumu yapılacaktır.

---

## 6. Her kamerayı tek başına doğrulama

Her kamera için ayrı ayrı:

1. Diğer ZED'leri çıkar.
2. Kamerayı doğrudan USB 3 porta bağla.
3. `ZED Diagnostic.exe` çalıştır.
4. USB bağlantısının USB 3 olarak görüldüğünü doğrula.
5. Kamera firmware'ini ve seri numarasını kaydet.
6. `ZED Explorer.exe` ile `HD720 / 60 FPS` aç.
7. Görüntüde yeşil/mor bozulma olmadığını kontrol et.
8. `ZED Sensor Viewer.exe` ile IMU verisinin aktığını kontrol et.
9. Kamerayı 10–15 dakika açık bırak.
10. Isındıktan sonra görüntü ve sensörleri tekrar kontrol et.
11. Kısa bir SVO2 kaydı al ve yeniden aç.

Araçları PowerShell'den açmak istersen:

```powershell
Start-Process -FilePath "C:\Program Files (x86)\ZED SDK\tools\ZED Diagnostic.exe"
Start-Process -FilePath "C:\Program Files (x86)\ZED SDK\tools\ZED Explorer.exe"
Start-Process -FilePath "C:\Program Files (x86)\ZED SDK\tools\ZED Sensor Viewer.exe"
```

Her kamera şu kapıları geçmelidir:

- seri numarası okunuyor;
- `HD720/60` açılıyor;
- görüntü bozuk değil;
- bağlantı kopmuyor;
- IMU mevcut;
- SVO2 yazılıyor ve tekrar açılıyor.

---

## 7. PC-1'de iki kamera stres testi

İki fiziksel USB port aynı controller'ı paylaşabilir. İki port görülmesi iki
bağımsız veri yolu olduğu anlamına gelmez.

### Test

1. C01 ve C02'yi PC-1'e bağla.
2. ZED Studio'yu aç:

```powershell
Start-Process -FilePath "C:\Program Files (x86)\ZED SDK\tools\ZED Studio.exe"
```

3. İki kameranın seri numarasını kontrol et.
4. İkisini de `HD720 / 60 FPS` aç.
5. İki stream'i `Shift` ile seç.
6. `Record` bölümünde hedef sıkıştırmayı seç.
7. Her dosyayı PC-1'in yerel SSD'sine yaz.
8. En az 30 dakika kesintisiz kaydet.
9. Kayıt sırasında ağır ViTPose/depth/Fusion çalıştırma.
10. Windows Görev Yöneticisinden disk, GPU encoder, CPU ve sıcaklığı izle.
11. Kayıtları güvenli biçimde kapat.
12. İki SVO2'yi ZED Studio'da baştan, ortadan ve sondan oynat.
13. Daha sonra tam SDK taramasıyla bütün kare/timestamp raporunu üret.

Test aşağıdaki durumda başarısızdır:

- kamera kopması;
- yeşil/mor veya bozuk kare;
- SDK `grab` hatası;
- normal kare aralığının `1.5 katını` aşan açıklanamayan boşluk;
- disk yazma yetişmemesi;
- ikinci H.264/H.265 encoder oturumunun açılamaması;
- uygulamanın takılması;
- bilgisayarın ısınma nedeniyle ciddi yavaşlaması.

Başarısızsa sırayla:

1. C01/C02'yi farklı USB portlarına taşı.
2. USB controller yollarını değiştir.
3. Bağımsız kanallı PCIe USB kartı kullan.
4. Sıkıştırma modunu kontrollü değiştir.
5. Kamera dağılımını ters çevir: başka bilgisayar iki kamerayı alsın.
6. Hiçbiri geçmezse üçüncü kayıt bilgisayarı kullan.

---

## 8. SVO2 kayıt biçimi ve sıkıştırma

SVO2:

- stereo görüntüyü;
- görüntü timestamp'lerini;
- IMU/sensör metadata'sını;
- yeniden oynatma bilgisini

korur. Depth ve ilgili SDK çıktıları playback sırasında yeniden üretilebilir.

SVO2 otomatik olarak lossless değildir:

| Mod | Kullanım |
|---|---|
| LOSSLESS | Kısa kalite/kalibrasyon referansı |
| H.264 / H.265 | Uzun kayıt için küçük dosya; kayıplı olabilir |
| H.264 LOSSLESS / H.265 LOSSLESS | Daha büyük fakat yüksek kaliteli seçenek |

İlk gün:

1. Aynı kısa hareketi bir lossless referansla çek.
2. Aynı hareketi hedef uzun-kayıt sıkıştırmasıyla çek.
3. Dosya boyutu, encoder yükü, timestamp gap ve ViTPose 2B kalitesini
   karşılaştır.
4. Sonuç ölçülmeden “H.265 yeterli” veya “lossless zorunlu” kararı verme.

Depth'i kayıt sırasında çalıştırmak zorunlu değildir. Kayıt bilgisayarlarının
ana görevi görüntüleri kare kaybetmeden yerel SVO2'ye yazmaktır.

---

## 9. Telefon hotspot ağı

Telefon hotspot'u eduroam'a göre daha kontrol edilebilir bir yerel ağ sağlar.
Mobil veri açık olmak zorunda değildir; doğrudan cihaz iletişimi yeterlidir.

Önerilen:

```text
SSID: TK3D-CAPTURE
PC-1: koordinatör
PC-2: kayıt worker
İnternet: isteğe bağlı
```

### Kurulum

1. Telefonda hotspot'u aç.
2. PC-1 ve PC-2'yi bu ağa bağla.
3. Windows'ta ağı `Özel ağ` olarak işaretle.
4. Telefon ekranı kapanınca hotspot'un kapanmadığını doğrula.
5. Mobil veri kapanınca yerel cihaz iletişiminin sürdüğünü test et.
6. İki bilgisayarın IP adresini kaydet.

IP bilgisini görmek için her bilgisayarda:

```powershell
Get-NetIPConfiguration |
  Where-Object { $_.IPv4Address -ne $null } |
  Select-Object InterfaceAlias,@{Name="IPv4";Expression={$_.IPv4Address.IPAddress}}
```

PC-1'den PC-2 temel erişim testi:

```powershell
Test-Connection -ComputerName "<PC2-IP-ADRESI>" -Count 4
```

Ping güvenlik duvarınca engellenebilir. Tek tuşlu kayıt servisi
geliştirildiğinde asıl test servis portuyla yapılacaktır:

```powershell
Test-NetConnection -ComputerName "<PC2-IP-ADRESI>" -Port <KAYIT-SERVISI-PORTU>
```

Eduroam cihazların birbirini görmesine izin verirse kullanılabilir; fakat
rehber bunu garanti kabul etmez. Hotspot da cihazlar arası trafiği engellerse
manuel kayıt akışına geç.

### PTP hakkında

Telefon hotspot'u veya eduroam Wi-Fi üzerinde PTP kullanma. Stereolabs,
Wi-Fi adaptörlerinin PTP için gerekli gönderim timestamp desteğini
sağlamadığını belirtir. Windows/NTP/Chrony kaba saati iyileştirebilir fakat
mikrosaniyelik kamera senkronizasyonu kanıtı değildir.

---

## 10. Kayıt yöntemleri

### 10.1 Bugün uygulanabilir manuel yöntem

Bu yöntem için özel TK3D kayıt yazılımı gerekmez.

1. PC-2'de C03'ü ZED Studio/Explorer ile aç.
2. C03 yerel SVO2 kaydını başlat.
3. Dosya boyutunun ve kare sayacının arttığını doğrula.
4. PC-1'de C01 ve C02'yi ZED Studio ile aç.
5. İki stream'i seçip iki yerel SVO2 kaydını başlat.
6. Üç kameranın kayıt yaptığını gör.
7. 10 saniye ön kayıt al.
8. Bölüm 13'teki senkron hareketini yap.
9. Performansı kaydet.
10. Bitiş senkron hareketini yap.
11. 10 saniye son kayıt al.
12. PC-1'de C01/C02 kayıtlarını güvenli kapat.
13. PC-2'de C03 kaydını güvenli kapat.

Başlatma ve durdurma düğmelerine aynı anda basman gerekmez. Ortak fiziksel
aralık sonradan bulunacaktır.

### 10.2 Geliştirilecek PC-1 tek-tuş yöntemi

Hedef kullanıcı deneyimi:

```text
PC-1 ekranı
  Hazırla
  ├─ C01 ARMED
  ├─ C02 ARMED
  └─ C03 ARMED

  Kaydı Başlat
  ├─ C01 RECORDING -> PC-1 SSD
  ├─ C02 RECORDING -> PC-1 SSD
  └─ C03 RECORDING -> PC-2 SSD

  Kaydı Durdur
  ├─ C01 FINALIZED
  ├─ C02 FINALIZED
  └─ C03 FINALIZED
```

Durum makinesi:

```text
IDLE -> PREFLIGHT -> ARMED -> RECORDING -> FINALIZING -> VERIFIED
Her aşama -> ERROR (neden manifestte)
```

Uygulama:

- kameraları seri numarasıyla açmalı;
- ortak `take_id` dağıtmalı;
- her kameranın yerel SSD'sini kontrol etmeli;
- üç `ARMED` cevabı olmadan çekimi başlatmamalı;
- dosya boyutu ve kare sayacını izlemeli;
- ağ koparsa yerel kaydı sürdürmeli;
- stop komutunda her SVO2'nin güvenli kapanmasını beklemeli;
- dosya, kare, timestamp ve hash raporu üretmeli.

Bu uygulama repository'de henüz yoktur. Hazır olmadan rehberde geçen tek-tuş
komutlarını varmış gibi çalıştırmaya çalışma.

### 10.3 Neden üç ana dosyayı PC-1'e ağdan kaydetmiyoruz?

ZED Studio birden fazla ağ stream'ini merkezde SVO2'ye kaydedebilir. Fakat
telefon Wi-Fi'sinde:

- paket kaybı;
- bağlantı kopması;
- değişken gecikme;
- PC-1'de iki USB kamera + ağ decode + üç kayıt yükü;
- C03 ana kaydının ağ kesintisine bağlı olması

ek risk oluşturur.

Bu nedenle:

```text
Merkezî kontrol: evet
Merkezî izleme: test geçerse evet
Üç ana kaydı yalnız PC-1'e yazma: hayır
Her kamerayı yerel SSD'ye yazma: evet
```

---

## 11. Kamera yerleşimi

Başlangıç düzeni:

```text
                    C03
              arka/yan çapraz

          +-------------------+
          |                   |
          |    aktif alan     |
          |                   |
          +-------------------+

       C01                     C02
   ön-sol çapraz           ön-sağ çapraz
```

Başlangıç önerisi:

- yaklaşık `1.4–1.8 m` kamera yüksekliği;
- C01/C02 arasında geniş fakat ortak görüşü koruyan açı;
- C03'ün arka tarafı ve C01/C02'de kapanan eklemleri görmesi;
- sporcunun baş, el ve ayaklarının aktif alanın tamamında kadrajda kalması;
- gövdenin mümkünse her anda üç kamerada, en az iki kamerada görünmesi;
- doğrudan güçlü ışığa bakmayan kameralar;
- mat ve kaymayan zemin;
- tripod ayaklarının bantla işaretlenmesi;
- bütün kameraların senkron hareket alanını görmesi.

Kritik sınır:

- Üç kamera temel triangulation yapabilir.
- Bir kamera kapanınca yalnız iki kamera kalır ve sonuç kırılganlaşır.
- Mevcut TK3D cross-view geri besleme bir hedef kamera için en az dört başka
  kamera ister; üç kamera bu özelliği tam çalıştırmaz.
- Üç kameralı sistem dış doğrulama olmadan puanlamaya hazır sayılmaz.

---

## 12. ZED360 ve Fusion kullanımı

### 12.1 Görev ayrımı

ZED360:

- ZED Body Tracking noktalarından kamera dizisinin ortak dünya pozlarını
  tahmin eder;
- birleşik gövdeyi görsel olarak kontrol etmeyi sağlar;
- kalibrasyon JSON'u üretir.

Fusion:

- ZED body/object/spatial verilerini publish/subscribe ile birleştirir;
- canlı sağlık önizlemesi ve ZED fused-body karşılaştırması sağlar.

TK3D:

- rectified görüntülerden ViTPose WholeBody 133 üretir;
- ZED depth ve çok-kamera triangulation'ı birleştirir;
- nihai `[T, 133, 3]` metre cinsinden ortak dünya çıktısını üretir.

Fusion Body sonucu TK3D'nin 133 noktalı çıktısının yerine geçmez.

### 12.2 Bugün en kolay ZED360 testi: Local Workflow

Kameraları oynatmadan ve PC-1 USB controller testi izin verirse:

1. Üç kamerayı geçici olarak PC-1'e bağla.
2. `ZED360.exe` aç:

```powershell
Start-Process -FilePath "C:\Program Files (x86)\ZED SDK\tools\ZED360.exe"
```

3. `Auto Discover` seç.
4. Üç seri numarasını doğrula.
5. `Setup the room` seç.
6. Kalibrasyon alanında yalnız bir kişi bırak.
7. Kişi alanın tamamında yavaşça yürüsün.
8. Ayak bilekleri görünür kalsın.
9. Yaklaşık her 10 saniyelik optimizasyonda gövdelerin hizalanmasını izle.
10. Gövdeler iyi hizalandığında `Finish calibration` seç.
11. JSON'u benzersiz adla kaydet.
12. Kameraları oynatmadan normal `2+1` bağlantıya geri dön.

USB kablolarını değiştirmek veya kamerayı fiziksel olarak oynatmak kalibrasyonu
geçersiz kılabilir. Bu nedenle bu yol ancak kameralar sabit kalırken kablo
erişimi mümkünse uygulanır.

### 12.3 İki bilgisayarlı ZED360 Network Workflow

Bu akış **deneyseldir** ve repository'de sender uygulaması henüz yoktur.

Gerekli yapı:

```text
PC-1:
  C01 Fusion publisher -> ayrı port
  C02 Fusion publisher -> ayrı port
  ZED360/Fusion subscriber

PC-2:
  C03 Fusion publisher -> ayrı port
```

Kurallar:

- bütün publisher'larda aynı ZED SDK ana sürümü;
- aynı body formatı;
- aynı coordinate system ve metre birimi;
- her kamera için seri numarası;
- her publisher için farklı port;
- genel ZED video streaming yerine Fusion API `startPublishing()`;
- PC-1'in iki publisher + subscriber yük testi;
- güvenlik duvarı port izinleri;
- telefon hotspot'unda en az 30 dakika ağ testi.

Stereolabs'ın web rehberi ZED Hub adımlarını göstermeye devam eder; fakat
Stereolabs desteği genel ZED Hub–ZED360 akışının deprecated olduğunu
bildirmiştir. Yeni sistem ZED Hub'a üretim bağımlılığı kurmamalıdır.

Ağ ZED360 çalışmazsa:

1. yerel SVO2 kaydı iptal edilmez;
2. hedef tabanlı kalibrasyon ana yol olur;
3. ZED360 ayrı subscriber veya doğrulanmış yerel ağ olduğunda yeniden denenir.

---

## 13. Görüntü tabanlı senkron olayı

Ön ve arka kameralar kişinin farklı yüzlerini görür; aynı görüntüyü görmeleri
gerekmez. Aynı fiziksel hareketin hız ve durma anlarını görmeleri yeterlidir.

Senkron dizisi:

```text
2 saniye sabit dur
iki kolu hızla baş üstüne kaldır
1 saniye sabit tut
iki kolu hızla aşağı indir
derin çömel ve hızla ayağa kalk
bu diziyi toplam 3 kez yap
2 saniye sabit dur
```

Mümkünse iki ucunda parlak renkli işaret bulunan bir çubuğu yatay tut ve
başının üzerine kaldır. Çubuğun ön/arka yüzü olmadığı için bütün açılardan
görülür.

Senkron olayını:

- performanstan önce;
- performanstan sonra;
- 10 dakikadan uzun kayıtta yaklaşık her 5 dakikada bir

yap.

Eller, omuzlar, kalça ve ayaklar üç kamerada da görünmelidir. Üç kamera aynı
kişiyi ortak bir bölgede göremiyorsa ZED360 kalibrasyonu ve triangulation da
sağlıklı değildir; kamera yerleşimini değiştir.

---

## 14. Standart çekim günü sırası

### Çekim öncesi

1. Telefon hotspot'unu aç.
2. PC-1 ve PC-2'yi bağla.
3. Bilgisayarları prize tak.
4. Windows uyku/yeniden başlatma durumunu kontrol et.
5. C01/C02/C03 seri numaralarını doğrula.
6. Tripod ve zemin işaretlerini kontrol et.
7. Lensleri uygun bezle temizle.
8. USB kablolarını sabitle.
9. SSD boş alanını kontrol et.
10. Kameraları ve ışıkları 10–15 dakika ısıt.
11. `HD720/60` modunu doğrula.
12. Pozlama, gain, white balance ve motion blur kontrolü yap.
13. Sporcunun aktif alanda baştan ayağa üç kamerada kaldığını kontrol et.
14. Günlük kalibrasyon doğrulamasını yap.
15. Benzersiz `take_id` oluştur.

### Kayıt

1. Üç kamerayı yerel SVO2 kayda al.
2. Üç dosyanın büyüdüğünü doğrula.
3. En az 10 saniye bekle.
4. Başlangıç senkron dizisini yap.
5. İki saniye sabit bekle.
6. Performansı çek.
7. İki saniye sabit bekle.
8. Bitiş senkron dizisini yap.
9. En az 10 saniye bekle.
10. Kayıtları güvenli biçimde kapat.

### Kayıt sonrası

1. Dosya boyutlarının sıfırdan büyük olduğunu kontrol et.
2. Her SVO2'yi ilk/orta/son kareden aç.
3. Her SVO2'yi SDK ile baştan sona tara.
4. Timestamp monotonluğu ve gap listesini üret.
5. Kamera seri numarası ve dosya adını eşleştir.
6. IMU metadata'sını kontrol et.
7. SHA-256 üret.
8. C03'ü harici SSD ile PC-1'e taşı.
9. Kopya SHA-256 değerini kaynakla karşılaştır.
10. Kaynak dosyaları hemen silme.

---

## 15. Take dizin yapısı

Ham take'leri repository içine koyma. Örnek harici SSD yapısı:

```text
E:\TK3D_TAKES\
  take_2026-07-31_001\
    manifest.json
    notes.md
    svo\
      C01_SN<serial>.svo2
      C02_SN<serial>.svo2
      C03_SN<serial>.svo2
    sync\
      network_clock_samples.json
      visual_sync_events.json
      frame_alignment.csv
      synchronization_report.json
    calibration\
      zed360_candidate.json
      charuco_candidate.json
      cameras_production.json
      calibration_report.json
    hashes\
      sha256.txt
```

Klasör oluşturma örneği:

```powershell
$TakeRoot = "E:\TK3D_TAKES\take_2026-07-31_001"
New-Item -ItemType Directory -Force -Path "$TakeRoot\svo" | Out-Null
New-Item -ItemType Directory -Force -Path "$TakeRoot\sync" | Out-Null
New-Item -ItemType Directory -Force -Path "$TakeRoot\calibration" | Out-Null
New-Item -ItemType Directory -Force -Path "$TakeRoot\hashes" | Out-Null
Write-Host "Take klasörü hazır: $TakeRoot"
```

`E:` örnektir. Gerçek harici SSD harfini kullan. Var olan take klasörünün
üzerine yazma; her çekime yeni `take_id` ver.

---

## 16. Dosya taşıma ve SHA-256

PC-2'de C03 kaynağının hash'ini al:

```powershell
$SourceFile = "D:\TK3D_TAKES\take_2026-07-31_001\svo\C03_SN12345678.svo2"
Get-FileHash -Algorithm SHA256 -LiteralPath $SourceFile |
  Format-List Algorithm,Hash,Path
```

Harici SSD'ye kopyala:

```powershell
$SourceFile = "D:\TK3D_TAKES\take_2026-07-31_001\svo\C03_SN12345678.svo2"
$DestinationFile = "E:\TK3D_TAKES\take_2026-07-31_001\svo\C03_SN12345678.svo2"
Copy-Item -LiteralPath $SourceFile -Destination $DestinationFile
Get-FileHash -Algorithm SHA256 -LiteralPath $SourceFile
Get-FileHash -Algorithm SHA256 -LiteralPath $DestinationFile
```

İki hash eşleşmeden:

- PC-2 kaynağını silme;
- dosyayı işleme için kabul etme;
- `latest_run.json` veya benzeri işaretçiyi güncelleme.

Ham kaydı mümkünse iki ayrı fiziksel diskte koru.

---

## 17. Offline zaman hizalama

### Neden yalnız aynı anda başlatmak yetmez?

60 FPS'te bir kare yaklaşık `16.67 ms` sürer. Ağ komutu, kamera açılışı veya
kayıp kare birkaç kare fark oluşturabilir. Hızlı tekmede bu fark yanlış 3B
üretir.

### Timestamp modeli

C01 ve C02 aynı host saatini, C03 başka host saatini kullanır. PC-2 zamanı
PC-1 ortak eksenine dönüştürülür:

```text
t_ortak = a * t_PC2 + b
```

- `b`: başlangıç saat farkı;
- `a`: kayıt boyunca saat drift'i.

Kanıtlar:

- ağ istek-cevap ölçümleri: kaba öncül;
- başlangıç senkron hareketi;
- ara senkron hareketleri;
- bitiş senkron hareketi;
- ortak gövde hareketinin hız/ivme korelasyonu.

### Kare eşleme kuralları

1. PC-1 ortak 60 Hz zaman ızgarası oluşturulur.
2. C01/C02 gerçek image timestamp'leriyle eşlenir.
3. C03 timestamp'leri `a` ve `b` ile düzeltilir.
4. En yakın kare yalnız tolerans içindeyse kabul edilir.
5. Eşleme monoton ve bire birdir.
6. Aynı kamera karesi iki fiziksel ana sessizce kopyalanmaz.
7. Tolerans aşılırsa o kamera gözlemi eksik işaretlenir.
8. Ham ve düzeltilmiş timestamp ile residual kaydedilir.

Ortak çıktı:

```text
T_start = üç kameranın doğrulanmış ortak başlangıcı
T_end   = üç kameranın doğrulanmış ortak bitişi
T       = T_start ... T_end, 60 Hz
```

Ham SVO2 dosyaları farklı uzunlukta kalabilir. Hizalı TK3D zaman çizelgesi
aynı uzunluktadır.

Başlangıç mühendislik hedefi görsel senkron residual'ının `<= 0.5 kare`
olmasıdır. Gerçek pilotta ölçülmeden “frame-synchronized” denmez.

---

## 18. Ortak dünya kalibrasyonu

### 18.1 ZED fabrika kalibrasyonu

Her ZED'in fabrika dosyası yalnız kendi sol/sağ kamera intrinsics ve stereo
geometrisini verir. Üç kameranın odadaki konumunu vermez.

### 18.2 ZED360 adayı

ZED360:

- yalnız bir kişiyle;
- ortak görüşte;
- alanın tamamında yavaş yürüyüşle;
- ayak bilekleri görünürken

kamera pozu adayı üretir. JSON seri numarası, tarih, SDK sürümü ve SHA-256 ile
saklanır.

### 18.3 ChArUco/AprilTag bağımsız çözümü

1. Büyük, rijit ve ölçüsü doğrulanmış hedef kullan.
2. Hedefi üç kameranın ortak gördüğü bölgelerde gezdir.
3. Alanın önü, arkası, merkezi ve kenarlarını kapsa.
4. Zemine yakın, bel ve göğüs yüksekliğinde örnekler al.
5. Farklı yaw/pitch açıları kullan.
6. Parlama ve motion blur olan kareleri ele.
7. Kalibrasyonda kullanılmayan ayrı doğrulama kareleri tut.

### 18.4 Ölçülü çubuk doğrulaması

1. Uzunluğu hassas ölçülmüş rijit çubuğun uçlarını belirgin işaretle.
2. Çubuğu farklı konum, yükseklik ve yönlerde tut.
3. Üç-kamera 3B uç mesafesini çöz.
4. Hesaplanan uzunluğu gerçek uzunlukla karşılaştır.
5. Alanın yalnız merkezinde değil kenarlarında da ölç.

Üretim kalibrasyonu:

- reprojection dağılımı;
- ölçülü çubuk 3B hatası;
- alan boyunca tutarlılık;
- seri numarası eşleşmesi

kapılarını geçmelidir.

ZED360 ve hedef çözümü farklıysa körlemesine ZED360 seçilmez. Ölçümde daha iyi
olan, provenance'i açık aday seçilir.

Kalibrasyon şu durumda geçersizdir:

- kamera veya tripod oynadı;
- kamera başka konuma taşındı;
- kamera değişti;
- çözünürlük/rectification sözleşmesi değişti;
- günlük doğrulama kapısı geçmedi.

---

## 19. TK3D işleme sırası

```text
3 SVO2
  -> dosya/seri/hash doğrulaması
  -> image timestamp çıkarımı
  -> offset + drift + görsel olay hizalama
  -> ortak 60 Hz zaman çizelgesi
  -> production calibration doğrulaması
  -> rectified sol RGB
  -> RF-DETR kişi tespiti
  -> ByteTrack kamera içi kimlik
  -> ViTPose-Huge WholeBody 133
  -> ham 2B ölçümlerin korunması
  -> offline 2B stabilizasyon
  -> ZED NEURAL depth + confidence
  -> kişi maskeli güvenli depth örnekleme
  -> üç-kamera robust triangulation
  -> reprojection/depth/anatomi/zaman kapıları
  -> ham triangulation korunarak optimizasyon
  -> [T, 133, 3] metre, ortak dünya
```

Üç kamera için:

- üç bağımsız görüntü: tercih edilen;
- iki güvenilir kamera: triangulation mümkün;
- tek kamera: ortak dünya triangulation değildir;
- tek-kamera ZED depth: yalnız işaretli RGB-D fallback/önizleme;
- görüntü kanıtı yok: nokta eksik.

Eksik değer:

- JSON: `null`;
- CSV: boş hücre.

`NaN` veya `inf` downstream çıktıya yazılmaz.

---

## 20. Kalite raporları ve kabul kapıları

Her take için:

- beklenen üç SVO2 mevcut;
- seri numarası doğru;
- SHA-256 mevcut ve kopyalar eşleşiyor;
- çözünürlük/FPS doğru;
- başarıyla okunan kare sayısı raporlu;
- timestamp monoton;
- timestamp gap listesi mevcut;
- başlangıç/bitiş senkron olayları bulundu;
- offset ve drift raporlu;
- hizalama residual'ı raporlu;
- üretim kalibrasyonu geçerli;
- reprojection dağılımı raporlu;
- ölçülü çubuk kontrolü raporlu;
- BODY-17/133 geçerli oranı raporlu;
- kamera kanıtı ve eksik kareler raporlu;
- ham triangulation korunmuş;
- optimize sonuç ve rollback raporlu;
- çıktı `[T, 133, 3]`, metre ve ortak dünya.

Şu durumda puanlama yapılmaz:

- bir kamera dosyası eksik;
- seri numarası/kalibrasyon uyuşmuyor;
- senkron residual kapıyı geçmiyor;
- kalibrasyon yaklaşık veya kamera oynadı;
- kritik eklem uzun süre tek kamerada;
- sonuç yalnız ZED depth veya Fusion gövdesine dayanıyor;
- dış 3B doğrulama/yetkilendirme yok.

İyi reprojection tek başına ground-truth doğruluğu değildir.

---

## 21. Sık sorunlar

| Sorun | Belirti | Çözüm |
|---|---|---|
| PC-1 USB yükü | Kamera kopması, bozuk kare | Farklı controller, PCIe USB kartı, dağılımı değiştir |
| SSD yetersiz | Kayıt durur/yavaşlar | Yerel hızlı SSD, alan aç, stres testi |
| Encoder sınırı | İkinci kayıt açılamaz | Başka sıkıştırma veya GPU/PC dağılımı |
| Hotspot kopması | PC-2 durumu kaybolur | Yerel kayıt sürsün, manuel stop |
| Eduroam izolasyonu | PC'ler birbirini görmez | Telefon/PC hotspot'u veya manuel akış |
| Saat offset'i | Aynı hareket farklı karede | Görsel olay + affine offset/drift |
| Kayıp kare | 33 ms veya daha büyük gap | Eksik işaretle, tekrar çekimi değerlendir |
| ZED360 ağ bağlantısı | Timeout/crash/yanlış body format | Aynı SDK/body format, `startPublishing`, portlar; olmazsa hedef kalibrasyonu |
| Kamera oynadı | Reprojection bölgesel artar | Kalibrasyonu geçersiz say, yeniden yap |
| Motion blur | Eklem kayması | Kısa pozlama, daha güçlü ışık |
| Depth arka plana sıçrar | El/bilek metrelerce uzak | Kişi maskesi, confidence, multiview residual |
| Ön/arka senkron hareketi zor | Görüntüler farklı | Düşey tam-vücut hareketi veya çift uçlu çubuk |
| SVO2 açılamıyor | Playback/grab hatası | Kaynağı silme, SDK sürümü ve tam tarama |
| C03 kopyası bozuk | Hash farklı | Yeniden kopyala, kaynak korunmalı |

---

## 22. Aşamalı uygulama planı

### Faz A — üç tekil kamera testi

Çıkış: Üç kamera ayrı ayrı `HD720/60` SVO2 kaydediyor.

### Faz B — PC-1 çift kamera stres testi

Çıkış: C01+C02 en az 30 dakika kararlı kayıt yapıyor.

### Faz C — manuel üç kamera take

Çıkış: Üç yerel SVO2, hash ve ortak senkron hareketi mevcut.

### Faz D — offline zaman hizalama

Çıkış: Ortak 60 Hz zaman çizelgesi ve residual raporu.

### Faz E — ortak dünya kalibrasyonu

Çıkış: ZED360 adayı, hedef tabanlı aday ve ölçülü doğrulama.

### Faz F — üç-kamera TK3D pilotu

Çıkış: ViTPose-133, depth, triangulation ve kalite raporları.

### Faz G — tek-tuş kayıt uygulaması

Çıkış: PC-1'den `ARM/START/STOP`, ağ kopma testi ve otomatik manifest.

### Faz H — uzun saha testi

Çıkış: Gerçek hedef süre, ara senkron olayları, sıcaklık/disk/ağ raporu.

### Faz I — dış doğrulama

Çıkış: MPJPE/açı/temporal ölçümler ve puanlama yetkilendirme kararı.

Her faz bir önceki fazın raporu geçmeden büyütülmez.

---

## 23. İlk gün uygulanacak net pilot

1. Üç kamerayı C01/C02/C03 olarak etiketle.
2. Seri numaralarını kaydet.
3. Her kamerayı tek başına Diagnostic/Explorer/Sensor Viewer ile test et.
4. PC-1'de C01+C02 ile 30 dakikalık stres kaydı yap.
5. Stres testi geçerse üç kamerayı saha konumlarına yerleştir.
6. Telefon hotspot'unu kur.
7. PC-1/PC-2 bağlantısını test et.
8. Üç kamerayı 10–15 dakika ısıt.
9. Kadraj, ışık ve motion blur kontrolü yap.
10. Üç yerel SVO2 kaydını manuel başlat.
11. 10 saniye bekle.
12. Başlangıç senkron dizisini üç kez yap.
13. Büyük kalibrasyon hedefini alan boyunca gezdir.
14. Ölçülü çubuğu farklı yer/yönlerde tut.
15. Yavaş hareket çek.
16. Hızlı tekme ve dönüş çek.
17. Bitiş senkron dizisini üç kez yap.
18. 10 saniye bekle.
19. Üç kaydı güvenli kapat.
20. Üç dosyayı baştan sona doğrula.
21. Hash al ve C03'ü PC-1'e doğrulanmış biçimde taşı.
22. Offset, drift, gap ve kalibrasyon raporlarını üret.
23. Raporlar geçerse ilk üç-kamera TK3D işleme pilotuna geç.

Bu pilot geçmeden önemli veya uzun poomsae çekimi yapma.

---

## 24. Resmî kaynaklar

- [Stereolabs — Çoklu kamera ve USB timestamp hizalama](https://docs.stereolabs.com/docs/development/zed-sdk/modules/camera/multi-camera)
- [Stereolabs — SVO/SVO2 kayıt](https://docs.stereolabs.com/docs/development/zed-sdk/modules/camera/recording)
- [Stereolabs — ZED Studio çoklu ve ağ stream kaydı](https://docs.stereolabs.com/docs/development/zed-tools/zed-studio)
- [Stereolabs — Sensör ve görüntü timestamp'leri](https://docs.stereolabs.com/docs/development/zed-sdk/modules/sensors/time-synchronization)
- [Stereolabs — Timestamp senkron penceresi](https://docs.stereolabs.com/docs/development/zed-sdk/modules/global-localization/data-synchronization)
- [Stereolabs — ZED360](https://docs.stereolabs.com/docs/development/zed-tools/zed-360)
- [Stereolabs — Fusion API](https://docs.stereolabs.com/docs/development/zed-sdk/modules/fusion)
- [Stereolabs Support — ZED Hub–ZED360 deprecated durumu](https://community.stereolabs.com/t/zed360-with-zed-hub/8670)

Repository içindeki teknik arka plan:

- `ZED2I_COKLU_KAMERA_SAHA_REHBERI.md`
- `docs/ZED2I_OFFLINE_MULTICAMERA_PLAN.md`
- `PROJECT_STATUS.md`
- `docs/ARCHITECTURE_DECISIONS.md`
