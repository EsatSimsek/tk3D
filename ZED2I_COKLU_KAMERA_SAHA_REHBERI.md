# TK3D — 3 ZED 2i / 2 Bilgisayar Saha Kurulum ve Kayıt Rehberi

> **FUTURE FIELD PLAN:** Bu rehber gelecekteki 3-ZED / 2-PC saha düzeni
> içindir. Güncel `CURRENT_ACTIVE` üretim/araştırma workflow'u değildir;
> mevcut kanonik hat tek bilgisayarda işlenen iki ZED 2i kaydını kullanır.

Durum: **uygulanacak saha yol haritası — kayıt ve senkron yazılımı henüz
geliştirilmedi**

Son güncelleme: **31 Temmuz 2026**

Bu belge şu kesin saha düzeni için hazırlanmıştır:

```text
PC-1 (ana bilgisayar) -> ZED C01 + ZED C02 -> PC-1 yerel SSD
PC-2 (kayıt düğümü)   -> ZED C03           -> PC-2 yerel SSD

PC-1 <---- telefon erişim noktası / yerel Wi-Fi ----> PC-2
```

Üç kamera aynı hareket alanını farklı açılardan çeker. İki bilgisayar aynı
odadadır ve tek operatör tarafından yönetilir. Ethernet yoktur. Bilgisayarlar
eduroam kullanabilir; fakat hassas senkronizasyon ve kayıt güvenliği eduroam'a
bağlı olmayacaktır.

Başlangıç kayıt standardı, aksi ölçülerek gerekmedikçe:

```text
1280x720 (HD720), 60 FPS, SVO2, yerel SSD, offline işleme
```

Kayıt süresi henüz belirtilmediği için depolama hesabı ve uzun stres testinin
süresi ilk pilotta ölçülecektir.

---

## 0. İncelenen seçenekler ve nihai mimari karar

Bu rehber, ZED360/Fusion hakkında yapılan dış sohbet ve güncel Stereolabs
belgeleri karşılaştırılarak revize edilmiştir.

### 0.1 Dış sohbette önerilen yol

Sohbette sırasıyla şu mimari önerilmiştir:

1. kameraları farklı bilgisayarlardan ZED360 Network Workflow'a bağlamak;
2. ZED Hub üzerinden cihazları merkezi bilgisayara getirmek;
3. ağ kameralarını merkezi ZED Studio'da birlikte SVO2 kaydetmek;
4. bilgisayar saatlerini PTP/Chrony ile eşitlemek;
5. Fusion'ın timestamp'lere bakarak kareleri kusursuz hizalamasına güvenmek;
6. nihai olarak SVO2'leri ZED SDK ile okuyup en yakın timestamp'li kareleri
   doğrudan TK3D'ye vermek.

Bu listenin **altıncı maddesindeki doğrudan SVO2 okuma fikri doğrudur**.
Diğer maddelerin bazıları bizim iki Windows bilgisayar + telefon/eduroam Wi-Fi
düzenimiz için uygun veya doğrulanmış değildir.

### 0.2 Düzeltilen yanlış veya aşırı kesin iddialar

| İddia | Teknik gerçek ve karar |
|---|---|
| ZED Hub zorunludur | Fusion API doğrudan IP/port ve `startPublishing()` ile yerel ağda çalışabilir. Ayrıca Stereolabs desteği, genel ZED Hub–ZED360 akışının geciktiğini ve deprecated olduğunu bildirmiştir. ZED Hub'a bağımlı olunmayacak. |
| Telefon Wi-Fi'sinde PTP ile mikrosaniye senkron yapılır | Stereolabs Wi-Fi adaptörlerinin PTP için gerekli timestamp desteğini sağlamadığını açıkça belirtir. PTP ancak uygun Ethernet/NIC düzeninde ayrı bir gelecek yükseltmesidir. |
| ZED 2i her kareye diğer bilgisayarlarla ortak donanım zamanı basar | ZED 2i USB kameralar arasında ortak hardware trigger yoktur. SDK görüntü timestamp'i host sistem saatine bağlı epoch zamanıdır; farklı host saatleri ayrıca hizalanmalıdır. |
| Fusion farklı host saatlerini otomatik düzeltip kusursuz fiziksel eşleme yapar | Fusion yakın timestamp'leri bir senkron penceresinde gruplar. Yanlış host saatini veya bilinmeyen drift'i sihirli biçimde düzeltemez. Girdi saatleri ortak değilse eşleşme fiziksel olarak yanlış olabilir. |
| Merkezi ZED Studio'da ağ kayıtları en güvenli yöntemdir | ZED Studio ağ akışlarını kaydedebilir; fakat Wi-Fi kaybı ana görüntü kaydını etkiler. Bizde her kamera bağlı olduğu bilgisayarın yerel SSD'sine kaydedecek. |
| SVO2 depth/3B veriyi hazır biçimde saklar ve her zaman kayıpsızdır | SVO2 temel olarak stereo görüntü ile timestamp/IMU gibi metadata'yı saklar; depth yeniden üretilebilir. H.264/H.265 modları kayıplı olabilir, lossless ayrıca seçilir. |
| Yalnız en yakın timestamp'i seçmek yeterlidir | Önce PC-1–PC-2 saat offset'i ve drift'i çözülmelidir. Sonra monoton, tekil ve tolerans kapılı kare eşleme yapılır; kayıp kareler boş bırakılır. |

### 0.3 Nihai önerilen mimari

```text
KAYIT — ana ve güvenli yol
  PC-1: C01 + C02 -> iki yerel SVO2
  PC-2: C03       -> bir yerel SVO2
  Wi-Fi: yalnız ARM/START/STOP/durum ve kaba saat ölçümü

SENKRON — üretim yolu
  SVO2 image timestamp'leri
  + PC-1/PC-2 offset-drift tahmini
  + başlangıç/ara/bitiş ortak tam-vücut hareketleri
  -> offline ortak zaman çizelgesi ve kalite raporu

KALİBRASYON
  ZED360/Fusion network pilotu -> kamera pozu adayı ve canlı tanı
  ChArUco/AprilTag + ölçülü çubuk -> bağımsız üretim doğrulaması

İŞLEME
  SVO2'den doğrudan rectified görüntü dizileri
  -> ViTPose WholeBody-133
  -> robust triangulation + ZED depth kontrolü
  -> [T, 133, 3] ortak dünya çıktısı
```

ZED360/Fusion kullanılacaktır; ancak kayıt güvenliğinin, hassas zaman
hizalamanın veya 133 noktalı nihai çıktının tek sahibi olmayacaktır.

---

## 1. En önemli kararlar

### 1.1 Kameralar nasıl bölünecek?

- **PC-1**, iki ZED 2i kaydeder ve merkezi kontrolü yürütür.
- **PC-2**, üçüncü ZED 2i'yi kaydeder.
- Her kamera kendi SVO2 dosyasını yalnız bağlı olduğu bilgisayarın yerel
  SSD'sine yazar.
- Canlı görüntü veya SVO2 dosyası Wi-Fi üzerinden taşınmaz.
- Çekimden sonra PC-2'deki dosya harici SSD ile ana işleme bilgisayarına
  taşınır ve SHA-256 ile doğrulanır.

İki kamerayı PC-1'e takmak fiziksel olarak mümkün olsa bile iki USB portunun
aynı USB controller'ı paylaşmadığı ölçülmeden kabul edilmez. Stereolabs, aynı
bilgisayardaki çoklu USB ZED'lerin farklı USB controller'larına dağıtılmasını
önerir.

### 1.2 Wi-Fi ne için kullanılacak?

Wi-Fi yalnız şunlar için kullanılır:

- PC-2'yi `hazırla`, `kaydı başlat`, `kaydı durdur` komutları;
- PC-2'den `hazırım`, `kayıt başladı`, `dosya büyüyor`, `hata var` cevapları;
- iki bilgisayar arasındaki kaba saat farkının ölçülmesi;
- ortak `take_id` ve çekim notlarının dağıtılması.

Wi-Fi şunların kanıtı değildir:

- kameraların aynı anda pozlandığının;
- karelerin hassas biçimde senkron olduğunun;
- kaydın kayıpsız olduğunun;
- SVO2 dosyalarının doğru ve eksiksiz kapandığının.

### 1.3 Aynı anda başlamak ve durmak zorunlu mu?

Hayır. Dosyaların aynı anda başlaması veya aynı uzunlukta olması gerekmez.
Bütün kameraların asıl performanstan önce ve sonra ortak görüntü kaydetmesi
yeterlidir.

Operatör açısından kayıt yine **PC-1'de tek tuşla** yönetilecektir. PC-1:

1. C01, C02 ve uzaktaki C03'e `ARM` gönderir;
2. üç kameradan hazır cevabı alır;
3. tek `START` olayıyla üç yerel kaydı başlatır;
4. tek `STOP` olayıyla üç yerel kaydı güvenli kapatır.

Bu merkezî kontrol uygulaması henüz repository'de geliştirilmemiştir. Rehber,
uygulanacak veri ve güvenlik sözleşmesini tanımlar.

Doğru yaklaşım:

1. Önce üç kamera da kayda girer.
2. En az 10 saniye ön kayıt alınır.
3. Görüntüde ortak senkron hareketi yapılır.
4. Performans çekilir.
5. Bitişte senkron hareketi tekrarlanır.
6. En az 10 saniye son kayıt alınır.
7. Kameralar sırayla güvenli biçimde durdurulur.

Sonradan yalnız üç dosyanın ortak ve doğrulanmış zaman aralığı işlenir. Ham
SVO2 dosyaları kesilmez veya yeniden kodlanmaz.

Ham SVO2 dosyalarının ilk/son timestamp'i ve kare sayısı birkaç kare farklı
olabilir. Teslim edilen hizalı TK3D zaman çizelgesi ise tek ortak aralıkta ve
aynı uzunlukta olacaktır:

```text
T_start = üç kameradaki doğrulanmış ortak başlangıç
T_end   = üç kameradaki doğrulanmış ortak bitiş
T       = T_start ... T_end, 60 Hz ortak zaman ızgarası
```

Bir kamerada ortak ızgaranın bir anına ait kare yoksa zaman çizelgesi
kısaltılmaz ve eski kare sessizce kopyalanmaz. O kamera gözlemi o anda eksik
olarak işaretlenir. Böylece üç çıktı aynı süreyi taşırken kare kaybı gizlenmez.

---

## 2. Neden telefon erişim noktası yararlı?

Eduroam, kurumun güvenlik ayarlarına göre istemcilerin birbirine doğrudan
erişmesini engelleyebilir. Bu durumda iki bilgisayar internete çıkabilse bile
birbirini göremez. Ayrıca ağ gecikmesi değişkendir.

Telefonun erişim noktası iki bilgisayara küçük ve kontrol edilebilir bir yerel
ağ sağlar. **Mobil veri zorunlu değildir**; erişim noktasının yerel ağ kısmı
yeterlidir. Önerilen saha ağı:

```text
Telefon erişim noktası: TK3D-CAPTURE
PC-1: ağa bağlı, ana kontrol uygulaması
PC-2: ağa bağlı, kayıt düğümü
İnternet: isteğe bağlı
```

İlk kurulumda şu test yapılır:

1. İki bilgisayar erişim noktasına bağlanır.
2. PC-2'nin yerel IP adresi bulunur.
3. PC-1'den PC-2 kayıt servisinin sağlık adresine erişilir.
4. PC-2 Windows Güvenlik Duvarı yalnız bu özel ağdaki uygulamaya izin verir.
5. Telefon ekranı kapanınca veya mobil veri kapanınca bağlantının sürüp
   sürmediği test edilir.
6. En az 30 dakika boyunca durum mesajlarının kopmadığı doğrulanır.

Telefon erişim noktası cihazlar arası trafiği engelliyorsa ikinci tercih,
PC-1'in Windows “Mobil etkin nokta” özelliğidir. O da çalışmazsa rehberdeki
manuel yedek akış kullanılır. Eduroam'a özel güvenlik kuralını aşmaya
çalışılmaz.

---

## 3. Zaman damgası ve gerçek senkronizasyon

### 3.1 ZED 2i'nin sınırı

ZED 2i bir USB kameradır ve farklı USB ZED kameralarını ortak pozlama anında
tetikleyecek bir donanım trigger girişi sunmaz. Aynı bilgisayara bağlı C01 ve
C02 bile tam olarak aynı anda pozlanmış kabul edilmez.

Stereolabs'a göre USB ZED kameralarında hizalama görüntü zaman damgalarıyla
yapılır. ZED görüntü paketleri ana bilgisayara geldiğinde, bilgisayarın sistem
saatine göre epoch zaman damgası alır. Bu nedenle:

- C01 ve C02 aynı PC saatini kullanır; hizalamaları daha kolaydır.
- C03, PC-2 saatini kullanır; PC-1 ile saat farkı ve drift ölçülmelidir.
- Telefon Wi-Fi'si donanımsal PTP değildir.
- Windows saatinin internette “doğru görünmesi” kare düzeyinde senkron
  garantisi vermez.

60 FPS'te bir kare yaklaşık `16.67 ms` sürer. Hızlı tekmelerde bir karelik
yanlış eşleme bile 3B sonucu belirgin biçimde bozabilir.

PTP bu saha düzeninde kurulmayacaktır. Stereolabs'ın PTP akışı Ethernet,
`linuxptp` ve uygun NIC timestamp desteğini esas alır; Wi-Fi adaptörlerinin PTP
için gerekli gönderim timestamp kabiliyetini sağlamadığını belirtir. Chrony
veya Windows/NTP saatini iyileştirebilir fakat telefon hotspot'unda
mikrosaniyelik kamera senkronizasyonu kanıtı değildir. Gelecekte kablolu,
PTP-doğrulanmış Linux ağ kurulursa bu karar yeniden değerlendirilebilir.

### 3.2 Üç katmanlı senkron modeli

Senkronizasyon üç ayrı kanıta dayanacaktır:

#### Katman A — ağ üzerinden kayıt koordinasyonu

PC-1 her iki düğüme ortak bir `take_id` gönderir. PC-2 komutu aldığını
onaylar. İki bilgisayar da önce `ARMED`, sonra `RECORDING` durumuna geçer.

Bu katman operatör işini kolaylaştırır; hassas kare eşlemesi sağlamaz.

#### Katman B — saat farkı ölçümü

Kayıt uygulaması, çekim öncesinde ve kayıt sırasında PC-1 ile PC-2 arasında
küçük istek-cevap paketleri gönderir. Her ölçümde:

- PC-1 gönderim zamanı;
- PC-2 alış ve cevap zamanı;
- PC-1 dönüş zamanı;
- gidiş-dönüş gecikmesi;
- tahmini saat farkı

manifestte saklanır. Yüksek gecikmeli ölçümler elenir; kalan ölçümler kaba
offset ve drift başlangıç tahmini verir.

Bu yazılım ölçümü PTP değildir ve tek başına kesin sonuç sayılmaz.

#### Katman C — görüntü içeriğinden doğrulama ve düzeltme

LED kullanılamadığı için aktif alanın ortasında, öndeki ve arkadaki
kameralardan da görülebilen belirgin bir **senkron hareket dizisi** yapılır:

```text
2 saniye sabit dur
iki kolu hızla yukarı kaldır
1 saniye sabit tut
iki kolu hızla aşağı indir
derin çömel ve hızla ayağa kalk
aynı diziyi iki kez daha tekrarla
2 saniye sabit dur
```

Ön ve arka görüntünün birbirine benzemesi gerekmez. Omuz, bilek ve kalçanın
aynı fiziksel anda yaptığı düşey hareketin hız/ivme tepe noktaları eşlenir.
Mümkünse iki ucunda parlak renkli işaret bulunan bir çubuk yatay tutulup baş
üstüne kaldırılır. Tek yüzlü bir telefon veya pano yerine her yönden görülen
tam-vücut hareketi tercih edilir. Elektrikli LED gerekmez. Hareket aktif alanın
merkezinde yapılır; eller, omuzlar, kalça ve ayaklar üç görüntüde de görünür
olmalıdır.

Bu dizi:

- asıl performanstan önce;
- performanstan sonra;
- 10 dakikadan uzun kayıtlarda yaklaşık her 5 dakikada bir

tekrarlanır. Başlangıç ve bitiş olaylarının farkı, bilgisayar saatlerindeki
drift'i ölçmeye yarar.

Offline senkron çözücüsü omuz, bilek, pano köşesi veya genel optik akıştaki
hareket tepe noktalarını eşler. Ağ ölçümünü yalnız arama öncülü sayar; nihai
offset ve drift görüntü kanıtıyla doğrulanır.

### 3.3 Offline zaman eşleme algoritması

Yalnız “C01 timestamp'ine en yakın C03 karesi” seçilmeyecektir. PC-2'nin
timestamp ekseni önce PC-1 ortak zamanına taşınır:

```text
t_ortak = a * t_PC2 + b

b = başlangıçtaki saat offset'i
a = kayıt boyunca saat drift'i
```

`a` ve `b` şu kanıtlardan birlikte kestirilir:

- ağ istek-cevap saat örnekleri: yalnız kaba başlangıç öncülü;
- başlangıç görsel hareketi;
- varsa ara görsel hareketler;
- bitiş görsel hareketi;
- performans boyunca ortak gövde hareketinin hız/ivme korelasyonu.

Sonra:

1. PC-1 zaman çizelgesi referans alınır.
2. C01 ve C02 aynı host saatinde olsalar bile gerçek timestamp'leriyle
   eşleştirilir.
3. C03 timestamp'leri düzeltilmiş `t_ortak` eksenine çevrilir.
4. Her referans kare için zaman olarak en yakın aday aranır.
5. Eşleme monoton ve bire bir olur; aynı kaynak kare sessizce iki kez
   kullanılamaz.
6. Fark izin verilen toleransı aşarsa kare uydurulmaz, o kamera o anda eksik
   sayılır.
7. Kayıp veya bozuk kareler çoğaltılarak gizlenmez.
8. Kare indeksleri, ham/düzeltilmiş timestamp'ler ve residual CSV/JSON'a
   yazılır.

Fusion API aynı veya yakın timestamp'leri kendi senkron penceresinde
gruplayabilir. 60 FPS'te pencere yaklaşık bir kare süresidir. Bu özellik
Fusion fused-body hattında kullanılabilir; fakat TK3D'nin ham görüntü eşleme
raporunun yerine geçmez ve yanlış host saatini düzelttiği varsayılmaz.

### 3.4 Kabul kapısı

Her çekim için şu değerler raporlanır:

- kamera başına ilk ve son timestamp;
- başarıyla okunan kare sayısı;
- timestamp boşlukları;
- PC-1–PC-2 ağ saat farkı ve gidiş-dönüş gecikmesi;
- başlangıç ve bitiş görsel olaylarının kareleri;
- kamera çiftleri için offset;
- kayıt boyunca tahmini drift;
- hizalama residual'ı;
- ortak kullanılabilir zaman aralığı.

Başlangıç mühendislik hedefi, görsel olayların kameralar arasında en yakın
kareyle eşleşmesi ve kalan hatanın `<= 0.5 kare` olmasıdır. Bu eşik gerçek
çekimlerle doğrulanmadan “frame-synchronized” denmez.

Görsel olay üç kamerada da güvenilir bulunamazsa:

- yalnız ağ komutuna güvenilmez;
- çekim otomatik puanlamaya verilmez;
- gerekirse manuel kare eşlemesi yapılır;
- olay tekrarlanarak yeni çekim alınır.

---

## 4. Kayıt uygulamasının çalışma biçimi

Geliştirilecek küçük Python/ZED SDK uygulaması iki rolden oluşacaktır.

### 4.1 PC-1 — koordinatör ve iki kamera kaydedici

PC-1 uygulaması:

1. bağlı ZED'leri otomatik listeler;
2. C01 ve C02'nin seri numaralarını okur;
3. kameraları açıkça seri numarasıyla açar;
4. iki kamerayı ayrı worker/thread ile yerel SSD'ye kaydeder;
5. PC-2 sağlık kontrolünü yapar;
6. ortak `take_id` oluşturur;
7. bütün kameraların `ARMED` cevabını bekler;
8. kaydı başlatır ve durumları tek ekranda gösterir;
9. hata olsa bile sağlam kameraların kaydını izinsiz silmez;
10. durdurmada her SVO2'nin güvenli kapanışını bekler.

### 4.2 PC-2 — tek kamera kayıt düğümü

PC-2 uygulaması:

1. C03 seri numarasını otomatik okur;
2. PC-1'den gelen komutları yerel ağda dinler;
3. dosyayı PC-2 yerel SSD'sine yazar;
4. kayıt durumu, kare sayacı, hata ve disk alanını PC-1'e yollar;
5. ağ koparsa kayda devam eder;
6. yeniden bağlantıda güncel durumunu bildirir;
7. yalnız açık bir `STOP` komutu veya yerel acil durdurma ile kaydı kapatır.

### 4.3 Durum makinesi

```text
IDLE
  -> PREFLIGHT
  -> ARMED
  -> RECORDING
  -> FINALIZING
  -> VERIFIED

Her aşama -> ERROR (neden manifestte)
```

`START` komutu yalnız üç kamera açılmış, yerel SSD yazılabilir ve beklenen seri
numaraları doğrulanmışsa kabul edilir.

### 4.4 Ağ koparsa ne olacak?

Ana kural: **Ağ kaybı yerel kaydı durdurmaz.**

- PC-1, C01 ve C02'yi kaydetmeye devam eder.
- PC-2, C03'ü kaydetmeye devam eder.
- Operatör performansı bitirir ve iki bilgisayarı sırayla yerelden durdurur.
- Ortak görüntü aralığı offline bulunur.
- Manifestte ağ kopma zamanı ve manuel durdurma açıkça işaretlenir.

### 4.5 Neden merkezi ZED Studio ağ kaydı ana yol değil?

ZED Studio birden fazla yerel veya ağ stream'ini seçip ayrı SVO2 dosyalarına
kaydedebilir. Bu özellik tanı ve kısa karşılaştırma deneyi için yararlıdır.
Ancak bizim düzende PC-2 görüntüsünü telefon/eduroam Wi-Fi üzerinden PC-1'e
taşıyıp yalnız merkezde kaydetmek şu riskleri ekler:

- Wi-Fi paket kaybı ve değişken gecikme;
- ağ kopunca C03 ana kaydının kaybolması;
- kodlama/stream ayarlarının orijinal yerel kayıtla karışması;
- PC-1 ağ, decode, iki yerel kayıt ve disk yükünün birleşmesi;
- merkezi dosyanın oluşmasının kaynak kamerada kayıp olmadığını kanıtlamaması.

Bu nedenle üretim kuralı:

```text
Her kamera -> bağlı olduğu hostta yerel SVO2
ZED Studio network record -> yalnız opsiyonel tanı kopyası
```

Opsiyonel merkezi kopya alınırsa yerel SVO2'nin yerine geçmez ve iki dosya
birbirine kalite/kare/timestamp raporuyla karşılaştırılır.

---

## 5. Seri numarası neden önemlidir?

Seri numarası dosya adı süsü değildir. Her fiziksel kameranın:

- fabrika intrinsics/stereo kalibrasyonunu;
- sahadaki `C01/C02/C03` rolünü;
- ortak dünya extrinsic kalibrasyonunu;
- hangi bilgisayara ve USB yoluna bağlı olduğunu

birbirine bağlayan kimliktir.

USB kamera indeksi (`0`, `1`) bilgisayar yeniden başladığında veya kablolar
yer değiştirince değişebilir. Bu nedenle uygulama kamerayı indeksle değil,
ZED SDK'dan otomatik okunan seri numarasıyla açacaktır.

İlk kurulumda bir kez şu eşleme yapılır:

| Mantıksal kamera | ZED seri numarası | Bilgisayar | Fiziksel konum |
|---|---|---|---|
| C01 | otomatik okunacak | PC-1 | ön-sol/çapraz |
| C02 | otomatik okunacak | PC-1 | ön-sağ/çapraz |
| C03 | otomatik okunacak | PC-2 | arka veya yan |

Bir kamera başka konuma taşınırsa seri numarası aynı kalır fakat extrinsic
kalibrasyonu geçersiz olur ve yeniden yapılır.

---

## 6. Kamera yerleşimi

Üç kameralı başlangıç düzeni:

```text
                    C03
             (arka/yan çapraz)

          +-------------------+
          |                   |
          |    aktif alan     |
          |                   |
          +-------------------+

       C01                     C02
   (ön-sol çapraz)       (ön-sağ çapraz)
```

Başlangıç önerileri:

- C01 ve C02 arasında mümkün olduğunca geniş fakat ortak görüşü koruyan açı;
- C03, C01/C02'de kapanan arka tarafı ve ayakları görecek konum;
- yaklaşık `1.4–1.8 m` kamera yüksekliği;
- sporcu aktif alanın her noktasında baştan ayağa kadrajda;
- en az iki, tercihen üç kamerada gövde ve kritik eklemler görünür;
- doğrudan parlak ışığa bakmayan kameralar;
- mat zemin ve mümkün olduğunca az yansıma;
- tripod ayakları ve kamera yönü zeminde işaretli.

Üç kamera, mevcut TK3D leave-one-camera-out cross-view düzeltmesinin dört
başka kamera isteyen güvenlik eşiğini karşılamaz. Bu nedenle üç kameralı
sistemde:

- temel robust triangulation kullanılabilir;
- ZED depth ek kalite kanıtı olabilir;
- üç kameradan biri kapanır veya aykırılaşırsa sonuç hızla kırılganlaşır;
- mevcut tam cross-view düzeltme özelliği etkin kabul edilemez;
- dış doğrulama olmadan puanlama yetkisi verilmez.

---

## 7. SVO2 kayıt sözleşmesi ve PC-1 stres testi

### 7.1 SVO2 tam olarak neyi korur?

SVO2, ZED stereo görüntülerini ve timestamp, IMU/sensör gibi metadata'yı
koruyan yeniden oynatılabilir kaynak biçimidir. Depth, point cloud ve gövde
takibi gibi SDK modülleri SVO2 oynatılırken yeniden çalıştırılabilir.

Şu ayrımlar korunmalıdır:

- timestamp'in nanosaniye çözünürlükte yazılması, farklı hostların fiziksel
  olarak nanosaniye doğrulukta senkron olduğu anlamına gelmez;
- SVO2 kullanmak otomatik olarak lossless anlamına gelmez;
- `H.264` ve `H.265` kayıtları kayıplı olabilir;
- `LOSSLESS`, `H.264 LOSSLESS` veya `H.265 LOSSLESS` ayrı modlardır;
- depth'i kayıt sırasında ağır biçimde çalıştırmak zorunlu değildir;
- TK3D görüntüleri MP4'e dönüştürmeden doğrudan ZED SDK ile SVO2'den
  okuyacaktır.

İlk pilotta aynı kısa hareket hem seçilen uzun-kayıt modu hem lossless referans
ile çekilir. 2B eklem başarımı, dosya boyutu, encoder yükü ve kare boşlukları
karşılaştırılmadan sıkıştırma modu sabitlenmez.

### 7.2 PC-1'de iki kameranın zorunlu stres testi

Bilgisayarların “yeterli seviyede” olması başlangıç için olumlu olsa da iki
ZED'in aynı anda kayıt yapabildiği ölçülmelidir.

İlk saha kullanımından önce:

1. C01 ve C02 iki farklı USB 3 porta bağlanır.
2. Windows Aygıt Yöneticisi/USB ağacıyla portların controller yolu incelenir.
3. İki kamera aynı anda `HD720/60` açılır.
4. Hedef sıkıştırma modunda en az 30 dakika SVO2 kaydı yapılır.
5. Bu sırada ağır pose/depth işlemi çalıştırılmaz.
6. İki dosya baştan sona ZED SDK ile taranır.
7. Kare sayısı, timestamp aralığı, boşluklar ve SDK hataları raporlanır.
8. Disk yazma hızı, sıcaklık ve GPU encoder durumu incelenir.

Şunlardan biri görülürse test geçmez:

- kamera kopması;
- yeşil/mor/bozuk kare;
- tekrarlanan veya monoton olmayan timestamp;
- normal kare aralığının `1.5 katını` aşan açıklanamayan boşluk;
- kayıt uygulamasının uzun süre cevap vermemesi;
- SSD'nin dolması veya sürdürülebilir yazma hızının düşmesi;
- ikinci H.264/H.265 kayıt oturumunun açılamaması.

Test geçmezse sırasıyla:

1. kameraları farklı USB controller'lara taşı;
2. PC-1'e bağımsız kanallı PCIe USB 3 kartı ekle;
3. farklı doğrulanmış sıkıştırma modunu dene;
4. iki kamera + bir kamera dağılımını tersine çevir;
5. son çare olarak üçüncü kayıt bilgisayarı kullan.

USB hub ancak tam stres testini geçerse kabul edilir.

---

## 8. Her çekimin saha sırası

### 8.1 Çekim öncesi

1. Telefon erişim noktasını aç ve iki bilgisayarı bağla.
2. PC-1'den PC-2 sağlık kontrolünü doğrula.
3. Üç kameranın seri numarası/rol eşlemesini doğrula.
4. Lens, tripod, kablo ve zemin işaretlerini kontrol et.
5. SSD boş alanını ve tahmini çekim süresini kontrol et.
6. Kameraları ve ışıkları 10–15 dakika ısıt.
7. Pozlama, gain, white balance ve motion blur kontrolü yap.
8. Üç görüntüde de aktif alanın tamamının göründüğünü doğrula.
9. Günlük ortak dünya kalibrasyon kontrolünü yap.
10. Benzersiz `take_id` oluştur.

### 8.2 Otomatik ağ akışı

1. PC-1'de `Hazırla` düğmesine bas.
2. Ekranda `C01 ARMED`, `C02 ARMED`, `C03 ARMED` görülmeden devam etme.
3. `Kaydı Başlat` düğmesine bas.
4. Üç dosyanın da büyüdüğünü ve kare sayaçlarının ilerlediğini gör.
5. En az 10 saniye bekle.
6. Aktif alanda başlangıç senkron hareketini üç kez yap.
7. İki saniye bekle ve performansı başlat.
8. Performans bitince iki saniye bekle.
9. Bitiş senkron hareketini üç kez yap.
10. En az 10 saniye bekle.
11. `Kaydı Durdur` düğmesine bas.
12. Üç kamerada `VERIFIED` veya açıklamalı hata durumu görülmeden uygulamayı
    kapatma.

### 8.3 Wi-Fi/eduroam çalışmazsa manuel yedek akış

1. PC-2'de C03 kaydını yerelden başlat.
2. C03 dosyasının büyüdüğünü doğrula.
3. PC-1'de C01 ve C02 kaydını başlat.
4. İki dosyanın da büyüdüğünü doğrula.
5. En az 10 saniye bekle.
6. Başlangıç senkron hareketini yap.
7. Performansı kaydet.
8. Bitiş senkron hareketini yap.
9. En az 10 saniye bekle.
10. Önce PC-1, sonra PC-2 kaydını güvenli biçimde durdur.

Bu yedek akışta düğmelere aynı anda basmaya çalışmak gerekmez.

### 8.4 Acil durum

- Kamera düşer veya kablo çıkarsa performans durdurulur.
- Kayıt uygulaması mümkün olan dosyaları güvenli biçimde kapatır.
- Bozuk çekimin üzerine yeni kayıt yazılmaz.
- Yeni `take_id` ile tekrar çekilir.
- Oynayan kameranın ortak dünya kalibrasyonu yeniden doğrulanır.

---

## 9. Dosya ve manifest yapısı

```text
takes/
  take_2026-07-30_001/
    manifest.json
    notes.md
    sync/
      network_clock_samples.json
      visual_sync_events.json
      synchronization_report.json
    calibration/
      cameras_world.json
      calibration_report.json
    svo/
      C01_SN<serial>.svo2
      C02_SN<serial>.svo2
      C03_SN<serial>.svo2
```

Her SVO2 kaydı benzersizdir. Var olan dosyanın üzerine yazılmaz.

Manifestte en az şunlar bulunur:

- session ve `take_id`;
- PC-1/PC-2 makine kimliği;
- mantıksal kamera adı ve otomatik okunan seri numarası;
- ZED SDK ve firmware sürümü;
- çözünürlük, hedef FPS ve sıkıştırma;
- bilgisayar, USB controller/yol ve hedef disk;
- ilk/son görüntü timestamp'i;
- raporlanan ve gerçekten okunan kare sayısı;
- timestamp boşlukları;
- başlangıç/durdurma komutu ve onay zamanları;
- ağ saat ölçümleri;
- görsel senkron olayları, offset ve drift;
- SVO2 dosya boyutu ve SHA-256;
- kullanılan extrinsic kalibrasyon kimliği ve SHA-256;
- ışık, kamera hareketi ve operatör notları.

---

## 10. Kayıt sonrası zorunlu doğrulama

Her dosya için:

1. SVO2'nin ZED SDK ile açıldığını doğrula.
2. İlk, orta ve son kareyi gerçekten oku.
3. Dosyayı baştan sona tara.
4. Başarıyla okunan kare sayısını kaydet.
5. Timestamp monotonluğunu ve boşluklarını denetle.
6. Seri numarası ile dosya/kalibrasyon eşlemesini doğrula.
7. IMU ve sensör metadata'sını kontrol et.
8. SHA-256 üret.

PC-2 dosyası harici SSD ile taşınırken:

1. Kaynak SHA-256 alınır.
2. Dosya ana bilgisayara kopyalanır.
3. Hedef SHA-256 yeniden alınır.
4. İki özet eşleşmeden PC-2'deki ham dosya silinmez.

Sonra:

1. ağ saat örneklerinden kaba PC offset/drift çözülür;
2. başlangıç ve bitiş senkron hareketleri üç kamerada bulunur;
3. görüntü kanıtıyla nihai offset ve drift çözülür;
4. kayıp kareler zaman çizelgesinde boş bırakılır;
5. üç kameranın ortak aralığı belirlenir;
6. senkron kalite raporu yazılır;
7. yalnız rapor geçerse çok-kamera 3B işleme başlar.

---

## 11. Ortak dünya kalibrasyonu

Her ZED'in fabrika kalibrasyonu yalnız kendi sol/sağ stereo geometrisini verir.
Üç kameranın odadaki birbirine göre konumu ayrıca ölçülmelidir.

### 11.1 ZED360 kullanılacak mı?

**Evet, kalibrasyon adayı ve canlı tanı aracı olarak kullanılacaktır.**
ZED360'ın ürettiği kamera pozları bağımsız geometrik doğrulama geçerse TK3D
üretim kalibrasyonuna dönüştürülebilir.

ZED360 kalibrasyonu görüntüdeki özel bir tahta yerine, her kameranın ZED Body
Tracking sonucunu kullanır. Bir kişi kameraların ortak gördüğü alanın tamamında
yavaşça yürür; araç farklı kameralardaki gövde noktalarını ortak WORLD
koordinatında hizalar. Kamera rolleri seri numarasıyla tutulur. Çıktıdaki kamera
dönüşleri ve metre cinsinden konumlar bir JSON yapılandırma dosyasına
kaydedilir.

Saha kalibrasyon akışı:

1. Üç kamera sabit konumlarına yerleştirilir ve ısıtılır.
2. Telefon erişim noktası üzerinden PC-1 ve PC-2'nin birbirini gördüğü
   doğrulanır.
3. PC-1'deki C01 ve C02 ayrı yayın portlarıyla, PC-2'deki C03 üçüncü portla
   Fusion publisher olarak çalıştırılır. Bunun için genel ZED video streaming
   düğmesi değil, Fusion API `startPublishing()` akışı kullanılır.
4. PC-1'de ZED360/Fusion subscriber çalıştırılır.
5. İki bilgisayarda aynı ZED SDK sürümü, aynı body formatı, aynı coordinate
   system/unit ve uyumlu body model ayarları kullanılır.
6. Üç kamerada aynı anda yalnız bir kişi görünür.
7. Kişi aktif alanın tamamında, kenarlarda ve merkezde yavaşça yürür.
8. Ayak bilekleri görünür tutularak zemin düzlemi tahminine yardım edilir.
9. Gövdeler ZED360 ekranında iyi hizalandığında kalibrasyon JSON'u kaydedilir.
10. JSON; seri numarası, SDK sürümü, tarih ve dosya SHA-256 değeriyle saklanır.
11. Kameralar hareket etmedikçe dosya yeniden kullanılabilir; herhangi bir
    kamera oynarsa kalibrasyon geçersiz olur.

Bu topoloji ZED360'ın en basit yerel akışı değildir. Yerel akışta bütün
kameralar aynı makine ve aynı işlem içindedir. Bizde iki kamera PC-1'de, üçüncü
kamera PC-2'dedir. Bu nedenle Fusion'ın `LOCAL_NETWORK` yayın/abone modeli ve
her kamera için farklı port kullanılacaktır. PC-1'in aynı anda iki publisher
ve subscriber olması CPU/GPU/USB/ağ yükünü artırır. Telefon erişim noktasında
bu akış gerçek stres testiyle doğrulanmadan çalışıyor kabul edilmez.

Stereolabs'ın güncel web sayfası ZED360 Network Workflow için ZED Hub
adımlarını göstermeye devam etse de Stereolabs destek ekibi 28 Nisan 2025'te
genel ZED Hub–ZED360 akışının deprecated olduğunu bildirmiştir. Bu nedenle:

- yeni bir ZED Hub hesabı/workspace'i üretim bağımlılığı yapılmaz;
- publisher'lar doğrudan `LOCAL_NETWORK` IP/port yapılandırmasıyla denenir;
- genel ZED video stream'i ile Fusion publisher birbirine karıştırılmaz;
- ZED360 UI ağ akışı kullanılan SDK sürümünde ayrıca smoke test edilir;
- ağ ZED360 çalışmazsa yerel SVO2 kayıt sistemi bundan etkilenmez.

Stereolabs'ın belgelenmiş ağ topolojisi ayrı publisher makineleri ve ayrı bir
subscriber makinesi varsayar. Bizde üçüncü bilgisayar yoktur. PC-1'in hem iki
publisher hem subscriber olduğu deney geçmezse zorlanmayacaktır. Yedek sırası:

1. kameraları oynatmadan, USB/controller kapasitesi uygunsa kalibrasyon için
   üç kamerayı geçici olarak PC-1'de ZED360 Local Workflow ile açmak;
2. bu mümkün değilse ChArUco/AprilTag ortak dünya kalibrasyonunu ana yöntem
   yapmak ve ZED360'ı yalnız erişilebilen tanı akışında kullanmak;
3. Fusion network akışını daha sonra ayrı bir subscriber bilgisayarı veya
   doğrulanmış yerel ağla tekrar denemek.

ZED360'ın ürettiği dosya doğrudan ve kontrolsüz biçimde TK3D üretim
kalibrasyonu sayılmaz. İnsan gövdesine dayalı optimizasyon, kapanma veya hatalı
ZED body noktalarından etkilenebilir.

### 11.2 Fusion API kullanılacak mı?

**Evet, iki amaçla kullanılacaktır:**

1. **Canlı kurulum/sağlık önizlemesi:** Üç kameradaki kişinin aynı ortak dünya
   konumunda birleşip birleşmediğini görmek; kötü örtüşme, ters kamera yönü
   veya açık kalibrasyon hatasını sahada fark etmek.
2. **Karşılaştırma hattı:** Stereolabs'ın fused body sonucunu TK3D'nin
   ViTPose-Huge WholeBody + robust triangulation sonucuyla aynı çekimde
   karşılaştırmak.

Fusion sonucu şu aşamada nihai TK3D pozunun yerine geçmez:

- Fusion'ın ZED Body Tracking iskelet sözleşmesi TK3D'nin
  `keypoints_3d_world[t, 133, 3]` WholeBody sözleşmesiyle aynı değildir.
- Mevcut TK3D el, yüz ve ayak dâhil 133 görüntü noktasını korumalıdır.
- Wi-Fi tabanlı canlı Fusion, yerel SVO2 kaydının yerine kullanılmayacaktır.
- Ağ kesilirse Fusion önizlemesi gidebilir fakat üç yerel SVO2 kaydı sürer.
- ZED fused body, dış ground-truth değildir ve tek başına puanlama yetkisi
  vermez.

İlk pilotta üç sonuç ayrı saklanacaktır:

```text
1. ZED360 kamera pozları ve kalibrasyon kalite görünümü
2. Fusion API fused body çıktısı
3. TK3D ViTPose-133 + çok-kamera triangulation çıktısı
```

Fusion ile TK3D arasındaki fark; ortak eklemler, zaman, birim ve koordinat
sistemi eşlendikten sonra ölçülür. Uyuşmaları yararlı çapraz kanıttır;
uyuşmamaları hangi sistemin doğru olduğunu tek başına göstermez.

### 11.3 Bağımsız kalibrasyon doğrulaması

ZED360'a ek olarak şu doğrulama yapılacaktır:

- büyük, rijit ve ölçüsü doğrulanmış ChArUco/AprilTag hedefi;
- hedefin üç kamerada mümkün olduğunca birlikte görünmesi;
- aktif alanın önü, arkası, merkezi, zemine yakın ve göğüs yüksekliğinde
  örnekler;
- kalibrasyonda kullanılmayan ayrı doğrulama görüntüleri;
- ölçüsü bilinen rijit çubukla bağımsız 3B kontrol.

ZED360 ve hedef tabanlı kalibrasyon iki ayrı aday üretirse körlemesine biri
seçilmez. Ayrılmış doğrulama karelerinde reprojection, ölçülü çubuk 3B hatası
ve alan boyunca tutarlılık karşılaştırılır. Üretim kalibrasyonu yalnız bu
kapıları geçen aday olur; yöntem ve provenance manifestte yazılır.

Kalibrasyon şu durumlarda geçersiz olur:

- tripod/kamera oynadı;
- kamera başka fiziksel konuma taşındı;
- kamera değiştirildi;
- çözünürlük/rectification sözleşmesi değişti;
- günlük reprojection ve ölçülü çubuk doğrulaması geçmedi.

Seri numarası doğru olsa bile kamera taşındıysa eski extrinsic kullanılmaz.

---

## 12. TK3D işleme hattı

```text
3 yerel SVO2
  -> dosya ve seri numarası doğrulaması
  -> timestamp + ağ ölçümü + görsel hareketle ortak zaman
  -> ortak dünya kalibrasyonu doğrulaması
  -> rectified sol RGB çıkarımı
  -> RF-DETR kişi tespiti + ByteTrack
  -> ViTPose-Huge WholeBody, 133 adet 2B nokta
  -> ham 2B ölçümlerin korunması
  -> offline 2B stabilizasyon
  -> ZED depth + confidence + kişi maskesi
  -> üç kameralı robust triangulation
  -> reprojection/depth/anatomi/zaman kalite kapıları
  -> ham triangulation korunarak güvenli optimizasyon
  -> [T, 133, 3] metre, ortak dünya koordinatı
  -> dış ground-truth doğrulaması varsa puanlama yetkisi
```

Üç kameralı düzende bir eklem için:

- üç bağımsız görüntü desteği: tercih edilen;
- iki güvenilir kamera: temel triangulation yapılabilir;
- tek kamera: ortak dünya triangulation değildir; ZED depth yalnız açıkça
  işaretli RGB-D fallback/önizleme olabilir;
- görüntü kanıtı yok: nokta eksik bırakılır.

Eksik değer JSON'da `null`, CSV'de boş hücre olmalıdır. `NaN` veya `inf`
downstream çıktıya yazılmaz.

---

## 13. Devreye alma yol haritası

### Faz 0 — donanım envanteri

- üç seri numarasını yazılım aracılığıyla otomatik al;
- C01/C02/C03 fiziksel etiketlerini yapıştır;
- PC-1'deki iki USB yolunu belirle;
- SDK, firmware, disk ve encoder bilgisini kaydet;
- erişim noktası ve güvenlik duvarı bağlantı testini yap.

Çıkış kapısı: üç kamera doğru seri numarasıyla ayrı ayrı açılıyor.

### Faz 1 — PC-1 çift kamera stres testi

- C01+C02 ile en az 30 dakika `HD720/60 SVO2`;
- bütün dosyayı tarama;
- kare boşluğu, bozuk kare, USB kopması, disk ve encoder raporu;
- gerekirse USB controller dağılımını değiştir.

Çıkış kapısı: iki kamera aynı anda kararlı kayıt yapıyor.

### Faz 2 — dağıtık kayıt kontrolü

- PC-1 koordinatör ve PC-2 worker uygulaması;
- `ARM/START/STOP/status` protokolü;
- ağ kopunca kayda devam;
- ortak `take_id`;
- güvenli dosya kapatma;
- manuel yedek akış testi.

Çıkış kapısı: otomatik ve manuel akışta üç sağlam SVO2 oluşuyor.

### Faz 3 — zaman senkronizasyonu

- ağ saat örneklerinin kaydı;
- başlangıç/ara/bitiş senkron hareketleri;
- otomatik olay algılama;
- kamera çiftleri offset ve drift raporu;
- bilerek gecikmeli başlatma ve Wi-Fi koparma testi.

Çıkış kapısı: başlangıç düğmesine farklı zamanlarda basılsa bile ortak zaman
çizelgesi tekrar üretilebiliyor ve residual hedefi geçiyor.

### Faz 4 — ortak dünya kalibrasyonu

- ZED Hub olmadan doğrudan IP/port `startPublishing()` smoke testi;
- aynı SDK/body format/coordinate system kontrolü;
- üç publisher + PC-1 Fusion subscriber yük ve ağ testi;
- ZED360 ile yavaş yürüyüş tabanlı ilk kamera pozları;
- ZED360 kalibrasyon JSON'unun seri numarasıyla saklanması;
- büyük rijit hedef;
- alanın tamamında görüntü toplama;
- ZED360 ve hedef tabanlı kamera pozlarının karşılaştırılması;
- ayrılmış karelerde reprojection;
- ölçülü rijit çubukla 3B doğrulama.

Çıkış kapısı: bağımsız doğrulama ölçümleri kabul sınırında.

### Faz 5 — ilk üç kameralı hareket pilotu

- yavaş hareket;
- hızlı tekme/dönüş;
- başlangıç ve bitiş senkron dizileri;
- Fusion fused body sonucunun ayrı kaydı;
- 2B, ZED depth ve triangulation karşılaştırması;
- Fusion ile TK3D'nin ortak eklemlerde karşılaştırılması;
- kamera başına sağlık ve senkron raporu.

Çıkış kapısı: aynı zaman, kalibrasyon ve kamera setiyle tekrarlanabilir 3B
sonuç.

### Faz 6 — uzun saha testi

- gerçek hedef süre kadar kesintisiz kayıt;
- her 5 dakikada ara senkron hareketi;
- telefon ekranı kapanması;
- kısa Wi-Fi kesintisi;
- disk doluluk ve sıcaklık takibi;
- kayıt sonrası tam SVO2 taraması.

Çıkış kapısı: ağ hatası yerel kayıt kaybına yol açmıyor; drift ölçülüyor.

### Faz 7 — puanlama öncesi dış doğrulama

- ölçülü veya bağımsız 3B ground-truth;
- BODY-17 geçerli oranı;
- reprojection dağılımı;
- kamera kanıtı;
- temporal/açı jitter;
- kamera çıkarma testi;
- tekrar çekim tutarlılığı.

İç geometri kapısının geçmesi tek başına resmî puanlama doğruluğu değildir.

---

## 14. Kabul edilmeyecek kısa yollar

- İki bilgisayarda aynı anda düğmeye basmayı hassas senkron saymak.
- Yalnız Windows saatlerinin aynı görünmesine güvenmek.
- Eduroam komut zamanını kamera pozlama zamanı saymak.
- Ağ kesildiğinde yerel kaydı durdurmak.
- SVO2'yi doğrudan Wi-Fi ağ diskine yazmak.
- Kamera indeksini seri numarası yerine kalıcı kimlik saymak.
- Kayıp kareleri çoğaltarak veya zaman çizelgesini sıkıştırarak gizlemek.
- Üç kameralı sonucu mevcut beş-kamera cross-view güvenlik tasarımıyla eşdeğer
  göstermek.
- ZED depth'i bağımsız ground-truth saymak.
- Kalibrasyon hedefi olmadan yaklaşık kamera konumlarını üretim extrinsic'i
  olarak kullanmak.

---

## 15. Günlük kısa kontrol listesi

### Kurulum

- [ ] PC-1: C01+C02, PC-2: C03 bağlı.
- [ ] Üç seri numarası beklenen C01/C02/C03 rolleriyle eşleşiyor.
- [ ] USB bağlantıları ve PC-1 çift kamera testi geçerli.
- [ ] Telefon/PC erişim noktası çalışıyor veya manuel akış hazır.
- [ ] Üç kamera aktif alanı ve senkron hareketini görüyor.
- [ ] Tripod ve zemin işaretleri eşleşiyor.
- [ ] SSD alanı hedef çekim süresi için yeterli.
- [ ] Kamera ve ışık ısınması tamamlandı.

### Kayıt

- [ ] Benzersiz `take_id` oluşturuldu.
- [ ] Üç kamera `RECORDING`.
- [ ] Üç dosya büyüyor ve kare sayacı ilerliyor.
- [ ] En az 10 saniye ön kayıt var.
- [ ] Başlangıç senkron hareketi kaydedildi.
- [ ] Uzun çekimde ara olaylar kaydedildi.
- [ ] Bitiş senkron hareketi kaydedildi.
- [ ] En az 10 saniye son kayıt var.
- [ ] Üç dosya güvenli kapandı.

### Kayıt sonrası

- [ ] Üç SVO2 baştan sona tarandı.
- [ ] Kare/timestamp gap raporu üretildi.
- [ ] SHA-256 değerleri alındı.
- [ ] C03 kopyası doğrulandı.
- [ ] Görsel senkron offset/drift raporu geçti.
- [ ] Ortak dünya kalibrasyonu geçti.
- [ ] Ham SVO2 ve ham triangulation korundu.
- [ ] Çıktı benzersiz `outputs/<session_id>/runs/<run_id>/` altında.

---

## 16. İlk uygulanacak net deney

İlk gün tam poomsae çekimine geçmeden şu 20–30 dakikalık pilot yapılmalıdır:

1. PC-1'e C01+C02, PC-2'ye C03 bağla.
2. İki bilgisayarı telefon erişim noktasına bağla.
3. Üç seri numarasını otomatik okut.
4. Üç Fusion publisher'ı `startPublishing()` ile farklı portlarda dene.
5. PC-1 ZED360'ın üç kaynağı görüp görmediğini ve hızı kontrol et.
6. Ağ pilotu geçerse yavaş yürüyüş kalibrasyonu yapıp JSON'u kaydet.
7. Ağ pilotu geçmezse bunu not et; yerel SVO2 kaydını iptal etme.
8. Üç kamerayı `HD720/60` SVO2 kayda al.
9. 10 saniye bekle.
10. Aktif alan merkezinde başlangıç senkron hareketini üç kez yap.
11. Büyük ChArUco/AprilTag hedefini alan boyunca gezdir.
12. Ölçüsü bilinen rijit çubuğu farklı yönlerde hareket ettir.
13. Yavaş, sonra hızlı bir tekme ve dönüş yap.
14. Bitiş senkron hareketini üç kez yap.
15. 10 saniye bekle ve kayıtları durdur.
16. Üç dosyayı baştan sona doğrula.
17. C03'ü SHA-256 kontrollü kopyala.
18. Offset, drift, reprojection ve ölçülü çubuk hata raporlarını üret.
19. ZED360 ve hedef tabanlı kamera pozlarını bağımsız ölçümlerde karşılaştır.

Bu pilot geçmeden uzun veya önemli çekime başlanmamalıdır.

---

## 17. Resmî teknik dayanaklar

- [Stereolabs — çoklu kamera kurulumu ve USB zaman hizalama](https://docs.stereolabs.com/docs/development/zed-sdk/modules/camera/multi-camera)
- [Stereolabs — SVO/SVO2 kayıt ve çoklu kamera kaydı](https://docs.stereolabs.com/docs/development/zed-sdk/modules/camera/recording)
- [Stereolabs — ZED Studio yerel/ağ stream kaydı](https://docs.stereolabs.com/docs/development/zed-tools/zed-studio)
- [Stereolabs — sensör zaman damgaları](https://docs.stereolabs.com/docs/development/zed-sdk/modules/sensors/time-synchronization)
- [Stereolabs — timestamp tabanlı senkron penceresi ve veri kaybı](https://docs.stereolabs.com/docs/development/zed-sdk/modules/global-localization/data-synchronization)
- [Stereolabs — seri numarasına bağlı kamera kalibrasyonu](https://docs.stereolabs.com/docs/development/zed-sdk/modules/camera/camera-calibration)
- [Stereolabs — ZED360 çoklu kamera kalibrasyonu](https://docs.stereolabs.com/docs/development/zed-tools/zed-360)
- [Stereolabs — Fusion API ve yapılandırma dosyaları](https://docs.stereolabs.com/docs/development/zed-sdk/modules/fusion)
- [Stereolabs Support — genel ZED Hub–ZED360 akışının deprecated durumu](https://community.stereolabs.com/t/zed360-with-zed-hub/8670)

Repository içindeki ilgili belgeler:

- `PROJECT_STATUS.md`
- `docs/ZED2I_OFFLINE_MULTICAMERA_PLAN.md`
- `docs/ARCHITECTURE_DECISIONS.md`
