# TK3D Doğrulama ve Geliştirme Planı

Oluşturma: **20 Eylül 2026** · Son güncelleme: **24 Eylül 2026** · Sürüm: **1.2**

Hedef: **Antrenör ve hakemin bir hareketi doğru anda, dayanağı görülebilen
ölçümlerle inceleyebildiği bir araştırma sistemi.** Otomatik puanlamaya geçiş,
hareket zamanı, ölçüm doğruluğu ve kural geçerliliği için ayrı kanıt gerektirir.

Bu belge yeni çalışmaların sırasını belirler. `PUANLAMA_PLANI.md` tarihsel
mimari bağlamdır; oradaki uzman doğrulamasını sonraya bırakan öncelik bu planla
değişmiştir. P0 ve P1 uygulanıp doğrulandı; gerçek uzman incelemesi olan P2
henüz tamamlanmadı. Kanıt yolları ve kalan işler son bölümde bulunur.

## 1. İlk somut teslim

**Mevcut iki kameralı kayıtta M01–M06 için incelenebilir zaman etiketleri ve
üç aday ölçütün doğrulama dosyası.** İnceleyen kişi her sonuç için şunları
görebilmeli:

1. Hangi hareket ve hangi değerlendirme aralığı kullanıldı?
2. Aralığı kim önerdi, kim inceledi; çözülemeyen belirsizlik var mı?
3. Hangi eklemler, koordinat doğrultusu ve formül kullanıldı?
4. Bağımsız referansla ölçülen hata ve ölçüm yapılamayan durumlar neler?
5. Kuralın yazılı kaynağı, kapsamı ve uzman yorumu ne?
6. Sonuç ölçüm mü, inceleme adayı mı; karar vermeyi engelleyen neden ne?

İlk aday ölçütler: **dirsek açısı, diz açısı ve gövde eğimi**. Bunlar henüz
onaylanmış seçimler değildir. Görünürlük ve bağımsız ölçüm imkânı yetersizse
seçim gerekçesi kaydedilerek değiştirilecektir. Üç ölçütün doğrulanması,
174 kurallık envanterin veya 133 noktanın tamamını doğrulamaz.

## 2. Başlangıç gerçeği ve korunan varlıklar

- İncelenen temel commit: `e21ce51` (`main`). Başlangıç çalışma ağacı temiz;
  `git pull --ff-only` sonucu güncel. İlk plan aşamasında commit/push istenmedi;
  sonraki kullanıcı talebi P0/P1 ve CI düzeltmesinin gönderimini kapsıyor.
- Mevcut hat: iki ZED 2i, RF-DETR Small, ByteTrack, ViTPose-Huge WholeBody-133,
  çoklu görüntüden 3B hesaplama, BODY-17 depth/optimizasyon ve inceleme raporları.
- Aktif zaman çizelgesi 741 örnek, 60 FPS, M01–M06 kapsamı tanımlar. Hareket
  kimlikleri, sınırlar ve fazlar uzman incelemesinde yeniden kontrol edilecek.
- YAML'daki `manual`, `confirmed` ve `confidence` değerleri mevcut beyanlardır.
  İnceleyen kimliği ve bağlı onay kaydı bulunmadığından insan doğrulaması veya
  kalibre edilmiş doğruluk olasılığı olarak kullanılamaz.
- Otomatik bölümleme hareket hızından aday aralıklar çıkarır; gerçek hareket
  tanıma ve faz doğruluğu bağımsız olarak kanıtlanmış değildir.
- Güncel v3 profilde `judge_validated_rules: []`. İç testler ve geometri
  tutarlılığı dış ölçüm doğruluğu ya da hakem uyumu değildir.
- Eski çıktılar, ham 2B/3B, kaynak videolar, kalibrasyon ve hash kayıtları
  korunur. Her yeni inceleme/analiz benzersiz run dizinine yazılır.

## 3. Takip kuralları

Durumlar: **tamamlandı / sürüyor / sırada / dış katkı bekliyor**.
Bir aşama ancak kabul ölçütü ve kanıt yolu yazıldığında tamamlandı sayılır.
Bir belge, şema veya test hazırlamak gerçek uzman incelemesini tamamlamaz.

- Her çalışma başında bu tablodaki ilk açık yazılım işi ele alınır.
- Dış katkı beklenirken bağımsız yazılım ve protokol işleri sürdürülebilir;
  eksik onay varsayılarak puanlama ya da başarı iddiası açılmaz.
- Her teslimde değişen dosyalar, çalıştırılan kontroller, çıktı yolu ve kalan
  engel kaydedilir. Planın durum tablosu güncellenir.
- Model, eşik veya kamera değişiklikleri aynı kayıt/ayarlarla karşılaştırılır.
  Ölçüm hatası anlaşılmadan filtre, yeni model veya kamera sayısı başarı diye
  sunulmaz. Tek deneyde mümkün olduğunca tek değişken değiştirilir.
- Commit/push kullanıcı talebiyle yapılır. Eski run veya kullanıcı etiketinin
  üzerine yazılmaz; videolar ve `outputs/` Git'e eklenmez.

| ID | Aşama | Bağımlılık | Durum | Tamamlandığında teslim |
| --- | --- | --- | --- | --- |
| P0 | Kanıt envanteri ve belge tutarlılığı | Mevcut repo | Tamamlandı | İddia/kanıt listesi, düzeltilmiş belgeler, kamera inceleme paketi |
| P1 | Zaman etiketi güven sözleşmesi | P0 | Tamamlandı | Onaysız etiketin karar yetkisi vermediği kod, 541 geçen test ve gerçek kayıt analizi |
| P2 | Altı hareketin uzman incelemesi | P0 paketi; kalıcı onay için P1 | Dış katkı bekliyor | Kaynağa bağlı, imzalanmamış boşlukları açık altı hareket kaydı |
| P3 | Üç ölçüt ve kural tanımı | P2; uzman ve ölçüm desteği | Sırada | Üç ölçüm/kural kartı, kabul protokolü |
| P4 | Ölçüm ve zaman duyarlılığı deneyi | P2–P3, bağımsız referans | Sırada | Hata, belirsizlik ve ölçülemeyen durumlar raporu |
| P5 | Yeni kayıt pilotu | P3 protokolü; P4 ilk bulguları | Sırada | Kayıt planı, kamera gerekçesi, geliştirme/test ayrımı |
| P6 | Otomatik önerilerin değerlendirilmesi | P1, P2, ayrı test kayıtları | Sırada | Referansa göre sınır/faz hatası ve insan düzeltme yükü |
| P7 | Kullanıcı denemesi ve araştırma dosyası | P4–P6 sonuçları | Sırada | Kullanılabilirlik sonuçları, yeniden üretilebilir deney ve makale taslağı |

## 4. P0 — İddiaları mevcut kanıtla eşitle

Sorumlu: yazılım/analiz çalışması. Uzman onayı gerektirmez.

- [x] Aktif timeline, profil, puanlama, readiness, şablon ve bölümleme
  yollarındaki güven kabullerini incele.
- [x] Güncel belgelerdeki dayanağı bulunmayan insan doğrulaması ifadelerini
  düzelt; tarihsel sonuçları silmeden kullanım sınırını yaz.
- [x] İki gerçek kameradan altı fixation adayı için çevre karelerini üret;
  bunun inceleme önerisi olduğunu açıkça göster.
- [x] Boş uzman inceleme/kural formunu ve sonraki uygulama görevini hazırla.

Kabul: kaynak yol ve hash'leri görülen inceleme paketi; hiçbir doldurulmamış
form insan onayı sayılmaz. **P0 bitmesi runtime güven açığının kapandığı
anlamına gelmez; bu P1'in kabul ölçütüdür.**

P1'de kapatılan somut riskler (ikinci sütun değişiklik öncesini anlatır):

| Yol | Mevcut kabul | Gerekli düzeltme |
| --- | --- | --- |
| `contracts.py` | Şema v2 `confirmed` alanını kabul ediyor; inceleyen/onay alanı yok | Eski dosyayı okuyabilen, onaysızlığı açık taşıyan güven sözleşmesi |
| `source_bound_accuracy.py` | Sayısal karar üretimi faz onayını sınamıyor | Zaman/faz güveni yoksa uygulanmış kesinti üretme |
| Aynı modülde duraklama türetimi | `manual` + `confirmed` + confidence bağımsız inceleme sayılabiliyor | İnsan incelemesini bağlı onay kaydıyla ayır |
| `technical_accuracy.py` | İmzalı hakem eşiğinin karar yetkisi faz incelemesinden ayrılmıyor | İmzalı eşik olsa da incelenmemiş hareketi puansız tut |
| `contracts.py` tam Accuracy kapısı | `complete` beyanı insan incelemesi yerine geçebiliyor | Tam değerlendirme için kapsama bağlı incelemeyi de zorunlu tut |
| `readiness.py`, `application.py`, `technical_conformance.py` | Bazı hazır/onaylı durumlar etikete dayanıyor | Tek merkezi güven kararı kullan; raporda nedenini göster |
| `build_poomsae_reference_templates.py` | `label_source=manual` referans için yeterli | Onaysız şablonun doğrulanmış referansa yükselmesini engelle |
| `automatic_segmentation.py` | Aynı taslakla karşılaştırma hata değerleri üretiyor | Referans güvenini taşı; taslak farkını doğruluk olarak sunma |

## 5. P1 — Etiket güvenini kodda koru

Sorumlu: yazılım çalışması. Bu aşama uzman görüşü uydurmadan tamamlanabilir.

Uygulama sırası:

1. Geriye uyumlu okuma: eski timeline veri olarak okunabilir; onay kaydı yoksa
   güveni doğrulanmamış olur. Dosya okunabilirliği ile karar yetkisi ayrılır.
2. Onay kaydı: inceleyen, rol, zaman dilimli tarih/saat, video inceleme yöntemi,
   kanıt referansı/hash'i, onaylanan etiket içeriğinin hash'i ve kapsadığı
   hareket/sınır/fazlar. Karar ve gerekçe bağlı inceleme formunda tutulur;
   belirsiz kapsam onay listesine alınmaz. Zaman ekseni ve kaynak pose bağı
   içerik hash'ine dahildir. İçerik değişirse eski onay uygulanamaz.
3. İsim/tarih girilmesi kimliğin veya ölçümün doğruluğunun kriptografik kanıtı
   değildir. Kayıt izlenebilirlik sağlar; gerçek inceleme ayrıca yapılır.
   Saf sözleşme kontrolü dış kanıt dosyasını açıp içeriğini doğrulamaz;
   `verified` yalnız kaydın mevcut içerik ve kapsamla eşleştiğini belirtir.
4. Bir bulguya verilen mevcut HTML “Doğru” cevabı, faz etiketinin veya kuralın
   onayı sayılmaz. Bu üç inceleme türünün kapsamı ayrılır.
5. Onaysız fazda ölçüm yapılabilir; sonuç koşullu ölçüm/inceleme adayıdır.
   Uygulanmış kesinti, onaylı referans veya skor yetkisi oluşmaz. Eksik bilgi
   `0 hata`/`başarılı` diye sunulmaz.
6. Yeni inference run'ına timeline aktarılırken video/zaman/içerik bağı yeniden
   sınanır; eski onay otomatik taşınmaz.
7. Yeni bir analiz run'ıyla JSON, HTML, video ve özetlerin aynı güven durumunu
   gösterdiği kontrol edilir. Eski run'lar tarihsel kalır.

Kabul testleri:

- Onaysız `manual + confirmed + confidence=1` onay ve kesinti yetkisi vermez.
- Eksik/kısmi onay yalnız kapsadığı alanı etkiler; hareket onayı tüm fazları
  veya diğer hareketleri otomatik onaylamaz.
- Anchor, sınır, kaynak, zaman ekseni değişikliği eski onayı geçersiz kılar.
- Geçerli, açıkça sentetik test onayı mevcut aritmetik testlerini çalıştırır;
  sentetik fixture gerçek sporcu doğrulaması diye belgelenmez.
- Onaysız şablon üretimi ve “doğrulanmış referans” başarı raporu engellenir.
- Ölçüm verileri ve 133 nokta korunur; null/JSON sözleşmesi bozulmaz.
- İlgili testler, Ruff, tam pytest ve `git diff --check` geçer. Pose inference
  değişmedikçe yeni GPU koşusu gerekmez; gerçek kayıtla puanlama/rapor smoke
  koşusu gerekir ve `latest` davranışı denetlenir.

## 6. P2 — Altı hareketi insanlar tarafından incelenebilir hale getir

Sorumlu: antrenör/hakem ile inceleme; araç ve kayıt desteği yazılım çalışması.

- Önce hareket kimliği ve segment sınırları, sonra değerlendirme aşaması
  incelenir. Sabitleme her teknik için aynı anlamda varsayılmaz.
- Tek bir kesin kare zorlamak yerine kabul edilebilir başlangıç/bitiş
  aralığı, temsil karesi ve “karar verilemedi” seçeneği kaydedilir.
- Temas sayfası kısa bir ön incelemedir; hareketin dinamiği ve geçişi için
  senkron videonun öncesi/sonrası izlenir. M06'nın kayıt sonunda kesilmesi
  sabitleme sonrası pencereyi sınırlar; daha uzun süre varsayılmaz.
- Mümkünse ikinci uzman bağımsız işaretler; anlaşmazlık saklanır. Yalnız tek
  uzman varsa bu sınırlama yazılır, uzmanlar arası uyum iddia edilmez.
- Önceki taslağı görmek inceleyeni yönlendirebilir. Bağımsız doğrulama alt
  kümesinde öneriler kapatılır; önerili inceleme ile bağımsız etiket ayrılır.

Kabul: altı hareketin her biri için **onaylandı / düzeltildi / belirsiz**
kararı ve kayıt bağı; belirsiz hareketin onaylı ölçüm testine alınmaması.
Henüz uzman seçilmedi ve hiçbir inceleme yapılmış sayılmıyor.

## 7. P3–P4 — Üç ölçütü ve kuralını doğrula

Her ölçüt için doldurulacak kart:

- Hareket/teknik, taraf, değerlendirilen faz ve değerlendirme aralığı.
- Eklem noktaları, açı/distance tanımı, koordinat referansı ve birim.
- Yazılı kaynak veya uzman görüşü; sayısal eşiğin nereden geldiği.
- Kabul edilebilir varyasyon, sınır durum ve ölçülemeyen örnek.
- Ölçüm belirsizliği ile teknik toleransın ayrı tanımı.
- Bağımsız referans yöntemi ve onun da hata sınırı.

Bağımsız referans mevcut pose çıktısından veya aynı işaretlerin başka
görselleştirmesinden türetilemez. İki boyutlu kamera üstünde çizilen açı,
uygun düzlem koşulları gösterilmeden gerçek 3B açı sayılmaz. Ölçülmüş sabit
pozlar geometriyi sınar; dinamik performansın tamamını doğrulamaz.

Deneyler:

1. Uzman aralığı sabitken ölçüm hatası: ortalama mutlak hata, sistematik sapma,
   yeterli örnek varsa hata dağılımı/yüzdelikleri ve ölçülebilir örnek oranı.
2. Değerlendirme aralığı değişirken duyarlılık: örneğin anchor ±1/±3/±5 kare
   kayınca değer ve uyarı değişiyor mu? Bunlar deney kaydırmalarıdır; geçerli
   faz toleransı veya yeni sistem varsayılanı değildir.
3. Görünürlük bozulunca davranış: ölçümün yanlış kesinlik yerine belirsiz veya
   ölçülemedi durumuna geçmesi; dışlanan tüm örneklerin gerekçesi.
4. Teknik hata yorumu: ancak uzman referansı varsa yanlış alarm, kaçırılan
   hata ve uzman uyumu; yalnız poz doğruluğundan puan doğruluğu türetilmez.

Kabul sınırları gerçek test sonuçlarına bakılarak sonradan seçilmez. Uzman,
amaçlanan kullanım ve referans belirsizliğiyle **deneyden önce** yazılır.
Şu anda “±X derece yeterli” veya “%Y başarılı” diye uydurulmuş bir kapı yoktur.
Referans sağlanamazsa yalnız duyarlılık/tutarlılık raporu teslim edilir ve
ölçüm doğruluğu aşaması açık kalır.

## 8. P5–P6 — Kayıt ve otomasyonu kanıtla büyüt

Yeni kayıt öncesi mevcut videoda hareket/eklem/kamera bazında kapanma tablosu
çıkarılır. Ek kamera açısı bu kayıplara göre gerekçelendirilir; dokuz kamera
veya ek sensör şimdiden zorunlu satın alma listesi değildir.

Kayıt protokolü: kalibrasyon, senkron kontrolü, kaynak FPS/zaman damgası,
ışık/alan, tam beden görünürlüğü, başlangıç-bitiş payı, kamera yerleşimi ve
değişiklik günlüğü. Katılımcı/kayıt kullanım koşulları üniversiteyle netleşir.

Pilot farklı sporcu, tekrar ve kayıt oturumu içermelidir. Örnek sayısı erişim,
ilk hata dağılımı ve araştırma sorusuyla belirlenir; küçük fizibilite pilotu
genellenebilirlik kanıtı olarak sunulmaz. Geliştirme ve son test ayrımı sporcu
ve oturum düzeyinde yapılır; aynı videonun komşu kareleri iki kümeye dağıtılmaz.

Otomasyon için önce mevcut hareket-enerjisi yöntemi sabit başlangıç olur.
Bilinen hareket sırasından yararlanıldığı açıkça yazılır. Yeni model ancak
ölçülmüş sorun bunu gerektiriyorsa denenir. Rapor: başlangıç/bitiş/faz farkı
(kare ve ms), kaçırılan/ek hareket, kararsızlık oranı, kullanıcı düzeltme sayısı
ve süresi. Öneriler uzman onayına veya skora kendiliğinden dönüşmez.

## 9. P7 — Kullanıcı ve araştırma teslimi

Antrenör/hakem ölçümü ilgili video anına kadar izleyebilmeli, itiraz veya
belirsizlik kaydedebilmeli. Bir bulguyu inceleme süresi ve yanlış anlaşılan
ifadeler kullanıcı denemesinde kaydedilir.

Makale konusu sonuçlara göre daraltılır: örneğin faz seçiminin açı ölçümüne
etkisi veya otomatik öneriyle inceleme yükünün değişmesi. Teslim; araştırma
sorusu, veri/protokol, bağımsız referans, sabit yöntem karşılaştırması, hata
analizi, sınırlamalar ve yeniden üretilebilir komutlardan oluşur. Yayın veya
resmî kullanım kabulü bu planla garanti edilmiş olmaz.

## 10. Kısa, orta ve uzun vade

| Vade | Öncelik | Geçiş koşulu |
| --- | --- | --- |
| Kısa: ilk çalışma döngüsü | P0–P1 ve P2 inceleme hazırlığı | Güven açığı kapalı; gerçek uzman girdisi için paket hazır |
| Orta: uzman ve referans erişimi sonrası | P2–P4; kayıt protokolü | Üç ölçütün tanımı, referansı ve raporlanmış hata sınırı |
| Uzun: ayrı test kayıtları sonrası | P5–P7; kapsamı adım adım 18 harekete büyütme | Yeni veri üzerindeki sonuçlar ve kullanım sınırı açık |

Takvim uzman, sporcu ve ölçüm ortamı erişimine bağlıdır. Bu erişim tarihleri
bilinmeden hafta veya bitiş günü taahhüt edilmez.

## 11. Çalışma günlüğü ve sonraki iş

**20 Eylül 2026 — plan:** Kod/değer/veri ayrımı incelendi; doğrulanmamış
etiketlerin etkilediği yollar yukarıda listelendi.

**22 Eylül 2026 — P0 ve P1 tamamlandı:**

- Merkezi içerik/kapsam kontrolü sayısal, kategorik, duraklama, hakem eşikli
  teknik karar, tam Accuracy, readiness, şablon ve inceleme çıktılarına bağlandı.
- Asıl taslak YAML değiştirilmedi. Ham `manual/confirmed` beyanları korunuyor;
  etkin inceleme durumu `unverified / missing_review_record`.
- Ruff temiz; tam pytest **541 passed in 120.23s**; `git diff --check` temiz.
  Test kaydı: `outputs/pytest-trust-full-20260922-01.log`.
- Mevcut gerçek 3B verisiyle yeni analiz tamamlandı; GPU inference yapılmadı.
  102 bulgu korundu, kesinti toplamı/Accuracy/resmî skor `null` kaldı.
- Aynı güncel 3.2.1 profil ve girdilerle önceki `e21ce51` koduna karşı
  **3.132 teknik kural değeri** eşit. WholeBody hareket ölçümleri de eski
  kaydedilmiş çıktıyla eşit. Eski `benim-denemem-21` teknik profili 3.2.0
  olduğundan onun iki settle-offset değerindeki fark bu regresyona katılmadı.
- Video **1920×1080, 741 kare, 60 FPS, 12,35 saniye**. Kaynak kareler korunuyor;
  uygulanmış kesinti olmayınca eski videodaki kesinti okuma araları eklenmiyor.
  218 ve 637. kareler görsel olarak, son kare okunabilirlik açısından kontrol
  edildi. HTML içeriği doğrulandı; tarayıcı önizlemesi erişim politikasıyla
  engellendiği için bu teslimde HTML için tarayıcı görsel kontrolü yapılmadı.
- İki kameradan altı hareket için 12 şerit ve 82 küçük görüntü üretildi.
  Kaynak yolları/hash'leri mevcut; hiçbir insan onayı eklenmedi.
- Kaynak pose, kontrol edilen eski raporlar ve `latest_run.json` hash'leri
  değişmedi. Yeni çıktılar Git dışındaki ayrı run dizinlerine yazıldı.

Yerel kanıtlar (repo köküne göre):

| Çıktı | Yol |
| --- | --- |
| Yeni analiz raporu | `outputs/poomsae_1_zed2i_20260731_trimmed/runs/trust-p1-20260922-01/review/poomsae_scoring_review.html` |
| İki kamera inceleme paketi | `outputs/poomsae_1_zed2i_20260731_trimmed/runs/annotation-review-20260922-01/review/fixation_review.html` |
| Karşılaştırma ve bütünlük kanıtı | `outputs/poomsae_1_zed2i_20260731_trimmed/runs/trust-p1-verification-20260922-02/verification.json` |

**23–24 Eylül 2026 — CI ve gönderim hazırlığı:** Ruff sürüm farkı yeniden
üretildi; 0.16.8 ve mevcut `E4/E7/E9/F` kapsamı sabitlendi. Araştırma dosyaları
olmayan kaynak kopyasında 540 test geçti, gerçek ZED verisine bağlı tek test
beklendiği gibi atlandı. Ruff ve bağımlılık kontrolü geçti. Gönderim sonrası
aynı commit'in GitHub koşusu ayrıca takip edilir; yerel test uzak CI başarısı
olarak sunulmaz. Ayrıntı `PROJECT_STATUS.md` içindedir.

**Sıradaki iş: P2 uzman incelemesi.** Önce altı hareketin kimliği, sınırları
ve değerlendirme aralıkları kaynak videodan kontrol edilir. Düzeltmeler eski
dosyaya yazılmadan ayrı kaydedilir; yalnız gerçekten incelenmiş kapsam için
bağlı onay hazırlanır. Ardından P3'te üç aday ölçütün tanımı ve bağımsız
referans yöntemi kesinleştirilir. Uzman erişimi henüz bildirilmedi; bu
nedenle P2–P4 tamamlandı veya ölçümler doğrulandı sayılmaz.

Uzmandan istenecek girdiler: [Pilot inceleme formu](PILOT_UZMAN_INCELEME_FORMU.md).
