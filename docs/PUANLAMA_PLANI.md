# TK3D Puanlama Planı

Durum: **uygulama için kabul edilen mimari plan**

Son kaynak doğrulama tarihi: **3 Ağustos 2026**

İlk hedef: **Taegeuk 1 Jang için açıklanabilir, resmî kaynaklara bağlı ve
uzman gerektirmeden çalışabilen provisional puanlama**

## 1. Karar özeti

TK3D puanlama sistemi ilk aşamada uçtan uca bir makine öğrenmesi modeliyle
kurulmayacaktır. İlk üretilecek yapı şunların birleşimi olacaktır:

1. Sürümlenmiş resmî kural paketi,
2. Taegeuk 1 Jang'ın resmî kaynaklara bağlı sıralı hareket tanımı,
3. Sporcuya göre normalize edilmiş 3B/2B ölçümler,
4. Bilinen hareket sırasına bağlı zaman eşleştirme,
5. Her kesintiyi ölçüm ve video kanıtıyla açıklayan kural motoru,
6. Ölçülemeyen durumlarda puan kesilmesini engelleyen güven kapısı,
7. Yeterli teknik etiket oluştuğunda eklenebilen hareket/faz modeli,
8. İleride imkân doğarsa ayrıca eklenebilecek hakem kalibrasyonu.

Kısa karar:

> Önce kural paketi ve Poomsae 1 hareket şeması, sonra ölçüm ve hizalama
> motoru, en son yeterli etiketli veriyle makine öğrenmesi.

Makine öğrenmesi reddedilmemektedir. Ancak resmî kuralların kaynağı, hareket
sırası veya ilk hata kararları ML modeline öğretilerek belirsiz bir kara kutuya
dönüştürülmeyecektir. ML; otomatik hareket bölümleme, görüntüden zor ayrıntıları
algılama ve görüntüden zor ayrıntıları çıkarma gibi veriyle gerçekten
öğrenilebilen alt problemlerde kullanılacaktır. Hakem puanlarına kalibrasyon
mevcut planın önkoşulu değildir; ileride uygun veri bulunursa ayrı bir katman
olarak eklenebilir.

Bu yaklaşımın eldeki koşullarda en doğru başlangıç olduğuna dair gerekçeler:

- Taegeuk 1 Jang serbest bir hareket dizisi değil, sırası önceden bilinen bir
  formdur. Bilinen sırayı modelde kullanmamak gereksiz veri ihtiyacı doğurur.
- Henüz büyük, çeşitli ve hakem etiketli bir poomsae veri setimiz yoktur.
- Puan kesintisinin hangi hareket ve kurala dayandığının açıklanması gerekir.
- Resmî kurallar değişebilir; model yeniden eğitmeden kural sürümü
  değiştirilebilmelidir.
- 3B iskelet her şeyi ölçemez. Bakış, yumruk biçimi, kihap ve ifade gibi
  unsurlar için çoklu veri kanalı ve `ölçülemedi` durumu gerekir.
- Birkaç iyi video yalnız ideal örnek sağlar; kabul edilebilir varyasyonu ve
  küçük/büyük hata sınırlarını tek başına öğretmez.

Hiçbir mimarinin gerçek veride karşılaştırılmadan matematiksel olarak “kesin
en iyi” olduğu iddia edilemez. Bu nedenle bu plan yalnız bir tasarım tercihi
değildir; hareket eşleme ve metrik doğruluğu için geliştirmede kullanılmayan
ayrı test videolarında kabul kontrolleri içerir. Hakem uyumu mevcut imkânlarla
zorunlu bir kapı değildir; yalnız gelecekte hakem eşdeğerliği iddia edilmek
istenirse devreye girer.

## 2. Hedef ürün ve iddia sınırı

İlk ürün “resmî hakemin yerine geçen otomatik sistem” değildir. İlk ürün:

- Taegeuk 1 Jang performansını sıralı hareketlere ayıran,
- hareket başına ölçümleri çıkaran,
- olası hataları kural maddesine bağlayan,
- kesinti adayını video ve sayısal kanıtla gösteren,
- ölçüm güveni yetersizse karar vermeyen,
- kullanıcı veya teknik inceleyicinin kararı gözden geçirip düzeltebildiği

bir **antrenman ve teknik analiz sistemi** olacaktır.

Yetkinlik seviyeleri birbirinden ayrılacaktır:

| Seviye | Anlamı | İzin verilen çıktı |
|---|---|---|
| `measurement_ready` | Poz, zaman, kamera ve iç kalite kapıları geçti | Hareket ve metrik raporu |
| `rule_scoring_ready` | Kural paketi ve Poomsae şeması resmî kaynaklara izlenebilir, iç testleri geçti | Açıklanabilir provisional kural puanı |
| `judge_calibrated_ready` | Opsiyonel; ileride bağımsız hakem verisinde kabul kapıları geçerse | Hakem kalibreli tahmin ve belirsizlik |
| `official_scoring_ready` | Opsiyonel; yetkili kurum/uzman kabulü olmadan daima kapalı | Kapsamı belirtilmiş resmî puan iddiası |

Mevcut `provisional_scoring_ready=true` yalnız ilk seviyenin altyapısına yakın
olduğumuzu gösterir. Tarihsel `71,7178/100` generic değeri WT puanı değildir;
onu üreten kaynaksız legacy motor 12 Ağustos 2026 tarihinde kaldırılmıştır.

### 2.1 Mevcut çalışma koşulu: uzman/hakem erişimi yok

Bu proje planında uzman, koç veya hakem etiketi geliştirme önkoşulu değildir.
Sistem aşağıdaki kaynaklarla ilerleyecektir:

- resmî WT kural ve interpretation belgeleri,
- resmî Kukkiwon Taegeuk 1 eğitim materyalleri,
- kaynak maddesi açık teknik tanımlar,
- aynı tanımın birden fazla resmî/referans gösterimde kontrolü,
- sentetik testler ve geliştirmede kullanılmayan ayrı teknik test videoları,
- her eşik ve formül için açık provenance ve belirsizlik etiketi.

Uzman erişimi olmamasının sonucu geliştirme durması değil, iddia seviyesinin
sınırlandırılmasıdır. Sistem `rule_based_provisional` ve
`provisional_not_judge_validated` sonuç üretebilir. `judge_calibrated_ready` ve
`official_scoring_ready` ise uygun dış doğrulama gelmediği sürece `false`
kalır. Bu iki kapının kapalı olması provisional 0–4 Accuracy, presentation
teşhisi veya açıkça işaretlenmiş provisional 0–10 toplamın üretilmesini
engellemez.

Bağımsız mocap veya ölçülmüş 3B ground-truth bulunmaması geliştirme amaçlı
kural puanlamasını durdurmayacaktır. Bununla birlikte:

- İç RGB/depth tutarlılığı dış 3B doğruluk değildir.
- Hakem etiketi 3B ground-truth değildir; scoring geçerliliğini ölçer.
- Mocap ölçüm doğruluğunu, hakem etiketleri puanlama doğruluğunu sınar.
- Bu iki doğrulama türü raporda ve yetkilendirmede ayrı tutulur.

## 3. Resmî kaynaklar ve yönetişim

### 3.1 Kaynak hiyerarşisi

Puanlama alanında aşağıdaki kaynak sırası uygulanacaktır:

1. Yürürlükteki World Taekwondo Poomsae Competition Rules & Interpretation,
2. İlgili WT teknik/scoring guideline ve ekleri,
3. Kukkiwon'un Taegeuk 1 Jang eğitim materyali,
4. Aynı tekniği açıklayan birden fazla resmî kaynak/gösterimin çapraz kontrolü,
5. Kaynağın sayısal sınır vermediği durumlarda açıkça “engineering provisional”
   işaretli, hassasiyet analizi yapılmış toleranslar,
6. İleride erişim oluşursa bağlayıcı olmayan uzman/hakem geri bildirimi.

2 Ağustos 2026 tarihinde erişilen resmî kaynaklar:

- [WT Poomsae Competition Rules & Interpretation — yürürlük tarihi 30 Eylül
  2024](https://www.worldtaekwondo.org/att_file/documents/Poomsae_Competition_Rules_and_Interpretation_%28In_force_as_of_September_30_2024%29.pdf)
- [World Taekwondo Rules and Regulations](https://www.worldtaekwondo.org/inside/documents/rar/list)
- [Kukkiwon Taegeuk 1 Jang resmî sayfası](https://www.kukkiwon.or.kr/eng/board/read?boardManagementNo=55&boardNo=1347&menuLevel=3&menuNo=72&page=2&searchCategory=&searchType=&searchWord=)
- [Kukkiwon renkli kuşak poomsae eğitim videoları duyurusu](https://kukkiwon.or.kr/eng/board/read?boardManagementNo=49&boardNo=3515&menuLevel=2&menuNo=91%3FselectMenuCode%3DPE020300&page=3&searchCategory=&searchType=&searchWord=)
- [Kukkiwon 2025 Taegeuk 1 ayrıntılı eğitimi](https://www.youtube.com/watch?v=__4pltpaPJo)
- [Kukkiwon Taegeuk 1 kısa tam sıra gösterimi](https://www.youtube.com/watch?v=FmlZHqV9Y-M)

WT'nin 2 Ağustos 2026 tarihindeki canlı `POOMSAE / CURRENT` listesi 30 Eylül
2024 belgesini güncel kural olarak göstermektedir. Yerel olarak doğrulanan
PDF'nin SHA-256 değeri
`3fc994363544ab2a1717d9d4d805b368e283a122d3f3c52fc444c70b8c206e24`'tür.
Article 16'nın atıf yaptığı ayrı scoring guideline dosyası güncel kamuya açık
listede bulunamamıştır. Ayrıntılı erişim, ücret ve kaynak boşluğu kaydı
[`SCORING_SOURCE_REGISTER.md`](SCORING_SOURCE_REGISTER.md) içindedir.

3 Ağustos 2026 derin taramasında aynı guideline'ın 2014 WT kural kitabına ekli
resmî tarihsel sürümü ve ayrı tam 43 sayfalık arşivi doğrulanmıştır.
Masaüstündeki 35 sayfalık kopya bu belgenin eksik kopyasıdır; ilk 35 sayfanın
metni tam dosyayla eşleşir. Tarihsel teknik tanımlar hata adayı ve metrik
tasarımını güçlendirir fakat güncel WT toleransı sayılmaz. Ayrıntılı hata ve
ölçülebilirlik matrisi
[`TAEGEUK1_ERROR_TAXONOMY.md`](TAEGEUK1_ERROR_TAXONOMY.md) içindedir.

Kural uygulamasına başlamadan ve her yayımdan önce WT sayfası tekrar kontrol
edilecektir. Dosyanın URL'si kadar SHA-256 özeti, yürürlük tarihi, erişim tarihi,
dil ve sayfa/madde bilgisi de kaydedilecektir.

### 3.2 PDF'den doğrudan puan verilmemesi

LLM veya PDF ayrıştırıcı, belgeyi okuyup aday kuralları tabloya dönüştürebilir;
ancak çıkarılan metin otomatik olarak aktif kural olmayacaktır. Akış:

```text
resmî PDF
  -> madde ve sayfa referanslı aday kural
  -> teknik anlam ve kaynak tutarlılığı kontrolü
  -> ölçülebilirlik kontrolü
  -> iç kaynak denetimi
  -> sürümlenmiş provisional RulePack
```

PDF metni her video çalışmasında yeniden yorumlanmayacaktır. Böylece aynı
video aynı kural sürümüyle deterministik sonuç verir ve sonradan “model kuralı
farklı yorumladı” sorunu oluşmaz.

Resmî metinde sayısal tolerans yoksa ortaya çıkan açı veya mesafe eşiği resmî
kural gibi gösterilmeyecektir. Provisional mühendislik sınırları:

1. Teknik tanım,
2. Birden fazla resmî veya açık provenance'lı referans performans,
3. Kabul edilebilir ve hatalı örnek dağılımları,
4. Eşik hassasiyet analizi,
5. Geliştirmede kullanılmayan ayrı video kontrolü

birlikte kullanılarak belirlenecektir. Bu eşikler
`engineering_provisional=true` taşır ve ileride uzman geri bildirimi gelirse
ayrı sürümde güncellenir.

## 4. WT puan yapısının sistemde temsili

30 Eylül 2024 yürürlük tarihli WT belgesindeki Recognized Poomsae yapısı
esas alındığında toplam puan `10,0` ve ana bölümler şöyledir:

- Accuracy: `4,0`
- Presentation: `6,0`
  - Speed and power: `2,0`
  - Control of power, speed and rhythm: `2,0`
  - Expression of energy: `2,0`

Accuracy tarafında belgelenen küçük hata `-0,1`, büyük hata `-0,3` ve yeniden
başlama `-0,6` anlamları kural paketinde doğrudan kaynak maddesine bağlanacaktır.
Belge sürümü değişirse eski koşular eski paketle yeniden üretilebilir kalmalı,
yeni paket mevcut çıktının üzerine yazmamalıdır.

### 4.1 Accuracy motoru

Accuracy motoru `4,0` puandan başlar ve yalnız kanıtlı olaylar için kesinti
adayı oluşturur. Her aday en az şu alanları taşır:

```yaml
deduction_id: D-000123
rule_id: WT-2024-A16-...
poomsae_id: taegeuk_1_jang
movement_id: M07
phase_id: execution
severity: minor
deduction: 0.1
metric_id: stance.length.normalized
observed_value: 0.82
accepted_range: [0.95, 1.10]
unit: leg_length_ratio
start_frame: 312
apex_frame: 321
end_frame: 329
measurement_confidence: 0.91
supporting_cameras: [camera_1, camera_2]
evidence_status: observed
source_ref: WT-2024-p24-A16
review_status: pending
```

Bu yalnız şema örneğidir; örnekteki hareket ve toleranslar onaylanmış teknik
kural sayılmaz.

Aynı geometrik sapma birden fazla metrik tarafından görülürse iki kez puan
kırılmamalıdır. Metrikler önce tek bir hata olayında birleştirilir, ardından
kural uygulanır. Ham ölçüm, teknik sapma, kesinti adayı ve nihai skor ayrı veri
katmanlarıdır.

### 4.2 Presentation motoru

Presentation puanı kare başına ceza ortalamasıyla hesaplanmayacaktır. Bütün
performans veya anlamlı hareket grupları boyunca şu özelliklerden yararlanır:

- zirve el/ayak hızları ve kontrollü sonlanma,
- hızlanma, yavaşlama ve hareketler arası kontrast,
- geçiş ve uygulama süreleri,
- ritim düzeni ve bilinçli tempo değişimleri,
- sağ/sol taraf zamanlama tutarlılığı,
- gövde-kalça transferi ve son pozisyon kararlılığı,
- bakış, ifade ve kihap gözlemleri,
- başlangıç/bitiş tutumu ve genel bütünlük.

3B iskelet gerçek kuvvet ölçmez. Kuvvet platformu olmadan üretilen değer
`power` değil `kinematic_power_proxy` olarak adlandırılacaktır. Benzer şekilde
COM ve ayak geometrisinden hesaplanan sonuç gerçek dinamik denge değil
`kinematic_balance_proxy` olacaktır.

İlk presentation çıktısının birincil ürünü bileşenlere ayrılmış teşhis profili
olmalıdır. Kullanıcıya toplam puan sunmak için ayrıca açık formüllü ve sürümlü
bir `presentation_provisional_0_6` üretilebilir. Bu değer
`judge_calibrated=false` ve `provisional_not_judge_validated` taşır; resmî hakem
puanı olarak gösterilmez. Çoklu hakem etiketi ileride bulunursa mevcut formül
silinmeden ayrı bir kalibrasyon profili eklenebilir.

## 5. Poomsae 1 hareket modeli

### 5.1 Sıralı hareket grafiği

Taegeuk 1 Jang, serbest sınıflandırma yerine sırası zorunlu bir hareket grafiği
olarak temsil edilecektir. Her düğüm tek bir görsel pozdan ibaret değildir;
duruş, dönüş ve eşzamanlı tekniği birlikte kapsayabilir.

Her hareket kaydı şunları içerir:

- `movement_id` ve resmî kaynağa bağlı adı,
- sıra ve izin verilen geçişler,
- aktif taraf ve beklenen yön,
- başlangıç ve bitiş duruşu,
- ana ve yardımcı teknikler,
- hazırlık, geçiş, uygulama, zirve/sabitleme ve toparlanma fazları,
- beklenen el/ayak yolu,
- vücuda göre hedef yüksekliği,
- el, ayak, kalça ve gövdenin bitiş ilişkisi,
- beklenen bakış yönü,
- varsa kihap olayı,
- ölçülebilir metrikler,
- küçük/büyük hata adayları,
- zorunlu sensör ve minimum kanıt şartı,
- kaynak referansı, çıkarım yöntemi ve iç inceleme durumu.

Tam hareket listesi bellekten veya genel internet özetinden yazılmayacaktır.
Resmî Kukkiwon materyali ve tekrar edilebilir kaynak denetimiyle çıkarılacaktır.
Kaynaklar arasında belirsiz kalan hareket veya teknik otomatik kesinti üretemez;
`source_ambiguity` olarak raporlanır.

### 5.2 Hareket fazları

Önerilen ortak faz sözleşmesi:

```text
preparation -> transition -> execution -> apex/fixation -> recovery
```

Her teknikte bütün fazların bulunması zorunlu değildir. Poomsae şeması hangi
fazların anlamlı olduğunu açıkça belirtir. Metrikler doğru fazda hesaplanır:

- duruş genişliği çoğunlukla `apex/fixation`,
- el/ayak yolu `execution`,
- hazırlık konumu `preparation`,
- denge ve sallanma `apex/fixation` sonrası,
- tempo `transition + execution`

üzerinden değerlendirilir. Böylece geçiş karesindeki doğal bükülme yanlışlıkla
son pozisyon hatası sayılmaz.

## 6. Girdi ve koordinat mimarisi

Ana sözleşme korunur:

```text
keypoints_3d_world[t, 133, 3]
metre
x = sağ, y = ileri, z = yukarı
```

Puanlama iki koordinat düzeyini birlikte kullanmalıdır:

1. **Dünya/mat koordinatı:** başlangıç-bitiş konumu, çizgi düzeni, dönüş ve
   yer değiştirme için.
2. **Sporcu-merkezli koordinat:** teknik açı, duruş, sağ/sol ve hedef yüksekliği
   için.

Sporcu-merkezli eksen; pelvis, kalça, omuz ve güvenilir ayak yönünden
hesaplanacaktır. Tek bir karede kararsız yön kullanılmayacak, zaman içinde
güvenli ve süreklilik taşıyan heading kestirilecektir.

Mutlak metre eşikleri doğrudan farklı boydaki sporculara uygulanmayacaktır.
Mesafeler uygun durumda omuz genişliği, kalça genişliği, bacak uzunluğu veya
boy tahminiyle normalize edilir. Referans sporcu iskeleti hedef kişiye kemik
oranları korunarak uyarlanır; tek bir referans sporcunun mutlak eklem koordinatları
“evrensel doğru” yapılmaz.

### 6.1 Puanlama öncesi zorunlu eksen düzeltmesi

`src/scoring_metrics.py` içindeki mevcut gövde eğimi hesabı `y=yukarı` ve
`z=ileri` varsaymaktadır. Projenin kanonik sözleşmesi ise `y=ileri`,
`z=yukarı`dır. Poomsae metrikleri geliştirilmeden önce:

- bütün scoring metrikleri `src/coordinate_system.py` sözleşmesine bağlanmalı,
- eksenler için sentetik testler eklenmeli,
- sağ/sol, ileri/geri ve yukarı/aşağı dönüşümler ayrı doğrulanmalı,
- eski provisional skor yeni motorun referansı yapılmamalıdır.

Bu düzeltme yapılmadan gövde eğimi, stance length veya yön tabanlı skor
uygulanmamalıdır.

## 7. Sensör kanıtı ve ölçülebilirlik

| Veri | Kullanım | Sınır |
|---|---|---|
| Çoklu RGB | 2B eklem, el/ayak görünümü, görsel kanıt | Örtüşme ve motion blur |
| Stereo depth | 3B derinlik desteği ve güven kapısı | Yüzey mesafesi eklem merkezi değildir |
| Kalibrasyon | Ortak dünya ve reprojection | İç doğruluk kanıtıdır, mocap değildir |
| Zaman damgası | Kamera ve hareket fazı senkronu | Kayıp/tekrar kare açıkça raporlanır |
| ZED IMU | Kamera yönelimi ve yerçekimi | Sporcunun IMU'su değildir |
| WholeBody el/ayak/yüz | Yumruk, ayak ve bakış için yardımcı kanıt | Mevcut cross-view optimizasyon BODY-17 ile sınırlı |
| Harici ses | Kihap olayı | ZED kaydı tek başına ses garantisi vermez |

Her metrik şu gözlem durumlarından birini üretir:

- `observed`: yeterli bağımsız görüntü/sensör kanıtı var,
- `partially_observed`: kısıtlı kanıt var, yalnız uyarı üretilebilir,
- `inferred`: model veya zaman bilgisinden tamamlandı, bağımsız kanıt değildir,
- `not_measurable`: karar verilemez.

`not_measurable` ve `inferred` durumları otomatik negatif kanıt veya kesinti
olamaz. Eksik eklem `0` kabul edilemez; JSON'da `null`, CSV'de boş kalır.

Üçüncü kamera eklendiğinde kural motoru değişmeyecektir. Kamera sayısı yalnız
kanıt ve observability katmanını etkiler. Üç kamera daha iyi görüş ve robust
triangulation sağlayabilir; ancak mevcut en az dört destek kamera gerektiren
cross-view 2B geri besleme koşulunu tek başına sağlamaz.

## 8. Hareket bölümleme ve sıralı eşleştirme

Mevcut `motion_candidate` segmentleri yalnız hareket enerjisi aralıklarıdır;
Taegeuk 1 Jang adımları değildir. Nihai akış:

```text
eklem hızları + ayak sabitliği + heading dönüşü + poz değişimi
  -> aday sınırlar ve apex olayları
  -> Taegeuk 1 sırasına kısıtlı monoton hizalama
  -> hareket/faz zaman çizelgesi
  -> güven ve alternatif hizalama raporu
```

İlk sürümde önerilen yöntem:

1. Mevcut video elle hareket `start/end` aralıkları ve o harekete özgü faz
   anchor'larıyla etiketlenir. Tekme-vuruş gibi bileşiklerde tek bir `apex`
   alanına zorlanmaz; örneğin `kick_apex`, `landing` ve `fixation` ayrı olabilir.
2. Kinematik olay algılayıcı sınır adayları üretir.
3. Bilinen Poomsae sırasına Dynamic Time Warping veya dinamik programlama ile
   monoton eşleştirme yapılır.
4. Atlama, tekrar, uzun duraklama ve yanlış sıra ayrı olaylar olarak tutulur.
5. Düşük güvenli eşleşme kullanıcı incelemesine gönderilir.

Etiketli veri büyüdüğünde olay algılayıcı, hareket/faz olasılıkları üreten bir
modelle değiştirilebilir. Ancak model çıktısı yine Poomsae grafiği içindeki
HSMM/Viterbi benzeri kısıtlı çözücüden geçmelidir. Model tek başına sıralamayı
serbestçe değiştirememelidir.

## 9. Metrik kataloğu

Her metrik ayrı bir kimlik, birim, koordinat düzeyi, gerekli eklemler,
uygulanabilir faz, güven hesabı ve kaynak taşır.

### 9.1 Duruş ve alt vücut

- normalize duruş genişliği ve uzunluğu,
- ön/arka ayak yönelimi,
- ayaklar arası açı,
- diz ve kalça fleksiyonu,
- ağırlık aktarımının kinematik göstergesi,
- pelvis yüksekliği ve yönü,
- diz-ayak hizası,
- duruş sonundaki ayak kayması.

Ayak yönü yalnız ankle noktalarından güvenilir çıkmıyorsa WholeBody ayak
noktaları veya RGB kanıtı zorunlu tutulmalıdır.

### 9.2 El-kol teknikleri

- hazırlık/chamber konumu,
- aktif elin vücuda göre yolu,
- hedef yüksekliği,
- omuz-dirsek-bilek açıları,
- bitiş uzanımı ve dirsek kontrolü,
- diğer elin toparlanma konumu,
- el ve duruşun eşzamanlı tamamlanması,
- yumruk/elin yönü ölçülebiliyorsa WholeBody el geometrisi.

2B iskelet veya BODY-17 tek başına yumruğun sıkılığını kanıtlayamaz. Bu tür
ölçümler RGB/el keypoint görünürlüğüne göre kapılanır.

### 9.3 Yön, yol ve sıra

- sporcu heading'i ve beklenen yön,
- dönüş açısı,
- mat üzerindeki yer değiştirme,
- başlangıç ve bitiş bölgesi,
- hareket sırası, tekrar ve atlama,
- bakış yönü ölçülebiliyorsa yüz/RGB desteği.

### 9.4 Zamanlama ve sunum göstergeleri

- hareket, geçiş ve sabitleme süreleri,
- zirve hız ve ivme,
- son pozisyonda frenleme,
- el/ayak/kalça bitiş senkronu,
- ritim aralıkları ve varyasyon,
- hareket sonrası sallanma,
- kinematik güç ve denge göstergeleri.

## 10. Makine öğrenmesi kullanım politikası

### 10.1 İlk aşamada ML kullanılmayacak kararlar

- Resmî kuralın ne olduğu,
- Poomsae hareket sırası,
- Küçük/büyük hata kesinti miktarı,
- Kaynak belgesiz sayısal tolerans,
- Ölçülemeyen bir hareketin doğru/yanlış sayılması,
- Tek bir nihai skordan geriye doğru açıklama üretmek.

### 10.2 Uygun ML görevleri

- hareket sınırı, apex ve faz olasılığı,
- görüş/örtüşme ve ölçüm güveni tahmini,
- el, ayak, bakış ve görsel teknik ayrıntıları,
- farklı vücutlar için kabul edilebilir varyasyon modeli,
- kural tabanlı teknik özelliklerden provisional skor kalibrasyonu,
- ileride veri bulunursa hakem puanı ve hakemler arası dağılım modellemesi.

### 10.3 Neden çok sayıda etiketsiz video yeterli değildir?

Etiketsiz videolar hareket görünümünü ve kamera çeşitliliğini öğretir; ancak
hangi tekniğin doğru olduğunu, hangi sapmanın küçük/büyük hata sayıldığını veya
hakemin presentation puanını öğretmez. Etiketsiz veri self-supervised ön
eğitim, pose sağlamlığı ve veri artırımı için kullanılabilir. İlk kural tabanlı
puanlama resmî kaynaklardan kurulabilir; yalnız “hakemle eşdeğer” kalibrasyon
iddiası için ileride hakem etiketi gerekir.

### 10.4 ML'ye geçiş kapısı

Bir alt problem için ML ancak şu koşullarda eklenir:

- açık etiket sözleşmesi vardır,
- sporcu bazında ayrılmış train/validation/test vardır,
- kural tabanlı basit baseline kaydedilmiştir,
- model geliştirmede kullanılmayan ayrı testte baseline'ı önceden seçilmiş
  metrikte geçmiştir,
- farklı boy, seviye, kıyafet ve kamera düzeninde alt grup hataları raporlanır,
- güven skoru gerçek hata olasılığıyla kalibre edilmiştir,
- başarısızlıkta güvenli kural/inceleme fallback'i vardır.

Modelin daha karmaşık olması tek başına kabul nedeni değildir.

## 11. Veri toplama ve etiketleme planı

### 11.1 Referans katmanları

Tek bir “mükemmel video” yerine üç veri katmanı hedeflenir:

1. **Resmî/referans gösterimler:** doğru sırayı ve teknik açıklamayı veren
   Kukkiwon materyalleri ve provenance'ı açık çoklu performanslar.
2. **Kabul edilebilir varyasyon:** farklı boy, yaş, seviye ve beden oranlarında
   kaynakla çelişmeyen performanslar.
3. **Kontrollü sapma örnekleri:** belirli bir ölçümün kasıtlı değiştirildiği
   veya sentetik olarak üretildiği teknik testler.

Mevcut kısa video ilk annotation ve pipeline örneği olarak kullanılabilir;
tek başına canonical doğru veya skor doğrulama seti olamaz.

### 11.2 Etiket şeması

Uzman gerektirmeyen zorunlu teknik etiketler:

- anonim `athlete_id`, seviye ve gerekli izinler,
- `session_id`, `run_id`, kamera/sensör profili,
- rule pack ve poomsae spec sürümü,
- hareket kimliği,
- `start/end` kareleri ve harekete özgü faz anchor'ları,
- görülen teknik/duruş/yön,
- kaynak tabanlı provisional hata adayı,
- hatanın kanıtlandığı kamera/kare,
- ölçüm güveni ve observability durumu,
- provisional accuracy kesintisi,
- üç provisional presentation özelliği

tutulmalıdır.

`judge_score`, `judge_deduction`, `judge_confidence` ve benzeri hakem alanları
şemada opsiyonel kalır ve mevcut koşulda `null` olur. Bunlar provisional motoru
çalıştırmak için zorunlu değildir.

İleride hakem erişimi oluşursa hakemler mümkün olduğunca birbirlerinin puanını
görmeden bağımsız etiketleme yapar. Sürekli puanlar için ICC, kategorik hata
etiketleri için uygun kappa/uyum metriği raporlanır. Bu gelecekteki veri mevcut
provisional sonuçların üzerine yazılmaz; yeni kalibrasyon sürümü oluşturur.

### 11.3 Yaklaşık veri hedefleri

Bunlar garanti veya resmî minimum değil, planlama aralığıdır:

- Şema/prototip: resmî gösterimler ve mevcut videonun elle etiketlenmesi,
- hareket/faz pilotu: yaklaşık `50–100` etiketli tam performans,
- erken çeşitlilik: yaklaşık `20–30` sporcu, kişi başına `2–3` tekrar,
- gelecekte opsiyonel hakem kalibrasyonu düşünülürse: yaklaşık `100–300`
  çeşitli performans ve video başına tercihen en az `3` bağımsız hakem.

Veri azsa nihai puan iddiası küçültülür; aynı test videoları tekrar tekrar eşik
ayarlamak için kullanılmaz.

### 11.4 Veri bölme

- Aynı sporcu train ve testte bulunmamalıdır.
- Aynı çekimin farklı kameraları farklı splitlere ayrılmamalıdır.
- Aynı oturumdan türetilmiş kırpılmış/augment edilmiş kayıtlar tek splitte
  kalmalıdır.
- Final test seti eşik ve model seçimi bitene kadar kilitli tutulmalıdır.

## 12. Yazılım bileşenleri

Mevcut `scoring_readiness.py`, `scoring_authorization.py`, kalite raporları ve
`keypoints_3d_world[t,133,3]` sözleşmesi yeniden kullanılmalıdır. Yeni alan
mantığı ayrı ve test edilebilir bir paket olarak planlanır:

```text
src/poomsae_scoring/
  rule_pack.py
  poomsae_spec.py
  body_frame.py
  observability.py
  event_detection.py
  sequence_alignment.py
  metric_registry.py
  accuracy_engine.py
  presentation_features.py
  judge_calibration.py
  report_builder.py

config/scoring/
  rules/wt_recognized_2024.yaml
  poomsae/taegeuk_1_jang_v1.yaml
  metrics/taegeuk_1_metrics_v1.yaml
  calibration/<profile>.yaml
```

Kaynaksız `src/scoring_engine.py`, yinelenen `config/scoring_config.yaml` ve
0-100 generic skor çıktıları kaldırılmıştır. Tarihsel sonuçlar yalnız eski run
provenance'ında korunur; yeni WT skoruyla aynı alan veya dosya adı kullanılmaz.

### 12.1 Katman sınırları

```text
Evidence
  -> Normalization
  -> Observability
  -> Event/Phase Detection
  -> Constrained Poomsae Alignment
  -> Metric Observations
  -> Technique Deviations
  -> Deduction Candidates
  -> Accuracy + Presentation
  -> Uncertainty Gate
  -> Human Review + Report
```

Bu katmanların hiçbiri kendinden önceki ham veriyi silmez. Örneğin kesinti
motoru keypointleri değiştiremez; alignment motoru düşük güvenli hareketi
zorla doğru kabul edemez.

## 13. Çıktı sözleşmesi

Her puanlama çalışması aynı pose koşusuna bağlanmış benzersiz bir alt koşu veya
run kimliğiyle saklanmalıdır:

```text
outputs/<session_id>/runs/<run_id>/scoring/<scoring_run_id>/
  rule_pack_manifest.json
  poomsae_spec_manifest.json
  evidence_summary.json
  movement_timeline.json
  metric_observations.json
  deduction_candidates.json
  presentation_features.json
  scoring_report.json
  scoring_review.html
  csv/movement_timeline.csv
  csv/metric_observations.csv
  csv/deduction_candidates.csv
  clips/<movement-or-deduction>.mp4
```

Her sonuçta şunlar bulunur:

- kaynak pose run kimliği ve SHA-256 özeti,
- kod commit'i,
- kural/spec/metrik/kalibrasyon sürümü ve özeti,
- kullanılan kamera ve sensör kanıtı,
- değerlendirme kapsamı ve kırpma bilgisi,
- ölçülebilen hareket/metrik oranı,
- her kesinti için kare ve video kanıtı,
- provisional/kalibre/resmî durum etiketi,
- belirsizlik ve inceleme gerektiren olaylar.

`NaN`/`inf` JSON veya CSV'ye yazılmaz. Eksikler JSON'da `null`, CSV'de boş
hücre olur. Başarısız veya yarım koşu `latest_run.json` hedefi yapılmaz.

## 14. Doğrulama stratejisi

### 14.1 Birim ve sentetik doğrulama

- eksen ve birim dönüşümleri,
- sağ/sol aynalama,
- sporcu heading sürekliliği,
- kemik oranı normalizasyonu,
- sentetik duruş açıları,
- faza göre metrik seçimi,
- küçük/büyük kesinti ve tekrar engelleme,
- eksik veride `not_measurable`,
- kural/spec şema doğrulaması,
- aynı girdi/sürümde deterministik sonuç.

### 14.2 Hareket eşleme doğrulaması

Geliştirmede eşik ayarlamak için kullanılmayan, hareket sınırları teknik olarak
elle işaretlenmiş ayrı test videolarında:

- hareket kimliği doğruluğu,
- başlangıç/apex/bitiş sınır hatası,
- atlanan ve tekrar edilen hareket yakalama,
- tam dizi hizalama başarısı,
- düşük güvenin gerçek hata ile kalibrasyonu

ölçülür. Kabul eşikleri pilot veriden önce yazılı olarak sabitlenir; final test
sonucuna bakılarak geriye dönük gevşetilmez.

### 14.3 Metrik doğrulaması

Örnek karelerde bağımsız hesap, görsel kontrol veya uygun ölçüm aracıyla elde
edilen açı/mesafe/zaman ile sistem sonucu karşılaştırılır. Kamera sayısı,
örtüşme, sporcu boyu ve teknik fazına göre hata dağılımı raporlanır. Aynı pose
hatasını paylaşan iki metrik birbirinin ground-truth'u sayılmaz.

### 14.4 Opsiyonel gelecek kapısı: hakem uyumu

Bu bölüm mevcut geliştirmeyi veya provisional puanı engellemez. Yalnız ileride
`judge_calibrated_ready` ya da `official_scoring_ready` iddiası istenirse:

- accuracy kesinti precision/recall ve hata türü karışıklığı,
- presentation bileşen MAE ve sıralama korelasyonu,
- toplam puan MAE/yanlılık,
- sistem-hakem ve hakem-hakem uyumu,
- hata şiddeti ve sporcu alt gruplarına göre sonuç,
- bootstrap güven aralıkları

raporlanır. Bu veri yokken ilgili alanlar `not_evaluated` kalır; provisional
kural puanına başarısızlık olarak yansımaz.

### 14.5 Ablation ve mimari doğrulama

“En doğru yol” iddiası aşağıdaki kontrollü karşılaştırmayla sınanacaktır:

1. Mevcut generic frame-score baseline,
2. Kural + sıralı hizalama sistemi,
3. Yeterli veri oluştuğunda öğrenilmiş faz modeli + kural motoru,
4. Yalnız araştırma karşılaştırması olarak end-to-end skor modeli.

Aynı geliştirmede kullanılmayan ayrı test verisi, aynı pose girdisi ve aynı
kapsam kullanılır. Hibrit yapı teknik doğruluk, tutarlılık veya
açıklanabilirlik kabul kapısını geçmezse ilgili karar yeniden gözden geçirilir;
başarısız sonuç saklanır.

## 15. Uygulama yol haritası

### Aşama 0 — Teknik sözleşmeyi düzelt

İlerleme: **Eksen düzeltmesi 2 Ağustos 2026 tarihinde tamamlandı.** Kaynaksız
generic baseline 12 Ağustos 2026 tarihinde kaldırıldı.

İşler:

- [tamamlandı] `scoring_metrics.py` eksen uyuşmazlığını düzeltmek ve sentetik
  eksen testlerini eklemek,
- [tamamlandı] scoring terimlerini ölçüm/rule/judge/official olarak ayırmak,
- [tamamlandı] eski `71,7178` skor motorunu kaldırıp sonucu tarihsel kayıt olarak etiketlemek,
- sentetik koordinat ve metrik testlerini eklemek.

Çıkış kapısı: Bütün sentetik yön, açı ve birim testleri geçmeli; yeni motor
eski provisional score alanını kullanmamalı.

### Aşama 1 — Kaynak kayıt sistemi ve RulePack

İşler:

- [tamamlandı] WT PDF kimliği, SHA-256 özeti ve kaynak sicilini kaydetmek,
- [tamamlandı] Article 15/16 kesinti semantiğini sürümlü RulePack'e geçirmek,
- [tamamlandı] Accuracy ve Presentation bütçelerini ayırmak,
- [tamamlandı] benzersiz YAML anahtarı ve fail-closed şema doğrulamasını
  kurmak,
- [kısmen kapatıldı] Article 16'nın işaret ettiği scoring guideline'ın 2014
  resmî tarihsel 43 sayfalık sürümünü edinmek; güncel ayrı ek hâlâ kamuya açık
  listede bulunmadığından tarihsel ölçüleri aktif tolerans yapmamak.

Çıkış kapısı: Her aktif kuralın belge, sayfa/madde, çıkarım türü ve kaynak
denetim kaydı bulunmalı. Belgesiz kural aktif olmamalı.

### Aşama 2 — Taegeuk 1 Jang PoomsaeSpec

İşler:

- [tamamlandı - taslak] 2022 tam sıra gösterimi ve 2025 ayrıntılı eğitimden
  18 sıralı hareket birimini çıkarmak,
- [tamamlandı - taslak] hareket, teknik, duruş, yön ve çoklu faz sözleşmesini
  oluşturmak,
- [tamamlandı - M01-M06 engineering v1] ölçülebilir/ölçülemez kriterleri ve
  ayrı, resmî olmayan mühendislik tolerans politikasını doldurmak,
- [devam ediyor] özellikle tekme-vuruş bileşiklerinin tablo sınırlarını ikinci
  resmî kaynakla çapraz kontrol etmek.

Çıkış kapısı: Sıra ve tekniklerin kaynak izi tam; belirsiz kural aktif puan
üretmiyor ve `source_ambiguity` olarak raporlanıyor.

### Aşama 3 — Annotation ve referans seti

İlerleme: ZED2i pose çıktısı SHA-256 ile taslak `MovementTimeline` sözleşmesine
bağlandı. İki kamera incelemesinde kısa kaydın yalnız M01-M06'yı içerdiği
doğrulandı; bu altı hareketin `24` faz anchor'ı girildi. İki kameralı temas
sayfası incelemesinden sonra zaman sınırları `confirmed` oldu; bu teknik
doğruluk onayı değildir.
M07-M18 videoda bulunmadığı için üretilmedi ve hazırlık kapısı kapalıdır.
MovementTimeline v2 bunu `partial_sequence` kapsamıyla etiket eksikliğinden
ayırır. Senkron iki-kamera HTML inceleme ekranı aynı zaman çizelgesine bağlıdır.

İşler:

- mevcut 12,35 saniyelik videoda hareket aralığı ve hareketin anlamlı faz
  anchor'larını etiketlemek,
- resmî/referans gösterimleri aynı şemada etiketlemek,
- annotation aracı ve gözden geçirme geçmişi oluşturmak,
- tek videodan evrensel tolerans üretmemek.

Çıkış kapısı: Etiket sözleşmesi tamamlanmış; hareket sınırları ikinci bir
zamanda tekrar incelenmiş ve değişiklik geçmişi saklanmış.

### Aşama 4 — Kısıtlı sequence alignment

İşler:

- olay adaylarını çıkarmak,
- bilinen sıraya monoton hizalama yapmak,
- tekrar, atlama, uzun durma ve düşük güveni raporlamak,
- elle etiketli videolarla boundary/sequence ölçmek.

Çıkış kapısı: Önceden belirlenen ayrı test videosu hareket eşleme kapıları
geçmeli.

### Aşama 5 — Metrik ve observability motoru

İlerleme: İlk BODY-17 v1 ölçümü yetersiz bulunarak sayısal sonuç üretme yetkisi
kaldırıldı. WholeBody-133 v2; gövde, ayak, yüz ve iki eli kullanarak stance,
ayak yönü, uygulayan taraf, hikite, el/bilek geometrisi, bakış-gövde yönü,
eşzaman, fixation, ağırlık aktarımı ve trajectory ölçer. Gerçek kayıtta eşikli
`96` metriğin `74` tanesi ölçülebilir, `22` tanesi ölçülemez kaldı; `%77,08`
kapsam `%90` kapısını geçmedi.

3 Ağustos 2026 genişletmesinde bütün M01-M18 hareketleri ölçülebilir kriter
listesine kavuştu. M14/M16 `ap-chagi + momtong-jireugi` hareketlerinde tekme
zirvesi, dizin yeniden toplanması, destek ayağı pivotu, iniş ve yumruk zamanı
ayrı metriklerdir. Güncel sayısal tolerans olmadığı için bu beş metrik yalnız
teşhis değeri üretir; aday/kesinti üretmez. `momtong-an-makki` ve
`eolgul-makki` yükseklik/dirsek değerleri de kaynak gelene kadar eşiksiz
teşhistir.

WholeBody v2.4, eski `iki ayağın birbirine göre yaw farkı` metriğini kaldırır.
Arka ayak yönü, arka ayak bileğinden ön ayak bileğine duruş doğrultusuna göre
ölçülür. Fixation penceresindeki sağlam MAD dağılımından `%95` belirsizlik ve
örnek sayısı taşınır. Arae-makki için yumruk merkezi-uyluk uzaklığı yumruk
genişliğine normalize edilir.

İşler:

- body-frame ve ölçek normalizasyonu,
- stance, teknik, yön, zaman ve denge göstergeleri,
- eklem/kamera/faz bazlı güven,
- `observed/partial/inferred/not_measurable` durumları,
- ham ölçüm ve video kanıtı çıktısı.

Çıkış kapısı: Ölçümler elle doğrulanan örneklerde kabul aralığında; eksik veri
puan kesmiyor.

### Aşama 6 — Accuracy 4,0 motoru

İlerleme: Kaynak bağlı `4,0` başlangıç ve `0,1 / 0,3 / 0,6` kesintilerini
uygulayan çekirdek motor tamamlandı. Motor yalnız aktif PoomsaeSpec, eksiksiz
MovementTimeline ve gözlenmiş/kuralca doğrulanmış olay kabul eder. Buna ek
olarak resmî motordan ayrılmış WholeBody-133 v2 teşhis motoru tamamlandı. İlk
BODY-17 `4,0` denemesi maksimum skor izlenimi verdiği için geçersizleştirildi;
artık `partial_engineering_trial_score=null` üretir. WholeBody koşusu `13`
video-inceleme adayı üretti fakat eşikler resmî/kalibre olmadığı için kesinti
ve skor üretmedi; `accuracy_score=null` kaldı.

Source-bound v1 katmanı, tarihsel resmî 2014 kılavuzundaki açık geometrileri
güncel WT eki diye sunmadan ayrı profile bağlar: ap-seogi/apkubi arka ayak
`30°`, arae-makki yumruk-uyluk `1–2 yumruk`, momtong-an-makki dirsek
`90–120°`. Ölçümün `%95` aralığı kaynak sınırının tamamıyla dışına çıkmadıkça
kesinti uygulanmaz. Sayısal geometri yalnız `minor=-0,1` oluşturabilir;
`major=-0,3` sayısal açı şiddetinden türetilemez ve doğrudan gözlenen kategorik
olay gerektirir. Aynı hareketin aynı fiziksel `error_unit` değeri tekilleştirilir.
Kısmi kayıtta yalnız `observed_scope_provisional_deduction_total` raporlanır;
tam `accuracy_score` üretilmez.

Güncel gerçek M01-M06 regresyonunda dokuz source-bound kararın dördü küçük hata
olarak uygulandı, biri kaynak aralığında, biri sınır-belirsiz kaldı, üçü
ölçülemedi. Gözlenen kapsam kesinti toplamı `0,4`; kategorik major olayı `0`;
`accuracy_score=null` oldu. Bu sayı
tam Poomsae puanı veya güncel WT tolerans doğrulaması değildir.

RulePack 1.1.0, süre ihlali ve yarışma alanı sınırını geçmeyi Accuracy
kesintilerinden ayırıp final skor metadata'sı olarak taşır. Her ikisi `-0,3`
olsa da sınır geçmenin tekrar frekansı WT metninde açık değildir; runtime olay
sözleşmesi netleşene kadar otomatik uygulanmaz.

İşler:

- küçük/büyük hata olayları,
- tekrar kesintisini engelleme,
- sıra/yön/duruş/teknik adayları,
- kural maddeli inceleme raporu,
- kullanıcı/teknik inceleme kabul-red kaydı.

Çıkış kapısı: Sentetik/kontrollü hata örneklerinde beklenen kural olayı
üretiliyor, doğru harekete bağlanıyor, aynı hata iki kez kesilmiyor ve düşük
kanıt otomatik kesintiye dönüşmüyor.

### Aşama 7 — Provisional Presentation teşhisi

İşler:

- hız-güç, ritim ve enerji özelliklerini ayrı üretmek,
- ölçülemeyen ifade/kihap/gaze alanlarını açık bırakmak,
- formülü ve ağırlıkları sürümlenmiş provisional 0–6 dönüşümü oluşturmak,
- eşik hassasiyetini ve belirsizliği raporlamak,
- gelecekteki hakem kalibrasyonunu ayrı ve opsiyonel profil olarak tasarlamak.

Çıkış kapısı: Birincil çıktı teşhis profilidir. 0–6 değer gösterilirse
`judge_calibrated=false`, `provisional_not_judge_validated` ve kullanılan
formül sürümü raporda görünür olmalı.

### Aşama 8 — ML destekli otomasyon

İşler:

- faz/sınır modeli,
- görsel el/ayak/bakış modelleri,
- confidence kalibrasyonu,
- baseline ve ablation karşılaştırması,
- güvenli fallback.

Çıkış kapısı: ML ilgili alt görevde basit baseline'ı geliştirmede kullanılmayan
ayrı testte geçmeli; kural motorunun açıklanabilirliğini bozmamalı.

### Aşama 9 — Yayın ve saha doğrulaması

İşler:

- farklı sporcu ve kamera düzeninde kör test,
- kullanıcı/teknik inceleme oturumu,
- hata/puan/uncertainty raporunun kullanılabilirliği,
- sürüm ve geri alma prosedürü,
- yalnız kanıtlanan kapsam için ürün iddiası.

Çıkış kapısı: `rule_scoring_ready` teknik ve kaynak kapılarıyla açılabilir.
Uzman/yetkili kurum olmadığı için `judge_calibrated_ready=false` ve
`official_scoring_ready=false` kalması bu aşamanın başarısızlığı değildir.

## 16. Riskler ve korumalar

| Risk | Koruma |
|---|---|
| PDF'nin yanlış yorumlanması | Madde/sayfa referansı, iki geçişli kaynak denetimi ve belirsizlik kapısı |
| Tek referans videosuna aşırı uyum | Çoklu referans ve vücut normalize aralıklar |
| Pose hatasının teknik hata sayılması | Observability, kamera kanıtı ve replay |
| Geçiş karesinin son pozisyon gibi puanlanması | Faz bazlı metrik |
| Aynı hatanın iki kez kesilmesi | Olay birleştirme ve benzersiz deduction kimliği |
| Depth'in ground-truth sanılması | İç/dış kanıtın ayrı raporlanması |
| Sunum puanının keyfî formülü | Formül provenance'ı, hassasiyet analizi ve `not_judge_validated` etiketi |
| ML'nin veri setini ezberlemesi | Sporcu/oturum bazlı split |
| Düşük güvenli tahminin kesin gösterilmesi | Belirsizlik kapısı ve `not_measurable` |
| Kural değişince eski sonucun bozulması | RulePack hash'i ve immutable scoring run |
| Üçüncü kameranın farklı sonuç semantiği yaratması | Kamera-bağımsız scoring giriş sözleşmesi |

## 17. İlk somut iş paketi

İlk uygulama sprintinin kapsamı yalnız şunlar olmalıdır:

1. [tamamlandı] Eksen hatasını ve sentetik metrik testlerini düzeltmek.
2. [tamamlandı] RulePack, PoomsaeSpec ve MovementTimeline YAML sözleşmelerini
   oluşturmak.
3. [tamamlandı] WT 2024-09-30 puan semantiğini kaynak referanslarıyla girmek.
4. [tamamlandı - taslak] Taegeuk 1 hareket tablosunu ücretsiz resmî Kukkiwon
   kaynaklarından çıkarmak.
5. [tamamlandı - kısa kayıt] Mevcut videodaki M01-M06'yı hareket aralığı ve
   faz anchor'larıyla etiketlemek; kayıtta olmayan M07-M18'i üretmemek.
6. [tamamlandı - puansız kanıt] Hareket zaman çizelgesi, iki etiketli inceleme
   videosu ve faz bazlı metrik/observability raporu üretmek.
7. [tamamlandı - inceleme yüzeyi] İki kamerayı senkron oynatan, hareket/faz
   seçimi, readiness engelleri, BODY-17 iptal uyarısı ve WholeBody adaylarını
   gösteren, kendi başına puan üretmeyen HTML raporu eklemek.
8. [tamamlandı - WholeBody teşhisi] M01-M06 için 133 noktalı gövde/ayak/yüz/el,
   trajectory, zamanlama ve fixation ölçülerini; grup/metric güven kapılarını
   ve gerçek kayıt koşusunu eklemek. Adayları ceza yapmamak, otomatik major ve
   bütün sayısal skorları kapalı tutmak.
9. [tamamlandı - source-bound observed scope] Tarihsel açık geometriyi
   hash/sayfa bağlı profile almak; `%95` belirsizlik kapısı, küçük-hata sınırı,
   kategorik major sözleşmesi, hata tekilleştirme ve kısmi-kayıt `score=null`
   korumasıyla gerçek M01-M06 regresyonunu üretmek.
10. [sıradaki] M01-M18'in tamamını içeren yeni kaydı aynı sözleşmeyle
   etiketlemek ve kısıtlı sıra hizalamasını sınamak.

Hazırlık kapısı kalıcı komutla denetlenir:

```powershell
.\.venv312\Scripts\python.exe scripts\assess_poomsae_scoring_readiness.py `
  --rule-pack config\scoring\rules\wt_recognized_2024-09-30.yaml `
  --poomsae-spec config\scoring\poomsae\taegeuk_1_jang_v0_draft.yaml `
  --timeline config\scoring\timelines\poomsae1_zed2i_rgbd_rerun_20260802_draft.yaml
```

Komutun `blocked` çıkması mevcut aşamada hata değildir: pose dosyasının doğru
koşuya ait olduğu ayrıca doğrulanırken eksik kaynak/etiketler puan üretmeden
listelenir.

Bu sprintte yapılmayacaklar:

- Etiketsiz videolardan nihai skor modeli eğitmek,
- Keyfî joint-angle eşikleri üretmek,
- Tarihsel `71,7178` değerini WT puanına çevirmek veya yeniden üretmek,
- Ölçülemeyen bakış/kihap/yumruk ayrıntısına kesinti vermek,
- “resmî” veya “hakemle eşdeğer” doğruluk iddiası.

İlk sprint tamamlandığında elimizde henüz nihai skor olmayabilir; fakat puanın
üzerine kurulacağı hareket, kural, ölçüm ve kanıt zinciri doğru olacaktır. Bu
zincir kurulmadan üretilecek daha erken bir sayı teknik olarak hızlı görünse
de güvenilir bir Poomsae puanı olmaz.

## 18. Kararı yeniden değerlendirme koşulları

Bu plan şu durumlarda yeniden incelenir:

- WT/Kukkiwon kural veya teknik tanımı değişirse,
- resmî kaynaklar hareket ontolojisi veya hata şiddetinde çelişirse,
- kural + hizalama baseline'ı ayrı test videolarında kabul kapısını geçemezse,
- yeterli etiketli veriyle end-to-end veya başka bir yöntem aynı kapsamda
  anlamlı biçimde daha doğru ve kalibre sonuç verirse,
- yeni sensör, örneğin sporcu IMU'su veya kuvvet platformu eklenirse,
- ürün hedefi antrenman desteğinden resmî müsabaka kullanımına değişirse.

Karar değişirse eski kural paketi, testler ve sonuçlar silinmez. Yeni mimari
aynı ayrı-test protokolünde önce/sonra ölçülür ve ayrı sürüm olarak yayımlanır.
