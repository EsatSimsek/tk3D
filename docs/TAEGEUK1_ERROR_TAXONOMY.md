# Taegeuk 1 Hata Taksonomisi ve Ölçülebilirlik Matrisi

Son araştırma ve doğrulama: **3 Ağustos 2026**
Son güncelleme: **18 Ağustos 2026** (15 Ağustos'ta eklenen eolgul-makki sayısal
kuralı ve otomatik duraklama tespiti belgelendi)

Bu belge, Taegeuk 1 için “hangi hataları aramalıyız?” sorusunu kaynak
otoritesi, puan semantiği ve sensörle ölçülebilirlik bakımından ayırır. Buradaki
bir kaydın bulunması otomatik kesinti yetkisi vermez. Aktif kesinti için ayrıca
RulePack, PoomsaeSpec, gözlenmiş kanıt ve kural doğrulama kaydı gerekir.

## 1. Kaynak sınıfları

| Sınıf | Kaynak | İzin verilen kullanım |
|---|---|---|
| `A1-current-official` | WT 30 Eylül 2024 Competition Rules, Articles 15-16 | Puan bütçesi, kesinti miktarı ve açık büyük hata örnekleri |
| `A2-current-official-technique` | Kukkiwon 2025 Taegeuk 1 eğitimi ve resmî tam sıra videosu | Hareket sırası, teknik/faz anlamı, yön ve kihap olayı |
| `B1-historical-official` | WTF 2014, 43 sayfalık International Referee Scoring Guidelines | Teknik geometri ve hata adayı tasarımı; güncel tolerans diye kullanılamaz |
| `B2-national-secondary` | British Taekwondo 2025 kuralları | WT metnini yorumlama ve araştırma çapraz kontrolü; genel WT RulePack'e doğrudan taşınamaz |
| `B3-training-secondary` | Swiss Taekwondo January 2025 judging manual | Hakem eğitim örnekleri ve araştırma ipucu; WT/Kukkiwon otoritesi değildir |
| `C-primary-research` | Hakemli poomsae/HPE çalışmaları | Hizalama, temsil ve anomali yöntemi; resmî kural veya tolerans kaynağı değildir |

Masaüstündeki `WTF---Poomsae-scoring-guidelines.pdf` 35 sayfadır ancak sayfa
altlıkları `35 / 43` ile biter. WT kaynaklı tam 43 sayfalık kopya
`output/pdf/scoring_sources/WT_Poomsae_Scoring_Guidelines_2014_43p.pdf`
konumuna kaydedildi. İki dosyanın ilk 35 sayfasındaki çıkarılabilir metin
eşleşir. Eksik sekiz sayfa terminoloji ve hakem puan formlarıdır; yeni teknik
tolerans eklemez.

## 2. Puanlama olayları

### 2.1 WT tarafından güncel olarak tanımlanan Accuracy olayları

| Olay | WT sonucu | Taegeuk 1 ilgisi | Otomatik gözlenebilirlik |
|---|---:|---|---|
| Bireysel harekette küçük hata | `-0,1` | Duruş/el tekniği ekli guideline dışına çıkarsa | Ancak aktif teknik kriter ve yeterli kanıt varsa |
| Yanlış veya guideline dışı hareket | `-0,3` | Yanlış blok, vuruş, duruş, taraf, sıra | Sıra/teknik sınıflandırması yüksek güvenliyse aday; otomatik major şimdilik kapalı |
| Kihap eksik veya yanlış anda | `-0,3` | M18 son yumrukta kihap | Ses kanalı yoksa `not_measurable`; ağız hareketi ses kanıtı değildir |
| Hareketler sırasında en az 3 saniye durma | `-0,3` | Tam performansta geçerli | Zaman çizelgesiyle ölçülebilir; bilinçli bitiş/fixation ile karıştırılmamalı |
| Bakışın hareket yönünü izlememesi | `-0,3` | Her yön değişiminde ilgili | Yüz keypointleri fiziksel kapıyı geçerse aday; baş yönü göz yönünün tam karşılığı değildir |
| Hakdari-seogi yükseltilmiş ayağın yere değmesi | `-0,3` | Taegeuk 1'de hakdari yok | Bu formda uygulanamaz |
| Jittzikgi'de güç/ses eksikliği | `-0,3` | Taegeuk 1'de jittzikgi yok | Bu formda uygulanamaz |
| Aşırı yüksek nefes sesi | `-0,3` | Teorik olarak tam performans | Ses yoksa `not_measurable`; sağlık/efor çıkarımı yapılmaz |
| Poomsae'yi yeniden başlatma | `-0,6` | Tam performans olayı | Tek video klibinde başlangıca dönüşü güvenle ayırt etmek gerekir |

WT 2024 metni “genel denge kaybı = her durumda -0,3” şeklinde açık bir
örnek vermez; özgül örnek hakdari ayağının yere değmesidir. British 2025 ve
Swiss 2025 belgeleri genel büyük denge kaybı/stumble örneğini ekler. Bu geniş
yorum, WT RulePack'e ayrı otorite doğrulaması olmadan aktarılmayacaktır.

**Otomatik türetilen tek kategorik gözlem: 3 saniye duraklama.** Yukarıdaki
kategorik olayların yalnız biri zaman çizelgesinden deterministik olarak
kanıtlanabilir. `derive_categorical_observations()`
(`src/poomsae_scoring/source_bound_accuracy.py`) etiketli segmentler arasındaki
boşlukları tarar ve eşiği (varsayılan `3,0` saniye, `>=` semantiği) karşılayan
her boşluğu, boşluktan **önce gelen** harekete bağlı bir
`pause_at_least_3_sec` gözlemine dönüştürür. Son segmentten sonraki, performans
bitişiyle sınırlandırılmamış kuyruk boşluğu taranmaz. `confirmation_method`
alanı `duration_measurement` olur. Manuel doğrulanmış iki komşu segmentte ve
yeterli gerçek güvende `evidence_status=observed`; otomatik timeline'da
`evidence_status=inferred` olur ve doğrulama olmadan kesinti uygulanmaz. Gözlem
güveni iki komşu segment güveninin küçüğüdür; taban değere yükseltilmez.

Yanlış hareket (`wrong_action`) ve yanlış duruş (`wrong_stance`) yalnız timeline
sırasından türetilmez; bu tautolojik olurdu. Bunun yerine
`build_categorical_diagnostics()` WholeBody-133 ölçümlerini beklenen ve
alternatif teknik/duruş mühendislik profilleriyle karşılaştırır. Beklenen aralık
en az bir gerekli ölçümde dışlanırken alternatif profil bütün gerekli
ölçümlerde uyuyorsa `kinematic_screening` adayı üretilir. Metrik belirsizliği
varsa dışlama bu aralığın tamamıyla yapılır; yoksa düşük güvenli nokta
karşılaştırması olarak işaretlenir. Adayın
`evidence_status=inferred` olması zorunludur; kaynak-bağlı Accuracy motoru
`not_directly_observed` nedeniyle kesinti uygulamaz. Doğrudan büyük kesinti için
hâlâ bağımsız video gözlemi veya ayrı veri üzerinde doğrulanmış bir
sınıflandırıcı gerekir.

### 2.2 Tam yarışma performansı cezaları

WT Article 16, Accuracy dışındaki nihai puan cezaları olarak belirlenen yarışma
süresinden erken/geç bitirmeyi ve alan sınırını geçmeyi ayrı ayrı `-0,3`
tanımlar. Mevcut kısa kayıt M01-M06 içeren kesilmiş bir teknik klip olduğundan
bu iki ceza uygulanamaz. Tam performans ve yarışma alanı kalibrasyonu olmadan
üretilmeleri engellenmelidir.

### 2.3 Presentation hataları tekil kesinti değildir

WT, `speed and power`, `rhythm & tempo` ve `expression of energy` başlıklarını
ikişer puanlık bütünsel değerlendirme olarak tanımlar. Her sallanma veya hız
sapmasına `-0,1` yazmak yanlış mimaridir. Sistem şu göstergeleri ayrı çıkarır:

- yumuşak başlangıçtan kritik noktada hız/güç zirvesine geçiş;
- gereksiz erken kasılma veya bir sonraki harekete aşırı karşı hareket;
- geçişler arası hız ve güç değişimi, tekdüze olmayan ritim;
- kontrollü fixation, canlılık, hareket büyüklüğü, bakış ve özgüven göstergeleri;
- kinematik hız/ivme vekilleri; bunlar gerçek kuvvet ölçümü değildir.

Hakem kalibrasyonu olmadığı için bu özelliklerin doğrudan resmî `0-6` puana
dönüşümü etkin değildir. Bu göstergeleri toplayan katman
`src/poomsae_scoring/presentation.py` içindedir ve sözleşme gereği
`total_score=None`, `score_claim_allowed=false` üretir; ayrıntısı
[`PRESENTATION_DIAGNOSTICS.md`](PRESENTATION_DIAGNOSTICS.md) belgesindedir.

## 3. Taegeuk 1 teknik hata aileleri

Aşağıdaki geometri, 2014 tarihsel resmî scoring guideline ile 2025 Kukkiwon
teknik eğitiminin aday üretmede kullanılabilecek kesişimidir. Sayısal sınırlar
güncel WT toleransı değildir.

Source-bound v1 yalnız kılavuzda açık sayı bulunan geometrileri tarihsel
provisional küçük-hata kararına açar: ap-seogi/apkubi arka ayak `30°`,
arae-makki yumruk-uyluk `1–2 yumruk`, momtong-an-makki dirsek `90–120°` ve
eolgul-makki yumruk-alın `0,5–1,5 yumruk`. Ölçümün `%95` aralığı sınırla
çakışırsa karar `boundary_uncertain` olur; kesinti yoktur. Hiçbir sayısal
geometri `-0,3` üretemez.

### 3.1 Ap-seogi

Tarihsel kılavuzdaki doğru teknik tanımı:

- başlangıçtan yaklaşık üç ayak uzunluğu;
- ön ayak ileri, arka ayak yaklaşık 30 derece;
- iki bacak düz;
- ayakların iç kenarları aynı çizgide;
- gövde doğal yaklaşık 45 derece;
- ağırlık iki bacağa eşit dağılmış.

Aranacak hata aileleri: adım boyu, ayak yaw, çizgi sapması, dizin gereksiz
bükülmesi, gövde yönü, ağırlık aktarımı ve bitişte kararsızlık. Ayak uzunluğu
WholeBody keypointlerinden güvenilir değilse span eşiği vücut oranıyla
provisional kalır.

### 3.2 Ap-gubi

Tarihsel tanım:

- yaklaşık `4-4,5` ayak uzunluğu;
- ön ayak ileri, arka ayak yaklaşık 30 derece;
- yaklaşık `%70 / %30` ön-arka ağırlık dağılımı;
- ayak iç kenarları arasında bir-iki yumruk;
- gövde dik ve doğal yaklaşık 30 derece.

RGBD iskelet ağırlık dağılımını doğrudan ölçmez. Pelvis/COM ve ayak desteğinden
üretilen değer yalnız `kinematic_weight_transfer_proxy` olabilir.

### 3.3 Arae-makki

Tarihsel tanım ve görsel hata örnekleri:

- blok eli karşı omuzdan hazırlanır; hammer-fist omuza temas eder;
- çeken kol karın önünde kemerin üzerinden geçer;
- bitiş yumruğu uyluktan yaklaşık iki yumruk uzakta ve bacağın yanında;
- blok kolu düz, dirsek gereksiz bükülü değildir;
- hikite bel yanındadır.

Ölçülebilir: taraf, dirsek açısı, bilek yüksekliği, uyluğa normalize mesafe,
hikite konumu, yol verimi ve fixation. Yumruğun gerçekten kapalı olması için
fiziksel olarak geçerli el keypointleri veya RGB crop gerekir.

### 3.4 Momtong jireugi

Tarihsel tanım:

- vuran el belden avuç yukarı başlar, karşı el solar-plexus yönünden çekilir;
- kol uzar, yumruk dönüşü hareketin son anında tamamlanır;
- hedef solar-plexus, temas yüzeyi iki büyük boğum;
- bilek düz, avuç aşağı; hikite bel yanındadır.

Ölçülebilir: doğru taraf, son el yüksekliği, dirsek uzaması, bilek-önkol
hizası, hikite, omuz-kalça eşgüdümü, el-ayak eşzamanı, yol ve fixation.
Boğum temas yüzeyi mevcut çözünürlükte çoğunlukla `not_measurable` kalır.

### 3.5 Momtong makki / an-makki

2014 kılavuzundaki genel momtong makki için hazırlık yumruğu omuzun biraz
üstünde, dirsek aşağı; bitiş bileği gövde merkezinde/solar-plexus düzeyinde,
kol yaklaşık `90-120` derece ve hikite bel yanındadır. Taegeuk 1'deki
`momtong-an-makki` için bu genel tanım tek başına bütün yön/yol ayrıntılarını
kanıtlamaz. Yalnız bitiş dirsek açısı, tarihsel kaynak statüsü ve `%95`
belirsizlik kapısıyla provisional küçük-hata kontrolüne açıktır; diğer yön/yol
ayrıntıları eşiksiz kalır.

### 3.6 Eolgul-makki

Tarihsel bitiş tanımı:

- blok bileği alın merkezinden yaklaşık bir yumruk uzakta;
- dirsek yukarı;
- hikite bel yanında.

Alın mesafesi yüz ölçeği ve bilek güveni geçerse ölçülebilir. Kameraya göre
2B piksel mesafesi tek başına kullanılmaz.

**Aktif sayısal kural (15 Ağustos 2026).** Bu geometri, profildeki beşinci
sayısal kural — dördüncü ayrı `metric_id` — olarak source-bound profile
alınmıştır:

| Alan | Değer |
|---|---|
| `rule_id` | `HIST-2014-EOLGUL-FIST-FOREHEAD-ONE` |
| Metrik | `eolgul_fist_to_forehead_fist_ratio` |
| Aralık | `0,5–1,5` yumruk genişliği (`operator: range`) |
| Belirsizlik tabanı | `0,20` yumruk |
| Sınır koruması | `0,15` yumruk |
| Kaynak | 2014 WTF guideline, sayfa `17`, `historical_official_not_current_attachment` |
| Kesinti türü | yalnız `minor` (`-0,1`) |
| İlgili hareketler | M13, M15 |

Kural, PoomsaeSpec kriter adına göre değil profildeki
`technique_ids: [eolgul_makki]` alanı üzerinden bağlanır; bu teknik yalnız M13
ve M15'te bulunur ve iki hareket de
`technique.eolgul_makki.forehead_distance` kriterini taşır.

Ölçüm `src/poomsae_scoring/wholebody_diagnostics.py` içindeki
`_eolgul_fist_to_forehead_ratio()` fonksiyonuyla yapılır. Ölçünün paydası
yumruk genişliğidir; alın merkezi **doğrudan ölçülen kaş çizgisidir** (iBUG
`17-26`), yukarı doğru ekstrapolasyon yapılmaz. Fonksiyon şu durumların
herhangi birinde **o kare için** `None` döner:

- vücut ölçeği yok veya sıfır/negatif;
- dört yumruk boğumundan (`index/middle/ring/pinky MCP`) biri bile eksik;
- yumruk genişliği sıfıra yakın (boğumlar çakışmış);
- yumruk genişliği/vücut ölçeği oranı el kalite kapısının
  (`hand_palm_length_body_scale`) dışında;
- on iki göz noktasından biri bile eksik;
- göz ayrımı/vücut ölçeği oranı yüz kalite kapısının
  (`face_eye_separation_body_scale`) dışında;
- geçerli kaş noktası sayısı dörtten az.

Fonksiyon tek kare üzerinde çalışır; metrik değeri fixation anchor'ı etrafındaki
pencereden (`anchor_window_radius_frames = 5`, yani en fazla 11 kare) sağlam
biçimde toplanır. Metriğin `None` olması ve kararın `not_measurable` üretilmesi
için pencerede geçerli örnek sayısının `min_anchor_window_valid_samples = 3`
altına düşmesi gerekir. Yani 11 karenin 8'i yukarıdaki kapılara takılsa bile
kural normal şekilde karar verebilir.

Bu kapılar arae-makki ölçümüne benzer fail-closed mantığındadır — eolgul
ölçümü ek olarak yüz/göz ayrımı kapısı ve sıfıra yakın yumruk genişliği
koruması taşır. Ortak ilke aynıdır: ölçüm yapılamıyorsa değer uydurulmaz.

### 3.7 Ap-chagi ve ardından yumruk

Tarihsel ön tekme tanımı:

- diz göğse doğru toplanır;
- ayak düz bir hatta hızlı uzatılır, hedef ayak tarak yastığıdır;
- ayak bileği/instep düz, parmaklar geri;
- destek ayağı doğal pivot yapar;
- diz yeniden göğse toplanır ve dengeyle indirilir.

M14/M16 bileşik harekettir. `kick_apex`, `rechamber`, `landing`,
`punch_execution` ve `fixation` ayrı anchor olmalıdır. Tekmeyi tek final poza
indirgemek; rechamber, denge ve yumruk zamanlama hatalarını kaçırır.

### 3.8 Sıra, yön ve bitiş

Kaynak transkripsiyonu 18 hareket birimidir. Sistem en az şunları aramalıdır:

- eksik, fazla, tekrar veya sıra dışı hareket;
- yanlış aktif taraf veya yanlış dönüş yönü;
- `ap-seogi` yerine `ap-gubi` gibi yanlış duruş;
- blok yerine yumruk veya yanlış blok yüksekliği;
- M14/M16 tekme-yumruk bileşiğinde eksik faz;
- M18 kihapın eksik/erken/geç olması;
- başlangıç/bitiş tutumunun bozulması ve yeniden başlama.

Mevcut kısa kayıt M01-M06'da bittiği için M07-M18 “eksik yapılmış hareket”
değil, kayıt kapsamı dışında kabul edilir.

## 4. Sensör ve kanıt matrisi

| Hata ailesi | Ana kanıt | Destek | Fail-closed nedeni |
|---|---|---|---|
| Sıra/yanlış hareket | Tam MovementTimeline + çoklu RGB | 3B poz | Kısa/trim kayıt yanlış eksik sayılmamalı |
| Duruş uzunluğu/çizgisi | Dünya 3B ayak/kalça | Depth, iki kamera | Ayak keypointi veya kalibrasyon zayıfsa karar yok |
| Ayak yönü | WholeBody ayak noktaları | RGB crop | Fiziksel ayak oranı kapısı geçmezse karar yok |
| Diz/dirsek/gövde açıları | 3B eklem | Çoklu 2B reprojection | Geçiş karesi final poz sayılmaz |
| Yumruk/elin kapanması | WholeBody el + RGB crop | 3B bilek | Parmak geometrisi fiziksel kapıyı geçmezse karar yok |
| Bakış | Yüz/göz + iki RGB görünümü | Baş-gövde yaw | Baş yönü göz yönü yerine otomatik kesinti olamaz |
| Kihap/nefes | Senkron ses | Ağız hareketi yalnız aday | Mevcut ZED video sesi doğrulanmadan ölçülemez |
| Ağırlık/denge | Kinematik COM ve ayak desteği | Depth | Kuvvet platformu yok; yalnız proxy |
| Güç | Hız/ivme ve kontrollü bitiş | Depth/IMU kamera hareketi | Sporcu kuvveti veya sporcu IMU'su değildir |
| Sınır ihlali | Kalibre yarışma alanı | Dünya 3B ayak | Mat sınırı tanımlı değilse uygulanamaz |

Depth; yüzey derinliğini doğrular, eklem merkezinin ground-truth'u değildir.
ZED IMU kameranın hareketi/yerçekimi için yararlıdır; sporcunun gövdesine takılı
IMU gibi teknik güç veya yön kanıtı sayılamaz.

## 5. Akademik yöntemlerden alınan mimari sonuç

Birincil çalışmalar şu yaklaşımı destekler:

- Kişiler arası uzuv boyu farkı nedeniyle ham 3B eklem mesafesi yerine
  vücut-şekli ayrıştırılmış eklem rotasyonları/açıları kullanılmalıdır.
- Hareket süreleri farklı olduğundan test ve referans sıra monoton DTW veya
  benzeri kısıtlı hizalamayla eşlenmelidir.
- Tek kamera 3B HPE hatası sonuçları belirgin etkiler; mevcut çoklu kamera ve
  depth kanıt kapısı korunmalıdır.
- Anomali modeli “referanstan farklı” olanı bulabilir, fakat farkın WT küçük
  hata mı, kabul edilebilir beden varyasyonu mu olduğunu tek başına bilemez.
- Hareket sınıflandırma doğruluğu, puan doğruluğu değildir. 2024 LSTM
  çalışmasında uzman karşılaştırmalı sonuç `%61` olduğundan küçük veriyle
  uçtan uca score modeli güvenli bir başlangıç değildir.

Bu nedenle önerilen sıra:

```text
çoklu RGBD kanıtı
  -> kaynak bağlı hareket/faz hizalama
  -> beden-normalize ölçümler
  -> observability ve fiziksel geçerlilik
  -> teknik sapma adayları
  -> yalnız aktif WT kriterinde kesinti
  -> bütünsel presentation teşhisi
```

## 6. Kalan gerçek bilgi açıkları

Araştırmayla kapatılamayan noktalar aşağıdadır; bunlar gizlenmeyecek ve skor
kapısını fail-closed tutacaktır:

1. WT 2024 Article 16'nın atıf yaptığı scoring guideline'ın ayrı, güncel ve
   kamuya açık sürümü bulunamadı. 2014 belgesi resmî tarihsel kaynaktır.
2. Kukkiwon Textbook Volume 3'ün güncel tam metni ücretsiz değildir; özellikle
   bileşik hareket tablo ayrıntıları ikinci resmî basılı kaynakla henüz
   doğrulanmadı.
3. WT küçük/büyük hata sınırlarının her teknik için güncel sayısal toleransı
   yayımlanmış kaynaklarda yoktur.
4. Mevcut kısa kayıtta M07-M18, tam performans süresi, alan sınırı ve ses yoktur.
5. Hakem/uzman kalibrasyonu yoktur; bu nedenle resmî hakem eşdeğerliği ve
   presentation puan doğruluğu iddia edilemez.

Bu açıkların hiçbiri M01-M06 için ölçüm ve hata adayı üretmeyi engellemez;
yalnız doğrulanmamış adayın otomatik puan kesmesine engel olur.

## 7. Kaynaklar

- World Taekwondo, *Poomsae Competition Rules and Interpretation*, yürürlük
  30 Eylül 2024, Articles 15-16.
- World Taekwondo Federation, *Poomsae Scoring Guidelines for International
  Referees*, 2014, 43 sayfa.
- Kukkiwon, *Taegeuk 1 Jang* resmî sayfası, kısa gösterim ve 16 Ocak 2025
  eğitim videosu.
- British Taekwondo, *Poomsae Competition Rules*, June 2025.
- Swiss Taekwondo, *Manual for Poomsae Scoring*, January 2025. İkincil eğitim
  kaynağıdır.
- Hoang & Ahn (2023), *An Evaluation Method of Taekwondo Poomsae
  Performance*, DOI `10.56977/jicce.2023.21.4.337`.
- Fernando, Sandaruwan & Athapaththu (2024), *Evaluation of Taekwondo
  Poomsae movements using skeleton points*, DOI
  `10.4038/jnsfsr.v52i1.11986`.
- Chaâbane, Ben Said & Chaari (2025), *Deep Learning-Based Autoencoder for
  Objective Assessment of Taekwondo Poomsae Movements*, icSPORTS 2025.
