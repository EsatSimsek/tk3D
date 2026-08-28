# Taegeuk 1: kurallar, hareket bitişi ve M01-M18 ayrımı

Bu belge TK3D repository'sindeki **mevcut çalışan sistemi** anlatır. Özellikle
şu üç soruya cevap verir:

1. Taegeuk 1 kuralları toplu olarak nerede tutuluyor?
2. Sistem bir hareketin veya yumruğun bittiğini nasıl anlıyor?
3. Taegeuk 1 nasıl `M01`, `M02`, `M03` ... hareketlerine ayrıldı?

En önemli kısa cevap şudur:

> Güncel gerçek Poomsae 1 kaydında sistem, yumruğun bittiği kareyi tamamen
> otomatik olarak keşfetmiyor. Hareket sınırları ve önemli faz kareleri iki
> kamera görüntüsünden **elle belirlenip doğrulanmış** bir zaman çizelgesinden
> okunuyor. 3B sistem daha sonra bu karelerin çevresindeki pozu ve hareketi
> ölçüyor. Otomatik hareket adayları ve sıra hizalama kodu vardır, fakat güncel
> tek-komutlu puanlama akışının doğrulanmış zaman çizelgesinin yerine geçmiş
> değildir.

## 1. Taegeuk 1 kuralları toplu olarak nerede?

Tek bir dosyada hem resmî müsabaka puanları, hem 18 hareketin sırası, hem teknik
geometri, hem de belirli videonun kare numaraları tutulmuyor. Bunlar farklı
anlamlara sahip olduğu için bilinçli olarak ayrı dosyalara bölünmüştür.

### 1.1. Resmî WT puanlama kuralları

Dosya:
[`config/scoring/rules/wt_recognized_2024-09-30.yaml`](../config/scoring/rules/wt_recognized_2024-09-30.yaml)

Bu dosyada World Taekwondo'nun 30 Eylül 2024 tarihli Recognized Poomsae
kurallarından alınan üst düzey puanlama sözleşmesi bulunur:

- toplam puan: `10,0`,
- Accuracy başlangıç puanı: `4,0`,
- küçük hata kesintisi: `0,1`,
- büyük hata kesintisi: `0,3`,
- poomsae'ye yeniden başlama kesintisi: `0,6`,
- Presentation toplamı: `6,0`,
- Presentation alt başlıkları: hız ve güç, ritim ve tempo, enerjinin ifadesi,
- süre ihlali ve sınır çizgisi gibi performans düzeyi olayları,
- yanlış hareket, yanlış duruş, yanlış/eksik kihap ve üç saniyelik duraklama
  gibi büyük hata örnekleri.

Kaynağın URL'si, erişim tarihi, ilgili madde/sayfalar ve PDF'nin SHA-256 özeti
de aynı dosyada saklanır. Böylece bir kural kararının hangi belgeye dayandığı
izlenebilir.

Bu dosya **Taegeuk 1'in hareket sırasını tarif etmez**. WT yarışma puanlama
mantığını tarif eder.

### 1.2. Taegeuk 1'in 18 hareketlik teknik tarifi

Dosya:
[`config/scoring/poomsae/taegeuk_1_jang_v0_draft.yaml`](../config/scoring/poomsae/taegeuk_1_jang_v0_draft.yaml)

Bu dosya projenin `PoomsaeSpec` dosyasıdır. Taegeuk 1'e özel olarak şunları
toplu biçimde taşır:

- `M01`-`M18` hareket sırası,
- her hareketin Türkçe görünen adı,
- yönü,
- beklenen duruşu,
- yapılan tekniği ve uygulayan tarafı,
- hareketin fazları,
- ölçülebilecek teknik ölçütlerin kimlikleri,
- her hareketin dayandığı Kukkiwon ve tarihsel WT kaynakları.

Bu nedenle “Poomsae 1'in bütün hareketleri toplu olarak nerede?” sorusunun en
doğrudan cevabı bu dosyadır.

Dosyanın durumu hâlâ `draft`, sürümü `0.6.0-draft`tır. Bunun sebebi 18 hareketin
sırasının bilinmemesi değildir. Güncel WT kurallarının atıf yaptığı ayrıntılı
teknik puanlama ekinin kamuya açık güncel kopyası bulunamadığı için bazı
sayısal teknik toleranslar resmî güncel tolerans olarak etkinleştirilememiştir.

### 1.3. Kaynak bağlı teknik geometri ve kesinti kararları

Dosya:
[`config/scoring/accuracy/taegeuk_1_source_bound_v1.yaml`](../config/scoring/accuracy/taegeuk_1_source_bound_v1.yaml)

Burada ölçülebilir bazı teknik geometriler ile bunların karar politikası
bulunur. Örneğin:

- ap-seogi ve ap-gubi arka ayak yönü,
- arae-makki yumruğunun uyluğa uzaklığı,
- momtong-an-makki dirsek açısı,
- yanlış hareket, yanlış duruş, kihap ve yeniden başlama gibi kategorik
  olaylar.

Önemli sınırlama: dosyadaki sayısal geometrilerin bir bölümü 2014 tarihli
resmî fakat **güncel olmayan** kılavuzdan gelir. Dosya bu durumu
`provisional_historical_geometry` olarak açıkça işaretler. Bunlar güncel WT'nin
bütün teknik toleranslarıymış gibi sunulamaz.

### 1.4. Belirli videonun hareket zamanları

Dosya:
[`config/scoring/timelines/poomsae1_zed2i_rgbd_rerun_20260802_draft.yaml`](../config/scoring/timelines/poomsae1_zed2i_rgbd_rerun_20260802_draft.yaml)

Bu dosya kural kitabı değil, mevcut `741` karelik videonun
`MovementTimeline` kaydıdır. Hangi hareketin hangi kareler arasında olduğu ve
önemli fazların temsil kareleri burada yazılıdır.

Kayıt Taegeuk 1'in tamamını değil yalnız `M01-M06` bölümünü içerir. Dosyada bu
durum açık biçimde:

- `recording_scope: partial_sequence`,
- gözlenen hareketler: `M01-M06`,
- kaynakta bulunmayan hareketler: `M07-M18`

olarak tutulur. Böylece sistem videoda hiç bulunmayan bir hareketi “model
algılayamadı” diye yorumlamaz.

### 1.5. Hepsini birbirine bağlayan profil

Dosya:
[`config/scoring/profiles/poomsae1_trimmed.yaml`](../config/scoring/profiles/poomsae1_trimmed.yaml)

Tek komutlu çalışma, yukarıdaki kural paketi, PoomsaeSpec, MovementTimeline,
teşhis profili, Accuracy profili, 3B pose dosyası ve kamera videolarını bu
profil üzerinden bir araya getirir.

Özetle dosyaların görevleri şöyledir:

| Soru | Cevabı veren dosya |
|---|---|
| Kaç puan kesilir? | WT `RulePack` |
| Taegeuk 1'de sırayla hangi hareketler vardır? | `PoomsaeSpec` |
| Teknik geometri için hangi kaynaklı sınırlar kullanılabilir? | Accuracy profili |
| Bu özel videoda M01 hangi karede başladı/bitti? | `MovementTimeline` |
| Hangi pose, video ve ayarlar birlikte çalıştırılır? | Çalıştırma profili |

## 2. Sistem hareketin veya yumruğun bittiğini nasıl anlıyor?

Burada “bitiş” kelimesinin üç ayrı anlamı vardır:

1. **Execution:** Elin veya ayağın tekniği icra ettiği an.
2. **Fixation:** Tekniğin hedef biçimine ulaşıp kısa süre sabitlendiği an.
3. **Segment sonu:** Bir sonraki hareket başlamadan önce mevcut harekete ayrılan
   son kare.

Bunlar aynı kare olmak zorunda değildir. Örneğin M02 için:

- hareket başlangıcı: kare `230`,
- adım fazı: kare `260`,
- yumruk icrası (`execution`): kare `289`,
- yumruğun/duruşun sabitlendiği temsil karesi (`fixation`): kare `301`,
- M02 segmentinin sonu: kare `353`,
- M03 başlangıcı: kare `354`.

Dolayısıyla sistem “yumruk bitti, hemen M02 bitti” demez. Yumruğun hedef
biçimini `fixation` çevresinde ölçebilir; M02'nin bütün zaman aralığı ise bir
sonraki hareketin başlangıcına kadar devam eder.

### 2.1. Güncel gerçek kayıtta kullanılan yöntem: doğrulanmış kare etiketleri

Mevcut kısa kaydın `MovementTimeline` dosyasında `label_source: manual` yazar.
İş akışı şöyledir:

1. İki senkron ZED kamera görüntüsü birlikte incelendi.
2. Kukkiwon'daki bilinen Taegeuk 1 sırasına bakıldı.
3. Her hareketin başladığı ve bir sonraki harekete geçtiği kareler seçildi.
4. Hareket içindeki `preparation`, `turn`/`step`, `execution` ve `fixation`
   temsil kareleri işaretlendi.
5. Başlangıç, geçiş ve fixation kareleri iki kamera temas sayfalarında tekrar
   kontrol edildi.
6. Bu nedenle etiketlere `label_status: confirmed` yazıldı.

Buradaki `confirmed`, yalnızca **zaman etiketinin iki kameradan kontrol
edildiği** anlamına gelir. Tekniğin doğru yapıldığı, resmî hakemin onayladığı
veya puan kesilmemesi gerektiği anlamına gelmez.

Program çalışırken bu kareleri yeniden tahmin etmek yerine zaman çizelgesini
okur. `contracts.py` şu güvenlik kontrollerini uygular:

- segmentler sıra ile `1`den başlamalıdır,
- hareket sırası PoomsaeSpec ile aynı olmalıdır,
- `start_frame <= end_frame` olmalıdır,
- segmentler çakışamaz ve geriye gidemez,
- faz anchor'ları segmentin içinde kalmalıdır,
- anchor'lar PoomsaeSpec'teki faz sırasını bozamaz,
- gözlenen ve eksik hareket listeleri birlikte tam `M01-M18` sırasını
  bölmelidir.

Bu doğrulamadan sonra ölçüm motoru örneğin M02 `fixation=301` bilgisini alır ve
tek kareye körü körüne bağlı kalmamak için çoğu fixation metriğini bu karenin
çevresindeki sağlam bir pencerede hesaplar. Güncel WholeBody teşhisinde bu
pencere varsayılan olarak yaklaşık `±5` karedir. Böylece tek bir karedeki küçük
pose gürültüsü doğrudan karar haline gelmez.

### 2.2. Yumruk gerçekten “oturdu mu” diye hangi işaretlere bakılıyor?

Zaman etiketi bize **nereye bakacağımızı** söyler. 3B/WholeBody ölçümleri ise o
bölgede **ne olduğunu** inceler. Yumruk veya blok için kullanılan ölçümler
harekete göre değişmekle birlikte şunları içerebilir:

- uygulayan bileğin hareketi ve tepe hızı,
- dirsek açısı,
- bilek-önkol hizası,
- yumruğun kapanması,
- yumruğun hedef yüksekliği veya hedef bölgeye uzaklığı,
- diğer elin hikite konumu,
- el ve ayağın yerleşme zamanları arasındaki fark,
- omuz-kalça dönüşü,
- duruş uzunluğu ve ön diz açısı,
- fixation sonrasındaki bilek titreşimi/jitter.

Örneğin “fixation kararlı mı?” ölçümü, seçilmiş fixation karesinden segment
sonuna kadar uygulayan bileğin ne kadar oynadığını vücut ölçeğine göre
normalize eder. “El ve ayak aynı anda oturdu mu?” ölçümü ise hazırlık ile
fixation arasındaki el ve ayak hareketlerinin yerleşme zamanlarını karşılaştırır.

Bu ölçümler hareketin semantik adını tek başına keşfetmez. M02'nin sağ
`momtong-jireugi` olduğu bilgisi PoomsaeSpec'ten gelir; timeline hangi karelere
bakılacağını söyler; pose ölçümleri de o karelerde sağ kol, el, gövde ve
bacakların geometrisini hesaplar.

### 2.3. Otomatik hareket adayı bulma kodu ne yapıyor?

[`src/scoring_readiness.py`](../src/scoring_readiness.py) içinde genel amaçlı
`movement_segments(...)` fonksiyonu vardır. Bu fonksiyon:

1. Her eklemin ardışık 3B kareler arasındaki hızını hesaplar.
2. BODY-17 eklemlerinin hızlarından kare başına ortalama “hareket enerjisi”
   çıkarır.
3. Medyan, MAD ve yüzde 25/yüzde 75 sınırlarını kullanan adaptif bir hareket
   eşiği hesaplar.
4. Enerjinin eşiğin üstünde kaldığı ardışık kareleri `motion_candidate` olarak
   gruplar.
5. Çok kısa adayları eler.

Bu, “sporcu burada hareket ediyor” diyebilir; fakat tek başına “bu sağ
momtong-jireugi'dir ve tam şu karede doğru biçimde bitti” diyemez. Ürettiği
adayların durumu bu yüzden `needs_poomsae_label`dır.

Sürekli poomsae hareketinde iki teknik arasında belirgin hareketsiz boşluk
olmayabilir. Bu nedenle salt hız eşiği birden fazla tekniği tek segmente
birleştirebilir veya küçük bir yavaşlamayı yanlış sınır sanabilir. Güncel
kaynak-bağlı puanlama akışının manuel timeline kullanmasının temel nedeni
budur.

### 2.4. Sıra hizalama kodu ne durumda?

[`src/poomsae_scoring/sequence_alignment.py`](../src/poomsae_scoring/sequence_alignment.py)
şu yardımcı parçaları içerir:

- bulunan segmentleri bilinen `M01-M18` sırasına doğrudan eşleme,
- segment sayısı azsa `missing_segments`, fazlaysa `extra_segments` raporlama,
- segmentin videodaki göreli konumuna dayalı DTW maliyeti,
- aday segment pozu ile beklenen referans poz arasındaki 3B mesafe maliyeti,
- bir hareketi zorla eşlemek yerine atlayabilen ceza mekanizması,
- atlama sınırına yakın hareketleri sporcu lehine tutup insan incelemesine
  gönderen belirsizlik bandı.

Ancak bu modül şu anda ana `run_poomsae_scoring.py` akışına bağlanmış tam bir
otomatik hareket/faz dedektörü değildir. Beklenen referans pozların üretim
kaynağı, faz sınırları ve tam performans doğrulaması tamamlanmadan manuel
timeline'ın yerini alamaz. Yani repository'de otomatik hizalama altyapısı
vardır; güncel gerçek koşunun M01-M06 sınırları onun çıktısı değildir.

## 3. Taegeuk 1'i M01, M02, M03 diye nasıl böldük?

### 3.1. M numarası ne demek?

`M`, “movement/hareket” kimliğidir. `M01` birinci, `M02` ikinci, `M18`
on sekizinci sıradaki hareketi ifade eder. Baştaki sıfır alfabetik ve sayısal
sıralamanın dosyalarda düzgün görünmesini sağlar.

M numaraları modelin videoda kendiliğinden uydurduğu sınıflar değildir.
Kukkiwon'un tam Taegeuk 1 gösterimi ve ayrıntılı eğitimindeki **bilinen resmî
sıra**, proje içinde kararlı kimliklere dönüştürülmüştür.

### 3.2. “Bir hareket” için kullandığımız pratik birim

Hareketleri yalnız “kol bir kere hızlandı” diye bölmedik. Bir hareket girdisi
genellikle şu paketi temsil eder:

- yeni yön veya dönüş,
- ayak adımı ve ortaya çıkan duruş,
- o adımla birlikte veya duruş içinde yapılan ana teknik,
- tekniğin sabitlendiği son biçim.

Bazen tek giriş bileşik olabilir. Örneğin M14 ve M16; tekme, ayağın geri
toplanması, yere iniş ve yumruğu bir hareket kimliği altında fakat birçok fazla
taşır. Bu, fazların neden ayrıca gerekli olduğunu gösterir.

### 3.3. Projedeki 18 hareketlik sıra

| ID | Hareket | Fazlar |
|---|---|---|
| M01 | Sol ap-seogi ve sol arae-makki | hazırlık, dönüş, icra, sabitleme |
| M02 | Sağ ap-seogi ve sağ momtong-jireugi | hazırlık, adım, icra, sabitleme |
| M03 | Sağ ap-seogi ve sağ arae-makki | hazırlık, dönüş, icra, sabitleme |
| M04 | Sol ap-seogi ve sol momtong-jireugi | hazırlık, adım, icra, sabitleme |
| M05 | Sol ap-gubi ve sol arae-makki | hazırlık, dönüş, ağırlık aktarımı, icra, sabitleme |
| M06 | Sol ap-gubi içinde sağ momtong-jireugi | hazırlık, icra, sabitleme |
| M07 | Sağ ap-seogi ve sol momtong-an-makki | hazırlık, dönüş, icra, sabitleme |
| M08 | Sol ap-seogi ve sağ momtong-jireugi | hazırlık, adım, icra, sabitleme |
| M09 | Sol ap-seogi ve sağ momtong-an-makki | hazırlık, dönüş, icra, sabitleme |
| M10 | Sağ ap-seogi ve sol momtong-jireugi | hazırlık, adım, icra, sabitleme |
| M11 | Sağ ap-gubi ve sağ arae-makki | hazırlık, dönüş, ağırlık aktarımı, icra, sabitleme |
| M12 | Sağ ap-gubi içinde sol momtong-jireugi | hazırlık, icra, sabitleme |
| M13 | Sol ap-seogi ve sol eolgul-makki | hazırlık, dönüş, icra, sabitleme |
| M14 | Sağ ap-chagi, sağ ap-seogi ve sağ momtong-jireugi | hazırlık, tekme, tepe, geri toplama, iniş, yumruk, sabitleme |
| M15 | Sağ ap-seogi ve sağ eolgul-makki | hazırlık, dönüş, icra, sabitleme |
| M16 | Sol ap-chagi, sol ap-seogi ve sol momtong-jireugi | hazırlık, tekme, tepe, geri toplama, iniş, yumruk, sabitleme |
| M17 | Sol ap-gubi ve sol arae-makki | hazırlık, dönüş, ağırlık aktarımı, icra, sabitleme |
| M18 | Sağ ap-gubi, sağ momtong-jireugi ve kihap | hazırlık, adım, icra, kihap, sabitleme |

### 3.4. Mevcut videoya bu sıra nasıl uygulandı?

İki kameradaki görüntü, yukarıdaki sıra ile yan yana incelendi. İlk `140` kare
hazırlık/bekleme olduğu için M01'e dahil edilmedi. Sonra geçişler şu şekilde
etiketlendi:

| Hareket | Kare aralığı | Temsil faz kareleri |
|---|---:|---|
| M01 | 140-229 | preparation 145, turn 178, execution 205, fixation 218 |
| M02 | 230-353 | preparation 230, step 260, execution 289, fixation 301 |
| M03 | 354-473 | preparation 354, turn 388, execution 432, fixation 448 |
| M04 | 474-569 | preparation 474, step 490, execution 517, fixation 529 |
| M05 | 570-685 | preparation 570, turn 588, weight_transfer 616, execution 628, fixation 637 |
| M06 | 686-740 | preparation 686, execution 722, fixation 728 |

Video M06 fixation sonrasında bittiği için M07-M18 için hayalî segment
üretilmedi. Bu ayrıntılı incelemenin önceki kaydı
[`docs/POOMSAE1_KISA_KAYIT_ETIKETLERI.md`](POOMSAE1_KISA_KAYIT_ETIKETLERI.md)
içindedir.

### 3.5. M01-M18 ile fazlar arasındaki fark

Bu ayrım çok önemlidir:

- `M02`, bütün “sağ ap-seogi + sağ momtong-jireugi” hareketidir.
- `execution`, M02 içindeki yumruğun icra anını temsil eder.
- `fixation`, yumruk ve duruşun hedef son biçimini ölçtüğümüz bölgedir.
- `start_frame/end_frame`, M02'ye ait bütün inceleme aralığını sınırlar.

Yani `M01/M02/M03` **hareket sırası**, `preparation/execution/fixation` ise
hareket içindeki **faz sırası**dır.

## 4. Çalışma sırasında veri akışı

Mevcut kaynak-bağlı akış sadeleştirilmiş biçimde şöyledir:

```text
iki senkron kamera videosu
    -> 2B WholeBody-133 eklem tahmini
    -> çok kameralı 3B pose [kare, 133, 3]
    -> PoomsaeSpec: beklenen M01-M18 sırası ve teknikler
    -> MovementTimeline: bu videodaki M01-M06 kareleri ve faz anchor'ları
    -> kalite/observability kapıları
    -> anchor ve fixation pencerelerinde 3B teknik ölçümler
    -> kaynak bağlı kural karşılaştırmaları
    -> karar kanıtı, işaretli video ve inceleme ekranı
```

Bir ölçümün puan kararına dönüşebilmesi için yalnız doğru kareyi bulmak yetmez.
Pose kanıtının gözlenebilir olması, yeterli kamera desteği taşıması, ilgili
kuralın kaynak durumunun uygun olması ve belirsizlik aralığının sınırı güvenli
biçimde aşması gerekir. Kayıt kısmi olduğu için güncel örnekte tam Accuracy
puanı yine de `null` kalır.

## 5. Bugünkü sistemin kesin sınırı

Bugün için doğru ifade şudur:

- Taegeuk 1'in `M01-M18` sırası projede tanımlıdır.
- Mevcut kısa video için `M01-M06` zamanları iki kameradan elle doğrulanmıştır.
- Sistem bu etiketli fazlarda ayrıntılı 3B/WholeBody ölçümleri yapar.
- Genel hareket enerjisiyle otomatik segment adayı üretebilir.
- Segmentleri bilinen sıraya hizalamak için deneysel yardımcı kod vardır.
- Fakat yeni ve bilinmeyen bir videoda bütün hareketleri ve yumruk bitişlerini
  uçtan uca otomatik, doğrulanmış ve resmî puanlamaya hazır biçimde bulan bir
  üretim hareket tanıma modeli henüz yoktur.

Tam otomasyon için yeni tam M01-M18 kayıtları, her faz için güvenilir etiketler,
referans poz/sekans temsili, otomatik sınır algılamanın bağımsız testleri ve
manuel timeline'a karşı ölçülmüş hata oranları gerekir.
