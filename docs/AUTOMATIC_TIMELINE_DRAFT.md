# Otomatik hareket zaman çizelgesi taslağı

Son güncelleme: **31 Ağustos 2026**

Bu belge, elle etiketlenmemiş bir kayıt için hareket zaman çizelgesi **önerisi**
üreten adımı tanımlar. Öneri insan onayı olmadan puanlamaya girmez.

## Neden ayrı bir komut

Kanonik `tk3d-poomsae` akışı zaman çizelgesini **girdi** olarak alır ve bütün
ölçümleri ona göre yapar. Taslak doğrudan oraya bağlansaydı sistem kendi
tahminini doğru kabul edip üstüne ölçüm yapardı; yanlış eşleşme raporda
sporcunun hatası gibi görünürdü. Bu yüzden akış iki adımdır:

```text
kayıt -> taslak komutu -> insan düzeltmesi -> kanonik puanlama akışı
```

Taslak komutu kanonik akışın hiçbir aşamasında çağrılmaz; bir test bunu koruma
altına alır.

## Zincir

1. **Segment tespiti.** `automatic_segmentation` hareket enerjisinden hareket
   ve faz sınırlarını önerir. Ölçülmüş isabeti AD-028'dedir.
2. **Segment duruşu.** Her segmentin fixation çapası etrafındaki pencerede
   eklem başına ortalama alınır — şablonların üretildiği tarifin aynısı.
3. **Eşleştirme.** Segment duruşları
   `config/scoring/templates/` altındaki referans şablonlarla karşılaştırılır.
   Karşılaştırma konumdan ve ölçekten bağımsızdır; bakış yönü korunur, çünkü
   formun hareketlerini ayıran şey odur.
4. **Taslak ve şüpheler.** Sonuç bir `label_source: automatic` MovementTimeline
   ve ayrı bir hizalama anomali raporudur.

## Eşik ve "düşük güven"

Eşik elle seçilmez, formun kendi şablonlarından çıkar: birbirine en çok benzeyen
iki şablonun farkının yaklaşık üçte ikisi. Bir segmentin eşleştiği harekete olan
farkı eşiği geçerse eşleştirme yapılmaz ve hareket `unmatched_movement` olarak
raporlanır. Fark eşiğin dörtte biri kadar yakınına girerse eşleştirme tutulur
ama `label_status: ambiguous` işaretlenir — sporcunun lehine karar verilir,
şüphe kaydedilir.

## Çapalar

Faz çapaları segmenti eşit aralıklara bölerek değil, dedektörün ölçtüğü
değerlerden alınır. Fixation çapası bütün sonraki ölçümlerin hangi kareleri
örnekleyeceğini belirlediği için bu fark önemlidir. Ölçülen çapalar yalnız
eşleşen hareketin fazlarını birebir adlandırıyor, birleşmiş aralığın içinde
kalıyor ve faz sırasını bozmuyorsa kullanılır; aksi hâlde eşit dağıtıma dönülür.

## Bilinen sınır: kayıt formun başından başlamalı

MovementTimeline sözleşmesi `observed + missing = spec sırası` ister. Bu, gözlenen
hareketlerin spec'in **baştan bir öneki** olması demektir. Yalnız M07-M18 içeren
bir kayıt bu formatta temsil edilemez; komut açık bir mesajla durur, sessizce
yanlış bir çizelge yazmaz.

**Sonuç:** eksik hareketler çekilirken form baştan sona bir kerede kaydedilmeli.
Ortadan başlayan bir kayıt için çizelge elle yazılmak zorundadır.

## Kullanım

```powershell
python scripts/build_poomsae_automatic_timeline_draft.py `
  --pose <vitpose_session_3d.json> `
  --poomsae-spec config/scoring/poomsae/taegeuk_1_jang_v0_draft.yaml `
  --templates config/scoring/templates/taegeuk_1_reference_poses_poomsae1_zed2i_20260802.json `
  --timeline-id <benzersiz-id> `
  --output-timeline <taslak.yaml> `
  --output-anomalies <anomaliler.json>
```

`--expect-movements` varsayılan olarak formun tamamıdır; kısmi bir kayıt için
spec sırasına uyan bir önek verilebilir (`M01,M02,M03`).

Şablonlar elle etiketlenmiş bir kayıttan gelmek zorundadır; otomatik bir
çizelgeden türetilmiş şablon reddedilir, yoksa hizalama kendi ödevini onaylamış
olur.

## Şüphelerin rapora düşmesi

Anomali raporu `scripts/build_poomsae_evidence_events.py` komutuna
`--alignment-anomalies` ile verilebilir. Anomaliler mavi inceleme adayı olur;
hiçbiri kesinti değildir ve hiçbiri sporcunun hatası olduğunu söylemez. Rapor
başka bir çizelge için üretilmişse bağlantı reddedilir.
