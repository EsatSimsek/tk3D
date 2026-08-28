# Otomatik hareket ve faz segmentasyonu

`src/poomsae_scoring/automatic_segmentation.py`, doğrulanmış
`keypoints_3d_world[t,133,3]` dizisinden puansız hareket ve faz sınırı önerileri
üretir. Onaylı `MovementTimeline` dosyasını değiştirmez.

## Algılama akışı

1. BODY-17 eklem hızlarının kare başına ortalaması hareket enerjisi olarak
   hesaplanır.
2. Sinyal kısa, ortalanmış bir medyan penceresiyle yumuşatılır.
3. Veri dağılımından yüksek hareket ve hareket başlangıcı eşikleri çıkarılır.
4. Kısa boşluklarla ayrılan aktif kareler tek hareket kümesinde birleştirilir.
5. Performans öncesi kamera/sporcu hareketi, kümeler arasındaki sessiz süre ve
   beklenen çalışma kapsamı kullanılarak gerçek hareket dizisinden ayrılır.
6. Hazırlık başlangıcı, son yüksek-hareket yavaşlaması ve düşük-harekete giriş
   noktalarından `preparation`, `execution` ve `fixation` ankrajları çıkarılır.
   `turn`, `step` ve `weight_transfer` gibi sıralı ara fazlar bu iki fiziksel
   ankraj arasında açıkça `ordered_interpolation` olarak işaretlenir.

## Güvenlik sözleşmesi

- Sınır algılama referans timeline karelerini kullanmaz.
- Referans timeline yalnız kayıt kapsamındaki hareket kimliklerini sırayla
  vermek ve algılama tamamlandıktan sonra hata ölçmek için kullanılır.
- Çıktı `automatic_segmentation_diagnostic_only` durumundadır.
- Onaylı timeline'ın otomatik değiştirilmesi, puan ve otomatik kesinti yasaktır.
- Pose SHA-256, frame sayısı ve FPS referans timeline ile uyuşmazsa komut kapalı
  biçimde hata verir.

## Çıktılar

- `json/automatic_segmentation_report.json`: hareket pencereleri, faz
  ankrajları, yöntem/provenance ve referans karşılaştırması.
- `csv/automatic_segmentation_signal.csv`: her kare için ham/yumuşatılmış
  hareket enerjisi, eşik durumları, hareket kimliği ve fixation işareti.
- HTML inceleme ekranı: otomatik ve referans sınır farkları ile her otomatik
  fixation anına giden senkron video düğmeleri.

23 Ağustos 2026 gerçek M01-M06 koşusunda 7 aday kümeden performans öncesi küme
ayrıldı ve 6/6 hareket seçildi. Referans karşılaştırmasında başlangıç MAE
11,33 kare, bitiş MAE 10,33 kare, tüm faz ankrajı MAE 6,04 kare (yaklaşık
0,10 saniye), en büyük faz hatası 19 kare oldu. Bu değerler tek kayıt üzerindeki
iç doğrulamadır; farklı sporcu/kamera verisinde dış doğrulama yerine geçmez.
