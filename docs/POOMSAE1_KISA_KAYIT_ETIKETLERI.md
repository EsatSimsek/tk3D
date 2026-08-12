# Poomsae 1 Kısa Kayıt Hareket Etiketleri

Son inceleme: **3 Ağustos 2026**

## Kapsam kararı

İki ZED2i kamera görüntüsü birlikte incelendi. `741` karelik kayıt Taegeuk 1
Jang'ın tamamını içermiyor; resmî 18 hareketlik sıranın yalnız ilk **6
hareketini** içeriyor. Bu nedenle M07-M18 etiketi üretilmedi ve kayıt için
toplam Accuracy puanı hesaplanmadı.

MovementTimeline v2 bu durumu `partial_sequence` olarak taşır. Böylece M07-M18
“etiketlenemedi” veya “algılanamadı” sayılmaz; kaynak kayıtta bulunmayan
hareketler olarak açıkça ayrılır.

İlk `140` kare hazırlık/bekleme alanıdır. M01-M06 etiketleri iki kamera ve
bilinen Kukkiwon sırası kullanılarak çıkarılmıştır. Başlangıç, hareket geçişi
ve fixation kareleri iki kamera temas sayfalarında yeniden incelendiği için
zaman çizelgesi etiketleri `confirmed` durumundadır. Bu yalnız zaman sınırı
onayıdır; teknik doğruluk veya hakem onayı değildir.

## Hareket aralıkları

| Hareket | Tanım | Kare | Zaman | Güven |
|---|---|---:|---:|---:|
| M01 | Sol ap-seogi + sol arae-makki | 140-229 | 2,333-3,817 s | 0,86 |
| M02 | Sağ ap-seogi + sağ momtong-jireugi | 230-353 | 3,833-5,883 s | 0,88 |
| M03 | Sağ ap-seogi + sağ arae-makki | 354-473 | 5,900-7,883 s | 0,86 |
| M04 | Sol ap-seogi + sol momtong-jireugi | 474-569 | 7,900-9,483 s | 0,89 |
| M05 | Sol ap-gubi + sol arae-makki | 570-685 | 9,500-11,417 s | 0,88 |
| M06 | Sol ap-gubi içinde sağ momtong-jireugi | 686-740 | 11,433-12,333 s | 0,87 |

## Faz anchor'ları

| Hareket | Faz ve kareler |
|---|---|
| M01 | preparation 145; turn 178; execution 205; fixation 218 |
| M02 | preparation 230; step 260; execution 289; fixation 301 |
| M03 | preparation 354; turn 388; execution 432; fixation 448 |
| M04 | preparation 474; step 490; execution 517; fixation 529 |
| M05 | preparation 570; turn 588; weight_transfer 616; execution 628; fixation 637 |
| M06 | preparation 686; execution 722; fixation 728 |

Anchor bir fazın tek doğru sınırı anlamına gelmez; o fazı ölçmek ve ekranda
göstermek için seçilen temsil karesidir. Hareket aralıkları çakışmaz ve kaynak
videonun `60 FPS` örnek indeksini kullanır.

## İnceleme çıktıları

- [Güncel WholeBody-133 senkron inceleme ekranı](../outputs/poomsae_1_zed2i_20260731_trimmed/runs/poomsae1-scoring-expand-v2_3-20260803-123907/review/poomsae1_scoring_review.html)
- [WholeBody-133 teşhis JSON'u](../outputs/poomsae_1_zed2i_20260731_trimmed/runs/poomsae1-scoring-expand-v2_3-20260803-123907/json/wholebody_diagnostics_report.json)
- [WholeBody metrik CSV'si](../outputs/poomsae_1_zed2i_20260731_trimmed/runs/poomsae1-scoring-expand-v2_3-20260803-123907/csv/wholebody_metrics.csv)
- [Kamera 35151067 hareket/faz videosu](../outputs/poomsae_1_zed2i_20260731_trimmed/runs/poomsae1-engineering-trial-20260802-234034/videos/zed_35151067_movement_labels.mp4)
- [Kamera 37137479 hareket/faz videosu](../outputs/poomsae_1_zed2i_20260731_trimmed/runs/poomsae1-engineering-trial-20260802-234034/videos/zed_37137479_movement_labels.mp4)
- [Ölçüm ve observability raporu](../outputs/poomsae_1_zed2i_20260731_trimmed/runs/poomsae1-engineering-trial-20260802-234034/json/movement_evidence_report.json)
- [Faz ölçümleri CSV](../outputs/poomsae_1_zed2i_20260731_trimmed/runs/poomsae1-engineering-trial-20260802-234034/csv/movement_phase_measurements.csv)
- [Geçersizleştirilmiş BODY-17 teşhis kaydı](../outputs/poomsae_1_zed2i_20260731_trimmed/runs/poomsae1-scoring-expand-v2_3-20260803-123907/json/deprecated_body17_screening.json)
- [WholeBody bağlı puanlama hazırlık raporu](../outputs/poomsae_1_zed2i_20260731_trimmed/runs/poomsae1-scoring-expand-v2_3-20260803-123907/json/accuracy_readiness_report.json)
- [Güncel hazırlık raporu](../outputs/poomsae_1_zed2i_20260731_trimmed/runs/poomsae1-engineering-trial-20260802-234034/json/accuracy_readiness_report.json)
- [Kaynak ve çıktı SHA-256 manifesti](../outputs/poomsae_1_zed2i_20260731_trimmed/runs/poomsae1-engineering-trial-20260802-234034/json/poomsae1_review_manifest.json)

Ölçüm raporunda `24` faz anchor'ının `22` tanesi `observed`, `2` tanesi
`partially_observed`, hiçbiri `not_measurable` değildir. Bu değer pose
gözlenebilirliğini anlatır; tekniğin doğru olduğu veya puan kesilmesi gerektiği
anlamına gelmez.

İlk BODY-17 denemesindeki `4,0/4,0` yorumu geçersizdir: el, ayak ve yüz
ayrıntılarını kullanmıyor, ölçülemeyen özellikleri de puandan düşmüyordu. Bu
motor artık skor üretmez. Önceki WholeBody-133 v2.3 koşusu eşikli `96` metriğin `74` tanesini
ölçebildi (`%77,08`); `%90` kapsam kapısı başarısızdır. Fixation, zamanlama,
ayak yönü, teknik dirsek, bakış ve ağırlık aktarımı ailelerinde toplam `13`
video-inceleme adayı bulundu.
Hiçbiri otomatik ceza değildir; gerçek `accuracy_score` `null`dır.

WholeBody v2.4 ve güncel source-bound v1 koşusu
[`poomsae1-source-bound-20260810-221705`](../outputs/poomsae_1_zed2i_20260731_trimmed/runs/poomsae1-source-bound-20260810-221705/json/source_bound_accuracy_decisions.json)
altındadır. Dokuz tarihsel kaynak-bağlı geometrinin dördü `%95` belirsizlik ve
koruma bandıyla küçük hata, biri aralık içi, biri sınır-belirsiz, üçü ölçülemez
çıktı. Yalnız gözlenen kapsam kesinti toplamı `0,4`; kısmi kayıt nedeniyle
`accuracy_score=null`dır.

## Sonraki kayıt gereksinimi

Tam Taegeuk 1 puan akışını sınamak için M01-M18'in tamamını, başlangıç ve bitiş
hazırlığı görünür olacak şekilde kesintisiz kaydeden yeni bir çekim gerekir.
Mevcut kısa kayıt M01-M06 hareket tanıma, faz ölçümü ve overlay doğrulaması için
korunacaktır.
