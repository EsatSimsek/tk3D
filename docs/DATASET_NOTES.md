# TK3D Veri Seti ve Yerel Koşu Notları

Bu belge veri setine özel gerçekleri genel mimariden ayırır.

## AIST / AIST++

Rolü:

- dokuz kameralı akış, senkron, kalibrasyon, triangulation ve video export
  smoke/regresyon testi
- opsiyonel SMPL mesh denemesi
- ground-truth ana doğruluk benchmark'ı değil
- poomsae verisi değil

Aktif örnek sekans:

```text
gBR_sBM_cAll_d04_mBR0_ch01
```

### c05 zaman ofseti

Bu sekansın c05 videosunda dans bölümü diğer kameralardan yaklaşık 268 kare
geç başlar.

- Geometri taraması: `+268`
- Bağımsız hareket sinyali: `+269`
- Session sözleşmesindeki ayar: `sync.offsets.c05: -268`
- Dokuz kameranın ortak fiziksel bölümü: 451 kare

Bu değer yalnız bu örnek sekansa aittir. Başka kayıt veya başka c05 kamera için
otomatik uygulanmamalıdır.

Kısa 20 karelik doğrulamada senkron düzeltmesi c05 medyan konsensüs hatasını
`45.57 px` değerinden `4.58 px` değerine indirdi ve kamera `healthy` göründü.
Bu kısa sonuç tam koşunun yerine geçmez.

Son 451 karelik tam koşuda:

- c05 sınıfı: `localized_2d_joint_detection`
- c05 target sayısı: `97`
- medyan konsensüs: yaklaşık `5.40 px`
- P95: yaklaşık `17.94 px`
- global kamera ağırlığı: yaklaşık `0.811`

Yorum: Senkron kök nedeni çözülmüştür; kalanlar lokal 2B eklem/occlusion
aykırılarıdır.

### Son korunan AIST run

```text
outputs/aist_test/runs/pose_2026-07-26_15-07-28_797
```

Ana dosyalar:

```text
videos/vitpose_skeleton_3d_world.mp4
viewer/pose3d_viewer.html
json/run_quality_report.json
json/camera_health_report.json
json/crossview_2d_feedback_report.json
json/global_pose_optimization_report.json
```

## MADS

Rolü:

- kalibre üç RGB kamera
- optik motion-capture ground truth
- bağlı RGB-only koşu için tarihsel dış 3B doğruluk benchmark'ı
- karate/tai-chi hareketleri nedeniyle poomsae'ye AIST dansından daha yakın

MADS metrikleri koşuya, kamera/profil yapılandırmasına ve tahmin dosyasına
bağlıdır. Başka bir koşuya, özellikle ZED stereo depth kullanan güncel RGBD
hattına devredilmez. ZED depth sistemin kendi sensör kanıtıdır; bağımsız
ground-truth değildir.

Split ilkesi:

- Kata F2 held-out testtir.
- Eğitim veya validation cache'ine girmemelidir.
- Model/adapter seçimi F2 global MPJPE sonucuyla kontrol edilir.

Son belgelenmiş temel model F2 sonucu:

```text
300 inference örneği, stride 2
Global MPJPE        90.409 mm
P95                162.504 mm
Root-relative      106.791 mm
PA-MPJPE            73.358 mm
PCK@100 mm           0.663
Açı MAE             13.426 derece
Geçerli eklem       %95.89
```

`50 mm` global MPJPE hedefi bu bağlı MADS koşusunda geçilmedi; bu koşunun
puanlama yetkisi kapalıdır. Sonuç güncel ZED RGBD koşusunun başarısızlık metriği
değildir. Denenen MADS adapter'ları 2B validation metriğini iyileştirse de
held-out 3B sonucu kötüleştirdiği için reddedildi.

Bu sonuç 27 Temmuz 2026'da güncel kodla yeniden üretildi. MADS yalnız üç
kameralı olduğu için en az dört destekleyici görüş isteyen cross-view düzeltme
aday üretmedi; MADS session profili global BODY-17 optimizasyonunu da kapalı
tuttu. Bu nedenle sonuç önceki temel model ölçümüyle aynıdır. Özelliklerin MADS
için eşikleri ayrıca doğrulanmadan yeni optimizasyonların dış doğruluğu
iyileştirdiği söylenemez.

## Gerçek poomsae verisi

Henüz doğrulanması gerekenler:

- senkron global-shutter çoklu kamera çekimi
- ortak checkerboard hacim kalibrasyonu ve drift kontrolü
- farklı sporcu, seviye, kıyafet ve poomsae
- mümkünse mocap referansı
- gerçek phase/step başlangıç-bitiş etiketleri
- en az birkaç uzman/hakemden teknik hata ve skor etiketi
- sporcu ve oturum bazlı ayrılmış train/validation/test

AIST veya MADS sonucundan gerçek poomsae saha doğruluğu çıkarılmamalıdır.
ZED RGB-vs-depth kapısından da bağımsız dış doğruluk sonucu çıkarılmamalıdır.

## SMPL

- 3B çubuk iskelet veya puanlama için zorunlu değildir.
- Yalnız gerçekçi mesh görselleştirmesi içindir.
- Lisanslı `.pkl` dosyaları repository'ye eklenmez.
- Kullanıcı modeli kendi hesabıyla `models/smpl/` altına koyar.

## Yerel ve Git'e girmeyen varlıklar

```text
weights/
models/
external/
data/aist_test/videos/
data/aist_test/annotations/
data/mads_test/local/
outputs/
.venv*
```

Bu dosyaları temizlemek veya yeniden indirmek kullanıcı onayı gerektirir.
