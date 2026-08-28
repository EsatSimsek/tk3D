# TK3D Veri Seti ve Yerel Koşu Notları

Bu belge veri setine ve yerel kayda özgü gerçekleri genel mimariden ayırır.
Sıralama güncel ürün/araştırma önceliğini izler: önce `CURRENT_ACTIVE`, sonra
`CURRENT_VALIDATION`, tarihsel benchmark ve legacy/opsiyonel varlıklar.

## CURRENT_ACTIVE — iki ZED 2i Poomsae kaydı

Aktif geliştirme ve birincil regresyon workflow'u:

`poomsae1_trimmed`: iki ZED 2i → RGBD multiview 3B → kaynak-bağlı Poomsae
analizi.

Aktif bağlar:

```text
config/scoring/profiles/poomsae1_trimmed.yaml
outputs/poomsae_1_zed2i_20260731_trimmed/source/session.yaml
outputs/poomsae_1_zed2i_20260731_trimmed/calibration/cameras.json
```

Session iki kamerayı tanımlar:

- `zed_35151067`
- `zed_37137479`

Kayıt 60 FPS'tir. Kullanıcı tarafından seçilmiş trim, parent kaydın kaynak
kare `190–930` aralığını ve 741 çıktı karesini kapsar.

### AVI RGB rolü

Her kamera için kayıpsız AVI, RF-DETR kişi tespiti, ByteTrack, ViTPose-Huge
WholeBody-133 heatmap/2B gözlemi, multiview triangulation ve review videosunun
görüntü kanıtıdır. Frame/timestamp kimliği session ve mapping raporlarıyla
korunur.

### SVO2 depth rolü

Her AVI kamera için bağlı SVO2 kaydı ve timestamp mapping raporu vardır. SVO2,
ZED SDK üzerinden stereo depth/confidence sağlar. Depth yalnız triangulation
sonrası, confidence/yüzey/residual ve final RGB-vs-depth kalite kapılarıyla
BODY-17 yardımcı fusion olarak kullanılır. RGB görüntü kanıtının veya bağımsız
ground truth'un yerine geçmez.

IMU kullanımı kamera gravity/orientation calibration içindir; kare-bazlı
sporcu hareket düzeltmesi değildir.

### Güncel rol ve sınırlamalar

- Bu kayıt CURRENT_ACTIVE geliştirme, davranış dondurma ve birincil regresyon
  örneğidir.
- Session, AVI, SVO2, timestamp raporları, calibration, checkpoint ve bağlı
  reference pose Git dışı yerel araştırma varlıklarıdır.
- Aktif session YAML makineye özgü SVO2/timestamp yolları içerebilir.
- İki kamera, en az dört başka destekleyici view isteyen cross-view guided
  ikinci geçişi çalıştırmaz; bu koşuda `ZERO_WORK` beklenebilir.
- İç reprojection/depth tutarlılığı bağımsız dış 3B doğruluğu değildir.
- Bu kayıt için bağımsız mocap/ölçüm ground truth yoktur.
- Elle doğrulanmış Poomsae timeline yalnız M01–M06 kapsamındadır.
- Uzman/hakem karar etiketi ve judge calibration yoktur.
- Gerçek saha/genelleme doğruluğu için farklı sporcu, seviye, kıyafet,
  poomsae, kamera düzeni ve oturumlar gerekir.

AIST veya MADS sonucundan bu ZED RGBD kaydının saha doğruluğu çıkarılmamalıdır.

## CURRENT_VALIDATION — AIST / AIST++

Sınıfı: `CURRENT_VALIDATION`. TK3D'nin aktif ürün/geliştirme workflow'u değildir.

Rolü:

- dokuz kameralı akış, senkron, kalibrasyon, triangulation ve video export
  smoke/regresyon testi;
- opsiyonel SMPL mesh denemesi;
- ground-truth ana doğruluk benchmark'ı değil;
- Poomsae verisi değil.

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

AIST sonucu CURRENT_ACTIVE ZED/Poomsae regresyonunun yerine kullanılamaz.

## HISTORICAL_BENCHMARK — MADS

Sınıfı: `HISTORICAL_BENCHMARK`.

Rolü:

- kalibre üç RGB kamera;
- optik motion-capture ground truth;
- bağlı RGB-only koşu için tarihsel dış 3B doğruluk benchmark'ı;
- karate/tai-chi hareketleri nedeniyle poomsae'ye AIST dansından daha yakın.

MADS metrikleri koşuya, kamera/profil yapılandırmasına ve tahmin dosyasına
bağlıdır. Başka bir koşuya, özellikle ZED stereo depth kullanan güncel RGBD
hattına devredilmez. ZED depth sistemin kendi sensör kanıtıdır; bağımsız
ground truth değildir.

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

## LEGACY / tarihsel yerel kayıtlar

Tek-ZED SVO2 pilotları ve tek-kamera RGB-D önizlemeleri ingest, depth ve zaman
çizelgesi davranışının geliştirilmesinde kullanılmış tarihsel teknik kayıtlardır.
Ortak-dünya multiview 3B veya güncel Poomsae workflow'u değildirler.

Eski sentetik `run_multiview_3d.py --dry-run` veri yolu
`SUPPORTED_COMPATIBILITY / LEGACY` sınıfındadır. Gerçek kamera, aktif model veya
bilimsel benchmark sayılmaz.

Bu varlıklara ait ölçüm ve pilot günlüğü
[`history/PROJECT_STATUS_PRE_FINAL_POLISH.md`](history/PROJECT_STATUS_PRE_FINAL_POLISH.md)
içinde korunur.

## Opsiyonel SMPL varlıkları

- 3B çubuk iskelet veya Poomsae analizi için zorunlu değildir.
- Yalnız AIST hareketlerinden gerçekçi mesh görselleştirmesi içindir.
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

Bu dosyaları temizlemek, taşımak veya yeniden indirmek kullanıcı onayı
gerektirir. Provenance ve tarihsel run artifact'leri kullanıcı açıkça istemeden
silinmez.
