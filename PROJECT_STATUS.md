# TK3D Güncel Proje Durumu

Son doğrulama tarihi: **27 Temmuz 2026**
Doğrulanan temel kod revizyonu: **`fa08d65`**
Dal: **`main`**

Bu dosya değişken proje durumunun tek kısa kaynağıdır. Yeni oturumlarda geçmiş
`sohbet*.md` dosyalarını baştan okumak yerine önce burası okunmalıdır.

## Projenin amacı ve mevcut sınırı

TK3D'nin hedefi çok kameralı videodan güvenilir 3B insan pozu üretip bunu
tekvando poomsae analiz ve puanlama sistemine girdi yapmaktır. Mevcut repository
güçlü bir görüntü işleme, 3B geometri, kalite kontrolü ve geçici teknik analiz
altyapısı içerir.

Sistem henüz resmî poomsae puanlamasına hazır değildir:

- MADS held-out ground-truth hedefi geçilmedi.
- Gerçek poomsae phase/step etiketleri ve hakem/koç onaylı hedefler yok.
- İç geometri raporunun geçmesi gerçek 3B doğruluğu tek başına kanıtlamaz.
- Normal çok-kameralı run raporu bu nedenle `scoring_ready: false` yazar.
- Ground-truth değerlendiricisi artık koşuya bağlı `scoring_authorization.json`
  üretir. Yalnız bütün dış doğruluk ve iç geometri kapıları geçerse burada
  `scoring_ready: true` olur; tahmin JSON'undaki elle yazılmış bir alan tek
  başına puanlama açamaz.

## Aktif üretim hattı

```text
senkronize kamera videoları
  -> RF-DETR-Small kişi tespiti
  -> ByteTrack kamera içi kimlik takibi
  -> adaptif ve stabilize kişi crop'u
  -> ViTPose-Huge WholeBody (133 nokta, flip-test)
  -> güven/hareket duyarlı temporal filtre
  -> sıfır-fazlı offline 2B stabilizasyon
  -> robust çok-kameralı ilk triangulation
  -> leave-one-camera-out çok-kameralı 2B geri besleme
  -> güvenli yeniden triangulation
  -> anatomik/zamansal 3B güvenilirlik kontrolü
  -> BODY-17 global çok-kameralı optimizasyon
  -> 3B stabilizasyon, export ve kalite raporları
```

Aktif yapılandırma `config/model_config.yaml` içindedir:

- 2B model: `ViTPose-Huge-WholeBody`
- Backend: `tk3d_vitpose_plus`
- Girdi: `256x192`
- Veri sözleşmesi: 133 COCO-WholeBody noktası
- Kişi dedektörü: RF-DETR Small
- Tek kamera RTMW3D helper: opsiyonel ve varsayılan kapalı
- Cross-view 2B feedback: açık
- Global BODY-17 optimization: açık

## Ana veri ve koordinat sözleşmeleri

- 3B çıktı şekli: `keypoints_3d_world[t, 133, 3]`
- Birim: metre
- Analiz eksenleri: `x=sağ`, `y=ileri`, `z=yukarı`
- 133 nokta export ve görselleştirmede korunur.
- Cross-view geri besleme ve global optimizasyon şu anda yalnız BODY-17 üzerinde
  çalışır. El, yüz ve ayak için eşdeğer çapraz-kamera optimizasyonu doğrulanmış
  değildir.
- Kalibrasyon canlı üretimde fail-closed'dur. Kabul edilen üretim modları ortak
  çok-kamera referansı veya resmî AIST çok-kamera kalibrasyonudur.

## Kritik güvenlik mekanizmaları

### Cross-view 2B feedback

- Bir kamera düzeltilirken o kamera 3B öncülden çıkarılır.
- En az dört başka kamera bağımsız öncülü destekler.
- Öncül hedef görüntüde yalnız ViTPose heatmap aramasını sınırlar.
- Adayın gerçek heatmap skoru ve geometrik hatası birlikte iyileşmelidir.
- Kabul edilmeyen aykırı görüntü ölçümü 3B geometriden çıkarılır.
- Yalnız izdüşümle gösterilen nokta turuncu ve
  `visualization_only` provenance ile işaretlenir.
- Yeniden triangulation geçerli nokta, medyan ve P95 reprojection kapılarını
  geçmezse ilk geometriye dönülür.

### Global BODY-17 optimization

- Tüm senkron sekansı birlikte çözer.
- Reprojection, kamera güveni, kemik uzunluğu, dirsek/diz limitleri, ivme,
  jerk ve kısa kapanma bilgilerini kullanır.
- Ham triangulation ayrı dosyada korunur.
- Solver, reprojection, ivme, kemik kararlılığı veya düzeltme sınırı
  kötüleşirse ham sonuca otomatik dönüş yapılır.
- Kısa kapanmadan tamamlanan nokta doğrudan gözlenmiş noktadan ayrı provenance
  taşır; uzun kanıtsız boşluklar doldurulmaz.

## Güncel doğrulama

### İlk ZED 2i SVO2 pilotu

28 Temmuz 2026 tarihinde
`HD720_SN39504762_13-06-55.svo2` gerçek ZED SDK 5.4.1 ile işlendi:

- ZED 2i seri `39504762`, firmware `1523`;
- kalibre odak `3.823 mm`, yatay görüş `67.73°`; 4 mm lens varyantıyla uyumlu;
- `1280×720`, `60 FPS`, 666 başarıyla okunan kare, `11.134046 s`;
- IMU verisi `666/666` karede mevcut;
- üç adet yaklaşık `33 ms` zaman aralığı görüldü; uzun stres testi henüz yok;
- NEURAL depth geçerli piksel medyanı `%99.945`, güçlü güven medyanı `%95.905`;
- RF-DETR + ViTPose smoke: 67 örnek karede BODY-17 algılama `%100`;
- kişi-depth kapısıyla BODY-17 ve ayak depth eşleşmesi `%100`;
- güvenli yüz depth eşleşmesi `%90.89`, el depth eşleşmesi `%81.69`;
- naif eklem-piksel depth örneklemesinin siluet kenarlarında arka plana
  sıçradığı gösterildi; üretim entegrasyonu kişi maskesi ve çok-kamera
  residual kapısı olmadan yapılmamalıdır;
- doğru zaman çizelgeli sol RGB ve teşhis raporları
  `outputs/zed2i_pilot/runs/zed2i-svo-diagnostic-20260728-180605/`
  altındadır.

Bu pilot tek-kamera RGB-D teknik doğrulamasıdır. Ortak dünya koordinatlı
çok-kamera 3B veya puanlama yetkisi sağlamaz. Kalıcı ZED ingest/depth-fusion
çalışma zamanı henüz `src/` hattına eklenmemiştir.

### Tam tek-kamera RGB-D kontrol koşusu

28 Temmuz 2026 tarihinde aynı SVO2'nin 666 karesinin tamamı, mevcut RF-DETR,
ViTPose, 2B stabilizasyon, güvenilirlik filtresi, 3B yumuşatma ve HTML görüntüleyici
modülleri yeniden kullanılarak izole bir çıktı koşusunda işlendi:

- kaynak zaman çizelgesi korundu: `666` kare, `60 FPS`, `11.134046 s`;
- güvenilir son BODY-17 oranı `%95.3983`;
- güvenilir son WHOLEBODY-133 oranı `%96.1245`;
- 2B video, kamera-koordinatlı 3B video, birleşik video, etkileşimli HTML,
  `[666, 133, 3]` JSON ve düz CSV üretildi;
- JSON/CSV çıktılarında `NaN`/`inf` sızıntısı görülmedi; eksikler `null`/boş
  hücre olarak korundu;
- koşu `outputs/zed2i_single_view/runs/zed2i-single-rgbd-20260728-183704/`
  altındadır ve ana `src/`/`config/` hattını değiştirmemiştir.

Bu çıktı sol kamera optik merkezine göre metre cinsinden
`x=sağ, y=ileri, z=yukarı` RGB-D önizlemesidir. Ortak dünya koordinatlı
çok-kamera triangulation sonucu değildir ve puanlama yetkisi taşımaz.

### Otomatik testler

`fa08d65` üzerinde 27 Temmuz 2026 tarihinde:

```text
132 passed in 21.31s
```

Komut:

```powershell
.\.venv312\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp outputs\pytest-scoring-auth-full-20260727
```

Önceki sohbetlerdeki `28`, `31`, `47`, `73`, `91` ve `128` sayıları kendi
tarihlerindeki sonuçlardır; güncel test sayısı değildir.

### Son korunan tam AIST koşusu

```text
outputs/aist_test/runs/pose_2026-07-26_15-07-28_797
```

- 9 kamera
- `stride 1`
- 451 ortak senkron kare
- 9 adet 2B overlay video
- 1 adet 3B iskelet video
- İç geometri durumu: `passed`
- Ground-truth değerlendirmesi: yapılmadı
- `scoring_ready`: `false`
- BODY-17 geçerli oranı: `1.0`
- Ortalama reprojection: `6.364 px`
- Cross-view hedefi: `814`
- Görüntü heatmap'iyle kabul edilen düzeltme: `51`
- Geometriden reddedilen aykırı gözlem: `763`
- Feedback medyan reprojection: `5.314 -> 5.275 px`
- Feedback P95 reprojection: `9.089 -> 8.953 px`
- Global BODY-17 optimization: uygulandı, fallback kullanılmadı

Tam koşudaki kamera sağlığı:

- `c01`, `c09`: `healthy`
- `c02`–`c08`: farklı yoğunluklarda `localized_2d_joint_detection`
- c05 kamera ağırlığı: yaklaşık `0.811`
- c05 medyan konsensüs hatası: yaklaşık `5.40 px`

c05 senkron sorunu çözülmüş olsa da tam koşuda bütün lokal 2B eklem
aykırılarının bittiği iddia edilmemelidir.

### Dış ground-truth benchmark

27 Temmuz 2026 tarihinde yeni 3B koduyla yeniden çalıştırılan MADS Kata F2
koşusu:

- 3 resmî kalibre kamera
- 300 inference örneği, stride 2
- Global MPJPE: `90.409 mm`
- P95: `162.504 mm`
- Açı MAE: `13.426°`
- Geçerli eklem oranı: `%95.89`
- Sonuç: ground-truth kalite kapısı başarısız
- Koşu:
  `outputs/mads_kata_f2/runs/mads-kata-f2-rescore-20260727-201327`
- Yetkilendirme: `decision=denied`, `scoring_ready=false`
- İç geometri: geçti; yetkilendirme dosya bütünlüğü: geçti
- Üretim modeli: temel ViTPose-Huge; deneysel MADS adapter'ları reddedildi

Sonuç önceki ölçümle aynıdır. Nedeni MADS'in yalnız üç kameralı olması nedeniyle
en az dört destekleyici kamera isteyen cross-view düzeltmenin aday üretememesi
ve MADS session yapılandırmasında global BODY-17 optimizasyonunun kapalı
olmasıdır. Bu özellikler MADS'e güvenli ve ölçülebilir bir profil eklenmeden
iyileşme iddia edilemez.

### Puanlama yetkilendirmesi

- `evaluate_ground_truth_3d.py` tahmin, ground truth, doğrulama config'i, kare
  eşleşmeleri, validation manifesti ve iç-geometri raporunu SHA-256 ile bağlar.
- `analyze_pose_for_scoring.py` normal modda yalnız bu sidecar'ı doğrular;
  tahmin JSON'undaki bağımsız `scoring_ready` değerini güven kaynağı saymaz.
- Herhangi bir bağlı dosya değişirse yetki kapanır.
- `scoring_ready`, dış 3B doğruluğu geçen geçici puanlama altyapısı içindir.
  `official_scoring_ready`, gerçek poomsae kural ve hakem doğrulaması için ayrı
  tutulur ve şu anda `false` değerindedir.

## Çalıştırma

En güçlü yerel AIST testi:

```powershell
.\scripts\run_full_aist_pose.ps1
```

Bu komut benzersiz tarihli run klasörü oluşturur ve `stride 1` kullanır.

Hızlı preview:

```powershell
.\.venv312\Scripts\python.exe scripts\run_vitpose_multiview_3d.py `
  --session data\aist_test\session.yaml `
  --stride 10 `
  --run-id preview-<benzersiz-id>
```

`--max-frames` yalnız bilinçli kısa preview için kullanılmalıdır. `stride`,
çıktı video süresini değiştirmemelidir.

## Temel çıktı dosyaları

```text
outputs/<session_id>/runs/<run_id>/
  videos/<camera>_vitpose_2d_overlay.mp4
  videos/vitpose_skeleton_3d_world.mp4
  viewer/pose3d_viewer.html
  json/run_quality_report.json
  json/camera_health_report.json
  json/crossview_2d_feedback_report.json
  json/global_pose_optimization_report.json
  csv/vitpose_keypoints_2d_prefeedback_flat.csv
  csv/vitpose_keypoints_2d_geometry_flat.csv
  csv/vitpose_keypoints_2d_feedback_provenance.csv
  csv/vitpose_keypoints_3d_world_triangulated_flat.csv
  csv/vitpose_keypoints_3d_world_global_optimized_flat.csv
  csv/vitpose_keypoints_3d_provenance.csv
  csv/vitpose_keypoints_3d_world_flat.csv
```

## Bilinen eksikler ve doğru öncelik

1. Yeni görüntü işleme mimarisini MADS F2 held-out ground truth üzerinde aynı
   protokolle yeniden benchmark etmek.
2. Gerçek Taekwondo BODY-17 görüntülerini etiketleyip yüksek çözünürlüklü
   alan-özel pose modeli eğitmek.
3. El, yüz ve ayak için güvenli cross-view düzeltme/optimizasyonu geliştirmek.
4. c02–c08 lokal 2B algılama aykırılarını kamera, eklem ve hareket fazı
   bazında azaltmak.
5. Gerçek poomsae kameralarında senkron checkerboard kalibrasyonu, drift ve
   uzun süreli stres testi yapmak.
6. Çok sporculu görüntülerde kamera-arası kimlik eşlemeyi doğrulamak.
7. Poomsae phase/step etiketleri ile hakem/koç onaylı teknik hedefler
   oluşturmak.

## Tarihsel belgeler

- `sohbet1.md`–`sohbet6.md`: oturum özetleri, kalıcı gerçek kaynağı değildir.
- `tk3d_architecture_deep_dive.md`: erken mimariyi anlatır; MMPose, basit DLT
  ve eski scoring örnekleri güncel hattı temsil etmeyebilir.
- `README.md`: kurulum ve kullanıcı komutları için ana belgedir; değişken
  benchmark/test durumu için bu dosya tercih edilmelidir.
