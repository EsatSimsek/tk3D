# TK3D

TK3D'nin asıl amacı, tekvando poomsae videolarını teknik olarak analiz edip puanlayabilen bir 3D poomsae scoring sistemi geliştirmektir.

Bu repository şu anda nihai puanlama motoruna giden ara katmanı kurar: çok kameralı poomsae videolarından kalibrasyonlu 3D insan pozu/iskeleti üretmek, bu çıktıyı kalite kontrolünden geçirmek, hareket segmentlerine hazırlamak ve puanlama algoritmasının kullanacağı veri sözleşmesini oluşturmak.

## AI Aracı İçin Hızlı Bağlam

Bu projeyi okuyan bir AI aracı şunu varsaymalıdır:

- Nihai ürün, poomsae performansını otomatik veya yarı otomatik puanlayan bir analiz sistemidir.
- 3D iskelet üretimi projenin asıl amacı değil, puanlama için gerekli ara çıktıdır.
- Ana ara veri sözleşmesi `keypoints_3d_world[t, 133, 3]` formatındaki COCO-WholeBody tabanlı 3D dünya koordinatlarıdır.
- Şu an odak, video -> 2D pose -> multi-view 3D pose -> kalite analizi -> biomekanik özellikler -> hareket segment adayları zincirini sağlamlaştırmaktır.
- Puanlama motoru henüz tamamlanmadı; sıradaki büyük iş phase/step detection, teknik hata metrikleri ve skor üretimidir.
- AIST Dance/AIST++ verisi gerçek poomsae videosu gelmeden kamera, triangulation, ViTPose inference, SMPL mesh ve scoring-readiness akışını test etmek için kullanılıyor.
- Kendi poomsae videoları geldiğinde checkerboard calibration, kişi takibi/eşleme, poomsae adım segmentasyonu ve kural tabanlı/öğrenmeli scoring katmanı eklenecek.

Ana ara hedef veri:

```python
keypoints_3d_world[t, 133, 3]
```

Bu ilk sürüm, nihai puanlama sistemine temel olacak şu bileşenleri içerir:

- Checkerboard tabanlı kamera kalibrasyonu için giriş noktası
- ViTPose-Huge 2D wholebody tahmin sınıfı için entegrasyon arayüzü
- Kalibrasyonlu multi-view triangulation
- Kamera `frame_offset` değerlerini dikkate alan global frame senkronizasyonu
- Normalize DLT triangulation, reprojection-error tabanlı triangulation quality score
- Sentetik 3 kamera dry-run verisi ile triangulation doğrulama
- 3D temporal smoothing
- 3D validation, kalite ölçümleri ve scoring-readiness analizi
- JSON, CSV, Excel ve figür export iskeleti
- Pytest tabanlı çekirdek algoritma testleri
- Gelecekteki 3D poomsae scoring motoruna uygun veri yapıları

## Kurulum

```powershell
cd C:\Users\WWWW\Desktop\tk3d
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

ViTPose-Huge WholeBody canlı 2D inference için ağırlık dosyası `weights/vitpose_huge_wholebody_256x192.pth` altında tutulur ve Git'e eklenmez. Resmi ViTPose kodu `external/vitpose` altında yerel runtime olarak kullanılır.

## İlk kontrol

```powershell
python scripts\inspect_session.py --session data\session_001\session.yaml
python scripts\preflight_session.py --session data\session_001\session.yaml
python scripts\probe_videos.py --session data\session_001\session.yaml
python scripts\check_models.py --session data\session_001\session.yaml
python scripts\run_multiview_3d.py --session data\session_001\session.yaml --dry-run
python -m pytest -q
```

Codex/sandbox ortamında pytest cache veya temp izinleri sorun çıkarırsa şu form kullanılabilir:

```powershell
python -m pytest -q -p no:cacheprovider --basetemp outputs\pytest-tmp
```

Son doğrulama sonucu:

```text
31 passed
```

`--dry-run`, gerçek video ve model olmadan sentetik dünya koordinatları üretir, bunları 3 kamera projection matrix ile 2D'ye projekte eder ve gerçek multi-view triangulation kodundan geçirerek beklenen output yapısını üretir.
Bu komut gerçek bir `outputs/session_001/videos/skeleton_3d_world.mp4` dosyası üretir. Video, sentetik 3D dünya iskeletinin frame frame render edilmiş halidir; siyah placeholder değildir.

Gerçek video/model dosyaları geldiğinde önce strict preflight çalıştırılır:

```powershell
python scripts\preflight_session.py --session data\session_001\session.yaml --require-videos --require-calibration-videos --require-model-files
```

## AIST Video Testi

AIST Dance Video DB videoları, poomsae videosu gelmeden çok kameralı görüntü akışını test etmek için kullanılabilir. AIST++ annotation dosyaları COCO 17 eklem formatındadır; bu nihai COCO-WholeBody 133 hedefini değiştirmez. Videolar bizim ViTPose-Huge WholeBody adapter yoluna girdiğinde hedef yine 133 eklemdir. AIST++ 17 eklem verisi sadece calibration, projection, triangulation ve hata ölçümü için opsiyonel doğrulama verisidir.

Yerel test klasörlerini hazırlamak:

```powershell
python scripts\setup_aist_test.py --sequence gBR_sBM_cAll_d04_mBR0_ch01 --cameras c01 c02
```

Bu komut şunları hazırlar:

- `data/aist_test/videos/`
- `data/aist_test/annotations/`
- `data/aist_test/session.yaml`
- `data/aist_test/aist_test_manifest.json`

Beklenen ilk video dosyaları:

```text
data/aist_test/videos/gBR_sBM_c01_d04_mBR0_ch01.mp4
data/aist_test/videos/gBR_sBM_c02_d04_mBR0_ch01.mp4
```

AIST++ API bu projede `external/aistplusplus_api` altına kurulur ve Git'e eklenmez. Büyük video ve annotation dosyaları da Git'e eklenmez.

## Kalibrasyon

```powershell
python scripts\calibrate_cameras.py --session data\session_001\session.yaml
```

Kalibrasyon scripti önce senkron checkerboard tespitleriyle tüm kameraları ortak bir referans kamera koordinat sistemine bağlayan `multiview_common_reference` modunu dener. Bu mod için:

- Her kamerada checkerboard aynı fiziksel anda görünmelidir.
- `session.yaml` içindeki `sync.offsets` değerleri kalibrasyon frame eşlemesine uygulanır.
- `config/calibration_config.yaml` içinde opsiyonel `checkerboard.min_common_frames` ve referans kamera için `extrinsics.world_origin_camera` veya `checkerboard.reference_camera_id` kullanılabilir.

Ortak frame bulunamazsa script `single_camera_fallback` moduna düşer ve rapora açık uyarı yazar. Bu fallback intrinsic üretmek için yararlıdır, ancak kameraların ortak dünya koordinatında metrik 3D verdiğini garanti etmez.

Kalibrasyon çıktıları:

- `outputs/session_001/calibration/cameras.json`
- `outputs/session_001/calibration/calibration_report.json`

## Çok Kameralı 3D Pipeline

```powershell
python scripts\run_multiview_3d.py --session data\session_001\session.yaml --dry-run
python scripts\run_vitpose_multiview_3d.py --session data\session_001\session.yaml
```

ViTPose multi-view pipeline:

- `session.yaml` içindeki kamera `frame_offset` değerlerini global frame zaman çizelgesine uygular.
- Aynı fiziksel ana denk gelen yerel kamera karelerini batch inference ile işler.
- Kalibrasyon yoksa yalnızca test amaçlı yaklaşık iki kamera kalibrasyonuna düşer ve `calibration_mode: approximate_test_calibration` yazar.
- Gerçek üretim çıktısı için `outputs/<session_id>/calibration/cameras.json` dosyasının ilgili kamera ID'leriyle uyumlu olması gerekir.

Beklenen ana çıktılar:

- `outputs/session_001/json/session_3d.json`
- `outputs/session_001/json/preflight_report.json`
- `outputs/session_001/json/video_probe_report.json`
- `outputs/session_001/json/model_runtime_report.json`
- `outputs/session_001/json/quality_summary.json`
- `outputs/session_001/json/artifact_manifest.json`
- `outputs/session_001/csv/keypoints_3d_world_flat.csv`
- `outputs/session_001/session_3d_analysis.xlsx`
- `outputs/session_001/figures/reprojection_error_timeline.png`
- `outputs/session_001/figures/keypoint_validity_heatmap.png`
- `outputs/session_001/figures/camera_usage_heatmap.png`

## Veri Mimarisi

```text
Session
-> CameraView
-> Frame
-> PersonPose2D
-> PersonPose3D
-> TriangulatedPose3D
-> Phase
-> Step
-> Validation
-> Scoring
```

Nihai scoring hiyerarşisi `Episode -> Task -> Phase -> Step -> Metric -> Error -> Score` şeklinde düşünülür.

Bu ilk sürümde gerçek puanlama motoru henüz uygulanmaz. Mevcut kod, scoring motorunun ihtiyaç duyacağı 3D poz, kalite, smoothing, biomekanik açı ve hareket segment adayı verilerini üretmeye odaklanır.

## Güncel Durum

Hazır olanlar:

- Proje iskeleti ve Git ignore kuralları
- `keypoints_3d_world[t, 133, 3]` veri sözleşmesi
- Checkerboard tabanlı multiview ortak referans kalibrasyonu ve single-camera fallback raporlaması
- Kamera frame offset senkronizasyonu
- ViTPose 2D ve RTMW3D adapter sınıfları
- Batch/multi-camera 2D inference arayüzü
- Hartley-style normalize edilmiş multi-view triangulation çekirdeği
- Reprojection error, görüş sayısı ve SVD conditioning kullanan triangulation quality score
- Sentetik 3 kamera dry-run pipeline
- JSON/CSV/Excel/PNG/MP4 output üretimi; CSV export eksik sayıları `NaN` stringi yerine boş hücre yazar
- Preflight raporu: eksik/açılamayan video, eksik kalibrasyon videosu, eksik model config/checkpoint kontrolü
- Video probe raporu: her kamera videosu için açılabilirlik, FPS, çözünürlük, frame count, duration
- Model runtime raporu: ViTPose-Huge WholeBody config/checkpoint hazır mı kontrolü
- AIST++ camera data importer: mapping.txt + setting_*.json dosyalarından gerçek 9 kamera intrinsic/extrinsic üretimi
- ViTPose-Huge gerçek inference ile AIST videolarından 133 eklemli 2D overlay ve kalibrasyonlu multi-view 3D çıktı
- Artifact manifest: her run için beklenen çıktılar, dosya boyutları ve SHA-256 özetleri
- Quality summary: valid frame/joint oranı, triangulation score, reprojection error, kullanılan kamera sayısı
- Yönlü torso lean, smoothing sonrası hız, ağırlıklı center-of-mass proxy, adaptif hareket segmentasyonu
- Triangulation, smoothing, validation, scoring-readiness ve pipeline testleri

Bekleyenler / sıradaki büyük işler:

- Kendi poomsae kameraları için senkron checkerboard calibration videoları ile ortak referans intrinsic/extrinsic üretimi
- Gerçek poomsae videolarında multi-person/person tracking eşlemesi
- Poomsae phase/step detection: hareketleri poomsae adımlarına bölme
- Teknik hata metrikleri: denge, açı, hizalama, yükseklik, zamanlama, simetri ve duruş kararlılığı gibi ölçümler
- Scoring motoru: kural tabanlı ve/veya öğrenmeli puan üretimi
- Hakem/koç tarafından anlaşılabilir rapor: hangi adımda hangi teknik hata var, puan neden kırıldı

## ViTPose Gerçek Video Testi

ViTPose gerçek video inference mevcut `.venv` ortamında çalışır. Ayrıntılı Windows sürüm notu: `docs/vitpose_windows_setup.md`.

```powershell
cd C:\Users\WWWW\Desktop\tk3d
.\.venv\Scripts\Activate.ps1
python scripts\check_models.py --session data\aist_test\session.yaml
python scripts\run_pose2d_overlays.py --session data\aist_test\session.yaml --camera c01 --stride 10
python scripts\run_vitpose_multiview_3d.py --session data\aist_test\session.yaml --stride 10
```

Not: `--max-frames` sadece kısa preview üretmek için kullanılır. Tam video ile aynı süreli çıktı istiyorsan `--max-frames` verme. `--stride` modelin kaç karede bir çalışacağını belirler; çıktı videosunun süresi korunur. Kameralar arası zaman farkları `session.yaml` içindeki `sync.offsets` alanından okunur.

Ana çıktılar:

- `outputs/aist_test/videos/c01_vitpose_2d_overlay.mp4`
- `outputs/aist_test/videos/vitpose_skeleton_3d_world.mp4`
- `outputs/aist_test/json/vitpose_session_3d.json`
- `outputs/aist_test/csv/vitpose_keypoints_3d_world_flat.csv`

Not: AIST++ camera data indirildiğinde `scripts\import_aist_cameras.py` sekansın `mapping.txt` kaydını okuyup `outputs/aist_test/calibration/cameras.json` üretir. Bu dosya varken ViTPose multi-view pipeline `calibration_mode: loaded` ile gerçek AIST++ intrinsic/extrinsic değerlerini kullanır. Kendi poomsae kameraların için senkron checkerboard calibration gerekir.
## SMPL Mesh İnsan Modeli

Çubuk iskelet yerine gerçek insan yüzeyi/mesh görmek için SMPL aşaması kullanılır. AIST++ motion dosyaları indirildi; ancak lisanslı SMPL body model dosyası repoda tutulmaz. Ayrıntılı kurulum: `docs/smpl_mesh_setup.md`.

SMPL model dosyasını koyduktan sonra:

```powershell
cd C:\Users\WWWW\Desktop\tk3d
.\.venv\Scripts\Activate.ps1
python scripts\render_aist_smpl_mesh.py --session data\aist_test\session_all.yaml --smpl-dir models\smpl --gender MALE --max-frames 120 --stride 1
```

Beklenen mesh çıktıları:

- `outputs/aist_test/videos/aist_smpl_mesh.mp4`
- `outputs/aist_test/figures/aist_smpl_mesh_frame0.obj`
- `outputs/aist_test/json/aist_smpl_mesh_report.json`


Mouse ile döndürülebilen oynayan Open3D viewer opsiyoneldir. Python 3.13 ortamında Open3D paketi bulunmayabilir; bu durumda tarayıcıdaki Three.js viewer kullanılmalıdır.

```powershell
python scripts\view_aist_smpl_mesh_open3d.py --session data\aist_test\session_all.yaml --smpl-dir models\smpl --gender MALE --max-frames 240 --stride 1
```


Tarayicida acilan interaktif Three.js viewer uretmek icin:

```powershell
python scripts\export_aist_smpl_threejs_viewer.py --session data\aist_test\session_all.yaml --smpl-dir models\smpl --gender MALE --max-frames 240 --stride 1
```

Cikti: `outputs/aist_test/viewer/aist_smpl_viewer.html`

## Puanlama Hazirlik Analizi

Puanlama motoruna gecmeden once 3D ciktiyi kalite, smoothing, biomekanik acilar ve hareket segment adaylari ile hazirlamak icin:

```powershell
cd C:\Users\WWWW\Desktop\tk3d
.\.venv\Scripts\Activate.ps1
python scripts\analyze_pose_for_scoring.py --session data\aist_test\session_all.yaml --smoothing-window 5
```

Ana ciktilar:

- `outputs/aist_test/json/scoring_readiness_report.json`
- `outputs/aist_test/json/vitpose_session_3d_smoothed.json`
- `outputs/aist_test/csv/pose_quality_frames.csv`
- `outputs/aist_test/csv/pose_quality_joints.csv`
- `outputs/aist_test/csv/biomechanics_timeseries.csv`
- `outputs/aist_test/csv/movement_segments.csv`
- `outputs/aist_test/scoring_readiness_analysis.xlsx`

Bu asama henuz puan vermez; puanlama motorunun kullanacagi guvenilir frame, guvenilir eklem, smoothing, diz/kalca/omuz/dirsek acilari ve hareket segment adaylarini hazirlar. Gercek poomsae videosu geldiginde scoring motoru bu dosyalarin uzerine kurulur.

Tek kamera 2D cubuk overlay gerekiyorsa zaten mevcut komut kullanilir:

```powershell
python scripts\run_pose2d_overlays.py --session data\aist_test\session_all.yaml --camera c01 --stride 1
```
