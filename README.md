# TK3D

TK3D, tekvando poomsae videolarından kalibrasyonlu çok kameralı 3D iskelet üretmek için hazırlanmış Python proje iskeletidir.

Ana hedef veri:

```python
keypoints_3d_world[t, 133, 3]
```

Bu ilk sürüm şu bileşenleri içerir:

- Checkerboard tabanlı kamera kalibrasyonu için giriş noktası
- RTMW-x-l 2D wholebody tahmin sınıfı için entegrasyon arayüzü
- RTMW3D-x single-view 3D yardımcı tahmin sınıfı için entegrasyon arayüzü
- Kalibrasyonlu multi-view triangulation
- Sentetik 3 kamera dry-run verisi ile triangulation doğrulama
- 3D temporal smoothing
- 3D validation ve kalite ölçümleri
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

RTMW-x-l ve RTMW3D-x entegrasyonu için MMPose/MMPRETRAIN ortamı ayrıca kurulmalıdır. Bu repository temel pipeline ve veri sözleşmesini hazırlar; model ağırlıkları `models/` veya `weights/` altında tutulur ve Git'e eklenmez.

## İlk kontrol

```powershell
python scripts\inspect_session.py --session data\session_001\session.yaml
python scripts\preflight_session.py --session data\session_001\session.yaml
python scripts\probe_videos.py --session data\session_001\session.yaml
python scripts\check_models.py --session data\session_001\session.yaml
python scripts\run_multiview_3d.py --session data\session_001\session.yaml --dry-run
python -m pytest -q
```

`--dry-run`, gerçek video ve model olmadan sentetik dünya koordinatları üretir, bunları 3 kamera projection matrix ile 2D'ye projekte eder ve gerçek multi-view triangulation kodundan geçirerek beklenen output yapısını üretir.
Bu komut gerçek bir `outputs/session_001/videos/skeleton_3d_world.mp4` dosyası üretir. Video, sentetik 3D dünya iskeletinin frame frame render edilmiş halidir; siyah placeholder değildir.

Gerçek video/model dosyaları geldiğinde önce strict preflight çalıştırılır:

```powershell
python scripts\preflight_session.py --session data\session_001\session.yaml --require-videos --require-calibration-videos --require-model-files
```

## AIST Video Testi

AIST Dance Video DB videoları, poomsae videosu gelmeden çok kameralı görüntü akışını test etmek için kullanılabilir. AIST++ annotation dosyaları COCO 17 eklem formatındadır; bu nihai COCO-WholeBody 133 hedefini değiştirmez. Videolar bizim RTMW-x-l adapter yoluna girdiğinde hedef yine 133 eklemdir. AIST++ 17 eklem verisi sadece calibration, projection, triangulation ve hata ölçümü için opsiyonel doğrulama verisidir.

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

Kalibrasyon çıktıları:

- `outputs/session_001/calibration/cameras.json`
- `outputs/session_001/calibration/calibration_report.json`

## Çok Kameralı 3D Pipeline

```powershell
python scripts\run_multiview_3d.py --session data\session_001\session.yaml
```

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

Puanlama motoru bu ilk sürümde uygulanmaz. Veri yapıları `Episode -> Task -> Phase -> Step -> Metric -> Error -> Score` hiyerarşisine hazır olacak şekilde tanımlanmıştır.

## Güncel Durum

Hazır olanlar:

- Proje iskeleti ve Git ignore kuralları
- `keypoints_3d_world[t, 133, 3]` veri sözleşmesi
- Kamera kalibrasyonu için checkerboard tabanlı script
- 2D/3D model adapter sınıfları
- Multi-view triangulation çekirdeği
- Sentetik 3 kamera dry-run pipeline
- JSON/CSV/Excel/PNG/MP4 output üretimi
- Preflight raporu: eksik video, eksik kalibrasyon videosu, eksik model config/checkpoint kontrolü
- Video probe raporu: her kamera videosu için açılabilirlik, FPS, çözünürlük, frame count, duration
- Model runtime raporu: RTMW-x-l / RTMW3D-x config, checkpoint ve MMPose ortam hazır mı
- Artifact manifest: her run için beklenen çıktılar, dosya boyutları ve SHA-256 özetleri
- Quality summary: valid frame/joint oranı, triangulation score, reprojection error, kullanılan kamera sayısı
- Triangulation, smoothing, validation ve pipeline testleri

Bekleyenler:

- RTMW-x-l gerçek MMPose inference bağlantısı
- RTMW3D-x gerçek inference bağlantısı
- Gerçek checkerboard calibration videoları ile intrinsic/extrinsic üretimi
- Gerçek poomsae videolarında multi-person/person tracking eşlemesi
- Phase/step detection ve scoring motoru
