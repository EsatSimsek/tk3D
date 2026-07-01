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
python scripts\run_multiview_3d.py --session data\session_001\session.yaml --dry-run
python -m pytest -q
```

`--dry-run`, gerçek video ve model olmadan sentetik dünya koordinatları üretir, bunları 3 kamera projection matrix ile 2D'ye projekte eder ve gerçek multi-view triangulation kodundan geçirerek beklenen output yapısını üretir.

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
- Triangulation, smoothing, validation ve pipeline testleri

Bekleyenler:

- RTMW-x-l gerçek MMPose inference bağlantısı
- RTMW3D-x gerçek inference bağlantısı
- Gerçek checkerboard calibration videoları ile intrinsic/extrinsic üretimi
- Gerçek poomsae videolarında multi-person/person tracking eşlemesi
- Phase/step detection ve scoring motoru
