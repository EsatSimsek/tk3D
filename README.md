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
- 3D temporal smoothing
- 3D validation ve kalite ölçümleri
- JSON, CSV, Excel ve figür export iskeleti
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
```

`--dry-run`, gerçek video ve model olmadan beklenen output yapısını üretir.

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
