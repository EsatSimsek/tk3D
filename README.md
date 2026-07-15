# TK3D

TK3D'nin asıl amacı, tekvando poomsae videolarını teknik olarak analiz edip puanlayabilen bir 3D poomsae scoring sistemi geliştirmektir.

Bu repository şu anda nihai puanlama motoruna giden ara katmanı kurar: çok kameralı poomsae videolarından kalibrasyonlu 3D insan pozu/iskeleti üretmek, bu çıktıyı kalite kontrolünden geçirmek, hareket segmentlerine hazırlamak ve puanlama algoritmasının kullanacağı veri sözleşmesini oluşturmak.

## AI Aracı İçin Hızlı Bağlam

Bu projeyi okuyan bir AI aracı şunu varsaymalıdır:

- Nihai ürün, poomsae performansını otomatik veya yarı otomatik puanlayan bir analiz sistemidir.
- 3D iskelet üretimi projenin asıl amacı değil, puanlama için gerekli ara çıktıdır.
- Ana ara veri sözleşmesi `keypoints_3d_world[t, 133, 3]` formatındaki COCO-WholeBody tabanlı 3D dünya koordinatlarıdır.
- Çalışan zincir: video -> 2D pose -> sağlamlaştırılmış multi-view 3D pose -> kalite analizi -> biomekanik özellikler -> hareket segment adayları -> açıklanabilir geçici teknik skor.
- Geçici skor resmi hakem puanı değildir. Sıradaki alan işi, gerçek poomsae kayıtlarında phase/step etiketleri ve onaylı teknik hedefler oluşturmaktır.
- AIST Dance/AIST++ verisi gerçek poomsae videosu gelmeden kamera, triangulation, ViTPose inference, SMPL mesh ve scoring-readiness akışını test etmek için kullanılıyor.
- MADS Karate/Tai-chi verisi kalibre üç kamera ve motion-capture ground truth ile 3B doğruluk benchmark'ı olarak kullanılıyor; F2 dizisi model uyarlamasından tamamen ayrı testtir.
- Kendi poomsae videoları geldiğinde ortak checkerboard kalibrasyonu yapılmalı; çok kişili çekimlerde kimlik eşleme, poomsae adım etiketleri ve hakem/koç onaylı puan hedefleri eklenmelidir.

Ana ara hedef veri:

```python
keypoints_3d_world[t, 133, 3]
```

Bu ilk sürüm, nihai puanlama sistemine temel olacak şu bileşenleri içerir:

- Checkerboard tabanlı kamera kalibrasyonu için giriş noktası
- ViTPose-Huge 2D wholebody tahmin sınıfı için entegrasyon arayüzü
- Kalibrasyonlu multi-view triangulation
- Kamera FPS'i ile saniye/frame offsetlerini dikkate alan ortak zaman çizelgesi senkronizasyonu
- Görüş aykırılıklarını eleyen sağlam triangulation, pozitif derinlik/açı kontrolleri ve robust reprojection optimizasyonu
- Sentetik 3 kamera dry-run verisi ile triangulation doğrulama
- 3D temporal smoothing
- 3D validation, kalite ölçümleri ve scoring-readiness analizi
- Açıklanabilir, kalite kapılı ve açıkça `provisional_not_official` olarak işaretlenen teknik ön skor
- JSON, CSV, Excel ve figür export iskeleti
- Pytest tabanlı çekirdek algoritma testleri
- Gelecekteki 3D poomsae scoring motoruna uygun veri yapıları

## Sıfırdan Kurulum ve Gerekli İndirmeler

Git deposu bilinçli olarak büyük video, veri seti, model ağırlığı ve lisanslı insan modeli içermez. Aynı testleri
yeniden üretmek için kodu klonladıktan sonra aşağıdaki varlıklar resmî kaynaklarından indirilmelidir.

### Hangi bileşen gerçekten gerekli?

| Bileşen | Ne için gerekli? | Durum |
| --- | --- | --- |
| Python 3.12 ve `requirements.txt` | Sentetik dry-run ve model gerektirmeyen çekirdek işlemler | Zorunlu başlangıç |
| `requirements-pose.txt` içindeki PyTorch ortamı | 73 otomatik testin tamamı ve gerçek inference | Tam doğrulama için zorunlu |
| NVIDIA GPU, ViTPose kaynak kodu ve WholeBody ağırlığı | Gerçek videodan 2B/3B iskelet üretimi | Gerçek inference için zorunlu |
| MADS multi-view | Kalibre üç kamera ve mocap ground-truth ile ana 3B benchmark | Güvenilirlik testi için zorunlu |
| MADS depth | İleride stereo-depth füzyonu ve depth GT incelemesi | Mevcut RGB benchmark için opsiyonel |
| AIST videoları ve AIST++ kamera verisi | Eski dans smoke/regresyon testi | Opsiyonel |
| SMPL model dosyası | Gerçekçi insan mesh'i çizmek | Opsiyonel; iskelet ve puanlama onsuz çalışır |

Test edilen ana Windows ortamı Python `3.12.13`, PyTorch `2.13.0+cu130`, CUDA 13 uyumlu NVIDIA sürücüsü ve RTX
4060 Laptop GPU'dur. Testlerin tamamı için PyTorch paketi gerekir; GPU, ViTPose kaynak ağacı, checkpoint, MADS, AIST
veya SMPL gerekmez. ViTPose-Huge gerçek inference için NVIDIA GPU kuvvetle önerilir; CPU üzerinde çalıştırmak pratik
olmayacak kadar yavaştır. MADS'in tamamı yaklaşık
24 GB, ViTPose checkpoint'i yaklaşık 3,47 GiB olduğundan en az 50 GB boş alan ayırmak güvenlidir.

### 1. Depoyu klonla ve Python ortamını kur

Windows'ta aşağıdaki araçlar kurulu olmalıdır:

- Git for Windows: https://git-scm.com/download/win
- 64-bit Python 3.12: https://www.python.org/downloads/windows/
- MADS split ZIP'leri için güncel 7-Zip: https://www.7-zip.org/
- Gerçek inference yapılacaksa güncel NVIDIA sürücüsü: https://www.nvidia.com/Download/index.aspx
- Yalnız AIST API'nin bazı görselleştirmeleri için opsiyonel FFmpeg: https://ffmpeg.org/download.html

```powershell
git clone https://github.com/EsatSimsek/tk3D.git
cd .\tk3D

py -3.12 -m venv .venv312
.\.venv312\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

PowerShell sanal ortam aktivasyonunu engellerse yalnız açık terminal için:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv312\Scripts\Activate.ps1
```

Gerçek ViTPose inference için NVIDIA/CUDA paketlerini ayrıca kur:

```powershell
python -m pip install -r requirements-pose.txt
```

`requirements-pose.txt` bu projede test edilen CUDA 13.0 PyTorch wheel'ini kullanır. Farklı CUDA/PyTorch ortamında
paket sürümlerini körlemesine değiştirmek yerine önce PyTorch'un resmî kurulum seçicisine göre uyumlu wheel kurulmalı,
ardından `requirements.txt`, `timm` ve `torchvision` tamamlanmalıdır: https://pytorch.org/get-started/locally/

### 2. Resmî ViTPose kaynak kodunu kur

TK3D, MMPose kurulumu yerine resmî ViTPose deposundan sınırlı bir yerel runtime kullanır. Yeniden üretilebilirlik için
bu projede test edilen commit sabitlenmiştir:

```powershell
New-Item -ItemType Directory -Force external | Out-Null
git clone https://github.com/ViTAE-Transformer/ViTPose.git external\vitpose
git -C external\vitpose checkout c050ed29112da7704797cc1a65af0234b525010d
```

Resmî depo ve lisans: https://github.com/ViTAE-Transformer/ViTPose

Beklenen klasör:

```text
external/vitpose/mmpose/
external/vitpose/configs/
```

### 3. ViTPose-Huge WholeBody ağırlığını indir ve doğrula

COCO-only 17 eklemli ViTPose-H ağırlığını kullanmayın. Gerekli dosya, resmî ViTPose WholeBody tablosundaki
`ViTPose++-H COCO+AIC+MPII+AP10K+APT36K+WholeBody 256x192` checkpoint'idir.

1. Resmî bağlantıyı tarayıcıda açın:
   https://1drv.ms/u/s!AimBgYV7JjTlgccoXv8rCUgVe7oD9Q?e=ZBw6gR
2. İndirilen gerçek `.pth` dosyasını aşağıdaki ada taşıyın:

```text
weights/vitpose_huge_wholebody_256x192.pth
```

3. Boyut ve SHA-256 değerini doğrulayın:

```powershell
New-Item -ItemType Directory -Force weights | Out-Null
Get-Item weights\vitpose_huge_wholebody_256x192.pth | Select-Object Name,Length
Get-FileHash weights\vitpose_huge_wholebody_256x192.pth -Algorithm SHA256
```

Beklenen değerler:

```text
Boyut  : 3723960207 byte
SHA-256: A714AE5F0B45F7A3F1A86624CF7382913454EE1D61A4AE5F06C40573D5B6A459
```

OneDrive birkaç KB/MB boyutunda HTML dosyası kaydettiyse bu checkpoint değildir; silip tarayıcıdan yeniden indirin.
`check_models.py` HTML veya uyumsuz model dosyasını reddeder.

### 4. MADS veri setini indir, çıkar ve hazırla

MADS ana güvenilirlik benchmark'ıdır. Proje veri setini yeniden dağıtmaz; resmî sayfadan indirin:

- https://visal.cs.cityu.edu.hk/downloads/
- Sayfada `Human Pose Datasets -> MADS -> download here`
- Doğrudan MADS Google Drive klasörü: https://drive.google.com/drive/folders/0B0AquUC4V8cFU2otR3l3WWRUVVk?resourcekey=0-KC-rxBAHiIIpylFRCTESNQ

Aynı ortamı tamamen yeniden üretmek için `MADS_multiview` ve `MADS_depth` arşivlerini indirin. Yalnız mevcut F2 RGB
benchmark'ını çalıştırmak için multi-view parçaları yeterlidir. Split ZIP'in tüm `.z01`, `.z02`, ... ve `.zip`
parçaları aynı klasörde olmalıdır. Güncel 7-Zip ile ana `.zip` dosyasını açıp örneğin `C:\datasets\MADS` altına
çıkarın.

7-Zip varsayılan konumdaysa komut satırından çıkarma örneği:

```powershell
New-Item -ItemType Directory -Force C:\datasets\MADS\MADS_multiview | Out-Null
& "C:\Program Files\7-Zip\7z.exe" x `
  C:\Downloads\MADS\MADS_multiview.zip `
  "-oC:\datasets\MADS\MADS_multiview"

# Depth arşivi indirildiyse:
New-Item -ItemType Directory -Force C:\datasets\MADS\MADS_depth | Out-Null
& "C:\Program Files\7-Zip\7z.exe" x `
  C:\Downloads\MADS\MADS_depth.zip `
  "-oC:\datasets\MADS\MADS_depth"
```

Arşiv yolu farklıysa yalnız `C:\Downloads\MADS` bölümünü değiştirin. Çıkarma bittikten sonra aşağıdaki kesin klasör
düzenini kontrol etmeden kurulum scriptini çalıştırmayın.

TK3D'nin beklediği kesin klasör düzeni:

```text
C:\datasets\MADS\
├── MADS_multiview\MADS\multi_view_data\
│   ├── Kata\Kata_F2_C0.avi
│   ├── Kata\Kata_F2_GT.mat
│   ├── Kata\Calib_Cam0.mat
│   └── Taichi\...
└── MADS_depth\MADS\depth_data\       # opsiyonel
```

Veriyi indeksle, resmî kamera kalibrasyonlarını içe aktar, metric ground-truth JSON'larını ve önizlemeleri üret:

```powershell
python scripts\setup_mads_test.py `
  --dataset-root C:\datasets\MADS `
  --actions Kata `
  --hash-files `
  --preview
```

Başarılı kurulumdan sonra en az şu dosyalar bulunmalıdır:

```text
data/mads_test/local/sessions/mads_kata_f2.yaml
data/mads_test/local/ground_truth/multiview/Kata_F2.json
outputs/mads_kata_f2/calibration/cameras.json
outputs/mads_setup/mads_setup_report.json
```

`data/mads_test/local/` makineye özgü mutlak yollar içerir ve Git'e eklenmez. Ayrıntılı MADS açıklaması:
`docs/mads_ground_truth_setup.md`.

### 5. AIST dans smoke testi için opsiyonel indirme

AIST, MADS'in yerine geçen doğruluk benchmark'ı değildir. Yalnız eski çok-kamera akışını tekrar test etmek isteyenler
için gerekir. Önce AIST kullanım koşullarını okuyun: https://aistdancedb.ongaaccel.jp/terms_of_use/

İki kameralı küçük örnek için klasörleri hazırla ve resmî AIST video URL'lerinden yalnız gerekli iki videoyu indir:

```powershell
python scripts\setup_aist_test.py `
  --sequence gBR_sBM_cAll_d04_mBR0_ch01 `
  --cameras c01 c02

$aistBase = "https://aistdancedb.ongaaccel.jp/video_raw/10M"
Invoke-WebRequest `
  "$aistBase/gBR_sBM_c01_d04_mBR0_ch01.mp4" `
  -OutFile data\aist_test\videos\gBR_sBM_c01_d04_mBR0_ch01.mp4
Invoke-WebRequest `
  "$aistBase/gBR_sBM_c02_d04_mBR0_ch01.mp4" `
  -OutFile data\aist_test\videos\gBR_sBM_c02_d04_mBR0_ch01.mp4
```

AIST++ kamera kalibrasyonunu resmî GitHub release'inden indir ve çıkar:

```powershell
Invoke-WebRequest `
  "https://github.com/google/aistplusplus_dataset/releases/download/v1.0/cameras.zip" `
  -OutFile data\aist_test\cameras.zip
Expand-Archive `
  -Path data\aist_test\cameras.zip `
  -DestinationPath data\aist_test\annotations `
  -Force

python scripts\import_aist_cameras.py --session data\aist_test\session.yaml
```

Beklenen kamera dosyası `data/aist_test/annotations/cameras/mapping.txt` konumundadır. AIST++ motion, 2B/3B
annotation veya API kodu yalnız bu ek özellikler kullanılacaksa gerekir. Resmî annotation indirmeleri:
https://google.github.io/aistplusplus_dataset/download.html

Opsiyonel API kodunu bu projede test edilen commit ile kurmak için:

```powershell
git clone https://github.com/google/aistplusplus_api.git external\aistplusplus_api
git -C external\aistplusplus_api checkout 2dd7b3e946b794fd0081c98e2e2433545abf8b87
```

### 6. SMPL insan modeli yalnız mesh için opsiyoneldir

SMPL dosyası 3B iskelet, MADS ground-truth ölçümü veya puanlama altyapısı için gerekli değildir. Yalnız gerçekçi insan
yüzeyi/mesh render etmek isteyen kullanıcı kurmalıdır. Standart SMPL modeli lisans nedeniyle bu repo tarafından
indirilemez veya üçüncü kişilere dağıtılamaz.

1. https://smpl.is.tue.mpg.de/ adresinde hesap açın.
2. Lisansı okuyup kendi hesabınızla modeli indirin.
3. İndirdiğiniz dosyaları şu adlarla yerleştirin:

```text
models/smpl/SMPL_MALE.pkl
models/smpl/SMPL_FEMALE.pkl   # opsiyonel
```

4. Mesh paketlerini kurun:

```powershell
python -m pip install -r requirements-smpl.txt
```

Open3D kurulamazsa tarayıcı tabanlı Three.js export için yalnız temel mesh paketleri yeterlidir:

```powershell
python -m pip install smplx trimesh chumpy --no-build-isolation
```

Lisans ve mesh komutları: `docs/smpl_mesh_setup.md`.

### 7. Kurulumu doğrula

Önce hiçbir model veya video gerektirmeyen core testleri ve sentetik üç kamera dry-run'ını çalıştırın:

```powershell
python -m pytest -q
python scripts\run_multiview_3d.py `
  --session data\session_001\session.yaml `
  --dry-run
```

Codex/sandbox ortamında pytest temp izni sorun çıkarırsa:

```powershell
python -m pytest -q -p no:cacheprovider --basetemp outputs\pytest-tmp
```

Beklenen test sonucu:

```text
73 passed
```

`--dry-run`, sentetik dünya koordinatlarını üç kameraya projekte edip gerçek triangulation kodundan geçirir. Çıktısı:

```text
outputs/session_001/runs/<run_id>/videos/skeleton_3d_world.mp4
```

ViTPose ve MADS kurulduktan sonra gerçek varlık kontrolü:

```powershell
python scripts\check_models.py `
  --session data\mads_test\local\sessions\mads_kata_f2.yaml `
  --model-config config\model_config.yaml
```

Beklenen satır:

```text
pose2d: ready - ready
```

Tam 300 örnekli MADS benchmark ve geçici puanlama komutları aşağıdaki `Puanlama Altyapısı: Güvenli Geliştirme
Akışı` bölümündedir. Gerçek kullanıcı videosunda önce strict preflight çalıştırılmalıdır:

```powershell
python scripts\preflight_session.py `
  --session data\session_001\session.yaml `
  --require-videos `
  --require-calibration-videos `
  --require-model-files
```

### 8. Git'e konmayan yerel varlıklar

Aşağıdaki dosyalar `.gitignore` ile yerelde tutulur; `git push` bunları GitHub'a yüklemez veya bilgisayardan silmez:

```text
weights/*.pth
external/vitpose/
external/aistplusplus_api/
data/mads_test/local/
data/aist_test/videos/*.mp4
data/aist_test/annotations/
models/smpl/*.pkl
outputs/**
```

## Ground-truth 3B Doğrulama

Poomsae'ye yakın hareket alanı ve optik motion-capture referansı nedeniyle birincil dış doğrulama veri seti olarak
MADS (Martial Arts, Dancing and Sports) seçildi. AIST mevcut çok-kamera smoke testleri için korunur; gerçek 3B hata
ölçümünün ana benchmark'ı değildir.

Ground-truth karşılaştırma katmanı global/pelvis-relative/PA-MPJPE, PCK-3D, açı, hız, ivme ve kemik kararlılığı
raporlarını üretir. Girdi koordinatı ve birimi açıkça doğrulanmadan çalışmaz. Kurulum, veri sözleşmesi ve resmî indirme
durumu: `docs/mads_ground_truth_setup.md`.

Yerel MADS arşivini (çoklu-görüş ve depth) indeksleyip Kata oturumlarını hazırlamak:

```powershell
python scripts\setup_mads_test.py --dataset-root C:\Users\WWWW\Desktop\MADS --actions Kata --hash-files --preview
```

Kurulum 30 çoklu-görüş ve 30 depth diziyi indeksler; seçilen diziler için resmî kalibrasyon, yerel session ve metre
cinsinden 3B ground-truth üretir. Makineye özgü yolları içeren `data/mads_test/local/` Git'e eklenmez.

```powershell
python scripts\evaluate_ground_truth_3d.py --prediction <tk3d_3d.json> --ground-truth <metric_gt.json> --output-dir outputs\ground_truth_validation
```

Komut, ground-truth kalite kapısı başarısızsa CI/otomasyonun bunu başarı sanmaması için sıfırdan farklı kodla çıkar.
Yalnız tanısal başarısız raporu bilinçli biçimde kabul etmek için `--allow-failed-quality-gate` kullanılabilir.

MADS domain-adaptation altyapısı da eklidir. Donmuş ViTPose omurga özelliklerinden 2B heatmap head eğitimi ve robust
eklem offset kalibrasyonu yapılabilir; ancak üretilen adapter varsayılan olarak onaysızdır. Normal çalışma, ayrı 3B
testte onaylanmamış adapter'ı reddeder. Mevcut deneylerde 2B doğrulama kaybı iyileşmesine rağmen hiç görülmemiş F2
3B testi kötüleştiği için MADS adapter üretime alınmadı; kullanılan model hâlâ daha iyi sonuç veren temel
ViTPose-Huge modelidir. Ayrıntılar ve sayısal sonuçlar: `docs/mads_ground_truth_setup.md`.

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

Kalibrasyon scripti senkron checkerboard tespitleriyle tüm kameraları ortak bir referans kamera koordinat sistemine bağlayan `multiview_common_reference` modunu üretir. Bu mod için:

- Her kamerada checkerboard aynı fiziksel anda görünmelidir.
- `session.yaml` içindeki `sync.offsets` değerleri kalibrasyon frame eşlemesine uygulanır.
- `config/calibration_config.yaml` içinde opsiyonel `checkerboard.min_common_frames` ve referans kamera için `extrinsics.world_origin_camera` veya `checkerboard.reference_camera_id` kullanılabilir.

Ortak kare bulunamazsa komut güvenli biçimde durur ve üretim `cameras.json` dosyası yazmaz. Yalnızca intrinsic teşhisi gerekiyorsa açık `--allow-intrinsics-only-fallback` seçeneği `intrinsics_only.json` üretir; canlı 3B akış bu dosyayı kabul etmez.

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
- Aynı fiziksel ana denk gelen yerel kamera karelerini işler ve her kameradaki kişi kutusunu zaman içinde takip eder.
- Üretim kalibrasyonu yoksa veya ortak dünya extrinsic bilgisi doğrulanmamışsa güvenli biçimde durur. Yaklaşık iki-kamera kalibrasyonu yalnızca açık `--allow-approximate-calibration` seçeneğiyle diagnostik preview için kullanılabilir.
- Gerçek üretim çıktısı için `outputs/<session_id>/calibration/cameras.json` dosyasının ilgili kamera ID'leriyle uyumlu olması gerekir.
- Her canlı çalışma `outputs/<session_id>/runs/<run_id>/` altında izole edilir; yalnızca kalite kapısını geçen çalışma `latest_run.json` olarak işaretlenir.
- Canlı çalışmanın kalite raporu yalnız iç geometrik kaliteyi ölçer ve `scoring_ready=false` yazar; ground-truth
  doğruluk raporu ayrıca geçmeden çıktı puanlama için güvenilir kabul edilmez.

Sentetik dry-run çıktıları `outputs/session_001/runs/<run_id>/` altında, canlı ViTPose çıktıları da aynı izole çalışma yapısında tutulur. Beklenen dry-run dosyaları:

- `outputs/session_001/runs/<run_id>/json/session_3d.json`
- `outputs/session_001/runs/<run_id>/json/preflight_report.json`
- `outputs/session_001/runs/<run_id>/json/video_probe_report.json`
- `outputs/session_001/runs/<run_id>/json/model_runtime_report.json`
- `outputs/session_001/runs/<run_id>/json/quality_summary.json`
- `outputs/session_001/runs/<run_id>/json/artifact_manifest.json`
- `outputs/session_001/runs/<run_id>/csv/keypoints_3d_world_flat.csv`
- `outputs/session_001/runs/<run_id>/session_3d_analysis.xlsx`
- `outputs/session_001/runs/<run_id>/figures/reprojection_error_timeline.png`
- `outputs/session_001/runs/<run_id>/figures/keypoint_validity_heatmap.png`
- `outputs/session_001/runs/<run_id>/figures/camera_usage_heatmap.png`

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

Kod, 3B poz ve kalite kapılarının üzerine açıklanabilir bir teknik ön skor üretir. Bu skor yalnızca altyapı/doğrulama içindir; onaylı poomsae adım şablonu ve hakem kriteri olmadan resmi puan olarak kullanılamaz.

## Güncel Durum

Hazır olanlar:

- Proje iskeleti ve Git ignore kuralları
- `keypoints_3d_world[t, 133, 3]` veri sözleşmesi
- Checkerboard tabanlı multiview ortak referans kalibrasyonu ve üretimde fail-closed davranış
- Farklı FPS, frame offset ve saniye offsetlerini destekleyen timestamp senkronizasyonu
- ViTPose-Huge WholeBody 2D runtime; opsiyonel RTMW3D yardımcı adapter'ı varsayılan olarak kapalı
- Batch/multi-camera 2D inference arayüzü
- Aykırı kamerayı eleyen, pozitif derinlik ve triangulation açısı kontrolü yapan robust multi-view triangulation
- Robust reprojection optimizasyonu; reprojection error ve inlier kamera sayısına dayalı kalite skoru
- Sentetik 3 kamera dry-run pipeline
- JSON/CSV/Excel/PNG/MP4 output üretimi; CSV export eksik sayıları `NaN` stringi yerine boş hücre yazar
- Preflight raporu: eksik/açılamayan video, eksik kalibrasyon videosu, eksik model config/checkpoint kontrolü
- Video probe raporu: her kamera videosu için açılabilirlik, FPS, çözünürlük, frame count, duration
- Model runtime raporu: ViTPose-Huge WholeBody config/checkpoint hazır mı kontrolü
- AIST++ camera data importer: mapping.txt + setting_*.json dosyalarından gerçek 9 kamera intrinsic/extrinsic üretimi
- ViTPose-Huge gerçek inference ile AIST videolarından 133 eklemli 2D overlay ve kalibrasyonlu multi-view 3D çıktı
- MADS multi-view/depth indeksleme, resmî üç kamera kalibrasyonu ve metre cinsinden mocap ground-truth dönüşümü
- F2'yi eğitimden ayıran domain-adaptation altyapısı ve onaysız adapter'ı üretimde reddeden güvenlik kilidi
- MADS F2 üzerinde 300 örnekli ölçülmüş `82,5 mm` MPJPE benchmark; `50 mm` hedef geçilmediği için resmî skor kapalı
- Artifact manifest: her run için beklenen çıktılar, dosya boyutları ve SHA-256 özetleri
- Quality summary: valid frame/joint oranı, triangulation score, reprojection error, kullanılan kamera sayısı
- Yönlü torso lean, smoothing sonrası hız, ağırlıklı center-of-mass proxy, adaptif hareket segmentasyonu
- Kalite kapılı geçici frame/step skoru ve puan kırılma nedenlerini listeleyen teknik hata raporu
- Triangulation, smoothing, validation, scoring-readiness ve pipeline testleri

Bekleyenler / sıradaki büyük işler:

- Kendi poomsae kameraları için senkron checkerboard calibration videoları ile ortak referans intrinsic/extrinsic üretimi
- Gerçek poomsae videolarında multi-person/person tracking eşlemesi
- Poomsae phase/step detection: hareketleri poomsae adımlarına bölme
- Gerçek poomsae için phase/step adlarını ve başlangıç-bitiş karelerini onaylayacak etiket veri seti
- Denge, açı, hizalama, yükseklik, zamanlama ve simetri hedeflerinin hakem/koç tarafından onaylanması
- Geçici teknik ön skoru resmi kurallara bağlayan sürümlü referans şablonları ve uzman doğrulaması

## ViTPose Gerçek Video Testi

ViTPose gerçek video inference `.venv312` ortamında çalışır. Ayrıntılı Windows sürüm notu: `docs/vitpose_windows_setup.md`.

```powershell
cd C:\Users\WWWW\Desktop\tk3d
.\.venv312\Scripts\Activate.ps1
python scripts\check_models.py --session data\aist_test\session.yaml
python scripts\run_pose2d_overlays.py --session data\aist_test\session.yaml --camera c01 --stride 10
python scripts\run_vitpose_multiview_3d.py --session data\aist_test\session.yaml --stride 10
```

Not: `--max-frames` sadece kısa preview üretmek için kullanılır. Tam video ile aynı süreli çıktı istiyorsan `--max-frames` verme. `--stride` modelin kaç karede bir çalışacağını belirler; çıktı videosunun süresi korunur. Kameralar arası zaman farkları `session.yaml` içindeki `sync.offsets` alanından okunur.

Ana çıktılar:

- `outputs/aist_test/runs/<run_id>/videos/c01_vitpose_2d_overlay.mp4`
- `outputs/aist_test/runs/<run_id>/videos/vitpose_skeleton_3d_world.mp4`
- `outputs/aist_test/runs/<run_id>/json/vitpose_session_3d.json`
- `outputs/aist_test/runs/<run_id>/csv/vitpose_keypoints_3d_world_flat.csv`

Not: AIST++ camera data indirildiğinde `scripts\import_aist_cameras.py` sekansın `mapping.txt` kaydını okuyup `outputs/aist_test/calibration/cameras.json` üretir. Bu dosya `aist_official_multiview` olarak işaretlenir ve gerçek AIST++ intrinsic/extrinsic değerlerini kullanır. Kendi poomsae kameraların için senkron checkerboard calibration gerekir.
## SMPL Mesh İnsan Modeli

Çubuk iskelet yerine gerçek insan yüzeyi/mesh görmek için SMPL aşaması kullanılır. Bunun için AIST++ motion annotation
dosyası ve kullanıcının kendi hesabıyla indirdiği lisanslı SMPL body model dosyası gerekir; ikisi de repoda tutulmaz.
Ayrıntılı kurulum: `docs/smpl_mesh_setup.md`.

SMPL model dosyasını koyduktan sonra:

```powershell
cd C:\Users\WWWW\Desktop\tk3d
.\.venv312\Scripts\Activate.ps1
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

## Puanlama Altyapısı: Güvenli Geliştirme Akışı

Puanlama altyapısı geliştirilebilir durumdadır; fakat mevcut MADS F2 sonucu `82,5 mm` MPJPE olduğu ve `50 mm`
ground-truth hedefini geçmediği için resmî puanlama kapalıdır. Sistem yalnız kaliteyi geçen karelerden açıklanabilir
teknik özellik ve `provisional_not_official` durumlu geliştirme skoru üretir. Geçersiz eklem veya yetersiz kamera
görüşü puana katılmaz.

### MADS F2 üzerinde baştan sona çalıştırma

Aşağıdaki PowerShell bloğu tek seferde kopyalanabilir. Her çalıştırmada benzersiz bir çalışma kimliği üretildiği için
eski sonuçların üzerine yazılmaz:

```powershell
cd C:\Users\WWWW\Desktop\tk3d
.\.venv312\Scripts\Activate.ps1

$runId = "mads-kata-f2-scoring-$(Get-Date -Format yyyyMMdd-HHmmss)"
$runRoot = "outputs\mads_kata_f2\runs\$runId"

python scripts\check_models.py `
  --session data\mads_test\local\sessions\mads_kata_f2.yaml `
  --model-config config\model_config.yaml

python scripts\run_vitpose_multiview_3d.py `
  --session data\mads_test\local\sessions\mads_kata_f2.yaml `
  --model-config config\model_config.yaml `
  --stride 2 `
  --max-frames 300 `
  --run-id $runId `
  --allow-low-quality-output

python scripts\evaluate_ground_truth_3d.py `
  --prediction "$runRoot\json\vitpose_session_3d.json" `
  --ground-truth data\mads_test\local\ground_truth\multiview\Kata_F2.json `
  --output-dir "$runRoot\ground_truth_validation" `
  --allow-failed-quality-gate

python scripts\analyze_pose_for_scoring.py `
  --session data\mads_test\local\sessions\mads_kata_f2.yaml `
  --input-json "$runRoot\json\vitpose_session_3d.json" `
  --smoothing-window 5 `
  --allow-unvalidated-provisional-score

Write-Host "Sonuç klasörü: $runRoot"
```

`--allow-low-quality-output`, `--allow-failed-quality-gate` ve `--allow-unvalidated-provisional-score` yalnız
geliştirme/teşhis içindir. Kalite kapılarını geçmiş gibi göstermezler. Bu izinler olmadan doğrulanmamış çalışma
puanlanmaz. Hızlı bir smoke test için yalnız 3B üretim komutunda `--stride 20 --max-frames 30` kullanılabilir; 30
kare güvenilirlik kararı veya model onayı için yeterli değildir.

### Çıktılar nerede?

Bütün sonuçlar `outputs/mads_kata_f2/runs/<run_id>/` altında aynı çalışmaya ait olacak şekilde tutulur.

| Öncelik | Dosya | Ne gösterir? |
| --- | --- | --- |
| 1 | `scoring_readiness_analysis.xlsx` | Kalite, biomekanik, segment, kare/adım skoru ve teknik hataları tek dosyada gösterir. |
| 2 | `ground_truth_validation/ground_truth_validation_report.json` | MPJPE, P95, PCK, açı hatası ve ground-truth kalite kapısı sonucunu gösterir. |
| 3 | `json/scoring_readiness_report.json` | Kaç kare ve eklemin değerlendirmeye uygun olduğunu gösterir. |
| 4 | `json/provisional_scoring_report.json` | Geçici toplam/bileşen skorları ve `provisional_not_official` durumunu gösterir. |
| 5 | `videos/vitpose_skeleton_3d_world.mp4` | Üretilen 3B iskeleti görsel olarak kontrol etmeyi sağlar. |

Diğer ayrıntılı çıktılar:

- `json/vitpose_session_3d.json`: filtrelenmiş ham 3B eklemler, güvenler ve kullanılan kamera sayıları
- `json/vitpose_session_3d_smoothed.json`: puanlama analizinde kullanılan yumuşatılmış 3B iskelet
- `json/run_quality_report.json`: kalibrasyon/reprojection/geçerlilik temelli iç geometri kontrolü
- `csv/pose_quality_frames.csv`: kare bazında kalite ve puanlamaya kabul durumu
- `csv/pose_quality_joints.csv`: eklem bazında kalite özeti
- `csv/biomechanics_timeseries.csv`: açı, hız, denge ve diğer biomekanik zaman serileri
- `csv/movement_segments.csv`: otomatik hareket segment adayları
- `csv/provisional_frame_scores.csv`: kare bazında geçici skorlar
- `csv/provisional_step_scores.csv`: segment/adım bazında geçici skorlar
- `csv/technical_errors.csv`: açıklanabilir teknik hata kodları ve açıklamaları
- `ground_truth_validation/ground_truth_frame_errors.csv`: kare bazında gerçek 3B hata
- `ground_truth_validation/ground_truth_joint_errors.csv`: eklem bazında gerçek 3B hata

`provisional_scoring_report.json`, kalite kapısını geçen kareler için duruş, alt vücut, kinematik denge ve kemik
uzunluğu kararlılığı bileşenlerinden 0-100 arası bir altyapı skoru verir. Hareket segmentleri gerçek poomsae
adımlarıyla etiketlenmeden, teknik hedefler uzman tarafından onaylanmadan ve hakem puanlarıyla dış doğrulama
yapılmadan bu değer resmî puan değildir.

### AIST üzerinde eski geliştirme akışı

AIST yalnız akış/smoke testi için korunur; MADS ground-truth doğruluk benchmark'ının yerini almaz:

```powershell
python scripts\analyze_pose_for_scoring.py `
  --session data\aist_test\session_all.yaml `
  --smoothing-window 5 `
  --allow-unvalidated-provisional-score
```

Tek kamera 2D cubuk overlay gerekiyorsa zaten mevcut komut kullanilir:

```powershell
python scripts\run_pose2d_overlays.py --session data\aist_test\session_all.yaml --camera c01 --stride 1
```
