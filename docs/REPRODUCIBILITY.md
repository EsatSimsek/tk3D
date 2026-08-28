# TK3D Reproducibility Contract

Bu belge clean checkout ile yerel araştırma makinesi arasındaki sınırı tanımlar.
Repository'nin klonlanması tek başına model, ZED kaydı, üretim kalibrasyonu veya
doğrulanmış pose artifact'i sağlamaz.

## Doğrulanmış ortam

Yerel `CURRENT_ACTIVE` araştırma koşusu şu ortamda doğrulandı:

| Bileşen | Doğrulanmış değer |
| --- | --- |
| İşletim sistemi | Windows 11 `10.0.26200`, AMD64 |
| Python | CPython `3.12.13` |
| NumPy | `2.5.1` |
| SciPy | `1.18.0` |
| OpenCV | `5.0.0.93` |
| PyTorch | `2.13.0+cu130` |
| TorchVision | `0.28.0+cu130` |
| CUDA runtime | `13.0` |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| RF-DETR | `1.7.0` |
| Supervision | `0.27.0` |
| timm | `1.0.28` |

Paket metadata'sı Python `>=3.11` ister. CI, Windows üzerinde Python 3.11 ve
CPU Torch hedefler; yerel araştırma doğrulaması Python 3.12 ile yapılmıştır.
Python 3.11 üzerinde gerçek CUDA/ZED inference doğrulanmış sayılmaz.

## Dependency kaynakları

- `pyproject.toml`: kurulabilir TK3D paketinin minimum core, `dev`, `pose` ve
  `smpl` dependency sözleşmesidir.
- `requirements.txt`: Windows Tier-1 geliştirme/test bileşimidir; yerel
  `xtcocotools_compat` paketini ve test araçlarını içerir. Testlerin Torch
  kullanan kısmı için CI ayrıca CPU Torch kurar.
- `requirements-pose.txt`: doğrulanmış CUDA 13 araştırma bileşimidir ve
  `torch==2.13.0+cu130` / `torchvision==0.28.0+cu130` seçimini sabitler.
- `requirements-smpl.txt`: yalnız opsiyonel mesh/görselleştirme katmanıdır.

Bu dosyalar farklı amaç taşır; `requirements-pose.txt` normal CI gereksinimi
değildir.

## Tier 1 — clean checkout / CI

Tier 1 GPU, model checkpoint, gerçek video, SVO, historical `outputs/` veya
üretim kalibrasyonu gerektirmez. Windows PowerShell clean-install yolu:

```powershell
py -3.12 -m venv .venv312
.\.venv312\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e . --no-deps --no-build-isolation
tk3d-check
python -m ruff check src scripts tests
python -m pytest -q -p no:cacheprovider --basetemp outputs\pytest-clean
python -m pip check
git diff --check
```

Clean checkout'ta `tk3d-check` için beklenen durum `PARTIALLY_READY` ve
`Tier 1 clean/lightweight: READY` değeridir. `PARTIALLY_READY`, dış araştırma
varlıklarının eksik olduğunu bildirir; Tier-1 başarısızlığı değildir.

Tier 1 şunları doğrular:

- paket ve application API importları;
- model config ve Poomsae profil/config sözleşmeleri;
- current/legacy artifact contract testleri;
- deterministic küçük fixture'lar;
- console entry point help ve readiness smoke;
- run isolation/lifecycle ve latest-success güvenliği.

## Tier 2 — CURRENT_ACTIVE yerel araştırma

Aktif workflow iki ZED 2i kamera ile RGBD çok-kamera 3B üretimi ve
`poomsae1_trimmed` analizidir. AIST yalnız `CURRENT_VALIDATION`, MADS ise
`HISTORICAL_BENCHMARK` sınıfındadır.

Tier 2 için gereken dış varlıklar:

| Gereksinim | Sınıf | Beklenen kaynak |
| --- | --- | --- |
| ViTPose-Huge WholeBody config | `INCLUDED_IN_REPOSITORY` | `config/mmpose_configs/...py` |
| ViTPose-Huge WholeBody checkpoint | `EXTERNAL_MODEL` | `weights/vitpose_huge_wholebody_256x192.pth` |
| RF-DETR Small checkpoint | `EXTERNAL_MODEL` | `%USERPROFILE%\.roboflow\models\rf-detr-small.pth` |
| ZED AVI/SVO2 ve timestamp mapping | `EXTERNAL_DATA` / `MACHINE_SPECIFIC` | Aktif session içindeki açık yollar |
| ZED SDK ve `pyzed` | `MACHINE_SPECIFIC` | Makineye kurulu uyumlu ZED SDK |
| Üretim calibration | `EXTERNAL_CALIBRATION` | `outputs/<session>/calibration/cameras.json` |
| Bağlı WholeBody-133 reference pose | `EXTERNAL_DATA` | Aktif profil `reference_pose` yolu |
| NVIDIA GPU/CUDA | `MACHINE_SPECIFIC` | Aktif config `device: cuda:0` |

RF-DETR'nin mevcut runtime politikası ilk gerçek inference sırasında kendi
checkpoint'ini Roboflow cache'ine indirebilir. `tk3d-check` hiçbir model indirmez
ve checkpoint yüklemez; yalnız seçilen logical model/variant ve beklenen yolu
raporlar. ViTPose checkpoint'i otomatik indirilmez. Her gerçek run, kullanılan
model dosyalarının SHA-256 değerini kendi manifestinde kaydeder.

Üretim calibration eksik veya production mode dışında olduğunda normal aktif
inference fail-closed kalır; readiness kontrolü approximate calibration'a
düşmez.

## Hafif readiness ve environment snapshot

```powershell
tk3d-check
tk3d-check --json
tk3d-check --write-report outputs\reproducibility\environment-<benzersiz-id>.json
```

Durumlar:

- `READY`: Tier 1 ve bütün `CURRENT_ACTIVE` dış varlıkları mevcut.
- `PARTIALLY_READY`: Tier 1 hazır, en az bir araştırma varlığı eksik.
- `NOT_READY`: paket/config/import/output gibi Tier-1 sözleşmesi bozuk.

Snapshot; Python, OS, Torch, CUDA, GPU, NumPy, SciPy, OpenCV, pandas,
TorchVision, RF-DETR, Supervision ve timm sürümlerini Phase A run-manifest
environment kolektörü üzerinden kaydeder.

## Aktif yerel smoke

Ana session'ın `latest_run.json` değerini kısa smoke ile değiştirmemek için ayrı
bir validation root kullanılır:

```powershell
$validationRoot = "outputs\phasec_current_active_validation"
New-Item -ItemType Directory -Force `
  "$validationRoot\poomsae_1_zed2i_20260731_trimmed\calibration" | Out-Null
Copy-Item `
  "outputs\poomsae_1_zed2i_20260731_trimmed\calibration\cameras.json" `
  "$validationRoot\poomsae_1_zed2i_20260731_trimmed\calibration\cameras.json"

tk3d-multiview `
  --session outputs\poomsae_1_zed2i_20260731_trimmed\source\session.yaml `
  --output-root $validationRoot `
  --stride 1 `
  --max-frames 30 `
  --run-id phasec-zed-smoke-<benzersiz-id> `
  --allow-low-quality-output

tk3d-poomsae `
  --profile poomsae1_trimmed `
  --run-id phasec-poomsae-smoke-<benzersiz-id>
```

30-kare smoke tam 741-kare run ile yalnız ortak ve adil zaman bölgesinde
karşılaştırılır. Offline filtre sağ sınırındaki son dört kare, gelecekteki
kareler kısa run'da bulunmadığı için byte-eşitlik iddiasına dahil edilmez.

## Run lifecycle

Yeni run dizini `run_state.json` ile `preparing` başlar. Uygulama bunu `running`,
başarılı teslim `completed`, açık hata işaretlemesi `failed` yapar. Failed veya
incomplete run `latest_run.json` değerini değiştiremez. Eski lifecycle dosyası
olmayan completed run'lar geriye uyumluluk için okunabilir.
