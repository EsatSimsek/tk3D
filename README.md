# TK3D

TK3D, iki senkron ZED 2i kaydından WholeBody-133 üç boyutlu insan pozu
üreten ve sonucu tekvando poomsae inceleme çıktısına dönüştüren Windows tabanlı
bir araştırma uygulamasıdır.

Bu depo kaynak kodunu, testleri ve çalışma ayarlarını içerir. Yaklaşık 6,9 GB
olan video, SVO2 ve model dosyaları GitHub'a konmaz. Projeyi paylaşan kişiden
ayrıca `TK3D_PAYLASIM_PAKETI` klasörünü almanız gerekir.

> TK3D resmî veya hakem-kalibre edilmiş bir poomsae puanı üretmez. Mevcut
> çıktı; ölçüm, teknik teşhis ve insan inceleme desteğidir. Faz onayı olmayan
> ölçümler kesintiye dönüştürülmez ve toplam puan `null` kalır.

## 1. Doğrulanmış çalışma ortamı

Bu sürüm aşağıdaki ortamda çalıştırılmıştır:

| Bileşen | Doğrulanmış sürüm |
| --- | --- |
| İşletim sistemi | Windows 11 x64 |
| Python | 3.12.14 x64 |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| PyTorch | 2.13.0+cu130 |
| TorchVision | 0.28.0+cu130 |
| CUDA runtime | 13.0 |
| ZED Python API | pyzed 5.4 |
| RF-DETR | 1.7.0 |
| Supervision | 0.27.0 |
| OpenCV | 5.0.0.93 |

Gerçek RGBD inference için native Windows, CUDA destekli NVIDIA GPU ve ZED SDK
gerekir. Stereolabs güncel gereksinimlerinde compute capability 7.5 veya daha
yeni NVIDIA GPU ister; 16 GB RAM ve 6 GB üzeri VRAM önerir. Resmî belgeler:

- [Windows için ZED SDK kurulumu](https://docs.stereolabs.com/docs/development/zed-sdk/windows)
- [ZED Python API kurulumu](https://docs.stereolabs.com/docs/development/api-languages/python)
- [ZED SDK donanım gereksinimleri](https://docs.stereolabs.com/docs/development/zed-sdk/specifications/)
- [PyTorch yerel kurulum sayfası](https://pytorch.org/get-started/locally/)

Kurulum, paylaşılan varlıklar ve yeni çıktılar için sistem diskinde en az
15 GB boş alan ayırın. Uzun koşular daha fazla alan kullanabilir.

## 2. Almanız gereken iki parça

1. GitHub deposu: kaynak kodu ve yapılandırmalar.
2. Projeyi paylaşan kişiden `TK3D_PAYLASIM_PAKETI`: Git'e giremeyen büyük ve
   makine-dışı varlıklar.

Paketin beklenen yapısı şöyledir:

```text
TK3D_PAYLASIM_PAKETI/
├── README.md
├── SHA256SUMS.txt
├── repo/
│   ├── data/poomsae_1_zed2i_20260731_trimmed/
│   │   ├── videos/                 # iki kırpılmış 741-kare AVI
│   │   ├── svo/                    # iki özgün SVO2 depth kaydı
│   │   ├── reports/                # AVI-SVO zaman eşleme raporları
│   │   └── reference/              # doğrulanmış 3B pose + gözlenen 2B CSV
│   ├── outputs/poomsae_1_zed2i_20260731_trimmed/
│   │   └── calibration/cameras.json
│   └── weights/
│       └── vitpose_huge_wholebody_256x192.pth
├── user_home/.roboflow/models/
│   └── rf-detr-small.pth
└── extras/
    └── zed_fusion_calibration.json
```

Dosya veya klasör adlarını değiştirmeyin. Kurucu betik tüm zorunlu dosyaları
`SHA256SUMS.txt` ile doğrular; eksik veya bozuk pakette kopyalama başlamaz.

## 3. Sıfırdan Windows kurulumu

### 3.1 Önkoşullar

- Git for Windows
- 64 bit Python 3.12 ve `py` launcher
- Güncel NVIDIA sürücüsü
- CUDA 13 ile uyumlu ZED SDK 5.x; bu proje pyzed 5.4 ile doğrulandı
- `TK3D_PAYLASIM_PAKETI` klasörü

ZED SDK'yı native Windows'a kurun ve kurulumdan sonra bilgisayarı yeniden
başlatın. WSL bu kayıt için desteklenen kurulum yolu değildir.

### 3.2 Depoyu clone etme ve sanal ortam

Aşağıdaki bloğu normal PowerShell'de çalıştırın. GitHub adresini size verilen
depo adresiyle değiştirin.

```powershell
Set-Location "$env:USERPROFILE\Desktop"
git clone https://github.com/EsatSimsek/tk3D.git
Set-Location "$env:USERPROFILE\Desktop\tk3D"

py -3.12 -m venv .venv312
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv312\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-pose.txt
python -m pip install -e . --no-deps --no-build-isolation
```

`requirements-pose.txt`, doğrulanan CUDA 13 PyTorch/TorchVision sürümlerini ve
RF-DETR bağımlılıklarını sabitler. Yalnız CPU ile hafif testler yapılabilir;
gerçek aktif inference ayarı `cuda:0` kullanır.

### 3.3 pyzed'i sanal ortama kurma

Önce `.venv312` etkin olmalıdır. ZED SDK'nın kendi kurucusunu aynı Python ile
çalıştırın:

```powershell
Set-Location "$env:USERPROFILE\Desktop\tk3D"
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv312\Scripts\Activate.ps1
python "C:\Program Files (x86)\ZED SDK\get_python_api.py"
python -c "import pyzed.sl as sl; print('pyzed hazır:', sl.__file__)"
```

`Permission denied` görülürse PowerShell'i yönetici olarak açın veya
`get_python_api.py` dosyasını masaüstüne kopyalayıp etkin sanal ortamın
`python` komutuyla çalıştırın. `pyzed`, ZED SDK kurulmadan bağımsız bir paket
gibi kullanılamaz.

### 3.4 Paylaşım paketini kurma

`$paket` değerini paketin bulunduğu tam Windows yoluyla değiştirin:

```powershell
Set-Location "$env:USERPROFILE\Desktop\tk3D"
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
$paket = "D:\TK3D_PAYLASIM_PAKETI"
.\scripts\install_shared_assets.ps1 -PackageRoot $paket
```

Kurucu şu işlemleri yapar:

- AVI, SVO2, zaman eşleme ve referans pose dosyalarını deponun `data` dizinine
  kopyalar;
- üretim kalibrasyonunu `outputs/<session_id>/calibration` altına koyar;
- ViTPose ağırlığını `weights` altına koyar;
- RF-DETR ağırlığını `%USERPROFILE%\.roboflow\models` altına koyar;
- kopyalama öncesi ve sonrası SHA-256 doğrulaması yapar;
- var olan farklı bir dosyanın üzerine `-Force` verilmeden yazmaz.

Yalnız paketi kontrol etmek, hiçbir şey kopyalamamak için:

```powershell
Set-Location "$env:USERPROFILE\Desktop\tk3D"
$paket = "D:\TK3D_PAYLASIM_PAKETI"
.\scripts\install_shared_assets.ps1 -PackageRoot $paket -VerifyOnly
```

## 4. Kurulum doğrulaması

Bu blok paket, model, CUDA, ZED, video, SVO2, kalibrasyon ve profile hash
bağlarını kontrol eder. Raporun tam yolu komut sonunda yazılır.

```powershell
Set-Location "$env:USERPROFILE\Desktop\tk3D"
.\.venv312\Scripts\Activate.ps1
$report = "outputs\reproducibility\environment-$(Get-Date -Format 'yyyyMMdd-HHmmss').json"
tk3d-check --write-report $report
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'YOK')"
Write-Host "Rapor: $((Resolve-Path $report).Path)"
```

Beklenen sonuç:

```text
READY
Tier 1 clean/lightweight: READY
CURRENT_ACTIVE local research: READY
```

`PARTIALLY_READY`, kaynak kodunun kurulabildiğini fakat bir veya daha fazla
haricî varlığın eksik olduğunu gösterir. Komutun `[WARN]` satırı eksik dosyanın
tam yolunu verir. `NOT_READY`, kaynak/config sözleşmesinin bozuk olduğunu
gösterir.

## 5. İlk güvenli smoke koşusu

Smoke, yalnız kurulumun çalıştığını sınar; tam kayıt kalitesiyle
karşılaştırılmaz. On karelik tanısal koşu kalite kapısını geçmek zorunda
olmadığı için `--allow-low-quality-output` kullanılır ve `latest_run.json`
güncellenmez.

```powershell
Set-Location "$env:USERPROFILE\Desktop\tk3D"
.\.venv312\Scripts\Activate.ps1
$runId = "clone-smoke-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
tk3d-multiview `
  --session config\sessions\poomsae1_zed2i_trimmed.yaml `
  --stride 1 `
  --max-frames 10 `
  --allow-low-quality-output `
  --run-id $runId
$runPath = "outputs\poomsae_1_zed2i_20260731_trimmed\runs\$runId"
Write-Host "Smoke çıktısı: $((Resolve-Path $runPath).Path)"
```

Smoke sonunda en az şu dosyalar oluşmalıdır:

```text
outputs/poomsae_1_zed2i_20260731_trimmed/runs/<run_id>/
├── run_state.json
├── config/
├── json/
│   ├── run_manifest.json
│   ├── vitpose_session_3d.json
│   └── run_quality_report.json
├── csv/
├── figures/
└── videos/
```

## 6. Tam 741-kare 3B koşu

Tam koşu GPU'da uzun sürebilir. Her seferinde benzersiz `run_id` kullanılır;
var olan bir run'ın üzerine yazılmaz.

```powershell
Set-Location "$env:USERPROFILE\Desktop\tk3D"
.\.venv312\Scripts\Activate.ps1
$runId = "full-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
tk3d-multiview `
  --session config\sessions\poomsae1_zed2i_trimmed.yaml `
  --stride 1 `
  --run-id $runId
$runPath = "outputs\poomsae_1_zed2i_20260731_trimmed\runs\$runId"
Write-Host "Tam koşu çıktısı: $((Resolve-Path $runPath).Path)"
```

Kalite kapısı başarısızsa tanısal dosyalar korunur, run `failed` olur ve önceki
başarılı `latest_run.json` değiştirilmez. `stride > 1` yalnız inference
örneklemesidir; çıktı zaman çizelgesi kaynak videonun süresini korur.

## 7. Mevcut doğrulanmış pose ile Poomsae analizi

Bu komut paketle gelen referans pose'u kullanır; ViTPose inference'ı yeniden
çalıştırmaz. İki kaynak video yine HTML ve işaretli inceleme videosu için
gereklidir.

```powershell
Set-Location "$env:USERPROFILE\Desktop\tk3D"
.\.venv312\Scripts\Activate.ps1
$runId = "poomsae-review-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
tk3d-poomsae --profile poomsae1_trimmed --run-id $runId
$runPath = "outputs\poomsae_1_zed2i_20260731_trimmed\runs\$runId"
Write-Host "Poomsae çıktısı: $((Resolve-Path $runPath).Path)"
```

Videodan 3B pose'u yeniden üretip ardından analiz etmek için:

```powershell
Set-Location "$env:USERPROFILE\Desktop\tk3D"
.\.venv312\Scripts\Activate.ps1
$runId = "poomsae-full-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
tk3d-poomsae --profile poomsae1_trimmed --process-video --run-id $runId
Write-Host "Poomsae tam çıktı: $((Resolve-Path "outputs\poomsae_1_zed2i_20260731_trimmed\runs\$runId").Path)"
```

Poomsae workflow'u ölçümler, otomatik segmentasyon önerileri, teknik teşhisler,
işaretli video ve HTML inceleme ekranı üretir. Bunlar resmî hakem puanı değildir.

## 8. Geliştirici doğrulaması

Kod değişikliği göndermeden önce:

```powershell
Set-Location "$env:USERPROFILE\Desktop\tk3D"
.\.venv312\Scripts\Activate.ps1
python -m ruff check src scripts tests
$testRoot = "outputs\pytest-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
python -m pytest -q -p no:cacheprovider --basetemp $testRoot
python -m pip check
git diff --check
Write-Host "Pytest geçici dizini: $((Resolve-Path $testRoot).Path)"
```

Git'e video, SVO2, model, sanal ortam veya `outputs/` içeriği eklemeyin.

## 9. Sık karşılaşılan sorunlar

### `py -3.12` bulunamıyor

Python 3.12 x64'ü yeniden kurun ve kurulumda Python Launcher seçeneğini
etkinleştirin. `py -0p` kurulu Python yollarını listeler.

### PowerShell betik çalıştırmayı engelliyor

Yalnız açık terminal süreci için izin verin:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### `No module named pyzed`

`.venv312` etkin değilken ZED Python API kurulmuş olabilir. Sanal ortamı
etkinleştirip `C:\Program Files (x86)\ZED SDK\get_python_api.py` betiğini aynı
`python` komutuyla yeniden çalıştırın.

### CUDA `False` veya GPU görünmüyor

NVIDIA sürücüsünü, ZED SDK/CUDA uyumunu ve
`requirements-pose.txt` kurulumunun hatasız tamamlandığını kontrol edin.
CPU PyTorch kurulmuşsa gerçek inference başlamaz.

### `tk3d-check` bir dosyada hash uyuşmazlığı bildiriyor

Dosyayı elle yeniden adlandırmayın veya düzenlemeyin. Paylaşım paketini
`-VerifyOnly` ile kontrol edin; bozuk dosyayı paket sahibinden yeniden alın.

### Kalibrasyon bulunamıyor

Beklenen tam yol:

```text
C:\Users\<kullanıcı>\Desktop\tk3D\outputs\poomsae_1_zed2i_20260731_trimmed\calibration\cameras.json
```

Kalibrasyonu yaklaşık/sentetik veriyle üretim kalibrasyonu gibi işaretlemeyin.

## 10. Veri sözleşmesi ve güvenlik sınırları

- Ana çıktı `keypoints_3d_world[t, 133, 3]` ve metre cinsinden TK3D analiz
  koordinatlarıdır: `x=sağ`, `y=ileri`, `z=yukarı`.
- Eksik değer JSON'da `null`, CSV'de boş hücredir; `NaN` ve `inf` downstream
  çıktıya yazılmaz.
- Global optimizasyon ham triangulation'ı silmez; kalite kapısı başarısızsa
  güvenli ham sonuca dönülür.
- 3B izdüşüm arama öncülü veya açık görselleştirme fallback'idir; bağımsız
  görüntü kanıtı sayılıp tekrar triangulation'a sokulmaz.
- Kalite karşılaştırmaları aynı video bölümü, kamera seti, stride, model ve
  yapılandırmayla yapılır. Smoke ve tam koşu kalite kıyası değildir.
- Geçerli iç geometri, dış 3B ground-truth doğruluğu veya resmî poomsae puan
  doğruluğu anlamına gelmez.
