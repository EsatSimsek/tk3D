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

### 6.1 Kalibrasyon kayıtla uyuşmuyor

Her 3B koşu, yüksek güvenli 2B noktaların işaretli epipolar sapmasını ölçer
(`run_quality_report.json` → `calibration_consistency`). Kalibrasyon kayıttan
sonra kameralar oynatılarak alınmışsa noktalar hep aynı yönde sapar; mutlak
medyan 3 pikseli aşarsa koşu `failed` olur ve `latest` değişmez.

Bu durumda doğru çözüm kameraları oynatmadan kalibrasyonu yenilemektir. Eski
bir kayıt için kameralardan birinin yönelimi kaydın kendi noktalarından yeniden
kestirilebilir. Komut yeni bir dosya yazar; kaynak kalibrasyon değişmez. Çift
karelerde kestirir, tek karelerde sonucu raporlar. Sonuç iki kameranın kendi
içinde tutarlı olduğunu gösterir, bağımsız 3B doğruluk kanıtı değildir.

```powershell
Set-Location "$env:USERPROFILE\Desktop\tk3D"
.\.venv312\Scripts\Activate.ps1
$session = "outputs\poomsae_1_zed2i_20260731_trimmed"
python scripts\refine_calibration_from_keypoints.py `
  --calibration "$session\calibration\cameras.json" `
  --keypoints-2d "data\poomsae_1_zed2i_20260731_trimmed\reference\csv\vitpose_keypoints_2d_flat.csv" `
  --reference-camera zed_35151067 `
  --output "$session\calibration\cameras_keypoint_refined_$(Get-Date -Format 'yyyyMMdd').json"
```

Seçilen kalibrasyonla 3B ve Poomsae analizi:

```powershell
Set-Location "$env:USERPROFILE\Desktop\tk3D"
.\.venv312\Scripts\Activate.ps1
$runId = "refined-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
tk3d-poomsae --profile poomsae1_trimmed --process-video --run-id $runId `
  --calibration "outputs\poomsae_1_zed2i_20260731_trimmed\calibration\cameras_keypoint_refined_20261113.json"
Write-Host "Çıktı: $((Resolve-Path "outputs\poomsae_1_zed2i_20260731_trimmed\runs\$runId").Path)"
```

`tk3d-multiview` de aynı `--calibration` seçeneğini kabul eder.

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

İnceleme ekranındaki **Otomatik sabitleme incelemesi** bölümünde bütün hareket
kümeleri kronolojik gösterilir. El, ayak, pelvis ve gövde için güvenilir iki
kamera/3B poz kanıtı ile en az kısa bir ortak kararlılık aralığı bulunursa zaman
önerilir. Hareket yavaşladıysa ama bütün gerekli kanıt bulunmadıysa tamamlanma
adayı ve eksik kanıt ayrı gösterilir. Kayıt sonu, eksik eklem, yüksek reprojection
ve stride ile üretilmiş tekrar kareler sabitlenme kanıtı sayılmaz.

Elle yazılmış timeline yalnız ayrı bir **uyum karşılaştırması** olarak okunur;
otomatik işaretleri veya eşikleri belirlemez ve bu etiketlerin doğru olduğu
varsayılmaz. Otomatik adayın hareket kimliği de kesin olarak tanınmış sayılmaz.
Her adayı iki kamerada açıp kareyi değiştirerek **kabul et / seçtiğim kareyi
kaydet / sabitleme yok / belirsiz** kararlarından birini kaydedebilirsiniz.
Hareket kimliğini isteğe bağlı seçip sabitleme incelemesini JSON olarak indirin.
Bu JSON öneriyi değiştirmez; otomatik olarak bir doğrulanmış timeline veya puan
üretmez. Aynı raporda JSON tekrar yüklenebilir. Yerel tarayıcı kaydı geçici
olabileceğinden saklamak için JSON indirilmelidir.

Mevcut referans pose ve iki kamera videosuyla yeni, benzersiz bir inceleme koşusu:

```powershell
Set-Location "$env:USERPROFILE\Desktop\tk3D"
$runId = "fixation-review-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
.\.venv312\Scripts\python.exe -m scripts.run_poomsae_scoring --profile poomsae1_trimmed --run-id $runId
if ($LASTEXITCODE -ne 0) { throw 'Poomsae inceleme koşusu başarısız oldu.' }
$review = "outputs\poomsae_1_zed2i_20260731_trimmed\runs\$runId\review\poomsae_scoring_review.html"
Write-Host "İnceleme sayfası: $((Resolve-Path $review).Path)"
```

Etiketsiz yeni bir pose için `scripts.build_poomsae_automatic_segmentation`
komutu `--reference-timeline` olmadan da JSON, kare-sinyali CSV ve isteğe bağlı
`--output-html` üretir. `--camera NAME=VIDEO` ile eklenecek tarayıcı uyumlu
MP4/WebM videoların kaynak pose ile kare sayısı ve FPS'i aynı olmalıdır.
Videolar verilirse hash'leri, HTML üretilmese de inceleme kaynak bağına yazılır.
Kamera kullanılmadığında inceleme ekranı kare numaralarını gösterir ancak
görüntüyle doğrulama yapılamaz. Çıktılar yeni run klasörüne yazılmalı;
komut var olan dosyanın üzerine yazmayı reddeder. Algoritmanın `confidence`
değeri kalibre edilmiş doğruluk olasılığı değildir.

### Bağımsız işaretleme, kanıt denetimi ve sürüm karşılaştırması

Ana inceleme sayfasında elle oluşturulmuş çizelge düğmeleri **Elle işaret**,
diğer çizelgeler **Çizelge etiketi** olarak gösterilir. İnceleme kaydı durumu
ayrıca belirtilir; etikette beyan edilen güven doğruluk yüzdesi değildir.
**Otomatik sabitlemeler** bağlantısı yeni dedektörün önerilerine gider.
Hareketten çıkış, ortak kararlı aralık ve bileşen zamanları ayrı açılabilir.

Otomatik rapordaki `fixation_timing_diagnostics`, ilk önerilen kararlı aralığı
ek hız yumuşatması olmadan tekrar kontrol eder. En az mevcut minimum süreyi
sağlayan ilk ortak aralık ve onay karesi raporlanır. Bu kontrol ilk aralığın
dışındaki daha geç bir düzeltmeyi benimsemez, asıl öneriyi otomatik değiştirmez.
Kaynak 3B pose önceden filtrelenmiş olabilir; kontrol bağımsız görüntü kanıtı
değildir. Eller için mutlak hız ve kalçaya göre hızın ölçülebilirliği ayrı
gösterilir. Kalça eksikken mutlak el hızı ortak sabitlemeyi onaylamaz.

Yeni Poomsae koşuları ayrıca `fixation_quality/` üretir. Ana inceleme ekranından
kanıt raporuna ve bağımsız işaretleme ekranına geçilebilir. `quality_report.html`
otomatik adayları ve kaynak ret nedenlerini gösterir. Bağımsız inceleyen kişi
doğrudan **`blind_annotation.html`** açmalıdır; bu dosya önceki timeline'ı,
otomatik adayları veya başka inceleyenin etiketlerini içermez.

Tamamlanma, kararlı duruş başlangıcı ve bitişi ayrı olaylardır. Emin olunmayan
an tek kareye zorlanmadan bir aralıkla işaretlenir. İnceleyen/sporcu/oturum kodu
girilir; incelenen kapsam ve tamamlanma beyanı kaydedilir. Tarayıcı destekliyorsa
görüntüden kare alma çözümlenen video karesinin zaman damgasını kullanır.
Her inceleyenin JSON'u ayrı indirilir. Başka inceleyen için **Yeni inceleme başlat**
kullanılır. İnceleyen kimliği ve bağımsızlık beyanı uygulama tarafından doğrulanmaz.

Tek inceleyen varsa **İşaretleme turu** alanıyla aynı kayıt farklı zamanlarda
`tur-1`, `tur-2` olarak yeniden işaretlenebilir. Her tur boş başlatılır; önceki
JSON yüklenmeden video yeniden incelenir. İnceleyen, sporcu ve oturum kodları
aynı kalır. Aynı kişinin turları `intra_rater_repeatability`, farklı kişilerin
incelemeleri `inter_rater_agreement` olarak raporlanır. İki tur iki uzman sayılmaz.
Tur başına ayrı tarayıcı kaydı ve JSON dosyası vardır. Sürüm 2 JSON tur kodunu
zorunlu kılar; eski sürüm 1 dosyalar `legacy-1` turuyla okunur.

Var olan kaynak-bağlı V1 raporuyla yeni bir kalite paketi üretmek için
`$sourceRunId` değerini kendi makinenizde daha önce tamamlanmış bir Poomsae
koşusunun kimliğiyle değiştirin (örneğin yukarıdaki komutun yazdığı `$runId`):

```powershell
Set-Location "$env:USERPROFILE\Desktop\tk3D"
$sourceRunId = '<tamamlanmış-poomsae-run-id>'
.\.venv312\Scripts\python.exe -m scripts.build_fixation_quality_workspace --report "outputs\poomsae_1_zed2i_20260731_trimmed\runs\$sourceRunId\json\automatic_segmentation_report.json"
```

Komut benzersiz `$env:USERPROFILE\Desktop\tk3D\outputs\<session_id>\runs\fixation-quality-<id>\`
klasörünü ve içindeki HTML/JSON dosyalarının tam yollarını yazdırır. Eski run
değiştirilmez. `fixation_audit.json` kaynak kalite, kamera sayısı, reprojection,
ret bitleri, kaynak triangulation puanı, ortak gözlem/düşük hareket süreleri ve
varsa dışa aktarılmış 2B CSV'nin kamera bazlı ölçümlerini içerir. Bu CSV'deki
geçerlilik ve güven değerleri bağımsız görüntü kanıtı olarak yeniden onaylanmaz.

`fixation_sensitivity.json` aynı pose, hareket kümeleri, filtre ve kalite kapısı
üzerinde bekleme süresini %75/%100/%125 ve hareket eşiklerini %80/%100/%120
oranlarında deneyerek karar duyarlılığını gösterir. Geçersiz parametreler
atlanır ve ayrı listelenir. İnsan etiketleri bu deneye girmez; deney sayısı
bir doğruluk olasılığı, gözlenen kare aralığı da istatistiksel güven aralığı
değildir. Araç parametre seçmez veya runtime varsayılanını değiştirmez.

İndirilen JSON dosyaları sonraki çalıştırmada `--annotation 'C:\tam\yol\inceleyen-a.json'`
ile eklenir; her inceleyen için bu seçenek tekrarlanabilir. `--compare-report`
aynı videolara ve zaman tabanına sahip başka dedektör raporunu aynı işaretlerle
karşılaştırır. Bağımsız işaretlerin kimliği video hash'leri ve zaman tabanına
bağlıdır; farklı pose/dedektör sürümleriyle kullanılabilir. Pose hash'i ayrıca
denetim provenance'ında korunur.

`fixation_benchmark.json` olayları bire bir eşleştirir; fazladan ve kaçırılan
olaylar, precision/recall/F1, eşleşen olayların etiket aralığına kare/ms uzaklığı,
belirsiz bırakılan adaylar ve inceleyenler arası uyuşmazlıklar ayrı raporlanır.
Varsayılan 50/100/200 ms toleransları betimleyici deney ayarlarıdır; uzman onaylı
kabul sınırı değildir. İnceleme tamamlanmadıysa metrik üretilmez; insan JSON'u
yokken durum `awaiting_independent_annotations` olur. Otomatik etkinleştirme ve
puanlama onayı bu raporla verilmez.

İsteğe bağlı `--split-plan` JSON'u `schema_version: 1` ve `recordings` listesi
taşır. Her satır `recording_id`, `athlete_id`, `session_id`, `split` alanlarını
içerir; `split` değeri `development`, `validation` veya `test` olur. Aynı sporcu,
oturum veya kayıt farklı bölümlere taşındığında araç değerlendirmeyi reddeder.
Bu grupların ayrılması tek başına dış doğruluk kanıtı değildir.

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
