# TK3D

TK3D, senkronize çok-kameralı videodan güvenilir 3B insan pozu üretip bu
kanıtı tekvando poomsae teknik analizine taşıyan bir araştırma ve mühendislik
projesidir. Güncel ürün hattı iki ZED 2i kamerayla kaydedilmiş RGBD poomsae
verisini işler.

## Belge haritası

- `README.md`: güncel ürün özeti, kurulum ve hızlı başlangıç.
- [`docs/PROJECT_CONTEXT.md`](docs/PROJECT_CONTEXT.md): mühendisler ve yeni
  AI/Codex oturumları için ayrıntılı güncel teknik bağlam.
- [`docs/history/README_PRE_FINAL_POLISH.md`](docs/history/README_PRE_FINAL_POLISH.md):
  Final Polish öncesindeki uzun README'nin değiştirilmemiş tarihsel kopyası.
- [`PROJECT_STATUS.md`](PROJECT_STATUS.md): son doğrulanmış çalışma ağacı,
  testler, ölçümler ve açık sınırlamalar.
- [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md): clean checkout, yerel
  araştırma ortamı ve tekrarlanabilirlik sözleşmesi.
- [`docs/ARCHITECTURE_DECISIONS.md`](docs/ARCHITECTURE_DECISIONS.md): korunması
  gereken tasarım kararları.

Yeni bir geliştirme oturumunda önce [`AGENTS.md`](AGENTS.md), sonra
[`PROJECT_STATUS.md`](PROJECT_STATUS.md) ve
[`docs/PROJECT_CONTEXT.md`](docs/PROJECT_CONTEXT.md) okunmalıdır.

## Güncel kapsam

Kanonik `CURRENT_ACTIVE` workflow:

```text
iki ZED 2i kamera
  -> senkron AVI RGB + eşlenmiş SVO2 depth
  -> RF-DETR Small kişi tespiti + ByteTrack takibi
  -> ViTPose-Huge WholeBody 2B (133 nokta)
  -> robust çok-kameralı triangulation
  -> kapılı ZED depth fusion (BODY-17)
  -> güvenilirlik + BODY-17 global optimizasyon + 3B stabilizasyon
  -> keypoints_3d_world[t, 133, 3]
  -> kaynak-bağlı Poomsae analiz ve insan inceleme çıktıları
```

Ana 3B sözleşme metre cinsinden TK3D analiz koordinatlarıdır:
`x=sağ`, `y=ileri`, `z=yukarı`. BODY-17 üzerinde çalışan bir iyileştirme tüm
133 noktalı çıktıyı küçültemez veya silemez.

Sistem şu anda resmî ya da hakem-kalibre edilmiş poomsae puanı üretmeye hazır
değildir. Güncel ZED workflow'u için bağımsız dış 3B ground truth ve uzman/hakem
etiketi yoktur; hareket zaman çizelgesinin doğrulanmış kapsamı da sınırlıdır.
İç geometri veya kalite kapısının geçmesi dış doğruluk kanıtı değildir.
`provisional_scoring_ready` yalnız kaynak-bağlı analiz için veri hazırlığını,
`official_scoring_ready:false` ise resmî puan iddiasının kapalı olduğunu belirtir.

## Gereksinimler

- Windows ve Python 3.11 veya 3.12
- Tier 1 geliştirme/test için `requirements.txt`
- Gerçek pose inference için NVIDIA GPU/CUDA ve `requirements-pose.txt`
- `weights/vitpose_huge_wholebody_256x192.pth`
- RF-DETR Small checkpoint/cache
- Aktif ZED çalışması için ZED SDK, `pyzed`, AVI/SVO2 kayıtları ve üretim
  kalibrasyonu

Video, SVO2, checkpoint, sanal ortam ve `outputs/` içeriği repository'ye dahil
değildir. Ayrıntılı ortam matrisi ve dış varlık sınırı için
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) kullanılmalıdır.

## Kurulum

Clean-checkout Tier 1 ortamı:

```powershell
py -3.12 -m venv .venv312
.\.venv312\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e . --no-deps --no-build-isolation
```

Yerel CUDA araştırma ortamının sabitlenmiş bağımlılıkları
`requirements-pose.txt` içindedir. Bu ortamın kurulumu ve ViTPose checkpoint
hazırlığı için [`docs/vitpose_windows_setup.md`](docs/vitpose_windows_setup.md)
izlenmelidir.

## Hazırlık kontrolü

`tk3d-check` model yüklemeden paket, giriş noktası, config, output yazılabilirliği
ve aktif dış varlıkları denetler:

```powershell
tk3d-check
tk3d-check --json
tk3d-check --write-report outputs\reproducibility\environment-local.json
```

Clean checkout'ta Tier 1 hazırken dış araştırma verileri eksikse genel durumun
`PARTIALLY_READY` olması beklenir. Yerel `CURRENT_ACTIVE` varlıklarının tümü
mevcutsa `READY` döner. `NOT_READY`, Tier 1 sözleşmesinin bozuk olduğunu gösterir.

## Çok-kameralı 3B üretim

Kanonik giriş noktası `tk3d-multiview` komutudur. `--session` bir session YAML
yoludur; gerçek inference üretim kalibrasyonu ve model varlıkları ister.

```powershell
tk3d-multiview `
  --session outputs\poomsae_1_zed2i_20260731_trimmed\source\session.yaml `
  --stride 1 `
  --run-id <benzersiz-run-id>
```

Hızlı bir smoke için `--max-frames` kullanılabilir; smoke sonucu tam koşuyla
yalnız aynı video aralığı, kamera seti, stride, model ve config altında
karşılaştırılmalıdır. `stride > 1` inference örneklemesidir; kaynak zaman
çizelgesini kısaltma yetkisi vermez.

Başlıca çıktılar:

```text
outputs/<session_id>/runs/<run_id>/
├── run_state.json
├── config/
├── json/
│   ├── run_manifest.json
│   ├── vitpose_session_3d.json
│   ├── run_quality_report.json
│   └── performance_report.json       # yalnız profiling açıkken
├── csv/
├── figures/
└── videos/
```

`vitpose_session_3d.json` kare/zaman kimliğini, WholeBody-133 3B noktaları,
geçerlilik/kalite alanlarını ve provenance bağını taşır. Eksik JSON değerleri
`null`, CSV değerleri boş hücre olmalıdır; `NaN` veya `inf` downstream çıktıya
sızmamalıdır.

## Poomsae analizi

Kanonik giriş noktası `tk3d-poomsae` komutudur. Varsayılan aktif profil
`poomsae1_trimmed` mevcut doğrulanmış 3B artifact'i kullanır:

```powershell
tk3d-poomsae `
  --profile poomsae1_trimmed `
  --run-id <benzersiz-run-id>
```

Bağlı videodan stride-1 3B üretimini de yeniden çalıştırmak için:

```powershell
tk3d-poomsae `
  --profile poomsae1_trimmed `
  --process-video `
  --run-id <benzersiz-run-id>
```

Bu workflow otomatik segmentasyon önerileri, WholeBody ölçümleri, hareket
kanıtı, kategorik ve teknik uygunluk teşhisleri, kaynak-bağlı provisional
kararlar, readiness, hata videosu ve HTML inceleme raporları üretir. Teşhis
adayları otomatik WT kesintisi veya resmî Accuracy skoru değildir.

V3 kapsamlı teknik-doğruluk katmanı M01–M18 hareket kontratı ve 174 kurallık
makine-okunur envanter üretir. Pipeline sınırındaki 8 özellik dışında 166
kuralın ölçüm evaluator yolu vardır; 133 landmarkın tamamı envanterde, 51'i
aktif eşikli kuralların zorunlu kümesindedir. Aktif videoda yalnız M01–M06
ölçülür; M07–M18 kontrat/sentetik kapsamdır. Geçici adaylar skor ve kesintiyi değiştirmez.
Baş/yüz çıktısı gerçek göz bakışı değil `head_orientation_proxy` olarak
yorumlanır. Sporcu-yerel yön referansı her run'da açılış duruşundan türetilir ve
oturum/pose hash'ine bağlanır; türetilemezse 17 yön kuralı fail-closed kalır ve
gerekçe `json/athlete_local_direction_reference_status.json` içine yazılır.
Ayrıntı:
[`docs/TECHNICAL_ACCURACY_DIAGNOSTICS.md`](docs/TECHNICAL_ACCURACY_DIAGNOSTICS.md).

Elle etiketlenmemiş bir kayıt için hareket zaman çizelgesi **önerisi** ayrı bir
komutla üretilir; kanonik akış bu taslağı kendiliğinden tüketmez. Taslak, tespit
edilen segmentleri `config/scoring/templates/` altındaki referans duruşlarla
eşleştirir, şüpheli eşleşmeleri `ambiguous` işaretler ve hizalama anomalilerini
ayrı bir rapora yazar. İnsan düzeltmesi olmadan puanlamaya girmez. Ayrıntı:
[`docs/AUTOMATIC_TIMELINE_DRAFT.md`](docs/AUTOMATIC_TIMELINE_DRAFT.md).

Kural motorunun sentetik yazılım doğrulaması ayrı ve puansızdır. Düzenek 174
kural ile 133-landmark kapsam envanterini, 33 aktif kural için 330
sınır/eksik/non-finite vakasını ve 12 WholeBody-133 geometri senaryosunu
hash'li manifest taşıyan makine-okunur JSON/CSV artifact'leri olarak üretir.
Bu sonuç hakem veya biomekanik doğruluk iddiası değildir. Çalıştırma komutu ve
yorum sınırları:
[`docs/TECHNICAL_ACCURACY_RULE_VALIDATION.md`](docs/TECHNICAL_ACCURACY_RULE_VALIDATION.md).

Önemli Poomsae çıktıları:

```text
json/poomsae_scoring_summary.json
json/automatic_segmentation_report.json
json/wholebody_diagnostics_report.json
json/athlete_local_direction_reference_status.json
json/technical_accuracy_diagnostics_report.json
csv/technical_accuracy_coverage_matrix.csv
csv/technical_accuracy_landmark_coverage.csv
json/movement_evidence_report.json
json/categorical_diagnostics_report.json
json/technical_conformance_report.json
json/source_bound_accuracy_decisions.json
json/rule_scoring_readiness.json
review/poomsae_scoring_review.html
review/run_history.html
videos/poomsae_scoring_annotated.mp4
```

## Aktif yapılandırma

- Model/runtime: [`config/model_config.yaml`](config/model_config.yaml)
- Poomsae profil:
  [`config/scoring/profiles/poomsae1_trimmed.yaml`](config/scoring/profiles/poomsae1_trimmed.yaml)
- ViTPose-Huge config:
  [`config/mmpose_configs/wholebody_2d_keypoint/vitpose/coco-wholebody/td-hm_ViTPose-huge_8xb64-210e_coco-wholebody-256x192.py`](config/mmpose_configs/wholebody_2d_keypoint/vitpose/coco-wholebody/td-hm_ViTPose-huge_8xb64-210e_coco-wholebody-256x192.py)
- RulePack: `config/scoring/rules/wt_recognized_2024-09-30.yaml`
- PoomsaeSpec: `config/scoring/poomsae/taegeuk_1_jang_v0_draft.yaml`
- Hareket zaman çizelgesi:
  `config/scoring/timelines/poomsae1_zed2i_rgbd_rerun_20260802_draft.yaml`
- WholeBody teşhis profili:
  `config/scoring/engineering/taegeuk_1_wholebody_diagnostics_v2.yaml`
- Kapsamlı teknik-doğruluk profili:
  `config/scoring/engineering/taegeuk_1_wholebody_diagnostics_v3.yaml`
- Teknik-doğruluk sentetik validation profili:
  `config/scoring/validation/taegeuk_1_rule_accuracy_validation_v1.yaml`
- Kaynak-bağlı Accuracy profili:
  `config/scoring/accuracy/taegeuk_1_source_bound_v1.yaml`

Aktif profil dış `outputs/` varlıklarına SHA-256 bağları içerir. Bu yolları,
hash'leri veya config'leri yalnız biçimsel temizlik amacıyla değiştirmeyin.

## Çıktı ve güvenlik kuralları

- Her çalışma `outputs/<session_id>/runs/<run_id>/` altında benzersiz ve
  üzerine yazılamaz bir dizin kullanır.
- `latest_run.json` yalnız başarıyla tamamlanan ve uygun koşuya ilerletilir;
  failed/incomplete run önceki başarılı işaretçiyi değiştiremez.
- Workflow içindeki bir subprocess aşaması hata verirse `run_state.json`
  `running` bırakılmaz; aşama adı, exit code ve `failed` durumu kaydedilir.
- Her güncel 3B artifact; session/run kimliği, calibration snapshot/hash ve run
  manifestiyle bağlanır. Sözleşme uyuşmazlığı fail-closed'dur.
- Yaklaşık calibration açıkça izin verilmedikçe üretim inference'ta kabul
  edilmez ve puanlama kanıtı sayılmaz.
- Ham triangulation korunur. Depth fusion veya global optimizer kalite kapısını
  geçemezse güvenli önceki sonuca dönülür.
- Hedef kameraya yapılan 2B geri beslemede hedef kamera kendi 3B öncülünden
  çıkarılmalıdır. Yalnız izdüşümden gelen nokta bağımsız görüntü kanıtı değildir.

## Workflow sınıfları

| Sınıf | Güncel anlamı |
| --- | --- |
| `CURRENT_ACTIVE` | İki ZED 2i RGBD → WholeBody-133 3B → `poomsae1_trimmed` analizi |
| `CURRENT_VALIDATION` | AIST/AIST++ korunmuş ikincil geometri ve uyumluluk smoke'u |
| `HISTORICAL_BENCHMARK` | MADS Kata F2 gibi kendi koşuluna bağlı tarihsel dış benchmark |
| `LEGACY` | Sentetik/önceki nesil uyumluluk yolları; kanonik ürün workflow'u değil |

Davranış dondurma ve regresyonda uygun veri varsa birincil karşılaştırma
`CURRENT_ACTIVE` üzerinde yapılır. AIST veya MADS sonucu güncel ZED RGBD
doğruluğuna devredilemez.

## Geliştirme ve doğrulama

Python değişikliklerinin varsayılan yerel teslim kapısı:

```powershell
.\.venv312\Scripts\python.exe -m ruff check src scripts tests
.\.venv312\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp outputs\pytest-local
.\.venv312\Scripts\python.exe -m pip check
git diff --check
```

Gerçek inference davranışı değiştiyse ilgili kısa gerçek smoke ve JSON kalite
raporu da incelenmelidir. En son gerçekten doğrulanmış test/benchmark sayıları
için yalnız [`PROJECT_STATUS.md`](PROJECT_STATUS.md) referans alınmalıdır.

## Daha derin belgeler

- Güncel teknik hafıza: [`docs/PROJECT_CONTEXT.md`](docs/PROJECT_CONTEXT.md)
- Mühendislik sırası: [`docs/ENGINEERING_WORKFLOW.md`](docs/ENGINEERING_WORKFLOW.md)
- Veri/workflow sınıfları: [`docs/DATASET_NOTES.md`](docs/DATASET_NOTES.md)
- Legacy sınırı: [`docs/LEGACY_COMPONENTS.md`](docs/LEGACY_COMPONENTS.md)
- Poomsae kural ve hareket sistemi:
  [`docs/POOMSAE1_HAREKET_VE_KURAL_SISTEMI.md`](docs/POOMSAE1_HAREKET_VE_KURAL_SISTEMI.md)
- Otomatik segmentasyon: [`docs/AUTOMATIC_SEGMENTATION.md`](docs/AUTOMATIC_SEGMENTATION.md)
- Teknik uygunluk: [`docs/TECHNICAL_CONFORMANCE.md`](docs/TECHNICAL_CONFORMANCE.md)
- Hata taksonomisi: [`docs/TAEGEUK1_ERROR_TAXONOMY.md`](docs/TAEGEUK1_ERROR_TAXONOMY.md)
- Korunmuş eski README:
  [`docs/history/README_PRE_FINAL_POLISH.md`](docs/history/README_PRE_FINAL_POLISH.md)
