# TK3D Workflow, Veri Seti ve Legacy Bileşen Sınırı

Bu kayıt benzer görünen workflow, adapter ve yardımcıların güncel rolünü
sınıflandırır. Sınıflandırma repository içi çağrı, config/profile bağı,
entry point, test, dokümantasyon, son yerel run ve artifact provenance
kanıtlarına dayanır.

## Sınıflar

| Sınıf | Anlam |
| --- | --- |
| `CURRENT_ACTIVE` | Güncel ürün/araştırma geliştirmesinin kanonik workflow veya doğrudan gerekli bileşeni |
| `CURRENT_VALIDATION` | Aktif workflow'un yerine geçmeyen, korunan ikincil validation/smoke yolu |
| `SUPPORTED_COMPATIBILITY` | Test edilen veya açıkça desteklenen geriye uyumluluk/adaptör yolu; kanonik değil |
| `HISTORICAL_BENCHMARK` | Sonucu yalnız kendi bağlı eski veri/koşuluna ait ölçüm yolu |
| `LEGACY` | Önceki nesil veya artık kanonik olmayan bileşen; yeni geliştirme hedefi değil |
| `UNCERTAIN / RETAIN` | Repository içi referansı bulunmayan fakat doğrudan script/dış kullanım ihtimali dışlanamayan yardımcı; denetimsiz silinmez |

`CURRENT_VALIDATION`, birincil `CURRENT_ACTIVE` regresyonunun yerine kullanılamaz.
`HISTORICAL_BENCHMARK` sonucu başka sensör/domain workflow'una taşınamaz.

## Workflow ve veri setleri

| Workflow / veri | Sınıf | Gerekçe |
| --- | --- | --- |
| `poomsae1_trimmed`: iki ZED 2i → AVI RGB + SVO2 depth → WholeBody-133 3B → Poomsae analizi | `CURRENT_ACTIVE` | Aktif profil, güncel config, yerel run'lar ve uygulama API'leri bu hattı bağlar; birincil davranış/regresyon örneğidir. |
| ViTPose-Huge iki-ZED multiview rekonstrüksiyon | `CURRENT_ACTIVE` | Kanonik 3B uygulama ve `tk3d-multiview` entry point'idir. |
| Kaynak-bağlı Poomsae application workflow'u | `CURRENT_ACTIVE` | Kanonik analiz/diagnostic/provisional-decision/review uygulaması ve `tk3d-poomsae` entry point'idir. |
| AIST/AIST++ dokuz-kamera dans sekansı | `CURRENT_VALIDATION` | Kalibrasyon, senkron, triangulation, export ve manifest uyumluluğu için korunan ikincil smoke'tur; Poomsae ürünü veya aktif accuracy benchmark'ı değildir. |
| MADS Kata F2 RGB-only + mocap | `HISTORICAL_BENCHMARK` | Bağlı üç-kamera RGB-only koşusunun tarihsel dış 3B doğruluk sonucudur; güncel ZED RGBD hattına devredilmez. |
| Sentetik `run_multiview_3d.py --dry-run` | `SUPPORTED_COMPATIBILITY` / `LEGACY` | İlk nesil triangulation/artifact sözleşmesini test eder; gerçek kamera veya kanonik inference workflow'u değildir. |

## CURRENT_ACTIVE giriş noktaları ve uygulamalar

| Bileşen | Sınıf | Rol |
| --- | --- | --- |
| `tk3d-check` / `scripts/check_reproducibility.py` | `CURRENT_ACTIVE` | Tier 1 ve yerel CURRENT_ACTIVE hazırlığını model yüklemeden denetleyen kanonik entry point |
| `tk3d-multiview` / `scripts/run_vitpose_multiview_3d.py` | `CURRENT_ACTIVE` | Kalibrasyonlu WholeBody-133 multiview 3B için ince CLI adaptörü |
| `src/multiview_application.py` | `CURRENT_ACTIVE` | Kanonik multiview uygulama API'si ve orchestration sınırı |
| `tk3d-poomsae` / `scripts/run_poomsae_scoring.py` | `CURRENT_ACTIVE` | Poomsae analiz/diagnostic/provisional karar workflow'u için ince CLI adaptörü |
| `src/poomsae_scoring/application.py` | `CURRENT_ACTIVE` | Kanonik Poomsae application API'si ve orchestration sınırı |
| `src/performance.py` | `CURRENT_ACTIVE` | Opsiyonel ve bilimsel artifact'leri değiştirmeyen performans enstrümantasyonu |

## Aktif Poomsae application yardımcıları

Aşağıdaki script'ler `src/poomsae_scoring/application.py` tarafından doğrudan
çağrılır veya aktif tek-komut workflow'unun zorunlu alt aşamasıdır. Ayrı CLI
dosyaları olmaları onları dead/legacy yapmaz:

| Bileşen | Sınıf | Aktif rol |
| --- | --- | --- |
| `scripts/check_models.py` | `CURRENT_ACTIVE` | model/session preflight |
| `scripts/run_vitpose_multiview_3d.py` | `CURRENT_ACTIVE` | opsiyonel bağlı video → 3B yeniden işleme |
| `scripts/analyze_pose_for_scoring.py` | `CURRENT_ACTIVE` | bağlı 3B pose kanıt analizi |
| `scripts/build_browser_review_videos.py` | `CURRENT_ACTIVE` | browser-compatible inceleme videoları |
| `scripts/build_poomsae_automatic_segmentation.py` | `CURRENT_ACTIVE` | hareket/faz segmentasyon önerileri |
| `scripts/run_wholebody_poomsae_diagnostics.py` | `CURRENT_ACTIVE` | WholeBody-133 ölçüm/diagnostic raporu |
| `scripts/analyze_poomsae_movement_evidence.py` | `CURRENT_ACTIVE` | hareket kanıtı |
| `scripts/run_categorical_poomsae_diagnostics.py` | `CURRENT_ACTIVE` | kategorik hareket/duruş teşhisleri |
| `scripts/build_poomsae_technical_conformance.py` | `CURRENT_ACTIVE` | teknik uygunluk füzyonu |
| `scripts/build_poomsae_presentation_diagnostics.py` | `CURRENT_ACTIVE` | presentation proxy teşhisleri |
| `scripts/assess_poomsae_scoring_readiness.py` | `CURRENT_ACTIVE` | fail-closed readiness |
| `scripts/build_source_bound_accuracy_decisions.py` | `CURRENT_ACTIVE` | kaynak-bağlı provisional Accuracy kararları |
| `scripts/build_poomsae_evidence_events.py` | `CURRENT_ACTIVE` | karar/inceleme kanıt olayları |
| `scripts/render_poomsae_error_video.py` | `CURRENT_ACTIVE` | işaretli inceleme videosu |
| `scripts/create_poomsae_review_report.py` | `CURRENT_ACTIVE` | HTML inceleme raporu |
| `scripts/build_poomsae_run_history.py` | `CURRENT_ACTIVE` | karşılaştırılabilir run geçmişi |

Bu listedeki dosyalar kozmetik temizlik veya “tek referanslı script” gerekçesiyle
taşınmamalı ya da silinmemelidir; uygulama subprocess yolları bu adlara bağlıdır.

## CURRENT_VALIDATION bileşenleri

| Bileşen | Sınıf | Rol |
| --- | --- | --- |
| `scripts/import_aist_cameras.py` | `CURRENT_VALIDATION` | Resmî AIST kamera verisini TK3D calibration sözleşmesine aktarır |
| `scripts/run_full_aist_pose.ps1` | `CURRENT_VALIDATION` | Korunan tam AIST smoke wrapper'ı |
| AIST session/setup yardımcıları | `CURRENT_VALIDATION` | İkincil multiview validation verisini hazırlar |
| `scripts/evaluate_ground_truth_3d.py` | `CURRENT_VALIDATION` | Bağlı tahmin/GT/config/manifest için genel fail-closed 3B değerlendirme yolu |
| `scripts/render_aist_smpl_mesh.py` | `CURRENT_VALIDATION` | Opsiyonel tarihsel AIST mesh görselleştirmesi |
| `scripts/export_aist_smpl_threejs_viewer.py` | `CURRENT_VALIDATION` | Opsiyonel browser mesh viewer |
| `scripts/view_aist_smpl_mesh_open3d.py` | `CURRENT_VALIDATION` | Opsiyonel Open3D viewer |

SMPL yardımcıları ana 3B veya Poomsae akışının önkoşulu değildir. Lisanslı
SMPL model dosyaları repository'ye dahil edilmez.

## SUPPORTED_COMPATIBILITY ve LEGACY bileşenleri

| Bileşen | Sınıf | Gerekçe |
| --- | --- | --- |
| `scripts/run_multiview_3d.py --dry-run` | `SUPPORTED_COMPATIBILITY` / `LEGACY` | Sentetik ilk nesil workflow ve artifact smoke'u |
| `src/synthetic_data.py` | `SUPPORTED_COMPATIBILITY` / `LEGACY` | Sentetik dry-run verisini üretir |
| `src/artifacts.py` | `SUPPORTED_COMPATIBILITY` / `LEGACY` | Yalnız eski sentetik artifact yolu; Phase A manifestinin alternatifi değil |
| `scripts/run_rtmw_multiview_3d.py` | `SUPPORTED_COMPATIBILITY` | Eski komut adını aktif ViTPose girişine yönlendiren alias |
| `src/pose2d_estimator.py::RTMW2DEstimator` | `LEGACY` | Aktif config ve repository içi kanonik çağrıda kullanılmıyor |
| `src/pose3d_estimator.py::RTMW3DEstimator` | `LEGACY` | Opsiyonel tek-görüş helper; aktif multiview config'te kapalı |
| `scripts/run_partial_poomsae_engineering_trial.py` | `LEGACY` | Güncel kaynak-bağlı Poomsae application öncesi kısmi mühendislik yolu |
| eski partial-engineering scoring profile/config'leri | `LEGACY` | Güncel `poomsae1_trimmed` application profilinin alternatifi değildir |

RTMW ve sentetik uyumluluk kodu bu fazda değiştirilmez. Dış kullanıcı ve
deprecation dönemi denetlenmeden kaldırma yapılmaz.

## HISTORICAL_BENCHMARK bileşenleri

| Bileşen | Sınıf | Gerekçe |
| --- | --- | --- |
| `scripts/setup_mads_test.py` | `HISTORICAL_BENCHMARK` | MADS yerel benchmark hazırlığı |
| `scripts/calibrate_mads_vitpose_offsets.py` | `HISTORICAL_BENCHMARK` | Yalnız bağlı MADS deney/calibration yolu |
| `scripts/train_mads_vitpose_adapter.py` | `HISTORICAL_BENCHMARK` | Reddedilmiş adapter deneylerini tekrar üretme desteği |
| MADS config, manifest ve sonuç raporları | `HISTORICAL_BENCHMARK` | Yalnız kendi F2 koşusuna bağlı tarihsel ölçüm/provenance |

Bu bileşenlerin test veya provenance referansları vardır. “Aktif model değil”
olmaları silinmeleri için yeterli kanıt değildir.

## UNCERTAIN / RETAIN doğrudan yardımcılar

Pre-polish referans taramasında aşağıdaki script'ler için repository içinde
import, entry point, test, config, doküman veya output-provenance referansı
bulunmadı:

- `scripts/analyze_pose3d_stability.py`
- `scripts/export_pose3d_html_viewer.py`
- `scripts/inspect_session.py`
- `scripts/preview_camera_pair.py`
- `scripts/probe_videos.py`
- `scripts/render_poomsae_movement_overlay.py`
- `scripts/render_pose_stability_diagnostics.py`

Sınıf: `UNCERTAIN / RETAIN`.

Bu yalnız repository içi zero-reference bulgusudur; doğrudan komut satırı veya
repository dışı kullanımın olmadığı kanıtlanmamıştır. Bu fazda hiçbiri `DEAD`
sayılmaz, taşınmaz veya silinmez. Kaldırma için ayrı kullanım denetimi,
deprecation kararı ve gerektiğinde migration notu gerekir.

## Kanıtlanmış gereksiz backup dosyaları

Pre-polish denetimi yalnız şu eski yedeklerin gereksiz olduğunu kanıtladı:

- `config/scoring_config.yaml.bak`
- `config/scoring/rules/wt_recognized_2024-09-30.yaml.bak2`
- `src/poomsae_scoring/contracts.py.bak`

Çalışma ağacında zaten deletion durumundadırlar ve geri yüklenmemelidirler.
Bu bulgu başka dosyalar için toplu deletion yetkisi oluşturmaz.
