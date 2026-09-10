# TK3D ayrıntılı proje incelemesi ve geliştirme yol haritası

**İnceleme tarihi:** 5–6 Eylül 2026 (inceleme 5 Eylül'de başladı; son kontrol 6 Eylül'de tamamlandı)
**İncelenen çalışma alanı:** `C:\Users\WWWW\Desktop\tk3d`
**HEAD:** `da322a3b9d7bb9c2c03ac49a21e9c62d6f6d0d3c`
**Çalışma ağacı:** kullanıcı değişiklikleri ve çözülmemiş Git birleştirme kayıtları içeriyor.
**Raporun amacı:** mevcut sorunları kanıtlarıyla belirlemek; kısa, orta ve uzun vadeli işleri öncelik, bağımlılık ve kabul ölçütleriyle tanımlamak.
**6 Eylül eki:** Bölüm 16'ya proje eleştirileri, yapısal sorunlar ve gelişim öncelikleri eklendi. İlk inceleme manifesti ilk rapor sürümünü temsil eder; güncel raporun hash'i [ek kayıt manifestinde](C:/Users/WWWW/Desktop/tk3d/outputs/project_audit/runs/audit-20260906-critique-01/manifest.json) tutulur.
**10 Eylül düzeltme eki:** İlk denetimdeki teslim ve profil uyumsuzluğu bulgularının çözüm durumu Bölüm 17'de kayıt altına alındı. İlk bölümlerdeki test ve çalışma ağacı ifadeleri denetim anının tarihsel görüntüsüdür.

## 1. Yönetici değerlendirmesi

TK3D, yalnız bir pose modeli çalıştıran prototipten daha ileri bir araştırma altyapısına sahip. Kalibrasyon, WholeBody-133 sözleşmesi, ham veri koruma, RGB/depth karşılaştırması, global optimizasyon, kaynak-bağlı teşhisler, inceleme ekranı ve tekrarlanabilirlik kontrolleri önemli bir temel oluşturuyor. Buna rağmen **mevcut çalışma ağacı teslim kapısından geçmiyor**. Tam test koşusunda **381 passed, 3 failed** sonucu alındı; ayrıca belgelerde birleştirme işaretleri duruyor.

Temel öncelik yeni model veya daha çok kural eklemek değildir. Önce çalışma ağacını tutarlı hale getirmek, yaşam döngüsü ve veri sözleşmesi açıklarını kapatmak, ölçülen büyüklüklerle kural adları arasındaki anlamı düzeltmek gerekiyor. Sonraki büyük ihtiyaç bağımsız veri üzerinde ölçüm ve uzman doğrulaması. Daha fazla aday üretmek, daha çok landmarkı envantere yazmak veya jitter azaltmak tek başına daha doğru teknik analiz anlamına gelmez.

Bu inceleme **11 somut bulgu**, ayrıca ayrı ele alınması gereken mimari/ürün riskleri ve bilimsel doğrulama eksikleri belirledi. Bulguların bir kısmı izole fonksiyon veya hata enjeksiyonu seviyesinde yeniden üretildi; bunlar gerçek kayıtta aynı hatanın yaşandığı iddiası değildir. Her bulgunun bu sınırı aşağıda belirtilmiştir.

Önerilen sıra:

1. Birleştirme, profil/test/belge uyumu ve doğrudan multiview hata yönetimi.
2. Artifact, zaman çizelgesi, durum geçişi ve provenance doğrulaması.
3. Kural anlamları, açık eşik envanteri ve ölçüm belirsizliği.
4. M01–M18 gerçek veri kapsamı, uzman etiketleme ve bağımsız değerlendirme.
5. Ölçülmüş darboğazlara yönelik performans ve ürünleştirme.

## 2. İncelemenin kapsamı ve kanıt düzeyleri

AGENTS.md, PROJECT_STATUS.md, README, güncel teknik bağlam, paket/bağımlılık ve CI tanımları incelendi. Kaynak ve test envanteri çıkarıldı; multiview uygulama, triangulation, depth fusion, optimizer, senkronizasyon, artifact/run yönetimi, puanlama uygulaması, teknik doğruluk, readiness, review ve ground-truth değerlendirme yollarındaki kritik bölümler okundu. Tüm test paketi çalıştırıldı; seçili güvenlik sınırları ayrıca sınandı.

Git tarafından izlenen `src`, `scripts`, `tests` altında **169 benzersiz Python dosyası ve 46.850 satır** var. Bu sayılar testleri ve boş/açıklama satırlarını da içerir; test kapsam yüzdesi değildir. Envanter için Git’in unmerged dosyaları birden fazla listelemesi giderildi. İlk `probe_results.json` içindeki ham envanter bu yüzden yüksek çıkar; doğru envanter `repository_evidence.json` içindeki `unique_inventory` alanıdır.

Kanıt sınıfları:

| Sınıf | Bu raporda anlamı |
| --- | --- |
| Doğrudan doğrulandı | Komut/test/izole örnek gerçekten çalıştırıldı ve sonuç görüldü. |
| Koddan doğrulandı | Belirtilen kaynak yolundaki davranış veya eksik kontrol okundu. |
| Kaydedilmiş kanıt | Önceden üretilmiş yerel JSON yeniden okundu ve dosya hash’i alındı; inference tekrar edilmedi. |
| Risk/öneri | Tasarım veya kullanım sınırından çıkarım; sahada gerçekleşmiş hata olarak sunulmuyor. |

Yeni GPU/ZED inference, bağımsız mocap karşılaştırması, insan/hakem pilotu, gerçek tarayıcı video oynatma testi veya temiz makinede wheel kurulumu yapılmadı. Tüm bağımlılıkların güvenlik/lisans denetimi ve güncel federasyon kural kaynağı doğrulaması da bu kod incelemesinin parçası değildi. Bu rapor kapsamlı bir mühendislik incelemesidir; olası bütün hataların tüketildiği garantisi değildir.

İnceleme kaynak kodunu veya mevcut profilleri düzeltmedi. Başlangıçtaki kullanıcı değişiklikleri korunmuştur. Commit/push yapılmamıştır.

### 2.1. Gerçekten çalıştırılan kontroller

| Kontrol | Sonuç | Yorum |
| --- | --- | --- |
| `git status --short` | 5 mevcut değişmiş/çakışmış yol | Teknik profil, iki belge, teknik doğruluk modülü, ilgili test dosyası. |
| `git pull --ff-only` | Başarısız | Unmerged dosyalar nedeniyle Git pull yapmadı; ağ/uzak dal güncelliği doğrulanmış sayılmaz. |
| Ruff: `src scripts tests` | Geçti | Python sözdizimi/lint kapısı temiz. |
| Tam pytest | **381 passed, 3 failed, 92.10 s** | Mevcut kirli çalışma ağacına aittir; HEAD-only sonucu değildir. |
| `python -m pip check` | `No broken requirements found` | Kurulu paket metadata tutarlılığı; sayısal sonuç doğruluğu değildir. |
| `check_reproducibility.py` | Tier 1 ve CURRENT_ACTIVE `READY` | Yerel varlık/ortam kontrolü; başarısız testleri veya resmi puan doğruluğunu kapsamaz. |
| `git diff --check` | Başarısız | İki belgede toplam altı çatışma işareti. |
| İzole sözleşme/lifecycle örnekleri | Açıklar yeniden üretildi | B03–B09 için aşağıdaki etki sınırlarına bakılmalı. |
| Teknik ölçüm anlamı örneği | Üç farklı sabit el yüksekliğinde `true` | B11; yalnız ilgili geometri evaluator’ı sınandı. |

### 2.2. Kanıt dosyaları

Bu incelemenin çıktıları şu benzersiz dizindedir:

`C:\Users\WWWW\Desktop\tk3d\outputs\project_audit\runs\audit-20260905-01\`

- [probe_results.json](C:/Users/WWWW/Desktop/tk3d/outputs/project_audit/runs/audit-20260905-01/probe_results.json): izole API ve hata enjeksiyonu sonuçları.
- [repository_evidence.json](C:/Users/WWWW/Desktop/tk3d/outputs/project_audit/runs/audit-20260905-01/repository_evidence.json): doğru benzersiz dosya envanteri, güncel kural dağılımı, kaydedilmiş run ölçümleri ve hash’ler.
- [semantic_probe_results.json](C:/Users/WWWW/Desktop/tk3d/outputs/project_audit/runs/audit-20260905-01/semantic_probe_results.json): hedefe ulaşma ile sabitlik ayrımını gösteren örnek.
- [environment.json](C:/Users/WWWW/Desktop/tk3d/outputs/project_audit/runs/audit-20260905-01/environment.json): gerçekten çalıştırılmış ortam kontrolü.
- [audit_manifest.json](C:/Users/WWWW/Desktop/tk3d/outputs/project_audit/runs/audit-20260905-01/audit_manifest.json): rapor ve kanıt dosyalarının hash'leri, test sonuç özeti, son kaynak kimlikleri ve rapor bağlantı kontrolü.
- Aynı dizindeki `audit_probes.py`, `collect_evidence.py`, `semantic_probe.py` örneklerin kaynaklarıdır. Bazıları aynı run adlarını kullandığından ikinci kez aynı dizinde çalıştırılmamalıdır; yeniden üretimde yeni run dizini kullanılmalıdır.

## 3. Mevcut sistemin gerçek durumu

### 3.1. Mimari ve korunması gereken güçlü taraflar

Kanonik hat iki ZED 2i RGB/SVO2 kaydından RF-DETR + ByteTrack ile kişiyi seçiyor, ViTPose-Huge ile 133 nokta üretiyor, kalibrasyonlu triangulation yapıyor, BODY-17 için depth ve global optimizasyon uyguluyor, ardından teknik analiz ve inceleme çıktıları oluşturuyor.

Korunması gerekenler:

- **WholeBody-133 ve metre/eksen sözleşmesi:** BODY-17 optimizasyonu el, yüz ve ayak exportunu küçültmüyor.
- **Görüntü kanıtının ayrılması:** cross-view izdüşümünün bağımsız ölçüm sayılmaması ve hedef kameranın kendi öncülünden çıkarılması doğru bir güvenlik sınırı.
- **Alternatif sonucun kaliteyle kabulü:** depth adayının RGB referansa karşı kontrolü, optimizer fallback’i ve ham verinin korunması.
- **Kaynak-bağlı analiz:** pose/timeline/config bağları ve review kimliği, yanlış run’a etiket taşıma riskini azaltıyor.
- **Resmî puan sınırı:** aktif kaynak-bağlı karar ve readiness katmanları resmî skoru kapalı tutuyor.
- **Kullanışlı test altyapısı:** hata senaryoları, sözleşmeler, triangulation, video süreleri, review JSON ve geometri için çok sayıda test mevcut.
- **Dürüst workflow ayrımı:** CURRENT_ACTIVE, CURRENT_VALIDATION, HISTORICAL_BENCHMARK ve LEGACY farklı bağlamlar olarak tanımlanmış.

Bu temeli büyük bir yeniden yazımla değiştirmek yerine açıkları dar değişikliklerle kapatmak daha düşük risk taşır.

### 3.2. Kaydedilmiş gerçek kayıt kanıtı

Referans run ve latest’in gösterdiği `post-polish-full-regression-20260828-021518` kalite JSON’larında şu değerler yeniden okundu:

| Ölçüm | Kaydedilmiş değer | Sınırı |
| --- | ---: | --- |
| Kare sayısı | 741 | Seçili kısa kayıt; tam 18 hareketlik performans değil. |
| Geçerli BODY-17 oranı | %97,8884 | Tüm 133 noktanın aynı kalitede olduğu anlamına gelmez. |
| Gözlenen BODY-17 oranı | %96,6024 | Zamansal geri kazanımla ayrı tutulmalı. |
| Zamansal geri kazanılan BODY-17 oranı | %1,2860 | Yeni bağımsız kamera gözlemi değildir. |
| Ortalama reprojection | 5,20961 px | Mutlak 3B doğruluk veya hakem kararı doğruluğu değildir. |
| Dış ground truth değerlendirmesi | `false` | Aktif ZED kaydı için dış doğruluk iddiası desteklenmiyor. |
| Resmî puan hazırlığı | `false` | Doğru biçimde kapalı. |

Stabilite JSON’unda açı yüksek-frekans medyanı yaklaşık **0,05086°**, bu rapordaki son stabilizasyon karşılaştırmasının azaltımı **%0** ve medyan/p95 düzeltmesi **0 mm**. Bu, ilgili son aşamanın o raporda ek değişiklik yapmadığını gösterir; tüm pipeline’ın hiç filtre uygulamadığı veya hareketin fiziksel olarak bu kadar doğru olduğu sonucu çıkarılamaz.

`run_quality_report.json` içindeki `max_reprojection_error_px: 25.0` alanı uygulamada eşik değişkeninden yazılıyor. Bunu gözlenen dağılımın maksimumu diye sunmamak gerekir. Rapor şemasında `configured_max_reprojection_error_px` gibi daha açık ad veya açıklama tercih edilmeli; gözlenen p50/p95/p99/max ayrı alanlar olmalı.

`benim-denemem-21` kaydedilmiş raporu **75 aktif, 74 yalnız ölçüm, 17 referansla bloke, 8 gözlenemeyen** kural ve **82 puansız aday** içeriyor. Denetim anındaki çalışma profilinin dağılımı ise **74 aktif, 75 yalnız ölçüm, 17 bloke, 8 gözlenemeyen** idi. Eski rapor farklı bir profil çalıştırılmış gibi yorumlanmamalıdır; 10 Eylül birleşik profil durumu Bölüm 17'dedir.

### 3.3. Hazırlık durumlarının ayrılması

| Alan | Değerlendirme |
| --- | --- |
| Yerel varlık ve kurulum | Kontrol komutu READY; kurulu bağımlılıklar tutarlı. |
| Güncel teslim kapısı | Başarısız: testler ve Git diff kontrolü temiz değil. |
| Tek oturum iç geometri kanıtı | Kaydedilmiş başarılı sonuç mevcut. |
| Yeni sporcu/kamera/oturum genellemesi | Bu incelemede doğrulanmış değil. |
| Tam Taegeuk 1 gerçek veri kapsamı | M07–M18 için aktif doğrulanmış kayıt kanıtı yok. |
| Uzman/hakem doğruluğu | Bağımsız etiketli pilot yok. |
| Resmî puanlama | Açılması için yeterli kanıt yok. |

## 4. Somut bulgular

Öncelik tanımı: **P1** mevcut teslimi veya önemli davranış güvenilirliğini etkiler; **P2** belirli girdilerde sözleşme/izlenebilirlik açığıdır; **P3** erken hata yakalama ve bakım iyileştirmesidir. Öncelikler gerçek dünyada kanıtlanmış zarar iddiası değildir.

| ID | Öncelik | Bulgu | Kanıt |
| --- | --- | --- | --- |
| B01 | P1 | Birleştirme tamamlanmamış; belge çatışmaları duruyor | Git status/pull/diff |
| B02 | P1 | Aktif kural envanteriyle test ve belgeler uyuşmuyor | 3 pytest hatası, profil diff’i |
| B03 | P1 | Doğrudan multiview hatasında run `running` kalıyor | Snapshot hata enjeksiyonu |
| B04 | P2 | Ana 3B sözleşmesi zaman/kare değerlerinin anlamını denetlemiyor | Dört geçersiz örnek kabul edildi |
| B05 | P2 | Artifact–manifest model-config hash eşitliği kontrol edilmiyor | İzole binding örneği |
| B06 | P2 | Lifecycle geçişleri başarısız run’ı tamamlanmışa çevirebiliyor | `failed → completed` örneği |
| B07 | P2 | Latest okuyucusu lifecycle’sız/kimliği uyuşmayan hedefi kabul ediyor | İzole latest örneği |
| B08 | P2 | Optimizer kabul kapısında eksik metrikler başarı gibi görünebiliyor | Eksik-metrik örneği |
| B09 | P3 | Model config doğrulayıcı bazı non-finite eşikleri kabul ediyor | `NaN` reprojection eşiği örneği |
| B10 | P2 | Aktif boolean kararların fiziksel eşikleri koda gömülü | Kaynak/config karşılaştırması |
| B11 | P1 | “Hedefe ulaştı” kontrolü yalnız sabitliği ölçüyor | Kaynak ve izole geometri örneği |

### B01 — Birleştirme tamamlanmamış

Başlangıçta `docs/ARCHITECTURE_DECISIONS.md`, `docs/TECHNICAL_ACCURACY_DIAGNOSTICS.md` ve `src/poomsae_scoring/technical_accuracy.py` Git indeksinde `UU` durumundaydı. Python dosyasında metinsel çatışma işareti kalmamış, dolayısıyla Ruff geçebiliyor; bu Git’in birleştirmeyi çözülmüş saydığı anlamına gelmiyor. İki belgede altı işaret gerçekten mevcut.

**Etki:** pull ilerlemiyor; teslim kontrolü bozuk; belgeler aynı anda iki farklı tasarım durumunu anlatıyor. Bu incelemenin sonucu remote HEAD’in güncelliğini kanıtlamıyor.

**Yapılacak:** iki tarafın değişikliklerini anlam açısından birleştir; kullanıcı değişikliklerini koru; yalnız ilgili dosyaların çözümünü tamamla. Commit/push ayrı kullanıcı talebine bağlı kalmalı.

**Kabul:** `git ls-files -u` boş, çatışma işareti yok, diff kontrolü temiz, ilgili test ve tam teslim kapısı geçiyor.

Kaynak: [mimari çatışma](C:/Users/WWWW/Desktop/tk3d/docs/ARCHITECTURE_DECISIONS.md:615), [teknik belge çatışması](C:/Users/WWWW/Desktop/tk3d/docs/TECHNICAL_ACCURACY_DIAGNOSTICS.md:159).

### B02 — 75/74 kural uyuşmazlığı ve eski teslim kanıtı

Üç başarısız test `75` aktif kural veya toplam `88` değerlendirilen metrik bekliyor; gerçek değerler `74` ve `87`. Kök neden `head_torso_settle_offset` kuralının aktif listeden çıkarılıp açıklamasıyla measurement-only yapılması. Gerekçe, settle anını hesaplayan **10°** sabitinin profilde olmaması. Değişiklik güvenli yönde olabilir; sırf test sayısını tutturmak için yeniden etkinleştirmek doğru çözüm değildir.

**Etki:** PROJECT_STATUS içindeki 371 geçmiş başarılı test, 75 aktif kural ve 980 vaka bilgisi güncel ağacın teslim kanıtı olarak kullanılamaz. Bu sayılar geçmiş run’a ait olarak korunabilir.

**Yapılacak:** beklenen davranışta karar ver; test envanteri, profil sürümü ve güncel belgeleri birlikte güncelle. Semantik kapsamı sabitleyen testleri koru; yalnız testleri gevşeterek yeşile dönme.

**Kabul:** gerçek kural dağılımı rapor/test/belgede aynı; sınır/eksik/geçersiz değer vakaları yeniden üretilmiş; tam pytest sıfır hatalı.

Kaynak: [profil](C:/Users/WWWW/Desktop/tk3d/config/scoring/engineering/taegeuk_1_wholebody_diagnostics_v3.yaml:529), [envanter testi](C:/Users/WWWW/Desktop/tk3d/tests/test_technical_accuracy.py:46), [validation testi](C:/Users/WWWW/Desktop/tk3d/tests/test_technical_accuracy_validation.py:74).

### B03 — Doğrudan multiview yaşam döngüsü hatası

`run_multiview_pose` run oluşturup `running` yazdıktan sonra config snapshot alıyor. Bütün fonksiyonu kapsayan hata/kesinti yakalama ve failed yazımı yok. İzole denemede `snapshot_file` bir OSError üretti; ilgili yeni run’ın durumu **running** kaldı. Model yüklenmedi veya inference yapılmadı. CLI yalnız `MultiviewQualityError` yakalıyor; bu genel açığı kapatmıyor.

Poomsae üst uygulamasında ayrı bir hata sarmalayıcısı var. Bu, doğrudan `tk3d-multiview`/uygulama API’sine aynı güvenceyi vermiyor.

**Etki:** bitmiş/hatalı iş çalışıyor sanılabilir; otomasyon, kullanıcı arayüzü ve temizlik politikaları yanlış karar verebilir. Kaynakların serbest bırakılması da tek bir yaşam döngüsü sahibiyle yönetilmeli.

**Yapılacak:** yeni run’ın sahipliğini alan dış uygulama katmanında `Exception` ve `KeyboardInterrupt` yönet; failed yazarken asıl hatayı koru. Capture/model/writer temizliğini `finally`/context manager ile garanti et. Kalite reddi ile işlem hatası için açık terminal durumları tanımla.

**Kabul:** snapshot, video açma, model yükleme, inference, export ve kesinti hatalarının her biri doğru terminal durum bırakır; eski run ve latest değişmez.

Kaynak: [multiview başlangıcı](C:/Users/WWWW/Desktop/tk3d/src/multiview_application.py:235), [CLI](C:/Users/WWWW/Desktop/tk3d/scripts/run_vitpose_multiview_3d.py).

### B04 — Zaman ve kare kimliği doğrulaması eksik

`validate_main_3d_artifact`, `frame_indices` ve `timestamps_sec` için yalnız liste uzunluğunu denetliyor. Fixture’dan türetilen metin timestamp, `null` timestamp, negatif frame index ve kesirli frame index örneklerinin tamamına `CURRENT` döndü.

**Etki:** bozuk artifact erken reddedilmeyebilir; downstream dönüşümler geç hata verir veya zaman/indeks bilgisini yanlış yorumlar. Bu deneme tüm downstream akışların bu girdileri kabul ettiğini kanıtlamaz; ortak giriş sözleşmesindeki boşluğu gösterir.

**Yapılacak:** frame indekslerine gerçek integer ve artış/tekrar politikası; zamanlara finite sayı ve monotonluk; FPS/örnekleme ile tutarlılık; kalite dizilerine şekil/değer denetimi ekle. Negatif ortak zamanın geçerli olduğu senaryolar varsa zamanın negatifliği için kör yasak yerine açık sözleşme kur.

**Kabul:** yanlış tip, bool, null zaman, duplicate/reversed zaman, kesirli frame ve kalite şekil uyuşmazlığı ayrı testlerle reddedilir; geçerli offset/stride örnekleri korunur.

Kaynak: [artifact doğrulaması](C:/Users/WWWW/Desktop/tk3d/src/artifact_contracts.py:53), [yalnız uzunluk kontrolü](C:/Users/WWWW/Desktop/tk3d/src/artifact_contracts.py:160).

### B05 — Model-config provenance bağı tamamlanmamış

`provenance.model_config_sha256` zorunlu bir metin alanı. Ancak `validate_artifact_manifest_binding` session, run ve calibration eşitliklerini kontrol ederken bunu manifestin model-config hash’iyle karşılaştırmıyor. İzole örnekte farklı model-config hash’i kabul edildi.

**Etki:** calibration doğru olsa bile artifact’in beyan ettiği model yapılandırması ile manifestin yapılandırması uyuşmayabilir. Manifest snapshot dosyalarının hash kontrolü mevcut; eksik olan artifact beyanıyla aralarındaki eşitliktir.

**Yapılacak:** kanonik config anahtarından model hash’ini eşleştir; hash formatını doğrula; model/adapter kimlikleri için de hangi bağların zorunlu olduğunu açıklaştır.

**Kabul:** artifact veya manifest model hash’inin tek başına değiştirilmesi load aşamasında reddedilir; valid eski format desteği açıkça sınıflandırılır.

Kaynak: [manifest bağı](C:/Users/WWWW/Desktop/tk3d/src/artifact_contracts.py:109).

### B06 — Terminal durumlar korunmuyor

`_update_run_state` kimliği doğruluyor fakat eski durumdan yeni duruma geçişi doğrulamıyor. İzole örnekte failed run’a `mark_run_complete` çağrılınca durum **completed** oldu ve test kökündeki latest yazıldı. Bu işlem üretim latest’ine dokunmadı.

**Etki:** API’yi yanlış sırayla kullanan gelecekteki kod başarısız işi başarılı gösterebilir. Tamamlanmış işin tekrar running/failed yapılmasına da açık bir geçiş politikası yok.

**Yapılacak:** durum makinesi tanımla. `preparing → running → completed/failed` ana yol; yeniden deneme yeni run kimliğiyle olmalı. Üst Poomsae işi ile multiview alt aşamasının aynı run içinde geçici completed kullanması ayrıca tasarlanmalı; doğrudan geçiş kilidi eklemek mevcut bileşimi bozabilir.

**Kabul:** geçersiz geçişler reddedilir; üst iş tamamlanmadan alt aşama tüm run’ı başarıyla kapatmaz; eşzamanlı durum yazma test edilir.

Kaynak: [run durum güncellemesi](C:/Users/WWWW/Desktop/tk3d/src/run_outputs.py:103), [Poomsae aşama bağlama](C:/Users/WWWW/Desktop/tk3d/src/poomsae_scoring/application.py:207).

### B07 — Latest çözümleyicisi eksik bağlamı kabul ediyor

`resolve_latest_run`, state dosyası varsa completed olmasını denetliyor; state yoksa hedefi kabul ediyor. Marker içindeki `run_id` ile hedef klasör/state kimliği eşitliği de zorunlu değil. İzole örnekte boş run klasörü ve farklı marker ID’si kabul edildi.

**Etki:** bozulmuş veya elle yanlış düzenlenmiş marker güvenilir tamamlanmış run gibi çözümlenebilir. **Mevcut gerçek latest’in bozuk olduğu bulunmadı**; yeniden okunan hedefin lifecycle’ı completed.

**Yapılacak:** güncel formatta state, session/run eşitliği, tam beklenen klasör ve gerekli artifact/kalite kontrolünü zorunlu kıl. Tarihsel lifecycle’sız run desteği gerekiyorsa açık legacy modu olarak ayır.

**Kabul:** eksik state, yanlış run/session, alt-alt klasör, eksik sonuç ve failed hedef reddedilir; geçerli legacy aktarımı sessiz gerçekleşmez.

Kaynak: [latest çözümleme](C:/Users/WWWW/Desktop/tk3d/src/run_outputs.py:86).

### B08 — Optimizer kapısı eksik metriklerde geçebiliyor

Optimizer `_acceptance_gate`, before/after metrikleri `None` ise ilgili kıyaslamayı atlıyor. `solver_success=True`, boş correction ve eksik reprojection/acceleration/bone değerleri verilen izole örnekte `passed=True` ve bütün check alanları true çıktı.

**Etki sınırı:** bu, iç fonksiyonun eksik kanıtı başarıya çevirebildiğini doğrular; mevcut gerçek optimizer run’ının eksik metrikle geçtiği gösterilmedi. Üst fonksiyonun giriş/observasyon kontrolleri bazı yolları zaten engelliyor.

**Yapılacak:** her zorunlu metrik için `evaluated/passed/reason` ayrımı; gereken kanıt yoksa fallback. Her ölçütün her kısa sekansa uygulanabilir olduğunu varsayma; koşullu ölçütler açıkça `not_applicable` olmalı.

**Kabul:** boş/all-invalid/nan aday, yetersiz temporal pencere ve kaybolan reprojection kanıtı “başarılı” sayılmaz; sağlam sonuçlar aynı kalır.

Kaynak: [optimizer kapısı](C:/Users/WWWW/Desktop/tk3d/src/multiview_pose_optimization.py:974).

### B09 — Non-finite yapılandırma erken reddedilmiyor

`triangulation.max_reprojection_error_px=NaN` verilen model config, `validate_model_config` tarafından kabul edildi; `<= 0` kontrolü NaN’ı yakalamıyor. Alt triangulation fonksiyonu finite kontrolü yapıyor; bu örnekte sorun öncelikle geç hata ve hazırlık kontrolüyle runtime’ın uyuşmaması.

**Yapılacak:** tüm sayısal config alanlarını finite/tip/aralık bakımından ortak yardımcılarla denetle; bool→int ve kesirli sayı→int sessiz dönüşümlerini de taramaya dahil et.

**Kabul:** YAML `.nan`, `.inf`, yanlış tip ve sınır dışı değerler run yaratılmadan açıklayıcı alan adıyla reddedilir.

Kaynak: [model config doğrulaması](C:/Users/WWWW/Desktop/tk3d/src/config_validation.py:91).

### B10 — Boolean çıktılar fiziksel eşikleri gizliyor

`_hand_settled` için **0,05**, `_premature_transition` için **0,08**, `_fixation_stable` için **0,05/0,04** sabitleri kodda. Profil boolean beklenen sonucu tanımlıyor; bu sonucu üreten bütün fiziksel karar sınırlarını taşımıyor. `head_torso_settle_offset` için yapılan measurement-only geçişi benzer sorunu yalnız bir ölçütte ele almış.

**Etki:** profil sürümü/hash’i tek başına eşik politikasını anlatmıyor; hassasiyet analizi zorlaşıyor. Numeric ve boolean kurallar aynı ölçümü farklı belirsizlik politikalarıyla kullanabilir. Örneğin late correction için profilde 0,08 ve uncertainty band varken premature boolean doğrudan 0,08 kıyası yapıyor.

**Yapılacak:** kullanılan fiziksel değer, normalizasyon birimi, operatör, belirsizlik bandı ve gerekçeyi profile taşı; boolean sonucun yanında ham ölçümü göster. Aynı fiziksel olaydan çıkan ilişkili adayları ayrıca işaretle.

**Kabul:** aktif kararların tamamı config üzerinden açıklanabilir; gizli karar sabiti envanteri yok; sınır-belirsiz örnekler benzer kurallar arasında çelişkili kesinlik üretmez.

Kaynak: [geometri yardımcıları](C:/Users/WWWW/Desktop/tk3d/src/poomsae_scoring/technical_accuracy_metrics.py:979), [profil eşikleri](C:/Users/WWWW/Desktop/tk3d/config/scoring/engineering/taegeuk_1_wholebody_diagnostics_v3.yaml:106).

### B11 — Sabitliği “hedefe ulaştı” diye yorumlama

`active_technique_reaches_target_by_fixation`, `_hand_settled` sonucundan üretiliyor. Bu fonksiyon el merkezinin zamansal dağılımını ölçeğe bölüyor; hedef yüksekliği, hedef yönü veya teknik kontratını almıyor. İzole örnekte el yüksekliği 0,3 m, 1,3 m veya 2,0 m olduğunda el sabit kaldığı için sonuçların üçü de true.

**Etki:** kontrol adı ve anlamı ölçümün desteklediğinden fazlasını söylüyor. Yanlış yerde sabit kalan el bu tek kontrol altında hedefe ulaşmış gibi görünebilir. Diğer hedef-hata metrikleri ayrıca aday üretebilir; bu bulgu toplam sistemin bütün yanlış teknikleri kaçırdığını veya resmî skor verdiğini göstermez.

**Yapılacak:** ölçümü `hand_settled_by_fixation` benzeri gerçek anlamıyla sun; hedefe ulaşmayı teknik hedef hatası + zaman penceresi + kanıt kalitesiyle ayrı kontrol et. Hedef referansı yoksa ölçülemez/measurement-only kalmalı.

**Kabul:** doğru hedefte sabit, yanlış hedefte sabit, doğru hedefe geç gelen ve geçersiz kanıtlı dört örnek farklı sonuç üretir; kural açıklaması ve review ekranı bunu tutarlı gösterir.

Kaynak: [metriğin bağlanması](C:/Users/WWWW/Desktop/tk3d/src/poomsae_scoring/technical_accuracy_metrics.py:182), [sabitlik hesabı](C:/Users/WWWW/Desktop/tk3d/src/poomsae_scoring/technical_accuracy_metrics.py:945).

## 5. Mimari, entegrasyon ve ürün riskleri

Bu bölümdeki maddeler gerçekleşmiş üretim hatası olarak sayılmamıştır.

### 5.1. Çalıştırılan kodun tam kimliği eksik

Run manifesti commit SHA ve `dirty` boolean kaydediyor; kirli değişikliklerin diff’i veya kaynak dosya hash manifesti yok. Kaydedilmiş latest run manifestinde gerçekten `dirty:true` bulunuyor. Aynı commit ve dirty=true, farklı iki kod kümesini ayıramaz. Ayrıca manifest üretimi çalışmanın sonuna yakın olduğundan süreç sürerken kaynak değişmesi ayrı bir risk.

Öneri: başlangıçta çalışma kodu kimliğini dondur; temiz commit veya kaynak hash manifesti kullan. Yerel araştırma için dirty diff özeti/hash’i sakla; dosya içeriği saklanacaksa sır/veri sınırını koru. Analiz-only Poomsae çalışmasına da kendi code/environment provenance’ını ekle; yalnız kaynak pose run’ının ortamını miras almak güncel analiz kodunu tanımlamaz.

Kaynak: [git_provenance](C:/Users/WWWW/Desktop/tk3d/src/run_manifest.py:175), [Poomsae özet bağları](C:/Users/WWWW/Desktop/tk3d/src/poomsae_scoring/application.py:799).

### 5.2. Büyük girişler metadata ile tanımlanıyor

512 MiB üzerindeki giriş dosyaları manifestte boyut/mtime ile, checksum olmadan kaydediliyor. Bu açıkça belgelenmiş bir performans tercihidir; hata diye gizlenmemiştir. Ancak büyük SVO2/video’nun içerik eşitliğini boyut ve mtime kanıtlayamaz.

Öneri: ingest sırasında bir defalık tam hash, doğrulanmış asset kimliği ve sonraki run’larda yeniden kullanılabilir hash kaydı. Her run’da çok gigabayt okumak zorunlu olmadan güçlü içerik kimliği sağlanmalı. Yeniden yerleştirme/import akışı hash’i doğrulamalı.

### 5.3. Review videoları ile analiz girdilerinin eşlemesi

Poomsae profilindeki video listesi session’daki kamera listesinden bağımsız. Profil doğrulaması video etiketlerinin benzersizliğini kontrol ediyor; kamera ID benzersizliği ve session kamera/video eşleşmesi aynı güçte bağlanmıyor. Browser proxy üreticisi giriş/çıkış hash’lerini yazıyor, ancak doğru kaynak videonun seçildiğini baştan kanıtlamak ayrı iştir.

Ayrıca inceleme videolarını aynı `currentTime` ile oynatmak, farklı başlangıç offset’i olan genel session’lar için tek başına senkronizasyon sağlamaz. Aktif trim zaten ortak zaman çizelgesine hazırlanmış; bu bulgu aktif videonun yanlış senkron olduğu anlamına gelmez.

Öneri: seçilen kamera kümesini session + pose provenance ile eşleştir; ortak zaman→kamera karesi dönüşümünü review’a taşı veya yalnız önceden hizalanmış proxy kabul et. Kamera ID çakışması ve yanlış video hash’i için preflight ekle.

### 5.4. Birden çok kişi için kamera-arası kimlik güvencesi yok

ByteTrack kimliği kamera içinde tutuluyor. İlk seçim ve reacquire confidence/alan üzerinden yapılıyor. Kamera A ve B’de seçilen kişilerin aynı sporcu olduğunu doğrulayan açık global kimlik sözleşmesi görülmedi. Tek sporcunun kontrollü kaydında bu kabul edilebilir; antrenör/seyirci/sahaya giriş durumunda genelleştirilemez.

Öneri: kısa vadede tek-sporcu çekim sınırını preflight/review’da açıklaştır; orta vadede ROI/başlangıç hedef seçimi, epipolar/geometrik kimlik tutarlılığı ve şüpheli durumda durma ekle. Çoklu-sporcu ürün talebi yoksa kapsamlı re-identification sistemi hemen gerekli değildir.

Kaynak: [kişi seçimi](C:/Users/WWWW/Desktop/tk3d/src/person_tracking.py:123).

### 5.5. Hakem kaydı ile doğrulanmış karar yetkisi farklı kavramlar

Çalışma ağacındaki yeni `judge_source` yolu ad, unvan, tarih ve approval_reference metinlerini doğruluyor; bunlar doğrulanmış dış onay belgesinin içeriğine kriptografik bağ veya bağımsız hakem performans kalibrasyonu değildir. Aktif `judge_validated_rules` listesi **boş**; mevcut profil üzerinden böyle bir kesinti etkinliği bulunmadı.

Bu özelliği kullanmadan önce kural versiyonu, kaynak belge hash’i, kapsam, geçerlilik durumu, onayı veren yetki ve deneysel hakem kalibrasyonu ayrı tutulmalı. Metin alanı doldurulması otomatik olarak bilimsel doğruluk kazanımı sayılmamalı. Teknik diagnostics ile kaynak-bağlı Accuracy arasında skor sahipliği net kalmalı.

Ayrıca [scoring_authorization.py](C:/Users/WWWW/Desktop/tk3d/src/scoring_authorization.py) içindeki eski/genel yetkilendirme yolunda `official_scoring_ready`, yapılandırma boolean’ı `poomsae_rules_validated` ile ilişkilendiriliyor. Kanonik readiness bunu false tutuyor. Ürünleşmede bu farklı readiness anlamları tek politika altında gözden geçirilmeli; aktif resmî puan açığı olarak yorumlanmamalıdır.

### 5.6. Paketleme halen repository merkezli

Paket metadata’sı `src`, `scripts` paketlerini içeriyor; varsayılan config’ler repository `config/` ağacına, ROOT ise kaynak dosyanın üst dizinine dayanıyor. CI editable kurulum ve repository içinden komut çalıştırıyor. Bu, bağımsız wheel’in farklı bir dizinden tam çalıştığını kanıtlamaz.

Öneri: hedef yalnız repo içi araştırmaysa bunu açık tut. Kurulabilir uygulama hedeflenirse config/resource paketleme, kullanıcı veri kökü ve çalışma kökü ayrımı, `importlib.resources` veya eşdeğer kaynak çözümleme ve repo dışı wheel smoke ekle. Python `>=3.11` beyanının desteklenen sürüm matrisiyle ilişkisini de netleştir.

### 5.7. Uzun modüller ve sunum bağımlılığı bakım maliyetini artırıyor

Örnek boyutlar: wholebody diagnostics 2.085, multiview application 1.823, technical accuracy 1.797, review report 1.287, Poomsae application 1.027 satır. Boyut tek başına kusur değildir; fakat yaşam döngüsü, hesaplama, serialization ve görselleştirme sorumluluklarının birleşmesi dar bir hatanın geniş etkisini artırıyor.

Poomsae uygulamasında browser video oluşturma analitik aşamalardan önce geliyor. Bir presentation/transcode hatası analiz üretimini de durdurabiliyor. Öneri: ölçüm/karar ile sunum aşamalarına ayrı durum ve yeniden üretim imkânı; bağımlılık grafiği üzerinden başarısız aşamayı açıklama. Yeni mimari önce mevcut davranış regresyonuyla korunmalı; mikroservis ayrımı gerekmiyor.

### 5.8. Sabit kayıt açıklamaları yeni veriyle yanlış hale gelebilir

Review ve teknik rapor metinlerinde “kayıt M06’da bitiyor/M07–M18 yalnız config kapsamı” gibi aktif session’a özgü açıklamalar bulunuyor. Aktif kayıt için doğru; genel profil desteğine geçildiğinde coverage alanından türetilmeli. Kullanıcıya yeni bir tam kayıt için eski sınırlama metni gösterilmemeli.

### 5.9. Kalite oranları tek başına yeterli değil

Grup kalite kapısı segment boyunca ortalama landmark oranı kullanıyor. Uzun bir iyi bölüm, kararın tam anchor anındaki yerel eksikliği gizleyebilir; helper’ların finite kontrolleri bir miktar koruma sağlar. Bunu mevcut hatalı sonuç iddiası olarak değil, ek test ihtiyacı olarak görmek gerekir.

Öneri: kuralın gerçekten kullandığı pencere için sample sayısı, joint başına geçerlilik, kesintisiz boşluk uzunluğu ve doğrudan kamera kanıtını raporla. `measured`, `recovered`, `interpolated`, `visualization_only` ayrımı kullanıcıya gerektiği yerde taşınmalı.

## 6. Bilimsel ve teknik doğrulama programı

### 6.1. Önce üç ayrı doğruluk sorusunu cevapla

1. **2B görüntü ölçümü doğru mu?** Bağımsız kamera annotation’ına göre nokta hatası/başarı oranı; görünürlük ve occlusion ayrı.
2. **3B geometri doğru mu?** Senkron bağımsız 3B ölçüme göre global MPJPE, p95, açı ve türev hataları; calibration/ölçek hatalarını saklayan hizalamalar ayrı diagnostic olarak.
3. **Teknik karar doğru mu?** Uzmanlarca etiketlenmiş olaylara göre yanlış pozitif, kaçırılan hata, precision/recall ve belirsiz kalma oranı.

Bu üç katmanın hiçbirini diğerinin metriğiyle ikame etmemek gerekiyor. İç reprojection geçişi dış 3B ground truth değildir; 3B doğruluğu da kuralın doğru teknik anlam taşıdığını kanıtlamaz.

### 6.2. Veri toplama ve ayrım

Pilot veri farklı sporcu, boy/vücut oranı, beceri seviyesi, kıyafet, ışık, kamera mesafesi/açısı, hız ve occlusion içermeli. Bütün 18 hareket, geçişler, başlangıç/bitiş ve kasıtlı kontrollü teknik varyasyonlar kaydedilmeli. Tek bir videonun çok sayıda karesini bağımsız örnek sayarak genelleme iddiası kurulamaz.

Eşik geliştirme, doğrulama ve kör test kümelerini **sporcu/oturum bazında** ayır. Aynı sporcunun komşu karelerinin eğitim ve testte karışmasını engelle. İlk pilot örnek sayısı bütçe ve etiketleme kapasitesiyle belirlenebilir; bunu yeterlilik kanıtı yerine iş planı olarak sun. Nihai örnek sayısını olay sıklığı ve hedef güven aralığına göre belirle.

### 6.3. Hakem/uzman annotation tasarımı

- Sistem adayını görmeden bağımsız ilk etiket; ardından gerekirse sistemle karşılaştırma.
- En az iki bağımsız inceleyici ve anlaşmazlık çözümü; tek uzman görüşünü mutlak ground truth olarak sunmama.
- Hareket/faz, eklem/teknik hata türü, şiddet, görülebilirlik ve eminlik ayrı alanlar.
- Sistem adayları yanında aday üretmeyen örnekler de incelenmeli; yalnız aday doğrulama precision hakkında bilgi verir, recall için kaçırılan olaylar gerekir.
- “Ölçülemiyor” ile “doğru teknik” farklı etiket olmalı.
- Kaynak video hash’i, sporcu/oturum anonim ID’si, annotation sürümü, reviewer ve tarih korunmalı.

### 6.4. Ölçüm belirsizliği

Mevcut uncertainty band’leri mühendislik hipotezi olarak tutulmalı. 2B/kalibrasyon/senkron hatasından teknik metriğe belirsizlik aktarımı hedeflenmeli. Yakın eşiklerde kesin aday yerine boundary/abstention üretilmeli; eşik duyarlılığı raporlanmalı.

Ground-truth kodunda frame ortalamalarından bağımsız yeniden örnekleme ile bootstrap CI var. Zamansal korelasyonlu videoda bu aralıkları bağımsız sporcu/oturum genellemesi için kullanmamak gerekir. Orta vadede blok/oturum/sporcu düzeyinde bootstrap ve kullanılan örnekleme birimini raporlayan değerlendirme eklenmeli. Bu değişiklik sonucu görmeden mevcut CI’nın ne kadar dar olduğu sayısal olarak iddia edilemez.

Kaynak: [bootstrap uygulaması](C:/Users/WWWW/Desktop/tk3d/src/ground_truth_validation.py:292).

### 6.5. Metrik paneli

| Katman | Birlikte raporlanacaklar |
| --- | --- |
| Kamera/senkron | Offset, timestamp residual p50/p95/max, drift, eksik/tekrar frame, kalibrasyon provenance |
| 2B | Kamera ve body/foot/face/hand grubu bazında bağımsız etiket hatası, görünürlük, track değişimi |
| 3B | Valid/observed/recovered oranları, kullanılan kamera, reprojection dağılımı, triangulation açıları |
| Stabilite | Temporal ve angle jitter, hız/ivme, kemik değişimi, correction dağılımı, hızlı hareket tepe korunumu |
| Dış 3B | Global ve root-relative MPJPE, p95, PCK, açı MAE; PA-MPJPE ayrı diagnostic |
| Segmentasyon | Sınır/anchor MAE ve dağılımı, tolerans içi oran, eksik/fazla segment, yanlış hareket eşlemesi |
| Teknik karar | Kural/olay bazında precision, recall, F1, false-positive/session, abstention, uzman uyumu |
| Ürün | İş başına süre, RAM/VRAM/disk, failed run oranı, review süresi, etiket kaybı/aktarım hatası |

## 7. Test ve CI geliştirmeleri

Mevcut 384 test sonucu önemli bir taban; test sayısı kapsam yüzdesi değildir. Yapılandırmada branch coverage/mutation ölçüm kapısı görülmedi. İlk hedef keyfi yüksek bir coverage yüzdesi değil, bu incelemede açık kalan tehlikeli dalların korunması olmalı.

Öncelikli ek test grupları:

| Grup | Gerekli senaryolar |
| --- | --- |
| Run lifecycle | Her aşamada hata, kesinti, state yazma hatası, terminal geçiş, aynı run ID yarışı |
| Artifact sözleşmesi | Yanlış frame/timestamp, kalite dizisi şekli, hash uyuşmazlığı, legacy/current ayrımı |
| Kalite kapıları | Boş/all-invalid metrik, eksik aday, kısa sekans, solver success fakat geçersiz çıktı |
| Teknik anlam | Yanlış yerde sabit el, doğru hedefe geç varış, ters yön, sınır-belirsiz değer, eksik kanıt |
| Senkronizasyon | Nonzero offset, farklı FPS, drift, kayıp frame, VFR/metadata hatası ve kamera seçimi |
| Review | Gerçek browser seek/play/pause, iki video drift’i, storage kapalı, bozuk/başka run JSON |
| Dağıtım | Repo dışı wheel kurulumu, desteklenen Python sürümleri, dış asset yokken açık hata |

CI Windows/Python 3.11 ve CPU Torch hedefliyor; yerel araştırma Python 3.12/CUDA/ZED. En az 3.11 ve 3.12 hafif test matrisi düşünülebilir. GPU/ZED kontrolleri ayrı uygun donanımlı iş olarak yürütülmeli; hafif CI’ın geçtiği yerde gerçek inference doğrulandı denmemeli. Başarısız test logları, junit özeti ve ilgili küçük kalite artifact’leri saklanmalı; ham video/checkpoint CI artifact’ine kör biçimde eklenmemeli.

## 8. Performans ve ölçekleme

PROJECT_STATUS’ta üç eş 260-kare run için kayıtlı ortalamalar: internal toplam **282,778 s**, core **218,503 s**, steady-state inference+geometry **1,833 FPS**. Bu incelemede benchmark tekrar edilmedi. Bölüm değerleri farklı/örtüşen zamanlama kapsamları taşıyabileceğinden hepsini toplayarak yeni toplam türetilmemeli.

Belgedeki önemli maliyetler depth fusion 70,688 s, ViTPose+causal 61,254 s, triangulation 52,057 s, serialization 38,546 s. Poomsae analizinde presentation/export 43,597 s, analiz/karar 12,821 s kayıtlı. Bunlar optimizasyon için başlangıç hipotezidir.

Önerilen sıra:

1. Aynı girdi/kamera/stride/config ile tekrarlı baz çizgi; p50/p95 süre ve RAM/VRAM/disk ölçümü.
2. Gereksiz tekrar video transcode ve rapor yeniden okuma/yazmalarını ölç; hash’e bağlı proxy cache tasarla.
3. Analiz sonucu hazırken sunumun ayrıca üretilebilmesini sağla.
4. Büyük JSON’ların tekrar eden envanter/kontrat kopyalarını ölç; kanonik değişmez envanter + referans yapısı veya sıkıştırılmış ara depolama dene. Mevcut tüketicilere sürümlü geçiş gerekebilir.
5. Triangulation’da yeniden kullanılan camera/undistort hesaplarını ve joint batch fırsatlarını profiler ile değerlendir.
6. Model hızlandırmayı ancak eş koşullarda doğruluk kaybı kapısıyla değerlendir. Flip test, model boyutu, mixed precision veya batch değişikliği davranış değişikliği olarak ele alınmalı.

Uzun sekans için chunk işlemenin temporal filtre/optimizer sınır etkileri tasarlanmalı; overlap ve birleşim kanıtı olmadan parçalamak sonucu değiştirir. Stride artırıp daha az kareyi “hızlandı” diye sunmak, output süresini korusa bile inference kanıtı yoğunluğunu değiştirir; karşılaştırmada açıkça raporlanmalı.

İki kameralı mevcut düzende `min_supporting_views=4` cross-view ikinci geçişinin sıfır iş yapması beklenen davranış. Üç kamera eklemek bu ayarda yeterli değildir: hedef dışındaki dört destek kamera için toplam en az beş gerekir. Üç kamera başka geometri/occlusion faydaları için ayrı pilot olabilir; mevcut eşiği sırf bu özellik çalışsın diye düşürmek genel güvenceyi değiştirir.

## 9. Kısa vade: ilk 0–4 hafta

Süreler tek odaklı mühendis için yaklaşık planlama aralıklarıdır; doğrulanmış teslim taahhüdü değildir. Veri/hakem erişimi ayrı takvim bağımlılığıdır.

| İş | Yaklaşık efor | Bağımlılık | Kabul ölçütü |
| --- | --- | --- | --- |
| K01 Birleştirmeyi ve kural dağılımını tutarlılaştır | 0,5–2 gün | Yok | B01/B02 kapanmış; test/diff temiz; tarihsel kanıt korunmuş |
| K02 Multiview lifecycle ve hata sahipliği | 2–4 gün | K01 | B03/B06; aşama hataları terminal durum bırakıyor |
| K03 Latest ve artifact sözleşmesini sıkılaştır | 2–4 gün | K01 | B04/B05/B07 negatif testleri geçiyor |
| K04 Eksik kalite metriklerini güvenle kapat | 1–3 gün | K01 | B08/B09; eksik veri başarıya dönüşmüyor |
| K05 Kural anlamı ve gizli eşik denetimi | 2–5 gün | K01 | B10/B11; ham ölçüm/karar/isim tutarlı |
| K06 Snapshot/code provenance ve güncel teslim kaydı | 1–3 gün | K02/K03 | Çalıştırılan kod kimliği ve test kanıtı bağlanmış |
| K07 Kör annotation protokolü ve tam kayıt listesi | 2–4 gün | Uzman erişimi | Etiket şeması, örnekleme ve uyuşmazlık çözümü yazılı |

**Kısa vade çıkış kapısı:** açık birleştirme yok; ilgili ve tam testler geçiyor; gerçek inference davranışı değişen düzeltmeler için kısa gerçek smoke ve JSON kalite incelemesi tamamlanmış; davranış değişiklikleri README/PROJECT_STATUS/ADR’a uygun biçimde yansıtılmış. Eski run’lar ve ham kanıtlar korunmuş.

## 10. Orta vade: 1–3 ay

| İş | Amaç | Bağımlılık | Kabul ölçütü |
| --- | --- | --- | --- |
| O01 Tam M01–M18 kayıt ve etiket kapsamı | Eksik hareket kanıtını tamamlama | K07 | Her hareket/faz için gerçek kaynak-bağlı etiket ve kapsam raporu |
| O02 Çift inceleyicili kör pilot | Teknik aday doğruluğunu ölçme | O01 | Kural bazında precision/recall, anlaşmazlık ve abstention raporu |
| O03 Bağımsız 2B annotation | Görüntü ölçüm hatasını ayırma | Veri protokolü | Kamera/grup/occlusion bazında sonuçlar |
| O04 Kalibrasyon ve senkron stres paketi | Kamera değişimine dayanıklılık | K03 | Offset/drift/kayıp frame/yeniden kalibrasyon testleri |
| O05 Ölçüm belirsizliği ve eşik duyarlılığı | Kesinlik iddiasını sınırlandırma | K05, O02/O03 | Sınır bölgeleri, hata kaynakları ve kalibrasyon eğrileri |
| O06 Timeline/faz düzenleyici | İnsan düzeltmesini güvenli kaydetme | K03, review bağları | Eski etiketi bozmadan yeni sürüm, audit izi, import/export |
| O07 Gerçek browser uçtan uca testi | Video/review işlevini doğrulama | Proxy eşleme | Oynatma, seek, JSON, storage ve senkron senaryoları |
| O08 Analiz/sunum aşama ayrımı ve cache | Tekrar çalışma maliyetini azaltma | K02/K06 | Aynı analiz çıktısı; ölçülmüş süre/disk kazancı |
| O09 Desteklenen ortam ve paket smoke’u | Başka makineye taşıma | Ürün hedefi | Repo dışında kurulum/çalıştırma; açık asset manifesti |

**Orta vade çıkış kapısı:** seçili tek oturumun dışından veri var; etiketli olayların kör test bölümü ayrılmış; hangi kuralların güvenilir, belirsiz veya desteklenmemiş olduğu sayısal olarak görülebiliyor. Kullanıcı “aday sayısı” yerine adayın kalitesini ve kanıtını değerlendirebiliyor.

## 11. Uzun vade: 3–12+ ay

| İş | Amaç | Önkoşul | Başarı kanıtı |
| --- | --- | --- | --- |
| U01 Senkron bağımsız 3B ground truth | Mutlak geometrik doğruluk | Ölçüm düzeni ve bütçe | Sporcu/oturum bazında dış 3B metrikleri ve güven aralıkları |
| U02 El/yüz/ayak özel doğrulama | 133 noktanın kalite dengesini iyileştirme | O03/U01 | Grup bazında dış hata, görünürlük ve fallback raporu |
| U03 Çok-kamera saha pilotu | Occlusion ve kötü açı dayanıklılığı | O04 | Eş koşullarda kamera sayısı/yerleşimi karşılaştırması |
| U04 Genellenebilir hareket/faz tanıma | Kısmi bilinen sıranın ötesine geçme | O01/O02 | Eksik/fazla/yanlış sırada hareket ve yeni sporcu testleri |
| U05 Hakem kalibrasyonu ve ayrı skor politikası | Doğrulanmış karar kapsamını belirleme | O02/U01 + kaynak onayı | Kör hakem karşılaştırması, yetki/kural sürüm bağı ve dış kabul |
| U06 Uzun kayıt/batch işleme | Araştırma koleksiyonlarını verimli işleme | O08 | Süre/RAM/disk/failure hedefleri; chunk sınır doğrulaması |
| U07 Çok kullanıcılı inceleme ve audit | Takım halinde annotation | Kanıt ve etiket sürümleme | Çakışma çözümü, erişim, geri alma, taşınabilir yedek |
| U08 Dağıtım paketi ve saha işletimi | Kurulum ve destek yükünü azaltma | O09 + ürün kararı | Tekrarlanabilir kurulum, açık lisans/asset sınırları ve geri dönüş |

Resmî puanlama, bu listenin sonunda otomatik açılacak bir bayrak değildir. Teknik doğruluk, kural kaynağı, uzman/hakem kalibrasyonu ve hedef kullanım için dış kabul ayrı ayrı sağlanmalı. Proje araştırma/koçluk incelemesi olarak kalacaksa bütün resmî puanlama yatırımlarını yapmak zorunlu değildir.

## 12. Eklenmesi gereken ürün özellikleri

### Öncelikli

- **Run merkezi:** running/completed/failed ve kalite reddini ayıran görünüm; stage, elapsed time, hata nedeni ve çıktı bağlantıları.
- **Kanıt kartı:** hangi kamera, hangi frame, ham/iyileştirilmiş ölçüm, beklenen aralık, belirsizlik ve aday gerekçesi.
- **Ölçülememe açıklaması:** kullanıcıyı model hatası, occlusion, kalibrasyon, eksik referans ve bilinmeyen teknik arasında yönlendirme.
- **Timeline düzenleme:** frame hassasiyetinde hareket/faz anchor düzeltmesi; versiyon ve eski etiket ilişkisi.
- **Eş koşul karşılaştırması:** farklı profil/stride/kamera setini fark edip karşılaştırılabilirlik uyarısı; sayısal fark ve örnek frame.
- **Veri ingest kontrolü:** doğru kamera/video eşlemesi, frame/timestamp kontrolü, asset hash’i, depolama konumu ve benzersiz session.

### Doğrulama verisinden sonra

- Sporcuya anlaşılır teknik geri bildirim; yalnız doğrulanmış kural kapsamı için.
- Oturumlar arası gelişim takibi; config ve kamera değişiminin etkisi ayrı gösterilerek.
- Koç/hakem inceleme kuyruğu; çok sayıda benzer adayın gruplandırılması.
- Kural bazında güvenilirlik sayfası; hangi veriyle test edildiği, hangi durumda çalışmadığı.

### Şimdilik ertelenmeli

- Yeni çok büyük model veya fine-tuning: bağımsız değerlendirme seti olmadan gelişim ölçülemez.
- Daha çok kuralı varsayılan aktif yapmak: mevcut kural anlamı ve false-positive oranı çözülmeli.
- Tüm sistemi mikroservislere bölmek: mevcut yerel workflow için ek operasyon maliyeti yaratır.
- Tam gerçek zaman iddiası: mevcut ölçümler offline araştırma iş akışına ait.
- Kontrolsüz eşik otomatik öğrenme veya skor açma: veri sızıntısı ve yetki kapsamı tasarlanmadan uygulanmamalı.

## 13. Veri, erişim ve dağıtım hazırlığı

Git dışındaki video, SVO2, model ve output varlıklarının korunması doğru. Bunun yanında bir asset kataloğu, içerik hash’i, yedekleme/geri yükleme denemesi ve referans run’ların saklama politikası gerekiyor. Ham triangulation ve 2B ölçümler kullanıcı istemeden temizlenmemeli; disk kazancı için önce yeniden üretilebilir sunum/cache çıktıları ele alınmalı.

Paylaşım hedeflenirse sporcu görüntüleri, yüz/eller ve reviewer kimlikleri için veri erişimi, anonimleştirme ve izin kayıtları tasarlanmalı. Bu rapor hukuki uygunluk değerlendirmesi yapmadı. Repository’de açık lisans dosyası bulunmadığına ilişkin mevcut proje notu dış dağıtım öncesi ele alınmalı; model ve veri lisansları da ayrı envanterlenmeli.

Yerel araştırma için kimlik doğrulamalı sunucu şart değildir. Çok kullanıcılı/uzaktan hizmet hedeflendiğinde güvenilmeyen dosya yükleme, yol sınırları, upload kotası, iş zaman aşımı, kaynak tüketimi ve erişim kayıtları ayrıca tasarlanmalıdır. Mevcut yerel araç bunların tümünü sağlayan bir web servisi gibi kabul edilmemeli.

## 14. Uygulama ve teslim stratejisi

İşleri tek büyük değişiklikte birleştirmemek gerekir. Önerilen inceleme paketleri:

1. Git/profil/test/belge uyumu.
2. Lifecycle ve latest sahipliği.
3. Artifact/provenance/config doğrulaması.
4. Teknik anlam ve eşik açıklanabilirliği.
5. Doğrulama verisi ve annotation araçları.
6. Sunum/performance/paketleme.

Her paket problem→yeniden üretim→dar düzeltme→ilgili test→riskine uygun geniş kontrol sırasını izlemeli. Gerçek inference etkilenirse kısa gerçek smoke ve ilgili kalite raporu şart. Önce/sonra karşılaştırmasında aynı video bölümü, kamera seti, stride, model ve yapılandırma korunmalı; değiştirilen faktör açıkça belirtilmeli.

Bu incelemenin bir sonraki en değerli uygulama paketi **K01–K05**: teslim engellerini kapatıp hangi ölçümün ne söylediğini güvenilir hale getirmek. Sonrasında en yüksek bilgi kazanımı yeni kurallardan değil, **tam kayıt + bağımsız annotation + kör değerlendirmeden** gelecektir.

## 15. Kullanıcının yeniden çalıştırabileceği teslim kontrolleri

Aşağıdaki blok yeni bir test klasörü üretir. Repo kaynaklarında düzeltme yapmaz. Bu rapordaki mevcut ağaçta üç pytest hatası ve belge conflict hataları beklenir; düzeltmeden sonra yeniden doğrulama için kullanılabilir.

```powershell
Set-Location 'C:\Users\WWWW\Desktop\tk3d'
$auditStamp = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
$auditTestRoot = "C:\Users\WWWW\Desktop\tk3d\outputs\pytest-audit-$auditStamp"
git status --short
.\.venv312\Scripts\python.exe -m ruff check src scripts tests
.\.venv312\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp $auditTestRoot
.\.venv312\Scripts\python.exe -m pip check
git diff --check
Write-Output "Test geçici çıktıları: $auditTestRoot"
```

Bu blok farklı komutların sonuçlarını gösterir; tek bir son exit code’a bakarak bütün kapıların geçtiği varsayılmamalı. Bu rapordaki asıl test koşusunun geçici çıktıları `C:\Users\WWWW\Desktop\tk3d\outputs\pytest-audit-20260905-01` altındadır.

## 16. Projeye eleştirel değerlendirme: yapısal sorunlar ve gelişim öncelikleri

**Ana eleştiri: yazılımın kapsamı, doğruluğunu gösterebildiği kapsamdan daha hızlı büyümüş.** Çok sayıda katman, kural ve rapor mevcut; bunların sporcu hakkında ne kadar doğru hüküm verdiğine ilişkin bağımsız kanıt daha dar. Bundan sonraki gelişim, özellik sayısından çok hangi sonuçlara hangi koşullarda güvenilebildiği üzerinden yönetilmeli.

Bu bölüm önceki teknik bulguların proje yönetimi ve ürün açısından yorumudur. Yeni test veya bağımsız saha doğrulaması yapılmış olduğu anlamına gelmez; mevcut 11 somut bulguya ek, ayrı bir hata sayısı olarak okunmamalıdır.

### 16.1. Kapsam genişliği başarı ölçüsüne dönüşme riski taşıyor

133 landmark, 174 kural ve 18 hareket kontratı geniş bir envanterdir. Fakat bir landmarkın kurala bağlanması güvenilir ölçüldüğünü; evaluator bulunması da teknik anlamının doğru olduğunu göstermez. Aynı fiziksel olayı farklı isimlerle değerlendiren kontroller de bağımsız doğrulama kapsamı gibi sayılmamalı.

**Sorun:** geliştirme ilerlemesi kural/özellik sayısıyla ölçülürse, doğruluğu belirsiz bir sistem giderek daha tamamlanmış görünebilir.

**Gerekli değişim:** ilerleme raporlarında doğrulanmış kural sayısı, bağımsız test edilen sporcu/oturum kapsamı, yanlış aday oranı ve belirsiz kalma oranı öne çıkarılmalı. Envanter büyüklüğü ayrı bir mühendislik ölçüsü olarak kalmalı. Bölüm 6'daki değerlendirme programı bu ayrımı sağlar.

### 16.2. Bazı teknik iddialar ölçümün desteklediğinden büyük

B11'deki “hedefe ulaştı” kontrolünün yalnız elin sabitliğini ölçmesi bunun doğrudan örneği. Sabit durmak ile doğru hedefe ulaşmak farklı özelliklerdir. Böyle bir anlam kayması, kod hata vermeden çalışırken de yanlış güven üretebilir.

**Sorun:** kullanıcı kural adını teknik bir değerlendirme olarak okur; implementasyonun daha dar bir proxy ölçtüğünü bilmeyebilir. Diğer kuralların ayrıca hata yakalaması, bu kontrolün anlamını düzeltmez.

**Gerekli değişim:** bütün aktif kurallar için “hesap gerçekten adında söylediği şeyi ölçüyor mu?” incelemesi yapılmalı. Her kurala tanım, kullanılan ölçüm, karşı örnek, gözlenebilirlik sınırı ve ölçülememe davranışı yazılmalı. B10/B11 düzeltmeleri bu denetimin başlangıcı olmalı.

### 16.3. Doğrulama uygulamanın kendi varsayımlarına fazla yakın

Sentetik testler, sınır vakaları ve veri sözleşmeleri gerekli. Ancak uygulamanın varsaydığı geometriden türetilen testler aynı varsayımın yanlışlığını yakalayamayabilir. Testlerin geçmesi hesaplamanın tutarlı olduğunu gösterebilir; gerçek sporcunun tekniğine ilişkin yorumun doğru olduğunu tek başına göstermez.

**Sorun:** çok sayıda başarılı test, bağımsız dış doğrulama yerine psikolojik güven oluşturabilir. Yalnız sistemin ürettiği adayları incelemek de kaçırılan hataları görünmez bırakır.

**Gerekli değişim:** algoritmadan bağımsız hazırlanmış gerçek örnekler ve karşı örnekler kullanılmalı; aday üretmeyen bölümler de kör etiketlenmeli. Yazılım regresyonu, ölçüm doğruluğu ve teknik karar doğruluğu ayrı raporlanmalı.

### 16.4. Tek kısa kayıt projeyi fazla şekillendirmiş

Aktif gerçek kanıt M01–M06 ağırlıklı, fakat mimari ve raporlama daha genel bir sistem görüntüsü veriyor. Yeni sporcu, farklı vücut oranı, hareket hızı veya kamera yerleşiminde hangi parçaların aynı güvenilirliği koruyacağı yeterince bilinmiyor. Bölüm 5.8'deki kayda özgü metinler de bu dar bağlamın uygulamaya yerleştiğini gösteriyor.

**Sorun:** tek örneği daha ayrıntılı açıklamak, genelleme problemini çözmeyebilir. Yeni kayıt için gereken manuel profil, yol ve referans hazırlığı da gerçek kullanım maliyetinin parçasıdır.

**Gerekli değişim:** bağımsız yeni bir oturumu içeri alma süresi ve gereken müdahaleler ölçülmeli. Yeni veri üzerinde başarısız olan kontrol için genel varsayılanı sessizce değiştirmek yerine veri koşulu ve başarısızlık nedeni raporlanmalı. O01–O04 işleri bu yüzden yeni kural geliştirmesinden önce gelmeli.

### 16.5. Güvenlik ilkeleri bütün girişlerde aynı güçte uygulanmıyor

Poomsae uygulamasındaki hata yönetimi doğrudan multiview yolunda aynı kapsamda bulunmuyor. B03'te run'ın running kalması, B05–B07'deki bağ ve durum açıkları bunun somut örnekleri. Proje doğru ilkeleri tanımlamış; fakat bunların bütün giriş noktalarına ortak altyapı üzerinden uygulanması tamamlanmamış.

**Sorun:** güvenlik davranışı hangi CLI/API yolunun çağrıldığına bağlı hale gelebilir. Yeni komut ve katmanlar bu farkı büyütebilir.

**Gerekli değişim:** lifecycle, artifact doğrulama ve başarı ilanının tek, açık sahipliği olmalı. Her giriş noktası aynı olumsuz senaryolarla sınanmalı. Yeni bir katman eklenmeden önce ortak sözleşme kullanılmalı.

### 16.6. Mimari bakım yükü büyümüş

Uzun uygulama modüllerinde hesaplama, dosya yönetimi, yaşam döngüsü ve sunum sorumlulukları birleşiyor. Analizin video üretimine takılabilmesi pratik bir örnek. Dosya uzunluğu tek başına kötü tasarım kanıtı değildir; fakat bir sorumluluktaki değişiklik diğerini gereksiz yere etkiliyorsa ayrım yetersizdir.

**Sorun:** küçük değişiklikler geniş regresyon yükü oluşturabilir; yeni geliştiricinin sistemi anlaması ve hata kökünü ayırması zorlaşır.

**Gerekli değişim:** hesaplama sonuçları, artifact yazımı ve sunum üretimi daha açık aşamalara ayrılmalı. Önce mevcut davranış korunarak dar ayrıştırmalar yapılmalı; büyük yeniden yazım veya mikroservisleşme varsayılan çözüm olmamalı.

### 16.7. Belgeler ayrıntılı, fakat güncel gerçeği takip etmek zorlaşmış

Tarihsel kanıtın korunması değerli. Buna karşılık 75 aktif kural yazarken mevcut profilin 74 üretmesi ve geçmiş başarılı test sayılarının güncel durumla yan yana bulunması okuyucunun yanlış sonuca varmasını kolaylaştırıyor. B01/B02 bunun güncel belirtileri.

**Sorun:** belge hacmi büyüdükçe aynı bilginin birden fazla yerde elle güncellenmesi zorlaşıyor. Ayrıntı arttığı halde hangi bilginin bugün geçerli olduğu belirsizleşebiliyor.

**Gerekli değişim:** değişken envanter ve sayılar mümkün olduğunca koddan/validation artifact'inden üretilmeli. Güncel teslim özeti commit, çalışma ağacı kimliği, test sonucu ve kullanılan run'a bağlanmalı; tarihsel bilgiler açıkça geçmiş olarak kalmalı.

### 16.8. Kullanıcı değeri yeterince ölçülmüş değil

Koç veya sporcu için yüzlerce kural satırından daha önemli sorular şunlardır: “Neyi düzeltmeliyim?”, “Hangi görüntü bunu gösteriyor?” ve “Sistem ne kadar emin?” Çok sayıda benzer veya yanlış aday, ayrıntılı raporu pratikte yorucu hale getirebilir.

**Sorun:** teknik çıktı zenginliği, kullanışlı geri bildirim kalitesiyle karıştırılabilir. Aday sayısının artması inceleme yükünü büyütürken faydayı artırmayabilir.

**Gerekli değişim:** kullanıcı pilotunda inceleme süresi, yararlı bulunan geri bildirim oranı, yanlış aday yükü ve anlaşılmayan açıklamalar ölçülmeli. Aynı olaya ait adaylar gruplandırılmalı; kanıt ve belirsizlik ilk bakışta anlaşılmalı. Bunlar mevcut pilot sonucu değil, ölçülmesi gereken ürün hedefleridir.

### 16.9. Proje yönetimi açısından önerilen yön değişikliği

Yeni kural ekleme hızı geçici olarak düşürülmeli. Önce K01–K05 tamamlanmalı; ardından az sayıda önemli teknik kontrol bağımsız uzman etiketleriyle sınanmalı. Kontrollerin hangilerinin güvenilir olduğu öğrenildikten sonra kapsam genişletilmeli.

| Öncelik | Yönetim kararı | Beklenen somut çıktı |
| --- | --- | --- |
| 1 | Teslim ve ortak sözleşme açıklarını kapat | Temiz test/diff kapısı, doğru terminal run durumları ve güçlü artifact bağları |
| 2 | Aktif kuralların anlamını denetle | Ölçümle uyumlu ad/açıklama, açık fiziksel eşikler ve karşı örnek testleri |
| 3 | Küçük ama bağımsız bir doğrulama pilotu yap | Kural bazında precision/recall, belirsizlik ve hata örnekleri |
| 4 | Yeni oturum ve kullanıcı deneyimini ölç | Kurulum/ingest ve review süresi, yararlı geri bildirim oranı |
| 5 | Kanıtlanan ihtiyaçlara göre genişlet | Ölçülebilir fayda sağlayan yeni kural, model veya performans değişikliği |

Projenin ham kanıtı koruması, fail-closed davranışı hedeflemesi ve resmî puan iddiasını sınırlaması güçlü yönleridir. Eleştirinin amacı bunları kaldırmak değil, uygulamanın tamamında tutarlı hale getirmektir. Sonraki gelişim hedefi **hangi sonuçlara, hangi koşullarda ve hangi kanıtla güvenilebileceğini göstermek** olmalıdır.

## 17. 10 Eylül 2026 düzeltme durumu

Bu teslimde devam eden merge, yerel geçici kural genişletmesi ile uzak daldaki
hakem-imzalı eşik sözleşmesini birlikte koruyacak biçimde çözüldü. Mimari karar
numaraları tekilleştirildi ve bütün çatışma işaretleri kaldırıldı.

Uzak daldaki sonraki düzeltme, baş/gövde oturma kararındaki `10°` sınırını ve
ölçüm katmanında tekrar eden diğer karar sabitlerini profile taşıdı.
`head_torso_settle_offset` böylece güvenli biçimde yeniden aktif edildi. Güncel
profil 3.2.1; envanter **75 `active_diagnostic`, 74 `measurement_only`, 17
`blocked_missing_reference`, 8 `not_observable_with_current_pipeline`** üretir.
Sentetik sınıflandırma kapsamı **88 metrik ve 980 vaka** olarak doğrulandı.

Doğrulama sonuçları:

- teknik doğruluk, validation ve Poomsae runner odaklı testler:
  **96 passed in 50.63s**;
- tam pytest: **389 passed in 76.29s**;
- Ruff: temiz;
- `git diff --check`: temiz.

Bu sonuçlarla B01'deki merge çatışması, B02'deki profil/test/belge sayım
uyuşmazlığı ve B10'daki gömülü karar sınırları bu teslim kapsamında kapatıldı.
Raporun lifecycle, artifact bağı, semantik doğruluk ve bağımsız dış doğrulama
hakkındaki diğer bulguları kısa, orta ve uzun vadeli çalışma listesi olarak
geçerliliğini korur.
