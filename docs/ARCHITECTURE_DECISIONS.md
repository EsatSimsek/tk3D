# TK3D Mimari Karar Kaydı

Bu belge, sonraki oturumların daha eski veya daha kolay görünen bir yaklaşıma
yanlışlıkla dönmesini engelleyen aktif kararları özetler.

## AD-001 — 133 noktalı veri sözleşmesi

Karar: Ana veri şekli `keypoints_3d_world[t, 133, 3]` olarak korunur.

Gerekçe: Poomsae için gövdenin yanında el, ayak ve yüz bilgisi gelecekte
gereklidir. BODY-17 kalite/optimizasyon katmanı tüm çıktıyı 17 noktaya
indiremez.

## AD-002 — Aktif 2B hat ViTPose-Huge WholeBody

Karar: Üretim hattı RF-DETR + ByteTrack + stabilize crop + ViTPose-Huge
WholeBody kullanır. RTMW 2B aktif hat değildir; `run_rtmw_multiview_3d.py`
yalnız geriye uyumluluk yönlendirmesidir.

Gerekçe: WholeBody 133 nokta sözleşmesini ve mevcut doğrulanmış runtime'ı
korumak.

## AD-003 — Kalibrasyon ve koordinat fail-closed

Karar: Canlı 3B yalnız ortak referanslı çok-kamera veya resmî AIST
kalibrasyonuyla üretim kalitesinde kabul edilir. Intrinsics-only, yaklaşık ve
sentetik kalibrasyon üretim sonucu sayılmaz.

Gerekçe: Bağımsız kamera koordinatlarını ortak dünya sanmak düşük
reprojection'a rağmen fiziksel olarak yanlış 3B üretebilir.

## AD-004 — Ortak zaman çizelgesi

Karar: Kamera FPS, frame offset ve saniye offsetleri fiziksel ortak zamana
eşlenir. Aynı yerel frame indexi otomatik olarak aynı fiziksel an sayılmaz.

Gerekçe: Senkron hatası pose modelinden bağımsız olarak triangulation'ı bozar.

## AD-005 — Robust triangulation

Karar: Normalize kamera koordinatlarında hipotez/inlier seçimi, pozitif
derinlik, minimum triangulation açısı, reprojection ve robust nonlineer
iyileştirme kullanılır.

Reddedilen yaklaşım: Bütün kamera gözlemlerini aykırı kontrolü olmadan tek DLT
çözümüne vermek.

## AD-006 — Döngüsüz cross-view 2B feedback

Karar: Hedef kamera leave-one-out 3B öncülden çıkarılır. Öncül yalnız heatmap
aramasını yönlendirir. Görüntü kanıtı yoksa nokta triangulation'a yeni kanıt
olarak dönmez.

Gerekçe: Kendi 3B sonucunu tekrar 2B kanıt gibi kullanmak hatayı küçültmüş
gösteren dairesel veri üretir.

## AD-007 — Global BODY-17 optimizasyon ve rollback

Karar: Tüm senkron sekans reprojection, kemik, eklem limiti ve temporal
kısıtlarla birlikte çözülür; ham triangulation korunur ve kabul kapısı
başarısızsa geri dönülür.

Reddedilen yaklaşım: Yalnız nihai videoyu ağır smoothing ile sakinleştirip ham
ölçümü silmek.

## AD-008 — Run izolasyonu ve provenance

Karar: Her çalışma benzersiz `runs/<run_id>` dizinindedir. Ham, stabilize,
feedback geometry, optimized ve visualization-only kaynaklar ayrı tutulur.

Gerekçe: Önceki sonucu ezmeyi, benchmark karışmasını ve veri kaynağının
belirsizleşmesini önlemek.

## AD-009 — Ground-truth olmadan resmî skor yok

Karar: İç geometri kapısı puanlama yetkisi vermez. Onaysız çıktı
`scoring_ready=false`; yalnız açık tanısal izinle
`provisional_not_official` skor üretilebilir.

Gerekçe: Reprojection, aynı hatayı paylaşan kameralar veya model yanlılığı
nedeniyle gerçek 3B hatayı ölçmez.

## AD-010 — Model adapter seçimi held-out 3B ile yapılır

Karar: 2B validation loss iyileşmesi üretim onayı değildir. MADS F2 eğitim ve
validation'dan ayrı tutulur; adapter held-out global MPJPE'yi kötüleştirirse
reddedilir.

Sonuç: Denenen MADS head ve offset adapter'ları üretime alınmadı; temel
ViTPose-Huge kullanılmaya devam ediyor.

## AD-011 — Veri setine özel ayarların izolasyonu

Karar: Kamera ofsetleri ve veri seti eşikleri ilgili session/manifestte
tutulur; genel config'e otomatik taşınmaz.

Örnek: AIST örnek sekansındaki `c05: -268`, bütün c05 kameraları veya başka
çekimler için evrensel değildir.

## AD-012 — Puanlama yetkisi koşuya ve dosya özetlerine bağlıdır

Karar: Tahmin JSON'undaki `scoring_ready` alanı tek başına yetki vermez.
Ground-truth değerlendiricisi tahmin, referans, eşik profili, eşleşen kareler,
doğrulama manifesti ve iç-geometri raporunun SHA-256 özetlerini
`scoring_authorization.json` içinde aynı koşuya bağlar. Puanlama girişinde bu
bağların tamamı yeniden doğrulanır.

Gerekçe: Başka bir benchmark raporunun yanlış koşuya bağlanmasını, doğrulama
sonrası tahminin değiştirilmesini ve bir boolean alanı elle değiştirerek kalite
kapısının aşılmasını önlemek.

Sınır: `scoring_ready` dış 3B ground-truth kapısını geçmiş
`provisional_not_official` altyapıyı açar. Resmî poomsae puanı için uzman
kural/hakem etiketleri ayrıca doğrulanmalı ve `official_scoring_ready` bağımsız
olarak geçmelidir.

## AD-013 — ZED SVO2 ingest donanım zamanına ve açık kamera bazına bağlıdır

Karar: Çoklu ZED SVO/SVO2 kayıtları aynı yerel kare indeksine göre değil,
kayıtlı görüntü zaman damgalarından ortak sabit-FPS çizelgeye eşlenir. Kayıplı
yeniden sıkıştırma ultra inference girdisi değildir; varsayılan hazırlama
formatı kayıpsız FFV1'dir. Yeniden kullanılan ve kullanılmayan kaynak kareler
ayrı senkron raporunda korunur.

ZED Fusion `camera_to_world` pozları metre ve `RIGHT_HANDED_Z_UP`
(`x=sağ, y=ileri, z=yukarı`) olarak okunur. `override_gravity=false` olduğunda
SDK sözleşmesindeki `Pose_abs = Pose_rel * Rot_IMU_camera` uygulanır. Dünya
pozunun tersi tek başına kamera projeksiyonu değildir; kamera bazının ayrıca
OpenCV optik eksenlerine (`x=sağ, y=aşağı, z=ileri`) çevrilmesi zorunludur.

Gerekçe: Ayrı kayıtlardaki tek-kare zaman boşlukları sabit frame offset ile
temsil edilemez. Ayrıca ZED dünya/kamera eksenini doğrudan `K[R|t]` içine vermek
pozitif derinliği ve 3B geçerliliği bozar; ilk smoke ölçümünde optik-baz düzeltmesi
BODY-17 geçerli oranını `%1.76 -> %68.33`, ortalama reprojection'ı
`10.53 -> 5.05 px` değiştirmiştir.

Sınır: ZED Fusion ortak dünya kalibrasyonu iç geometri için kabul edilen bir
üretim kalibrasyon modudur; dış ground-truth doğruluğu veya resmî puanlama
yetkisi sağlamaz. Stereo depth-fusion için ek güvenlik kararı AD-014'tedir.

## AD-014 — ZED depth yardımcı kanıttır ve saf RGB dalına karşı kapılanır

Karar: ZED stereo depth, RGB çok-kamera triangulation'ın yerine geçmez.
Yalnız confidence, beklenen derinlik, yerel robust patch ve koşuya özel
kamera/eklem yüzey-ofset kapılarını geçen BODY-17 ölçümleri yardımcı kısıt
olarak kullanılır. El, yüz ve ayak bu ilk entegrasyonun dışındadır.

Depth adayı ve aynı gözlemlerden üretilen saf RGB referans adayı ayrı global
BODY-17 optimizasyonlarından geçirilir. Son reprojection, ivme, kemik
kararlılığı veya solver durumu tanımlı sınırı aşarsa tüm depth dalı reddedilip
saf RGB sonucuna dönülür. Ham triangulation ve ham depth gözlemleri ayrı
çıktılarda korunur.

Gerekçe: ZED yüzey mesafesi eklem merkezi değildir; siluet kenarlarında
arka-plan derinliğine sıçrayabilir ve kare bazlı küçük düzeltmeler temporal
gürültü üretebilir. Nokta kapısı tek başına sekans kalitesini garanti etmez.

IMU sınırı: Sabit kameralarda IMU, Fusion kamera yönelimi ve yerçekimi
kalibrasyonu için kullanılır. Kare bazlı IMU dönüşü sporcunun pozu gibi
uygulanmaz. Kamera hareketi algılama/düzeltme ayrı bir doğrulama gerektirir.

Doğrulama sınırı: İç RGB-vs-depth kapısının geçmesi dış 3B doğruluk kanıtı
değildir. `scoring_ready` ancak bağlı ground-truth profili de geçerse açılır.

## AD-015 — Dış doğruluk koşuya bağlıdır ve tarihsel metrik devralınmaz

Karar: Dış doğruluk sonucu tahmin dosyasına, kamera/sensör profiline,
ground-truth kaynağına ve değerlendirme yapılandırmasına bağlıdır. Bir koşunun
MPJPE/P95 gibi metrikleri başka bir koşunun raporuna kopyalanmaz veya puanlama
kararında o koşunun metriğiymiş gibi kullanılmaz. Uyumlu bağımsız referans yoksa
`external_accuracy.status=not_evaluated_for_this_run`, `metrics=null` ve
`historical_benchmark_inherited=false` yazılır.

ZED RGB, stereo depth, confidence, kalibrasyon, zaman damgaları ve IMU aynı
sistem içindeki sensör kanıtlarıdır. Bunların kendi aralarındaki reprojection,
depth residual, temporal ve kemik kararlılığı kontrolleri
`internal_sensor_consistency` altında raporlanır; bağımsız dış 3B doğruluk
iddiası oluşturmaz.

Gerekçe: Aynı stereo çift, kalibrasyon veya zamanlama hatası RGB ve depth
kanallarını birlikte etkileyebilir. Tarihsel MADS RGB-only sonucu ise güncel ZED
RGBD sensör profili ve koşusunu ölçmez. İki kanıt türünü ayırmak hem yanlış
başarısızlık atamasını hem de yanlış doğruluk iddiasını önler.

Puanlama etkisi: İç sensör tutarlılığının geçmesi geliştirme amaçlı
`provisional_not_official` analize izin verebilir; `scoring_ready` yalnız aynı
koşuya kriptografik olarak bağlı bağımsız dış doğrulama yetkisiyle açılır.

## AD-016 — Provisional puanlama dış ground-truth beklemez

Durum: 12 Ağustos 2026 tarihinde AD-023 tarafından kısmen geçersiz kılındı.
`provisional_scoring_ready` veri hazırlık kapısı olarak korunur; bu kararda
tanımlanan kaynaksız 0-100 skor üretimi kaldırılmıştır.

Karar: Mocap veya bağımsız ölçülmüş 3B referans bulunmaması normal ZED çalışma
akışını durdurmaz. Aynı koşunun `run_quality_report.json` dosyasında üretim
kalibrasyonu, iç geometri ve sensör tutarlılığı geçtiğinde
`provisional_scoring_ready=true` olur. Puanlama analizi tahmin ve kalite raporu
session/run kimlikleriyle SHA-256 özetlerini bağlayarak
`provisional_not_official` çıktı üretir.

Dış doğruluk bu karara katılmaz, tarihsel MADS metriği devralınmaz ve
`official_scoring_ready=false` kalır. Kullanıcı açıkça dış yetkilendirme dosyası
verirse o dosya hâlâ katı biçimde doğrulanır; başka koşuya ait veya bozulmuş
dosya kabul edilmez.

Gerekçe: Sahada mocap elde edilemeyecek olması açıklanabilir iç kaliteye dayalı
geliştirme skorunu gereksiz yere durdurmamalıdır. Buna karşılık iç sensörlerin
birbirini doğrulaması bağımsız dış doğruluk veya resmî hakem geçerliliği olarak
sunulmamalıdır.

## AD-017 — Poomsae puanlaması önce kural ve sıralı hareket modeliyle kurulur

Karar: Nihai Poomsae motoru ilk aşamada uçtan uca skor tahmin eden bir ML
modeli olmayacaktır. Sürümlenmiş WT kural paketi, resmî kaynaklara izlenebilir
Taegeuk 1 Jang hareket/faz şeması, sporcu-merkezli normalize ölçümler, bilinen
sıraya kısıtlı hizalama, observability kapısı ve kanıtlı kesinti adayları
kullanılacaktır.

ML yalnız yeterli etiketli veri bulunduğunda hareket/faz algılama, görüntüden
zor teknik ayrıntıları çıkarma ve çoklu hakem puanlarına kalibrasyon gibi alt
görevlerde eklenir. Öğrenilmiş sonuç Poomsae sırası ve kural motorunu atlayamaz;
düşük güvenli veya ölçülemeyen durum puan kesintisine dönüşemez.

Gerekçe: Taegeuk 1 Jang sırası bilinen bir formdur; bu bilgiyi kullanmak veri
ihtiyacını azaltır ve kesintiyi hareket/kural/video kanıtıyla açıklanabilir
kılar. Etiketsiz çok sayıda video doğru teknik, hata şiddeti veya hakem puanı
ground-truth'u sağlamaz.

Uzman/hakem erişimi mevcut geliştirme planının önkoşulu değildir. Resmî
WT/Kukkiwon kaynakları, ayrı teknik test videoları ve açıkça işaretlenmiş
mühendislik toleranslarıyla `rule_based_provisional` puan üretilebilir.
Presentation 0–6 dönüşümü yapılırsa `judge_calibrated=false` ve
`provisional_not_judge_validated` olarak işaretlenir. Hakem kalibrasyonu ve
`official_scoring_ready` gelecekteki opsiyonel kapılardır; bunların kapalı
olması provisional puanlamayı durdurmaz.

Uygulama ayrımı üç sözleşmeyle yapılır:

- `RulePack`: yürürlükteki WT puan bütçesi ve kesinti miktarları,
- `PoomsaeSpec`: Kukkiwon kaynaklı hareket, teknik, yön ve faz sırası,
- `MovementTimeline`: belirli pose run'ının SHA-256 ile bağlanmış hareket/faz
  zamanları ve kaydın `complete_performance`/`partial_sequence` kapsamı.

MovementTimeline v2'de kapsam zorunludur. `partial_sequence` kaydı, gözlenen ve
kaynakta bulunmayan hareketleri sıralı iki küme olarak taşır ve nedenini yazar.
Bu ayrım sayesinde videoda olmayan bir hareket, algılama/etiketleme hatası gibi
raporlanmaz; kısmi kayıt tam performans puanına giremez.

Hareket zaman çizelgesi tek bir `apex` alanına indirgenmez. Tekme ardından
vuruş gibi bileşik hareketlerde `kick_apex`, `rechamber`, `landing`,
`punch_execution` ve `fixation` gibi birden fazla sıralı anchor taşınabilir.
Ekrandaki hareket/faz yazısı bu zaman çizelgesinden türetilir; overlay kuralın
veya puanın kaynağı değildir.

2 Ağustos 2026 kaynak denetiminde WT canlı `CURRENT` listesi 30 Eylül 2024
Poomsae kuralını güncel gösterdi. Article 16'nın atıf yaptığı ayrı scoring
guideline eki kamuya açık güncel listede bulunmadığı için kesinti miktarları
aktif, teknik bazlı küçük/büyük hata eşikleri ise fail-closed taslaktır.
Kukkiwon'un 2022 tam sıra gösterimi ve 2025 ayrıntılı Taegeuk 1 eğitimi ücretsiz
resmî hareket kaynakları olarak kullanılır. Kaynak ve erişim durumu
[`SCORING_SOURCE_REGISTER.md`](SCORING_SOURCE_REGISTER.md) içinde tutulur.

Uygulamadaki `assess_accuracy_readiness` kapısı RulePack, PoomsaeSpec,
MovementTimeline ve bağlanan pose dosyasının SHA-256 değerini birlikte
doğrular. Hazırlık raporunun `blocked` olması depth/ground-truth eksikliğiyle
otomatik olarak eş tutulmaz; her engel ayrı kodla raporlanır. Mevcut ZED2i
koşusunda pose bağı doğrulanmış, kaynak aktivasyonu ve 18 hareketin zaman/faz
etiketleri açık kalmıştır.

Kısa veya yarım Poomsae kaydı tam performans gibi puanlanmaz. Bilinen sıranın
videoda gerçekten görülen başlangıç bölümü hareket/faz ölçüm kanıtı üretebilir;
eksik hareketler otomatik eklenmez, toplam Accuracy puanı `null` kalır ve durum
`not_scored_partial_recording` olarak raporlanır. Böyle bir kayıt hareket
etiketleme ve metrik geliştirme için kullanılabilir fakat tam Poomsae skor
testinin yerine geçemez.

İnceleme yüzeyi puan motorundan ayrıdır. Senkron video, hareket/faz atlama,
ölçüm özeti, hazırlık engelleri ve önceden üretilmiş mühendislik denemesini
gösterebilir; kendisi deduction olayı veya Accuracy değeri üretemez. Gösterdiği
sonuçlar kaynak bağlı evidence/readiness/trial JSON'larından gelir ve ayrı
manifestte hash'lenir.

Sayısal mühendislik hipotezleri resmî RulePack'e yazılmaz; ayrı
`EngineeringToleranceProfile` sözleşmesinde sürümlenir. Profil yalnız gözlenen
M01-M06'ya, fixation fazına ve geniş body-normalize aralıklara uygulanır.
Düşük BODY-17, iki kameradan az kanıt, yüksek reprojection veya düşük timeline
güveni `not_measurable` üretir. Aynı hareket/aile/fazdaki sapmalar tek trial
olayında birleştirilir. Sayısal sapma WT'nin “yanlış hareket” anlamındaki büyük
hatasını kanıtlamadığı için otomatik major kapalıdır.

BODY-17 v1 denemesinin sayısal alanı sonradan geçersizleştirilmiştir;
`partial_engineering_trial_score=null`, `accuracy_score=null` ve
`applied_deductions=[]` kalır. Tarihsel dosyalar silinmez fakat güncel sonuç
olarak gösterilmez.

## AD-018 — Poomsae tekniği WholeBody-133 ölçülür ve skor fail-closed kalır

Karar: Pose hattının BODY-17 optimizasyonu puanlama girişini 17 noktaya
daraltamaz. Poomsae teşhisi `keypoints_3d_world[t,133,3]` içindeki gövde,
ayak, yüz, sol el ve sağ el gruplarını açıkça tüketir. Ayak yönü, bakış/gövde
yönü, hikite, bilek-el hizası, yumruk kapanması, zamanlama, fixation ve
trajectory hareket/faz boyunca ayrı metriklerdir.

Gerekçe: İlk v1 denemesi yalnız beş kaba BODY-17 ölçüsüyle hata bulamayınca
başlangıç değerini koruyup yanlış bir maksimum izlenimi verdi. Ölçülemeyen veya
eşikte kalan özellik doğruluk kanıtı değildir.

Koruma: WholeBody v2 `numeric_score_enabled=false` sözleşmesini doğrular; 17
noktalı giriş reddedilir, düşük grup/metric kapsamı raporlanır, eşik aşımı
yalnız `review_candidate_not_deduction` üretir. Resmî/kalibre tolerans ve kabul
edilmiş hata olayı olmadan Accuracy veya WT kesintisi üretilemez.

## AD-019 — Ölçüm adayı doğrudan kesinti olayı değildir

Karar: WholeBody ölçüsü `metric_id` ile, hareket sözleşmesindeki anlamı ayrı
`criterion_id` ile taşınır. Accuracy motoru yalnız ilgili hareketin
`measurable_criteria` listesinde bulunan kriteri; kesinti türüyle eşleşen WT
`rule_id`, yetkili `source_ref`, doğrulama yöntemi ve `review_record_id` ile
birlikte kabul eder.

Gerekçe: Yalnız `decision_status=confirmed_by_rule` yazılmış serbest bir JSON
olayı, önceki sözleşmede keyfî metriği kesintiye sokabiliyordu. Ölçüm eşiğinin
aşılması, o eşiğin WT küçük/büyük hata tanımı olduğu anlamına gelmez.

Koruma: Sahte/uyuşmayan kural kimliği sözleşme hatasıdır; hareket için yetkisiz
kriter `metric_not_authorized_for_movement` ile uygulanmaz. WholeBody adayları
`human_review_status=pending`, `rule_eligibility=blocked_unvalidated_screening_threshold`
ve `score_effect=null` taşır. Hazırlık kapısı ayrıca WholeBody raporunun timeline,
pose SHA-256, 133-nokta grupları ve minimum kapsam bağını doğrular.

## AD-020 — Yeni scoring kaynağı otomatik kural aktivasyonu yapamaz

Karar: Kullanıcıdan, web'den veya yerel arşivden gelen her yeni PDF önce
`SourceIntake` aday manifestine alınır. PDF imzası, SHA-256, kurum, belge ve
yürürlük tarihi, otorite sınıfı, kullanım amacı ve talep edilen iddialar
doğrulanır. Başarılı intake sonucu yalnız
`ready_for_manual_activation_review` olabilir; `automatic_activation_allowed`
daima `false` kalır.

Gerekçe: Resmî logolu tarihsel bir belge, ulusal federasyon yorumu veya hakem
eğitim notu güncel WT toleransı olmayabilir. Dosyanın gerçek ve faydalı olması,
içindeki her ölçünün güncel RulePack'e taşınabileceği anlamına gelmez.

Koruma: Tarihsel/ikincil/akademik kaynak güncel sayısal tolerans talep ederse
`numeric_threshold_authority_insufficient` engeli oluşur. Hash'i sabitlenmeyen,
otoritesi sınıflandırılmayan veya PDF olmayan kaynak manuel aktivasyon
incelemesine giremez. İnceleme sonrasında bile madde/sayfa iddiası, ölçüm
sözleşmesi, sentetik test ve gerçek regresyon ayrı olarak tamamlanır.

## AD-021 — Sayısal Accuracy kararı belirsizlik kapılı ve küçük hatayla sınırlıdır

Karar: Açık bir kaynak geometrisine bağlı sayısal ölçüm, `%95` ölçüm aralığının
tamamı kaynak sınırının dışında kaldığında yalnız `minor=-0,1` oluşturabilir.
Aralık sınırla çakışırsa sonuç `boundary_uncertain` olur ve puan kesilmez.
Sayısal sapmanın büyüklüğü hiçbir zaman kendiliğinden `major=-0,3` olmaz.

Tarihsel 2014 resmî geometri ölçüleri sürümlü, hash ve sayfa bağlı ayrı
`provisional_historical_geometry` profilde kullanılabilir; her karar kaynağın
güncel WT eki olmadığını taşır. Güncel WT `-0,3` yalnız yanlış hareket/duruş,
kihap, üç saniyelik duraklama ve bakış yönü gibi doğrudan gözlenen kategorik
olay sözleşmesiyle; restart `-0,6` performans kapsamlı gözlemle uygulanır.

Gerekçe: Kural kitabı kesinti miktarını verir ama açı sapmasının kaç derecede
“major” olacağını tanımlamaz. Pose oynaklığını yok saymak sınırdaki ölçümleri
yanlış hataya, büyük sayısal sapmayı `-0,3` yapmak ise kaynakta olmayan bir
şiddet formülüne dönüştürürdü.

Koruma: Arka ayak açısı iki ayağın birbirine göre değil, arka ayak bileğinden
ön ayak bileğine duruş doğrultusuna göre ölçülür. Fixation penceresinin sağlam
MAD belirsizliği ile metrik tabanı birlikte kullanılır. Hareket ve `error_unit`
anahtarı aynı fiziksel hatayı iki kez kesmeyi engeller. Kısmi timeline yalnız
`observed_scope_provisional_deduction_total` üretir; `accuracy_score=null`
kalır.

## AD-022 — Süre ve sınır ihlali Accuracy değil final skor kesintisidir

Karar: WT Article 16.3.1 süre ihlali ve 16.3.2 yarışma alanı sınırını geçme
olayları `accuracy.deductions` içine karıştırılmaz. RulePack 1.1.0 bunları ayrı
`final_score_deductions` alanında, final skordan `-0,3` ve performans kapsamlı
olaylar olarak taşır.

Gerekçe: Bu kesintiler bireysel hareket tekniği hatası değildir. Accuracy
skorundan düşülmeleri `4,0` bütçesini yanlış değiştirir; doğru uygulama jüri
Accuracy ve Presentation birleşiminden sonraki final skordur.

Koruma: RulePack doğrulayıcısı iki final kesintisinin anahtarlarını, pozitif
miktarını, `applies_to=final_score`, `scope=performance`, kaynak referansını ve
frekans sözleşmesini doğrular. Article 16.3.2 aynı performansta tekrar eden
sınır geçişlerinin sıklığını açıkça söylemediği için `per_performance` şimdilik
provisional metadata'dır; açıklık sağlanana kadar otomatik runtime kesintisi
uygulanmaz.

## AD-023 — Kaynaksız generic 0-100 puan motoru kaldırılmıştır

Karar: `src/scoring_engine.py`, ona ait `config/scoring_config.yaml` ve
`provisional_scoring_report.json` üretimi kaldırılmıştır. 3B hazırlık aracı
yalnız yumuşatılmış iskelet, kalite, biomekanik ve hareket-segmenti kanıtları
üretir; hiçbir puan hesaplamaz.

Gerekçe: Eski motorun 10°/130°/0,70 eşikleri ve bileşen ağırlıkları resmî WT
kaynağına bağlı değildi. Açıkça provisional etiketlenmiş olsa da yeni
RulePack tabanlı motorla birlikte bulunması iki farklı puan gerçeği yaratıyor
ve tarihsel `71,7178/100` değerinin WT puanı sanılmasına yol açıyordu.

Koruma: Görüntü işleme, RGBD fusion, WholeBody-133, kalite kapıları,
biomekanik zaman serileri ve segment adayları korunur. Sayısal Accuracy kararı
yalnız sürümlü kaynak, PoomsaeSpec, timeline, observability ve belirsizlik
kapılarını kullanan `src/poomsae_scoring/` hattında üretilebilir. Eski run
çıktıları provenance amacıyla silinmez fakat yeniden üretilmez.

## AD-024 — Puan kararı ve görsel hata kanıtı tek yönlü ayrılır

Karar: Kaynak-bağlı Accuracy kararları değişmez bir `EvidenceEvent`
sözleşmesine çevrilir. Olay; hareket/faz, kare-zaman aralığı, karar durumu,
kalibre 3B ölçüm, `%95` belirsizlik aralığı, kaynak sınırı, kesinti ve yalnız
görselleştirmede kullanılacak eklem geometrisini taşır. İşaretli MP4 ve HTML
inceleme ekranı bu olayları tüketir; hiçbir görselleştirme çıktısı puan motoruna
geri beslenmez.

Gerekçe: Kullanıcı “yumruk/dirsek neden hatalı?” sorusunun cevabını doğrudan
iki kamera üzerinde görebilmelidir. Bununla birlikte kamera üstüne çizilen 2B
iskelet, çok kameralı 3B kararın bağımsız doğrulaması değildir. 2B çizimden
yeniden açı veya kesinti hesaplamak iki farklı puan gerçeği ve dairesel kanıt
oluştururdu.

Koruma: İşaretli video kaynak videoların bütün karelerini aynı sırayla korur;
kare atmaz ve ham videoların üzerine yazmaz. Okunabilirlik için her benzersiz
doğrulanmış kesinti anchor'ından sonra 3 saniyelik tekrar kareleri eklenebilir.
Bu ek süre manifestte açıkça kayıtlıdır; uzatılmış video ham kameralarla aynı
zaman eksenindeymiş gibi HTML senkronizasyonuna sokulmaz. Renkler yalnız karar durumunu
anlatır: kırmızı doğrulanmış kesinti, amber sınır-belirsiz, gri ölçülemez,
yeşil kaynak aralığında. Panel açıkça sayısal kararın 3B'den, çizimin ise
kayıtlı ViTPose 2B izinden geldiğini yazar. Perspektifli kamera düzleminde
`30°` kabul geometrisi uydurulmaz; kural ve ölçüm ayrı üstten-görünüş 3B
şemasında karşılaştırılır. Kullanıcının `Doğru / Yanlış /
Belirsiz` inceleme girdisi ayrı JSON olarak dışa aktarılır; özgün karar JSON'u
değiştirilmez ve otomatik olarak RulePack'e geri yazılmaz.

## AD-025 — Mevcut geliştirme kapsamı M01-M06 ve WholeBody teşhisi puandan ayrıdır

Karar: Mevcut kısa kayıtta aktif geliştirme kapsamı M01-M06'dır. Çalıştırıcı
bu kapsamı `6/6` olarak raporlar; bütün Taegeuk 1 kapsamındaki `6/18` bilgisi
yalnız provenance ve tam Poomsae skoru koruması için ayrı tutulur. M07-M18'in
yokluğu M01-M06 ölçüm, karar ve kanıt üretimini engellemez; kısa kayıttan 4,0
tam Accuracy puanı üretmeye de izin vermez.

WholeBody-133 el/yüz/gövde eşik dışı ölçümleri `diagnostic_review_candidate`
olarak görsel olay zincirine katılabilir. Bu olayların `deduction_points` ve
`score_effect` alanları `null` kalır, mavi gösterilir ve puan-kesintisi
dondurması oluşturmaz. Kaynak-bağlı kesintiler kırmızı ve ayrı kalır. Yumruk
kapalılığı 21 el noktasından; baş/yüz–gövde yön farkı 68 yüz noktası ile omuz
hattından ölçülür. İkincisi göz küresi bakış takibi olarak adlandırılmaz.

Gerekçe: COCO-WholeBody noktaları zaten mevcutken el ve yüz ölçülerini yalnız
ham JSON'da bırakmak kullanıcıya sistemin bunları kullanmadığı izlenimini
veriyordu. Buna karşılık mühendislik tarama eşiklerini kaynak-bağlı WT
kesintisine çevirmek de ikinci, kaynaksız bir puan motoru yaratırdı.

Koruma: HTML bütün seçili el/yüz ölçülerini değer ve ölçülebilirlik durumuyla
gösterir. Video yalnız eşik dışı teşhis adaylarını ekler ve kart üzerinde
`puan yok` yazar. Manifest kesinti dondurma karelerini ayrı kaydeder; teşhis
olayları bu listeye giremez. Tam kayıt uygunluğu sözleşmedeki
`status=complete` ve `recording_scope=complete_performance` adlarıyla sınanır.
İzole eklem boşluğu yalnız aynı hareket/fixation penceresindeki yeterli gerçek
örneklerden tamamlanabilir. Bütün pencere boyunca kayıp omuz, bilek veya ayak
bileği simetri, komşu eklem ya da 3B projeksiyonla uydurulmaz. Ölçülemez karar
kritik eklem adlarını ve pencere örnek sayılarını kanıt alanında taşır.

Uygulama ve doğrulama aşamaları:
[`docs/PUANLAMA_PLANI.md`](PUANLAMA_PLANI.md).

## AD-026 — Performans telemetrisi opsiyonel ve bilimsel artifact'lerden ayrıdır

Karar: Multiview ve Poomsae uygulamalarındaki performans telemetrisi varsayılan
olarak kapalıdır. Açıldığında hiyerarşik CPU duvar-zamanı, desteklenen PyTorch
CUDA aşamalarında CUDA Event süresi ve Torch peak allocated/reserved VRAM
ayrı `json/performance_report.json` artifact'ine yazılır. Parent süreleri child
sürelerini kapsar; rapor tüketicisi bunları birbirine ekleyemez. Opaque
RF-DETR `predict` GPU kuyruğunu kapsayan senkronize duvar-zamanıyla, ZED SDK
çağrıları ise doğal senkron CPU duvar-zamanıyla etiketlenir. Torch ölçümü ZED
SDK veya driver'ın toplam GPU belleği olarak sunulamaz.

Gerekçe: Optimizasyon kararı, aynı protokolde tekrar edilen ve gerçek aktif
workflow'dan alınmış ölçüme dayanmalıdır. Ölçüm kodunun ana 3B, kalite,
depth-fusion, optimizer, readiness veya Poomsae karar artifact'lerine alan
eklemesi davranış-dondurma karşılaştırmasını belirsizleştirirdi. Sistem RAM/CPU
için yeni ağır runtime bağımlılığı eklemek de salt ölçüm aşamasının kapsamını
aşardı; güvenilir kaynak yoksa değer açıkça `unavailable` kalır.

Koruma: Profil açmak model/algoritma parametresi, frame/timestamp kimliği,
stride, eklem sayısı, kamera kanıtı, kalite kapısı veya fallback kararını
değiştiremez. CUDA peak reset ancak modeller yüklendikten sonra yapılır; model
resident allocation başlangıç tabanında korunur. Benchmark penceresi yalnız
raporlama filtresidir ve kare atlamaz. Poomsae analiz/karar ile
presentation/export süreleri ayrılır. Önce/sonra iddiası aynı veri bölümü,
kamera seti, stride, model ve config ile; değişken run/provenance alanları
hariç bilimsel artifact eşitliğiyle doğrulanır. Phase D ölçüm sonuçları tek
başına bir optimizasyonu yetkilendirmez.

## AD-027 — Kapsamlı teknik doğruluk ayrı, puansız ve yön bağı fail-closed katmandır

Karar: Taegeuk 1'in mevcut WholeBody-133 ile gözlenebilir teknik doğruluk alanı,
v2 ölçüm ve source-bound karar katmanlarını değiştirmeden ayrı v3 diagnostic
profilinde tanımlanır. M01–M18 hareket kontratı, 174 kural envanteri,
measurement/evaluation ayrımı ve kural-hareket kapsam matrisi kanonik
`tk3d-poomsae` uygulamasının artifact'idir.

Sporcu-yerel semantik yön, session kimliği ve immutable reference pose hash'i
ile bağlı ayrı bir referans olmadan dünya eksenlerine eşlenmez. Bağ yoksa
mutlak yön kuralları `blocked_missing_reference`, bağıl gövde geometrisi ise
ölçülebilir kalır. Manuel referans yalnız diagnostic provenance'dır; üretim
kalibrasyonu sayılmaz.

Baş/yüz geometrisi `head_orientation_proxy` olarak adlandırılır; gerçek pupil,
gaze, dikkat veya rakip farkındalığı iddiası yasaktır. Pelvis/body-centre ve
fixation dağılımları geometrik proxy'dir; merkez kütle, basınç, ağırlık
dağılımı, kuvvet veya güç değildir.

Koruma: Bütün v3 adaylar `review_candidate_not_deduction`, `score_effect=null`,
`deduction_points=null`, `numeric_score_enabled=false` ve
`deduction_enabled=false` taşır. Eşikler yalnız v3 YAML'da sürümlenir ve
`self_authored_temporary_accuracy_rule` olarak işaretlenir. Source-bound karar
dizileri, Accuracy toplamı, readiness ve Presentation bileşenleri v3'ten veri
almaz. Kanıt zinciri yalnız JSON/CSV → mavi EvidenceEvent → video/HTML yönünde
ilerler; render çıktısı yeniden değerlendirilmez. M07–M18 aktif videoda yoktur
ve yalnız kontrat/config/sentetik test kapsamıdır.

Ölçüm uygulama sözleşmesi: Pipeline sınırındaki 8 özellik dışında kalan 166
kuralın her biri bir evaluator kimliği taşır. Eksik evaluator, eksik sayısal
teknik hedef ve yetersiz video kanıtı aynı durum olarak raporlanamaz. Zorunlu
landmark grubu profilin minimum geçerlilik oranını karşılamazsa sonlu birkaç
örnek bulunsa bile ölçüm aday üretmeden `unmeasurable` olur. Ayrıntılı el
kararlılığı palm/wrist WholeBody noktalarından, ayak kararlılığı heel/toe
noktalarından hesaplanır; BODY wrist/ankle bunların yerine kanıt sayılmaz.

Gerekçe: Eski v2, ölçülebilir birkaç metriği sağlıyordu fakat eksik kuralları,
yön referansı boşluğunu ve M01–M18 sözleşme kapsamını tek makine-okunur yerde
göstermiyordu. Bu ayrım eksik kanıtı sessizce atlamak yerine açık durum koduyla
korurken ikinci bir skor gerçeği yaratmaz.

Ayrıntı:
[`TECHNICAL_ACCURACY_DIAGNOSTICS.md`](TECHNICAL_ACCURACY_DIAGNOSTICS.md).

## AD-028 — Segment tespiti tek yöntemde birleşir: ölçülen hareket-enerjisi dedektörü

Karar: Hareket ve faz sınırı önerisi yalnız `automatic_segmentation.py` tarafından
üretilir. Paralel geliştirilen hold (duruş) tabanlı dedektör
`src/poomsae_scoring/segmentation.py` ve eş işlevli
`scripts/derive_poomsae_categorical_observations.py` kaldırılmıştır.

Gerekçe: İki dedektör aynı elle etiketlenmiş M01–M06 kaydına karşı aynı ölçütlerle
ölçüldü (31 Ağustos 2026; `poomsae1-zed2i-rgbd-rerun-20260802-draft` timeline'ı,
741 kare, 60 fps). Kare cinsinden ortalama mutlak hata:

| Dedektör | Başlangıç | Bitiş | Fixation ort. | Fixation en kötü |
| --- | ---: | ---: | ---: | ---: |
| Hareket enerjisi (`automatic_segmentation`) | **11,3** | **10,3** | **6,8** | **13** |
| Hold tabanlı (`segmentation`) | 34,2 | 10,7 | 12,5 | 22 |
| Karışım (hareket sınırı + hold fixation) | 11,3 | 10,3 | 12,5 | 22 |

Hareket dedektörü her iki metrikte de öndedir. "Her yöntemin iyi yanını al"
varsayımı ayrıca ölçüldü: sınırları hareket dedektöründen, fixation anını hold
dedektöründen alan karışım denendi ve hareket dedektöründen kesin olarak kötü
çıktı; fixation hatasını 6,8'den 12,5'e taşıdı. Bu yüzden karışım da
uygulanmamıştır. Hold dedektörünün belgelenmiş amacı — otomatik hizalama için
segment kaynağı olmak — hareket dedektörü tarafından daha iyi karşılanmaktadır.

Sınır: Bu sonuç hiçbir dedektörü elle etiketlemenin yerine koymaz. Fixation ölçüm
penceresi ±5 karedir; kazanan dedektörün 6,8 karelik ortalaması ve 13 karelik en
kötü hatası bu pencereden büyüktür. Öneriler insan onayı gerektirir ve
`confirmed_timeline_replacement_allowed: False` korunur. Ölçüm tek sporcunun tek
kaydının altı hareketine dayanır; genel bir doğruluk iddiası değildir.

Kategorik gözlemler: Duraklama, yanlış hareket ve yanlış duruş gözlemleri
`run_categorical_poomsae_diagnostics.py` üzerinden üretilir ve kanonik akışta
`--observations` girdisi olarak source-bound karar katmanına gider. Timeline'dan
tek başına duraklama türeten eski script bu gözlemlerin dar bir alt kümesini
üretiyordu ve hiçbir yerden çağrılmıyordu.

## AD-029 — Kural doğruluğu yazılım kontratı ve dış doğruluk olarak ikiye ayrılır

Karar: Taegeuk 1 v3 teknik-doğruluk kuralları için sentetik, sürümlü ve
makine-okunur ayrı bir validation düzeneği kullanılır. Düzenek 174 kuralın
durum/evaluator/puan-sızlık envanterini, 33 aktif kuralın eşik sınırlarını ve
WholeBody-133 kanıt bozulmalarını doğrular. Runtime analiz artifact'i veya
source-bound kararları bu düzeneğin çıktısından beslenmez.

Koruma: NaN ve sonsuzluk her sayısal eşikte `unmeasurable` olur ve JSON'a ham
non-finite sayı yazılmaz. BODY-17 reddedilir; eksik yüz/el/ayak, kamera veya
reprojection kanıtı aday üretmez; yön kuralları session-bound referans olmadan
kapalı kalır. Sağ-sol ayna iki kez uygulandığında fixture geri dönmeli,
kontrollü fixation drift'i hedef kararlılık metriklerini baz çizginin üzerine
taşımalıdır. Eksik kanıt testi ancak aynı hedef baz çizgide ölçülebilirken
başarılı sayılır; ayna testi sıfır olmayan metrikleri ve bütün koordinat/kalite
dizilerini kapsar. 133 landmarkın her biri ayrı coverage satırı taşır. Girdi,
uygulama kaynakları ve çıktı artifact'leri SHA-256 ile bağlanır; staging dizini
tamamlanmadan final run görünür olmaz. Çıktı yollarının üzerine yazılmaz ve
bütün sonuçlar puansızdır.

Sınır: `status=passed` yalnız bu yazılım davranışlarını kanıtlar. Biomekanik
geçerlilik, resmî kural uyumu, hakem anlaşması, gerçek videoda precision/recall
ve sporcular arası genelleme iddiası için bağımsız etiketli dış doğrulama
gerekir. Sentetik fixture, gerçek hata doğruluğunun yerine kullanılamaz.

Ayrıntı:
[`TECHNICAL_ACCURACY_RULE_VALIDATION.md`](TECHNICAL_ACCURACY_RULE_VALIDATION.md).

## Karar değiştirme süreci

Bu kararlardan biri değiştirilecekse:

1. Değişiklik nedeni yazılır.
2. Aynı protokolde önce/sonra ölçümü yapılır.
3. Veri kaybı ve circular evidence riski değerlendirilir.
4. İlgili test ve gerçek koşu tamamlanır.
5. Bu belge, `PROJECT_STATUS.md` ve gerekiyorsa README aynı committe
   güncellenir.
