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

Uygulama ve doğrulama aşamaları:
[`docs/PUANLAMA_PLANI.md`](PUANLAMA_PLANI.md).

## Karar değiştirme süreci

Bu kararlardan biri değiştirilecekse:

1. Değişiklik nedeni yazılır.
2. Aynı protokolde önce/sonra ölçümü yapılır.
3. Veri kaybı ve circular evidence riski değerlendirilir.
4. İlgili test ve gerçek koşu tamamlanır.
5. Bu belge, `PROJECT_STATUS.md` ve gerekiyorsa README aynı committe
   güncellenir.
