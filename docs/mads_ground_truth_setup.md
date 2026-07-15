# MADS ground-truth 3B doğrulama

## Neden MADS?

TK3D için birincil dış doğrulama veri seti MADS (Martial Arts, Dancing and Sports) olarak seçildi.
Veri seti Karate ve Tai-chi gibi poomsae'ye yakın hızlı, dönüşlü ve kendi kendini örten hareketleri içerir.
Üç RGB kamera senkronize ve kalibredir; 3B referans pozlar 60 Hz optik motion-capture sistemiyle ölçülmüştür.
Kamera, video ve motion-capture verileri ortak koordinat ve zaman referansına kalibre edilmiştir. Yayın 15 fps ve
1024x768 değerlerini belirtse de indirilen, çıkarılmış çoklu-görüş AVI dosyalarının gerçek başlığı 30 fps ve 512x384'tür.
Kurulum bu yüzden yayın metnini varsaymak yerine dosya başlığını kaydeder ve video/GT'yi kare indeksiyle eşler.

Bu seçim, AIST'in kötü bir veri seti olduğu anlamına gelmez. AIST dans ve çok-kamera akış testi için değerlidir;
MADS ise TK3D'nin dövüş sanatı hareket alanına daha yakındır.

Karşılaştırılan başlıca seçenekler:

| Veri seti | Güçlü tarafı | TK3D için eksik tarafı |
| --- | --- | --- |
| MADS | Karate/Tai-chi, 3 kalibre kamera, optik 3B ground truth | Az sporcu; yayın 15 fps dese de yerel AVI başlıkları 30 fps |
| TotalCapture | 8 kamera, 60 Hz, Vicon ve IMU, yaklaşık 1.9 milyon kare | Poomsae/karate hareketi yok; kayıt ve araştırma lisansı gerekiyor |
| Fit3D | 3 milyona yakın 3B iskelet, egzersiz geri bildirimi, SMPL-X | Dövüş sanatı hareketi yok; hesapla giriş gerekiyor |
| TUHAD | Gerçek taekwondo teknikleri ve uzman sporcular | Ön/yan çekimler eşzamanlı değil; optik 3B ground truth değil |

MADS, alan yakınlığı nedeniyle birincil testtir. TotalCapture ileride kamera geometrisi için ikincil genel benchmark
olarak eklenebilir. Hiçbir dış veri seti, kendi kamera düzenimizde çekilecek motion-capture eşlenmiş poomsae testinin
yerini tamamen tutmaz.

## Resmî indirme

Yalnızca üreticinin resmî sayfası kullanılmalıdır:

- https://visal.cs.cityu.edu.hk/downloads/
- Sayfadaki `Human Pose Datasets -> MADS -> download here` bağlantısı

Yerel kurulumda hem çoklu-görüş hem stereo-depth arşivleri mevcuttur. 3B RGB pipeline'ın ana benchmark'ı çoklu-görüş
verisidir; depth verisi de indekslenir, 19 eklemli ground-truth JSON'a dönüştürülür ve ileride depth füzyonu için
saklanır. Yalnız RGB benchmark kurulacaksa çoklu kamera parçaları yeterlidir:

```text
MADS_multiview.z01
MADS_multiview.z02
...
MADS_multiview.z10
MADS_multiview.zip
```

Tüm parçalar aynı klasördeyken güncel 7-Zip ile arşivleri açın. Büyük arşivler ve çıkarılmış veri Git'e eklenmez.

## Yerel MADS kurulumu

Mevcut veri kökü `C:\Users\WWWW\Desktop\MADS` olarak algılandı. Aşağıdaki komut tüm 60 diziyi indeksler; seçilen
Kata dizileri için yerel oturum dosyalarını, resmî kamera kalibrasyonlarını, metre cinsinden ground-truth JSON'larını,
SHA-256 özetlerini ve projeksiyon önizlemelerini üretir:

```powershell
python scripts\setup_mads_test.py `
  --dataset-root C:\Users\WWWW\Desktop\MADS `
  --actions Kata `
  --hash-files `
  --preview
```

Üretilen yerel dosyalar:

- `data/mads_test/local/sessions/mads_kata_*.yaml`
- `data/mads_test/local/ground_truth/multiview/Kata_*.json`
- `data/mads_test/local/ground_truth/depth/Kata_*.json`
- `data/mads_test/local/mads_manifest.json`
- `outputs/mads_kata_*/calibration/cameras.json`
- `outputs/mads_setup/previews/*_gt_overlay.png`
- `outputs/mads_setup/mads_setup_report.json`

Yerel manifest ve veri yolları makineye özgü olduğu için Git'e eklenmez. Kurulumda bulunan tüm 30 çoklu-görüş ve
30 depth dizi manifestte kayıtlıdır. Kata çoklu-görüş dizileri 15, Kata depth dizileri 19 eklemlidir; dönüştürücü bu
iki resmî şemayı ayrı ayrı destekler. NaN içeren geçersiz mocap kareleri manifestte sayılır ve değerlendirmede
kullanılmaz.

Arşiv geldikten sonra ilk işlem dosya adlarını, annotation anahtarlarını, eklem sırasını, kamera matrislerini ve
koordinat birimini kontrol etmektir. Bu bilgiler tahmin edilmemeli; arşivdeki resmî README ve metadata esas alınmalıdır.

## TK3D ground-truth veri sözleşmesi

Değerlendirici, hem tahminin hem referansın metre cinsinden TK3D analiz koordinatına dönüştürülmüş olmasını zorunlu
tutar. Yanlış eksen veya birim sessizce kabul edilmez.

Referans JSON'un asgari alanları:

```json
{
  "schema_version": 1,
  "dataset": "MADS",
  "sequence_id": "sequence_name",
  "coordinate_system": {
    "name": "tk3d_analysis",
    "unit": "meter",
    "axes": {"x": "right", "y": "forward", "z": "up"},
    "handedness": "right"
  },
  "fps": 60.0,
  "frame_indices": [0, 1],
  "timestamps_sec": [0.0, 0.0166666667],
  "joint_names": ["left_shoulder", "right_shoulder"],
  "keypoints_3d_ground_truth": [
    [[-0.2, 0.0, 1.5], [0.2, 0.0, 1.5]],
    [[-0.2, 0.01, 1.5], [0.2, 0.01, 1.5]]
  ]
}
```

MADS dönüştürücüsü resmî 15/19 eklem sırasını anatomik adlara çevirir. Ortak adlar otomatik eşlenir; farklı bir
model/şema kullanılırsa anahtar tahmin eklemi, değer MADS eklemi olacak şekilde açık YAML eşlemesi verilebilir:

```yaml
joint_map:
  left_shoulder: MADS_LEFT_SHOULDER_NAME
  right_shoulder: MADS_RIGHT_SHOULDER_NAME
  left_hip: MADS_LEFT_HIP_NAME
```

## Doğruluk raporunu çalıştırma

```powershell
python scripts\run_vitpose_multiview_3d.py `
  --session data\mads_test\local\sessions\mads_kata_f2.yaml `
  --stride 2 `
  --run-id mads-kata-f2-stride2-reliable-v1 `
  --allow-low-quality-output

python scripts\evaluate_ground_truth_3d.py `
  --prediction outputs\mads_kata_f2\runs\mads-kata-f2-stride2-reliable-v1\json\vitpose_session_3d.json `
  --ground-truth data\mads_test\local\ground_truth\multiview\Kata_F2.json `
  --output-dir outputs\mads_kata_f2\runs\mads-kata-f2-stride2-reliable-v1\ground_truth_validation `
  --allow-failed-quality-gate
```

Üretilen çıktılar:

- `ground_truth_validation_report.json`
- `ground_truth_frame_errors.csv`
- `ground_truth_joint_errors.csv`
- `ground_truth_angle_errors.csv`
- `ground_truth_frame_matches.csv`
- `validation_manifest.json` (girdi ve çıktıların SHA-256 özeti)

MADS arşivindeki hazırlanmış GT dizisi her video karesi için bir poz içerir. Dönüştürücü gerçek AVI fps değerini
kaydeder; değerlendirici frame/timestamp eşleşme farkını ayrıca raporlar.

## Ölçülen sonuçlar

Kata F2 üzerinde 600 kareye eşit aralıklı 30 örnekle yapılan diagnostik çalışmada resmî üç kamera kalibrasyonu
kullanıldı. Seyrek örneklerde uzak anları birbirine karıştırmamak için smoothing otomatik olarak kapatıldı.

- Ortak anatomik eklem: 12
- Global MPJPE: 149.0 mm
- Median hata: 79.7 mm
- Root-relative MPJPE: 123.0 mm
- PA-MPJPE: 94.7 mm
- PCK@100 mm: %70
- 2B GT projeksiyonuna medyan uzaklık: yaklaşık 8 piksel

Bu bir başarı sonucu değildir: ground-truth kalite kapısı doğru biçimde `failed_ground_truth_quality_gate` verdi.
Önceki yanlış smoothing davranışı aynı testte 276.1 mm üretmişti; düzeltmeden sonra hata 149.0 mm'ye indi. Kalan
hata ağırlıklı olarak birkaç 2B eklem sapmasının triangulation sırasında derinlik hatasına büyümesidir. Kalibrasyon
projeksiyonları üç kamerada görüntüyle örtüşmekte ve sağ/sol eklem eşlemesi doğrulanmıştır. 30 kare yalnız diagnostik
örnektir.

Asgari örnek koşulunu karşılayan ikinci çalışmada F2'nin 300 karesi (stride 2) ölçüldü:

- Değerlendirilen nokta: 3.598 (12 eklem)
- Geçerli eklem oranı: %99,94
- Global MPJPE: 100,6 mm
- Median hata: 78,9 mm
- P95 hata: 154,3 mm
- Root-relative MPJPE: 116,5 mm
- PA-MPJPE: 85,1 mm
- PCK@100 mm: %73,6
- Açı MAE: 13,8 derece

300 kare koşulu ve geçerli eklem oranı geçti; doğruluk, PCK, açı, dinamik ve kemik kararlılığı kapıları geçmedi.
Sonuç `failed_ground_truth_quality_gate` olduğundan sistem henüz sayısal poomsae puanı üretmeye hazır değildir.
Canlı pipeline'ın kendi `run_quality_report.json` raporu yalnız kalibrasyon/reprojection/geçerlilik gibi iç geometrik
kontrolleri kapsar; `quality_scope=internal_geometry_only`, `ground_truth_accuracy_evaluated=false` ve
`scoring_ready=false` alanları bu ayrımı açıkça kaydeder. Resmî kıyas için kalan Kata/Taichi sekansları da sporcu ve
sekans ayrımlı olarak çalıştırılmalıdır.

UDP/DARK kod çözme, `window_size=1` geçerlilik-maskesi düzeltmesi ve anatomik/zamansal güvenilirlik filtresi eklenen
güncel `reliable-v1` çalışması aynı 300 karede şu sonucu verdi:

- Değerlendirilen nokta: 3.530 (12 eklem)
- Geçerli eklem oranı: %98,06
- Global MPJPE: 82,5 mm
- Median hata: 76,3 mm
- P95 hata: 141,1 mm
- Root-relative MPJPE: 102,1 mm
- PA-MPJPE: 68,5 mm
- PCK@100 mm: %77,5
- Açı MAE: 13,1 derece
- Kemik uzunluğu CV: %5,6 (önceki %28,8)

Filtre, 5.100 vücut noktasının 5.025'ini güvenilir bıraktı; 25 temel kalite, 38 kemik uzunluğu ve 12 zamansal ret
kaydetti. Dolayısıyla iyileşme kapsama oranını çökertmeden elde edildi. Buna rağmen 50 mm MPJPE kalite hedefi
geçilmediği için durum hâlâ `failed_ground_truth_quality_gate` ve `scoring_ready=false` olarak kalır. Kalan hata
özellikle COCO görüntü eklemi ile MADS mocap eklem merkezi tanımı farklı olan kalça/dizlerde sistematiktir.

## MADS domain-adaptation deneyi ve üretim kararı

Adapter eğitimi için F2 daha baştan test kümesi olarak ayrıldı ve özellik cache'ine girmesi kod seviyesinde
engellendi. Doğrulama dizileri `Kata:F3` ve `Taichi:S6`; eğitim dizileri kalan Kata/Taichi dizileridir. MADS metadata'sı
açık sporcu kimliği vermediği için bu ayrım sekans bazlıdır; kişi bazlı sızıntısızlık ancak kendi veri setimizde açık
sporcu kimlikleriyle garanti edilebilir.

İki güvenli aday denendi:

1. ViTPose-Huge omurgası dondurularak yalnız heatmap head'in son katmanı eğitildi. Doğrulama heatmap kaybı
   `0,00160479` değerinden `0,00137724` değerine indi; buna rağmen hiç görülmemiş F2'nin 30 karelik 3B kontrolünde
   geçerli eklem oranı `%52,5`, MPJPE `473,1 mm` oldu. Aday reddedildi.
2. Eğitim dizilerinden robust eklem-offset kalibrasyonu yapıldı. 2B doğrulama merkez hatası `1,8455` heatmap
   pikselinden `1,6860` değerine indi (`%8,6`). Aynı F2 3B kontrolünde temel model `100,170 mm` MPJPE ve
   `144,601 mm` P95 üretirken offset adayı `105,568 mm` MPJPE ve `140,768 mm` P95 üretti. P95 azalsa da ana metrik
   kötüleştiği için bu aday da reddedildi.

Bu sonuç önemli bir güvenilirlik kontrolüdür: yalnız eğitim/2B doğrulama metriği iyileşti diye model üretime
alınmamıştır. Üretim yapılandırması hiçbir MADS adapter'a bağlı değildir ve doğrulanmış temel `ViTPose-Huge-WholeBody`
modelini kullanır. Adapter checkpoint'leri `production_approved=false` taşır; normal runtime bunları reddeder.
`allow_unapproved_adapter=true` yalnız açık tanısal benchmark için kullanılabilir ve puanlama onayı anlamına gelmez.

Deneyleri yeniden üretmek için:

```powershell
python scripts\train_mads_vitpose_adapter.py `
  --dataset-root C:\Users\WWWW\Desktop\MADS `
  --test-sequences Kata:F2 `
  --validation-sequences Kata:F3 Taichi:S6

python scripts\calibrate_mads_vitpose_offsets.py
```

Yerel ağırlıklar ve feature cache'leri Git'e eklenmez. Yeni bir aday ancak en az 300 F2 örneğinde temel modelden daha
düşük MPJPE, daha kötü olmayan P95 ve en az `%95` geçerli eklem oranı gösterirse adapter adayı olarak değerlendirilebilir.
Bu yine 50 mm ground-truth kalite kapısını veya kendi taekwondo saha doğrulamasını otomatik olarak geçmiş sayılmaz.

Ground-truth değerlendiricisi artık kalite kapısı başarısızsa varsayılan olarak başarısız süreç koduyla çıkar. Yalnız
tanısal rapor üretiminde açık `--allow-failed-quality-gate` kullanılabilir. Puanlama analizi de doğrulanmamış 3B
çalışmaları varsayılan olarak reddeder; geliştirme amaçlı geçici skor ancak açık
`--allow-unvalidated-provisional-score` seçeneğiyle üretilebilir.

## Ölçülen güvenilirlik

Ana metrik global MPJPE'dir. Pelvis-relative MPJPE ve PA-MPJPE yalnızca tanı amaçlıdır; global konum, dönüş veya ölçek
hatalarını gizleyebilecekleri için birincil başarı ölçüsü olarak kullanılmazlar.

Rapor ayrıca şunları üretir:

- MPJPE, median ve 95. yüzdelik hata (mm)
- MPJPE için bootstrap %95 güven aralığı
- 50 mm ve 100 mm PCK-3D
- Eklem bazında geçerlilik ve hata
- Diz, kalça, omuz ve dirsek açı hataları
- Hız ve ivme hataları
- Kemik uzunluğu kararlılığı
- Kare bazında hata ve timestamp eşleşme denetimi

Başlangıç mühendislik eşikleri `config/ground_truth_validation.yaml` içindedir. Bunlar resmî spor standardı değildir;
uzman etiketleri ve kendi motion-capture kayıtlarımız geldikçe sürümlenerek güncellenmelidir.

## Maksimum güvenilirlik için tamamlanması gereken saha doğrulaması

1. En az 4-8 global-shutter kamera, 60 fps veya üzeri, kısa pozlama ve donanımsal senkron kullanın.
2. Her çekim gününde ortak hacim kalibrasyonu yapın; kalibrasyon driftini başlangıç ve bitişte kontrol edin.
3. Aynı çekimde Vicon/OptiTrack benzeri optik motion capture ile kamera timestamp'lerini ortak tetikleyin.
4. Marker-to-anatomical-joint dönüşümünü fonksiyonel kalibrasyonla belirleyin; yalnız marker konumunu eklem merkezi saymayın.
5. Farklı boy, kıyafet, seviye ve vücut yapılarında sporcuları; tüm Taegeuk ve hedef siyah kuşak poomsae'lerini çekin.
6. Eğitim, ayar ve test bölümlerini sporcu ve çekim oturumu bazında ayırın; aynı kişi iki bölüme sızmamalıdır.
7. Hızlı tekme, dönüş, çapraz kol, kamera kapanması ve görüntü dışına çıkmayı ayrı zorluk kümeleri olarak raporlayın.
8. En az üç hakemin bağımsız adım/hata/puan etiketini toplayın; hakemler arası uyumu da ölçün.
9. Sistem belirsizliğini gerçek hatayla kalibre edin ve kalite kapısını geçmeyen kare/adım için puan üretmeyin.
10. Model, kalibrasyon, eşik, veri ve kod sürümünü her raporda hash ile sabitleyin; zaman içinde drift testi çalıştırın.

Bu dış benchmark ve saha protokolü tamamlanmadan sistem yalnızca `provisional_not_official` puan üretmelidir.
