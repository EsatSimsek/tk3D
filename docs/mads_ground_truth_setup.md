# MADS ground-truth 3B doğrulama

## Neden MADS?

TK3D için birincil dış doğrulama veri seti MADS (Martial Arts, Dancing and Sports) olarak seçildi.
Veri seti Karate ve Tai-chi gibi poomsae'ye yakın hızlı, dönüşlü ve kendi kendini örten hareketleri içerir.
Üç RGB kamera senkronize ve kalibredir; 3B referans pozlar 60 Hz optik motion-capture sistemiyle ölçülmüştür.
RGB video 15 fps ve 1024x768 çözünürlüktedir. Kamera, video ve motion-capture verileri ortak koordinat ve zaman
referansına kalibre edilmiştir.

Bu seçim, AIST'in kötü bir veri seti olduğu anlamına gelmez. AIST dans ve çok-kamera akış testi için değerlidir;
MADS ise TK3D'nin dövüş sanatı hareket alanına daha yakındır.

Karşılaştırılan başlıca seçenekler:

| Veri seti | Güçlü tarafı | TK3D için eksik tarafı |
| --- | --- | --- |
| MADS | Karate/Tai-chi, 3 kalibre kamera, optik 3B ground truth | Az sporcu ve yalnızca 15 fps RGB |
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

15 Temmuz 2026 kontrolünde resmî sunucu zaman aşımına uğradığı için otomatik indirme tamamlanamadı. Lisansı ve dosya
bütünlüğü doğrulanamayan üçüncü taraf aynalar projeye alınmadı.

TK3D için stereo-depth arşivi gerekli değildir. Resmî bağlantı tekrar çalıştığında yalnızca çoklu kamera parçalarını
indirmek yeterlidir:

```text
MADS_multiview.z01
MADS_multiview.z02
...
MADS_multiview.z10
MADS_multiview.zip
```

Dosyaları `data/mads_test/raw/` altına koyun. Tüm parçalar aynı klasördeyken güncel 7-Zip ile
`MADS_multiview.zip` dosyasını `data/mads_test/extracted/` içine açın. `raw/` ve `extracted/` Git tarafından
özellikle izlenmez.

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

MADS eklem adları COCO adlarıyla aynı değilse açık bir YAML eşlemesi verilmelidir. Anahtar tahmin eklemi, değer MADS
eklemidir:

```yaml
joint_map:
  left_shoulder: MADS_LEFT_SHOULDER_NAME
  right_shoulder: MADS_RIGHT_SHOULDER_NAME
  left_hip: MADS_LEFT_HIP_NAME
```

Gerçek MADS eklem adları arşiv görülmeden bu dosyaya yazılmamalıdır.

## Doğruluk raporunu çalıştırma

```powershell
python scripts\evaluate_ground_truth_3d.py `
  --prediction outputs\<session>\runs\<run_id>\json\vitpose_session_3d.json `
  --ground-truth data\mads_test\ground_truth\<sequence>.json `
  --joint-map data\mads_test\mads_to_coco.yaml `
  --output-dir outputs\mads_validation\<sequence>
```

Üretilen çıktılar:

- `ground_truth_validation_report.json`
- `ground_truth_frame_errors.csv`
- `ground_truth_joint_errors.csv`
- `ground_truth_angle_errors.csv`
- `ground_truth_frame_matches.csv`
- `validation_manifest.json` (girdi ve çıktıların SHA-256 özeti)

Video 15 fps, motion capture 60 fps olsa da değerlendirici timestamp üzerinden en yakın referans karesini eşler ve
eşleşme farkını ayrıca raporlar.

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
