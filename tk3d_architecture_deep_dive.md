# TK3D Projesi: Uçtan Uca Sistem Mimarisi, Kod Analizi ve Matematiksel Temeller (Master-Class Dokümantasyon)

Bu doküman, TK3D (Taekwondo 3D) projesini bir mühendis, sistem mimarı ve matematikçi gözüyle en ince ayrıntısına kadar incelemektedir. Amacımız sadece hangi kodun çalıştığını değil; **o kodun arkasındaki lineer cebir denklemlerini, optimizasyon tercihlerini ve derin öğrenme stratejilerini** sıfır eksikle anlatmaktır. Bunu okuyan biri, projeyi sıfırdan kendisi yazabilecek seviyeye gelecektir.

---

## 1. Yüksek Seviye Mimari ve Proje Vizyonu

### 1.1. Neden 3D Çoklu Kamera (Multi-View)?
Tekvando poomsae (hayali dövüş) performanslarında sporcu sürekli döner. Tek bir kamera kullanıldığında, sporcunun arkasında kalan kolları veya bacakları görünmez (Occlusion - Kapanma problemi). Ayrıca 2D bir videodan eklem açılarını (örneğin dizin 180 derece mi 160 derece mi açıldığını) ölçmek perspektif yanılsaması yüzünden imkansızdır. Bu nedenle sistem; birden fazla kameradan 2D görüntüleri alıp bunları uzayda 3D koordinatlara (X, Y, Z - metre cinsinden) dönüştüren bir laboratuvar gibi tasarlanmıştır.

### 1.2. Sistemin Rolü: Veri Boru Hattı (Data Pipeline)
Bu proje nihai puanı veren kural motoru değildir; puanı verecek olan motora **kusursuz, temizlenmiş, biyomekanik olarak hesaplanmış 3D iskelet verisini hazırlayan Ara Katmandır (Middleware)**. 

Akış şu şekildedir:
1. **Veri Alma & Kalibrasyon**: Kameraların uzaydaki konumu belirlenir.
2. **2D Inference**: MMPose AI modeli videodan 133 eklem noktasını bulur.
3. **Triangulation**: Kameralardan gelen pikseller uzayda kesiştirilip 3D'ye çevrilir.
4. **Validation**: Hatalı noktalar (outliers) filtrelenir.
5. **Biomechanics**: Sporcunun hızı, açıları ve hareket adımları hesaplanır.

---

## 2. Sistemin Temeli: Veri Yapıları (`src/data_structures.py`)

Projede nesne mutasyonlarından kaçınmak ve RAM verimliliğini sağlamak için standart Python sınıfları yerine katı tipli (strongly typed) `@dataclass(slots=True)` yapısı kullanılmıştır.

```python
from dataclasses import dataclass
import numpy as np

# Neden 133? Çünkü Tekvando'da ellerin açık/kapalı olması, yüzün baktığı yön kritiktir.
# Standart 17 noktalı COCO modeli yerine, 133 noktalı COCO-WholeBody modeli kullanılmıştır.
COCO_WHOLEBODY_KEYPOINTS = 133

@dataclass(slots=True)
class PersonPose2D:
    camera_id: str
    frame_idx: int
    keypoints_xy: np.ndarray # Shape: (133, 2) -> Her eklemin X ve Y pikseli.
    scores: np.ndarray       # Shape: (133,)  -> Modelin bu noktaya güveni (0.0 - 1.0)
    valid_mask: np.ndarray   # Threshold'u geçenler (True/False)

    def __post_init__(self) -> None:
        # Kodun kırılmaması için matris boyutları instance oluşurken denetlenir. (Fail-fast prensibi)
        self.keypoints_xy = _as_shape(self.keypoints_xy, (133, 2), "keypoints_xy")

@dataclass(slots=True)
class CameraCalibration:
    camera_id: str
    image_size: tuple[int, int]
    intrinsic_matrix: np.ndarray        # Kameranın iç mercek yapısı (K matrisi)
    distortion_coefficients: np.ndarray # Mercek bükülmeleri
    rotation_vector: np.ndarray         # Kameranın açısı
    translation_vector: np.ndarray      # Kameranın konumu
    projection_matrix: np.ndarray       # P = K[R|t] -> 3D'den 2D'ye düşüren matris.
```
**Mimari Karar:** Bellek sızıntılarını önlemek ve Numpy dizilerinin standart dışı boyutlarda sisteme sızmasını engellemek için `__post_init__` kullanılmıştır.

---

## 3. Pinhole Kamera Modeli ve Kalibrasyon (`src/camera_calibration.py`)

Triangulation (üçgenleme) yapılabilmesi için kameraların fiziksel yapısı bilinmelidir.

### 3.1. Matematiksel Temel (P = K[R|t])
Bir kameranın dünya üzerindeki X,Y,Z koordinatını 2D ekran pikseline (u,v) nasıl dönüştürdüğü **Projeksiyon Matrisi (P)** ile hesaplanır.
*   **K (Intrinsic Matrix - İç Matris):** Kameranın odak uzaklığı ($f_x, f_y$) ve optik merkezi ($c_x, c_y$). Lensin içine aittir.
*   **[R|t] (Extrinsic Matrix - Dış Matris):** Kameranın odaya göre hangi açıda durduğu (R - Rotation 3x3) ve nerede durduğu (t - Translation 3x1).

### 3.2. Kod Analizi
```python
def calibrate_single_camera(video_path, pattern_size, square_size_m):
    # 1. Satranç tahtası (checkerboard) köşeleri bulunur ve alt-piksel hassasiyetine getirilir.
    object_points, image_points, image_size = collect_checkerboard_points(...)

    # 2. OpenCV'nin iteratif optimizasyon algoritması bu noktaları kullanarak K ve R|t'yi çözer.
    rms, intrinsic, distortion, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )
    
    # rvecs (Rotation Vector) 3x1'lik bir vektördür. Bunu 3x3 matrise çevirmek için Rodrigues dönüşümü yapılır.
    rotation_matrix, _ = cv2.Rodrigues(rvecs[0])
    
    # 3x3 Rotasyon Matrisi ile 3x1 Çeviri Matrisi yanyana birleştirilir (hstack). Çıkan matris 3x4'tür.
    extrinsic = np.hstack([rotation_matrix, tvecs[0].reshape(3, 1)])
    
    # İŞTE SİSTEMİN KALBİ: Projeksiyon matrisi K (3x3) ile Extrinsic (3x4) çarpılarak bulunur (Sonuç 3x4).
    projection = intrinsic @ extrinsic 
    
    return CameraCalibration(..., projection_matrix=projection, ...)
```

---

## 4. Yapay Zeka ile İskelet Çıkarımı (`src/pose2d_estimator.py`)

Videodaki sporcunun eklem yerlerini tespit eden yapay zeka entegrasyonu.

### 4.1. Kod Analizi ve Outlier Rejection
MMPose modeli bir frame'de birden fazla insan (örneğin arkada yürüyen hakem) bulabilir. Sistemin kafasının karışmaması için en net sporcuyu seçen bir filtre (`_extract_mmpose_wholebody`) tasarlanmıştır.

```python
class ViTPose2DEstimator:
    # Model belleği (VRAM) yormasın diye sadece predict anında veya "__init__" içinde dry_run=False ise yüklenir.
    def predict(self, frame: np.ndarray, camera_id: str, frame_idx: int) -> PersonPose2D:
        result = self._model(frame) # Modele resim verilir, karmaşık bir dict döner.
        keypoints_xy, scores = _extract_mmpose_wholebody(result, allow_padding=False)
        return pose2d_from_arrays(..., keypoints_xy, scores, self.config.score_threshold)

def _extract_mmpose_wholebody(result: Any) -> tuple[np.ndarray, np.ndarray]:
    predictions = result.get("predictions", [])
    
    # Eğer birden çok insan varsa, tüm eklemlerin güven (score) ortalaması en yüksek olan kişi "best" olarak seçilir.
    # Bu, arka plandaki insanları silmek için basit ama çok etkili bir yoldur.
    best = max(predictions, key=_prediction_score)
    keypoints = np.asarray(best["keypoints"], dtype=float)
    scores = np.asarray(best.get("keypoint_scores", np.ones(keypoints.shape[0])), dtype=float)
    
    # Veri Standartlaştırması: legacy adapter eksik noktaları NaN ile doldurabilir.
    # ViTPose-Huge WholeBody yolunda ise 133 nokta zorunludur; 17-eklem COCO modeli
    # yanlışlıkla bağlanırsa sistem erken hata verir.
    if keypoints.shape[0] < COCO_WHOLEBODY_KEYPOINTS:
        padded_xy = np.full((133, 2), np.nan, dtype=float)
        padded_scores = np.zeros(133, dtype=float)
        padded_xy[: keypoints.shape[0]] = keypoints[:, :2]
        padded_scores[: scores.shape[0]] = scores
        return padded_xy, padded_scores
        
    return keypoints[:133, :2], scores[:133]
```

---

## 5. Sistemin Kalbi: 3D Triangulation ve DLT/SVD (`src/triangulation.py`)

Burası piksellerin dünya koordinatlarına dönüştüğü sihirli matematik modülüdür. 

### 5.1. Matematiksel Temel (Epipolar Geometri ve DLT)
Her kamera lensinden çıkan bir ışın (ray) düşünün. Bu ışın kameradan çıkıp sporcunun eline gider. İki kameranın ışını uzayda mükemmel bir şekilde kesişmelidir. Ancak kameralarda çözünürlük sınırı ve lens bükülmesi olduğu için ışınlar tam kesişmez, birbirini teğet geçer. **SVD (Singular Value Decomposition)** algoritması, bu iki (veya daha fazla) ışının birbirine en yakın olduğu o hayali orta noktayı bulur (Least Squares Error - En Küçük Kareler Hatası).

### 5.2. `triangulate_n_view` Fonksiyonunun Analizi
```python
def triangulate_n_view(points_2d: list[np.ndarray], projection_mats: list[np.ndarray]) -> np.ndarray | None:
    # Eğer sadece 2 kameramız varsa OpenCV'nin hazır fonksiyonunu kullan. (Hız optimizasyonu)
    if len(points_2d) == 2:
        point_a = np.asarray(points_2d[0], dtype=float).reshape(2, 1)
        point_b = np.asarray(points_2d[1], dtype=float).reshape(2, 1)
        homogeneous = cv2.triangulatePoints(projection_mats[0], projection_mats[1], point_a, point_b)
        if abs(homogeneous[3, 0]) < 1e-12: return None
        return (homogeneous[:3, 0] / homogeneous[3, 0]).astype(float)

    # DLT (Direct Linear Transformation) Algoritmasının Manuel Kurulumu (3 ve üzeri kamera için)
    rows = []
    for point, projection in zip(points_2d, projection_mats):
        x, y = point
        # A matrisi (Design Matrix) oluşturuluyor.
        # Bu formül: Noktanın pikselleri ile Projeksiyon matrisi satırlarının çarpımıdır (Kesişim denklemi)
        # Her bir 2D nokta (x,y) için uzayda bir doğru denklemi yazılır (2 satır).
        rows.append(x * projection[2, :] - projection[0, :])
        rows.append(y * projection[2, :] - projection[1, :])
        
    design = np.asarray(rows, dtype=float)
    
    # A * X = 0 Sisteminin SVD ile çözülmesi. 
    # U, S (Singular Values), V^T döner.
    _, _, vt = np.linalg.svd(design)
    
    # Aradığımız çözüm, V^T matrisinin en alt satırıdır (En küçük özdeğere karşılık gelen özvektör).
    # Bu vektör Homojen Koordinat formatındadır: (X, Y, Z, W)
    homogeneous = vt[-1]
    
    if abs(homogeneous[3]) < 1e-12: 
        return None # Paralel ışınlar veya hatalı matris (Sıfıra bölme koruması)
        
    # Gerçek dünya (Öklid) koordinatına geçmek için (X/W, Y/W, Z/W) yapılır.
    return (homogeneous[:3] / homogeneous[3]).astype(float)
```

### 5.3. Reprojection Error (Doğrulama)
Bulduğumuz (X,Y,Z) noktası ne kadar doğru? Noktayı tekrar kameralara yansıtıp (project) eski piksellerine olan uzaklığını ölçeriz.
```python
def mean_reprojection_error(point_3d, points_2d, projection_mats):
    # Homojenleştir: (X,Y,Z) -> (X,Y,Z,1.0)
    homogeneous = np.array([point_3d[0], point_3d[1], point_3d[2], 1.0], dtype=float)
    errors = []
    
    for observed, projection in zip(points_2d, projection_mats):
        # 3D Uzayı 2D Kameraya çarp!
        projected = projection @ homogeneous
        if abs(projected[2]) < 1e-12: continue
            
        # Pikseller (u,v) bulunur
        xy = projected[:2] / projected[2]
        
        # Orijinal nokta (observed) ile yansıtılan nokta (xy) arasındaki farkı Öklid Uzaklığı ile hesapla.
        # Pisagor teoremi: Karekök((x2-x1)^2 + (y2-y1)^2) = np.linalg.norm
        errors.append(float(np.linalg.norm(xy - observed)))
        
    return float(np.mean(errors)) if errors else float("nan")
```
*(Bu değer 25 pikselin üzerindeyse, `validation_3d.py` modülü bu noktayı çöpe atıp `NaN` koyacaktır.)*

---

## 6. Puanlamaya Hazırlık: Biyomekanik ve Kinetik Matrisler (`src/scoring_readiness.py`)

Poomsae motoru için sporcunun "ne zaman tekme attığını" (segmentasyon) ve "omzunun ne kadar dik olduğunu" (biyomekanik) bulmamız gerekir. Bu işlemler onbinlerce frame için tek tek yapılmak yerine **Numpy Vektörizasyonu** kullanılarak matris düzeyinde milisaniyeler içinde halledilir.

### 6.1. Hız ve Vektörizasyon (Kinetik)
```python
def joint_speed(keypoints_3d: np.ndarray, fps: float) -> np.ndarray:
    # keypoints_3d matrisi = (Zaman, Eklemler, X-Y-Z) şeklindedir.
    speed = np.full(keypoints_3d.shape[:2], np.nan, dtype=float)
    
    # np.diff() zaman ekseninde t(2) - t(1) diyerek yer değiştirmeyi bulur.
    # np.linalg.norm() X,Y,Z değişiminin hipotenüsünü (bileşke büyüklüğünü) bulur.
    # Sonuç FPS ile çarpılarak Metre/Saniye (m/s) hızına ulaşılır.
    # Bütün bu işlemler onbinlerce kare için tek bir C++ tabanlı komutta biter (For döngüsü kullanılmaz!)
    diffs = np.linalg.norm(np.diff(keypoints_3d, axis=0), axis=-1) * max(fps, 0.0)
    
    speed[1:] = diffs
    return speed
```

### 6.2. Hareket Adımları Segmentasyonu (Dynamic Thresholding)
Videoyu "Ap-chagi (Ön tekme)" ve "Jumbi (Hazır)" gibi safhalara ayırmak gerekir.
```python
def movement_segments(keypoints_3d: np.ndarray, fps: float) -> list[dict]:
    speeds = joint_speed(keypoints_3d, fps=fps)
    
    # Sadece ilk 17 noktayı (Ana gövdeler) alıp hızı birleştirerek "Motion Energy" bulur.
    energy = _nanmean_axis1(speeds[:, :17]) 
    
    # Videonun genel enerji düzeyinin %60'lık kısmını bir baraj (threshold) olarak belirler.
    # Bu, statik baraj kullanmanın önüne geçer (her sporcu farklı hızda yapabilir).
    threshold = float(np.nanpercentile(energy, 60))
    
    # Barajın üstündekiler aktif (True), altındakiler duraklamış (False) kabul edilir.
    active = np.isfinite(energy) & (energy >= threshold) & (energy > 0)
    
    # ... Sonrasında bu array okunur, start ve end noktaları saniye bazında çıkarılır.
```

### 6.3. Biyomekanik Açılar (Gövde Eğimi)
Sporcunun ne kadar öne eğildiğini bulmak için 3D vektör matematiği kullanılır (Dot Product).
```python
def torso_lean_deg(frame: np.ndarray) -> float:
    # Omuzların ortası ve kalçaların ortası hesaplanır
    shoulder_center = _mean_points(frame, ["left_shoulder", "right_shoulder"])
    hip_center = _mean_points(frame, ["left_hip", "right_hip"])
    
    # Kalçadan omuza giden Torso (Gövde) Vektörü: A vektörü
    torso = shoulder_center - hip_center
    denom = np.linalg.norm(torso) 
    
    # Kusursuz dik duruşu temsil eden Z ekseni Vektörü: B vektörü (0,0,1)
    vertical = np.array([0.0, 0.0, 1.0], dtype=float)
    
    # İki vektör arasındaki açıyı bulmak için Nokta Çarpımı (Dot Product) formülü kullanılır: 
    # A • B = |A| * |B| * cos(Theta)  --->  cos(Theta) = (A • B) / (|A| * |B|)
    cosine = np.clip(np.dot(torso, vertical) / denom, -1.0, 1.0)
    
    # Ark Kosinüs (arccos) ile açı radyan cinsinden bulunur, degrees ile dereceye çevrilir.
    return float(np.degrees(np.arccos(abs(cosine))))
```

---

## 7. Özet ve Mühendislik Vizyonu

Bu kod yapısı okunduğunda görüleceği üzere proje, derin öğrenmenin yetersiz kaldığı noktalarda (kapanma/occlusion problemleri) **Lineer Cebir ve Klasik Fizik/Kinetik** kurallarına sarılmaktadır. 

Sistemdeki **Separation of Concerns (Sorumlulukların Ayrılığı)** o kadar iyidir ki; gelecekte Poomsae Kuralları değişse sadece `scoring_readiness.py`'a yeni bir kural fonksiyonu yazılır. Kamera modelleri değişse sadece `camera_calibration.py` güncellenir. Sistemin çekirdek Triangulation veya Inference katmanına asla dokunulmaz. 

TK3D kod tabanı, bir Poomsae maçını sıfır hatayla dijital ortama aktarabilecek, matematiksel olarak kanıtlanabilir (provable) bir veri mimarisidir.
