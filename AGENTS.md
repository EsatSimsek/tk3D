# TK3D Agent Çalışma Sözleşmesi

Bu dosya repository kökünün tamamı için geçerlidir. Yeni bir kodlama oturumunda
önce bu dosyayı, ardından `PROJECT_STATUS.md` ve görevle ilgili kaynak dosyaları
oku.

## Oturuma başlama

1. Her yeni sohbet/çalışma oturumunda ilk değişiklikten önce
   `git status --short` çalıştır.
2. Kullanıcı aksini söylemedikçe `git pull --ff-only` ile güncel durumu al.
3. Çalışma ağacı kirliyse kullanıcı değişikliklerini koru. Stash, reset,
   checkout, revert veya toplu yeniden yazma yapma.
4. `PROJECT_STATUS.md` içindeki doğrulanmış commit, test ve benchmark
   bilgilerinin hâlâ geçerli olup olmadığını kontrol et.
5. Görevin kapsamını koru. Görüntü işleme görevi puanlama sistemini değiştirme
   yetkisi vermez; puanlama görevi de pose hattını sebepsiz değiştirmez.

## Kullanıcıyla çalışma

- Türkçe, açık ve ölçüme dayalı iletişim kur.
- Sonucu “kusursuz”, “üretime hazır” veya “puanlamaya hazır” diye nitelemek
  için uygun dış doğrulama gerekir.
- Commit ve push yalnız kullanıcı istediğinde yapılır. Push öncesi test, diff
  ve stage kapsamı doğrulanır.
- Kullanıcının çalıştıracağı komutları tek, kopyalanabilir PowerShell blokları
  halinde ver ve çıktıların tam Windows yolunu belirt.

## Kodlama ilkeleri

- Önce kök nedeni bul; yalnız görsel belirtiyi maskeleyen filtre ekleme.
- Mevcut modülleri ve veri sözleşmelerini kullan. Büyük yeni dosya veya
  bağımlılık eklemeden önce benzer işlevin varlığını ara.
- Ana 3B veri sözleşmesi `keypoints_3d_world[t, 133, 3]` ve metre cinsinden
  TK3D analiz koordinatlarıdır. BODY-17 optimizasyonları 133 noktalı çıktıyı
  küçültmemelidir.
- Eksik değer JSON'da `null`, CSV'de boş hücre olmalıdır. `NaN`/`inf` metni
  downstream çıktılara sızmamalıdır.
- Yaklaşık veya sentetik kalibrasyonu üretim kalibrasyonu gibi işaretleme.
- Onaysız model adapter'ını normal runtime'da etkinleştirme.
- Veri setine özel eşik, kamera ofseti veya çözümü genel varsayılana taşıma.

## Görüntü işleme güvenlik sınırları

- Stabilizasyon kare atarak, video süresini kısaltarak veya eklem sayısını
  azaltarak “iyileşmiş” gösterilemez.
- `stride > 1` yalnız inference örneklemesidir; çıktı zaman çizelgesi kaynak
  videonun süresini korumalıdır.
- Çok-kameralı 2B geri beslemede hedef kamera, kendi 3B öncülünün
  triangulation hesabından çıkarılmalıdır.
- 3B izdüşüm yalnız arama öncülü veya açıkça işaretli görselleştirme fallback'i
  olabilir; bağımsız görüntü kanıtı gibi tekrar triangulation'a sokulamaz.
- Görüntü kanıtı bulunmayan aykırı ölçüm geometriden çıkarılabilir fakat
  “model tarafından bulundu” diye etiketlenemez.
- Global optimizasyon ham triangulation'ı silmez. Kalite kapısı başarısızsa
  güvenli ham sonuca dönülmelidir.
- Kalite karşılaştırmaları aynı video bölümü, kamera seti, stride, model ve
  yapılandırmayla yapılmalıdır. Smoke test ile tam koşu karşılaştırılmamalıdır.

## Çıktı ve veri yönetimi

- Her çalışma benzersiz bir `run_id` ile
  `outputs/<session_id>/runs/<run_id>/` altında oluşturulur.
- Var olan run'ın üzerine yazma. `latest_run.json` yalnız başarıyla tamamlanan
  uygun koşuyu göstermelidir.
- Video, veri seti, checkpoint, SMPL modeli, sanal ortam ve `outputs/`
  içeriklerini Git'e ekleme.
- Provenance ve önce/sonra raporlarını koru. Ham triangulation veya ham 2B
  ölçümleri yalnız kullanıcı açıkça isterse temizlenebilir.

## Doğrulama ve teslim

- En küçük ilgili testi önce, ardından değişikliğin riskine uygun daha geniş
  testi çalıştır.
- Python değişikliklerinde varsayılan teslim kapısı:

```powershell
.\.venv312\Scripts\python.exe -m ruff check src scripts tests
.\.venv312\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp outputs\pytest-<benzersiz-id>
git diff --check
```

- Gerçek inference davranışı değiştiyse en az kısa gerçek smoke koşusu ve
  ilgili JSON kalite raporu incelenmelidir. Yalnız unit test yeterli değildir.
- Kalite iddiasında geçerli eklem oranı, reprojection dağılımı, kamera kanıtı,
  temporal/angle jitter ve varsa ground-truth metriği birlikte raporlanır.
- İç geometri kapısının geçmesi ground-truth doğruluğu veya resmî puanlama
  doğruluğu anlamına gelmez.
- Kullanıcıya yalnız gerçekten çalıştırılan testleri ve gerçekten oluşan
  çıktıları bildir.
- Kullanıcıya dönük davranış, kurulum, komut veya çıktı yapısı değiştiyse
  `README.md`; güncel proje gerçeği değiştiyse `PROJECT_STATUS.md`; mimari
  karar değiştiyse `docs/ARCHITECTURE_DECISIONS.md` güncellenmelidir.

## Doküman önceliği

Çelişki olduğunda şu sıra kullanılır:

1. Güncel kod, yapılandırma ve testler
2. `AGENTS.md`
3. `PROJECT_STATUS.md`
4. `docs/ARCHITECTURE_DECISIONS.md`
5. `docs/ENGINEERING_WORKFLOW.md`
6. `docs/DATASET_NOTES.md`
7. `README.md`
8. `sohbet*.md` ve `tk3d_architecture_deep_dive.md` tarihsel belgeleri
