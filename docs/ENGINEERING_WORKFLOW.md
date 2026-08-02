# TK3D Mühendislik İş Akışı

Bu belge, hızlı “vibe coding” yapılırken doğruluk ve veri güvenliğini koruyan
uygulama sırasını tanımlar.

## 1. Görevi sınıflandır

Değişiklikten önce görevi birincil alana ayır:

- görüntü işleme / 2B pose
- senkronizasyon / kalibrasyon
- triangulation / global 3B optimizasyon
- export / görselleştirme
- ground-truth doğrulama
- puanlama altyapısı
- dokümantasyon / bakım

Bir alandaki görev, diğer alanlarda geniş yeniden tasarım izni değildir.

## 2. Başlangıç kontrolü

```powershell
git status --short
git pull --ff-only
git log -5 --oneline
```

Ardından `AGENTS.md`, `PROJECT_STATUS.md`, ilgili config ve testleri oku.
Çalışma ağacı kirliyse kullanıcı değişikliğini geri alma veya stash etme.

## 3. Kök neden ve ölçüm

Bir kalite problemi için kod yazmadan önce:

1. Aynı kare aralığını ve kamera setini sabitle.
2. Ham 2B, stabilize 2B, geometri girdisi ve nihai 3B çıktıyı ayır.
3. Reprojection, geçerli oran, kullanılan kamera, temporal jitter ve açı
   jitter metriklerini çıkar.
4. Senkron, kalibrasyon, kişi crop'u, sol/sağ kimliği, occlusion ve heatmap
   lokalizasyon hipotezlerini ayrı değerlendir.
5. Düzeltmenin ölçülebilir kabul kriterini yaz.

Video daha sakin göründü diye algoritma başarılı sayılmaz.

## 4. Uygulama kuralları

- Küçük ve geri alınabilir değişiklik yap.
- Eşik gevşetmek yerine hatalı kanıtın kaynağını düzelt.
- Yeni config alanını `src/config_validation.py` ile doğrula.
- Yeni çıktı kararına provenance ekle.
- Eski/ham sonucu koru ve yeni sonucu ayrı adla export et.
- Kalite kapısı ve rollback olmadan global düzeltme ekleme.
- Dataset ofsetlerini ilgili session/manifest içinde tut.
- 133 noktalı sözleşmeyi BODY-17 optimizasyonuna indirgeme.

## 5. Test matrisi

### Doküman veya yorum

- bağlantı ve komut yollarını kontrol et
- `git diff --check`

### Saf Python mantığı

```powershell
.\.venv312\Scripts\python.exe -m pytest -q tests\<ilgili_test>.py `
  -p no:cacheprovider --basetemp outputs\pytest-target-<benzersiz-id>
```

### Repository genelini etkileyen Python değişikliği

```powershell
.\.venv312\Scripts\python.exe -m ruff check src scripts tests
.\.venv312\Scripts\python.exe -m pytest -q `
  -p no:cacheprovider `
  --basetemp outputs\pytest-full-<benzersiz-id>
git diff --check
```

### Model veya gerçek pipeline değişikliği

Yukarıdakilere ek olarak:

```powershell
.\.venv312\Scripts\python.exe scripts\check_models.py `
  --session data\aist_test\session.yaml
```

Önce kısa benzersiz smoke run, risk yüksekse tam `stride 1` run çalıştır.
Rapor JSON'larını programatik ve videoları görsel olarak incele.

### Ground-truth doğruluk iddiası

Değerlendirilen sensör/mimari profiliyle uyumlu held-out protokolünü çalıştır.
MADS yalnız bağlı RGB-only profil için kullanılabilir; ZED RGBD koşusuna tarihsel
MADS metriği devredilmez. İç geometri ve ZED RGB-vs-depth tutarlılığı bağımsız
ground-truth yerine geçmez. Global MPJPE ana karar metriğidir; PA-MPJPE tek
başına model onayı için kullanılmaz.

## 6. Adil önce/sonra karşılaştırması

Şunların tümü aynı olmalıdır:

- session ve fiziksel zaman aralığı
- kamera seti ve sync offsetleri
- model/checkpoint ve kişi dedektörü
- stride ve max-frame davranışı
- kalibrasyon
- kalite eşikleri

En az şu tablo raporlanmalıdır:

- BODY-17 valid ratio
- reprojection median / P95 / mean
- temporal joint ve angle jitter
- kamera başına target ve hata dağılımı
- kabul edilen, reddedilen ve visualization-only düzeltme sayıları
- global optimization applied/fallback durumu
- varsa MPJPE/PCK/açı ground-truth sonuçları

## 7. Çıktı yaşam döngüsü

- Run ID benzersiz olmalıdır.
- Eski run'ın üzerine yazılmaz.
- `latest_run.json` yalnız tamamlanmış başarılı run'a güncellenir.
- Test çıktıları ve cache Git'e eklenmez.
- Eski çıktıları temizlemek için önce korunacak run kullanıcıyla
  kesinleştirilir.
- Silme işlemi sonrası korunan videolar ve Git durumu tekrar doğrulanır.

## 8. Commit ve teslim

Kullanıcı commit/push istediğinde:

```powershell
git status --short
git diff --check
git diff --stat
git diff
```

Yalnız görev dosyalarını stage et. Sohbet özeti, model, video, output veya
başka kullanıcı dosyasını yanlışlıkla stage etme. Commit sonrası test sonucu,
commit hash'i ve pushlanan dalı bildir.

## Definition of Done

Bir görev ancak aşağıdakiler sağlanınca tamamdır:

- kök neden açıklanmış veya istenen özellik uygulanmış
- veri sözleşmesi ve provenance korunmuş
- ilgili testler geçmiş
- gerçek runtime değiştiyse gerçek smoke sonucu incelenmiş
- önce/sonra iddiası adil metriklerle kanıtlanmış
- kullanıcı komutu ve çıktı yolu net
- gerekli README/status/decision belgeleri güncel
- kullanıcı istemediyse commit, push veya silme yapılmamış
