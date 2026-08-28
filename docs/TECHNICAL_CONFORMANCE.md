# M01–M06 Teknik Uygunluk Motoru

Son güncelleme: **23 Ağustos 2026**

`src/poomsae_scoring/technical_conformance.py`, doğrulanmış hareket zaman
çizelgesini, WholeBody-133 ölçümlerini ve kategorik yanlış hareket/duruş
teşhislerini tek bir hareket bazlı raporda birleştirir. Bu rapor hakem puanı,
resmî uygunluk kararı veya otomatik kesinti değildir.

## Hareket sonucu

Her gözlenen hareket şu sonuçlardan birini alır:

- `mismatch_candidate`: kinematik tarama alternatif hareket veya duruş
  profiline daha açık uyar;
- `review_candidate`: en az bir ölçülebilir teknik kriter mühendislik tarama
  aralığının dışındadır;
- `ambiguous`: `%95` ölçüm aralığı karar sınırına biner veya kimlik kanıtları
  çelişir;
- `consistent_within_measured_scope`: ölçülebilen ve eşikle
  değerlendirilebilen kriterlerde çelişki bulunmamıştır;
- `not_measurable`: eşikle değerlendirilebilen kriter yoktur.

`consistent_within_measured_scope`, eksik kriterlerin doğru olduğu anlamına
gelmez. Bu nedenle her harekette beklenen, ölçülebilir, eşikle
değerlendirilebilir, yalnız tanısal ve ölçülemeyen kriter sayıları ayrı tutulur.

## Kanıt güveni

Motor yeni bir model güveni uydurmaz. `fused_evidence_confidence` aşağıdaki
mevcut kanıtların en düşük değeridir:

- timeline hareket etiketi güveni;
- gerekli WholeBody grubunun geçerli örnek kapsamı;
- ilgili pencere içindeki gerekli eklem örnek oranı;
- kategorik kontrolün kendi güveni.

Minimum kullanımı kasıtlıdır: tek bir zayıf kanıt, diğer yüksek değerlerin
ortalaması içinde gizlenmez. Değer puan veya doğruluk olasılığı değildir;
inceleme önceliği için kanıt yeterliliği göstergesidir.

## Belirsizlik

Bir metrik `uncertainty_95` taşıyorsa tarama kuralı yalnız merkez değerle değil,
`value ± uncertainty_95` aralığıyla karşılaştırılır:

- aralığın tamamı kabul bölgesindeyse `within_screening_range`;
- tamamı dışarıdaysa `review_candidate`;
- sınırı kesiyorsa `boundary_uncertain`.

Belirsizlik bulunmayan metrikte nokta değer kullanılır ve bu durum JSON'da
açıkça görülebilir.

## Teknik boyutlar

Beklenen PoomsaeSpec kriterleri üç görünümde gruplanır:

- duruş ve postür;
- teknik uygulama;
- zamanlama ve kontrol.

Ham ölçümler silinmez. Her kriter satırı kaynak metrik değerini, birimini,
eşik kuralını, karşılaştırma aralığını, kanıt penceresini ve birleşik kanıt
güvenini taşır.

## Güvenlik sözleşmesi

Raporun değişmez sınırları:

- `official_conformance_claim_allowed=false`;
- `score_claim_allowed=false`;
- `automatic_deduction_allowed=false`;
- çıkarımsal yanlış hareket/duruş adayı bağımsız doğrulama gerektirir;
- kısmi kayıt yalnız gözlenen M01–M06 için raporlanır.

## Tek-komut çıktısı

```powershell
.\.venv312\Scripts\python.exe scripts\run_poomsae_scoring.py --profile poomsae1_trimmed
```

Rapor benzersiz run altında `json/technical_conformance_report.json` olarak
oluşur. Aynı hareket kartları, teknik boyutlar ve kanıta atlama düğmeleri HTML
inceleme ekranında gösterilir.
