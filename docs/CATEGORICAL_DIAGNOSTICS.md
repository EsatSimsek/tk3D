# Kategorik Hareket ve Duruş Teşhisleri

Son güncelleme: **23 Ağustos 2026**

`src/poomsae_scoring/categorical_diagnostics.py`, mevcut WholeBody-133
ölçümlerinden üç kategorik teşhis türü üretir:

- iki doğrulanmış segment arasındaki en az 3 saniyelik duraklama;
- beklenen duruş yerine alternatif duruş profiline uyum;
- beklenen teknik yerine alternatif teknik profiline uyum.

## Güvenlik sözleşmesi

Yanlış hareket ve yanlış duruş sonuçları `evidence_status=inferred` ve
`confirmation_method=kinematic_screening` taşır. Bunlar görüntüden türetilen
inceleme adaylarıdır fakat doğrudan gözlem değildir. Kaynak-bağlı Accuracy
motoru bu nedenle `reason=not_directly_observed` yazar ve `deduction_points`
alanını `null` bırakır.

Bir adayın oluşması için gerekli bütün metriklerin ölçülebilir olması,
alternatif profilin bütün aralıklarına uyulması ve en az bir metrik değerinin
beklenen profili tamamen dışlaması gerekir. Metrik `%95` belirsizliği taşıyorsa
karşılaştırma tek değer yerine bu aralığın tamamıyla yapılır; belirsizlik yoksa
bu durum raporda açıkça işaretlenir ve aday güveni daha düşük tutulur. Diğer
sonuçlar `consistent`, `ambiguous`, `not_measurable` veya `unsupported` olarak açıkça
raporlanır. `consistent`, hareketin kesin doğru olduğu anlamına gelmez; yalnız
ölçülen kinematiklerin beklenen mühendislik profiliyle çelişmediğini belirtir.

## Şu an desteklenen ayrımlar

- duruş: `ap_seogi` ↔ `ap_gubi`, `stance_span_ratio` ve `front_knee_deg`;
- teknik: `arae_makki` ↔ `momtong_jireugi`, uygulayan bilek yüksekliği ve
  dirsek açısı.

Eolgul-makki, momtong-an-makki, ap-chagi ve bileşik hareketlerde yeterli
alternatif profil yoksa sistem tahmin uydurmaz ve `unsupported` döndürür.

## Tek-komut çıktısı

```powershell
.\.venv312\Scripts\python.exe scripts\run_poomsae_scoring.py --profile poomsae1_trimmed
```

Rapor benzersiz run altında
`json/categorical_diagnostics_report.json` olarak oluşur. Aynı kontroller
beklenen ve alternatif ölçüm aralıklarıyla HTML inceleme ekranında gösterilir.
