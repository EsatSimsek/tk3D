# Teknik Doğruluk Kural Doğrulama Düzeneği

Son güncelleme: **31 Ağustos 2026**

Bu düzenek, Taegeuk 1 v3 teknik-doğruluk sisteminin yazılım davranışını
tekrarlanabilir sentetik vakalarla doğrular. Kuralın biomekanik olarak doğru
olduğunu, hakemle uyuştuğunu veya eşiklerin gerçek sporcularda geçerli olduğunu
kanıtlamaz. Çıktının durumu bu nedenle
`synthetic_engineering_validation_only`; skor ve kesinti etkisi her zaman
`null` kalır.

## Doğrulanan katmanlar

1. **174 kural envanteri:** Her kuralın durumu, evaluator sözleşmesi,
   doğrulama derinliği ve puansızlığı ayrı satırdır.
2. **133 landmark envanteri:** 0–132 arasındaki her noktanın en az bir kural
   sözleşmesine bağlı olduğu ayrıca doğrulanır. Bu kontrol landmark başına
   bağımsız aktif hata kuralı bulunduğu anlamına gelmez.
3. **33 aktif kuralın eşik sınıflandırması:** Her aktif kural için `pass`,
   üst/ana `boundary`, karşı taraf `opposite_boundary`, üst/ana `fail`, karşı
   taraf `opposite_fail`, `missing`, `nan`, `positive_infinity`,
   `negative_infinity` ve `wrong_type` çalıştırılır. Böylece toplam 330
   sınıflandırma vakası oluşur. Range operatörünün iki sınırı ve `abs_max`
   operatörünün pozitif/negatif yönü ayrı sınanır. Tek taraflı veya boolean
   kontratta uygulanamayan karşı sınır açıkça `not_applicable` kaydedilir.
4. **WholeBody-133 geometri senaryoları:** Tam fixture; yüz/el/ayak kanıtının
   kaybı; kamera ve reprojection kapıları; dejenere yüz; fixation sonrası
   kontrollü drift; sağ-sol ayna; BODY-17 reddi ve yön bağı davranışı sınanır.
5. **Artifact güvenliği:** JSON yazımı `allow_nan=false` ile yapılır. NaN ve
   sonsuzluk test girdileri ham sayı olarak artifact'e yazılmaz; tür etiketi ve
   `null` değerle temsil edilir. Dört config girdisinin mutlak yolu ve SHA-256
   özeti rapora bağlanır. Validator/evaluator/metric/CLI kaynak dosyalarının
   SHA-256 özetleri ve runtime sürümleri de rapora girer. Beş ana artifact'in
   hash ve boyutu `validation_manifest.json` içinde tutulur. Çıktılar önce
   geçici staging dizinine yazılıp tek adımda benzersiz run dizinine taşınır;
   var olan dizinin üzerine yazılmaz.

## Çalıştırma

Aşağıdaki komutta run kimliğini her çalıştırmada benzersiz seçin:

```powershell
$runId = "taegeuk1-rule-validation-20260831-r1"
$out = "outputs\validation\runs\$runId"
.\.venv312\Scripts\python.exe scripts\run_technical_accuracy_rule_validation.py `
  --validation-profile config\scoring\validation\taegeuk_1_rule_accuracy_validation_v1.yaml `
  --technical-profile config\scoring\engineering\taegeuk_1_wholebody_diagnostics_v3.yaml `
  --poomsae-spec config\scoring\poomsae\taegeuk_1_jang_v0_draft.yaml `
  --timeline config\scoring\timelines\poomsae1_zed2i_rgbd_rerun_20260802_draft.yaml `
  --output-json "$out\rule_accuracy_validation_report.json" `
  --rule-inventory-csv "$out\rule_validation_inventory.csv" `
  --landmark-inventory-csv "$out\landmark_validation_inventory.csv" `
  --classification-csv "$out\active_rule_classification_cases.csv" `
  --scenario-csv "$out\geometry_scenarios.csv"
```

Komut, bütün kontroller geçerse `0`, herhangi biri başarısızsa `1` döndürür.
Mevcut çıktı yollarından biri varsa üzerine yazmayı reddeder.

## Başarı ölçütü ve yorum

Rapor ancak aşağıdakilerin hepsi doğruysa `status=passed` olur:

- 174 envanter satırının tamamı kendi durum sözleşmesini geçer;
- 133 landmark satırının tamamı en az bir açık kural bağı taşır;
- 330 aktif-kural vakasının tamamı beklenen sınıfa düşer;
- yapılandırılmış 12 geometri senaryosunun tamamı geçer;
- BODY-17 girdisi kabul edilmez, eksik kanıt aday üretmez ve yön kuralları
  session-bound referans olmadan açılmaz;
- kontrollü drift ilgili metrikleri baz çizginin üstüne taşır ve ayna işlemi
  sıfır olmayan baz metriklerde değişmez kalır; ayna işlemi iki kez
  uygulandığında koordinat, geçerlilik, kamera ve reprojection dizileri eksiksiz
  geri döner;
- eksik kanıt senaryosundaki her hedef, bozulmadan önce baz çizgide ölçülebilir
  olmalıdır; zaten ölçülemeyen bir metrik yanlış başarı üretemez;
- geçerli session-bound bağ verildiğinde 17 yön kuralının tamamı ölçülür ve
  değerlendirilir.

`passed`, “yazılım bu tanımlı kontratlarda beklenen biçimde davrandı” demektir.
Rapor bunu ayrı `readiness` nesnesiyle korur:
`synthetic_contract_validation_ready=true` iken
`external_rule_accuracy_ready`, `judge_calibrated_ready`,
`production_threshold_ready` ve `official_scoring_ready=false` kalır.
Şunlar için ayrıca dış veri gerekir:

- uzman etiketli gerçek teknik hata örnekleri;
- kural başına pozitif/negatif vaka ve kör hakem karşılaştırması;
- precision, recall, false-positive/false-negative ve belirsizlik analizi;
- farklı sporcu, beden, dobok, kamera yerleşimi ve hareket hızında genelleme;
- geçici eşiklerin sürüm kontrollü kalibrasyonu.

Bu dış doğrulamalar yapılana kadar hiçbir harness sonucu resmî Accuracy puanı,
WT/Kukkiwon uyumu, üretim kalibrasyonu veya bilimsel ground-truth doğruluğu
olarak sunulamaz.
