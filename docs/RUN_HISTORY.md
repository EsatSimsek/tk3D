# Koşu Geçmişi ve Regresyon Teşhisi

Son güncelleme: **23 Ağustos 2026**

`src/poomsae_scoring/run_history.py`, her tek-komut koşusunu aynı session
altındaki önceki puanlama özetleriyle karşılaştırır. Çıktılar:

- `json/run_history_report.json`;
- `review/run_history.html`.

## Karşılaştırılabilirlik

Bir önceki koşunun tabloya alınması için aşağıdakilerin aynı olması gerekir:

- workflow profili;
- seçili kayıt kapsamı (`M01-M06` gibi);
- çalışma modu (`score_verified_pose` veya `process_video_then_score`).

Bu koşullar sağlansa bile otomatik regresyon uyarısı yalnız pose SHA-256 da
aynıysa üretilebilir. Farklı poz/video koşuları yalnız bağlam amaçlı yan yana
gösterilir; “daha iyi” veya “daha kötü” diye sınıflandırılmaz.

## İzlenen alanlar

- WholeBody ölçüm kapsamı;
- teknik kriter ölçüm kapsamı;
- ölçülemeyen ve sınır-belirsiz Accuracy kararları;
- WholeBody inceleme adayları;
- yanlış hareket/duruş adayları;
- teknik inceleme gereken hareket sayısı;
- gözlenen kapsam provisional kesintisi;
- Rule scoring readiness kapısı.

## Regresyon uyarıları

Aynı pose için aşağıdaki durumlar uyarı üretir:

- WholeBody ölçüm kapsamının azalması;
- teknik kriter kapsamının azalması;
- ölçülemeyen Accuracy kararlarının artması;
- Rule readiness değerinin `true` iken `false` olması.

İnceleme adayı veya provisional kesinti sayısının artması/azalması tek başına
kalite regresyonu sayılmaz. Daha az aday, hataların kaçırılması anlamına da
gelebilir; ground-truth olmadan bu yönde otomatik hüküm verilmez.

## Güvenlik sözleşmesi

- resmî doğruluk sıralaması yoktur;
- birleşik bir kalite puanı üretilmez;
- farklı kayıt kapsamları karşılaştırılmaz;
- farklı pose girdilerinde regresyon alarmı verilmez;
- bozuk veya başka workflow'a ait özetler fail-closed biçimde atlanır.
