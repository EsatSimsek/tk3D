# Puanlama Kaynak Sicili

Son doğrulama: **3 Ağustos 2026**

Bu sicil, kural motoruna giren bilginin nereden geldiğini ve hangi iddia için
kullanılabileceğini kaydeder. Bir kaynağın burada bulunması, içeriğinin otomatik
olarak aktif puan kuralı olduğu anlamına gelmez.

## 1. Aktif yarışma kuralı

| Alan | Değer |
|---|---|
| Kurum | World Taekwondo |
| Belge | Poomsae Competition Rules and Interpretation |
| Yürürlük | 30 Eylül 2024 |
| Erişim | Ücretsiz, resmî PDF |
| WT canlı liste durumu | `CURRENT` - 2 Ağustos 2026 tarihinde doğrulandı |
| Yerel doğrulanan boyut | 2.960.548 bayt |
| SHA-256 | `3fc994363544ab2a1717d9d4d805b368e283a122d3f3c52fc444c70b8c206e24` |
| İlgili sayfalar | Article 15, belge sayfası 23; Article 16, belge sayfası 25 |

Resmî bağlantılar:

- [WT Rules and Regulations](https://www.worldtaekwondo.org/inside/documents/rar/list)
- [WT Poomsae Competition Rules and Interpretation - 30 Eylül 2024](https://www.worldtaekwondo.org/att_file/documents/Poomsae_Competition_Rules_and_Interpretation_%28In_force_as_of_September_30_2024%29.pdf)
- [Doğrulanmış yerel PDF kopyası](../output/pdf/scoring_sources/WT_Poomsae_Competition_Rules_2024-09-30.pdf)
- [Kullanılan bütün kaynakların yerel indeksi](scoring_sources/README.md)

Doğrulanan puan yapısı:

- Recognized Poomsae toplamı `10,0`;
- Accuracy başlangıcı `4,0`;
- Presentation toplamı `6,0`;
- küçük hareket hatası `-0,1`;
- büyük hareket hatası `-0,3`;
- poomsae'yi yeniden başlatma `-0,6`.

Article 16 küçük ve büyük hatayı ekli **Poomsae Competition Scoring
Guidelines** belgesine bağlar. 2 Ağustos 2026 tarihinde WT'nin canlı
`POOMSAE / CURRENT` listesinde ayrı bir güncel scoring guideline dosyası
bulunmamıştır. 3 Ağustos 2026 yeniden kontrolünde de ayrı güncel ek bulunmadı.
Bu nedenle kesinti miktarları aktiftir; ancak her teknik için
hangi sayısal sapmanın `0,1` veya `0,3` sayılacağı resmî olarak
aktifleştirilemez. M01-M06 için eklenen açık mühendislik profili yalnız ayrı
geliştirme denemesinde kullanılır; RulePack'in resmî eşiği sayılmaz.

## 2. Taegeuk 1 resmî ücretsiz kaynakları

### 2.0 WT 2014 tam teknik scoring guideline — tarihsel resmî kaynak

| Alan | Değer |
|---|---|
| Belge | Poomsae Scoring Guidelines for International Referees |
| Kurum | World Taekwondo Federation |
| Statü | Resmî fakat tarihsel; 2014 yarışma kurallarına ek |
| Sayfa | 43 |
| Yerel dosya | `output/pdf/scoring_sources/WT_Poomsae_Scoring_Guidelines_2014_43p.pdf` |
| Boyut | 1.474.523 bayt |
| SHA-256 | `46686f9a58c7d3dbc50bd9a1d284d9f6d2a56dbab6f70c365c39f029f8e01c8f` |
| Resmî/tarihsel erişim | [WT 2014 Rules + ek](https://www.worldtaekwondo.org/wp-content/uploads/2016/10/WTF-Poomsae-Competition-Rules-Interpretation-March-19-2014.pdf) |
| Ayrı arşiv kopyası | [43 sayfalık PDF](https://d17nlwiklbtu7t.cloudfront.net/983/document/Poomsae_scoring_guidelines.pdf) |

Masaüstündeki `WTF---Poomsae-scoring-guidelines.pdf` `35` sayfa ve
`1.403.573` bayttır; SHA-256 özeti
`701543a7c509a38eb8f87dc174b2a832618b0c655fa6cdbf4727403044fbc3dd`
değeridir. Sayfa altlıkları belgenin aslında 43 sayfa olduğunu gösterir. Tam
kopyayla ilk 35 sayfanın çıkarılabilir metni eşleşmiştir. Eksik 36-43.
sayfalar terminoloji, basic movement listesi ve score sheet'lerdir.

Bu belge; `ap-seogi`, `ap-gubi`, `arae-makki`, `eolgul-makki`,
`momtong-makki`, `jireugi` ve `ap-chagi` teknik geometrisi için güçlü bir
tarihsel kaynaktır. Ancak güncel WT canlı listesinde ayrı current guideline
olarak yayımlanmadığı için sayısal ölçüleri güncel resmî tolerans veya doğrudan
tam skor eşiği yapılmaz. Açık geometriler yalnız kaynağı tarihsel olarak
işaretleyen ayrı source-bound profilde, `%95` belirsizlik kapısıyla provisional
`-0,1` gözlenen-kapsam kararı için kullanılabilir. WholeBody adaylarının
semantiği ve hata taksonomisi için de kullanılır.

### 2.1 Kukkiwon 2025 ayrıntılı eğitim videosu

| Alan | Değer |
|---|---|
| Başlık | `유급자 품새 교육 \| 태극 1장` |
| Yayıncı | KUKKIWON WORLD TAEKWONDO HEADQUARTERS - doğrulanmış kanal |
| Tarih | 16 Ocak 2025 |
| Süre | 21:12 |
| Erişim | Ücretsiz |
| Altyazı | Resmî İngilizce ve İspanyolca; Korece otomatik altyazı da mevcut |
| Bağlantı | [YouTube videosu](https://www.youtube.com/watch?v=__4pltpaPJo) |

Teknik tanım bakımından en güçlü ücretsiz kaynak budur. Kaynakta özellikle:

- başlangıç hazırlığı ve nefes,
- eksen etrafında 90 derece dönüş,
- `ap-seogi` ve `ap-gubi` merkez aktarımı,
- `arae-makki`, `momtong-an-makki`, `eolgul-makki` bitiş ilişkileri,
- `momtong-jireugi` sırasında gövde kullanımı,
- `ap-chagi` diz toplama ve yeniden toplama,
- son `kihap`ın hareketle eşzamanı ve bitiş tutumu

açıklanır. Altyazıda verilen sayısal veya geometrik ifadeler önce ölçülebilirlik
ve çeviri tutarlılığı denetiminden geçmeden doğrudan eşik yapılmayacaktır.

### 2.2 Kukkiwon kısa tam sıra gösterimi

| Alan | Değer |
|---|---|
| Başlık | `품새_태극1장` |
| Tarih | 6 Ocak 2022 |
| Süre | 00:41 |
| İlgili aralık | 00:12-00:36 |
| Erişim | Ücretsiz, liste dışı video |
| Bağlantı | [YouTube videosu](https://www.youtube.com/watch?v=FmlZHqV9Y-M) |

Video rastgele bir üçüncü taraf kaynağı değildir. Kukkiwon'un resmî Taegeuk 1
sayfasındaki QR kod bu videoya yönlendirir. Tam hareket sırasının görsel
transkripsiyonunda kullanılır; tek başına sayısal tolerans kaynağı değildir.

### 2.3 Kukkiwon Taegeuk 1 bilgi sayfası

[Kukkiwon Taegeuk 1 Jang](https://www.kukkiwon.or.kr/eng/board/read?boardManagementNo=55&boardNo=1347&menuLevel=3&menuNo=72&page=2&searchCategory=&searchType=&searchWord=)
sayfası formun anlamını ve temel hareketleri doğru öğrenme amacını açıklar.
Hareket tablosu veya yarışma kesinti eşiği içermez.

## 3. Ücretli veya erişimi sınırlı kaynaklar

### 3.1 Kukkiwon Taekwondo Textbook 3 - Poomsae

Bu kitap resmî ve güçlü bir ikinci çapraz doğrulama kaynağıdır. Kukkiwon'un
2022 [İngilizce edisyon duyurusuna](https://kukkiwon.or.kr/eng/board/read?boardManagementNo=49&boardNo=1701&menuLevel=2&menuNo=91&page=10&searchCategory=&searchType=&searchWord=)
göre İngilizce beş cilt set `120.000 KRW`, her e-kitap cildi
`16.800 KRW` olarak yayımlanmıştır. [YES24 Volume 3 kaydı](https://www.yes24.com/Product/Goods/110586882)
`16.800 KRW`, DRM'li PDF ve yaklaşık 290 sayfa bilgisi gösterir; aynı sayfa
ürünü şu anda `절판`/satış dışı olarak da işaretlemektedir. Dolayısıyla fiyat
bilgisi doğrulanmış olsa da güncel satın alınabilirlik kesin değildir.

Karar: **şimdilik satın alma gerekmiyor**. Ücretsiz 2025 Kukkiwon eğitimi ilk
transkripsiyon ve teknik adaylar için yeterlidir. Kitap; bileşik hareketlerin
resmî tablo sınırlarını, yön işaretlerini veya terminolojiyi ikinci resmî
kaynakla doğrulamak gerekirse kullanıcı onayıyla alınabilir.

### 3.2 Kukkiwon kısaltılmış ders kitabı

Kukkiwon'un 2022 duyurusunda tek cilt kısaltılmış sürüm `50.000 KRW` olarak
listelenmiştir. Yön renkleri, dönüş ekseni, ayrıntılı hareket açıklamaları ve
QR videoları içerdiği belirtilir. Volume 3'e kıyasla yalnız Taegeuk 1 kural
çıkarımı için daha pahalı ve daha az hedeflidir; ilk satın alma tercihi
değildir.

### 3.3 WT kurs materyalleri

WT International Poomsae Referee/Coach eğitimleri puanlama uygulamasını
içerebilir; ancak güncel ders materyali ve scoring attachment kamuya açık
belge listesinde bulunmamıştır. Kurs içeriği elde edilmeden sistem bu kaynağı
varmış gibi kabul etmeyecektir.

### 3.4 Aramada bulunan fakat güncel kural eki olmayan belge

WT'nin 2022 çevrim içi Poomsae Open Challenge sayfasında ayrı bir
[Poomsae Deduction](https://m.worldtaekwondo.org/competition/view.html?mcd=K1&nid=139335&page=26&sc=in)
iframe'ı bulunmuştur. Bu dosya belirli bir çevrim içi yarışmanın kayıt/yayın
kurallarıyla birlikte yayımlanmıştır; 30 Eylül 2024 kural kitabındaki güncel
`Poomsae Competition Scoring Guidelines` eki olduğu doğrulanamamıştır. Bu
nedenle tarihsel araştırma izi olarak kaydedilir, RulePack veya teknik eşik
kaynağı yapılmaz.

### 3.5 Masaüstünde incelenen ikincil/ulusal belgeler

| Dosya | Statü | SHA-256 | Karar |
|---|---|---|---|
| `C:\Users\WWWW\Desktop\tk\Poomsae-Judgement-Handbook.pdf` | Swiss Taekwondo logolu, January 2025 hakem eğitim el kitabı; WT resmî belgesi değil | `7baa05019a7f034896b9a99b24d399ca9f170d5d5c1e87f3f9efa85e0988df14` | Accuracy/presentation örnekleri araştırma çapraz kontrolü; aktif RulePack kaynağı değil |
| `C:\Users\WWWW\Desktop\tk\BT-Poomsae-Competition-Rules-June-2025-1.pdf` | British Taekwondo MNA kuralı | `7bd3b59239ff410d8aac241e06381ba53d8b5e6540ef6e4244016891273471f9` | Ulusal uygulama farklarını gösterir; genel WT kuralına taşınmaz |
| `C:\Users\WWWW\Desktop\tk\British-Taekwondo-Poomsae-Competition-Rules-2024-approved-27062024.pdf` | British Taekwondo önceki sürüm | `1219e6308291d93f4c9e6a7fe513169e4c7de5e2319dbd78b7d8b98d49e80002` | Tarihsel ulusal çapraz kontrol |
| `C:\Users\WWWW\Desktop\tk\Poomsae Competition Rules and Interpretation (In force as of June 14 2024).pdf` | WT resmî, taranmış/görüntü PDF | `b89f796d7d52ab4eb6018289c1702d9f9735508cc4177bda6e697823906802a4` | 30 Eylül sürümünün önceki resmî kuralı |

Swiss el kitabı `4,0 + 6,0`, küçük/büyük hata örnekleri ve Taegeuk 1 son
yumrukta tek kihap bilgisini açıklar. Ancak belgenin kapak iddiası ve güncel
tarihi onu WT/Kukkiwon otoritesi yapmaz. British 2025 metni genel büyük denge
kaybı/stumble örneğini ekler; güncel WT 2024 İngilizce metnindeki özgül denge
örneği ise Hakdari-seogi ayağının yere değmesidir. Bu fark nedeniyle genel
denge kaybı otomatik `-0,3` yapılmamıştır.

## 4. Kaynak kabul durumları

| Bilgi | Durum | Kullanım |
|---|---|---|
| WT `4,0 / 6,0` dağılımı | aktif | RulePack |
| WT `0,1 / 0,3 / 0,6` miktarları | aktif | Accuracy kesinti motoru |
| Taegeuk 1 sıra transkripsiyonu | taslak, kaynak bağlı | PoomsaeSpec ve video etiketi |
| Teknik bitiş biçimleri | aday, kaynak bağlı | metrik tasarımı |
| 2014 WT/WTF teknik geometrisi | tarihsel resmî | hash/sayfa bağlı provisional küçük-hata geometrisi; güncel WT eki değil |
| Güncel sayısal toleranslar | resmî kaynak eksik | güncel WT toleransı iddia edilemez |
| Tarihsel küçük hata eşlemesi | belirsizlik kapılı provisional | yalnız `%95` aralığı bütünüyle sınır dışındaysa `-0,1` |
| Büyük hata eşlemesi | eksik | otomatik major kapalı |
| Hakem eşdeğerliği | doğrulanmadı | iddia edilemez |

## 5. Fail-closed kuralı

Gerçek `accuracy_score` kesintisi ancak aşağıdakilerin tamamı varsa uygulanabilir:

1. yürürlükteki RulePack kesinti türünü ve miktarını tanımlar;
2. aktif PoomsaeSpec hareketi ve fazı tanımlar;
3. metrik resmî/aktif ölçüte bağlıdır;
4. gerekli kamera/sensör kanıtı gerçekten gözlenmiştir;
5. olay güven eşiğini geçer ve aynı hata daha önce kesilmemiştir.

Kaynak boşluğu, görünürlük yetersizliği veya faz belirsizliği halinde sonuç
`not_measurable`, `source_gap` ya da `candidate` olur; puan kesilmez.
BODY-17 engineering v1 sayısal denemesi geçersizleştirilmiştir ve artık skor
üretmez. Sürümlü WholeBody v2 mühendislik hipotezleri yalnız
`review_candidate_not_deduction` olayı üretebilir. Ayrı source-bound profil,
yalnız hash/sayfa bağlı tarihsel açık geometrilerde `%95` belirsizlik aralığının
tamamı sınır dışındaysa gözlenen kapsam için `-0,1` uygulayabilir; bu sonuç
güncel WT toleransı veya tam skor değildir. `accuracy_score`,
`partial_engineering_trial_score`, `applied_deductions` ve resmî hazır olma
kapılarını açamaz.

Tüm hata aileleri, sensör kanıtı ve kalan açıklar
[`TAEGEUK1_ERROR_TAXONOMY.md`](TAEGEUK1_ERROR_TAXONOMY.md) içinde tutulur.

## 6. Kullanıcının getireceği yeni kaynaklar

Yeni PDF doğrudan RulePack'e veya eşik profiline yazılmaz. Önce
[`scoring_sources/SOURCE_INTAKE_TEMPLATE.yaml`](scoring_sources/SOURCE_INTAKE_TEMPLATE.yaml)
kopyalanır; kurum, başlık, belge/yürürlük tarihi, dosya yolu, erişim biçimi,
otorite sınıfı ve amaçlanan iddialar doldurulur. Ardından
`scripts/validate_scoring_source_intake.py` ile dosya imzası ve SHA-256 bağı
çıkarılır.

Kabul sırası:

1. Dosya gerçekten PDF mi ve hash'i sabitlendi mi?
2. Yayınlayan kurum WT/Kukkiwon mu, ulusal federasyon mu, eğitim kaynağı mı?
3. Belge tarihi ile yürürlük tarihi ayrı mı?
4. İddia hangi sayfa/madde/görsel tarafından destekleniyor?
5. Bilgi hareket semantiği mi, ölçüm tanımı mı, güncel sayısal tolerans mı?
6. Mevcut resmî kaynakla çelişiyor mu?
7. Ölçülebilir ve test edilebilir mi?

Araç hiçbir durumda otomatik aktivasyon yapmaz. Tarihsel, ulusal, eğitim veya
araştırma kaynağı güncel sayısal tolerans talep ederse
`numeric_threshold_authority_insufficient` engeli üretir. Kaynak doğrulansa
bile sonraki adım sayfa/madde bazlı manuel iddia incelemesidir.
