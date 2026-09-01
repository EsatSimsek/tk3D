# Referans duruş şablonları

Bu dizin, otomatik hizalamanın "bu segment hangi hareket" sorusunu cevaplarken
kıyasladığı referans duruş kütüphanesini taşır.

Şablonlar `scripts/build_poomsae_reference_templates.py` ile **elle etiketlenmiş**
bir kayıttan üretilir: her hareketin fixation çapası etrafındaki pencerede eklem
başına ortalama alınır, az görülen eklem NaN kalır. Otomatik bir timeline'dan
üretilmesi script tarafından reddedilir — hizalama kendi ödevini kontrol edemez.

Kıyaslama `pose_distance` ile yapılır ve konumdan/ölçekten bağımsızdır: poz pelvis
ortasına taşınır ve ölçeğe bölünür. Bakış yönü **normalize edilmez**, çünkü formun
hareketlerini birbirinden ayıran şey odur.

Sınırlar:

- Tek sporcunun tek kaydı. Doğrulanmış bir referans standart değil, eşleştirme
  geometrisi.
- Kaydın içermediği hareketler `missing_movement_ids` altında listelenir ve
  hizalamada eşleşmemiş olarak raporlanır.
- Şablonlar tolerans taşımaz; tek başına hiçbir kesintiyi gerekçelendiremez.
