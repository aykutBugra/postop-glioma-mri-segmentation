# Ameliyat sonrası glioma MR segmentasyonu

TEKNOFEST 2026 kapsamında geliştirilen bu araştırma projesi, dört MR modalitesinden ameliyat sonrası glioma bileşenlerini ve rezeksiyon boşluğunu 3B olarak segmentler. Girdiler T1c, T1n, T2w ve T2-FLAIR. Çıkışlar non-enhancing tümör çekirdeği (NETC), çevresel FLAIR hiperintensitesi (SNFH), kontrastlanan doku (ET) ve rezeksiyon boşluğu (RC) etiketleridir.

## Doğrulanan sonuçlar

**Phase 1F fold-0 validasyonunda 31 hastadan 96 zaman noktası için ortalama foreground Dice 0,9153.** Foreground dört etiketin birleşimi. Sınıf Dice’ında referans veya tahminde etiketin bulunduğu vakalar hesaba katıldı. Bağımsız test ve klinik performans bu değerlendirme kapsamında ölçülmedi.

| Ölçüm | Phase 1F | Experiment A |
| --- | ---: | ---: |
| Foreground Dice | 0,9153 | 0,9172 |
| NETC aktif vaka Dice | 0,6102 | 0,5962 |
| SNFH aktif vaka Dice | 0,9108 | 0,9119 |
| ET aktif vaka Dice | 0,7935 | 0,7884 |
| RC aktif vaka Dice | 0,6494 | 0,6265 |
| Foreground hacim MAE (mL) | 2,1400 | 2,3522 |

![Phase 1F validasyonunda sınıf bazlı Dice dağılımı](../results/figures/class-dice.png)

Experiment A sınıf odaklı patch örnekleme kullandı. Foreground Dice hafif yükselirken NETC ve RC Dice düştü. Hacim hatası karar eşiğini aştığı için Phase 1F referans olarak korundu. Experiment B ağırlıklı çapraz entropi ekledi ve erken durduruldu. [Deney notları](experiments.md)

## Yöntem ve sınırlar

Veri hazırlığı aynı hastaya ait zaman noktalarını birlikte böler. Yerel 596 zaman noktasının 594’ü gerekli görüntü ve maskeleri içeriyordu. nnU-Net v2 `3d_fullres` eğitimi 128³ patch, batch 2 ve 1.000 epoch kullandı. Modelin tam hacim validasyonu, eğitim sırasında izlenen patch tabanlı pseudo Dice’tan ayrı bir geçişte yapıldı.

Tarihsel planlayıcı, ayrılmış 84 vaka dahil 594 vakanın tamamını gördü. Bu 84 vaka kayıtlı optimizasyon splitinde yer almıyor. Ayrılmış 30 hasta daha önceki custom-model test bölümünde yer almıştı. Sonuçları tek fold validasyonu olarak okuyun. Küçük NETC ve RC odaklarındaki hatalar sürüyor.

## Yerel kullanım

Python 3.12 ile `requirements.txt` dosyasındaki paketleri kurun. `python scripts/plot_results.py` mevcut toplu sonuçlardan grafikleri oluşturur. MR işlemek için TCIA’dan verinin yerel olarak edinilmesi gerekir. GPU eğitimi ve ağırlıkla inference için ayrıca uyumlu PyTorch/nnU-Net ortamı gerekir. Bu pakette görüntü, vaka düzeyi dosya ve checkpoint bulunmaz. [Kurulum ve deneyi yeniden üretme kılavuzu](reproducibility.md) · [İngilizce README ve atıflar](../README.md)
