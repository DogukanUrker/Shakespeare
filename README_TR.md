# 🚀 Shakespeare: PyTorch Kullanarak Nesne Tanımlama Gücünü Açığa Çıkarın

[🇬🇧 English](README.md) | **🇹🇷 Türkçe**

Shakespeare, popüler mimariler kullanarak (ResNet, EfficientNet, VGG, DenseNet veya MobileNet gibi) görsel olarak nesneleri tanımlayacak bir model eğitmenize yardımcı olmak için tasarlanmış hafif bir PyTorch projesidir.

<p align="center">
    <img src="https://github.com/DogukanUrker/Shakespeare/assets/62756402/76d2fe03-1c5e-474f-99ed-f8683ec66a97" alt="logo">
</p>

## 📹 Eğitim Videosu

Adım adım kılavuz için [Eğitim Videom](https://youtu.be/347KrcnjJQI)'a göz atın!

## 📂 Kurulum

Depoyu klonlayın:

```bash
git clone https://github.com/DogukanUrker/Shakespeare.git
cd Shakespeare
```

Gerekli klasörleri oluşturmak ve gerekli modülleri yüklemek için setup.py'ı çalıştırın:

```bash
python3 setup.py
```

## ⚡️ Başlarken

1. 💿 **Verilerinizi Hazırlayın**:

   - Görsellerinizi şu varsayılan dizinlere yerleştirin:
     - `data/object/`: Sınıflandırmak istediğiniz nesne görsellerini içerir.
     - `data/notObject/`: Nesne sınıfına ait olmayan görselleri içerir.
     - `data/test/`: Eğitilen modeli test etmek için ek görseller içerir.

2. ⚙️ **Modeli Yapılandırın**:

   - `defaults.py` dosyasını açın ve `MODEL_NAME` parametresini istediğiniz model mimarisiyle (`resnet`, `efficientnet`, `vgg`, `densenet`, `mobilenet`) ayarlayın.

3. 🏋️ **Modeli Eğitin**:

   - Eğitime başlayın ve bir `.pkl` dosyası oluşturmak için şunu çalıştırın:
     ```bash
     python3 main.py
     ```

4. 📝 **Test Etme**:
   - Eğitimin ardından, modelinizin performansını değerlendirmek için şunu çalıştırın:
     ```bash
     python3 test.py
     ```

## 🎨 Özelleştirme

- Hiperparametreleri ayarlamak veya farklı model mimarileriyle denemeler yapmak için `defaults.py` veya `train.py` dosyalarını değiştirin.
- `train.py` dosyasına ön işleme adımları veya veri artırma ekleyerek işlevselliği genişletin.

## 💞 Katkıda Bulunun

Katkılar memnuniyetle karşılanır! Önerileriniz, hata raporlarınız veya özellik eklemek isterseniz, lütfen bir pull request gönderin.

## ⚖️ Lisans

Bu proje MIT Lisansı altında lisanslanmıştır - detaylar için [LICENSE](./LICENSE) dosyasına bakın.

## 🌟 Projemizi Destekleyin!

Shakespeare'i projelerinizde yararlı bulursanız ve geliştirme ve bakımını desteklemek isterseniz, projemizin sürdürülebilirliğine katkıda bulunabilirsiniz.

### Nasıl Yardımcı Olabilirsiniz:

- **GitHub'da Bize Bir Yıldız Verin**: Takdirinizi [GitHub depomuzu](https://github.com/DogukanUrker/Shakespeare) yıldızlayarak gösterin. Bu, daha fazla geliştiriciye ulaşmamıza yardımcı olur!

- **Bağış sayfamı ziyaret ederek** doğrudan çalışmalarıma destek olabilirsiniz: [bağış sayfam](https://dogukanurker.com/donate).

Desteğiniz çok önemlidir ve Shakespeare'in topluluk için geliştirilmesine devam etmemize yardımcı olur. Düşündüğünüz için teşekkürler!

Oluşturan: [Doğukan Ürker](https://dogukanurker.com)
