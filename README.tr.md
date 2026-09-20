# APTOS 2019 — Diyabetik Retinopati Evre Tespiti

🔗 **Canlı Demo:** [diabetic-retinopathy-app.streamlit.app](https://diabetic-retinopathy-app.streamlit.app) 

Retina fundus (göz dibi) görüntülerinden diyabetik retinopati (DR) evresini (0–4 arası, 5 sınıflı) tahmin eden bir derin öğrenme projesi. EfficientNet-B0, ResNet50 ve DenseNet121 mimarileri transfer learning ile eğitilmiş, en iyi performansı veren DenseNet121 kademeli (discriminative) fine-tuning ile daha da geliştirilmiş; Grad-CAM ile model kararlarının klinik olarak anlamlı bölgelere dayandığı doğrulanmış ve sonuçlar bir Streamlit arayüzü üzerinden sunulmuştur.

## İçindekiler

- [Veri Seti](#veri-seti)
- [Ön İşleme](#ön-i̇şleme)
- [Modeller ve Sonuçlar](#modeller-ve-sonuçlar)
- [Eğitim Grafikleri](#eğitim-grafikleri)
- [Fine-Tuning](#fine-tuning)
- [Grad-CAM ile Açıklanabilirlik](#grad-cam-ile-açıklanabilirlik)
- [Sağlamlık Testi: Dairesel Maskeleme](#sağlamlık-testi-dairesel-maskeleme)
- [Proje Yapısı](#proje-yapısı)
- [Kurulum ve Çalıştırma](#kurulum-ve-çalıştırma)
- [Streamlit Arayüzü](#streamlit-arayüzü)

## Veri Seti

[APTOS 2019 Blindness Detection](https://www.kaggle.com/datasets/mariaherrerot/aptos2019/data) veri seti kullanılmıştır — retina fundus görüntüleri, 5 sınıflı DR evrelendirmesiyle etiketlenmiştir:

| Evre | Sınıf |
|---|---|
| 0 | No DR |
| 1 | Mild |
| 2 | Moderate |
| 3 | Severe |
| 4 | Proliferative DR |

| Bölüm | Görüntü Sayısı |
|---|---|
| Train | 2.930 |
| Validation | 366 |
| Test | 366 |

Sınıf dağılımı dengesiz (No DR sınıfı çoğunlukta, Severe en azınlıkta, ~9:1 oranında) — bu, eğitim sırasında **weighted Cross-Entropy Loss** ile ele alınmıştır.

<table>
<tr>
<td><img src="images/train_class_distribution.png" alt="Train Sınıf Dağılımı"></td>
<td><img src="images/validation_class_distribution.png" alt="Validation Sınıf Dağılımı"></td>
<td><img src="images/test_class_distribution.png" alt="Test Sınıf Dağılımı"></td>
</tr>
</table>

Ham görüntüler 17 farklı çözünürlükte, 474×358 ile 4288×2848 piksel arasında değişmektedir.

## Ön İşleme

Her görüntü, modele verilmeden önce şu adımlardan geçer:

1. **Auto-Crop** — görüntünün siyah kenarlıklarını otomatik tespit edip kırpar (parlaklık eşiğine dayalı bounding-box).
2. **CLAHE** (Contrast Limited Adaptive Histogram Equalization) — LAB renk uzayının L (parlaklık) kanalına uygulanarak kılcal damarlar ve lezyonlar netleştirilir.
3. **Albumentations** ile augmentasyon (train setinde) — yatay/dikey çevirme, 90° rotasyon, shift/scale/rotate; ardından 224×224'e yeniden boyutlandırma ve ImageNet istatistikleriyle normalize etme.

İşleme mantığı `dataset.py` içindeki `auto_crop()`, `apply_clahe()` ve `APTOSDataset` sınıfında toplanmıştır; macOS'taki multiprocessing (spawn) sorunlarını önlemek için bu fonksiyonlar ayrı bir modüle taşınmıştır.

Görüntüleri her epoch'ta tekrar tekrar işlememek için, isteğe bağlı bir **ön-işleme (pre-caching) betiği** ile crop+CLAHE uygulanmış görüntüler bir kere işlenip diske kaydedilebilir.

**Orijinal vs. işlenmiş (Auto-Crop + CLAHE) karşılaştırması:**

![Orijinal vs İşlenmiş Karşılaştırması](images/dataset_original_vs_processed.png)

![Orijinal vs İşlenmiş Karşılaştırması 2](images/comparison_original_vs_processed.png)

CLAHE sonrası kılcal damarların ve lezyon sınırlarının belirgin şekilde netleştiği, özellikle Moderate/Proliferative DR örneklerinde eksüda ve kanama bölgelerinin daha ayırt edici hale geldiği görülmektedir.

## Modeller ve Sonuçlar

Üç mimari, ImageNet ön-eğitimli ağırlıklarla başlatılıp son katmanı 5 sınıfa uyarlanarak eğitildi (weighted Cross-Entropy Loss, AdamW, ReduceLROnPlateau, early stopping).

### Validation Sonuçları

| Model | Accuracy | Macro F1 | Weighted F1 | QWK |
|---|---|---|---|---|
| EfficientNet-B0 | 80.33% | 0.684 | 0.808 | 0.888 |
| ResNet50 | 79.51% | 0.675 | 0.804 | 0.890 |
| **DenseNet121** | **81.69%** | **0.712** | **0.820** | **0.902** |

**QWK (Quadratic Weighted Kappa)**, DR evrelendirmesinin sıralı (ordinal) yapısını dikkate aldığı için bu projede en kritik metrik olarak kabul edilmiştir — bir evrelik hata, iki evrelik hatadan daha az cezalandırılır.

DenseNet121, üç mimari arasında hem Accuracy hem QWK hem Macro F1'de en iyi sonucu vermiş ve sonraki fine-tuning çalışmaları için temel model olarak seçilmiştir.

> **Not — Baseline CNN:** Sıfırdan eğitilen basit bir custom CNN (baseline) de denenmiş, ancak öğrenme eğrisinin düzleştiği (train/val loss ~1.61 civarında sabit kalmış, 5 sınıf için rastgele tahminin teorik loss'una — ln(5) ≈ 1.609 — yakın) ve modelin anlamlı bir şey öğrenemediği görülmüştür. Bu nedenle baseline, karşılaştırmalardan ve sonraki adımlardan çıkarılmış, transfer learning yaklaşımına odaklanılmıştır (bkz. [Eğitim Grafikleri](#eğitim-grafikleri)).

## Eğitim Grafikleri

**Baseline CNN — öğrenemedi, loss sabit kaldı (bu yüzden transfer learning'e geçildi):**

![Baseline CNN Loss](images/baseline_cnn_loss_15epochs.png)

**EfficientNet-B0 — 12 epoch, early stopping devreye girmeden hemen önce val loss'ta ani bir sıçrama görülmüştür; en iyi model (sıçramadan önceki epoch) ayrıca kaydedildiği için nihai değerlendirme bundan etkilenmemiştir:**

![EfficientNet-B0 Loss](images/efficientnet_b0_loss_12epochs.png)

**ResNet50 — 11 epoch; EfficientNet'e benzer şekilde son epoklarda bir kararsızlık (val loss sıçraması) görülmüştür, en iyi model checkpoint'i sıçramadan önceki epoch'a aittir:**

![ResNet50 Loss](images/resnet50_loss_epochs11.png)

**DenseNet121 — 8 epoch, en düşük val loss'a erken epoklarda ulaşmıştır:**

![DenseNet121 Loss](images/densenet121_loss_8epochs.png)

> EfficientNet-B0 ve ResNet50'de görülen bu geç-epoch kararsızlığı, muhtemelen `ReduceLROnPlateau` scheduler'ının learning rate'i yeterince düşürmeden önce modelin sınıf ağırlıklı loss yüzeyinde dar bir minimumdan sıçramasından kaynaklanmaktadır; DenseNet121'in daha kısa sürede ve daha kararlı yakınsaması, bu mimarinin bu veri seti için seçilme gerekçelerinden biri olmuştur.

## Fine-Tuning

DenseNet121 üzerinde, **QWK-odaklı, kademeli (discriminative) fine-tuning** stratejisi uygulanmıştır:

- **WeightedRandomSampler** — sınıf ağırlıklandırmasına ek olarak örnekleme seviyesinde de denge sağlanır.
- **2 aşamalı kademeli katman çözme** — önce sadece son yoğun blok (`denseblock4`) + sınıflandırma katmanı, sonra `denseblock3` da dahil edilerek daha derin bir fine-tuning yapılır.
- **Discriminative learning rate** — sınıflandırma katmanına yüksek, gövdeye düşük learning rate uygulanarak ImageNet'ten gelen genel özellikler korunur.
- **Cosine Annealing + Warmup** scheduler.
- **Mixed precision (AMP)** ile hızlandırılmış eğitim.
- En iyi model seçimi `val_loss` yerine **QWK'ya göre** yapılır.
- **Test-Time Augmentation (TTA)** — nihai değerlendirmede orijinal, yatay/dikey flip ve 180° rotasyon tahminlerinin ortalaması alınır.

### Test Seti Sonucu (Fine-tuned DenseNet121, TTA ile)

| Metrik | Değer |
|---|---|
| Accuracy | **83.61%** |
| QWK | **0.894** |
| Macro F1 | 0.666 |
| Weighted F1 | 0.831 |

Fine-tuning, base modele göre test setinde ~2 puanlık bir accuracy artışı sağlamıştır.

**DenseNet121 — Aşama 1 (denseblock4 + classifier) — Loss ve Val QWK:**

![Fine-Tuning Aşama 1](images/densenet121_stage1_loss_qwk.png)

**DenseNet121 — Aşama 2 (denseblock3 + denseblock4 + classifier) — Loss ve Val QWK:**

![Fine-Tuning Aşama 2](images/densenet121_stage2_loss_qwk.png)

QWK, iki aşama boyunca dalgalı ama yükselen bir eğilim izleyerek 0.882'den 0.916'ya (en iyi epoch) çıkmıştır.

Aynı kademeli fine-tuning stratejisi karşılaştırma amacıyla **ResNet50** üzerinde de denenmiştir (`layer4` → `layer3+layer4`):

**ResNet50 — Aşama 1 (layer4 + fc) — Loss ve Val QWK:**

![ResNet50 Fine-Tuning Aşama 1](images/resnet50_stage1_loss_epochs10.png)

**ResNet50 — Aşama 2 (layer3 + layer4 + fc) — Loss ve Val QWK:**

![ResNet50 Fine-Tuning Aşama 2](images/resnet50_stage2_loss_epochs5.png)

ResNet50'nin fine-tuning'i de QWK'yı 0.87 civarından ~0.91'e yükseltmiş, ancak nihai model seçiminde DenseNet121 (daha yüksek ve daha kararlı QWK) tercih edilmiştir.

## Grad-CAM ile Açıklanabilirlik

Modelin karar mekanizmasını doğrulamak için `pytorch-grad-cam` kullanılarak DenseNet121'in son evrişim bloğu üzerinden ısı haritaları (heatmap) üretilmiştir:

- Her DR evresinden örnek görüntülerle model tahminlerinin görsel olarak doğru bölgelere (eksüda, kanama, optik disk çevresi) odaklandığı gözlemlenmiştir.
- No DR görüntülerinde ısı haritasının dağınık/odaksız kalması beklenen ve sağlıklı bir davranış olarak yorumlanmıştır (görüntüde belirgin bir lezyon yoktur).
- `target_class` parametresi elle 0-4 arası değiştirilerek, aynı görüntü için farklı sınıflara ait ısı haritaları karşılaştırılmış; gerçek sınıfa karşılık gelen haritanın en anlamlı/odaklı olduğu görülmüştür.

**Tek bir örnek üzerinde Orijinal → Auto-Crop+CLAHE → Grad-CAM akışı** (Moderate, doğru tahmin edilmiş):

![Orijinal-CLAHE-GradCAM](images/original_clahe_gradcam_visualization.png)

**Her DR evresinden örnek görüntüler ve karşılık gelen Grad-CAM ısı haritaları:**

![Grad-CAM Sağlamlık Kontrolü](images/gradcam_sanity_check.png)

Isı haritalarının genel olarak eksüda kümeleri, kanama bölgeleri ve optik disk çevresi gibi klinik açıdan anlamlı bölgelere odaklandığı görülmektedir.

**`target_class` parametresi 0–4 arası elle sabitlenerek, aynı (Moderate) görüntü için her sınıfa ait ısı haritası üretilmiştir:**

![Target Class Manuel Değişim](images/target_class_manual_override.png)

Gerçek sınıfa (target_class=2, Moderate) karşılık gelen ısı haritasının, görünür lezyon bölgesiyle en tutarlı ve en geniş odaklanmayı gösterdiği; diğer sınıflar için haritanın farklı (ve genelde daha az anlamlı) bölgelere kaydığı görülmektedir.

## Sağlamlık Testi: Dairesel Maskeleme

Modelin gerçekten klinik bulgulara mı odaklandığını, yoksa görüntü köşelerindeki artefaktlara mı ("shortcut learning") güvendiğini test etmek amacıyla, 366 validation görüntüsünün tamamına dairesel maske (köşeleri siyaha boyayan) uygulanarak tahminler karşılaştırılmıştır:

| Maske yarıçap ölçeği | Tahmin uyumu | Doğruluk (maskesiz) | Doğruluk (maskeli) |
|---|---|---|---|
| 1.08 | 79.8% | 85.5% | 74.9% |
| 1.20 | 89.6% | 85.5% | 79.8% |

Maske yarıçapı büyütüldükçe (retina dokusunun daha az kesilmesi sağlandıkça) tahmin uyumu belirgin şekilde artmıştır (79.8% → 89.6%). Uyuşmayan örneklerin Grad-CAM incelemesinde modelin büyük çoğunlukla gerçek lezyonlara odaklandığı, köşe artefaktlarına anlamlı bir bağımlılık olmadığı sonucuna varılmıştır — kalan fark, modelin eğitim sırasında hiç dairesel maskelenmiş görüntü görmemiş olmasından (dağılım kayması) kaynaklanmaktadır.

> **Not:** Dairesel maskeleme yalnızca bir **doğrulama/sağlamlık testi** olarak uygulanmıştır — modelin karar mekanizmasını denetlemek amacıyla sonradan yapılmış bir deneydir. Ön işleme pipeline'ına (`dataset.py` / `APTOSDataset`) veya eğitim sürecine dahil edilmemiştir; eğitim ve nihai model hâlâ maskesiz (sadece Auto-Crop + CLAHE uygulanmış) görüntülerle çalışmaktadır.

**Maskesiz vs. maskeli Grad-CAM karşılaştırması, radius_scale=1.08 (5 örnek, farklı evrelerden):**

![Dairesel Maskeleme Grad-CAM Karşılaştırması](images/circular_masking_gradcam_comparison.png)

**Yarıçap büyütülünce (radius_scale=1.20), aynı örneklerde tahmin uyumu belirgin şekilde artmıştır:**

![Dairesel Maskeleme v2 — Büyük Yarıçap](images/test_v2_larger_radius_circular_masking.png)

**Tahminin değiştiği örnekler (radius_scale=1.08)** — çoğunlukla sınır çizgisindeki (borderline) evreler arasında (örn. Moderate ↔ Mild, Mild ↔ No DR) geçişler görülmüştür; ısı haritaları maskeleme sonrası da lezyon bölgelerine yakın kalmaya devam etmiştir:

<table>
<tr>
<td><img src="images/masked_vs_unmasked_prediction_diff_1.png" alt="Diff 1"></td>
</tr>
<tr>
<td><img src="images/masked_vs_unmasked_prediction_diff_2.png" alt="Diff 2"></td>
</tr>
<tr>
<td><img src="images/masked_vs_unmasked_prediction_diff_5.png" alt="Diff 5"></td>
</tr>
<tr>
<td><img src="images/masked_vs_unmasked_prediction_diff_6.png" alt="Diff 6"></td>
</tr>
<tr>
<td><img src="images/masked_vs_unmasked_prediction_diff_7.png" alt="Diff 7"></td>
</tr>
</table>

## Proje Yapısı

```
├── aptos_2019.ipynb          # Ana notebook — EDA, ön işleme, eğitim, değerlendirme, Grad-CAM
├── dataset.py                 # auto_crop, apply_clahe, APTOSDataset
├── model_utils.py              # Model yükleme, tahmin, ön işleme yardımcıları
├── gradcam_utils.py            # Grad-CAM üretimi ve görselleştirme
├── app.py                     # Streamlit arayüzü
├── requirements.txt
├── checkpoints/
│   ├── efficientnet_b0_best_model.pth
│   ├── resnet50_best_model.pth
│   ├── densenet121_best_model.pth
│   └── densenet121_stage2_best_qwk.pth   # Fine-tuned nihai model
└── README.md
```

## Kurulum ve Çalıştırma

```bash
pip install -r requirements.txt
```

Notebook, ortamı otomatik algılayarak (Kaggle / Google Colab / yerel) veri ve checkpoint yollarını buna göre ayarlar.

Görüntü veri seti ve eğitilmiş checkpoint'ler boyut kısıtları nedeniyle bu repoda yer almaz; Kaggle Datasets veya Google Drive üzerinden ayrıca sağlanmalıdır.

## Streamlit Arayüzü

```bash
streamlit run app.py
```

Arayüz, yüklenen bir retina görüntüsü için:
1. Auto-Crop + CLAHE ön işlemesini uygular,
2. DenseNet121 (fine-tuned) modeliyle DR evresi ve güven skorunu tahmin eder,
3. Grad-CAM ısı haritasını orijinal görüntünün üzerine bindirerek gösterir.

> ⚠️ Bu araç yalnızca eğitim/araştırma amaçlıdır, kesin tıbbi teşhis yerine geçmez.
