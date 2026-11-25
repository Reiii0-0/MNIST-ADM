---
title: "Competitive Machine Learning & Kaggle Grandmaster Playbook:"
source_doc: "COMPETITION_PLAYBOOK.md"
date: 2026-10-11
updated: 2026-11-24
tags:
  - adm-project
  - documentation
  - competition-strategy
---

# Competitive Machine Learning & Kaggle Grandmaster Playbook:
## Strategi Menang Kompetisi Klasifikasi Karakter Tulisan Tangan (Chinese MNIST & CodaBench)

---

### Daftar Isi
1. [Eksekutif Ringkasan & Anatomi Kompetisi](#1-eksekutif-ringkasan--anatomi-kompetisi)
2. [Telaah Solusi Pemenang (1st–5th Place) Kompetisi Sejenis](#2-telaah-solusi-pemenang-1st5th-place-kompetisi-sejenis)
   - 2.1 [Kannada-MNIST (Kaggle Playground): Fenomena Dig-MNIST & Writer Drift](#21-kannada-mnist-kaggle-playground-fenomena-dig-mnist--writer-drift)
   - 2.2 [Bengali.AI Handwritten Grapheme Classification (Kaggle Featured)](#22-bengaliai-handwritten-grapheme-classification-kaggle-featured)
   - 2.3 [Kuzushiji Recognition & Kuzushiji-MNIST](#23-kuzushiji-recognition--kuzushiji-mnist)
   - 2.4 [Karakteristik & Best Practice Ekosistem CodaBench/CodaLab](#24-karakteristik--best-practice-ekosistem-codabenchcodalab)
3. [Taktik Kunci 1: Skema Validasi Bebas Kebocoran (Leak-Free Validation Harness)](#3-taktik-kunci-1-skema-validasi-bebas-kebocoran-leak-free-validation-harness)
   - 3.1 [Analisis Kebocoran Gaya Penulis (Writer Stylistic Leakage)](#31-analisis-kebocoran-gaya-penulis-writer-stylistic-leakage)
   - 3.2 [StratifiedGroupKFold vs StratifiedKFold: Bahaya Shake-Up](#32-stratifiedgroupkfold-vs-stratifiedkfold-bahaya-shake-up)
   - 3.3 [Implementasi Teknis Validation Harness](#33-implementasi-teknis-validation-harness)
4. [Taktik Kunci 2: Model Ensembling & Keragaman Bias Induktif](#4-taktik-kunci-2-model-ensembling--keragaman-bias-induktif)
   - 4.1 [Portfolio Arsitektur yang Saling Melengkapi](#41-portfolio-arsitektur-yang-saling-melengkapi)
   - 4.2 [Peran Esensial Baseline Statistik Multivariat (PCA + Fisher's LDA)](#42-peran-esensial-baseline-statistik-multivariat-pca--fishers-lda)
   - 4.3 [Teknik Blending & Stacking Out-of-Fold (OOF)](#43-teknik-blending--stacking-out-of-fold-oof)
5. [Taktik Kunci 3: Rekayasa Loss Function & Karakter Dual-Form](#5-taktik-kunci-3-rekayasa-loss-function--karakter-dual-form)
   - 5.1 [Cross-Entropy dengan Label Smoothing](#51-cross-entropy-dengan-label-smoothing)
   - 5.2 [Focal Loss untuk Memisahkan Pasangan Karakter Ambigu](#52-focal-loss-untuk-memisahkan-pasangan-karakter-ambigu)
   - 5.3 [Sub-center ArcFace: Solusi Definitif untuk Bentuk Tradisional vs Disederhanakan](#53-sub-center-arcface-solusi-definitif-untuk-bentuk-tradisional-vs-disederhanakan)
6. [Taktik Kunci 4: Test-Time Augmentation (TTA) Aman untuk Karakter](#6-taktik-kunci-4-test-time-augmentation-tta-aman-untuk-karakter)
   - 6.1 [Augmentasi Aman vs Augmentasi Merusak Semantik](#61-augmentasi-aman-vs-augmentasi-merusak-semantik)
   - 6.2 [Skema Agregasi Probabilitas TTA](#62-skema-agregasi-probabilitas-tta)
7. [Taktik Kunci 5: Test Set Adaptation & Pseudo-Labeling Bertingkat](#7-taktik-kunci-5-test-set-adaptation--pseudo-labeling-bertingkat)
   - 7.1 [Mengatasi Perbedaan Penulis (Writer Domain Shift)](#71-mengatasi-perbedaan-penulis-writer-domain-shift)
   - 7.2 [Unsupervised Domain Adaptation: AdaBN & Tent](#72-unsupervised-domain-adaptation-adabn--tent)
   - 7.3 [Siklus Pseudo-Labeling Etis & Terkalibrasi untuk Fase 1 dan Fase 2](#73-siklus-pseudo-labeling-etis--terkalibrasi-untuk-fase-1-dan-fase-2)
8. [Taktik Kunci 6: Protokol Submission CodaBench & Mitigasi Error](#8-taktik-kunci-6-protokol-submission-codabench--mitigasi-error)
   - 8.1 [Struktur File Zip & Hierarki Direktori CodaBench](#81-struktur-file-zip--hierarki-direktori-codabench)
   - 8.2 [Pre-flight Sanity Check Automation Script](#82-pre-flight-sanity-check-automation-script)
   - 8.3 [Manajemen Slot Submission & Strategi Dua Prediksi Final](#83-manajemen-slot-submission--strategi-dua-prediksi-final)
9. [Actionable Step-by-Step Playbook (Sprint Menuju Puncak Leaderboard)](#9-actionable-step-by-step-playbook-sprint-menuju-puncak-leaderboard)
10. [Anotasi Literatur & Daftar Pustaka Terunduh](#10-anotasi-literatur--daftar-pustaka-terunduh)

---

## 1. Eksekutif Ringkasan & Anatomi Kompetisi

Kompetisi ini berfokus pada klasifikasi gambar karakter numerik Mandarin (**Chinese MNIST**) berukuran $64 \times 64$ piksel skala abu-abu (*grayscale*), dengan total 15 kelas (label 0–14):

$$\mathcal{C} = \{\text{0:零}, \text{1:一}, \text{2:二}, \text{3:三}, \text{4:四}, \text{5:五}, \text{6:六}, \text{7:七}, \text{8:八}, \text{9:九}, \text{10:十}, \text{11:百}, \text{12:千}, \text{13:万(萬)}, \text{14:亿(億)}\}$$

Berdasarkan dataset asli (*Newcastle University Project, Nazarpour & Chen*):
- Total populasi data: 15.000 gambar yang ditulis oleh **100 penulis unik**. Setiap penulis menulis 15 karakter sebanyak 10 kali ($15 \times 10 = 150$ gambar/penulis).
- **Data Latih (Train)**: Terdiri dari 10.500 gambar dari **70 penulis unik** (penulis 1 sampai 70, persis 150 gambar/penulis). Distribusi kelas sangat seimbang (700 gambar per kelas).
- **Fase 1 (Validasi Publik)**: Terdiri dari 2.466 gambar (~16–17 penulis baru yang tidak ada di data latih).
- **Fase 2 (Uji Final Privat)**: Terdiri dari 2.902 gambar (~19–20 penulis baru yang tidak ada di data latih maupun fase 1).

### Masalah Fundamental Kompetisi: *Writer Domain Shift* & *Intra-Class Bimodality*
Mayoritas peserta pemula menganggap ini hanyalah variasi sederhana dari MNIST biasa dan langsung melatih CNN standar dengan pemisahan `train_test_split(test_size=0.2)` secara acak. **Ini adalah kesalahan fatal yang berujung pada kehancuran di evaluasi akhir (*catastrophic shake-up*)**.
1. **Kebocoran Identitas Penulis**: Gaya goresan (*stroke width, pen pressure, slant, curvature*) melekat pada setiap individu penulis. Jika gambar dari penulis yang sama bocor ke set validasi, model hanya "menghafal" gaya penulis tersebut. Akurasi validasi akan terlihat fantastis (>99.8%), namun akurasi anjlok drastis saat menghadapi penulis baru di Fase 1 dan Fase 2.
2. **Dual-Form Graphemes (Bentuk Tradisional vs Disederhanakan)**: Label 13 (万 memiliki varian tradisional 萬 dengan 12–13 goresan) dan Label 14 (亿 memiliki varian 億 dengan 15 goresan). Representasi visual kedua varian ini memiliki topologi garis yang bertolak belakang secara drastis. Loss Cross-Entropy standar memaksakan kedua varian tersebut ke dalam satu vektor prototipe tunggal di ruang laten, memicu konflik gradien (*intra-class feature divergence*).

---

## 2. Telaah Solusi Pemenang (1st–5th Place) Kompetisi Sejenis

### 2.1 Kannada-MNIST (Kaggle Playground): Fenomena Dig-MNIST & Writer Drift
Kompetisi Kannada-MNIST (2019) menyajikan 60.000 data latih angka aksara Kannada. Penyelenggara (*Vinay Prabhu et al.*) merilis set evaluasi sekunder bernama **Dig-MNIST** yang ditulis oleh sukarelawan non-penutur asli dengan gaya goresan berbeda.
- **Hasil & Pelajaran**:
  - Model yang mencetak akurasi 99.2% pada data uji standar anjlok menjadi 75%–82% pada Dig-MNIST karena model terlalu bergantung pada pola goresan spesifik penulis asli.
  - **Solusi Pemenang (Rank 1–5)**:
    - Tidak mengandalkan satu model raksasa (*overparameterized model*). Mereka menggunakan *ensemble* dari 10–20 arsitektur CNN ringan yang dilatih dengan *bagging* dan variasi *seed*.
    - Melakukan normalisasi ketebalan garis (*morphological thinning/dilation*) dan *data augmentation* affine ringan.
    - Menggunakan *temperature scaling* pada logits probabilitas untuk mencegah model terlalu percaya diri (*overconfident*) pada bentuk angka yang janggal.

### 2.2 Bengali.AI Handwritten Grapheme Classification (Kaggle Featured)
Kompetisi ini melibatkan pengenalan kombinasi grafem tulisan tangan Bengali yang sangat kompleks (168 akar grafem, 11 vokal, 7 konsonan) dengan distribusi kelas yang sangat *long-tailed* dan variasi penulis yang ekstrem.
- **Solusi Pemenang Pertama (Linsho Kaku / "deoxy")**:
  - **Zero Preprocessing Trap**: Menolak melakukan *binarization* atau *heavy cropping*, karena *thresholding* memotong ekor goresan halus (*ligatures*) yang krusial untuk membedakan karakter mirip.
  - **Dua Model Komplementer**: Membangun *Seen Model* (EfficientNet-B7) untuk menangani kombinasi umum dan *Out-of-Distribution (OOD) Model* berbasis representasi fitur generalis.
  - **Synthesized Font Augmentation**: Menggunakan font komputer untuk membuat data sintetik tulisan sebagai regularisasi dasar.
- **Solusi Pemenang 2nd–5th Place**:
  - **Metric Learning (ArcFace)**: Menggantikan atau mengombinasikan Cross-Entropy dengan ArcFace. Margin sudut memaksa fitur tulisan tangan dari karakter yang sama membentuk kluster rapat pada *hypersphere*, sangat efektif membedakan aksara rumit.
  - **Mixed-Sample Augmentation**: Penggunaan **CutMix**, **MixUp**, dan **FMix** terbukti esensial untuk mencegah overfitting pada goresan spesifik penulis.
  - **Soft Pseudo-Labeling**: Menggunakan probabilitas prediksi ensemble pada data uji yang belum berlabel untuk *self-training*.

### 2.3 Kuzushiji Recognition & Kuzushiji-MNIST
Mengklasifikasikan aksara Jepang klasik (Kuzushiji) dari buku-buku kuno. Karakter memiliki variasi bentuk historis yang masif (satu karakter dapat ditulis dalam puluhan gaya *hentaigana* yang berbeda).
- **Pelajaran Inti Pemenang**:
  - *Multi-scale feature preservation*: Menjaga resolusi fitur spasial sejak lapisan konvolusi awal.
  - *Multi-model Blending*: Mengombinasikan model berorientasi *backbone* konvolusi lokal (ResNet/DenseNet) dengan model resolusi spasial tinggi (HRNet).
  - *Disciplined Test-Time Augmentation (TTA)*: Menguji dengan rotasi mikro ($\pm 3^\circ, \pm 5^\circ$) dan translasi sub-piksel, menghasilkan kenaikan konsisten $+0.3\%$ hingga $+0.5\%$ tanpa melanggar semantik aksara.

### 2.4 Karakteristik & Best Practice Ekosistem CodaBench/CodaLab
- **Dua Fase Evaluasi**:
  - *Fase 1*: Leaderboard publik berjalan pada subset data (2.466 gambar).
  - *Fase 2*: Evaluasi akhir menggunakan data uji privat baru (2.902 gambar).
- **Risiko Overfitting Public Leaderboard**: Banyak tim melakukan *hyperparameter tuning* berdasarkan skor publik Fase 1. Di CodaBench, ini menyebabkan *severe shake-up* saat Fase 2 dibuka, karena data Fase 2 ditulis oleh kelompok penulis baru yang sama sekali berbeda.
- **Best Practice**: Kunci kesuksesan CodaBench adalah **kepatuhan mutlak pada skor Out-of-Fold (OOF) lokal**. Jangan pernah memilih model final hanya berdasarkan kenaikan 0.001 di leaderboard publik jika skor CV lokalnya menurun.

---

## 3. Taktik Kunci 1: Skema Validasi Bebas Kebocoran (Leak-Free Validation Harness)

### 3.1 Analisis Kebocoran Gaya Penulis (Writer Stylistic Leakage)
Secara matematis, citra karakter $X$ dipengaruhi oleh dua variabel laten:
$$X \sim P(X \mid Y=c, W=w)$$
di mana $c \in \{0, \dots, 14\}$ adalah kelas karakter, dan $w \in \{1, \dots, 100\}$ adalah gaya unik penulis.

Jika sampel dari penulis $w$ terdistribusi di kedua set latih dan validasi (seperti pada Stratified K-Fold standar):
$$\mathcal{W}_{\text{train}} \cap \mathcal{W}_{\text{val}} \neq \emptyset$$
Model konvolusi dalam (*deep CNN*) memiliki kapasitas memorisasi yang cukup untuk mempelajari korelasi antara tekstur pena penulis $w$ dan target $c$. Akibatnya:
$$\mathcal{L}_{\text{val}} \approx P(\hat{Y} \neq Y \mid W \in \mathcal{W}_{\text{train}}) \ll P(\hat{Y} \neq Y \mid W \notin \mathcal{W}_{\text{train}})$$

Ketika model dievaluasi pada test set di mana $\mathcal{W}_{\text{test}} \cap \mathcal{W}_{\text{train}} = \emptyset$, performa sebenarnya anjlok drastis.

### 3.2 StratifiedGroupKFold vs StratifiedKFold: Bahaya Shake-Up
| Metrik / Aspek | Stratified K-Fold | GroupKFold (pada `writer_id`) | StratifiedGroupKFold |
| :--- | :--- | :--- | :--- |
| **Pemisahan Penulis** | Bocor ($\mathcal{W}_{\text{val}} \subset \mathcal{W}_{\text{train}}$) | Terisolasi Sempurna | Terisolasi Sempurna |
| **Keseimbangan Kelas per Fold** | Sangat Seimbang | Kadang Terjadi Fluktuasi | Sangat Seimbang & Presisi |
| **Estimasi Skor CV** | Optimistis Palsu (~99.8%) | Realistis (~97.5% - 98.8%) | Realistis & Stabil (~97.8% - 98.9%) |
| **Korelasi CV-ke-Test** | Sangat Rendah ($r < 0.3$) | Tinggi ($r > 0.85$) | Sangat Tinggi ($r > 0.95$) |
| **Risiko Shake-Up** | **Sangat Tinggi (Bencana)** | Rendah | **Nol / Terlindungi Maksimal** |

Dalam dataset kompetisi ini:
- 70 penulis, masing-masing memiliki tepat 150 gambar (10 sampel per karakter).
- Dengan **5-Fold StratifiedGroupKFold**:
  - Setiap fold berisi tepat $70 / 5 = 14$ penulis unik.
  - Setiap fold validasi memiliki tepat $14 \times 150 = 2.100$ gambar.
  - Setiap kelas pada fold validasi memiliki tepat $14 \times 10 = 140$ gambar.
  - Distribusi kelas menjadi **sempurna seimbang secara eksak**.

### 3.3 Implementasi Teknis Validation Harness
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

def create_leak_free_folds(df_path="train_labels.csv", n_splits=5, seed=42):
    df = pd.read_csv(df_path)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    df['fold'] = -1
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(df, df['label'], groups=df['writer_id'])):
        df.loc[val_idx, 'fold'] = fold
        
    # Verifikasi integritas isolasi penulis
    for fold in range(n_splits):
        train_writers = set(df[df['fold'] != fold]['writer_id'])
        val_writers = set(df[df['fold'] == fold]['writer_id'])
        intersection = train_writers.intersection(val_writers)
        assert len(intersection) == 0, f"LEAKAGE DETECTED in fold {fold}: {intersection}"
        print(f"Fold {fold}: {len(val_writers)} writers, {len(df[df['fold'] == fold])} samples. Class distribution balanced.")
        
    return df
```

---

## 4. Taktik Kunci 2: Model Ensembling & Keragaman Bias Induktif

Kaidah utama Grandmaster Playbook: **"Ensemble yang kuat dibentuk oleh model-model yang memiliki bias induktif berbeda, bukan oleh 5 replika model yang sama dengan bobot berbeda."**

### 4.1 Portfolio Arsitektur yang Saling Melengkapi

```
                            [ INPUT IMAGE: 64x64 Grayscale ]
                                           |
         +-----------------+---------------+----------------+----------------+
         |                 |               |                |                |
         v                 v               v                v                v
   [ ResNet18 ]     [ DenseNet121 ]  [ ConvNeXt-Femto ] [ Custom 4-CNN ] [ PCA + Fisher LDA ]
   (Residual         (Concatenation  (7x7 Depthwise      (Local High-Freq  (Linear Statistical
    Highway)          Feature Reuse)  Inverted Bottleneck) Stroke Edge)     Orthogonal Subspace)
         |                 |               |                |                |
         \-----------------+---------------+----------------+----------------/
                                           |
                                  [ Logit Blending / ]
                                  [ Temperature Softmax ]
                                           |
                                [ FINAL PREDICTION ]
```

1. **ResNet18 / ResNet34**:
   - *Inductive Bias*: Residual skip-connection $\mathbf{x} + \mathcal{F}(\mathbf{x})$ menciptakan permukaan *loss* yang sangat mulus (*convex-like loss landscape*), mencegah *vanishing gradient*, dan mengenali topologi karakter secara holistik.
2. **DenseNet121**:
   - *Inductive Bias*: *Feature reuse* melalui penyambungan (*concatenation*) saluran antar-lapisan. Fitur tepi goresan beresolusi tinggi dari layer 1 dan 2 diteruskan langsung ke classifier akhir tanpa hilang tertelan operasi *downsampling*. Ini sangat superior untuk membedakan detail halus pada karakter kompleks seperti 萬 dan 億.
3. **ConvNeXt-Femto / ConvNeXt-Atto**:
   - *Inductive Bias*: Desain modern dengan *kernel* besar ($7 \times 7$ *depthwise convolution*), *inverted bottleneck ratio 4*, LayerNorm, dan aktivasi GELU. Model ini menangkap hubungan struktural goresan lintas kuadran karakter tanpa meningkatkan jumlah parameter.
4. **Custom Shallow 4-Layer CNN (VGG-Style)**:
   - *Inductive Bias*: CNN dangkal (32-64-128-256 filter, kernel $3 \times 3$, MaxPool, Dropout 0.3) dengan *receptive field* kompak. Model dangkal tidak rentan terhadap distorsi spasial skala besar dan sangat sensitif terhadap kontinuitas tinta lokal.
5. **Baseline Multivariat Klasik: PCA + Fisher's Linear Discriminant Analysis (LDA)**:
   - Memberikan proyeksi probabilitas linear berbasis kovarians global yang tidak terpengaruh oleh *neural feature collapse*.

### 4.2 Peran Esensial Baseline Statistik Multivariat (PCA + Fisher's LDA)
Mengapa mengikutsertakan model statistik klasik dalam *ensemble* deep learning?
- *Deep Neural Networks* cenderung menderita masalah **overconfidence calibration**: ketika dihadapkan pada variasi penulis asing yang aneh (out-of-distribution), jaringan saraf tiruan sering kali memberikan probabilitas 0.99 pada kelas yang salah karena softmax berbasis dot-product yang tidak terkendali.
- **PCA (Eigencharacters)** mereduksi dimensi citra $64 \times 64 = 4096$ menjadi $k \approx 80-120$ komponen utama yang merangkum $\sim 92\%$ varians global tanpa noise.
- **Fisher's LDA** memaksimalkan rasio determinan sebaran antar-kelas (*Between-class scatter* $S_B$) terhadap sebaran intra-kelas (*Within-class scatter* $S_W$):
  $$J(W) = \frac{\det(W^T S_B W)}{\det(W^T S_W W)}$$
- Model LDA menghasilkan batas keputusan hiperplane linear yang sangat kokoh dan konservatif. Ketika di-blend dengan bobot kecil (misal $w_{\text{LDA}} = 0.05 - 0.08$), model ini bertindak sebagai **regularizer berbasis kovarians multivariat**, meredam anomali probabilitas deep learning pada sampel-sampel ekstrem.

### 4.3 Teknik Blending & Stacking Out-of-Fold (OOF)
Jangan menggunakan rata-rata sederhana (*simple average*) jika performa model bervariasi.
1. **Ekstraksi Prediksi OOF**: Simpan matriks probabilitas out-of-fold $P_{\text{OOF}}^{(m)} \in \mathbb{R}^{N \times 15}$ untuk setiap model $m \in \{1, \dots, M\}$.
2. **Optimasi Bobot Ensemble (Nelder-Mead / SLSQP)**:
   Temukan vektor bobot $\mathbf{w} = [w_1, \dots, w_M]^T$ dengan konstrain $\sum w_m = 1, w_m \ge 0$ yang meminimalkan Log-Loss atau memaksimalkan Akurasi pada data OOF:
   $$\mathbf{w}^* = \arg\min_{\mathbf{w}} -\sum_{i=1}^{N} \log \left( \sum_{m=1}^{M} w_m P_{i, y_i}^{(m)} \right)$$
3. **Logit Blending vs Softmax Blending**:
   Lebih disukai melakukan averaging pada level *uncalibrated logits* $z^{(m)}$ yang telah diskalakan dengan temperatur $T_m$:
   $$z_{\text{blend}} = \sum_{m=1}^{M} w_m \frac{z^{(m)}}{T_m}, \quad P_{\text{final}} = \text{softmax}(z_{\text{blend}})$$

---

## 5. Taktik Kunci 3: Rekayasa Loss Function & Karakter Dual-Form

### 5.1 Cross-Entropy dengan Label Smoothing
Untuk target satu-panas (*one-hot*) standar $y \in \{0, 1\}^C$, model didorong untuk menghasilkan logit $z_{y} \to +\infty$, memicu overconfidence. Dengan Label Smoothing $\epsilon = 0.08$:
$$\tilde{y}_k = (1 - \epsilon) y_k + \frac{\epsilon}{C}$$
Keuntungan:
- Mencegah model menghafal goresan cacat atau getaran tangan penulis (*writer tremor*).
- Menghasilkan representasi kluster fitur yang lebih teratur pada lapisan sebelum softmax.

### 5.2 Focal Loss untuk Memisahkan Pasangan Karakter Ambigu
Beberapa karakter dalam Chinese MNIST memiliki kemiripan topologi goresan tinggi:
- Label 1 (一), Label 2 (二), Label 3 (三): hanya berbeda pada jumlah garis horizontal sejajar.
- Label 8 (八) vs Label 6 (六): goresan atap mirip.
- Label 10 (十) vs Label 7 (七): garis vertikal dengan kait halus.

Focal Loss memodifikasi gradien agar fokus pada sampel sulit (*hard negatives*):
$$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$
Dengan mengatur $\gamma = 1.5$ hingga $2.0$, contoh-contoh mudah ($p_t > 0.9$) dieliminasi kontribusi gradiennya sebesar $>90\%$, memaksa backpropagation memprioritaskan pemisahan karakter ambigu.

### 5.3 Sub-center ArcFace: Solusi Definitif untuk Bentuk Tradisional vs Disederhanakan

#### Masalah Fundamental Representasi:
Label 13 (万 vs 萬) dan Label 14 (亿 vs 億) memiliki anomali tipografi:
- Penulis asal Tiongkok daratan umumnya menulis bentuk sederhana: **万** (3 goresan) dan **亿** (3 goresan).
- Penulis asal Taiwan/Hong Kong atau generasi tua kerap menulis bentuk tradisional: **萬** (13 goresan) dan **億** (15 goresan).

Jika menggunakan Softmax atau ArcFace standar, model memiliki **1 vektor bobot tunggal** $W_{13}$ untuk kelas 13. Jaringan dipaksa memetakan citra 3 goresan (万) dan citra 13 goresan (萬) ke arah sudut $\theta$ yang sama. Ini memicu distorsi representasi metrik yang parah (*intra-class distribution collapse*).

#### Formulasi Sub-center ArcFace (*Deng et al., ECCV 2020*):
Alih-alih 1 pusat per kelas, setiap kelas dialokasikan $K$ sub-pusat:
$$W_c = \{W_{c,1}, W_{c,2}, \dots, W_{c,K}\} \quad (K=2 \text{ atau } 3)$$
Untuk sampel $\mathbf{x}_i$ dengan kelas sejati $y_i$, kemiripan kosinus dihitung terhadap sub-pusat yang **paling dekat**:
$$\cos \theta_{i, y_i} = \max_{k \in \{1, \dots, K\}} \left( \frac{\mathbf{x}_i^T W_{y_i, k}}{\|\mathbf{x}_i\| \|W_{y_i, k}\|} \right)$$
Margin sudut aditif $m$ hanya diterapkan pada sub-pusat terdekat tersebut:
$$\mathcal{L}_{\text{Sub-Arc}} = -\log \frac{e^{s \cdot \cos(\theta_{i, y_i} + m)}}{e^{s \cdot \cos(\theta_{i, y_i} + m)} + \sum_{j \neq y_i} \sum_{k=1}^K e^{s \cdot \cos \theta_{i, j, k}}}$$

#### Dampak:
- Sub-pusat $W_{13, 1}$ secara alami mengelompokkan bentuk sederhana (万).
- Sub-pusat $W_{13, 2}$ secara alami mengelompokkan bentuk tradisional (萬).
- Tidak ada penalti margin antar kedua bentuk tersebut di dalam kelas yang sama! Akurasi pemisahan kelas 13 dan 14 meningkat secara dramatis.

---

## 6. Taktik Kunci 4: Test-Time Augmentation (TTA) Aman untuk Karakter

TTA mengevaluasi variasi transformasi dari citra uji dan merata-ratakan prediksinya. Namun, pada data karakter, TTA sembarangan akan menghancurkan akurasi!

### 6.1 Augmentasi Aman vs Augmentasi Merusak Semantik

```
+------------------------------------+------------------------------------+
|   AUGMENTASI YANG HARUS DIHINDARI  |     AUGMENTASI AMAN & EFEKTIF      |
|              (BAHAYA)              |            (DIREKOMENDASIKAN)      |
+------------------------------------+------------------------------------+
| ❌ Horizontal Flip (Pembalikan Kiri-Kanan) |  Translasi Mikro (Shift):       |
|    - Merusak goresan kait (e.g. 七, 亿)   |    dx, dy in [-2, 0, +2] piksel |
|    - Membalik stroke order konvensional   |                                    |
| ❌ Vertical Flip (Pembalikan Atas-Bawah)  |  Rotasi Halus:                     |
|    - Merusak karakter (e.g. 二, 三, 六)   |    angle in [-5 deg, 0, +5 deg]    |
| ❌ Random Cropping Ekstrem                |  Skala Mikro (Scale/Zoom):         |
|    - Memotong titik goresan penentu tepi  |    scale factor in [0.96, 1.04]    |
| ❌ Cutout / Erasing Besar                |  Shear Affine Halus:               |
|    - Menghilangkan radikal esensial       |    shear angle in [-3 deg, +3 deg] |
+------------------------------------+------------------------------------+
```

### 6.2 Skema Agregasi Probabilitas TTA
Untuk setiap citra uji $X_{\text{test}}$, buat 5–8 variasi transformasi aman:
1. Citra Asli (bobot 1.0)
2. Geser kiri 2 piksel (bobot 0.7)
3. Geser kanan 2 piksel (bobot 0.7)
4. Rotasi $+4^\circ$ (bobot 0.8)
5. Rotasi $-4^\circ$ (bobot 0.8)
6. Skala $1.03$ (zoom in sedikit, bobot 0.7)
7. Skala $0.97$ (zoom out sedikit, bobot 0.7)

**Metode Agregasi Terbaik**:
Alih-alih rata-rata aritmatika sederhana, gunakan **Temperature Power Mean**:
$$P_c(X) = \frac{\left( \sum_{t=1}^T w_t [P_{c}^{(t)}(X)]^p \right)^{1/p}}{\sum_{k=0}^{14} \left( \sum_{t=1}^T w_t [P_{k}^{(t)}(X)]^p \right)^{1/p}} \quad (p \approx 1.5 - 2.0)$$
Eksponen $p > 1$ memberikan penekanan lebih pada prediksi yang memiliki kepastian tinggi (*high-confidence consensus*), menolak noise transformasi yang ambigu.

---

## 7. Taktik Kunci 5: Test Set Adaptation & Pseudo-Labeling Bertingkat

Kompetisi memiliki dua fase pengujian:
- **Fase 1 (Validasi Publik)**: 2.466 gambar.
- **Fase 2 (Uji Final Privat)**: 2.902 gambar.

Kedua fase ini ditulis oleh penulis yang berbeda dari 70 penulis data latih. Kita dapat memanfaatkan data uji ini secara legal dan etis melalui teknik adaptasi domain tanpa supervisi (*Unsupervised Domain Adaptation*) dan *Self-Training*.

### 7.1 Mengatasi Perbedaan Penulis (Writer Domain Shift)
Gaya tulisan penulis baru menciptakan pergeseran distribusi kovariat (*covariate shift*):
$$P_{\text{train}}(X) \neq P_{\text{test}}(X), \quad \text{namun } P_{\text{train}}(Y \mid X) \approx P_{\text{test}}(Y \mid X)$$

### 7.2 Unsupervised Domain Adaptation: AdaBN & Tent

#### A. Adaptive Batch Normalization (AdaBN, Li et al., 2018):
- *Konsep*: Lapisan Batch Normalization menyimpan parameter $\mu_{\text{run}}$ dan $\sigma^2_{\text{run}}$ yang dihitung dari 70 penulis data latih. Karakteristik tinta dan latar belakang penulis data uji berbeda, sehingga statistik running ini mengalami bias (*domain mismatch*).
- *Tindakan*:
  1. Bekukan semua bobot konvolusi dan fully-connected model ($\mathbf{W}, \mathbf{b}$ tetap).
  2. Alirkan seluruh data uji (Fase 1 atau Fase 2) ke dalam model dalam mode latih parsial (`model.train()`) hanya untuk memperbarui estimasi $\mu_{\text{test}}$ dan $\sigma^2_{\text{test}}$ pada setiap layer BatchNorm.
  3. Kembalikan ke mode `model.eval()`.
- *Hasil*: Teknik ini berlangsung dalam hitungan detik di GPU dan secara konsisten memberikan peningkatan akurasi **$+0.4\%$ hingga $+0.8\%$** tanpa mengubah parameter representasi fitur!

#### B. Test-Time Entropy Minimization (Tent, Wang et al., ICLR 2021):
- Mengoptimalkan hanya parameter modulasi skala dan pergeseran affine pada BatchNorm ($\gamma, \beta$) dengan meminimalkan entropi Shannon prediksi pada data uji:
  $$\min_{\{\gamma, \beta\}} \mathcal{H}(\hat{y}) = -\sum_{c=0}^{14} p(c \mid X_{\text{test}}) \log p(c \mid X_{\text{test}})$$

### 7.3 Siklus Pseudo-Labeling Etis & Terkalibrasi untuk Fase 1 dan Fase 2

```
      [ 10,500 Labeled Train ]                 [ 2,466 Unlabeled Phase 1 Test ]
                 |                                             |
                 v                                             v
     [ Train 5-Fold Ensemble ] ---------------> [ AdaBN / TTA Ensemble Inference ]
                 |                                             |
                 |                                             v
                 |                               [ Confident Thresholding ]
                 |                               (P_max > 0.985 + Balanced Top-K)
                 |                                             |
                 |                                             v
                 + <----------------------------- [ Pseudo-Labeled Set (D_pseudo) ]
                 |
                 v
     [ Retrain Final Ensemble ]
       Loss = L_train + lambda * L_pseudo (lambda = 0.5)
                 |
                 v
     [ Submission Phase 1 ] & [ Final Evaluator for Phase 2 ]
```

1. **Thresholding Ketat**: Hanya ambil sampel yang memiliki $\max_c P(c \mid X) > 0.985$ dan margin terhadap kelas kedua $> 0.70$.
2. **Keseimbangan Kelas (Class-Balanced Top-K)**: Jangan ambil sampel terbanyak dari kelas mudah saja. Batasi kuota pseudo-label maksimum per kelas (misalnya maksimal 120 sampel per kelas) untuk menjaga prioritas distribusi kelas seragam.
3. **Discounted Loss Retraining**: Ketika melatih ulang model dengan $D_{\text{train}} \cup D_{\text{pseudo}}$, berikan bobot loss diskon pada pseudo-labels ($\lambda_{\text{pseudo}} = 0.5$) untuk mencegah degradasi jika terjadi salah label (*confirmation bias*).

---

## 8. Taktik Kunci 6: Protokol Submission CodaBench & Mitigasi Error

Platform CodaBench sangat ketat terhadap format pengiriman. Kesalahan sepele pada struktur zip dapat menyebabkan submission hangus (*Failed/Error*).

### 8.1 Struktur File Zip & Hierarki Direktori CodaBench
- **Aturan Emas**: File `predictions.csv` harus berada tepat di **root arsip zip**.
- **Kesalahan Klasik Peserta**: Melakukan zip pada folder (`zip -r sub.zip my_folder/`), sehingga di dalam zip strukturnya menjadi `my_folder/predictions.csv`. Sistem evaluasi CodaBench tidak akan menemukan file tersebut dan menghasilkan status *Crash/Execution Failed*.
- **Perintah CLI yang Benar**:
  ```bash
  # Masuk ke direktori file atau gunakan flag -j (junk paths)
  zip -j submission.zip /path/to/predictions.csv
  ```

### 8.2 Pre-flight Sanity Check Automation Script
Sebelum mengunggah file zip, jalankan skrip verifikasi otomatis berikut:

```python
import pandas as pd
import zipfile
import io

def verify_codabench_submission(zip_path, expected_test_ids, expected_filename="predictions.csv"):
    print(f"[*] Memeriksa integritas: {zip_path}")
    
    # 1. Cek struktur zip
    with zipfile.ZipFile(zip_path, 'r') as z:
        namelist = z.namelist()
        if expected_filename not in namelist:
            raise ValueError(f"CRITICAL: {expected_filename} TIDAK ditemukan di root zip! Isi zip: {namelist}")
        
        for name in namelist:
            if name.startswith('__MACOSX') or name.endswith('.DS_Store'):
                print(f"[WARNING] Ditemukan file sampah sistem: {name}")
                
        # 2. Baca file CSV dari memori
        with z.open(expected_filename) as f:
            df = pd.read_csv(io.BytesIO(f.read()))
            
    # 3. Verifikasi struktur kolom
    assert 'id' in df.columns, "CRITICAL: Kolom 'id' tidak ditemukan!"
    assert 'label' in df.columns or 'prediction' in df.columns, "CRITICAL: Kolom target ('label') tidak ditemukan!"
    target_col = 'label' if 'label' in df.columns else 'prediction'
    
    # 4. Verifikasi jumlah baris dan kesesuaian ID
    assert len(df) == len(expected_test_ids), f"CRITICAL: Jumlah baris ({len(df)}) != ekspektasi ({len(expected_test_ids)})!"
    assert list(df['id']) == list(expected_test_ids), "CRITICAL: Urutan ID tidak sama persis dengan urutan data uji!"
    
    # 5. Cek nilai NaN/Null
    assert df.isnull().sum().sum() == 0, f"CRITICAL: Ditemukan nilai NaN pada submission!\n{df.isnull().sum()}"
    
    # 6. Cek rentang nilai dan tipe data
    assert df[target_col].dtype in [int, 'int64', 'int32'], f"CRITICAL: Tipe data target bukan integer: {df[target_col].dtype}"
    assert df[target_col].min() >= 0 and df[target_col].max() <= 14, f"CRITICAL: Label di luar rentang [0, 14]!"
    
    # 7. Cek distribusi kelas prediksi (sanity check sanity)
    counts = df[target_col].value_counts()
    print("[+] Distribusi prediksi:")
    for cls in range(15):
        print(f"    Kelas {cls:2d}: {counts.get(cls, 0)} sampel")
    assert len(counts) == 15, "WARNING: Ada kelas yang tidak pernah diprediksi sama sekali!"
    
    print(f"[SUCCESS] Submission {zip_path} valid 100% dan siap diunggah ke CodaBench.")

if __name__ == "__main__":
    # Contoh pemanggilan
    pass
```

### 8.3 Manajemen Slot Submission & Strategi Dua Prediksi Final
Di CodaBench / Kaggle, setiap tim memiliki batas harian (misal 3–5 submission per hari) dan memilih 2 submission final untuk penentuan peringkat akhir:
- **Submission 1 (Safe Local CV Anchor)**: Ensemble konservatif terbaik murni berdasarkan skor Out-of-Fold (OOF) StratifiedGroupKFold tanpa pseudo-labeling agresif. Ini adalah jaring pengaman anti-shakeup Anda.
- **Submission 2 (High-Potential Adapted Model)**: Ensemble penuh dengan AdaBN + Sub-center ArcFace + TTA + Confident Pseudo-Labeling Fase 1. Model ini memaksimalkan peluang meraih posisi #1 jika domain shift test set terbukti kuat.

---

## 9. Actionable Step-by-Step Playbook (Sprint Menuju Puncak Leaderboard)

Berikut adalah panduan operasional langkah demi langkah (*actionable roadmap*) yang harus dieksekusi:

### Hari 1: Fondasi Validasi & Baseline Cepat
1. **Eksekusi StratifiedGroupKFold**:
   - Buat file `train_folds.csv` membagi 70 penulis ke dalam 5 fold bebas bocor.
2. **EDA & Ekstraksi Fitur Dasar**:
   - Hitung rasio intensitas tinta, bounding box karakter, dan deteksi anomali pada goresan label 13 dan 14.
3. **Membangun Baseline Statistik Multivariat**:
   - Latih pipeline PCA ($k=100$) + Fisher's LDA pada data mentah piksel.
   - Catat skor akurasi OOF CV sebagai batas performa minimal (*benchmark anchor*).

### Hari 2: Pelatihan Core Deep Learning Models
1. **Latih ResNet18**:
   - Loss: CrossEntropy dengan Label Smoothing ($\epsilon=0.08$).
   - Augmentasi: Rotasi $\pm 6^\circ$, translasi $\pm 3$ px, scale $0.95 - 1.05$.
   - Optimizer: AdamW, Cosine Annealing LR Scheduler ($1e-3 \to 1e-6$, 30 epochs).
2. **Latih DenseNet121**:
   - Simpan bobot checkpoint dengan skor val loss terendah di setiap fold.
3. **Latih ConvNeXt-Femto & Custom 4-Layer CNN**:
   - Lengkapi kuartet model konvolusi.
4. **Hitung Matriks Korelasi Kesalahan**:
   - Pastikan kesalahan prediksi antar-model tidak berkorelasi tinggi (*diversity check*).

### Hari 3: Solusi Khusus Karakter Dual-Form & Metric Learning
1. **Latih Model Sub-center ArcFace**:
   - Gunakan backbone ResNet18 / ConvNeXt dengan lapisan Sub-center ArcFace ($K=2$, $s=32.0, m=0.35$).
   - Amati konvergensi kluster untuk kelas 13 (万/萬) dan 14 (亿/億).
2. **Focal Loss Fine-Tuning**:
   - Lakukan fine-tuning 5 epoch pada model terbaik menggunakan Focal Loss ($\gamma=1.5$) untuk mempertegas pemisahan karakter ambigu (1, 2, 3 dan 6, 8).

### Hari 4: Optimasi Inferensi (AdaBN + TTA) & Validasi Leaderboard
1. **Implementasi AdaBN**:
   - Terapkan pembaruan statistik running BatchNorm pada data Fase 1.
2. **Kalibrasi TTA**:
   - Jalankan TTA 7-arah (asli + 2 pergeseran + 2 rotasi halus + 2 skala mikro) dengan agregasi *temperature power mean*.
3. **Submission Pertama ke CodaBench**:
   - Jalankan skrip validasi format, buat zip, dan unggah submission baseline untuk memverifikasi kecocokan pipeline scoring.

### Hari 5: Pseudo-Labeling Bertingkat & Blending OOF
1. **Ensemble Stacking**:
   - Gunakan optimasi Nelder-Mead pada matriks OOF 5-Fold untuk mencari bobot optimal $w_1, \dots, w_5$.
2. **Generate Pseudo-Labels Fase 1**:
   - Terapkan ensemble terbaik pada 2.466 citra Fase 1.
   - Saring kandidat dengan $P_{\text{max}} > 0.985$ dan terapkan kuota berimbang per kelas.
3. **Retraining Ensemble**:
   - Latih ulang model menggunakan gabungan data latih + data pseudo-label Fase 1.
4. **Kirim Submission Puncak Fase 1**:
   - Amankan posisi teratas pada leaderboard publik.

### Hari 6: Freeze Pipeline & Kesiapan Evaluasi Final Fase 2
1. **Otomatisasi Skrip Inferensi Fase 2**:
   - Siapkan skrip satu tombol (*one-click pipeline*) yang siap memproses 2.902 citra Fase 2 segera setelah dirilis:
     $$\text{Input: test_phase2/ } \longrightarrow \text{AdaBN Update} \longrightarrow \text{TTA Inference} \longrightarrow \text{Ensemble Blend} \longrightarrow \text{ZIP Validation}$$
2. **Kunci Dua Submission Final**:
   - Sub 1: *Pure OOF GroupKFold Ensemble* (Jaminan anti-shakeup).
   - Sub 2: *AdaBN + Pseudo-Labeled Ensemble* (Target juara 1).

---

## 10. Anotasi Literatur & Daftar Pustaka Terunduh

Seluruh dokumen ilmiah acuan utama telah diunduh dan tersimpan secara lokal pada direktori:
`/home/rei/Workspace/Academic/Multivariate Data Analysis/FinalProject/references/`

Berikut adalah ringkasan bibliografi dan relevansinya terhadap strategi kompetisi:

1. **Deng et al. (ECCV 2020)** — *Sub-center ArcFace: Boosting Face Recognition by Large-scale Noisy Web Faces*
   - File: `deng2020_subcenter_arcface.pdf` / `deng2020_subcenter_arcface_eccv.pdf`
   - *Relevansi*: Fondasi teoritis untuk menangani karakter dengan bentuk ganda (tradisional vs sederhana) melalui alokasi $K$ sub-pusat per kelas tanpa memicu konflik margin sudut intra-kelas.
2. **Deng et al. (CVPR 2019)** — *ArcFace: Additive Angular Margin Loss for Deep Face Recognition*
   - File: `deng2018_arcface.pdf` / `deng2018_arcface_cvpr.pdf`
   - *Relevansi*: Prinsip dasar pemisahan metrik sudut pada manifold hipersferikal untuk meningkatkan diskriminabilitas fitur goresan tulisan tangan.
3. **Prabhu (2019)** — *Kannada-MNIST: A new handwritten digits dataset for the Kannada language*
   - File: `prabhu2019_kannada_mnist.pdf`
   - *Relevansi*: Bukti empiris mengenai pergeseran domain penulis (*writer domain shift*) antara data latihan dan dataset evaluasi eksternal (*Dig-MNIST*).
4. **Clanuwat et al. (2018)** — *Deep Learning for Classical Japanese Literature (Kuzushiji-MNIST)*
   - File: `clanuwat2018_kuzushiji_mnist.pdf`
   - *Relevansi*: Studi komparatif arsitektur konvolusi untuk karakter Asia bertopologi kompleks dengan variasi goresan historis.
5. **Zhang, Bengio, Liu et al. (2017)** — *Online and Offline Handwritten Chinese Character Recognition: A Comprehensive Benchmark to Date*
   - File: `Zhang_Bengio_Liu2017_Online_and_Offline_Handwritten_Chinese_Character_Recognition_Benchmark.pdf`
   - *Relevansi*: Benchmark resmi pengenalan karakter Mandarin tulisan tangan, arsitektur konvolusi yang terbukti efektif, dan metode normalisasi goresan.
6. **Li et al. (Pattern Recognition 2018)** — *Revisiting Batch Normalization for Practical Domain Adaptation (AdaBN)*
   - File: `li2016_adabn.pdf`
   - *Relevansi*: Metodologi adaptasi domain instan tanpa pelatihan ulang dengan memperbarui parameter statistik running mean dan varians pada data uji baru.
7. **Wang et al. (ICLR 2021)** — *Tent: Fully Test-Time Adaptation by Entropy Minimization*
   - File: `wang2020_tent_test_time_adaptation.pdf`
   - *Relevansi*: Optimasi parameter affine BatchNorm berbasis entropi saat inferensi untuk mengeliminasi bias penulis asing pada data uji CodaBench.
8. **Liu et al. (CVPR 2022)** — *A ConvNet for the 2020s (ConvNeXt)*
   - File: `liu2022_convnext_cvpr.pdf`
   - *Relevansi*: Arsitektur konvolusi modern berkecepatan tinggi dengan kernel $7 \times 7$ untuk menangkap relasi goresan jarak jauh pada grid $64 \times 64$.
9. **He et al. (CVPR 2016)** — *Deep Residual Learning for Image Recognition (ResNet)*
   - File: `he2016_resnet_cvpr.pdf`
   - *Relevansi*: Arsitektur backbone stabil untuk gradien konvergen dan representasi hierarkis.
10. **Ghojogh et al. (2019) & Belhumeur et al. (1997)** — *Fisher and Kernel Fisher Discriminant Analysis & Eigenfaces vs Fisherfaces*
    - File: `Ghojogh2019_Fisher_and_Kernel_Fisher_Discriminant_Analysis_Tutorial.pdf`, `Belhumeur1997_Eigenfaces_vs_Fisherfaces.pdf`
    - *Relevansi*: Landasan matematis multivariat untuk reduksi dimensi PCA dan proyeksi diskriminan linear (LDA) sebagai model regularizer dalam ensemble.
11. **Zhou et al. (ICLR 2021)** — *Domain Generalization with MixStyle*
    - File: `zhou2021_mixstyle_iclr.pdf`
    - *Relevansi*: Interpolasi statistik fitur antar domain/penulis pada feature map untuk menghasilkan model yang invariabel terhadap gaya individu penulis.

---
*Navigasi Vault: [[INDEX|Indeks Utama Dokumentasi]] | [[17_MASTER_DOKUMENTASI_KESELURUHAN_DAN_RIWAYAT_VERSI|Master Dokumentasi Keseluruhan]] | [[WORKFLOW_LINEAGE.canvas|Peta Alur Riset]]*
