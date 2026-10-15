# Chinese Handwritten Character Recognition (Chinese MNIST 15 Classes)
## Proyek Akhir Analisis Data Multivariat (ADM) — Departemen Teknik Informatika ITS
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![Kaggle GPU](https://img.shields.io/badge/Kaggle-Tesla%20T4-20BEFF.svg)](https://www.kaggle.com/)
[![OOF Accuracy](https://img.shields.io/badge/OOF%20Accuracy-99.79%25-brightgreen.svg)]()
[![Ensemble Accuracy](https://img.shields.io/badge/Ensemble%20%2B%20TTA-99.914%25-success.svg)]()

Repository proyek akhir mata kuliah **Analisis Data Multivariat (ADM)** yang mengintegrasikan metode **Statistika Multivariat Klasik** (PCA, MANOVA, Fisher LDA, Regularized QDA) dengan arsitektur **State-of-the-Art (SOTA) Deep Learning** (ResNet18-CBAM, ConvNeXt-Femto, Sub-Center ArcFace, MixStyle) untuk klasifikasi 15 karakter angka tulisan tangan aksara Mandarin (*Chinese MNIST*) pada kompetisi CodaBench.

---

## 📊 Ringkasan Hasil & Benchmark Performa

### 1. Perbandingan Lintas Pendekatan (Klasik vs SOTA Deep Learning)
Seluruh evaluasi dilakukan dengan protokol **5-Fold StratifiedGroupKFold pada 70 penulis unik (*unseen writers*)** tanpa kebocoran data (*zero writer leakage*):

| Metodologi | Model / Algoritma | Dimensi Fitur | Akurasi OOF (%) | Catatan Metodologis |
| :--- | :--- | :---: | :---: | :--- |
| **Batas Manusia (SOTA)** | Human Bound (CASIA-HWDB 2013) | - | 96,13% | Batas persepsi visual manusia pada karakter Mandarin |
| **Multivariat Klasik** | Nearest Centroid (Euclidean) | 4.096 | 80,33% | Asumsi kovarians identitas sferis $\mathbf{I}_p$ |
| **Multivariat Klasik** | Two-Stage PCA (100) + Fisher LDA | 100 | 85,13% | Terikat asumsi kovarians homogen ($\boldsymbol{\Sigma}_k = \boldsymbol{\Sigma}$) |
| **Multivariat Klasik** | Two-Stage PCA (100) + Logistic Reg. | 100 | 91,08% | Diskriminan linear non-parametrik |
| **Multivariat Klasik** | **Two-Stage PCA (40) + Regularized QDA** | 40 | **96,09%** | **Unggul +10,96% vs LDA** (menangkap heterogenitas kovarians) |
| **Deep Learning** | ConvNeXt-Femto + SubCenter ArcFace | 256 | **99,55%** | Arsitektur modern $7 \times 7$ depthwise conv + LayerNorm |
| **Deep Learning** | **ResNet18-CBAM + MixStyle (Single Fold)** | 256 | **99,79%** | **Top-Tier SOTA** (Hanya 22 error dari 10.500 citra) |
| **Deep Learning** | **5-Fold Ensemble + 5-Way TTA Blended** | 256 | **99,914%** | **Hanya 9 error dari 10.500 citra** (1 di antaranya label noise) |
| **Dual-Form Hanzi** | Traditional Variants ('萬' & '億') | 256 | **100,00%** | Sempurna 700/700 citra sintetik benar |

---

## 🔬 Temuan Kunci Analisis Multivariat Klasik (Jalur Akademik ADM)

1. **Reduksi Dimensi PCA (Eigen-characters):**
   - Nilai eigen rata-rata (Kaiser Threshold): $0,087497$.
   - Kriteria Kaiser mempertahankan **275 Principal Components**.
   - Proporsi Variansi Kumulatif (CPVE):
     - $80\%$ variansi dijelaskan oleh **82 PC**
     - $85\%$ variansi dijelaskan oleh **109 PC**
     - $90\%$ variansi dijelaskan oleh **154 PC**
     - $95\%$ variansi dijelaskan oleh **252 PC**
2. **Uji Hipotesis MANOVA (Top 20 PC):**
   - **Wilks' Lambda ($\Lambda$):** $8,89 \times 10^{-5}$ ($F \approx 184,3, p < 10^{-15}$), membuktikan perbedaan vektor nilai tengah 15 karakter sangat signifikan secara statistik.
   - **Pillai's Trace ($V$):** $5,8418$ ($p < 10^{-15}$), uji paling robust terhadap pelanggaran multivariat.
   - **Partial $\eta_p^2$:** $0,4864$ (48,64% variabilitas fitur laten dijelaskan oleh kelas karakter).
3. **Justifikasi Teoretis Keunggulan QDA atas LDA:**
   - Uji Box's M ($p < 0,001$) membuktikan matriks kovarians antarkelas **heterogen secara signifikan** ($\boldsymbol{\Sigma}_i \neq \boldsymbol{\Sigma}_j$).
   - Fisher LDA memaksakan *pooled covariance* $\boldsymbol{\Sigma}$, sedangkan Regularized QDA mengestimasi kovarians unik per kelas, sehingga akurasinya melonjak dari $85,13\%$ ke **$96,09\%$**.

---

## 🛡️ Uji Ketahanan Model (Stress-Testing 8 Skenario)

Evaluasi ketahanan terhadap manipulasi citra dan catatan panitia CodaBench (*"goresan tebal + blur ringan"*):

| Skenario Pengujian | Parameter Gangguan | ResNet18 Ens (%) | ResNet + TTA (%) | ConvNeXt Ens (%) | Status Ketahanan |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Clean Baseline** | Citra asli tanpa gangguan | 99,64% | 99,64% | 99,64% | Standar acuan |
| **Mild Blur** | Gaussian $k=3, \sigma=0,8$ | 99,64% | 99,64% | 99,64% | **Kebal Mutlak (0% Degradasi)** |
| **Heavy Blur** | Gaussian $k=7, \sigma=1,8$ | 99,64% | 99,64% | 99,64% | **Kebal Mutlak (0% Degradasi)** |
| **Stroke Dilation** | Penebalan $2 \times 2$ | 99,64% | 99,64% | 99,64% | **Kebal Mutlak (0% Degradasi)** |
| **Heavy Dilation** | Penebalan $3 \times 3$ | 99,64% | 99,64% | 99,64% | **Kebal Mutlak (0% Degradasi)** |
| **Stroke Erosion** | Penipisan $2 \times 2$ | **100,00%** | **100,00%** | 99,64% | **ResNet Sempurna 100%** |
| **Committee Compound** | **Penebalan + Blur (Panitia)** | **99,64%** | **99,64%** | **99,64%** | **Simulasi Panitia: KEBAL** |
| **Additive Noise** | Gaussian $\mathcal{N}(0, 15)$ | **100,00%** | **100,00%** | 99,64% | **ResNet Sempurna 100%** |
| **Ink Specks** | 40 noda bintik tinta | **97,09%** | **96,36%** | 93,09% | **ResNet Lebih Tahan (+4,00%)** |

---

## 📁 Struktur Direktori Proyek

```text
FinalProject/
├── README.md                                    # Dokumentasi utama proyek akhir
├── requirements.txt                             # Dependensi Python
├── classes.csv                                  # Metadata 15 kelas karakter angka Mandarin
├── train_labels.csv                             # Label 10.500 citra latih asli
├── splits_kfold.csv                             # Split 5-Fold bebas kebocoran (StratifiedGroupKFold)
├── train_synthetic_labels.csv                   # Label 700 citra sintetik tradisional ('萬' & '億')
│
├── src/                                         # Kode sumber pipeline
│   ├── models.py                                # Arsitektur ResNet18-CBAM, ConvNeXt-Femto, SubCenterArcFace
│   ├── split_data.py                            # Generator split 5-fold bebas kebocoran penulis
│   ├── generate_traditional_data.py             # Generator citra sintetik '萬' & '億' (Elastic Deformation)
│   ├── multivariate_analysis.py                 # Pipeline PCA, MANOVA, LDA, QDA, Box's M, & Scree Plot
│   ├── train_cnn.py                             # Pipeline training neural network lokal
│   ├── prepare_submission.py                    # Script inferensi CodaBench dengan AdaBN & 5-Way TTA
│   ├── run_phase1_pipeline.py                   # Runner otomatisasi satu perintah untuk Fase 1
│   ├── stress_test_blur_dilation.py             # Uji cepat ketahanan blur & dilatasi
│   ├── comprehensive_stress_test.py             # Uji ketahanan komprehensif 8 skenario degradasi
│   └── compare_stress_test_architectures.py     # Uji head-to-head ResNet vs ConvNeXt (ter-vektorisasi)
│
├── dokumentasi/                                 # Laporan riset & audit akademis mendalam
│   ├── 00_PRA_FASE.md                           # Fondasi, mitigasi data hilang, & ringkasan pra-fase
│   ├── 01_ERROR_ANALYSIS_CONFUSION_MATRIX_REPORT.md # Bedah morfologi 22 sampel salah & pemulihan TTA
│   ├── 02_STRESS_TEST_ROBUSTNESS_REPORT.md      # Laporan empiris uji ketahanan 8 degradasi
│   ├── 03_ARCHITECTURAL_COMPARISON_RESNET_VS_CONVNEXT.md # Studi komparasi ResNet vs ConvNeXt
│   ├── LITERATURE_BENCHMARK_ACADEMIC_RIGOR_EVALUATION.md # Benchmark CASIA-HWDB & uji asumsi MANOVA
│   ├── AUDIT_KERENTANAN_FASE1_FASE2.md          # Audit blind spots CodaBench & Action Checklist
│   └── ROADMAP_MENUNGGU_FASE_1.md               # Rencana kerja sprint fase 1 & 2
│
├── outputs/                                     # Visualisasi grafik & metrik JSON
│   ├── scree_plot.png                           # Grafik Scree Plot & Kaiser Threshold PCA
│   ├── eigen_characters.png                     # Visualisasi 20 Eigen-characters teratas
│   ├── lda_canonical_projection.png             # Proyeksi kanonikal 2D Fisher LDA
│   ├── mahalanobis_qq_plot.png                  # Chi-Square Q-Q plot jarak Mahalanobis
│   ├── head_to_head_stress_test.json            # Metrik evaluasi komparatif ResNet vs ConvNeXt
│   └── stress_test_metrics.json                 # Metrik uji ketahanan 8 skenario
│
├── kaggle_kernel/                               # Kernel pelatihan Kaggle Cloud (ResNet18-CBAM)
│   ├── kernel-metadata.json
│   └── mnist-adm.py
│
├── kaggle_kernel_convnext/                      # Kernel pelatihan Kaggle Cloud (ConvNeXt-Femto)
│   ├── kernel-metadata.json
│   └── mnist-adm-convnext.py
│
├── kaggle_dataset/                              # Metadata dataset Kaggle (reiiihel/chinese-mnist-adm)
│   └── dataset-metadata.json
│
└── references/                                  # Koleksi 30+ paper ilmiah bereputasi tinggi (TPAMI, CVPR, PR)
```

---

## 🚀 Panduan Eksekusi & Reproduksi

### 1. Instalasi Environment
```bash
cd FinalProject
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Menjalankan Analisis Multivariat Klasik (PCA, MANOVA, LDA, QDA)
```bash
python src/multivariate_analysis.py
# Grafik otomatis tersimpan di folder outputs/
```

### 3. Menguji Ketahanan Model (Stress-Testing Komparatif)
```bash
python src/compare_stress_test_architectures.py
```

### 4. Eksekusi Otomatisasi Fase 1 CodaBench (15 Oktober)
Setelah data validasi diekstrak ke `FinalProject/data_validasi/`, jalankan:
```bash
python src/run_phase1_pipeline.py
# Otomatis menghasilkan 2 arsip: submission_phase1_anchor.zip & submission_phase1_adabn_tta.zip
```

### 5. Mengunduh Bobot Model dari Kaggle Cloud (Opsional)
Bobot model hasil pelatihan cloud dapat diunduh langsung via Kaggle CLI:
```bash
# Model Utama (ResNet18-CBAM 5-Fold, 99.79% OOF)
kaggle kernels output reiiihel/mnist-adm -p models/kaggle_outputs

# Model Pembanding (ConvNeXt-Femto 5-Fold, 99.55% OOF)
kaggle kernels output reiiihel/mnist-adm-convnext -p models/kaggle_outputs_convnext
```

---

## 📚 Publikasi Rujukan Utama
1. **Deng et al. (ECCV 2020):** *Sub-center ArcFace: Boosting Face Recognition by Large-Scale Noisy Web Faces.*
2. **Zhou et al. (ICLR 2021):** *Domain Generalization with MixStyle.*
3. **Woo et al. (ECCV 2018):** *CBAM: Convolutional Block Attention Module.*
4. **Liu et al. (CVPR 2022):** *A ConvNet for the 2020s (ConvNeXt).*
5. **Zhang, Bengio, Liu (IEEE TPAMI 2017):** *Online and Offline Handwritten Chinese Character Recognition: A Comprehensive Study and New Benchmark.*
6. **Li et al. (Pattern Recognition 2018):** *Adaptive Batch Normalization for practical domain adaptation.*
7. **Belhumeur et al. (IEEE TPAMI 1997):** *Eigenfaces vs. Fisherfaces: Recognition Using Class Specific Linear Projection.*
8. **Johnson & Wichern (Pearson 2007):** *Applied Multivariate Statistical Analysis (6th Edition).*
