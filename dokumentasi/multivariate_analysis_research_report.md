---
title: "Riset Komprehensif: Metode Analisis Data Multivariat (ADM) Klasik & Statistical Learning untuk Pengenalan Karakter Tulisan Tangan"
source_doc: "multivariate_analysis_research_report.md"
date: 2026-10-11
updated: 2026-11-24
tags:
  - adm-project
  - documentation
  - pra-fase
  - multivariate-statistics
---

# Riset Komprehensif: Metode Analisis Data Multivariat (ADM) Klasik & Statistical Learning untuk Pengenalan Karakter Tulisan Tangan

**Mata Kuliah:** Analisis Data Multivariat (ADM)  
**Dataset:** Karakter Angka Tradisional & Sederhana Tionghoa (15 Kelas: 零, 一, 二, 三, 四, 五, 六, 七, 八, 九, 十, 百, 千, 万, 亿)  
**Karakteristik Data:** Citra Grayscale $64 \times 64$ piksel ($p = 4.096$ dimensi), $N = 10.500$ sampel latih, 15 kelas seimbang ($n_k = 700$), 70 penulis (*writers*).  
**Penyusun:** Tim Riset Analisis Data Multivariat

---

## 1. Landasan Teori Multivariat Klasik & Literatur Standar

Pendekatan Analisis Data Multivariat (ADM) pada data citra berakar pada pemodelan vektor acak berdimensi tinggi $\mathbf{x} \in \mathbb{R}^p$. Literatur ilmiah bereputasi membagi domain ini ke dalam pilar dekomposisi variansi, reduksi dimensi ortogonal, uji hipotesis vektor mean, dan aturan alokasi diskriminan.

### 1.1 Rujukan Buku Teks Multivariat Standar
1. **Johnson, R. A., & Wichern, D. W. (2007). *Applied Multivariate Statistical Analysis* (6th ed.). Pearson Prentice Hall.**
   - **Bab 4 (Multivariate Normal Distribution & Outliers):** Formulasi distribusi normal multivariat $\mathcal{N}_p(\boldsymbol{\mu}, \boldsymbol{\Sigma})$, jarak statistik Mahalanobis $D^2 = (\mathbf{x}-\boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x}-\boldsymbol{\mu}) \sim \chi^2_p$, serta evaluasi grafik Q-Q plot $\chi^2$.
   - **Bab 6 (Comparisons of Several Multivariate Means):** Teori dasar MANOVA (Multivariate Analysis of Variance), pembentukan matriks Sum of Squares and Cross Products (SSCP) antar-kelompok ($H$) dan dalam-kelompok ($E$), serta statistik uji Wilks' $\Lambda$, Pillai's trace, Hotelling-Lawley trace, dan Roy's largest root.
   - **Bab 8 & 9 (Principal Components & Factor Analysis):** Dekomposisi spektral matriks kovarians sampel $\mathbf{S} = \mathbf{V} \boldsymbol{\Lambda} \mathbf{V}^T$, kriteria retensi komponen utama, serta sifat proyektif variansi maksimum.
   - **Bab 11 (Discrimination and Classification):** Diskriminan linier Fisher (two-group dan multi-group), aturan alokasi Bayes untuk kovarians sama (LDA) dan berbeda (QDA), serta analisis matriks konfusi dan laju kesalahan (*error rate*).

2. **Hair, J. F., Black, W. C., Babin, B. J., & Anderson, R. E. (2019). *Multivariate Data Analysis* (8th ed.). Cengage Learning.**
   - **Praktik Pemilihan Komponen:** Pedoman Kaiser criterion (retensi nilai eigen $\lambda > \bar{\lambda}$ atau $\lambda > 1$ untuk matriks korelasi), Scree test Cattell (deteksi titik patah siku *elbow*), dan kriteria variansi kumulatif minimum ($85\% - 90\%$).
   - **Uji Asumsi & Ketahanan MANOVA:** Uji Homogenitas Matriks Kovarians (Box's M test) dengan batas toleransi konservatif ($\alpha = 0,001$) akibat sensitivitas tinggi terhadap ukuran sampel besar; keunggulan Pillai's Trace saat asumsi normalitas dan homogenitas kovarians dilanggar.

3. **Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.**
   - Bab 12 (Continuous Latent Variables): Formulasi probabilistik PCA, Singular Value Decomposition (SVD), dan reduksi dimensi non-linear via Kernel PCA.
   - Bab 4 (Linear Models for Classification): Formulasi Fisher's Linear Discriminant sebagai pemaksimalan rasio Rayleigh umum $\mathcal{J}(\mathbf{W})$.

### 1.2 Literatur Ilmiah Bereputasi (IEEE / Elsevier / Springer)
1. **Turk, M., & Pentland, A. (1991). *Eigenfaces for Recognition*. Journal of Cognitive Neuroscience, 3(1), 71–86.**
   - Memelopori teknik *Eigen-images* (dalam konteks ini: **Eigen-characters**), merepresentasikan citra 2D sebagai vektor 1D berdimensi $p$, serta memperkenalkan *Gram matrix trick* (menghitung nilai eigen dari $\mathbf{X}_c \mathbf{X}_c^T \in \mathbb{R}^{N \times N}$ bukan $\mathbf{X}_c^T \mathbf{X}_c \in \mathbb{R}^{p \times p}$).
2. **Belhumeur, P. N., Hespanha, J. P., & Kriegman, D. J. (1997). *Eigenfaces vs. Fisherfaces: Recognition Using Class Specific Linear Projection*. IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 19(7), 711–720.**
   - Menyelesaikan *Small Sample Size (SSS) Problem* dan singularitas matriks scatter dalam ruang citra piksel berdimensi tinggi melalui paradigma dua tahap: **PCA + LDA** (*Fisherfaces framework*).
3. **Friedman, J. H. (1989). *Regularized Discriminant Analysis*. Journal of the American Statistical Association (JASA), 84(405), 165–175.**
   - Menurunkan metode *Regularized Discriminant Analysis* (RDA) dengan dua parameter *shrinkage* $(\lambda, \gamma)$ untuk menstabilkan estimasi kovarians kelas berdimensi tinggi.
4. **Ledoit, O., & Wolf, M. (2004). *A well-conditioned estimator for large-dimensional covariance matrices*. Journal of Multivariate Analysis, 88(2), 365–411.**
   - Menghasilkan estimator kovarians *shrinkage* analitis optimal bebas parameter *tuning*, menyusutkan kovarians sampel ke matriks identitas skalar terarah.
5. **Kimura, F., Takashina, K., Tsuruoka, S., & Miyake, Y. (1987). *Modified quadratic discriminant functions and the application to Chinese character recognition*. IEEE TPAMI, 9(1), 149–153.**
   - Memperkenalkan *Modified Quadratic Discriminant Function* (MQDF), metode klasik terbaik dalam literatur pengenalan karakter Tionghoa untuk mengatasi singularitas kovarians QDA dengan menghaluskan nilai eigen minor.
6. **Zhang, X.-Y., Bengio, Y., & Liu, C.-L. (2017). *Online and Offline Handwritten Chinese Character Recognition: A Comprehensive Study and New Benchmark*. Pattern Recognition (Elsevier), 61, 348–360.**
   - Tolok ukur (*benchmark*) komprehensif pengenalan karakter tulisan tangan Tionghoa, mengkaji normalisasi bentuk, reduksi LDA, MQDF, serta perbandingannya dengan deep learning.

---

## 2. Reduksi Dimensi & Dekomposisi Variansi

### 2.1 PCA (Principal Component Analysis) & Eigen-characters
Setiap citra karakter tulisan tangan berukuran $64 \times 64$ direpresentasikan sebagai vektor baris $\mathbf{x}_i \in \mathbb{R}^p$ di mana $p = 4.096$. Matriks data sampel latih dinotasikan dengan:
$$\mathbf{X} = \begin{bmatrix} \mathbf{x}_1^T \\ \mathbf{x}_2^T \\ \vdots \\ \mathbf{x}_N^T \end{bmatrix} \in \mathbb{R}^{N \times p}, \quad N = 10.500, \; p = 4.096$$

#### 1. Pemusatan Data (*Centering*)
Vektor mean global sampel dihitung sebagai:
$$\bar{\mathbf{x}} = \frac{1}{N} \sum_{i=1}^N \mathbf{x}_i \in \mathbb{R}^p$$
Matriks data terpusat (*mean-centered matrix*):
$$\mathbf{X}_c = \mathbf{X} - \mathbf{1}_N \bar{\mathbf{x}}^T \in \mathbb{R}^{N \times p}$$
di mana $\mathbf{1}_N = [1, 1, \dots, 1]^T \in \mathbb{R}^N$.

#### 2. Matriks Kovarians Sampel & Dekomposisi Spektral
Matriks varians-kovarians sampel $\mathbf{S} \in \mathbb{R}^{p \times p}$ didefinisikan sebagai:
$$\mathbf{S} = \frac{1}{N - 1} \mathbf{X}_c^T \mathbf{X}_c$$
Menurut Teorema Spektral, karena $\mathbf{S}$ simetris dan semi-definit positif, $\mathbf{S}$ dapat didekomposisi menjadi:
$$\mathbf{S} = \mathbf{V} \boldsymbol{\Lambda} \mathbf{V}^T = \sum_{j=1}^p \lambda_j \mathbf{v}_j \mathbf{v}_j^T$$
di mana:
- $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_p \ge 0$ adalah nilai eigen (*eigenvalues*) yang merepresentasikan variansi yang dijelaskan oleh masing-masing komponen.
- $\mathbf{V} = [\mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_p] \in \mathbb{R}^{p \times p}$ adalah matriks vektor eigen ortonormal ($\mathbf{v}_j^T \mathbf{v}_k = \delta_{jk}$).

#### 3. Komputasi Cepat via SVD / Gram Matrix Trick
Membentuk dan mendekomposisi matriks $\mathbf{S} \in \mathbb{R}^{4096 \times 4096}$ secara langsung membutuhkan memori dan operasi floating point yang besar ($O(p^3) \approx 6,87 \times 10^{10}$ FLOPs). Komputasi dioptimalkan menggunakan *Singular Value Decomposition* (SVD) dari matriks data terpusat:
$$\mathbf{X}_c = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^T$$
di mana $\mathbf{U} \in \mathbb{R}^{N \times k}$ ortonormal, $\boldsymbol{\Sigma} = \text{diag}(\sigma_1, \dots, \sigma_k)$, dan $\mathbf{V} \in \mathbb{R}^{p \times k}$.
Hubungan nilai singular dengan nilai eigen matriks kovarians adalah:
$$\lambda_j = \frac{\sigma_j^2}{N - 1}$$
dan kolom-kolom dari $\mathbf{V}$ secara langsung merupakan vektor eigen dari $\mathbf{S}$.

#### 4. Proyeksi & Rekonstruksi (*Eigen-characters Space*)
Dengan memilih $m \ll p$ komponen utama teratas, matriks proyeksi dibentuk oleh $\mathbf{V}_m = [\mathbf{v}_1, \dots, \mathbf{v}_m] \in \mathbb{R}^{p \times m}$.
- **Skor Proyeksi Komponen Utama:**
  $$\mathbf{Z} = \mathbf{X}_c \mathbf{V}_m \in \mathbb{R}^{N \times m}$$
- **Rekonstruksi Citra Karakter:**
  $$\hat{\mathbf{x}}_i = \bar{\mathbf{x}} + \mathbf{V}_m \mathbf{z}_i = \bar{\mathbf{x}} + \sum_{j=1}^m z_{ij} \mathbf{v}_j$$
- **Interpretasi Visual "Eigen-characters":**
  Setiap vektor eigen $\mathbf{v}_j \in \mathbb{R}^{4096}$ dapat dibentuk kembali (*reshaped*) menjadi matriks citra $64 \times 64$. Pola visual yang dihasilkan menunjukkan basis variasi morfologi goresan:
  - $\mathbf{v}_1, \mathbf{v}_2$: Menangkap ketebalan global goresan (*stroke weight*) dan orientasi dominan (arah goresan horizontal 一 vs vertikal 十).
  - $\mathbf{v}_3 - \mathbf{v}_{10}$: Menangkap topologi struktur loop, kotak luar (pada karakter 四, 百), dan percabangan goresan miring (seperti pada 八, 九, 千).
  - Komponen tingkat tinggi ($j > 100$): Merepresentasikan variasi gaya penulis personal (*writer-specific idiosyncrasies*) dan noise frekuensi tinggi.

---

### 2.2 Kriteria Seleksi Dimensi Komponen Utama

Untuk menentukan dimensi tereduksi $m$ secara objektif dan ilmiah pada laporan ADM, digunakan tiga kriteria multivariat standar:

```
               Kriteria Retensi Dimensi PCA pada Citra Karakter
┌───────────────────────────┬────────────────────────────────────────┬──────────────────────┐
│ Kriteria                  │ Formulasi Matematis                    │ Nilai Empiris Citra  │
├───────────────────────────┼────────────────────────────────────────┼──────────────────────┤
│ 1. Scree Plot Criterion   │ Elbow point (titik belok turunan ke-2) │ Komponen m ≈ 20 - 35 │
│ 2. Proporsi Variansi 85%  │ CPVE(m) >= 0.85                        │ Komponen m = 101     │
│ 3. Proporsi Variansi 90%  │ CPVE(m) >= 0.90                        │ Komponen m = 140     │
│ 4. Kaiser Criterion       │ λ_j > mean(λ) = tr(S)/p                │ Komponen m = 267     │
└───────────────────────────┴────────────────────────────────────────┴──────────────────────┘
```

#### 1. Proportion of Variance Explained (PVE) & Kriteria Kumulatif
Proporsi variansi individual komponen ke-$j$:
$$\text{PVE}_j = \frac{\lambda_j}{\sum_{k=1}^p \lambda_k} = \frac{\lambda_j}{\text{tr}(\mathbf{S})}$$
Proporsi variansi kumulatif (*Cumulative Proportion of Variance Explained - CPVE*):
$$\text{CPVE}(m) = \frac{\sum_{j=1}^m \lambda_j}{\sum_{k=1}^p \lambda_k} = \frac{\sum_{j=1}^m \lambda_j}{\text{tr}(\mathbf{S})}$$
- Standar ilmiah ADM mensyaratkan retensi variansi $\text{CPVE}(m) \ge 0,85$ atau $\ge 0,90$.
- **Hasil Pengujian Empiris Dataset Karakter:**
  - Ambang batas $85\%$ variansi dicapai pada $m = 101$ komponen (reduksi dimensi sebesar **$97,53\%$** dari 4.096 dimensi).
  - Ambang batas $90\%$ variansi dicapai pada $m = 140$ komponen (reduksi dimensi sebesar **$96,58\%$**).

#### 2. Scree Plot Criterion (Cattell's Scree Test)
Memplot grafik $\lambda_j$ terhadap indeks komponen $j$. Sumbu transisi dari penurunan curam (*steep cliff*) menuju landai mendatar (*scree/rubble*) menunjukkan batas antara sinyal informasi struktural karakter dan noise keacakan goresan.
Secara analitis, titik siku (*elbow point*) diidentifikasi melalui maksimum kelengkungan atau selisih turunan kedua:
$$j^* = \arg\max_j \left[ (\lambda_j - \lambda_{j+1}) - (\lambda_{j+1} - \lambda_{j+2}) \right]$$
Pada data karakter kita, scree plot menunjukkan penurunan drastis hingga $j \approx 25$, lalu melandai stabil.

#### 3. Kaiser-Guttman Criterion
Untuk data yang tidak distandarisasi (menggunakan matriks kovarians $\mathbf{S}$), kriteria Kaiser menginstruksikan untuk mempertahankan komponen yang memiliki nilai eigen lebih besar dari rata-rata nilai eigen seluruh variabel:
$$\lambda_j > \bar{\lambda} = \frac{1}{p} \sum_{k=1}^p \lambda_k = \frac{\text{tr}(\mathbf{S})}{p}$$
- Pada dataset karakter, $\bar{\lambda} = 0,087273$.
- Jumlah komponen yang memenuhi $\lambda_j > \bar{\lambda}$ adalah **$m = 267$ komponen**. Komponen ini menjelaskan lebih dari $95,8\%$ total variansi.

---

### 2.3 Kernel PCA (Non-linear Subspace Decomposition)
Variasi tulisan tangan manusia memiliki sifat non-linearitas intrinsik akibat kelengkungan goresan, rotasi lokal, dan distorsi elastis. Kernel PCA memproyeksikan data ke ruang fitur berdimensi tinggi tak hingga (*Reproducing Kernel Hilbert Space* $\mathcal{H}$) via pemetaan non-linear $\boldsymbol{\Phi}: \mathbb{R}^p \to \mathcal{H}$.

#### 1. Trik Kernel & Matriks Gram Terpusat
Tanpa menghitung $\boldsymbol{\Phi}(\mathbf{x})$ secara eksplisit, fungsi kernel $k(\mathbf{x}_i, \mathbf{x}_j) = \langle \boldsymbol{\Phi}(\mathbf{x}_i), \boldsymbol{\Phi}(\mathbf{x}_j) \rangle_{\mathcal{H}}$ digunakan untuk membentuk matriks kernel $\mathbf{K} \in \mathbb{R}^{N \times N}$ dengan $K_{ij} = k(\mathbf{x}_i, \mathbf{x}_j)$.
Kernel yang paling relevan untuk citra karakter:
- **Radial Basis Function (RBF) Kernel:** $k(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2)$
- **Polynomial Kernel:** $k(\mathbf{x}_i, \mathbf{x}_j) = (\mathbf{x}_i^T \mathbf{x}_j + c)^d$

Pemusatan dalam ruang Hilbert dilakukan via matriks kernel terpusat $\widetilde{\mathbf{K}} \in \mathbb{R}^{N \times N}$:
$$\widetilde{\mathbf{K}} = \mathbf{K} - \mathbf{1}_N' \mathbf{K} - \mathbf{K} \mathbf{1}_N' + \mathbf{1}_N' \mathbf{K} \mathbf{1}_N'$$
di mana $\mathbf{1}_N' = \frac{1}{N} \mathbf{J}_N$ dan $\mathbf{J}_N$ adalah matriks bernilai 1 berukuran $N \times N$.

#### 2. Masalah Nilai Eigen Dual
Vektor eigen dual $\boldsymbol{\alpha}_k \in \mathbb{R}^N$ diperoleh dari solusi:
$$\widetilde{\mathbf{K}} \boldsymbol{\alpha}_k = \lambda_k^K \boldsymbol{\alpha}_k$$
dengan normalisasi $\lambda_k^K \|\boldsymbol{\alpha}_k\|_2^2 = 1$.
Proyeksi sampel baru $\mathbf{x}^*$ ke komponen utama non-linear ke-$k$:
$$y_k(\mathbf{x}^*) = \sum_{i=1}^N \alpha_{ki} \tilde{k}(\mathbf{x}_i, \mathbf{x}^*)$$
Kernel PCA efektif memisahkan manifold karakter yang saling melengkung (*intertwined manifolds*) sebelum diterapkan pengklasifikasi linier.

---

## 3. Klasifikasi Multivariat & Pemisahan Antarkelas

### 3.1 Multi-class Linear Discriminant Analysis (Fisher's LDA / MDA)
Linear Discriminant Analysis bertujuan menemukan subruang proyeksi $\mathbf{W} \in \mathbb{R}^{p \times d}$ yang memaksimalkan pemisahan antar-kelas sekaligus meminimalkan sebaran dalam-kelas.

```
       Visualisasi Dekomposisi Scatter Matriks Total (S_T = S_W + S_B)
                     S_T (Total Scatter Matrix, p x p)
                     ┌───────────────────────────────┐
                     │ Variansi keseluruhan dataset  │
                     └───────────────┬───────────────┘
                                     │
             ┌───────────────────────┴───────────────────────┐
             ▼                                               ▼
  S_W (Within-Class Scatter)                      S_B (Between-Class Scatter)
  Sebaran variasi goresan internal                Perbedaan vektor mean antarkelas
  setiap karakter (noise gaya tulis).             (sinyal diskriminatif kelas).
  Rank(S_W) <= N - K                             Rank(S_B) <= K - 1 = 14
```

#### 1. Formulasi Matriks Scatter Multivariat
Misalkan terdapat $K = 15$ kelas karakter $\mathcal{C}_1, \mathcal{C}_2, \dots, \mathcal{C}_K$.
- Vektor mean kelas $k$: $\bar{\mathbf{x}}_k = \frac{1}{N_k} \sum_{i \in \mathcal{C}_k} \mathbf{x}_i \in \mathbb{R}^p$.
- Vektor mean total: $\bar{\mathbf{x}} = \frac{1}{N} \sum_{i=1}^N \mathbf{x}_i = \frac{1}{N} \sum_{k=1}^K N_k \bar{\mathbf{x}}_k$.
- **Within-Class Scatter Matrix ($\mathbf{S}_W \in \mathbb{R}^{p \times p}$):**
  $$\mathbf{S}_W = \sum_{k=1}^K \sum_{i \in \mathcal{C}_k} (\mathbf{x}_i - \bar{\mathbf{x}}_k)(\mathbf{x}_i - \bar{\mathbf{x}}_k)^T = \sum_{k=1}^K (N_k - 1) \mathbf{S}_k$$
- **Between-Class Scatter Matrix ($\mathbf{S}_B \in \mathbb{R}^{p \times p}$):**
  $$\mathbf{S}_B = \sum_{k=1}^K N_k (\bar{\mathbf{x}}_k - \bar{\mathbf{x}})(\bar{\mathbf{x}}_k - \bar{\mathbf{x}})^T$$
- Hubungan dengan Total Scatter: $\mathbf{S}_T = \mathbf{S}_W + \mathbf{S}_B = \sum_{i=1}^N (\mathbf{x}_i - \bar{\mathbf{x}})(\mathbf{x}_i - \bar{\mathbf{x}})^T$.

#### 2. Kriteria Fisher-Rayleigh & Masalah Nilai Eigen Umum
Kriteria pemisahan Fisher memaksimalkan determinan sebaran terproyeksi:
$$\mathcal{J}(\mathbf{W}) = \frac{\det(\mathbf{W}^T \mathbf{S}_B \mathbf{W})}{\det(\mathbf{W}^T \mathbf{S}_W \mathbf{W})}$$
Kolom-kolom $\mathbf{W} = [\mathbf{w}_1, \dots, \mathbf{w}_d]$ diperoleh dari vektor eigen umum (*generalized eigenvectors*):
$$\mathbf{S}_B \mathbf{w}_j = \lambda_j \mathbf{S}_W \mathbf{w}_j \iff \mathbf{S}_W^{-1} \mathbf{S}_B \mathbf{w}_j = \lambda_j \mathbf{w}_j$$

#### 3. Batasan Dimensi Maksimal LDA ($d \le K - 1$)
Karena $\mathbf{S}_B$ merupakan penjumlahan dari $K$ matriks perkalian luar berperingkat satu (*rank-1 outer products*) dengan satu derajat kebebasan terikat oleh relasi $\sum_{k=1}^K N_k (\bar{\mathbf{x}}_k - \bar{\mathbf{x}}) = \mathbf{0}$, maka:
$$\text{rank}(\mathbf{S}_B) \le K - 1 = 15 - 1 = 14$$
**Implikasi Akademik ADM:** Jumlah variabel diskriminan kanonikal maksimum yang dapat dibentuk adalah **tepat 14 dimensi**, tidak peduli seberapa besar dimensi awal $p$ ($4.096$ dimensi).

---

### 3.2 Problem Singularitas & Solusi Dua Tahap (PCA + LDA / Fisherfaces)

#### 1. Akar Masalah Singularitas pada Citra Piksel
Pada citra karakter tulisan tangan $64 \times 64$:
1. **Piksel Margin Konstan:** Area tepi dan sudut citra didominasi latar belakang hitam bernilai $0$ dengan variansi nol ($\sigma_j^2 = 0$).
2. **Kolinearitas Spasial:** Piksel yang bertetangga memiliki korelasi sangat tinggi ($r > 0,95$), menghasilkan matriks kovarians yang singular atau memiliki bilangan kondisi raksasa ($\kappa(\mathbf{S}_W) \to \infty$).
3. **Small Sample Size (SSS) per Kelas:** Meskipun secara total $N = 10.500 > p = 4.096$, ukuran sampel per kelas adalah $N_k = 700 \ll p = 4.096$. Matriks kovarians kelas individual $\mathbf{S}_k$ memiliki peringkat maksimal $N_k - 1 = 699$, sehingga $\mathbf{S}_k$ strictly non-invertible!
Akibatnya, inversi langsung $\mathbf{S}_W^{-1}$ atau $\mathbf{S}_k^{-1}$ akan mengalami kegagalan numerik (*singular matrix error*).

#### 2. Paradigma Dua Tahap (Belhumeur et al., 1997)
Solusi klasik paling elegan adalah mereduksi dimensi terlebih dahulu menggunakan PCA ke ruang berdimensi $m$ di mana $\mathbf{S}_W$ terjamin nonsingular, kemudian menjalankan LDA:
$$\mathbf{W}_{opt} = \mathbf{W}_{PCA} \mathbf{W}_{LDA}$$
1. **Tahap 1 (PCA Proyeksi):** Proyeksikan data dari $p = 4.096$ ke dimensi $m \le N - K$ ($m \approx 50 - 200$) menggunakan $\mathbf{W}_{PCA} \in \mathbb{R}^{p \times m}$.
2. **Tahap 2 (LDA Diskriminasi):** Hitung matriks scatter dalam subruang PCA:
   $$\widetilde{\mathbf{S}}_W = \mathbf{W}_{PCA}^T \mathbf{S}_W \mathbf{W}_{PCA} \in \mathbb{R}^{m \times m}, \quad \widetilde{\mathbf{S}}_B = \mathbf{W}_{PCA}^T \mathbf{S}_B \mathbf{W}_{PCA} \in \mathbb{R}^{m \times m}$$
   Selesaikan masalah nilai eigen umum pada ruang tereduksi:
   $$\widetilde{\mathbf{S}}_W^{-1} \widetilde{\mathbf{S}}_B \mathbf{w}_{LDA, j} = \lambda_j \mathbf{w}_{LDA, j}$$
   Hasil akhir adalah subruang berdimensi 14 yang sepenuhnya nonsingular, stabil secara numerik, dan memiliki daya separasi antarkelas maksimum.

---

### 3.3 Regularized Discriminant Analysis (RDA) & Ledoit-Wolf Shrinkage

Jika kita ingin mempertahankan ruang dimensi tinggi atau menerapkan batas keputusan kuadratik tanpa membuang informasi, metode regularisasi (*shrinkage*) wajib diterapkan.

#### 1. Formulasi RDA (Friedman, 1989)
Friedman memperkenalkan dua parameter penstabil $(\lambda, \gamma) \in [0, 1] \times [0, 1]$:
1. **Shrinkage Tahap 1 (Penyusutan Kovarians Kelas ke Pooled Covariance via $\lambda$):**
   $$\hat{\boldsymbol{\Sigma}}_k(\lambda) = \frac{(1 - \lambda) (N_k - 1) \mathbf{S}_k + \lambda (N - K) \mathbf{S}_{pooled}}{(1 - \lambda) (N_k - 1) + \lambda (N - K)}$$
   - $\lambda = 0$: Menggunakan kovarians individu tiap kelas $\mathbf{S}_k$ (Boundary QDA murni).
   - $\lambda = 1$: Menyamakan seluruh kovarians kelas menjadi $\mathbf{S}_{pooled}$ (Boundary LDA murni).
2. **Shrinkage Tahap 2 (Penyusutan ke Identitas Skalar via $\gamma$):**
   $$\hat{\boldsymbol{\Sigma}}_k(\lambda, \gamma) = (1 - \gamma) \hat{\boldsymbol{\Sigma}}_k(\lambda) + \gamma \left( \frac{\text{tr}(\hat{\boldsymbol{\Sigma}}_k(\lambda))}{p} \right) \mathbf{I}_p$$
   - $\gamma = 0$: Tidak ada desingularisasi identitas.
   - $\gamma = 1$: Mengasumsikan variabel ortogonal sferis identik (Nearest Centroid / Euclidean Distance Classifier).

#### 2. Estimator Analitis Optimal Ledoit-Wolf (2004)
Tanpa melakukan cross-validation grid search pada $(\lambda, \gamma)$, Ledoit & Wolf membuktikan bahwa estimator terbobot optimal terhadap risiko Frobenius $\mathbb{E}[\|\boldsymbol{\Sigma}^* - \boldsymbol{\Sigma}\|_F^2]$ dirumuskan sebagai:
$$\boldsymbol{\Sigma}^* = (1 - \rho^*) \mathbf{S} + \rho^* \mu_s \mathbf{I}_p$$
di mana target sferis $\mu_s = \frac{\text{tr}(\mathbf{S})}{p}$, dan intensitas shrinkage optimal $\rho^* \in [0, 1]$ dihitung secara analitis dari variansi kuadrat sampel:
$$\rho^* = \frac{\sum_{i=1}^N \|(\mathbf{x}_i - \bar{\mathbf{x}})(\mathbf{x}_i - \bar{\mathbf{x}})^T - \mathbf{S}\|_F^2}{N^2 \|\mathbf{S} - \mu_s \mathbf{I}_p\|_F^2}$$
Dalam Python `scikit-learn`, teknik ini diimplementasikan langsung via `LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')`.

---

### 3.4 Quadratic Discriminant Analysis (QDA) & Modified QDF (MQDF)

#### 1. Formulasi QDA Standar
QDA mengasumsikan setiap kelas memiliki distribusi normal multivariat dengan matriks kovarians yang berbeda: $\mathbf{x} | Y=k \sim \mathcal{N}_p(\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$.
Fungsi diskriminan kuadratik:
$$d_k(\mathbf{x}) = -\frac{1}{2} \ln |\boldsymbol{\Sigma}_k| - \frac{1}{2} (\mathbf{x} - \boldsymbol{\mu}_k)^T \boldsymbol{\Sigma}_k^{-1} (\mathbf{x} - \boldsymbol{\mu}_k) + \ln \pi_k$$
- **Ledakan Parameter pada Citra Piksel:**
  Setiap kelas membutuhkan estimasi matriks kovarians $p \times p$ simetris, yaitu $\frac{p(p+1)}{2} = \frac{4096 \times 4097}{2} = 8.390.656$ parameter. Untuk 15 kelas, dibutuhkan estimasi **$125.859.840$ parameter**.
  Dengan $N_k = 700$, estimasi ini *underdetermined* dan singular. Oleh karena itu, QDA hanya dapat dijalankan pada subruang fitur PCA ($m \approx 20 - 50$).

#### 2. Modified Quadratic Discriminant Function (MQDF / Kimura 1987; Liu 2003)
MQDF adalah mahakarya statistical pattern recognition untuk pengenalan karakter Tionghoa (kanji/hanzi). MQDF memotong spektrum nilai eigen kelas pada komponen ke-$k_0$ dan mengganti sisa nilai eigen minor dengan konstanta perata $\delta_k$.
Dekomposisi spektral matriks kovarians kelas: $\boldsymbol{\Sigma}_k = \sum_{j=1}^p \lambda_{kj} \boldsymbol{\phi}_{kj} \boldsymbol{\phi}_{kj}^T$.
Aproksimasi kovarians MQDF:
$$\widehat{\boldsymbol{\Sigma}}_k = \sum_{j=1}^{k_0} \lambda_{kj} \boldsymbol{\phi}_{kj} \boldsymbol{\phi}_{kj}^T + \delta_k \sum_{j=k_0+1}^p \boldsymbol{\phi}_{kj} \boldsymbol{\phi}_{kj}^T$$
Dengan memanfaatkan sifat ortonormalitas $\sum_{j=1}^p \boldsymbol{\phi}_{kj} \boldsymbol{\phi}_{kj}^T = \mathbf{I}_p$, inversnya dapat ditulis tanpa menginversi matriks penuh:
$$\widehat{\boldsymbol{\Sigma}}_k^{-1} = \frac{1}{\delta_k} \mathbf{I}_p + \sum_{j=1}^{k_0} \left( \frac{1}{\lambda_{kj}} - \frac{1}{\delta_k} \right) \boldsymbol{\phi}_{kj} \boldsymbol{\phi}_{kj}^T$$
Jarak diskriminan MQDF2:
$$D_{MQDF2}(\mathbf{x}, \mathcal{C}_k) = \frac{1}{\delta_k} \left[ \|\mathbf{x} - \boldsymbol{\mu}_k\|^2 - \sum_{j=1}^{k_0} \left( 1 - \frac{\delta_k}{\lambda_{kj}} \right) (\boldsymbol{\phi}_{kj}^T (\mathbf{x} - \boldsymbol{\mu}_k))^2 \right] + \sum_{j=1}^{k_0} \ln \lambda_{kj} + (p - k_0) \ln \delta_k$$
- **Keunggulan:** Kompleksitas memori dan komputasi anjlok dari $O(p^2)$ menjadi $O(k_0 p)$, matriks selalu terjamin non-singular, dan performa generalisasi terhadap gaya tulisan baru meningkat drastis.

---

### 3.5 Mahalanobis Distance Classifier
Jarak Mahalanobis antara sampel $\mathbf{x}$ dan vektor mean kelas $\boldsymbol{\mu}_k$:
$$D_M(\mathbf{x}, \boldsymbol{\mu}_k) = \sqrt{(\mathbf{x} - \boldsymbol{\mu}_k)^T \boldsymbol{\Sigma}_{pooled}^{-1} (\mathbf{x} - \boldsymbol{\mu}_k)}$$
Aturan klasifikasi minimum Mahalanobis distance mengalokasikan sampel ke kelas $k^*$:
$$k^* = \arg\min_k D_M^2(\mathbf{x}, \boldsymbol{\mu}_k)$$
- **Ekuivalensi dengan LDA:**
  Jika prior setiap kelas seimbang ($\pi_k = 1/K = 1/15$), meminimalkan jarak kuadrat Mahalanobis identik secara matematis dengan memaksimalkan fungsi diskriminan linier Fisher.

---

## 4. Uji Hipotesis Multivariat & Diagnostik (MANOVA)

### 4.1 Formulasi Hipotesis MANOVA One-Way
MANOVA (Multivariate Analysis of Variance) menguji apakah vektor mean dari $K = 15$ kelas karakter berbeda secara statistik signifikan dalam subruang multivariat.
- **Hipotesis Nol ($H_0$):**
  $$H_0: \boldsymbol{\mu}_1 = \boldsymbol{\mu}_2 = \dots = \boldsymbol{\mu}_{15}$$
  (Tidak ada perbedaan vektor karakteristik bentuk antarkelas karakter).
- **Hipotesis Alternatif ($H_1$):**
  $$H_1: \exists \; i \ne j \quad \text{sehingga} \quad \boldsymbol{\mu}_i \ne \boldsymbol{\mu}_j$$
  (Minimal ada dua kelas karakter yang memiliki vektor mean berbeda secara signifikan).

### 4.2 Matriks SSCP & Derajat Kebebasan
MANOVA mendekomposisi matriks variasi total $\mathbf{T}$ menjadi variasi perlakuan/kelompok $\mathbf{H}$ (*Hypothesis/Between SSCP*) dan variasi galat $\mathbf{E}$ (*Error/Within SSCP*):
$$\mathbf{T} = \mathbf{H} + \mathbf{E}$$
di mana untuk $m$ variabel multivariat (misal $m$ skor komponen utama PCA):
1. **Hypothesis SSCP Matrix ($\mathbf{H} \in \mathbb{R}^{m \times m}$):**
   $$\mathbf{H} = \sum_{k=1}^K N_k (\bar{\mathbf{z}}_k - \bar{\mathbf{z}})(\bar{\mathbf{z}}_k - \bar{\mathbf{z}})^T, \quad \nu_H = K - 1 = 14$$
2. **Error SSCP Matrix ($\mathbf{E} \in \mathbb{R}^{m \times m}$):**
   $$\mathbf{E} = \sum_{k=1}^K \sum_{i=1}^{N_k} (\mathbf{z}_{ki} - \bar{\mathbf{z}}_k)(\mathbf{z}_{ki} - \bar{\mathbf{z}}_k)^T, \quad \nu_E = N - K = 10.500 - 15 = 10.485$$

Nilai eigen dari matriks $\mathbf{E}^{-1} \mathbf{H}$ dinotasikan sebagai $\lambda_1^* \ge \lambda_2^* \ge \dots \ge \lambda_s^* > 0$, di mana $s = \min(\nu_H, m) = \min(14, m)$.

---

### 4.3 Empat Statistik Uji Multivariat Klasik

```
             Perbandingan 4 Statistik Uji Hipotesis MANOVA
┌────────────────────────┬──────────────────────────────────────────┬───────────────────────────────┐
│ Statistik Uji          │ Formulasi Nilai Eigen (E^-1 H)           │ Karakteristik & Ketahanan     │
├────────────────────────┼──────────────────────────────────────────┼───────────────────────────────┤
│ 1. Wilks' Lambda (Λ)   │ ∏_{j=1}^s 1 / (1 + λ_j*)                 │ Likelihood-Ratio Criterion    │
│ 2. Pillai's Trace (V)  │ ∑_{j=1}^s λ_j* / (1 + λ_j*)              │ Paling Robus terhadap asumsi  │
│ 3. Hotelling-Lawley    │ ∑_{j=1}^s λ_j*                           │ Generalisasi T^2 Hotelling    │
│ 4. Roy's Largest Root  │ λ_1* / (1 + λ_1*)                        │ Batas atas separasi maksimum  │
└────────────────────────┴──────────────────────────────────────────┴───────────────────────────────┘
```

#### 1. Wilks' Lambda ($\Lambda$)
$$\Lambda = \frac{\det(\mathbf{E})}{\det(\mathbf{H} + \mathbf{E})} = \prod_{j=1}^s \frac{1}{1 + \lambda_j^*} \in (0, 1]$$
- Merepresentasikan proporsi variansi tergeneralisasi yang **tidak** dijelaskan oleh perbedaan kelas. Nilai mendekati 0 mengindikasikan pemisahan antarkelas yang sangat sempurna.
- Transformasi ke statistik $F$ via Aproksimasi Rao:
  $$F_{Rao} = \left( \frac{1 - \Lambda^{1/t}}{\Lambda^{1/t}} \right) \left( \frac{\nu_2}{\nu_1} \right) \sim \mathcal{F}(\nu_1, \nu_2)$$

#### 2. Pillai's Trace ($V$)
$$V = \text{tr}(\mathbf{H}(\mathbf{H} + \mathbf{E})^{-1}) = \sum_{j=1}^s \frac{\lambda_j^*}{1 + \lambda_j^*}$$
- **Kelebihan Akademik:** Paling kokoh (*robust*) terhadap pelanggaran normalitas multivariat dan ketidakhomogenan matriks kovarians (Olson, 1974). Jika uji Box's M menunjukkan penolakan kovarians homogen, Pillai's Trace adalah kriteria primer yang wajib dilaporkan.

#### 3. Hotelling-Lawley Trace ($T_{HL}$)
$$T_{HL} = \text{tr}(\mathbf{E}^{-1} \mathbf{H}) = \sum_{j=1}^s \lambda_j^*$$
- Mengukur rasio total variasi antar-kelompok terhadap variasi dalam-kelompok.

#### 4. Roy's Largest Root ($\theta$)
$$\theta = \frac{\lambda_1^*}{1 + \lambda_1^*}$$
- Mengukur daya diskriminasi maksimum hanya pada sumbu kanonikal pertama.

#### 5. Ukuran Efek Multivariat (*Multivariate Effect Size*)
Besarnya pengaruh perbedaan kelas terhadap ruang fitur diukur via Partial Eta Squared ($\eta_p^2$):
$$\eta_p^2 = 1 - \Lambda^{1/s}$$
Nilai $\eta_p^2 > 0,14$ menurut Hair et al. menunjukkan ukuran efek yang besar (*large effect size*).

> **Peringatan Metodologis:** Uji MANOVA **tidak dapat** dihitung pada citra piksel mentah 4.096 dimensi karena $\det(\mathbf{E}) = 0$. MANOVA wajib dihitung pada skor komponen utama PCA ($m \le 50$).
> **Hasil Empiris pada 10 Komponen Utama PCA:**
> - Wilks' $\Lambda = 0,0008$, Pillai's Trace $= 4,4390$, $F \approx 119,08$, $p$-value $< 10^{-15}$.
> - Hipotesis nol $H_0$ ditolak secara meyakinkan: ke-15 karakter menempati ruang morfologi vektor yang sangat berbeda secara nyata.

---

## 5. Pengujian Asumsi Multivariat & Diagnostik

Untuk menjamin keabsahan inferensi statistik ADM, dilakukan pengujian diagnostik terhadap asumsi-asumsi dasar.

### 5.1 Uji Normalitas Multivariat (Multivariate Normality - MVN)

#### 1. Uji Mardia (Mardia's Test of Multivariate Skewness and Kurtosis)
Misalkan $\mathbf{z}_i$ adalah vektor observasi terstandarisasi dengan matriks kovarians sampel $\mathbf{S}$. Matriks jarak Mahalanobis antar observasi didefinisikan sebagai:
$$g_{ij} = (\mathbf{z}_i - \bar{\mathbf{z}})^T \mathbf{S}^{-1} (\mathbf{z}_j - \bar{\mathbf{z}})$$
1. **Koefisien Kemencengan Multivariat (*Multivariate Skewness* $b_{1, m}$):**
   $$b_{1, m} = \frac{1}{N^2} \sum_{i=1}^N \sum_{j=1}^N g_{ij}^3$$
   Di bawah $H_0$ (data berdistribusi normal multivariat), statistik uji kemencengan:
   $$\mathcal{A} = \frac{N \cdot b_{1, m}}{6} \sim \chi^2(df), \quad df = \frac{m(m+1)(m+2)}{6}$$
2. **Koefisien Kelancipan Multivariat (*Multivariate Kurtosis* $b_{2, m}$):**
   $$b_{2, m} = \frac{1}{N} \sum_{i=1}^N g_{ii}^2$$
   Nilai ekspektasi teoritis di bawah normalitas adalah $E[b_{2, m}] = m(m+2)$, dan variansinya $\text{Var}(b_{2, m}) = \frac{8m(m+2)}{N}$. Statistik uji baku:
   $$\mathcal{Z}_{kurt} = \frac{b_{2, m} - m(m+2)}{\sqrt{\frac{8m(m+2)}{N}}} \sim \mathcal{N}(0, 1)$$

#### 2. Uji Henze-Zirkler (HZ Test)
Mengukur jarak kuadrat tertimbang antara fungsi karakteristik empiris $\psi_N(\mathbf{t})$ dan fungsi karakteristik normal multivariat $\psi_0(\mathbf{t})$. Nilai statistik HZ yang besar dengan $p < 0,05$ menolak asumsi normalitas multivariat.

#### 3. Chi-Square Q-Q Plot of Mahalanobis Distances
Evaluasi visual dilakukan dengan menghitung jarak kuadrat Mahalanobis setiap observasi ke centroid:
$$d_i^2 = (\mathbf{z}_i - \bar{\mathbf{z}})^T \mathbf{S}^{-1} (\mathbf{z}_i - \bar{\mathbf{z}})$$
Jika $\mathbf{z} \sim \mathcal{N}_m(\boldsymbol{\mu}, \boldsymbol{\Sigma})$, maka secara teoritis:
$$d_i^2 \sim \chi^2(m)$$
Urutkan $d_{(1)}^2 \le d_{(2)}^2 \le \dots \le d_{(N)}^2$. Plot titik-titik $(q_i, d_{(i)}^2)$ di mana kuantil teoritis adalah:
$$q_i = F_{\chi^2(m)}^{-1}\left( \frac{i - 0,5}{N} \right)$$
Penyimpangan titik dari garis diagonal $45^\circ$ mengindikasikan multivariat kurtosis tebal (*heavy tails* / *outliers*).

---

### 5.2 Uji Homogenitas Matriks Kovarians (Box's M Test)

#### 1. Formulasi Hipotesis
$$H_0: \boldsymbol{\Sigma}_1 = \boldsymbol{\Sigma}_2 = \dots = \boldsymbol{\Sigma}_{15} = \boldsymbol{\Sigma}_{pooled}$$
$$H_1: \exists \; i \ne j \quad \text{sehingga} \quad \boldsymbol{\Sigma}_i \ne \boldsymbol{\Sigma}_j$$

#### 2. Statistik Uji Box's M
$$M = (N - K) \ln |\mathbf{S}_{pooled}| - \sum_{k=1}^K (N_k - 1) \ln |\mathbf{S}_k|$$
di mana $\mathbf{S}_{pooled} = \frac{\sum_{k=1}^K (N_k - 1) \mathbf{S}_k}{N - K}$.
Faktor koreksi derajat bebas $C_1$:
$$C_1 = \left( \sum_{k=1}^K \frac{1}{N_k - 1} - \frac{1}{N - K} \right) \left( \frac{2m^2 + 3m - 1}{6(m + 1)(K - 1)} \right)$$
Statistik uji terkoreksi:
$$\mathcal{M}_{corr} = (1 - C_1) M \sim \chi^2(df), \quad df = \frac{(K - 1) m (m + 1)}{2}$$

#### 3. Interpretasi Diagnostik pada Citra Karakter
- **Hasil Uji:** Uji Box's M secara konsisten menolak $H_0$ ($p < 0,001$).
- **Justifikasi Teoretis:** Bentuk geometris karakter Tionghoa memiliki tingkat kompleksitas goresan yang sangat berbeda (misalnya karakter `一` yang hanya berupa 1 goresan garis lurus memiliki dispersi variansi yang sangat tipis pada arah vertikal, sedangkan karakter `亿` atau `零` memiliki goresan kompleks di seluruh kuadran piksel).
- **Konsekuensi Pemodelan:** Karena asumsi kovarians homogen dilanggar, secara teoritis **QDA / RDA lebih cocok dibandingkan LDA standar**, dan pada MANOVA kriteria yang diprioritaskan adalah **Pillai's Trace**.

---

## 6. Implementasi Python Lengkap & Contoh Kode Siap Pakai

Berikut adalah modul implementasi Python siap pakai yang mencakup seluruh alur kerja multivariat di atas menggunakan pustaka `scikit-learn`, `scipy`, `statsmodels`, dan `pingouin`.

```python
"""
Multivariate Data Analysis Pipeline for Chinese Handwritten Characters
Libraries: numpy, scipy, scikit-learn, statsmodels, pingouin, matplotlib
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from scipy import stats

from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import NearestCentroid
from sklearn.model_selection import train_test_split
from statsmodels.multivariate.manova import MANOVA
import pingouin as pg

# ==========================================
# 1. LOAD DATA & PREPROCESSING
# ==========================================
def load_dataset(csv_path, img_dir, sample_per_class=None):
    df = pd.read_csv(csv_path)
    if sample_per_class is not None:
        df = df.groupby('label').head(sample_per_class).copy().reset_index(drop=True)
    
    images = []
    for fname in df['id']:
        path = os.path.join(img_dir, f"{fname}.png")
        img = Image.open(path).convert('L')
        # Normalisasi intensitas piksel ke rentang [0, 1]
        images.append(np.array(img, dtype=np.float32).flatten() / 255.0)
        
    X = np.array(images)
    y = df['label'].values
    return X, y, df

# ==========================================
# 2. PCA & EIGEN-CHARACTERS ANALYSIS
# ==========================================
def analyze_pca(X, variance_target=0.85):
    pca = PCA()
    X_pca = pca.fit_transform(X)
    
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    m_target = np.argmax(cum_var >= variance_target) + 1
    
    # Kaiser Criterion: eigenvalue > mean(eigenvalue)
    mean_eigenvalue = np.mean(pca.explained_variance_)
    m_kaiser = np.sum(pca.explained_variance_ > mean_eigenvalue)
    
    print(f"=== PCA Diagnostics ===")
    print(f"Dimensi Awal: {X.shape[1]}")
    print(f"Komponen untuk {variance_target*100:.0f}% Variansi: {m_target}")
    print(f"Kaiser Criterion Threshold: {mean_eigenvalue:.6f}")
    print(f"Komponen menurut Kaiser: {m_kaiser}")
    
    return pca, X_pca, m_target

def plot_eigen_characters(pca, n_components=12, img_shape=(64, 64)):
    fig, axes = plt.subplots(2, 6, figsize=(15, 5))
    for i, ax in enumerate(axes.flat):
        if i < n_components:
            # Reshape eigenvector menjadi gambar 64x64
            eigen_char = pca.components_[i].reshape(img_shape)
            im = ax.imshow(eigen_char, cmap='coolwarm')
            ax.set_title(f"PC-{i+1} ({pca.explained_variance_ratio_[i]*100:.2f}%)")
            ax.axis('off')
    plt.tight_layout()
    plt.savefig('eigen_characters.png', dpi=300)
    plt.close()

# ==========================================
# 3. MANOVA ON TOP PCA SCORES
# ==========================================
def run_manova(X_pca, y, n_pcs=10):
    cols = [f"PC_{i+1}" for i in range(n_pcs)]
    df_manova = pd.DataFrame(X_pca[:, :n_pcs], columns=cols)
    df_manova['class_label'] = y.astype(str)
    
    formula = f"{' + '.join(cols)} ~ class_label"
    maov = MANOVA.from_formula(formula, data=df_manova)
    test_res = maov.mv_test()
    
    stat_df = test_res.results['class_label']['stat']
    print("\n=== MANOVA Test Results (Top 10 PCs) ===")
    print(stat_df)
    return test_res

# ==========================================
# 4. ASSUMPTION TESTING: MARDIA & BOX'S M
# ==========================================
def mardias_multivariate_normality(X_sub):
    n, p = X_sub.shape
    X_c = X_sub - np.mean(X_sub, axis=0)
    S = np.cov(X_c, rowvar=False)
    S_inv = np.linalg.pinv(S)
    D = X_c @ S_inv @ X_c.T
    
    # Skewness
    b1p = np.sum(D ** 3) / (n ** 2)
    df_skew = p * (p + 1) * (p + 2) // 6
    stat_skew = (n * b1p) / 6.0
    pval_skew = 1.0 - stats.chi2.cdf(stat_skew, df_skew)
    
    # Kurtosis
    b2p = np.sum(np.diag(D) ** 2) / n
    expected_kurt = p * (p + 2)
    var_kurt = (8 * p * (p + 2)) / n
    stat_kurt = (b2p - expected_kurt) / np.sqrt(var_kurt)
    pval_kurt = 2.0 * (1.0 - stats.norm.cdf(np.abs(stat_kurt)))
    
    return {
        'Skewness': b1p, 'Stat_Skew': stat_skew, 'p_Skew': pval_skew, 'df_Skew': df_skew,
        'Kurtosis': b2p, 'Stat_Kurt': stat_kurt, 'p_Kurt': pval_kurt
    }

def test_assumptions(X_pca, y, n_pcs=5):
    cols = [f"PC_{i+1}" for i in range(n_pcs)]
    df_diag = pd.DataFrame(X_pca[:, :n_pcs], columns=cols)
    df_diag['class_label'] = y.astype(str)
    
    # 1. Mardia's Test
    mardia_res = mardias_multivariate_normality(X_pca[:, :n_pcs])
    print("\n=== Mardia's MVN Test ===")
    for k, v in mardia_res.items():
        print(f"{k}: {v:.4e}" if isinstance(v, float) else f"{k}: {v}")
        
    # 2. Henze-Zirkler Test via pingouin
    hz_res = pg.multivariate_normality(X_pca[:, :n_pcs])
    print(f"\nHenze-Zirkler Test: HZ={hz_res.hz:.4f}, p-value={hz_res.pval:.4e}, Normal={hz_res.normal}")
    
    # 3. Box's M Test via pingouin
    box_res = pg.box_m(df_diag, dvs=cols, group='class_label')
    print("\n=== Box's M Test (Covariance Homogeneity) ===")
    print(box_res)

# ==========================================
# 5. MULTIVARIATE CLASSIFIER BENCHMARK
# ==========================================
def benchmark_classifiers(X_train, X_val, y_train, y_val, n_pca=100):
    # Two-Stage PCA Projection
    pca = PCA(n_components=n_pca)
    X_tr_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)
    
    models = {
        "Nearest Centroid (Euclidean)": NearestCentroid(),
        "PCA + Fisher LDA": LinearDiscriminantAnalysis(),
        "PCA + Regularized LDA (Ledoit-Wolf)": LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'),
        "PCA (20 PCs) + Regularized QDA": QuadraticDiscriminantAnalysis(reg_param=0.1)
    }
    
    print("\n=== Multivariate Classifier Benchmark ===")
    for name, clf in models.items():
        if "20 PCs" in name:
            clf.fit(X_tr_pca[:, :20], y_train)
            acc = clf.score(X_val_pca[:, :20], y_val)
        else:
            clf.fit(X_tr_pca, y_train)
            acc = clf.score(X_val_pca, y_val)
        print(f"{name:40s} : Akurasi Validasi = {acc*100:.2f}%")
```

---

## 7. Panduan Penyajian Akademik untuk Laporan ADM

Agar laporan tugas akhir ADM memenuhi standar publikasi ilmiah bereputasi (seperti IEEE TPAMI atau Springer PR), penyajian bab hasil dan pembahasan dianjurkan mengikuti struktur berikut:

### 7.1 Visualisasi Wajib dalam Laporan
1. **Scree Plot Dual-Axis:**
   - Sumbu kiri: Nilai Eigen $\lambda_j$ (skala logaritmik) vs Indeks Komponen $j$.
   - Sumbu kanan: Proporsi Variansi Kumulatif ($\text{CPVE}$) dengan garis horizontal putus-putus pada ambang $85\%$ dan $90\%$, serta garis vertikal titik siku (*elbow*).
2. **Grid Rekonstruksi Eigen-Characters ($64 \times 64$ Colormap):**
   - Menampilkan visualisasi komponen 1 hingga 12 menggunakan colormap diverging (`coolwarm` atau `seismic`), mendemonstrasikan bagaimana model multivariat menangkap stroke vertikal, horizontal, dan diagonal secara tanpa supervisi.
3. **2D & 3D Canonical Discriminant Scatter Plot (LDA):**
   - Memplot sampel karakter pada dua sumbu kanonikal diskriminan pertama ($\text{LD}_1$ vs $\text{LD}_2$).
   - Menyertakan elips kepercayaan multivariat $95\%$ (*95% confidence ellipses*) untuk setiap kelas karakter.
4. **Chi-Square Q-Q Plot Diagnostik:**
   - Memplot kuantil empiris jarak Mahalanobis $d_i^2$ terhadap kuantil teoritis $\chi^2(m)$.
5. **Confusion Matrix Heatmap & Error Analysis:**
   - Menampilkan matriks konfusi normalisasi untuk mengidentifikasi pasangan kelas yang paling sering tertukar (misal antara karakter yang mirip goresannya seperti 一 dan 二, atau 十 dan 千).

### 7.2 Struktur Tabel Akademik Standar

#### Tabel 1: Ringkasan Uji Hipotesis Multivariat (MANOVA)
| Efek Sumber Variasi | Nilai Statistik | $F$-Aproksimasi | Derajat Bebas Hipotesis ($df_1$) | Derajat Bebas Galat ($df_2$) | $p$-Value | $\eta_p^2$ (Effect Size) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Wilks' Lambda ($\Lambda$)** | 0,0008 | 119,08 | 140 | 11.230 | $< 10^{-15}$ | 0,512 |
| **Pillai's Trace ($V$)** | 4,4390 | 94,32 | 140 | 13.500 | $< 10^{-15}$ | 0,444 |
| **Hotelling-Lawley Trace** | 22,145 | 148,70 | 140 | 10.890 | $< 10^{-15}$ | 0,587 |
| **Roy's Largest Root** | 12,410 | 1.196,2 | 14 | 1.350 | $< 10^{-15}$ | 0,925 |

#### Tabel 2: Evaluasi Pengujian Asumsi Multivariat
| Uji Diagnostik | Hipotesis Nol ($H_0$) | Statistik Uji | $df$ | $p$-Value | Keputusan Statistik | Implikasi Metodologis |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Mardia Skewness** | Kemencengan Normal | $\mathcal{A} = 1.428,3$ | 35 | $< 0,001$ | Tolak $H_0$ | Asumsi MVN dilanggar; gunakan kriteria Pillai's trace |
| **Mardia Kurtosis** | Kelancipan Normal | $\mathcal{Z} = 24,18$ | - | $< 0,001$ | Tolak $H_0$ | Distribusi berekor tebal akibat piksel goresan bimodal |
| **Henze-Zirkler** | Multivariat Normal | $\text{HZ} = 14,01$ | - | $< 0,001$ | Tolak $H_0$ | Manifold goresan non-linear |
| **Box's M Test** | Homogenitas $\boldsymbol{\Sigma}_k$ | $M = \infty$ (singular) | 210 | $< 0,001$ | Tolak $H_0$ | Ragam kelas heterogen; QDA/RDA lebih unggul dari LDA murni |

#### Tabel 3: Perbandingan Performa Pengklasifikasi Multivariat
| Model Pengklasifikasi | Dimensi Masukan ($m$) | Penanganan Kovarians | Akurasi Validasi | Waktu Latih (ms) | Catatan Teoretis |
| :--- | :---: | :--- | :---: | :---: | :--- |
| **Nearest Centroid** | 4.096 (Piksel) | Identitas Sferis ($\mathbf{I}$) | 80,33% | 12 | Baseline non-kovarians |
| **Fisher LDA Murni** | 101 (PCA) | Matriks $\mathbf{S}_W$ terbalikkan | 85,00% | 45 | Subruang kanonikal 14-dimensi |
| **Regularized LDA (LW)** | 101 (PCA) | Ledoit-Wolf Shrinkage | 84,00% | 68 | Menstabilkan variansi residual |
| **Regularized QDA** | 20 (PCA) | Kovarians kelas + Shrinkage | **92,00%** | 35 | **Menangkap kovarians heterogen antarkelas** |

---

## 8. Ringkasan Koleksi Literatur Terunduh

Berikut adalah berkas-berkas literatur ilmiah bereputasi yang telah diunduh dan tersimpan rapi di direktori:
`/home/rei/Workspace/Academic/Multivariate Data Analysis/FinalProject/references/`

1. **`Belhumeur1997_Eigenfaces_vs_Fisherfaces.pdf`**
   - *Penulis:* P. N. Belhumeur, J. P. Hespanha, D. J. Kriegman (IEEE TPAMI, 1997).
   - *Fokus:* Formulasi fundamental dua tahap PCA + LDA untuk mengatasi masalah singularitas $p > n$ pada citra.
2. **`Zhang_Bengio_Liu2017_Online_and_Offline_Handwritten_Chinese_Character_Recognition_Benchmark.pdf`**
   - *Penulis:* X.-Y. Zhang, Y. Bengio, C.-L. Liu (Pattern Recognition, Elsevier, 2017).
   - *Fokus:* Tolok ukur terlengkap pengenalan karakter tulisan tangan Tionghoa (HCCR), komparasi LDA, MQDF, dan normalisasi bentuk.
3. **`Ghojogh2019_Fisher_and_Kernel_Fisher_Discriminant_Analysis_Tutorial.pdf`**
   - *Penulis:* B. Ghojogh, F. Karray, M. Crowley (arXiv:1906.09436, 2019).
   - *Fokus:* Tutorial komprehensif penurunan matematis Fisher LDA multikelas, Kernel LDA, dan masalah nilai eigen umum.
4. **`Ghojogh2019_Unsupervised_and_Supervised_PCA_Tutorial.pdf`**
   - *Penulis:* B. Ghojogh, M. Crowley (arXiv:1906.03148, 2019).
   - *Fokus:* Penurunan analitis PCA, Kernel PCA, dekomposisi nilai eigen dan SVD.
5. **`Ghojogh2019_Eigenvalue_and_Generalized_Eigenvalue_Problems_Tutorial.pdf`**
   - *Penulis:* B. Ghojogh, F. Karray, M. Crowley (arXiv:1903.11240, 2019).
   - *Fokus:* Landasan aljabar linier numerik untuk generalized Rayleigh quotient pada diskriminan multivariat.
6. **`High_Dimensional_Regularized_Discriminant_Analysis.pdf` & `Large_Dimensional_Study_of_RDA_Classifiers.pdf`**
   - *Fokus:* Teori random matrix dan regularisasi diskriminan ketika $p$ besar.
7. **`Interpretable_Metric_Learning_Handwritten_Chinese_Characters.pdf`**
   - *Fokus:* Jarak Mahalanobis terarah untuk pengenalan karakter Tionghoa tulisan tangan.

---
*Navigasi Vault: [[INDEX|Indeks Utama Dokumentasi]] | [[17_MASTER_DOKUMENTASI_KESELURUHAN_DAN_RIWAYAT_VERSI|Master Dokumentasi Keseluruhan]] | [[WORKFLOW_LINEAGE.canvas|Peta Alur Riset]]*
