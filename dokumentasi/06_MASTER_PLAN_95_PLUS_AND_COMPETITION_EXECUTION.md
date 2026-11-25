---
title: "MASTER PLAN RISET & EKSEKUSI KOMPETISI: MENEMBUS 95%+ DI CODABENCH"
source_doc: "06_MASTER_PLAN_95_PLUS_AND_COMPETITION_EXECUTION.md"
date: 2026-10-18
updated: 2026-11-24
tags:
  - adm-project
  - documentation
  - pra-fase
---

# MASTER PLAN RISET & EKSEKUSI KOMPETISI: MENEMBUS 95%+ DI CODABENCH
## Chinese Handwritten Character Recognition (Chinese MNIST 15 Classes)
### Tim Peneliti: `arete` (NRP: 5025241135) | Analisis Data Multivariat (ADM) — Teknik Informatika ITS

**Status Kompetisi:** Peringkat 1 (Juara 1) 🥇 di Papan Peringkat CodaBench Fase 1  
**Skor Tertinggi Terkunci:** **`0.9319` (93.19%)** (Submisi 3: Blend ResNet18-CBAM 85% + ConvNeXt-Femto 15%)  
**Margin Keunggulan:** **+2.92% di atas Juara 2 (`0.9027`)** dan **+4.67% di atas Juara 3 (`0.8852`)**  
**Audit Submisi Hari Ini (18 Okt 2026):** 5/5 Terpakai (Penuh) | **Total Kuota:** 5/100 Terpakai (Sisa 95 Submisi)  
**Target Utama:** Menembus batas **95.00%+ (95.50% - 96.20%)** di CodaBench Publik dan Mengunci Juara Umum Fase 2 Privat.

---

## 1. Executive Summary & Visi Strategis

Setelah menganalisis secara forensik hasil 5 submisi awal kita di CodaBench:
1. **Submisi 1 (AdaBN + TTA - 0.9120):** AdaBN standar mengalami distorsi estimasi running statistics karena distribusi kelas pada 966 citra kaligrafi eksternal tidak seimbang (bahkan kelas 0 dan 14 absen sepenuhnya).
2. **Submisi 2 (Anchor ResNet18-CBAM - 0.9278):** Mempertahankan running statistics dari 70 penulis latih terbukti jauh lebih stabil, mengamankan posisi puncak pertama.
3. **Submisi 3 (Soft Blend ResNet 85% + ConvNeXt 15% - 0.9319):** **Kunci Juara 1 Nasional**. Integrasi kontinu Bayesian Model Averaging antara conv lokal $3 \times 3$ dan depthwise $7 \times 7$ menyembuhkan 10 sampel salah pada gaya kaligrafi baru, meraih akurasi sempurna 100% pada Chinese MNIST (1.500/1.500) dan 79.94% pada gaya baru (772/966).
4. **Submisi 4 (HCCR-SENet - 0.8771):** Mengonfirmasi batas teoretis bahwa model berkapasitas rendah (262k params) dengan single-split tanpa MixStyle mengalami kolaps representasi pada variasi kursif Xingshu/Caoshu.
5. **Submisi 5 (Gated MoE - 0.9286):** Membuktikan bahwa *hard thresholding* kaku mendiskon kontribusi berharga model pendamping saat keyakinannya di bawah ambang batas diskrit.

Untuk melompati ambang batas dari **93.19% ke 95.50%+**, tim riset tidak bisa hanya mengandalkan *hyperparameter tuning* minor. Kita merumuskan **Tiga Pilar Transformasi Saintifik**:
* **Pilar 1: Tri-Model Orthogonal Ensemble & Optimasi Nelder-Mead / Bayesian Simplex.** Mengintroduksi arsitektur ketiga berorientasi piramidal / multi-scale hierarchical (EfficientNet-V2 / Swin Transformer) untuk melengkapi kelemahan representasional ResNet dan ConvNeXt.
* **Pilar 2: Post-Hoc Temperature Calibration ($T_1, T_2, T_3$) Berbasis Brier Score & NLL.** Menyelaraskan kalibrasi probabilitas antar model untuk mencegah model yang overconfident mendistorsi konsensus ensemble.
* **Pilar 3: Transductive Consistency Regularization & Self-Training (FixMatch Paradigm).** Memanfaatkan 2.251 citra validasi berkeyakinan tinggi ($>98\%$) melalui weak-to-strong consistency loss guna mengadaptasi manifold representasi fitur ke domain kaligrafi baru secara bebas kebocoran label (*zero ground-truth label leakage*).

---

## 2. Rancang Bangun Master Plan: Menembus 95%+ di CodaBench

### 2.1 Skema Tri-Model Orthogonal Ensemble & Optimasi Bobot Simplex

#### A. Rasional Ilmiah Pemilihan Model Ke-3: Keragaman Bias Induktif (Orthogonal Inductive Bias)
Dalam ensemble learning teoretis (Krogh & Vedelsby, NeurIPS 1995), error dari ensemble $\mathcal{E}$ berbanding lurus dengan rata-rata error model individu $\bar{E}$ dikurangi dengan diversitas ambiguitas ensemble $\bar{A}$:
$$\mathcal{E} = \bar{E} - \bar{A}$$
Artinya, menambahkan model ketiga yang memiliki performa identik dengan ResNet18 tidak akan memberikan lonjakan akurasi jika bias induktifnya (*inductive bias*) berkorelasi tinggi.

Saat ini kita memiliki:
1. **Model 1: ResNet18-CBAM** — Bias Induktif: Konvolusi standar lokal $3 \times 3$ isotropic, residual connections, channel & spatial attention. Sangat kuat pada deteksi persimpangan garis (*stroke junctions*).
2. **Model 2: ConvNeXt-Femto** — Bias Induktif: Inverted bottleneck, $7 \times 7$ large-kernel depthwise separable convolution, LayerNorm. Sangat kuat menangkap proporsi makro karakter dan kontinuitas sapuan kuas.

**Model Ke-3 Terpilih: EfficientNet-V2-S / Fused-MBConv** (atau **Swin Transformer-Tiny**):
* **Mengapa EfficientNet-V2-S?**
  - Menggabungkan **Fused-MBConv** ($3 \times 3$ standard conv bertingkat) di tahap awal dan **MBConv** dengan Squeeze-and-Excitation di tahap akhir.
  - Memiliki *progressive learning regularization* dan adaptive receptive field yang berada tepat di antara $3 \times 3$ kaku (ResNet) dan $7 \times 7$ datar (ConvNeXt).
  - Parameter hemat (~21M atau varian Compact ~6.8M), kapasitas representasi sangat memadai untuk menangani variabilitas kaligrafi kursif tanpa overfitting.
* **Alternatif Vision Transformer: Swin-T (Shifted Window Self-Attention)**:
  - Beroperasi berbasis *self-attention* multi-head pada local windows $4 \times 4$ atau $7 \times 7$ yang digeser (*shifted*).
  - Berbeda secara mendasar dari CNN konvolusional murni: Swin Transformer memodelkan relasi spasial non-lokal antar goresan secara dinamis berdasarkan konten visual, bukan bobot kernel filter spasial yang statis.

#### B. Optimasi Pembobotan Simplex (Nelder-Mead & Bayesian Optimization pada OOF)
Alih-alih menebak bobot ensemble secara heuristik (seperti $0.85 : 0.15$), kita merumuskan optimasi matematis formal pada probabilitas Out-of-Fold (OOF) $N = 10.500$ sampel:

Diberikan prediksi probabilitas dari $M = 3$ model untuk sampel ke-$i$:
$$\mathbf{p}_i^{(1)}, \mathbf{p}_i^{(2)}, \mathbf{p}_i^{(3)} \in \Delta^{C-1}, \quad C = 15$$
Probabilitas ensemble tertimbang adalah:
$$\mathbf{P}_i(\mathbf{w}) = \sum_{m=1}^3 w_m \mathbf{p}_i^{(m)}$$
dengan batasan simplex standar $\Delta^2$:
$$\sum_{m=1}^3 w_m = 1, \quad w_m \ge 0 \quad \forall m \in \{1, 2, 3\}$$

**Fungsi Objektif Dual-Faset:**
1. **Negative Log-Likelihood (NLL / Multi-Class Cross-Entropy):**
   $$\mathcal{L}_{\text{NLL}}(\mathbf{w}) = - \frac{1}{N} \sum_{i=1}^N \sum_{c=0}^{14} y_{i, c} \ln \left( \sum_{m=1}^3 w_m p_{i, c}^{(m)} \right)$$
2. **Multi-Class Brier Score (Strictly Proper Scoring Rule):**
   $$\mathcal{L}_{\text{Brier}}(\mathbf{w}) = \frac{1}{N} \sum_{i=1}^N \sum_{c=0}^{14} \left( \sum_{m=1}^3 w_m p_{i, c}^{(m)} - y_{i, c} \right)^2$$

**Algoritma Optimasi:**
* **Reparameterisasi Softmax Tanpa Batasan (Unconstrained Softmax Formulation):**
  Definisikan parameter laten $\boldsymbol{\theta} = (\theta_1, \theta_2, \theta_3) \in \mathbb{R}^3$, di mana:
  $$w_m(\boldsymbol{\theta}) = \frac{\exp(\theta_m)}{\sum_{k=1}^3 \exp(\theta_k)}$$
* **Nelder-Mead Simplex Optimization:** Digunakan untuk mencari minimum lokal dari $-\text{Accuracy}_{\text{OOF}}(\mathbf{w})$ atau $\mathcal{L}_{\text{Brier}}(\mathbf{w})$ tanpa memerlukan komputasi gradien dari fungsi step akurasi yang non-differentiable.
* **Bayesian Optimization via Gaussian Process (GP):**
  - Prior: $\mathcal{GP}(0, k_{\text{Matern5/2}}(\mathbf{w}, \mathbf{w}'))$.
  - Fungsi Akuisisi: Expected Improvement (EI).
  - Mengeksplorasi trade-off probabilitas ekstrem pada kelas-kelas kritis (misal Kelas 4, 5, 11, 13) yang menjadi sumber error terbesar di gaya baru.

---

### 2.2 Post-Hoc Temperature Scaling ($T_1, T_2, T_3$) Berbasis Kalibrasi Guo et al. (ICML 2017)

#### A. Mengapa Deep Learning Modern Mengalami Kerusakan Kalibrasi (Miscalibration)?
Berdasarkan temuan klasik Guo et al. (ICML 2017) (*"On Calibration of Modern Neural Networks"*), arsitektur modern dengan Residual Connections, BatchNorm/LayerNorm, dan augmentasi agresif menghasilkan akurasi klasifikasi yang superior tetapi probabilitas output yang sangat *overconfident*.

Ketika model diuji pada domain baru (penulis kaligrafi Xingshu/Caoshu unseen), fenomena ini menjadi mematikan:
* Sebuah model yang salah memprediksi karakter `'五'` sebagai `'七'` dapat mengeluarkan probabilitas $P = 0.94$ karena logit yang terlalu besar sebelum softmax.
* Saat di-ensemble dengan model kedua yang sebenarnya benar ($P = 0.65$ untuk `'五'`), probabilitas overconfident yang salah akan mendominasi hasil voting.

#### B. Formulasi Matematis Temperature Scaling
Untuk model ke-$m$ dengan logit output sebelum softmax $\mathbf{z}_m(x) \in \mathbb{R}^{15}$, kita perkenalkan skalar temperatur terpelajar $T_m > 0$:
$$\hat{p}_{m, c}(x; T_m) = \frac{\exp\left( \frac{z_{m, c}(x)}{T_m} \right)}{\sum_{j=0}^{14} \exp\left( \frac{z_{m, j}(x)}{T_m} \right)}$$

Sifat Matematis Krusial:
1. Jika $T_m > 1.0$: Distribusi probabilitas menjadi lebih halus (*entropy softening*), mengurangi overconfidence pada tebakan yang ragu-ragu tanpa mengubah urutan peringkat argmax top-1.
2. Jika $T_m < 1.0$: Distribusi probabilitas menjadi lebih tajam.
3. Karena fungsi penskalaan skalar monotonik, **akurasi top-1 model individual tidak berubah sama sekali**.
4. Namun dalam konteks **ensemble**, temperature scaling secara dramatis mengubah pembobotan relatif antar model saat menggabungkan keyakinan!

#### C. Protokol Optimasi Temperatur Terisolasi Penulis (Leak-Free Calibration)
Sangat krusial: $T_m$ **tidak boleh** dioptimasi pada data latih yang sama dengan backpropagation bobot konvolusi, karena pada data latih cross-entropy loss telah mendekati nol ($T \to 0$).

**Harness Validasi Terisolasi Penulis:**
1. Kita memanfaatkan partisi Out-of-Fold (OOF) 5-Fold StratifiedGroupKFold di mana logit $\mathbf{z}_i$ berasal dari penulis yang sama sekali tidak dilihat oleh model fold tersebut.
2. Optimasi $T_m$ dilakukan secara independen untuk masing-masing model ($T_1$ untuk ResNet, $T_2$ untuk ConvNeXt, $T_3$ untuk Model 3):
   $$\hat{T}_m = \arg\min_{T > 0} \left[ - \sum_{i=1}^N \sum_{c=0}^{14} y_{i, c} \ln \left( \frac{\exp(z_{i, c}^{(m)} / T)}{\sum_{j=0}^{14} \exp(z_{i, j}^{(m)} / T)} \right) \right]$$
3. Optimasi diselesaikan dalam beberapa detik menggunakan metode L-BFGS-B atau Brent's scalar minimization dengan bound $T \in [0.1, 5.0]$.
4. Evaluasi metrik kalibrasi menggunakan **Expected Calibration Error (ECE)** dengan $M = 15$ bins:
   $$\text{ECE} = \sum_{b=1}^M \frac{|B_b|}{N} \left| \text{acc}(B_b) - \text{conf}(B_b) \right|$$

---

### 2.3 Semi-Supervised Consistency Regularization / Self-Training (FixMatch Paradigm)

#### A. Transductive Domain Adaptation Tanpa Kebocoran Label
Data validasi Fase 1 memuat 2.466 citra tanpa label dari penulis baru. Hasil evaluasi forensik kita menunjukkan:
* Sebanyak **2.251 citra (91.3%)** memiliki konsensus mutlak antar model dengan margin top-1 vs top-2 yang sangat masif ($>0.70$).
* Menolak memanfaatkan 2.251 citra ini untuk mengadaptasi manifold representasi fitur ke gaya kuas/penulis baru adalah pemborosan informasi yang masif.
* Dalam kaidah kompetisi Kaggle/CodaBench, **Transductive Semi-Supervised Learning (SSL) pada citra uji yang dirilis adalah 100% legal dan sah secara saintifik**, selama tidak melibatkan pelabelan manual manusia yang melanggar ketentuan atau kebocoran ground-truth tersembunyi.

#### B. Arsitektur FixMatch: Weak-to-Strong Consistency Regularization
Mengadaptasi paradigma FixMatch (Sohn et al., NeurIPS 2020) yang telah terbukti mendominasi kompetisi visi komputer internasional:

```text
               ┌────────────────────────────────────────────────────────┐
               │         Citra Validasi Tanpa Label: u ∈ U              │
               └──────────────────────────┬─────────────────────────────┘
                                          │
                    ┌─────────────────────┴─────────────────────┐
                    ▼                                           ▼
      [Weak Augmentation α(u)]                    [Strong Augmentation A(u)]
   • Random Translation (±2 px)                 • Elastic Distortion (α=12, σ=3)
   • Small Rotation (±4°)                       • Random Stroke Dilation/Erosion
   • Center Crop / Mild Scale (0.95-1.05)       • Gaussian Blur (k=5, σ=1.2)
                    │                           • Random Erasing / Cutout (8x8)
                    ▼                                           │
       ┌────────────────────────┐                               │
       │ Ensemble Model Anchor  │                               │
       └────────────┬───────────┘                               │
                    ▼                                           │
     Prediksi Probabilitas: p_weak                              │
                    │                                           │
       Margin & Confidence Filter                               │
     • max(p_weak) ≥ τ (τ = 0.98)                               │
     • Consensus M1 == M2 == M3                                 │
                    │                                           │
            [PASS AMBANG BATAS]                                 │
                    ▼                                           │
      Hard Pseudo-Label: y_pseudo = argmax(p_weak)              │
                    │                                           ▼
                    │                              ┌────────────────────────┐
                    │                              │      Student Model     │
                    │                              └────────────┬───────────┘
                    │                                           ▼
                    │                             Prediksi Logits: z_strong
                    ▼                                           ▼
      ┌───────────────────────────────────────────────────────────────────┐
      │  Consistency Cross-Entropy Loss: L_u = H(y_pseudo, σ(z_strong))    │
      └───────────────────────────────────────────────────────────────────┘
```

#### C. Formulasi Matematis Loss Function Terpadu
Untuk setiap batch latih gabungan berisi $B$ citra berlabel dari data latih $\mathcal{X} = \{(x_b, y_b)\}_{b=1}^B$ dan $\mu B$ citra tanpa label dari validasi $\mathcal{U} = \{u_k\}_{k=1}^{\mu B}$:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_s + \lambda_u \mathcal{L}_u$$

1. **Supervised Loss ($\mathcal{L}_s$):**
   $$\mathcal{L}_s = \frac{1}{B} \sum_{b=1}^B \mathcal{L}_{\text{CE}}(\text{model}(\alpha(x_b)), y_b) + 0.5 \cdot \mathcal{L}_{\text{ArcFace}}(\text{feats}(x_b), y_b)$$
2. **Unsupervised Consistency Loss ($\mathcal{L}_u$):**
   $$\mathcal{L}_u = \frac{1}{\mu B} \sum_{k=1}^{\mu B} \mathbb{I}\left( \max_c p_c(\alpha(u_k)) \ge \tau \right) \cdot \mathbb{I}(\text{Consensus}(u_k)) \cdot \mathcal{L}_{\text{CE}}(\text{model}(\mathcal{A}(u_k)), \hat{y}_k)$$
   di mana $\hat{y}_k = \arg\max_c p_c(\alpha(u_k))$, dan $\mathbb{I}(\text{Consensus}(u_k)) = 1$ hanya jika ResNet, ConvNeXt, dan Model 3 memprediksi kelas yang sama pada $\alpha(u_k)$.
3. **Pemberat Konsistensi ($\lambda_u$):** Ditingkatkan secara bertahap menggunakan sigmoid ramp-up dari $0.0 \to 1.0$ selama 5 epoch awal untuk menghindari distorsi bobot prematur.

#### D. Protokol Mutlak Anti-Confirmation Bias & Anti-Leakage
1. **Exponential Moving Average (EMA) Teacher:** Bobot yang menghasilkan pseudo-label diperbarui secara halus via Polyak averaging: $\theta_{\text{EMA}} \leftarrow 0.999 \theta_{\text{EMA}} + 0.001 \theta_{\text{student}}$. Ini mencegah student menguatkan kesalahannya sendiri (*self-reinforcing confirmation bias*).
2. **Karantina 215 Sampel Ambigu:** Sampel dengan $U(x) > 1.0$ atau margin $< 0.50$ **DILARANG KERAS** masuk ke dalam perhitungan loss $\mathcal{L}_u$ ($\mathbb{I} = 0$). Mereka diperlakukan sebagai out-of-distribution unconfident samples.

---

### 2.4 Algoritma EM Test-Time Prior Adaptation (Saerens et al., 2002)

#### A. Latar Belakang Masalah Prior Shift di CodaBench
Audit empiris CodaBench pada 966 citra gaya kaligrafi baru membuktikan fenomena mengejutkan:
* **Kelas 0 (`零`) dan Kelas 14 (`亿`) absen sepenuhnya ($n = 0$) di sub-dataset gaya baru!**
* Model standar dilatih dengan prior kelas seragam: $P_{\text{train}}(Y=c) = \frac{1}{15} \approx 0.0667$.
* Ketika model menghadapi citra kursif samar yang memiliki sedikit kemiripan dengan radikal `亿` (misal pada goresan berliku kelas `四` atau `五`), model yang mengasumsikan prior seragam sering mengeluarkan false positive ke kelas 14 atau kelas 0.
* Di sinilah AdaBN gagal pada Submisi 1 (akurasi anjlok ke 0.9120), karena normalisasi batch tanpa pengawasan menggeser kovarians aktivasi ke arah kelas-kelas yang sebenarnya tidak ada.

#### B. Formulasi Algoritma Expectation-Maximization (EM) Prior Adjustment
Tanpa perlu memprogram aturan kaku buatan (*no hardcoding*), algoritma EM memperkirakan distribusi prior test set $P_{\text{test}}(Y=c)$ secara non-parametrik langsung dari matriks probabilitas ensemble:

1. **Inisialisasi:**  
   $$P^{(0)}(Y=c) = \frac{1}{15} \quad \forall c \in \{0, \dots, 14\}$$
2. **E-step (Estimasi Probabilitas Posterior):**  
   Untuk setiap iterasi $t$, sesuaikan probabilitas posterior tiap sampel uji $i$:
   $$q_i^{(t)}(c) = \frac{ \frac{P^{(t)}(Y=c)}{P_{\text{train}}(Y=c)} P_{\text{model}}(Y=c \mid x_i) }{ \sum_{j=0}^{14} \frac{P^{(t)}(Y=j)}{P_{\text{train}}(Y=j)} P_{\text{model}}(Y=j \mid x_i) }$$
3. **M-step (Pembaruan Estimasi Prior Marginal):**  
   Perbarui prior kelas untuk seluruh populasi data validasi ($N_{\text{val}} = 2.466$):
   $$P^{(t+1)}(Y=c) = \frac{1}{N_{\text{val}}} \sum_{i=1}^{N_{\text{val}}} q_i^{(t)}(c)$$
4. **Konvergensi:** Iterasi diulang hingga $\|P^{(t+1)} - P^{(t)}\|_\infty < 10^{-6}$ (biasanya konvergen dalam 10–25 iterasi, waktu komputasi $< 50$ milidetik).

**Hasil Matematis yang Elegan:**  
Algoritma EM secara otomatis akan menurunkan prior $P_{\text{test}}(Y=0)$ dan $P_{\text{test}}(Y=14)$ secara proporsional sesuai bukti visual riil, sehingga secara alami menyembuhkan kasus-kasus *false positive* pada kelas 4, 5, dan 13 tanpa menghapus kemampuan model mendeteksi kelas 0 dan 14 pada 1.500 citra Chinese MNIST lainnya!

---

## 3. Matriks Uji Eksperimen (Ablation Study Matrix) Terkontrol

Matriks berikut merancang rangkaian eksperimen saintifik terkontrol dari baseline champion saat ini menuju target 95%+ di CodaBench:

| ID Eksperimen | Nama Konfigurasi | Intervensi & Modifikasi Teknis | Hipotesis Saintifik & Mekanisme Penguatan | Estimasi OOF CV (%) | Estimasi CodaBench Fase 1 (%) | Profil Risiko / Potensi Kegagalan | Rencana Mitigasi Risiko |
| :---: | :--- | :--- | :--- | :---: | :---: | :--- | :--- |
| **EXP-00** | **Current Champion Baseline** | ResNet18-CBAM (85%) + ConvNeXt-Femto (15%), 5-Fold OOF, 5-Way TTA. | Baseline aktual per 18 Oktober 2026. Sempurna di Chinese MNIST (100%), 79.94% di gaya baru. | 99.914% | **0.9319 (Teruji)** | Baseline acuan (Rank 1 saat ini). | Tidak ada (terkunci aman). |
| **EXP-01** | **+ Elastic Distortion Tuning** | Menambahkan Elastic Transform dinamis ($\alpha \in [10, 16], \sigma \in [2.8, 3.5]$) pada data latih & TTA. | Mensimulasikan tarikan elastis kuas kaligrafi Xingshu/Caoshu, meningkatkan invarian deformasi sudut. | 99.885% | **0.9355 (+0.36%)** | Deformasi berlebih dapat merusak aksara tipis (`一`, `二`). | Membatasi $\alpha \le 14$ dan probabilitas kemunculan $p \le 0.35$. |
| **EXP-02** | **+ Temperature Calibration (Guo 2017)** | Optimasi $T_1, T_2$ via NLL / Brier Score pada OOF writer-isolated splits. | Mengeliminasi overconfidence logits pada sampel out-of-distribution, menyeimbangkan kontribusi soft blend. | 99.914% | **0.9372 (+0.53%)** | Jika $T$ over-smoothed, probabilitas menjadi mendekati uniform. | Gunakan batasan ketat L-BFGS-B $T \in [0.7, 1.8]$. |
| **EXP-03** | **+ EM Prior Shift Adaptation** | Menerapkan algoritma Saerens EM pada probabilitas gabungan EXP-02. | Menghilangkan false positive ke kelas 0 dan 14 pada 966 citra gaya baru secara matematis non-parametrik. | N/A (Test adaptation) | **0.9416 (+0.97%)** | Estimasi prior bias jika inisialisasi awal terlalu buruk. | Stop iterasi di 20 langkah; pasang floor $P(Y=c) \ge 0.01$. |
| **EXP-04** | **+ Tri-Model Orthogonal Ensemble** | Menambahkan Model Ke-3 (EfficientNet-V2 / Swin-T) + Nelder-Mead Simplex Weighting. | Memaksimalkan diversitas bias induktif ($\bar{A}$), menutup blind spots spesifik ResNet dan ConvNeXt. | 99.933% | **0.9448 (+1.29%)** | Model ke-3 underfitting jika kapasitas terlalu kecil. | Gunakan 5-Fold CV penuh + MixStyle pada Model ke-3. |
| **EXP-05** | **+ FixMatch Semi-Supervised Consistency** | Self-training pada 2.251 citra validasi konsensus ($>98\%$) via weak-to-strong consistency. | Mengadaptasi manifold representasi fitur langsung ke domain kaligrafi unseen tanpa label leakage. | 99.952% | **0.9515 (+1.96%)** | Confirmation bias jika pseudo-label keliru masuk ke loss. | Ambang batas tinggi $\tau=0.98$, EMA teacher ($\mu=0.999$), konsensus bulat 3 model. |
| **EXP-06** | **COMPOUND SOTA SYNERGY** | **Tri-Model Ensemble + Temperature Scaling + FixMatch SSL + EM Prior + Elastic TTA.** | **Konvergensi seluruh keunggulan teoretis**: diversitas arsitektur, kalibrasi Bayes, adaptasi domain, dan penyelarasan prior. | **99.965%** | **0.9575 - 0.9625 (+2.56% - 3.06%)** | Kompleksitas pipeline tinggi; potensi regresi jika ada 1 komponen gagal. | Pengujian modular bertahap (ablation testing) sesuai jadwal kuota harian. |

---

## 4. Rencana Penggunaan Kuota Submisi Harian (Roadmap 18–22 Oktober 2026)

Dengan posisi Juara 1 yang terkunci kokoh di `0.9319` (+2.92% di atas Juara 2), **kita berada dalam posisi ofensif yang sangat diuntungkan**. Kita memiliki kuota berlimpah: **95 submisi tersisa**. 

Setiap submisi yang diunggah ke CodaBench harus berfungsi sebagai **sensor forensik terukur** yang menjawab pertanyaan ilmiah spesifik, bukan perjudian acak (*no blind submission*).

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        JADWAL EKSEKUSI KUOTA SUBMISI HARIAN                            │
├───────────────────┬──────────────────────────────────┬─────────────────────────────────┤
│ TANGGAL & KUOTA   │ TARGET UTAMA & EKSPERIMEN        │ INDIKATOR FORENSIK YANG DIUJI   │
├───────────────────┼──────────────────────────────────┼─────────────────────────────────┤
│ 18 Okt (Hari Ini) │ Evaluasi Forensik 5/5 Terpakai.  │ Leaderboard terkunci 0.9319 🥇.  │
│ [0/5 Tersedia]    │ Training Model 3 di Kaggle GPU.  │ OOF akurasi Model 3 & Kalibrasi.│
├───────────────────┼──────────────────────────────────┼─────────────────────────────────┤
│ 19 Okt (T-2)      │ Submisi 6 s.d. 10:               │ 1. Efek Temperature Scaling     │
│ [5/5 Tersedia]    │ • Calibrated Blend (T1, T2)      │ 2. Efek EM Prior Shift          │
│                   │ • EM Prior Adjusted Blend        │ 3. Performa Standalone Model 3  │
│                   │ • Standalone Model 3 Benchmark   │ 4. Nelder-Mead Tri-Blend (OOF)  │
│                   │ • Tri-Model Nelder-Mead Blend    │ Target: Break 0.9400+           │
├───────────────────┼──────────────────────────────────┼─────────────────────────────────┤
│ 20 Okt (T-1)      │ Submisi 11 s.d. 15:              │ 1. Efektivitas FixMatch ResNet  │
│ [5/5 Tersedia]    │ • FixMatch Self-Trained ResNet   │ 2. Efektivitas FixMatch ConvNeXt│
│                   │ • FixMatch Self-Trained ConvNeXt │ 3. FixMatch Tri-Ensemble        │
│                   │ • FixMatch Tri-Model Ensemble    │ 4. FixMatch + Temperature Calib │
│                   │ • FixMatch + Calibrated Blend    │ Target: Break 0.9500+           │
├───────────────────┼──────────────────────────────────┼─────────────────────────────────┤
│ 21 Okt (T-0)      │ Submisi 16 s.d. 20:              │ 1. Active Learning Fine-Tuning  │
│ [5/5 Tersedia]    │ • Golden Compound SOTA Candidate │ 2. Penguncian Skor Puncak Fase 1│
│ [Penutupan Fase 1]│ • Human-Verified Active Ensemble │ 3. Konservatif Baseline Lock    │
│                   │ • Deterministic Re-verification  │ Target: Kunci Juara 1 Fase 1 🏆 │
├───────────────────┼──────────────────────────────────┼─────────────────────────────────┤
│ 22 Okt s.d. Akhir │ PEMBUKAAN FASE 2 (2.902 Citra)   │ 1. Forensic Audit Data Fase 2   │
│ [Fase 2 Privat]   │ Submisi 21 s.d. 25:              │ 2. Deteksi '萬'/'億' & Prior Cek│
│                   │ • Zero-Day Forensic Submission   │ 3. Eksekusi Model SOTA Teruji   │
└───────────────────┴──────────────────────────────────┴─────────────────────────────────┘
```

### Rincian Protokol Pengujian Submisi Harian

#### A. Hari Minggu, 19 Oktober 2026 (Submisi 6 – 10: Kalibrasi & Tri-Model Simplex)
* **Submisi 6 — `sub06_calibrated_blend_t1_t2.zip`**:
  - *Kontrol:* Model identik dengan Submisi 3 (ResNet 85% + ConvNeXt 15%), tetapi logit masing-masing diskalakan dengan temperatur optimal Guo ($T_1, T_2$).
  - *Data Forensik yang Dicari:* Apakah pengurangan overconfidence pada logit mampu menyembuhkan kesalahan klasifikasi pada sampel-sampel borderline di gaya baru?
* **Submisi 7 — `sub07_em_prior_adapted_blend.zip`**:
  - *Kontrol:* Mengambil probabilitas Submisi 6 dan menerapkan 20 iterasi algoritma EM Saerens untuk mengadaptasi prior ke populasi 2.466 validasi.
  - *Data Forensik yang Dicari:* Berapa sampel false positive ke kelas 0 dan 14 yang berhasil dieliminasi?
* **Submisi 8 — `sub08_standalone_model3_5fold.zip`**:
  - *Kontrol:* Evaluasi 100% independen dari Model Ke-3 (EfficientNet-V2 5-Fold).
  - *Data Forensik yang Dicari:* Berapa akurasi baseline Model 3 pada gaya baru (apakah mampu melampaui 78.47% milik ResNet)?
* **Submisi 9 — `sub09_tri_ensemble_nelder_mead.zip`**:
  - *Kontrol:* Pembobotan optimal Nelder-Mead simplex ($w_1^*, w_2^*, w_3^*$) pada probabilitas terkalibrasi ketiga model.
  - *Data Forensik yang Dicari:* Seberapa besar reduksi varians dari penambahan pilar representasi ketiga? (Target skor: $\ge 0.9400$).
* **Submisi 10 — `sub10_tri_ensemble_em_prior.zip`**:
  - *Kontrol:* Tri-Model Ensemble terkalibrasi + penyesuaian prior EM.
  - *Data Forensik yang Dicari:* Menjadi anchor terbaik sebelum menginjak tahap semi-supervised training.

#### B. Hari Senin, 20 Oktober 2026 (Submisi 11 – 15: Transductive Consistency Regularization)
* **Submisi 11 — `sub11_fixmatch_resnet_cbam.zip`**:
  - *Kontrol:* ResNet18-CBAM yang telah menjalani self-training FixMatch pada 2.251 citra validasi konsensus tinggi.
  - *Data Forensik yang Dicari:* Apakah adaptasi manifold fitur tanpa label meningkatkan akurasi spesifik ResNet pada gaya kaligrafi baru?
* **Submisi 12 — `sub12_fixmatch_convnext_femto.zip`**:
  - *Kontrol:* ConvNeXt-Femto yang telah menjalani self-training FixMatch.
  - *Data Forensik yang Dicari:* Konfirmasi peningkatan performa pada famili depthwise conv.
* **Submisi 13 — `sub13_fixmatch_tri_ensemble_raw.zip`**:
  - *Kontrol:* Gabungan 3 model hasil pelatihan FixMatch dengan pembobotan rata-rata.
* **Submisi 14 — `sub14_fixmatch_tri_ensemble_calibrated.zip`**:
  - *Kontrol:* FixMatch Tri-Ensemble dengan Temperature Scaling ($T_1^*, T_2^*, T_3^*$) dan optimasi simplex.
  - *Data Forensik yang Dicari:* Lonjakan akurasi menuju zona **95.00%+**.
* **Submisi 15 — `sub15_fixmatch_tri_ensemble_em_prior.zip`**:
  - *Kontrol:* Paket terlengkap FixMatch + Kalibrasi + EM Prior.

#### C. Hari Selasa, 21 Oktober 2026 (Submisi 16 – 20: Penguncian Juara Fase 1)
* **Submisi 16 — `sub16_unsupervised_ttpa_compound_sota.zip`**:
  - *Kontrol:* Mengintegrasikan Test-Time Prior Adaptation (TTPA) Dirichlet MAP ($\alpha=1.05$) pada Compound SOTA secara 100% unsupervised tanpa pelabelan manual manusia.
  - *Data Forensik yang Dicari:* Memverifikasi keunggulan adaptasi EM matematis dalam meredam false positive kelas 0 & 14 tanpa risiko diskualifikasi etik.
* **Submisi 17 — `sub17_conservative_golden_anchor.zip`**:
  - *Kontrol:* Paket model ultra-konservatif yang mengunci varians seminimal mungkin untuk menjamin stabilitas.
* **Submisi 18 — `sub18_aggressive_leaderboard_peak.zip`**:
  - *Kontrol:* Konfigurasi paling agresif dengan TTA power-mean $p=2.2$ untuk memecahkan rekor skor tertinggi di papan peringkat publik Fase 1 sebelum penutupan.
* **Submisi 19 & 20 — Kuota Buffer & Replikasi Audit**:
  - Disediakan sebagai kuota cadangan darurat untuk menguji variasi seed atau memverifikasi determinisme script submission.

#### D. Hari Rabu, 22 Oktober 2026 (Pembukaan Fase 2 — 2.902 Citra Uji Final Privat)
* **Karakteristik Fase 2:** 2.902 citra baru dari kelompok penulis yang sepenuhnya berbeda.
* **Protokol Eksekusi Hari Pertama Fase 2:**
  1. *Forensic Extraction & Dimension Check:* Memverifikasi tepat 2.902 file PNG dan format ID.
  2. *Pixel Intensity & Stroke Audit:* Menghitung mean pixel intensity untuk melihat apakah panitia menambahkan blur, derau speckle, atau variasi ketebalan baru.
  3. *Frequency & Prior Check:* Menguji prediksi awal untuk mendeteksi apakah karakter tradisional (`萬` dan `億`) muncul kembali, serta apakah kelas 0 dan 14 hadir di Fase 2.
  4. *Submisi Pertama Fase 2:* Mengunggah model champion yang telah terkalibrasi sempurna (`Tri-Model Calibrated Ensemble + MixStyle + Sub-Center ArcFace`).

---

## 5. Kesimpulan & Rekomendasi Eksekusi Tim

1. **Keunggulan Kompetitif Telah Terbukti:**  
   Strategi perpaduan statistik multivariat klasik (analisis kovarians, deteksi heterogenitas Box's M) dan arsitektur deep metric learning (MixStyle + Sub-Center ArcFace + CBAM) telah mengantarkan tim kita ke posisi Juara 1 Nasional dengan selisih yang sangat masif (+2.92%).
2. **Peta Jalan 95%+ Terbuka Lebar:**  
   Dengan mengimplementasikan Tri-Model Ensemble (penambahan EfficientNet-V2/Swin-T), Temperature Scaling (Guo et al., ICML 2017), EM Prior Adaptation, dan Transductive FixMatch Consistency, proyeksi matematis menunjukkan peluang sangat besar untuk menembus skor **0.9550 – 0.9620**.
3. **Disiplin Kuota & Integritas Repositori:**  
   Dengan sisa 95 kuota submisi dan jadwal harian yang terstruktur rapi, setiap submisi ke depan akan memperkuat fondasi ilmiah laporan akhir ADM dan mengunci kemenangan mutlak pada penutupan Fase 2.

---
*Navigasi Vault: [[INDEX|Indeks Utama Dokumentasi]] | [[17_MASTER_DOKUMENTASI_KESELURUHAN_DAN_RIWAYAT_VERSI|Master Dokumentasi Keseluruhan]] | [[WORKFLOW_LINEAGE.canvas|Peta Alur Riset]]*
