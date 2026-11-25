---
title: "21: Panduan Master Bulletproof - Matematika, Alur Sistem Lengkap, Bedah Kode 10 Notebook & Q&A Sidang"
source_doc: "21_PANDUAN_PAHAM_BULLETPROOF_PRESENTASI_DOSEN.md"
date: 2026-11-25
updated: 2026-11-25
tags:
  - presentation-defense
  - bulletproof-qa
  - deep-walkthrough
  - notebook-code-guide
  - multivariate-analysis
  - knowledge-distillation
  - adm-project
aliases:
  - Panduan Master Bulletproof Presentasi ADM
  - Bedah Rinci Sistem, Rumus, & Kode 10 Notebook
status: verified
---

# 21: Panduan Master Bulletproof Presentasi Dosen & Asisten Dosen

Dokumen ini disusun sebagai **referensi satu pintu yang sangat rinci** sebelum Anda mempresentasikan dek Canva 29 slide (`FP_Analisis_Data_Multivariat_Tim_Arete_5025241135_FINAL.pdf`) serta menghadapi sesi tanya jawab dosen dan pemeriksaan kode oleh asisten dosen. Isinya membedah seluruh proses dari awal hingga akhir (matematika ADM klasik sampai distilasi model akhir), dimensi tensor baris demi baris, kode tiap notebook `01` hingga `10` beserta fungsi di `src/`, dan 18 pertanyaan jebakan paling tajam.

---

## Bagian 1: Cara Membawakan PPT (`Slide 1-29`) Secara Efektif Sesuai Arahan Asdos

Asisten dosen menyampaikan bahwa tidak ada format baku wajib; intinya adalah menjelaskan **apa saja yang Anda kerjakan dari awal hingga akhir** serta **kode pendukungnya**. Dengan dek 29 slide yang lengkap, Anda **tidak perlu menceritakan riwayat Sub 1 sampai Sub 32 secara berurutan nomor per nomor**. Gunakan grafik trajektori sebagai visual pendukung di layar, lalu bagi narasi lisan ke dalam **4 babak** (~7 menit total):

1. **Babak 1 - Masalah Data & Fondasi ADM Klasik (`Slide 3-11`, ~2 menit)**: Porsi penilaian paling utama untuk mata kuliah ADM. Tekankan transformasi gambar `64 x 64` -> vektor `4.096-D`, alasan matriks kovarians singular (`det(S) = 0`), PCA (`109` komponen = `85%`, `154` = `90%`, `252` = `95%`, Kaiser = `275`), MANOVA (`Wilks' Lambda = 8,89e-5`), Box's M (`M = 214.542,08`), mengapa QDA (`96,09%`) menang `+10,96%` atas Fisher LDA (`85,13%`), dan bagaimana keempat temuan ini langsung menentukan desain jaringan saraf.
2. **Babak 2 - Kendala Domain Shift Kursif & Tiga Terobosan (`Slide 12-20`, ~1,5 menit)**: Lewati daftar nomor submit secara cepat. Fokus saja pada masalah nyata saat Fase 1 dibuka (`1.500` Chinese MNIST hampir `100%`, tetapi `966` Gaya Baru kaligrafi kursif jatuh di `五`, `百`, `四`, `千`), lalu jelaskan 3 obat utamanya: **DenseNet-121 CBAM**, **Biomekanik + 2.952 Kurikulum NCCU** (`lecturer_exact_filter.py`), dan **Bayesian EM TTPA** (`alpha = 1,05`). Tutup babak ini dengan alasan ilmiah mengapa ensemble `3,0 GB` + tomografi MILP (`Sub 30: 99,76%`) sengaja dihentikan karena lambat (`135x` *forward pass*) dan bersifat transduktif.
3. **Babak 3 - Model Final Terbaru: Knowledge Distillation `14,75 MiB` (`Slide 21-24`, ~2 menit)**: Bedah cara `MeanTeacherEMA` ($\alpha = 0,995$) pada citra bersih membimbing satu model murid `StudentDenseNet121_CBAM` pada citra terdistorsi biomekanik lewat **5-Loss `UnifiedDistillationLoss`** + **SWA Epoch 17-25** + kompresi **FP16 (`14,75 MiB`)** -> **`Sub 32 = 99,84%` (`4` error)**.
4. **Babak 4 - Kejujuran Uji Buta Fase 2 & Bedah Kode GitHub (`Slide 25-29`, ~1,5 menit)**: Tegaskan pemisahan **Fase 1 (*in-sample development benchmark*)** vs **Fase 2 tersegel (`2.902` citra baru, `0` duplikat, `0` MD5 overlap -> `98,62%` dalam `5,44 detik`)**, uji derau Gaussian $\sigma = 25$ (`95,98%`), dan peta `src/` + `10` notebook `.ipynb` di `https://github.com/Reiii0-0/MNIST-ADM`.

---

## Bagian 2: Penjelasan Rinci Matematika, Intuisi, & Dimensi Sistem (Dari Nol Sampai Selesai)

### 2.1 Titik Nol: Anatomi Data, Praproses, dan Kunci Validasi Anti-Bocor
1. **Representasi Data Citra -> Vektor Multivariat**:
   - Dataset inti `Chinese MNIST` terdiri atas `10.500` citra dari `70` penulis (`15` kelas x `10` sampel/penulis/kelas = `700` sampel per kelas).
   - Tiap gambar berukuran `64 x 64` piksel grayscale (`0-255`, tinta putih di atas latar hitam setelah normalisasi polaritas). Saat diratakan (*flatten*), satu gambar menjadi vektor acak:
     $$\mathbf{x}_i \in \mathbb{R}^{4096}, \qquad \mathbf{X}_{\text{train}} \in \mathbb{R}^{10500 \times 4096}$$
   - Perbedaan skala normalisasi antara dua jalur kode: di jalur ADM klasik ([`src/multivariate_analysis.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/multivariate_analysis.py)), piksel diskalakan ke **`[0, 1]`** (`img / 255.0`), sedangkan di jalur PyTorch Deep Learning (`Notebook 01-09` dan `src/predict_phase2_distillation.py`), piksel diskalakan ke **`[-1, 1]`** (`img / 127.5 - 1.0`).
2. **Audit Awal Data (Pra-Fase)**:
   - **Bimodalitas Aksara & Kekosongan Data Latih**: Kelas `13` (`万` sederhana 3 goresan vs `萬` tradisional 12 goresan) dan Kelas `14` (`亿` 3 goresan vs `億` 15 goresan) memiliki bentuk berlainan dalam satu label. Audit pada `10.500` data latih awal menunjukkan **0 contoh aksara tradisional `萬`/`億`**, sehingga disintesis `700` citra tradisional ([`src/generate_traditional_data.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/generate_traditional_data.py)) yang tepinya disamakan dengan blur pengampu (`GaussianBlur 3x3, sigma=0.5`).
   - **Isolasi Penulis Mutlak (`src/split_data.py`)**: Fungsi `create_folds(n_splits=5, random_state=42)` mempartisi data dengan **`StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`** menggunakan `groups=df['writer_id']` dan `assert len(train_writers.intersection(val_writers)) == 0`. Tiap fold melatih tepat `56` penulis (`8.400` citra murni, atau `9.100` bila ditambah `700` sintetik tradisional) dan menguji `14` penulis yang benar-benar belum pernah terlihat (`2.100` citra).

### 2.2 Jalur Analisis Data Multivariat Klasik (`src/multivariate_analysis.py`)
Mengapa tidak langsung memfitting LDA/QDA pada `4.096` piksel mentah?
1. **Fenomena Singularitas ($\det(\mathbf{S}) \approx 0$) meski $N > p$**:
   - Secara aljabar, $N = 10.500 > p = 4.096$ tampak mencukupi. Namun pada citra huruf berpusat di tengah, ribuan piksel di sudut/bingkai kanvas `64 x 64` bernilai `0` hampir di semua gambar (varians $\approx 0$), dan piksel yang bersebelahan berkorelasi nyaris `1,0`. Akibatnya matriks kovarians sampel $\mathbf{S} \in \mathbb{R}^{4096 \times 4096}$ kehilangan peringkat penuh (*rank-deficient*) dan invers $\mathbf{S}^{-1}$ tidak terdefinisi.
2. **Dekomposisi Spektral PCA (`run_pca_analysis`)**:
   - Data dipusatkan ($\mathbf{X}_c = \mathbf{X} - \bar{\mathbf{x}}$), lalu didekomposisi lewat `PCA(n_components=300, svd_solver='randomized', random_state=42)`.
   - `4` *Eigen-Character* pertama menyumbang `6,6667%` (`PC1`), `5,6249%` (`PC2`), `4,9330%` (`PC3`), dan `4,0909%` (`PC4`).
   - Kurva *Cumulative Proportion of Variance Explained (CPVE)* menunjukkan: `80%` variansi diraih pada **`82` PC**, `85%` pada **`109` PC**, `90%` pada **`154` PC**, `95%` pada **`252` PC**.
   - Kriteria Kaiser (`eigenvalues > mean(var(X, axis=0)) = 0,087497`) mempertahankan **`275` PC** dari `4.096` dimensi, membuktikan $>93\%$ dimensi piksel mentah redundan secara linier.
3. **Uji Hipotesis Vektor Mean MANOVA (`run_manova_and_diagnostics`, $m = 20$ PC)**:
   - Di subruang `20` PC teratas (bebas kolinearitas), dibangun matriks hipotesis antar-kelas $\mathbf{H}$ dan matriks galat dalam-kelas $\mathbf{E}$:
     $$\mathbf{H} = \sum_{c=0}^{14} N_c (\bar{\mathbf{x}}_c - \bar{\mathbf{x}})(\bar{\mathbf{x}}_c - \bar{\mathbf{x}})^\top, \qquad \mathbf{E} = \sum_{c=0}^{14} (\mathbf{X}_c - \bar{\mathbf{x}}_c)^\top (\mathbf{X}_c - \bar{\mathbf{x}}_c)$$
   - Nilai eigen $\lambda_i$ dari $\mathbf{E}^{-1}\mathbf{H}$ (`np.linalg.pinv(E) @ H`) menghasilkan keempat statistik uji MANOVA:
     - **Wilks' Lambda**: $\Lambda = \prod_{i} \frac{1}{1 + \lambda_i} = \mathbf{8{,}8900 \times 10^{-5}}$ (sangat dekat ke nol -> tolak $H_0: \boldsymbol{\mu}_0 = \dots = \boldsymbol{\mu}_{14}$).
     - **Pillai's Trace**: $V = \sum_i \frac{\lambda_i}{1 + \lambda_i} = \mathbf{5{,}8418}$ (statistik paling tahan pelanggaran homogenitas kovarians).
     - **Hotelling-Lawley Trace**: $T = \sum_i \lambda_i = \mathbf{17{,}6700}$.
     - **Roy's Largest Root**: $\theta = \frac{\lambda_{\max}}{1 + \lambda_{\max}} = \mathbf{0{,}8096}$.
     - **Partial Eta Squared**: $\eta_p^2 = 1 - \Lambda^{1/\min(K-1, m)} = \mathbf{0{,}4864}$ (ambang efek besar standar = `0,14`).
4. **Uji Homogenitas Kovarians Box's M (`compute_box_m`) & Plot Q-Q Mahalanobis**:
   - Untuk menghindari *floating-point underflow* saat menghitung determinan matriks $20 \times 20$ atau $40 \times 40$, kode memakai log-determinan **`np.linalg.slogdet(S_c)`** (dengan *ridge* kecil `1e-8 * I` jika nyaris singular):
     $$M = (N - K)\ln|\mathbf{S}_{\text{pooled}}| - \sum_{c=0}^{14} (n_c - 1)\ln|\mathbf{S}_c|$$
   - Hasil dengan faktor koreksi Bartlett `c_factor`:
     - Pada **`m = 20` PC**: **$M = 214.542,08$**, $\chi^2 = 212.310,12$, $df = 2.940$, **$p = 0{,}0$**.
     - Pada **`m = 40` PC**: **$M = 535.061,64$**, $\chi^2 = 524.045,55$, $df = 11.480$, **$p = 0{,}0$**.
   - Plot Q-Q jarak kuadrat Mahalanobis $d_i^2 = (\mathbf{x}_i - \hat{\boldsymbol{\mu}}_{y_i})^\top \mathbf{S}_{\text{pooled}}^{-1} (\mathbf{x}_i - \hat{\boldsymbol{\mu}}_{y_i})$ melawan kuantil teoretis $\chi^2_{df=20}$ juga memperlihatkan ekor berat yang menyimpang dari normalitas multivariat.
5. **Konsekuensi Empiris pada Klasifikator Klasik (5-Fold Unseen Writer CV)**:
   - **`Two-Stage PCA (100) + Fisher LDA`** (`Sigma` tunggal): **`85,13%`** ($\pm 2,00\%$; per-fold: `82,38%`, `88,05%`, `85,48%`, `83,52%`, `86,24%`).
   - **`Two-Stage PCA (100) + Logistic Regression`**: **`91,08%`** ($\pm 1,68\%$).
   - **`Two-Stage PCA (40) + Regularized QDA`** (`reg_param = 0,1`, $\hat{\boldsymbol{\Sigma}}_k = 0{,}9\mathbf{S}_k + 0{,}1\mathbf{S}_{\text{pooled}}$): **`96,09%`** ($\pm 0,98\%$; per-fold: `94,62%`, `97,29%`, `96,24%`, `95,38%`, `96,90%`).
   - **Mengapa QDA memakai `m = 40` PC sedangkan LDA memakai `m = 100` PC?** Setiap matriks kovarians kelas di QDA butuh estimasi $\frac{m(m+1)}{2}$ parameter bebas. Di `m = 100`, tiap kelas perlu menaksir `5.050` parameter pada `~560` sampel latih per fold (`p > n_c` lokal!). Di `m = 40`, parameternya menyusut menjadi `820` dan distabilkan oleh `reg_param = 0,1`, sehingga unggul `+10,96%` atas LDA.

### 2.3 Jembatan 4 Masalah Statistik ADM ke Solusi Deep Learning

| Masalah Statistik Era 0 | Bukti Numerik ADM | Solusi Arsitektural Modern & Kode Implementasi |
| :--- | :--- | :--- |
| 1. Kolinearitas ekstrem piksel `4.096-D` & sensitif translasi PCA | PCA butuh `109-252` komponen; PCA berhenti di `96,09%` | Ekstraktor konvolusi hierarkis nonlinier `DenseNet-121 CBAM` ($\mathbb{R}^{4096} \to \mathbb{R}^{256}$) menjaga tepi mikro via konkatenasi fitur |
| 2. Separabilitas mean antar-15 kelas (`H` vs `E`) | MANOVA $\Lambda = 8{,}89\times 10^{-5}$, $\eta_p^2 = 0{,}4864$ | Penarikan margin pada hipersfer satuan $\mathbb{S}^{255}$ melalui `CRD` (InfoNCE) dan `RKD` (jarak & sudut) |
| 3. Heterogenitas matriks kovarians & bimodalitas bentuk | Box's M $M = 214.542,08$ ($p=0$); QDA `+10,96%` > LDA | **`Sub-Center ArcFace (K = 2)`**: memberi 2 vektor sub-pusat per kelas (satu untuk Kaishu formal, satu untuk kursif/tradisional) |
| 4. Pergeseran gaya penulis & frekuensi kemunculan kelas | Gaya Baru absen kelas `0` & `14` (`45` false positives awal) | **`MixStyle` pasca-stem** (`p=0,5, alpha=0,1`) untuk gaya penulis + **`Bayesian EM TTPA` (`alpha=1,05`)** untuk adaptasi prior tanpa label |

### 2.4 Evolusi Era I-IV: Mengapa Harus Beralih ke DenseNet-CBAM, Kurikulum NCCU, TTPA, dan Akhirnya Meninggalkan MILP?

1. **Mengapa ResNet-18 & ConvNeXt-Femto Rontok di Gaya Baru?**
   - Di Fase 1 (`2.466` citra = `1.500` Chinese MNIST formal penulis 71-80 + `966` kaligrafi Gaya Baru semi-kursif/kursif), `Sub 1` turun ke `91,20%` dan `Sub 3` (`85% ResNet-18 + 15% ConvNeXt-Femto`) baru meraih **`93,19%`** (meskipun Chinese MNIST sudah `1.500/1.500 = 100%`).
   - Pada `966` citra Gaya Baru, akurasi kelas tersulit masih rendah pada `Sub 3`: Kelas `5` (`五`) = `51,0%`, Kelas `11` (`百`) = `31,0%`, Kelas `13` (`万`) = `64,9%`, Kelas `4` (`四`) = `73,0%`, Kelas `12` (`千`) = `73,3%`.
   - Penyebab arsitektural: di ResNet (`x + F(x)`), penjumlahan antar-blok berisiko melarutkan goresan tipis `1-2` piksel (seperti coretan miring atas pada `千` agar tidak tampak seperti `十`).
2. **Obat Morfologi & Kurikulum Data (`lecturer_exact_filter.py` & `biomechanical_augmentation.py`)**:
   - `biomechanical_augmentation.py`: mengaplikasikan *Simard Elastic Distortion* ($\alpha \in [6,0, 10,0]$, $\sigma \in [3,5, 4,5]$ dengan `cv2.remap` interpolasi bilinear), *DynamicBrushMorphology* (kernel diagonal $45^\circ$, $135^\circ$, silang, dan $2\times 2$ untuk meniru tekanan kuas tebal dan tinta kering *Feibai*), serta `TargetedStrokePerturbation`.
   - `lecturer_exact_filter.py`: menyelaraskan **`2.952` citra kursif eksternal NCCU** memakai `apply_lecturer_exact_filter()`: konversi grayscale -> normalisasi polaritas (jika sudut kanvas `median > 120`, balik `255 - img` jadi putih di latar hitam) -> mask ambang `> 25` -> crop *tight bounding-box* -> padding bujur sangkar mempertahankan rasio aspek + margin `8%` -> `cv2.resize(..., INTER_AREA)` ke `64x64` -> `cv2.GaussianBlur(..., (3,3), sigma=0.5)`. Penambahan data kursif terfilter inilah yang memecahkan plafon `94,73%` menjadi **`97,57%` (`Sub 25`)**.
3. **Bayesian EM Test-Time Prior Adaptation (`src/test_time_prior_adaptation.py`)**:
   - Di Fase 1 (`2.466` citra), `1.500` Chinese MNIST punya masing-masing `100` sampel untuk Kelas `0` dan Kelas `14`, sedangkan di `966` Gaya Baru kedua kelas itu `0` sampel. Akibat prior seragam `1/15`, `Champion_Blend_85_15` memprediksi Kelas `0` sebanyak `112` kali (`+12` di atas 100) dan Kelas `14` sebanyak `133` kali (`+33` di atas 100).
   - Alih-alih melakukan kecurangan *hardcoding* peluang nol, `TestTimePriorAdapter(alpha_dirichlet=1.05)` memperkirakan prior populasi target $\boldsymbol{\pi}$ tanpa label via EM:
     - **Inisialisasi**: $q_{i,k} = \frac{P_s(y=k \mid \mathbf{x}_i)}{P_s(y=k)}$, $\pi_k^{(0)} = 1/15$, pseudo-count $\gamma = \alpha_{\text{Dir}} - 1 = 0{,}05$.
     - **E-step**: tanggung jawab posterior $w_{i,k}^{(t)} = \frac{\pi_k^{(t)} q_{i,k}}{\sum_{j=0}^{14} \pi_j^{(t)} q_{i,j}}$.
     - **M-step**: pembaruan MAP Dirichlet $\pi_k^{(t+1)} = \frac{\sum_{i=1}^N w_{i,k}^{(t)} + \gamma}{N + K\gamma}$.
   - Karena $\gamma = 0{,}05 > 0$, prior tidak pernah kolaps ke nol sehingga aman 100% saat Kelas `0` dan `14` hadir kembali di Fase 2. Pada `Champion_Blend_85_15` Fase 1, EM konvergen dalam **`20` iterasi (`5,51 ms`)**, menekan Kelas `14` (`133 -> 122`), Kelas `0` (`112 -> 110`), dan memulihkan **`23` prediksi Kelas `4` (`四`, `180 -> 203`)**.
4. **Tomografi MILP (`Sub 30 = 99,76%`) & Alasan Berpindah ke Distilasi**:
   - Pada `src/synthesize_sub30_grandmaster_anchored_fusion.py`, seluruh `2.466` citra dikelompokkan berdasarkan vektor prediksi historis lintas `22` submisi resmi menjadi `G` kelas ekuivalensi (`unique_trajs`, `n_vars = 30 * G` variabel bilangan bulat $C[g,c]$ dan $B[g,c]$). Kendala persamaan dibentuk dari ukuran grup (`G`), total kelas (`30`), dan seluruh sel matriks konfusi historis (`22 x (15 + 195) = 4.620`), lalu diselesaikan oleh `scipy.optimize.milp` (HiGHS) untuk memaksimalkan $\log p_{\text{master}}$.
   - Audit ini berhasil membongkar bug `val_00459` & kontaminasi `chinese_mnist_exact_1500.csv` pada Sub 29, serta mencetak **`Sub 30 = 99,76%`** (`2.460 / 2.466`).
   - **Mengapa tidak dipakai untuk Fase 2?** (1) MILP butuh matriks konfusi historis dari dataset target (transduktif, mustahil dilakukan pada `2.902` citra buta Fase 2 dengan kuota hanya `6` submit); (2) Ensemble guru pra-distilasi berat (`3,0 GB` pool riset, `444,55 MiB` 15-model adaptasi, beban `135x` *forward pass* per citra).

### 2.5 Arsitektur Final `StudentDenseNet121_CBAM`, Aliran Tensor & 5-Loss `UnifiedDistillationLoss`

Berikut alur dimensi tensor nyata di `StudentDenseNet121_CBAM` (`Notebook 09`, `src/evaluate_student_vs_ensemble.py`, dan `src/predict_phase2_distillation.py`):

| Tahap Layer (`StudentDenseNet121_CBAM`) | Operasi Utama | Ukuran Tensor Keluaran `[B, C, H, W]` | Fungsi Fisik pada Karakter `64 x 64` |
| :--- | :--- | :--- | :--- |
| **Input Citra** | Grayscale norm `(img / 127.5) - 1.0` | `[B, 1, 64, 64]` | Tinta putih `[-1, 1]` resolusi penuh |
| **Custom Stem** | `Conv2d(1->64, 3x3, s=1, p=1) + BN + ReLU` | `[B, 64, 64, 64]` | Menjaga resolusi `64x64` utuh tanpa MaxPool awal |
| **MixStyle** (`train` only) | `p=0.5, alpha=0.1`, campur `mu` & `sigma` antar-batch | `[B, 64, 64, 64]` | Membiasakan model kebal terhadap ketebalan kuas antarpenulis |
| **DenseBlock 1** (`6` layer) + **CBAM 1** | Konkatenasi `k=32` + Atensi Kanal & Spasial `7x7` | `[B, 256, 64, 64]` | Menyimpan fitur tepi & sudut mikro (`64 + 6*32 = 256`) |
| **Transition 1** | `BN + ReLU + Conv 1x1 (256->128) + AvgPool 2x2` | `[B, 128, 32, 32]` | Kompresi kanal `theta=0,5` & reduksi spasial halus |
| **DenseBlock 2** (`12` layer) + **CBAM 2** | Konkatenasi `k=32` + `CBAMBlock(512)` | `[B, 512, 32, 32]` | Menangkap gabungan goresan lokal (`128 + 12*32 = 512`) |
| **Transition 2** | `Conv 1x1 (512->256) + AvgPool 2x2` | `[B, 256, 16, 16]` | Abstraksi sub-radikal menengah |
| **DenseBlock 3** (`24` layer) + **CBAM 3** | Konkatenasi `k=32` + `CBAMBlock(1024)` | `[B, 1024, 16, 16]` | Menangkap ligatur kursif menyeluruh (`256 + 24*32 = 1024`) |
| **Transition 3** | `Conv 1x1 (1024->512) + AvgPool 2x2` | `[B, 512, 8, 8]` | Pemadatan representasi global |
| **DenseBlock 4** (`16` layer) + **CBAM 4** | Konkatenasi `k=32` + `CBAMBlock(1024)` | `[B, 1024, 8, 8]` | Fitur semantik tingkat tinggi (`512 + 16*32 = 1024`) |
| **Final Pool & Embed** | `BN + ReLU + AdaptiveAvgPool2d(1,1) -> Linear(1024->256) -> BN1d -> PReLU` | **`[B, 256]`** | Vektor fitur manifold laten $\mathbf{f}_S \in \mathbb{R}^{256}$ |
| **Linear Head + SubCenterArcFace** | `fc_linear(256->15)` & `K=2, s=32.0, m=0.30` | **`[B, 15]`** | Prediksi akhir: `logits = 0.5 * z_lin + 0.5 * z_arc` |

*(Catatan teknis jujur apabila ditanya perbedaan skrip lokal [`src/model_densenet_cbam.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/model_densenet_cbam.py) vs arsitektur murid final `Notebook 09` / [`src/evaluate_student_vs_ensemble.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/evaluate_student_vs_ensemble.py): pada prototipe Era II `src/model_densenet_cbam.py`, `CBAMBlock` ditempatkan di dalam blok transisi dan proyeksi memakai `Linear -> ReLU`, sedangkan pada model final `StudentDenseNet121_CBAM` di `Notebook 09` `CBAMBlock` diletakkan langsung seusai tiap `DenseBlock` dan proyeksinya ditingkatkan ke `Linear -> BatchNorm1d -> PReLU` yang jauh lebih stabil bagi norma fitur ArcFace.)*

**Rincian Matematis 5 Komponen `UnifiedDistillationLoss` ([`src/distillation_loss.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/distillation_loss.py))**:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \alpha \mathcal{L}_{\text{KD}} + \beta_{\text{CRD}} \mathcal{L}_{\text{CRD}} + \gamma_{\text{dist}} \mathcal{L}_{\text{RKD-dist}} + \delta_{\text{angle}} \mathcal{L}_{\text{RKD-angle}}$$
1. **$\mathcal{L}_{\text{task}}$ (`Label Smoothing = 0,05`)**: `CrossEntropyLoss(label_smoothing=0.05)` pada fusi logit murid, menjaga akurasi terhadap label keras Super-Dataset `15.918` citra.
2. **$\mathcal{L}_{\text{KD}}$ (`alpha = 0,70`, suhu `T = 3,5`)**: `KLDivLoss(batchmean)(log_softmax(z_S / T), softmax(z_T / T)) * T^2`. Faktor $T^2 = 12,25$ menyeimbangkan besaran gradien saat logit diperlunak oleh $T = 3,5$, mentransfer distribusi kemiripan antarkarakter dari `MeanTeacherEMA`.
3. **$\mathcal{L}_{\text{CRD}}$ (`beta_crd = 0,10`, `tau = 0,07`)**: Fitur guru & murid dinormalisasi ke hipersfer satuan $\mathbb{S}^{255}$ ($\hat{\mathbf{f}} = \mathbf{f} / \|\mathbf{f}\|_2$), lalu dihitung matriks kemiripan `[B, B]`: $\mathbf{M} = (\hat{\mathbf{F}}_S \hat{\mathbf{F}}_T^\top) / \tau$, dan dioptimasi lewat `cross_entropy(M, arange(B))` (InfoNCE) agar fitur gambar ke-$i$ murid tertarik ke guru ke-$i$ dan menjauhi $B-1$ gambar negatif lain.
4. **$\mathcal{L}_{\text{RKD-dist}}$ (`gamma_dist = 1,00`)**: Menghitung jarak Euclidean berpasangan antar-sampel dalam batch `ds[i,j] = ||f_Si - f_Sj||_2`, membuang diagonal (`~eye(B)`), menormalkan dengan mean jarak off-diagonal batch (`ds_off / mu_s`, `dt_off / mu_t`) agar tidak tergantung skala absolut, lalu menyamakannya dengan `F.smooth_l1_loss`.
5. **$\mathcal{L}_{\text{RKD-angle}}$ (`delta_angle = 2,00`)**: Membentuk vektor arah ternormalisasi $\mathbf{t}_{i,j} = (\mathbf{f}_i - \mathbf{f}_j)/\|\mathbf{f}_i - \mathbf{f}_j\|_2$, menghitung produk dalam kosinus sudut triplet via `torch.bmm(td_s, td_s.transpose(1,2))` (`[B, B, B]`), membuang seluruh triplet degeneratif (`i == j`, `k == j`, `i == k`), lalu menyamakan sudut guru vs murid dengan `F.smooth_l1_loss`.

**Guru Online `MeanTeacherEMA` + `SWA` + `FP16`**:
- `MeanTeacherEMA` (`alpha_ema = 0,995`): memperbarui bobot guru tiap iterasi lewat $\theta_T \leftarrow 0,995\,\theta_T + 0,005\,\theta_S$ tanpa *backpropagation*, sehingga guru menghasilkan target $z_T, \mathbf{f}_T$ bebas osilasi dari citra bersih `x_clean`.
- `SWA` (Epoch `17-25`, `swa_lr = 1e-4`): merata-ratakan parameter `AveragedModel(student)` dan memanggil `update_bn` untuk menyelaraskan statistik BatchNorm di dasar lembah kurva loss yang datar (*flat minima*).
- Hasil checkpoint [`densenet121_cbam_student_fp16.pt`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/models/densenet121_cbam_student_fp16.pt) berukuran **`14,75 MiB`** (`7.522.968` parameter), prediksi **100% identik** dengan FP32 `29,28 MiB`, meraih **`0,9984` (`Sub 32`, `4` error)** di Fase 1 dan **`0,9862` (`5,44 detik`)** pada `2.902` citra blind test Fase 2 (`0` duplikat, `0` MD5 overlap), serta mempertahankan stabilitas **`95,98% ± 0,02%`** pada noise Gaussian $\sigma = 25$ (unggul `+1,61%` atas checkpoint Epoch 13 `densenet121_cbam_student_best.pt`).

---

## Bagian 3: Bedah Kode Rinci Semua 10 Notebook (`01` s.d. `10`)

Seluruh 10 notebook berada di folder [`FinalProject/notebooks/`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/notebooks/README.md) dan tersinkron di GitHub (`https://github.com/Reiii0-0/MNIST-ADM/tree/main/notebooks`). Notebook `01` s.d. `09` menyematkan `execution_count` dan log `stdout` asli Kaggle GPU Tesla T4, sedangkan Notebook `10` menyematkan output eksekusi lokal berikut `7` figur `image/png`.

### 3.1 `01_mnist_adm_resnet18_cbam.ipynb` (`43` sel: `22` markdown, `21` kode | `2` sel berisi `57` baris stdout)
- **Asal Kaggle & Peran**: `reiiihel/mnist-adm` (Era I - Baseline Kuat & Uji Domain Shift awal).
- **Susunan Sel Utama**:
  - `C02-C06`: Imports, `seed_everything(42)`, setup `DEVICE`, mencetak info `2 x Tesla T4`.
  - `C08-C10`: `discover_and_extract_data()` memindai `/kaggle/input` (`train`, `train_synthetic`, `splits_kfold.csv`, `train_synthetic_labels.csv`).
  - `C12-C18`: `BiomechanicalAugmentor` dan `ChineseMNISTDataset` (normalisasi `[-1, 1]`).
  - `C20-C22`: Arsitektur **`ResNet18_CBAM`** (stem `3x3 stride 1`, 4 stage `BasicBlockCBAM` berkanal `(64, 128, 256, 512)`, `MixStyle(p=0.5, alpha=0.1)`, proyeksi `512 -> 256`) + `SubCenterArcFace(in_features=256, num_classes=15, sub_centers=2)`.
  - `C24-C26`: Kelas `TestTimePriorAdapter(alpha=1.05)` (implementasi EM TTPA mandiri di dalam kernel).
  - `C28-C30`: `train_and_evaluate_folds(epochs=15, batch_size=64, lr=5e-4)` memakai `AdamW(weight_decay=1e-4)`, melatih 5-Fold `StratifiedGroupKFold` (`9.100` train + `2.100` val per fold) dan menyimpan `resnet18_cbam_fold0..4.pt`.
  - `C32-C38`: `apply_adabn_to_model` (rekalibrasi BatchNorm test-time), `tta_variants`, `run_test_inference`, dan `certify_and_package_codabench` (pemeriksa integritas CSV/ZIP).
  - `C40-C42`: `main()` menjalankan pelatihan lengkap 5-Fold.
- **Output Log Tertanam (`C06`, `C42`)**: Fold 1-5 mencapai puncak validasi hingga Fold 5 = `99,90%`, dengan **`OVERALL 5-FOLD OUT-OF-FOLD ACCURACY: 99.80%`** (catatan: versi run awal 13 Okt di `models/kaggle_outputs/` menghasilkan OOF `99,79%` dan file `oof_probabilities.npy`).

### 3.2 `02_mnist_adm_convnext_femto.ipynb` (`12` sel: `6` markdown, `6` kode | `3` sel berisi `124` baris stdout)
- **Asal Kaggle & Peran**: `reiiihel/mnist-adm-convnext` (Era I - Arsitektur Makro Berkernel `7x7` untuk Blend Rank 1 `Sub 3`).
- **Susunan Sel & Kode**:
  - `C01-C05`: Setup seed `42`, deteksi `Tesla T4` (`15.64 GB`), ekstraksi dataset (`10.500` train + `700` sintetik = `11.200` file PNG di `/tmp/data`).
  - `C07`: Arsitektur **`ConvNeXt_Femto`** (`LayerNorm2d`, `ConvNeXtBlock` berisi `7x7 depthwise conv -> LayerNorm2d -> Conv 1x1 4x expand -> GELU -> Conv 1x1 -> LayerScale gamma=1e-6`, stage `dims=(48, 96, 192, 384)` dengan `depths=(2, 2, 4, 2)`), plus `SubCenterArcFace(256, 15, sub_centers=2, s=32.0, m=0.30)`.
  - `C09`: `elastic_transform_np(alpha=12, sigma=3)`, `augment_character`, `ChineseCharacterDataset`, serta `train_and_eval_fold(epochs=15, batch_size=64)` dengan optimizer `AdamW(lr=1e-3)`.
  - `C11`: Loop eksekusi 5-Fold dan agregasi probabilitas OOF.
- **Output Log Tertanam (`C03`, `C05`, `C11`)**: `CONVNEXT-FEMTO 5-FOLD OVERALL OOF ACCURACY: 99.55%` (per-fold: `99,57%`, `99,71%`, `99,67%`, `99,10%`, `99,71%`). Ketika di-blend `15%` dengan ResNet-18 `85%`, merebut Rank 1 pertama (`Sub 3: 0,9319`).

### 3.3 `03_mnist_adm_densenet121_cbam.ipynb` (`14` sel: `7` markdown, `7` kode | `2` sel berisi `141` baris stdout)
- **Asal Kaggle & Peran**: `reiiihel/mnist-adm-densenet121` (Era II - Arsitektur DenseNet-121 CBAM 5-Fold).
- **Susunan Sel & Kode**:
  - `C01-C05`: `seed_everything(42)` + `setup_datasets()` mengindeks `19.666` file (dari `chinese-mnist-adm` + `chinese-mnist-calligraphy-v2` + `2.466` val).
  - `C07-C09`: Kelas augmentasi terstruktur `ElasticDistortion`, `DynamicBrushMorphology`, `TargetedStrokePerturbation`, `BiomechanicalAugmentationPipeline`, dan `BiomechanicalChineseDataset`.
  - `C11`: **`DenseNet121_CBAM`** custom (`growth_rate=32`, `block_config=(6, 12, 24, 16)`, `_DenseLayer` 1x1->3x3, `_DenseBlock`, `_Transition` dengan `CBAM` di dalamnya, `MixStyle(p=0.4, alpha=0.1)`) + `SubCenterArcFace(256, 15, sub_centers=2, s=32.0, m=0.30)`.
  - `C13`: `train_densenet_pipeline(epochs=15, batch_size=64, lr=5e-4)` + inferensi 5-Way TTA pada `2.466` citra Fase 1.
- **Output Log Tertanam (`C03`, `C13`)**: **`Mean OOF Accuracy: 99.76%`** (`[99.71, 99.81, 99.81, 99.71, 99.76]`), ekspor `oof_probabilities_densenet121_cbam_v2.npy`, `probabilities_densenet121_cbam_v2_5fold.npy`, dan `submission_phase1_densenet121_cbam_v2.zip`.

### 3.4 `04_mnist_adm_sota_v3_biomechanical.ipynb` (`14` sel: `7` markdown, `7` kode | `2` sel berisi `159` baris stdout)
- **Asal Kaggle & Peran**: `reiiihel/mnist-adm-sota-v3` (Era II - Pipeline v3 dengan Augmentasi Biomekanik & Calligraphy v2).
- **Susunan Sel & Kode**:
  - `C01-C05`: Setup & `setup_datasets()` mengindeks `39.332` entri path gambar dari `splits_kfold.csv`, `train_synthetic_labels.csv`, dan `train_synthetic_calligraphy_v2.csv`.
  - `C07`: `apply_elastic_distortion(alpha=6.0, sigma=3.5)` dan `apply_anisotropic_brush_erosion`.
  - `C09-C11`: `DenseNet121_CBAM` (`growth_rate=32`, `block_config=(6, 12, 24, 16)`, `CBAM`, `MixStyle(p=0.4, alpha=0.1)`) + `SubCenterArcFace(256, 15, sub_centers=2, s=32.0, m=0.30)`.
  - `C13`: `train_sota_v3_pipeline(epochs=15, batch_size=64, lr=5e-4)` + 5-Way TTA.
- **Catatan Jujur**: Judul markdown `C00` di kernel menyebut `ResNet-18 CBAM` secara historis, tetapi kode sel `C09` sebenarnya menjalankan `DenseNet121_CBAM`.
- **Output Log Tertanam (`C03`, `C13`)**: Ekspor `probabilities_sota_v3_5fold.npy` + paket `sub_sota_v3.zip` (`2.466` prediksi lengkap dengan distribusi Kelas `0..14`).

### 3.5 `05_mnist_adm_fixmatch.ipynb` (`22` sel: `11` markdown, `11` kode | `2` sel berisi `147` baris output)
- **Asal Kaggle & Peran**: `reiiihel/mnist-adm-fixmatch` (Era III Track 2 - Semi-Supervised Consistency Regularization pada ResNet-18).
- **Susunan Sel & Kode**:
  - `C05`: `EMBEDDED_PSEUDO_B64` (konsensus pseudo-label konfidensi tinggi terkompresi base64 agar kernel mandiri).
  - `C07-C11`: `setup_datasets()`, `apply_weak_augmentation` (geser/rotasi halus), `apply_strong_augmentation` + `apply_elastic_distortion(alpha=6.0, sigma=3.5)`, `SupervisedDataset`, dan `FixMatchDualDataset` (mengembalikan pasangan `(img_weak, img_strong)` untuk sampel uji tanpa label).
  - `C13`: `ResNet18_CBAM` (`MixStyle(p=0.5, alpha=0.1)`) + `SubCenterArcFace(256, 15, sub_centers=2)`.
  - `C15-C19`: Tahap 1 `get_or_train_base_model(epochs=10, batch_size=64, lr=5e-4)` (`Val Acc ~99,67-99,90%`); Tahap 2 `run_fixmatch_fine_tuning(epochs=4, batch_size=64, lr=1e-5)` mengoptimasi `Sup_Loss + 0.5 * Consistency_Loss`; `get_9way_tta_variants`.
- **Catatan Audit Jujur pada Output (`C03`, `C21`)**: Eksekusi cloud selesai penuh pada **Fold 1 s.d. Fold 4** (menyimpan `resnet18_cbam_fixmatch_fold0..3.pt` dengan Joint Loss turun ke `0,3934-0,4008`), dan dihentikan pada **Fold 5/5 Base Epoch 3/10** (`CANCEL_ACKNOWLEDGED`, disertai catatan resmi pada output `stderr`) saat tim beralih memprioritaskan `06` STM dan `07` ConvNeXt-FixMatch.

### 3.6 `06_mnist_adm_stm_adaptation.ipynb` (`16` sel: `8` markdown, `8` kode | `2` sel berisi `122` baris stdout)
- **Asal Kaggle & Peran**: `reiiihel/mnist-adm-stm-adaptation` (Era III Track 1 - Style Transfer Mapping oleh Bengio/Zhang TPAMI 2017 -> `Sub 16 = 0,9469`).
- **Susunan Sel & Kode**:
  - `C07-C09`: Augmentasi biomekanik (`alpha=5.0, sigma=3.0`) + definisi kedua arsitektur `ResNet18_CBAM` (`MixStyle(p=0.4, alpha=0.1)`) dan `ConvNeXt_Femto`.
  - `C11`: `find_or_train_checkpoints` (memakai `AdamW lr=1e-3` untuk ResNet, `lr=8e-4` untuk ConvNeXt jika checkpoint belum terpasang).
  - `C13`: Mesin analitik tanpa gradien: `compute_class_prototypes` menghitung mean vektor fitur 256-D $\boldsymbol{\mu}_k$ per kelas dari data latih; `solve_closed_form_stm(beta=50.0, gamma=1.0)` memecahkan persamaan aljabar linier kuadrat tertutup untuk matriks rotasi/skala $\mathbf{A} \in \mathbb{R}^{256 \times 256}$ dan vektor pergeseran $\mathbf{b} \in \mathbb{R}^{256}$ lalu memperbarui `z' = A*z + b` selama beberapa iterasi *self-training*.
  - `C15`: `run_stm_self_training_pipeline` & `main()`.
- **Output Log Tertanam (`C03`, `C15`)**: Mengekspor `submission_stm.csv`, `sub_stm.zip` (`7.377` byte), `probabilities_stm.npy`, dan `stm_matrices.npz`.

### 3.7 `07_mnist_adm_convnext_fixmatch.ipynb` (`22` sel: `11` markdown, `11` kode | `2` sel berisi `197` baris stdout)
- **Asal Kaggle & Peran**: `reiiihel/mnist-adm-convnext-fixmatch` (Era III Track 3 - Cabang Konvolusi Makro `ConvNeXt-Femto` dengan FixMatch).
- **Susunan Sel & Kode**:
  - Strukturnya paralel dengan `05`, namun memakai arsitektur **`ConvNeXt_Femto`** (`LayerNorm2d`, `ConvNeXtBlock` `7x7`, `SubCenterArcFace(256, 15, sub_centers=2)`).
  - `C15-C19`: `get_or_train_base_model(epochs=10, batch_size=128, lr=5e-4)`, `run_fixmatch_fine_tuning(epochs=4, batch_size=128, lr=1e-5)` (`0.5 * L_cons`), serta `get_9way_tta_batch` / `evaluate_batched_9way_tta_ensemble` yang memproses 9 varian TTA secara *batched*.
- **Output Log Tertanam (`C03`, `C21`)**: Berbeda dari `05`, `07` **selesai utuh seluruh 5 Fold** hingga mengekspor `sub_convnext_fixmatch.zip` dan `probabilities_convnext_fixmatch_5fold.npy` untuk `2.466` citra.

### 3.8 `08_mnist_adm_massive_sota.ipynb` (`22` sel: `11` markdown, `11` kode | `4` sel berisi `134` baris stdout)
- **Asal Kaggle & Peran**: `reiiihel/mnist-adm-massive-sota` (Era IV - Eksperimen Skala Besar Pra-Distilasi pada `15.811` Citra).
- **Susunan Sel & Kode**:
  - `C03-C09`: Konstanta `SEED=42`, `BATCH_SIZE=64`, `NUM_EPOCHS=20`, `LEARNING_RATE=3e-4`, `WEIGHT_DECAY=1e-4`. Ekstraksi dinamis mengindeks `22.618` citra dan menyusun manifest **`15.811` sampel latih** (`10.500` CM Train + `2.952` NCCU Cursive + `2.359` Consensus Pseudo-Labels).
  - `C11-C15`: `BiomechanicalAugmenter`, `MassiveHCCRDataset`, serta **`DenseNet121_CBAM_64`** (menggunakan `models.densenet121(weights=None)` dengan modifikasi stem `3x3 stride 2`, CBAM, dan **`feature_dim=512`**) + **`SubCenterArcFace(in_features=512, out_features=15, K=3, s=32.0, m=0.35)`**.
  - `C17-C21`: `CosineAnnealingLR(T_max=20, eta_min=1e-6)`, `save_checkpoint` & `try_load_latest_checkpoint` tiap 2 epoch agar kebal timeout Kaggle, serta `generate_submission`.
- **Catatan Kode Penting**: Banner print log di `C21` bertulis `"DenseNet-161 CBAM"` dan `"7.80 Million Parameters"`, tetapi definisi kelas aktual di sel `C13` & `C15` menginstansiasi **`DenseNet121_CBAM_64`** dengan `feature_dim=512` dan `K=3` sub-pusat.
- **Output Log Tertanam (`C03`, `C07`, `C09`, `C21`)**: Selesai penuh `20` epoch (`Epoch 20 Train Acc: 99,89%`, `39,7s/epoch`), mengekspor `val_probabilities_latest.npy` (dipakai menjadi salah satu komponen `p_massive` berbobot `0,35` pada MILP Sub 30!) dan `submission_massive_sota.zip`.

### 3.9 `09_mnist_adm_distillation_final.ipynb` (`20` sel: `10` markdown, `10` kode | `3` sel berisi `59` baris stdout)
- **Asal Kaggle & Peran**: `reiiihel/mnist-adm-distillation` (Era V-VI - **Pipeline Utama Model Juara Tunggal Final `14,75 MiB FP16`**).
- **Susunan Sel & Kode**:
  - `C03-C07`: `SEED=42`, `BATCH_SIZE=64`, `NUM_EPOCHS=25`, `LEARNING_RATE=5e-4`, `WEIGHT_DECAY=1e-4`; memuat `PHASE1_LABELS` (`2.466` label terverifikasi Sub 31), lalu menyusun **Master Super-Dataset `15.918` sampel** (`10.500` CM + `2.952` NCCU + `2.466` Fase 1).
  - `C09`: `BiomechanicalAugmenter` dan `HCCRSuperDataset` (mengembalikan `x_clean` untuk guru dan `x_aug` untuk murid).
  - `C11`: **`DenseNet121_CBAM`** (custom stem `3x3 stride 1`, `growth_rate=32`, `block_config=(6, 12, 24, 16)`, `MixStyle(p=0.5, alpha=0.1)`, 4 CBAM, `256-D` projection) + **`SubCenterArcFace(in_features=256, num_classes=15, sub_centers=2)`**.
  - `C13`: **`UnifiedDistillationLoss(T=3.5, alpha=0.70, beta_crd=0.10, gamma_dist=1.00, delta_angle=2.00, tau=0.07, label_smoothing=0.05)`** (keseluruhan 5 loss seperti dibedah pada Bagian 2.5).
  - `C15`: **`MeanTeacherEMA(student, metric_head, alpha_ema=0.995)`** memperbarui parameter guru tanpa gradien.
  - `C17`: **`EMPriorShiftAdapter`** (adaptasi prior Bayesian EM saat evaluasi).
  - `C19`: `main()` menjalankan 25 epoch dengan AMP `GradScaler`, `CosineAnnealingLR` di Epoch 1-16, lalu masuk fase **SWA** pada `swa_start = int(25 * 0.70) = 17` sampai Epoch 25 (`SWALR(optimizer, swa_lr=1e-4)`), diakhiri `update_bn`, penyimpanan **`checkpoint_student_fp16.pt (14.75 MB)`**, dan evaluasi TTPA.
- **Output Log Tertanam (`C03`, `C07`, `C19`)**: Epoch 1 (`Train Acc 76,24% | Val 95,01%`) -> Epoch 13 menyentuh `100,00% (2466/2466)` (momen tersimpannya `densenet121_cbam_student_best.pt` FP32 `29,28 MiB`) -> Epoch 17-25 fase SWA -> tersimpannya model final FP16 `14,75 MiB` (`Sub 32 = 99,84%` Fase 1 & `P2-Sub1 = 98,62%` Fase 2).

### 3.10 `10_mnist_adm_visual_evaluation_and_diagnostics.ipynb` (`14` sel: `7` markdown, `7` kode | semua `7` sel kode memiliki output + **`7` gambar PNG tertanam**)
- **Peran**: Menyediakan bukti grafik visual dan audit tabel terpadu langsung di GitHub (melengkapi `01-09` yang tanpa overhead plotting).
- **Isi Eksekusi Cell `01-07`**:
  - `Exec 1`: Setup tema Royal Blue & membaca 4 berkas terverifikasi (`multivariate_diagnostics_summary.json`, `multivariate_benchmark_results.json`, `verified_empirical_metrics.json`, `master_all_submissions_audit.csv`).
  - `Exec 2` (**2 PNG**): Mencetak ringkasan PCA, MANOVA, Box's M, CV 5-Fold (`LDA 85,13%` vs `QDA 96,09%` vs `LogReg 91,08%`), memplot *Scree Top-16 + Bar CV*, serta menampilkan galeri `2x2` figur ADM asli (`scree_plot.png`, `eigen_characters.png`, `mahalanobis_qq_plot.png`, `lda_canonical_projection.png`).
  - `Exec 3` (**1 PNG**): Mencetak seluruh `28` baris audit `Sub 1-32 + Phase 2` dan memplot kurva progresi *Overall vs Chinese MNIST vs Gaya Baru + P2-Sub1*.
  - `Exec 4` (**1 PNG**): Mencetak tabel kemajuan `13` kelas aktif pada `966` citra Gaya Baru dan memplot perbandingan `Sub 1` -> `Sub 8` -> `Sub 26` (terobosan kelas `4, 5, 11, 12, 13`).
  - `Exec 5` (**1 PNG**): Memplot stress-test perturbasi (`Clean`, `Gaussian sigma=25`, `Erosi 2x2`, `Dilasi 2x2`, `Elastis`) dan bar horizontal kompresi memori `773,94 MiB -> 14,75 MiB`.
  - `Exec 6` (**1 PNG**): Memplot diagram reliabilitas kalibrasi Fase 1 (`Pre-KD ECE 5,18%` vs `Student SWA 0,00%`) serta log-histogram confidence `2.902` citra Fase 2 (`Raw` vs `+ EM TTPA`).
  - `Exec 7` (**1 PNG**): Menampilkan galeri `3x2` infografik metodologi dan arsitektur dari `outputs/presentation_figures/`.

---

## Bagian 4: Bedah Skrip Produksi Modular di `FinalProject/src/`

Jika dosen atau asdos meminta bukti implementasi terstruktur di luar notebook Kaggle, tunjukkan alur eksekusi modular di [`FinalProject/src/`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/config.py):

1. [`src/split_data.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/split_data.py): Membuat partisi `splits_kfold.csv` memakai `StratifiedGroupKFold(n_splits=5, random_state=42)` dan memverifikasi nol irisan `writer_id` antar-fold.
2. [`src/multivariate_analysis.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/multivariate_analysis.py): Mengeksekusi seluruh jalur ADM klasik (`PCA(300)` terpusat -> MANOVA $\mathbf{H}, \mathbf{E}$ pada `m=20` -> Box's M via `slogdet` pada `m=20` dan `m=40` -> CV 5-Fold `LDA`, `LogReg`, dan `Reg-QDA reg_param=0.1`) dan menghasilkan 4 figur `outputs/*.png` + `multivariate_diagnostics_summary.json`.
3. [`src/lecturer_exact_filter.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/lecturer_exact_filter.py) & [`src/biomechanical_augmentation.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/biomechanical_augmentation.py): Standarisasi morfologi kurasi eksternal (crop Otsu + margin `8%` + blur $\sigma=0,5$) dan simulasi otot-kuas kaligrafi (elastis Simard + erosi/dilasi kuas $45^\circ/135^\circ$).
4. [`src/models.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/models.py) & [`src/model_densenet_cbam.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/model_densenet_cbam.py): Definisi modular `MixStyle`, `CBAMBlock`, `ResNet18_CBAM`, `ConvNeXt_Femto`, `DenseNet121_CBAM`, dan `SubCenterArcFace`.
5. [`src/distillation_loss.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/distillation_loss.py) & [`src/test_time_prior_adaptation.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/test_time_prior_adaptation.py): Implementasi teruji `UnifiedDistillationLoss` dan `TestTimePriorAdapter` (`EM MAP Dirichlet`).
6. [`src/synthesize_sub30_grandmaster_anchored_fusion.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/synthesize_sub30_grandmaster_anchored_fusion.py): Solver Joint MILP HiGHS lintas 22 matriks konfusi historis CodaBench + fusi posterior `p_master` untuk rekonstruksi Fase 1 (`99,76%`).
7. [`src/evaluate_student_vs_ensemble.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/evaluate_student_vs_ensemble.py): Mesin benchmark kepala-ke-kepala (`StudentDenseNet121_CBAM` vs ResNet-18/ConvNeXt) untuk uji keandalan *Clean*, *Gaussian Noise $\sigma=25$*, *Erosion*, *Dilation*, dan *Elastic Distortion*.
8. [`src/predict_phase2_distillation.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/predict_phase2_distillation.py): Pipeline inferensi produksi akhir (mendukung folder maupun `.zip`), memuat `densenet121_cbam_student_fp16.pt` (`14,75 MiB`), menjalankan `1x forward pass` Batch-128 -> `0.5 * z_lin + 0.5 * z_arc` -> `TestTimePriorAdapter` memakai `source_counts` dari checkpoint -> mengekspor `predictions.csv` siap kirim dalam **`5,44 detik`**.
9. [`src/generate_detailed_infographics.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/generate_detailed_infographics.py) & [`src/generate_presentation_figures.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/generate_presentation_figures.py): Menghasilkan ke-`20` infografik dan grafik empiris di `outputs/presentation_figures/` bertema Canva Royal Blue (`#2254c5`).

---

## Bagian 5: Pemisahan Tiga Kelompok Akurasi (Wajib Hafal Saat Sidang)

| Jenis Evaluasi | Sampel & Isolasi Data | Skor Terukur | Posisi Klaim yang Sah di Depan Dosen |
| :--- | :--- | :---: | :--- |
| **OOF 5-Fold CV (`Pra-Fase`)** | `10.500` Kaishu, `70` penulis (`56` latih vs `14` uji via `StratifiedGroupKFold`) | LDA **`85,13%`**, LogReg **`91,08%`**, Reg-QDA **`96,09%`**, ResNet-18 **`99,79%`**, ConvNeXt **`99,55%`**, DenseNet v2 **`99,76%`** | Bukti validasi lokal bebas kebocoran penulis sebelum uji publik |
| **Fase 1 (`Development Benchmark`)** | `2.466` citra (`1.500` CM formal + `966` Gaya Baru kursif; kelas `0` & `14` = `0` di Gaya Baru) | Sub 1 `91,20%`, Sub 3 `93,19%`, Sub 25 `97,57%`, Sub 30 `99,76%`, **Sub 32 Student SWA `99,84%`** | Sampai Sub 30 adalah benchmark Fase 1 + MILP; untuk `Sub 32`, karena `2.466` citra Fase 1 masuk ke Super-Dataset `15.918`, statusnya **harus disebut jujur sebagai *development benchmark*** |
| **Fase 2 (`Sealed Unseen Blind Test`)** | `2.902` citra penulis baru (`0` duplikat internal, `0` irisan hash MD5 vs `15.918` citra) | **`0,9862`** (~`2.862 / 2.902` benar, **`5,44 detik`** Batch-128 RTX 4060, kuota `1/6`, `ID 1048381`) | **Bukti generalisasi OOD sejati** pada data yang benar-benar belum pernah dilihat model maupun MILP |

---

## Bagian 6: Bank 18 Pertanyaan Jebakan Dosen & Asdos (Jawaban Bulletproof)

### Kelompok A: Statistik & Matematika Multivariat Klasik (ADM)

1. **Tanya**: *"Mengapa matriks kovarians kalian singular padahal jumlah baris $N = 10.500$ lebih banyak dari kolom $p = 4.096$?"*
   - **Jawaban**: *"$N = 10.500$ memang lebih besar daripada $p = 4.096$, tetapi $\det(\mathbf{S}) \approx 0$ akibat dua hal: piksel di pinggiran kanvas `64x64` bernilai konstan nol hampir di semua citra sehingga variansinya nyaris nol, dan piksel bertetangga membentuk kolinearitas spasial yang ekstrem. Itulah alasan PCA wajib mereduksi dimensi dulu ke `20-100` komponen."*
2. **Tanya**: *"Di Box's M kovarians tidak homogen ($p=0$) dan di Q-Q plot Mahalanobis normalitas multivariat ditolak. Bukankah uji MANOVA kalian jadi tidak sah?"*
   - **Jawaban**: *"Tidak gugur karena dua alasan yang terverifikasi di `multivariate_analysis.py`. Pertama, jumlah sampel sangat besar dan seimbang (`700` per kelas) sehingga teorema batas pusat menstabilkan uji vektor mean; kedua, selain Wilks' Lambda ($\Lambda = 8{,}89\times 10^{-5}$), kami melaporkan **Pillai's Trace** (`5,8418`) yang paling kebal terhadap pelanggaran homogenitas kovarians. Penolakan Box's M justru membuktikan secara matematis mengapa QDA (`96,09%`) jauh melampaui LDA (`85,13%`)."*
3. **Tanya**: *"Mengapa pada klasifikasi QDA dipakai `m = 40` PC, pada LDA & Regresi Logistik dipakai `m = 100` PC, dan pada MANOVA dipakai `m = 20` PC?"*
   - **Jawaban**: *"Matriks kovarians simetris berdimensi $m \times m$ memiliki $\frac{m(m+1)}{2}$ parameter bebas per kelas. Untuk QDA yang mengestimasi `15` matriks kovarians terpisah pada `~560-600` sampel per fold, memakai `m = 100` berarti menaksir `5.050` parameter per kelas sehingga tidak stabil; dengan `m = 40` (`820` parameter) plus shrinkage `reg_param = 0,1`, QDA stabil dan memuncak di `96,09%`. Sementara Box's M pada `m = 20` dan `m = 40` dihitung via `np.linalg.slogdet` agar bebas underflow."*
4. **Tanya**: *"Apa arti fisik dari empat Eigen-Character pertama (`PC1-PC4`), dan kenapa basis PCA tidak cukup?"*
   - **Jawaban**: *"`PC1` (`6,67%`) s.d. `PC4` (`4,09%`) memisahkan intensitas pusat tinta serta kontras goresan datar vs vertikal. Tetapi basis PCA bersifat linier dan tidak invarian terhadap pergeseran beberapa piksel atau kelengkungan kursif Caoshu, sehingga skor proyeksi antarkelas tetap bertumpuk dan batas akurasinya berhenti di `96,09%`."*

### Kelompok B: Integritas Data, Leakage, TTPA & Etika Kompetisi

5. **Tanya**: *"Pada Notebook `09`, ke-`2.466` data Fase 1 (`PHASE1_LABELS`) dimasukkan ke dalam Super-Dataset `15.918` citra saat melatih Student. Itu data leakage, kan?"*
   - **Jawaban**: *"Betul apabila `0,9984` Fase 1 diklaim sebagai pengujian baru, Pak/Bu, dan karena itu di Slide 25 kami menyatakannya sebagai *in-sample development benchmark*. Bukti generalisasi sejati bebas kebocoran adalah pengujian **Fase 2 (`2.902` citra blind test penulis baru)** yang sudah diaudit memiliki `0` duplikat internal dan `0` overlap hash MD5 terhadap ke-`15.918` citra pelatihan, dengan skor **`0,9862`** sekali pass dalam `5,44 detik`."*
6. **Tanya**: *"Untuk apa memakai Tomografi MILP pada Sub 30 jika sifatnya memanfaatkan feedback matriks konfusi CodaBench?"*
   - **Jawaban**: *"MILP HiGHS pada Fase 1 memberi dua manfaat saintifik: menetapkan batas atas kesalahan di Fase 1 serta mengaudit dua bug sistem yang tersembunyi di data/skrip (`val_00459` dan kontaminasi masker `148` citra Kaishu di Sub 29). Namun karena MILP bersifat transduktif dan mustahil diterapkan di ujian buta Fase 2, teknik itu tidak dipakai di produksi akhir."*
7. **Tanya**: *"Di grafik TTPA kenapa Kelas 0 dan 14 jumlahnya masih `110` dan `122`, padahal katanya absen di Gaya Baru?"*
   - **Jawaban**: *"Total validasi Fase 1 adalah `2.466` citra = `1.500` Chinese MNIST formal (yang memang punya `100` sampel Kelas 0 dan `100` sampel Kelas 14) + `966` Gaya Baru (yang kelas 0 & 14-nya nol). Jadi kelebihan di atas 100 sebelum TTPA (`112` dan `133`, total `+45`) adalah tebakan palsu di Gaya Baru. EM TTPA (`alpha = 1,05`, `20` iterasi) menekannya menuju proporsi sesungguhnya dan menyelamatkan `23` prediksi Kelas `4` (`四`), tanpa meng-hardcode nol agar kelas `0` & `14` tetap otomatis terdeteksi di Fase 2."*
8. **Tanya**: *"Dari mana jaminan `5-Fold` validasi lokal benar-benar terisolasi penulisnya?"*
   - **Jawaban**: *"Di `src/split_data.py`, `StratifiedGroupKFold` mengelompokkan seluruh `10.500` sampel berdasarkan kolom `writer_id` (`70` penulis), dilengkapi `assert len(train_writers.intersection(val_writers)) == 0` untuk tiap fold (`56` penulis train vs `14` penulis val)."*

### Kelompok C: Arsitektur Deep Learning, Loss Distilasi & Kuantisasi

9. **Tanya**: *"Jelaskan beda posisi `CBAM` dan `fc_embed` pada `StudentDenseNet121_CBAM` dibanding `ResNet18_CBAM`!"*
   - **Jawaban**: *"`ResNet18_CBAM` menjumlahkan fitur dengan blok residual dan memproyeksikan `512 -> 256` lewat `ReLU`. Sebaliknya, `StudentDenseNet121_CBAM` menjaga resolusi `64x64` di stem `3x3 stride 1`, menyambung (concat) fitur bertahap lewat `growth_rate=32` (`256 -> 512 -> 1024 -> 1024` kanal), menempatkan 4 `CBAMBlock` sesudah tiap `DenseBlock`, lalu memadatkannya lewat `Linear(1024->256) -> BatchNorm1d(256) -> PReLU`."*
10. **Tanya**: *"Kenapa pakai dua kepala (`fc_linear` dan `SubCenterArcFace`) lalu logit-nya digabung `0.5 * z_lin + 0.5 * z_arc`?"*
    - **Jawaban**: *"`SubCenterArcFace (K=2, s=32, m=0,30)` membentuk geometri jarak sudut terpisah antara gaya balok dan gaya kursif pada bola hipersfer `256-D`, sementara `fc_linear` menjaga batas keputusan linear lembut untuk kalibrasi probabilitas softmax; penggabungan `50/50` menggabungkan ketajaman pemisahan metrik dan kestabilan kalibrasi."*
11. **Tanya**: *"Mengapa 1 model murid `14,75 MiB` bisa mengungguli ensemble gurunya (`99,84%` vs `99,76%`)?"*
    - **Jawaban**: *"Kombinasi tiga faktor: target lunak `MeanTeacherEMA` pada $T = 3,5$ mencegah murid menghafal 4 error label keras yang masih tersisa di `Sub 31`; `CRD` + `RKD-dist` + `RKD-angle` menjaga geometri relasional fitur di ruang `256-D`; dan `SWA` (Epoch 17-25) merata-ratakan bobot ke pusat lembah kurva loss yang datar."*
12. **Tanya**: *"Apakah kuantisasi `FP16` (`29,28 MiB -> 14,75 MiB`) membuat prediksi berubah atau rentan terhadap noise?"*
    - **Jawaban**: *"Tidak sama sekali: pada uji silang `2.902` citra Fase 2, hasil prediksi `FP16` sama persis `2.902 / 2.902 (100,00%)` dengan `FP32`. Di uji noise Gaussian $\sigma = 25$ (3 seed `42, 43, 44`), SWA `FP16` mempertahankan **`95,98% ± 0,02%`**, mengungguli checkpoint Epoch 13 `densenet121_cbam_student_best.pt` (`94,37% ± 0,24%`)."*
13. **Tanya**: *"Bagaimana persisnya fungsi `rkd_dist` dan `rkd_angle` mencegah perbedaan skala antara fitur guru dan murid?"*
    - **Jawaban**: *"Sebelum menghitung `smooth_l1_loss` di `src/distillation_loss.py`, seluruh vektor fitur dinormalisasi L2 ke $\mathbb{S}^{255}$, lalu pada `rkd_dist` matriks jarak pairwise off-diagonal dibagi dengan rata-rata jarak masing-masing batch (`ds_off / mu_s`, `dt_off / mu_t`) sehingga hanya struktur jarak relatif dan sudut triplet yang ditransfer."*

### Kelompok D: Pemeriksaan Kode, ke-10 Notebook & Reproduksi Asdos

14. **Tanya**: *"Mengapa di GitHub notebook `01-09` hanya berisi log teks tanpa grafik `matplotlib`?"*
    - **Jawaban**: *"Notebook `01-09` dieksekusi di Kaggle GPU khusus untuk komputasi training, OOF `.npy`, dan pembuatan submisi tanpa membuang kuota waktu GPU untuk plotting (`execution_count` dan log `stdout` asli Tesla T4 sudah tertanam di sel `01-09`). Seluruh visualisasi ADM klasik dan evaluasi empirisnya disusun terpadu pada Notebook `10` (`10_mnist_adm_visual_evaluation_and_diagnostics.ipynb`) yang sudah menampilkan `7` grafik `image/png` langsung di GitHub."*
15. **Tanya**: *"Pada Notebook `08`, judul log bertulis `DenseNet-161 CBAM`, tetapi kelas kodenya bernama `DenseNet121_CBAM_64`. Mana yang benar?"*
    - **Jawaban**: *"Kelas aktual yang dijalankan adalah `DenseNet121_CBAM_64` berbasis `models.densenet121(weights=None)` dengan stem `3x3 stride 2`, `feature_dim=512`, dan `SubCenterArcFace(K=3)`. Teks `DenseNet-161` hanya keterangan historis pada baris `print` awal ketika skrip disusun."*
16. **Tanya**: *"Mengapa Notebook `05` (`mnist-adm-fixmatch`) berhenti di Fold 5 Epoch 3 dengan status `CANCEL_ACKNOWLEDGED`?"*
    - **Jawaban**: *"Riwayat asli Kaggle tetap kami tampilkan utuh tanpa diedit: Fold 1 sampai 4 selesai lengkap menghasilkan `resnet18_cbam_fixmatch_fold0..3.pt`, lalu eksekusi dihentikan di Fold 5 Epoch 3 setelah tim beralih memfokuskan GPU ke `06` STM (`Sub 16: 0,9469`) dan `07` ConvNeXt-FixMatch (selesai utuh 5-fold)."*
17. **Tanya**: *"Mengapa pada Notebook `09` nilai awal `source_counts` disimpan ke dalam file checkpoint `.pt`?"*
    - **Jawaban**: *"Di `predict_phase2_distillation.py`, `load_student_model` membaca kembali vektor `source_counts` empiris Super-Dataset (`15.918` sampel: `[905, 1014, 1011, ..., 1105]`) langsung dari checkpoint sehingga `TestTimePriorAdapter` memakai prior latih $P_s(y)$ yang akurat, bukan asumsi seragam `1/15` yang keliru."*
18. **Tanya**: *"Jika kami ingin menjalankan inferensi model final atau membangkitkan ulang statistik ADM di mesin lokal, perintah tepatnya apa?"*
    - **Jawaban**: *"Cukup buat virtualenv dari `requirements.txt`, jalankan `.venv/bin/python src/multivariate_analysis.py` untuk menguji ulang PCA/MANOVA/Box's M/LDA vs QDA, jalankan `.venv/bin/python src/predict_phase2_distillation.py` untuk menguji inferensi `5,44 detik` model Student FP16, atau buka langsung `notebooks/10_mnist_adm_visual_evaluation_and_diagnostics.ipynb`."*

---

## Bagian 7: Cheat Sheet Peta 29 Slide & Dua Aturan Emas Sidang

| Slide | Judul Inti | Poin Wajib Terucap (1 Kalimat Kunci) |
| :---: | :--- | :--- |
| **1-2** | Sampul & Identitas | Tim `arete` (`5025241135`): dari ADM klasik menuju model tunggal `14,75 MiB` (`99,84%` F1, `98,62%` F2). |
| **3-4** | Objek 15 Kelas & Masalah Multivariat | `64x64` = `4.096-D`; kendala kolinearitas, bimodalitas `万/萬`, domain shift `Kaishu->Caoshu`, dan label shift. |
| **5-6** | Peta 6 Era & 5 Sumber Data | Sintesis `700` tradisional, isolasi `StratifiedGroupKFold` (`70` penulis), Super-Dataset `15.918` + `2.902` Fase 2. |
| **7-8** | Era 0A-0B: PCA, Eigen-Char & Scree | `85%` = `109` PC, `90%` = `154` PC, `95%` = `252` PC, Kaiser = `275` PC (`>93%` dimensi redundan linier). |
| **9-10** | Era 0C-0D: MANOVA & Box's M (`LDA` vs `QDA`) | Mean beda nyata ($\Lambda=8{,}89\times 10^{-5}$), kovarians heterogen ($M=214.542, p=0$); QDA (`96,09%`) menang `+10,96%`. |
| **11** | Jembatan Matematis ke Deep Learning | 4 temuan ADM -> `DenseNet-121 CBAM`, `CRD`/`RKD`, `Sub-Center ArcFace (K=2)`, dan `EM TTPA`. |
| **12-13** | Era I Kejutan Fase 1 & 5 Karakter Sulit | `1.500` CM = `100%`, tapi `966` Gaya Baru gagal di `五` (`51%`), `百` (`31%`), `四`, `千`, dan `十`. |
| **14-16** | Era II DenseNet-CBAM, Biomekanik & EM TTPA | Konkatenasi fitur + elastis ($\alpha=7,\sigma=4$) + kuas `45°` + EM TTPA (`20` iter, pulihkan `+23` kelas 4 tanpa hardcode). |
| **17-18** | Era III Plafon `94,73%` & Terobosan NCCU | Ablasi `drop D1` & STM tertahan di `94,73%`; pecah lewat `2.952` kursif NCCU terfilter (`Sub 25: 97,57%`). |
| **19-20** | Era IV MILP (`99,76%`) & Alasan Berhenti | MILP bongkar bug `val_00459` & `148` masker, tapi transduktif + ensemble `3,0 GB` -> beralih ke Distilasi. |
| **21-22** | Era V Pipeline 5-Loss KD & `MeanTeacherEMA` | Guru EMA bersih bimbing murid terdistorsi via `Task + KL (T=3,5) + CRD + RKD-dist + RKD-angle` + SWA Ep 17-25. |
| **23-24** | Rekor `Sub 32` (`99,84%`) & Audit `best` vs `fp16` | Murid `14,75 MiB` turun ke `4` error (`五`/`百` 100%); SWA `FP16` `+1,61%` lebih tahan noise $\sigma=25$ dibanding Ep 13. |
| **25-27** | Kejujuran 4 Lapis & Blind Test Fase 2 (`98,62%`) | Pisahkan Fase 1 (dev) dari `2.902` citra tersegel Fase 2 (`0` MD5 overlap, `5,44 detik`, TTPA otonom, kuota `5/6`). |
| **28-29** | Peta Kode, 10 Notebook & Penutup | Kode `src/`, `9` notebook Kaggle (`+ log T4`) + `Notebook 10` (`7` PNG), publik di `https://github.com/Reiii0-0/MNIST-ADM`. |

**Dua Aturan Emas Saat Sidang**:
1. **Jangan pernah mengarang angka yang belum diuji**: Jika dosen menanyakan skenario ablasi yang tidak dijalankan di kuota Fase 2, sampaikan jujur bahwa kombinasi tersebut belum diuji dan tunjukkan metrik paling dekat yang terukur di `outputs/` atau `eval_results/`.
2. **Selalu dahului isu kebocoran data**: Sebelum dosen mempertanyakan mengapa Fase 1 masuk ke Super-Dataset, Anda sudah menjelaskan duluan pada Slide 25 bahwa Fase 1 adalah *development benchmark* dan generalisasi sejati dibuktikan oleh skor `0,9862` (`5,44 detik`) pada `2.902` citra blind test Fase 2.

---
*Navigasi Vault: [[INDEX|Indeks Utama Dokumentasi]] | [[19_TEKS_PRESENTER_PPT_VERBATIM|Naskah Presenter Verbatim (Slide 1-29)]] | [[20_PENJELASAN_SISTEM_DARI_TITIK_PALING_DASAR|Penjelasan Sistem dari Titik Nol]] | [[17_MASTER_DOKUMENTASI_KESELURUHAN_DAN_RIWAYAT_VERSI|Master Dokumentasi]]*
