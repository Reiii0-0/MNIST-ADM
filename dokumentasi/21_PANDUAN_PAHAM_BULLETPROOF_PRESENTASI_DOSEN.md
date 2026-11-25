---
title: "21: Panduan Pemahaman Utuh, Bedah 10 Notebook & Pertahanan Bulletproof Presentasi Dosen"
source_doc: "21_PANDUAN_PAHAM_BULLETPROOF_PRESENTASI_DOSEN.md"
date: 2026-11-25
updated: 2026-11-25
tags:
  - presentation-defense
  - bulletproof-qa
  - notebook-walkthrough
  - multivariate-analysis
  - knowledge-distillation
  - adm-project
aliases:
  - Panduan Bulletproof Presentasi ADM
  - Bedah Lengkap Sistem & 10 Notebook
status: verified
---

# 21: Panduan Pemahaman Utuh, Bedah 10 Notebook & Pertahanan Bulletproof Presentasi Dosen

Dokumen ini merupakan bekal lengkap untuk menguasai **seluruh cerita sistem dari nol sampai akhir** beserta **isi kode ke-10 notebook** (`01` s.d. `10`) dan skrip `src/` sebelum maju mempresentasikan **29 slide** dek Canva (`FP_Analisis_Data_Multivariat_Tim_Arete_5025241135_FINAL.pdf`) di hadapan dosen pengampu dan asisten dosen. Seluruh angka, nama kelas, hyperparameter, dan jalur berkas telah diverifikasi langsung terhadap repositori [`FinalProject/`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject) dan repositori publik `https://github.com/Reiii0-0/MNIST-ADM`.

---

## Bagian 1: Strategi Membawakan PPT Sesuai Arahan Asisten Dosen

Asisten dosen menyampaikan bahwa tidak ada format kaku: hal terpenting adalah menjelaskan **apa saja yang dikerjakan tim** dan **kode pendukungnya**. Dengan 29 slide yang ada, **Anda tidak perlu membacakan riwayat Sub 1 sampai Sub 32 satu per satu**. Gunakan grafik evolusi sebagai pelengkap visual di layar, lalu bagi penjelasan lisan ke dalam **4 babak** (~6-8 menit total):

1. **Babak 1 - Masalah & Fondasi ADM Klasik (`Slide 3-11`, ~2 menit)**: Bagian terpenting untuk nilai mata kuliah Analisis Data Multivariat. Jelaskan data `4.096-D`, PCA, MANOVA, Box's M, kemenangan Regularized QDA atas LDA (`96,09%` vs `85,13%`), serta bagaimana temuan statistik ini melahirkan desain jaringan saraf kita.
2. **Babak 2 - Kejutan Kaligrafi Kursif & Terobosan Utama (`Slide 12-20`, ~1,5 menit)**: Sampaikan intinya secara ringkas tanpa bertele-tele menyebut tiap nomor Sub: model awal turun di `966` gambar kaligrafi kursif, lalu dipecahkan oleh **DenseNet-121 CBAM + Augmentasi Biomekanik + Kurikulum NCCU (`2.952` citra) + Bayesian EM TTPA (`alpha=1,05`)**; sebutkan singkat bahwa eksplorasi ensemble `3,0 GB` + MILP (`99,76%`) sengaja dihentikan karena berat dan tidak bisa dibawa ke Fase 2.
3. **Babak 3 - Sistem Terbaru: Single Student SWA FP16 (`Slide 21-24`, ~2 menit)**: Jelaskan mekanisme **5-Loss Knowledge Distillation** + guru online `MeanTeacherEMA` + perataan bobot **SWA Epoch 17-25** yang memadatkan seluruh kecerdasan riset menjadi 1 file model `14,75 MiB` (`Sub 32: 99,84%`).
4. **Babak 4 - Kejujuran Blind Test Fase 2 & Peta Kode GitHub (`Slide 25-29`, ~1,5 menit)**: Tekankan pemisahan tegas antara Fase 1 (`in-sample development benchmark`) dengan **Fase 2 tersegel (`2.902` citra baru -> `98,62%` dalam `5,44 detik`)**, plus tunjukkan struktur `src/` dan `10` notebook di GitHub.

---

## Bagian 2: Penjelasan Utuh Dari Titik Nol Sampai Model Akhir

### 1. Titik Nol: Apa Sebenarnya Data dan Masalahnya?
- **Masukan sistem**: Gambar tulisan tangan **15 kelas angka Mandarin** (`0` = `零` sampai `100 juta` = `亿`), berukuran `64 x 64` piksel skala keabuan (`0-255`). Jika matriks gambar itu diratakan (*flatten*), setiap satu gambar adalah satu pengamatan acak multivariat berdimensi **`p = 4.096`**: $\mathbf{x} \in \mathbb{R}^{4096}$.
- **Tiga masalah berat pada data**:
  1. **Satu kelas punya dua rupa berlainan (bimodalitas aksara)**: Kelas `13` mencakup `万` sederhana (3 goresan) dan `萬` tradisional (12 goresan), sementara Kelas `14` mencakup `亿` dan `億`. Pada `10.500` gambar latih awal, bentuk tradisional `萬`/`億` sama sekali tidak ada (`0` contoh). Karena itu kita menyintesis **`700` gambar tradisional** lebih dulu ([`generate_traditional_data.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/generate_traditional_data.py)).
  2. **Tulisan balok rapi vs kaligrafi cakar ayam (*domain shift*)**: `10.500` data latih ditulis dengan gaya balok formal rapi (**Kaishu**), sedangkan ujian Fase 1 (`966` citra Gaya Baru) dan Fase 2 (`2.902` citra) menghadirkan tulisan semi-kursif (**Xingshu**) dan kursif cepat (**Caoshu**). Contoh ekstrem: angka `5` (`五`) meliuk seperti `Z`/`S`, angka `11` (`百`) kotak bawahnya menyatu menyerupai `万`, dan angka `12` (`千`) kerap tertukar `十`.
  3. **Risiko bocor gaya penulis**: `10.500` gambar berasal dari `70` penulis. Validasi lokal dikunci dengan **`StratifiedGroupKFold`** berdasarkan `writer_id` (`56` penulis latih vs `14` penulis uji di tiap fold), sehingga model tidak pernah diuji pada tulisan orang yang sama.

### 2. Babak ADM Klasik: PCA, MANOVA, Box's M & LDA vs QDA (Era 0)
Seluruh uji klasik berada di [`multivariate_analysis.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/multivariate_analysis.py):
- **Mengapa kovarians mentah singular (`det(S) = 0`) padahal `N = 10.500 > p = 4.096`?** Bukan karena kekurangan baris, melainkan karena piksel bingkai luar kanvas hampir selalu kosong konstan (varians mendekati nol) dan piksel yang bersebelahan saling berkorelasi tinggi (**multikolinearitas spasial**). Akibatnya nilai eigen terkecil jatuh ke nol dan $\mathbf{S}^{-1}$ gagal dihitung.
- **Principal Component Analysis (PCA)**: Memutar `4.096` piksel ke sumbu ortogonal baru (*Eigen-Characters*). Hasilnya:
  - `4` komponen pertama menangkap energi kontras utama (`PC1 6,67%`, `PC2 5,62%`, `PC3 4,93%`, `PC4 4,09%`).
  - `82` komponen menangkap `80%` variansi, **`109` komponen mencapai `85%`**, **`154` mencapai `90%`**, **`252` mencapai `95%`**, dan kriteria Kaiser ($\bar{\lambda} = 0{,}0875$) mempertahankan `275` komponen. Artinya $>93\%$ dimensi piksel mentah bersifat redundan secara linier.
- **MANOVA pada `m = 20` PC**: Menguji apakah ke-15 kelas punya pusat rata-rata yang berbeda nyata. Hasilnya mutlak menolak $H_0$: **`Wilks' Lambda = 8,8900e-5`**, **`Pillai's Trace = 5,8418`**, **`Hotelling-Lawley = 17,6700`**, **`Roy's Root = 0,8096`**, dan **`Partial Eta Squared = 0,4864`** (efek pemisahan sangat besar, jauh di atas `0,14`).
- **Uji Box's M via `slogdet` & Plot Q-Q Mahalanobis**: Membuktikan asumsi kesamaan matriks kovarians **ditolak total** ($M = 214.542,08$, $\chi^2 = 212.310,12$, $df = 2.940$, $p = 0$ pada $m=20$; $M = 535.061,64$ pada $m=40$) dan ekor sebaran Mahalanobis menyimpang dari normalitas murni. Secara fisik, huruf `一` punya elips sebaran sempit, sedangkan `萬`/`億` menyebar lebar ke banyak arah.
- **Bukti Empiris 5-Fold Unseen Writer CV**:
  - `Two-Stage PCA (100) + Fisher LDA` (memaksa satu kovarians gabungan untuk 15 kelas): **`85,13%`** ($\pm 2,00\%$).
  - `Two-Stage PCA (100) + Logistic Regression`: **`91,08%`** ($\pm 1,68\%$).
  - `Two-Stage PCA (40) + Regularized QDA` (`reg_param = 0,1`, memberi elips kovarians terpisah per kelas): **`96,09%`** ($\pm 0,98\%$, unggul **`+10,96%`** di atas LDA!).
- **Alasan `m = 40` di QDA dan `m = 100` di LDA**: Satu matriks kovarians $m \times m$ memiliki $\frac{m(m+1)}{2}$ parameter bebas per kelas. Pada `m = 100`, QDA harus menaksir `5.050` parameter kovarians di setiap kelas untuk ratusan sampel saja; menurunkan ke `m = 40` (`820` parameter) plus regularisasi `0,1` membuat kovarians tiap kelas tetap stabil secara numerik.
- **Jembatan ke Deep Learning**: Karena proyeksi piksel linier PCA berhenti di `96,09%` saat menghadapi pergeseran posisi maupun liukan kaligrafi kursif, arsitektur modern kita dirancang langsung menjawab 4 temuan tadi:
  1. Kolinearitas `4.096-D` -> Ekstraktor nonlinier `DenseNet-121 CBAM` ke manifold $\mathbb{R}^{256}$.
  2. Separabilitas mean MANOVA -> Penarikan margin ruang fitur oleh `CRD` + `RKD` pada hipersfer $\mathbb{S}^{255}$.
  3. Heterogenitas kovarians Box's M (`QDA >> LDA`) -> Kepala klasifikasi **`Sub-Center ArcFace (K = 2)`** (dua pusat per kelas untuk gaya formal dan gaya kursif/tradisional).
  4. Pergeseran frekuensi kelas -> **`Bayesian EM TTPA`** tanpa label dan tanpa hardcode.

### 3. Mengapa Tidak Berhenti di ResNet/ConvNeXt Maupun Ensemble MILP? (Era I-IV)
- **Kejutan Fase 1 (`Sub 1-5`)**: Model awal `ResNet-18 CBAM` meraih `99,79%` OOF lokal, tetapi saat Fase 1 CodaBench (`2.466` citra) dibuka, `Sub 1` turun ke `91,20%` (`Sub 3` blend ResNet 85% + ConvNeXt 15% merebut Rank 1 di `93,19%`). Pembedahan membuktikan `1.500` Chinese MNIST formal sudah `100%`, sedangkan `966` kaligrafi Gaya Baru baru `~77-83%` karena **penjumlahan residu ResNet menenggelamkan detail goresan 1-2 piksel** pada `五` (51%), `百` (31%), `四` (73%), dan `千` (73,3%).
- **Tiga Terobosan Era II-III**:
  1. **Arsitektur [`model_densenet_cbam.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/model_densenet_cbam.py)**: DenseNet-121 melakukan *concatenation* peta fitur antar-lapisan agar fitur tepi mikro awal tetap utuh sampai ujung, dilengkapi stem `3x3 stride-1` (tanpa MaxPool awal), 1 modul **`MixStyle`** pasca-stem (`p=0,5, alpha=0,1`) untuk menukar statistik gaya ketebalan tinta antarpenulis, 4 modul **`CBAM`**, dan `Sub-Center ArcFace`.
  2. **Fisika Kuas + Kurikulum Kursif NCCU**: [`biomechanical_augmentation.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/biomechanical_augmentation.py) melenturkan huruf Kaishu lewat *Elastic Distortion* ($\alpha = 7, \sigma = 4$) dan dilasi `45°`/erosi `2x2`. Lalu [`lecturer_exact_filter.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/lecturer_exact_filter.py) menyaring **`2.952` citra kursif NCCU** memakai binarisasi Otsu, *tight bounding-box crop*, dan blur ringan agar morfologinya identik dengan profil data pengampu, yang memecahkan plafon `94,73%` menjadi **`97,57%` (`Sub 25`)**.
  3. **Bayesian EM TTPA ([`test_time_prior_adaptation.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/test_time_prior_adaptation.py))**: Pada `2.466` citra Fase 1, `1.500` Chinese MNIST punya tepat `100` sampel Kelas `0` dan `100` sampel Kelas `14`, sedangkan pada `966` Gaya Baru kedua kelas itu `0` sampel. Akibatnya prior seragam `1/15` melempar banyak tebakan palsu (misalnya `Champion_Blend_85_15` memprediksi Kelas `0` sebanyak `112` kali dan Kelas `14` sebanyak `133` kali). Dengan **EM + Dirichlet `alpha = 1,05`** (`20` iterasi, `5,51 ms`), sistem menekan `11` prediksi palsu Kelas `14` (`133 -> 122`), `2` prediksi Kelas `0` (`112 -> 110`), dan **memulihkan `23` prediksi Kelas `4` (`四`, `180 -> 203`)** tanpa satu pun angka nol manual.
- **Puncak Era IV (`Sub 30 = 99,76%`) & Krisis Menuju Fase 2**: Solver MILP SciPy HiGHS (`4.993` kendala persamaan lintas 22 matriks konfusi historis) berhasil menemukan bug indeks `val_00459.png` + kontaminasi masker `148` citra di Sub 29, lalu mengangkat `Sub 30` ke **`99,76%` (`2.460 / 2.466`, hanya 6 salah)** dan menghasilkan label kunci `Sub 31`. Namun cara itu **tidak bisa dibawa ke Fase 2**: MILP butuh feedback matriks konfusi (transduktif) dan pool ensemble riset menumpuk hingga **`3,0 GB`** (`444,55 MiB` untuk 15 model adaptasi, `135x` *forward pass* per gambar).

### 4. Sistem Final: Knowledge Distillation 5-Loss & Single Student SWA FP16 (Era V-VI)
Implementasi lengkap ada di [`distillation_loss.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/distillation_loss.py), `Notebook 09`, dan inferensi [`predict_phase2_distillation.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/predict_phase2_distillation.py):
- **Data latih (`15.918` citra)**: `10.500` Chinese MNIST + `2.952` NCCU kursif + `2.466` label terverifikasi Fase 1.
- **Guru Online (`MeanTeacherEMA`, `alpha_EMA = 0,995`)**: Bobot bergerak lambat milik guru mengevaluasi **citra bersih** (`x_clean`) menghasilkan logit lunak $z_T$ dan fitur laten L2 $f_T \in \mathbb{R}^{256}$.
- **Murid Belajar di Badai (`UnifiedDistillationLoss`)**: Murid mengevaluasi **citra yang didistorsi biomekanik + MixStyle** (`x_aug`) lalu mengoptimasi 5 komponen loss serentak:
  1. **Task Loss** (`Label Smoothing = 0,05` + `SubCenterArcFace K=2, s=32,0, m=0,30`, fusi logit `0,5 * z_lin + 0,5 * z_arc`).
  2. **Response Dark Knowledge (`alpha = 0,70, T = 3,5`)**: Menyalin probabilitas lunak guru via KL Divergence berskala $T^2$ sehingga murid mempelajari kemiripan halus antarkarakter dan kebal terhadap 4 sisa noise label di `Sub 31`.
  3. **Contrastive CRD (`beta_crd = 0,10, tau = 0,07`)**: InfoNCE pada hipersfer satuan $\mathbb{S}^{255}$ agar fitur laten murid dan guru berpasangan saling menarik dan pasangan berbeda saling menolak.
  4. **Distance RKD (`gamma_dist = 1,00`)**: *Smooth-L1* jarak Euclidean berpasangan antar-gambar di batch agar skala jarak ruang laten guru dan murid sama.
  5. **Angle RKD (`delta_angle = 2,00`)**: *Smooth-L1* sudut kosinus triplet agar kelengkungan geometri manifold ikut terwariskan.
- **SWA & Kuantisasi FP16**: Pada Epoch `17-25` (`swa_lr = 1e-4`), bobot 9 epoch terakhir dirata-ratakan plus `update_bn`, lalu dikompresi ke **FP16** ([`densenet121_cbam_student_fp16.pt`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/models/densenet121_cbam_student_fp16.pt), **`14,75 MiB`**, `7.522.968` parameter). Prediksi FP16 terbukti **100% identik (`2.902 / 2.902`)** dengan FP32.
- **Hasil Akhir**:
  - **Fase 1 (`Sub 32`, ID `1014243`)**: **`0,9984`** (`2.462 / 2.466`, hanya `4` salah; 12 dari 13 kelas aktif Gaya Baru mencapai `100,0%`).
  - **Fase 2 Blind Test (`P2-Sub1`, ID `1048381`)**: `2.902` citra penulis baru (`0` duplikat internal, `0` irisan hash MD5 vs `15.918` citra latih), dijalankan `1x forward pass` Batch-128 dalam **`5,44 detik`** di RTX 4060 -> skor resmi **`0,9862`** (~`2.862 / 2.902` benar, masih tersisa `5` dari `6` kuota).

---

## Bagian 3: Bedah Lengkap Isi Kode & Hasil Seluruh 10 Notebook (`notebooks/01-10`)

Seluruh file notebook berada di `FinalProject/notebooks/` dan dapat langsung dilihat beserta outputnya di `https://github.com/Reiii0-0/MNIST-ADM/tree/main/notebooks`.

### Ringkasan Tabel ke-10 Notebook

| No. | File Notebook | Asal / Lingkup | Inti Kelas & Arsitektur | Hyperparameter Kunci | Output Tertanam (Log / Visual) |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **`01`** | [`01_mnist_adm_resnet18_cbam.ipynb`](./../notebooks/01_mnist_adm_resnet18_cbam.ipynb) | Kaggle `reiiihel/mnist-adm` (`21` sel kode) | `ResNet18_CBAM`, `BasicBlockCBAM`, `MixStyle(p=0.5, alpha=0.1)`, `SubCenterArcFace(K=2)`, `TestTimePriorAdapter` | `5-Fold`, `15` ep, `bs=64`, `lr=5e-4`, `alpha_TTPA=1.05` | Log Tesla T4 (`57` baris): **OOF `99,80%`** |
| **`02`** | [`02_mnist_adm_convnext_femto.ipynb`](./../notebooks/02_mnist_adm_convnext_femto.ipynb) | Kaggle `reiiihel/mnist-adm-convnext` (`6` sel kode) | `ConvNeXt_Femto` (`ConvNeXtBlock`, `LayerNorm2d`), `SubCenterArcFace(K=2, s=32, m=0.30)` | `5-Fold`, `15` ep, `bs=64`, `AdamW lr=1e-3`, elastis `alpha=12, sigma=3` | Log Tesla T4 (`124` baris): **OOF `99,55%`** |
| **`03`** | [`03_mnist_adm_densenet121_cbam.ipynb`](./../notebooks/03_mnist_adm_densenet121_cbam.ipynb) | Kaggle `reiiihel/mnist-adm-densenet121` (`7` sel kode) | `DenseNet121_CBAM` (`growth_rate=32`, `(6,12,24,16)`), `CBAM`, `MixStyle(p=0.4, alpha=0.1)`, `SubCenterArcFace` | `5-Fold`, `15` ep, `bs=64`, `lr=5e-4` | Log Tesla T4 (`141` baris): **OOF `99,76%`** + 5-Way TTA `2.466` citra |
| **`04`** | [`04_mnist_adm_sota_v3_biomechanical.ipynb`](./../notebooks/04_mnist_adm_sota_v3_biomechanical.ipynb) | Kaggle `reiiihel/mnist-adm-sota-v3` (`7` sel kode) | `DenseNet121_CBAM`, `CBAM`, `MixStyle(p=0.4)`, `apply_elastic_distortion`, `apply_anisotropic_brush_erosion` | `5-Fold`, `15` ep, `bs=64`, `lr=5e-4`, elastis `alpha=6.0, sigma=3.5` | Log Tesla T4 (`159` baris): distribusi `2.466` prediksi + `sub_sota_v3.zip` |
| **`05`** | [`05_mnist_adm_fixmatch.ipynb`](./../notebooks/05_mnist_adm_fixmatch.ipynb) | Kaggle `reiiihel/mnist-adm-fixmatch` (`11` sel kode) | `ResNet18_CBAM`, `FixMatchDualDataset`, `MixStyle(p=0.5)`, `SubCenterArcFace`, `EMBEDDED_PSEUDO_B64` | Base `10` ep (`lr=5e-4`) + FixMatch `4` ep (`lr=1e-5`), `bs=64`, `+ 0.5 * L_cons` | Log Tesla T4 (`147` baris): selesai Fold 1-4; berhenti `CANCEL_ACKNOWLEDGED` di Fold 5 Ep 3 |
| **`06`** | [`06_mnist_adm_stm_adaptation.ipynb`](./../notebooks/06_mnist_adm_stm_adaptation.ipynb) | Kaggle `reiiihel/mnist-adm-stm-adaptation` (`8` sel kode) | `ResNet18_CBAM` + `ConvNeXt_Femto` + `solve_closed_form_stm` ($\mathbf{z}'=\mathbf{A}\mathbf{z}+\mathbf{b}$) | `beta=50.0, gamma=1.0`, `lr=1e-3` / `8e-4` | Log Tesla T4 (`122` baris): ekspor `sub_stm.zip` (`Sub 16: 0,9469`) |
| **`07`** | [`07_mnist_adm_convnext_fixmatch.ipynb`](./../notebooks/07_mnist_adm_convnext_fixmatch.ipynb) | Kaggle `reiiihel/mnist-adm-convnext-fixmatch` (`11` sel kode) | `ConvNeXt_Femto`, `FixMatchDualDataset`, `FastValDataset`, `evaluate_batched_9way_tta_ensemble` | Base `10` ep (`lr=5e-4`) + FixMatch `4` ep (`lr=1e-5`), `bs=128` | Log Tesla T4 (`197` baris): 5-Fold selesai + 9-Way TTA `2.466` citra |
| **`08`** | [`08_mnist_adm_massive_sota.ipynb`](./../notebooks/08_mnist_adm_massive_sota.ipynb) | Kaggle `reiiihel/mnist-adm-massive-sota` (`11` sel kode) | `DenseNet121_CBAM_64` (`feature_dim=512`), `CBAM`, `SubCenterArcFace(in=512, K=3, s=32, m=0.35)` | `20` ep, `bs=64`, `lr=3e-4`, CosineLR, `15.811` sampel pra-oracle | Log Tesla T4 (`134` baris): Train Acc `99,89%`, `submission_massive_sota.zip` |
| **`09`** | [`09_mnist_adm_distillation_final.ipynb`](./../notebooks/09_mnist_adm_distillation_final.ipynb) | Kaggle `reiiihel/mnist-adm-distillation` (`10` sel kode) | `DenseNet121_CBAM` (`256-D`), `UnifiedDistillationLoss`, `MeanTeacherEMA`, `EMPriorShiftAdapter` | `25` ep, `bs=64`, `lr=5e-4`, `SWA Ep 17-25 (swa_lr=1e-4)`, `T=3.5, alpha=0.70, beta=0.10, gamma=1.00, delta=2.00` | Log Tesla T4 (`59` baris): **`14,75 MB` FP16**, Peak Val `100,00%`, **`Sub 32: 0,9984`** & **`Fase 2: 0,9862`** |
| **`10`** | [`10_mnist_adm_visual_evaluation_and_diagnostics.ipynb`](./../notebooks/10_mnist_adm_visual_evaluation_and_diagnostics.ipynb) | Lokal `FinalProject/` (`7` sel kode) | Analisis `pandas`, `matplotlib`, `PIL` atas JSON/CSV terverifikasi (`outputs/`, `eval_results/`) | Membaca `multivariate_diagnostics_summary.json`, `master_all_submissions_audit.csv`, `verified_empirical_metrics.json` | **`7` grafik `image/png` tertanam + tabel `stdout`** langsung tampil di GitHub |

### Detail Kode Per Notebook yang Perlu Anda Kuasai Saat Ditanya Asdos

1. **Notebook `01` (`ResNet-18 CBAM Baseline & AdaBN/TTPA Engine`)**:
   - Memuat alur penuh: `discover_and_extract_data` -> `BiomechanicalAugmentor` -> `ResNet18_CBAM` (menggunakan `MixStyle(p=0.5, alpha=0.1)` pasca-lapisan awal + 4 `CBAMBlock` + proyeksi `256-D`) -> `SubCenterArcFace(K=2)`.
   - Di fungsi inferensi (`run_test_inference`), notebook ini mendemonstrasikan **AdaBN** (`apply_adabn_to_model` menghitung ulang rata-rata & varians BatchNorm pada gambar uji tanpa label), **TTA**, dan **TTPA** (`TestTimePriorAdapter`, `alpha=1.05`), serta `certify_and_package_codabench` untuk memastikan format CSV/ZIP bebas NaN dan tepat `N` baris.

2. **Notebook `02` (`ConvNeXt-Femto Complementary Backbone`)**:
   - Sengaja dibangun dengan blok **`ConvNeXtBlock`** berkernel spasial besar (`7x7` depthwise) dan `LayerNorm2d`, kontras dengan konvolusi kecil `3x3` milik ResNet-18.
   - Menggunakan fungsi augmentasi `elastic_transform_np(alpha=12, sigma=3)` dan `train_and_eval_fold(epochs=15, batch_size=64)` dengan `AdamW(lr=1e-3)`.
   - Alasan diblend `15%` ke ResNet-18 (`85%`) di `Sub 3`: ResNet kuat menangkap sudut lokal tajam, sedangkan ConvNeXt menangkap kerangka global karakter.

3. **Notebook `03` & `04` (`Peralihan ke DenseNet-121 CBAM + Biomekanik Kursif`)**:
   - Kedua notebook ini meninggalkan penjumlahan residu dan mengimplementasikan **DenseNet-121 custom** (`growth_rate=32`, `block_config=(6, 12, 24, 16)`) yang melakukan *concatenation* fitur lintas lapisan agar detail goresan tipis `1-2` piksel tetap terpelihara, diikuti `CBAM` di setiap akhir *DenseBlock*.
   - Di **`03`**, augmentasi dipecah menjadi kelas OOP terstruktur (`ElasticDistortion`, `DynamicBrushMorphology`, `TargetedStrokePerturbation`) dengan `MixStyle(p=0.4, alpha=0.1)` dan menghasilkan OOF `99,76%`.
   - Di **`04`**, ditambahkan kurasi *Synthetic Calligraphy v2* dengan fungsi `apply_elastic_distortion(alpha=6.0, sigma=3.5)` dan `apply_anisotropic_brush_erosion` sebelum diekspor via 5-Way TTA menjadi `sub_sota_v3.zip`.
   - *(Catatan konsistensi parameter)*: Di skrip lokal `src/biomechanical_augmentation.py` default elastisnya `alpha=7.0, sigma=4.0`, sedangkan di kernel Kaggle `04` disetel `alpha=6.0, sigma=3.5`. Keduanya berada di rentang aman agar karakter sederhana `一`/`二`/`三` tidak terputus topologinya.

4. **Notebook `05`, `06`, & `07` (`Era III: Tiga Cabang Adaptasi Domain Tanpa Label`)**:
   - **`05` (`Track 2: ResNet-18 FixMatch`)** & **`07` (`Track 3: ConvNeXt-Femto FixMatch`)**: Mengimplementasikan *semi-supervised consistency regularization* via `FixMatchDualDataset` pada `EMBEDDED_PSEUDO_B64` (sampel konsensus konfidensi tinggi). Tahap 1 melatih base model `10` epoch (`lr=5e-4`), lalu Tahap 2 melakukan *fine-tuning* halus hanya `4` epoch (`lr=1e-5`) agar tidak merusak representasi dasar dengan kerugian gabungan `Sup_Loss + 0.5 * Consistency_Loss` antara augmentasi lemah (`apply_weak_augmentation`) dan augmentasi kuat (`apply_strong_augmentation`). Di `05`, eksekusi selesai Fold 1-4 dan dihentikan (`CANCEL_ACKNOWLEDGED`) pada Fold 5 Ep 3 saat fokus beralih ke `06` STM dan `07` ConvNeXt-FixMatch (yang selesai utuh 5-fold dengan `batch_size=128` dan `evaluate_batched_9way_tta_ensemble`).
   - **`06` (`Track 1: Style Transfer Mapping / STM`)**: Berbeda dari jaringan yang di-fine-tune lewat gradien, fungsi `solve_closed_form_stm(beta=50.0, gamma=1.0)` menghitung matriks transformasi linier tertutup $\mathbf{A}$ (`256x256`) dan vektor bias $\mathbf{b}$ (`256`) secara analitik (*Quadratic Programming* ber-regularisasi ridge) untuk menarik vektor fitur `256-D` kaligrafi uji agar mendekati prototipe rata-rata kelas latih $\boldsymbol{\mu}_k$ (`compute_class_prototypes`), menghasilkan **`Sub 16 = 0,9469`**.

5. **Notebook `08` (`Eksperimen Pra-Distilasi pada Super-Dataset`)**:
   - Menggabungkan dataset menjadi `15.811` sampel (`10.500` Train + `2.952` NCCU + `2.359` konsensus pseudo-label) di kelas `MassiveHCCRDataset`.
   - **Catatan jujur tentang kode `08`**: Pada teks judul/print awal skrip tertera label historis `"DenseNet-161 CBAM"`, **tetapi jika dosen mengecek baris kode kelasnya**, arsitektur yang benar-benar diinstansiasi adalah **`DenseNet121_CBAM_64`** (berbasis `models.densenet121(weights=None)` dengan stem `3x3 stride 2`, ruang fitur **`feature_dim=512`**, dan **`SubCenterArcFace(in_features=512, out_features=15, K=3, s=32.0, m=0.35)`** selama `20` epoch, `lr=3e-4`). Mengungkap fakta kode ini langsung membuat Anda terlihat sangat menguasai kode nyata tim sendiri.

6. **Notebook `09` (`Final Knowledge Distillation Champion Pipeline`)**:
   - Inilah notebook terpenting di repositori. Ia melatih **satu model tunggal `DenseNet121_CBAM`** (custom stem `3x3`, `growth_rate=32`, `block_config=(6,12,24,16)`, proyeksi `256-D`, `SubCenterArcFace K=2`, `MixStyle p=0.5, alpha=0.1`) pada `15.918` sampel (`10.500 + 2.952 + 2.466 verified PHASE1_LABELS`) selama **`25` epoch** (`bs=64`, `lr=5e-4`, `weight_decay=1e-4`, AMP `GradScaler`).
   - Kelas `UnifiedDistillationLoss` menghitung gabungan 5 loss: `ce(label_smoothing=0.05)`, `kld` (`T=3.5, alpha=0.70`), CRD InfoNCE (`beta_crd=0.10, tau=0.07`), `rkd_dist(gamma_dist=1.00)`, dan `rkd_angle(delta_angle=2.00)` berpasangan dengan guru `MeanTeacherEMA(alpha_ema=0.995)`.
   - Mulai `swa_start = int(25 * 0.70) = 17` sampai Epoch `25`, `AveragedModel` + `SWALR(swa_lr=1e-4)` aktif merata-ratakan bobot, dilanjutkan `update_bn` dan kompresi **FP16** (`checkpoint_student_fp16.pt`, **`14,75 MB`**), lalu dievaluasi oleh `EMPriorShiftAdapter`.

7. **Notebook `10` (`Visual Evaluation, ADM Diagnostics & Audits`)**:
   - Mengapa dibuat? Karena notebook `01-09` dijalankan di Kaggle GPU tanpa plot `matplotlib` (`matplotlib_or_plot=False`), seluruh visualisasi dikerjakan secara lokal di `src/` lalu dikompilasi ke Notebook `10` (`7` sel kode dieksekusi end-to-end dengan **`7` grafik `image/png` tertanam + tabel `stdout`**):
     1. *PCA Scree Plot Top-16* & *5-Fold Unseen Writer CV (`LDA 85,13%` vs `QDA 96,09%` vs `LogReg 91,08%`)*.
     2. Galeri `2x2` figur ADM asli (`scree_plot.png`, `eigen_characters.png`, `mahalanobis_qq_plot.png`, `lda_canonical_projection.png`).
     3. Kurva progresi `28` baris audit `Sub 1-32` Fase 1 + titik blind test Fase 2 (`98,62%`).
     4. Lonjakan per-kelas `13` kelas aktif pada `966` citra kaligrafi Gaya Baru (`Sub 1 -> Sub 8 -> Sub 26`).
     5. Stress test ketahanan derau (`Clean`, `Gaussian sigma=25`, `Erosi 2x2`, `Dilasi 2x2`, `Elastis`) & kompresi memori `773,94 MiB -> 14,75 MiB`.
     6. Kurva reliabilitas kalibrasi (`Pre-KD ECE 5,18%` vs `Student SWA ECE 0,00%`) + histogram confidence Fase 2 (`Raw` vs `+ EM TTPA`).
     7. Galeri `3x2` ringkasan infografik arsitektur dan metodologi dari `outputs/presentation_figures/`.

### Pemetaan Cepat Notebook ke Skrip Produksi Modular (`FinalProject/src/`)
Saat asdos bertanya *"Mana file `.py` modular di lokalnya dibanding notebook Kaggle?"*, tunjukkan padanan langsung ini:
- Analisis multivariat klasik (`Era 0`) -> [`src/multivariate_analysis.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/multivariate_analysis.py)
- Penyelarasan morfologi data kursif (`NCCU`) -> [`src/lecturer_exact_filter.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/lecturer_exact_filter.py) & [`src/biomechanical_augmentation.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/biomechanical_augmentation.py)
- Arsitektur murid final & fungsi loss 5 komponen -> [`src/model_densenet_cbam.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/model_densenet_cbam.py) & [`src/distillation_loss.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/distillation_loss.py)
- Adaptasi prior Bayesian tanpa label -> [`src/test_time_prior_adaptation.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/test_time_prior_adaptation.py)
- Rekonstruksi tomografis MILP Fase 1 -> [`src/synthesize_sub30_grandmaster_anchored_fusion.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/synthesize_sub30_grandmaster_anchored_fusion.py)
- Inferensi `5,44 detik` Blind Test Fase 2 -> [`src/predict_phase2_distillation.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/predict_phase2_distillation.py)
- Pembangkit `24` grafik Keynote Royal Blue -> [`src/generate_detailed_infographics.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/generate_detailed_infographics.py) & [`src/generate_presentation_figures.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/generate_presentation_figures.py)

---

## Bagian 4: Tiga Kelompok Angka yang Wajib Anda Bedakan Secara Tegas

Kesalahan paling fatal saat sidang adalah mencampuradukkan tiga tolok ukur berikut:

| Jenis Evaluasi | Jumlah Sampel & Sifat Data | Skor Terukur | Bagaimana Cara Menjelaskannya ke Dosen |
| :--- | :--- | :---: | :--- |
| **OOF 5-Fold CV (`Pra-Fase`)** | `10.500` citra Kaishu, `70` penulis (`56` latih vs `14` uji via `StratifiedGroupKFold`) | LDA **`85,13%`**, LogReg **`91,08%`**, Reg-QDA **`96,09%`**, ResNet-18 **`99,79%`** (`99,80%` di `01`), ConvNeXt **`99,55%`**, DenseNet v2 **`99,76%`** | Validasi lokal bebas kebocoran penulis sebelum menyentuh Fase 1/2 |
| **Fase 1 (`Development Benchmark`)** | `2.466` citra (`1.500` Chinese MNIST penulis 71-80 + `966` Gaya Baru kursif; kelas `0` & `14` = `0` di Gaya Baru) | Sub 1: `91,20%`, Sub 3 Rank 1: `93,19%`, Sub 25: `97,57%`, Sub 30 MILP: `99,76%`, **Sub 32 Student SWA: `99,84%`** | Sampai Sub 30 bersifat *out-of-sample* plus perbaikan transduktif MILP; pada `Sub 32`, `2.466` citra Fase 1 ikut masuk ke Super-Dataset `15.918` sehingga `0,9984` kami klaim jujur sebagai *development benchmark* |
| **Fase 2 (`Sealed Unseen Blind Test`)** | `2.902` citra penulis baru (`0` duplikat internal, `0` irisan hash MD5 vs `15.918` citra latih/Fase 1) | **`0,9862`** (`~2.862 / 2.902` benar, **`5,44 detik`** Batch-128, kuota `1/6`, `ID 1048381`) | **Bukti generalisasi OOD sejati** yang benar-benar belum pernah dilihat model maupun solver MILP sebelumnya |

---

## Bagian 5: Bank 15 Pertanyaan Jebakan Dosen & Asdos (Jawaban Bulletproof)

### Kelompok A: Teori & Diagnostik ADM Klasik

1. **Tanya**: *"Kenapa matriks kovarians kalian singular padahal $N = 10.500$ lebih besar dari $p = 4.096$?"*
   - **Jebakan**: Menjawab asal *"karena $n < p$"*.
   - **Jawaban Bulletproof**: *"$N = 10.500$ memang lebih besar daripada $p = 4.096$, Pak/Bu, tetapi matriks kovarians sampel tetap singular secara numerik (`det(S) ~ 0`) karena dua hal: piksel tepi kanvas 64x64 bernilai konstan nol di ribuan sampel sehingga variansinya nyaris nol, dan piksel bertetangga membentuk kolinearitas spasial yang ekstrem. Itulah alasan PCA wajib mereduksi dimensi dulu sebelum invers kovarians dihitung pada MANOVA, Box's M, dan LDA/QDA."*

2. **Tanya**: *"Di plot Q-Q Mahalanobis normalitas multivariat menyimpang di ekor, lalu di Box's M kovarians tidak homogen ($p=0$). Apakah hasil MANOVA kalian gugur?"*
   - **Jawaban Bulletproof**: *"Tidak gugur, Pak/Bu. Pertama, pada sampel besar ($N = 10.500$, 700 sampel per kelas), teorema batas pusat multivariat menstabilkan uji vektor mean; kedua, kami tidak bergantung pada Wilks' Lambda semata ($\Lambda = 8{,}89 \times 10^{-5}$), melainkan juga memeriksa **Pillai's Trace** (`5,8418`) yang paling robust terhadap pelanggaran homogenitas kovarians serta **Partial Eta Squared** (`0,4864`). Penolakan Box's M justru kami jadikan bukti empiris mengapa QDA (`96,09%`) jauh mengungguli LDA (`85,13%`)."*

3. **Tanya**: *"Kenapa di QDA kalian pakai `m=40` PC, sementara di LDA dan LogReg pakai `m=100` PC, dan MANOVA pakai `m=20`?"*
   - **Jawaban Bulletproof**: *"Di QDA setiap kelas mengestimasi matriks kovariansnya sendiri berjumlah $\frac{m(m+1)}{2}$ parameter bebas. Pada `m=100`, ada `5.050` parameter per kelas untuk sekitar `560-600` sampel per fold sehingga tidak stabil. Pada `m=40`, jumlah parameternya turun ke `820` dan distabilkan oleh `reg_param = 0,1`. Sedangkan uji MANOVA dan Box's M pada `m=20` dan `m=40` dihitung dengan `slogdet` agar stabil secara numerik."*

4. **Tanya**: *"Apa yang terlihat pada `16 Eigen-Characters` dan apa batas kemampuan PCA di proyek ini?"*
   - **Jawaban Bulletproof**: *"Empat eigen-character teratas (`PC1 6,67%`, `PC2 5,62%`, `PC3 4,93%`, `PC4 4,09%`) memisahkan orientasi global goresan horizontal terhadap vertikal dan energi bingkai luar, sedangkan PC selanjutnya menangkap lekukan persimpangan. Batasnya, PCA bersifat linier tak-berawasi: begitu goresan bergeser beberapa piksel atau hurufnya meliuk di gaya Caoshu, skor PCA mencampur identitas huruf dengan gaya kuas penulis."*

### Kelompok B: Integritas Data, Leakage & Etika Kompetisi

5. **Tanya**: *"Data validasi Fase 1 (2.466 citra) kalian masukkan ke Super-Dataset 15.918 citra saat training Student `Sub 32`. Itu bocor data (data leakage), kan?"*
   - **Jawaban Bulletproof**: *"Betul, Pak/Bu, `0,9984` Fase 1 tidak boleh diklaim sebagai uji out-of-sample murni, dan justru karena itu di Slide 25 kami memisahkannya secara eksplisit. Evaluasi Fase 1 hanya berfungsi sebagai **development benchmark**, sedangkan bukti generalisasi sejati bebas kebocoran ada di **Fase 2 (`2.902` citra blind test penulis baru)** yang terbukti memiliki **0 duplikat internal dan 0 irisan hash MD5** terhadap ke-15.918 citra pelatihan. Pada data tersegel Fase 2 itu, model Student tunggal yang sama mencetak **`0,9862`** sekali jalan dalam `5,44 detik`."*

6. **Tanya**: *"MILP Tomografi di Sub 26-30 memanfaatkan feedback matriks konfusi CodaBench. Bukankah itu eksploitasi leaderboard?"*
   - **Jawaban Bulletproof**: *"MILP itu kami pakai di Fase 1 untuk dua tujuan analitis: mengetahui batas kesalahan pada 966 citra kursif dan membongkar dua bug tersembunyi di data/skrip (`val_00459.png` dan masker 148 citra Kaishu di Sub 29). Namun metode transduktif seperti MILP tidak sah secara saintifik maupun tidak mungkin diterapkan pada ujian buta Fase 2, sehingga seluruhnya kami tinggalkan dan kami bangun model murni induktif via Knowledge Distillation `14,75 MiB`."*

7. **Tanya**: *"Kenapa validasi lokal membagi data memakai `StratifiedGroupKFold` berdasarkan `writer_id`, bukan `StratifiedKFold` biasa?"*
   - **Jawaban Bulletproof**: *"Karena masing-masing dari `70` penulis menulis seluruh `15` kelas sebanyak `10` kali (`10.500` citra). Jika diacak Stratified biasa, tulisan orang yang sama berada di train dan validasi sekaligus sehingga model cukup menghafal gaya kuas penulis. `StratifiedGroupKFold` memisahkan `56` penulis latih dari `14` penulis uji secara mutlak."*

8. **Tanya**: *"Di grafik TTPA, kenapa Kelas 0 dan Kelas 14 masih punya ~110-122 prediksi padahal katanya nol di Gaya Baru?"*
   - **Jawaban Bulletproof**: *"Karena total `2.466` citra Fase 1 merupakan gabungan `1.500` citra Chinese MNIST formal (yang memang memiliki tepat `100` sampel Kelas 0 dan `100` sampel Kelas 14) ditambah `966` citra Gaya Baru (yang kelas 0 dan 14-nya nol). Jadi segala prediksi di atas 100 sebelum TTPA (`112` dan `133`, total `+45`) adalah tebakan palsu di Gaya Baru. Bayesian EM TTPA dengan Dirichlet `alpha = 1,05` menekannya menjadi `110` dan `122` serta memulihkan `23` sampel Kelas 4 (`四`), tanpa pernah meng-hardcode nol agar tetap otomatis mendeteksi Kelas 0 dan 14 ketika keduanya hadir kembali di Fase 2."*

### Kelompok C: Arsitektur Deep Learning, Distilasi & Kuantisasi

9. **Tanya**: *"Mengapa 1 model murid `14,75 MiB` bisa mengalahkan ensemble gurunya (`0,9984` vs `0,9976`)?"*
   - **Jawaban Bulletproof**: *"Pertama, target lunak dari `MeanTeacherEMA` pada suhu $T = 3{,}5$ meredam 4 sisa noise label di `Sub 31` sehingga murid tidak ikut menghafal error. Kedua, 5 loss sekaligus menjaga logit (`KL`) maupun geometri jarak dan sudut di ruang hipersfer 256-D (`CRD` + `RKD`). Ketiga, **SWA Epoch 17-25** menaruh bobot di lembah kurva yang datar (*flat minima*)."*

10. **Tanya**: *"Dari `densenet121_cbam_student_best.pt` (29,28 MiB FP32) ke `densenet121_cbam_student_fp16.pt` (14,75 MiB FP16), apakah penyusutan presisi menurunkan akurasi?"*
    - **Jawaban Bulletproof**: *"Tidak sama sekali, Pak/Bu. Audit langsung pada `2.902` citra Fase 2 menunjukkan prediksi argmax `densenet121_cbam_student_fp16.pt` **100,00% identik (`2.902 / 2.902` sama)** dengan FP32. Saat diuji gangguan Gaussian noise $\sigma = 25$ di 3 seed berbeda (`42, 43, 44`), SWA FP16 mencatat stabilitas **`95,98% ± 0,02%`**, unggul **`+1,61%`** dibanding `densenet121_cbam_student_best.pt` Epoch 13 pra-SWA (`94,37% ± 0,24%`)."*

11. **Tanya**: *"Kenapa memilih DenseNet-121 CBAM dan tidak pakai Transformer seperti ViT atau Swin?"*
    - **Jawaban Bulletproof**: *"Resolusi input kita kecil (`64 x 64`) dan dataset kurasi inti berkisar `15.918` citra. Untuk menangkap goresan setebal `1-2` piksel (pembeda `千` vs `十`), konkatenasi fitur pada `DenseNet-121` dengan stem `3x3` stride-1 tanpa MaxPool awal jauh lebih hemat data (`7,52M` parameter) daripada ViT murni yang minim bias induktif lokal."*

12. **Tanya**: *"Apa perbedaan `MixStyle`, `Sub-Center ArcFace`, dan `MeanTeacherEMA`?"*
    - **Jawaban Bulletproof**: *"`MixStyle` bekerja di fitur awal pasca-stem (`p=0,5, alpha=0,1`) untuk mencampur statistik gaya kuas antarpenulis; `Sub-Center ArcFace` bekerja di kepala klasifikasi menyediakan $K=2$ pusat prototipe per kelas (Kaishu vs Kursif/Tradisional); sedangkan `MeanTeacherEMA` (`alpha=0,995`) adalah guru ber-bobot halus saat distilasi yang membaca gambar bersih sebagai acuan murid."*

### Kelompok D: Kode, Ke-10 Notebook & Reproduksibilitas Asdos

13. **Tanya**: *"Mengapa di notebook `01-09` tidak ada gambar plot `matplotlib`, dan kenapa di GitHub ada Notebook `10`?"*
    - **Jawaban Bulletproof**: *"Notebook `01-09` adalah kernel GPU Tesla T4 Kaggle yang fokus pada training 5-fold dan menghasilkan checkpoint/log tanpa membuang kuota waktu GPU untuk plotting; semua output run `stdout` aslinya sudah disematkan ke sel notebook `01-09`. Seluruh plot diagnostik ADM dan grafik evaluasi empiris kami kumpulkan di skrip lokal `src/` dan dijalankan menjadi [`10_mnist_adm_visual_evaluation_and_diagnostics.ipynb`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/notebooks/10_mnist_adm_visual_evaluation_and_diagnostics.ipynb) yang langsung menampilkan `7` figur `image/png` di GitHub."*

14. **Tanya**: *"Mengapa pada Notebook `08` judul log tertulis `DenseNet-161 CBAM` sedangkan kelas kodenya `DenseNet121_CBAM_64`, dan mengapa Notebook `05` log-nya berhenti di Fold 5 Ep 3 (`CANCEL_ACKNOWLEDGED`)?"*
    - **Jawaban Bulletproof**: *"Dua-duanya merupakan riwayat eksperimen asli yang kami jaga apa adanya di GitHub, Pak/Mas. Pada Notebook `08`, banner print awal memang bertulis `DenseNet-161 CBAM` sementara kelas aktualnya memakai `models.densenet121` (`DenseNet121_CBAM_64`, `feature_dim=512`, `K=3`). Pada Notebook `05`, `Track 2 FixMatch` ResNet-18 menyelesaikan penuh Fold 1-4 dan dihentikan di Fold 5 Epoch 3 karena tim langsung memprioritaskan `Track 1 STM (0,9469)` (`Notebook 06`) dan `Track 3 ConvNeXt-FixMatch` (`Notebook 07`, selesai utuh 5-fold)."*

15. **Tanya**: *"Bagaimana cara dosen atau asdos memverifikasi ulang kode dan prediksi Fase 2 dari repositori GitHub?"*
    - **Jawaban Bulletproof**: *"Repositori publik `https://github.com/Reiii0-0/MNIST-ADM` sudah menyertakan seluruh kode modular di `src/`, 10 notebook `.ipynb` lengkap dengan output run, `requirements.txt`, dokumentasi indeks, serta PDF/PPTX presentasi. Untuk memverifikasi statistik ADM klasik cukup menjalankan `src/multivariate_analysis.py`, untuk menghasilkan `24` figur Keynote jalankan `src/generate_detailed_infographics.py` dan `src/generate_presentation_figures.py`, dan untuk mengulang inferensi Fase 2 `5,44 detik` gunakan `src/predict_phase2_distillation.py` dengan checkpoint FP16 terlampir."*

---

## Bagian 6: Cheat Sheet Peta 29 Slide & Dua Aturan Emas

| Slide | Judul Inti | Poin Wajib Terucap |
| :---: | :--- | :--- |
| **1-2** | Sampul & Identitas | Tim `arete` (`5025241135`): dari teori ADM klasik ke model `14,75 MiB` (`99,84%` F1, `98,62%` F2). |
| **3-4** | Objek 15 Kelas & Masalah Multivariat | Gambar `64x64` = vektor `4.096-D`; kolinearitas, bimodalitas `万/萬`, domain shift `Kaishu->Caoshu`, label shift. |
| **5-6** | Peta 6 Era & 5 Sumber Data | Sintesis `700` tradisional, kunci `StratifiedGroupKFold` per `writer_id`, Super-Dataset `15.918` + `2.902` Fase 2. |
| **7-8** | Era 0A-0B: PCA, Eigen-Char & Scree | `85%` = `109` PC, `90%` = `154` PC, `95%` = `252` PC, Kaiser = `275` PC; $>93\%$ piksel redundan linier. |
| **9-10** | Era 0C-0D: MANOVA & Box's M (`LDA` vs `QDA`) | Mean beda nyata ($\Lambda=8{,}89\times 10^{-5}$), kovarians beda jauh ($M=214.542$); QDA (`96,09%`) menang `+10,96%`. |
| **11** | Jembatan Matematis ke Deep Learning | Empat masalah ADM klasik -> DenseNet-121 CBAM, CRD/RKD, Sub-Center ArcFace (`K=2`), dan EM TTPA. |
| **12-13** | Era I Kejutan Fase 1 & 5 Karakter Sulit | Chinese MNIST `100%`, tapi 966 Gaya Baru memunculkan error di `五` (`51%`), `百` (`31%`), `四`, `千`, dan `十`. |
| **14-16** | Era II DenseNet-CBAM, Biomekanik & EM TTPA | Konkatenasi fitur + elastis ($\alpha=7,\sigma=4$) + kuas `45°` + EM TTPA (`20` iter, pulihkan `+23` kelas 4). |
| **17-18** | Era III Plafon `94,73%` & Terobosan NCCU | Ablasi `drop D1` & STM mentok di `94,73%`; terpecahkan `2.952` kursif NCCU terfilter (`Sub 25: 97,57%`). |
| **19-20** | Era IV MILP (`99,76%`) & Mengapa Berhenti | MILP temukan bug `val_00459.png` dan masker `148` citra, tapi transduktif + `3,0 GB` ensemble -> pindah ke Distilasi. |
| **21-22** | Era V Pipeline 5-Loss KD & `MeanTeacherEMA` | Guru EMA bersih bimbing murid teraugmentasi via `Task + KL (T=3,5) + CRD + RKD-dist + RKD-angle` + SWA Ep 17-25. |
| **23-24** | Rekor `Sub 32` (`99,84%`) & Audit `best` vs `fp16` | Murid `14,75 MiB` turun ke `4` error (`五`/`百` 100%); SWA `FP16` `+1,61%` lebih tahan noise $\sigma=25$ dibanding Ep 13. |
| **25-27** | Kejujuran 4 Lapis & Blind Test Fase 2 (`98,62%`) | Pisahkan Fase 1 (dev) dari `2.902` citra tersegel Fase 2 (`0` MD5 overlap, `5,44 detik`, TTPA otonom, kuota `5/6`). |
| **28-29** | Peta Kode, 10 Notebook & Penutup | Kode `src/`, `9` notebook Kaggle (`+ log T4`) + `Notebook 10` (`7` grafik PNG), publik di `https://github.com/Reiii0-0/MNIST-ADM`. |

**Dua Aturan Emas Saat Menjawab Dosen**:
1. **Jangan pernah mengarang angka yang belum diuji**: Jika ditanya skenario ablasi yang tidak pernah dijalankan di Fase 2, akui tegas belum diuji pada kuota Fase 2 dan tunjukkan angka terdekat yang terukur.
2. **Selalu bedakan Fase 1 vs Fase 2 lebih dulu**: Sebelum dosen sempat menuduh kebocoran data pada `99,84%` Fase 1, Anda yang lebih dulu menyatakan bahwa bukti generalisasi sejatinya adalah `0,9862` di `2.902` citra tersegel Fase 2.

---
*Navigasi Vault: [[INDEX|Indeks Utama Dokumentasi]] | [[19_TEKS_PRESENTER_PPT_VERBATIM|Naskah Presenter Verbatim (Slide 1-29)]] | [[20_PENJELASAN_SISTEM_DARI_TITIK_PALING_DASAR|Penjelasan Sistem dari Titik Nol]] | [[17_MASTER_DOKUMENTASI_KESELURUHAN_DAN_RIWAYAT_VERSI|Master Dokumentasi]]*
