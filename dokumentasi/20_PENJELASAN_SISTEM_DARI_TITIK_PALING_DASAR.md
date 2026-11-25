---
title: "20: Penjelasan Sistem Lengkap dari Titik Paling Dasar"
source_doc: "20_PENJELASAN_SISTEM_DARI_TITIK_PALING_DASAR.md"
date: 2026-11-25
updated: 2026-11-25
tags:
  - architecture-guide
  - first-principles
  - bahasa-skills
  - adm-project
aliases:
  - Penjelasan Dasar Sistem ADM
  - First Principles System Guide
status: verified
---

# 20: Penjelasan Sistem Lengkap dari Titik Paling Dasar

Dokumen ini menjelaskan seluruh sistem proyek akhir Analisis Data Multivariat Tim Arete (`5025241135`) mulai dari nol: dari cara komputer melihat satu gambar angka Mandarin hingga bagaimana satu file model berukuran `14,75 MiB` bisa mengenali `2.902` tulisan tangan baru dalam `5,44 detik` dengan akurasi `98,62%`.

---

## 1. Titik Nol: Apa Sebenarnya Data Kita?

Bagi mata manusia, gambar di dataset kita adalah tulisan tangan angka Mandarin dari `0` (`零`) sampai `100.000.000` (`亿`/`億`), berjumlah **15 kelas** (`0` sampai `14`).

Bagi komputer dan matematika multivariat, satu gambar berukuran `64 x 64` piksel keabuan (*grayscale*) bukanlah gambar, melainkan **matriks `64 x 64` angka** intensitas tinta dari `0` sampai `255`. Jika matriks itu diratakan menjadi satu baris panjang (`flatten`), setiap satu gambar berubah menjadi **satu vektor berdimensi `p = 4.096`**:

$$\mathbf{x} = [x_1, x_2, \dots, x_{4096}]^\top \in \mathbb{R}^{4096}$$

Pada tahap pelatihan inti, kita memiliki **10.500 gambar** dari **70 penulis** (`700` gambar per kelas). Artinya, seluruh data latih awal adalah satu tabel raksasa berukuran `10.500 baris x 4.096 kolom`.

### Mengapa Tugas Ini Tidak Sesederhana MNIST Biasa?

1. **Multikolinearitas ekstrem**: Piksel pinggiran gambar hampir selalu kosong (bernilai seragam), dan piksel yang berdekatan saling berkorelasi kuat. Jika langsung dimasukkan ke rumus statistik klasik, matriks kovarians $\mathbf{S}$ menjadi **singular** ($\det(\mathbf{S}) \approx 0$) sehingga tidak bisa diinvers.
2. **Satu kelas punya dua rupa (bimodalitas aksara)**: Kelas `13` (`万` sederhana 3 goresan vs `萬` tradisional 12 goresan) dan kelas `14` (`亿` vs `億`) memiliki bentuk sangat berbeda tetapi memakai label kelas yang sama. Lebih gawat lagi, `10.500` data latih awal sama sekali tidak memiliki contoh bentuk tradisional `萬` dan `億`.
3. **Pergeseran gaya tulisan (*domain shift*)**: Data latih ditulis dengan gaya balok formal rapi (**Kaishu**), sedangkan data uji Fase 1 dan Fase 2 menghadirkan gaya semi-kursif (**Xingshu**) dan kursif cepat (**Caoshu**) di mana beberapa goresan menyatu menjadi satu lengkungan (contoh ekstrem: angka `5` / `五` melingkar seperti huruf `Z` atau `S`, angka `11` / `百` menyusut mirip `万`).
4. **Pergeseran jumlah kemunculan kelas (*label shift*)**: Pada `966` citra kaligrafi Gaya Baru di Fase 1, kelas `0` (`零`) dan kelas `14` (`亿`) tidak muncul sama sekali (`0` sampel), padahal di data uji akhir Fase 2 (`2.902` citra) keduanya muncul kembali.

---

## 2. Lapisan Fondasi: Mengapa Kita Memulai dengan Statistik Multivariat Klasik (Era 0)

Implementasi berada di [`multivariate_analysis.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/multivariate_analysis.py). Sebelum menyentuh *deep learning*, sistem ini menguji data `10.500 x 4.096` secara berjenjang agar kita paham geometri datanya dan bebas kebocoran penulis (*zero writer leakage*).

1. **Kunci Validasi Anti-Bocor (`StratifiedGroupKFold`)**:
   Jika `10.500` gambar diacak biasa, tulisan tangan penulis yang sama bisa masuk ke latih dan validasi sekaligus sehingga akurasi terlihat palsu. Oleh karena itu, seluruh `70` penulis dipartisi per `writer_id`: setiap fold melatih `56` penulis dan menguji `14` penulis yang belum pernah dilihat.
2. **PCA (*Principal Component Analysis*)**:
   PCA memutar dan memadatkan `4.096` sumbu piksel menjadi sumbu-sumbu baru (*Eigen-Characters*) yang saling tegak lurus dan tidak lagi berkorelasi. Hasil terukur pada data kita:
   - `4` komponen teratas (`PC1` `6,67%`, `PC2` `5,62%`, `PC3` `4,93%`, `PC4` `4,09%`) menangkap kontras goresan horizontal, vertikal, dan bingkai utama.
   - `109` komponen sudah cukup menangkap **85%** variansi total; `154` komponen menangkap **90%**; `252` komponen menangkap **95%**; dan kriteria Kaiser ($\bar{\lambda} = 0{,}0875$) mempertahankan `275` komponen. Lebih dari **93%** dimensi piksel asli ternyata redundan secara linier.
3. **MANOVA (*Multivariate Analysis of Variance*)**:
   Pada ruang `20` komponen utama teratas, MANOVA menguji apakah rata-rata ke-15 kelas memang berbeda nyata. Hasilnya sangat tegas: `Wilks' Lambda = 8,8900e-5`, `Pillai's Trace = 5,8418`, `Hotelling-Lawley = 17,6700`, `Roy's Largest Root = 0,8096`, dan `Partial Eta Squared = 0,4864` (jauh di atas ambang efek besar `0,14`). Artinya, pusat ke-15 kelas terpisah nyata.
4. **Uji Box's M & Plot Q-Q Mahalanobis**:
   Walaupun rata-ratanya berbeda, apakah sebaran (*matriks kovarians*) ke-15 kelas seragam? Uji Box's M via `slogdet` menghasilkan `M = 214.542,08` (`m = 20`, `df = 2.940`, `p = 0`) dan `M = 535.061,64` (`m = 40`, `df = 11.480`, `p = 0`). Plot Q-Q Mahalanobis juga memperlihatkan ekor berat yang menyimpang dari normalitas multivariat. Artinya, **matriks kovarians tiap kelas berbeda jauh** ($\boldsymbol{\Sigma}_i \neq \boldsymbol{\Sigma}_j$).
5. **Bukti Empiris: QDA Mengalahkan LDA (+10,96%)**:
   - `PCA(100) + Fisher LDA` (memaksa satu kovarians gabungan untuk semua kelas): **`85,13%`**.
   - `PCA(100) + Logistic Regression`: **`91,08%`**.
   - `PCA(40) + Regularized QDA` (`reg_param = 0,1`, memberi matriks kovarians sendiri per kelas): **`96,09%`**.

**Kesimpulan dari Titik Nol**: Proyeksi piksel linier (PCA) berhenti di `96,09%` karena tidak tahan terhadap pergeseran posisi goresan atau liukan kaligrafi kursif, dan batas keputusannya **wajib non-linier serta mendukung lebih dari satu bentuk per kelas**.

---

## 3. Lapisan Arsitektur Visi & Pengayaan Data (Era I s.d. Era III)

Temuan klasik Era 0 diterjemahkan menjadi desain jaringan saraf dan kurikulum data yang terarah.

### A. Mengapa `StudentDenseNet121_CBAM`? ([`model_densenet_cbam.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/model_densenet_cbam.py))

Pada jaringan ResNet biasa, fitur lapisan awal dijumlahkan (`x + F(x)`), sehingga detail goresan tipis setebal `1–2` piksel (seperti ujung atas `千` agar tidak tertukar dengan `十`) berisiko tertindih. DenseNet-121 memakai **konkatenasi fitur** (*channel concatenation*): peta fitur dari lapisan awal terus diteruskan dan digabung hingga lapisan akhir. Empat modifikasi utama pada arsitektur kita:
1. **Stem Rapat `3x3` (`stride=1`, tanpa `MaxPool` awal)**: Gambar `64 x 64` langsung diproses tanpa pembuangan resolusi mendadak di awal.
2. **MixStyle Pasca-Stem (`p=0,5`, `alpha=0,1`)**: Menukar rata-rata dan simpangan baku fitur antarpenulis di dalam satu mini-batch agar jaringan mengabaikan gaya ketebalan tinta individu dan fokus pada bentuk struktur huruf.
3. **4 Blok Atensi CBAM (`7x7`)**: Dipasang setelah tiap *DenseBlock* untuk memberi bobot lebih besar pada kanal dan lokasi spasial persimpangan goresan yang krusial.
4. **Kepala Ganda (`Sub-Center ArcFace K=2` + `Linear Head`)**: Fitur diringkas ke vektor `256` dimensi ber-normalisasi L2 (`Linear(1024->256) -> BatchNorm1d -> PReLU`). Jawaban langsung atas uji Box's M: setiap kelas diberi **`K = 2` sub-pusat** di `Sub-Center ArcFace` (`s = 32,0`, `m = 0,30`) agar gaya balok Kaishu dan gaya kursif/tradisional punya pusat masing-masing dalam satu kelas yang sama. Saat inferensi, logit keduanya digabung `50/50`.

### B. Kurikulum Data & Augmentasi Fisik

1. **Data Sintetik Tradisional (`700` citra)** ([`generate_traditional_data.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/generate_traditional_data.py)): Menutup kekosongan aksara tradisional `萬` dan `億` sesuai profil blur pengampu.
2. **Augmentasi Biomekanik** ([`biomechanical_augmentation.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/biomechanical_augmentation.py)):
   - *Simard Elastic Distortion* ($\alpha = 7, \sigma = 4$): Melenturkan garis lurus Kaishu secara halus menyerupai tarikan otot kursif tanpa memutus goresan.
   - *Anisotropic Brush Pressure*: Dilasi diagonal `45°` meniru tekanan kuas tebal, erosi `2x2` meniru sapuan cepat kuas kering (*Feibai*).
3. **Kurikulum Kursif NCCU (`2.952` citra terfilter)** ([`lecturer_exact_filter.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/lecturer_exact_filter.py)): Menambahkan contoh kaligrafi kursif asli yang disaring dengan binarisasi Otsu, *tight bounding-box crop* berasio aspek asli, dan *Gaussian blur* ringan agar seragam dengan praproses dosen. Langkah inilah yang memecahkan plafon `94,73%` menuju **`97,57%`** (`Sub 25`).

### C. Adaptasi Prior Otomatis Tanpa Label ([`test_time_prior_adaptation.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/test_time_prior_adaptation.py))

Saat menguji data tanpa label, model standar mengasumsikan semua kelas muncul dengan peluang sama (`1/15`). Padahal pada Gaya Baru Fase 1 kelas `0` dan `14` absen (`0` sampel). Karena *hardcoding* nol dilarang (di Fase 2 kelas `0` dan `14` bisa muncul lagi), sistem memakai **Bayesian EM TTPA** dengan penghalusan Dirichlet (`alpha = 1,05`):
- **E-step**: Menghitung probabilitas posterior tiap gambar memakai prior populasi saat ini.
- **M-step**: Memperbarui perkiraan prior populasi $\pi_k$ dari rata-rata posterior seluruh data uji tanpa menyentuh satu pun label asli.
Dalam `20` iterasi (`5,51 ms`), mekanisme ini otomatis meredam alarm palsu kelas yang jarang/absen saat di Fase 1 dan otomatis mengangkatnya kembali saat kelas `0` dan `14` hadir di Fase 2.

---

## 4. Jembatan ke Model Tunggal: Tomografi MILP & Distilasi 5-Loss (Era IV s.d. Era V)

Pada akhir Fase 1 (`2.466` citra validasi), kita menggabungkan matriks konfusi historis dari `22` submisi ke dalam **Joint Mixed-Integer Linear Programming (MILP)** lewat solver SciPy HiGHS (`4.993` persamaan kendala linier dan `10.290` variabel biner). Audit tomografis ini menemukan bug indeks `val_00459.png` dan kontaminasi masker `148` citra Kaishu pada `Sub 29`, lalu melahirkan **`Sub 30`** dengan akurasi **`99,76%`** (`2.460 / 2.466`, hanya `6` error) serta label referensi **`Sub 31`**.

Namun `Sub 30` memiliki tiga krisis praktis untuk ujian buta Fase 2:
1. MILP bersifat transduktif (butuh matriks konfusi historis, tidak bisa dipakai langsung pada `2.902` citra baru).
2. Pool model riset menumpuk hingga **`3,0 GB`** (`444,55 MiB` untuk ensemble `15` model adaptasi).
3. Beban komputasinya **`135x` *forward pass* per gambar** (`15` model dikali `9-way TTA`).

### Solusi Akhir: Knowledge Distillation (`UnifiedDistillationLoss` + `MeanTeacherEMA` + `SWA`)

Implementasi inti berada di [`distillation_loss.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/distillation_loss.py). Seluruh kecerdasan ensemble guru dipadatkan ke **satu model murid tunggal `StudentDenseNet121_CBAM`** (`7.522.968` parameter) di atas **Super-Dataset `15.918` citra** (`10.500` Train + `2.952` NCCU + `2.466` Fase 1).

1. **Guru Online (`MeanTeacherEMA`, `alpha_EMA = 0,995`)**: Guru melihat gambar bersih (`x_clean`) untuk menghasilkan prediksi probabilitas lunak ($z_T$) dan vektor fitur laten L2 ($f_T \in \mathbb{R}^{256}$) yang stabil.
2. **Murid Belajar pada Kondisi Sulit (`x_aug`)**: Murid menerima gambar yang sudah didistorsi secara biomekanik, lalu dipaksa menyamai guru melalui gabungan **5 fungsi loss serentak**:
   - **Task Loss** (`Label Smoothing = 0,05`): Menjaga akurasi klasifikasi keras terhadap label Super-Dataset dan margin `Sub-Center ArcFace`.
   - **Response Dark Knowledge (`alpha = 0,70`, suhu `T = 3,5`)**: Menyalin probabilitas lunak guru via *KL Divergence* berskala $T^2$ agar murid tahu hubungan kemiripan halus antarhuruf (misal seberapa mirip lengkungan `五` dengan `六` dibanding `一`).
   - **Contrastive Representation Distillation / CRD (`beta = 0,10`, `tau = 0,07`)**: Memakai InfoNCE pada bola satuan $\mathbb{S}^{255}$ agar fitur laten murid dan guru untuk gambar yang sama saling tarik-menarik, sementara terhadap gambar lain saling tolak.
   - **Distance Relational KD (`gamma = 1,00`)**: Menjaga jarak Euclidean berpasangan antar-gambar dalam batch agar sama antara ruang laten guru dan murid.
   - **Angle Relational KD (`delta = 2,00`)**: Menjaga sudut kosinus triplet antar-gambar agar bentuk geometri manifold tidak berubah saat dikompresi.
3. **Stochastic Weight Averaging (`SWA`, Epoch `17–25`, `swa_lr = 1e-4`) & `FP16`**: Alih-alih berhenti di satu titik tajam, bobot dari 9 epoch terakhir dirata-ratakan, statistik *BatchNorm* diperbarui ulang (`update_bn`), lalu disimpan dalam presisi `FP16` ([`densenet121_cbam_student_fp16.pt`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/models/densenet121_cbam_student_fp16.pt), **`14,75 MiB`**). Prediksi `FP16` terverifikasi **100% identik** dengan `FP32`.

---

## 5. Cara Kerja Sistem Saat Menerima Gambar Baru (Inferensi Produksi Fase 2)

Implementasi berada di [`predict_phase2_distillation.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/predict_phase2_distillation.py). Ketika sistem menerima `2.902` gambar baru yang benar-benar belum pernah dilihat (`0` irisan hash MD5), alurnya sangat ringkas:

1. **Praproses**: Gambar `64 x 64` grayscale dinormalisasi ke skala `[-1, 1]`.
2. **1x Forward Pass (Tanpa TTA)**: Seluruh `2.902` gambar melewati `densenet121_cbam_student_fp16.pt` dalam `Batch-128` di GPU RTX 4060 hanya sekali jalan (`5,44 detik` total), menghasilkan `z_S = 0,5 * z_lin + 0,5 * z_arc`.
3. **Kalibrasi Prior Dinamis (TTPA)**: Softmax awal disesuaikan secara otomatis melalui Bayesian EM (`alpha = 1,05`, `source_counts` empiris) untuk mendeteksi proporsi kelas aktual di batch uji (termasuk mendeteksi hadirnya kembali kelas `0` dan `14`).
4. **Keluaran Akhir**: Menghasilkan `predictions.csv` siap kirim ke CodaBench (`0,9984` Fase 1; `0,9862` Fase 2), sedangkan seluruh plot evaluasi statistik dan kurva pembuktian terangkum di [`10_mnist_adm_visual_evaluation_and_diagnostics.ipynb`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/notebooks/10_mnist_adm_visual_evaluation_and_diagnostics.ipynb).
   - **Uji Ketahanan Noise Gaussian ($\sigma = 25$)**: SWA `FP16` mempertahankan stabilitas **`95,98% (±0,02%)`** pada Fase 2 (unggul `+1,61%` atas `densenet121_cbam_student_best.pt` Epoch 13 pra-SWA), dan pada stress-test Fase 1 bertahan di **`96,19%`** dibanding ConvNeXt (`42,70%`) maupun ResNet-18 (`36,90%`).

---
*Navigasi Vault: [[INDEX|Indeks Utama Dokumentasi]] | [[21_PANDUAN_PAHAM_BULLETPROOF_PRESENTASI_DOSEN|Panduan Bulletproof Dosen]] | [[19_TEKS_PRESENTER_PPT_VERBATIM|Naskah Presenter PPT]] | [[17_MASTER_DOKUMENTASI_KESELURUHAN_DAN_RIWAYAT_VERSI|Master Dokumentasi]]*
