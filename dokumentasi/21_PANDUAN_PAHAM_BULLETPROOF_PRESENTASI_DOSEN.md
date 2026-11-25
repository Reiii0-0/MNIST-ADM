---
title: "21: Panduan Pemahaman Mendalam & Pertahanan Bulletproof Presentasi Dosen"
source_doc: "21_PANDUAN_PAHAM_BULLETPROOF_PRESENTASI_DOSEN.md"
date: 2026-11-25
updated: 2026-11-25
tags:
  - presentation-defense
  - bulletproof-qa
  - multivariate-analysis
  - knowledge-distillation
  - adm-project
aliases:
  - Panduan Bulletproof Presentasi ADM
  - Bekal Paham Sidang PPT
status: verified
---

# 21: Panduan Pemahaman Mendalam & Pertahanan Bulletproof Presentasi Dosen

Dokumen ini disusun sebagai bekal persiapan sebelum Anda mempresentasikan **29 slide** dek Canva (`FP_Analisis_Data_Multivariat_Tim_Arete_5025241135_FINAL.pdf`) maupun menjawab pertanyaan dosen pengampu dan asisten dosen. Fokusnya bukan menghafal kalimat satu per satu, melainkan memastikan Anda menguasai **hubungan sebab-akibat di balik setiap keputusan**, **angka kunci yang wajib dibedakan**, serta **15 pertanyaan jebakan paling tajam** beserta jawaban jujur berdasar bukti empiris repositori [`FinalProject/`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject).

---

## 1. Inti Cerita dalam 60 Detik (Jangan Pernah Terputus)

Jika dosen hanya memberi waktu satu menit untuk merangkum seluruh proyek, benang merahnya terdiri atas lima mata rantai sebab-akibat:

1. **Titik Nol (Data 4.096-D)**: Satu gambar `64 x 64` piksel keabuan adalah vektor multivariat `p = 4.096`. Matriks kovarians mentahnya singular karena piksel tepi bernilai nol seragam dan piksel tetangga berkorelasi kuat, sementara pada `10.500` citra latih awal aksara tradisional `萬`/`億` belum ada sama sekali dan gaya tulis uji bergeser ke kursif (*domain shift* + *label shift*).
2. **Temuan ADM Klasik (Era 0)**: Setelah direduksi lewat PCA (85% variansi pada `109` komponen, 90% pada `154`, 95% pada `252`, Kaiser `275`), uji MANOVA ($m=20$ PC) membuktikan rata-rata ke-15 kelas berbeda sangat nyata ($\Lambda_{\text{Wilks}} = 8{,}8900 \times 10^{-5}$, $\eta_p^2 = 0{,}4864$). Namun uji Box's M via `slogdet` menolak kesamaan kovarians ($M = 214.542,08$, $p = 0$), yang langsung terbukti pada CV 5-fold terisolasi penulis: **Regularized QDA (`96,09%`)** melampaui **Fisher LDA (`85,13%`)** sebesar **`+10,96%`**.
3. **Jembatan ke Arsitektur Deep Learning (Era I-III)**: Basis PCA linier mentok di `96,09%` saat menghadapi liukan kaligrafi kursif (*Xingshu/Caoshu*). Karena itu kita mengganti PCA dengan **DenseNet-121 CBAM** (menjaga detail goresan tipis lewat konkatenasi fitur), mengganti asumsi kovarians tunggal dengan **Sub-Center ArcFace ($K = 2$)** untuk dua rupa (balok Kaishu dan kursif/tradisional), menambah **MixStyle + Augmentasi Biomekanik + Kurikulum NCCU (`2.952` citra)**, dan mengatasi *label shift* tanpa *hardcoding* lewat **Bayesian EM TTPA** (`alpha = 1,05`).
4. **Alasan Meninggalkan `Sub 30` (Era IV -> Era V)**: Tomografi MILP pada `Sub 30` memang meraih **`99,76%`**, tetapi sifatnya transduktif (butuh matriks konfusi historis) dan memakai pool model `3,0 GB` (`444,55 MiB` untuk `15` model adaptasi dengan `135x` *forward pass*). Cara itu tidak bisa langsung dibawa ke data buta Fase 2 yang kuotanya ketat (`6` submisi total).
5. **Solusi Final (`Sub 32` & Fase 2)**: Seluruh kecerdasan ensemble dipadatkan lewat **Knowledge Distillation 5-Loss** (`Task + Response KL + CRD + RKD-dist + RKD-angle`) dengan guru online `MeanTeacherEMA` dan perataan bobot **SWA Epoch 17-25** ke **satu file `densenet121_cbam_student_fp16.pt` berukuran `14,75 MiB`**. Hasilnya **`99,84%`** (`4` error dari `2.466`) di Fase 1 dan **`98,62%`** (~`2.862 / 2.902` benar) di uji buta tersegel Fase 2 hanya dalam **`5,44 detik`** sekali jalan (`1x forward pass`).

---

## 2. Tiga Kelompok Angka yang Wajib Anda Bedakan Secara Tegas

Kesalahan paling berbahaya saat sidang adalah mencampuradukkan tiga jenis akurasi di bawah ini. Hafalkan batasnya:

| Jenis Evaluasi | Jumlah Sampel & Sifat Data | Skor Terukur | Bagaimana Cara Menjelaskannya ke Dosen |
| :--- | :--- | :---: | :--- |
| **OOF 5-Fold CV (`Pra-Fase`)** | `10.500` citra Kaishu, `70` penulis (`56` latih vs `14` uji via `StratifiedGroupKFold`) | LDA **`85,13%`**, LogReg **`91,08%`**, Reg-QDA **`96,09%`**, ResNet-18 **`99,79%`**, ConvNeXt **`99,55%`**, DenseNet v2 **`99,76%`** | Validasi lokal bebas kebocoran penulis sebelum menyentuh Fase 1/2 |
| **Fase 1 (`Development Benchmark`)** | `2.466` citra (`1.500` Chinese MNIST penulis 71-80 + `966` Gaya Baru kursif; kelas `0` & `14` = `0` di Gaya Baru) | Sub 1: `91,20%`, Sub 3 Rank 1: `93,19%`, Sub 25: `97,57%`, Sub 30 MILP: `99,76%`, **Sub 32 Student SWA: `99,84%`** | Sampai Sub 30 bersifat *out-of-sample* plus perbaikan transduktif MILP; pada `Sub 32`, `2.466` citra Fase 1 ikut masuk ke Super-Dataset `15.918` sehingga `0,9984` kami klaim jujur sebagai *development benchmark* |
| **Fase 2 (`Sealed Unseen Blind Test`)** | `2.902` citra penulis baru (`0` duplikat internal, `0` irisan hash MD5 vs `15.918` citra latih/Fase 1) | **`0,9862`** (`~2.862 / 2.902` benar, **`5,44 detik`** Batch-128, kuota `1/6`, `ID 1048381`) | **Bukti generalisasi OOD sejati** yang benar-benar belum pernah dilihat model maupun solver MILP sebelumnya |

---

## 3. Konsep Teknis Kunci yang Harus Benar-Benar Anda Pahami

### A. Mengapa `det(S) = 0` Padahal `N = 10.500 > p = 4.096`?
Biasanya matriks singular terjadi saat sampel lebih sedikit dari fitur ($N < p$). Di proyek kita $N = 10.500$ lebih besar dari $p = 4.096$, tetapi matriks kovarians sampel $\mathbf{S}$ tetap singular (`nearly rank-deficient`). Penyebabnya ada dua:
1. Piksel tepi kanvas `64 x 64` hampir selalu putih/nol konstan pada seluruh `10.500` gambar sehingga variansinya nyaris nol.
2. Piksel yang bertetangga membentuk kombinasi linier yang sangat mirip (**multikolinearitas spasial ekstrem**). Akibatnya nilai eigen terkecil jatuh mendekati nol sehingga invers $\mathbf{S}^{-1}$ meledak.

### B. Mengapa QDA Unggul `+10,96%` Atas LDA, dan Mengapa Dimensi PCA-nya Berbeda (`m=40` vs `m=100`)?
- **Fisher LDA** mengasumsikan semua 15 kelas punya elips sebaran yang sama ($\boldsymbol{\Sigma}_0 = \boldsymbol{\Sigma}_1 = \dots = \boldsymbol{\Sigma}_{14} = \boldsymbol{\Sigma}_{\text{pooled}}$). Padahal secara fisik karakter `一` (1 goresan datar) elipsnya gepeng sempit, sedangkan `萬`/`億` (12-15 goresan) menyebar ke banyak arah. Uji **Box's M** membuktikan asumsi homogenitas itu salah total ($M = 214.542,08$, $p = 0$).
- **Regularized QDA** memberi matriks kovarians sendiri $\hat{\boldsymbol{\Sigma}}_k(\gamma) = (1-\gamma)\mathbf{S}_k + \gamma\mathbf{S}_{\text{pooled}}$ (`reg_param = 0,1`) untuk tiap kelas sehingga batasnya melengkung kuadratik dan akurasinya melompat dari `85,13%` ke `96,09%`.
- **Alasan `m = 40` untuk QDA dan `m = 100` untuk LDA**: Satu matriks kovarians simetris berdimensi $m \times m$ memiliki $\frac{m(m+1)}{2}$ parameter bebas per kelas. Jika memakai `m = 100`, QDA harus mengestimasi `5.050` parameter kovarians di setiap kelas padahal sampel per kelas per fold di CV hanya sekitar ratusan citra. Dengan menurunkan `m = 40` (`820` parameter kovarians per kelas) plus shrinkage `reg_param = 0,1`, QDA stabil secara numerik.

### C. Kenapa Ada 112 dan 133 Prediksi Kelas 0/14 Padahal Katanya Absen di Gaya Baru?
Jangan sampai terpeleset di angka ini saat menjelaskan Slide 16 atau Grafik TTPA (`info_ttpa_prior_shift.png`):
- Di total **`2.466` citra validasi Fase 1**, terdapat **`1.500` citra Chinese MNIST formal** (100 sampel per kelas x 15 kelas -> sehingga **memang ada 100 sampel asli Kelas `0` dan 100 sampel asli Kelas `14`**) ditambah **`966` citra kaligrafi Gaya Baru** (yang di dalamnya kelas `0` dan `14` berjumlah **0 sampel**).
- Karena itu pada `Champion_Blend_85_15`, model pra-TTPA memprediksi Kelas `0` sebanyak `112` kali (`+12` di atas `100`) dan Kelas `14` sebanyak `133` kali (`+33` di atas `100`), total **`45` kesalahan prediksi yang terbuang**.
- Setelah **Bayesian EM TTPA (`alpha = 1,05`, `20` iterasi, `5,51 ms`)** dijalankan tanpa menyentuh label sama sekali, sistem menekan **`11`** prediksi palsu Kelas `14` (`133 -> 122`), menekan **`2`** prediksi palsu Kelas `0` (`112 -> 110`), dan **memulihkan `23` prediksi Kelas `4` (`四`, `180 -> 203`)** karena struktur radikalnya sering tertukar dengan `亿`.

### D. Mengapa Student Tunggal SWA `14,75 MiB` Mengalahkan Ensemble `3,0 GB`?
1. **Efek Pemulusan Suhu (`T = 3,5`) + `MeanTeacherEMA` (`alpha_EMA = 0,995`)**: Kunci referensi `Sub 31` masih menyisakan `4` kesalahan residu bawaan di kasus kursif paling ekstrem. Target lunak guru EMA tidak memaksa murid menghafal empat error keras tersebut (`Label Smoothing 0,05` + `KL T=3,5`), sehingga murid turun dari `6` error (`Sub 30`: `99,76%`) menjadi hanya **`4` error** (`Sub 32`: `99,84%`).
2. **Menyatukan Kekuatan Antar-Domain Tanpa Konflik Logit**: Alih-alih menggabung dua keluarga model lewat rata-rata kasar pasca-inferensi, murid belajar ujung-ke-ujung dengan **Contrastive CRD (`tau = 0,07`)**, **Relational RKD Jarak & Sudut**, serta **`Sub-Center ArcFace` ($K = 2$)**.
3. **Stochastic Weight Averaging (`SWA` Epoch 17-25) + `update_bn`**: Merata-ratakan lintasan bobot 9 epoch terakhir membawa parameter ke dasar lembah loss yang datar (*flat minima*), lalu presisi diringkas ke **FP16** (`29,28 MiB -> 14,75 MiB`) dengan prediksi **100% identik (`2.902 / 2.902`)** terhadap FP32.

### E. Mengapa Kode di GitHub Punya `9` Kernel Kaggle Tanpa Grafik + `Notebook 10` Visual?
Agar siap bila dosen atau asisten dosen membuka https://github.com/Reiii0-0/MNIST-ADM secara langsung:
- **Notebook `01` s.d. `09`** adalah kernel pelatihan GPU Tesla T4 Kaggle yang memang ditulis tanpa *plotting* `matplotlib` agar tidak membebani waktu GPU saat melatih 5-fold dan distilasi (`execution_count` dan log `stdout` asli Tesla T4 sudah disematkan ke semua selnya).
- **Notebook `10`** ([`10_mnist_adm_visual_evaluation_and_diagnostics.ipynb`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/notebooks/10_mnist_adm_visual_evaluation_and_diagnostics.ipynb)) merangkum dan menjalankan secara lokal seluruh evaluasi visual (`7` figur `image/png` tertanam langsung di sel notebook) membaca keluaran nyata dari `outputs/` dan `eval_results/`.

---

## 4. Bank 15 Pertanyaan Jebakan Dosen & Asdos (Jawaban Bulletproof)

### Kelompok A: Teori & Diagnostik ADM Klasik

1. **Tanya**: *"Kenapa matriks kovarians kalian singular padahal $N = 10.500$ lebih besar dari $p = 4.096$?"*
   - **Jebakan**: Menjawab asal *"karena dimensinya besar"* atau keliru menyebut $N < p$.
   - **Jawaban Bulletproof**: *"$N = 10.500$ memang lebih besar daripada $p = 4.096$, Pak/Bu, tetapi matriks kovarians sampel tetap singular secara numerik (`det(S) ~ 0`) karena dua hal: piksel tepi kanvas 64x64 bernilai konstan nol di ribuan sampel sehingga variansinya nyaris nol, dan piksel bertetangga membentuk kolinearitas spasial yang ekstrem. Itulah alasan PCA wajib mereduksi dimensi dulu ke 20-100 komponen sebelum invers kovarians dihitung pada MANOVA, Box's M, dan LDA/QDA."*

2. **Tanya**: *"Di plot Q-Q Mahalanobis normalitas multivariat menyimpang di ekor, lalu di Box's M kovarians tidak homogen ($p=0$). Apakah hasil MANOVA kalian gugur?"*
   - **Jebakan**: Panik dan mengatakan *"iya MANOVA tidak valid."*
   - **Jawaban Bulletproof**: *"Tidak gugur, Pak/Bu, karena ada dua pertimbangan yang kami verifikasi langsung. Pertama, pada sampel besar ($N = 10.500$, 700 sampel per kelas), teorema batas pusat multivariat menstabilkan uji vektor mean; kedua, kami tidak bergantung pada Wilks' Lambda semata ($\Lambda = 8{,}89 \times 10^{-5}$), melainkan juga memeriksa **Pillai's Trace** (`5,8418`) yang dikenal paling robust terhadap pelanggaran homogenitas kovarians serta **Partial Eta Squared** (`0,4864`, efek pemisahan sangat besar). Penolakan Box's M justru kami jadikan bukti empiris mengapa QDA (`96,09%`) jauh mengungguli LDA (`85,13%`)."*

3. **Tanya**: *"Kenapa di Box's M dan QDA kalian pakai `m=20` atau `m=40` PC, sementara di LDA dan LogReg kalian pakai `m=100` PC?"*
   - **Jebakan**: Mengaku *"cuma tebak-tebakan parameter."*
   - **Jawaban Bulletproof**: *"Di QDA setiap kelas mengestimasi matriks kovariansnya sendiri berjumlah $\frac{m(m+1)}{2}$ parameter bebas. Pada `m=100`, ada `5.050` parameter per kelas untuk sekitar `560-600` sampel per fold sehingga kovarians tidak stabil. Pada `m=40`, jumlah parameternya turun ke `820` dan distabilkan dengan shrinkage `reg_param = 0,1`. Sedangkan untuk MANOVA dan uji Box's M lintas 15 kelas, `m=20` dan `m=40` dihitung via `slogdet` agar tetap stabil secara logaritmik."*

4. **Tanya**: *"Apa yang terlihat pada `16 Eigen-Characters` dan apa batas kemampuan PCA di proyek ini?"*
   - **Jawaban Bulletproof**: *"Empat eigen-character teratas (`PC1 6,67%`, `PC2 5,62%`, `PC3 4,93%`, `PC4 4,09%`) memisahkan orientasi global goresan horizontal terhadap vertikal dan energi bingkai luar, sedangkan PC selanjutnya menangkap lekukan persimpangan. Batasnya, PCA adalah proyeksi linier tak-berawasi: begitu posisi goresan bergeser beberapa piksel atau hurufnya meliuk di gaya Caoshu, skor proyeksi PCA bercampur antara identitas huruf dan gaya kuas penulis."*

### Kelompok B: Integritas Data, Leakage & Etika Kompetisi

5. **Tanya**: *"Data validasi Fase 1 (2.466 citra) kalian masukkan ke Super-Dataset 15.918 citra saat training Student `Sub 32`. Itu bocor data (data leakage), kan?"*
   - **Jebakan**: Membela diri bahwa `99,84%` Fase 1 adalah bukti generalisasi murni.
   - **Jawaban Bulletproof**: *"Jika `99,84%` Fase 1 diklaim sebagai uji *out-of-sample*, betul itu tidak sah, Pak/Bu, dan justru karena itu di Slide 25 kami memisahkannya secara eksplisit. Evaluasi Fase 1 (`0,9984`) hanya kami posisikan sebagai **development benchmark**, sedangkan bukti generalisasi murni tanpa kebocoran ada di **Fase 2 (`2.902` citra blind test penulis baru)** yang sudah diaudit memiliki **0 duplikat internal dan 0 irisan hash MD5** terhadap ke-15.918 citra pelatihan. Di data tersegel Fase 2 itu, model Student yang sama langsung meraih **`0,9862`** sekali jalan dalam `5,44 detik`."*

6. **Tanya**: *"MILP Tomografi di Sub 26-30 memanfaatkan feedback matriks konfusi CodaBench. Bukankah itu eksploitasi leaderboard?"*
   - **Jawaban Bulletproof**: *"MILP itu kami lakukan pada tahap eksplorasi Fase 1 untuk dua hal: memahami batas atas kesalahan di 966 citra kursif dan mengaudit dua bug sistem yang tersembunyi di data/skrip (`val_00459.png` serta kontaminasi masker 148 citra Kaishu di Sub 29). Namun kami menyadari penuh bahwa metode transduktif seperti MILP **tidak sah secara saintifik maupun tidak mungkin diterapkan** pada ujian buta Fase 2. Karena itu seluruh proses MILP ditinggalkan di Fase 1 dan kami membangun model murni induktif via Knowledge Distillation `14,75 MiB`."*

7. **Tanya**: *"Kenapa di validasi lokal kalian membagi data pakai `StratifiedGroupKFold` berdasarkan `writer_id`, bukan `StratifiedKFold` biasa?"*
   - **Jawaban Bulletproof**: *"Karena setiap dari `70` penulis menulis seluruh `15` kelas masing-masing `10` kali (`10.500` citra). Jika diacak dengan Stratified biasa, tulisan orang yang sama berada di train dan validation sehingga model cukup mengenali ketebalan kuas penulisnya. Dengan `StratifiedGroupKFold`, `56` penulis di train benar-benar terpisah dari `14` penulis di validation (100% simulasi penulis baru)."*

8. **Tanya**: *"Di Fase 1 kelas 0 dan 14 tidak muncul pada 966 citra Gaya Baru. Bagaimana kami yakin kalian tidak meng-hardcode nol untuk kedua kelas itu sehingga model hancur di Fase 2?"*
   - **Jawaban Bulletproof**: *"Buktinya terlihat di distribusi prediksi Fase 2 (`2.902` citra) dan di kode `test_time_prior_adaptation.py`, Pak/Bu. Kami memakai **Bayesian EM TTPA dengan penghalusan Dirichlet `alpha = 1,05`** (`alpha > 1` secara matematis menjamin prior tidak pernah bernilai nol). Di Fase 1, EM otomatis menurunkan prior kelas 0 dan 14 yang jarang, sementara saat Fase 2 dijalankan tanpa perubahan kode apa pun, EM kembali mendeteksi kemunculan kelas 0 (5 kali) dan kelas 14 (1 kali sebagai kandidat akhir bersih) beserta distribusi kelas lainnya secara otonom."*

### Kelompok C: Arsitektur Deep Learning, Distilasi & Kuantisasi

9. **Tanya**: *"Bagaimana ceritanya 1 model murid `14,75 MiB` lebih akurat (`0,9984`, 4 error) daripada ensemble gurunya (`0,9976`, 6 error)?"*
   - **Jawaban Bulletproof**: *"Ada tiga penyebab terukur. Pertama, target lunak dari `MeanTeacherEMA` dengan suhu $T = 3{,}5$ memperhalus 4 label residu keras yang masih bermasalah di `Sub 31` sehingga murid tidak ikut menghafal noise label. Kedua, 5 komponen loss menjaga bukan hanya output akhir (`KL`), melainkan juga geometri jarak Euclidean dan sudut triplet (`RKD` + `CRD`) di ruang hipersfer 256-D. Ketiga, **SWA Epoch 17-25** menaruh bobot di pusat lembah datar (*flat minima*), mengurangi varian prediksi individu."*

10. **Tanya**: *"Dari `densenet121_cbam_student_best.pt` (29,28 MiB FP32) ke `densenet121_cbam_student_fp16.pt` (14,75 MiB FP16), apakah penyusutan presisi menurunkan akurasi?"*
    - **Jawaban Bulletproof**: *"Tidak sama sekali di data kita, Pak/Bu. Sudah kami audit perbandingan langsung pada `2.902` citra Fase 2: prediksi argmax `densenet121_cbam_student_fp16.pt` **100,00% identik (`2.902 / 2.902` sama)** dengan model FP32, sementara ukurannya menyusut separuh dan kecepatannya mencapai `5,44 detik` di RTX 4060. Bahkan ketika diberi derau Gaussian $\sigma = 25$ di 3 seed berbeda (`42, 43, 44`), SWA FP16 mencatat stabilitas **`95,98% ± 0,02%`**, unggul **`+1,61%`** dibanding `densenet121_cbam_student_best.pt` Epoch 13 pra-SWA (`94,37% ± 0,24%`)."*

11. **Tanya**: *"Kenapa memilih DenseNet-121 CBAM dan tidak pakai Transformer seperti ViT atau Swin?"*
    - **Jawaban Bulletproof**: *"Resolusi citra kita kecil (`64 x 64`) dan dataset inti setelah kurasi berkisar `15.918` citra. Pada rezim resolusi $64 \times 64$ dengan goresan tipis `1-2` piksel (seperti pembeda `千` vs `十`), mekanisme *feature concatenation* DenseNet dengan stem `3x3` stride-1 (tanpa `MaxPool` awal) menjaga peta fitur tepi halus tetap hidup sampai lapisan akhir, jauh lebih efisien (`7,52M` parameter) dibanding ViT murni yang minim bias induktif spasial pada data terbatas."*

12. **Tanya**: *"Apa perbedaan `MixStyle`, `Sub-Center ArcFace`, dan `MeanTeacherEMA`? Jangan sampai fungsi ketiganya tertukar."*
    - **Jawaban Bulletproof**: *"Ketiganya beroperasi di titik yang berbeda: **`MixStyle`** menyentuh statistik peta fitur di awal jaringan pasca-stem (`p=0,5, alpha=0,1`) untuk mencampur gaya ketebalan kuas antarpenulis; **`Sub-Center ArcFace`** berada di ujung kepala klasifikasi untuk menyediakan $K=2$ pusat prototipe per kelas (menjawab bimodalitas Kaishu vs Kursif/Tradisional); sedangkan **`MeanTeacherEMA`** adalah mekanisme guru saat distilasi (`alpha=0,995`) yang memproses citra bersih untuk membimbing murid yang sedang mempelajari citra teraugmentasi biomekanik."*

### Kelompok D: Pertanyaan Praktis Asdos Soal Kode, Notebook & Reproduksi

13. **Tanya**: *"Kenapa di notebook Kaggle `01` sampai `09` tidak muncul grafik `matplotlib`, malah hanya log teks per epoch?"*
    - **Jawaban Bulletproof**: *"Di Kaggle kesembilan skrip tersebut dijalankan murni untuk memaksimalkan kuota GPU saat training 5-fold, menghasilkan bobot `.pt`, OOF `.npy`, dan paket `.zip` submission tanpa overhead rendering gambar; semua hasil eksekusi Tesla T4 aslinya sudah tertanam di tiap sel `01-09`. Seluruh plot statistik dan evaluasi empirisnya disusun terpisah secara modular di `src/` lalu kami satukan menjadi Notebook `10` ([`10_mnist_adm_visual_evaluation_and_diagnostics.ipynb`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/notebooks/10_mnist_adm_visual_evaluation_and_diagnostics.ipynb)) agar `7` figur evaluasinya bisa langsung dilihat di GitHub."*

14. **Tanya**: *"Di GitHub kenapa ada 1 file notebook `05_mnist_adm_fixmatch.ipynb` yang status log-nya `CANCEL_ACKNOWLEDGED` di Fold 5 Epoch 3?"*
    - **Jawaban Bulletproof**: *"Transparansi riwayat eksperimen kami pertahankan apa adanya, Pak/Mas. Pada `Track 2 FixMatch` untuk ResNet-18, Fold 1 sampai 4 selesai penuh hingga checkpoint tersimpan, lalu dihentikan di Fold 5 Epoch 3 ketika strategi kami beralih mengutamakan cabang **Track 1 STM (`Sub 16: 0,9469`)**, **Track 3 ConvNeXt-FixMatch** (selesai penuh 5-fold), dan **Kurikulum NCCU** yang hasilnya jauh lebih menentukan."*

15. **Tanya**: *"Apa bukti bahwa proyek ini tidak memakai alat atau skrip tersembunyi di luar repositori?"*
    - **Jawaban Bulletproof**: *"Seluruh kode reproduksi mulai dari ADM klasik (`src/multivariate_analysis.py`), filter morfologi (`src/lecturer_exact_filter.py`), arsitektur dan 5-loss KD (`src/model_densenet_cbam.py`, `src/distillation_loss.py`), inferensi Fase 2 (`src/predict_phase2_distillation.py`), pembangkit `24` figur Keynote, hingga 10 notebook `.ipynb` terbuka publik di `https://github.com/Reiii0-0/MNIST-ADM` lengkap dengan `requirements.txt` dan indeks dokumentasi."*

---

## 5. Checklist Peta Presentasi 29 Slide (1 Kalimat Kunci per Slide)

Gunakan daftar ini 3 menit sebelum giliran Anda presentasi agar urutan alur slide selalu menempel di kepala:

| Slide | Judul Inti | Poin Wajib Terucap |
| :---: | :--- | :--- |
| **1-2** | Sampul & Identitas | Tim `arete` (`5025241135`): menjembatani teori ADM klasik ke model `14,75 MiB` (`99,84%` Fase 1, `98,62%` Fase 2). |
| **3-4** | Objek 15 Kelas & Masalah Multivariat | Gambar `64x64` = vektor `4.096-D`; kendala kolinearitas, bimodalitas `万/萬`, domain shift `Kaishu->Caoshu`, dan label shift. |
| **5-6** | Peta 6 Era & 5 Sumber Data | Sintesis `700` aksara tradisional, kunci `StratifiedGroupKFold` per `writer_id`, hingga Super-Dataset `15.918` + `2.902` Fase 2. |
| **7-8** | Era 0A-0B: PCA, Eigen-Char & Scree | `85%` = `109` PC, `90%` = `154` PC, `95%` = `252` PC, Kaiser = `275` PC; $>93\%$ dimensi piksel redundan secara linier. |
| **9-10** | Era 0C-0D: MANOVA & Box's M (`LDA` vs `QDA`) | Mean 15 kelas beda nyata ($\Lambda=8{,}89\times 10^{-5}$), tapi kovarians beda jauh ($M=214.542$); QDA (`96,09%`) menang `+10,96%` atas LDA. |
| **11** | Jembatan Matematis ke Deep Learning | Empat penyakit ADM klasik diterjemahkan menjadi DenseNet-121 CBAM, CRD/RKD, Sub-Center ArcFace (`K=2`), dan EM TTPA. |
| **12-13** | Era I Kejutan Fase 1 & 5 Karakter Sulit | Chinese MNIST `100%`, tapi 966 Gaya Baru memunculkan 168 error di `五` (`51%`), `百` (`31%`), `四`, `千`, dan `十`. |
| **14-16** | Era II DenseNet-CBAM, Biomekanik & EM TTPA | Konkatenasi fitur + Simard elastis ($\alpha=7,\sigma=4$) + kuas `45°` + EM TTPA (`20` iter, pulihkan `+23` kelas 4 tanpa hardcode). |
| **17-18** | Era III Plafon `94,73%` & Terobosan NCCU | Ablasi `drop D1` dan STM mentok di `94,73%`; terpecahkan oleh `2.952` kursif NCCU terfilter (`Sub 25: 97,57%`). |
| **19-20** | Era IV MILP (`99,76%`) & Mengapa Berhenti | MILP mengungkap bug `val_00459.png` dan masker `148` citra, tapi transduktif + `3,0 GB` ensemble -> wajib beralih ke Distilasi. |
| **21-22** | Era V Pipeline 5-Loss KD & `MeanTeacherEMA` | Guru EMA bersih membimbing murid teraugmentasi lewat `Task + KL (T=3,5) + CRD + RKD-dist + RKD-angle` + SWA Ep 17-25. |
| **23-24** | Rekor `Sub 32` (`99,84%`) & Audit `best` vs `fp16` | Murid `14,75 MiB` turun ke `4` error (`五`/`百` 100%); SWA `FP16` `+1,61%` lebih tahan noise $\sigma=25$ dibanding Epoch 13. |
| **25-27** | Kejujuran 4 Lapis & Blind Test Fase 2 (`98,62%`) | Pisahkan Fase 1 (in-sample dev) dari `2.902` citra tersegel Fase 2 (`0` MD5 overlap, `5,44 detik`, TTPA otonom, sisa kuota `5/6`). |
| **28-29** | Peta Kode, 10 Notebook & Penutup | Kode `src/`, `9` notebook Kaggle (`+ stdout`) + `Notebook 10` visual, serta dokumentasi publik di `https://github.com/Reiii0-0/MNIST-ADM`. |

---

## 6. Dua Aturan Emas Saat Menjawab Dosen
1. **Jangan pernah mengarang angka yang belum diuji**: Jika dosen menanyakan ablasi yang tidak kita jalankan (misalnya *"Bagaimana jika `CRD` dimatikan tapi `T=5` di Fase 2?"*), jawab lugas: *"Kombinasi parameter spesifik tersebut belum kami uji di kuota submisi Fase 2; yang terukur saat ini adalah perbandingan SWA vs Epoch 13 di Fase 2 (`+1,61%` stabilitas noise) serta ablasi Fase 1 (Sub 1-32)."*
2. **Selalu pisahkan klaim Fase 1 vs Fase 2**: Jika dosen menguji soal data leakage, Anda justru mendahului dengan menunjukkan Slide 25 bahwa tim sudah mengaudit `densenet121_cbam_student_best.pt` vs `densenet121_cbam_student_fp16.pt` dan memvalidasinya pada `2.902` citra blind test Fase 2.

---
*Navigasi Vault: [[INDEX|Indeks Utama Dokumentasi]] | [[19_TEKS_PRESENTER_PPT_VERBATIM|Naskah Presenter Verbatim (Slide 1-29)]] | [[20_PENJELASAN_SISTEM_DARI_TITIK_PALING_DASAR|Penjelasan Sistem dari Titik Nol]] | [[17_MASTER_DOKUMENTASI_KESELURUHAN_DAN_RIWAYAT_VERSI|Master Dokumentasi]]*
