---
title: "19: Teks Presenter Verbatim PPT (Slide 1 s.d. Slide 29 + Q&A Sidang)"
source_doc: "19_TEKS_PRESENTER_PPT_VERBATIM.md"
date: 2026-11-24
updated: 2026-11-25
tags:
  - presentation-script
  - bahasa-skills
  - antigravity-workflow
  - adm-project
aliases:
  - Naskah Presenter PPT ADM
  - Speaker Notes 29 Slide
status: verified
---

# 19: Teks Presenter Verbatim PPT (Slide 1 s.d. Slide 29 + Tanya Jawab Sidang)

Naskah ini disusun untuk dibacakan langsung saat presentasi akademik mata kuliah Analisis Data Multivariat. Kalimat dirancang lugas dalam ragam lisan konsultatif-ilmiah sesuai EYD V, tanpa klise pembuka, dan merujuk hanya pada angka serta modul kode yang terverifikasi di repositori [`FinalProject/`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject). Bagian penutup juga menjelaskan secara terbuka pemanfaatan Google Antigravity, subagen riset paralel, validasi GPU lokal, dan integrasi Model Context Protocol (MCP) Canva.

---

## Bagian I: Pemaparan Slide demi Slide (Slide 1 s.d. Slide 29)

### Slide 1: Sampul Utama (`Representasi Laten & Knowledge Distillation Pengenalan Karakter Mandarin`)
Selamat pagi Bapak dan Ibu dosen serta rekan-rekan sekalian. Hari ini saya Farikh Muhammad Fauzan, NRP 5025241135, dari tim arete akan memaparkan perjalanan riset proyek akhir Analisis Data Multivariat kami: bagaimana analisis statistik multivariat klasik dari ruang 4.096 piksel memandu perancangan model tunggal DenseNet-121 CBAM berukuran 14,75 MiB yang meraih akurasi 99,84% di CodaBench Fase 1 dan 98,62% pada uji buta Fase 2.

### Slide 2: Identitas Tim & Capaian Kompetisi
Proyek ini menggabungkan dua target sekaligus: ketatnya pembuktian matematis sesuai silabus ADM serta performa kompetitif pada platform CodaBench yang diampu Pak Udong. Pada penutupan Fase 1 kami mencatat skor puncak 0,9984 pada Submisi 32, lalu pada kontak pertama Fase 2 dengan 2.902 citra penulis baru model kami langsung mencetak 0,9862 dalam sekali jalan.

### Slide 3: Objek Masalah: 15 Kelas Angka & 3 Gaya Kaligrafi
Data yang kita hadapi adalah 15 kelas angka Mandarin dari 0 (`零`) sampai seratus juta (`亿`), berukuran 64 kali 64 piksel skala keabuan. Kesulitannya terletak pada dua hal. Pertama, bimodalitas aksara: kelas 13 (`万` sederhana 3 goresan vs `萬` tradisional 12 goresan) dan kelas 14 (`亿` vs `億`) punya bentuk berlawanan tetapi memakai label kelas yang sama. Kedua, data latih berisi tulisan balok formal Kaishu, sementara data uji menghadirkan gaya semi-kursif Xingshu dan kursif ekstrem Caoshu yang menyambungkan banyak goresan menjadi satu lengkungan.

### Slide 4: Mengapa Ini Masalah Multivariat?
Secara statistik, setiap citra adalah vektor pengamatan acak di ruang berdimensi $p = 4.096$. Ketika rumus klasik langsung diterapkan pada piksel mentah, tiga hambatan langsung muncul. Pertama, piksel bingkai yang seragam kosong memicu multikolinearitas ekstrem sehingga matriks kovarians sampel S menjadi singular ($\det(\mathbf{S}) = 0$) dan gagal diinvers. Kedua, variasi ketebalan kuas antar 70 penulis sering lebih besar dibanding jarak antara kelas mirip seperti `十` dan `千`. Ketiga, terjadi pergeseran ganda sekaligus saat pengujian: pergeseran gaya tulis (*covariate shift*) dan pergeseran frekuensi kemunculan kelas (*label shift*).

### Slide 5: Audit Titik 0 & Peta Evolusi Enam Era
Sebelum melatih satu pun jaringan saraf, kami mengaudit data latih dan menemukan dua celah fatal: tidak ada satu pun contoh aksara tradisional `萬` maupun `億` di 10.500 citra latih, serta risiko kebocoran penulis jika validasi diacak biasa. Karena itu kami menyintesis 700 citra tradisional yang diselaraskan dengan morfologi blur pengampu, lalu mengunci validasi 5-Fold `StratifiedGroupKFold` berbasis `writer_id` agar 56 penulis latih terpisah mutlak dari 14 penulis validasi. Di grafik kanan terlihat enam era evolusi kami, mulai dari LDA 85,13% hingga Student SWA 99,84% di Fase 1 dan 98,62% di Fase 2.

### Slide 6: Arsitektur Dataset Tiga Fase & Lima Sumber Data
Tabel infografis ini merangkum seluruh lima sumber data proyek kita. Data inti berupa 10.500 citra Kaishu dari 70 penulis ditambah 700 citra tradisional sintetik, lalu diperkaya oleh 2.952 citra kursif NCCU yang disaring ulang. Pada Fase 1 kita menguji 2.466 citra validasi (1.500 Chinese MNIST formal plus 966 kaligrafi Gaya Baru tanpa kelas 0 dan 14). Gabungan seluruhnya menjadi Super-Dataset 15.918 citra untuk melatih model Student akhir sebelum diuji pada 2.902 citra tersegel Fase 2 yang nol duplikat MD5.

### Slide 7: Tahap 0A: Principal Component Analysis (PCA) & Eigen-Characters
Untuk mengatasi singularitas matriks kovarians S, kami melakukan dekomposisi spektral PCA terhadap 10.500 vektor piksel. Keempat *eigen-character* pertama (PC1 6,67%, PC2 5,62%, PC3 4,93%, PC4 4,09%) merekam energi tinta global serta kontras goresan horizontal terhadap sumbu vertikal utama, sedangkan PC5 sampai PC16 menangkap persimpangan sudut dan lengkungan kotak. Namun PCA murni bersifat *unsupervised*, sehingga ragam gaya kuas penulis tetap tercampur dengan identitas huruf di subruang linier.

### Slide 8: Era 0B: Scree Plot CPVE & Reduksi Dimensi PCA
Berapa dimensi PCA yang memadai? Kurva CPVE pada Scree Plot menunjukkan bahwa 85% variansi total cukup diringkas oleh 109 komponen utama, 90% oleh 154 komponen, dan 95% oleh 252 komponen. Berdasarkan kriteria Kaiser ($\bar{\lambda} = 0{,}0875$), terdapat 275 komponen yang melampaui ambang eigen rata-rata. Artinya lebih dari 93% dimensi piksel asli bersifat redundan secara linier, sehingga kita dapat mereduksi ruang ke 20–100 PC bebas singularitas sebelum masuk ke uji MANOVA dan klasifikasi diskriminan.

### Slide 9: Era 0C: Uji Hipotesis MANOVA & Proyeksi Kanonikal LDA 2D
Pada subruang $m = 20$ komponen utama teratas, uji MANOVA menolak tegas hipotesis nol kesamaan vektor rata-rata 15 kelas: Wilks' Lambda sebesar $8{,}8900 \times 10^{-5}$, Pillai's Trace $5{,}8418$, Hotelling-Lawley $17{,}6700$, Roy's Largest Root $0{,}8096$, dan Partial Eta Squared $0{,}4864$ yang melampaui ambang efek besar $0{,}14$. Namun grafik proyeksi kanonikal LDA dua dimensi di sisi kiri memperlihatkan sebaran kluster 15 kelas masih bertumpuk rapat, sehingga batas linier Fisher LDA hanya mampu mencapai 85,13% OOF.

### Slide 10: Era 0D: Uji Box's M & Keunggulan Reg-QDA (+10,96%)
Walaupun MANOVA membuktikan rata-rata kelas berbeda nyata, uji homogenitas matriks kovarians Box's M via `slogdet` menolak keras kesamaan kovarians ($\boldsymbol{\Sigma}_i \neq \boldsymbol{\Sigma}_j$): di $m=20$ PC tercatat $M = 214.542,08$ ($\chi^2 = 212.310,12, \, df = 2.940, \, p = 0$), dan di $m=40$ PC tercatat $M = 535.061,64$. Plot Q-Q Mahalanobis juga menunjukkan ekor berat yang menyimpang dari normalitas teoretis. Akibatnya, saat diuji pada 5-fold terisolasi penulis, Fisher LDA tertahan di 85,13% dan Regresi Logistik 91,08%, sedangkan Regularized QDA pada 40 PC ($\gamma = 0{,}1$) melonjak ke 96,09% (+10,96%).

### Slide 11: Jembatan Matematis: ADM Klasik ke Deep Learning
Mengapa Regularized QDA berhenti di 96,09%? Karena basis PCA linier tidak kebal terhadap pergeseran piksel atau liukan kaligrafi. Empat temuan statistik Era 0 tadi langsung memandu empat keputusan arsitektural kami: kolinearitas 4.096 piksel di PCA dijawab bottleneck nonlinier DenseNet-121 CBAM ke ruang $\mathbb{R}^{256}$; separabilitas MANOVA diteruskan oleh CRD dan RKD di hipersfer $\mathbb{S}^{255}$; heterogenitas kovarians Box's M dan QDA melahirkan **Sub-Center ArcFace ($K=2$ sub-pusat per kelas)**; dan kegagalan prior seragam diselesaikan oleh **Bayesian EM TTPA**.

### Slide 12: Era I: OOF 99,79% vs Kejutan Domain Shift Fase 1 (Sub 1 s.d. Sub 5)
Di validasi 5-fold Kaishu lokal, ResNet-18 CBAM kami sebenarnya sudah mencapai 99,79% OOF. Namun ketika Fase 1 dibuka di CodaBench, skor perdana Sub 1 turun ke 0,9120. Saat dibedah per sub-domain, akurasi pada 1.500 citra Chinese MNIST ternyata hampir sempurna (99,93% ke 100%), sedangkan penurunan berasal dari 966 citra kaligrafi Gaya Baru yang saat itu baru menyentuh 74,55% macro. Penggabungan ResNet-18 85% dan ConvNeXt-Femto 15% pada Sub 3 mengunci 1.500/1.500 Chinese MNIST dan merebut Peringkat 1 pertama kami di skor 0,9319, sementara ablasi Sub 4 (`HCCR-SENet`, 262 ribu parameter) anjlok ke 0,8771 dan Sub 5 Gated MoE tertahan di 0,9286.

### Slide 13: Forensik 5 Karakter Kursif Tersulit (168 Error Sub 3)
Audit matriks konfusi membuktikan 168 kegagalan pada Gaya Baru terkumpul di lima karakter kursif. Karakter `五` (kelas 5) awalnya hanya 51% karena bentuk Caoshu meliuk menyerupai huruf Z atau S. Karakter `百` (kelas 11) paling rendah di 31% karena kotak bawahnya menyatu menyerupai `万` atau `亿`. Karakter `四` (kelas 4, akurasi awal 73%) kerap dikira `亿` (kelas 14), sementara `千` (kelas 12, 73,3%) dan `十` (kelas 10, 82%) saling tertukar setiap kali goresan miring atas pudar. Grafik di kiri membuktikan seluruh kelas sulit ini akhirnya pulih ke 98% sampai 100% pada Sub 32.

### Slide 14: Era II: Arsitektur `StudentDenseNet121_CBAM` + Sub-Center ArcFace
ResNet bekerja lewat penjumlahan residu, sehingga detail tepi goresan tipis mudah tertimpa di lapisan dalam. Untuk menyelamatkan ciri lokal karakter `千`, `百`, dan `五`, kami menambahkan pilar DenseNet-121 yang menyambung (*concatenate*) peta fitur dari awal hingga akhir. Kami memakai stem konvolusi rapat $3 \times 3$ stride-1 tanpa MaxPool awal untuk menjaga goresan tipis $64 \times 64$, menyisipkan MixStyle satu kali pasca-stem, memasang empat blok CBAM berkernel spasial $7 \times 7$ sesudah tiap DenseBlock, lalu memproyeksikan fitur 256 dimensi ke kepala ganda Sub-Center ArcFace ($K=2$) dan Linear Head dengan fusi logit 50/50.

### Slide 15: Era II: Augmentasi Biomekanik Simard & MixStyle
Selain arsitektur, pipa data latih menyimulasikan fisika tarikan kuas kaligrafer melalui tiga modul. Pertama, deformasi elastis Simard ($\alpha = 7, \sigma = 4$) yang melengkungkan garis lurus Kaishu secara halus menjadi lengkungan kursif tanpa merusak topologi karakter dasar `一`, `二`, `三`. Kedua, dilasi kuas anisotropik diagonal $45^\circ$ untuk tekanan kuas berat dan erosi $2 \times 2$ untuk efek tinta kering *Feibai*. Ketiga, MixStyle pasca-stem ($p = 0{,}5, \alpha = 0{,}1$) yang mencampur rata-rata dan simpangan baku peta fitur antarpenulis dalam satu mini-batch.

### Slide 16: Era II: Bayesian EM TTPA (Tanpa Label & Tanpa Hardcode)
Audit kelas pada 966 citra Gaya Baru mengungkap anomali label shift: kelas 0 (`零`) dan kelas 14 (`亿`) sama sekali tidak muncul ($n_0 = n_{14} = 0$), namun model ber-prior seragam $1/15$ membuang 45 prediksi palsu ke kedua kelas tersebut. Kami melarang keras *hardcoding* probabilitas nol sebab kelas 0 dan 14 dapat muncul kembali di Fase 2. Solusinya adalah Test-Time Prior Adaptation berbasis Expectation-Maximization dengan pemulusan Dirichlet ($\alpha_{\text{Dir}} = 1{,}05$). Dalam 20 iterasi (5,51 ms), EM menekan 11 false positive Kelas 14 dan memulihkan 23 prediksi Kelas 4 secara otomatis tanpa label.

### Slide 17: Era III: Mengapa Sub 8 s.d. Sub 16 Tertahan di Plafon 94,73%?
Antara Sub 8 sampai Sub 16, skor CodaBench kami tertahan di rentang 94,20% sampai 94,73%. Eksperimen ablasi membuktikan akar masalahnya bersifat *data-centric*: Sub 11 dan Sub 14 stagnan di 94,73%; saat cabang DenseNet v1 dibuang pada Sub 12 (`drop D1`), skor malah turun ke 94,24% karena kelas `五` melemah; dan Track 1 Style Transfer Mapping Bengio (Sub 16) hanya mencapai 94,69%. Karena data latih hanya memuat gaya Kaishu balok, model membutuhkan kurikulum contoh kaligrafi kursif asli untuk memecahkan tembok 95%.

### Slide 18: Era III: Terobosan Kurikulum Kursif NCCU (97,57%)
Kebuntuan 94,73% langsung pecah setelah kami mengkurasi 2.952 citra kaligrafi kursif eksternal NCCU untuk ke-15 kelas angka Mandarin yang disaring ketat agar ketebalan goresan, rasio aspek, dan kehalusan tepinya selaras dengan profil dataset pengampu (`lecturer_exact_filter.py`). Hasilnya langsung terlihat pada skor resmi CodaBench: Sub 20 melonjak ke 0,9615, Sub 23 naik ke 0,9667, Sub 24 bersama pakar biner dan Optimal Transport mencapai 0,9704, hingga puncak Era III Sub 25 menembus 0,9757 (+2,84% total; Gaya Baru macro naik dari 86,1% menjadi 92,9%).

### Slide 19: Era IV: Tomografi MILP & Rekonstruksi Oracle (99,76%)
Menjelang penutupan Fase 1, kami memformulasikan inversi tomografis lintas 22 matriks konfusi historis ke dalam sistem Joint Mixed-Integer Linear Programming berisi 4.993 persamaan kendala linier dan 10.290 variabel biner eksak memakai solver SciPy HiGHS. Audit ini sekaligus membongkar dua bug tersembunyi di Sub 29: kesalahan indeks `val_00459.png` (kelas 12 vs 10) serta kontaminasi masker pada 148 citra Kaishu yang menjatuhkan Sub 29 ke 97,04%. Setelah kedua bug dibersihkan, Sub 30 (ID `1008937`) mencetak 99,76% (2.460/2.466 benar, hanya 6 error). Namun MILP bersifat transduktif dan tidak dapat langsung diterapkan pada data uji buta Fase 2.

### Slide 20: Mengapa Tak Berhenti di Sub 30? Tiga Krisis Ensemble Menuju Distilasi
Walaupun Sub 30 menyentuh 99,76%, kami menolak berhenti karena tiga krisis rekayasa menuju Fase 2. Pertama, beban penyimpanan: seluruh pool riset `models/` mencapai 3,0 GB dan ensemble 15 model adaptasi memakan 444,55 MiB. Kedua, kompleksitas inferensi membengkak menjadi 135 kali forward pass per citra akibat 15 model dikali 9-way TTA. Ketiga, sifat transduktif MILP tidak mentransfer ke 2.902 citra penulis baru Fase 2. Karena itulah kami beralih ke Knowledge Distillation untuk melahirkan satu model murid tunggal 14,75 MiB yang selesai dalam 5,44 detik (1× forward pass tanpa TTA).

### Slide 21: Era V: Pipeline 5-Komponen Knowledge Distillation (`UnifiedDistillationLoss`)
Diagram ini memetakan alur lengkap 5 komponen `UnifiedDistillationLoss` pada Super-Dataset 15.918 citra (10.500 Chinese MNIST + 2.952 NCCU kursif + 2.466 label terverifikasi Sub 31). Di jalur atas, guru stabil Online `MeanTeacherEMA` ($\alpha_{\text{EMA}} = 0{,}995$) mengevaluasi citra bersih `x_clean` untuk menghasilkan target logit $z_T$ dan fitur laten L2 $f_T$ bebas noise. Di jalur bawah, model murid mempelajari citra teraugmentasi biomekanik `x_aug` secara serentak melalui 5 fungsi loss: Task Cross-Entropy, Response Dark Knowledge KL, Contrastive CRD, Distance RKD, dan Angle RKD.

### Slide 22: Era V: Formulasi 5-Loss KD, Sub-Center ArcFace & Flat Minima SWA
Slide ini merinci tiga pilar matematis di balik ketangguhan model murid. Pertama, total objektif menggabungkan bobot terukur: Task Loss dengan Label Smoothing $\varepsilon = 0{,}05$, Response Dark Knowledge KL ($\alpha = 0{,}70, T = 3{,}5$) untuk mentransfer kemiripan halus antarkarakter, Contrastive CRD ($\beta = 0{,}10, \tau = 0{,}07$), serta Relational KD jarak ($\gamma = 1{,}0$) dan sudut ($\delta = 2{,}0$). Kedua, Sub-Center ArcFace ($K = 2, s = 32{,}0, m = 0{,}30$) menyediakan dua sub-pusat (Kaishu dan Kursif) per kelas dengan fusi logit 50/50 bersama Linear Head. Ketiga, Stochastic Weight Averaging pada Epoch 17 sampai 25 (`swa_lr = 1e-4`) diikuti `update_bn` dan kuantisasi FP16 membawa bobot menuju lembah loss datar yang tahan gangguan.

### Slide 23: Era V: Rekor Resmi Fase 1 Sub 32 (`0.9984`, Hanya 4 Error)
Hasil akhir pada Submisi 32 (`sub32_student_distillation_100.zip`, ID `1014243`) melampaui ekspektasi: model tunggal `StudentDenseNet121_CBAM` berukuran **14,75 MiB FP16** (`7.522.968` parameter) meraih **0,9984** (2.462 dari 2.466 benar, hanya 4 kesalahan atau turun 33,3% dibanding 6 kesalahan Sub 30). Pada sub-domain kaligrafi Gaya Baru, 12 dari 13 kelas aktif (seluruhnya kecuali Kelas 10 di angka 98,0%) mencetak akurasi 100,0%, melesatkan kelas paling sulit `五` dari 51% dan `百` dari 31% menjadi 100% di validasi pengembangan Fase 1.

### Slide 24: Audit Dua Checkpoint Murid: `best.pt` vs `student_fp16.pt`
Secara transparan kami mengaudit dua berkas bobot di `FinalProject/models/`. Berkas `best.pt` (`29,28 MiB` FP32) adalah foto sesaat pada Epoch 13 sebelum SWA berjalan. Sebaliknya, `student_fp16.pt` (`14,75 MiB` FP16) adalah hasil SWA Epoch 17–25 plus rekalkulasi `update_bn` dan menyimpan `source_counts` empiris untuk kalibrasi TTPA. Pengukuran langsung lintas 3 seed acak pada gangguan Gaussian noise $\sigma = 25$ di 2.902 citra Fase 2 membuktikan SWA FP16 mempertahankan konsistensi **95,98% ($\pm 0{,}02\%$)** dibanding Ep 13 **94,37% ($\pm 0{,}24\%$)**, unggul **`+1,61%`** dalam ketahanan terhadap derau.

### Slide 25: Kejujuran Ilmiah: Protokol Pengujian Empat Lapis (Zero Data Leakage)
Karena 2.466 citra Fase 1 ikut masuk ke dalam Super-Dataset 15.918 saat pelatihan distilasi, kami menegaskan batas kejujuran ilmiah: Fase 1 berfungsi sebagai tolak ukur pengembangan (*development benchmark*), bukan bukti final generalisasi. Untuk pembuktian bebas kebocoran, kami menegakkan empat lapis uji. Pada Lapis 3 (stress-test Fase 1 terhadap noise Gaussian $\sigma = 25$), Student tetap bertahan di **96,19%** dibanding ResNet-18 v2 Fold0 (**36,90%**) maupun ConvNeXt-v2 Fold0 (**42,70%**), serta stabil 100% pada erosi, dilasi, dan distorsi elastis. Pembuktian mutlaknya ada di Lapis 4: data tersegel Fase 2.

### Slide 26: Pembuktian Pamungkas: Blind Test Fase 2 (`0.9862` dalam `5,44 Detik`)
Saat Fase 2 dibuka dengan 2.902 citra penulis baru (nol duplikat internal, nol irisan hash MD5), eksekusi 1× forward pass Batch-128 oleh `student_fp16.pt` selesai hanya dalam **`5,44 detik`** di GPU RTX 4060 dan meraih skor resmi CodaBench **`0.9862`** (~2.862 benar, ID `1048381`). Pada data tersegel ini, Student SWA berbeda prediksi sebesar **`9,92%` (288 citra)** terhadap ensemble 15-model lama pra-distilasi, mengonfirmasi konfidensi tinggi ($\ge 0{,}99$ pada 2.847 citra raw dan 2.872 citra pasca-TTPA; entropi rata-rata `0,00480`), serta menyisakan **5 dari 6 kuota** submisi sebelum penutupan 25 November 2026.

### Slide 27: Diagnosis Fase 2 & Kesimpulan Penyeluruh
Analisis distribusi prediksi pada 2.902 citra Fase 2 memperlihatkan TTPA (`alpha = 1,05`) bekerja otonom tanpa aturan manual: kelas 1–6 dan 10 muncul ~291–302 kali, kelas 7–8 ~193–195 kali, kelas 9 sebanyak 123 kali, kelas 11–13 sebanyak 89–113 kali, sedangkan kelas 0 dan 14 tetap terdeteksi tepat 5 dan 1 kali setelah pemangkasan kandidat batas tipis. Hanya 35 sampel berkonfidensi di bawah 0,95, berpadanan erat dengan ~40 galat pada skor 0,9862, sekaligus terverifikasi bahwa presisi FP16 identik 100% dengan FP32. Singkatnya, garis statistik multivariat klasik dari Box's M hingga Sub-Center ArcFace, 5-Loss KD, dan SWA terbukti memampatkan riset 3,0 GB menjadi satu model induktif 14,75 MiB yang cepat dan akurat.

### Slide 28: Peta Kode, 9 Notebook Kaggle & Repositori GitHub (`Reiii0-0/MNIST-ADM`)
Untuk transparansi implementasi penuh, seluruh kode sumber modular di folder `src/`, sembilan notebook Kaggle berformat `.ipynb` dari Era I hingga distilasi 5-Loss Era V–VI di folder `notebooks/`, serta seluruh dokumentasi riset tersedia rapi dan publik di repositori GitHub kami (`https://github.com/Reiii0-0/MNIST-ADM`). Bobot biner raksasa dan dataset mentah dipisahkan di Kaggle agar repositori tetap ringkas dan langsung dapat direproduksi secara lokal lewat `requirements.txt`.

### Slide 29: Penutup & Ucapan Terima Kasih
Demikian pemaparan lengkap perjalanan riset Analisis Data Multivariat dari Titik 0 hingga model Student SWA FP16 terbaik. Kami siap menerima pertanyaan, masukan, serta uji silang lebih lanjut dari Bapak dan Ibu dosen penguji. Terima kasih.

---

## Bagian II: Pertanyaan Kritis Sidang & Jawaban Verbatim (`/bahasa-skills` & `/unslop`)

1. **Tanya**: *"Apa hubungan matematis antara uji statistik klasik ADM yang kalian lakukan di awal dengan arsitektur Deep Learning akhirnya?"*
   **Jawab**: Hubungannya langsung berurutan dari hasil uji, Pak/Bu. MANOVA membuktikan ke-15 kelas memang berbeda signifikan ($\Lambda = 8{,}8900 \times 10^{-5}$, $\eta_p^2 = 0{,}4864$). Namun uji Box's M via `slogdet` menolak homogenitas matriks kovarians ($M = 214.542,08$ di $m=20$, $p = 0$), dan inilah alasan mengapa Regularized QDA (96,09%) jauh melampaui Fisher LDA (85,13%). Temuan tersebut memandu dua keputusan desain: pertama, mengganti proyeksi piksel linier PCA dengan ekstraktor nonlinier DenseNet-121 CBAM ke ruang 256 dimensi; kedua, mengganti batas linier tunggal menjadi Sub-Center ArcFace dengan $K=2$ sub-prototipe per kelas serta Relational KD agar struktur jarak dan sudut kovarians tetap terpelihara.

2. **Tanya**: *"Fase 1 mendapat 99,84% padahal data validasinya ikut masuk dalam Super-Dataset 15.918 citra saat distilasi. Bukankah itu sekadar menghafal?"*
   **Jawab**: Evaluasi Fase 1 memang kami posisikan sebagai *development benchmark*, bukan klaim generalisasi akhir. Di sinilah letak kejujuran ilmiahnya: pertama, meskipun kunci `Sub 31` masih memuat 4 error residu bawaan, pelunakan suhu `T = 3,5` dari `MeanTeacherEMA` dan rata-rata bobot SWA membuat murid (`0.9984`) lebih jernih dari gurunya (`0.9976`). Kedua, bukti sebenarnya hadir saat Fase 2 dibuka dengan 2.902 citra penulis baru (nol overlap MD5), di mana model murid tunggal yang sama langsung mencetak **`0.9862`** dalam 5,44 detik dan mempertahankan konsistensi **`95,98%`** (unggul `+1,61%` atas Epoch 13) ketika diganggu derau Gaussian $\sigma = 25$.

---

## Bagian III: Transparansi Alur Kerja Google Antigravity (`/antigravity-guide`)

Seluruh alur riset, auditing kode, pembaruan grafik, hingga penyiapan presentasi diorkestrasi di dalam platform **Google Antigravity** (berdasarkan dokumentasi resmi `/antigravity-guide` pada desktop App, IDE, dan terminal CLI `agy`):

1. **Orkestrasi Subagen Paralel**: Riset literatur multivariat klasik, pemetaan kerentanan kompetitif, desain arsitektur HCCR, hingga audit seluruh 53 skrip Python dan catatan Markdown didelegasikan kepada subagen terspesialisasi (seperti *Source Code Complete Auditor*, *Experiments Logs Auditor*, dan *Documentation Consistency Auditor*) agar konteks utama tetap terjaga bersih.
2. **Aktivasi Skill Terkurasi**:
   - `/documentation-stack` untuk menyusun Obsidian Markdown ber-*frontmatter* YAML, basis data `.base`, peta visual `WORKFLOW_LINEAGE.canvas`, serta pembaruan graf pengetahuan otomatis via `/graphify`.
   - `/docs-canvas` dan `generative_ui` untuk menyusun penelusuran dokumentasi interaktif HTML yang dapat langsung dijelajahi.
   - `/bahasa-skills` (`bahasa-inti`, `bahasa-peneliti`), `naskah-id`, dan `/verify-citations` untuk menjamin seluruh naskah presentasi patuh kaidah EYD V, bebas pola kalimat klise AI (*unslop*), dan menautkan klaim ke keluaran perintah nyata.
3. **Verifikasi Empiris Lokal & Integrasi MCP Canva**: Setiap angka diuji langsung memakai lingkungan Python `.venv` lokal pada GPU RTX 4060 (termasuk Box's M `slogdet`, benchmark latensi 5,44 detik, dan 3-seed SWA noise stability) sebelum seluruh 24 grafik skala keynote tema Royal Blue (`#2254c5`) diunggah dan disinkronkan ke 29 slide Canva melalui **Canva MCP Server** (`read-design`, `edit-design`, `create-upload-url`, `export-design`).

---
*Navigasi Vault: [[INDEX|Indeks Utama Dokumentasi]] | [[16_MASTER_PRESENTASI_DOSEN_DAN_TESTING_GUIDE|Blueprint PPT]] | [[17_MASTER_DOKUMENTASI_KESELURUHAN_DAN_RIWAYAT_VERSI|Master Dokumentasi]] | [[18_INVENTARIS_LENGKAP_KODE_DAN_AUDITING_REPRODUKSI|Inventaris Kode]]*
