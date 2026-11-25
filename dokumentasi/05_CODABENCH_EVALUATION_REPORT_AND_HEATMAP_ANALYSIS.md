---
title: "Laporan Evaluasi Komprehensif CodaBench & Audit 5 Submisi (Fase 1: Validasi)"
source_doc: "05_CODABENCH_EVALUATION_REPORT_AND_HEATMAP_ANALYSIS.md"
date: 2026-10-18
updated: 2026-11-24
tags:
  - adm-project
  - documentation
  - phase-1
---

# Laporan Evaluasi Komprehensif CodaBench & Audit 5 Submisi (Fase 1: Validasi)
## Analisis Forensik Submisi 1 s.d. 5, Dekonstruksi Heatmap, & Rekomendasi SOTA

**Tanggal Evaluasi:** 18 Oktober 2026 (02:30 WIB)  
**Status Leaderboard CodaBench:** **Peringkat 1 (Juara 1)** 🥇  
**Skor Tertinggi Terkunci:** **`0.9319` (93.19%)** — Tim `arete` (`5025241135`)  
**Margin Keunggulan:** **+2.92% di atas Juara 2 (`0.9027`)** dan **+4.67% di atas Juara 3 (`0.8852`)**  
**Kuota Submisi Terpakai:** **5 dari 5 (Hari Ini: Penuh)** | **5 dari 100 (Total Keseluruhan Kompetisi)**  
**Sumber Artefak Evaluasi:** CodaBench Scoring Packages (`scoring_result_1.zip` s.d. `scoring_result_5.zip`)

---

## 1. Papan Peringkat Resmi & Matriks Evaluasi 5 Submisi

Berdasarkan log kontainer evaluasi resmi CodaBench (`codalab/codalab-legacy:py312`), seluruh 5 submisi kelompok kita berhasil dievaluasi tanpa error:

| No | File Arsip | ID CodaBench | Waktu Submit | Akurasi Total | Akurasi Angka Tiongkok (1.500 citra) | Akurasi Gaya Baru (966 citra) | Total Benar / 2.466 | Status / Peringkat Leaderboard |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | `submission_phase1_adabn_tta.zip` | 995975 | 17 Okt 22:32 | 0.9120 (91.20%) | 0.9993 (1.499 / 1.500) | 0.7455 (720 / 966) | 2.249 / 2.466 | Benchmark Awal |
| **2** | `submission_phase1_anchor.zip` | 995976 | 17 Okt 22:33 | 0.9278 (92.78%) | 0.9993 (1.499 / 1.500) | 0.7847 (758 / 966) | 2.288 / 2.466 | Peringkat 1 🥇 (Mengunci Puncak) |
| **3** | `submission_phase1_blend_resnet_conv.zip` | 996005 | 17 Okt 22:52 | **0.9319 (93.19%)** | **1.0000 (1.500 / 1.500)** | **0.7994 (772 / 966)** | **2.298 / 2.466** | **Peringkat 1 🥇 (JUARA 1 TERTINGGI)** |
| **4** | `submission_phase1_hccr_senet.zip` | 996076 | 17 Okt 23:37 | 0.8771 (87.71%) | 0.9967 (1.495 / 1.500) | 0.6661 (643 / 966) | 2.163 / 2.466 | Eksperimen Model Ringan |
| **5** | `submission_phase1_gated_moe.zip` | 996080 | 17 Okt 23:40 | 0.9286 (92.86%) | 0.9993 (1.499 / 1.500) | 0.7876 (761 / 966) | 2.290 / 2.466 | Gated Mixture of Experts |

### Posisi Papan Peringkat Terkini (CodaBench Public Leaderboard):
1. 🥇 **Tim Kita (`arete` / `5025241135`) : `0.9319` (93.19%)** — **TERKUNCI AMAN DI PUNCAK**
2. 🥈 Peringkat 2 (`5025241058_5025241104`) : `0.9027` (90.27%) — *Tertinggal 72 sampel (+2.92%)*
3. 🥉 Peringkat 3 (`5025241130_5025241131`) : `0.8852` (88.52%) — *Tertinggal 115 sampel (+4.67%)*

> [!NOTE]
> Leaderboard CodaBench beroperasi dengan prinsip **Strict Best-Score Retained**. Nilai tertinggi yang pernah dicapai tim (`0.9319` dari Submisi 3) terkunci permanen di puncak papan peringkat dan tidak terpengaruh oleh submisi berikutnya.

---

## 2. Bedah Komparatif Performa per Kelas (966 Citra Gaya Tulisan Baru)

Berikut adalah perbandingan akurasi per kelas pada sub-dataset paling menantang (966 citra gaya kaligrafi eksternal / unseen writers):

| Label | Karakter | Jumlah ($n$) | Sub 1 (AdaBN) | Sub 2 (Anchor) | Sub 3 (Blend) 🏆 | Sub 4 (HCCR-SENet) | Sub 5 (Gated MoE) | Analisis Dinamika & Karakteristik |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **0** | **零** | 0 | - | - | - | - | - | **Absen di gaya baru** (hanya ada di 1.500 citra Chinese MNIST). |
| **1** | **一** | 100 | 96.0% | 97.0% | **97.0%** | 96.0% | **97.0%** | Sangat konsisten di seluruh model. |
| **2** | **二** | 100 | 96.0% | 98.0% | **98.0%** | 87.0% | **98.0%** | HCCR-SENet drop (-11 sampel); model utama stabil. |
| **3** | **三** | 100 | 92.0% | **98.0%** | 95.0% | 69.0% | 97.0% | HCCR-SENet kolaps (-29 sampel tertukar ke `6` dan `7`). |
| **4** | **四** | 100 | 59.0% | 66.0% | **73.0% 🟢** | 41.0% | 69.0% | **Keunggulan Telak Blend:** Menyelamatkan 7 sampel dari ilusi radikal `亿`. |
| **5** | **五** | 100 | 45.0% | **57.0%** | 51.0% | 41.0% | **57.0%** | Titik tersulit; stroke kaligrafi kursif sangat dinamis. |
| **6** | **六** | 100 | 79.0% | 75.0% | **81.0% 🟢** | 56.0% | 76.0% | Blend unggul; menangkap siluet segitiga titik atas. |
| **7** | **七** | 64 | 90.6% | **96.9%** | **96.9%** | 92.2% | **96.9%** | ResNet & ConvNeXt konsisten tinggi. |
| **8** | **八** | 64 | **98.4%** | **98.4%** | **98.4%** | 89.1% | **98.4%** | 63 dari 64 sampel benar pada model utama. |
| **9** | **九** | 42 | 92.9% | **97.6%** | **97.6%** | 78.6% | **97.6%** | 41 dari 42 sampel benar pada model utama. |
| **10** | **十** | 100 | 76.0% | 82.0% | 82.0% | **89.0%** | 80.0% | HCCR-SENet menangkap cross stroke sederhana lebih agresif. |
| **11** | **百** | 29 | 24.1% | 24.1% | **31.0% 🟢** | 58.6%* | 24.1% | *Catatan: HCCR-SENet memprediksi `百` over-agresif (97 kali). |
| **12** | **千** | 30 | 63.3% | **73.3%** | **73.3%** | 33.3% | **73.3%** | HCCR-SENet kolaps parah (hanya 10/30 benar). |
| **13** | **万** | 37 | 56.8% | 56.8% | **64.9% 🟢** | 35.1% | 59.5% | Blend unggul (+3 sampel diselamatkan dari false `零`). |
| **14** | **亿** | 0 | - | - | - | - | - | **Absen di gaya baru** (hanya ada di 1.500 citra Chinese MNIST). |

---

## 3. Analisis Forensik Mendalam: Mengapa Submisi 4 (HCCR-SENet) Kurang Optimal?

Submisi 4 menggunakan arsitektur ringkas **HCCR-SENet** (~262.951 parameter) dengan Inverted Residuals dan Squeeze-and-Excitation (SE Block). Saat divalidasi lokal, model ini mencatat akurasi spektakuler **99.85%** (2.097/2.100). Namun, di CodaBench akurasinya anjlok menjadi **87.71%** (hanya 66.61% pada gaya tulisan baru).

Berikut adalah diagnosis akar penyebab ilmiahnya:

### A. Defisit Kapasitas Representasi (*Severe Under-Capacity*)
* **Rasio Parameter:** ResNet18 memiliki ~11,2 juta parameter, ConvNeXt-Femto memiliki ~5,2 juta parameter, sedangkan HCCR-SENet hanya memiliki **0,26 juta parameter (43x lebih kecil dari ResNet18)**.
* **Fenomena:** Untuk dataset Kaishu bersih (Chinese MNIST reguler), 262k parameter sudah cukup untuk "menghafal" pola ortogonal. Namun, ketika berhadapan dengan manifold kaligrafi Xingshu (semi-kursif) dan Caoshu (kursif penuh) yang memiliki rotasi sudut ekstrem, ketebalan goresan heterogen, dan sambungan ligatur tak teratur, ruang hipotesis model 262k parameter **tidak memiliki kapasitas representasi (*representational capacity*) yang cukup** untuk memisahkan fitur invarian gaya dari fitur semantik kelas.

### B. Single Split vs. 5-Fold Stratified Cross Validation
* **ResNet18 & ConvNeXt:** Dilatih dengan **5-Fold Cross Validation** di mana 5 model independen melatih 100% data penulis dan probabilitasnya di-ensemble. Ini memberikan efek *variance reduction* yang sangat kuat.
* **HCCR-SENet:** Hanya dilatih pada **1 split acak** (hanya 56 penulis latih, 14 penulis diabaikan dari training). Ketiadaan ensemble menyebabkan varians prediksi tinggi terhadap gaya penulis baru yang tidak pernah dilihat sebelumnya.

### C. Ketiadaan MixStyle Domain Generalization
* Model champion kita mengintegrasikan **MixStyle (Zhou et al., ICLR 2021)** pada lapisan awal, yang menginterpolasi statistik fitur ($\mu, \sigma$) antarsampel pada batch yang sama. Hal ini melatih model untuk **kebal terhadap ketebalan kuas dan gaya goresan**. HCCR-SENet tidak memiliki modul MixStyle, sehingga ia terlalu sensitif terhadap intensitas goresan.

### D. Patologi False Positive pada Kelas 7 (`七`) dan Kelas 11 (`百`)
Berdasarkan audit confusion matrix:
* HCCR-SENet memprediksi Kelas 7 (`七`) sebanyak **138 kali** (padahal sampel aslinya hanya 64). Sebanyak 23 sampel `五`, 16 sampel `六`, 10 sampel `三`, dan 8 sampel `千` secara keliru didiagnosis sebagai `七`.
* HCCR-SENet memprediksi Kelas 11 (`百`) sebanyak **97 kali** (padahal sampel aslinya hanya 29). Sebanyak 32 sampel `四`, 21 sampel `六`, dan 9 sampel `五` secara keliru diserap menjadi `百`.
* **Akar Masalah Morfologi:** Kernel konvolusi depthwise $3 \times 3$ yang kecil gagal menangkap konteks global siluet karakter. Sambungan goresan kursif disalahartikan sebagai goresan horizontal-vertikal sederhana milik `七` dan kotak milik `百`.

---

## 4. Analisis Teoretis: Mengapa Soft Blend (0.9319) Melampaui Gated MoE (0.9286)?

Pada Submisi 5, kita menguji **Gated Mixture of Experts (MoE)** yang menerapkan aturan tegas (*hard thresholding*): ConvNeXt hanya boleh mengambil alih keputusan jika $Margin_{\text{ResNet}} \le 0.10$ DAN $P_{\text{ConvNeXt}} \ge 0.60$.

Hasil evaluasi menunjukkan:
* Gated MoE berhasil mengungguli Anchor ResNet (**92.86% vs 92.78%**, surplus +2 sampel benar).
* Namun, Gated MoE masih berada di bawah Soft Blend (**92.86% vs 93.19%**, selisih 8 sampel benar).

### Rasional Teoretis (Continuous Bayesian Smoothing vs. Hard Discretization):
1. **Reduksi Varians Kontinu:**  
   Pada domain shift dengan deformasi goresan non-linear, probabilitas output model mencerminkan ketidakpastian epistemik (*epistemic uncertainty*). Penggabungan kontinu:
   $$P_{\text{blend}}(y = c \mid \mathbf{x}) = 0.85 P_{\text{ResNet}}(y = c \mid \mathbf{x}) + 0.15 P_{\text{ConvNeXt}}(y = c \mid \mathbf{x})$$
   memberikan efek *Bayesian model averaging* yang merata ke seluruh 15 dimensi kelas.
2. **Kekurangan Ambang Keras (Hard Gating Boundary):**  
   Ketika ConvNeXt memiliki keyakinan $0.55$ pada kelas yang benar (di bawah batas ambang $0.60$), Gated MoE langsung mengabaikan opini ConvNeXt. Padahal, probabilitas $0.55$ tersebut—bila digabungkan secara proporsional dengan ResNet—sering kali cukup untuk mengoreksi bias tunggal ResNet tanpa menimbulkan risiko *false override*.
3. **Bukti Empiris:**  
   Dari 34 sampel yang diubah oleh Soft Blend tetapi ditolak oleh Gated MoE, Soft Blend mencetak **+8 sampel benar bersih (*net positive accuracy*)**. Ini membuktikan bahwa pembobotan probabilitas lembut (*soft ensembling*) secara konsisten lebih optimal daripada pembagian domain kaku (*hard gating*) pada data dengan derau gaya kaligrafi.

---

## 5. Rangkuman Tindakan Pembersihan & Integritas Repositori

Untuk memastikan repositori tetap ramping, profesional, dan siap untuk pengumpulan akademik serta evaluasi Fase 2:

1. **Eliminasi File Redundan:**
   * Menghapus file sementara di root: `predictions.csv` dan `predictions.zip` (semua arsip resmi telah tersimpan rapi dan terverifikasi di folder `submissions/`).
2. **Optimalisasi `.gitignore`:**
   * Mengabaikan file hasil kemasan prediksi sementara di root (`predictions.csv`, `predictions.zip`, `predictions_*.zip`).
   * Mengabaikan folder artefak model lokal tambahan (`models/kaggle_outputs_hccr_senet/`, `kaggle_pulled/`).
   * Menjaga file kode inti, generator pengiriman (`src/generate_champion_submission.py`), dan dokumentasi riset tetap terlacak rapi.
3. **Status Integritas Git:**
   * Repositori berada dalam kondisi bersih (*clean working tree*), tersinkronisasi dengan GitHub (`origin/main`), dan siap diaudit.

---

## 6. Kesimpulan & Rencana Kerja Menuju Fase 2 (22 Oktober 2026)

1. **Posisi Juara 1 Aman Terkendali:**  
   Dengan skor **93.19%**, kita memimpin dengan selisih yang sangat nyaman (+2.92% dari peringkat 2). Kuota hari ini (5/5) telah dimaksimalkan dengan sempurna untuk memetakan seluruh batas performa model.
2. **Chinese MNIST Selesai Sempurna (100.00%):**  
   Arsitektur ensemble kita telah menuntaskan 1.500 citra angka Tiongkok tanpa cacat (1.500/1.500 benar).
3. **Fokus Riset untuk Fase 2 (2.902 Citra Uji):**  
   * Pada Fase 2 (22 Oktober – 26 November 2026), karakter tradisional (`萬`, `億`) dan kelas 0 serta 14 berpotensi muncul kembali dalam komposisi berbeda.
   * Model ensemble champion kita (`ResNet18-CBAM + ConvNeXt-Femto + MixStyle + Sub-Center ArcFace`) adalah satu-satunya model yang terbukti **tahan banting di seluruh skenario**: sempurna di karakter standar, unggul di karakter tradisional sintetis, dan meraih akurasi tertinggi di kaligrafi baru.

---
*Navigasi Vault: [[INDEX|Indeks Utama Dokumentasi]] | [[17_MASTER_DOKUMENTASI_KESELURUHAN_DAN_RIWAYAT_VERSI|Master Dokumentasi Keseluruhan]] | [[WORKFLOW_LINEAGE.canvas|Peta Alur Riset]]*
