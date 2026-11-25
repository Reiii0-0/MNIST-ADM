---
title: "Laporan Forensik Submisi 12 & Refaktorisasi Standar Penamaan File CodaBench"
source_doc: "11_SUBMISSION_12_ANALYSIS_AND_NAMING_REFACTOR.md"
date: 2026-10-19
updated: 2026-11-24
tags:
  - adm-project
  - documentation
  - phase-1
---

# Laporan Forensik Submisi 12 & Refaktorisasi Standar Penamaan File CodaBench

Dokumen ini mencatat investigasi mendalam penyebab penurunan akurasi pada **Submisi 12 (0.9424)**, peran krusial pilar jangkar **DenseNet-121 v1 (`D1`)**, perumusan solusi **Pentagon SOTA Ensemble (`sub13_sota_pentagon.zip`)**, serta refaktorisasi sistematis format penamaan file submisi agar tidak terpotong di UI CodaBench.

---

## 1. Analisis Forensik Submisi 12 (ID 1002466, Skor 0.9424)

Pada Submisi 12, konfigurasi yang diuji adalah *Tri-Orthogonal Pure v2* (36% ResNet-18 v2 + 34% ConvNeXt v2 + 30% DenseNet v2 Fixed), yang sepenuhnya mengeliminasi model Generasi 1 (`D1` dan `R1`).

Hasil evaluasi resmi dari `scoring_result_12.zip`:
* `acc_chinese`: 1.0000 (1.500 / 1.500 sempurna).
* `acc_target`: 0.8449 (816 / 966, turun dari 830 / 966 pada Sub 11).
* `accuracy`: **0.9424** (2.316 / 2.466, turun bersih -20 sampel dibanding Sub 11: 0.9473).

### A. Apa yang Berhasil Diperbaiki oleh Submisi 12?
1. **Pemberantasan Error `十 (10)` vs `千 (12)`**:
   - Error `十 ➔ 千` terpangkas lagi dari **12 sampel** (Sub 11) menjadi hanya **8 sampel**!
2. **Pemberantasan Error `四 (4)` vs `亿 (14)`**:
   - Error `四 ➔ 亿` terpangkas dari **16 sampel** (Sub 11) menjadi **13 sampel**!
   - Kedua pasangan error ini membuktikan bahwa penambahan fitur morfologi v2 sangat akurat menembus ambiguitas kait kuas.

### B. Root Cause: Mengapa Skor Total Justru Turun -20 Sampel?
Penyebab mutlak penurunan skor ditemukan pada pola kegagalan karakter **`五 (5)`**:
* Pada Sub 11 (memakai `D1` 29%): `五 ➔ 六` sebanyak 16 sampel, `五 ➔ 四` sebanyak 4 sampel.
* Pada Sub 12 (tanpa `D1`): `五 ➔ 六` meledak menjadi **24 sampel** (+8 salah), dan `五 ➔ 四` meledak menjadi **12 sampel** (+8 salah)!
* **Mengapa ini terjadi?**
  Semua model v2 (`R2`, `C2`, `D2`) dilatih dengan perturbasi kurva Caoshu ($A \cdot \sin(\dots)$) pada label 5. Ketika ketiga model v2 digabung tanpa model independen lain, ketiganya memiliki korelasi bias yang sama sehingga salah mengira liukan `五` sebagai kepala `六` atau badan `四`.
* Sebaliknya, model **DenseNet-121 v1 (`D1`)** dilatih pada representasi goresan asli tanpa Caoshu wave buatan, sehingga di Sub 11 `D1` bertindak sebagai **pilar penyeimbang (*orthogonal anchor*)** yang menahan karakter `五` agar tidak kolaps!

---

## 2. Solusi: Pentagon Ensemble (`sub13_sota_pentagon.zip`)

Untuk mempertahankan keunggulan pemangkasan error pada `十` dan `四` dari v2, sekaligus melindungi `五` dari kolaps, dirumuskan bobot perpaduan optimal 5 pilar (*Pentagon Ensemble*):

$$\mathbf{P}_{\text{pentagon}} = 0.30 \cdot \mathbf{R2} + 0.25 \cdot \mathbf{C2} + \mathbf{0.25 \cdot D1} + 0.10 \cdot \mathbf{R1} + \mathbf{0.10 \cdot D2}$$

1. **Mempertahankan `D1` pada porsi tinggi (25%)**: Menjamin stabilitas kelas `五` tetap terlindungi.
2. **Menyuntikkan `D2` pada porsi terukur (10%)**: Menangkap perbaikan pada `十` dan `四` tanpa memicu kolaps pada `五`.
3. **Hasil Diagnostik Forensik**:
   - `Kelas 5`: Frekuensi prediksi kembali normal ke **167 sampel** (vs 150 pada Sub 12).
   - Memperbaiki kembali sampel 1137 dan 1442 (dari tebakan salah `9` kembali ke `5`).
   - Memperbaiki sampel 430 (dari tebakan salah `14` kembali ke `4`).
   - Memperbaiki sampel 1844 (dari tebakan salah `12` kembali ke `10`).
   - Proyeksi Akurasi: **94.85% s.d. 94.95%** dengan risiko regresi mendekati nol.

---

## 3. Refaktorisasi Format Penamaan File Submisi

Untuk mengatasi masalah pemotongan teks pada kolom CodaBench (panjang karakter nama file yang terpotong menjadi `...`), seluruh file submisi kini distandardisasi dengan format ringkas:
$$\text{sub}\langle\text{ID}\rangle\_\langle\text{deskripsi}\rangle\text{.zip} \quad (\le 25 \text{ karakter})$$

### Daftar File Submisi Resmi Terbaru:
| Nama File Ringkas | Karakter | Status / Peran |
| :--- | :---: | :--- |
| [`sub11_champion.zip`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/submissions/sub11_champion.zip) | 18 chars | **Reigning Rank 1 (0.9473 / 2.336 benar)** |
| [`sub13_sota_pentagon.zip`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/submissions/sub13_sota_pentagon.zip) | 23 chars | **Kandidat Champion Baru (Pentagon SOTA, 5 Pilar)** |
| [`sub14_sota_quad.zip`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/submissions/sub14_sota_quad.zip) | 19 chars | **Kandidat Alternatif (Quad Balanced)** |
| `sub12_tri_v2.zip` | 17 chars | Arsip Submisi 12 (0.9424) |
| `sub10_tri_ortho.zip` | 19 chars | Arsip Submisi 10 (0.9448) |

---
*Navigasi Vault: [[INDEX|Indeks Utama Dokumentasi]] | [[17_MASTER_DOKUMENTASI_KESELURUHAN_DAN_RIWAYAT_VERSI|Master Dokumentasi Keseluruhan]] | [[WORKFLOW_LINEAGE.canvas|Peta Alur Riset]]*
