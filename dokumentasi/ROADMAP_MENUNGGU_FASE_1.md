---
title: "ROADMAP & ACTION PLAN: MENUNGGU PEMBUKAAN FASE 1 (13 - 14 OKTOBER 2026)"
source_doc: "ROADMAP_MENUNGGU_FASE_1.md"
date: 2026-10-13
updated: 2026-11-24
tags:
  - adm-project
  - documentation
  - pra-fase
---

# ROADMAP & ACTION PLAN: MENUNGGU PEMBUKAAN FASE 1 (13 - 14 OKTOBER 2026)
## Proyek Analisis Data Multivariat — CodaBench Chinese Handwritten Character Classification

**Periode:** 13 Oktober 2026 s.d. 14 Oktober 2026 (Menjelang Rilis Fase 1: 15 Oktober 2026 pukul 00:00 WIB)  
**Tujuan:** Memaksimalkan kesiapan akademik dan keunggulan teknis kompetisi tanpa terburu-buru melakukan submisi prematur.

---

## 1. Status Terkini Pipeline (Baseline Ready)
- **Metrik SOTA:** 5-Fold OOF Accuracy **99,79%** (hanya 22 error dari 10.500 sampel).
- **Blended Ensemble:** 5-Fold Ensemble + 5-Way TTA menyembuhkan 13 error $\to$ akurasi meningkat ke **99,914%** (sisa 9 error).
- **Bentuk Tradisional ('萬' & '億'):** Akurasi **100,00%** (700/700 sampel sintetik benar).
- **Kesiapan Script Inferensi:** [`prepare_submission.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/prepare_submission.py) bebas bug, dilengkapi modul AdaBN dan validasi integritas ZIP CodaBench otomatis.

---

## 2. Empat Agenda Strategis Selama Menunggu (48 Jam ke Depan)

### Agenda 1: Persiapan Jalur Akademik ADM (Laporan Akhir & Slide Presentasi)
*Prioritas: SANGAT TINGGI untuk Mengunci Nilai A+ Mata Kuliah*
1. **Menyusun Draft Makalah Ilmiah Format Standar (IEEE/Springer ADM):**
   - Menghubungkan teori buku teks Johnson & Wichern (*Applied Multivariate Statistical Analysis*) dengan hasil empiris:
     - Dekomposisi Spectral & SVD matriks kovarians $\mathbf{S}$.
     - Kriteria Kaiser ($275$ PC) dan Scree Plot (CPVE $80\%, 85\%, 90\%, 95\%$).
     - MANOVA: Signifikansi Wilks' Lambda ($\Lambda = 8,89 \times 10^{-5}$, $p < 10^{-15}$) dan Pillai's Trace ($V = 5,8418$).
     - Uji Box's M ($p < 0,001$) untuk heterogenitas matriks kovarians sebagai justifikasi mengapa QDA ($96,09\%$) mengungguli LDA ($85,13\%$).
2. **Menyiapkan Draft Slide Presentasi Sidang Akhir (Marp / PowerPoint):**
   - Membuat outline presentasi 10–15 menit yang memadukan visualisasi multivariat klasik (eigen-characters, Q-Q plot, 2D canonical projection) dengan rekayasa deep learning SOTA.

### Agenda 2: Pembangunan Script Otomatisasi Fase 1 (`run_phase1_pipeline.py`)
*Prioritas: TINGGI untuk Eksekusi Hari-H Tanpa Keraguan (Zero Hesitation)*
1. Membuat script eksekusi satu-perintah (*single-command runner*) yang menangani:
   - Verifikasi integritas folder `FinalProject/data_validasi/` (memastikan tepat 2.466 citra).
   - Eksekusi instan AdaBN Domain Adaptation (<5 detik).
   - Eksekusi 5-Model Ensemble + 5-Way Power-Mean TTA ($p=1.8$).
   - Pembuatan dua paket submisi otomatis:
     - `submission_sub1_anchor.zip` (baseline tanpa AdaBN sebagai pembanding leaderboard).
     - `submission_sub2_adabn_tta.zip` (SOTA utama untuk klaim Peringkat #1).
   - Ekstraksi kandidat pseudo-labeling berkeyakinan tinggi ($P_{\max} > 0.985$, $\Delta P > 0.70$) untuk persiapan Fase 2.

### Agenda 3: Stress Testing Ketahanan Model terhadap Blur & Ketebalan Ekstrem
*Prioritas: MENENGAH untuk Verifikasi Pertahanan Model*
1. Catatan panitia menyebutkan: *"Goresannya tidak dipertebal karena sudah tebal, tetapi diberi blur ringan agar tepinya sehalus data latih"*.
2. Kita dapat membuat script uji lokal untuk menguji ketahanan bobot model kita terhadap berbagai tingkat Gaussian Blur ($\sigma \in [0.5, 2.0]$) dan pelebaran goresan (*dilation*).
3. Mengukur seberapa besar AdaBN mampu mengembalikan performa jika citra uji mengalami degradasi kehalusan tepi yang drastis.

### Agenda 4: Eksplorasi Model Diversitas Kedua (Opsional di Kaggle Cloud)
*Prioritas: CADANGAN / STRATEGIS*
1. ResNet18-CBAM kita sudah sangat kuat ($99,79\%$).
2. Namun dalam prinsip ensemble machine learning tingkat tinggi, memadukan dua arsitektur berbeda famili (misalnya ResNet + Compact CNN multi-scale atau ConvNeXt-Femto) memberikan pengurangan varians estimasi yang lebih kuat daripada sekadar model sejenis.
3. Menyiapkan kernel pelatihan model alternatif di Kaggle CLI untuk dijalankan secara asynchronous jika diperlukan.

---

## 3. Timeline Eksekusi Menuju 15 Oktober 2026

```
+-----------------------------------------------------------------------------------------+
| HARI / TANGGAL        FOKUS UTAMA                  OUTPUT / DELIVERABLE                 |
+-----------------------------------------------------------------------------------------+
| Selasa, 13 Okt 2026   Dokumentasi & Automasi       • ROADMAP & Dokumentasi Pra-Fase     |
|                       Script Hari-H                • run_phase1_pipeline.py siap uji    |
|                                                    • Stress-test blur & ketebalan       |
+-----------------------------------------------------------------------------------------+
| Rabu, 14 Okt 2026     Jalur Akademik ADM           • Draft Laporan Ilmiah ADM Lengkap   |
| (T - 1 Hari)          & Kesiapan Browser           • Draft Slide Presentasi Sidang      |
|                                                    • Akun & environment CodaBench siap  |
+-----------------------------------------------------------------------------------------+
| Kamis, 15 Okt 2026    HARI-H PEMBUKAAN FASE 1      • Unduh 2.466 citra validasi         |
| (00:00 WIB)           Eksekusi Submisi             • Jalankan run_phase1_pipeline.py    |
|                                                    • Upload Submisi 1 & 2 ke CodaBench  |
|                                                    • Validasi Peringkat #1 Leaderboard  |
+-----------------------------------------------------------------------------------------+
```

---
*Navigasi Vault: [[INDEX|Indeks Utama Dokumentasi]] | [[17_MASTER_DOKUMENTASI_KESELURUHAN_DAN_RIWAYAT_VERSI|Master Dokumentasi Keseluruhan]] | [[WORKFLOW_LINEAGE.canvas|Peta Alur Riset]]*
