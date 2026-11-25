---
sub_id: 13
codabench_id: "1002479"
timestamp: "2026-10-19 15:26"
title: "Sub 13: sub13_sota_pentagon.zip"
phase: 1
score: 0.9469
total_correct: 2335
total_samples: 2466
errors: 131
architecture: "Pentagon 5-Fold Ensemble"
status: "OFFICIAL_CODABENCH"
tags:
  - submission
  - phase1
---

# Sub 13: sub13_sota_pentagon.zip

> [!INFO] Ringkasan Submisi Resmi CodaBench
> - **CodaBench Submission ID**: `1002479`
> - **Timestamp Server (GMT+7)**: `2026-10-19 15:26`
> - **Nomor Submisi**: `Sub 13` (`Fase 1`)
> - **Arsitektur / Metode**: `Pentagon 5-Fold Ensemble`
> - **Skor Leaderboard**: `==0.9469==` (2335 / 2466 benar, 131 kesalahan)
> - **Status Resmi**: `OFFICIAL_CODABENCH`

## 1. Detail Kinerja & Dekomposisi Metrik

| Metrik Evaluasi | Nilai Capaian | Catatan Teknis |
|:---|:---:|:---|
| **Akurasi Keseluruhan** | **`0.9469`** | 2335 dari 2466 sampel |
| **Jumlah Kesalahan** | **`131` sampel** | Evaluasi resmi CodaBench |
| **Chinese MNIST (1.500 citra)** | `1.0` | Evaluasi sub-domain formal |
| **Gaya Baru Micro (966 citra)** | `0.8644` | Evaluasi sub-domain kaligrafi |
| **Gaya Baru Macro (`acc_target`)** | `0.8588` | Bobot seragam 13 kelas aktif |

## 2. Inovasi & Signifikansi Metodologis
- **Metodologi Utama**: Multi-scale 5-fold cross-validation blending
- **Korelasi Terhadap Garis Waktu**: Bagian integral dari evolusi riset yang didokumentasikan di [[17_MASTER_DOKUMENTASI_KESELURUHAN_DAN_RIWAYAT_VERSI|Master Dokumentasi]].
- **Visualisasi Alur**: Dapat dilihat pada peta koneksi [[WORKFLOW_LINEAGE.canvas]].

---
*Tercatat dalam basis data resmi [[OBSIDIAN_BASE_SUBMISSIONS.base]] dan indeks utama [[INDEX]].*
