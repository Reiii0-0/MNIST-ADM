---
sub_id: 8
codabench_id: "1001089"
timestamp: "2026-10-18 21:33"
title: "Sub 8: submission_phase1_bayes_calibrated_.zip"
phase: 1
score: 0.9465
total_correct: 2334
total_samples: 2466
errors: 132
architecture: "Bayes Calibrated Blend"
status: "OFFICIAL_CODABENCH"
tags:
  - submission
  - phase1
---

# Sub 8: submission_phase1_bayes_calibrated_.zip

> [!INFO] Ringkasan Submisi Resmi CodaBench
> - **CodaBench Submission ID**: `1001089`
> - **Timestamp Server (GMT+7)**: `2026-10-18 21:33`
> - **Nomor Submisi**: `Sub 8` (`Fase 1`)
> - **Arsitektur / Metode**: `Bayes Calibrated Blend`
> - **Skor Leaderboard**: `==0.9465==` (2334 / 2466 benar, 132 kesalahan)
> - **Status Resmi**: `OFFICIAL_CODABENCH`

## 1. Detail Kinerja & Dekomposisi Metrik

| Metrik Evaluasi | Nilai Capaian | Catatan Teknis |
|:---|:---:|:---|
| **Akurasi Keseluruhan** | **`0.9465`** | 2334 dari 2466 sampel |
| **Jumlah Kesalahan** | **`132` sampel** | Evaluasi resmi CodaBench |
| **Chinese MNIST (1.500 citra)** | `1.0` | Evaluasi sub-domain formal |
| **Gaya Baru Micro (966 citra)** | `0.8634` | Evaluasi sub-domain kaligrafi |
| **Gaya Baru Macro (`acc_target`)** | `0.8228` | Bobot seragam 13 kelas aktif |

## 2. Inovasi & Signifikansi Metodologis
- **Metodologi Utama**: Temperature Scaling (T=1.15) on OOF splits
- **Korelasi Terhadap Garis Waktu**: Bagian integral dari evolusi riset yang didokumentasikan di [[17_MASTER_DOKUMENTASI_KESELURUHAN_DAN_RIWAYAT_VERSI|Master Dokumentasi]].
- **Visualisasi Alur**: Dapat dilihat pada peta koneksi [[WORKFLOW_LINEAGE.canvas]].

---
*Tercatat dalam basis data resmi [[OBSIDIAN_BASE_SUBMISSIONS.base]] dan indeks utama [[INDEX]].*
