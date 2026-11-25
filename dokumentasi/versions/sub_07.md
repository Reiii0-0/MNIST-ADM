---
sub_id: 7
codabench_id: "1000818"
timestamp: "2026-10-18 17:46"
title: "Sub 7: submission_phase1_blend_prior_adapt.zip"
phase: 1
score: 0.9327
total_correct: 2300
total_samples: 2466
errors: 166
architecture: "Blend + Bayesian Prior (TTPA)"
status: "OFFICIAL_CODABENCH"
tags:
  - submission
  - phase1
---

# Sub 7: submission_phase1_blend_prior_adapt.zip

> [!INFO] Ringkasan Submisi Resmi CodaBench
> - **CodaBench Submission ID**: `1000818`
> - **Timestamp Server (GMT+7)**: `2026-10-18 17:46`
> - **Nomor Submisi**: `Sub 7` (`Fase 1`)
> - **Arsitektur / Metode**: `Blend + Bayesian Prior (TTPA)`
> - **Skor Leaderboard**: `==0.9327==` (2300 / 2466 benar, 166 kesalahan)
> - **Status Resmi**: `OFFICIAL_CODABENCH`

## 1. Detail Kinerja & Dekomposisi Metrik

| Metrik Evaluasi | Nilai Capaian | Catatan Teknis |
|:---|:---:|:---|
| **Akurasi Keseluruhan** | **`0.9327`** | 2300 dari 2466 sampel |
| **Jumlah Kesalahan** | **`166` sampel** | Evaluasi resmi CodaBench |
| **Chinese MNIST (1.500 citra)** | `1.0` | Evaluasi sub-domain formal |
| **Gaya Baru Micro (966 citra)** | `0.8282` | Evaluasi sub-domain kaligrafi |
| **Gaya Baru Macro (`acc_target`)** | `0.7931` | Bobot seragam 13 kelas aktif |

## 2. Inovasi & Signifikansi Metodologis
- **Metodologi Utama**: Unsupervised EM algorithm on target distribution
- **Korelasi Terhadap Garis Waktu**: Bagian integral dari evolusi riset yang didokumentasikan di [[17_MASTER_DOKUMENTASI_KESELURUHAN_DAN_RIWAYAT_VERSI|Master Dokumentasi]].
- **Visualisasi Alur**: Dapat dilihat pada peta koneksi [[WORKFLOW_LINEAGE.canvas]].

---
*Tercatat dalam basis data resmi [[OBSIDIAN_BASE_SUBMISSIONS.base]] dan indeks utama [[INDEX]].*
