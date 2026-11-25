---
sub_id: 6
codabench_id: "1000807"
timestamp: "2026-10-18 17:42"
title: "Sub 6: submission_phase1_retrained_sota.zip"
phase: 1
score: 0.9148
total_correct: 2256
total_samples: 2466
errors: 210
architecture: "ResNet-18 CBAM Retrained"
status: "OFFICIAL_CODABENCH"
tags:
  - submission
  - phase1
---

# Sub 6: submission_phase1_retrained_sota.zip

> [!INFO] Ringkasan Submisi Resmi CodaBench
> - **CodaBench Submission ID**: `1000807`
> - **Timestamp Server (GMT+7)**: `2026-10-18 17:42`
> - **Nomor Submisi**: `Sub 6` (`Fase 1`)
> - **Arsitektur / Metode**: `ResNet-18 CBAM Retrained`
> - **Skor Leaderboard**: `==0.9148==` (2256 / 2466 benar, 210 kesalahan)
> - **Status Resmi**: `OFFICIAL_CODABENCH`

## 1. Detail Kinerja & Dekomposisi Metrik

| Metrik Evaluasi | Nilai Capaian | Catatan Teknis |
|:---|:---:|:---|
| **Akurasi Keseluruhan** | **`0.9148`** | 2256 dari 2466 sampel |
| **Jumlah Kesalahan** | **`210` sampel** | Evaluasi resmi CodaBench |
| **Chinese MNIST (1.500 citra)** | `0.9993` | Evaluasi sub-domain formal |
| **Gaya Baru Micro (966 citra)** | `0.7836` | Evaluasi sub-domain kaligrafi |
| **Gaya Baru Macro (`acc_target`)** | `0.7547` | Bobot seragam 13 kelas aktif |

## 2. Inovasi & Signifikansi Metodologis
- **Metodologi Utama**: Trained with Biomechanical Elastic Distortion
- **Korelasi Terhadap Garis Waktu**: Bagian integral dari evolusi riset yang didokumentasikan di [[17_MASTER_DOKUMENTASI_KESELURUHAN_DAN_RIWAYAT_VERSI|Master Dokumentasi]].
- **Visualisasi Alur**: Dapat dilihat pada peta koneksi [[WORKFLOW_LINEAGE.canvas]].

---
*Tercatat dalam basis data resmi [[OBSIDIAN_BASE_SUBMISSIONS.base]] dan indeks utama [[INDEX]].*
