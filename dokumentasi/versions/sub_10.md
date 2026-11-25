---
sub_id: 10
codabench_id: "1001367"
timestamp: "2026-10-19 01:17"
title: "Sub 10: submission_phase1_tri_orthogonal_en.zip"
phase: 1
score: 0.9448
total_correct: 2330
total_samples: 2466
errors: 136
architecture: "Tri-Ensemble Orthogonal Inductive Bias"
status: "OFFICIAL_CODABENCH"
tags:
  - submission
  - phase1
---

# Sub 10: submission_phase1_tri_orthogonal_en.zip

> [!INFO] Ringkasan Submisi Resmi CodaBench
> - **CodaBench Submission ID**: `1001367`
> - **Timestamp Server (GMT+7)**: `2026-10-19 01:17`
> - **Nomor Submisi**: `Sub 10` (`Fase 1`)
> - **Arsitektur / Metode**: `Tri-Ensemble Orthogonal Inductive Bias`
> - **Skor Leaderboard**: `==0.9448==` (2330 / 2466 benar, 136 kesalahan)
> - **Status Resmi**: `OFFICIAL_CODABENCH`

## 1. Detail Kinerja & Dekomposisi Metrik

| Metrik Evaluasi | Nilai Capaian | Catatan Teknis |
|:---|:---:|:---|
| **Akurasi Keseluruhan** | **`0.9448`** | 2330 dari 2466 sampel |
| **Jumlah Kesalahan** | **`136` sampel** | Evaluasi resmi CodaBench |
| **Chinese MNIST (1.500 citra)** | `1.0` | Evaluasi sub-domain formal |
| **Gaya Baru Micro (966 citra)** | `0.8592` | Evaluasi sub-domain kaligrafi |
| **Gaya Baru Macro (`acc_target`)** | `0.8486` | Bobot seragam 13 kelas aktif |

## 2. Inovasi & Signifikansi Metodologis
- **Metodologi Utama**: 50% ResNet-18 + 20% ConvNeXt + 30% DenseNet-121 (multi-family orthogonal inductive bias, tanpa regularisasi loss ortogonal)
- **Korelasi Terhadap Garis Waktu**: Bagian integral dari evolusi riset yang didokumentasikan di [[17_MASTER_DOKUMENTASI_KESELURUHAN_DAN_RIWAYAT_VERSI|Master Dokumentasi]].
- **Visualisasi Alur**: Dapat dilihat pada peta koneksi [[WORKFLOW_LINEAGE.canvas]].

---
*Tercatat dalam basis data resmi [[OBSIDIAN_BASE_SUBMISSIONS.base]] dan indeks utama [[INDEX]].*
