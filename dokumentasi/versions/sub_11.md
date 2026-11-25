---
sub_id: 11
codabench_id: "1002441"
timestamp: "2026-10-19 15:04"
title: "Sub 11: submission_phase1_tri_orthogonal_v2.zip"
phase: 1
score: 0.9473
total_correct: 2336
total_samples: 2466
errors: 130
architecture: "Hybrid Tri-Ensemble v2+v1 Pure Visual"
status: "OFFICIAL_CODABENCH"
tags:
  - submission
  - phase1
---

# Sub 11: submission_phase1_tri_orthogonal_v2.zip

> [!INFO] Ringkasan Submisi Resmi CodaBench
> - **CodaBench Submission ID**: `1002441`
> - **Timestamp Server (GMT+7)**: `2026-10-19 15:04`
> - **Nomor Submisi**: `Sub 11` (`Fase 1`)
> - **Arsitektur / Metode**: `Hybrid Tri-Ensemble v2+v1 Pure Visual`
> - **Skor Leaderboard**: `==0.9473==` (2336 / 2466 benar, 130 kesalahan)
> - **Status Resmi**: `OFFICIAL_CODABENCH`

## 1. Detail Kinerja & Dekomposisi Metrik

| Metrik Evaluasi | Nilai Capaian | Catatan Teknis |
|:---|:---:|:---|
| **Akurasi Keseluruhan** | **`0.9473`** | 2336 dari 2466 sampel |
| **Jumlah Kesalahan** | **`130` sampel** | Evaluasi resmi CodaBench |
| **Chinese MNIST (1.500 citra)** | `1.0` | Evaluasi sub-domain formal |
| **Gaya Baru Micro (966 citra)** | `0.8654` | Evaluasi sub-domain kaligrafi |
| **Gaya Baru Macro (`acc_target`)** | `0.859` | Bobot seragam 13 kelas aktif |

## 2. Inovasi & Signifikansi Metodologis
- **Metodologi Utama**: 33% ResNet-18 v2 + 29% ConvNeXt v2 + 29% DenseNet v1 + 10% ResNet-18 v1 (pure visual argmax tanpa TTPA)
- **Korelasi Terhadap Garis Waktu**: Bagian integral dari evolusi riset yang didokumentasikan di [[17_MASTER_DOKUMENTASI_KESELURUHAN_DAN_RIWAYAT_VERSI|Master Dokumentasi]].
- **Visualisasi Alur**: Dapat dilihat pada peta koneksi [[WORKFLOW_LINEAGE.canvas]].

---
*Tercatat dalam basis data resmi [[OBSIDIAN_BASE_SUBMISSIONS.base]] dan indeks utama [[INDEX]].*
