---
sub_id: 12
codabench_id: "1002466"
timestamp: "2026-10-19 15:17"
title: "Sub 12: submission_phase1_tri_orthogonal_v2.zip"
phase: 1
score: 0.9424
total_correct: 2324
total_samples: 2466
errors: 142
architecture: "Pure v2 Tri-Ensemble (Drop D1 Ablation)"
status: "OFFICIAL_CODABENCH"
tags:
  - submission
  - phase1
---

# Sub 12: submission_phase1_tri_orthogonal_v2.zip

> [!INFO] Ringkasan Submisi Resmi CodaBench
> - **CodaBench Submission ID**: `1002466`
> - **Timestamp Server (GMT+7)**: `2026-10-19 15:17`
> - **Nomor Submisi**: `Sub 12` (`Fase 1`)
> - **Arsitektur / Metode**: `Pure v2 Tri-Ensemble (Drop D1 Ablation)`
> - **Skor Leaderboard**: `==0.9424==` (2324 / 2466 benar, 142 kesalahan)
> - **Status Resmi**: `OFFICIAL_CODABENCH`

## 1. Detail Kinerja & Dekomposisi Metrik

| Metrik Evaluasi | Nilai Capaian | Catatan Teknis |
|:---|:---:|:---|
| **Akurasi Keseluruhan** | **`0.9424`** | 2324 dari 2466 sampel |
| **Jumlah Kesalahan** | **`142` sampel** | Evaluasi resmi CodaBench |
| **Chinese MNIST (1.500 citra)** | `1.0` | Evaluasi sub-domain formal |
| **Gaya Baru Micro (966 citra)** | `0.853` | Evaluasi sub-domain kaligrafi |
| **Gaya Baru Macro (`acc_target`)** | `0.8449` | Bobot seragam 13 kelas aktif |

## 2. Inovasi & Signifikansi Metodologis
- **Metodologi Utama**: 36% ResNet-18 v2 + 34% ConvNeXt v2 + 30% DenseNet v2 hasil hotfix; drop D1 memicu kolaps kelas lima (五)
- **Korelasi Terhadap Garis Waktu**: Bagian integral dari evolusi riset yang didokumentasikan di [[17_MASTER_DOKUMENTASI_KESELURUHAN_DAN_RIWAYAT_VERSI|Master Dokumentasi]].
- **Visualisasi Alur**: Dapat dilihat pada peta koneksi [[WORKFLOW_LINEAGE.canvas]].

---
*Tercatat dalam basis data resmi [[OBSIDIAN_BASE_SUBMISSIONS.base]] dan indeks utama [[INDEX]].*
