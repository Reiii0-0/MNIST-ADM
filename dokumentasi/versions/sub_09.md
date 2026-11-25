---
sub_id: 9
codabench_id: "1001160"
timestamp: "2026-10-18 22:46"
title: "Sub 9: submission_phase1_tri_ensemble_sota.zip"
phase: 1
score: 0.9420
total_correct: 2323
total_samples: 2466
errors: 143
architecture: "Tri-Ensemble SOTA"
status: "OFFICIAL_CODABENCH"
tags:
  - submission
  - phase1
---

# Sub 9: submission_phase1_tri_ensemble_sota.zip

> [!INFO] Ringkasan Submisi Resmi CodaBench
> - **CodaBench Submission ID**: `1001160`
> - **Timestamp Server (GMT+7)**: `2026-10-18 22:46`
> - **Nomor Submisi**: `Sub 9` (`Fase 1`)
> - **Arsitektur / Metode**: `Tri-Ensemble SOTA`
> - **Skor Leaderboard**: `==0.9420==` (2323 / 2466 benar, 143 kesalahan)
> - **Status Resmi**: `OFFICIAL_CODABENCH`

## 1. Detail Kinerja & Dekomposisi Metrik

| Metrik Evaluasi | Nilai Capaian | Catatan Teknis |
|:---|:---:|:---|
| **Akurasi Keseluruhan** | **`0.9420`** | 2323 dari 2466 sampel |
| **Jumlah Kesalahan** | **`143` sampel** | Evaluasi resmi CodaBench |
| **Chinese MNIST (1.500 citra)** | `1.0` | Evaluasi sub-domain formal |
| **Gaya Baru Micro (966 citra)** | `0.852` | Evaluasi sub-domain kaligrafi |
| **Gaya Baru Macro (`acc_target`)** | `0.8237` | Bobot seragam 13 kelas aktif |

## 2. Inovasi & Signifikansi Metodologis
- **Metodologi Utama**: ResNet-18 + ConvNeXt-Femto + DenseNet-121 CBAM
- **Korelasi Terhadap Garis Waktu**: Bagian integral dari evolusi riset yang didokumentasikan di [[17_MASTER_DOKUMENTASI_KESELURUHAN_DAN_RIWAYAT_VERSI|Master Dokumentasi]].
- **Visualisasi Alur**: Dapat dilihat pada peta koneksi [[WORKFLOW_LINEAGE.canvas]].

---
*Tercatat dalam basis data resmi [[OBSIDIAN_BASE_SUBMISSIONS.base]] dan indeks utama [[INDEX]].*
