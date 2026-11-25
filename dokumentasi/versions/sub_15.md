---
sub_id: 15
codabench_id: "1004105"
timestamp: "2026-10-20 08:02"
title: "Sub 15: sub15_sota_v3_grandmaster.zip"
phase: 1
score: 0.9448
total_correct: 2330
total_samples: 2466
errors: 136
architecture: "Grandmaster 15-Model Ensemble"
status: "OFFICIAL_CODABENCH"
tags:
  - submission
  - phase1
---

# Sub 15: sub15_sota_v3_grandmaster.zip

> [!INFO] Ringkasan Submisi Resmi CodaBench
> - **CodaBench Submission ID**: `1004105`
> - **Timestamp Server (GMT+7)**: `2026-10-20 08:02`
> - **Nomor Submisi**: `Sub 15` (`Fase 1`)
> - **Arsitektur / Metode**: `Grandmaster 15-Model Ensemble`
> - **Skor Leaderboard**: `==0.9448==` (2330 / 2466 benar, 136 kesalahan)
> - **Status Resmi**: `OFFICIAL_CODABENCH`

## 1. Detail Kinerja & Dekomposisi Metrik

| Metrik Evaluasi | Nilai Capaian | Catatan Teknis |
|:---|:---:|:---|
| **Akurasi Keseluruhan** | **`0.9448`** | 2330 dari 2466 sampel |
| **Jumlah Kesalahan** | **`136` sampel** | Evaluasi resmi CodaBench |
| **Chinese MNIST (1.500 citra)** | `1.0` | Evaluasi sub-domain formal |
| **Gaya Baru Micro (966 citra)** | `0.8592` | Evaluasi sub-domain kaligrafi |
| **Gaya Baru Macro (`acc_target`)** | `0.8522` | Bobot seragam 13 kelas aktif |

## 2. Inovasi & Signifikansi Metodologis
- **Metodologi Utama**: 15-model v1+v2+DenseNet v2 across 3 architecture families (~444.55 MiB local checkpoints)
- **Korelasi Terhadap Garis Waktu**: Bagian integral dari evolusi riset yang didokumentasikan di [[17_MASTER_DOKUMENTASI_KESELURUHAN_DAN_RIWAYAT_VERSI|Master Dokumentasi]].
- **Visualisasi Alur**: Dapat dilihat pada peta koneksi [[WORKFLOW_LINEAGE.canvas]].

---
*Tercatat dalam basis data resmi [[OBSIDIAN_BASE_SUBMISSIONS.base]] dan indeks utama [[INDEX]].*
