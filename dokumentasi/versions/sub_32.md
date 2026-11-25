---
sub_id: 32
codabench_id: "1014243"
timestamp: "2026-10-28 06:18"
title: "Sub 32: sub32_student_distillation_100.zip"
phase: 1
score: 0.9984
total_correct: 2462
total_samples: 2466
errors: 4
architecture: "Knowledge Distillation Student SWA FP16"
status: "OFFICIAL_CODABENCH_PEAK"
tags:
  - submission
  - phase1
---

# Sub 32: sub32_student_distillation_100.zip

> [!INFO] Ringkasan Submisi Resmi CodaBench
> - **CodaBench Submission ID**: `1014243`
> - **Timestamp Server (GMT+7)**: `2026-10-28 06:18`
> - **Nomor Submisi**: `Sub 32` (`Fase 1`)
> - **Arsitektur / Metode**: `Knowledge Distillation Student SWA FP16`
> - **Skor Leaderboard**: `==0.9984==` (2462 / 2466 benar, 4 kesalahan)
> - **Status Resmi**: `OFFICIAL_CODABENCH_PEAK`

## 1. Detail Kinerja & Dekomposisi Metrik

| Metrik Evaluasi | Nilai Capaian | Catatan Teknis |
|:---|:---:|:---|
| **Akurasi Keseluruhan** | **`0.9984`** | 2462 dari 2466 sampel |
| **Jumlah Kesalahan** | **`4` sampel** | Evaluasi resmi CodaBench |
| **Chinese MNIST (1.500 citra)** | `0.9987` | Evaluasi sub-domain formal |
| **Gaya Baru Micro (966 citra)** | `0.9979` | Evaluasi sub-domain kaligrafi |
| **Gaya Baru Macro (`acc_target`)** | `0.9985` | Bobot seragam 13 kelas aktif |

## 2. Inovasi & Signifikansi Metodologis
- **Metodologi Utama**: Single DenseNet-121 CBAM student: 0.9984 (only 4 errors, 14.75 MiB)
- **Korelasi Terhadap Garis Waktu**: Bagian integral dari evolusi riset yang didokumentasikan di [[17_MASTER_DOKUMENTASI_KESELURUHAN_DAN_RIWAYAT_VERSI|Master Dokumentasi]].
- **Visualisasi Alur**: Dapat dilihat pada peta koneksi [[WORKFLOW_LINEAGE.canvas]].

---
*Tercatat dalam basis data resmi [[OBSIDIAN_BASE_SUBMISSIONS.base]] dan indeks utama [[INDEX]].*
