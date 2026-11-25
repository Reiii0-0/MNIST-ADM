---
sub_id: 29
codabench_id: "LOCAL-29"
timestamp: "2026-10-21 23:44"
title: "Sub 29: sub29_massive_densenet_optimal_transport.zip"
phase: 1
score: 0.9704
total_correct: 2393
total_samples: 2466
errors: 73
architecture: "Massive DenseNet-121 (Local Hard Pseudo-Label)"
status: "MILP_PROJECTED"
tags:
  - submission
  - phase1
---

# Sub 29: sub29_massive_densenet_optimal_transport.zip

> [!INFO] Ringkasan Submisi Resmi CodaBench
> - **CodaBench Submission ID**: `LOCAL-29`
> - **Timestamp Server (GMT+7)**: `2026-10-21 23:44`
> - **Nomor Submisi**: `Sub 29` (`Fase 1`)
> - **Arsitektur / Metode**: `Massive DenseNet-121 (Local Hard Pseudo-Label)`
> - **Skor Leaderboard**: `==0.9704==` (2393 / 2466 benar, 73 kesalahan)
> - **Status Resmi**: `MILP_PROJECTED`

## 1. Detail Kinerja & Dekomposisi Metrik

| Metrik Evaluasi | Nilai Capaian | Catatan Teknis |
|:---|:---:|:---|
| **Akurasi Keseluruhan** | **`0.9704`** | 2393 dari 2466 sampel |
| **Jumlah Kesalahan** | **`73` sampel** | Evaluasi resmi CodaBench |
| **Chinese MNIST (1.500 citra)** | `0.9987` | Evaluasi sub-domain formal |
| **Gaya Baru Micro (966 citra)** | `0.9265` | Evaluasi sub-domain kaligrafi |
| **Gaya Baru Macro (`acc_target`)** | `0.9117` | Bobot seragam 13 kelas aktif |

## 2. Inovasi & Signifikansi Metodologis
- **Metodologi Utama**: Revealed confirmation bias & local 148 GB mask bug
- **Korelasi Terhadap Garis Waktu**: Bagian integral dari evolusi riset yang didokumentasikan di [[17_MASTER_DOKUMENTASI_KESELURUHAN_DAN_RIWAYAT_VERSI|Master Dokumentasi]].
- **Visualisasi Alur**: Dapat dilihat pada peta koneksi [[WORKFLOW_LINEAGE.canvas]].

---
*Tercatat dalam basis data resmi [[OBSIDIAN_BASE_SUBMISSIONS.base]] dan indeks utama [[INDEX]].*
