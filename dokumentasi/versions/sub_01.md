---
sub_id: 1
codabench_id: "995975"
timestamp: "2026-10-17 22:32"
title: "Sub 1: submission_phase1_adabn_tta.zip"
phase: 1
score: 0.9120
total_correct: 2249
total_samples: 2466
errors: 217
architecture: "ResNet-18 CBAM + AdaBN + TTA"
status: "OFFICIAL_CODABENCH"
tags:
  - submission
  - phase1
---

# Sub 1: submission_phase1_adabn_tta.zip

> [!INFO] Ringkasan Submisi Resmi CodaBench
> - **CodaBench Submission ID**: `995975`
> - **Timestamp Server (GMT+7)**: `2026-10-17 22:32`
> - **Nomor Submisi**: `Sub 1` (`Fase 1`)
> - **Arsitektur / Metode**: `ResNet-18 CBAM + AdaBN + TTA`
> - **Skor Leaderboard**: `==0.9120==` (2249 / 2466 benar, 217 kesalahan)
> - **Status Resmi**: `OFFICIAL_CODABENCH`

## 1. Detail Kinerja & Dekomposisi Metrik

| Metrik Evaluasi | Nilai Capaian | Catatan Teknis |
|:---|:---:|:---|
| **Akurasi Keseluruhan** | **`0.9120`** | 2249 dari 2466 sampel |
| **Jumlah Kesalahan** | **`217` sampel** | Evaluasi resmi CodaBench |
| **Chinese MNIST (1.500 citra)** | `0.9993` | Evaluasi sub-domain formal |
| **Gaya Baru Micro (966 citra)** | `0.7764` | Evaluasi sub-domain kaligrafi |
| **Gaya Baru Macro (`acc_target`)** | `0.7455` | Bobot seragam 13 kelas aktif |

## 2. Inovasi & Signifikansi Metodologis
- **Metodologi Utama**: First contact with Gaya Baru cursive shift
- **Korelasi Terhadap Garis Waktu**: Bagian integral dari evolusi riset yang didokumentasikan di [[17_MASTER_DOKUMENTASI_KESELURUHAN_DAN_RIWAYAT_VERSI|Master Dokumentasi]].
- **Visualisasi Alur**: Dapat dilihat pada peta koneksi [[WORKFLOW_LINEAGE.canvas]].

---
*Tercatat dalam basis data resmi [[OBSIDIAN_BASE_SUBMISSIONS.base]] dan indeks utama [[INDEX]].*
