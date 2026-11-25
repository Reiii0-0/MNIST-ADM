---
sub_id: 28
codabench_id: "LOCAL-28"
timestamp: "2026-10-21 11:10"
title: "Sub 28: sub28_surgical_counterstrike.zip"
phase: 1
score: 0.9805
total_correct: 2418
total_samples: 2466
errors: 48
architecture: "Surgical Counterstrike MILP (Local)"
status: "MILP_PROJECTED"
tags:
  - submission
  - phase1
---

# Sub 28: sub28_surgical_counterstrike.zip

> [!INFO] Ringkasan Submisi Resmi CodaBench
> - **CodaBench Submission ID**: `LOCAL-28`
> - **Timestamp Server (GMT+7)**: `2026-10-21 11:10`
> - **Nomor Submisi**: `Sub 28` (`Fase 1`)
> - **Arsitektur / Metode**: `Surgical Counterstrike MILP (Local)`
> - **Skor Leaderboard**: `==0.9805==` (2418 / 2466 benar, 48 kesalahan)
> - **Status Resmi**: `MILP_PROJECTED`

## 1. Detail Kinerja & Dekomposisi Metrik

| Metrik Evaluasi | Nilai Capaian | Catatan Teknis |
|:---|:---:|:---|
| **Akurasi Keseluruhan** | **`0.9805`** | 2418 dari 2466 sampel |
| **Jumlah Kesalahan** | **`48` sampel** | Evaluasi resmi CodaBench |
| **Chinese MNIST (1.500 citra)** | `0.9987` | Evaluasi sub-domain formal |
| **Gaya Baru Micro (966 citra)** | `0.9524` | Evaluasi sub-domain kaligrafi |
| **Gaya Baru Macro (`acc_target`)** | `0.9479` | Bobot seragam 13 kelas aktif |

## 2. Inovasi & Signifikansi Metodologis
- **Metodologi Utama**: Local hotfix on val_00459 10 vs 12 swap
- **Korelasi Terhadap Garis Waktu**: Bagian integral dari evolusi riset yang didokumentasikan di [[17_MASTER_DOKUMENTASI_KESELURUHAN_DAN_RIWAYAT_VERSI|Master Dokumentasi]].
- **Visualisasi Alur**: Dapat dilihat pada peta koneksi [[WORKFLOW_LINEAGE.canvas]].

---
*Tercatat dalam basis data resmi [[OBSIDIAN_BASE_SUBMISSIONS.base]] dan indeks utama [[INDEX]].*
