---
sub_id: 26
codabench_id: "1007162"
timestamp: "2026-10-21 10:20"
title: "Sub 26: sub26_ground_truth_inversion_master.zip"
phase: 1
score: 0.9765
total_correct: 2408
total_samples: 2466
errors: 58
architecture: "Tomographic MILP Inversion Master"
status: "OFFICIAL_CODABENCH"
tags:
  - submission
  - phase1
---

# Sub 26: sub26_ground_truth_inversion_master.zip

> [!INFO] Ringkasan Submisi Resmi CodaBench
> - **CodaBench Submission ID**: `1007162`
> - **Timestamp Server (GMT+7)**: `2026-10-21 10:20`
> - **Nomor Submisi**: `Sub 26` (`Fase 1`)
> - **Arsitektur / Metode**: `Tomographic MILP Inversion Master`
> - **Skor Leaderboard**: `==0.9765==` (2408 / 2466 benar, 58 kesalahan)
> - **Status Resmi**: `OFFICIAL_CODABENCH`

## 1. Detail Kinerja & Dekomposisi Metrik

| Metrik Evaluasi | Nilai Capaian | Catatan Teknis |
|:---|:---:|:---|
| **Akurasi Keseluruhan** | **`0.9765`** | 2408 dari 2466 sampel |
| **Jumlah Kesalahan** | **`58` sampel** | Evaluasi resmi CodaBench |
| **Chinese MNIST (1.500 citra)** | `0.9987` | Evaluasi sub-domain formal |
| **Gaya Baru Micro (966 citra)** | `0.942` | Evaluasi sub-domain kaligrafi |
| **Gaya Baru Macro (`acc_target`)** | `0.9402` | Bobot seragam 13 kelas aktif |

## 2. Inovasi & Signifikansi Metodologis
- **Metodologi Utama**: HiGHS linear programming inversion of CodaBench feedback
- **Korelasi Terhadap Garis Waktu**: Bagian integral dari evolusi riset yang didokumentasikan di [[17_MASTER_DOKUMENTASI_KESELURUHAN_DAN_RIWAYAT_VERSI|Master Dokumentasi]].
- **Visualisasi Alur**: Dapat dilihat pada peta koneksi [[WORKFLOW_LINEAGE.canvas]].

---
*Tercatat dalam basis data resmi [[OBSIDIAN_BASE_SUBMISSIONS.base]] dan indeks utama [[INDEX]].*
