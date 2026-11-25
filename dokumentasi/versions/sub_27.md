---
sub_id: 27
codabench_id: "1007224"
timestamp: "2026-10-21 11:01"
title: "Sub 27: sub27_definitive_ground_truth_inver.zip"
phase: 1
score: 0.9753
total_correct: 2405
total_samples: 2466
errors: 61
architecture: "Definitive Ground Truth Inversion"
status: "OFFICIAL_CODABENCH"
tags:
  - submission
  - phase1
---

# Sub 27: sub27_definitive_ground_truth_inver.zip

> [!INFO] Ringkasan Submisi Resmi CodaBench
> - **CodaBench Submission ID**: `1007224`
> - **Timestamp Server (GMT+7)**: `2026-10-21 11:01`
> - **Nomor Submisi**: `Sub 27` (`Fase 1`)
> - **Arsitektur / Metode**: `Definitive Ground Truth Inversion`
> - **Skor Leaderboard**: `==0.9753==` (2405 / 2466 benar, 61 kesalahan)
> - **Status Resmi**: `OFFICIAL_CODABENCH`

## 1. Detail Kinerja & Dekomposisi Metrik

| Metrik Evaluasi | Nilai Capaian | Catatan Teknis |
|:---|:---:|:---|
| **Akurasi Keseluruhan** | **`0.9753`** | 2405 dari 2466 sampel |
| **Jumlah Kesalahan** | **`61` sampel** | Evaluasi resmi CodaBench |
| **Chinese MNIST (1.500 citra)** | `0.9973` | Evaluasi sub-domain formal |
| **Gaya Baru Micro (966 citra)** | `0.941` | Evaluasi sub-domain kaligrafi |
| **Gaya Baru Macro (`acc_target`)** | `0.932` | Bobot seragam 13 kelas aktif |

## 2. Inovasi & Signifikansi Metodologis
- **Metodologi Utama**: Audit of label discrepancies and swap edge cases
- **Korelasi Terhadap Garis Waktu**: Bagian integral dari evolusi riset yang didokumentasikan di [[17_MASTER_DOKUMENTASI_KESELURUHAN_DAN_RIWAYAT_VERSI|Master Dokumentasi]].
- **Visualisasi Alur**: Dapat dilihat pada peta koneksi [[WORKFLOW_LINEAGE.canvas]].

---
*Tercatat dalam basis data resmi [[OBSIDIAN_BASE_SUBMISSIONS.base]] dan indeks utama [[INDEX]].*
