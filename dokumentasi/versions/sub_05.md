---
sub_id: 5
codabench_id: "996080"
timestamp: "2026-10-17 23:40"
title: "Sub 5: submission_phase1_gated_moe.zip"
phase: 1
score: 0.9286
total_correct: 2290
total_samples: 2466
errors: 176
architecture: "Gated Mixture-of-Experts"
status: "OFFICIAL_CODABENCH"
tags:
  - submission
  - phase1
---

# Sub 5: submission_phase1_gated_moe.zip

> [!INFO] Ringkasan Submisi Resmi CodaBench
> - **CodaBench Submission ID**: `996080`
> - **Timestamp Server (GMT+7)**: `2026-10-17 23:40`
> - **Nomor Submisi**: `Sub 5` (`Fase 1`)
> - **Arsitektur / Metode**: `Gated Mixture-of-Experts`
> - **Skor Leaderboard**: `==0.9286==` (2290 / 2466 benar, 176 kesalahan)
> - **Status Resmi**: `OFFICIAL_CODABENCH`

## 1. Detail Kinerja & Dekomposisi Metrik

| Metrik Evaluasi | Nilai Capaian | Catatan Teknis |
|:---|:---:|:---|
| **Akurasi Keseluruhan** | **`0.9286`** | 2290 dari 2466 sampel |
| **Jumlah Kesalahan** | **`176` sampel** | Evaluasi resmi CodaBench |
| **Chinese MNIST (1.500 citra)** | `0.9993` | Evaluasi sub-domain formal |
| **Gaya Baru Micro (966 citra)** | `0.8188` | Evaluasi sub-domain kaligrafi |
| **Gaya Baru Macro (`acc_target`)** | `0.7876` | Bobot seragam 13 kelas aktif |

## 2. Inovasi & Signifikansi Metodologis
- **Metodologi Utama**: Confidence-based router between CNNs (0.9286)
- **Korelasi Terhadap Garis Waktu**: Bagian integral dari evolusi riset yang didokumentasikan di [[17_MASTER_DOKUMENTASI_KESELURUHAN_DAN_RIWAYAT_VERSI|Master Dokumentasi]].
- **Visualisasi Alur**: Dapat dilihat pada peta koneksi [[WORKFLOW_LINEAGE.canvas]].

---
*Tercatat dalam basis data resmi [[OBSIDIAN_BASE_SUBMISSIONS.base]] dan indeks utama [[INDEX]].*
