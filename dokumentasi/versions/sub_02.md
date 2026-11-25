---
sub_id: 2
codabench_id: "995976"
timestamp: "2026-10-17 22:33"
title: "Sub 2: submission_phase1_anchor.zip"
phase: 1
score: 0.9278
total_correct: 2288
total_samples: 2466
errors: 178
architecture: "ResNet-18 CBAM Anchor"
status: "OFFICIAL_CODABENCH"
tags:
  - submission
  - phase1
---

# Sub 2: submission_phase1_anchor.zip

> [!INFO] Ringkasan Submisi Resmi CodaBench
> - **CodaBench Submission ID**: `995976`
> - **Timestamp Server (GMT+7)**: `2026-10-17 22:33`
> - **Nomor Submisi**: `Sub 2` (`Fase 1`)
> - **Arsitektur / Metode**: `ResNet-18 CBAM Anchor`
> - **Skor Leaderboard**: `==0.9278==` (2288 / 2466 benar, 178 kesalahan)
> - **Status Resmi**: `OFFICIAL_CODABENCH`

## 1. Detail Kinerja & Dekomposisi Metrik

| Metrik Evaluasi | Nilai Capaian | Catatan Teknis |
|:---|:---:|:---|
| **Akurasi Keseluruhan** | **`0.9278`** | 2288 dari 2466 sampel |
| **Jumlah Kesalahan** | **`178` sampel** | Evaluasi resmi CodaBench |
| **Chinese MNIST (1.500 citra)** | `0.9993` | Evaluasi sub-domain formal |
| **Gaya Baru Micro (966 citra)** | `0.8168` | Evaluasi sub-domain kaligrafi |
| **Gaya Baru Macro (`acc_target`)** | `0.7847` | Bobot seragam 13 kelas aktif |

## 2. Inovasi & Signifikansi Metodologis
- **Metodologi Utama**: Pure visual single model anchor
- **Korelasi Terhadap Garis Waktu**: Bagian integral dari evolusi riset yang didokumentasikan di [[17_MASTER_DOKUMENTASI_KESELURUHAN_DAN_RIWAYAT_VERSI|Master Dokumentasi]].
- **Visualisasi Alur**: Dapat dilihat pada peta koneksi [[WORKFLOW_LINEAGE.canvas]].

---
*Tercatat dalam basis data resmi [[OBSIDIAN_BASE_SUBMISSIONS.base]] dan indeks utama [[INDEX]].*
