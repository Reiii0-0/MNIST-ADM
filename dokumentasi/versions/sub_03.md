---
sub_id: 3
codabench_id: "996005"
timestamp: "2026-10-17 22:52"
title: "Sub 3: submission_phase1_blend_resnet_conv.zip"
phase: 1
score: 0.9319
total_correct: 2298
total_samples: 2466
errors: 168
architecture: "ResNet-18 CBAM (85%) + ConvNeXt-Femto (15%)"
status: "OFFICIAL_CODABENCH"
tags:
  - submission
  - phase1
---

# Sub 3: submission_phase1_blend_resnet_conv.zip

> [!INFO] Ringkasan Submisi Resmi CodaBench
> - **CodaBench Submission ID**: `996005`
> - **Timestamp Server (GMT+7)**: `2026-10-17 22:52`
> - **Nomor Submisi**: `Sub 3` (`Fase 1`)
> - **Arsitektur / Metode**: `ResNet-18 CBAM (85%) + ConvNeXt-Femto (15%)`
> - **Skor Leaderboard**: `==0.9319==` (2298 / 2466 benar, 168 kesalahan)
> - **Status Resmi**: `OFFICIAL_CODABENCH`

## 1. Detail Kinerja & Dekomposisi Metrik

| Metrik Evaluasi | Nilai Capaian | Catatan Teknis |
|:---|:---:|:---|
| **Akurasi Keseluruhan** | **`0.9319`** | 2298 dari 2466 sampel |
| **Jumlah Kesalahan** | **`168` sampel** | Evaluasi resmi CodaBench |
| **Chinese MNIST (1.500 citra)** | `1.0` | Evaluasi sub-domain formal |
| **Gaya Baru Micro (966 citra)** | `0.8261` | Evaluasi sub-domain kaligrafi |
| **Gaya Baru Macro (`acc_target`)** | `0.7994` | Bobot seragam 13 kelas aktif |

## 2. Inovasi & Signifikansi Metodologis
- **Metodologi Utama**: Rank 1 CodaBench capture (0.9319)
- **Korelasi Terhadap Garis Waktu**: Bagian integral dari evolusi riset yang didokumentasikan di [[17_MASTER_DOKUMENTASI_KESELURUHAN_DAN_RIWAYAT_VERSI|Master Dokumentasi]].
- **Visualisasi Alur**: Dapat dilihat pada peta koneksi [[WORKFLOW_LINEAGE.canvas]].

---
*Tercatat dalam basis data resmi [[OBSIDIAN_BASE_SUBMISSIONS.base]] dan indeks utama [[INDEX]].*
