---
sub_id: 16
codabench_id: "1004844"
timestamp: "2026-10-20 15:29"
title: "Sub 16: sub_track1_stm.zip"
phase: 1
score: 0.9469
total_correct: 2335
total_samples: 2466
errors: 131
architecture: "Style Transfer Mapping (Track 1)"
status: "OFFICIAL_CODABENCH"
tags:
  - submission
  - phase1
---

# Sub 16: sub_track1_stm.zip

> [!INFO] Ringkasan Submisi Resmi CodaBench
> - **CodaBench Submission ID**: `1004844`
> - **Timestamp Server (GMT+7)**: `2026-10-20 15:29`
> - **Nomor Submisi**: `Sub 16` (`Fase 1`)
> - **Arsitektur / Metode**: `Style Transfer Mapping (Track 1)`
> - **Skor Leaderboard**: `==0.9469==` (2335 / 2466 benar, 131 kesalahan)
> - **Status Resmi**: `OFFICIAL_CODABENCH`

## 1. Detail Kinerja & Dekomposisi Metrik

| Metrik Evaluasi | Nilai Capaian | Catatan Teknis |
|:---|:---:|:---|
| **Akurasi Keseluruhan** | **`0.9469`** | 2335 dari 2466 sampel |
| **Jumlah Kesalahan** | **`131` sampel** | Evaluasi resmi CodaBench |
| **Chinese MNIST (1.500 citra)** | `0.9987` | Evaluasi sub-domain formal |
| **Gaya Baru Micro (966 citra)** | `0.8665` | Evaluasi sub-domain kaligrafi |
| **Gaya Baru Macro (`acc_target`)** | `0.8467` | Bobot seragam 13 kelas aktif |

## 2. Inovasi & Signifikansi Metodologis
- **Metodologi Utama**: Bengio et al. TPAMI 2017 affine alignment (Az+b)
- **Korelasi Terhadap Garis Waktu**: Bagian integral dari evolusi riset yang didokumentasikan di [[17_MASTER_DOKUMENTASI_KESELURUHAN_DAN_RIWAYAT_VERSI|Master Dokumentasi]].
- **Visualisasi Alur**: Dapat dilihat pada peta koneksi [[WORKFLOW_LINEAGE.canvas]].

---
*Tercatat dalam basis data resmi [[OBSIDIAN_BASE_SUBMISSIONS.base]] dan indeks utama [[INDEX]].*
