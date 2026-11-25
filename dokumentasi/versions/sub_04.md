---
sub_id: 4
codabench_id: "996076"
timestamp: "2026-10-17 23:37"
title: "Sub 4: submission_phase1_hccr_senet.zip"
phase: 1
score: 0.8771
total_correct: 2163
total_samples: 2466
errors: 303
architecture: "HCCR-SENet (Custom 262k params)"
status: "OFFICIAL_CODABENCH"
tags:
  - submission
  - phase1
---

# Sub 4: submission_phase1_hccr_senet.zip

> [!INFO] Ringkasan Submisi Resmi CodaBench
> - **CodaBench Submission ID**: `996076`
> - **Timestamp Server (GMT+7)**: `2026-10-17 23:37`
> - **Nomor Submisi**: `Sub 4` (`Fase 1`)
> - **Arsitektur / Metode**: `HCCR-SENet (Custom 262k params)`
> - **Skor Leaderboard**: `==0.8771==` (2163 / 2466 benar, 303 kesalahan)
> - **Status Resmi**: `OFFICIAL_CODABENCH`

## 1. Detail Kinerja & Dekomposisi Metrik

| Metrik Evaluasi | Nilai Capaian | Catatan Teknis |
|:---|:---:|:---|
| **Akurasi Keseluruhan** | **`0.8771`** | 2163 dari 2466 sampel |
| **Jumlah Kesalahan** | **`303` sampel** | Evaluasi resmi CodaBench |
| **Chinese MNIST (1.500 citra)** | `0.9967` | Evaluasi sub-domain formal |
| **Gaya Baru Micro (966 citra)** | `0.6915` | Evaluasi sub-domain kaligrafi |
| **Gaya Baru Macro (`acc_target`)** | `0.6661` | Bobot seragam 13 kelas aktif |

## 2. Inovasi & Signifikansi Metodologis
- **Metodologi Utama**: Ablation study on under-capacity CNN (0.8771)
- **Korelasi Terhadap Garis Waktu**: Bagian integral dari evolusi riset yang didokumentasikan di [[17_MASTER_DOKUMENTASI_KESELURUHAN_DAN_RIWAYAT_VERSI|Master Dokumentasi]].
- **Visualisasi Alur**: Dapat dilihat pada peta koneksi [[WORKFLOW_LINEAGE.canvas]].

---
*Tercatat dalam basis data resmi [[OBSIDIAN_BASE_SUBMISSIONS.base]] dan indeks utama [[INDEX]].*
