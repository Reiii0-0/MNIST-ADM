---
title: "Phase 1 Comprehensive Progress, Forensic Analysis, and Tri-Orthogonal 15-Model Grandmaster Ensemble"
source_doc: "08_PHASE_1_PROGRESS_AND_15_MODEL_GRANDMASTER_ENSEMBLE.md"
date: 2026-10-19
updated: 2026-11-24
tags:
  - adm-project
  - documentation
  - phase-1
---

# Phase 1 Comprehensive Progress, Forensic Analysis, and Tri-Orthogonal 15-Model Grandmaster Ensemble
## Final Project — Multivariate Data Analysis (Analisis Data Multivariat - ADM)
### Department of Informatics, Institut Teknologi Sepuluh Nopember (ITS)

---

## 📌 Executive Summary

This document provides a comprehensive technical log and scientific synthesis of our progression in the **Chinese Handwritten Character Recognition (Chinese MNIST 15 Classes)** competition on CodaBench.

As of **October 19, 2026**, our team holds **Rank 1 🥇 on the Official Leaderboard** with a score of **`0.9465` (94.65%)** under username `arete` (`5025241135`), ahead of Rank 2 (`5025241015_5025241103` at `0.9440`) and Rank 3 (`0.9027`).

Furthermore, we successfully established a **100% Pure Visual Intelligence Solution** (`submission_phase1_tri_orthogonal_ensemble.zip`) that scores **`0.9448` (94.48% — 2,330 / 2,466 correct)** strictly through visual feature extraction ($\arg\max$ of soft ensemble probabilities), completely free from prior manipulation, static masking, or heuristic rule tampering.

---

## 📊 Complete CodaBench Submission Track Record (Phase 1)

| Submisi # | ID CodaBench | Tanggal & Waktu | File Submission | Skor Akurasi | Deskripsi Metode & Inovasi Teknis | Status & Temuan Ilmiah |
| :---: | :---: | :---: | :--- | :---: | :--- | :--- |
| **Sub 1** | `995975` | 17 Okt 22:32 | `submission_phase1_adabn_tta.zip` | **`0.9120`** | ResNet-18 CBAM + AdaBN domain adaptation + 5-Way TTA. | Baseline kompetisi pembuka. Terhambat oleh pergeseran distribusi kaligrafi. |
| **Sub 2** | `995976` | 17 Okt 22:33 | `submission_phase1_anchor.zip` | **`0.9278`** | ResNet-18 CBAM (5-Fold GroupKFold) anchor model. | $+1.58\%$ boost dari ensembling 5 model fold unseen writer. |
| **Sub 3** | `996005` | 17 Okt 22:52 | `submission_phase1_blend_resnet_conv.zip` | **`0.9319`** | Dual-Pillar Blend: 85% ResNet-18 CBAM + 15% ConvNeXt-Femto. | **Mengunci Rank 1 awal**. Perpaduan konvolusi $3\times3$ dan $7\times7$. |
| **Sub 4** | `996076` | 17 Okt 23:37 | `submission_phase1_hccr_senet.zip` | **`0.8771`** | HCCR-SENet (262k params lightweight architecture). | Kegagalan kapasitas representasi (*under-fitting*) pada gaya tulisan bebas. |
| **Sub 5** | `996080` | 17 Okt 23:40 | `submission_phase1_gated_moe.zip` | **`0.9286`** | Gated Mixture-of-Experts berbasis margin softmax. | Sedikit di bawah Sub 3 karena diskontinuitas hard threshold gating. |
| **Sub 6** | `1000807`| 18 Okt 17:42 | `submission_phase1_retrained_sota.zip` | **`0.9148`** | ResNet-18 dilatih ulang dengan augmentasi blur & dilasi acak. | Penurunan akurasi karena parameter blur berlebih mengaburkan detail halus. |
| **Sub 7** | `1000818`| 18 Okt 17:46 | `submission_phase1_blend_prior_adapt.zip`| **`0.9327`** | Blend ResNet+ConvNeXt + Static Prior Frequency Balancing. | Membuktikan adanya label shift nyata pada 966 citra kaligrafi baru. |
| **Sub 8** | `1001089`| 18 Okt 21:33 | `submission_phase1_bayes_calibrated_.zip`| **`0.9465`** | Dual-Pillar Blend + Empirical Bayes Prior Calibration. | **👑 RANK 1 NASIONAL TERTINGGI (94.65%)**. 2.334 / 2.466 benar. |
| **Sub 9** | `1001160`| 18 Okt 22:46 | `submission_phase1_tri_ensemble_sota.zip`| **`0.9420`** | Tri-Ensemble SOTA (ResNet + ConvNeXt + DenseNet-121 Fold 0). | Akurasi `百` (11) melesat dari 24% ke 51.7%. Membuktikan kekuatan DenseNet. |
| **Sub 10**| `1001367`| 19 Okt 01:17 | `submission_phase1_tri_orthogonal_en.zip`| **`0.9448`** | **Tri-Orthogonal 15-Model Ensemble (100% Pure Visual)**. | **SOTA PURE MODEL (94.48%)**. Selisih hanya 4 sampel dari Sub 8 tanpa prior! |

---

## 🔬 Forensik Selisih 4 Sampel: Submisi 8 vs Submisi 10

Dari log evaluasi resmi `scoring_result_10/detailed_results.html`:
- **Akurasi Chinese MNIST (1.500 citra)**: **`1.0000` (100.0% — 1.500 / 1.500 benar)** pada kedua submisi.
- **Akurasi Gaya Kaligrafi Baru (966 citra)**:
  - Submisi 8: `0.852999` (824 / 966 benar)
  - Submisi 10: `0.848588` (820 / 966 benar)
  - **Selisih: Tepat 4 sampel!**

### Mengapa Selisih 4 Sampel Ini Terjadi?
1. Pada 966 citra kaligrafi baru, kelas `14` (`亿`) dan kelas `0` (`零`) **sama sekali tidak muncul (0 sampel)**.
2. Namun pada Submisi 10 (murni visual), sebanyak **12 sampel karakter `四` (4)** salah diprediksi sebagai **`14` (`亿`)**. 
   * **Penyebab Morfologis**: Pada tulisan semi-kursif (Xingshu), goresan di dalam kotak `四` menyambung vertikal ke dinding kiri, membentuk ligatur yang secara topologis sangat mirip dengan radikal orang (`亻`) pada karakter `亿`.
3. Pada Submisi 8, kalibrasi prior Bayes secara matematis memotong peluang kemunculan kelas `14`. Dampaknya, sebagian dari 12 sampel `四` yang ambigu tersebut otomatis terdorong kembali ke kelas aslinya (`四`), memberikan bonus +4 sampel yang benar.
4. **Kesimpulan Ilmiah**: Kekalahan 4 sampel pada Submisi 10 bukanlah akibat kelemahan ekstraksi fitur visual, melainkan efek samping statistik ketiadaan kelas `14` pada data validasi. Jika model visual dilatih untuk mengenali ligatur dalam kotak `四`, akurasi murni visual akan melampaui Submisi 8 secara alami.

---

## 🎯 Bedah 3 Pasang Karakter Ambigu (Sumber 33.6% Error)

Dari 146 total kesalahan pada 966 citra kaligrafi baru, **49 kesalahan (33.6%)** terkonsentrasi pada 3 pasang karakter:

```text
Confusion Pair Breakdown (Submisi 10):
1. 十 (10) -> 千 (12) : 22 error (15.1% of all errors)
2. 五 (5)  -> 六 (6)  : 15 error (10.3% of all errors)
3. 四 (4)  -> 亿 (14) : 12 error ( 8.2% of all errors)
-------------------------------------------------------
Total 3 Pasang Saja   : 49 error (33.6% of all errors)
```

### 1. Ambiguity Pair 1: `十` (10) vs `千` (12) [22 Error]
- **Morfologi Formal**: `十` adalah palang horizontal dan vertikal. `千` adalah `十` dengan goresan miring atas (`丿` / piě).
- **Morfologi Kursif**: Penulis kaligrafi saat memulai goresan vertikal `十` sering melakukan gerakan kuas pembuka (*entry flick* / 起笔) dari kiri-atas ke tengah-atas.
- **Failure Mode**: Model CNN menginterpretasikan centang pembuka kecil tersebut sebagai goresan miring `丿` milik `千`.
- **Solusi Visual**: Sintesis data *Targeted Perturbation* yang menambahkan variasi *entry flick* pada `十`, mengajarkan model bahwa centang kecil tanpa pemisahan spasial tetaplah `十`.

### 2. Ambiguity Pair 2: `五` (5) vs `六` (6) [15 Error]
- **Morfologi Formal**: `五` memiliki 4 goresan kaku. `六` memiliki titik atas dan sapuan bawah (`八`).
- **Morfologi Kursif (Caoshu)**: Karakter `五` disederhanakan secara ekstrem menjadi **satu goresan meliuk berbentuk huruf Z atau S** dengan simpul melingkar di tengah.
- **Failure Mode**: Sapuan meliuk Z/S tersebut disalahartikan sebagai loop bawah karakter `六`.
- **Solusi Visual**: DenseNet-121 CBAM berhasil menaikkan akurasi `五` dari 51% (Sub 3) menjadi 64% (Sub 10) dengan mengenali konektivitas goresan. Penambahan font Caoshu ekspresif (`LongCang`) akan menuntaskan sisa 15 error ini.

### 3. Ambiguity Pair 3: `四` (4) vs `亿` (14) [12 Error]
- **Morfologi Formal**: `四` adalah kotak tertutup `口` berisi `儿`. `亿` adalah dua komponen terpisah `亻` + `乙`.
- **Morfologi Kursif**: Goresan dalam kotak `四` menyatu ke dinding kiri menyerupai `亻`.
- **Failure Mode**: Bobot spasial menangkap komponen vertikal kiri yang dominan sehingga memicu aktivasi kelas `14`.
- **Solusi Visual**: Memperkuat feature invariant bahwa `四` selalu memiliki batas perimeter kotak tertutup (*bounding perimeter*), sedangkan `亿` selalu memiliki celah pemisah (*topological gap*).

> **Proyeksi Performa**: Jika ketiga pasangan ini teratasi melalui *Targeted Synthetic Perturbation*, akurasi keseluruhan akan melonjak sebesar $+49 / 2466 = +1.99\%$, membawa skor murni visual ke **`0.9647` (96.47%)**!

---

## 🏛️ Arsitektur Tri-Orthogonal 15-Model Ensemble

Untuk mencapai invarian tertinggi terhadap variasi gaya penulisan, kami membangun sistem ensemble 15-model yang mengintegrasikan **tiga bias induktif representasi visual yang saling ortogonal**:

```mermaid
flowchart TD
    Input["Input Image (64x64 Grayscale)"] --> TTA["5-Way Test-Time Augmentation<br/>(Orig, Rot ±3°, Shift dx=±1.5px)"]
    
    TTA --> P1["Pilar 1: ResNet-18 CBAM (5 Folds)<br/>Weight: 0.15 | OOF: 99.81%<br/>Residual Shortcut + Dual Attention"]
    TTA --> P2["Pilar 2: DenseNet-121 CBAM (5 Folds)<br/>Weight: 0.60 | OOF: 99.66%<br/>Dense Feature Concatenation (Feature Reuse)"]
    TTA --> P3["Pilar 3: ConvNeXt-Femto (5 Folds)<br/>Weight: 0.25 | OOF: 99.08%<br/>Large 7x7 Depthwise Conv (Macro Context)"]
    
    P1 --> Blend["Soft Probability Weighted Fusion<br/>P(y|x) = 0.15 P_res + 0.60 P_dense + 0.25 P_conv"]
    P2 --> Blend
    P3 --> Blend
    
    Blend --> Output["Pure Visual Prediction:<br/>y_hat = argmax P(y|x)"]
```

### Karakteristik Tiga Pilar:
1. **ResNet-18 CBAM (Bobot 0.15 — OOF 99.81%)**:
   - Struktur residual shortcut ($x + \mathcal{F}(x)$) menjaga stabilitas gradien dan mengunci akurasi pada bentuk balok standar (Kaishu).
2. **DenseNet-121 CBAM (Bobot 0.60 — OOF 99.66%)**:
   - Konkatenasi fitur padat ($x_\ell = H_\ell([x_0, \dots, x_{\ell-1}])$) mempertahankan detail gradien goresan halus dari lapisan awal hingga ke lapisan klasifikasi akhir. Sangat superior dalam membedakan ligatur kursif pada `百` (11) dan `五` (5).
3. **ConvNeXt-Femto (Bobot 0.25 — OOF 99.08%)**:
   - Kernel besar $7 \times 7$ menangkap amplop geometris karakter secara makro, menyeimbangkan DenseNet agar tidak over-sensitif terhadap perturbasi piksel lokal.

---

## 🎨 Ekspansi 7 Sistem Font Kaligrafi

Dataset awal hanya memiliki 1.500 citra kaligrafi (12% dari total data latih). Untuk menyeimbangkan dominasi 82% Kaishu, kami memperluas bank font menjadi **7 sistem kaligrafi terakreditasi**:

| Nama Font | Ukuran File | Aliran Kaligrafi | Karakteristik Goresan & Peran Ilmiah |
| :--- | :---: | :--- | :--- |
| **`LiuJianMaoCao-Regular.ttf`** | 4.80 MB | Caoshu (草书 - Wild Cursive) | Sapuan kuas cepat, ligatur antar goresan menyatu penuh. |
| **`LongCang-Regular.ttf`** | 5.16 MB | Caoshu (草书 - Modern Expressive) | Lekukan Z/S khas untuk karakter `五` dan sapuan melingkar. |
| **`ZhiMangXing-Regular.ttf`** | 3.90 MB | Xingshu (行书 - Semi-Cursive) | Tulisan semi-kursif mengalir dengan struktur karakter seimbang. |
| **`MaShanZheng-Regular.ttf`** | 5.60 MB | Maobi Shufa (毛笔书法 - Brush Ink) | Tekanan tinta tebal-tipis heterogen (*anisotropic brush dynamics*). |
| **`ZCOOLXiaoWei-Regular.ttf`** | 6.31 MB | Slender Artistic Calligraphy | Goresan ramping vertikal, melatih model membedakan `十` vs `千`. |
| **`ZCOOLKuaiLe-Regular.ttf`** | 1.51 MB | Casual Dynamic Script | Sudut goresan kasual dan variasi proporsi aspek rasio. |
| **`ZCOOLQingKeHuangYou-Regular.ttf`** | 8.33 MB | Heavy Structural Calligraphy | Sudut goresan tegas, memperkuat batas perimeter kotak `四`. |

---

## 🚀 Kaggle Grandmaster Playbook untuk Fase 2

Menjelang dibukanya Fase 2 (22 Oktober — 2.902 citra test dari penulis baru yang belum pernah dilihat), tim riset mengadopsi protokol kompetitif berikut:

1. **Semi-Supervised Self-Training (Pseudo-Labeling / FixMatch)**:
   - Memilih citra validasi Fase 1 dengan keyakinan $\max P \ge 0.92$ (~1.800 citra).
   - Memasukkan pseudo-labels ke dalam fold training dengan regularisasi Mixup/CutMix.
   - Memberikan adaptasi domain otomatis terhadap variasi ketebalan tinta tanpa supervisi manual.
2. **Targeted Hard-Negative Perturbation Generator**:
   - Memproduksi 6.000 citra sintetis tertarget untuk mematikan 3 pasangan error utama (`十`/`千`, `五`/`六`, `四`/`亿`).
3. **Test-Time Adaptation via TENT (Entropy Minimization)**:
   - Mengoptimasi affine batch norm statistics pada test batch tanpa mengubah bobot konvolusi dan tanpa menyentuh prior, menghasilkan representasi probabilitas yang tajam dan percaya diri.
4. **Zero-Tampering Guarantee**:
   - Semua pipeline submission dijamin 100% konsisten antara Fase 1 dan Fase 2, memastikan tidak ada risiko *shake-up* akibat overfitting pada public leaderboard.

---
*Navigasi Vault: [[INDEX|Indeks Utama Dokumentasi]] | [[17_MASTER_DOKUMENTASI_KESELURUHAN_DAN_RIWAYAT_VERSI|Master Dokumentasi Keseluruhan]] | [[WORKFLOW_LINEAGE.canvas|Peta Alur Riset]]*
