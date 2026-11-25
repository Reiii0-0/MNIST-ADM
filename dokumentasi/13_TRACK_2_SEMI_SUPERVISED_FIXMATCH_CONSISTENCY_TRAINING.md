---
title: "TRACK 2: Semi-Supervised Consistency Self-Training (FixMatch Paradigm)"
source_doc: "13_TRACK_2_SEMI_SUPERVISED_FIXMATCH_CONSISTENCY_TRAINING.md"
date: 2026-10-20
updated: 2026-11-24
tags:
  - adm-project
  - documentation
  - phase-1
---

# TRACK 2: Semi-Supervised Consistency Self-Training (FixMatch Paradigm)

**Tim Riset**: Arete (NRP: 5025241135 / 5025241148)  
**Mata Kuliah**: Analisis Data Multivariat (ADM) — Teknik Informatika ITS  
**Kompetisi**: CodaBench Chinese MNIST Challenge  
**Tanggal**: 20 Oktober 2026  
**Status Eksekusi**: **RUNNING on Kaggle Cloud GPU (Tesla T4 / P100)**  
**Kernel Kaggle**: [`reiiihel/mnist-adm-fixmatch`](https://www.kaggle.com/code/reiiihel/mnist-adm-fixmatch)

---

## 1. Latar Belakang & Motivasi Teoretis

Pada evaluasi forensik terhadap 2.466 citra validasi Fase 1 CodaBench Chinese MNIST:
1. **Chinese MNIST Standar (1.500 citra)**: Model kita telah mencapai akurasi **100.00% (1.500/1.500)** tanpa satu pun kesalahan.
2. **Target Kaligrafi Baru (966 citra)**: Menghasilkan 130 kesalahan pada submisi champion (Sub 11), terutama pada klaster morfologi Caoshu/Xingshu (misal goresan meliuk pada `五`, dinding sambung pada `四`, dan kait atas pada `十`).
3. **Keterbatasan Pure Supervised Learning**: Model yang hanya dilatih pada dataset berlabel rentan terhadap *distribution shift* ketika dihadapkan pada gaya kaligrafi liar dari penulis target.
4. **Prinsip FixMatch (Sohn et al., NeurIPS 2020)**:
   FixMatch menggabungkan dua pilar semi-supervised learning modern:
   - *Pseudo-labeling*: Memanfaatkan prediksi berkonfidensi sangat tinggi sebagai label pengawas semu.
   - *Consistency Regularization*: Memaksa jaringan menghasilkan representasi dan prediksi probabilitas yang invarian terhadap variasi perturbasi deformasi bentuk (*weak augmentation* vs *strong augmentation*).

---

## 2. Formulasi Matematis FixMatch

### A. Labeled Loss ($\mathcal{L}_{\text{supervised}}$)
Dataset berlabel $\mathcal{D}_L = \mathcal{D}_{\text{train}} \cup \mathcal{D}_{\text{synth}} \cup \mathcal{D}_{\text{pseudo}}$ memuat:
- Data latih asli 5-Fold Stratified Group K-Fold (~8.400 sampel per fold).
- Data sintetis tradisional (`萬`, `億`) (700 sampel).
- Data sintetis kaligrafi v2 dari 7 Google Fonts (6.000 sampel).
- **Dataset Pseudo-Label Konsensus**: Sebanyak 2.344 citra validasi yang disepakati secara mutlak oleh 4 model champion (Sub 11, Sub 14, Sub 10, Sub 8) dengan konfidensi $>98\%$ dan presisi empiris $>99.8\%$.

Fungsi rugi supervised menggabungkan Cross-Entropy linear head dan angular margin Sub-Center ArcFace ($K=2$ sub-center):
$$\mathcal{L}_{\text{supervised}} = 0.5 \cdot \mathcal{L}_{\text{CE}}(\hat{\mathbf{y}}_{\text{lin}}, y) + 0.5 \cdot \mathcal{L}_{\text{CE}}(\hat{\mathbf{y}}_{\text{arc}}, y)$$

### B. Unlabeled Consistency Regularization Loss ($\mathcal{L}_{\text{consistency}}$)
Untuk setiap sampel pada kumpulan data validasi tanpa label $u \in \mathcal{D}_U$:
1. **Weak View Generator** ($u_w$):
   $$\alpha_w \sim \mathcal{U}(-2^\circ, 2^\circ), \quad \Delta x, \Delta y \sim \mathcal{U}(-1\text{ px}, 1\text{ px})$$
   Menghasilkan distribusi prediksi target tanpa gradient (*detached*):
   $$p_w = \text{Softmax}\left(\frac{f(u_w)}{T}\right), \quad \hat{y} = \arg\max(p_w), \quad q = \max(p_w)$$
2. **Strong View Generator** ($u_s$):
   - *Anisotropic Brush Pressure Erosion* ($p = 0.30$) pada kernel diagonal $45^\circ / 135^\circ$ untuk menyimulasikan penipisan tinta kuas kaligrafi.
   - *Affine Stretch/Shear* ($p = 0.25$) dengan faktor peregangan $0.90\times - 1.10\times$ dan shear $\pm 0.15$.
   - Rotasi $\pm 4^\circ$ dan translasi $\pm 1.5$ px.
3. **Consistency Objective**:
   $$\mathcal{L}_{\text{consistency}} = \mathbb{I}(q \ge \tau) \cdot \mathcal{H}\left(\hat{y}, \text{Softmax}(f(u_s))\right)$$
   dengan ambang batas konfidensi $\tau = 0.90$.

### C. Total Joint Objective
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{supervised}} + 0.5 \cdot \mathcal{L}_{\text{consistency}}$$

---

## 3. Protokol Anti-Catastrophic Forgetting

Untuk memastikan model tidak mengalami kerusakan pada bobot representasi yang telah stabil:
1. **Warm-Start Initialization**: Backbone diinisialisasi dari model pristine ResNet-18 CBAM v2.
2. **Ultra-Low Learning Rate**: Fine-tuning dilakukan dengan $lr = 1\times 10^{-5}$ menggunakan Cosine Annealing scheduler hingga $\eta_{\min} = 1\times 10^{-7}$.
3. **Strict Epoch Capping**: Latihan dibatasi ketat pada 4 epoch saja (rentang 3–5 epoch) per fold.

---

## 4. Evaluasi 9-Way Multi-Scale Morphological TTA

Setelah fine-tuning selesai pada seluruh 5 fold, inferensi dilakukan dengan 9 varian morfologis untuk setiap citra:
1. Citra Asli (Original)
2. Rotasi $+3^\circ$
3. Rotasi $-3^\circ$
4. Translasi $+1.5$ px
5. Translasi $-1.5$ px
6. Peregangan Aspek Rasio X ($1.05\times$)
7. Peregangan Aspek Rasio Y ($1.05\times$)
8. Erosi / Penipisan Goresan (Min-pooling $2\times 2$)
9. Dilasi / Penebalan Goresan (Max-pooling $2\times 2$)

Setiap varian dievaluasi melalui Dual-Head ($0.5 \times \text{Linear} + 0.5 \times \text{SubCenterArcFace}$), lalu dirata-ratakan pada ke-9 varian dan ke-5 fold.

---

## 5. Artefak Keluaran & Integrasi Submisi

Artefak yang dihasilkan langsung di direktori kerja Kaggle (`/kaggle/working/`):
1. `probabilities_fixmatch_5fold.npy`: Matriks probabilitas 5-Fold ukuran $(2.466, 15)$.
2. `submission_fixmatch.csv`: File prediksi CSV dengan kolom `id` dan `label`.
3. `sub_fixmatch.zip`: Berkas arsip zip siap submit CodaBench yang memuat `predictions.csv`.
4. Checkpoint Model: `models/resnet18_cbam_fixmatch_fold{0..4}.pt`.

---
*Navigasi Vault: [[INDEX|Indeks Utama Dokumentasi]] | [[17_MASTER_DOKUMENTASI_KESELURUHAN_DAN_RIWAYAT_VERSI|Master Dokumentasi Keseluruhan]] | [[WORKFLOW_LINEAGE.canvas|Peta Alur Riset]]*
