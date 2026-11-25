---
title: "STUDI ABLASI ARSITEKTUR: ResNet18-CBAM vs ConvNeXt-Femto"
source_doc: "03_ARCHITECTURAL_COMPARISON_RESNET_VS_CONVNEXT.md"
date: 2026-10-13
updated: 2026-11-24
tags:
  - adm-project
  - documentation
  - pra-fase
---

# STUDI ABLASI ARSITEKTUR: ResNet18-CBAM vs ConvNeXt-Femto
## Evaluasi Performa 5-Fold, Perbandingan Mekanisme Kerja, dan Uji Ketahanan Komparatif (Stress-Testing)

**Mata Kuliah:** Analisis Data Multivariat (ADM)  
**Status Evaluasi:** SELESAI & TERDOKUMENTASI 100%  
**Objek Komparasi:** 
1. **ResNet18-CBAM:** Arsitektur residual dengan Channel & Spatial Attention (CBAM) + MixStyle Domain Generalization.
2. **ConvNeXt-Femto:** Arsitektur modern pure-convolutional (Liu et al., CVPR 2022) berbasis $7 \times 7$ Depthwise Separable Convolutions + LayerNorm2d.

---

## 1. Perbandingan Performa 5-Fold Out-of-Fold (OOF)

Pelatihan kedua model dilakukan pada Cloud GPU Kaggle (Tesla T4) dengan protokol identik: 5-Fold StratifiedGroupKFold (70 unseen writers), 15 epochs/fold, AdamW optimizer, Cosine Annealing, dan Sub-Center ArcFace Head ($K=2$).

| Metrik Evaluasi | ResNet18-CBAM | ConvNeXt-Femto | Delta Keunggulan |
| :--- | :---: | :---: | :---: |
| **Fold 0 Accuracy** | **99,86%** (3 error) | 99,57% (9 error) | ResNet +0,29% |
| **Fold 1 Accuracy** | **99,76%** (5 error) | 99,71% (6 error) | ResNet +0,05% |
| **Fold 2 Accuracy** | **99,86%** (3 error) | 99,67% (7 error) | ResNet +0,19% |
| **Fold 3 Accuracy** | **99,62%** (8 error) | 99,10% (19 error) | ResNet +0,52% |
| **Fold 4 Accuracy** | **99,86%** (3 error) | 99,71% (6 error) | ResNet +0,15% |
| **Overall OOF Accuracy** | **99,790%** (22 error) | **99,552%** (47 error) | **ResNet +0,238%** (25 error lebih sedikit) |
| **Macro F1-Score** | **0,9979** | **0,9955** | ResNet +0,0024 |
| **Total Parameter** | 11,27 Juta | 4,32 Juta | ConvNeXt lebih ringkas |

---

## 2. Uji Ketahanan Komparatif (Head-to-Head Stress Test)

Dievaluasi pada 275 sampel terstratifikasi mencakup seluruh 15 kelas pada 9 skenario degradasi morfologis dan spasial:

| Skenario Pengujian | ResNet18 Ens (%) | ResNet + TTA (%) | ConvNeXt Ens (%) | ConvNeXt + TTA (%) | Selisih (Delta Ensemble) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Clean Baseline** | 99,64% | 99,64% | 99,64% | 99,64% | $0,00\%$ (Imbang) |
| **Mild Blur ($k=3, \sigma=0,8$)** | 99,64% | 99,64% | 99,64% | 99,64% | $0,00\%$ (Imbang) |
| **Heavy Blur ($k=7, \sigma=1,8$)** | 99,64% | 99,64% | 99,64% | 99,64% | $0,00\%$ (Imbang) |
| **Stroke Dilation ($2 \times 2$)** | 99,64% | 99,64% | 99,64% | 99,64% | $0,00\%$ (Imbang) |
| **Heavy Dilation ($3 \times 3$)** | 99,64% | 99,64% | 99,64% | 99,64% | $0,00\%$ (Imbang) |
| **Stroke Erosion ($2 \times 2$)** | **100,00%** | **100,00%** | 99,64% | 99,64% | **ResNet +0,36%** |
| **Committee Compound (Dilate+Blur)** | 99,64% | 99,64% | 99,64% | 99,64% | $0,00\%$ (Imbang) |
| **Additive Noise ($\mathcal{N}(0, 15)$)** | **100,00%** | **100,00%** | 99,64% | 99,64% | **ResNet +0,36%** |
| **Salt & Pepper Ink Specks** | **97,09%** | **96,36%** | 93,09% | 92,73% | **ResNet +4,00%** |

---

## 3. Analisis Mekanisme Kerja & Mengapa ResNet18-CBAM Lebih Unggul

### 3.1 Perbedaan Arsitektur Mendasar
1. **Receptive Field & Kernel Konvolusi:**
   - *ResNet18:* Menggunakan filter konvolusi $3 \times 3$ standar yang sangat terfokus.
   - *ConvNeXt:* Menggunakan filter *depthwise* besar $7 \times 7$ yang meniru mekanisme *local self-attention* pada Vision Transformer (Swin-T).
2. **Normalisasi & Statistik Kanal:**
   - *ResNet18:* Menggunakan **BatchNorm2d** yang dinormalisasi lintas-batch.
   - *ConvNeXt:* Menggunakan **LayerNorm2d** yang menormalisasi antarkanal per sampel spasial secara independen dari batch.
3. **Mekanisme Atensi Spasial:**
   - *ResNet18-CBAM:* Memiliki modul atensi eksplisit (CBAM) yang menghitung *Channel Attention* (relasi antarkanal) dan *Spatial Attention* (fokus pada titik simpul/sudut goresan).
   - *ConvNeXt:* Mengandalkan *receptive field* pasif tanpa modul atensi eksplisit.
4. **Generalisasi Domain Penulis:**
   - *ResNet18:* Dilengkapi **MixStyle (Zhou et al., ICLR 2021)** yang mencampur $\mu$ dan $\sigma$ antarsampel penulis saat pelatihan.

### 3.2 Mengapa ResNet18-CBAM Lebih Unggul pada Karakter Hanzi 64x64?
1. **Resolusi Spasial Mikro ($64 \times 64$):**
   - Pada citra $64 \times 64$, goresan karakter Hanzi memiliki ketebalan hanya 1–3 piksel, dan jarak antargoresan paralel (seperti pada `三`, `二`, atau celah kotak `四` dan `百`) sangat rapat.
   - Kernel besar $7 \times 7$ pada ConvNeXt cenderung membaurkan (*bleed/blur*) garis-garis sejajar yang rapat pada layer awal. Sebaliknya, kernel $3 \times 3$ ResNet mempertahankan detail kontur tepi mikro secara tajam.
2. **Peran Kritis MixStyle pada Penulis Baru (*Unseen Writers*):**
   - Gaya penulis (*handwriting style*) tercermin secara dominan dalam rata-rata intensitas dan varians aktivasi fitur. MixStyle pada ResNet secara aktif melatih jaringan agar kebal terhadap gaya individual penulis, menghasilkan generalisasi superior pada Fold 3 (penulis dengan gaya paling kursif).
3. **Ketahanan Terhadap Ink Specks (+4.00% pada Derau Tinta):**
   - Modul Spatial Attention $7 \times 7$ pada CBAM mampu menyaring bintik tinta acak di luar kontur utama karakter, sementara LayerNorm pada ConvNeXt memperkuat noise specks karena menormalkan seluruh kanal secara lokal.

---

## 4. Keputusan Strategis untuk Submisi CodaBench & Laporan Kuliah
1. **Model Utama yang Disubmit:** **ResNet18-CBAM + 5-Way TTA + AdaBN** (memiliki akurasi tertinggi $99,79\% \to 99,91\%$ dan ketahanan derau terbaik).
2. **Laporan Kuliah ADM:** Menyajikan perbandingan ResNet18-CBAM vs ConvNeXt-Femto sebagai **Bab Studi Ablasi Arsitektur SOTA (Deep Learning Benchmark)** yang membuktikan eksplorasi mendalam berbagai famili model modern.

---
*Navigasi Vault: [[INDEX|Indeks Utama Dokumentasi]] | [[17_MASTER_DOKUMENTASI_KESELURUHAN_DAN_RIWAYAT_VERSI|Master Dokumentasi Keseluruhan]] | [[WORKFLOW_LINEAGE.canvas|Peta Alur Riset]]*
