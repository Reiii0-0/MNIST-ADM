---
title: "DOKUMENTASI PROGRESS: 00 - PRA-FASE (PERSIAPAN & FONDASI)"
source_doc: "00_PRA_FASE.md"
date: 2026-11-24
updated: 2026-11-24
tags:
  - adm-project
  - documentation
  - pra-fase
  - multivariate-statistics
---

# DOKUMENTASI PROGRESS: 00 - PRA-FASE (PERSIAPAN & FONDASI)
**Mata Kuliah:** Analisis Data Multivariat (ADM)  
**Kompetisi:** CodaBench - Klasifikasi Karakter Aksara Mandarin (Chinese MNIST Variant)  
**Platform Eksekusi:** Kaggle CLI & Local Workstation  
**Status Fase:** SELESAI (Menunggu Rilis Fase 1: 15 Oktober)  
**Terakhir Diperbarui:** 11 Oktober 2026  

---

## 1. Ringkasan Eksekutif Pra-Fase

Pada tahap Pra-Fase ini, fokus utama adalah membedah karakteristik dataset secara matematis, menutup celah data latih (*critical data loophole*), merancang skema validasi yang kebal terhadap kebocoran gaya penulis (*zero-leakage validation*), dan melakukan pembuktian konsep (*proof-of-concept*) untuk metode Analisis Data Multivariat klasik serta SOTA Deep Learning.

---

## 2. Temuan Kritis (*Critical Discoveries*)

1. **Absennya Bentuk Tradisional di Data Latih:**
   - Di `train_labels.csv`, seluruh 700 citra Label 13 adalah bentuk sederhana `万` (3 goresan), dan seluruh 700 citra Label 14 adalah bentuk sederhana `亿` (3 goresan).
   - Penyelenggara mencatat di `classes.csv` bahwa bentuk tradisional `萬` (13 goresan) dan `億` (15 goresan) memiliki label yang sama (13 dan 14) dan dapat muncul di data uji.
   - **Tindakan:** Berhasil menyintesis 700 citra tulisan tradisional realistis menggunakan font sistem CJK + augmentasi deformasi elastis (*elastic distortion*).

2. **Pergeseran Domain Penulis (*Writer Shift*):**
   - Data latih ditulis oleh 70 orang penulis (`writer_id`), masing-masing menulis 15 karakter sebanyak 10 kali ($70 \times 150 = 10.500$ gambar).
   - Penulis di data validasi dan data uji adalah orang baru.
   - **Tindakan:** Membagi dataset menggunakan **`StratifiedGroupKFold(n_splits=5)`** berdasarkan kolom `writer_id`.

---

## 3. Artefak & Modul yang Berhasil Dibangun

| Modul / Berkas | Fungsi Utama | Status |
| :--- | :--- | :--- |
| [`src/split_data.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/split_data.py) | Membagi 10.500 citra ke 5 fold validasi bebas kebocoran penulis. Menghasilkan [`splits_kfold.csv`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/data/metadata/splits_kfold.csv). | Selesai & Terverifikasi |
| [`src/generate_traditional_data.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/generate_traditional_data.py) | Membuat 700 citra sintetik `萬` dan `億` di [`data/train_synthetic/`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/data/train_synthetic). | Selesai & Terverifikasi |
| [`src/multivariate_analysis.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/multivariate_analysis.py) | Ekstraksi PCA, visualisasi Eigen-characters, uji hipotesis MANOVA, dan benchmark klasifikasi (LDA, QDA, LR). | Selesai & Terverifikasi |
| [`src/models.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/models.py) | Arsitektur ResNet18-CBAM, SubCenterArcFace ($K=2$), MixStyle, dan CompactCNN. | Selesai & Terverifikasi |
| [`src/train_cnn.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/train_cnn.py) | Pipeline pelatihan neural network dengan augmentasi elastis dan loss campuran. | Selesai & Terverifikasi |
| [`src/prepare_submission.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/prepare_submission.py) | Generator submission CodaBench dengan 5-way TTA dan validasi pre-flight. | Selesai & Terverifikasi |

---

## 4. Hasil Empiris Eksperimen Pra-Fase

### A. Metode Multivariat Klasik (Bahan Laporan Akademik ADM)
*Evaluasi menggunakan 5-Fold Stratified Group K-Fold Out-of-Fold (OOF) pada 70 penulis:*

1. **Reduksi Dimensi PCA (Eigen-characters):**
   - Rata-rata nilai eigen (Kaiser Threshold): $0,087497$.
   - Kriteria Kaiser mempertahankan **275 komponen**.
   - Proporsi variansi kumulatif: $80\% \to 82$ PC, $85\% \to 109$ PC, $90\% \to 154$ PC, $95\% \to 252$ PC.
   - Grafik tersimpan di: [`outputs/scree_plot.png`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/outputs/scree_plot.png) dan [`outputs/eigen_characters.png`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/outputs/eigen_characters.png).
2. **Uji Hipotesis MANOVA (Top 20 PC):**
   - **Wilks' Lambda ($\Lambda$):** $8,89 \times 10^{-5}$ ($p < 10^{-15}$, pemisahan antarkelas sangat masif).
   - **Pillai's Trace ($V$):** $5,8418$ ($p < 10^{-15}$).
   - **Partial Eta Squared ($\eta_p^2$):** $0,4864$ (efek ukuran sangat besar).
   - Diagnostik Normalitas: [`outputs/mahalanobis_qq_plot.png`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/outputs/mahalanobis_qq_plot.png).
3. **Akurasi Model Multivariat:**
   - Two-Stage PCA (100) + Linear Discriminant Analysis (LDA): **85,13%**
   - Two-Stage PCA (100) + Logistic Regression: **91,08%**
   - Two-Stage PCA (40) + Regularized QDA: **96,09%** *(Keberhasilan QDA mengonfirmasi bahwa kovarians antar-karakter tulisan tangan bersifat heterogen)*.
   - Proyeksi diskriminan 2D: [`outputs/lda_canonical_projection.png`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/outputs/lda_canonical_projection.png).

### B. Uji Coba Deep Learning SOTA (Lokal GPU RTX 4060)
*Model: ResNet18-CBAM + MixStyle + Sub-center ArcFace pada Fold 0:*
```text
Epoch 01/05 | Train Loss: 3.0036 | Train Acc: 61.76% | Val Acc (Unseen Writers): 91.43%
Epoch 02/05 | Train Loss: 0.7029 | Train Acc: 99.02% | Val Acc (Unseen Writers): 98.95%
Epoch 03/05 | Train Loss: 0.5671 | Train Acc: 99.56% | Val Acc (Unseen Writers): 99.14%
Epoch 04/05 | Train Loss: 0.5203 | Train Acc: 99.80% | Val Acc (Unseen Writers): 99.67%
Epoch 05/05 | Train Loss: 0.5068 | Train Acc: 99.85% | Val Acc (Unseen Writers): 99.67%
```
* **Akurasi Validasi:** **99,67%** pada 14 penulis baru (2.100 citra uji, hanya 7 sampel salah tebak).
* Bobot model tersimpan di: [`models/resnet18_cbam_fold0.pt`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/models/resnet18_cbam_fold0.pt).

---

## 5. Hasil Pelatihan Skala Penuh di Kaggle Cloud (STATUS: 100% SUKSES)

Pelatihan skala penuh (Full 5-Fold Stratified Group K-Fold, 15 Epochs per Fold = 75 Epochs total) telah **selesai 100% dengan status `COMPLETE`** di Cloud GPU Kaggle (Tesla T4 16GB VRAM):

* **Kaggle Kernel:** [`reiiihel/mnist-adm`](https://www.kaggle.com/code/reiiihel/mnist-adm) (Durasi: ~49 menit)
* **Log Eksekusi:** Terverifikasi bersih, lolos pengecekan integritas tensor tanpa *warning* maupun *error*.
* **Hasil Akurasi Per Fold pada Penulis Baru (Unseen Real Writers):**
  - **Fold 0:** **$99,86\%$** (2.097 benar / 2.100 gambar) $\to$ [`resnet18_cbam_fold0.pt`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/models/kaggle_outputs/models/resnet18_cbam_fold0.pt)
  - **Fold 1:** **$99,76\%$** (2.095 benar / 2.100 gambar) $\to$ [`resnet18_cbam_fold1.pt`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/models/kaggle_outputs/models/resnet18_cbam_fold1.pt)
  - **Fold 2:** **$99,86\%$** (2.097 benar / 2.100 gambar) $\to$ [`resnet18_cbam_fold2.pt`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/models/kaggle_outputs/models/resnet18_cbam_fold2.pt)
  - **Fold 3:** **$99,62\%$** (2.092 benar / 2.100 gambar) $\to$ [`resnet18_cbam_fold3.pt`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/models/kaggle_outputs/models/resnet18_cbam_fold3.pt)
  - **Fold 4:** **$99,86\%$** (2.097 benar / 2.100 gambar) $\to$ [`resnet18_cbam_fold4.pt`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/models/kaggle_outputs/models/resnet18_cbam_fold4.pt)

* **Hasil Evaluasi Global Out-of-Fold (OOF) pada Seluruh 10.500 Gambar (70 Penulis Baru):**
  $$\text{Overall OOF Accuracy} = \mathbf{99,79\%} \quad (\text{Macro F1-Score} = \mathbf{0,9979})$$
  $$\text{Mean Fold Accuracy} = \mathbf{99,79\% \pm 0,09\%}$$
  *(Hanya 22 kesalahan dari total 10.500 gambar di seluruh dataset dosen!)*


* **Artefak yang Telah Diunduh ke Lokal:**
  - 5 Berkas Bobot Model: [`FinalProject/models/kaggle_outputs/models/`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/models/kaggle_outputs/models/)
  - Matriks Probabilitas OOF: [`FinalProject/models/kaggle_outputs/oof_probabilities.npy`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/models/kaggle_outputs/oof_probabilities.npy)
  - Laporan Metrik JSON: [`FinalProject/models/kaggle_outputs/oof_metrics.json`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/models/kaggle_outputs/oof_metrics.json)
  - Log Lengkap Eksekusi: [`FinalProject/models/kaggle_outputs/mnist-adm.log`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/models/kaggle_outputs/mnist-adm.log)

---

## 6. Laporan Audit & Meta-Analisis Pelengkap Pra-Fase

Seluruh audit mendalam pasca-pelatihan didokumentasikan dalam laporan spesialis berikut:
1. [**01_ERROR_ANALYSIS_CONFUSION_MATRIX_REPORT.md**](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/dokumentasi/01_ERROR_ANALYSIS_CONFUSION_MATRIX_REPORT.md): Bedah detail 22 sampel salah, morfologi goresan Hanzi, dan pemulihan 13 error via 5-Fold Ensemble + TTA (Akurasi 99,914%).
2. [**LITERATURE_BENCHMARK_ACADEMIC_RIGOR_EVALUATION.md**](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/dokumentasi/LITERATURE_BENCHMARK_ACADEMIC_RIGOR_EVALUATION.md): Perbandingan komparatif dengan CASIA-HWDB, isolasi kebocoran penulis, serta uji asumsi MANOVA & QDA vs LDA.
3. [**AUDIT_KERENTANAN_FASE1_FASE2.md**](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/dokumentasi/AUDIT_KERENTANAN_FASE1_FASE2.md): Audit ketahanan model terhadap catatan panitia (blur, ketebalan, class imbalance), verifikasi AdaBN, dan checklist Hari-H 15 Oktober.

---
*Pra-Fase telah resmi tuntas 100% dengan hasil SOTA sempurna. Tim berada dalam kesiapan penuh menyongsong pembukaan Fase 1 (Validasi) pada 15 Oktober.*


---
*Navigasi Vault: [[INDEX|Indeks Utama Dokumentasi]] | [[17_MASTER_DOKUMENTASI_KESELURUHAN_DAN_RIWAYAT_VERSI|Master Dokumentasi Keseluruhan]] | [[WORKFLOW_LINEAGE.canvas|Peta Alur Riset]]*
