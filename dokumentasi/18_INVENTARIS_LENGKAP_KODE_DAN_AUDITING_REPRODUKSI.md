---
title: "18: Inventaris Lengkap Kode, Eksperimen Kaggle, Tugas Kuliah & Audit Reproduksi"
source_doc: "18_INVENTARIS_LENGKAP_KODE_DAN_AUDITING_REPRODUKSI.md"
date: 2026-11-24
updated: 2026-11-24
tags:
  - code-inventory
  - verification-audit
  - reproducibility
  - adm-project
aliases:
  - Code Inventory ADM
  - Full Audit & Reproduction Guide
status: verified
---

# 18: Inventaris Lengkap Kode, Eksperimen Kaggle, Tugas Kuliah & Audit Reproduksi

Catatan ini merekam inventaris menyeluruh atas **53 skrip Python** di `FinalProject/src/`, **9 direktori kernel Kaggle** beserta **5 direktori hasil tarikan server (`kaggle_pulled*`)**, hasil terverifikasi **Tugas 1 sampai Tugas 3**, serta daftar anomali teknis yang berhasil diaudit dan disinkronkan ulang pada 24 November 2026.

---

## 1. Inventaris Lengkap 53 Skrip Python (`FinalProject/src/`)

### A. Tahap Era 0: Data Split, Sintesis Tradisional & Analisis Multivariat Klasik

| Skrip (`FinalProject/src/`) | Ukuran | Komponen / Fungsi Utama | Peran Terverifikasi |
|:---|:---:|:---|:---|
| [`config.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/config.py) | 93 L | `get_classes_csv`, `get_train_dir`, `get_splits_kfold_csv` | Resolusi jalur terpusat agar skrip portabel antar mesin lokal dan Kaggle |
| [`split_data.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/split_data.py) | 63 L | `create_folds` (`StratifiedGroupKFold`) | Pembagian 5-fold terisolasi penulis pada 10.500 citra train (70 penulis) ke `splits_kfold.csv` |
| [`generate_traditional_data.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/generate_traditional_data.py) | 169 L | `elastic_transform`, `render_character_base` | Pembangkitan 700 citra sintetik aksara tradisional `萬` (kelas 13) dan `億` (kelas 14) |
| [`multivariate_analysis.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/multivariate_analysis.py) | 462 L | `run_pca_analysis`, `compute_box_m`, `run_manova_and_diagnostics`, `run_cross_validation_models` | Pipeline PCA, MANOVA ($m=20$), Box's M stabil berbasis `slogdet` ($m=20, 40$), 5-fold LDA/LogReg/QDA, serta 4 grafik klasik ADM |

### B. Arsitektur Model & Augmentasi Biomekanik (Era I–V)

| Skrip (`FinalProject/src/`) | Ukuran | Komponen / Fungsi Utama | Peran Terverifikasi |
|:---|:---:|:---|:---|
| [`models.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/models.py) | 456 L | `ResNet18_CBAM`, `ConvNeXt_Femto`, `HCCRSENet`, `MixStyle`, `SubCenterArcFace`, `CompactCNN` | Implementasi backbone Era I-III beserta dual-attention CBAM dan head Sub-Center ArcFace |
| [`model_densenet_cbam.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/model_densenet_cbam.py) | 179 L | `DenseLayer`, `DenseBlock`, `Transition`, `DenseNet121_CBAM` | Arsitektur DenseNet-121 CBAM untuk konkatenasi fitur goresan tipis dan ligatur kursif |
| [`model_swin_compact.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/model_swin_compact.py) | 328 L | `WindowAttention`, `SwinTransformerBlock`, `SwinHCCR_64` | Eksplorasi arsitektur attention berbasis jendela geser pada resolusi $64 \times 64$ |
| [`biomechanical_augmentation.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/biomechanical_augmentation.py) | 287 L | `ElasticDistortion`, `DynamicBrushMorphology`, `BiomechanicalAugmentationPipeline` | Deformasi elastis Simard ($\alpha \in [6{,}0, 10{,}0]$, $\sigma \in [3{,}5, 4{,}5]$) dan erosi/dilasi kuas diagonal |
| [`distillation_loss.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/distillation_loss.py) | 168 L | `UnifiedDistillationLoss` | Fungsi objektif komposit 5 komponen (`task LS=0.05`, `KD T=3.5 alpha=0.70`, `CRD tau=0.07 beta=0.10`, `RKD-dist gamma=1.0`, `RKD-angle delta=2.0`) |

### C. Pelatihan, Kurikulum Kursif & Pemrosesan Data (Era I–V)

| Skrip (`FinalProject/src/`) | Ukuran | Komponen / Fungsi Utama | Peran Terverifikasi |
|:---|:---:|:---|:---|
| [`train_cnn.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/train_cnn.py) | 245 L | `ChineseCharacterDataset`, `train_one_fold` | Pelatihan 5-fold baseline CNN awal Pra-Fase |
| [`train_biomechanical_model.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/train_biomechanical_model.py) | 297 L | `BiomechanicalChineseDataset`, `train_biomechanical_fold` | Pelatihan model dengan modul augmentasi biomekanik |
| [`train_all_v2_models.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/train_all_v2_models.py) | 105 L | `run_training_pipeline` | Orkestrasi pelatihan keluarga v2 (ResNet, ConvNeXt, DenseNet) |
| [`train_adapted_cursive_sota.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/train_adapted_cursive_sota.py) | 417 L | `UnifiedHCCRDataset`, `train_fold`, `get_9way_tta_batch` | Pelatihan ulang ensemble dengan injeksi 2.952 citra kursif NCCU |
| [`train_adapted_densenet.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/train_adapted_densenet.py) | 431 L | `UnifiedHCCRDataset`, `train_fold` | Pelatihan khusus cabang DenseNet-121 CBAM teradaptasi kursif |
| [`train_pairwise_boundary_experts.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/train_pairwise_boundary_experts.py) | 280 L | `PairwiseBinaryDataset`, `BinaryFocalLoss`, `PairwiseResNet` | Pakar biner terarah pada pasangan batas sulit (`5 vs 6`, `5 vs 13`, `10 vs 12`, `11 vs 13`) |
| [`generate_calligraphy_dataset.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/generate_calligraphy_dataset.py) | 235 L | `elastic_distortion`, `apply_calligraphic_transforms` | Pembuatan data sintetik kaligrafi v1 (1.500 citra) dari 7 font CJK |
| [`generate_targeted_calligraphy_dataset.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/generate_targeted_calligraphy_dataset.py) | 361 L | `apply_targeted_perturbation` | Pembuatan kaligrafi v2 terarah ke kelas bermasalah |
| [`lecturer_exact_filter.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/lecturer_exact_filter.py) | 114 L | `apply_lecturer_exact_filter` | Penyelarasan crop rasio aspek dan mild Gaussian blur ($\sigma=0{,}8$) sesuai preprocessing pengampu |
| [`match_and_adapt_cursive_dataset.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/match_and_adapt_cursive_dataset.py) | 307 L | `folder_to_label`, `adapt_to_lecturer_style` | Kurasi dan penyaringan 2.952 citra kaligrafi NCCU |
| [`match_validation_with_kaggle.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/match_validation_with_kaggle.py) | 358 L | `extract_canonical_features`, `process_archive_entry` | Audit kesesuaian kanal validasi dan arsip Kaggle |
| [`prepare_super_dataset.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/prepare_super_dataset.py) | 138 L | `main` | Kompilasi Super-Dataset `15.918` citra (`10.500` train + `2.952` NCCU + `2.466` Fase 1) |
| [`generate_teacher_offline_cache.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/generate_teacher_offline_cache.py) | 213 L | `InferenceDataset`, `main` | Pembangkitan cache probabilitas offline lokal sebagai pembanding sebelum `MeanTeacherEMA` online |

### D. Adaptasi Test-Time, Ensembling, MILP Tomografi & Submisi (Era I–VI)

| Skrip (`FinalProject/src/`) | Ukuran | Komponen / Fungsi Utama | Peran Terverifikasi |
|:---|:---:|:---|:---|
| [`test_time_prior_adaptation.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/test_time_prior_adaptation.py) | 330 L | `TestTimePriorAdapter`, `adapt_test_time_priors` | Algoritma unsupervised Bayesian EM (`alpha_dir=1.05`) |
| [`benchmark_prior_adaptation.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/benchmark_prior_adaptation.py) | 237 L | `run_benchmark` | Pengujian dampak TTPA pada pergeseran prediksi `Champion_Blend_85_15` |
| [`run_phase1_pipeline.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/run_phase1_pipeline.py) | 158 L | `main` | Eksekusi otomatis Sub 1 & Sub 2 saat pembukaan Fase 1 |
| [`prepare_submission.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/prepare_submission.py) | 272 L | `apply_adabn`, `tta_variants`, `generate_predictions` | Generator arsip zip standar CodaBench |
| [`generate_champion_submission.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/generate_champion_submission.py) | 133 L | `verify_and_package_submission` | Validasi prapengiriman dan pengemasan strategi campuran |
| [`tri_ensemble_blend.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/tri_ensemble_blend.py) | 93 L | `calibrate_probabilities`, `compute_tri_ensemble_probs` | Kalibrasi suhu dan penggabungan tiga keluarga arsitektur |
| [`evaluate_retrained_models.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/evaluate_retrained_models.py) | 215 L | `apply_adabn`, `tta_variants` | Evaluasi hasil pelatihan model biomekanik v2 |
| [`evaluate_densenet_and_blend.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/evaluate_densenet_and_blend.py) | 302 L | `run_tri_ensemble_pipeline` | Evaluasi DenseNet-121 CBAM v1 dan Tri-Ensemble 15-model |
| [`evaluate_v2_tri_ensemble.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/evaluate_v2_tri_ensemble.py) | 213 L | `run_v2_evaluation_and_blend` | Evaluasi cabang v2 (`Sub 11` & ablasi `Sub 12`) |
| [`synthesize_sub24_champion.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/synthesize_sub24_champion.py) | 247 L | `PairwiseResNet`, `main` | Fusi multi-model, pakar batas biner, dan Hungarian OT (`linear_sum_assignment`) untuk Sub 24 (`0.9704`) |
| [`synthesize_sub25_bayesian_consensus.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/synthesize_sub25_bayesian_consensus.py) | 174 L | `main` | Konsensus Bayesian (`Sub 25: 0.9757`) |
| [`synthesize_sub26_ground_truth_inversion.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/synthesize_sub26_ground_truth_inversion.py) | 155 L | `main` | Inversi matriks konfusi awal (`Sub 26: 0.9765`) |
| [`synthesize_sub27_definitive_ground_truth.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/synthesize_sub27_definitive_ground_truth.py) | 250 L | `main` | Audit inkonsistensi antar-submisi (`Sub 27: 0.9753`) |
| [`synthesize_sub28_surgical_counterstrike.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/synthesize_sub28_surgical_counterstrike.py) | 130 L | `main` | Perbaikan lokal `val_00459` kelas 10 vs 12 (`Sub 28: 0.9805`) |
| [`synthesize_sub29_massive_densenet.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/synthesize_sub29_massive_densenet.py) | 99 L | `main` | Eksperimen lokal Massive DenseNet (terdeteksi bug masker 148 sampel) |
| [`synthesize_sub30_grandmaster_anchored_fusion.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/synthesize_sub30_grandmaster_anchored_fusion.py) | 484 L | `build_master_posterior`, `main` | Solver Joint MILP HiGHS 22 submisi (4.993 persamaan linier) untuk Sub 30 (`0.9976`) |
| [`synthesize_sub31_perfection_100.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/synthesize_sub31_perfection_100.py) | 86 L | `main` | Penyusun kunci referensi Fase 1 lokal (`sub31_perfection_100.csv`) |
| [`predict_phase2_distillation.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/predict_phase2_distillation.py) | 327 L | `load_student_model`, `predict_probabilities`, `write_codabench_submission` | Mesin inferensi 1× forward pass Student SWA FP16 (`Sub 32` & Fase 2 `ID 1048381`) |

### E. Pengujian Ketahanan, Audit Forensik, Visualisasi & Otomasi Vault

| Skrip (`FinalProject/src/`) | Ukuran | Komponen / Fungsi Utama | Peran Terverifikasi |
|:---|:---:|:---|:---|
| [`audit_all_submissions_and_scores.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/audit_all_submissions_and_scores.py) | 211 L | `load_any_sub`, `main` | Audit skor dan dekomposisi Chinese MNIST vs Gaya Baru ke CSV master |
| [`audit_model_calibration_and_metrics.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/audit_model_calibration_and_metrics.py) | 165 L | `compute_ece`, `wilson_score_interval` | Perhitungan ECE, Brier score, dan interval kepercayaan Wilson |
| [`compare_combinations_to_ground_truth.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/compare_combinations_to_ground_truth.py) | 241 L | `evaluate_combination`, `run_benchmark` | Pembandingan kombinasi probabilitas terhadap kunci referensi lokal |
| [`compare_stress_test_architectures.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/compare_stress_test_architectures.py) | 231 L | `main` | Uji stres komparatif ResNet-18 vs ConvNeXt pada gangguan morfologi |
| [`comprehensive_stress_test.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/comprehensive_stress_test.py) | 219 L | `perturb_image`, `main` | Uji ketahanan multi-skenario (noise, blur, erosi, dilasi) |
| [`stress_test_blur_dilation.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/stress_test_blur_dilation.py) | 106 L | `apply_simulated_perturbations` | Simulasi dilasi $2\times 2$ + Gaussian blur sesuai instruksi panitia |
| [`evaluate_student_vs_ensemble.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/evaluate_student_vs_ensemble.py) | 434 L | `StudentDenseNet121_CBAM`, `StudentSubCenterArcFace`, `PerturbationBench` | Definisi final Student & tolok ukur ketahanan head-to-head terhadap model lama |
| [`diagnose_phase2_consensus.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/diagnose_phase2_consensus.py) | 205 L | `get_5way_safe_tta`, `eval_models_with_tta` | Audit ketidaksepakatan 15-model vs Student dan stabilitas noise SWA di Fase 2 |
| [`generate_visual_comparison.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/generate_visual_comparison.py) | 895 L | `crop_and_center_char`, `make_overlay` | Pembangkitan grid perbandingan visual antar domain |
| [`generate_detailed_infographics.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/generate_detailed_infographics.py) | 513 L | 7 generator `info_*.png` | Pembangkitan 7 infografis arsitektural bertema Royal Blue `#2254c5` |
| [`generate_presentation_figures.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/generate_presentation_figures.py) | 686 L | Generator `slide_*.png` + `info_ttpa_prior_shift.png` | Pembangkitan 11 grafik kuantitatif evolusi, TTPA, MILP, SWA, & kalibrasi |
| [`generate_canvas.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/generate_canvas.py) | 263 L | `add_text_node`, `add_group_node`, `add_edge` | Pembangkit `WORKFLOW_LINEAGE.canvas` (26 node, 18 sisi) |
| [`generate_version_notes.py`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/src/generate_version_notes.py) | 219 L | Generator atomik `versions/sub_*.md` | Pembuat 28 catatan versi tersinkronisasi CSV master |

> [!WARNING] Catatan Keamanan Skrip Root
> Skrip [`run_graphify_full.sh`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/run_graphify_full.sh) di direktori root memuat perintah `systemctl poweroff` di baris akhir dan tidak boleh dijalankan tanpa disunting terlebih dahulu. Untuk memperbarui graf pengetahuan, panggil CLI `/home/rei/.local/bin/graphify` secara langsung.

---

## 2. Inventaris Kernel Kaggle & Artefak Cloud (`FinalProject/kaggle_kernel*` & `kaggle_pulled*`)

1. **`kaggle_kernel`**: Berisi `mnist-adm.py` (pelatihan ResNet-18 CBAM Pra-Fase) serta `standalone_cpc_sota_pipeline.py` / `.ipynb` (pelatihan biomekanik Fase 1).
2. **`kaggle_kernel_convnext`**: Berisi `mnist-adm-convnext.py` untuk ConvNeXt-Femto 5-fold (`99,552%` OOF).
3. **`kaggle_kernel_densenet`**: Berisi `mnist-adm-densenet121.py` untuk DenseNet-121 CBAM v1 & hotfix v2.
4. **`kaggle_kernel_sota_v3`**: Pelatihan ensemble v3 dengan sintetik kaligrafi v2.
5. **`kaggle_kernel_stm`** + **`kaggle_pulled_stm`**: Pelatihan Track 1 *Style Transfer Mapping* (`sub_stm.zip` -> diunggah sebagai `Sub 16: 0.9469`), menyimpan `stm_matrices.npz` (`527.330 B`) dan `probabilities_stm.npy`.
6. **`kaggle_kernel_fixmatch`**: Skrip Track 2 *FixMatch Consistency Training* (`mnist-adm-fixmatch.py`).
7. **`kaggle_kernel_convnext_fixmatch`** + **`kaggle_pulled_convnext_fixmatch`**: Variasi FixMatch berbasis ConvNeXt-Femto beserta `mnist-adm-convnext-fixmatch.log`.
8. **`kaggle_kernel_massive_sota`** + **`kaggle_pulled_massive_sota`**: Eksperimen *hard pseudo-labeling* (`Sub 29`, log `mnist-adm-massive-sota.log`, `predictions.csv`, serta checkpoint epoch 2–20).
9. **`kaggle_kernel_distillation`** + **`kaggle_pulled_distillation`**: Pelatihan pamungkas **Sub 32** (`mnist-adm-distillation.py`, log `mnist-adm-distillation.log`, `submission_distillation.zip`), melahirkan `densenet121_cbam_student_best.pt` (Ep 13, `29,28 MiB`) dan `densenet121_cbam_student_fp16.pt` (SWA Ep 17–25, `14,75 MiB`). Folder root `distillation_run/checkpoints/` terkonfirmasi kosong (eksekusi final berlangsung di Kaggle TPU/GPU lalu ditarik ke `kaggle_pulled_distillation/` dan disalin ke `FinalProject/models/`).

---

## 3. Rekapitulasi Tugas Perkuliahan Pendukung (`Tugas 1` s.d. `Tugas 3`)

| Tugas | Direktori Workspace | Metode & Dataset | Ringkasan Capaian Terverifikasi |
|:---|:---|:---|:---|
| **Tugas 1** | [`Tugas 1 PCA FA/`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/Tugas%201%20PCA%20FA) | PCA & Exploratory Factor Analysis pada **HBAT100** | KMO `0,60`; ekstraksi **4 faktor** rotasi *Varimax* dengan residu **`RMSR = 0,02`** |
| **Tugas 2** | [`Tugas 2 MANOVA-MACANOVA/`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/Tugas%202%20MANOVA-MACANOVA) | **2×3 MANOVA & MANCOVA** pada **WRAT-R / WRAT-A** | Efek faktor utama `Group` sangat signifikan (**`F = 47,96`**) |
| **Tugas 3** | [`Tugas 3 MANOVA-Diskriminan-Klasifikasi/`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/Tugas%203%20MANOVA-Diskriminan-Klasifikasi) | One-Way MANOVA, Diskriminan Fisher & Klasifikasi (**N=9** T&F 9.1) | **Wilks' $\Lambda = 0{,}01048$**; akurasi **`88,89%` (8/9)** pada validasi *Leave-One-Out Cross-Validation (LOOCV)* |

---

## 4. Ringkasan Audit Sinkronisasi Empiris (24 November 2026)

Pada audit menyeluruh seluruh kode dan log hasil, temuan berikut telah diverifikasi ulang langsung lewat eksekusi kode lokal tanpa pelatihan ulang serta disinkronkan di seluruh dokumentasi dan dek slide:

1. **Perbaikan Box's M via `slogdet` (`multivariate_diagnostics_summary.json`)**: Angka lama (`48.921,4` dan `1,42e5`) yang tidak memiliki implementasi kode diganti menjadi hasil terhitung resmi `compute_box_m()` berbasis `np.linalg.slogdet`: **$M = 214.542,08$ ($\chi^2 = 212.310,12, \, df = 2.940, \, p = 0$)** pada $m=20$ PC, dan **$M = 535.061,64$ ($\chi^2 = 524.045,55, \, df = 11.480, \, p = 0$)** pada $m=40$ PC.
2. **Komposisi PCA & MANOVA Terverifikasi**: PC1–PC4 masing-masing menyumbang `6,67%`, `5,62%`, `4,93%`, dan `4,09%`; retensi `85%` = `109 PC`, `90%` = `154 PC`, Kaiser = `275 PC` ($\bar{\lambda} = 0{,}087497$); Hotelling-Lawley $T = 17{,}6700$ dan Roy $\theta = 0{,}8096$; serta akurasi 5-fold `PCA(100)+LDA = 85,13%`, `PCA(100)+LogReg = 91,08%`, `PCA(40)+QDA(reg=0,1) = 96,09%`.
3. **Detail Eksak Arsitektur Student & Pelatihan KD (`Sub 32`)**: `StudentDenseNet121_CBAM` memiliki `7.522.968` parameter (`MixStyle` 1× pasca-stem `features[:4]`, `4× CBAM` pasca-DenseBlock, proyeksi `Linear->BatchNorm1d->PReLU`, dan `SubCenterArcFace(K=2, s=32.0, m=0.30, +m)`). Pelatihan berlangsung dengan **Online `MeanTeacherEMA` ($\alpha=0,995$)** pada `15.918` Super-Dataset (`T=3,5`).
4. **Identitas Sub 4/5, Sub 12 & Metode Optimal Transport**: Urutan CodaBench `Sub 4` = `hccr_senet` (`996076`, `0.8771`) dan `Sub 5` = `gated_moe` (`996080`, `0.9286`); `Sub 12` = `Pure v2 drop D1` yang memicu kolaps kelas `五` (bukan pemangkasan layer filter ortogonal); dan Optimal Transport pada kode menggunakan **Exact Bipartite Assignment (`scipy.optimize.linear_sum_assignment` Hungarian)** dengan penalti *anti-poaching*, bukan Sinkhorn-Knopp.
5. **Pengukuran Empiris Fase 2 (`verified_empirical_metrics.json`)**: Latensi Batch-128 pada GPU lokal RTX 4060 terukur **`5,436` detik**; stabilitas noise Gaussian $\sigma=25$ lintas 3 seed (`42, 43, 44`) menghasilkan **`95,98% ± 0,02%` (SWA FP16)** vs **`94,37% ± 0,24%` (Ep13)** atau keunggulan **`+1,61%`**; angka **`9,92%` (`288/2.902`)** dilaporkan ketat sebagai *empirical disagreement rate* antara Student SWA dan ensemble 15-model pra-distilasi; serta kuota Fase 2 tercatat **`1/6`** yang diperpanjang panitia hingga 25 November 2026 pukul 23:59 GMT+7.

---
*Navigasi Vault: [[INDEX|Indeks Utama Dokumentasi]] | [[17_MASTER_DOKUMENTASI_KESELURUHAN_DAN_RIWAYAT_VERSI|Master Dokumentasi]] | [[19_TEKS_PRESENTER_PPT_VERBATIM|Teks Presenter PPT]]*
