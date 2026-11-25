---
title: "INDEX: Indeks Pusat Vault Dokumentasi Proyek Akhir ADM"
source_doc: "INDEX.md"
date: 2026-11-18
updated: 2026-11-24
tags:
  - obsidian/moc
  - index
  - adm-project
aliases:
  - Indeks Dokumentasi ADM
  - Master TOC
status: verified
---

# 🗂️ Indeks Pusat Vault Dokumentasi ADM (Chinese MNIST -> Distilled Student)

> [!TIP] Ringkasan Status Proyek Akhir & Kompetisi
> - **Identitas Tim**: `arete` (`5025241135`: Farikh Muhammad Fauzan) | Pengampu: `udong`
> - **Puncak Fase 1 (`Sub 32`, ID `1014243`)**: **`0.9984`** (`2.462 / 2.466` benar, 4 error, model tunggal `14,75 MiB`)
> - **First Strike Fase 2 (`P2-Sub1`, ID `1048381`)**: **`0.9862`** (`~2.862 / 2.902` benar pada 2.902 citra OOD tersegel, **`5,44 s`** Batch-128, kuota `1/6` s.d. 25 Nov 2026)

---

## 1. Dokumen Induk & Presentasi (Terverifikasi 24 Nov 2026)

| Kode | Dokumen | Fokus Pembahasan |
|:---:|:---|:---|
| **16** | [[16_MASTER_PRESENTASI_DOSEN_DAN_TESTING_GUIDE]] | Blueprint presentasi teknis dosen, metrik 24 grafik Royal Blue, & protokol Q&A |
| **17** | [[17_MASTER_DOKUMENTASI_KESELURUHAN_DAN_RIWAYAT_VERSI]] | Master keseluruhan evolusi Era 0 s.d. VI, taksonomi matematis, & riwayat Sub 1–32 + Fase 2 |
| **18** | [[18_INVENTARIS_LENGKAP_KODE_DAN_AUDITING_REPRODUKSI]] | Inventaris lengkap 53 skrip Python, 9 Kaggle kernel, Tugas 1–3, & ringkasan audit empiris |
| **19** | [[19_TEKS_PRESENTER_PPT_VERBATIM]] | Naskah bicara presenter verbatim Slide 1–29 (`/bahasa-skills`, `/unslop`, `/antigravity-guide`) |
| **20** | [[20_PENJELASAN_SISTEM_DARI_TITIK_PALING_DASAR]] | Penjelasan sistem lengkap dari nol (piksel 4.096-D, ADM klasik, DenseNet-CBAM, hingga 5-Loss KD & Fase 2) |

---

## 2. Basis Data, Peta Visual & Graf Pengetahuan (`/documentation-stack`)

- **Basis Data Interaktif Submisi**: [[OBSIDIAN_BASE_SUBMISSIONS.base]] (mengindeks 28 catatan atomik di `versions/sub_01.md` s.d. `versions/sub_phase2_01.md`).
- **Peta Alur Riset Visual**: [[WORKFLOW_LINEAGE.canvas]] (JSON Canvas 1.0 dengan 26 node & 18 sisi tersinkronisasi).
- **Graf Pengetahuan Repositori (`/graphify`)**:
  - [`GRAPH_REPORT.md`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/graphify-out/GRAPH_REPORT.md)
  - [`graph.html`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/graphify-out/graph.html)
- **README Repositori**:
  - [`README.md` Root](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/README.md)
  - [`FinalProject/README.md`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/README.md)

---

## 3. Arsip Kronologis & Laporan Riset Mendalam (Pra-Fase s.d. Era V)

### A. Catatan Kemajuan Bernomor (`00` s.d. `15`)
- [[00_PRA_FASE]]: Fondasi data, sintesis `萬`/`億`, `StratifiedGroupKFold`, & baseline ADM klasik
- [[01_ERROR_ANALYSIS_CONFUSION_MATRIX_REPORT]]: Bedah 22 error OOF Pra-Fase & topologi goresan mirip
- [[02_STRESS_TEST_ROBUSTNESS_REPORT]]: Simulasi dilasi/blur panitia & keutuhan fitur ResNet vs ConvNeXt
- [[03_ARCHITECTURAL_COMPARISON_RESNET_VS_CONVNEXT]]: Perbandingan bias induktif kernel $3\times 3$ vs $7\times 7$
- [[04_PHASE_1_VALIDATION_ANALYSIS_AND_ACTIVE_LEARNING]]: Analisis pembukaan Fase 1 (2.466 citra) & domain shift
- [[05_CODABENCH_EVALUATION_REPORT_AND_HEATMAP_ANALYSIS]]: Evaluasi Sub 1–5 & perebutan Rank 1 pertama (`0.9319`)
- [[06_MASTER_PLAN_95_PLUS_AND_COMPETITION_EXECUTION]]: Rencana strategis menembus 95%+
- [[06_SOTA_BIOMECHANICAL_AUGMENTATION_AND_TRI_ENSEMBLE_ARCHITECTURAL_BLUEPRINT]]: Deformasi elastis Simard & Tri-Ensemble
- [[06_TEST_TIME_LABEL_SHIFT_ADAPTATION_RESEARCH_REPORT]]: Penurunan matematis Bayesian EM TTPA tanpa zero-masking
- [[07_TEORI_ILMIAH_DENSENET_DENSERAN_HCCR]]: Pembuktian teoritis DenseNet/DenseRAN (Wang et al. 2018)
- [[08_PHASE_1_PROGRESS_AND_15_MODEL_GRANDMASTER_ENSEMBLE]]: Tri-Orthogonal 15-model ensemble (`Sub 9–10`)
- [[09_TACTIC_A_V2_RESULTS_AND_GROUND_TRUTH_ALIGNMENT]]: Evaluasi pelatihan v2 & penyelarasan distribusi
- [[10_KAGGLE_DENSENET_V2_AUDIT_AND_HOTFIX]]: Audit dan perbaikan cabang DenseNet v2
- [[11_SUBMISSION_12_ANALYSIS_AND_NAMING_REFACTOR]]: Audit ablasi Pure v2 `drop D1` (`Sub 12: 0.9424`)
- [[12_MASTER_PROGRESS_ROADMAP_96_TO_99_PERCENT_AND_KAGGLE_V3]]: Roadmap melampaui plafon 94,73%
- [[13_TRACK_2_SEMI_SUPERVISED_FIXMATCH_CONSISTENCY_TRAINING]]: Laporan eksperimen Track 2 FixMatch
- [[14_MASTER_PROGRESS_ROADMAP_TRACK1_TRACK2_AND_GROUND_TRUTH_DISCOVERY]]: Track 1 STM, Kurikulum NCCU, & awal temuan MILP
- [[15_MASTER_DOCUMENTATION_SUB1_TO_SUB30_AND_99_PERCENT_BLUEPRINT]]: Dokumentasi transisi Sub 1–30 menuju distilasi

### B. Laporan Riset, Strategi & Audit Khusus (8 Dokumen Tematik)
- [[multivariate_analysis_research_report]]: Laporan teoritis lengkap PCA, MANOVA, Box's M, Mardia's Test, LDA, & QDA
- [[distillation_research_report]]: Blueprint SOTA 5-Loss Knowledge Distillation (`Sub 32`)
- [[COMPETITION_PLAYBOOK]]: Playbook teknis kompetisi HCCR & pencegahan kebocoran data
- [[LITERATURE_BENCHMARK_ACADEMIC_RIGOR_EVALUATION]]: Meta-analisis posisi SOTA dibanding CASIA-HWDB & Chinese MNIST
- [[MASTER_PLAN_WINNING_STRATEGY]]: Desain sinergi jalur ganda (Nilai Akademik A+ & Juara CodaBench)
- [[ROADMAP_MENUNGGU_FASE_1]]: Daftar periksa persiapan menjelang rilis validasi Fase 1
- [[AUDIT_KERENTANAN_FASE1_FASE2]]: Analisis celah blur/ketebalan, pergeseran kelas, & adaptasi tanpa label
- [[AUDIT_INTEGRITAS_DAN_KESIAPAN_KAGGLE_CODABENCH]]: Verifikasi bebas kebocoran penulis & kepatuhan etika kompetisi

---

## 4. Direktori 28 Catatan Submisi Atomik (`versions/`)

[[sub_01]] · [[sub_02]] · [[sub_03]] · [[sub_04]] · [[sub_05]] · [[sub_06]] · [[sub_07]] · [[sub_08]] · [[sub_09]] · [[sub_10]] · [[sub_11]] · [[sub_12]] · [[sub_13]] · [[sub_14]] · [[sub_15]] · [[sub_16]] · [[sub_20]] · [[sub_23]] · [[sub_24]] · [[sub_25]] · [[sub_26]] · [[sub_27]] · [[sub_28]] · [[sub_29]] · [[sub_30]] · [[sub_31]] · [[sub_32]] · [[sub_phase2_01]]

---

## 5. Katalog 22 Grafik Ilmiah & Infografis Tema Royal Blue (`#2254c5`)

### A. 4 Grafik Statistik Multivariat Klasik (`FinalProject/outputs/`)
1. [`scree_plot.png`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/outputs/scree_plot.png)
2. [`eigen_characters.png`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/outputs/eigen_characters.png)
3. [`mahalanobis_qq_plot.png`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/outputs/mahalanobis_qq_plot.png)
4. [`lda_canonical_projection.png`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/outputs/lda_canonical_projection.png)

### B. 7 Infografis Arsitektural & Dataset (`FinalProject/outputs/presentation_figures/`)
5. [`info_15_classes_and_calligraphy.png`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/outputs/presentation_figures/info_15_classes_and_calligraphy.png)
6. [`info_domain_shift_comparison.png`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/outputs/presentation_figures/info_domain_shift_comparison.png)
7. [`info_dataset_architecture_table.png`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/outputs/presentation_figures/info_dataset_architecture_table.png)
8. [`info_adm_to_dl_bridge.png`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/outputs/presentation_figures/info_adm_to_dl_bridge.png)
9. [`info_densenet_cbam_architecture.png`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/outputs/presentation_figures/info_densenet_cbam_architecture.png)
10. [`info_biomechanical_augmentation.png`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/outputs/presentation_figures/info_biomechanical_augmentation.png)
11. [`info_kd_5loss_pipeline.png`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/outputs/presentation_figures/info_kd_5loss_pipeline.png)

### C. 11 Grafik Kuantitatif Evolusi, Adaptasi & Audit (`FinalProject/outputs/presentation_figures/`)
12. [`slide_evolution_trajectory_0_to_student.png`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/outputs/presentation_figures/slide_evolution_trajectory_0_to_student.png)
13. [`slide_era1_phase1_jump.png`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/outputs/presentation_figures/slide_era1_phase1_jump.png)
14. [`info_ttpa_prior_shift.png`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/outputs/presentation_figures/info_ttpa_prior_shift.png)
15. [`slide_era3_ceiling_and_ablations.png`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/outputs/presentation_figures/slide_era3_ceiling_and_ablations.png)
16. [`slide_era4_milp_tomography.png`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/outputs/presentation_figures/slide_era4_milp_tomography.png)
17. [`slide_difficult_classes_breakthrough.png`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/outputs/presentation_figures/slide_difficult_classes_breakthrough.png)
18. [`slide_model_efficiency.png`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/outputs/presentation_figures/slide_model_efficiency.png)
19. [`slide_perturbation_robustness.png`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/outputs/presentation_figures/slide_perturbation_robustness.png)
20. [`slide_swa_vs_ep13_audit.png`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/outputs/presentation_figures/slide_swa_vs_ep13_audit.png)
21. [`slide_phase2_ood_verification.png`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/outputs/presentation_figures/slide_phase2_ood_verification.png)
22. [`slide_calibration_reliability.png`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/FinalProject/outputs/presentation_figures/slide_calibration_reliability.png)

---

## 6. Keterkaitan Tugas Kuliah Analisis Data Multivariat (`Tugas 1–3`)

- **Tugas 1 (`PCA & Exploratory Factor Analysis`)**: [`5025241135_PCAFAADM.ipynb`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/Tugas%201%20PCA%20FA/5025241135_PCAFAADM.ipynb): Dataset **HBAT100**, KMO `0,60`, 4 faktor *Varimax* dengan `RMSR = 0,02`.
- **Tugas 2 (`MANOVA & MANCOVA 2×3`)**: [`Tugas2_MANOVA_MANCOVA.Rmd`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/Tugas%202%20MANOVA-MACANOVA/Tugas2_MANOVA_MANCOVA.Rmd): Desain **WRAT-R / WRAT-A**, efek utama `Group` **`F = 47,96`**.
- **Tugas 3 (`MANOVA, Diskriminan Fisher & Klasifikasi`)**: [`5025241135_MANOVA_Diskriminan_Klasifikasi.ipynb`](file:///home/rei/Workspace/Academic/Multivariate%20Data%20Analysis/Tugas%203%20MANOVA-Diskriminan-Klasifikasi/5025241135_MANOVA_Diskriminan_Klasifikasi.ipynb): Sampel T&F 9.1 ($N = 9$), **Wilks' $\Lambda = 0{,}01048$**, akurasi LOOCV **`88,89%` (8/9)**.
