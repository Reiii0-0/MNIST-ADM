# Kaggle Notebooks & Visual Evaluation (`reiiihel/mnist-adm*` + Lokal ADM)

Folder ini berisi salinan lengkap **9 kernel Kaggle** (`reiiihel/mnist-adm*`, format `nbformat 4.5`) yang **sudah menyematkan output eksekusi asli (`execution_count` dan log `stdout` dari Kaggle Tesla T4 GPU)**, ditambah **1 notebook evaluasi visual lokal (`10`)** yang menyematkan grafik matplotlib dan figur diagnostik ADM langsung di dalam sel `.ipynb`. Seluruh skrip modular lokal berada di [`../src/`](../src/).

| No. | Berkas Notebook | Kernel / Lingkup | Era | Peran & Hasil Terverifikasi |
|:---:|:---|:---|:---:|:---|
| `01` | [`01_mnist_adm_resnet18_cbam.ipynb`](./01_mnist_adm_resnet18_cbam.ipynb) | `reiiihel/mnist-adm` | Era I | ResNet-18 CBAM + MixStyle + SubCenter ArcFace (`99,80%` OOF 5-Fold di log akhir / `99,790%` run awal; `Sub 2: 0.9278`) |
| `02` | [`02_mnist_adm_convnext_femto.ipynb`](./02_mnist_adm_convnext_femto.ipynb) | `reiiihel/mnist-adm-convnext` | Era I | ConvNeXt-Femto 5-Fold (`99,55%` OOF; diblend `15%` ke `Sub 3 Rank 1: 0.9319`) |
| `03` | [`03_mnist_adm_densenet121_cbam.ipynb`](./03_mnist_adm_densenet121_cbam.ipynb) | `reiiihel/mnist-adm-densenet121` | Era II | DenseNet-121 CBAM (`99,76%` OOF 5-Fold; `submission_phase1_densenet121_cbam_v2.zip`) |
| `04` | [`04_mnist_adm_sota_v3_biomechanical.ipynb`](./04_mnist_adm_sota_v3_biomechanical.ipynb) | `reiiihel/mnist-adm-sota-v3` | Era II | Augmentasi biomekanik Simard Elastic + dilasi kuas anisotropik `45°`/`135°` (`sub_sota_v3.zip`, `2.466` prediksi) |
| `05` | [`05_mnist_adm_fixmatch.ipynb`](./05_mnist_adm_fixmatch.ipynb) | `reiiihel/mnist-adm-fixmatch` | Era III | Track 2 FixMatch semi-supervised (log Kaggle T4 selesai Fold 1-4, berhenti `CANCEL_ACKNOWLEDGED` di Fold 5 Ep 3) |
| `06` | [`06_mnist_adm_stm_adaptation.ipynb`](./06_mnist_adm_stm_adaptation.ipynb) | `reiiihel/mnist-adm-stm-adaptation` | Era III | Track 1 Style Transfer Mapping closed-form QP (`Sub 16: 0.9469`, `sub_stm.zip`) |
| `07` | [`07_mnist_adm_convnext_fixmatch.ipynb`](./07_mnist_adm_convnext_fixmatch.ipynb) | `reiiihel/mnist-adm-convnext-fixmatch` | Era III | Fine-tuning konsistensi pada cabang ConvNeXt-Femto 5-Fold + 9-way TTA |
| `08` | [`08_mnist_adm_massive_sota.ipynb`](./08_mnist_adm_massive_sota.ipynb) | `reiiihel/mnist-adm-massive-sota` | Era IV | Pelatihan DenseNet-161 CBAM 20 epoch pada Super-Dataset 15.811 citra (`submission_massive_sota.zip`) |
| `09` | [`09_mnist_adm_distillation_final.ipynb`](./09_mnist_adm_distillation_final.ipynb) | `reiiihel/mnist-adm-distillation` | Era V-VI | **Final Single Student (`14,75 MiB FP16`)**: 5-Loss KD + EMA + SWA + EM TTPA (`100,00%` val internal 25 epoch; **Fase 1 `Sub 32: 0.9984`**, **Fase 2 `P2-Sub1: 0.9862`**, **`5,44 s`**) |
| `10` | [`10_mnist_adm_visual_evaluation_and_diagnostics.ipynb`](./10_mnist_adm_visual_evaluation_and_diagnostics.ipynb) | Lokal `FinalProject/` | Era 0-VI | **Evaluasi Visual & Diagnostik ADM Lengkap**: `7` figur `image/png` tertanam langsung (PCA/Scree, 4 plot klasik, kurva Sub 1-32 + Fase 2, terobosan 13 kelas gaya baru, stress test + kompresi >52x, kalibrasi `ECE`, galeri arsitektur) |

## Dataset Kaggle Terkait
- `reiiihel/chinese-mnist-adm`
- `reiiihel/chinese-mnist-calligraphy-v2`
