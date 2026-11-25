# Kaggle Notebooks & Cloud Training Pipeline (`reiiihel/mnist-adm*`)

Folder ini berisi salinan lengkap **9 kernel Kaggle** yang telah ditarik dan divalidasi (`nbformat 4.5`) agar mudah dibaca langsung di GitHub atau dijalankan ulang di Kaggle GPU (`P100` / `T4`). Seluruh eksperimen lokal modular berada di [`../src/`](../src/).

| No. | Berkas Notebook | Kaggle Kernel ID | Era | Peran & Hasil Terverifikasi |
|:---:|:---|:---|:---:|:---|
| `01` | [`01_mnist_adm_resnet18_cbam.ipynb`](./01_mnist_adm_resnet18_cbam.ipynb) | `reiiihel/mnist-adm` | Era I | ResNet-18 CBAM + MixStyle + SubCenter ArcFace (`99,790%` OOF 5-Fold; `Sub 2: 0.9278`) |
| `02` | [`02_mnist_adm_convnext_femto.ipynb`](./02_mnist_adm_convnext_femto.ipynb) | `reiiihel/mnist-adm-convnext` | Era I | ConvNeXt-Femto 5-Fold (`99,552%` OOF; diblend `15%` ke `Sub 3 Rank 1: 0.9319`) |
| `03` | [`03_mnist_adm_densenet121_cbam.ipynb`](./03_mnist_adm_densenet121_cbam.ipynb) | `reiiihel/mnist-adm-densenet121` | Era II | DenseNet-121 CBAM (konkatenasi fitur menjaga goresan tipis `64x64`) |
| `04` | [`04_mnist_adm_sota_v3_biomechanical.ipynb`](./04_mnist_adm_sota_v3_biomechanical.ipynb) | `reiiihel/mnist-adm-sota-v3` | Era II | Augmentasi biomekanik Simard Elastic + dilasi kuas anisotropik `45°`/`135°` |
| `05` | [`05_mnist_adm_fixmatch.ipynb`](./05_mnist_adm_fixmatch.ipynb) | `reiiihel/mnist-adm-fixmatch` | Era III | Track 2 FixMatch semi-supervised consistency regularization |
| `06` | [`06_mnist_adm_stm_adaptation.ipynb`](./06_mnist_adm_stm_adaptation.ipynb) | `reiiihel/mnist-adm-stm-adaptation` | Era III | Track 1 Style Transfer Mapping closed-form QP (`Sub 16: 0.9469`) |
| `07` | [`07_mnist_adm_convnext_fixmatch.ipynb`](./07_mnist_adm_convnext_fixmatch.ipynb) | `reiiihel/mnist-adm-convnext-fixmatch` | Era III | Fine-tuning konsistensi pada cabang ConvNeXt-Femto untuk ensemble |
| `08` | [`08_mnist_adm_massive_sota.ipynb`](./08_mnist_adm_massive_sota.ipynb) | `reiiihel/mnist-adm-massive-sota` | Era IV | Eksperimen pra-distilasi di atas Super-Dataset kurikuler |
| `09` | [`09_mnist_adm_distillation_final.ipynb`](./09_mnist_adm_distillation_final.ipynb) | `reiiihel/mnist-adm-distillation` | Era V–VI | **Final Single Student (`14,75 MiB FP16`)**: 5-Loss `UnifiedDistillationLoss` + `MeanTeacherEMA` + `SWA` + `EM TTPA` (**Fase 1 `Sub 32: 0.9984`**, **Fase 2 `P2-Sub1: 0.9862`**, **`5,44 s`**) |

## Dataset Kaggle Terkait
- `reiiihel/chinese-mnist-adm`
- `reiiihel/chinese-mnist-calligraphy-v2`
