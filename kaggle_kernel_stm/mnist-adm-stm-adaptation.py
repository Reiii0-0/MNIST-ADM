#!/usr/bin/env python3
"""
================================================================================
CodaBench Chinese MNIST Challenge — Track 1: Style Transfer Mapping (STM)
Based on: Xu-Yao Zhang, Fei Yin, Yan-Ming Zhang, Cheng-Lin Liu, Yoshua Bengio.
"Drawing and Recognizing Chinese Characters with Recurrent Neural Network",
IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI 2017).
Section 6: Unsupervised Writer Adaptation of ConvNet (pp. 7-8).

Mathematical Formulation:
-------------------------
Let phi(x) in R^d (d = 256) be the deep feature representation extracted from the
penultimate bottleneck layer of a deep neural network (ResNet-18 CBAM / ConvNeXt-Femto).

1. Class-Specific Prototypes (Source Layer Means):
   mu_k = (1 / N_k) * sum_{j: y_j = k} phi(x_j^{trn}) in R^d, for k in {0, ..., C-1}.

2. Unsupervised Style Transfer Mapping:
   For n unlabeled test images {x_i}_{i=1}^n with pseudo-labels y_i and
   softmax confidences f_i in [0, 1], the affine adaptation parameters A in R^{d x d}
   and offset b in R^d are determined by solving the convex Quadratic Program (QP):
   
     min_{A, b} sum_{i=1}^n f_i * ||A * phi(x_i) + b - mu_{y_i}||_2^2
                + beta * ||A - I||_F^2 + gamma * ||b||_2^2

   Setting gradients dL/db = 0 and dL/dA = 0 yields the analytical closed-form system:
     b = sum_{i=1}^n f_i * (mu_{y_i} - A * z_i) / (sum_{i=1}^n f_i + gamma)
     A = (sum_{i=1}^n f_i * (mu_{y_i} - b) * z_i^T + beta * I)
         * inv(sum_{i=1}^n f_i * z_i * z_i^T + beta * I)

3. Self-Training Iteration (Algorithm 1, Bengio et al. 2017):
   Initialize A = I, b = 0.
   For iter = 1 to iterNum:
     z_i' = A * phi(x_i) + b
     logits_i = W * z_i' + w_0
     p_i = softmax(logits_i)
     y_i = argmax_c p_{ic}
     f_i = max_c p_{ic}
     Update A, b by closed-form QP solution.
================================================================================
"""

import os
import sys
import glob
import math
import json
import zipfile
import random
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score

# ==============================================================================
# 1. HARDWARE & REPRODUCIBILITY CONFIGURATION
# ==============================================================================
def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

seed_everything(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 80)
print(f"🚀 KAGGLE HARDWARE SETUP | Active Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"   • Device Name  : {torch.cuda.get_device_name(0)}")
    print(f"   • VRAM Cap     : {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    print(f"   • PyTorch Ver  : {torch.__version__}")
print("=" * 80)

# Paths configuration
if os.path.exists("/kaggle"):
    TMP_DATA_DIR = "/tmp/data"
    WORKING_DIR = "/kaggle/working"
else:
    TMP_DATA_DIR = "/tmp/mnist_adm_data"
    WORKING_DIR = "./FinalProject/outputs/kaggle_stm"

os.makedirs(TMP_DATA_DIR, exist_ok=True)
os.makedirs(WORKING_DIR, exist_ok=True)
MODELS_DIR = os.path.join(WORKING_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


# ==============================================================================
# 2. DYNAMIC DATASET SCANNING & EXTRACTION
# ==============================================================================
def setup_datasets():
    print("\n[*] Scanning & extracting dataset files in search paths...")
    if os.path.exists("/kaggle/input"):
        search_roots = ["/kaggle/input"]
    elif os.path.exists("./FinalProject/data"):
        search_roots = ["./FinalProject/data", "./FinalProject/kaggle_dataset", "./FinalProject/kaggle_calligraphy_dataset"]
    elif os.path.exists("./data"):
        search_roots = ["./data"]
    else:
        search_roots = ["."]

    # 1. Extract any zip archives found across search_roots into TMP_DATA_DIR
    for r in search_roots:
        for z in glob.glob(f"{r}/**/*.zip", recursive=True):
            zname = os.path.basename(z).lower()
            target_sub = os.path.join(TMP_DATA_DIR, os.path.splitext(zname)[0])
            if not os.path.exists(target_sub) or len(os.listdir(target_sub)) == 0:
                print(f"  -> Extracting {z} to {target_sub} ...")
                os.makedirs(target_sub, exist_ok=True)
                try:
                    with zipfile.ZipFile(z, 'r') as zip_ref:
                        zip_ref.extractall(target_sub)
                except Exception as e:
                    print(f"     Warning extracting {z}: {e}")

    # 2. Locate CSV Metadata
    splits_csv = None
    synth_trad_csv = None
    synth_callig_csv = None

    for r in search_roots + [TMP_DATA_DIR]:
        for f in glob.glob(f"{r}/**/*.csv", recursive=True):
            fname = os.path.basename(f).lower()
            if "splits_kfold" in fname and splits_csv is None:
                splits_csv = f
            elif "train_synthetic_labels" in fname and synth_trad_csv is None:
                synth_trad_csv = f
            elif "train_synthetic_calligraphy_v2" in fname and synth_callig_csv is None:
                synth_callig_csv = f

    print(f"  • Splits CSV          : {splits_csv}")
    print(f"  • Synthetic Trad CSV  : {synth_trad_csv}")
    print(f"  • Synthetic Callig CSV: {synth_callig_csv}")

    # 3. Universal Recursive Image Indexing
    print("\n[*] Building universal path index of all available images...")
    id_to_path = {}
    valid_exts = {".png", ".jpg", ".jpeg"}

    for r in search_roots + [TMP_DATA_DIR]:
        for root, _, files in os.walk(r):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in valid_exts:
                    full_p = os.path.join(root, file)
                    base_id = os.path.splitext(file)[0]
                    id_to_path[base_id] = full_p
                    id_to_path[file] = full_p

    print(f"  • Total unique image IDs indexed: {len(id_to_path)}")

    # 4. Locate 2,466 Validation Images
    val_files = []
    for r in search_roots + [TMP_DATA_DIR]:
        matches = glob.glob(f"{r}/**/val_*.png", recursive=True) + glob.glob(f"{r}/**/val_*.jpg", recursive=True)
        if len(matches) >= 2466:
            val_files = sorted(list(set(matches)))
            break

    if not val_files:
        matches = [p for name, p in id_to_path.items() if name.startswith("val_")]
        val_files = sorted(list(set(matches)))

    print(f"  • Found {len(val_files)} validation images for test-time adaptation.")
    return splits_csv, synth_trad_csv, synth_callig_csv, id_to_path, val_files


# ==============================================================================
# 3. BIOMECHANICAL AUGMENTATION & DATASET CLASS
# ==============================================================================
def apply_elastic_distortion(img_arr: np.ndarray, alpha: float = 6.0, sigma: float = 3.5) -> np.ndarray:
    """Simard et al. (ICDAR 2003) Elastic Distortion for handwriting variation."""
    H, W = img_arr.shape
    dx = gaussian_filter((np.random.rand(H, W) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((np.random.rand(H, W) * 2 - 1), sigma) * alpha
    grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
    map_x = (grid_x + dx).astype(np.float32)
    map_y = (grid_y + dy).astype(np.float32)
    return cv2.remap(img_arr, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def apply_anisotropic_brush_erosion(img_arr: np.ndarray) -> np.ndarray:
    """Simulates variable brush pressure and ink depletion along strokes."""
    k = np.array([[1, 0], [0, 1]], dtype=np.uint8) if random.random() > 0.5 else np.array([[0, 1], [1, 0]], dtype=np.uint8)
    return cv2.erode(img_arr, k, iterations=1)


class ChineseMNISTDataset(Dataset):
    def __init__(self, df: pd.DataFrame, id_to_path: Dict[str, str], is_train: bool = True):
        self.df = df.reset_index(drop=True)
        self.id_to_path = id_to_path
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_id = str(row['id'])

        path = self.id_to_path.get(file_id)
        if path is None:
            path = self.id_to_path.get(f"{file_id}.jpg", self.id_to_path.get(f"{file_id}.png"))

        if path is None or not os.path.exists(path):
            img = np.zeros((64, 64), dtype=np.uint8)
        else:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                img = np.zeros((64, 64), dtype=np.uint8)

        if img.shape != (64, 64):
            img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)

        if self.is_train:
            if random.random() < 0.40:
                img = apply_elastic_distortion(img, alpha=5.0, sigma=3.0)
            if random.random() < 0.30:
                img = apply_anisotropic_brush_erosion(img)

            angle = random.uniform(-6.0, 6.0)
            tx = random.uniform(-2.0, 2.0)
            ty = random.uniform(-2.0, 2.0)
            M = cv2.getRotationMatrix2D((32, 32), angle, 1.0)
            M[0, 2] += tx
            M[1, 2] += ty
            img = cv2.warpAffine(img, M, (64, 64), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        # Standard Chinese MNIST normalization [-1.0, 1.0]
        tensor_img = torch.from_numpy(img.astype(np.float32) / 127.5 - 1.0).unsqueeze(0)
        label = int(row.get('label', -1))
        writer_id = int(row.get('writer_id', 999))

        return tensor_img, label, file_id


# ==============================================================================
# 4. NEURAL ARCHITECTURES: ResNet-18 CBAM & ConvNeXt-Femto
# ==============================================================================

# --- Attention & ResNet-18 CBAM ---
class MixStyle(nn.Module):
    def __init__(self, p=0.4, alpha=0.1, eps=1e-6):
        super().__init__()
        self.p = p
        self.alpha = alpha
        self.eps = eps

    def forward(self, x):
        if not self.training or random.random() > self.p:
            return x
        B = x.size(0)
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x - mu) / sig

        lmda = np.random.beta(self.alpha, self.alpha)
        perm = torch.randperm(B)
        mu2, sig2 = mu[perm], sig[perm]
        mu_mix = mu * lmda + mu2 * (1 - lmda)
        sig_mix = sig * lmda + sig2 * (1 - lmda)
        return x_normed * sig_mix + mu_mix


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        reduced = max(in_planes // ratio, 8)
        self.fc1 = nn.Conv2d(in_planes, reduced, 1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(reduced, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return self.sigmoid(out)


class CBAMBlock(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class BasicBlockCBAM(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.cbam = CBAMBlock(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = self.cbam(out)
        out += self.shortcut(x)
        out = F.relu(out, inplace=True)
        return out


class ResNet18_CBAM(nn.Module):
    """ResNet-18 with CBAM for 64x64 Grayscale Characters (feature_dim = 256)."""
    def __init__(self, in_channels=1, num_classes=15, feature_dim=256, use_mixstyle=True):
        super().__init__()
        self.in_planes = 64
        self.use_mixstyle = use_mixstyle
        self.mixstyle = MixStyle(p=0.4, alpha=0.1) if use_mixstyle else nn.Identity()

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, feature_dim)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlockCBAM(self.in_planes, planes, s))
            self.in_planes = planes * BasicBlockCBAM.expansion
        return nn.Sequential(*layers)

    def extract_features(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        if self.use_mixstyle:
            out = self.mixstyle(out)
        out = self.layer1(out)
        if self.use_mixstyle:
            out = self.mixstyle(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        feats = F.relu(self.fc(out))
        return feats

    def forward(self, x):
        feats = self.extract_features(x)
        logits = self.classifier(feats)
        return logits, feats


# --- ConvNeXt-Femto ---
class LayerNorm2d(nn.Module):
    """LayerNorm channels first (N, C, H, W)."""
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim, 1, 1)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x):
        inp = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        return inp + x


class ConvNeXt_Femto(nn.Module):
    """ConvNeXt-Femto tailored for 64x64 Grayscale Characters (feature_dim = 256)."""
    def __init__(self, in_channels=1, num_classes=15, feature_dim=256, depths=(2, 2, 4, 2), dims=(48, 96, 192, 384)):
        super().__init__()
        self.downsample_layers = nn.ModuleList()

        # Stem: 64x64 -> 32x32
        stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=2, padding=1),
            LayerNorm2d(dims[0])
        )
        self.downsample_layers.append(stem)

        for i in range(3):
            downsample = nn.Sequential(
                LayerNorm2d(dims[i]),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2)
            )
            self.downsample_layers.append(downsample)

        self.stages = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(*[ConvNeXtBlock(dims[i]) for _ in range(depths[i])])
            self.stages.append(stage)

        self.norm = LayerNorm2d(dims[-1])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(dims[-1], feature_dim)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def extract_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        x = self.norm(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        feats = F.gelu(self.fc(x))
        return feats

    def forward(self, x):
        feats = self.extract_features(x)
        logits = self.classifier(feats)
        return logits, feats


# ==============================================================================
# 5. MODEL CHECKPOINT MANAGER (SEARCH & FAST FALLBACK TRAIN)
# ==============================================================================
def find_or_train_checkpoints(df_splits, id_to_path, synth_trad_csv=None, synth_callig_csv=None):
    """
    Finds existing checkpoints for ResNet-18 CBAM and ConvNeXt-Femto.
    If checkpoints are not present, trains them quickly on GPU.
    """
    search_dirs = [
        "/kaggle/input",
        "/kaggle/working/models",
        "./FinalProject/models",
        "./models",
        "/tmp/adm_out/models",
        MODELS_DIR
    ]

    r18_ckpts = []
    cnxt_ckpts = []

    for d in search_dirs:
        if os.path.exists(d):
            r18_matches = sorted(glob.glob(f"{d}/**/resnet18_cbam_fold*.pt", recursive=True))
            cnxt_matches = sorted(glob.glob(f"{d}/**/convnext_femto_fold*.pt", recursive=True))
            if r18_matches and not r18_ckpts:
                r18_ckpts = r18_matches
            if cnxt_matches and not cnxt_ckpts:
                cnxt_ckpts = cnxt_matches

    def deduplicate_folds(paths):
        fold_dict = {}
        for p in paths:
            base = os.path.basename(p)
            for f in range(5):
                if f"fold{f}" in base and f not in fold_dict:
                    fold_dict[f] = p
        return [fold_dict[f] for f in sorted(fold_dict.keys())]

    r18_ckpts = deduplicate_folds(r18_ckpts)
    cnxt_ckpts = deduplicate_folds(cnxt_ckpts)

    print("\n[*] Pretrained Checkpoints Inventory (Deduplicated Folds):")
    print(f"    • ResNet-18 CBAM Checkpoints Found : {len(r18_ckpts)}")
    for p in r18_ckpts:
        print(f"      - {p}")
    print(f"    • ConvNeXt-Femto Checkpoints Found: {len(cnxt_ckpts)}")
    for p in cnxt_ckpts:
        print(f"      - {p}")

    # Fallback: if either architecture has no checkpoints, train fold 0 on GPU
    need_train_r18 = (len(r18_ckpts) == 0)
    need_train_cnxt = (len(cnxt_ckpts) == 0)

    if need_train_r18 or need_train_cnxt:
        print("\n⚠️ Checkpoints missing in input! Initiating fast high-precision GPU training...")
        train_df = df_splits[df_splits['fold'] != 0].copy()
        val_df = df_splits[df_splits['fold'] == 0].copy()

        if synth_trad_csv and os.path.exists(synth_trad_csv):
            df_s1 = pd.read_csv(synth_trad_csv)
            train_df = pd.concat([train_df, df_s1], ignore_index=True)
        if synth_callig_csv and os.path.exists(synth_callig_csv):
            df_s2 = pd.read_csv(synth_callig_csv)
            train_df = pd.concat([train_df, df_s2], ignore_index=True)

        train_ds = ChineseMNISTDataset(train_df, id_to_path, is_train=True)
        val_ds = ChineseMNISTDataset(val_df, id_to_path, is_train=False)

        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

        if need_train_r18:
            print("\n🚀 Training ResNet-18 CBAM (12 Epochs)...")
            m = ResNet18_CBAM(in_channels=1, num_classes=15, feature_dim=256, use_mixstyle=True).to(DEVICE)
            opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=12, eta_min=1e-5)
            crit = nn.CrossEntropyLoss(label_smoothing=0.05)
            scaler = GradScaler('cuda') if torch.cuda.is_available() else None

            best_acc = 0.0
            save_p = os.path.join(MODELS_DIR, "resnet18_cbam_fold0.pt")

            for ep in range(1, 13):
                m.train()
                for imgs, lbls, _ in train_loader:
                    imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                    opt.zero_grad()
                    with autocast('cuda', enabled=torch.cuda.is_available()):
                        logits, _ = m(imgs)
                        loss = crit(logits, lbls)
                    if scaler:
                        scaler.scale(loss).backward()
                        scaler.step(opt)
                        scaler.update()
                    else:
                        loss.backward()
                        opt.step()
                sch.step()

                m.eval()
                correct, total = 0, 0
                with torch.no_grad():
                    for imgs, lbls, _ in val_loader:
                        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                        logits, _ = m(imgs)
                        correct += (logits.argmax(dim=1) == lbls).sum().item()
                        total += imgs.size(0)
                acc = correct / total
                if acc > best_acc:
                    best_acc = acc
                    torch.save({"model_state_dict": m.state_dict(), "val_acc": acc, "fold": 0}, save_p)
                print(f"  • ResNet-18 Ep {ep:02d}/12 | Val Acc: {acc*100:.2f}% (Best: {best_acc*100:.2f}%)")

            r18_ckpts = [save_p]

        if need_train_cnxt:
            print("\n🚀 Training ConvNeXt-Femto (12 Epochs)...")
            m = ConvNeXt_Femto(in_channels=1, num_classes=15, feature_dim=256).to(DEVICE)
            opt = torch.optim.AdamW(m.parameters(), lr=8e-4, weight_decay=1e-4)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=12, eta_min=1e-5)
            crit = nn.CrossEntropyLoss(label_smoothing=0.05)
            scaler = GradScaler('cuda') if torch.cuda.is_available() else None

            best_acc = 0.0
            save_p = os.path.join(MODELS_DIR, "convnext_femto_fold0.pt")

            for ep in range(1, 13):
                m.train()
                for imgs, lbls, _ in train_loader:
                    imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                    opt.zero_grad()
                    with autocast('cuda', enabled=torch.cuda.is_available()):
                        logits, _ = m(imgs)
                        loss = crit(logits, lbls)
                    if scaler:
                        scaler.scale(loss).backward()
                        scaler.step(opt)
                        scaler.update()
                    else:
                        loss.backward()
                        opt.step()
                sch.step()

                m.eval()
                correct, total = 0, 0
                with torch.no_grad():
                    for imgs, lbls, _ in val_loader:
                        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                        logits, _ = m(imgs)
                        correct += (logits.argmax(dim=1) == lbls).sum().item()
                        total += imgs.size(0)
                acc = correct / total
                if acc > best_acc:
                    best_acc = acc
                    torch.save({"model_state_dict": m.state_dict(), "val_acc": acc, "fold": 0}, save_p)
                print(f"  • ConvNeXt Ep {ep:02d}/12 | Val Acc: {acc*100:.2f}% (Best: {best_acc*100:.2f}%)")

            cnxt_ckpts = [save_p]

    return r18_ckpts, cnxt_ckpts


# ==============================================================================
# 6. STYLE TRANSFER MAPPING (STM) ANALYTICAL CLOSED-FORM ENGINE
# ==============================================================================
def compute_class_prototypes(
    model: nn.Module,
    train_loader: DataLoader,
    num_classes: int = 15,
    feature_dim: int = 256
) -> np.ndarray:
    """
    Extracts training features z = phi(x) and calculates class prototype means:
    mu_k = (1 / N_k) * sum_{j: y_j = k} phi(x_j^{trn}) in R^d.
    """
    model.eval()
    all_feats = []
    all_labels = []

    with torch.no_grad():
        for imgs, lbls, _ in train_loader:
            imgs = imgs.to(DEVICE)
            with autocast('cuda', enabled=torch.cuda.is_available()):
                f = model.extract_features(imgs)
            all_feats.append(f.cpu().numpy())
            all_labels.append(lbls.numpy())

    Z_trn = np.concatenate(all_feats, axis=0)
    Y_trn = np.concatenate(all_labels, axis=0)

    prototypes = np.zeros((num_classes, feature_dim), dtype=np.float32)
    for c in range(num_classes):
        mask = (Y_trn == c)
        if np.sum(mask) > 0:
            prototypes[c] = np.mean(Z_trn[mask], axis=0)
        else:
            print(f"Warning: Class {c} has no training samples in prototype loader!")

    return prototypes


def solve_closed_form_stm(
    Z_val: np.ndarray,
    y_pred: np.ndarray,
    f_conf: np.ndarray,
    mu: np.ndarray,
    beta: float = 50.0,
    gamma: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bengio TPAMI 2017 Closed-Form Quadratic Programming (QP) Solver:
      min_{A, b} sum_{i=1}^n f_i * ||A * z_i + b - mu_{y_i}||_2^2
                 + beta * ||A - I||_F^2 + gamma * ||b||_2^2

    Analytic Solution:
      b = sum(f_i * (mu_{y_i} - A * z_i)) / (sum(f_i) + gamma)
      A = (sum(f_i * (mu_{y_i} - b) * z_i^T) + beta * I)
          * inv(sum(f_i * z_i * z_i^T) + beta * I)
    """
    n, d = Z_val.shape
    Sf = np.sum(f_conf)

    # Weighted center of mass
    bar_mu = (mu[y_pred].T @ f_conf) / (Sf + gamma)
    bar_z = (Z_val.T @ f_conf) / (Sf + gamma)

    # Weighted covariance matrices
    Z_f = np.sqrt(f_conf[:, None]) * Z_val
    cov_z = Z_f.T @ Z_f
    cov_yz = (mu[y_pred] * f_conf[:, None]).T @ Z_val

    # Substitution of b = bar_mu - A * bar_z into the matrix gradient:
    LHS = cov_z - (Sf + gamma) * np.outer(bar_z, bar_z) + beta * np.eye(d)
    RHS = cov_yz - (Sf + gamma) * np.outer(bar_mu, bar_z) + beta * np.eye(d)

    # Invert LHS with ridge stabilization
    try:
        A = RHS @ np.linalg.inv(LHS)
    except np.linalg.LinAlgError:
        A = RHS @ np.linalg.pinv(LHS)

    b = bar_mu - A @ bar_z

    # Coordinate Refinement (2 block coordinate descent steps for numerical perfection)
    inv_cov_z_beta = np.linalg.inv(cov_z + beta * np.eye(d))
    for _ in range(2):
        b = (mu[y_pred].T @ f_conf - A @ (Z_val.T @ f_conf)) / (Sf + gamma)
        numer = (mu[y_pred] - b).T @ (f_conf[:, None] * Z_val) + beta * np.eye(d)
        A = numer @ inv_cov_z_beta

    return A.astype(np.float32), b.astype(np.float32)


def run_stm_self_training_pipeline(
    model: nn.Module,
    model_name: str,
    val_loader: DataLoader,
    mu_prototypes: np.ndarray,
    num_iters: int = 4,
    beta: float = 50.0,
    gamma: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Executes Algorithm 1 (Unsupervised Adaptation of ConvNet, Bengio et al. 2017).
    Returns (unadapted_probs, adapted_probs, final_A, final_b).
    """
    print(f"\n{'='*75}")
    print(f"🔬 RUNNING STM ADAPTATION: {model_name}")
    print(f"   Hyperparameters: beta={beta}, gamma={gamma}, self_training_iters={num_iters}")
    print(f"{'='*75}")

    model.eval()
    val_feats = []
    with torch.no_grad():
        for imgs, _, _ in val_loader:
            imgs = imgs.to(DEVICE)
            with autocast('cuda', enabled=torch.cuda.is_available()):
                f = model.extract_features(imgs)
            val_feats.append(f.cpu().numpy())

    Z_val = np.concatenate(val_feats, axis=0)
    n, d = Z_val.shape
    print(f"  • Extracted {n} test feature vectors of dimension {d}.")

    W = model.classifier.weight.detach().cpu().numpy()
    w0 = model.classifier.bias.detach().cpu().numpy()

    # Step 0: Baseline unadapted predictions
    logits0 = Z_val @ W.T + w0
    probs0 = np.exp(logits0 - np.max(logits0, axis=1, keepdims=True))
    probs0 /= np.sum(probs0, axis=1, keepdims=True)

    y_curr = np.argmax(probs0, axis=1)
    f_curr = np.max(probs0, axis=1)
    print(f"  • Baseline Unadapted Mean Confidence : {np.mean(f_curr):.4f}")
    print(f"  • Baseline Predicted Class Counts     : {np.bincount(y_curr, minlength=15)}")

    A = np.eye(d, dtype=np.float32)
    b = np.zeros(d, dtype=np.float32)
    probs_adapted = probs0.copy()

    # Algorithm 1: Self-Training Iteration Loop
    for it in range(1, num_iters + 1):
        # 1. Closed-Form Quadratic Programming Optimization
        A, b = solve_closed_form_stm(Z_val, y_curr, f_curr, mu_prototypes, beta=beta, gamma=gamma)

        # 2. Transform Features z' = A * z + b
        Z_adapted = Z_val @ A.T + b

        # 3. Predict with Linear Classifier on Adapted Features
        logits_adapted = Z_adapted @ W.T + w0
        probs_adapted = np.exp(logits_adapted - np.max(logits_adapted, axis=1, keepdims=True))
        probs_adapted /= np.sum(probs_adapted, axis=1, keepdims=True)

        y_new = np.argmax(probs_adapted, axis=1)
        f_new = np.max(probs_adapted, axis=1)

        shifts = np.sum(y_new != y_curr)
        shift_pct = (shifts / n) * 100.0
        diff_A = np.linalg.norm(A - np.eye(d))
        norm_b = np.linalg.norm(b)

        print(f"  ⭐ Iter {it:02d}/{num_iters:02d} | Shifts: {shifts:3d}/{n} ({shift_pct:.2f}%) | "
              f"Mean Conf: {np.mean(f_new):.4f} | ||A - I||_F: {diff_A:.4f} | ||b||_2: {norm_b:.4f}")

        y_curr = y_new
        f_curr = f_new

    print(f"  • Final Adapted Class Counts          : {np.bincount(y_curr, minlength=15)}")
    return probs0, probs_adapted, A, b


def load_state_dict_flexible(model: nn.Module, state_dict: dict):
    """Robust state dict loader resolving minor layer naming differences (conv vs conv1)."""
    model_keys = set(model.state_dict().keys())
    remapped = {}
    for k, v in state_dict.items():
        if k in model_keys:
            remapped[k] = v
        elif k.replace(".sa.conv1.", ".sa.conv.") in model_keys:
            remapped[k.replace(".sa.conv1.", ".sa.conv.")] = v
        elif k.replace(".sa.conv.", ".sa.conv1.") in model_keys:
            remapped[k.replace(".sa.conv.", ".sa.conv1.")] = v
        else:
            remapped[k] = v
    model.load_state_dict(remapped)


# ==============================================================================
# 7. MAIN STM PIPELINE ORCHESTRATOR
# ==============================================================================
def main():
    splits_csv, synth_trad_csv, synth_callig_csv, id_to_path, val_files = setup_datasets()
    if not splits_csv or not os.path.exists(splits_csv):
        raise FileNotFoundError(f"Missing splits_kfold.csv!")
    if len(val_files) < 2466:
        print(f"Warning: Expected 2,466 validation images, found {len(val_files)}.")

    df_splits = pd.read_csv(splits_csv)

    # 1. Checkpoint Resolution
    r18_ckpts, cnxt_ckpts = find_or_train_checkpoints(
        df_splits, id_to_path, synth_trad_csv, synth_callig_csv
    )

    # 2. Build Datasets & Loaders
    train_proto_df = df_splits.copy()
    if synth_trad_csv and os.path.exists(synth_trad_csv):
        df_trad = pd.read_csv(synth_trad_csv)
        train_proto_df = pd.concat([train_proto_df, df_trad], ignore_index=True)

    train_proto_ds = ChineseMNISTDataset(train_proto_df, id_to_path, is_train=False)
    train_proto_loader = DataLoader(train_proto_ds, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

    val_df = pd.DataFrame({"id": [os.path.splitext(os.path.basename(f))[0] for f in val_files]})
    val_ds = ChineseMNISTDataset(val_df, id_to_path, is_train=False)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

    val_ids = [str(i) for i in val_df["id"]]

    # 3. Multi-Fold STM Adaptation for ResNet-18 CBAM
    print("\n" + "=" * 80)
    print("💠 STAGE 1: RESNET-18 CBAM STYLE TRANSFER MAPPING (STM)")
    print("=" * 80)

    r18_probs_unadapted = []
    r18_probs_adapted = []
    r18_A_list, r18_b_list = [], []

    for fold_idx, ckpt_p in enumerate(r18_ckpts):
        print(f"\n▶ Loading ResNet-18 CBAM (Fold {fold_idx}) from: {ckpt_p}")
        model = ResNet18_CBAM(in_channels=1, num_classes=15, feature_dim=256, use_mixstyle=False).to(DEVICE)
        ckpt = torch.load(ckpt_p, map_location=DEVICE)
        state_dict = ckpt.get("model_state_dict", ckpt)
        load_state_dict_flexible(model, state_dict)

        # Compute source prototypes
        print(f"  • Calculating training prototypes mu_k for ResNet-18...")
        mu_r18 = compute_class_prototypes(model, train_proto_loader, num_classes=15, feature_dim=256)

        # Run closed-form STM adaptation
        p0, p_stm, A, b = run_stm_self_training_pipeline(
            model=model,
            model_name=f"ResNet-18 CBAM (Fold {fold_idx})",
            val_loader=val_loader,
            mu_prototypes=mu_r18,
            num_iters=4,
            beta=50.0,
            gamma=1.0
        )
        r18_probs_unadapted.append(p0)
        r18_probs_adapted.append(p_stm)
        r18_A_list.append(A)
        r18_b_list.append(b)

    avg_p0_r18 = np.mean(r18_probs_unadapted, axis=0)
    avg_p_stm_r18 = np.mean(r18_probs_adapted, axis=0)

    # 4. Multi-Fold STM Adaptation for ConvNeXt-Femto
    print("\n" + "=" * 80)
    print("💠 STAGE 2: CONVNEXT-FEMTO STYLE TRANSFER MAPPING (STM)")
    print("=" * 80)

    cnxt_probs_unadapted = []
    cnxt_probs_adapted = []
    cnxt_A_list, cnxt_b_list = [], []

    for fold_idx, ckpt_p in enumerate(cnxt_ckpts):
        print(f"\n▶ Loading ConvNeXt-Femto (Fold {fold_idx}) from: {ckpt_p}")
        model = ConvNeXt_Femto(in_channels=1, num_classes=15, feature_dim=256).to(DEVICE)
        ckpt = torch.load(ckpt_p, map_location=DEVICE)
        state_dict = ckpt.get("model_state_dict", ckpt)
        load_state_dict_flexible(model, state_dict)

        print(f"  • Calculating training prototypes mu_k for ConvNeXt-Femto...")
        mu_cnxt = compute_class_prototypes(model, train_proto_loader, num_classes=15, feature_dim=256)

        p0, p_stm, A, b = run_stm_self_training_pipeline(
            model=model,
            model_name=f"ConvNeXt-Femto (Fold {fold_idx})",
            val_loader=val_loader,
            mu_prototypes=mu_cnxt,
            num_iters=4,
            beta=50.0,
            gamma=1.0
        )
        cnxt_probs_unadapted.append(p0)
        cnxt_probs_adapted.append(p_stm)
        cnxt_A_list.append(A)
        cnxt_b_list.append(b)

    avg_p0_cnxt = np.mean(cnxt_probs_unadapted, axis=0)
    avg_p_stm_cnxt = np.mean(cnxt_probs_adapted, axis=0)

    # 5. Dual-Architecture Ensemble Integration
    print("\n" + "=" * 80)
    print("🏆 STAGE 3: DUAL-ARCHITECTURE STM ENSEMBLE SYNTHESIS")
    print("=" * 80)

    blend_unadapted = 0.5 * avg_p0_r18 + 0.5 * avg_p0_cnxt
    blend_adapted = 0.5 * avg_p_stm_r18 + 0.5 * avg_p_stm_cnxt

    pred_unadapted = np.argmax(blend_unadapted, axis=1)
    pred_adapted = np.argmax(blend_adapted, axis=1)

    discrepancies = np.sum(pred_unadapted != pred_adapted)
    print(f"[*] Total Pseudo-Label Refinements via STM : {discrepancies}/{len(pred_adapted)} ({discrepancies/len(pred_adapted)*100:.2f}%)")
    print(f"[*] Adapted Ensemble Class Distribution:")
    for c in range(15):
        print(f"    Class {c:2d}: {(pred_adapted == c).sum():4d} samples")

    # 6. Export Submission Artifacts
    print("\n" + "=" * 80)
    print("📦 STAGE 4: EXPORTING SUBMISSION ARTIFACTS")
    print("=" * 80)

    # Save CSV
    sub_df = pd.DataFrame({"id": val_ids, "label": pred_adapted})
    csv_path = os.path.join(WORKING_DIR, "submission_stm.csv")
    sub_df.to_csv(csv_path, index=False)
    print(f"  • Saved CSV Submission to : {csv_path}")

    # Save ZIP (containing predictions.csv for CodaBench evaluation)
    zip_path = os.path.join(WORKING_DIR, "sub_stm.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path, arcname="predictions.csv")
    print(f"  • Saved ZIP Submission to : {zip_path} ({os.path.getsize(zip_path):,} bytes)")

    # Save Numerical Probability & Adaptation Matrices
    probs_path = os.path.join(WORKING_DIR, "probabilities_stm.npy")
    np.save(probs_path, blend_adapted)
    print(f"  • Saved Probabilities to  : {probs_path}")

    matrices_path = os.path.join(WORKING_DIR, "stm_matrices.npz")
    np.savez(
        matrices_path,
        r18_A=np.mean(r18_A_list, axis=0),
        r18_b=np.mean(r18_b_list, axis=0),
        cnxt_A=np.mean(cnxt_A_list, axis=0),
        cnxt_b=np.mean(cnxt_b_list, axis=0)
    )
    print(f"  • Saved STM Matrices to   : {matrices_path}")

    print("\n" + "=" * 80)
    print("🎉 STYLE TRANSFER MAPPING (STM) PIPELINE EXECUTED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
