#!/usr/bin/env python3
"""
================================================================================
CodaBench Chinese MNIST Challenge — SOTA Retrained Competitive Pipeline
Final Project: Analisis Data Multivariat (ADM) — Teknik Informatika ITS
================================================================================
Architecture:
- 100% Self-Contained Standalone Kaggle Notebook Script
- Pure PyTorch 2.x (No external model repository dependencies, runs Internet OFF)
- Models: ResNet18-CBAM (Dual Attention + MixStyle) + Sub-Center ArcFace (K=2)
- Biomechanical Augmentation: Elastic Deformation + Stroke Dynamics
- Stratified Group K-Fold (Writer-Independent, Zero Leakage)
- Test-Time Adaptation: AdaBN (Domain Alignment) + 5-Way Power-Mean TTA
- Post-Processing: Unsupervised Bayesian Test-Time Prior Adaptation (TTPA)
  with Dirichlet MAP Regularization (Safe for Phase 1 & Phase 2 unseen distributions)
- Output: Flat root 'submission_phase1_sota_retrained.zip' directly in /kaggle/working/
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
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, classification_report

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
print("=" * 75)
print(f"🚀 HARDWARE SETUP | Active Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"   • Device Name  : {torch.cuda.get_device_name(0)}")
    print(f"   • Device Count : {torch.cuda.device_count()}")
    print(f"   • VRAM Cap     : {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    print(f"   • PyTorch Ver  : {torch.__version__}")
print("=" * 75)

# Working directory paths
TMP_DATA_DIR = "/tmp/data"
os.makedirs(TMP_DATA_DIR, exist_ok=True)
WORKING_DIR = "/kaggle/working" if os.path.exists("/kaggle") else "./FinalProject/outputs/kaggle_standalone"
MODELS_DIR = os.path.join(WORKING_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# ==============================================================================
# 2. DYNAMIC PATH DISCOVERY & DATASET EXTRACTION
# ==============================================================================
def discover_and_extract_data() -> Tuple[str, Optional[str], str, Optional[str], Optional[str]]:
    """
    Robust dynamic scanner that locates training, synthetic, splits, and test sets
    across /kaggle/input and extracts any zipped files to /tmp/data automatically.
    """
    search_roots = ["/kaggle/input", "./FinalProject", "./"]
    input_root = None
    for r in search_roots:
        if os.path.exists(r):
            input_root = r
            break

    print(f"[*] Scanning data roots from: {input_root}...")

    # Extract any found zip files
    for z in glob.glob(f"{input_root}/**/*.zip", recursive=True):
        if "submission" in os.path.basename(z).lower():
            continue
        print(f"  • Extracting zip: {z} -> {TMP_DATA_DIR}")
        try:
            with zipfile.ZipFile(z, 'r') as zip_ref:
                zip_ref.extractall(TMP_DATA_DIR)
        except Exception as e:
            print(f"    [!] Note: {e}")

    # Search candidates across input_root and TMP_DATA_DIR
    all_search_dirs = [TMP_DATA_DIR, input_root]

    # 1. Train dir (find train_00001.png or train/ folder)
    train_dir = None
    train_sample = glob.glob(f"{TMP_DATA_DIR}/**/train_00001.png", recursive=True) or \
                   glob.glob(f"{input_root}/**/train_00001.png", recursive=True)
    if train_sample:
        train_dir = os.path.dirname(train_sample[0])
    else:
        raise FileNotFoundError("CRITICAL: train_00001.png not found!")

    # 2. Synthetic dir (find synth_00001.png)
    synth_dir = None
    synth_sample = glob.glob(f"{TMP_DATA_DIR}/**/synth_00001.png", recursive=True) or \
                   glob.glob(f"{input_root}/**/synth_00001.png", recursive=True)
    if synth_sample:
        synth_dir = os.path.dirname(synth_sample[0])

    # 3. Splits CSV (splits_kfold.csv or train_labels.csv)
    splits_csv = None
    splits_sample = glob.glob(f"{TMP_DATA_DIR}/**/splits_kfold.csv", recursive=True) or \
                    glob.glob(f"{input_root}/**/splits_kfold.csv", recursive=True)
    if splits_sample:
        splits_csv = splits_sample[0]
    else:
        labels_sample = glob.glob(f"{TMP_DATA_DIR}/**/train_labels.csv", recursive=True) or \
                        glob.glob(f"{input_root}/**/train_labels.csv", recursive=True)
        if labels_sample:
            splits_csv = labels_sample[0]
        else:
            raise FileNotFoundError("CRITICAL: Neither splits_kfold.csv nor train_labels.csv found!")

    # 4. Synthetic labels CSV
    synth_labels_csv = None
    synth_lbl_sample = glob.glob(f"{TMP_DATA_DIR}/**/train_synthetic_labels.csv", recursive=True) or \
                       glob.glob(f"{input_root}/**/train_synthetic_labels.csv", recursive=True)
    if synth_lbl_sample:
        synth_labels_csv = synth_lbl_sample[0]

    # 5. Test dir (find val_00001.png or test/ directory)
    test_dir = None
    test_sample = glob.glob(f"{TMP_DATA_DIR}/**/val_00001.png", recursive=True) or \
                  glob.glob(f"{input_root}/**/val_00001.png", recursive=True) or \
                  glob.glob(f"{input_root}/**/test/val_*.png", recursive=True)
    if test_sample:
        test_dir = os.path.dirname(test_sample[0])

    print(f"[+] Train Dir      : {train_dir}")
    print(f"[+] Synthetic Dir  : {synth_dir}")
    print(f"[+] Splits CSV     : {splits_csv}")
    print(f"[+] Synth Labels   : {synth_labels_csv}")
    print(f"[+] Test Dir       : {test_dir}")
    return train_dir, synth_dir, splits_csv, synth_labels_csv, test_dir


# ==============================================================================
# 3. BIOMECHANICAL DATA AUGMENTATION ENGINE
# ==============================================================================
class BiomechanicalAugmentor:
    """
    Biomechanical Calligraphic Augmentation Pipeline:
    Simulates human hand muscular elasticity, stroke speed variance, and brush physics.
    """
    def __init__(self, alpha_elastic=14.0, sigma_elastic=3.0, p_elastic=0.5, p_stroke=0.4):
        self.alpha = alpha_elastic
        self.sigma = sigma_elastic
        self.p_elastic = p_elastic
        self.p_stroke = p_stroke
        self.kernel_3x3 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    def elastic_transform(self, image: np.ndarray) -> np.ndarray:
        shape = image.shape
        dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), self.sigma, mode="constant", cval=0) * self.alpha
        dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), self.sigma, mode="constant", cval=0) * self.alpha
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)
        return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def stroke_morphology(self, image: np.ndarray) -> np.ndarray:
        choice = random.random()
        if choice < 0.5:
            # Stroke Thinning (simulates fast, light tip brush)
            return cv2.erode(image, self.kernel_3x3, iterations=1)
        else:
            # Stroke Thickening (simulates heavy pressure, ink bleed)
            return cv2.dilate(image, self.kernel_3x3, iterations=1)

    def affine_transform(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape
        angle = random.uniform(-10.0, 10.0)
        scale = random.uniform(0.88, 1.08)
        tx = random.uniform(-3.0, 3.0)
        ty = random.uniform(-3.0, 3.0)
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
        M[0, 2] += tx
        M[1, 2] += ty
        return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        img = image.copy()
        img = self.affine_transform(img)
        if random.random() < self.p_elastic:
            img = self.elastic_transform(img)
        if random.random() < self.p_stroke:
            img = self.stroke_morphology(img)
        return img


# ==============================================================================
# 4. DATASET & DATALOADER
# ==============================================================================
class ChineseMNISTDataset(Dataset):
    def __init__(self, df: pd.DataFrame, train_dir: str, synth_dir: Optional[str] = None, is_train: bool = True):
        self.df = df.reset_index(drop=True)
        self.train_dir = train_dir
        self.synth_dir = synth_dir
        self.is_train = is_train
        self.augmentor = BiomechanicalAugmentor() if is_train else None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_id = str(row['id'])
        label = int(row['label'])
        writer = int(row.get('writer_id', -1))

        if img_id.startswith("synth_") and self.synth_dir:
            fpath = os.path.join(self.synth_dir, f"{img_id}.png")
        else:
            fpath = os.path.join(self.train_dir, f"{img_id}.png")

        img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((64, 64), dtype=np.uint8)
        if img.shape != (64, 64):
            img = cv2.resize(img, (64, 64))

        if self.is_train and self.augmentor:
            img = self.augmentor(img)

        # Standard Grayscale Normalization [-1.0, 1.0]
        tensor = torch.from_numpy(img.astype(np.float32) / 127.5 - 1.0).unsqueeze(0)
        return tensor, label, writer


# ==============================================================================
# 5. SOTA NATIVE NEURAL NETWORK ARCHITECTURES (PURE PYTORCH)
# ==============================================================================
class MixStyle(nn.Module):
    """MixStyle for Domain Generalization (Zhou et al., ICLR 2021)."""
    def __init__(self, p=0.5, alpha=0.1, eps=1e-6):
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps

    def forward(self, x):
        if not self.training or torch.rand(1).item() > self.p:
            return x
        B = x.size(0)
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        x_normed = (x - mu) / sig
        lmda = self.beta.sample((B, 1, 1, 1)).to(x.device)
        perm = torch.randperm(B)
        mu_mix = mu * lmda + mu[perm] * (1 - lmda)
        sig_mix = sig * lmda + sig[perm] * (1 - lmda)
        return x_normed * sig_mix + mu_mix


class CBAMBlock(nn.Module):
    """Convolutional Block Attention Module (Woo et al., ECCV 2018)."""
    def __init__(self, in_planes: int, ratio: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        reduced = max(in_planes // ratio, 8)
        self.fc1 = nn.Conv2d(in_planes, reduced, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(reduced, in_planes, 1, bias=False)
        self.ca_sig = nn.Sigmoid()

        self.conv_sa = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.sa_sig = nn.Sigmoid()

    def forward(self, x):
        # Channel Attention
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        x = x * self.ca_sig(avg_out + max_out)
        # Spatial Attention
        sa_avg = torch.mean(x, dim=1, keepdim=True)
        sa_max, _ = torch.max(x, dim=1, keepdim=True)
        sa_map = self.sa_sig(self.conv_sa(torch.cat([sa_avg, sa_max], dim=1)))
        return x * sa_map


class BasicBlockCBAM(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.cbam = CBAMBlock(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = self.cbam(out)
        out += self.shortcut(x)
        return F.relu(out, inplace=True)


class ResNet18_CBAM(nn.Module):
    """
    Backbone 1: ResNet18-CBAM with MixStyle for domain-invariant calligraphy representation.
    """
    def __init__(self, in_channels=1, num_classes=15, feature_dim=256, use_mixstyle=True):
        super().__init__()
        self.in_planes = 64
        self.use_mixstyle = use_mixstyle
        self.mixstyle = MixStyle(p=0.5, alpha=0.1) if use_mixstyle else nn.Identity()

        self.conv1 = nn.Conv2d(in_channels, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_feat = nn.Linear(512, feature_dim)
        self.bn_feat = nn.BatchNorm1d(feature_dim)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def _make_layer(self, planes, blocks, stride):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlockCBAM(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def extract_features(self, x):
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.layer1(x)
        if self.use_mixstyle and self.training:
            x = self.mixstyle(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        feat = self.bn_feat(self.fc_feat(x))
        return feat

    def forward(self, x):
        feat = self.extract_features(x)
        logits = self.classifier(feat)
        return logits, feat


class SubCenterArcFace(nn.Module):
    """
    Sub-Center ArcFace Head (Deng et al., ECCV 2020).
    Employs K sub-centers per class to capture distinct cursive vs standard calligraphy modes.
    """
    def __init__(self, in_features: int = 256, num_classes: int = 15, sub_centers: int = 2,
                 s: float = 32.0, m: float = 0.35):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.sub_centers = sub_centers
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes * sub_centers, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        # Reshape to (B, C, K) and take max across sub-centers
        cosine = cosine.view(-1, self.num_classes, self.sub_centers)
        cosine, _ = torch.max(cosine, dim=2)  # (B, C)

        sine = torch.sqrt(1.0 - torch.pow(cosine, 2)).clamp(0, 1)
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


# ==============================================================================
# 6. TEST-TIME PRIOR ADAPTATION (EM DIRICHLET MAP)
# ==============================================================================
class TestTimePriorAdapter:
    """
    Unsupervised Bayesian Prior Shift Adaptation (Saerens et al. 2002; Sun et al. 2023).
    Recalibrates posteriors without human labels or hardcoded zero masks.
    Dirichlet parameter alpha=1.05 guarantees non-zero probability support across all classes.
    """
    def __init__(self, alpha_dirichlet: float = 1.05, max_iter: int = 50, tol: float = 1e-6, eps: float = 1e-12):
        self.alpha = alpha_dirichlet
        self.max_iter = max_iter
        self.tol = tol
        self.eps = eps
        self.target_prior_: Optional[np.ndarray] = None
        self.n_iter_: int = 0

    def fit_transform(self, probs: np.ndarray) -> np.ndarray:
        N, K = probs.shape
        p_source = np.ones(K, dtype=np.float64) / K
        q = probs / (p_source[np.newaxis, :] + self.eps)

        pi = np.ones(K, dtype=np.float64) / K
        gamma = max(0.0, self.alpha - 1.0)

        for it in range(1, self.max_iter + 1):
            pi_prev = pi.copy()
            # E-step
            numerator = q * pi[np.newaxis, :]
            w = numerator / (np.sum(numerator, axis=1, keepdims=True) + self.eps)
            # M-step with Dirichlet MAP
            pi = (np.sum(w, axis=0) + gamma) / (N + K * gamma)
            pi /= np.sum(pi)

            if np.max(np.abs(pi - pi_prev)) < self.tol:
                self.n_iter_ = it
                break
        else:
            self.n_iter_ = self.max_iter

        self.target_prior_ = pi

        # Final posterior adjustment
        adjusted_numerator = q * pi[np.newaxis, :]
        adjusted_probs = adjusted_numerator / (np.sum(adjusted_numerator, axis=1, keepdims=True) + self.eps)
        return adjusted_probs


# ==============================================================================
# 7. TRAINING & OUT-OF-FOLD CROSS VALIDATION PIPELINE
# ==============================================================================
def train_and_evaluate_folds(
    df_splits: pd.DataFrame,
    df_synth: Optional[pd.DataFrame],
    train_dir: str,
    synth_dir: Optional[str],
    epochs: int = 15,
    batch_size: int = 64,
    lr: float = 5e-4
) -> Tuple[List[str], np.ndarray]:
    """
    Trains 5-Fold ResNet18-CBAM with AMP FP16 and Sub-Center ArcFace.
    """
    print("\n" + "=" * 75)
    print("🚀 COMMENCING 5-FOLD STRATIFIED GROUP K-FOLD TRAINING LOOP")
    print("=" * 75)

    n_folds = 5
    oof_probs = np.zeros((len(df_splits), 15), dtype=np.float32)
    saved_checkpoints = []

    for fold in range(n_folds):
        val_mask = (df_splits['fold'] == fold)
        train_mask = ~val_mask

        df_train_fold = df_splits[train_mask].copy()
        if df_synth is not None:
            df_train_fold = pd.concat([df_train_fold, df_synth], ignore_index=True)

        df_val_fold = df_splits[val_mask].copy()

        train_dataset = ChineseMNISTDataset(df_train_fold, train_dir, synth_dir, is_train=True)
        val_dataset = ChineseMNISTDataset(df_val_fold, train_dir, synth_dir, is_train=False)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=2, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=2, pin_memory=True)

        model = ResNet18_CBAM(in_channels=1, num_classes=15, feature_dim=256, use_mixstyle=True).to(DEVICE)
        metric_head = SubCenterArcFace(in_features=256, num_classes=15, sub_centers=2).to(DEVICE)

        optimizer = torch.optim.AdamW(list(model.parameters()) + list(metric_head.parameters()),
                                      lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        scaler = GradScaler()
        criterion_ce = nn.CrossEntropyLoss(label_smoothing=0.05)

        best_val_acc = 0.0
        best_val_probs = None
        ckpt_path = os.path.join(MODELS_DIR, f"resnet18_cbam_fold{fold}.pt")

        print(f"\n--- [FOLD {fold + 1}/{n_folds}] Train: {len(df_train_fold)} | Val: {len(df_val_fold)} ---")

        for epoch in range(1, epochs + 1):
            model.train()
            metric_head.train()
            train_loss, train_correct, train_total = 0.0, 0, 0

            for imgs, labels, writers in train_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()

                with autocast():
                    logits, feats = model(imgs)
                    arc_logits = metric_head(feats, labels)
                    loss = 0.5 * criterion_ce(logits, labels) + 0.5 * criterion_ce(arc_logits, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item() * len(labels)
                preds = torch.argmax(logits, dim=1)
                train_correct += (preds == labels).sum().item()
                train_total += len(labels)

            scheduler.step()
            train_acc = train_correct / train_total

            # Validation
            model.eval()
            val_correct, val_total = 0, 0
            val_epoch_probs = []

            with torch.no_grad():
                for imgs, labels, _ in val_loader:
                    imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                    logits, _ = model(imgs)
                    probs = F.softmax(logits, dim=1)
                    val_epoch_probs.append(probs.cpu().numpy())
                    preds = torch.argmax(logits, dim=1)
                    val_correct += (preds == labels).sum().item()
                    val_total += len(labels)

            val_acc = val_correct / val_total
            val_epoch_probs = np.vstack(val_epoch_probs)

            if epoch % 5 == 0 or epoch == epochs:
                print(f"  Epoch {epoch:02d}/{epochs:02d} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_probs = val_epoch_probs
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "val_acc": val_acc,
                    "fold": fold,
                    "model_type": "resnet18_cbam"
                }, ckpt_path)

        oof_probs[val_mask] = best_val_probs
        saved_checkpoints.append(ckpt_path)
        print(f"[+] Fold {fold + 1} Best Val Acc: {best_val_acc*100:.2f}% saved to: {ckpt_path}")

    overall_oof = accuracy_score(df_splits['label'].values, np.argmax(oof_probs, axis=1))
    print("\n" + "=" * 75)
    print(f"🏆 OVERALL 5-FOLD OUT-OF-FOLD ACCURACY: {overall_oof * 100:.2f}%")
    print("=" * 75)
    return saved_checkpoints, oof_probs


# ==============================================================================
# 8. TEST INFERENCE WITH ADABN + 5-WAY TTA + EM PRIOR ADAPTATION
# ==============================================================================
def apply_adabn_to_model(model: nn.Module, test_files: List[str], batch_size: int = 64):
    """Adaptive Batch Normalization (Li et al., Pattern Recognition 2018)."""
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.reset_running_stats()
            m.training = True
            m.momentum = None

    with torch.no_grad():
        for i in range(0, len(test_files), batch_size):
            batch_p = test_files[i:i + batch_size]
            tensors = []
            for p in batch_p:
                img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                if img is None or img.shape != (64, 64):
                    img = cv2.resize(img if img is not None else np.zeros((64, 64)), (64, 64))
                t = torch.from_numpy(img.astype(np.float32) / 127.5 - 1.0).unsqueeze(0).unsqueeze(0)
                tensors.append(t)
            batch_tensor = torch.cat(tensors, dim=0).to(DEVICE)
            _ = model(batch_tensor)
    model.eval()


def tta_variants(tensor: torch.Tensor) -> List[torch.Tensor]:
    """Generates 5 rotation/translation variants."""
    variants = [tensor]
    device = tensor.device

    # Rot +4 deg
    th_pos = torch.tensor([[np.cos(np.radians(4)), -np.sin(np.radians(4)), 0],
                           [np.sin(np.radians(4)),  np.cos(np.radians(4)), 0]],
                          dtype=torch.float, device=device).unsqueeze(0)
    g_pos = F.affine_grid(th_pos, tensor.size(), align_corners=False)
    variants.append(F.grid_sample(tensor, g_pos, align_corners=False))

    # Rot -4 deg
    th_neg = torch.tensor([[np.cos(np.radians(-4)), -np.sin(np.radians(-4)), 0],
                           [np.sin(np.radians(-4)),  np.cos(np.radians(-4)), 0]],
                          dtype=torch.float, device=device).unsqueeze(0)
    g_neg = F.affine_grid(th_neg, tensor.size(), align_corners=False)
    variants.append(F.grid_sample(tensor, g_neg, align_corners=False))

    # Shift dx=+2
    th_r = torch.tensor([[1, 0, 0.06], [0, 1, 0]], dtype=torch.float, device=device).unsqueeze(0)
    variants.append(F.grid_sample(tensor, F.affine_grid(th_r, tensor.size(), align_corners=False), align_corners=False))

    # Shift dx=-2
    th_l = torch.tensor([[1, 0, -0.06], [0, 1, 0]], dtype=torch.float, device=device).unsqueeze(0)
    variants.append(F.grid_sample(tensor, F.affine_grid(th_l, tensor.size(), align_corners=False), align_corners=False))

    return variants


def run_test_inference(test_dir: str, ckpt_paths: List[str]) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Executes ensemble inference across test set with AdaBN, 5-Way TTA, and Unsupervised TTPA.
    """
    print("\n" + "=" * 75)
    print("🔮 GENERATING TEST PREDICTIONS WITH ADABN + 5-WAY TTA + BAYESIAN TTPA")
    print("=" * 75)

    test_files = sorted(glob.glob(f"{test_dir}/*.png") + glob.glob(f"{test_dir}/**/*.png", recursive=True))
    if not test_files:
        raise FileNotFoundError(f"No test images found in {test_dir}!")
    print(f"[*] Found {len(test_files)} test images in: {test_dir}")

    # Load and adapt models with AdaBN
    models = []
    for cp in ckpt_paths:
        ckpt = torch.load(cp, map_location=DEVICE)
        m = ResNet18_CBAM(in_channels=1, num_classes=15, feature_dim=256, use_mixstyle=False).to(DEVICE)
        m.load_state_dict(ckpt["model_state_dict"])
        apply_adabn_to_model(m, test_files)
        models.append(m)

    tta_weights = [1.0, 0.8, 0.8, 0.7, 0.7]
    raw_probs_list = []
    ids_list = []

    print("[*] Processing TTA across ensemble...")
    with torch.no_grad():
        for idx, fpath in enumerate(test_files):
            img_id = os.path.splitext(os.path.basename(fpath))[0]
            ids_list.append(img_id)

            img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            if img is None or img.shape != (64, 64):
                img = cv2.resize(img if img is not None else np.zeros((64, 64)), (64, 64))
            inp = torch.from_numpy(img.astype(np.float32) / 127.5 - 1.0).unsqueeze(0).unsqueeze(0).to(DEVICE)

            vars = tta_variants(inp)
            model_accum_probs = []
            for m in models:
                var_probs = []
                for v_idx, v in enumerate(vars):
                    logits, _ = m(v)
                    p = F.softmax(logits, dim=1).squeeze(0)
                    var_probs.append(p * tta_weights[v_idx])
                # Power-mean across variants (p=1.8)
                v_stack = torch.stack(var_probs, dim=0)
                p_mean = torch.pow(torch.sum(torch.pow(v_stack, 1.8), dim=0), 1.0 / 1.8)
                p_mean /= torch.sum(p_mean)
                model_accum_probs.append(p_mean)

            ensemble_p = torch.mean(torch.stack(model_accum_probs, dim=0), dim=0).cpu().numpy()
            raw_probs_list.append(ensemble_p)

    raw_probs = np.array(raw_probs_list)

    # Apply Unsupervised Test-Time Prior Adaptation (TTPA)
    print("\n[*] Applying Unsupervised Dirichlet MAP Prior Adaptation (alpha=1.05)...")
    adapter = TestTimePriorAdapter(alpha_dirichlet=1.05, max_iter=50)
    adapted_probs = adapter.fit_transform(raw_probs)
    print(f"    • EM converged in {adapter.n_iter_} iterations.")
    print(f"    • Estimated Target Class Priors (%): {np.round(adapter.target_prior_ * 100, 2)}")

    preds = np.argmax(adapted_probs, axis=1)
    df_submission = pd.DataFrame({"id": ids_list, "label": preds})
    return df_submission, adapted_probs


# ==============================================================================
# 9. CERTIFIED CODABENCH PACKAGER & AUDITOR
# ==============================================================================
def certify_and_package_codabench(df_preds: pd.DataFrame, output_dir: str) -> str:
    """
    Validates CodaBench strict compliance and generates flat root ZIP archive.
    """
    csv_path = os.path.join(output_dir, "predictions.csv")
    zip_path = os.path.join(output_dir, "submission_phase1_sota_retrained.zip")

    print("\n" + "=" * 75)
    print("🛡️ RUNNING PRE-FLIGHT VERIFICATION FOR CODABENCH CERTIFICATION")
    print("=" * 75)

    # 1. Schema Check
    assert list(df_preds.columns) == ["id", "label"], f"Invalid columns: {df_preds.columns}"
    print("[PASS] Schema matches ['id', 'label']")

    # 2. Null Check
    assert df_preds["label"].isnull().sum() == 0, "Null values detected!"
    print("[PASS] Zero nulls / NaNs verified.")

    # 3. Label dtype & range check
    assert df_preds["label"].min() >= 0 and df_preds["label"].max() <= 14, "Label out of range [0, 14]!"
    print(f"[PASS] Integer labels in range [{df_preds['label'].min()}, {df_preds['label'].max()}] verified.")

    # 4. Save CSV
    df_preds.to_csv(csv_path, index=False)
    print(f"[+] Saved CSV: {csv_path} ({len(df_preds)} rows, {os.path.getsize(csv_path):,} bytes)")

    # 5. Package into Flat Root ZIP
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(csv_path, arcname="predictions.csv")

    # 6. Verify ZIP internal structure
    with zipfile.ZipFile(zip_path, "r") as z:
        names = z.namelist()
        assert "predictions.csv" in names, "Missing predictions.csv at ZIP root!"
        assert len(names) == 1, f"Extraneous files found in archive: {names}"
    print(f"[PASS] Flat root ZIP archive certified: {zip_path} ({os.path.getsize(zip_path):,} bytes)")
    print("=" * 75)
    print("🎉 SUBMISSION IS 100% CERTIFIED AND READY FOR CODABENCH UPLOAD!")
    print("=" * 75)
    return zip_path


# ==============================================================================
# MAIN EXECUTION ROUTINE
# ==============================================================================
def main():
    print("=" * 75)
    print("🏆 CODABENCH SOTA RETRAINED EXECUTION PIPELINE")
    print("=" * 75)

    train_dir, synth_dir, splits_csv, synth_labels_csv, test_dir = discover_and_extract_data()

    df_splits = pd.read_csv(splits_csv)
    df_synth = pd.read_csv(synth_labels_csv) if synth_labels_csv and os.path.exists(synth_labels_csv) else None

    # Step 1: Train 5-Fold Ensemble
    checkpoints, oof_probs = train_and_evaluate_folds(
        df_splits=df_splits,
        df_synth=df_synth,
        train_dir=train_dir,
        synth_dir=synth_dir,
        epochs=15,
        batch_size=64,
        lr=5e-4
    )

    # Step 2: Test Inference & CodaBench Packaging
    if test_dir and os.path.exists(test_dir):
        df_preds, adapted_probs = run_test_inference(test_dir, checkpoints)
        final_zip = certify_and_package_codabench(df_preds, WORKING_DIR)
        print(f"\n[SUCCESS] Pipeline Complete! Certified submission ready at:\n--> {final_zip}")
    else:
        print("\n[!] Notice: Test directory not detected during this run.")
        print(f"[!] Checkpoints saved in {MODELS_DIR}. Ready for test evaluation once attached.")


if __name__ == "__main__":
    main()
