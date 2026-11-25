#!/usr/bin/env python3
"""
================================================================================
CodaBench Chinese MNIST Challenge — SOTA DenseNet-121 CBAM 5-Fold Pipeline (v2)
Dataset: 15,100 images (Kaishu + 7 Google Fonts Caoshu/Xingshu/Maobi + Biomechanical Aug)
Hardware: Kaggle GPU (Tesla T4 / P100 with 16GB VRAM)
Zero Prior Tampering — Strictly Inductive Deep Learning
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

# ------------------------------------------------------------------------------
# 1. HARDWARE & REPRODUCIBILITY CONFIGURATION
# ------------------------------------------------------------------------------
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
print(f"🚀 KAGGLE HARDWARE SETUP | Active Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"   • Device Name  : {torch.cuda.get_device_name(0)}")
    print(f"   • VRAM Cap     : {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    print(f"   • PyTorch Ver  : {torch.__version__}")
print("=" * 75)

TMP_DATA_DIR = "/tmp/data"
os.makedirs(TMP_DATA_DIR, exist_ok=True)
WORKING_DIR = "/kaggle/working" if os.path.exists("/kaggle") else "./FinalProject/outputs/kaggle_densenet"
MODELS_DIR = os.path.join(WORKING_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# ------------------------------------------------------------------------------
# 2. DYNAMIC DATASET SCANNING & EXTRACTION
# ------------------------------------------------------------------------------
def setup_datasets():
    print("\n[*] Scanning & extracting dataset files in search paths...")
    if os.path.exists("/kaggle/input"):
        search_roots = ["/kaggle/input"]
    elif os.path.exists("./FinalProject/data"):
        search_roots = ["./FinalProject/data"]
    elif os.path.exists("./data"):
        search_roots = ["./data"]
    else:
        search_roots = ["."]

    # 1. Extract any zip archives found across search_roots into TMP_DATA_DIR
    for r in search_roots:
        for z in glob.glob(f"{r}/**/*.zip", recursive=True):
            zname = os.path.basename(z).lower()
            if "submission" in zname:
                continue
            print(f"  • Extracting zip: {z} -> {TMP_DATA_DIR}")
            try:
                with zipfile.ZipFile(z, 'r') as zip_ref:
                    zip_ref.extractall(TMP_DATA_DIR)
            except Exception as e:
                print(f"    [!] Error extracting {z}: {e}")

    # 2. Build complete index of all PNG files (id_to_path) across TMP_DATA_DIR and search_roots
    id_to_path = {}
    scan_dirs = [TMP_DATA_DIR] + search_roots
    for d in scan_dirs:
        for p in glob.glob(f"{d}/**/*.png", recursive=True):
            stem = os.path.splitext(os.path.basename(p))[0]
            if stem not in id_to_path:
                id_to_path[stem] = p

    print(f"  [+] Indexed {len(id_to_path):,} unique image files across search paths.")

    # 3. Locate CSV metadata files
    def find_file(filename):
        for d in [TMP_DATA_DIR] + search_roots:
            matches = glob.glob(f"{d}/**/{filename}", recursive=True)
            if matches:
                return matches[0]
        return None

    splits_csv = find_file("splits_kfold.csv")
    if not splits_csv:
        raise FileNotFoundError("splits_kfold.csv not found anywhere!")
    print(f"  [+] Found splits_kfold.csv: {splits_csv}")

    synth_trad_csv = find_file("train_synthetic_labels.csv")
    synth_callig_csv = find_file("train_synthetic_calligraphy_v2.csv")
    if not synth_callig_csv:
        synth_callig_csv = find_file("train_synthetic_calligraphy.csv")

    print(f"  [+] Found Traditional CSV : {synth_trad_csv}")
    print(f"  [+] Found Calligraphy CSV : {synth_callig_csv}")

    # 4. Locate validation files (val_00001.png to val_02466.png)
    val_files = [p for k, p in id_to_path.items() if k.startswith("val_") and k[4:].isdigit()]
    if not val_files:
        val_files = [p for k, p in id_to_path.items() if "val" in k.lower() or "test" in k.lower()]
    val_files = sorted(val_files, key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x)))) if any(c.isdigit() for c in os.path.basename(x)) else 0)
    print(f"  [+] Located {len(val_files)} validation images.")

    return splits_csv, synth_trad_csv, synth_callig_csv, id_to_path, val_files


# ------------------------------------------------------------------------------
# 3. BIOMECHANICAL DATA AUGMENTATION MODULE
# ------------------------------------------------------------------------------
class ElasticDistortion:
    def __init__(self, alpha_range=(6.0, 9.5), sigma_range=(3.6, 4.4), p=0.45):
        self.alpha_range = alpha_range
        self.sigma_range = sigma_range
        self.p = p

    def __call__(self, img_np: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return img_np
        H, W = img_np.shape[:2]
        alpha = random.uniform(*self.alpha_range)
        sigma = random.uniform(*self.sigma_range)
        noise_x = np.random.uniform(-1.0, 1.0, size=(H, W))
        noise_y = np.random.uniform(-1.0, 1.0, size=(H, W))
        dx = gaussian_filter(noise_x, sigma, mode="constant", cval=0.0) * alpha
        dy = gaussian_filter(noise_y, sigma, mode="constant", cval=0.0) * alpha
        grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
        map_x = (grid_x + dx).astype(np.float32)
        map_y = (grid_y + dy).astype(np.float32)
        return cv2.remap(img_np, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


class DynamicBrushMorphology:
    def __init__(self, p_dilate=0.30, p_erode=0.25):
        self.p_dilate = p_dilate
        self.p_erode = p_erode
        self.k45 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.uint8)
        self.k135 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.uint8)
        self.kcross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
        self.krect = np.ones((2, 2), dtype=np.uint8)

    def __call__(self, img_np: np.ndarray) -> np.ndarray:
        r = random.random()
        k = random.choice([self.k45, self.k135, self.kcross, self.krect])
        if r < self.p_dilate:
            return cv2.dilate(img_np, k, iterations=1)
        elif r < self.p_dilate + self.p_erode:
            return cv2.erode(img_np, k, iterations=1)
        return img_np


class TargetedStrokePerturbation:
    def __init__(self, p_perturb=0.35):
        self.p_perturb = p_perturb

    def __call__(self, img_np: np.ndarray, label: int = -1) -> np.ndarray:
        if random.random() > self.p_perturb:
            return img_np
        out = img_np.copy()
        H, W = out.shape[:2]
        if label == 10:  # 十: entry hook simulation
            col_sums = np.sum(out[:24, :], axis=0)
            if np.max(col_sums) > 0:
                mid_x = int(np.argmax(col_sums))
                rows = np.where(out[:, mid_x] > 50)[0]
                if len(rows) > 0:
                    top_y = rows[0]
                    cv2.line(out, (max(0, mid_x - 4), max(0, top_y - 2)), (mid_x, top_y), color=200, thickness=1)
        elif label == 5:  # 五: Caoshu wave
            A = random.uniform(1.8, 3.2)
            period = random.uniform(28.0, 44.0)
            phase = random.uniform(0, 2 * math.pi)
            y_indices = np.arange(H).reshape(H, 1)
            dx = (A * np.sin(2 * np.pi * y_indices / period + phase)).astype(np.float32)
            dx_grid = np.repeat(dx, W, axis=1)
            grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
            out = cv2.remap(out, (grid_x + dx_grid).astype(np.float32), grid_y.astype(np.float32), interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        elif label == 4:  # 四: inner ligature
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            closed = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)
            out = cv2.addWeighted(out, 0.4, closed, 0.6, 0)
        return out


class BiomechanicalAugmentationPipeline:
    def __init__(self):
        self.elastic = ElasticDistortion()
        self.morph = DynamicBrushMorphology()
        self.perturb = TargetedStrokePerturbation()

    def __call__(self, img_np: np.ndarray, label: int = -1) -> np.ndarray:
        H, W = img_np.shape[:2]
        # Affine jitter
        if random.random() < 0.85:
            angle = random.uniform(-7.0, 7.0)
            scale = random.uniform(0.93, 1.05)
            tx = random.uniform(-1.5, 1.5)
            ty = random.uniform(-1.5, 1.5)
            M = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), angle, scale)
            M[0, 2] += tx
            M[1, 2] += ty
            img_np = cv2.warpAffine(img_np, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        img_np = self.perturb(img_np, label=label)
        img_np = self.elastic(img_np)
        img_np = self.morph(img_np)
        return img_np


# ------------------------------------------------------------------------------
# 4. DATASET & DATALOADER
# ------------------------------------------------------------------------------
class BiomechanicalChineseDataset(Dataset):
    def __init__(self, df, id_to_path, is_train=True):
        self.df = df.reset_index(drop=True)
        self.id_to_path = id_to_path
        self.is_train = is_train
        self.aug = BiomechanicalAugmentationPipeline() if is_train else None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = str(row['id'])
        label = int(row['label'])
        writer = int(row.get('writer_id', -1))

        if img_id not in self.id_to_path:
            raise FileNotFoundError(f"Image ID '{img_id}' not found in indexed disk paths!")

        p = self.id_to_path[img_id]
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise IOError(f"Corrupt or unreadable image at path: {p}")

        if self.is_train and self.aug is not None:
            img = self.aug(img, label=label)

        tensor = torch.from_numpy(img.astype(np.float32) / 127.5 - 1.0).unsqueeze(0)
        return tensor, label, writer


# ------------------------------------------------------------------------------
# 5. DENSENET-121 CBAM & SUB-CENTER ARCFACE ARCHITECTURE
# ------------------------------------------------------------------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out) * x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        scale = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return scale * x


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        return self.sa(self.ca(x))


class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(num_input_features)
        self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        self.drop_rate = float(drop_rate)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x), inplace=True))
        out = self.conv2(F.relu(self.bn2(out), inplace=True))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        return torch.cat([x, out], 1)


class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate=0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            for i in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class _Transition(nn.Module):
    def __init__(self, num_input_features, num_output_features):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_input_features)
        self.conv = nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x), inplace=True))
        return self.pool(out)


class MixStyle(nn.Module):
    def __init__(self, p=0.5, alpha=0.1, eps=1e-6):
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


class DenseNet121_CBAM(nn.Module):
    def __init__(self, in_channels=1, num_classes=15, feature_dim=256, use_mixstyle=True):
        super().__init__()
        self.feature_dim = feature_dim
        self.use_mixstyle = use_mixstyle
        self.growth_rate = 32
        block_config = (6, 12, 24, 16)
        num_init_features = 64

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, num_init_features, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True)
        )
        self.mixstyle = MixStyle(p=0.4, alpha=0.1) if use_mixstyle else nn.Identity()

        self.blocks = nn.ModuleList()
        self.cbams = nn.ModuleList()
        num_features = num_init_features

        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers, num_features, bn_size=4, growth_rate=self.growth_rate, drop_rate=0.0)
            self.blocks.append(block)
            num_features = num_features + num_layers * self.growth_rate
            self.cbams.append(CBAM(num_features, ratio=16))

            if i != len(block_config) - 1:
                trans = _Transition(num_features, num_features // 2)
                self.blocks.append(trans)
                num_features = num_features // 2

        self.final_bn = nn.BatchNorm2d(num_features)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_feat = nn.Linear(num_features, feature_dim)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def extract_features(self, x):
        x = self.features(x)
        if self.use_mixstyle:
            x = self.mixstyle(x)
        cbam_idx = 0
        for block in self.blocks:
            x = block(x)
            if isinstance(block, _DenseBlock):
                x = self.cbams[cbam_idx](x)
                cbam_idx += 1
        x = F.relu(self.final_bn(x), inplace=True)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc_feat(x)
        return F.normalize(x, p=2, dim=1)

    def forward(self, x):
        feats = self.extract_features(x)
        logits = self.classifier(feats)
        return logits, feats


class SubCenterArcFace(nn.Module):
    def __init__(self, in_features=256, num_classes=15, sub_centers=2, s=32.0, m=0.30):
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

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        cosine = cosine.view(-1, self.num_classes, self.sub_centers)
        cosine, _ = torch.max(cosine, dim=2)
        sine = torch.sqrt(1.0 - torch.clamp(cosine ** 2, 0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


# ------------------------------------------------------------------------------
# 6. TRAINING ENGINE
# ------------------------------------------------------------------------------
def train_densenet_pipeline(epochs=15, batch_size=64, lr=5e-4):
    splits_csv, synth_trad_csv, synth_callig_csv, id_to_path, val_files = setup_datasets()
    df_splits = pd.read_csv(splits_csv)

    print("\n=======================================================")
    print("🏆 TRAINING 5-FOLD DENSENET-121 CBAM SOTA V2 PIPELINE")
    print(f"Device: {DEVICE} | Batch Size: {batch_size} | Epochs: {epochs}")
    print("=======================================================")

    oof_probs = np.zeros((len(df_splits), 15), dtype=np.float32)
    scores = []

    for fold in range(5):
        print(f"\n▶ Starting DenseNet-121 CBAM Fold {fold}/4 ...")
        train_df = df_splits[df_splits['fold'] != fold].copy()
        val_df = df_splits[df_splits['fold'] == fold].copy()

        # Inject traditional synthetic data
        if synth_trad_csv and os.path.exists(synth_trad_csv):
            df_synth = pd.read_csv(synth_trad_csv)
            train_df = pd.concat([train_df, df_synth], ignore_index=True)
            print(f"  • Injected {len(df_synth)} traditional Hanzi samples.")

        # Inject 7-font targeted calligraphy v2 data
        if synth_callig_csv and os.path.exists(synth_callig_csv):
            df_callig = pd.read_csv(synth_callig_csv)
            train_df = pd.concat([train_df, df_callig], ignore_index=True)
            print(f"  • Injected {len(df_callig)} targeted calligraphy v2 samples.")

        print(f"  • Total Train Samples: {len(train_df):,} | Val Samples (Unseen Writers): {len(val_df):,}")

        # Pre-flight data integrity verification
        missing = [str(i) for i in train_df['id'] if str(i) not in id_to_path]
        print(f"  • Pre-flight Integrity Check: Missing samples from disk: {len(missing)}")
        if missing:
            raise RuntimeError(f"CRITICAL ERROR: {len(missing)} training images could not be located! Examples: {missing[:5]}")

        train_ds = BiomechanicalChineseDataset(train_df, id_to_path, is_train=True)
        val_ds = BiomechanicalChineseDataset(val_df, id_to_path, is_train=False)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

        model = DenseNet121_CBAM(in_channels=1, num_classes=15, feature_dim=256, use_mixstyle=True).to(DEVICE)
        metric_head = SubCenterArcFace(in_features=256, num_classes=15, sub_centers=2, s=32.0, m=0.30).to(DEVICE)

        criterion = nn.CrossEntropyLoss(label_smoothing=0.08)
        optimizer = torch.optim.AdamW(list(model.parameters()) + list(metric_head.parameters()), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        scaler = GradScaler('cuda') if torch.cuda.is_available() else None

        best_val_acc = 0.0
        best_val_probs = None
        ckpt_path = os.path.join(MODELS_DIR, f"densenet121_cbam_v2_fold{fold}.pt")

        for ep in range(1, epochs + 1):
            model.train()
            metric_head.train()
            t_loss, correct, total = 0.0, 0, 0

            for imgs, lbls, wrs in train_loader:
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                optimizer.zero_grad()
                with autocast('cuda', enabled=torch.cuda.is_available()):
                    logits, feats = model(imgs)
                    arc_logits = metric_head(feats, lbls)
                    loss = 0.5 * criterion(logits, lbls) + 0.5 * criterion(arc_logits, lbls)

                if scaler:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                t_loss += loss.item() * len(lbls)
                correct += (logits.argmax(dim=1) == lbls).sum().item()
                total += len(lbls)

            scheduler.step()
            train_acc = correct / total
            train_loss = t_loss / total

            # Validation
            model.eval()
            val_correct, val_total = 0, 0
            ep_probs = []
            with torch.no_grad():
                for imgs, lbls, wrs in val_loader:
                    imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                    with autocast('cuda', enabled=torch.cuda.is_available()):
                        logits, _ = model(imgs)
                    probs = F.softmax(logits, dim=1)
                    ep_probs.append(probs.cpu().numpy())
                    val_correct += (logits.argmax(dim=1) == lbls).sum().item()
                    val_total += len(lbls)

            val_acc = val_correct / val_total
            print(f"  Epoch {ep:02d}/{epochs:02d} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%", flush=True)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_probs = np.concatenate(ep_probs, axis=0)
                torch.save({"model_state_dict": model.state_dict(), "val_acc": val_acc, "fold": fold}, ckpt_path)

        scores.append(best_val_acc)
        oof_probs[val_df.index.values] = best_val_probs
        print(f"✅ Fold {fold} Complete! Best Val Acc: {best_val_acc*100:.2f}% (Saved: {ckpt_path})")

    mean_acc = np.mean(scores)
    print("\n=======================================================")
    print(f"🎯 ALL 5 FOLDS COMPLETE! Mean OOF Accuracy: {mean_acc*100:.2f}%")
    print(f"Per-Fold Accuracies: {[round(s*100, 2) for s in scores]}")
    oof_path = os.path.join(WORKING_DIR, "oof_probabilities_densenet121_cbam_v2.npy")
    np.save(oof_path, oof_probs)
    print(f"OOF Probabilities saved to: {oof_path}")
    print("=======================================================")

    # --------------------------------------------------------------------------
    # 7. VALIDATION SET 5-WAY TTA EVALUATION
    # --------------------------------------------------------------------------
    if val_files and len(val_files) > 0:
        print(f"\n🔍 Running 5-Way TTA Evaluation on {len(val_files)} Validation Images...")

        def get_tta_variants(tensor):
            variants = [tensor]
            B = tensor.size(0)
            dev = tensor.device
            for deg in [3, -3]:
                rad = math.radians(deg)
                theta = torch.tensor([[math.cos(rad), -math.sin(rad), 0], [math.sin(rad), math.cos(rad), 0]], dtype=torch.float, device=dev).unsqueeze(0).repeat(B, 1, 1)
                grid = F.affine_grid(theta, tensor.size(), align_corners=False)
                variants.append(F.grid_sample(tensor, grid, align_corners=False))
            for dx in [0.045, -0.045]:
                theta = torch.tensor([[1, 0, dx], [0, 1, 0]], dtype=torch.float, device=dev).unsqueeze(0).repeat(B, 1, 1)
                grid = F.affine_grid(theta, tensor.size(), align_corners=False)
                variants.append(F.grid_sample(tensor, grid, align_corners=False))
            return variants

        all_models_probs = []
        for fold in range(5):
            ckpt_path = os.path.join(MODELS_DIR, f"densenet121_cbam_v2_fold{fold}.pt")
            m = DenseNet121_CBAM(in_channels=1, num_classes=15, feature_dim=256, use_mixstyle=False).to(DEVICE)
            ckpt = torch.load(ckpt_path, map_location=DEVICE)
            m.load_state_dict(ckpt["model_state_dict"])
            m.eval()

            single_fold_probs = []
            with torch.no_grad():
                for f in val_files:
                    img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        raise IOError(f"Corrupt or missing validation image at: {f}")
                    t = torch.from_numpy(img.astype(np.float32) / 127.5 - 1.0).unsqueeze(0).unsqueeze(0).to(DEVICE)
                    tta_batch = torch.cat(get_tta_variants(t), dim=0)
                    with autocast('cuda', enabled=torch.cuda.is_available()):
                        l, _ = m(tta_batch)
                    probs = F.softmax(l, dim=1).mean(dim=0).cpu().numpy()
                    single_fold_probs.append(probs)
            all_models_probs.append(np.array(single_fold_probs))

        avg_val_probs = np.mean(all_models_probs, axis=0)
        prob_save_path = os.path.join(WORKING_DIR, "probabilities_densenet121_cbam_v2_5fold.npy")
        np.save(prob_save_path, avg_val_probs)
        print(f"✅ Saved 5-Fold DenseNet-121 v2 Probabilities to: {prob_save_path}")

        # Generate Pure Visual Submission
        val_ids = [os.path.splitext(os.path.basename(f))[0] for f in val_files]
        pred_labels = np.argmax(avg_val_probs, axis=1)
        sub_df = pd.DataFrame({"id": val_ids, "label": pred_labels})
        csv_path = os.path.join(WORKING_DIR, "submission_phase1_densenet121_cbam_v2.csv")
        zip_path = os.path.join(WORKING_DIR, "submission_phase1_densenet121_cbam_v2.zip")
        sub_df.to_csv(csv_path, index=False)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(csv_path, arcname="predictions.csv")
        print(f"📦 Generated Submission Package: {zip_path}")
        print("Class Distribution:", sub_df["label"].value_counts().sort_index().to_dict())

    print("\n🏁 PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    train_densenet_pipeline(epochs=15, batch_size=64, lr=5e-4)
