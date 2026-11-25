#!/usr/bin/env python3
"""
================================================================================
Scientific Step 1: Pairwise Boundary Discriminator Training Engine
Specialized Binary & Triplet Metric Classifiers for Ambiguous Cursive Hanzi:
1. Expert Pair A: 五 (Class 5) vs 六 (Class 6) -> 13 known errors in Sub 23
2. Expert Pair B: 十 (Class 10) vs 千 (Class 12) -> 4 known errors in Sub 23
3. Expert Pair C: 百 (Class 11) vs 四 (Class 4) / 万 (Class 13) -> 6 known errors
4. Expert Pair D: 三 (Class 3) vs 六 (Class 6) / 二 (Class 2) -> 6 known errors

Architecture: ResNet-18 CBAM Binary Boundary Classifier with Focal Loss
Hardware: NVIDIA RTX 4060 GPU
================================================================================
"""

import os
import sys
import glob
import math
import random
import time
from pathlib import Path
from typing import List, Tuple, Dict

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.models import ResNet18_CBAM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 80)
print(f"🚀 INITIALIZING PAIRWISE BOUNDARY EXPERT TRAINING ENGINE | Device: {DEVICE}")
print("=" * 80)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
SYNTH_DIR = os.path.join(DATA_DIR, "train_synthetic")
CURSIVE_DIR = os.path.join(DATA_DIR, "train_cursive_adapted")
VAL_DIR = os.path.join(DATA_DIR, "validasi/test")
EXPERT_DIR = os.path.join(PROJECT_ROOT, "models/pairwise_experts")
os.makedirs(EXPERT_DIR, exist_ok=True)

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything(42)

class PairwiseBinaryDataset(Dataset):
    def __init__(self, records: List[Dict], is_train: bool = True):
        self.records = records
        self.is_train = is_train

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        item = self.records[idx]
        img_path = item["path"]
        lbl = item["binary_label"]

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((64, 64), dtype=np.uint8)
        elif img.shape != (64, 64):
            img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)

        if self.is_train:
            # 1. Subtle rotation
            if random.random() < 0.5:
                angle = random.uniform(-4.0, 4.0)
                M = cv2.getRotationMatrix2D((32, 32), angle, 1.0)
                M[0, 2] += random.uniform(-1.5, 1.5)
                M[1, 2] += random.uniform(-1.5, 1.5)
                img = cv2.warpAffine(img, M, (64, 64), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

            # 2. Anisotropic Brush Pressure
            if random.random() < 0.3:
                kernel_angle = random.choice([45, 135])
                if kernel_angle == 45:
                    k = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.uint8)
                else:
                    k = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.uint8)
                img = cv2.erode(img, k, iterations=1)

        t = torch.from_numpy(img.astype(np.float32) / 127.5 - 1.0).unsqueeze(0)
        return t, lbl

class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.5):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction='none')
        probs = torch.sigmoid(logits)
        p_t = probs * targets.float() + (1.0 - probs) * (1.0 - targets.float())
        focal_weight = (1.0 - p_t) ** self.gamma
        alpha_weight = self.alpha * targets.float() + (1.0 - self.alpha) * (1.0 - targets.float())
        return (alpha_weight * focal_weight * bce_loss).mean()

class PairwiseResNet(nn.Module):
    def __init__(self):
        super().__init__()
        base = ResNet18_CBAM(in_channels=1, num_classes=15, feature_dim=256, use_mixstyle=False)
        self.features = base
        # Replace output projection with binary classifier head
        self.binary_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features.extract_features(x)
        feat_norm = F.normalize(feat, dim=1)
        out = self.binary_head(feat_norm).squeeze(-1)
        return out

def collect_records_for_pair(class_0: int, class_1: int) -> Tuple[List[Dict], List[Dict]]:
    """
    Collects training and validation images from Chinese MNIST + Adapted Cursive.
    Returns: (train_records, val_records)
    """
    # 1. Chinese MNIST
    splits_csv = os.path.join(DATA_DIR, "metadata/splits_kfold.csv")
    df_splits = pd.read_csv(splits_csv)
    
    # 2. Cursive
    cursive_csv = os.path.join(DATA_DIR, "train_cursive_adapted.csv")
    df_cursive = pd.read_csv(cursive_csv)

    train_records = []
    val_records = []

    # Process Chinese MNIST
    for _, row in df_splits.iterrows():
        lbl = int(row["label"])
        if lbl not in [class_0, class_1]:
            continue
        p = os.path.join(TRAIN_DIR, f"{row['id']}.png")
        b_lbl = 0 if lbl == class_0 else 1
        rec = {"path": p, "orig_label": lbl, "binary_label": b_lbl}
        if row["fold"] == 0:
            val_records.append(rec)
        else:
            train_records.append(rec)

    # Process Cursive (add to train_records)
    for _, row in df_cursive.iterrows():
        lbl = int(row["label"])
        if lbl not in [class_0, class_1]:
            continue
        p = row["saved_path"]
        b_lbl = 0 if lbl == class_0 else 1
        rec = {"path": p, "orig_label": lbl, "binary_label": b_lbl}
        train_records.append(rec)

    return train_records, val_records

def train_pairwise_expert(class_0: int, class_1: int, name_0: str, name_1: str, epochs: int = 15) -> str:
    print(f"\n{'='*70}")
    print(f"🔥 TRAINING PAIRWISE EXPERT: Class {class_0} ({name_0}) vs Class {class_1} ({name_1})")
    print(f"{'='*70}")

    train_recs, val_recs = collect_records_for_pair(class_0, class_1)
    print(f"  • Training samples: {len(train_recs)} | Validation samples: {len(val_recs)}")

    train_ds = PairwiseBinaryDataset(train_recs, is_train=True)
    val_ds = PairwiseBinaryDataset(val_recs, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

    model = PairwiseResNet().to(DEVICE)
    # Initialize backbone with pretrained adapted weights if available
    ckpt_path = os.path.join(PROJECT_ROOT, "models/adapted_cursive/resnet18_cbam_adapted_fold0.pt")
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=DEVICE)["model"]
        model.features.load_state_dict(state)
        print("  [+] Initialized backbone with Pretrained Adapted ResNet-18 CBAM.")

    criterion = BinaryFocalLoss(gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = GradScaler("cuda") if torch.cuda.is_available() else None

    best_acc = 0.0
    best_weights = None

    for ep in range(epochs):
        model.train()
        t_loss, correct, total = 0.0, 0, 0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad()
            with autocast("cuda", enabled=torch.cuda.is_available()):
                logits = model(imgs)
                loss = criterion(logits, lbls)

            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            t_loss += loss.item() * len(lbls)
            preds = (torch.sigmoid(logits) >= 0.5).long()
            correct += (preds == lbls).sum().item()
            total += len(lbls)

        scheduler.step()
        train_acc = correct / total

        model.eval()
        v_corr, v_tot = 0, 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                with autocast("cuda", enabled=torch.cuda.is_available()):
                    logits = model(imgs)
                preds = (torch.sigmoid(logits) >= 0.5).long()
                v_corr += (preds == lbls).sum().item()
                v_tot += len(lbls)

        val_acc = v_corr / v_tot if v_tot > 0 else train_acc
        if val_acc >= best_acc:
            best_acc = val_acc
            best_weights = {k: v.cpu() for k, v in model.state_dict().items()}

        if (ep + 1) % 3 == 0 or ep == epochs - 1:
            print(f"    Epoch {ep+1:2d}/{epochs:2d} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")

    out_ckpt = os.path.join(EXPERT_DIR, f"expert_{class_0}_vs_{class_1}.pt")
    torch.save({"model": best_weights, "class_0": class_0, "class_1": class_1, "acc": best_acc}, out_ckpt)
    print(f"  ⭐ Expert saved to: {out_ckpt} (Best Acc: {best_acc*100:.2f}%)")
    return out_ckpt

def main():
    # Train the 4 critical pairwise experts
    if not os.path.exists(os.path.join(EXPERT_DIR, "expert_5_vs_6.pt")):
        expert_5_vs_6 = train_pairwise_expert(5, 6, "五", "六", epochs=15)
    else:
        print("[*] expert_5_vs_6.pt already trained and cached.")

    if not os.path.exists(os.path.join(EXPERT_DIR, "expert_10_vs_12.pt")):
        expert_10_vs_12 = train_pairwise_expert(10, 12, "十", "千", epochs=15)
    else:
        print("[*] expert_10_vs_12.pt already trained and cached.")

    expert_4_vs_11 = train_pairwise_expert(4, 11, "四", "百", epochs=15)
    expert_3_vs_6 = train_pairwise_expert(3, 6, "三", "六", epochs=15)

    print("\n" + "=" * 80)
    print("✅ ALL PAIRWISE BOUNDARY DISCRIMINATORS TRAINED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    main()
