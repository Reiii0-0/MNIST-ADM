#!/usr/bin/env python3
"""
Production Training Pipeline: Biomechanical Augmentation + SOTA Model #3 (DenseNet-121 CBAM / SwinHCCR)
Integrated with Sub-Center ArcFace and MixStyle for CodaBench Chinese MNIST HCCR.

Features:
- Stratified Group K-Fold (Writer-Independent splitting).
- Biomechanical Augmentation Pipeline (Elastic Deformation + Calligraphic Brush Dynamics + Targeted Stroke Perturbation).
- Sub-Center ArcFace (K=2 sub-centers) + Softmax Label Smoothing.
- Multi-scale gradient optimization via AdamW & CosineAnnealingLR.
"""

import os
import sys
import random
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.models import ResNet18_CBAM, ConvNeXt_Femto, DenseNet121_CBAM, SwinHCCR_64, SubCenterArcFace
from src.biomechanical_augmentation import BiomechanicalAugmentationPipeline
from src.config import (
    get_splits_kfold_csv,
    get_train_synthetic_labels_csv,
    get_train_dir,
    get_train_synthetic_dir,
    MODELS_DIR
)


class BiomechanicalChineseDataset(Dataset):
    """
    Dataset wrapper injecting Biomechanical Data Augmentations into training pipeline.
    """
    def __init__(self, df, img_dir=None, synth_dir=None, is_train=True):
        self.df = df.reset_index(drop=True)
        self.img_dir = str(img_dir or get_train_dir())
        self.synth_dir = str(synth_dir or get_train_synthetic_dir())
        self.is_train = is_train
        self.aug = BiomechanicalAugmentationPipeline() if is_train else None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = str(row['id'])
        label = int(row['label'])
        writer = int(row.get('writer_id', -1))

        # Check synthetic vs authentic path
        if img_id.startswith("synth_"):
            path = os.path.join(self.synth_dir, f"{img_id}.png")
        else:
            path = os.path.join(self.img_dir, f"{img_id}.png")

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((64, 64), dtype=np.uint8)

        if self.is_train and self.aug is not None:
            # Apply Biomechanical Transformation
            img = self.aug(img, label=label)

        # Normalize to float tensor in [-1.0, 1.0]
        tensor = torch.from_numpy(img.astype(np.float32) / 127.5 - 1.0).unsqueeze(0)
        return tensor, label, writer


def train_biomechanical_fold(
    fold=0,
    model_type="densenet121_cbam",
    use_synthetic=True,
    epochs=15,
    batch_size=64,
    lr=5e-4,
    device="cuda"
):
    print(f"\n=======================================================")
    print(f"🚀 Training SOTA Model: {model_type} | Fold: {fold} | Device: {device}")
    print(f"=======================================================")

    splits_csv = get_splits_kfold_csv()
    if not splits_csv.exists():
        raise FileNotFoundError(f"Splits file not found: {splits_csv}")
    df_splits = pd.read_csv(splits_csv)

    train_df = df_splits[df_splits['fold'] != fold].copy()
    val_df = df_splits[df_splits['fold'] == fold].copy()

    if use_synthetic:
        synth_csv = get_train_synthetic_labels_csv()
        if synth_csv.exists():
            df_synth = pd.read_csv(synth_csv)
            train_df = pd.concat([train_df, df_synth], ignore_index=True)
            print(f"Injected {len(df_synth)} synthetic traditional Hanzi samples.")

    print(f"Training samples: {len(train_df):,} | Validation samples (Unseen Writers): {len(val_df):,}")

    train_dataset = BiomechanicalChineseDataset(train_df, is_train=True)
    val_dataset = BiomechanicalChineseDataset(val_df, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Initialize Architecture
    if model_type == "densenet121_cbam":
        model = DenseNet121_CBAM(in_channels=1, num_classes=15, feature_dim=256, use_mixstyle=True)
        metric_head = SubCenterArcFace(in_features=256, num_classes=15, sub_centers=2, s=32.0, m=0.30)
    elif model_type == "swin_hccr_64":
        model = SwinHCCR_64(in_channels=1, num_classes=15, patch_size=4, embed_dim=48, feature_dim=256)
        metric_head = SubCenterArcFace(in_features=256, num_classes=15, sub_centers=2, s=32.0, m=0.30)
    elif model_type == "resnet18_cbam":
        model = ResNet18_CBAM(in_channels=1, num_classes=15, feature_dim=256, use_mixstyle=True)
        metric_head = SubCenterArcFace(in_features=256, num_classes=15, sub_centers=2, s=32.0, m=0.30)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model = model.to(device)
    metric_head = metric_head.to(device)

    # Loss and Optimizer
    criterion_ce = nn.CrossEntropyLoss(label_smoothing=0.08)
    params = list(model.parameters()) + list(metric_head.parameters())
    optimizer = optim.AdamW(params, lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_acc = 0.0
    os.makedirs(MODELS_DIR, exist_ok=True)
    ckpt_path = str(MODELS_DIR / f"{model_type}_fold{fold}.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        metric_head.train()
        total_loss, correct, total = 0.0, 0, 0

        for imgs, labels, writers in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            logits, feats = model(imgs)
            arc_logits = metric_head(feats, labels)
            loss = 0.5 * criterion_ce(logits, labels) + 0.5 * criterion_ce(arc_logits, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(labels)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)

        scheduler.step()
        train_acc = correct / total
        train_loss = total_loss / total

        # Validation on Unseen Writers
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, labels, writers in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits, feats = model(imgs)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += len(labels)

        val_acc = val_correct / val_total
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "metric_head_state_dict": metric_head.state_dict(),
                "val_acc": val_acc,
                "epoch": epoch,
                "fold": fold,
                "model_type": model_type
            }, ckpt_path)

    print(f"\n--> Fold {fold} Complete! Best Validation Accuracy: {best_val_acc * 100:.2f}%")
    print(f"Checkpoint saved to: {ckpt_path}")
    return best_val_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0, help="Fold index (0-4)")
    parser.add_argument("--model", type=str, default="densenet121_cbam",
                        choices=["densenet121_cbam", "swin_hccr_64", "resnet18_cbam"])
    parser.add_argument("--epochs", type=int, default=15, help="Epoch count")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_biomechanical_fold(
        fold=args.fold,
        model_type=args.model,
        use_synthetic=True,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device
    )
