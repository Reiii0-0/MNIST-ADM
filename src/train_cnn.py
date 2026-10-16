import os
import sys
import random
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import cv2
from scipy.ndimage import gaussian_filter

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.models import ResNet18_CBAM, CompactCNN, SubCenterArcFace
from src.config import (
    get_splits_kfold_csv,
    get_train_synthetic_labels_csv,
    get_train_dir,
    get_train_synthetic_dir,
    MODELS_DIR
)

def elastic_transform_np(image, alpha=12, sigma=3):
    shape = image.shape
    dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)
    distorted = cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    return distorted

class ChineseCharacterDataset(Dataset):
    def __init__(self, df, img_dir=None, synth_dir=None, is_train=True):
        self.df = df.reset_index(drop=True)
        self.img_dir = str(img_dir or get_train_dir())
        self.synth_dir = str(synth_dir or get_train_synthetic_dir())
        self.is_train = is_train


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = str(row['id'])
        label = int(row['label'])
        
        # Check if synthetic or real
        if img_id.startswith("synth_"):
            path = os.path.join(self.synth_dir, f"{img_id}.png")
        else:
            path = os.path.join(self.img_dir, f"{img_id}.png")

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((64, 64), dtype=np.uint8)

        if self.is_train:
            # 1. Random Affine (Rotation, slight scale, translation)
            angle = random.uniform(-8.0, 8.0)
            scale = random.uniform(0.92, 1.05)
            tx = random.uniform(-2.0, 2.0)
            ty = random.uniform(-2.0, 2.0)
            M = cv2.getRotationMatrix2D((32, 32), angle, scale)
            M[0, 2] += tx
            M[1, 2] += ty
            img = cv2.warpAffine(img, M, (64, 64), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

            # 2. Elastic Deformation
            if random.random() < 0.4:
                img = elastic_transform_np(img, alpha=random.uniform(8.0, 14.0), sigma=random.uniform(2.5, 3.2))

            # 3. Dilation / Erosion
            morph = random.random()
            if morph < 0.25:
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                img = cv2.dilate(img, kernel, iterations=1)
            elif morph < 0.45:
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                img = cv2.erode(img, kernel, iterations=1)

        # Normalize to float tensor [-1, 1]
        tensor = torch.from_numpy(img).float() / 255.0
        tensor = (tensor - 0.5) / 0.5
        tensor = tensor.unsqueeze(0)  # Shape (1, 64, 64)

        return tensor, label, int(row.get('writer_id', 999))

def train_one_fold(
    fold=0,
    model_type="resnet18_cbam",
    use_synthetic=True,
    epochs=15,
    batch_size=64,
    lr=5e-4,
    device="cuda"
):
    print("=" * 60)
    print(f"TRAINING {model_type.upper()} ON FOLD {fold}")
    print(f"Device: {device} | Synthetic Data: {use_synthetic} | Epochs: {epochs}")
    print("=" * 60)

    splits_df = pd.read_csv(get_splits_kfold_csv())
    train_df = splits_df[splits_df['fold'] != fold].copy()
    val_df = splits_df[splits_df['fold'] == fold].copy()

    # Append synthetic traditional samples to train set only
    synth_csv = get_train_synthetic_labels_csv()
    if use_synthetic and synth_csv.exists():
        synth_df = pd.read_csv(synth_csv)
        train_df = pd.concat([train_df, synth_df], ignore_index=True)
        print(f"Added {len(synth_df)} synthetic samples to Train set. Total train: {len(train_df)}")

    print(f"Train samples: {len(train_df)} | Val samples: {len(val_df)} (from {val_df['writer_id'].nunique()} unseen writers)")

    train_ds = ChineseCharacterDataset(train_df, is_train=True)
    val_ds = ChineseCharacterDataset(val_df, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Initialize Model
    if model_type == "resnet18_cbam":
        model = ResNet18_CBAM(in_channels=1, num_classes=15, feature_dim=256, use_mixstyle=True)
        metric_head = SubCenterArcFace(in_features=256, num_classes=15, sub_centers=2, s=32.0, m=0.30)
    else:
        model = CompactCNN(in_channels=1, num_classes=15)
        metric_head = None

    model = model.to(device)
    if metric_head is not None:
        metric_head = metric_head.to(device)

    # Loss and Optimizer
    criterion_ce = nn.CrossEntropyLoss(label_smoothing=0.08)
    params = list(model.parameters()) + (list(metric_head.parameters()) if metric_head is not None else [])
    optimizer = optim.AdamW(params, lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_acc = 0.0
    os.makedirs(MODELS_DIR, exist_ok=True)
    ckpt_path = str(MODELS_DIR / f"{model_type}_fold{fold}.pt")


    for epoch in range(1, epochs + 1):
        model.train()
        if metric_head is not None:
            metric_head.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for imgs, labels, writers in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            logits, feats = model(imgs)

            if metric_head is not None:
                arc_logits = metric_head(feats, labels)
                loss = 0.5 * criterion_ce(logits, labels) + 0.5 * criterion_ce(arc_logits, labels)
            else:
                loss = criterion_ce(logits, labels)

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
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for imgs, labels, writers in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits, feats = model(imgs)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += len(labels)

        val_acc = val_correct / val_total
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Val Acc (Unseen Writers): {val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "metric_head_state_dict": metric_head.state_dict() if metric_head is not None else None,
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
    parser.add_argument("--model", type=str, default="resnet18_cbam", choices=["resnet18_cbam", "compact_cnn"])
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_one_fold(
        fold=args.fold,
        model_type=args.model,
        use_synthetic=True,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device
    )
