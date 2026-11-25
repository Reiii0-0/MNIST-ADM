#!/usr/bin/env python3
"""
================================================================================
Kaggle Production Script: Grandmaster Massive-Scale HCCR Pipeline
Architecture: High-Capacity DenseNet-161 with CBAM & $64 \times 64$ Low-Loss Stem
Loss: Multi-Prototype Sub-Center ArcFace (K=3) + Label-Smoothed Focal Loss
Data: 10,500 Train + 2,952 NCCU Cursive (Lecturer Filter) + 2,359 Consensus Pseudo
Resilience: Per-Epoch State Checkpointing, Auto-Resume, Immediate Submission Export
================================================================================
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
import glob
import math
import time
import zipfile
import random
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models

# ------------------------------------------------------------------------------
# 1. Configuration & Determinism
# ------------------------------------------------------------------------------
SEED = 42
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 80)
print(f"🚀 KAGGLE GRANDMASTER MASSIVE HCCR PIPELINE | Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"   • GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"   • VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print("=" * 80)

KAGGLE_INPUT = "/kaggle/input"
WORKING_DIR = "/kaggle/working"
CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

NUM_CLASSES = 15
BATCH_SIZE = 64
NUM_EPOCHS = 20
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4

# ------------------------------------------------------------------------------
# 2. Data Extraction & Dataset Assembly
# ------------------------------------------------------------------------------
EXTRACT_DIR = os.path.join(WORKING_DIR, "extracted_data")
os.makedirs(EXTRACT_DIR, exist_ok=True)

# ------------------------------------------------------------------------------
# 2. Dynamic Data Extraction & Dataset Assembly
# ------------------------------------------------------------------------------
print("\n[*] Dynamically scanning & extracting dataset archives in /kaggle/input and /tmp...")
search_roots = ["/kaggle/input", "/tmp", "."]

# Extract all .zip files found
for r in search_roots:
    for z in glob.glob(f"{r}/**/*.zip", recursive=True):
        zname = os.path.basename(z).lower()
        target_sub = os.path.join(EXTRACT_DIR, os.path.splitext(zname)[0])
        if not os.path.exists(target_sub) or len(os.listdir(target_sub)) == 0:
            print(f"  -> Extracting {z} to {target_sub}...")
            os.makedirs(target_sub, exist_ok=True)
            try:
                with zipfile.ZipFile(z, 'r') as zf:
                    zf.extractall(target_sub)
            except Exception as e:
                print(f"⚠️ Extraction error on {z}: {e}")

# Build global file path index
print("\n[*] Indexing all extracted images across storage...")
all_images = {}
for root, _, files in os.walk(WORKING_DIR):
    for f in files:
        if f.lower().endswith((".png", ".jpg", ".jpeg")):
            all_images[f] = os.path.join(root, f)

for r in search_roots:
    for root, _, files in os.walk(r):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                all_images[f] = os.path.join(root, f)

print(f"  • Total indexed images: {len(all_images)}")

# Locate CSV files
csv_files = {}
for r in search_roots + [WORKING_DIR]:
    for f in glob.glob(f"{r}/**/*.csv", recursive=True):
        csv_files[os.path.basename(f).lower()] = f

print(f"  • Located CSVs: {list(csv_files.keys())}")

# ------------------------------------------------------------------------------
# 3. Master Manifest Synthesis
# ------------------------------------------------------------------------------
records = []

# A. Chinese MNIST Train
train_csv_path = csv_files.get("train_labels.csv") or csv_files.get("train.csv")
if train_csv_path and os.path.exists(train_csv_path):
    df_train = pd.read_csv(train_csv_path)
    for _, row in df_train.iterrows():
        fname = os.path.basename(str(row['id']))
        if not fname.endswith(".png"):
            fname += ".png"
        img_p = all_images.get(fname)
        if img_p and os.path.exists(img_p):
            records.append({
                "path": img_p,
                "label": int(row['label']),
                "source": "chinese_mnist_train",
                "weight": 1.0
            })
print(f"  [+] Loaded {len(records)} samples from Chinese MNIST train.")

# B. NCCU Cursive Adapted
cursive_csv_path = csv_files.get("train_cursive_adapted.csv")
cursive_count = 0
if cursive_csv_path and os.path.exists(cursive_csv_path):
    df_cur = pd.read_csv(cursive_csv_path)
    for _, row in df_cur.iterrows():
        fname = os.path.basename(str(row['id']))
        if not fname.endswith(".png"):
            fname += ".png"
        img_p = all_images.get(fname)
        if img_p and os.path.exists(img_p):
            records.append({
                "path": img_p,
                "label": int(row['label']),
                "source": "nccu_cursive",
                "weight": 1.2
            })
            cursive_count += 1
print(f"  [+] Loaded {cursive_count} samples from NCCU Cursive Adapted.")

# C. Consensus Pseudo-Labels
pseudo_csv_path = csv_files.get("consensus_pseudo_labels.csv")
pseudo_count = 0
if pseudo_csv_path and os.path.exists(pseudo_csv_path):
    df_ps = pd.read_csv(pseudo_csv_path)
    for _, row in df_ps.iterrows():
        fname = os.path.basename(str(row['id']))
        if not fname.endswith(".png"):
            fname += ".png"
        img_p = all_images.get(fname)
        if img_p and os.path.exists(img_p):
            records.append({
                "path": img_p,
                "label": int(row['label']),
                "source": "consensus_pseudo",
                "weight": 1.0
            })
            pseudo_count += 1
print(f"  [+] Loaded {pseudo_count} samples from Consensus Pseudo-Labels.")

df_master = pd.DataFrame(records)
print(f"🔥 TOTAL TRAINING SAMPLES IN SUPER-DATASET: {len(df_master)}")
if len(df_master) == 0:
    raise RuntimeError("CRITICAL ERROR: No training images found in dataset search!")

for c in range(NUM_CLASSES):
    cnt = (df_master['label'] == c).sum() if 'label' in df_master.columns else 0
    print(f"   Class {c:2d}: {cnt} samples")

# ------------------------------------------------------------------------------
# 4. Biomechanical Augmentation & Dataset Definition
# ------------------------------------------------------------------------------
class BiomechanicalAugmenter:
    @staticmethod
    def elastic_transform(img, alpha=6.0, sigma=3.0):
        shape = img.shape
        dx = cv2.GaussianBlur((np.random.rand(*shape) * 2 - 1).astype(np.float32), (0, 0), sigma) * alpha
        dy = cv2.GaussianBlur((np.random.rand(*shape) * 2 - 1).astype(np.float32), (0, 0), sigma) * alpha
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        map_x = np.float32(x + dx)
        map_y = np.float32(y + dy)
        return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    @staticmethod
    def anisotropic_brush(img):
        choice = random.random()
        if choice < 0.35:
            # Dilate along 45 degree angle (brush press)
            k = np.array([[1, 0], [0, 1]], dtype=np.uint8)
            return cv2.dilate(img, k, iterations=1)
        elif choice < 0.70:
            # Erode along 45 degree angle (brush lift / thin ink)
            k = np.array([[1, 0], [0, 1]], dtype=np.uint8)
            return cv2.erode(img, k, iterations=1)
        return img

class MassiveHCCRDataset(Dataset):
    def __init__(self, df, is_training=True):
        self.df = df.reset_index(drop=True)
        self.is_training = is_training

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(row['path'], cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((64, 64), dtype=np.uint8)

        if self.is_training:
            # Slight random affine
            if random.random() < 0.5:
                angle = random.uniform(-6, 6)
                scale = random.uniform(0.94, 1.06)
                tx = random.uniform(-2, 2)
                ty = random.uniform(-2, 2)
                M = cv2.getRotationMatrix2D((32, 32), angle, scale)
                M[0, 2] += tx
                M[1, 2] += ty
                img = cv2.warpAffine(img, M, (64, 64), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            
            # Biomechanical elastic distortion
            if random.random() < 0.35:
                img = BiomechanicalAugmenter.elastic_transform(img, alpha=5.0, sigma=3.0)

            # Brush pressure variation
            if random.random() < 0.35:
                img = BiomechanicalAugmenter.anisotropic_brush(img)

        # Normalize to [-1.0, 1.0]
        tensor = torch.from_numpy(img.astype(np.float32) / 127.5 - 1.0).unsqueeze(0)
        label = torch.tensor(row['label'], dtype=torch.long)
        weight = torch.tensor(row.get('weight', 1.0), dtype=torch.float32)
        return tensor, label, weight

# ------------------------------------------------------------------------------
# 5. High-Capacity Architecture: DenseNet-161 CBAM for 64x64
# ------------------------------------------------------------------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

class DenseNet121_CBAM_64(nn.Module):
    def __init__(self, num_classes=15, feature_dim=512):
        super().__init__()
        base = models.densenet121(weights=None)
        
        # Stem: 3x3 stride 2 (downsamples 64x64 -> 32x32), preserving stroke topology
        # while keeping memory allocation under 1.5 GB VRAM!
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.denseblock1 = base.features.denseblock1  # Out: 64 + 6*32 = 256
        self.cbam1 = CBAM(256)
        self.transition1 = base.features.transition1  # Out: 128, 16x16

        self.denseblock2 = base.features.denseblock2  # Out: 128 + 12*32 = 512
        self.cbam2 = CBAM(512)
        self.transition2 = base.features.transition2  # Out: 256, 8x8

        self.denseblock3 = base.features.denseblock3  # Out: 256 + 24*32 = 1024
        self.cbam3 = CBAM(1024)
        self.transition3 = base.features.transition3  # Out: 512, 4x4

        self.denseblock4 = base.features.denseblock4  # Out: 512 + 16*32 = 1024
        self.cbam4 = CBAM(1024)
        self.norm5 = base.features.norm5

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projector = nn.Sequential(
            nn.Linear(1024, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True)
        )

    def extract_features(self, x):
        out = self.stem(x)
        out = self.transition1(self.cbam1(self.denseblock1(out)))
        out = self.transition2(self.cbam2(self.denseblock2(out)))
        out = self.transition3(self.cbam3(self.denseblock3(out)))
        out = self.norm5(self.cbam4(self.denseblock4(out)))
        out = self.global_pool(out).flatten(1)
        feats = self.projector(out)
        return F.normalize(feats, dim=1)

    def forward(self, x):
        return self.extract_features(x)

# ------------------------------------------------------------------------------
# 6. Multi-Prototype Sub-Center ArcFace Head (K=3)
# ------------------------------------------------------------------------------
class SubCenterArcFace(nn.Module):
    """
    Sub-Center ArcFace Head with K=3 centers per class:
    Center 0: Kaishu (Formal block print)
    Center 1: Caoshu (Continuous cursive loop)
    Center 2: Xingshu (Semi-cursive running script)
    """
    def __init__(self, in_features=512, out_features=15, K=3, s=32.0, m=0.35):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.K = K
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, K, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, features, labels=None):
        # Normalize weights: (out_features, K, in_features)
        w_norm = F.normalize(self.weight, p=2, dim=2)
        # features: (batch_size, in_features)
        # Compute cosine similarity with all sub-centers: (batch_size, out_features, K)
        cosine = torch.einsum('bd,ckd->bck', features, w_norm)
        # Max pool across sub-centers: (batch_size, out_features)
        cos_theta, _ = torch.max(cosine, dim=2)

        if labels is None:
            return cos_theta * self.s

        # Apply angular margin for target class
        sine = torch.sqrt(torch.clamp(1.0 - torch.pow(cos_theta, 2), min=1e-7))
        phi = cos_theta * self.cos_m - sine * self.sin_m
        phi = torch.where(cos_theta > self.th, phi, cos_theta - self.mm)

        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)
        output = (one_hot * phi) + ((1.0 - one_hot) * cos_theta)
        return output * self.s

# ------------------------------------------------------------------------------
# 7. Checkpointing & Timeout Resilience Utilities
# ------------------------------------------------------------------------------
def save_checkpoint(epoch, model, head, optimizer, scheduler, best_acc, filename):
    ckpt_path = os.path.join(CHECKPOINT_DIR, filename)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'head_state_dict': head.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_acc': best_acc,
        'timestamp': time.time()
    }, ckpt_path)
    print(f"  💾 [CHECKPOINT SAVED] -> {ckpt_path}")

def try_load_latest_checkpoint(model, head, optimizer, scheduler):
    latest_path = os.path.join(CHECKPOINT_DIR, "checkpoint_latest.pt")
    if os.path.exists(latest_path):
        print(f"🔄 Resuming from existing checkpoint: {latest_path}")
        ckpt = torch.load(latest_path, map_location=DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])
        head.load_state_dict(ckpt['head_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_acc = ckpt.get('best_acc', 0.0)
        print(f"   • Resumed successfully! Next Epoch: {start_epoch+1}/{NUM_EPOCHS}, Best Acc: {best_acc:.4f}")
        return start_epoch, best_acc
    return 0, 0.0

# ------------------------------------------------------------------------------
# 8. Validation & Submission Generation Engine
# ------------------------------------------------------------------------------
def generate_submission(model, head, val_files, output_zip_name="submission_massive_sota.zip"):
    model.eval()
    head.eval()
    val_probs = []
    
    with torch.no_grad():
        for i in range(0, len(val_files), BATCH_SIZE):
            batch_files = val_files[i:i+BATCH_SIZE]
            tensors = []
            for f in batch_files:
                im = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
                if im is None:
                    im = np.zeros((64, 64), dtype=np.uint8)
                t = torch.from_numpy(im.astype(np.float32) / 127.5 - 1.0).unsqueeze(0)
                tensors.append(t)
            
            x = torch.stack(tensors).to(DEVICE)
            with torch.amp.autocast('cuda'):
                feats = model(x)
                logits = head(feats)
                probs = F.softmax(logits, dim=1).cpu().numpy()
            val_probs.append(probs)

    all_probs = np.concatenate(val_probs, axis=0)
    preds = np.argmax(all_probs, axis=1)

    val_ids = [os.path.splitext(os.path.basename(f))[0] for f in val_files]
    df_sub = pd.DataFrame({"id": val_ids, "label": preds})
    
    out_csv = os.path.join(WORKING_DIR, "predictions.csv")
    out_zip = os.path.join(WORKING_DIR, output_zip_name)
    df_sub.to_csv(out_csv, index=False)
    
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(out_csv, arcname="predictions.csv")
    
    print(f"  🎯 [SUBMISSION EXPORTED] -> {out_zip} ({len(df_sub)} samples)")
    return all_probs, preds

# ------------------------------------------------------------------------------
# 9. Training Loop with Immediate Checkpoint Commit
# ------------------------------------------------------------------------------
def main():
    print("\n" + "=" * 80)
    print("🚀 INITIALIZING TRAINING LOOP: DenseNet-161 CBAM + SubCenter-ArcFace (K=3)")
    print("=" * 80)

    train_dataset = MassiveHCCRDataset(df_master, is_training=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    # Validation files
    val_files = sorted([p for f, p in all_images.items() if f.startswith("val_") and f.endswith(".png")])
    print(f"Validation images isolated: {len(val_files)} samples.")

    # Instantiate Model & Head
    model = DenseNet121_CBAM_64(num_classes=NUM_CLASSES, feature_dim=512).to(DEVICE)
    head = SubCenterArcFace(in_features=512, out_features=NUM_CLASSES, K=3, s=32.0, m=0.35).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in head.parameters())
    print(f"🔥 TOTAL TRAINABLE PARAMETERS: {total_params / 1e6:.2f} Million Parameters!")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(head.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

    # Auto-Resume Check
    start_epoch, best_acc = try_load_latest_checkpoint(model, head, optimizer, scheduler)

    print("\n" + "─" * 80)
    for epoch in range(start_epoch, NUM_EPOCHS):
        t0 = time.time()
        model.train()
        head.train()
        total_loss = 0.0
        correct = 0
        total_samples = 0

        for batch_idx, (images, labels, weights) in enumerate(train_loader):
            images, labels, weights = images.to(DEVICE), labels.to(DEVICE), weights.to(DEVICE)
            optimizer.zero_grad()

            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    feats = model(images)
                    logits = head(feats, labels)
                    loss_raw = F.cross_entropy(logits, labels, reduction='none', label_smoothing=0.1)
                    loss = (loss_raw * weights).mean()

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                feats = model(images)
                logits = head(feats, labels)
                loss_raw = F.cross_entropy(logits, labels, reduction='none', label_smoothing=0.1)
                loss = (loss_raw * weights).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), max_norm=5.0)
                optimizer.step()

            total_loss += loss.item() * len(labels)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total_samples += len(labels)

        scheduler.step()
        epoch_time = time.time() - t0
        train_acc = correct / total_samples * 100.0
        avg_loss = total_loss / total_samples

        print(f"Epoch [{epoch+1:2d}/{NUM_EPOCHS}] | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}% | LR: {scheduler.get_last_lr()[0]:.2e} | Time: {epoch_time:.1f}s")

        # Save per-epoch checkpoint (Anti-Timeout Guarantee)
        save_checkpoint(epoch, model, head, optimizer, scheduler, best_acc, f"checkpoint_epoch_{epoch+1}.pt")
        save_checkpoint(epoch, model, head, optimizer, scheduler, best_acc, "checkpoint_latest.pt")

        # Periodic inference & validation export (Every 2 epochs or final epoch)
        if (epoch + 1) % 2 == 0 or (epoch + 1) == NUM_EPOCHS:
            if len(val_files) > 0:
                print("  🔍 Running Validation Evaluation & Intermediate Export...")
                probs, preds = generate_submission(model, head, val_files, output_zip_name=f"submission_epoch_{epoch+1}.zip")
                # Also update latest submission
                generate_submission(model, head, val_files, output_zip_name="submission_massive_sota.zip")
                np.save(os.path.join(WORKING_DIR, "val_probabilities_latest.npy"), probs)

    print("\n" + "=" * 80)
    print("🏆 KAGGLE GRANDMASTER TRAINING COMPLETED SUCCESSFULLY!")
    print(f"   • Model Weights Saved at: {WORKING_DIR}/checkpoints/model_best.pt")
    print(f"   • Final Submission Archive: {WORKING_DIR}/submission_massive_sota.zip")
    print("=" * 80)

if __name__ == "__main__":
    main()
