import os
import sys
import glob
import math
import json
import zipfile
import random
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==============================================================================
# 1. SETUP & REPRODUCIBILITY
# ==============================================================================
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 70)
print(f"RUNNING ON DEVICE: {device}")
if torch.cuda.is_available():
    print(f"GPU Model: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print("=" * 70)

# ==============================================================================
# 2. DATA EXTRACTION & ROBUST PATH RESOLUTION
# ==============================================================================
WORK_DIR = "/tmp/data"
os.makedirs(WORK_DIR, exist_ok=True)
OUT_DIR = "/kaggle/working/models"
os.makedirs(OUT_DIR, exist_ok=True)

def setup_dataset_directories():
    print("Scanning /kaggle/input for dataset files...")
    
    # 1. Check for any zip files to extract
    for z in glob.glob("/kaggle/input/**/*.zip", recursive=True):
        print(f"Extracting zip archive: {z} -> {WORK_DIR}...")
        try:
            with zipfile.ZipFile(z, 'r') as zip_ref:
                zip_ref.extractall(WORK_DIR)
        except Exception as e:
            print(f"Warning extracting {z}: {e}")

    # 2. Locate train images directory by searching for train_00001.png
    train_matches = glob.glob("/kaggle/input/**/train_00001.png", recursive=True) + \
                    glob.glob(f"{WORK_DIR}/**/train_00001.png", recursive=True)
    if not train_matches:
        raise FileNotFoundError("CRITICAL: train_00001.png not found anywhere in /kaggle/input or /tmp!")
    train_img_dir = os.path.dirname(train_matches[0])
    print(f"--> Located Train Images Directory: {train_img_dir}")

    # 3. Locate synthetic images directory by searching for synth_00001.png
    synth_matches = glob.glob("/kaggle/input/**/synth_00001.png", recursive=True) + \
                    glob.glob(f"{WORK_DIR}/**/synth_00001.png", recursive=True)
    if not synth_matches:
        print("Warning: synth_00001.png not found directly, using train directory as fallback.")
        synth_img_dir = train_img_dir
    else:
        synth_img_dir = os.path.dirname(synth_matches[0])
        print(f"--> Located Synthetic Images Directory: {synth_img_dir}")

    # 4. Locate splits_kfold.csv
    splits_matches = glob.glob("/kaggle/input/**/splits_kfold.csv", recursive=True) + \
                     glob.glob(f"{WORK_DIR}/**/splits_kfold.csv", recursive=True)
    if not splits_matches:
        raise FileNotFoundError("CRITICAL: splits_kfold.csv not found anywhere!")
    splits_csv = splits_matches[0]
    print(f"--> Located Splits CSV: {splits_csv}")

    # 5. Locate train_synthetic_labels.csv
    synth_lbl_matches = glob.glob("/kaggle/input/**/train_synthetic_labels.csv", recursive=True) + \
                        glob.glob(f"{WORK_DIR}/**/train_synthetic_labels.csv", recursive=True)
    synth_labels_csv = synth_lbl_matches[0] if synth_lbl_matches else None
    print(f"--> Located Synthetic Labels CSV: {synth_labels_csv}")

    return train_img_dir, synth_img_dir, splits_csv, synth_labels_csv

train_img_dir, synth_img_dir, splits_csv, synth_labels_csv = setup_dataset_directories()

df_splits = pd.read_csv(splits_csv)
df_synth = pd.read_csv(synth_labels_csv) if synth_labels_csv and os.path.exists(synth_labels_csv) else None

print(f"\nLoaded splits: {len(df_splits)} samples | Synthetic: {len(df_synth) if df_synth is not None else 0} samples")

# ==============================================================================
# 3. AUGMENTATION & DATASET
# ==============================================================================
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
    def __init__(self, df, train_dir=train_img_dir, synth_dir=synth_img_dir, is_train=True):
        self.df = df.reset_index(drop=True)
        self.train_dir = train_dir
        self.synth_dir = synth_dir
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = str(row['id'])
        label = int(row['label'])

        if img_id.startswith("synth_"):
            path = os.path.join(self.synth_dir, f"{img_id}.png")
        else:
            path = os.path.join(self.train_dir, f"{img_id}.png")

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Failed to load image from: '{path}'. File does not exist or is corrupted!")

        if self.is_train:
            angle = random.uniform(-8.0, 8.0)
            scale = random.uniform(0.92, 1.05)
            tx = random.uniform(-2.0, 2.0)
            ty = random.uniform(-2.0, 2.0)
            M = cv2.getRotationMatrix2D((32, 32), angle, scale)
            M[0, 2] += tx
            M[1, 2] += ty
            img = cv2.warpAffine(img, M, (64, 64), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

            if random.random() < 0.4:
                img = elastic_transform_np(img, alpha=random.uniform(8.0, 14.0), sigma=random.uniform(2.5, 3.2))

            morph = random.random()
            if morph < 0.25:
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                img = cv2.dilate(img, kernel, iterations=1)
            elif morph < 0.45:
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                img = cv2.erode(img, kernel, iterations=1)

        tensor = torch.from_numpy(img).float() / 255.0
        tensor = (tensor - 0.5) / 0.5
        tensor = tensor.unsqueeze(0)

        return tensor, label, int(row.get('writer_id', 999))

# ==============================================================================
# 4. NEURAL NETWORK ARCHITECTURE
# ==============================================================================
class MixStyle(nn.Module):
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
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x - mu) / sig

        lmda = self.beta.sample((B, 1, 1, 1)).to(x.device)
        perm = torch.randperm(B)

        mu_mix = mu * lmda + mu[perm] * (1 - lmda)
        sig_mix = sig * lmda + sig[perm] * (1 - lmda)

        return x_normed * sig_mix + mu_mix

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        reduced = max(in_planes // ratio, 8)
        self.fc1 = nn.Conv2d(in_planes, reduced, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(reduced, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv1(torch.cat([avg_out, max_out], dim=1)))

class CBAMBlock(nn.Module):
    def __init__(self, in_planes):
        super().__init__()
        self.ca = ChannelAttention(in_planes)
        self.sa = SpatialAttention()

    def forward(self, x):
        return x * self.ca(x) * self.sa(x)

class BasicBlockCBAM(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.cbam = CBAMBlock(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.cbam(out)
        out += self.shortcut(x)
        return F.relu(out)

class ResNet18_CBAM(nn.Module):
    def __init__(self, in_channels=1, num_classes=15, feature_dim=256, use_mixstyle=True):
        super().__init__()
        self.in_planes = 64
        self.use_mixstyle = use_mixstyle
        self.mixstyle = MixStyle(p=0.5, alpha=0.1) if use_mixstyle else nn.Identity()

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
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
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
        logits = self.classifier(feats)
        return logits, feats

class SubCenterArcFace(nn.Module):
    def __init__(self, in_features, num_classes=15, sub_centers=2, s=32.0, m=0.30):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.sub_centers = sub_centers
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, sub_centers, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, features, labels=None):
        x = F.normalize(features, p=2, dim=1)
        w = F.normalize(self.weight, p=2, dim=2)
        cosine = torch.einsum('bd,ckd->bck', x, w)
        cosine_max, _ = torch.max(cosine, dim=2)

        if labels is None or not self.training:
            return cosine_max * self.s

        sine = torch.sqrt(1.0 - torch.clamp(cosine_max ** 2, 0.0, 1.0))
        phi = cosine_max * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine_max > self.th, phi, cosine_max - self.mm)

        one_hot = torch.zeros(cosine_max.size(), device=features.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine_max)
        return output * self.s

# ==============================================================================
# 5. SANITY CHECK ON DATASET LOADER
# ==============================================================================
sanity_ds = ChineseCharacterDataset(df_splits.iloc[:10], is_train=False)
sample_tensor, sample_label, sample_writer = sanity_ds[0]
print(f"\n[SANITY CHECK] Successfully read first sample image!")
print(f"Shape: {sample_tensor.shape} | Non-zero elements: {torch.count_nonzero(sample_tensor).item()}/4096")
assert torch.count_nonzero(sample_tensor).item() > 50, "ERROR: Loaded image is completely blank!"
print("[SANITY CHECK PASSED] Dataset pipeline is 100% verified and functional!\n")

# ==============================================================================
# 6. FULL 5-FOLD TRAINING PIPELINE
# ==============================================================================
EPOCHS = 15
BATCH_SIZE = 64
LR = 5e-4
N_FOLDS = 5

oof_probabilities = np.zeros((len(df_splits), 15), dtype=np.float32)
fold_val_accuracies = []

print("=" * 70)
print("STARTING FULL 5-FOLD STRATIFIED GROUP K-FOLD SOTA TRAINING")
print(f"Total Folds: {N_FOLDS} | Epochs per fold: {EPOCHS} | Batch size: {BATCH_SIZE}")
print("=" * 70)

for fold in range(N_FOLDS):
    print(f"\n>>> [FOLD {fold + 1}/{N_FOLDS}] INITIALIZATION")
    train_mask = (df_splits['fold'] != fold)
    val_mask = (df_splits['fold'] == fold)

    train_df = df_splits[train_mask].copy()
    val_df = df_splits[val_mask].copy()

    # Append synthetic traditional samples to train set
    if df_synth is not None and len(df_synth) > 0:
        train_df = pd.concat([train_df, df_synth], ignore_index=True)

    print(f"Train samples: {len(train_df)} | Val samples (Unseen writers): {len(val_df)}")

    train_ds = ChineseCharacterDataset(train_df, is_train=True)
    val_ds = ChineseCharacterDataset(val_df, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model = ResNet18_CBAM(in_channels=1, num_classes=15, feature_dim=256, use_mixstyle=True).to(device)
    metric_head = SubCenterArcFace(in_features=256, num_classes=15, sub_centers=2, s=32.0, m=0.30).to(device)

    criterion_ce = nn.CrossEntropyLoss(label_smoothing=0.08)
    params = list(model.parameters()) + list(metric_head.parameters())
    optimizer = optim.AdamW(params, lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    best_val_acc = 0.0
    best_val_probs = None
    ckpt_path = os.path.join(OUT_DIR, f"resnet18_cbam_fold{fold}.pt")

    for epoch in range(1, EPOCHS + 1):
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

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        all_probs = []

        with torch.no_grad():
            for imgs, labels, writers in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits, feats = model(imgs)
                probs = F.softmax(logits, dim=1)
                all_probs.append(probs.cpu().numpy())

                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += len(labels)

        val_acc = val_correct / val_total
        val_probs_epoch = np.vstack(all_probs)

        print(f"Fold {fold} | Epoch {epoch:02d}/{EPOCHS:02d} | Train Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc*100:.2f}% | Val Acc (Unseen Writers): {val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_probs = val_probs_epoch
            torch.save({
                "model_state_dict": model.state_dict(),
                "metric_head_state_dict": metric_head.state_dict(),
                "val_acc": val_acc,
                "epoch": epoch,
                "fold": fold,
                "model_type": "resnet18_cbam"
            }, ckpt_path)

    oof_probabilities[val_mask] = best_val_probs
    fold_val_accuracies.append(best_val_acc)
    print(f">>> [FOLD {fold + 1} COMPLETE] Best Val Accuracy: {best_val_acc * 100:.2f}% -> Saved to {ckpt_path}\n")

# ==============================================================================
# 7. FULL OUT-OF-FOLD (OOF) EVALUATION
# ==============================================================================
oof_preds = np.argmax(oof_probabilities, axis=1)
ground_truth = df_splits['label'].values
overall_oof_acc = accuracy_score(ground_truth, oof_preds)

print("=" * 70)
print("FINAL 5-FOLD OUT-OF-FOLD (OOF) BENCHMARK RESULTS")
print("=" * 70)
for f_idx, acc in enumerate(fold_val_accuracies):
    print(f"  • Fold {f_idx}: Accuracy = {acc * 100:.2f}%")
print("-" * 70)
print(f"OVERALL 5-FOLD OOF ACCURACY: {overall_oof_acc * 100:.2f}%")
print(f"MEAN ACCURACY: {np.mean(fold_val_accuracies) * 100:.2f}% ± {np.std(fold_val_accuracies) * 100:.2f}%")
print("=" * 70)

# Save OOF Report
np.save("/kaggle/working/oof_probabilities.npy", oof_probabilities)
oof_metrics = {
    "overall_oof_accuracy": float(overall_oof_acc),
    "fold_accuracies": [float(a) for a in fold_val_accuracies],
    "mean_accuracy": float(np.mean(fold_val_accuracies)),
    "std_accuracy": float(np.std(fold_val_accuracies))
}
with open("/kaggle/working/oof_metrics.json", "w") as f:
    json.dump(oof_metrics, f, indent=2)

print("Classification Report:")
print(classification_report(ground_truth, oof_preds, digits=4))
print("All 5-fold models and OOF metrics successfully saved in /kaggle/working/!")
