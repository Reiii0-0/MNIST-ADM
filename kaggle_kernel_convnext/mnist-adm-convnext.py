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
    for z in glob.glob("/kaggle/input/**/*.zip", recursive=True):
        print(f"Extracting zip archive: {z} -> {WORK_DIR}...")
        try:
            with zipfile.ZipFile(z, 'r') as zip_ref:
                zip_ref.extractall(WORK_DIR)
        except Exception as e:
            print(f"Warning extracting {z}: {e}")

    dataset_root = None
    for root, dirs, files in os.walk("/kaggle/input"):
        if "splits_kfold.csv" in files or "train_labels.csv" in files:
            dataset_root = root
            break
    if not dataset_root:
        dataset_root = "/kaggle/input"
    print(f"Resolved Dataset Root: {dataset_root}")
    return dataset_root

dataset_root = setup_dataset_directories()

splits_csv_candidates = [
    os.path.join(dataset_root, "splits_kfold.csv"),
    "/kaggle/input/chinese-mnist-adm/splits_kfold.csv",
    "/kaggle/input/splits_kfold.csv"
]
splits_path = None
for c in splits_csv_candidates:
    if os.path.exists(c):
        splits_path = c
        break

if not splits_path:
    found = glob.glob("/kaggle/input/**/splits_kfold.csv", recursive=True)
    if found:
        splits_path = found[0]

print(f"Using splits_kfold.csv from: {splits_path}")
df_splits = pd.read_csv(splits_path)
print(f"Total training images in splits: {len(df_splits)}")

synth_candidates = [
    os.path.join(dataset_root, "train_synthetic_labels.csv"),
    "/kaggle/input/chinese-mnist-adm/train_synthetic_labels.csv",
    "/kaggle/input/train_synthetic_labels.csv"
]
synth_path = None
for c in synth_candidates:
    if os.path.exists(c):
        synth_path = c
        break
if not synth_path:
    found_s = glob.glob("/kaggle/input/**/train_synthetic_labels.csv", recursive=True)
    if found_s:
        synth_path = found_s[0]

df_synth = None
if synth_path and os.path.exists(synth_path):
    print(f"Using train_synthetic_labels.csv from: {synth_path}")
    df_synth = pd.read_csv(synth_path)
    print(f"Synthetic traditional samples loaded: {len(df_synth)}")

print("Indexing all PNG image files in /kaggle and /tmp...")
image_path_map = {}
for search_dir in ["/kaggle/input", WORK_DIR]:
    if os.path.exists(search_dir):
        for p in glob.glob(os.path.join(search_dir, "**", "*.png"), recursive=True):
            stem = os.path.splitext(os.path.basename(p))[0]
            if stem not in image_path_map:
                image_path_map[stem] = p

print(f"Total mapped unique image files: {len(image_path_map)}")

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

def augment_character(img, p=0.8):
    if random.random() > p:
        return img
    img = img.copy()
    # Rotation (-10 to +10 deg)
    angle = random.uniform(-10.0, 10.0)
    # Scale (0.88 to 1.12)
    scale = random.uniform(0.88, 1.12)
    # Translation
    tx = random.uniform(-3.0, 3.0)
    ty = random.uniform(-3.0, 3.0)

    M = cv2.getRotationMatrix2D((32, 32), angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    img = cv2.warpAffine(img, M, (64, 64), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # Elastic warp (30% chance)
    if random.random() < 0.30:
        img = elastic_transform_np(img, alpha=random.uniform(8.0, 14.0), sigma=random.uniform(2.5, 3.2))

    # Dilation or erosion (30% chance)
    r = random.random()
    if r < 0.15:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        img = cv2.dilate(img, kernel, iterations=1)
    elif r < 0.30:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        img = cv2.erode(img, kernel, iterations=1)

    if img.shape != (64, 64):
        img = cv2.resize(img, (64, 64))

    return img

class ChineseCharacterDataset(Dataset):
    def __init__(self, df, path_map, is_train=True):
        self.df = df.reset_index(drop=True)
        self.path_map = path_map
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = str(row["id"])
        label = int(row["label"])

        img_path = self.path_map.get(img_id, None)
        if img_path and os.path.exists(img_path):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        else:
            img = np.zeros((64, 64), dtype=np.uint8)

        if img.shape != (64, 64):
            img = cv2.resize(img, (64, 64))

        if self.is_train:
            img = augment_character(img, p=0.85)

        if img.shape != (64, 64):
            img = cv2.resize(img, (64, 64))

        tensor = torch.from_numpy(img).float() / 255.0
        tensor = (tensor - 0.5) / 0.5
        tensor = tensor.unsqueeze(0)
        return tensor, torch.tensor(label, dtype=torch.long)


# ==============================================================================
# 4. CONVNEXT-FEMTO ARCHITECTURE (Liu et al., CVPR 2022)
# ==============================================================================
class LayerNorm2d(nn.Module):
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
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        return input + x

class ConvNeXt_Femto(nn.Module):
    def __init__(self, in_channels=1, num_classes=15, feature_dim=256, depths=(2, 2, 4, 2), dims=(48, 96, 192, 384)):
        super().__init__()
        self.downsample_layers = nn.ModuleList()

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
            stage = nn.Sequential(
                *[ConvNeXtBlock(dims[i]) for _ in range(depths[i])]
            )
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
        output *= self.s
        return output

# ==============================================================================
# 5. TRAINING LOOP (5-FOLD GROUP-KFOLD)
# ==============================================================================
def train_and_eval_fold(fold, df_train_fold, df_val_fold, epochs=15, batch_size=64):
    print("\n" + "=" * 70)
    print(f"STARTING TRAINING: FOLD {fold} (ConvNeXt-Femto SOTA)")
    print(f"Train samples: {len(df_train_fold)} | Val samples (Unseen Writers): {len(df_val_fold)}")
    print("=" * 70)

    train_ds = ChineseCharacterDataset(df_train_fold, image_path_map, is_train=True)
    val_ds = ChineseCharacterDataset(df_val_fold, image_path_map, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = ConvNeXt_Femto(in_channels=1, num_classes=15, feature_dim=256).to(device)
    metric_head = SubCenterArcFace(in_features=256, num_classes=15, sub_centers=2, s=32.0, m=0.30).to(device)

    optimizer = optim.AdamW(list(model.parameters()) + list(metric_head.parameters()), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    ce_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.08)

    best_val_acc = 0.0
    best_probs = None
    save_path = os.path.join(OUT_DIR, f"convnext_femto_fold{fold}.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        metric_head.train()
        total_loss = 0.0
        correct_train = 0
        total_train = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            logits, feats = model(imgs)
            metric_logits = metric_head(feats, labels)

            loss = 0.5 * ce_loss_fn(logits, labels) + 0.5 * ce_loss_fn(metric_logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(list(model.parameters()) + list(metric_head.parameters()), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            correct_train += (preds == labels).sum().item()
            total_train += imgs.size(0)

        scheduler.step()
        train_acc = correct_train / total_train
        train_loss = total_loss / total_train

        # Validation
        model.eval()
        metric_head.eval()
        correct_val = 0
        total_val = 0
        fold_probs = []

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits, feats = model(imgs)
                metric_out = metric_head(feats)
                # Ensemble probability between linear head and metric head
                p1 = F.softmax(logits, dim=1)
                p2 = F.softmax(metric_out, dim=1)
                p = 0.5 * (p1 + p2)

                preds = p.argmax(dim=1)
                correct_val += (preds == labels).sum().item()
                total_val += imgs.size(0)
                fold_probs.append(p.cpu().numpy())

        val_acc = correct_val / total_val
        fold_probs = np.concatenate(fold_probs, axis=0)

        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_probs = fold_probs
            torch.save({
                "model_state_dict": model.state_dict(),
                "metric_head_state_dict": metric_head.state_dict(),
                "model_type": "convnext_femto",
                "val_acc": val_acc,
                "fold": fold,
                "epoch": epoch
            }, save_path)

    print(f"[+] Fold {fold} Best Val Acc: {best_val_acc * 100:.2f}% saved to {save_path}")
    return best_val_acc, best_probs

# ==============================================================================
# 6. FULL 5-FOLD EXECUTION
# ==============================================================================
oof_probabilities = np.zeros((len(df_splits), 15), dtype=np.float32)
fold_accuracies = []

for fold in range(5):
    val_mask = (df_splits["fold"] == fold)
    train_mask = ~val_mask

    df_train_fold = df_splits[train_mask].copy()
    if df_synth is not None:
        df_train_fold = pd.concat([df_train_fold, df_synth], ignore_index=True)

    df_val_fold = df_splits[val_mask].copy()

    val_acc, best_probs = train_and_eval_fold(fold, df_train_fold, df_val_fold, epochs=15, batch_size=64)
    fold_accuracies.append(val_acc)
    oof_probabilities[val_mask] = best_probs

overall_oof_acc = accuracy_score(df_splits["label"].values, oof_probabilities.argmax(axis=1))
print("\n" + "=" * 70)
print(f"CONVNEXT-FEMTO 5-FOLD OVERALL OOF ACCURACY: {overall_oof_acc * 100:.2f}%")
print(f"Fold Accuracies: {[f'{a*100:.2f}%' for a in fold_accuracies]}")
print("=" * 70)

# Save OOF outputs
np.save("/kaggle/working/oof_probabilities_convnext.npy", oof_probabilities)
oof_metrics = {
    "model_type": "ConvNeXt_Femto",
    "overall_oof_accuracy": overall_oof_acc,
    "fold_accuracies": fold_accuracies,
    "macro_f1": float(classification_report(df_splits["label"].values, oof_probabilities.argmax(axis=1), output_dict=True)["macro avg"]["f1-score"])
}
with open("/kaggle/working/oof_metrics_convnext.json", "w") as f:
    json.dump(oof_metrics, f, indent=2)

print("ALL TRAINING COMPLETE! Checkpoints ready in /kaggle/working/models.")
