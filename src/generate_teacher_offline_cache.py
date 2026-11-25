#!/usr/bin/env python3
"""
================================================================================
Generate Offline Teacher Ensemble Cache (15,918 Samples)
================================================================================
Precomputes:
1. `cached_teacher_logits.npy` (15918, 15) - Ensemble calibrated soft probabilities/logits
2. `cached_teacher_feats.npy` (15918, 256) - Mean normalized latent feature embeddings
Across 15 Grandmaster Checkpoints:
- 5 Folds ResNet-18 CBAM v2 (weight 0.40)
- 5 Folds DenseNet-121 CBAM (weight 0.35)
- 5 Folds ConvNeXt-Femto v2 (weight 0.25)
================================================================================
"""

import os
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from src.models import ResNet18_CBAM, ConvNeXt_Femto, SubCenterArcFace
from src.model_densenet_cbam import DenseNet121_CBAM

class InferenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.paths = df['absolute_path'].tolist()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((64, 64), dtype=np.uint8)
        if img.shape != (64, 64):
            img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)

        # Standard normalization [-1.0, 1.0]
        tensor = torch.from_numpy(img.astype(np.float32) / 127.5 - 1.0).unsqueeze(0)
        return tensor, idx

def main():
    base_dir = Path("/home/rei/Workspace/Academic/Multivariate Data Analysis/FinalProject")
    data_dir = base_dir / "data"
    manifest_csv = data_dir / "metadata" / "super_dataset_15918.csv"

    if not manifest_csv.exists():
        raise FileNotFoundError(f"Missing {manifest_csv}")

    df = pd.read_csv(manifest_csv)
    print(f"[*] Loaded super-dataset manifest: {len(df)} images.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    dataset = InferenceDataset(df)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    # 1. Load All 15 Models
    models_dict = {
        'resnet': [],
        'densenet': [],
        'convnext': []
    }

    print("\n[*] Loading 15 Teacher Ensemble checkpoints...")
    for fold in range(5):
        # ResNet18-CBAM v2
        r_path = base_dir / f"models/resnet18_cbam_v2_fold{fold}.pt"
        if r_path.exists():
            m_r = ResNet18_CBAM(num_classes=15).to(device).eval()
            h_r = SubCenterArcFace(in_features=256, num_classes=15, sub_centers=2).to(device).eval()
            ckpt_r = torch.load(r_path, map_location=device)
            m_r.load_state_dict(ckpt_r['model_state_dict'])
            h_r.load_state_dict(ckpt_r['metric_head_state_dict'])
            models_dict['resnet'].append((m_r, h_r))

        # DenseNet121-CBAM
        d_path = base_dir / f"models/densenet121_cbam_fold{fold}.pt"
        if d_path.exists():
            m_d = DenseNet121_CBAM(num_classes=15).to(device).eval()
            h_d = SubCenterArcFace(in_features=256, num_classes=15, sub_centers=2).to(device).eval()
            ckpt_d = torch.load(d_path, map_location=device)
            m_d.load_state_dict(ckpt_d['model_state_dict'])
            h_d.load_state_dict(ckpt_d['metric_head_state_dict'])
            models_dict['densenet'].append((m_d, h_d))

        # ConvNeXt-Femto v2
        c_path = base_dir / f"models/convnext_femto_v2_fold{fold}.pt"
        if c_path.exists():
            m_c = ConvNeXt_Femto(num_classes=15).to(device).eval()
            h_c = SubCenterArcFace(in_features=256, num_classes=15, sub_centers=2).to(device).eval()
            ckpt_c = torch.load(c_path, map_location=device)
            m_c.load_state_dict(ckpt_c['model_state_dict'])
            h_c.load_state_dict(ckpt_c['metric_head_state_dict'])
            models_dict['convnext'].append((m_c, h_c))

    print(f"  • Loaded ResNet folds: {len(models_dict['resnet'])}")
    print(f"  • Loaded DenseNet folds: {len(models_dict['densenet'])}")
    print(f"  • Loaded ConvNeXt folds: {len(models_dict['convnext'])}")

    total_teachers = len(models_dict['resnet']) + len(models_dict['densenet']) + len(models_dict['convnext'])
    if total_teachers == 0:
        raise RuntimeError("No teacher models found!")

    # 2. Forward Pass & Feature Caching with 3-Way TTA
    # Clean, Slight Rotation (+2 deg), Slight Rotation (-2 deg)
    print("\n[*] Precomputing Teacher Ensemble representations with TTA...")
    t0 = time.time()

    all_ensemble_probs = np.zeros((len(df), 15), dtype=np.float32)
    all_ensemble_feats = np.zeros((len(df), 256), dtype=np.float32)

    with torch.no_grad():
        for batch_imgs, batch_indices in loader:
            batch_imgs = batch_imgs.to(device)
            B = batch_imgs.size(0)

            batch_probs = torch.zeros(B, 15, device=device)
            batch_feats = torch.zeros(B, 256, device=device)

            # TTA Variations: 1. Clean, 2. Rot +2.5 deg, 3. Rot -2.5 deg
            tta_variants = [batch_imgs]
            # Affine rotations
            theta_pos = torch.tensor([
                [0.9990, -0.0436, 0.0],
                [0.0436,  0.9990, 0.0]
            ], device=device).unsqueeze(0).repeat(B, 1, 1)
            grid_pos = F.affine_grid(theta_pos, batch_imgs.size(), align_corners=False)
            tta_variants.append(F.grid_sample(batch_imgs, grid_pos, align_corners=False))

            theta_neg = torch.tensor([
                [ 0.9990, 0.0436, 0.0],
                [-0.0436, 0.9990, 0.0]
            ], device=device).unsqueeze(0).repeat(B, 1, 1)
            grid_neg = F.affine_grid(theta_neg, batch_imgs.size(), align_corners=False)
            tta_variants.append(F.grid_sample(batch_imgs, grid_neg, align_corners=False))

            # Weight assignments
            w_res = 0.40 / max(1, len(models_dict['resnet']))
            w_den = 0.35 / max(1, len(models_dict['densenet']))
            w_cnx = 0.25 / max(1, len(models_dict['convnext']))

            for v in tta_variants:
                # ResNets
                for m, h in models_dict['resnet']:
                    lin, f = m(v)
                    arc = h(f)
                    p = F.softmax(0.5 * lin + 0.5 * arc, dim=1)
                    batch_probs += (w_res / len(tta_variants)) * p
                    batch_feats += (w_res / len(tta_variants)) * F.normalize(f, p=2, dim=1)

                # DenseNets
                for m, h in models_dict['densenet']:
                    lin, f = m(v)
                    arc = h(f)
                    p = F.softmax(0.5 * lin + 0.5 * arc, dim=1)
                    batch_probs += (w_den / len(tta_variants)) * p
                    batch_feats += (w_den / len(tta_variants)) * F.normalize(f, p=2, dim=1)

                # ConvNeXts
                for m, h in models_dict['convnext']:
                    lin, f = m(v)
                    arc = h(f)
                    p = F.softmax(0.5 * lin + 0.5 * arc, dim=1)
                    batch_probs += (w_cnx / len(tta_variants)) * p
                    batch_feats += (w_cnx / len(tta_variants)) * F.normalize(f, p=2, dim=1)

            # Store in numpy arrays
            idx_np = batch_indices.numpy()
            all_ensemble_probs[idx_np] = batch_probs.cpu().numpy()
            all_ensemble_feats[idx_np] = batch_feats.cpu().numpy()

    elapsed = time.time() - t0
    print(f"\n[+] Teacher inference completed in {elapsed:.1f} s ({elapsed / len(df) * 1000:.2f} ms / image).")

    # Convert ensemble probabilities to scaled logits for distillation
    epsilon = 1e-7
    all_ensemble_probs = np.clip(all_ensemble_probs, epsilon, 1.0 - epsilon)
    all_ensemble_probs = all_ensemble_probs / all_ensemble_probs.sum(axis=1, keepdims=True)
    all_ensemble_logits = np.log(all_ensemble_probs)

    # Save to disk
    out_dir = data_dir / "teacher_cache"
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "cached_teacher_logits.npy", all_ensemble_logits)
    np.save(out_dir / "cached_teacher_probs.npy", all_ensemble_probs)
    np.save(out_dir / "cached_teacher_feats.npy", all_ensemble_feats)

    size_logits = os.path.getsize(out_dir / "cached_teacher_logits.npy") / (1024 * 1024)
    size_feats = os.path.getsize(out_dir / "cached_teacher_feats.npy") / (1024 * 1024)

    print("\n✅ [OFFLINE TEACHER CACHE CREATED]")
    print(f"  • Logits shape: {all_ensemble_logits.shape} ({size_logits:.2f} MB)")
    print(f"  • Features shape: {all_ensemble_feats.shape} ({size_feats:.2f} MB)")
    print(f"  • Saved in: {out_dir}")

    # Accuracy audit against 100% verified ground truth
    ground_truth = df['label'].values
    pred_labels = np.argmax(all_ensemble_probs, axis=1)
    acc = np.mean(pred_labels == ground_truth) * 100.0
    print(f"\n[*] Teacher Ensemble Super-Dataset Alignment: {acc:.2f}% ({np.sum(pred_labels == ground_truth)} / {len(df)})")

if __name__ == "__main__":
    main()
