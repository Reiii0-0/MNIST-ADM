#!/usr/bin/env python3
"""
Evaluate newly retrained SOTA 5-Fold ResNet-18 CBAM models from Kaggle
on the 2,466 Phase 1 validation images.
Applies:
- 5-Fold Ensemble
- 5-Way Test-Time Augmentation (TTA)
- AdaBN Domain Adaptation
- Bayesian Prior Adaptation (Dirichlet MAP EM)
- Comparison against previous Sub 3 Champion Blend
"""

import os
import glob
import zipfile
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "kaggle_pulled" / "retrained_sota" / "models"
VAL_DIR = PROJECT_ROOT / "data" / "validasi" / "test"
OUT_SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
OUT_SUBMISSIONS_DIR.mkdir(exist_ok=True)

import sys
sys.path.insert(0, str(PROJECT_ROOT))
from kaggle_kernel.standalone_cpc_sota_pipeline import ResNet18_CBAM
from src.test_time_prior_adaptation import TestTimePriorAdapter

def load_and_preprocess_img(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        img = np.zeros((64, 64), dtype=np.uint8)
    if img.shape != (64, 64):
        img = cv2.resize(img, (64, 64))
    tensor = torch.from_numpy(img.astype(np.float32) / 127.5 - 1.0)
    return tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, 64, 64)

def tta_variants(tensor):
    variants = [tensor]
    B = tensor.size(0)
    device = tensor.device

    # Rot +3 deg
    theta_pos = torch.tensor([[np.cos(np.radians(3)), -np.sin(np.radians(3)), 0],
                              [np.sin(np.radians(3)),  np.cos(np.radians(3)), 0]], dtype=torch.float, device=device).unsqueeze(0).repeat(B, 1, 1)
    grid_pos = F.affine_grid(theta_pos, tensor.size(), align_corners=False)
    variants.append(F.grid_sample(tensor, grid_pos, align_corners=False))

    # Rot -3 deg
    theta_neg = torch.tensor([[np.cos(np.radians(-3)), -np.sin(np.radians(-3)), 0],
                              [np.sin(np.radians(-3)),  np.cos(np.radians(-3)), 0]], dtype=torch.float, device=device).unsqueeze(0).repeat(B, 1, 1)
    grid_neg = F.affine_grid(theta_neg, tensor.size(), align_corners=False)
    variants.append(F.grid_sample(tensor, grid_neg, align_corners=False))

    # Shift dx=+1.5
    theta_r = torch.tensor([[1, 0, 0.045], [0, 1, 0]], dtype=torch.float, device=device).unsqueeze(0).repeat(B, 1, 1)
    grid_r = F.affine_grid(theta_r, tensor.size(), align_corners=False)
    variants.append(F.grid_sample(tensor, grid_r, align_corners=False))

    # Shift dx=-1.5
    theta_l = torch.tensor([[1, 0, -0.045], [0, 1, 0]], dtype=torch.float, device=device).unsqueeze(0).repeat(B, 1, 1)
    grid_l = F.affine_grid(theta_l, tensor.size(), align_corners=False)
    variants.append(F.grid_sample(tensor, grid_l, align_corners=False))
    return variants

def apply_adabn(model, img_files, device, batch_size=64):
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.reset_running_stats()
            m.training = True
            m.momentum = None
    with torch.no_grad():
        for i in range(0, len(img_files), batch_size):
            batch_paths = img_files[i:i + batch_size]
            tensors = [load_and_preprocess_img(p) for p in batch_paths]
            batch_tensor = torch.cat(tensors, dim=0).to(device)
            _ = model(batch_tensor)
    model.eval()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Running inference on device: {device}")

    # Discover test images sorted numerically
    test_files = sorted(list(VAL_DIR.glob("*.png")), key=lambda p: int(p.stem.split('_')[-1]) if '_' in p.stem and p.stem.split('_')[-1].isdigit() else p.stem)
    img_ids = [p.stem for p in test_files]
    N = len(test_files)
    print(f"[*] Found {N} validation test images in {VAL_DIR}")
    assert N == 2466, f"Expected 2466 images, found {N}"

    raw_probs_path = OUT_SUBMISSIONS_DIR / "probabilities_phase1_retrained_resnet18.npy"
    if raw_probs_path.exists():
        print(f"[+] Found cached retrained probabilities: {raw_probs_path}")
        ensemble_probs = np.load(raw_probs_path)
    else:
        # Load 5-Fold checkpoints
        ckpt_files = sorted(list(MODELS_DIR.glob("resnet18_cbam_fold*.pt")))
        print(f"[*] Loaded {len(ckpt_files)} retrained fold checkpoints from {MODELS_DIR}")
        assert len(ckpt_files) == 5, f"Expected 5 checkpoints, found {len(ckpt_files)}"

        fold_probs = []

        for fold_idx, ckpt_path in enumerate(ckpt_files):
            print(f"\n---> Evaluating Retrained Fold {fold_idx}: {ckpt_path.name}")
            model = ResNet18_CBAM(in_channels=1, num_classes=15, feature_dim=256, use_mixstyle=True)
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            model = model.to(device)
            model.eval()

            # Apply AdaBN
            apply_adabn(model, test_files, device, batch_size=64)

            # Infer with 5-Way TTA
            probs_list = []
            batch_size = 64
            with torch.no_grad():
                for i in range(0, N, batch_size):
                    batch_paths = test_files[i:i + batch_size]
                    batch_tensors = torch.cat([load_and_preprocess_img(p) for p in batch_paths], dim=0).to(device)

                    # Generate 5 TTA variants
                    variants = tta_variants(batch_tensors)
                    var_probs = []
                    for var in variants:
                        logits, _ = model(var)
                        p = F.softmax(logits, dim=1)
                        var_probs.append(p)

                    # Power-mean TTA (power=1.8)
                    p_stack = torch.stack(var_probs, dim=0) ** 1.8
                    avg_p = p_stack.mean(dim=0)
                    avg_p = avg_p / avg_p.sum(dim=1, keepdim=True)
                    probs_list.append(avg_p.cpu().numpy())

            fold_p = np.vstack(probs_list)
            fold_probs.append(fold_p)

        # 5-Fold Mean Probabilities
        ensemble_probs = np.mean(fold_probs, axis=0)
        print(f"\n[+] 5-Fold ResNet18-CBAM Ensemble Computed! Shape: {ensemble_probs.shape}")
        np.save(raw_probs_path, ensemble_probs)
        print(f"[+] Saved raw retrained probabilities: {raw_probs_path}")


    # Check for ConvNeXt probabilities to form Blend
    convnext_probs_path = OUT_SUBMISSIONS_DIR / "probabilities_convnext.npy"
    if convnext_probs_path.exists():
        convnext_probs = np.load(convnext_probs_path)
        print(f"[+] Loaded ConvNeXt-Femto probabilities: shape {convnext_probs.shape}")
        # Blend: 85% Retrained ResNet-18 + 15% ConvNeXt
        blended_probs = 0.85 * ensemble_probs + 0.15 * convnext_probs
        blended_probs = blended_probs / blended_probs.sum(axis=1, keepdims=True)
    else:
        blended_probs = ensemble_probs

    # Apply Bayesian Prior Adaptation (TTPA via Dirichlet MAP EM)
    adapter = TestTimePriorAdapter(alpha_dirichlet=1.05, max_iter=100)
    adapted_probs = adapter.fit_transform(blended_probs)
    target_priors = adapter.target_prior_

    print("\n[+] Estimated Target Class Priors (TTPA):")
    for c in range(15):
        print(f"    Class {c:2d}: {target_priors[c]*100:5.2f}%")

    # Generate Final Predictions CSV
    preds = np.argmax(adapted_probs, axis=1)
    df_pred = pd.DataFrame({"id": img_ids, "label": preds})

    out_csv = OUT_SUBMISSIONS_DIR / "submission_phase1_retrained_sota.csv"
    df_pred.to_csv(out_csv, index=False)
    print(f"[+] Predictions CSV saved: {out_csv}")

    # Create Flat Root ZIP
    out_zip = OUT_SUBMISSIONS_DIR / "submission_phase1_retrained_sota.zip"
    with zipfile.ZipFile(out_zip, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(out_csv, arcname="predictions.csv")
    print(f"[+] Flat Root ZIP saved: {out_zip} ({out_zip.stat().st_size:,} bytes)")

    # Compare with Previous Sub 3 Champion Blend
    sub3_csv = OUT_SUBMISSIONS_DIR / "submission_phase1_blend_resnet_conv.csv"
    if sub3_csv.exists():
        df_sub3 = pd.read_csv(sub3_csv)
        agree_mask = df_pred['label'] == df_sub3['label']
        agree_count = agree_mask.sum()
        disagree_count = (~agree_mask).sum()
        print("\n" + "=" * 70)
        print("COMPARISON: NEW SOTA RETRAINED vs SUBMISSION 3 CHAMPION (0.9319)")
        print("=" * 70)
        print(f"Total Images         : {N}")
        print(f"Agreement Count      : {agree_count} ({agree_count/N*100:.2f}%)")
        print(f"Refined / Diverged   : {disagree_count} ({disagree_count/N*100:.2f}%)")

        print("\nTop Shifts (Sub 3 -> Retrained SOTA):")
        df_comp = pd.DataFrame({
            "sub3": df_sub3['label'],
            "new": df_pred['label']
        })
        shifts = df_comp[df_comp['sub3'] != df_comp['new']].groupby(['sub3', 'new']).size().reset_index(name='count')
        shifts = shifts.sort_values(by='count', ascending=False)
        print(shifts.head(15).to_string(index=False))

    print("\n" + "=" * 70)
    print("SUCCESS: New Retrained SOTA Submission is 100% Prepared!")
    print("=" * 70)

if __name__ == "__main__":
    main()
