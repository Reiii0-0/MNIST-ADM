#!/usr/bin/env python3
"""
Direct Comparative Stress-Testing (High-Speed Batched): ResNet18-CBAM vs ConvNeXt-Femto
Evaluates both 5-Fold architectures across 9 spatial and morphological degradation regimes:
1. Clean Baseline
2. Mild Blur (k=3, s=0.8)
3. Heavy Blur (k=7, s=1.8)
4. Stroke Dilation (2x2)
5. Heavy Dilation (3x3)
6. Stroke Erosion (2x2)
7. Committee Compound (Dilation 2x2 + Blur)
8. Additive Noise (Normal 15)
9. Salt & Pepper Ink Specks
"""

import os
import glob
import json
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn.functional as F
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT.parent))

from FinalProject.src.models import ResNet18_CBAM, ConvNeXt_Femto
from FinalProject.src.prepare_submission import tta_variants
from FinalProject.src.comprehensive_stress_test import perturb_image

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 90)
    print("HIGH-SPEED COMPARATIVE STRESS-TEST: ResNet18-CBAM vs ConvNeXt-Femto")
    print(f"Device: {device}")
    print("=" * 90)

    # 1. Load ResNet18-CBAM 5-Fold models
    resnet_ckpts = sorted(glob.glob(str(PROJECT_ROOT / "models" / "kaggle_outputs" / "models" / "resnet18_cbam_fold*.pt")))
    resnet_models = []
    for p in resnet_ckpts:
        m = ResNet18_CBAM(in_channels=1, num_classes=15, feature_dim=256, use_mixstyle=False).to(device)
        ckpt = torch.load(p, map_location=device)
        m.load_state_dict(ckpt["model_state_dict"])
        m.eval()
        resnet_models.append(m)
    print(f"Loaded {len(resnet_models)} ResNet18-CBAM models.")

    # 2. Load ConvNeXt-Femto 5-Fold models
    convnext_ckpts = sorted(glob.glob(str(PROJECT_ROOT / "models" / "kaggle_outputs_convnext" / "models" / "convnext_femto_fold*.pt")))
    if not convnext_ckpts:
        convnext_ckpts = sorted(glob.glob(str(PROJECT_ROOT / "models" / "kaggle_outputs_convnext" / "*.pt")))
    convnext_models = []
    for p in convnext_ckpts:
        m = ConvNeXt_Femto(in_channels=1, num_classes=15, feature_dim=256).to(device)
        ckpt = torch.load(p, map_location=device)
        m.load_state_dict(ckpt["model_state_dict"])
        m.eval()
        convnext_models.append(m)
    print(f"Loaded {len(convnext_models)} ConvNeXt-Femto models.")

    # 3. Prepare identical evaluation set
    df_splits = pd.read_csv(PROJECT_ROOT / "splits_kfold.csv")
    df_val = df_splits[df_splits["fold"] == 0]

    raw_images = []
    true_labels = []

    for cls in range(15):
        cls_sub = df_val[df_val["label"] == cls].head(15)
        for _, row in cls_sub.iterrows():
            img_path = PROJECT_ROOT / "train" / f"{row['id']}.png"
            if img_path.exists():
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    if img.shape != (64, 64):
                        img = cv2.resize(img, (64, 64))
                    raw_images.append(img)
                    true_labels.append(int(cls))

    synth_imgs = sorted(glob.glob(str(PROJECT_ROOT / "train_synthetic" / "*.png")))[:50]
    df_synth = pd.read_csv(PROJECT_ROOT / "train_synthetic_labels.csv").set_index("id")
    for sp in synth_imgs:
        stem = Path(sp).stem
        if stem in df_synth.index:
            img = cv2.imread(sp, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                if img.shape != (64, 64):
                    img = cv2.resize(img, (64, 64))
                raw_images.append(img)
                true_labels.append(int(df_synth.loc[stem, "label"]))

    total = len(raw_images)
    true_labels_tensor = torch.tensor(true_labels, device=device)
    print(f"Loaded {total} evaluation images across 15 classes.\n")

    scenarios = [
        ("Clean Baseline", "clean"),
        ("Mild Blur (k=3, s=0.8)", "mild_blur"),
        ("Heavy Blur (k=7, s=1.8)", "heavy_blur"),
        ("Stroke Dilation (2x2)", "stroke_dilation"),
        ("Heavy Dilation (3x3)", "heavy_dilation"),
        ("Stroke Erosion (2x2)", "stroke_erosion"),
        ("Committee Compound (Dilate+Blur)", "committee_compound"),
        ("Additive Noise (Normal 15)", "additive_noise"),
        ("Salt & Pepper Ink Specks", "ink_specks")
    ]

    tta_weights = [1.0, 0.8, 0.8, 0.7, 0.7]
    comparison_table = []

    for name, mode in scenarios:
        # Preprocess all perturbed images into a single tensor batch (N, 1, 64, 64)
        tensors = []
        for raw in raw_images:
            pert = perturb_image(raw, mode=mode)
            t = torch.from_numpy(pert).float() / 255.0
            t = (t - 0.5) / 0.5
            tensors.append(t.unsqueeze(0))
        batch = torch.stack(tensors, dim=0).to(device)  # (N, 1, 64, 64)

        # Batch TTA variants: list of (N, 1, 64, 64)
        # Using tta_variants on the entire batch directly!
        theta_pos = torch.tensor([[np.cos(np.radians(4)), -np.sin(np.radians(4)), 0],
                                  [np.sin(np.radians(4)),  np.cos(np.radians(4)), 0]], dtype=torch.float, device=device).unsqueeze(0).repeat(total, 1, 1)
        grid_pos = F.affine_grid(theta_pos, batch.size(), align_corners=False)
        v_rot_pos = F.grid_sample(batch, grid_pos, align_corners=False)

        theta_neg = torch.tensor([[np.cos(np.radians(-4)), -np.sin(np.radians(-4)), 0],
                                  [np.sin(np.radians(-4)),  np.cos(np.radians(-4)), 0]], dtype=torch.float, device=device).unsqueeze(0).repeat(total, 1, 1)
        grid_neg = F.affine_grid(theta_neg, batch.size(), align_corners=False)
        v_rot_neg = F.grid_sample(batch, grid_neg, align_corners=False)

        theta_sr = torch.tensor([[1, 0, 0.06], [0, 1, 0]], dtype=torch.float, device=device).unsqueeze(0).repeat(total, 1, 1)
        grid_sr = F.affine_grid(theta_sr, batch.size(), align_corners=False)
        v_shift_r = F.grid_sample(batch, grid_sr, align_corners=False)

        theta_sl = torch.tensor([[1, 0, -0.06], [0, 1, 0]], dtype=torch.float, device=device).unsqueeze(0).repeat(total, 1, 1)
        grid_sl = F.affine_grid(theta_sl, batch.size(), align_corners=False)
        v_shift_l = F.grid_sample(batch, grid_sl, align_corners=False)

        variants = [batch, v_rot_pos, v_rot_neg, v_shift_r, v_shift_l]

        # --- ResNet18-CBAM Inference ---
        with torch.no_grad():
            # Single (Fold 0)
            logits_r0, _ = resnet_models[0](batch)
            pred_r_single = logits_r0.argmax(dim=1)
            acc_r_single = (pred_r_single == true_labels_tensor).float().mean().item() * 100

            # 5-Fold Ens
            r_probs = [F.softmax(m(batch)[0], dim=1) for m in resnet_models]  # list of (N, 15)
            ens_r = torch.stack(r_probs, dim=0).mean(dim=0)  # (N, 15)
            pred_r_ens = ens_r.argmax(dim=1)
            acc_r_ens = (pred_r_ens == true_labels_tensor).float().mean().item() * 100

            # 5-Fold Ens + 5-Way TTA
            r_var_list = []
            for v_idx, v in enumerate(variants):
                v_p = torch.stack([F.softmax(m(v)[0], dim=1) for m in resnet_models]).mean(dim=0)
                r_var_list.append((v_p ** 1.8) * tta_weights[v_idx])
            pred_r_tta = torch.stack(r_var_list).sum(dim=0).argmax(dim=1)
            acc_r_tta = (pred_r_tta == true_labels_tensor).float().mean().item() * 100

        # --- ConvNeXt-Femto Inference ---
        with torch.no_grad():
            # Single (Fold 0)
            logits_c0, _ = convnext_models[0](batch)
            pred_c_single = logits_c0.argmax(dim=1)
            acc_c_single = (pred_c_single == true_labels_tensor).float().mean().item() * 100

            # 5-Fold Ens
            c_probs = [F.softmax(m(batch)[0], dim=1) for m in convnext_models]
            ens_c = torch.stack(c_probs, dim=0).mean(dim=0)
            pred_c_ens = ens_c.argmax(dim=1)
            acc_c_ens = (pred_c_ens == true_labels_tensor).float().mean().item() * 100

            # 5-Fold Ens + 5-Way TTA
            c_var_list = []
            for v_idx, v in enumerate(variants):
                v_p = torch.stack([F.softmax(m(v)[0], dim=1) for m in convnext_models]).mean(dim=0)
                c_var_list.append((v_p ** 1.8) * tta_weights[v_idx])
            pred_c_tta = torch.stack(c_var_list).sum(dim=0).argmax(dim=1)
            acc_c_tta = (pred_c_tta == true_labels_tensor).float().mean().item() * 100

        delta_ens = acc_r_ens - acc_c_ens
        delta_tta = acc_r_tta - acc_c_tta

        comparison_table.append({
            "scenario": name,
            "resnet_single": acc_r_single,
            "resnet_ens": acc_r_ens,
            "resnet_tta": acc_r_tta,
            "convnext_single": acc_c_single,
            "convnext_ens": acc_c_ens,
            "convnext_tta": acc_c_tta,
            "delta_ens": delta_ens,
            "delta_tta": delta_tta
        })

        print(f"[{name:<32}] ResNet Ens: {acc_r_ens:6.2f}% (TTA: {acc_r_tta:6.2f}%) | ConvNeXt Ens: {acc_c_ens:6.2f}% (TTA: {acc_c_tta:6.2f}%) | Delta: {delta_ens:+5.2f}%")

    with open(PROJECT_ROOT / "outputs" / "head_to_head_stress_test.json", "w") as f:
        json.dump(comparison_table, f, indent=2)

    print("\n[+] Head-to-head comparison saved to FinalProject/outputs/head_to_head_stress_test.json")

if __name__ == "__main__":
    main()
