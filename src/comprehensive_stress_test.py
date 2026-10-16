#!/usr/bin/env python3
"""
Comprehensive Stress-Testing & Robustness Evaluation
Tests 5-Fold Ensemble & TTA against all forms of spatial/morphological degradation:
- Gaussian Blur (Mild & Heavy)
- Stroke Dilation (Thickening)
- Stroke Erosion (Thinning/Fading)
- Compound Committee Degradation (Thickening + Blur)
- Additive Noise / Specks
"""

import os
import sys
import glob
import json
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn.functional as F

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.models import ResNet18_CBAM
from src.prepare_submission import tta_variants
from src.config import (
    get_splits_kfold_csv,
    get_train_dir,
    get_train_synthetic_dir,
    get_train_synthetic_labels_csv,
    OUTPUTS_DIR,
    MODELS_DIR
)


def perturb_image(img, mode="clean"):
    """
    Apply precise controlled perturbations to a 64x64 uint8 grayscale image.
    """
    img_mod = img.copy()
    if mode == "clean":
        pass
    elif mode == "mild_blur":
        img_mod = cv2.GaussianBlur(img_mod, (3, 3), 0.8)
    elif mode == "heavy_blur":
        img_mod = cv2.GaussianBlur(img_mod, (7, 7), 1.8)
    elif mode == "stroke_dilation":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        img_mod = cv2.dilate(img_mod, kernel, iterations=1)
    elif mode == "heavy_dilation":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        img_mod = cv2.dilate(img_mod, kernel, iterations=1)
    elif mode == "stroke_erosion":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        img_mod = cv2.erode(img_mod, kernel, iterations=1)
    elif mode == "committee_compound":
        # Committee noted: Already thick strokes + light blur
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        img_mod = cv2.dilate(img_mod, kernel, iterations=1)
        img_mod = cv2.GaussianBlur(img_mod, (3, 3), 0.8)
    elif mode == "additive_noise":
        noise = np.random.normal(0, 15, img_mod.shape).astype(np.float32)
        img_mod = np.clip(img_mod.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    elif mode == "ink_specks":
        # Random salt-and-pepper noise
        num_specks = 40
        for _ in range(num_specks):
            rx, ry = np.random.randint(0, 64, size=2)
            img_mod[ry, rx] = 255 if np.random.rand() > 0.5 else 0
    return img_mod

def to_tensor(img, device):
    t = torch.from_numpy(img).float() / 255.0
    t = (t - 0.5) / 0.5
    return t.unsqueeze(0).unsqueeze(0).to(device)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 80)
    print("COMPREHENSIVE MULTI-SCENARIO STRESS-TESTING (ROBUSTNESS AUDIT)")
    print(f"Device: {device}")
    print("=" * 80)

    # Load 5-Fold models
    ckpt_paths = sorted(glob.glob(str(PROJECT_ROOT / "models" / "kaggle_outputs" / "models" / "*.pt")))
    if not ckpt_paths:
        ckpt_paths = sorted(glob.glob(str(PROJECT_ROOT / "models" / "*.pt")))
    print(f"Loaded {len(ckpt_paths)} checkpoint models.")

    models = []
    for p in ckpt_paths:
        m = ResNet18_CBAM(in_channels=1, num_classes=15, feature_dim=256, use_mixstyle=False).to(device)
        ckpt = torch.load(p, map_location=device)
        m.load_state_dict(ckpt["model_state_dict"])
        m.eval()
        models.append(m)

    # Prepare a robust evaluation set of 300 diverse samples (20 per class across unseen writers + synthetic traditional)
    df_splits = pd.read_csv(get_splits_kfold_csv())
    df_val = df_splits[df_splits["fold"] == 0]  # Fold 0 validation set (14 unseen writers, 2100 images)

    eval_samples = []
    train_dir = get_train_dir()
    for cls in range(15):
        cls_sub = df_val[df_val["label"] == cls].head(15)
        for _, row in cls_sub.iterrows():
            img_path = train_dir / f"{row['id']}.png"
            if img_path.exists():
                eval_samples.append((str(img_path), int(cls)))

    # Add 50 synthetic traditional samples for classes 13 & 14
    synth_dir = get_train_synthetic_dir()
    synth_csv = get_train_synthetic_labels_csv()
    synth_imgs = sorted(glob.glob(str(synth_dir / "*.png")))[:50]
    if synth_csv.exists():
        df_synth = pd.read_csv(synth_csv).set_index("id")
        for sp in synth_imgs:
            stem = Path(sp).stem
            if stem in df_synth.index:
                eval_samples.append((sp, int(df_synth.loc[stem, "label"])))


    print(f"Evaluating {len(eval_samples)} stratified samples across 15 classes under 8 degradation regimes.\n")

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
    summary_results = {}

    for name, mode in scenarios:
        correct_single = 0
        correct_ensemble = 0
        correct_ensemble_tta = 0
        total = 0

        for img_path, true_label in eval_samples:
            raw = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if raw is None:
                continue
            if raw.shape != (64, 64):
                raw = cv2.resize(raw, (64, 64))

            pert = perturb_image(raw, mode=mode)
            inp = to_tensor(pert, device)

            # 1. Single Model (Fold 0)
            with torch.no_grad():
                l_single, _ = models[0](inp)
                pred_single = l_single.argmax(dim=1).item()
                if pred_single == true_label:
                    correct_single += 1

            # 2. 5-Model Ensemble (No TTA)
            model_probs = []
            with torch.no_grad():
                for m in models:
                    logits, _ = m(inp)
                    prob = F.softmax(logits, dim=1).squeeze(0)
                    model_probs.append(prob)
                ens_prob = torch.stack(model_probs, dim=0).mean(dim=0)
                pred_ens = ens_prob.argmax().item()
                if pred_ens == true_label:
                    correct_ensemble += 1

            # 3. 5-Model Ensemble + 5-Way TTA (Power-Mean p=1.8)
            variants = tta_variants(inp)
            var_probs_list = []
            with torch.no_grad():
                for v_idx, v in enumerate(variants):
                    v_model_probs = []
                    for m in models:
                        l_v, _ = m(v)
                        v_model_probs.append(F.softmax(l_v, dim=1).squeeze(0))
                    v_prob = torch.stack(v_model_probs, dim=0).mean(dim=0)
                    var_probs_list.append((v_prob ** 1.8) * tta_weights[v_idx])
                power_sum = torch.stack(var_probs_list, dim=0).sum(dim=0)
                pred_tta = (power_sum ** (1.0 / 1.8)).argmax().item()
                if pred_tta == true_label:
                    correct_ensemble_tta += 1

            total += 1

        acc_single = (correct_single / total) * 100
        acc_ens = (correct_ensemble / total) * 100
        acc_tta = (correct_ensemble_tta / total) * 100

        summary_results[name] = {
            "single_acc": acc_single,
            "ensemble_acc": acc_ens,
            "ensemble_tta_acc": acc_tta,
            "total": total
        }

        print(f"[{name:<32}] Single: {acc_single:6.2f}% | 5-Fold Ens: {acc_ens:6.2f}% | Ens+TTA: {acc_tta:6.2f}%")

    # Save metrics JSON
    with open(OUTPUTS_DIR / "stress_test_metrics.json", "w") as f:
        json.dump(summary_results, f, indent=2)
    print(f"\n[+] Full metrics saved to {OUTPUTS_DIR / 'stress_test_metrics.json'}")


if __name__ == "__main__":
    main()
