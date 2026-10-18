#!/usr/bin/env python3
"""
Production Evaluator and Tri-Ensemble Blending Pipeline:
Evaluates DenseNet-121 CBAM (trained on Kaishu + Caoshu + Xingshu) on 2,466 Phase 1 Validation Images,
then combines with ResNet-18 CBAM and ConvNeXt-Femto to form the Champion Tri-Ensemble.
Finally applies Dynamic Unsupervised Expectation-Maximization (TTPA) with Dirichlet MAP smoothing.
"""

import os
import sys
import glob
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.models import DenseNet121_CBAM, ResNet18_CBAM, ConvNeXt_Femto
from src.config import get_validation_dir, MODELS_DIR, SUBMISSIONS_DIR
from src.test_time_prior_adaptation import adapt_test_time_priors


def load_img(path: Path) -> torch.Tensor:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        img = np.zeros((64, 64), dtype=np.uint8)
    if img.shape != (64, 64):
        img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(img.astype(np.float32) / 127.5 - 1.0)
    return t.unsqueeze(0).unsqueeze(0)  # (1, 1, 64, 64)


def get_tta_variants(tensor: torch.Tensor) -> list[torch.Tensor]:
    """5-Way TTA: Original, Rot +3°, Rot -3°, Shift right 1.5px, Shift left 1.5px."""
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

    # Shift dx=+1.5 px
    theta_r = torch.tensor([[1, 0, 0.045], [0, 1, 0]], dtype=torch.float, device=device).unsqueeze(0).repeat(B, 1, 1)
    grid_r = F.affine_grid(theta_r, tensor.size(), align_corners=False)
    variants.append(F.grid_sample(tensor, grid_r, align_corners=False))

    # Shift dx=-1.5 px
    theta_l = torch.tensor([[1, 0, -0.045], [0, 1, 0]], dtype=torch.float, device=device).unsqueeze(0).repeat(B, 1, 1)
    grid_l = F.affine_grid(theta_l, tensor.size(), align_corners=False)
    variants.append(F.grid_sample(tensor, grid_l, align_corners=False))

    return variants


def evaluate_densenet(val_files: list[Path], ckpt_path: Path, device: str = "cuda") -> np.ndarray:
    print(f"\n🧠 Evaluating DenseNet-121 CBAM from: {ckpt_path.name}")
    model = DenseNet121_CBAM(in_channels=1, num_classes=15, feature_dim=256, use_mixstyle=False)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    all_probs = []
    with torch.no_grad():
        for i, f in enumerate(val_files):
            t = load_img(f).to(device)
            tta_tensors = get_tta_variants(t)
            batch = torch.cat(tta_tensors, dim=0)  # (5, 1, 64, 64)

            with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                logits, _ = model(batch)
                probs = F.softmax(logits, dim=1).mean(dim=0).cpu().numpy()
            all_probs.append(probs)

            if (i + 1) % 500 == 0 or (i + 1) == len(val_files):
                print(f"Processed {i+1}/{len(val_files)} validation images...")

    return np.array(all_probs)  # (2466, 15)


def evaluate_resnet18(val_files: list[Path], ckpt_path: Path, device: str = "cuda") -> np.ndarray:
    print(f"\n🧠 Evaluating ResNet-18 CBAM from: {ckpt_path.name}")
    model = ResNet18_CBAM(in_channels=1, num_classes=15, feature_dim=256, use_mixstyle=False)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    all_probs = []
    with torch.no_grad():
        for i, f in enumerate(val_files):
            t = load_img(f).to(device)
            tta_tensors = get_tta_variants(t)
            batch = torch.cat(tta_tensors, dim=0)

            with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                logits, _ = model(batch)
                probs = F.softmax(logits, dim=1).mean(dim=0).cpu().numpy()
            all_probs.append(probs)

            if (i + 1) % 500 == 0 or (i + 1) == len(val_files):
                print(f"Processed {i+1}/{len(val_files)} validation images...")

    return np.array(all_probs)


def evaluate_convnext(val_files: list[Path], ckpt_path: Path, device: str = "cuda") -> np.ndarray:
    print(f"\n🧠 Evaluating ConvNeXt-Femto from: {ckpt_path.name}")
    model = ConvNeXt_Femto(in_channels=1, num_classes=15, feature_dim=256)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    all_probs = []
    with torch.no_grad():
        for i, f in enumerate(val_files):
            t = load_img(f).to(device)
            tta_tensors = get_tta_variants(t)
            batch = torch.cat(tta_tensors, dim=0)

            with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                logits, _ = model(batch)
                probs = F.softmax(logits, dim=1).mean(dim=0).cpu().numpy()
            all_probs.append(probs)

            if (i + 1) % 500 == 0 or (i + 1) == len(val_files):
                print(f"Processed {i+1}/{len(val_files)} validation images...")

    return np.array(all_probs)


def run_tri_ensemble_pipeline():
    val_dir = get_validation_dir()
    val_files = sorted(list(val_dir.glob("*.png")), key=lambda p: int(p.stem.split("_")[1]) if "_" in p.stem else 0)
    print(f"Found {len(val_files)} validation images in {val_dir}")
    assert len(val_files) == 2466, f"Expected 2,466 validation images, got {len(val_files)}"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Find trained DenseNet checkpoints
    densenet_ckpts = sorted(list(MODELS_DIR.glob("densenet121_cbam_fold*.pt")))
    if not densenet_ckpts:
        raise FileNotFoundError(f"No DenseNet checkpoints found in {MODELS_DIR}")
    print(f"Found {len(densenet_ckpts)} DenseNet checkpoint(s): {[c.name for c in densenet_ckpts]}")

    densenet_prob_path = SUBMISSIONS_DIR / "probabilities_densenet121.npy"
    if densenet_prob_path.exists() and len(densenet_ckpts) == 5:
        print(f"Loading cached 5-Fold DenseNet probabilities from {densenet_prob_path.name}")
        densenet_probs = np.load(str(densenet_prob_path))
    else:
        densenet_prob_list = []
        for ckpt in densenet_ckpts:
            p = evaluate_densenet(val_files, ckpt, device=device)
            densenet_prob_list.append(p)
        densenet_probs = np.mean(densenet_prob_list, axis=0)
        np.save(str(densenet_prob_path), densenet_probs)
        print(f"✅ Saved DenseNet probabilities to {densenet_prob_path}")

    # 2. Check for newly trained 5-fold ResNet-18 checkpoints
    resnet_ckpts = sorted([c for c in MODELS_DIR.glob("resnet18_cbam_fold*.pt") if not c.name.startswith("._")])
    resnet_prob_path = SUBMISSIONS_DIR / "probabilities_resnet18_cbam_5fold.npy"
    if resnet_prob_path.exists() and len(resnet_ckpts) == 5:
        print(f"Loading cached 5-Fold ResNet-18 probabilities from {resnet_prob_path.name}")
        p_resnet = np.load(str(resnet_prob_path))
    elif len(resnet_ckpts) == 5:
        print(f"Found 5 newly trained ResNet-18 checkpoints: {[c.name for c in resnet_ckpts]}")
        resnet_prob_list = []
        for ckpt in resnet_ckpts:
            p = evaluate_resnet18(val_files, ckpt, device=device)
            resnet_prob_list.append(p)
        p_resnet = np.mean(resnet_prob_list, axis=0)
        np.save(str(resnet_prob_path), p_resnet)
        print(f"✅ Saved 5-Fold ResNet-18 probabilities to {resnet_prob_path}")
    else:
        p_resnet = np.load(str(SUBMISSIONS_DIR / "probabilities_phase1_anchor.npy"))
        print("Using anchor ResNet-18 probabilities.")

    # 3. ConvNeXt-Femto
    convnext_ckpts = sorted([c for c in MODELS_DIR.glob("convnext_femto_fold*.pt") if not c.name.startswith("._")])
    convnext_prob_path = SUBMISSIONS_DIR / "probabilities_convnext_femto_5fold.npy"
    if convnext_prob_path.exists() and len(convnext_ckpts) == 5:
        print(f"Loading cached 5-Fold ConvNeXt probabilities from {convnext_prob_path.name}")
        p_convnext = np.load(str(convnext_prob_path))
    elif len(convnext_ckpts) == 5:
        print(f"Found 5 newly trained ConvNeXt-Femto checkpoints: {[c.name for c in convnext_ckpts]}")
        convnext_prob_list = []
        for ckpt in convnext_ckpts:
            p = evaluate_convnext(val_files, ckpt, device=device)
            convnext_prob_list.append(p)
        p_convnext = np.mean(convnext_prob_list, axis=0)
        np.save(str(convnext_prob_path), p_convnext)
        print(f"✅ Saved 5-Fold ConvNeXt probabilities to {convnext_prob_path}")
    else:
        p_convnext = np.load(str(SUBMISSIONS_DIR / "probabilities_convnext.npy"))
        print("Using anchor ConvNeXt probabilities.")

    print("\n=======================================================")
    print("🏆 Forming Tri-Model Orthogonal Ensemble")
    print("=======================================================")
    print(f"Pilar 1 (ResNet-18 CBAM 5-Fold Anchor):    Shape {p_resnet.shape}")
    print(f"Pilar 2 (ConvNeXt-Femto 5-Fold SOTA):      Shape {p_convnext.shape}")
    print(f"Pilar 3 (DenseNet-121 CBAM Caoshu-Trained): Shape {densenet_probs.shape}")

    # Tri-Ensemble Blending Weights (Optimized: 50% ResNet, 20% ConvNeXt, 30% DenseNet)
    w_resnet = 0.50
    w_convnext = 0.20
    w_densenet = 0.30

    p_tri_visual = w_resnet * p_resnet + w_convnext * p_convnext + w_densenet * densenet_probs
    np.save(str(SUBMISSIONS_DIR / "probabilities_phase1_tri_ensemble.npy"), p_tri_visual)

    # Standalone visual prediction
    pred_visual = np.argmax(p_tri_visual, axis=1)

    # Apply Dynamic Unsupervised EM (TTPA with Dirichlet MAP smoothing alpha=1.05)
    print("\n⚡ Applying Dynamic Unsupervised EM Prior Adaptation (TTPA, alpha=1.05)...")
    p_ttpa, target_priors = adapt_test_time_priors(
        probs=p_tri_visual,
        alpha_dirichlet=1.05,
        temperature=1.05,
        max_iter=100
    )
    pred_ttpa = np.argmax(p_ttpa, axis=1)

    print(f"Estimated Target Priors (15 classes):\n{np.round(target_priors, 4)}")
    print(f"Total samples flipped by TTPA: {np.sum(pred_visual != pred_ttpa)}")

    # Generate Submission Files
    val_ids = [f.stem for f in val_files]

    # 1. Pure Visual Submission (100% Argmax, Zero Prior Modification)
    df_pure_visual = pd.DataFrame({"id": val_ids, "label": pred_visual})
    pure_csv = SUBMISSIONS_DIR / "submission_phase1_pure_visual_grandmaster_15model.csv"
    pure_zip = SUBMISSIONS_DIR / "submission_phase1_pure_visual_grandmaster_15model.zip"
    df_pure_visual.to_csv(pure_csv, index=False)
    with zipfile.ZipFile(pure_zip, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(pure_csv, arcname="predictions.csv")

    # Also keep legacy filename updated
    legacy_zip = SUBMISSIONS_DIR / "submission_phase1_pure_visual_5fold_densenet.zip"
    with zipfile.ZipFile(legacy_zip, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(pure_csv, arcname="predictions.csv")

    print(f"\n🎨 PURE VISUAL 15-MODEL GRANDMASTER SUBMISSION CREATED (Zero Prior Modification):")
    print(f"   CSV: {pure_csv}")
    print(f"   ZIP: {pure_zip} ({pure_zip.stat().st_size:,} bytes)")
    print("   Pure Visual Class Distribution:")
    print(df_pure_visual["label"].value_counts().sort_index().to_dict())

    # 2. Prior-Adapted Submission (Dynamic Bayesian EM / TTPA)
    df_sub = pd.DataFrame({"id": val_ids, "label": pred_ttpa})
    sub_csv = SUBMISSIONS_DIR / "submission_phase1_tri_ensemble_5fold_ttpa.csv"
    sub_zip = SUBMISSIONS_DIR / "submission_phase1_tri_ensemble_5fold_ttpa.zip"
    df_sub.to_csv(sub_csv, index=False)
    with zipfile.ZipFile(sub_zip, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(sub_csv, arcname="predictions.csv")

    print(f"\n⚡ DYNAMIC TTPA 5-FOLD SUBMISSION CREATED:")
    print(f"   CSV: {sub_csv}")
    print(f"   ZIP: {sub_zip} ({sub_zip.stat().st_size:,} bytes)")
    print("   TTPA Class Distribution:")
    print(df_sub["label"].value_counts().sort_index().to_dict())

    # Also overwrite the standard sota zip for convenience
    std_zip = SUBMISSIONS_DIR / "submission_phase1_tri_ensemble_sota.zip"
    with zipfile.ZipFile(std_zip, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(pure_csv, arcname="predictions.csv")

    # Comparison with Previous Sub 8 (Rank 1: 0.9465)
    sub8_csv = SUBMISSIONS_DIR / "submission_phase1_bayes_calibrated_blend.csv"
    if sub8_csv.exists():
        df_sub8 = pd.read_csv(sub8_csv)
        diff_pure = np.sum(df_pure_visual["label"] != df_sub8["label"])
        diff_ttpa = np.sum(df_sub["label"] != df_sub8["label"])
        print(f"\n📊 Differences against Sub 8 (Rank 1 @ 0.9465):")
        print(f"   Pure Visual vs Sub 8: {diff_pure} samples changed.")
        print(f"   TTPA vs Sub 8:        {diff_ttpa} samples changed.")



if __name__ == "__main__":
    run_tri_ensemble_pipeline()
