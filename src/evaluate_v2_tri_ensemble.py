#!/usr/bin/env python3
"""
Production Evaluator and Tri-Orthogonal Ensemble Blending Pipeline for v2 Models (Taktik A).
Evaluates 15-Model Ensemble (5-Fold ResNet-18 CBAM + 5-Fold ConvNeXt-Femto + 5-Fold DenseNet-121 CBAM)
trained on the expanded 15,100 sample dataset with 7 Google Fonts and Targeted Stroke Perturbations.

Performs:
1. 5-Way Test-Time Augmentation (TTA) for all 15 models.
2. Orthogonal Probability Blending:
   P_tri = w_resnet * P_resnet + w_convnext * P_convnext + w_densenet * P_densenet
3. Pure Visual Prediction (strictly argmax, ZERO prior modification/filtering).
4. Detailed forensic comparison against Sub 8 (Rank 1 @ 0.9465) and Sub 10 (SOTA Pure Visual @ 0.9448).
5. Generation of submission zip package ready for CodaBench upload upon quota reset:
   submissions/submission_phase1_tri_orthogonal_v2_pure.zip
"""

import os
import sys
import zipfile
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

from src.models import ResNet18_CBAM, ConvNeXt_Femto, DenseNet121_CBAM
from src.config import get_validation_dir, MODELS_DIR, SUBMISSIONS_DIR

CLASS_NAMES = {
    0: "零 (0)", 1: "一 (1)", 2: "二 (2)", 3: "三 (3)", 4: "四 (4)",
    5: "五 (5)", 6: "六 (6)", 7: "七 (7)", 8: "八 (8)", 9: "九 (9)",
    10: "十 (10)", 11: "百 (11)", 12: "千 (12)", 13: "万 (13)", 14: "亿 (14)"
}

def load_img(path: Path) -> torch.Tensor:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        img = np.zeros((64, 64), dtype=np.uint8)
    if img.shape != (64, 64):
        img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(img.astype(np.float32) / 127.5 - 1.0)
    return t.unsqueeze(0).unsqueeze(0)


def get_tta_variants(tensor: torch.Tensor) -> list[torch.Tensor]:
    """5-Way TTA: Original, Rot +3°, Rot -3°, Shift dx=+1.5px, Shift dx=-1.5px."""
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


def evaluate_model_family(model_cls, ckpts: list[Path], val_files: list[Path], name: str, device: str = "cuda") -> np.ndarray:
    print(f"\n=======================================================")
    print(f"🧠 Evaluating 5-Fold Family: {name} ({len(ckpts)} checkpoints) with 5-Way TTA")
    print(f"=======================================================")

    fold_probs = []
    for ckpt_idx, ckpt_path in enumerate(ckpts):
        print(f"--> [{ckpt_idx+1}/{len(ckpts)}] Evaluating {ckpt_path.name} ...", end=" ", flush=True)
        if "densenet" in name.lower():
            model = model_cls(in_channels=1, num_classes=15, feature_dim=256, use_mixstyle=False)
        elif "resnet" in name.lower():
            model = model_cls(in_channels=1, num_classes=15, feature_dim=256, use_mixstyle=False)
        else:
            model = model_cls(in_channels=1, num_classes=15, feature_dim=256)

        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(device)
        model.eval()

        single_fold_probs = []
        with torch.no_grad():
            for f in val_files:
                t = load_img(f).to(device)
                tta_tensors = get_tta_variants(t)
                batch = torch.cat(tta_tensors, dim=0)

                with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                    logits, _ = model(batch)
                    probs = F.softmax(logits, dim=1).mean(dim=0).cpu().numpy()
                single_fold_probs.append(probs)

        fold_probs.append(np.array(single_fold_probs))
        print("Done!")

    avg_probs = np.mean(fold_probs, axis=0)  # (2466, 15)
    return avg_probs


def run_v2_evaluation_and_blend(w_resnet=0.45, w_convnext=0.25, w_densenet=0.30):
    val_dir = get_validation_dir()
    val_files = sorted(list(val_dir.glob("*.png")), key=lambda p: int(p.stem.split("_")[1]) if "_" in p.stem else 0)
    print(f"Loaded {len(val_files)} validation images from {val_dir}")
    assert len(val_files) == 2466, f"Expected 2,466 validation images, got {len(val_files)}"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. ResNet-18 CBAM v2
    resnet_ckpts = sorted([c for c in MODELS_DIR.glob("resnet18_cbam_v2_fold*.pt") if not c.name.startswith("._")])
    if not resnet_ckpts:
        raise FileNotFoundError(f"No ResNet-18 v2 checkpoints found in {MODELS_DIR}")
    p_resnet_v2 = evaluate_model_family(ResNet18_CBAM, resnet_ckpts, val_files, "ResNet-18 CBAM v2", device=device)
    np.save(str(SUBMISSIONS_DIR / "probabilities_resnet18_cbam_v2_5fold.npy"), p_resnet_v2)

    # 2. ConvNeXt-Femto v2
    convnext_ckpts = sorted([c for c in MODELS_DIR.glob("convnext_femto_v2_fold*.pt") if not c.name.startswith("._")])
    if convnext_ckpts:
        p_convnext_v2 = evaluate_model_family(ConvNeXt_Femto, convnext_ckpts, val_files, "ConvNeXt-Femto v2", device=device)
        np.save(str(SUBMISSIONS_DIR / "probabilities_convnext_femto_v2_5fold.npy"), p_convnext_v2)
    else:
        print("ConvNeXt-Femto v2 not found, using v1 fallback.")
        p_convnext_v2 = np.load(str(SUBMISSIONS_DIR / "probabilities_convnext_femto_5fold.npy"))

    # 3. DenseNet-121 CBAM v2
    densenet_ckpts = sorted([c for c in MODELS_DIR.glob("densenet121_cbam_v2_fold*.pt") if not c.name.startswith("._")])
    if densenet_ckpts:
        p_densenet_v2 = evaluate_model_family(DenseNet121_CBAM, densenet_ckpts, val_files, "DenseNet-121 CBAM v2", device=device)
        np.save(str(SUBMISSIONS_DIR / "probabilities_densenet121_cbam_v2_5fold.npy"), p_densenet_v2)
    else:
        print("DenseNet-121 CBAM v2 not found, using v1 fallback.")
        p_densenet_v2 = np.load(str(SUBMISSIONS_DIR / "probabilities_densenet121.npy"))

    print("\n=======================================================")
    print("🏆 Forming SOTA v2 Tri-Orthogonal Grandmaster Ensemble")
    print(f"Weights: ResNet={w_resnet:.2f}, ConvNeXt={w_convnext:.2f}, DenseNet={w_densenet:.2f}")
    print("=======================================================")

    p_tri_v2 = w_resnet * p_resnet_v2 + w_convnext * p_convnext_v2 + w_densenet * p_densenet_v2
    np.save(str(SUBMISSIONS_DIR / "probabilities_phase1_tri_orthogonal_v2.npy"), p_tri_v2)

    pred_v2_pure = np.argmax(p_tri_v2, axis=1)
    val_ids = [f.stem for f in val_files]

    # Generate Pure Visual Submission
    df_pure = pd.DataFrame({"id": val_ids, "label": pred_v2_pure})
    out_csv = SUBMISSIONS_DIR / "submission_phase1_tri_orthogonal_v2_pure.csv"
    out_zip = SUBMISSIONS_DIR / "submission_phase1_tri_orthogonal_v2_pure.zip"
    df_pure.to_csv(out_csv, index=False)

    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(out_csv, arcname="predictions.csv")

    print(f"\n✅ PURE VISUAL SOTA V2 SUBMISSION GENERATED:")
    print(f"   CSV: {out_csv}")
    print(f"   ZIP: {out_zip} ({out_zip.stat().st_size:,} bytes)")
    print("   Class Distribution:")
    print(df_pure["label"].value_counts().sort_index().to_dict())

    # Forensic Comparison with Sub 10 (SOTA Pure Visual: 0.9448)
    sub10_csv = SUBMISSIONS_DIR / "submission_phase1_tri_orthogonal_ensemble.csv"
    if not sub10_csv.exists():
        sub10_csv = SUBMISSIONS_DIR / "submission_phase1_tri_orthogonal_en.csv"

    if sub10_csv.exists():
        df_sub10 = pd.read_csv(sub10_csv)
        diff_mask = df_pure["label"] != df_sub10["label"]
        n_diff = np.sum(diff_mask)
        print(f"\n🔍 Forensic Comparison vs Sub 10 (Pure Visual SOTA @ 0.9448):")
        print(f"   Total predictions updated: {n_diff} / 2,466 samples")

        # Check the 3 critical confusion pairs
        # 1. 十 (10) vs 千 (12)
        c10_to_12 = np.sum((df_sub10["label"] == 12) & (df_pure["label"] == 10))
        c12_to_10 = np.sum((df_sub10["label"] == 10) & (df_pure["label"] == 12))
        print(f"   • 十 (10) <-> 千 (12) shifts: {c10_to_12} shifted to 十, {c12_to_10} shifted to 千")

        # 2. 五 (5) vs 六 (6)
        c6_to_5 = np.sum((df_sub10["label"] == 6) & (df_pure["label"] == 5))
        c5_to_6 = np.sum((df_sub10["label"] == 5) & (df_pure["label"] == 6))
        print(f"   • 五 (5) <-> 六 (6) shifts: {c6_to_5} shifted to 五, {c5_to_6} shifted to 六")

        # 3. 四 (4) vs 亿 (14)
        c14_to_4 = np.sum((df_sub10["label"] == 14) & (df_pure["label"] == 4))
        c4_to_14 = np.sum((df_sub10["label"] == 4) & (df_pure["label"] == 14))
        print(f"   • 四 (4) <-> 亿 (14) shifts: {c14_to_4} shifted to 四, {c4_to_14} shifted to 亿")

    return df_pure

if __name__ == "__main__":
    run_v2_evaluation_and_blend()
