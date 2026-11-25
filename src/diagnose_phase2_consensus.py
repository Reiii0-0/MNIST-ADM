#!/usr/bin/env python3
"""
================================================================================
Phase 2 Consensus & Error Forensics Diagnostic (Zero Quota Cost)
================================================================================
Evaluates ALL diverse model families locally on the 2,902 unseen Phase 2 images:
1. Student SWA (1x pass, FP32-cast) -> matches Sub #1 (0.9862)
2. Student SWA + 5-Way Safe TTA
3. Student Epoch 13 + 5-Way Safe TTA
4. 5-Fold Adapted DenseNet-121 CBAM (incl. Traditional 萬/億 + NCCU, [0, 1]) + TTA
5. 5-Fold Adapted ConvNeXt-Femto (incl. Traditional 萬/億 + NCCU, [-1, 1]) + TTA
6. 5-Fold Adapted ResNet-18 CBAM (incl. Traditional 萬/億 + NCCU, [-1, 1]) + TTA

Outputs disagreement counts, class histograms, and confusion-pair candidates
WITHOUT modifying or burning CodaBench submission quota.
================================================================================
"""

import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.model_densenet_cbam import DenseNet121_CBAM
from src.models import ConvNeXt_Femto, ResNet18_CBAM, SubCenterArcFace
from src.predict_phase2_distillation import (
    CLASS_NAMES,
    UnlabeledImageDataset,
    load_student_model,
    resolve_test_inputs,
)
from src.evaluate_student_vs_ensemble import DEVICE


def get_5way_safe_tta(x: torch.Tensor) -> List[torch.Tensor]:
    """Conservative 5-way affine TTA (identity, ±3 deg, ±1.5 px) preserving stroke topology."""
    B, _, _, _ = x.shape
    dev = x.device
    variants = [x]
    for deg in (3.0, -3.0):
        rad = math.radians(deg)
        c, s = math.cos(rad), math.sin(rad)
        theta = torch.tensor([[c, -s, 0.0], [s, c, 0.0]], dtype=x.dtype, device=dev).unsqueeze(0).repeat(B, 1, 1)
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        variants.append(F.grid_sample(x, grid, align_corners=False, padding_mode='border'))
    for dx in (0.045, -0.045):
        theta = torch.tensor([[1.0, 0.0, dx], [0.0, 1.0, 0.0]], dtype=x.dtype, device=dev).unsqueeze(0).repeat(B, 1, 1)
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        variants.append(F.grid_sample(x, grid, align_corners=False, padding_mode='border'))
    return variants


def eval_models_with_tta(
    models_and_heads: List[Tuple[torch.nn.Module, torch.nn.Module]],
    loader: DataLoader,
    use_01_scale: bool = False,
    use_tta: bool = True,
) -> np.ndarray:
    """Computes ensemble-averaged softmax probabilities across folds and optional 5-way TTA."""
    fold_probs = []
    for model, head in models_and_heads:
        model.eval()
        head.eval()
        batches = []
        with torch.no_grad():
            for x in loader:
                x = x.to(DEVICE)
                if use_01_scale:
                    x = (x + 1.0) * 0.5  # [-1, 1] -> [0, 1]
                variants = get_5way_safe_tta(x) if use_tta else [x]
                var_probs = []
                for v in variants:
                    with autocast("cuda", enabled=torch.cuda.is_available()):
                        lin, feats = model(v)
                        arc = head(feats)
                    p = 0.5 * F.softmax(lin.float(), dim=1) + 0.5 * F.softmax(arc.float(), dim=1)
                    var_probs.append(p)
                batches.append(torch.stack(var_probs, dim=0).mean(dim=0).cpu().numpy())
        fold_probs.append(np.concatenate(batches, axis=0))
    return np.mean(fold_probs, axis=0)


def load_adapted_family(family: str) -> List[Tuple[torch.nn.Module, torch.nn.Module]]:
    """Loads 5-fold adapted_cursive models for dense/convnext/resnet."""
    models = []
    ckpt_dir = BASE_DIR / "models" / "adapted_cursive"
    for fold in range(5):
        if family == "densenet":
            ckpt_p = ckpt_dir / f"densenet121_cbam_adapted_fold{fold}.pt"
            m = DenseNet121_CBAM(in_channels=1, num_classes=15, feature_dim=256, use_mixstyle=True).to(DEVICE)
        elif family == "convnext":
            ckpt_p = ckpt_dir / f"convnext_femto_adapted_fold{fold}.pt"
            m = ConvNeXt_Femto(in_channels=1, num_classes=15, feature_dim=256).to(DEVICE)
        elif family == "resnet":
            ckpt_p = ckpt_dir / f"resnet18_cbam_adapted_fold{fold}.pt"
            m = ResNet18_CBAM(in_channels=1, num_classes=15, feature_dim=256, use_mixstyle=True).to(DEVICE)
        else:
            raise ValueError(family)

        h = SubCenterArcFace(in_features=256, num_classes=15, sub_centers=2).to(DEVICE)
        ck = torch.load(ckpt_p, map_location=DEVICE, weights_only=False)
        m.load_state_dict(ck["model"])
        h.load_state_dict(ck["head"])
        m.eval()
        h.eval()
        models.append((m, h))
    return models


def main():
    t_start = time.perf_counter()
    ids, img_paths, _ = resolve_test_inputs(BASE_DIR / "data" / "test_phase2", None, BASE_DIR / "data" / "_extracted_test")
    loader = DataLoader(UnlabeledImageDataset(img_paths), batch_size=128, shuffle=False, num_workers=2)

    sub1_df = pd.read_csv(BASE_DIR / "submissions" / "sub_phase2_student_swa_fp16_ttpa.csv")
    sub1_preds = sub1_df["label"].values

    print("\n[*] Loading Student models...")
    m_swa, h_swa, _ = load_student_model(BASE_DIR / "models" / "densenet121_cbam_student_fp16.pt", DEVICE, use_fp16=False)
    m_ep13, h_ep13, _ = load_student_model(BASE_DIR / "models" / "densenet121_cbam_student_best.pt", DEVICE, use_fp16=False)

    p_swa_1x = eval_models_with_tta([(m_swa, h_swa)], loader, use_01_scale=False, use_tta=False)
    p_swa_tta = eval_models_with_tta([(m_swa, h_swa)], loader, use_01_scale=False, use_tta=True)
    p_ep13_tta = eval_models_with_tta([(m_ep13, h_ep13)], loader, use_01_scale=False, use_tta=True)

    print("[*] Evaluating 5-Fold Adapted DenseNet-121 CBAM (TTA)...")
    p_ad_dense = eval_models_with_tta(load_adapted_family("densenet"), loader, use_01_scale=True, use_tta=True)
    print("[*] Evaluating 5-Fold Adapted ConvNeXt-Femto (TTA)...")
    p_ad_conv = eval_models_with_tta(load_adapted_family("convnext"), loader, use_01_scale=False, use_tta=True)
    print("[*] Evaluating 5-Fold Adapted ResNet-18 CBAM (TTA)...")
    p_ad_res = eval_models_with_tta(load_adapted_family("resnet"), loader, use_01_scale=False, use_tta=True)

    p_ad_tri = 0.40 * p_ad_dense + 0.30 * p_ad_conv + 0.30 * p_ad_res
    p_grand_blend = 0.45 * p_swa_tta + 0.15 * p_ep13_tta + 0.40 * p_ad_tri

    candidates: Dict[str, np.ndarray] = {
        "Sub1 (SWA 1x + TTPA)": sub1_preds,
        "Student SWA 1x (Raw)": np.argmax(p_swa_1x, axis=1),
        "Student SWA 5x-TTA": np.argmax(p_swa_tta, axis=1),
        "Student Ep13 5x-TTA": np.argmax(p_ep13_tta, axis=1),
        "Adapted DenseNet 5f TTA": np.argmax(p_ad_dense, axis=1),
        "Adapted ConvNeXt 5f TTA": np.argmax(p_ad_conv, axis=1),
        "Adapted ResNet 5f TTA": np.argmax(p_ad_res, axis=1),
        "Adapted Tri-Orthogonal 15f": np.argmax(p_ad_tri, axis=1),
        "Grand Blend (SWA+Ep13+Tri15)": np.argmax(p_grand_blend, axis=1),
    }

    print(f"\n[*] Completed all 17 models in {time.perf_counter() - t_start:.1f}s")
    print("=" * 86)
    print("📊 PHASE 2 MULTI-FAMILY AGREEMENT VS SUB #1 (Score: 0.9862 / ~40 errors)")
    print("=" * 86)
    for name, pr in candidates.items():
        diff = int(np.sum(pr != sub1_preds))
        c0 = int(np.sum(pr == 0))
        c14 = int(np.sum(pr == 14))
        c13 = int(np.sum(pr == 13))
        print(f"{name:<32s} | Diffs vs Sub1: {diff:4d} ({diff/len(sub1_preds)*100:5.2f}%) | c0={c0:3d} c13={c13:3d} c14={c14:3d}")

    # Analyze frequent disagreement pairs between Sub1 and Student SWA 5x-TTA / Tri15 / Grand Blend
    for cmp_name, pr in [
        ("Student SWA 5x-TTA", candidates["Student SWA 5x-TTA"]),
        ("Adapted Tri-Orthogonal 15f", candidates["Adapted Tri-Orthogonal 15f"]),
        ("Grand Blend (SWA+Ep13+Tri15)", candidates["Grand Blend (SWA+Ep13+Tri15)"]),
    ]:
        diff_idx = np.where(pr != sub1_preds)[0]
        pairs: Dict[Tuple[int, int], int] = {}
        for i in diff_idx:
            key = (int(sub1_preds[i]), int(pr[i]))
            pairs[key] = pairs.get(key, 0) + 1
        top_pairs = sorted(pairs.items(), key=lambda kv: -kv[1])[:8]
        pair_str = ", ".join([f"{CLASS_NAMES[k[0]]}->{CLASS_NAMES[k[1]]}: {v}" for k, v in top_pairs])
        print(f"\n🔍 Top shifts [{cmp_name} vs Sub1] ({len(diff_idx)} total):")
        print(f"   {pair_str}")

    out_npz = BASE_DIR / "submissions" / "phase2_multi_family_probs.npz"
    np.savez(
        out_npz,
        ids=np.array(ids),
        sub1_preds=sub1_preds,
        p_swa_1x=p_swa_1x,
        p_swa_tta=p_swa_tta,
        p_ep13_tta=p_ep13_tta,
        p_ad_dense=p_ad_dense,
        p_ad_conv=p_ad_conv,
        p_ad_res=p_ad_res,
        p_ad_tri=p_ad_tri,
        p_grand_blend=p_grand_blend,
    )
    print(f"\n✅ Saved multi-family probability tensors -> {out_npz}")


if __name__ == "__main__":
    main()
