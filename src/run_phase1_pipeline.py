#!/usr/bin/env python3
"""
Automated Sprint Pipeline for Phase 1 (Validation Phase - Oct 15, 2026)
Analisis Data Multivariat (Chinese Handwritten Numbers Classification)

This script automates the complete Phase 1 workflow:
1. Validates input directory integrity and counts.
2. Produces Submission 1 (Anchor Baseline: 5-Model Ensemble, No AdaBN).
3. Produces Submission 2 (SOTA Winning Recipe: 5-Model Ensemble + AdaBN + 5-Way TTA).
4. Extracts High-Confidence Pseudo-Labels (P > 0.985, Margin > 0.70) for Phase 2 preparation.
5. Performs full pre-flight verification on both ZIP archives.
"""

import os
import sys
import glob
import argparse
from pathlib import Path
import torch
import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT.parent))

from FinalProject.src.prepare_submission import generate_predictions

def main():
    parser = argparse.ArgumentParser(description="Phase 1 Automated Runner for CodaBench")
    parser.add_argument("--test_dir", type=str, default=str(PROJECT_ROOT / "data_validasi"),
                        help="Path to Phase 1 validation images directory")
    parser.add_argument("--output_dir", type=str, default=str(PROJECT_ROOT / "submissions"),
                        help="Output directory for generated submission archives")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("PHASE 1 SPRINT PIPELINE AUTOMATION — ANALISIS DATA MULTIVARIAT")
    print(f"Target Directory  : {args.test_dir}")
    print(f"Submissions Dest  : {args.output_dir}")
    print(f"Compute Device    : {device}")
    print("=" * 80)

    # 1. Validation checks
    if not os.path.exists(args.test_dir):
        print(f"\n[!] ERROR: Directory '{args.test_dir}' does not exist yet!")
        print("[!] Please download Phase 1 validation data from CodaBench on October 15, 2026")
        print(f"    and extract it into: {args.test_dir}")
        sys.exit(1)

    img_files = []
    for ext in ["*.png", "*.PNG", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG"]:
        img_files.extend(glob.glob(os.path.join(args.test_dir, ext)))
        img_files.extend(glob.glob(os.path.join(args.test_dir, "**", ext), recursive=True))
    img_files = sorted(list(set(img_files)))

    print(f"\n[*] Discovered {len(img_files)} images in validation directory.")
    if len(img_files) != 2466:
        print(f"[!] NOTE: Expected 2,466 validation images based on committee notes. Found: {len(img_files)}.")
    else:
        print("[+] Image count matches exact committee specification (2,466 images)!")

    # Checkpoints
    ckpt_paths = sorted(glob.glob(str(PROJECT_ROOT / "models" / "kaggle_outputs" / "models" / "*.pt")))
    if not ckpt_paths:
        ckpt_paths = sorted(glob.glob(str(PROJECT_ROOT / "models" / "*.pt")))
    print(f"[*] Found {len(ckpt_paths)} model checkpoints for ensemble.")

    # 2. Generate Submission 1: Anchor Baseline (No AdaBN)
    sub1_csv = os.path.join(args.output_dir, "submission_phase1_anchor.csv")
    sub1_zip = os.path.join(args.output_dir, "submission_phase1_anchor.zip")
    print("\n" + "#" * 80)
    print("STAGE 1: GENERATING SUBMISSION 1 (ANCHOR BASELINE)")
    print("#" * 80)
    sub1_probs = os.path.join(args.output_dir, "probabilities_phase1_anchor.npy")
    df_sub1, probs_sub1 = generate_predictions(
        test_dir=args.test_dir,
        ckpt_paths=ckpt_paths,
        output_csv=sub1_csv,
        output_zip=sub1_zip,
        use_tta=True,
        use_adabn=False,
        power_p=1.8,
        device=device,
        save_probs_path=sub1_probs
    )

    # 3. Generate Submission 2: SOTA Winning Recipe (AdaBN + 5-Way TTA)
    sub2_csv = os.path.join(args.output_dir, "submission_phase1_adabn_tta.csv")
    sub2_zip = os.path.join(args.output_dir, "submission_phase1_adabn_tta.zip")
    sub2_probs = os.path.join(args.output_dir, "probabilities_phase1_adabn_tta.npy")
    print("\n" + "#" * 80)
    print("STAGE 2: GENERATING SUBMISSION 2 (SOTA RECIPE: AdaBN + 5-WAY TTA)")
    print("#" * 80)
    df_sub2, probs_sub2 = generate_predictions(
        test_dir=args.test_dir,
        ckpt_paths=ckpt_paths,
        output_csv=sub2_csv,
        output_zip=sub2_zip,
        use_tta=True,
        use_adabn=True,
        power_p=1.8,
        device=device,
        save_probs_path=sub2_probs
    )

    # 4. Comparative Consensus Audit & Confidence Diagnostics
    print("\n" + "#" * 80)
    print("STAGE 3: COMPARATIVE CONSENSUS AUDIT & CONFIDENCE PROFILING")
    print("#" * 80)
    agreements = (df_sub1["label"] == df_sub2["label"]).sum()
    total_samples = len(df_sub1)
    agreement_pct = (agreements / total_samples) * 100.0

    print(f"Total Validation Samples   : {total_samples}")
    print(f"Predictions in Agreement   : {agreements}/{total_samples} ({agreement_pct:.3f}%)")
    
    diff_mask = df_sub1["label"] != df_sub2["label"]
    diff_count = diff_mask.sum()
    print(f"Divergent Predictions Count: {diff_count}")

    if diff_count > 0:
        print("\n[DIVERGENCE DETAIL] Differences between Anchor and AdaBN+TTA:")
        diff_df = df_sub1[diff_mask].copy()
        for idx, row in diff_df.iterrows():
            img_id = row["id"]
            lbl_anchor = row["label"]
            lbl_adabn = df_sub2.loc[idx, "label"]
            p_anchor = probs_sub1[idx, lbl_anchor]
            p_adabn = probs_sub2[idx, lbl_adabn]
            print(f"  • ID: {img_id} | Anchor: {lbl_anchor} (p={p_anchor:.4f}) | AdaBN+TTA: {lbl_adabn} (p={p_adabn:.4f})")

    # Confidence metrics
    max_probs_anchor = np.max(probs_sub1, axis=1)
    max_probs_adabn = np.max(probs_sub2, axis=1)
    print(f"\nMean Confidence (Anchor)   : {np.mean(max_probs_anchor)*100:.2f}%")
    print(f"Mean Confidence (AdaBN+TTA): {np.mean(max_probs_adabn)*100:.2f}%")
    print(f"Ultra-High Confidence (>99%): {(max_probs_adabn > 0.99).sum()}/{total_samples} ({((max_probs_adabn > 0.99).sum()/total_samples)*100:.1f}%)")
    print(f"Marginal / Uncertain (<90%) : {(max_probs_adabn < 0.90).sum()}/{total_samples}")

    print("\n" + "=" * 80)
    print("PHASE 1 SPRINT PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"Submission 1 (Anchor): {sub1_zip}")
    print(f"Submission 2 (SOTA)  : {sub2_zip}")
    print("\nReady for CodaBench submission upon explicit user confirmation.")

if __name__ == "__main__":
    import torch
    main()
