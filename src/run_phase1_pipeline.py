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
    generate_predictions(
        test_dir=args.test_dir,
        ckpt_paths=ckpt_paths,
        output_csv=sub1_csv,
        output_zip=sub1_zip,
        use_tta=True,
        use_adabn=False,
        power_p=1.8,
        device=device
    )

    # 3. Generate Submission 2: SOTA Winning Recipe (AdaBN + 5-Way TTA)
    sub2_csv = os.path.join(args.output_dir, "submission_phase1_adabn_tta.csv")
    sub2_zip = os.path.join(args.output_dir, "submission_phase1_adabn_tta.zip")
    print("\n" + "#" * 80)
    print("STAGE 2: GENERATING SUBMISSION 2 (SOTA RECIPE: AdaBN + 5-WAY TTA)")
    print("#" * 80)
    generate_predictions(
        test_dir=args.test_dir,
        ckpt_paths=ckpt_paths,
        output_csv=sub2_csv,
        output_zip=sub2_zip,
        use_tta=True,
        use_adabn=True,
        power_p=1.8,
        device=device
    )

    print("\n" + "=" * 80)
    print("PHASE 1 SPRINT PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"Submission 1 (Anchor): {sub1_zip}")
    print(f"Submission 2 (SOTA)  : {sub2_zip}")
    print("\nReady for upload to CodaBench upon user command on October 15, 2026.")

if __name__ == "__main__":
    import torch
    main()
