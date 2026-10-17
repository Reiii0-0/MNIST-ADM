#!/usr/bin/env python3
"""
Champion Submission Generator for CodaBench Chinese MNIST Challenge
Final Project — Analisis Data Multivariat (ADM), Teknik Informatika ITS

This script provides a unified interface to generate, verify, and package
CodaBench-compliant submission archives (.csv and .zip).
"""

import os
import sys
import zipfile
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.config import SUBMISSIONS_DIR, get_validation_dir


def verify_and_package_submission(df_preds, output_csv, output_zip):
    """
    Strict CodaBench Pre-Flight Verification & Packaging:
    1. Schema check: ['id', 'label']
    2. Null value check: 0 NaNs
    3. Row count check: exactly 2,466 rows
    4. Label validity check: integers in [0, 14]
    5. Flat ZIP root check: 'predictions.csv' at zip root without nested folders
    """
    print("\n" + "=" * 70)
    print("RUNNING STRICT PRE-FLIGHT VERIFICATION FOR CODABENCH")
    print("=" * 70)

    # 1. Schema
    assert list(df_preds.columns) == ["id", "label"], f"Invalid columns! Found: {df_preds.columns}"
    print("[PASS] Schema matches specification: ['id', 'label']")

    # 2. Nulls
    assert df_preds["label"].isnull().sum() == 0, "CRITICAL: Found null values in predictions!"
    print("[PASS] Zero null / NaN values found.")

    # 3. Label dtype & range
    assert df_preds["label"].dtype in [int, np.int64, np.int32], "CRITICAL: Label is not integer!"
    assert df_preds["label"].min() >= 0 and df_preds["label"].max() <= 14, "CRITICAL: Label out of [0, 14]!"
    print(f"[PASS] Integer data type verified. Range: [{df_preds['label'].min()}, {df_preds['label'].max()}].")

    # 4. Count
    assert len(df_preds) == 2466, f"Count mismatch! Expected 2,466, got {len(df_preds)}"
    print(f"[PASS] Row count matches exact Phase 1 test set: {len(df_preds)} rows.")

    # 5. Profiling
    print("\n[PROFILING] Predicted Class Distribution:")
    counts = df_preds["label"].value_counts().sort_index()
    for cls in range(15):
        cnt = counts.get(cls, 0)
        pct = (cnt / len(df_preds)) * 100
        print(f"  • Class {cls:2d}: {cnt:4d} samples ({pct:5.2f}%)")

    # Write CSV
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    df_preds.to_csv(output_csv, index=False)
    print(f"\n[+] Saved CSV: {output_csv} ({os.path.getsize(output_csv):,} bytes)")

    # Write ZIP
    os.makedirs(os.path.dirname(output_zip) or ".", exist_ok=True)
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(output_csv, arcname="predictions.csv")

    # Verify ZIP
    with zipfile.ZipFile(output_zip, "r") as z:
        namelist = z.namelist()
        assert "predictions.csv" in namelist, "CRITICAL: 'predictions.csv' missing from ZIP root!"
        assert len(namelist) == 1, f"CRITICAL: Extraneous files in ZIP: {namelist}"
    print(f"[PASS] Flat root ZIP archive certified: {output_zip} ({os.path.getsize(output_zip):,} bytes)")
    print("=" * 70)
    print("SUBMISSION IS 100% CERTIFIED AND READY FOR CODABENCH UPLOAD!")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Generate Champion CodaBench Submission")
    parser.add_argument(
        "--strategy",
        type=str,
        default="blend",
        choices=["blend", "moe", "human_verified", "anchor", "adabn", "hccr_senet", "quad_ensemble"],
        help="Submission recipe to build (default: blend, which holds Rank 1 score 0.9319)",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=str(PROJECT_ROOT / "predictions.csv"),
        help="Path for output predictions.csv",
    )
    parser.add_argument(
        "--output_zip",
        type=str,
        default=str(PROJECT_ROOT / "predictions.zip"),
        help="Path for output predictions.zip",
    )
    args = parser.parse_args()

    strategy_files = {
        "blend": SUBMISSIONS_DIR / "submission_phase1_blend_resnet_convnext.csv",
        "moe": SUBMISSIONS_DIR / "submission_phase1_gated_moe.csv",
        "human_verified": SUBMISSIONS_DIR / "submission_phase1_human_verified.csv",
        "anchor": SUBMISSIONS_DIR / "submission_phase1_anchor.csv",
        "adabn": SUBMISSIONS_DIR / "submission_phase1_adabn_tta.csv",
        "hccr_senet": SUBMISSIONS_DIR / "submission_phase1_hccr_senet.csv",
        "quad_ensemble": SUBMISSIONS_DIR / "submission_phase1_quad_ensemble.csv",
    }

    src_csv = strategy_files[args.strategy]
    if not src_csv.exists():
        raise FileNotFoundError(f"Source predictions not found at {src_csv}!")

    print(f"[*] Compiling CodaBench Submission Package using strategy: '{args.strategy}'")
    print(f"[*] Source predictions: {src_csv}")
    df_preds = pd.read_csv(src_csv)

    verify_and_package_submission(df_preds, args.output_csv, args.output_zip)



if __name__ == "__main__":
    main()
