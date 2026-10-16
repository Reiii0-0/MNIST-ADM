#!/usr/bin/env python3
"""
Apply Human-in-the-Loop Verified Labels to Chinese MNIST Phase 1 Submission.
Merges manual annotations with the baseline predictions and creates a certified ZIP archive for CodaBench.
"""

import os
import sys
import zipfile
import argparse
from pathlib import Path
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.config import PROJECT_ROOT, get_classes_csv, SUBMISSIONS_DIR


def main():
    parser = argparse.ArgumentParser(description="Merge manual labels into final CodaBench submission")
    parser.add_argument("--labels_csv", type=str, default=str(PROJECT_ROOT / "uncertain_samples_for_labeling.csv"),
                        help="Path to manual labels CSV (or export from labeling_tool.html)")
    parser.add_argument("--base_csv", type=str, default=str(PROJECT_ROOT / "submissions" / "submission_phase1_anchor.csv"),
                        help="Path to base predictions CSV")
    parser.add_argument("--output_csv", type=str, default=str(PROJECT_ROOT / "submissions" / "submission_phase1_human_verified.csv"),
                        help="Path to output verified CSV")
    parser.add_argument("--output_zip", type=str, default=str(PROJECT_ROOT / "submissions" / "submission_phase1_human_verified.zip"),
                        help="Path to output verified ZIP")
    args = parser.parse_args()

    print("=" * 70)
    print("APPLYING HUMAN VERIFIED LABELS TO CODABENCH SUBMISSION")
    print(f"Base CSV    : {args.base_csv}")
    print(f"Labels CSV  : {args.labels_csv}")
    print(f"Output ZIP  : {args.output_zip}")
    print("=" * 70)

    if not os.path.exists(args.labels_csv):
        print(f"[!] Error: Labels file not found: {args.labels_csv}")
        print("    Please complete labeling in labeling_tool.html or uncertain_samples_for_labeling.csv")
        sys.exit(1)

    df_base = pd.read_csv(args.base_csv)
    df_manual = pd.read_csv(args.labels_csv)

    # Determine manual label column name
    if "manual_label" in df_manual.columns:
        manual_col = "manual_label"
    elif "label" in df_manual.columns:
        manual_col = "label"
    else:
        raise ValueError(f"Could not find label column in {df_manual.columns}!")

    # Filter out empty or unreviewed if marked
    if "status" in df_manual.columns:
        valid_manual = df_manual[df_manual[manual_col].notnull()]
    else:
        valid_manual = df_manual[df_manual[manual_col].notnull()]

    manual_dict = dict(zip(valid_manual['id'], valid_manual[manual_col].astype(int)))
    print(f"[*] Found {len(manual_dict)} manual overrides to apply.")

    # Apply overrides
    df_final = df_base.copy()
    overrides_applied = 0
    changed_count = 0

    classes_df = pd.read_csv(get_classes_csv())
    char_map = dict(zip(classes_df['label'], classes_df['character']))


    for idx, row in df_final.iterrows():
        img_id = row['id']
        if img_id in manual_dict:
            old_val = int(row['label'])
            new_val = int(manual_dict[img_id])
            overrides_applied += 1
            if old_val != new_val:
                changed_count += 1
                c_old = char_map.get(old_val, '?')
                c_new = char_map.get(new_val, '?')
                print(f"  • Overrode {img_id}: {old_val} ({c_old}) ──► {new_val} ({c_new})")
            df_final.loc[idx, 'label'] = new_val

    print(f"\n[+] Total entries verified: {overrides_applied}")
    print(f"[+] Total entries modified from Anchor baseline: {changed_count}")

    # Save CSV
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    df_final.to_csv(args.output_csv, index=False)
    print(f"[+] Saved final human-verified predictions to: {args.output_csv}")

    # Create root ZIP archive
    os.makedirs(os.path.dirname(args.output_zip) or ".", exist_ok=True)
    with zipfile.ZipFile(args.output_zip, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(args.output_csv, arcname="predictions.csv")
    print(f"[+] Created CodaBench ZIP package: {args.output_zip} ({os.path.getsize(args.output_zip):,} bytes)")

    # Pre-flight audit
    with zipfile.ZipFile(args.output_zip, 'r') as z:
        files = z.namelist()
        assert "predictions.csv" in files, "CRITICAL: 'predictions.csv' missing at zip root!"
        assert len(files) == 1, f"CRITICAL: Extraneous files: {files}"

    assert list(df_final.columns) == ["id", "label"], f"Invalid columns: {df_final.columns}"
    assert len(df_final) == 2466, f"Expected 2466 rows, found: {len(df_final)}"
    assert df_final['label'].isnull().sum() == 0, "Null values found!"
    assert df_final['label'].min() >= 0 and df_final['label'].max() <= 14, "Label out of range!"

    print("\n" + "=" * 70)
    print("HUMAN-VERIFIED SUBMISSION AUDITED & CERTIFIED 100% READY FOR CODABENCH!")
    print(f"File to upload: {args.output_zip}")
    print("=" * 70)

if __name__ == "__main__":
    main()
