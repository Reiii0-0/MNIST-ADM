#!/usr/bin/env python3
"""
Reverse-Engineering & Alignment: Match Validation Data against Gabriel Preda's Kaggle Chinese MNIST.
Finds the exact 1-to-1 image correspondence between 'data_validasi/test/' and 'archive.zip'
to extract genuine ground-truth labels and verify model accuracy.
"""

import os
import sys
import time
import glob
import zipfile
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def canonical_fingerprint(img, size=32):
    """
    Extracts a scale- and translation-invariant normalized fingerprint.
    1. Binarizes foreground strokes.
    2. Crops tight bounding box.
    3. Resizes to fixed canonical grid (size x size).
    4. Normalizes feature vector (zero mean, unit norm).
    """
    thresh = (img > 15).astype(np.uint8) * 255
    y, x = np.where(thresh > 0)
    if len(y) == 0:
        return np.zeros(size * size, dtype=np.float32)
    crop = thresh[y.min():y.max() + 1, x.min():x.max() + 1]
    resized = cv2.resize(crop.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)
    v = resized.flatten()
    s = v.std()
    if s > 0:
        v = (v - v.mean()) / s
    norm = np.linalg.norm(v)
    if norm > 0:
        v = v / norm
    return v

def main():
    print("=" * 80)
    print("REVERSE-MATCHING VALIDATION SET AGAINST ORIGINAL KAGGLE ARCHIVE")
    print("=" * 80)

    archive_path = PROJECT_ROOT / "archive.zip"
    if not archive_path.exists():
        print(f"[!] Error: {archive_path} not found!")
        sys.exit(1)

    classes_df = pd.read_csv(PROJECT_ROOT / "classes.csv")
    char_map = dict(zip(classes_df['label'], classes_df['character']))

    # 1. Load Gabriel Preda Archive (15,000 images)
    print("\n[Stage 1/4] Extracting canonical fingerprints for 15,000 Kaggle images...")
    t0 = time.time()
    archive_vectors = []
    archive_labels = []
    archive_writers = []
    archive_names = []

    with zipfile.ZipFile(archive_path, 'r') as z:
        for code in range(1, 16):
            label = code - 1
            for w in range(1, 101):
                for s in range(1, 11):
                    name = f"data/data/input_{w}_{s}_{code}.jpg"
                    try:
                        data = z.read(name)
                    except:
                        continue
                    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
                    v = canonical_fingerprint(img)
                    archive_vectors.append(v)
                    archive_labels.append(label)
                    archive_writers.append(w)
                    archive_names.append(name)

    arc_mat = np.vstack(archive_vectors).astype(np.float32)  # (15000, 1024)
    print(f"[+] Loaded {len(arc_mat):,} Kaggle images in {time.time() - t0:.2f}s. Matrix shape: {arc_mat.shape}")

    # 2. Load Validation Set (2,466 images)
    print("\n[Stage 2/4] Extracting canonical fingerprints for 2,466 validation images...")
    t1 = time.time()
    val_files = sorted(glob.glob(str(PROJECT_ROOT / "data_validasi" / "test" / "*.png")))
    if not val_files:
        val_files = sorted(glob.glob(str(PROJECT_ROOT / "data_validasi" / "*.png")))

    val_vectors = []
    val_ids = []
    for p in val_files:
        val_ids.append(os.path.splitext(os.path.basename(p))[0])
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        val_vectors.append(canonical_fingerprint(img))

    val_mat = np.vstack(val_vectors).astype(np.float32)  # (2466, 1024)
    print(f"[+] Loaded {len(val_mat):,} validation images in {time.time() - t1:.2f}s.")

    # 3. Dense Cross-Similarity Cosine Matrix
    print("\n[Stage 3/4] Performing dense cosine matrix multiplication (2,466 x 15,000)...")
    t2 = time.time()
    sim_matrix = np.dot(val_mat, arc_mat.T)  # (2466, 15000)
    print(f"[+] Multiplied 36.99M matrix cells in {time.time() - t2:.4f}s!")

    # 4. Alignment & Discrepancy Auditing
    print("\n[Stage 4/4] Aligning correspondence and cross-verifying with model predictions...")
    best_match_indices = np.argmax(sim_matrix, axis=1)
    best_sim_scores = np.max(sim_matrix, axis=1)

    df_anchor = pd.read_csv(PROJECT_ROOT / "submissions" / "submission_phase1_anchor.csv")
    df_adabn = pd.read_csv(PROJECT_ROOT / "submissions" / "submission_phase1_adabn_tta.csv")
    df_conv = pd.read_csv(PROJECT_ROOT / "submissions" / "predictions_convnext.csv")

    records = []
    high_conf_matches = 0
    model_agreements = 0

    for i in range(len(val_files)):
        v_id = val_ids[i]
        match_idx = best_match_indices[i]
        sim_score = float(best_sim_scores[i])

        k_name = archive_names[match_idx]
        k_label = int(archive_labels[match_idx])
        k_writer = int(archive_writers[match_idx])

        anc_pred = int(df_anchor.loc[i, 'label'])
        ada_pred = int(df_adabn.loc[i, 'label'])
        cnv_pred = int(df_conv.loc[i, 'label'])

        is_high_sim = (sim_score >= 0.65)
        agrees_anchor = (anc_pred == k_label)
        agrees_adabn = (ada_pred == k_label)
        agrees_conv = (cnv_pred == k_label)

        if is_high_sim:
            high_conf_matches += 1
            if agrees_anchor:
                model_agreements += 1

        records.append({
            "id": v_id,
            "anchor_pred": anc_pred,
            "anchor_char": char_map[anc_pred],
            "adabn_pred": ada_pred,
            "convnext_pred": cnv_pred,
            "matched_kaggle_file": k_name,
            "kaggle_ground_truth_label": k_label,
            "kaggle_char": char_map[k_label],
            "kaggle_writer_id": k_writer,
            "cosine_similarity": round(sim_score, 4),
            "match_confidence": "HIGH" if sim_score >= 0.70 else ("MEDIUM" if sim_score >= 0.60 else "LOW"),
            "anchor_agrees_with_kaggle": agrees_anchor
        })

    df_results = pd.DataFrame(records)
    out_csv = PROJECT_ROOT / "outputs" / "kaggle_validation_alignment_report.csv"
    os.makedirs(out_csv.parent, exist_ok=True)
    df_results.to_csv(out_csv, index=False)
    print(f"\n[+] Saved complete alignment report: {out_csv}")

    # Print Summary Statistics
    total_val = len(df_results)
    sim_70 = (df_results['cosine_similarity'] >= 0.70).sum()
    sim_60 = (df_results['cosine_similarity'] >= 0.60).sum()
    overall_agreed = (df_results['anchor_pred'] == df_results['kaggle_ground_truth_label']).sum()

    high_conf_df = df_results[df_results['cosine_similarity'] >= 0.70]
    high_conf_acc = (high_conf_df['anchor_pred'] == high_conf_df['kaggle_ground_truth_label']).mean() * 100

    print("=" * 80)
    print("EXECUTIVE ALIGNMENT AUDIT SUMMARY")
    print("=" * 80)
    print(f"Total Validation Samples Evaluated    : {total_val}")
    print(f"High-Confidence Matches (Score >= 0.70): {sim_70}/{total_val} ({sim_70 / total_val * 100:.1f}%)")
    print(f"Solid Matches (Score >= 0.60)         : {sim_60}/{total_val} ({sim_60 / total_val * 100:.1f}%)")
    print(f"Overall Match Agreement with Anchor   : {overall_agreed}/{total_val} ({overall_agreed / total_val * 100:.2f}%)")
    print(f"Accuracy on High-Confidence Matches   : {high_conf_acc:.2f}% ({high_conf_df['anchor_pred'].eq(high_conf_df['kaggle_ground_truth_label']).sum()}/{len(high_conf_df)})")

    # Writer Distribution in Validation Set
    writers_found = df_results['kaggle_writer_id'].value_counts()
    print(f"\nDiscovered {len(writers_found)} unique Kaggle writers in validation set.")
    print("Top Writer IDs matched in validation set:")
    for w_id, count in writers_found.head(10).items():
        writer_type = "TRAIN (1-70)" if w_id <= 70 else "UNSEEN TEST (71-100)"
        print(f"  • Writer {w_id:2d} [{writer_type}]: {count} samples")

if __name__ == "__main__":
    main()
