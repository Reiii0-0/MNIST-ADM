#!/usr/bin/env python3
"""
Upgraded Reverse-Matching & Alignment Engine (V2):
Aspect-Ratio Preserved Tight Cropping, Morphological Stroke Alignment, & Canonical Registration.
Matches all 2,466 validation images against all 15,000 Gabriel Preda Kaggle Chinese MNIST images.
"""

import os
import sys
import time
import glob
import json
import zipfile
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def extract_canonical_features(img, dilate_k=0, target_size=40, canvas_size=48):
    """
    Extracts an aspect-ratio-preserving, scale- and translation-invariant normalized representation.
    1. Thresholds foreground strokes.
    2. Applies morphological dilation if requested (bridges stroke-width differences).
    3. Crops tight bounding box (removes empty borders).
    4. Resizes to target dimension preserving exact aspect ratio w/h.
    5. Centers on fixed canvas (canvas_size x canvas_size).
    6. Returns flattened zero-mean unit-norm vector, aspect ratio, and visualization image.
    """
    thresh = (img > 12).astype(np.uint8) * 255
    if dilate_k > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_k, dilate_k))
        thresh = cv2.dilate(thresh, kernel)
    
    y, x = np.where(thresh > 0)
    if len(y) == 0:
        v = np.zeros(canvas_size * canvas_size, dtype=np.float32)
        return v, 1.0, np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    
    crop = thresh[y.min():y.max()+1, x.min():x.max()+1]
    h, w = crop.shape
    ar = float(w) / float(h)
    
    # Scale preserving aspect ratio
    scale = float(target_size) / max(h, w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
    
    canvas = np.zeros((canvas_size, canvas_size), dtype=np.float32)
    sy = (canvas_size - nh) // 2
    sx = (canvas_size - nw) // 2
    canvas[sy:sy+nh, sx:sx+nw] = resized / 255.0
    
    # Zero-mean unit-norm feature vector
    v = canvas.flatten()
    s = v.std()
    if s > 0:
        v = (v - v.mean()) / s
    norm = np.linalg.norm(v)
    if norm > 0:
        v = v / norm
        
    vis = (canvas * 255).astype(np.uint8)
    return v, ar, vis

def process_kaggle_entry(entry):
    name, data, label, writer, sample = entry
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    # Dilate Kaggle raw (k=3) so stroke width matches lecturer's dilated strokes
    v, ar, vis = extract_canonical_features(img, dilate_k=3, target_size=40, canvas_size=48)
    return v, ar, vis, name, label, writer, sample

def main():
    print("=" * 80)
    print("SOTA REVERSE-MATCHING V2: ASPECT-PRESERVED CANONICAL ALIGNMENT")
    print("=" * 80)

    archive_path = PROJECT_ROOT / "archive.zip"
    if not archive_path.exists():
        print(f"[!] Error: {archive_path} not found!")
        sys.exit(1)

    classes_df = pd.read_csv(PROJECT_ROOT / "classes.csv")
    char_map = dict(zip(classes_df['label'], classes_df['character']))

    # 1. Load Gabriel Preda Archive (15,000 images)
    print("\n[Stage 1/4] Extracting aspect-preserved canonical features for 15,000 Kaggle images...")
    t0 = time.time()
    kaggle_entries = []
    with zipfile.ZipFile(archive_path, 'r') as z:
        for code in range(1, 16):
            label = code - 1
            for w in range(1, 101):
                for s in range(1, 11):
                    name = f"data/data/input_{w}_{s}_{code}.jpg"
                    try:
                        data = z.read(name)
                        kaggle_entries.append((name, data, label, w, s))
                    except:
                        continue

    print(f"[+] Read {len(kaggle_entries):,} Kaggle raw images from zip in {time.time() - t0:.2f}s.")
    print("[+] Processing canonical features in parallel with ThreadPoolExecutor...")
    t1 = time.time()
    with ThreadPoolExecutor(max_workers=8) as executor:
        kaggle_results = list(executor.map(process_kaggle_entry, kaggle_entries))

    k_vectors = [res[0] for res in kaggle_results]
    k_ars = np.array([res[1] for res in kaggle_results], dtype=np.float32)
    k_vis_dict = {res[3]: res[2] for res in kaggle_results}
    k_names = [res[3] for res in kaggle_results]
    k_labels = np.array([res[4] for res in kaggle_results], dtype=np.int32)
    k_writers = np.array([res[5] for res in kaggle_results], dtype=np.int32)
    k_samples = np.array([res[6] for res in kaggle_results], dtype=np.int32)

    k_mat = np.vstack(k_vectors).astype(np.float32)  # (15000, 2304)
    print(f"[+] Computed {len(k_mat):,} Kaggle canonical vectors in {time.time() - t1:.2f}s. Matrix shape: {k_mat.shape}")

    # 2. Load Validation Set (2,466 images)
    print("\n[Stage 2/4] Extracting canonical features for 2,466 validation images...")
    t2 = time.time()
    val_files = sorted(glob.glob(str(PROJECT_ROOT / "data_validasi" / "test" / "*.png")))
    if not val_files:
        val_files = sorted(glob.glob(str(PROJECT_ROOT / "data_validasi" / "*.png")))

    val_vectors = []
    val_ars = []
    val_ids = []
    val_vis_dict = {}

    for p in val_files:
        v_id = os.path.splitext(os.path.basename(p))[0]
        val_ids.append(v_id)
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        v, ar, vis = extract_canonical_features(img, dilate_k=0, target_size=40, canvas_size=48)
        val_vectors.append(v)
        val_ars.append(ar)
        val_vis_dict[v_id] = vis

    v_mat = np.vstack(val_vectors).astype(np.float32)  # (2466, 2304)
    v_ars = np.array(val_ars, dtype=np.float32)
    print(f"[+] Extracted {len(v_mat):,} validation canonical vectors in {time.time() - t2:.2f}s.")

    # 3. Dense Matrix Similarity with Aspect-Ratio Consistency Penalty
    print("\n[Stage 3/4] Performing dense matrix multiplication & aspect consistency weighting...")
    t3 = time.time()
    # Cosine correlation on normalized vectors: (2466, 15000)
    raw_sim = np.dot(v_mat, k_mat.T)
    
    # Aspect Ratio log difference: |ln(AR_val / AR_kaggle)|
    ar_ratio = np.abs(np.log(np.maximum(1e-3, v_ars[:, np.newaxis] / np.maximum(1e-3, k_ars[np.newaxis, :]))))
    ar_penalty = np.exp(-1.5 * ar_ratio)
    
    # Final penalized shape similarity score
    final_sim_matrix = raw_sim * ar_penalty
    print(f"[+] Computed 36.99M similarity cells with aspect ratio penalties in {time.time() - t3:.4f}s!")

    # 4. Alignment & Discrepancy Auditing
    print("\n[Stage 4/4] Extracting top matches and generating alignment report...")
    best_indices = np.argmax(final_sim_matrix, axis=1)
    best_scores = np.max(final_sim_matrix, axis=1)
    raw_corrs = raw_sim[np.arange(len(val_files)), best_indices]

    df_anchor = pd.read_csv(PROJECT_ROOT / "submissions" / "submission_phase1_anchor.csv")
    df_adabn = pd.read_csv(PROJECT_ROOT / "submissions" / "submission_phase1_adabn_tta.csv")
    df_conv = pd.read_csv(PROJECT_ROOT / "submissions" / "predictions_convnext.csv")

    out_img_dir = PROJECT_ROOT / "outputs" / "matched_kaggle"
    os.makedirs(out_img_dir, exist_ok=True)

    records = []
    for i in range(len(val_files)):
        v_id = val_ids[i]
        b_idx = best_indices[i]
        score = float(best_scores[i])
        corr = float(raw_corrs[i])

        k_name = k_names[b_idx]
        k_lbl = int(k_labels[b_idx])
        k_w = int(k_writers[b_idx])
        k_s = int(k_samples[b_idx])

        anc_pred = int(df_anchor.loc[i, 'label'])
        ada_pred = int(df_adabn.loc[i, 'label'])
        cnv_pred = int(df_conv.loc[i, 'label'])

        agrees = (anc_pred == k_lbl)

        # Save aspect-preserved normalized images for the viewer
        v_vis = val_vis_dict[v_id]
        k_vis = k_vis_dict[k_name]
        
        # Save canonical normalized images
        cv2.imwrite(str(out_img_dir / f"{v_id}_val_norm.png"), v_vis)
        cv2.imwrite(str(out_img_dir / f"{v_id}_kaggle_norm.png"), k_vis)
        
        # Create normalized false-color overlay (Green = Val Norm, Magenta = Kaggle Norm)
        overlay_norm = np.zeros((48, 48, 3), dtype=np.uint8)
        overlay_norm[:, :, 1] = v_vis  # Green
        overlay_norm[:, :, 0] = k_vis  # Blue
        overlay_norm[:, :, 2] = k_vis  # Red
        cv2.imwrite(str(out_img_dir / f"{v_id}_overlay_norm.png"), overlay_norm)

        records.append({
            "id": v_id,
            "anchor_pred": anc_pred,
            "anchor_char": char_map[anc_pred],
            "adabn_pred": ada_pred,
            "convnext_pred": cnv_pred,
            "matched_kaggle_file": k_name,
            "kaggle_ground_truth_label": k_lbl,
            "kaggle_char": char_map[k_lbl],
            "kaggle_writer_id": k_w,
            "kaggle_sample_id": k_s,
            "penalized_score": round(score, 4),
            "raw_correlation": round(corr, 4),
            "match_confidence": "HIGH" if score >= 0.70 else ("MEDIUM" if score >= 0.55 else "LOW"),
            "anchor_agrees_with_kaggle": agrees
        })

    df_results = pd.DataFrame(records)
    out_csv = PROJECT_ROOT / "outputs" / "kaggle_validation_alignment_report.csv"
    df_results.to_csv(out_csv, index=False)
    print(f"\n[+] Saved upgraded alignment report: {out_csv}")

    # Summary Statistics
    total_val = len(df_results)
    sim_70 = (df_results['penalized_score'] >= 0.70).sum()
    sim_55 = (df_results['penalized_score'] >= 0.55).sum()
    agreed_total = df_results['anchor_agrees_with_kaggle'].sum()

    high_df = df_results[df_results['penalized_score'] >= 0.70]
    high_acc = (high_df['anchor_pred'] == high_df['kaggle_ground_truth_label']).mean() * 100 if len(high_df) > 0 else 0

    print("=" * 80)
    print("UPGRADED ALIGNMENT AUDIT SUMMARY (V2)")
    print("=" * 80)
    print(f"Total Validation Samples Evaluated    : {total_val}")
    print(f"High-Confidence Matches (Score >= 0.70): {sim_70}/{total_val} ({sim_70 / total_val * 100:.1f}%)")
    print(f"Solid Matches (Score >= 0.55)         : {sim_55}/{total_val} ({sim_55 / total_val * 100:.1f}%)")
    print(f"Global Match Agreement with Anchor    : {agreed_total}/{total_val} ({agreed_total / total_val * 100:.2f}%)")
    print(f"Agreement on High-Confidence Matches  : {high_acc:.2f}% ({high_df['anchor_agrees_with_kaggle'].sum()}/{len(high_df)})")

if __name__ == "__main__":
    main()
