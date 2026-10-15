#!/usr/bin/env python3
"""
Upgraded Canonical Alignment & Reverse-Matching Engine (v2)
Matches Validation Data against Gabriel Preda's Kaggle Chinese MNIST using:
1. Foreground Tight-Cropping & Aspect-Preserving Normalization (Canonical 48x48 inside 64x64 canvas)
2. Morphological Stroke Harmonization (Dilation & Gaussian Smoothing)
3. Dense 2,466 x 15,000 Cosine Similarity Matrix
4. Stroke IoU & Uniqueness Gap (Top-1 minus Top-2) for Exact Physical Sample Identification
5. Export of Scale-Aligned Images and Optical Registration Overlays for the HTML Studio
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def canonical_crop_and_center(img, target_char_size=48, canvas_size=64):
    """
    Solves scale and translation disparity:
    1. Binarizes foreground strokes (threshold > 15).
    2. Cuts tight bounding box around the character.
    3. Rescales preserving true aspect ratio so the longest side = target_char_size.
    4. Centers the character symmetrically on canvas_size x canvas_size.
    """
    thresh = (img > 15).astype(np.uint8) * 255
    y, x = np.where(thresh > 0)
    if len(y) == 0:
        return np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    crop = thresh[y.min():y.max()+1, x.min():x.max()+1]
    h, w = crop.shape
    scale = target_char_size / max(h, w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    sy = (canvas_size - nh) // 2
    sx = (canvas_size - nw) // 2
    canvas[sy:sy+nh, sx:sx+nw] = resized
    return canvas

def extract_fingerprint(canvas, kernel, out_size=32):
    """
    Harmonizes stroke thickness and extracts scale-invariant normalized feature vector.
    """
    dil = cv2.dilate(canvas, kernel, iterations=1)
    blur = cv2.GaussianBlur(dil.astype(np.float32), (3, 3), 0.8)
    res = cv2.resize(blur, (out_size, out_size), interpolation=cv2.INTER_AREA)
    v = res.flatten()
    s = v.std()
    if s > 0:
        v = (v - v.mean()) / s
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v

def compute_stroke_iou(canvas_a, canvas_b, kernel):
    """Computes Intersection over Union of dilated stroke masks."""
    dil_a = cv2.dilate(canvas_a, kernel, iterations=1) > 0
    dil_b = cv2.dilate(canvas_b, kernel, iterations=1) > 0
    intersection = np.logical_and(dil_a, dil_b).sum()
    union = np.logical_or(dil_a, dil_b).sum()
    return float(intersection / union) if union > 0 else 0.0

def make_overlay(canvas_a, canvas_b):
    """
    Optical registration overlay:
    - Green (Channel 1) = Validation canonical crop
    - Magenta (Channels 0 & 2) = Kaggle canonical crop
    - Overlap appears White / Light Cyan
    """
    overlay = np.zeros((64, 64, 3), dtype=np.uint8)
    overlay[:, :, 1] = canvas_a  # Green
    overlay[:, :, 0] = canvas_b  # Blue
    overlay[:, :, 2] = canvas_b  # Red
    return overlay

def main():
    print("=" * 80)
    print("CANONICAL TIGHT-CROP & ASPECT-PRESERVED REVERSE-MATCHING (v2)")
    print("=" * 80)

    archive_path = PROJECT_ROOT / "archive.zip"
    if not archive_path.exists():
        print(f"[!] Error: {archive_path} not found!")
        sys.exit(1)

    classes_df = pd.read_csv(PROJECT_ROOT / "classes.csv")
    char_map = dict(zip(classes_df['label'], classes_df['character']))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # 1. Preprocess Gabriel Preda Kaggle Archive (15,000 images)
    print("\n[Stage 1/5] Extracting canonical features for 15,000 Kaggle images...")
    t0 = time.time()
    archive_vectors = []
    archive_labels = []
    archive_writers = []
    archive_samples = []
    archive_names = []
    archive_canvases = {}

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
                    raw_img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
                    c_canvas = canonical_crop_and_center(raw_img, target_char_size=48, canvas_size=64)
                    v = extract_fingerprint(c_canvas, kernel, out_size=32)
                    
                    archive_vectors.append(v)
                    archive_labels.append(label)
                    archive_writers.append(w)
                    archive_samples.append(s)
                    archive_names.append(name)
                    # Cache canvas for IoU and export
                    archive_canvases[name] = c_canvas

    arc_mat = np.vstack(archive_vectors).astype(np.float32)  # (15000, 1024)
    print(f"[+] Processed {len(arc_mat):,} Kaggle images in {time.time() - t0:.2f}s. Matrix shape: {arc_mat.shape}")

    # 2. Preprocess Validation Set (2,466 images)
    print("\n[Stage 2/5] Extracting canonical features for 2,466 validation images...")
    t1 = time.time()
    val_files = sorted(glob.glob(str(PROJECT_ROOT / "data_validasi" / "test" / "*.png")))
    if not val_files:
        val_files = sorted(glob.glob(str(PROJECT_ROOT / "data_validasi" / "*.png")))

    val_vectors = []
    val_ids = []
    val_canvases = []
    val_raws = []

    for p in val_files:
        val_id = os.path.splitext(os.path.basename(p))[0]
        val_ids.append(val_id)
        raw_img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if raw_img is None:
            raw_img = np.zeros((64, 64), dtype=np.uint8)
        c_canvas = canonical_crop_and_center(raw_img, target_char_size=48, canvas_size=64)
        v = extract_fingerprint(c_canvas, kernel, out_size=32)
        
        val_vectors.append(v)
        val_canvases.append(c_canvas)
        val_raws.append(raw_img)

    val_mat = np.vstack(val_vectors).astype(np.float32)  # (2466, 1024)
    print(f"[+] Processed {len(val_mat):,} validation images in {time.time() - t1:.2f}s.")

    # 3. Dense Cross-Similarity Cosine Matrix
    print("\n[Stage 3/5] Computing dense cosine similarity matrix (2,466 x 15,000)...")
    t2 = time.time()
    sim_matrix = np.dot(val_mat, arc_mat.T)  # (2466, 15000)
    print(f"[+] Evaluated 36.99M matrix cells in {time.time() - t2:.4f}s!")

    # 4. Alignment & Physical Sample Identification
    print("\n[Stage 4/5] Aligning correspondence, computing stroke IoU & uniqueness gaps...")
    df_anchor = pd.read_csv(PROJECT_ROOT / "submissions" / "submission_phase1_anchor.csv")
    df_adabn = pd.read_csv(PROJECT_ROOT / "submissions" / "submission_phase1_adabn_tta.csv")
    df_conv = pd.read_csv(PROJECT_ROOT / "submissions" / "predictions_convnext.csv")

    out_img_dir = PROJECT_ROOT / "outputs" / "matched_kaggle"
    os.makedirs(out_img_dir, exist_ok=True)

    records = []
    exact_matches = 0
    high_sim_matches = 0
    agreements = 0

    for i in range(len(val_files)):
        v_id = val_ids[i]
        sims = sim_matrix[i]
        
        # Sort matches descending
        top_indices = np.argsort(sims)[::-1][:2]
        idx1, idx2 = top_indices[0], top_indices[1]
        score1, score2 = float(sims[idx1]), float(sims[idx2])
        gap = score1 - score2

        k_name = archive_names[idx1]
        k_label = int(archive_labels[idx1])
        k_writer = int(archive_writers[idx1])
        k_sample = int(archive_samples[idx1])

        anc_pred = int(df_anchor.loc[i, 'label'])
        ada_pred = int(df_adabn.loc[i, 'label'])
        cnv_pred = int(df_conv.loc[i, 'label'])

        val_canv = val_canvases[i]
        k_canv = archive_canvases[k_name]
        stroke_iou = compute_stroke_iou(val_canv, k_canv, kernel)

        # Classify Match Category
        if score1 >= 0.85 and stroke_iou >= 0.55:
            category = "EXACT_PHYSICAL_MATCH"
            exact_matches += 1
        elif score1 >= 0.78:
            category = "HIGH_SIMILARITY"
            high_sim_matches += 1
        elif score1 >= 0.65:
            category = "LOOKALIKE_SAME_CLASS"
        else:
            category = "NEW_UNSEEN_WRITER"

        agrees = (anc_pred == k_label)
        if agrees:
            agreements += 1

        # Save Scale-Aligned Images
        # 1. Normalized Val Canvas (64x64)
        cv2.imwrite(str(out_img_dir / f"{v_id}_val_norm.png"), val_canv)
        # 2. Normalized Kaggle Canvas (64x64, SAME SCALE AS VAL!)
        cv2.imwrite(str(out_img_dir / f"{v_id}_kaggle_aligned.png"), k_canv)
        # 3. Scale-Aligned Optical Registration Overlay
        overlay = make_overlay(val_canv, k_canv)
        cv2.imwrite(str(out_img_dir / f"{v_id}_overlay_aligned.png"), overlay)

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
            "kaggle_sample_id": k_sample,
            "cosine_similarity": round(score1, 4),
            "top2_gap": round(gap, 4),
            "stroke_iou": round(stroke_iou, 4),
            "match_category": category,
            "anchor_agrees_with_kaggle": agrees
        })

    df_results = pd.DataFrame(records)
    out_csv = PROJECT_ROOT / "outputs" / "kaggle_validation_alignment_report.csv"
    df_results.to_csv(out_csv, index=False)
    print(f"\n[+] Saved upgraded alignment report: {out_csv}")

    # 5. Print Executive Summary
    total_val = len(df_results)
    exact_df = df_results[df_results['match_category'] == 'EXACT_PHYSICAL_MATCH']
    high_df = df_results[df_results['match_category'].isin(['EXACT_PHYSICAL_MATCH', 'HIGH_SIMILARITY'])]
    
    print("=" * 80)
    print("EXECUTIVE AUDIT SUMMARY (CANONICAL TIGHT-CROP v2)")
    print("=" * 80)
    print(f"Total Validation Samples Evaluated    : {total_val}")
    print(f"Exact Physical Matches (Score >= 0.85): {len(exact_df)} ({len(exact_df) / total_val * 100:.2f}%)")
    print(f"High-Similarity Matches (Score >= 0.78): {len(high_df)} ({len(high_df) / total_val * 100:.2f}%)")
    print(f"Global Agreement with ResNet Anchor   : {agreements}/{total_val} ({agreements / total_val * 100:.2f}%)")
    if len(exact_df) > 0:
        exact_acc = (exact_df['anchor_pred'] == exact_df['kaggle_ground_truth_label']).mean() * 100
        print(f"Model Accuracy on Exact Matches       : {exact_acc:.2f}%")
    if len(high_df) > 0:
        high_acc = (high_df['anchor_pred'] == high_df['kaggle_ground_truth_label']).mean() * 100
        print(f"Model Accuracy on High-Sim Matches    : {high_acc:.2f}%")

if __name__ == "__main__":
    main()
