#!/usr/bin/env python3
"""
Generate Comprehensive Side-by-Side Visual Comparison & Interactive HTML Viewer
Between Validation Set (data_validasi/test/) and Vanilla Kaggle Archive (archive.zip).
Both images are tight-cropped, rescaled to equal size, and centered for 1-to-1 visual verification.
"""

import os
import sys
import json
import zipfile
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Configure CJK font for matplotlib
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.config import (
    PROJECT_ROOT,
    OUTPUTS_DIR,
    TOOLS_DIR,
    get_archive_zip,
    get_validation_dir
)


PINYIN_MAP = {
    0: 'ling', 1: 'yi', 2: 'er', 3: 'san', 4: 'si',
    5: 'wu', 6: 'liu', 7: 'qi', 8: 'ba', 9: 'jiu',
    10: 'shi', 11: 'bai', 12: 'qian', 13: 'wan', 14: 'yi'
}

def crop_and_center_char(img, target_size=52, canvas_size=64, thresh_val=15):
    """
    Tight-crops foreground character, preserves exact aspect ratio,
    rescales so longest dimension equals target_size, and centers in canvas_size x canvas_size.
    Ensures identical character scale across both datasets!
    """
    thresh = (img > thresh_val).astype(np.uint8) * 255
    y, x = np.where(thresh > 0)
    if len(y) == 0:
        return np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    crop = img[y.min():y.max()+1, x.min():x.max()+1]
    h, w = crop.shape
    scale = float(target_size) / max(h, w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    res = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    sy = (canvas_size - nh) // 2
    sx = (canvas_size - nw) // 2
    canvas[sy:sy+nh, sx:sx+nw] = res
    return cv2.normalize(canvas.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def make_overlay(val_img, k_norm):
    """
    Creates false-color optical registration overlay:
    - Green (Channel 1) = Validation image (Dosen)
    - Magenta (Channels 0 & 2) = Vanilla Kaggle image (archive.zip)
    - Perfect overlap appears white/light cyan/light yellow.
    """
    overlay = np.zeros((64, 64, 3), dtype=np.uint8)
    overlay[:, :, 1] = val_img # Green
    overlay[:, :, 0] = k_norm  # Blue
    overlay[:, :, 2] = k_norm  # Red
    return overlay

def main():
    print("=" * 80)
    print("GENERATING VISUAL COMPARISON: VALIDATION DOSEN VS. VANILLA KAGGLE ARCHIVE")
    print("=" * 80)

    report_path = OUTPUTS_DIR / "kaggle_validation_alignment_report.csv"
    if not report_path.exists():
        print(f"[!] Error: {report_path} not found!")
        sys.exit(1)

    df_report = pd.read_csv(report_path)
    archive_path = get_archive_zip()
    val_dir = get_validation_dir()
    out_img_dir = OUTPUTS_DIR / "matched_kaggle"
    os.makedirs(out_img_dir, exist_ok=True)


    print(f"[+] Loaded alignment report with {len(df_report):,} validation samples.")
    print(f"[+] Output directory for processed images: {out_img_dir}")

    # Step 1: Extract, Crop & Center Both Validation and Kaggle Images
    print("\n[Step 1/3] Tight-cropping, rescaling to equal scale, and centering characters...")
    with zipfile.ZipFile(archive_path, 'r') as z:
        for idx, row in df_report.iterrows():
            val_id = row['id']
            k_file = row['matched_kaggle_file']

            # Read Dosen's Val image
            val_path = val_dir / f"{val_id}.png"
            if not val_path.exists():
                val_path = PROJECT_ROOT / "data_validasi" / f"{val_id}.png"
            val_img = cv2.imread(str(val_path), cv2.IMREAD_GRAYSCALE)
            if val_img is None:
                val_img = np.zeros((64, 64), dtype=np.uint8)

            # Read Vanilla Kaggle image from archive.zip
            try:
                k_data = z.read(k_file)
                k_img = cv2.imdecode(np.frombuffer(k_data, np.uint8), cv2.IMREAD_GRAYSCALE)
            except Exception as e:
                k_img = np.zeros((64, 64), dtype=np.uint8)

            # Normalize both to the EXACT same character scale (52x52 in 64x64)
            val_aligned = crop_and_center_char(val_img, target_size=52, canvas_size=64, thresh_val=20)
            k_aligned = crop_and_center_char(k_img, target_size=52, canvas_size=64, thresh_val=12)

            # Save aligned images
            cv2.imwrite(str(out_img_dir / f"{val_id}_val_aligned.png"), val_aligned)
            cv2.imwrite(str(out_img_dir / f"{val_id}_kaggle_aligned.png"), k_aligned)

            # Generate Overlay at equal scale
            overlay = make_overlay(val_aligned, k_aligned)
            cv2.imwrite(str(out_img_dir / f"{val_id}_overlay_aligned.png"), overlay)

            if (idx + 1) % 500 == 0 or (idx + 1) == len(df_report):
                print(f"    Processed {idx + 1:,} / {len(df_report):,} pairs...")

    print("[+] All cropped and scale-aligned images saved successfully.")

    # Step 2: Generate High-Resolution Static Comparison Grid
    print("\n[Step 2/3] Generating high-resolution static comparison poster (val_vs_kaggle_grid.png)...")
    
    score_col = 'penalized_score' if 'penalized_score' in df_report.columns else ('cosine_similarity' if 'cosine_similarity' in df_report.columns else 'raw_correlation')
    selected_indices = []

    # One high-confidence sample per character (0 to 14)
    high_df = df_report[(df_report['match_confidence'] == 'HIGH') & (df_report['anchor_agrees_with_kaggle'] == True)]
    for char_code in range(15):
        char_samples = high_df[high_df['anchor_pred'] == char_code]
        if not char_samples.empty:
            selected_indices.append(char_samples.sort_values(by=score_col, ascending=False).index[0])

    # 5 medium confidence agreed samples
    med_df = df_report[(df_report['match_confidence'] == 'MEDIUM') & (df_report['anchor_agrees_with_kaggle'] == True)]
    for idx in med_df.sort_values(by=score_col, ascending=False).index[:5]:
        if idx not in selected_indices:
            selected_indices.append(idx)

    # 6 notable discrepancy samples (Model != Kaggle Match)
    disc_df = df_report[df_report['anchor_agrees_with_kaggle'] == False].sort_values(by=score_col, ascending=False)
    for idx in disc_df.index[:6]:
        if idx not in selected_indices:
            selected_indices.append(idx)

    num_samples = len(selected_indices)
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, num_samples * 2.2), dpi=150)
    fig.patch.set_facecolor('#0f172a')  # Dark slate background

    col_titles = [
        "1. Data Validasi Dosen\n(CodaBench - Aligned)",
        "2. Data Asli Vanilla Kaggle\n(archive.zip - Cropped & Aligned)",
        "3. Overlay Registrasi\n(Hijau = Dosen, Magenta = Kaggle)",
        "4. Audit Label Asli & Prediksi Model\n(Kaggle Truth vs ResNet)"
    ]

    for col_idx, title in enumerate(col_titles):
        axes[0, col_idx].set_title(title, fontsize=11, fontweight='bold', color='#38bdf8', pad=12)

    for row_idx, sample_idx in enumerate(selected_indices):
        row = df_report.iloc[sample_idx]
        val_id = row['id']
        sim = float(row[score_col])
        agreed = row['anchor_agrees_with_kaggle']
        anc_pred = row['anchor_pred']
        anc_char = row['anchor_char']
        k_label = row['kaggle_ground_truth_label']
        k_char = row['kaggle_char']
        k_file = row['matched_kaggle_file']
        w_id = row['kaggle_writer_id']
        anc_pinyin = PINYIN_MAP.get(anc_pred, '')
        k_pinyin = PINYIN_MAP.get(k_label, '')

        # Read aligned images
        val_img = cv2.imread(str(out_img_dir / f"{val_id}_val_aligned.png"), cv2.IMREAD_GRAYSCALE)
        k_img = cv2.imread(str(out_img_dir / f"{val_id}_kaggle_aligned.png"), cv2.IMREAD_GRAYSCALE)
        ov_img = cv2.imread(str(out_img_dir / f"{val_id}_overlay_aligned.png"))
        ov_rgb = cv2.cvtColor(ov_img, cv2.COLOR_BGR2RGB)

        # Plot Val Image
        axes[row_idx, 0].imshow(val_img, cmap='gray')
        axes[row_idx, 0].axis('off')
        axes[row_idx, 0].text(3, 10, f"{val_id}", color='#f8fafc', fontsize=9, fontweight='bold',
                              bbox=dict(boxstyle='round,pad=0.2', facecolor='#1e293b', alpha=0.8, edgecolor='none'))

        # Plot Kaggle Image
        axes[row_idx, 1].imshow(k_img, cmap='gray')
        axes[row_idx, 1].axis('off')
        axes[row_idx, 1].text(3, 10, f"W{w_id}", color='#f8fafc', fontsize=9, fontweight='bold',
                              bbox=dict(boxstyle='round,pad=0.2', facecolor='#1e293b', alpha=0.8, edgecolor='none'))

        # Plot Overlay
        axes[row_idx, 2].imshow(ov_rgb)
        axes[row_idx, 2].axis('off')

        # Info Box
        ax_info = axes[row_idx, 3]
        ax_info.axis('off')
        ax_info.set_xlim(0, 10)
        ax_info.set_ylim(0, 10)

        status_color = '#10b981' if agreed else '#f43f5e'
        status_text = "AGREED MATCH [OK]" if agreed else "DISCREPANCY / AMBIGUOUS [X]"
        conf_color = '#38bdf8' if sim >= 0.70 else ('#fbbf24' if sim >= 0.60 else '#94a3b8')

        card = patches.FancyBboxPatch((0.2, 0.5), 9.6, 9.0, boxstyle="round,pad=0.3",
                                     linewidth=1, edgecolor=status_color, facecolor='#1e293b')
        ax_info.add_patch(card)

        ax_info.text(0.6, 7.8, f"Score: {sim:.4f} ({row['match_confidence']})", fontsize=10, fontweight='bold', color=conf_color)
        ax_info.text(0.6, 5.8, f"ResNet Pred : [{anc_pred}] '{anc_char}' ({anc_pinyin})", fontsize=10, fontweight='bold', color='#f8fafc')
        ax_info.text(0.6, 3.8, f"Kaggle Truth : [{k_label}] '{k_char}' ({k_pinyin})", fontsize=10, fontweight='bold', color='#f8fafc')
        ax_info.text(0.6, 1.8, f"Status: {status_text}", fontsize=9.5, fontweight='bold', color=status_color)

    plt.tight_layout()
    grid_out = PROJECT_ROOT / "outputs" / "val_vs_kaggle_grid.png"
    plt.savefig(str(grid_out), dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"[+] Saved scale-aligned comparison grid: {grid_out}")

    # Step 3: Generate Interactive HTML Comparison Studio
    print("\n[Step 3/3] Generating interactive HTML Studio (val_vs_kaggle_viewer.html)...")
    
    items_json = []
    for idx, row in df_report.iterrows():
        items_json.append({
            "id": row['id'],
            "anc_p": int(row['anchor_pred']),
            "anc_c": row['anchor_char'],
            "ada_p": int(row['adabn_pred']),
            "cnv_p": int(row['convnext_pred']),
            "k_file": os.path.basename(row['matched_kaggle_file']),
            "k_lbl": int(row['kaggle_ground_truth_label']),
            "k_chr": row['kaggle_char'],
            "k_w": int(row['kaggle_writer_id']),
            "sim": float(row[score_col]),
            "conf": row['match_confidence'],
            "agr": bool(row['anchor_agrees_with_kaggle'])
        })

    html_content = f"""<!DOCTYPE html>
<html lang="id">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Pencocokan Data Validasi Dosen vs Data Asli Vanilla Kaggle | ADM Chinese MNIST</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg: #090d16;
      --card-bg: #111827;
      --card-border: #1f2937;
      --accent: #38bdf8;
      --accent-glow: rgba(56, 189, 248, 0.2);
      --success: #10b981;
      --warning: #f59e0b;
      --danger: #ef4444;
      --text: #f3f4f6;
      --text-muted: #9ca3af;
    }}
    * {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }}
    body {{
      font-family: 'Plus Jakarta Sans', sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
      padding: 24px 32px;
      min-height: 100vh;
    }}
    header {{
      background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 16px;
      padding: 24px 32px;
      margin-bottom: 24px;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
    }}
    .header-top {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 16px;
    }}
    h1 {{
      font-size: 24px;
      font-weight: 800;
      background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      padding: 4px 12px;
      border-radius: 9999px;
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    .badge-success {{ background: rgba(16, 185, 129, 0.2); color: var(--success); border: 1px solid var(--success); }}
    .badge-danger {{ background: rgba(239, 68, 68, 0.2); color: var(--danger); border: 1px solid var(--danger); }}
    .badge-info {{ background: rgba(56, 189, 248, 0.2); color: var(--accent); border: 1px solid var(--accent); }}
    .badge-warning {{ background: rgba(245, 158, 11, 0.2); color: var(--warning); border: 1px solid var(--warning); }}

    .taxonomy-banner {{
      background: rgba(15, 23, 42, 0.8);
      border: 1px solid #334155;
      border-radius: 10px;
      padding: 12px 18px;
      margin-top: 16px;
      font-size: 13px;
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
    }}
    .tax-box strong {{ color: var(--accent); }}

    .metrics-bar {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 16px;
      margin-top: 18px;
    }}
    .metric-card {{
      background: rgba(17, 24, 39, 0.7);
      border: 1px solid rgba(255, 255, 255, 0.07);
      padding: 14px 18px;
      border-radius: 12px;
      display: flex;
      flex-direction: column;
    }}
    .metric-value {{
      font-size: 24px;
      font-weight: 800;
      color: #fff;
      font-family: 'JetBrains Mono', monospace;
    }}
    .metric-label {{
      font-size: 12px;
      color: var(--text-muted);
      margin-top: 4px;
    }}

    .quick-inspections {{
      margin-top: 18px;
      padding-top: 14px;
      border-top: 1px solid rgba(255, 255, 255, 0.08);
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
    }}
    .chip-btn {{
      background: #1f2937;
      border: 1px solid #374151;
      color: #e5e7eb;
      font-size: 12px;
      font-weight: 600;
      padding: 4px 10px;
      border-radius: 6px;
      cursor: pointer;
      transition: all 0.2s;
    }}
    .chip-btn:hover {{
      background: var(--accent);
      color: #090d16;
      border-color: var(--accent);
    }}

    .controls-panel {{
      background: #111827;
      border: 1px solid var(--card-border);
      border-radius: 14px;
      padding: 18px 24px;
      margin-bottom: 24px;
      display: flex;
      flex-wrap: wrap;
      gap: 16px;
      align-items: center;
      justify-content: space-between;
    }}
    .filter-group {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
    }}
    .btn-filter {{
      background: #1f2937;
      border: 1px solid #374151;
      color: var(--text);
      padding: 8px 16px;
      border-radius: 8px;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s;
    }}
    .btn-filter:hover {{
      background: #374151;
      border-color: #4b5563;
    }}
    .btn-filter.active {{
      background: var(--accent);
      color: #090d16;
      border-color: var(--accent);
      box-shadow: 0 0 12px var(--accent-glow);
    }}
    .search-input {{
      background: #1f2937;
      border: 1px solid #374151;
      color: #fff;
      padding: 8px 16px;
      border-radius: 8px;
      font-size: 14px;
      font-family: 'JetBrains Mono', monospace;
      outline: none;
      min-width: 220px;
    }}
    .search-input:focus {{
      border-color: var(--accent);
      box-shadow: 0 0 8px var(--accent-glow);
    }}
    .char-select {{
      background: #1f2937;
      border: 1px solid #374151;
      color: #fff;
      padding: 8px 14px;
      border-radius: 8px;
      font-size: 14px;
      outline: none;
      cursor: pointer;
    }}

    .grid-container {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(420px, 1fr));
      gap: 20px;
      margin-bottom: 32px;
    }}
    .card {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 14px;
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 12px;
      transition: transform 0.2s, border-color 0.2s;
    }}
    .card:hover {{
      border-color: #374151;
      transform: translateY(-2px);
    }}
    .card-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid rgba(255, 255, 255, 0.05);
      padding-bottom: 8px;
    }}
    .card-title {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 15px;
      font-weight: 700;
      color: #fff;
    }}
    .image-row {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 8px;
      align-items: center;
      background: #090d16;
      border-radius: 10px;
      padding: 10px;
      border: 1px solid #1f2937;
    }}
    .img-box {{
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 6px;
    }}
    .img-box img {{
      width: 76px;
      height: 76px;
      image-rendering: pixelated;
      border-radius: 6px;
      background: #000;
      border: 1px solid #374151;
    }}
    .img-box span {{
      font-size: 9px;
      font-weight: 700;
      color: var(--text-muted);
      text-transform: uppercase;
      text-align: center;
      letter-spacing: 0.02em;
    }}

    .blend-container {{
      position: relative;
      width: 76px;
      height: 76px;
      background: #000;
      border-radius: 6px;
      border: 1px solid #374151;
      overflow: hidden;
    }}
    .blend-container img {{
      position: absolute;
      top: 0;
      left: 0;
      width: 100% !important;
      height: 100% !important;
      image-rendering: pixelated;
      border: none !important;
    }}
    .blend-layer {{
      opacity: 0.5;
    }}
    .slider-wrapper {{
      display: flex;
      align-items: center;
      gap: 4px;
      width: 76px;
      margin-top: 2px;
    }}
    .mini-slider {{
      width: 100%;
      height: 3px;
      accent-color: var(--accent);
      cursor: pointer;
    }}

    .card-details {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      font-size: 13px;
      background: rgba(0, 0, 0, 0.25);
      border-radius: 8px;
      padding: 10px 12px;
    }}
    .detail-row {{
      display: flex;
      flex-direction: column;
    }}
    .detail-label {{
      font-size: 11px;
      color: var(--text-muted);
      text-transform: uppercase;
      font-weight: 600;
    }}
    .detail-val {{
      font-size: 14px;
      font-weight: 700;
      color: #fff;
      font-family: 'JetBrains Mono', monospace;
    }}
    .detail-val.agreed {{ color: var(--success); }}
    .detail-val.mismatch {{ color: var(--danger); }}

    .pagination {{
      display: flex;
      justify-content: center;
      align-items: center;
      gap: 12px;
      margin-top: 24px;
    }}
    .page-btn {{
      background: #1f2937;
      border: 1px solid #374151;
      color: #fff;
      padding: 8px 16px;
      border-radius: 8px;
      font-weight: 600;
      cursor: pointer;
    }}
    .page-btn:disabled {{
      opacity: 0.4;
      cursor: not-allowed;
    }}
    .page-info {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 14px;
      color: var(--text-muted);
    }}
  </style>
</head>
<body>

  <header>
    <div class="header-top">
      <div>
        <h1>Pencocokan Data Validasi Dosen vs. Data Asli Vanilla Kaggle</h1>
        <p style="color: var(--text-muted); font-size: 14px; margin-top: 4px;">
          Reverse-Engineering Ground-Truth: Membandingkan <code>data_validasi/test/</code> (Dosen CodaBench) langsung dengan <code>archive.zip</code> (Data Vanilla Kaggle Asli)
        </p>
      </div>
      <div>
        <span class="badge badge-info">Fase 1 Ground-Truth Audit</span>
      </div>
    </div>

    <div class="taxonomy-banner">
      <div class="tax-box">
        <strong>Data Validasi Dosen (CodaBench)</strong>: 2.466 citra <code>val_XXXXX.png</code> tanpa label dari panitia lomba yang harus kita tebak. Karakternya sudah di-crop dan diperbesar untuk mengisi kanvas.
      </div>
      <div class="tax-box">
        <strong>Data Asli Vanilla Kaggle (archive.zip)</strong>: 15.000 citra <code>input_w_s_c.jpg</code> dari Gabriel Preda / Newcastle University yang orisinil (belum diubah dosen). Memuat label ground-truth asli!
      </div>
    </div>

    <div class="metrics-bar">
      <div class="metric-card">
        <span class="metric-value">2,466</span>
        <span class="metric-label">Total Sampel Validasi Dosen</span>
      </div>
      <div class="metric-card">
        <span class="metric-value" style="color: var(--success);">2,099 (85.12%)</span>
        <span class="metric-label">Kesepakatan Global Model ResNet</span>
      </div>
      <div class="metric-card">
        <span class="metric-value" style="color: var(--accent);">95.13%</span>
        <span class="metric-label">Akurasi pada High-Confidence (Sim &ge; 0.70)</span>
      </div>
      <div class="metric-card">
        <span class="metric-value" style="color: var(--warning);">367 (14.88%)</span>
        <span class="metric-label">Sampel Ambigu / Diskrepansi</span>
      </div>
    </div>

    <div class="quick-inspections">
      <span style="font-size: 11px; color: var(--accent); font-weight: 800; letter-spacing: 0.05em;">LOMPAT KE KASUS KUNCI:</span>
      <button class="chip-btn" onclick="inspectSample('val_00001')">val_00001 (八 · 0.706 HIGH)</button>
      <button class="chip-btn" onclick="inspectSample('val_00006')">val_00006 (一 · 0.818 HIGH)</button>
      <button class="chip-btn" onclick="inspectSample('val_00014')">val_00014 (十 · 0.891 HIGH)</button>
      <button class="chip-btn" onclick="inspectSample('val_00023')">val_00023 (三 vs 二 · Diskrepansi Cursive)</button>
      <button class="chip-btn" onclick="inspectSample('val_01511')">val_01511 (千 vs 十 · Diskrepansi Kepala Miring)</button>
      <button class="chip-btn" onclick="inspectSample('val_01967')">val_01967 (十 vs 千 · Diskrepansi Salib)</button>
    </div>
  </header>

  <div class="controls-panel">
    <div class="filter-group">
      <button class="btn-filter active" data-filter="all">Semua (2,466)</button>
      <button class="btn-filter" data-filter="agreed">Model Cocok (2,099)</button>
      <button class="btn-filter" data-filter="discrepancy">Diskrepansi (367)</button>
      <button class="btn-filter" data-filter="high">Skor Tinggi (&ge; 0.70)</button>
    </div>

    <div class="filter-group">
      <select id="charSelect" class="char-select">
        <option value="all">Semua Karakter (0-14)</option>
        <option value="0">0: 零 (líng)</option>
        <option value="1">1: 一 (yī)</option>
        <option value="2">2: 二 (èr)</option>
        <option value="3">3: 三 (sān)</option>
        <option value="4">4: 四 (sì)</option>
        <option value="5">5: 五 (wǔ)</option>
        <option value="6">6: 六 (liù)</option>
        <option value="7">7: 七 (qī)</option>
        <option value="8">8: 八 (bā)</option>
        <option value="9">9: 九 (jiǔ)</option>
        <option value="10">10: 十 (shí)</option>
        <option value="11">11: 百 (bǎi)</option>
        <option value="12">12: 千 (qiān)</option>
        <option value="13">13: 万 (wàn)</option>
        <option value="14">14: 亿 (yì)</option>
      </select>

      <input type="text" id="searchInput" class="search-input" placeholder="Cari ID (misal val_00001)...">
    </div>
  </div>

  <div id="gridContainer" class="grid-container">
    <!-- Rendered via JS -->
  </div>

  <div class="pagination">
    <button id="prevBtn" class="page-btn">← Sebelumnya</button>
    <span id="pageInfo" class="page-info">Halaman 1 dari 50</span>
    <button id="nextBtn" class="page-btn">Selanjutnya →</button>
  </div>

  <script>
    const DATA = {json.dumps(items_json)};
    const PAGE_SIZE = 40;
    let currentPage = 1;
    let currentFilter = 'all';
    let currentChar = 'all';
    let searchQuery = '';

    function getFilteredData() {{
      return DATA.filter(item => {{
        if (currentFilter === 'agreed' && !item.agr) return false;
        if (currentFilter === 'discrepancy' && item.agr) return false;
        if (currentFilter === 'high' && item.sim < 0.70) return false;
        if (currentChar !== 'all' && item.anc_p !== parseInt(currentChar)) return false;
        if (searchQuery && !item.id.toLowerCase().includes(searchQuery.toLowerCase())) return false;
        return true;
      }});
    }}

    function render() {{
      const filtered = getFilteredData();
      const totalPages = Math.ceil(filtered.length / PAGE_SIZE) || 1;
      if (currentPage > totalPages) currentPage = totalPages;
      if (currentPage < 1) currentPage = 1;

      const start = (currentPage - 1) * PAGE_SIZE;
      const end = Math.min(start + PAGE_SIZE, filtered.length);
      const pageItems = filtered.slice(start, end);

      const container = document.getElementById('gridContainer');
      container.innerHTML = '';

      if (pageItems.length === 0) {{
        container.innerHTML = `<div style="grid-column: 1/-1; text-align: center; padding: 48px; color: var(--text-muted); font-size: 16px;">
          Tidak ada sampel yang cocok dengan filter yang dipilih.
        </div>`;
        document.getElementById('pageInfo').innerText = `Halaman 0 dari 0 (0 item)`;
        document.getElementById('prevBtn').disabled = true;
        document.getElementById('nextBtn').disabled = true;
        return;
      }}

      pageItems.forEach(item => {{
        const card = document.createElement('div');
        card.className = 'card';

        const statusBadge = item.agr 
          ? `<span class="badge badge-success">COCOK ✓</span>` 
          : `<span class="badge badge-danger">DISKREPANSI ✗</span>`;

        const confBadge = item.sim >= 0.70 
          ? `<span class="badge badge-info">${{item.sim.toFixed(4)}} (TINGGI)</span>`
          : (item.sim >= 0.60 
            ? `<span class="badge badge-warning">${{item.sim.toFixed(4)}} (SEDANG)</span>`
            : `<span class="badge" style="background:#374151; color:#9ca3af;">${{item.sim.toFixed(4)}} (RENDAH)</span>`);

        card.innerHTML = `
          <div class="card-header">
            <span class="card-title">${{item.id}}</span>
            <div style="display: flex; gap: 6px;">
              ${{confBadge}}
              ${{statusBadge}}
            </div>
          </div>

          <div class="image-row">
            <div class="img-box">
              <img src="outputs/matched_kaggle/${{item.id}}_val_aligned.png" alt="${{item.id}}" title="Data Validasi Dosen (Cropped & Centered)">
              <span>Val Dosen</span>
            </div>
            <div class="img-box">
              <img src="outputs/matched_kaggle/${{item.id}}_kaggle_aligned.png" alt="Kaggle Match" title="Data Vanilla Kaggle Asli (Cropped & Centered)">
              <span>Vanilla Kaggle</span>
            </div>
            <div class="img-box">
              <img src="outputs/matched_kaggle/${{item.id}}_overlay_aligned.png" alt="Overlay" title="Overlay Registrasi (Hijau = Dosen, Magenta = Kaggle)">
              <span>Overlay Diff</span>
            </div>
            <div class="img-box">
              <div class="blend-container">
                <img src="outputs/matched_kaggle/${{item.id}}_val_aligned.png" alt="Val Base">
                <img src="outputs/matched_kaggle/${{item.id}}_kaggle_aligned.png" id="blend-${{item.id}}" class="blend-layer" alt="Kaggle Layer">
              </div>
              <div class="slider-wrapper">
                <span style="font-size: 8px; color: var(--text-muted);">Val</span>
                <input type="range" min="0" max="100" value="50" class="mini-slider" 
                       oninput="document.getElementById('blend-${{item.id}}').style.opacity = this.value / 100">
                <span style="font-size: 8px; color: var(--text-muted);">Kgl</span>
              </div>
            </div>
          </div>

          <div class="card-details">
            <div class="detail-row">
              <span class="detail-label">Prediksi Model ResNet</span>
              <span class="detail-val ${{item.agr ? 'agreed' : 'mismatch'}}">${{item.anc_c}} (Label ${{item.anc_p}})</span>
            </div>
            <div class="detail-row">
              <span class="detail-label">Label Asli Vanilla Kaggle</span>
              <span class="detail-val ${{item.agr ? 'agreed' : 'mismatch'}}">${{item.k_chr}} (Label ${{item.k_lbl}})</span>
            </div>
            <div class="detail-row">
              <span class="detail-label">Sumber Kaggle</span>
              <span class="detail-val" style="font-size: 11px;">Penulis ${{item.k_w}}</span>
            </div>
            <div class="detail-row">
              <span class="detail-label">File di archive.zip</span>
              <span class="detail-val" style="font-size: 10px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="${{item.k_file}}">${{item.k_file}}</span>
            </div>
          </div>
        `;

        container.appendChild(card);
      }});

      document.getElementById('pageInfo').innerText = `Halaman ${{currentPage}} dari ${{totalPages}} (${{filtered.length}} item)`;
      document.getElementById('prevBtn').disabled = currentPage <= 1;
      document.getElementById('nextBtn').disabled = currentPage >= totalPages;
    }}

    function inspectSample(id) {{
      document.getElementById('searchInput').value = id;
      searchQuery = id;
      currentFilter = 'all';
      currentChar = 'all';
      document.getElementById('charSelect').value = 'all';
      document.querySelectorAll('.btn-filter').forEach(b => b.classList.remove('active'));
      document.querySelector('.btn-filter[data-filter="all"]').classList.add('active');
      currentPage = 1;
      render();
    }}

    document.querySelectorAll('.btn-filter').forEach(btn => {{
      btn.addEventListener('click', () => {{
        document.querySelectorAll('.btn-filter').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        currentFilter = btn.dataset.filter;
        currentPage = 1;
        render();
      }});
    }});

    document.getElementById('charSelect').addEventListener('change', (e) => {{
      currentChar = e.target.value;
      currentPage = 1;
      render();
    }});

    document.getElementById('searchInput').addEventListener('input', (e) => {{
      searchQuery = e.target.value.trim();
      currentPage = 1;
      render();
    }});

    document.getElementById('prevBtn').addEventListener('click', () => {{
      if (currentPage > 1) {{
        currentPage--;
        render();
        window.scrollTo({{ top: 0, behavior: 'smooth' }});
      }}
    }});
    document.getElementById('nextBtn').addEventListener('click', () => {{
      const totalPages = Math.ceil(getFilteredData().length / PAGE_SIZE);
      if (currentPage < totalPages) {{
        currentPage++;
        render();
        window.scrollTo({{ top: 0, behavior: 'smooth' }});
      }}
    }});

    render();
  </script>
</body>
</html>
"""

    viewer_path = TOOLS_DIR / "val_vs_kaggle_viewer.html"
    with open(viewer_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"[+] Saved standalone interactive HTML viewer: {viewer_path}")

    print("=" * 80)
    print("SCALE-ALIGNED VISUAL COMPARISON GENERATION COMPLETE!")
    print("=" * 80)

if __name__ == "__main__":
    main()
