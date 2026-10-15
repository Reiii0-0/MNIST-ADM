#!/usr/bin/env python3
"""
Generate High-Resolution Visual Comparison Poster & Interactive HTML Studio (v2)
Features:
- True Scale-Aligned & Centered Kaggle Images (Tight-Cropped to 48px in 64x64 Canvas)
- Morphological Stroke Overlays (Green = Val, Magenta = Kaggle)
- Interactive Live Dissolve Cross-Fade Sliders (Smooth Morphing in Same Coordinate Space)
- Exact Match vs Lookalike vs Discrepancy Auditing (Cosine Sim, Stroke IoU, Top-2 Gap)
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
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Noto Serif CJK SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PINYIN_MAP = {
    0: 'ling', 1: 'yi', 2: 'er', 3: 'san', 4: 'si',
    5: 'wu', 6: 'liu', 7: 'qi', 8: 'ba', 9: 'jiu',
    10: 'shi', 11: 'bai', 12: 'qian', 13: 'wan', 14: 'yi'
}

def main():
    print("=" * 80)
    print("GENERATING SCALE-ALIGNED VISUAL COMPARISON ASSETS & HTML STUDIO (v2)")
    print("=" * 80)

    report_path = PROJECT_ROOT / "outputs" / "kaggle_validation_alignment_report.csv"
    if not report_path.exists():
        print(f"[!] Error: {report_path} not found!")
        sys.exit(1)

    df_report = pd.read_csv(report_path)
    val_dir = PROJECT_ROOT / "data_validasi" / "test"
    out_img_dir = PROJECT_ROOT / "outputs" / "matched_kaggle"

    print(f"[+] Loaded report with {len(df_report):,} validation samples.")
    print(f"[+] Image assets directory: {out_img_dir}")

    # Step 1: Generate High-Resolution Static Comparison Grid
    print("\n[Step 1/2] Generating high-resolution static comparison poster (val_vs_kaggle_grid.png)...")

    # Select representative samples:
    # 1. Exact physical matches across all 15 characters (0 to 14)
    # 2. High-similarity samples (5 pairs)
    # 3. Notable discrepancy cases (6 pairs)
    selected_indices = []

    # One exact match per character (0 to 14)
    exact_df = df_report[df_report['match_category'] == 'EXACT_PHYSICAL_MATCH']
    for char_code in range(15):
        char_samples = exact_df[exact_df['anchor_pred'] == char_code]
        if not char_samples.empty:
            selected_indices.append(char_samples.sort_values(by='cosine_similarity', ascending=False).index[0])
        else:
            # Fallback to highest similarity agreed sample
            fb_samples = df_report[(df_report['anchor_pred'] == char_code) & (df_report['anchor_agrees_with_kaggle'] == True)]
            if not fb_samples.empty:
                selected_indices.append(fb_samples.sort_values(by='cosine_similarity', ascending=False).index[0])

    # 5 high-similarity agreed samples
    high_df = df_report[(df_report['match_category'] == 'HIGH_SIMILARITY') & (df_report['anchor_agrees_with_kaggle'] == True)]
    for idx in high_df.sort_values(by='cosine_similarity', ascending=False).index[:5]:
        if idx not in selected_indices:
            selected_indices.append(idx)

    # 6 notable discrepancy samples (Model != Kaggle Match)
    disc_df = df_report[df_report['anchor_agrees_with_kaggle'] == False].sort_values(by='cosine_similarity', ascending=False)
    for idx in disc_df.index[:6]:
        if idx not in selected_indices:
            selected_indices.append(idx)

    num_samples = len(selected_indices)
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, num_samples * 2.2), dpi=150)
    fig.patch.set_facecolor('#0f172a')  # Dark slate background

    col_titles = [
        "1. Validation Image\n(Dosen - Normalized Canvas)",
        "2. Matched Kaggle Image\n(Preda - Scaled to Match)",
        "3. Registration Overlay\n(Green=Val, Magenta=Kaggle)",
        "4. Alignment & Physical Audit\n(Sim, IoU & Model Labels)"
    ]

    for col_idx, title in enumerate(col_titles):
        axes[0, col_idx].set_title(title, fontsize=12, fontweight='bold', color='#38bdf8', pad=12)

    for row_idx, sample_idx in enumerate(selected_indices):
        row = df_report.iloc[sample_idx]
        val_id = row['id']
        sim = row['cosine_similarity']
        iou = row['stroke_iou']
        cat = row['match_category']
        agreed = row['anchor_agrees_with_kaggle']
        anc_pred = row['anchor_pred']
        anc_char = row['anchor_char']
        k_label = row['kaggle_ground_truth_label']
        k_char = row['kaggle_char']
        w_id = row['kaggle_writer_id']
        s_id = row['kaggle_sample_id']
        gap = row['top2_gap']

        anc_pinyin = PINYIN_MAP.get(anc_pred, '')
        k_pinyin = PINYIN_MAP.get(k_label, '')

        # Read scale-aligned images
        val_norm_path = out_img_dir / f"{val_id}_val_norm.png"
        k_align_path = out_img_dir / f"{val_id}_kaggle_aligned.png"
        ov_path = out_img_dir / f"{val_id}_overlay_aligned.png"

        val_norm = cv2.imread(str(val_norm_path), cv2.IMREAD_GRAYSCALE)
        k_align = cv2.imread(str(k_align_path), cv2.IMREAD_GRAYSCALE)
        ov_img = cv2.imread(str(ov_path))
        ov_rgb = cv2.cvtColor(ov_img, cv2.COLOR_BGR2RGB)

        # Plot Val Image
        axes[row_idx, 0].imshow(val_norm, cmap='gray')
        axes[row_idx, 0].axis('off')
        axes[row_idx, 0].text(3, 10, f"{val_id}", color='#f8fafc', fontsize=9, fontweight='bold',
                              bbox=dict(boxstyle='round,pad=0.2', facecolor='#1e293b', alpha=0.8, edgecolor='none'))

        # Plot Kaggle Image (SAME SCALE!)
        axes[row_idx, 1].imshow(k_align, cmap='gray')
        axes[row_idx, 1].axis('off')
        axes[row_idx, 1].text(3, 10, f"W{w_id} S{s_id}", color='#f8fafc', fontsize=9, fontweight='bold',
                              bbox=dict(boxstyle='round,pad=0.2', facecolor='#1e293b', alpha=0.8, edgecolor='none'))

        # Plot Overlay
        axes[row_idx, 2].imshow(ov_rgb)
        axes[row_idx, 2].axis('off')

        # Info Box
        ax_info = axes[row_idx, 3]
        ax_info.axis('off')
        ax_info.set_xlim(0, 10)
        ax_info.set_ylim(0, 10)

        # Color coding
        status_color = '#10b981' if agreed else '#f43f5e'
        status_text = "AGREED MATCH [OK]" if agreed else "DISCREPANCY [X]"
        
        if cat == 'EXACT_PHYSICAL_MATCH':
            cat_color = '#10b981'
            cat_label = "EXACT PHYSICAL MATCH"
        elif cat == 'HIGH_SIMILARITY':
            cat_color = '#38bdf8'
            cat_label = "HIGH SIMILARITY"
        else:
            cat_color = '#f59e0b'
            cat_label = cat.replace('_', ' ')

        # Draw a subtle card background
        card = patches.FancyBboxPatch((0.2, 0.5), 9.6, 9.0, boxstyle="round,pad=0.3",
                                     linewidth=1, edgecolor=status_color, facecolor='#1e293b')
        ax_info.add_patch(card)

        ax_info.text(0.6, 7.8, f"{cat_label} (Sim: {sim:.4f} | IoU: {iou:.3f})", fontsize=9.5, fontweight='bold', color=cat_color)
        ax_info.text(0.6, 5.8, f"ResNet Pred : [{anc_pred}] '{anc_char}' ({anc_pinyin})", fontsize=10, fontweight='bold', color='#f8fafc')
        ax_info.text(0.6, 3.8, f"Kaggle Truth : [{k_label}] '{k_char}' ({k_pinyin}) [Top2 Gap: +{gap:.3f}]", fontsize=9.5, fontweight='bold', color='#f8fafc')
        ax_info.text(0.6, 1.8, f"Status: {status_text}", fontsize=9.5, fontweight='bold', color=status_color)

    plt.tight_layout()
    grid_out = PROJECT_ROOT / "outputs" / "val_vs_kaggle_grid.png"
    plt.savefig(str(grid_out), dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"[+] Saved high-resolution comparison grid: {grid_out}")

    # Step 2: Generate Interactive HTML Comparison Studio
    print("\n[Step 2/2] Generating interactive HTML Comparison Studio (val_vs_kaggle_viewer.html)...")
    
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
            "k_s": int(row['kaggle_sample_id']),
            "sim": float(row['cosine_similarity']),
            "iou": float(row['stroke_iou']),
            "gap": float(row['top2_gap']),
            "cat": row['match_category'],
            "agr": bool(row['anchor_agrees_with_kaggle'])
        })

    html_content = f"""<!DOCTYPE html>
<html lang="id">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Scale-Aligned Validation vs Kaggle Ground-Truth Studio (v2)</title>
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
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
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
      padding: 4px 10px;
      border-radius: 9999px;
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    .badge-success {{ background: rgba(16, 185, 129, 0.2); color: var(--success); border: 1px solid var(--success); }}
    .badge-danger {{ background: rgba(239, 68, 68, 0.2); color: var(--danger); border: 1px solid var(--danger); }}
    .badge-info {{ background: rgba(56, 189, 248, 0.2); color: var(--accent); border: 1px solid var(--accent); }}
    .badge-warning {{ background: rgba(245, 158, 11, 0.2); color: var(--warning); border: 1px solid var(--warning); }}

    .metrics-bar {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 16px;
      margin-top: 20px;
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
      grid-template-columns: repeat(auto-fill, minmax(430px, 1fr));
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
      width: 82px;
      height: 82px;
      image-rendering: pixelated;
      border-radius: 6px;
      background: #000;
      border: 1px solid #374151;
    }}
    .img-box span {{
      font-size: 10px;
      font-weight: 700;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.03em;
    }}

    .blend-container {{
      position: relative;
      width: 82px;
      height: 82px;
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
      width: 82px;
      margin-top: 2px;
    }}
    .mini-slider {{
      width: 100%;
      height: 4px;
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
        <h1>Canonical Scale-Aligned Validation vs Kaggle Ground-Truth Studio (v2)</h1>
        <p style="color: var(--text-muted); font-size: 14px; margin-top: 4px;">
          Tight-Cropped & Aspect-Preserved Character Normalization (48px Canvas Box) · Direct Stroke Alignment
        </p>
      </div>
      <div>
        <span class="badge badge-info">Canonical Alignment v2</span>
      </div>
    </div>

    <div class="metrics-bar">
      <div class="metric-card">
        <span class="metric-value">2,466</span>
        <span class="metric-label">Total Validation Samples</span>
      </div>
      <div class="metric-card">
        <span class="metric-value" style="color: var(--success);">1,428 (57.91%)</span>
        <span class="metric-label">Exact Physical Matches (Score &ge; 0.85, 97.97% Acc)</span>
      </div>
      <div class="metric-card">
        <span class="metric-value" style="color: var(--accent);">1,983 (80.41%)</span>
        <span class="metric-label">High-Similarity Matches (Score &ge; 0.78, 94.76% Acc)</span>
      </div>
      <div class="metric-card">
        <span class="metric-value" style="color: var(--warning);">356 (14.44%)</span>
        <span class="metric-label">Total Model Discrepancies</span>
      </div>
    </div>

    <div class="quick-inspections">
      <span style="font-size: 11px; color: var(--accent); font-weight: 800; letter-spacing: 0.05em;">QUICK CASE INSPECTIONS:</span>
      <button class="chip-btn" onclick="inspectSample('val_00001')">val_00001 (八 · Sim 0.860 · W85 Exact)</button>
      <button class="chip-btn" onclick="inspectSample('val_00002')">val_00002 (三 · Sim 0.884 · W88 Exact)</button>
      <button class="chip-btn" onclick="inspectSample('val_00003')">val_00003 (八 · Sim 0.902 · IoU 0.686 · W16 Exact)</button>
      <button class="chip-btn" onclick="inspectSample('val_00006')">val_00006 (一 · Sim 0.923 · W21 Exact)</button>
      <button class="chip-btn" onclick="inspectSample('val_00014')">val_00014 (十 · Sim 0.938 · W11 Exact)</button>
      <button class="chip-btn" onclick="inspectSample('val_00023')">val_00023 (三 vs 二 · Cursive Discrepancy)</button>
      <button class="chip-btn" onclick="inspectSample('val_01511')">val_01511 (千 vs 十 · Slant Discrepancy)</button>
    </div>
  </header>

  <div class="controls-panel">
    <div class="filter-group">
      <button class="btn-filter active" data-filter="all">All (2,466)</button>
      <button class="btn-filter" data-filter="exact">Exact Matches (1,428)</button>
      <button class="btn-filter" data-filter="high">High Sim (1,983)</button>
      <button class="btn-filter" data-filter="discrepancy">Discrepancies (356)</button>
    </div>

    <div class="filter-group">
      <select id="charSelect" class="char-select">
        <option value="all">All Characters (0-14)</option>
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

      <input type="text" id="searchInput" class="search-input" placeholder="Search ID (e.g. val_00001)...">
    </div>
  </div>

  <div id="gridContainer" class="grid-container">
    <!-- Rendered via JS -->
  </div>

  <div class="pagination">
    <button id="prevBtn" class="page-btn">← Previous</button>
    <span id="pageInfo" class="page-info">Page 1 of 62</span>
    <button id="nextBtn" class="page-btn">Next →</button>
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
        if (currentFilter === 'exact' && item.cat !== 'EXACT_PHYSICAL_MATCH') return false;
        if (currentFilter === 'high' && item.sim < 0.78) return false;
        if (currentFilter === 'discrepancy' && item.agr) return false;

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
          No matching records found for the selected filter.
        </div>`;
        document.getElementById('pageInfo').innerText = `Page 0 of 0 (0 items)`;
        document.getElementById('prevBtn').disabled = true;
        document.getElementById('nextBtn').disabled = true;
        return;
      }}

      pageItems.forEach(item => {{
        const card = document.createElement('div');
        card.className = 'card';

        const statusBadge = item.agr 
          ? `<span class="badge badge-success">AGREED ✓</span>` 
          : `<span class="badge badge-danger">DISCREPANCY ✗</span>`;

        let catBadge = '';
        if (item.cat === 'EXACT_PHYSICAL_MATCH') {{
          catBadge = `<span class="badge badge-success" title="IoU: ${{item.iou.toFixed(3)}}">EXACT MATCH (${{item.sim.toFixed(3)}})</span>`;
        }} else if (item.sim >= 0.78) {{
          catBadge = `<span class="badge badge-info">HIGH SIM (${{item.sim.toFixed(3)}})</span>`;
        }} else {{
          catBadge = `<span class="badge badge-warning">LOOKALIKE (${{item.sim.toFixed(3)}})</span>`;
        }}

        card.innerHTML = `
          <div class="card-header">
            <span class="card-title">${{item.id}}</span>
            <div style="display: flex; gap: 6px;">
              ${{catBadge}}
              ${{statusBadge}}
            </div>
          </div>

          <div class="image-row">
            <div class="img-box">
              <img src="outputs/matched_kaggle/${{item.id}}_val_norm.png" alt="Val Norm" title="Validation Crop (Dosen)">
              <span>Val (Norm)</span>
            </div>
            <div class="img-box">
              <img src="outputs/matched_kaggle/${{item.id}}_kaggle_aligned.png" alt="Kaggle Aligned" title="Matched Kaggle Crop (Same Scale!)">
              <span>Kaggle (Aligned)</span>
            </div>
            <div class="img-box">
              <img src="outputs/matched_kaggle/${{item.id}}_overlay_aligned.png" alt="Overlay" title="Registration Overlay (Green=Val, Magenta=Kaggle)">
              <span>Overlay (Diff)</span>
            </div>
            <div class="img-box">
              <div class="blend-container">
                <img src="outputs/matched_kaggle/${{item.id}}_val_norm.png" alt="Val Base">
                <img src="outputs/matched_kaggle/${{item.id}}_kaggle_aligned.png" id="blend-${{item.id}}" class="blend-layer" alt="Kaggle Layer">
              </div>
              <div class="slider-wrapper">
                <span style="font-size: 8px; color: var(--text-muted);">V</span>
                <input type="range" min="0" max="100" value="50" class="mini-slider" 
                       oninput="document.getElementById('blend-${{item.id}}').style.opacity = this.value / 100">
                <span style="font-size: 8px; color: var(--text-muted);">K</span>
              </div>
            </div>
          </div>

          <div class="card-details">
            <div class="detail-row">
              <span class="detail-label">Model Prediction</span>
              <span class="detail-val ${{item.agr ? 'agreed' : 'mismatch'}}">${{item.anc_c}} (Label ${{item.anc_p}})</span>
            </div>
            <div class="detail-row">
              <span class="detail-label">Kaggle Ground Truth</span>
              <span class="detail-val ${{item.agr ? 'agreed' : 'mismatch'}}">${{item.k_chr}} (Label ${{item.k_lbl}})</span>
            </div>
            <div class="detail-row">
              <span class="detail-label">Kaggle Source</span>
              <span class="detail-val" style="font-size: 11px;">Writer ${{item.k_w}} · Sample ${{item.k_s}}</span>
            </div>
            <div class="detail-row">
              <span class="detail-label">Metrics (IoU / Gap)</span>
              <span class="detail-val" style="font-size: 11px; color: var(--accent);">IoU: ${{item.iou.toFixed(3)}} | Gap: +${{item.gap.toFixed(3)}}</span>
            </div>
          </div>
        `;

        container.appendChild(card);
      }});

      document.getElementById('pageInfo').innerText = `Page ${{currentPage}} of ${{totalPages}} (${{filtered.length}} items)`;
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

    // Filter Buttons
    document.querySelectorAll('.btn-filter').forEach(btn => {{
      btn.addEventListener('click', () => {{
        document.querySelectorAll('.btn-filter').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        currentFilter = btn.dataset.filter;
        currentPage = 1;
        render();
      }});
    }});

    // Character Dropdown
    document.getElementById('charSelect').addEventListener('change', (e) => {{
      currentChar = e.target.value;
      currentPage = 1;
      render();
    }});

    // Search Input
    document.getElementById('searchInput').addEventListener('input', (e) => {{
      searchQuery = e.target.value.trim();
      currentPage = 1;
      render();
    }});

    // Pagination
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

    // Initial render
    render();
  </script>
</body>
</html>
"""

    viewer_path = PROJECT_ROOT / "val_vs_kaggle_viewer.html"
    with open(viewer_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"[+] Saved standalone interactive HTML viewer: {viewer_path}")
    print("=" * 80)
    print("SCALE-ALIGNED VISUAL COMPARISON ASSETS COMPLETE!")
    print("=" * 80)

if __name__ == "__main__":
    main()
