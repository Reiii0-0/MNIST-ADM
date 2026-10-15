#!/usr/bin/env python3
"""
Generate Comprehensive Side-by-Side Visual Comparison & Interactive HTML Viewer (V2)
Features:
- Aspect-Ratio Preserved Tight Cropping & Centering (both images are rendered at the exact SAME scale!)
- Morphological Stroke Dilation matching on Kaggle raw strokes
- Optical Registration Overlay (Green = Val, Magenta = Kaggle)
- Interactive Cross-Fade Dissolve Slider for real-time stroke verification
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
    print("GENERATING VISUAL COMPARISON ASSETS & HTML STUDIO (V2)")
    print("=" * 80)

    report_path = PROJECT_ROOT / "outputs" / "kaggle_validation_alignment_report.csv"
    if not report_path.exists():
        print(f"[!] Error: {report_path} not found!")
        sys.exit(1)

    df_report = pd.read_csv(report_path)
    val_dir = PROJECT_ROOT / "data_validasi" / "test"
    out_img_dir = PROJECT_ROOT / "outputs" / "matched_kaggle"

    # Step 1: High-Resolution Static Comparison Poster
    print("\n[Step 1/2] Generating high-resolution static comparison poster (val_vs_kaggle_grid.png)...")
    
    # Select representative samples:
    # 1. High-confidence agreed matches across all 15 characters
    selected_indices = []
    high_df = df_report[(df_report['match_confidence'] == 'HIGH') & (df_report['anchor_agrees_with_kaggle'] == True)]
    for char_code in range(15):
        char_samples = high_df[high_df['anchor_pred'] == char_code]
        if not char_samples.empty:
            selected_indices.append(char_samples.sort_values(by='penalized_score', ascending=False).index[0])

    # 2. Medium confidence agreed samples
    med_df = df_report[(df_report['match_confidence'] == 'MEDIUM') & (df_report['anchor_agrees_with_kaggle'] == True)]
    for idx in med_df.sort_values(by='penalized_score', ascending=False).index[:5]:
        if idx not in selected_indices:
            selected_indices.append(idx)

    # 3. Discrepancy samples (Model != Kaggle Match)
    disc_df = df_report[df_report['anchor_agrees_with_kaggle'] == False].sort_values(by='penalized_score', ascending=False)
    for idx in disc_df.index[:6]:
        if idx not in selected_indices:
            selected_indices.append(idx)

    num_samples = len(selected_indices)
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, num_samples * 2.3), dpi=150)
    fig.patch.set_facecolor('#0f172a')  # Dark slate background

    col_titles = [
        "1. Validation Image\n(Lecturer / Panitia - Raw)",
        "2. Matched Kaggle (Preda)\n(Cropped & Scaled to Same Size!)",
        "3. Registration Overlay\n(Green=Val, Magenta=Kaggle)",
        "4. Alignment & Model Audit\n(Shape Score & Predictions)"
    ]

    for col_idx, title in enumerate(col_titles):
        axes[0, col_idx].set_title(title, fontsize=12, fontweight='bold', color='#38bdf8', pad=12)

    for row_idx, sample_idx in enumerate(selected_indices):
        row = df_report.iloc[sample_idx]
        val_id = row['id']
        score = row['penalized_score']
        raw_corr = row.get('raw_correlation', score)
        agreed = row['anchor_agrees_with_kaggle']
        anc_pred = row['anchor_pred']
        anc_char = row['anchor_char']
        k_label = row['kaggle_ground_truth_label']
        k_char = row['kaggle_char']
        w_id = row['kaggle_writer_id']
        s_id = row.get('kaggle_sample_id', '?')
        anc_pinyin = PINYIN_MAP.get(anc_pred, '')
        k_pinyin = PINYIN_MAP.get(k_label, '')

        # Read images
        val_img = cv2.imread(str(val_dir / f"{val_id}.png"), cv2.IMREAD_GRAYSCALE)
        k_norm_img = cv2.imread(str(out_img_dir / f"{val_id}_kaggle_norm.png"), cv2.IMREAD_GRAYSCALE)
        ov_img = cv2.imread(str(out_img_dir / f"{val_id}_overlay_norm.png"))
        ov_rgb = cv2.cvtColor(ov_img, cv2.COLOR_BGR2RGB)

        # 1. Plot Val Image (Raw)
        axes[row_idx, 0].imshow(val_img, cmap='gray')
        axes[row_idx, 0].axis('off')
        axes[row_idx, 0].text(3, 10, f"{val_id}", color='#f8fafc', fontsize=9, fontweight='bold',
                              bbox=dict(boxstyle='round,pad=0.2', facecolor='#1e293b', alpha=0.8, edgecolor='none'))

        # 2. Plot Kaggle Matched Image (Normalized to Same Scale!)
        axes[row_idx, 1].imshow(k_norm_img, cmap='gray')
        axes[row_idx, 1].axis('off')
        axes[row_idx, 1].text(3, 8, f"W{w_id}-S{s_id}", color='#f8fafc', fontsize=9, fontweight='bold',
                              bbox=dict(boxstyle='round,pad=0.2', facecolor='#1e293b', alpha=0.8, edgecolor='none'))

        # 3. Plot Normalized Overlay
        axes[row_idx, 2].imshow(ov_rgb)
        axes[row_idx, 2].axis('off')

        # 4. Info Card
        ax_info = axes[row_idx, 3]
        ax_info.axis('off')
        ax_info.set_xlim(0, 10)
        ax_info.set_ylim(0, 10)

        status_color = '#10b981' if agreed else '#f43f5e'
        status_text = "AGREED MATCH [OK]" if agreed else "DISCREPANCY / AMBIGUOUS [X]"
        conf_color = '#38bdf8' if score >= 0.70 else ('#fbbf24' if score >= 0.55 else '#94a3b8')

        card = patches.FancyBboxPatch((0.2, 0.5), 9.6, 9.0, boxstyle="round,pad=0.3",
                                     linewidth=1, edgecolor=status_color, facecolor='#1e293b')
        ax_info.add_patch(card)

        ax_info.text(0.6, 7.8, f"Shape Score: {score:.4f} (Raw Corr: {raw_corr:.4f})", fontsize=9.5, fontweight='bold', color=conf_color)
        ax_info.text(0.6, 5.8, f"ResNet Pred : [{anc_pred}] '{anc_char}' ({anc_pinyin})", fontsize=10, fontweight='bold', color='#f8fafc')
        ax_info.text(0.6, 3.8, f"Kaggle Truth : [{k_label}] '{k_char}' ({k_pinyin}) - W{w_id}", fontsize=10, fontweight='bold', color='#f8fafc')
        ax_info.text(0.6, 1.8, f"Status: {status_text}", fontsize=9.5, fontweight='bold', color=status_color)

    plt.tight_layout()
    grid_out = PROJECT_ROOT / "outputs" / "val_vs_kaggle_grid.png"
    plt.savefig(str(grid_out), dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"[+] Saved high-resolution comparison poster: {grid_out}")

    # Step 2: Generate Interactive HTML Studio (V2)
    print("\n[Step 2/2] Generating interactive HTML Studio (val_vs_kaggle_viewer.html)...")

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
            "k_s": int(row.get('kaggle_sample_id', 1)),
            "sim": float(row['penalized_score']),
            "raw_corr": float(row.get('raw_correlation', row['penalized_score'])),
            "conf": row['match_confidence'],
            "agr": bool(row['anchor_agrees_with_kaggle'])
        })

    html_content = f"""<!DOCTYPE html>
<html lang="id">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Validation Set vs Kaggle Chinese MNIST Inspector (V2 - Canonical Normalized)</title>
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
      width: 80px;
      height: 80px;
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
      text-align: center;
    }}

    .blend-container {{
      position: relative;
      width: 80px;
      height: 80px;
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
      width: 80px;
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
        <h1>Validation Set vs Kaggle Chinese MNIST Inspector (V2)</h1>
        <p style="color: var(--text-muted); font-size: 14px; margin-top: 4px;">
          Aspect-Preserved Canonical Normalization &amp; Centering (Both Images Aligned to the Same Scale!)
        </p>
      </div>
      <div>
        <span class="badge badge-info">Canonical Registration V2</span>
      </div>
    </div>

    <div class="metrics-bar">
      <div class="metric-card">
        <span class="metric-value">2,466</span>
        <span class="metric-label">Total Validation Samples</span>
      </div>
      <div class="metric-card">
        <span class="metric-value" style="color: var(--success);">1,474 (59.8%)</span>
        <span class="metric-label">High-Confidence Matches (&ge; 0.70)</span>
      </div>
      <div class="metric-card">
        <span class="metric-value" style="color: var(--accent);">92.47%</span>
        <span class="metric-label">Agreement on High-Confidence Matches</span>
      </div>
      <div class="metric-card">
        <span class="metric-value" style="color: #c084fc;">2,388 (96.8%)</span>
        <span class="metric-label">Solid Matches (&ge; 0.55)</span>
      </div>
    </div>

    <div class="quick-inspections">
      <span style="font-size: 11px; color: var(--accent); font-weight: 800; letter-spacing: 0.05em;">QUICK VERIFICATION CASES:</span>
      <button class="chip-btn" onclick="inspectSample('val_00001')">val_00001 (八 · 0.821 HIGH)</button>
      <button class="chip-btn" onclick="inspectSample('val_00003')">val_00003 (八 · 0.842 HIGH)</button>
      <button class="chip-btn" onclick="inspectSample('val_00006')">val_00006 (一 · 0.835 HIGH)</button>
      <button class="chip-btn" onclick="inspectSample('val_00014')">val_00014 (十 · 0.892 HIGH)</button>
      <button class="chip-btn" onclick="inspectSample('val_00023')">val_00023 (三 · Cursive)</button>
      <button class="chip-btn" onclick="inspectSample('val_01511')">val_01511 (千 vs 十)</button>
    </div>
  </header>

  <div class="controls-panel">
    <div class="filter-group">
      <button class="btn-filter active" data-filter="all">All (2,466)</button>
      <button class="btn-filter" data-filter="agreed">Agreed ({df_report['anchor_agrees_with_kaggle'].sum()})</button>
      <button class="btn-filter" data-filter="discrepancy">Discrepancies ({(~df_report['anchor_agrees_with_kaggle']).sum()})</button>
      <button class="btn-filter" data-filter="high">High Sim (&ge; 0.70)</button>
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
    <span id="pageInfo" class="page-info">Page 1 of 50</span>
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

        const confBadge = item.sim >= 0.70 
          ? `<span class="badge badge-info">${{item.sim.toFixed(4)}} (HIGH)</span>`
          : (item.sim >= 0.55 
            ? `<span class="badge badge-warning">${{item.sim.toFixed(4)}} (MED)</span>`
            : `<span class="badge" style="background:#374151; color:#9ca3af;">${{item.sim.toFixed(4)}} (LOW)</span>`);

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
              <img src="data_validasi/test/${{item.id}}.png" alt="${{item.id}}" title="Validation Image (Panitia)">
              <span>1. Val Raw</span>
            </div>
            <div class="img-box">
              <img src="outputs/matched_kaggle/${{item.id}}_kaggle_norm.png" alt="Kaggle Match" title="Gabriel Preda Kaggle (Cropped & Scaled to Same Size)">
              <span>2. Kaggle (Scaled)</span>
            </div>
            <div class="img-box">
              <img src="outputs/matched_kaggle/${{item.id}}_overlay_norm.png" alt="Overlay" title="Canonical Registration (Green=Val, Magenta=Kaggle)">
              <span>3. Overlay</span>
            </div>
            <div class="img-box">
              <div class="blend-container">
                <img src="outputs/matched_kaggle/${{item.id}}_val_norm.png" alt="Val Base">
                <img src="outputs/matched_kaggle/${{item.id}}_kaggle_norm.png" id="blend-${{item.id}}" class="blend-layer" alt="Kaggle Layer">
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
              <span class="detail-val" style="font-size: 11px;">Writer ${{item.k_w}}, Sample ${{item.k_s}}</span>
            </div>
            <div class="detail-row">
              <span class="detail-label">Raw Pixel Corr</span>
              <span class="detail-val" style="font-size: 12px; color: var(--accent);">${{item.raw_corr.toFixed(4)}}</span>
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

    viewer_path = PROJECT_ROOT / "val_vs_kaggle_viewer.html"
    with open(viewer_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"[+] Saved standalone interactive HTML viewer (V2): {viewer_path}")
    print("=" * 80)
    print("VISUAL COMPARISON GENERATION COMPLETE!")
    print("=" * 80)

if __name__ == "__main__":
    main()
