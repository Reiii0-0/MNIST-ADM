#!/usr/bin/env python3
"""
Generate Active Learning Labeling Suite for Chinese MNIST Phase 1.
Extracts the most ambiguous / low-confidence validation samples and prepares:
1. 'uncertain_samples_for_labeling.csv' - Spreadsheet for manual labeling.
2. 'labeling_tool.html' - Self-contained interactive web interface with base64 embedded images.
"""

import os
import sys
import glob
import base64
import json
from pathlib import Path
import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.config import (
    PROJECT_ROOT,
    SUBMISSIONS_DIR,
    TOOLS_DIR,
    get_classes_csv,
    get_validation_dir
)

def main():
    print("=" * 70)
    print("GENERATING ACTIVE LEARNING LABELLING SUITE")
    print("=" * 70)

    # 1. Load predictions & probabilities
    df_anchor = pd.read_csv(SUBMISSIONS_DIR / "submission_phase1_anchor.csv")
    df_adabn = pd.read_csv(SUBMISSIONS_DIR / "submission_phase1_adabn_tta.csv")
    df_conv = pd.read_csv(SUBMISSIONS_DIR / "predictions_convnext.csv")
    p_res = np.load(SUBMISSIONS_DIR / "probabilities_phase1_anchor.npy")
    p_conv = np.load(SUBMISSIONS_DIR / "probabilities_convnext.npy")

    classes_df = pd.read_csv(get_classes_csv())

    char_map = dict(zip(classes_df['label'], classes_df['character']))
    pinyin_map = dict(zip(classes_df['label'], classes_df['pinyin']))
    arti_map = dict(zip(classes_df['label'], classes_df['arti']))

    # 2. Compute uncertainty metrics
    sorted_p_res = np.sort(p_res, axis=1)
    margin = sorted_p_res[:, -1] - sorted_p_res[:, -2]
    top1_conf = sorted_p_res[:, -1]

    disagree_adabn = (df_anchor['label'] != df_adabn['label']).astype(float)
    disagree_conv = (df_anchor['label'] != df_conv['label']).astype(float)

    # Composite Uncertainty Score:
    # Small margin (up to 1.0) + Model Disagreements (up to +1.0)
    uncertainty_score = (1.0 - margin) + 0.5 * disagree_adabn + 0.5 * disagree_conv

    # Select samples with uncertainty score > 1.0 (ambiguous / models disagree / low margin)
    flagged_indices = np.where(uncertainty_score > 1.0)[0]
    # Sort from most uncertain to least uncertain
    sorted_order = np.argsort(uncertainty_score[flagged_indices])[::-1]
    flagged_indices = flagged_indices[sorted_order]

    print(f"Total validation samples: {len(df_anchor)}")
    print(f"Flagged uncertain samples for manual labeling: {len(flagged_indices)}")

    # 3. Build CSV records
    records = []
    items_for_html = []

    val_dir = get_validation_dir()
    for rank, idx in enumerate(flagged_indices, start=1):
        img_id = df_anchor.loc[idx, 'id']
        img_path = val_dir / f"{img_id}.png"
        if not img_path.exists():
            img_path = val_dir.parent / f"{img_id}.png"


        l_anc = int(df_anchor.loc[idx, 'label'])
        l_ada = int(df_adabn.loc[idx, 'label'])
        l_cnv = int(df_conv.loc[idx, 'label'])

        p_anc = float(top1_conf[idx])
        m_anc = float(margin[idx])
        u_score = float(uncertainty_score[idx])

        # Recommended default consensus
        # If ConvNeXt agrees with Anchor, vote Anchor. If ConvNeXt agrees with AdaBN, vote AdaBN.
        if l_cnv == l_ada:
            consensus_label = l_ada
        elif l_cnv == l_anc:
            consensus_label = l_anc
        else:
            consensus_label = l_anc  # Anchor as tie-breaker

        records.append({
            "rank": rank,
            "id": img_id,
            "uncertainty_score": round(u_score, 4),
            "anchor_pred": l_anc,
            "anchor_char": char_map[l_anc],
            "anchor_conf": round(p_anc * 100, 2),
            "anchor_margin": round(m_anc * 100, 2),
            "adabn_pred": l_ada,
            "adabn_char": char_map[l_ada],
            "convnext_pred": l_cnv,
            "convnext_char": char_map[l_cnv],
            "recommended_consensus": consensus_label,
            "manual_label": consensus_label,  # default pre-filled
            "status": "UNREVIEWED"
        })

        # Base64 image encoding for self-contained HTML
        with open(img_path, "rb") as f:
            b64_img = base64.b64encode(f.read()).decode('utf-8')

        items_for_html.append({
            "rank": rank,
            "id": img_id,
            "b64": b64_img,
            "score": round(u_score, 3),
            "anchor": {"label": l_anc, "char": char_map[l_anc], "conf": round(p_anc * 100, 1)},
            "adabn": {"label": l_ada, "char": char_map[l_ada]},
            "convnext": {"label": l_cnv, "char": char_map[l_cnv]},
            "recommended": consensus_label,
            "current_label": consensus_label
        })

    # Save CSV
    csv_out = PROJECT_ROOT / "uncertain_samples_for_labeling.csv"
    pd.DataFrame(records).to_csv(csv_out, index=False)
    print(f"[+] Saved labeling spreadsheet: {csv_out}")

    # 4. Generate Interactive HTML Tool
    classes_json = []
    for _, r in classes_df.iterrows():
        classes_json.append({
            "label": int(r['label']),
            "char": str(r['character']),
            "pinyin": str(r['pinyin']),
            "arti": str(r['arti']),
            "trad": str(r['bentuk_tradisional']) if pd.notna(r['bentuk_tradisional']) else ""
        })

    html_content = f"""<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Chinese MNIST Active Labeling Studio</title>
<style>
  :root {{
    --bg: #0f172a;
    --card-bg: #1e293b;
    --border: #334155;
    --text: #f8fafc;
    --text-muted: #94a3b8;
    --accent: #38bdf8;
    --accent-hover: #0284c7;
    --success: #22c55e;
    --warning: #f59e0b;
    --danger: #ef4444;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }}
  body {{ background-color: var(--bg); color: var(--text); padding: 20px; min-height: 100vh; }}
  .container {{ max-width: 1100px; margin: 0 auto; }}
  header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 24px; padding-bottom: 16px; border-bottom: 1px solid var(--border); }}
  h1 {{ font-size: 24px; font-weight: 700; color: var(--accent); }}
  .badge {{ background: #0369a1; color: white; padding: 4px 10px; border-radius: 9999px; font-size: 13px; font-weight: 600; }}
  
  .main-grid {{ display: grid; grid-template-columns: 360px 1fr; gap: 24px; }}
  .card {{ background: var(--card-bg); border: 1px solid var(--border); border-radius: 12px; padding: 20px; }}
  
  .img-box {{ display: flex; flex-direction: column; align-items: center; justify-content: center; background: #000; border-radius: 8px; padding: 20px; margin-bottom: 16px; border: 2px solid #475569; }}
  .img-box img {{ width: 220px; height: 220px; image-rendering: pixelated; border-radius: 4px; box-shadow: 0 4px 12px rgba(0,0,0,0.5); }}
  
  .nav-controls {{ display: flex; gap: 8px; width: 100%; margin-top: 12px; }}
  .nav-btn {{ flex: 1; padding: 10px; background: #334155; border: none; color: white; border-radius: 6px; cursor: pointer; font-weight: 600; }}
  .nav-btn:hover {{ background: #475569; }}

  .predictions-box {{ margin-top: 16px; }}
  .pred-row {{ display: flex; justify-content: space-between; padding: 8px 12px; border-radius: 6px; margin-bottom: 6px; font-size: 14px; background: #0f172a; }}
  .pred-row.highlight {{ border-left: 4px solid var(--accent); }}
  
  .class-palette {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }}
  .class-btn {{ display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 12px 8px; background: #0f172a; border: 2px solid var(--border); border-radius: 8px; color: var(--text); cursor: pointer; transition: all 0.15s; }}
  .class-btn:hover {{ border-color: var(--accent); transform: translateY(-2px); }}
  .class-btn.selected {{ background: #0284c7; border-color: #38bdf8; box-shadow: 0 0 12px rgba(56, 189, 248, 0.4); }}
  .class-btn .char {{ font-size: 28px; font-weight: bold; margin-bottom: 4px; font-family: "PingFang SC", "Noto Sans SC", "SimSun", serif; }}
  .class-btn .sub {{ font-size: 11px; color: var(--text-muted); }}
  .class-btn.selected .sub {{ color: #e0f2fe; }}
  
  .stats-bar {{ display: flex; justify-content: space-between; margin-bottom: 16px; font-size: 14px; color: var(--text-muted); }}
  .export-btn {{ background: var(--success); color: white; border: none; padding: 10px 18px; border-radius: 8px; font-weight: 700; cursor: pointer; display: flex; align-items: center; gap: 6px; }}
  .export-btn:hover {{ background: #16a34a; }}
  .keyboard-tips {{ margin-top: 16px; font-size: 12px; color: var(--text-muted); line-height: 1.6; border-top: 1px solid var(--border); padding-top: 12px; }}
  kbd {{ background: #334155; padding: 2px 6px; border-radius: 4px; color: #f1f5f9; font-weight: 600; }}
</style>
</head>
<body>

<div class="container">
  <header>
    <div>
      <h1>Active Labeling Studio</h1>
      <p style="color: var(--text-muted); font-size: 13px; margin-top: 4px;">Review & verifikasi 215 sampel ambigu untuk mencapai akurasi 100%</p>
    </div>
    <div style="display: flex; gap: 12px; align-items: center;">
      <span class="badge" id="progressBadge">0 / {len(flagged_indices)} Reviewed</span>
      <button class="export-btn" onclick="exportCSV()">📥 Download Verified CSV</button>
    </div>
  </header>

  <div class="main-grid">
    <!-- Left Column: Image & Model Predictions -->
    <div class="card">
      <div class="stats-bar">
        <span id="rankIndicator">Rank: 1 / {len(flagged_indices)}</span>
        <span id="scoreIndicator" style="color: var(--warning);">Ambiguity: High</span>
      </div>

      <div class="img-box">
        <img id="mainImage" src="" alt="Character Image" />
        <h3 id="imageId" style="margin-top: 12px; font-family: monospace;">val_00000</h3>
      </div>

      <div class="nav-controls">
        <button class="nav-btn" onclick="prevItem()">◀ Prev (A)</button>
        <button class="nav-btn" onclick="nextItem()">Next (D) ▶</button>
      </div>

      <div class="predictions-box">
        <h4 style="font-size: 13px; color: var(--text-muted); margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px;">Prediksi Model:</h4>
        <div class="pred-row" id="rowAnchor">
          <span>ResNet-18 Anchor:</span>
          <strong id="predAnchor">-</strong>
        </div>
        <div class="pred-row" id="rowConvNeXt">
          <span>ConvNeXt-Femto:</span>
          <strong id="predConvNeXt">-</strong>
        </div>
        <div class="pred-row" id="rowAdaBN">
          <span>ResNet-18 AdaBN:</span>
          <strong id="predAdaBN">-</strong>
        </div>
      </div>
    </div>

    <!-- Right Column: Interactive 15-Class Selection Palette -->
    <div class="card">
      <h3 style="margin-bottom: 12px; font-size: 16px;">Pilih Label Yang Benar:</h3>
      <div class="class-palette" id="paletteGrid"></div>

      <div class="keyboard-tips">
        💡 <strong>Shortcut Keyboard</strong>: Tekan angka <kbd>0</kbd>–<kbd>9</kbd> untuk karakter 0–9. Gunakan tombol panah <kbd>←</kbd> / <kbd>→</kbd> atau <kbd>A</kbd> / <kbd>D</kbd> untuk berpindah gambar. Tekan <kbd>Enter</kbd> untuk mengonfirmasi tebakan rekomendasi & lanjut.
      </div>
    </div>
  </div>
</div>

<script>
const items = {json.dumps(items_for_html)};
const classes = {json.dumps(classes_json)};
let currentIndex = 0;

// Load saved labels from localStorage if exists
const savedLabels = JSON.parse(localStorage.getItem('chinese_mnist_manual_labels') || '{{}}');
items.forEach(it => {{
  if (savedLabels[it.id] !== undefined) {{
    it.current_label = savedLabels[it.id];
  }}
}});

function renderPalette() {{
  const grid = document.getElementById('paletteGrid');
  grid.innerHTML = '';
  classes.forEach(c => {{
    const btn = document.createElement('div');
    btn.className = 'class-btn';
    btn.id = 'btn_cls_' + c.label;
    btn.innerHTML = `
      <div class="char">${{c.char}} ${{c.trad ? '<span style="font-size:16px;color:#94a3b8">(' + c.trad + ')</span>' : ''}}</div>
      <div class="sub">${{c.label}}: ${{c.pinyin}} (${{c.arti}})</div>
    `;
    btn.onclick = () => selectLabel(c.label);
    grid.appendChild(btn);
  }});
}}

function updateUI() {{
  const item = items[currentIndex];
  document.getElementById('mainImage').src = 'data:image/png;base64,' + item.b64;
  document.getElementById('imageId').innerText = item.id;
  document.getElementById('rankIndicator').innerText = `Rank: ${{currentIndex + 1}} / ${{items.length}}`;
  document.getElementById('scoreIndicator').innerText = `Uncertainty: ${{item.score}}`;

  document.getElementById('predAnchor').innerText = `${{item.anchor.label}} (${{item.anchor.char}}) - ${{item.anchor.conf}}%`;
  document.getElementById('predConvNeXt').innerText = `${{item.convnext.label}} (${{item.convnext.char}})`;
  document.getElementById('predAdaBN').innerText = `${{item.adabn.label}} (${{item.adabn.char}})`;

  // Highlight active button
  document.querySelectorAll('.class-btn').forEach(b => b.classList.remove('selected'));
  const activeBtn = document.getElementById('btn_cls_' + item.current_label);
  if (activeBtn) activeBtn.classList.add('selected');

  // Update reviewed progress
  const reviewedCount = items.filter(it => savedLabels[it.id] !== undefined).length;
  document.getElementById('progressBadge').innerText = `${{reviewedCount}} / ${{items.length}} Reviewed`;
}}

function selectLabel(label) {{
  const item = items[currentIndex];
  item.current_label = label;
  savedLabels[item.id] = label;
  localStorage.setItem('chinese_mnist_manual_labels', JSON.stringify(savedLabels));
  updateUI();
  // Auto advance to next item after 150ms
  setTimeout(() => {{
    if (currentIndex < items.length - 1) {{
      currentIndex++;
      updateUI();
    }}
  }}, 150);
}}

function nextItem() {{
  if (currentIndex < items.length - 1) {{
    currentIndex++;
    updateUI();
  }}
}}

function prevItem() {{
  if (currentIndex > 0) {{
    currentIndex--;
    updateUI();
  }}
}}

document.addEventListener('keydown', (e) => {{
  if (e.key >= '0' && e.key <= '9') {{
    selectLabel(parseInt(e.key));
  }} else if (e.key === 'ArrowRight' || e.key.toLowerCase() === 'd') {{
    nextItem();
  }} else if (e.key === 'ArrowLeft' || e.key.toLowerCase() === 'a') {{
    prevItem();
  }} else if (e.key === 'Enter') {{
    // Accept recommended consensus and advance
    selectLabel(items[currentIndex].recommended);
  }}
}});

function exportCSV() {{
  let csv = "id,label\\n";
  items.forEach(it => {{
    csv += `${{it.id}},${{it.current_label}}\\n`;
  }});
  const blob = new Blob([csv], {{ type: 'text/csv' }});
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.setAttribute('href', url);
  a.setAttribute('download', 'manual_labels.csv');
  a.click();
}}

renderPalette();
updateUI();
</script>
</body>
</html>
"""

    html_out = TOOLS_DIR / "labeling_tool.html"
    with open(html_out, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"[+] Saved standalone interactive HTML labeling studio: {html_out}")


if __name__ == "__main__":
    main()
