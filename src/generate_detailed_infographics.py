#!/usr/bin/env python3
"""
================================================================================
Generate Cohesive Keynote-Scale Infographics & Diagrams for Canva Presentation
================================================================================
Strictly harmonizes with Canva template `DAHboouTey8` (#2254c5 Royal Blue,
#0F2764 Midnight Navy, #FFFFFF Pure White) with high-contrast semantic accents
and LARGE keynote-scale typography (10.5–16.5 pt) for crisp 1920x1080 display.

Generates 7 infographics in FinalProject/outputs/presentation_figures/:
1. info_15_classes_and_calligraphy.png
2. info_domain_shift_comparison.png
3. info_dataset_architecture_table.png
4. info_adm_to_dl_bridge.png
5. info_densenet_cbam_architecture.png
6. info_biomechanical_augmentation.png
7. info_kd_5loss_pipeline.png
================================================================================
"""

import json
import os
from pathlib import Path

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK TC', 'DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300

ROYAL = '#2254c5'
NAVY = '#0F2764'
COBALT = '#3B6FD8'
CYAN = '#0284C7'
INDIGO = '#4F46E5'
VIOLET = '#7C3AED'
ICE = '#EAF0FC'
ICE_ALT = '#F4F7FE'
SLATE = '#1E293B'
MUTED = '#475569'
BORDER = '#CBD5E1'
TEAL = '#0D9488'
EMERALD = '#059669'
AMBER = '#D97706'
CRIMSON = '#DC2626'

BASE_DIR = Path("/home/rei/Workspace/Academic/Multivariate Data Analysis/FinalProject")
OUTPUT_DIR = BASE_DIR / "outputs" / "presentation_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------------------
# 1. Visual Gallery of 15 Classes (Kaishu vs Cursive Gaya Baru) — Keynote Scale
# ------------------------------------------------------------------------------
def generate_15_classes_gallery():
    df_super = pd.read_csv(BASE_DIR / "data/metadata/super_dataset_15918.csv")
    char_names = [
        "0:零", "1:一", "2:二", "3:三", "4:四",
        "5:五", "6:六", "7:七", "8:八", "9:九",
        "10:十", "11:百", "12:千", "13:万", "14:亿"
    ]
    severe_classes = {5, 11}
    medium_classes = {4, 10, 12, 13}
    hard_classes = severe_classes | medium_classes

    fig, axes = plt.subplots(4, 15, figsize=(14.6, 5.4), facecolor='white')
    fig.suptitle("Anatomi 15 Kelas Angka Mandarin: Kaishu Formal vs Kaligrafi Kursif",
                 fontsize=16.0, fontweight='bold', color=NAVY, y=0.97)

    row_titles = [
        "Kaishu\nFormal",
        "Variasi\nTrain",
        "Semi-\nKursif",
        "Caoshu\nEkstrem"
    ]

    for c in range(15):
        sub_tr = df_super[(df_super['source'] == 'chinese_mnist_train') & (df_super['label'] == c)]
        sub_gb = df_super[(df_super['stratum'].isin(['val_gaya_baru', 'nccu_cursive'])) & (df_super['label'] == c)]
        if len(sub_gb) < 2:
            sub_gb = sub_tr

        samples = [
            sub_tr.iloc[0]['absolute_path'],
            sub_tr.iloc[min(15, len(sub_tr) - 1)]['absolute_path'],
            sub_gb.iloc[0]['absolute_path'],
            sub_gb.iloc[min(25, len(sub_gb) - 1)]['absolute_path']
        ]

        accent = CRIMSON if c in severe_classes else (AMBER if c in medium_classes else NAVY)

        for r in range(4):
            ax = axes[r, c]
            img_path = samples[r]
            if os.path.exists(img_path):
                img = np.array(Image.open(img_path).convert('L'))
                inv = 255 - img
                ax.imshow(inv, cmap='gray', vmin=0, vmax=255)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                is_hard_gb = (c in hard_classes and r >= 2)
                spine.set_edgecolor(accent if is_hard_gb else BORDER)
                spine.set_linewidth(2.8 if is_hard_gb else 1.0)

            if r == 0:
                ax.set_title(char_names[c], fontsize=11.0, fontweight='bold', color=accent, pad=6)
            if c == 0:
                ax.set_ylabel(row_titles[r], fontsize=10.8, fontweight='bold', color=NAVY,
                              rotation=0, labelpad=38, va='center')

    fig.text(0.53, 0.018,
             "Bingkai Merah = Kritis (5 五: 51%, 11 百: 31% di Sub 3)  |  Oranye = Ambigu (4 四, 10 十, 12 千, 13 万) -> Sub 32: 98–100%",
             ha='center', fontsize=11.0, fontweight='bold', color=NAVY,
             bbox=dict(boxstyle='round,pad=0.3', fc='#FFFBEB', ec=AMBER, lw=1.4))

    plt.tight_layout(rect=[0.06, 0.07, 0.995, 0.91])
    save_path = OUTPUT_DIR / "info_15_classes_and_calligraphy.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Generated: {save_path}")


# ------------------------------------------------------------------------------
# 2. Slide 4: Large Side-by-Side Domain Shift Comparison — Keynote Scale
# ------------------------------------------------------------------------------
def generate_domain_shift_comparison():
    df_super = pd.read_csv(BASE_DIR / "data/metadata/super_dataset_15918.csv")
    focus = [
        (5, "5: 五 (wǔ)", "Z/S loop kursif\n51% -> 100%", CRIMSON, '#FEF2F2'),
        (4, "4: 四 (sì)", "Radikal menyatu\n73% -> 100%", AMBER, '#FFFBEB'),
        (11, "11: 百 (bǎi)", "Mirip 万 / 亿\n31% -> 100%", CRIMSON, '#FEF2F2'),
        (12, "12: 千 (qiān)", "Goresan atas pudar\n73,3% -> 100%", ROYAL, ICE),
    ]

    fig, axes = plt.subplots(4, 3, figsize=(7.6, 7.6), facecolor='white',
                             gridspec_kw={'width_ratios': [1.35, 1.0, 1.0]})
    fig.suptitle("Pergeseran Domain: Kaishu vs Kursif",
                 fontsize=15.5, fontweight='bold', color=NAVY, y=0.975)

    for r, (cls_id, title, note, ec_col, bg_col) in enumerate(focus):
        ax_txt = axes[r, 0]
        ax_txt.axis('off')
        box = patches.FancyBboxPatch((0.02, 0.08), 0.95, 0.84, boxstyle="round,pad=0.04",
                                     facecolor=bg_col, edgecolor=ec_col, lw=2.0)
        ax_txt.add_patch(box)
        ax_txt.text(0.09, 0.68, title, fontsize=13.5, fontweight='bold', color=NAVY, va='center')
        ax_txt.text(0.09, 0.28, note, fontsize=11.0, fontweight='bold', color=SLATE, va='center')

        sub_tr = df_super[(df_super['source'] == 'chinese_mnist_train') & (df_super['label'] == cls_id)]
        sub_gb = df_super[(df_super['stratum'] == 'val_gaya_baru') & (df_super['label'] == cls_id)]

        for c_idx, (df_src, hdr, ec) in enumerate([
            (sub_tr, "Train Kaishu", '#94A3B8'),
            (sub_gb, "Gaya Baru", ec_col),
        ]):
            ax = axes[r, c_idx + 1]
            p = df_src.iloc[min(5 if c_idx == 1 else 2, len(df_src) - 1)]['absolute_path']
            im = 255 - np.array(Image.open(p).convert('L'))
            ax.imshow(im, cmap='gray', vmin=0, vmax=255)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor(ec); sp.set_linewidth(2.8 if c_idx == 1 else 1.4)
            if r == 0:
                ax.set_title(hdr, fontsize=12.2, fontweight='bold', color=NAVY if c_idx == 1 else MUTED, pad=7)

    plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.935])
    save_path = OUTPUT_DIR / "info_domain_shift_comparison.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Generated: {save_path}")


# ------------------------------------------------------------------------------
# 3. Slide 6: Complete 5-Strata Dataset Architecture Table PNG — Keynote Scale
# ------------------------------------------------------------------------------
def generate_dataset_architecture_table():
    fig, ax = plt.subplots(figsize=(12.2, 5.2), facecolor='white')
    ax.set_xlim(0, 12.2)
    ax.set_ylim(0, 5.2)
    ax.axis('off')

    cols = [0.15, 2.85, 7.25, 9.15, 12.05]
    headers = ["Sub-Dataset", "Penulis & Morfologi", "Citra", "Peran Utama"]
    row_h = 0.82
    y_top = 5.05

    hdr_rect = patches.FancyBboxPatch((0.15, y_top - 0.68), 11.9, 0.68,
                                      boxstyle="round,pad=0.02,rounding_size=0.08",
                                      facecolor=NAVY, edgecolor=NAVY, lw=1)
    ax.add_patch(hdr_rect)
    col_centers = [(cols[i] + cols[i + 1]) / 2 for i in range(4)]
    for cx, htxt in zip(col_centers, headers):
        ax.text(cx, y_top - 0.34, htxt, ha='center', va='center', fontsize=13.0, fontweight='bold', color='white')

    rows = [
        ("1. Train Primer\n(CM Formal)",
         "70 penulis (ID 1–70), 700/kelas,\n0 tradisional; gaya Kaishu balok",
         "10.500",
         "ADM klasik & 5-Fold\nStratifiedGroupKFold"),
        ("2. Pengayaan\n(Trad + NCCU)",
         "700 Tradisional (萬, 億) + 2.952\nNCCU terfilter + 6.000 Kaligrafi v2",
         "3.652 –\n9.652",
         "Tutup celah tradisional &\nkursif Xingshu/Caoshu"),
        ("3. Validasi Fase 1\n(Dev Benchmark)",
         "1.500 CM (penulis 71–80) + 966\nGaya Baru (Kelas 0 & 14 = 0)",
         "2.466",
         "32 submisi CodaBench\n(91,20% -> 99,84%)"),
        ("4. Super-Dataset\n(Era V KD)",
         "10.500 Train + 2.952 NCCU kursif\n+ 2.466 label Sub 31 Oracle",
         "15.918",
         "Latih Single Student SWA\nFP16 (14,75 MiB)"),
        ("5. Blind Test F2\n(Final OOD)",
         "Penulis baru (0 overlap MD5);\nKelas 0 & 14 aktif lagi (TTPA)",
         "2.902",
         "Sub #1 resmi tanpa TTA:\n98,62% (5,44 detik)")
    ]

    for r_idx, (c1, c2, c3, c4) in enumerate(rows):
        y = y_top - 0.74 - (r_idx + 1) * row_h + 0.04
        bg = ICE if r_idx % 2 == 1 else '#FFFFFF'
        ec = EMERALD if r_idx == 4 else (ROYAL if r_idx in (2, 3) else BORDER)
        row_box = patches.FancyBboxPatch((0.15, y), 11.9, row_h - 0.08,
                                         boxstyle="round,pad=0.01,rounding_size=0.06",
                                         facecolor=bg, edgecolor=ec, lw=2.0 if r_idx == 4 else 1.2)
        ax.add_patch(row_box)
        ym = y + (row_h - 0.08) / 2
        ax.text(col_centers[0], ym, c1, ha='center', va='center', fontsize=11.2, fontweight='bold', color=NAVY)
        ax.text(cols[1] + 0.14, ym, c2, ha='left', va='center', fontsize=10.8, color=SLATE, fontweight='bold')
        ax.text(col_centers[2], ym, c3, ha='center', va='center', fontsize=12.4, fontweight='bold', color=EMERALD if r_idx == 4 else ROYAL)
        ax.text(cols[3] + 0.14, ym, c4, ha='left', va='center', fontsize=10.8, color=SLATE, fontweight='bold')

    plt.tight_layout(pad=0.15)
    save_path = OUTPUT_DIR / "info_dataset_architecture_table.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Generated: {save_path}")


# ------------------------------------------------------------------------------
# 4. Conceptual Bridge: Classical ADM -> Deep Metric Learning & Distillation
# ------------------------------------------------------------------------------
def generate_adm_to_dl_bridge():
    diag_p = BASE_DIR / "outputs/multivariate_diagnostics_summary.json"
    box_m_20 = "214.542,08"
    wilks_s = "8,89e-5"
    cpve85 = 109
    if diag_p.exists():
        d = json.loads(diag_p.read_text())
        box_m_20 = f"{d['manova_m20']['box_m']['box_m']:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        cpve85 = d['pca']['cpve_85_components']

    fig, ax = plt.subplots(figsize=(12.0, 5.9), facecolor='white')
    ax.set_xlim(0, 12.0)
    ax.set_ylim(0, 6.2)
    ax.axis('off')

    ax.text(6.0, 5.92, "Jembatan Matematis: ADM Klasik -> Deep Representation Learning",
            ha='center', va='center', fontsize=15.5, fontweight='bold', color=NAVY)

    rows = [
        ("1. Kolinearitas\n(p = 4.096)",
         f"PCA Eigen-Characters\n85% var = {cpve85} PC; QDA 40 PC",
         "DenseNet-121 + CBAM\nManifold φ_θ: R^4096 -> R^256"),
        ("2. Separabilitas\nAntar-Kelas",
         f"MANOVA (m = 20 PC)\nΛ = {wilks_s} | η² = 0,4864",
         "CRD + RKD pada S^255\nInfoNCE (τ=0,07) + Sudut Triplet"),
        ("3. Kovarians\nHeterogen",
         f"Box's M = {box_m_20} (p = 0)\nQDA 96,09% >> LDA 85,13%",
         "Sub-Center ArcFace (K = 2)\nPusat 1: Kaishu | Pusat 2: Kursif"),
        ("4. Pergeseran\nPrior Populasi",
         "Prior Seragam 1/15 Gagal\nKelas 0 & 14 = 0 di Gaya Baru",
         "Bayesian EM TTPA (α = 1,05)\nEstimasi π_k dinamis tanpa label")
    ]

    headers = [(1.45, "Masalah ADM", MUTED, 0.15, 2.6),
               (4.95, "Temuan Klasik (Era 0)", ROYAL, 3.05, 3.8),
               (9.55, "Solusi Deep Learning & KD", NAVY, 7.25, 4.6)]
    for cx, htxt, col, x0, w in headers:
        rect = patches.FancyBboxPatch((x0, 5.05), w, 0.56,
                                      boxstyle="round,pad=0.05", facecolor=col, edgecolor=NAVY, lw=1.4)
        ax.add_patch(rect)
        ax.text(cx, 5.33, htxt, ha='center', va='center', fontsize=12.2, fontweight='bold', color='white')

    row_accents = [
        (ROYAL, ICE),
        (VIOLET, '#F5F3FF'),
        (EMERALD, '#ECFDF5'),
        (AMBER, '#FFFBEB'),
    ]
    y_positions = [3.82, 2.62, 1.42, 0.22]
    for idx, (prob, adm, dl) in enumerate(rows):
        y = y_positions[idx]
        ec_r, bg_r = row_accents[idx]
        b1 = patches.FancyBboxPatch((0.15, y), 2.6, 0.96, boxstyle="round,pad=0.05", facecolor='#F8FAFC', edgecolor=MUTED, lw=1.4)
        ax.add_patch(b1)
        ax.text(1.45, y + 0.48, prob, ha='center', va='center', fontsize=11.4, fontweight='bold', color=NAVY)

        b2 = patches.FancyBboxPatch((3.05, y), 3.8, 0.96, boxstyle="round,pad=0.05", facecolor=ICE_ALT, edgecolor=ROYAL, lw=1.4)
        ax.add_patch(b2)
        ax.text(4.95, y + 0.48, adm, ha='center', va='center', fontsize=11.0, color=NAVY, fontweight='bold')

        ax.annotate('', xy=(7.22, y + 0.48), xytext=(6.92, y + 0.48),
                    arrowprops=dict(facecolor=ec_r, edgecolor=NAVY, shrink=0.02, width=3.2, headwidth=9.5))

        b3 = patches.FancyBboxPatch((7.25, y), 4.6, 0.96, boxstyle="round,pad=0.05", facecolor=bg_r, edgecolor=ec_r, lw=2.2)
        ax.add_patch(b3)
        ax.text(9.55, y + 0.48, dl, ha='center', va='center', fontsize=11.2, color=NAVY, fontweight='bold')

    plt.tight_layout(pad=0.15)
    save_path = OUTPUT_DIR / "info_adm_to_dl_bridge.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Generated: {save_path}")


# ------------------------------------------------------------------------------
# 5. Exact StudentDenseNet121_CBAM Architecture Diagram — Keynote Scale
# ------------------------------------------------------------------------------
def generate_student_architecture_diagram():
    fig, ax = plt.subplots(figsize=(12.2, 5.4), facecolor='white')
    ax.set_xlim(0, 12.2)
    ax.set_ylim(0, 5.6)
    ax.axis('off')

    ax.text(6.1, 5.30, "Arsitektur StudentDenseNet121_CBAM + Sub-Center ArcFace (14,75 MiB FP16)",
            ha='center', va='center', fontsize=14.8, fontweight='bold', color=NAVY)

    blocks = [
        (0.15, 1.65, 1.85, 2.75, MUTED, "Input\n64×64×1",
         "• Skala [-1, 1]\n• Elastic & Kuas\n  saat training\n• 1× Pass Uji\n  (Tanpa TTA)"),
        (2.30, 1.65, 2.15, 2.75, CYAN, "Stem 3×3\n+ MixStyle",
         "• Conv 3×3 s=1\n  (No MaxPool)\n• MixStyle 1×\n  (p=0,5, α=0,1)\n  pasca-stem"),
        (4.75, 1.65, 2.15, 2.75, ROYAL, "4 DenseBlock\n+ 4 CBAM",
         "• (6, 12, 24, 16)\n• Konkatenasi\n  1.024 kanal\n• CBAM 7×7 tiap\n  DenseBlock"),
        (7.20, 1.65, 2.15, 2.75, VIOLET, "Manifold\n256-D L2",
         "• GAP -> 256-D\n• BN1d + PReLU\n• Norm L2 S^255\n• Target CRD\n  & RKD Loss"),
        (9.65, 3.12, 2.40, 1.38, EMERALD, "ArcFace (K=2)",
         "• Bobot 30×256\n• s=32, m=0,30\n• Kaishu & Kursif"),
        (9.65, 1.50, 2.40, 1.38, AMBER, "Linear Head",
         "• 256 -> 15 kelas\n• Fusi 50/50:\n  0,5 z_lin+0,5 z_arc")
    ]

    for x, y, w, h, col, title, desc in blocks:
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", facecolor='#FFFFFF', edgecolor=col, lw=2.4)
        ax.add_patch(box)
        header_h = 0.72 if h > 2.0 else 0.46
        header = patches.FancyBboxPatch((x, y + h - header_h), w, header_h, boxstyle="round,pad=0.03", facecolor=col, edgecolor=col, lw=1)
        ax.add_patch(header)
        ax.text(x + w / 2, y + h - header_h / 2, title, ha='center', va='center', fontsize=11.4, fontweight='bold', color='white')
        ax.text(x + w / 2, y + (h - header_h) / 2, desc, ha='center', va='center', fontsize=10.3, color=SLATE, fontweight='bold')

    for x1, x2 in [(2.04, 2.27), (4.49, 4.72), (6.94, 7.17)]:
        ax.annotate('', xy=(x2, 3.02), xytext=(x1, 3.02),
                    arrowprops=dict(facecolor=NAVY, edgecolor=NAVY, shrink=0.02, width=3.0, headwidth=9.0))

    ax.annotate('', xy=(9.62, 3.80), xytext=(9.38, 3.25),
                arrowprops=dict(facecolor=EMERALD, edgecolor=EMERALD, shrink=0.02, width=2.8, headwidth=8.5))
    ax.annotate('', xy=(9.62, 2.18), xytext=(9.38, 2.80),
                arrowprops=dict(facecolor=AMBER, edgecolor=AMBER, shrink=0.02, width=2.8, headwidth=8.5))

    bottom_box = patches.FancyBboxPatch((0.15, 0.22), 11.9, 0.96, boxstyle="round,pad=0.05", facecolor=ICE, edgecolor=ROYAL, lw=2.0)
    ax.add_patch(bottom_box)
    ax.text(6.1, 0.70,
            "Produksi: 7.522.968 Param | 14,75 MiB FP16 | 5,44 Detik / 2.902 Citra GPU\n"
            "MeanTeacherEMA (α=0,995)  +  SWA Ep 17–25  +  Logit Fusion 50/50  +  EM TTPA (α=1,05)",
            ha='center', va='center', fontsize=11.2, fontweight='bold', color=NAVY)

    plt.tight_layout(pad=0.15)
    save_path = OUTPUT_DIR / "info_densenet_cbam_architecture.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Generated: {save_path}")


# ------------------------------------------------------------------------------
# 6. Visual Biomechanical Augmentation Showcase — Keynote Scale
# ------------------------------------------------------------------------------
def generate_biomechanical_augmentation_visual():
    df_super = pd.read_csv(BASE_DIR / "data/metadata/super_dataset_15918.csv")
    classes_demo = [(5, "5: 五"), (4, "4: 四"), (11, "11: 百"), (12, "12: 千")]

    fig, axes = plt.subplots(4, 4, figsize=(8.2, 7.3), facecolor='white')
    fig.suptitle("Augmentasi Biomekanik Simard & Kuas Anisotropik",
                 fontsize=14.8, fontweight='bold', color=NAVY, y=0.97)
    col_titles = [
        "1. Asli\n(Kaishu)",
        "2. Elastic\n(α=7, σ=4)",
        "3. Dilasi 45°\n(Kuas Tebal)",
        "4. Erosi 2×2\n(Feibai)"
    ]

    rng = np.random.default_rng(7)
    k45 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.uint8)
    k2 = np.ones((2, 2), dtype=np.uint8)

    for r, (cls_id, lbl) in enumerate(classes_demo):
        p = df_super[(df_super['source'] == 'chinese_mnist_train') & (df_super['label'] == cls_id)].iloc[4]['absolute_path']
        im = np.array(Image.open(p).convert('L'))
        H, W = im.shape
        dx = gaussian_filter(rng.uniform(-1, 1, (H, W)), 4.0, mode='constant') * 7.0
        dy = gaussian_filter(rng.uniform(-1, 1, (H, W)), 4.0, mode='constant') * 7.0
        gx, gy = np.meshgrid(np.arange(W), np.arange(H))
        im_el = cv2.remap(im, (gx + dx).astype(np.float32), (gy + dy).astype(np.float32),
                          interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        im_dil = cv2.dilate(im_el, k45, iterations=1)
        im_ero = cv2.erode(im_el, k2, iterations=1)

        col_borders = [BORDER, ROYAL, EMERALD, AMBER]
        col_title_cols = [MUTED, ROYAL, '#065F46', '#92400E']
        for c, variant in enumerate([im, im_el, im_dil, im_ero]):
            ax = axes[r, c]
            ax.imshow(255 - variant, cmap='gray', vmin=0, vmax=255)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor(col_borders[c])
                sp.set_linewidth(2.6 if c > 0 else 1.4)
            if r == 0:
                ax.set_title(col_titles[c], fontsize=11.4, fontweight='bold', color=col_title_cols[c], pad=6)
            if c == 0:
                ax.set_ylabel(lbl, fontsize=12.5, fontweight='bold', color=NAVY, rotation=0, labelpad=36, va='center')

    plt.tight_layout(rect=[0.07, 0.01, 0.99, 0.925])
    save_path = OUTPUT_DIR / "info_biomechanical_augmentation.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Generated: {save_path}")


# ------------------------------------------------------------------------------
# 7. Conceptual 5-Component Unified KD Pipeline — Keynote Scale
# ------------------------------------------------------------------------------
def generate_kd_5loss_diagram():
    fig, ax = plt.subplots(figsize=(12.2, 5.8), facecolor='white')
    ax.set_xlim(0, 12.2)
    ax.set_ylim(0, 6.0)
    ax.axis('off')

    ax.text(6.1, 5.72, "Aliran 5-Komponen Knowledge Distillation (UnifiedDistillationLoss)",
            ha='center', va='center', fontsize=15.5, fontweight='bold', color=NAVY)

    # Top Left: Oracle Labels / Super-Dataset
    gt_box = patches.FancyBboxPatch((0.15, 4.48), 3.35, 0.90, boxstyle="round,pad=0.05", facecolor='#F8FAFC', edgecolor=MUTED, lw=1.8)
    ax.add_patch(gt_box)
    ax.text(1.825, 4.93, "SUPER-DATASET (15.918)\nCM + NCCU + Sub 31 Oracle\nLabel Keras (y_true)",
            ha='center', va='center', fontsize=10.3, fontweight='bold', color=NAVY)

    # Middle Left: Teacher (Online MeanTeacherEMA)
    t_box = patches.FancyBboxPatch((0.15, 2.52), 3.35, 1.66, boxstyle="round,pad=0.05", facecolor=ICE, edgecolor=ROYAL, lw=2.3)
    ax.add_patch(t_box)
    ax.text(1.825, 3.84, "GURU: MeanTeacherEMA\n(α_EMA = 0,995 | x_clean)",
            ha='center', va='center', fontsize=10.8, fontweight='bold', color=NAVY)
    ax.text(1.825, 3.00,
            "• Output z_T (15-D) & f_T (256-D)\n"
            "• Bobot halus bebas noise batch",
            ha='center', va='center', fontsize=9.8, color=SLATE, fontweight='bold')

    ax.text(1.825, 2.22, "Pasangan Sinyal: (z_T, z_S) & (f_T, f_S)",
            ha='center', va='center', fontsize=9.4, fontweight='bold', color=NAVY,
            bbox=dict(boxstyle='round,pad=0.22', fc='#FFFBEB', ec=AMBER, lw=1.3))

    # Bottom Left: Student
    s_box = patches.FancyBboxPatch((0.15, 0.24), 3.35, 1.68, boxstyle="round,pad=0.05", facecolor='#ECFDF5', edgecolor=EMERALD, lw=2.3)
    ax.add_patch(s_box)
    ax.text(1.825, 1.58, "MURID: Sub 32 (SWA)\nDenseNet-121 (14,75 MiB)",
            ha='center', va='center', fontsize=10.8, fontweight='bold', color='#065F46')
    ax.text(1.825, 0.76,
            "• Input x_aug + MixStyle + ArcFace\n"
            "• Output z_S = 0,5(z_lin+z_arc), f_S",
            ha='center', va='center', fontsize=9.8, color=SLATE, fontweight='bold')

    # Vertical Convergence Bus
    ax.plot([3.95, 3.95], [0.70, 4.72], color=ROYAL, lw=2.8)
    ax.annotate('', xy=(3.95, 3.35), xytext=(3.54, 3.35), arrowprops=dict(facecolor=ROYAL, edgecolor=ROYAL, width=2.6, headwidth=8))
    ax.annotate('', xy=(3.95, 1.10), xytext=(3.54, 1.10), arrowprops=dict(facecolor=EMERALD, edgecolor=EMERALD, width=2.6, headwidth=8))

    ax.annotate('', xy=(4.42, 5.02), xytext=(3.54, 5.02), arrowprops=dict(facecolor=MUTED, edgecolor=MUTED, width=2.4, headwidth=7.5))
    ax.annotate('', xy=(4.42, 4.72), xytext=(3.95, 4.72), arrowprops=dict(facecolor=EMERALD, edgecolor=EMERALD, width=2.4, headwidth=7.5))

    losses = [
        (4.46, 4.44, NAVY, '#F8FAFC', "1. Task & ArcFace Margin (L_task, LS ε = 0,05)",
         "Cross-Entropy z_S vs y_true untuk margin sudut keras 15 kelas."),
        (4.46, 3.39, ROYAL, ICE_ALT, "2. Response Dark Knowledge (α = 0,70 · L_KD, T = 3,5)",
         "T² · KL(Softmax(z_T/T) || Softmax(z_S/T)): transfer kemiripan halus."),
        (4.46, 2.34, VIOLET, '#F5F3FF', "3. Contrastive Representation (β = 0,10 · L_CRD, τ = 0,07)",
         "InfoNCE di S^255: selaraskan fitur laten f_S dengan f_T guru."),
        (4.46, 1.29, TEAL, '#F0FDFA', "4. Distance Relational KD (γ = 1,00 · L_RKD-dist)",
         "Smooth-L1 jarak Euclidean berpasangan guru vs murid dalam batch."),
        (4.46, 0.24, AMBER, '#FFFBEB', "5. Angle Relational KD (δ = 2,00 · L_RKD-angle)",
         "Smooth-L1 kosinus sudut triplet untuk menjaga geometri manifold.")
    ]

    for x, y, col, bg_l, title, desc in losses:
        l_box = patches.FancyBboxPatch((x, y), 7.55, 0.88, boxstyle="round,pad=0.04", facecolor=bg_l, edgecolor=col, lw=2.2)
        ax.add_patch(l_box)
        ax.text(x + 0.16, y + 0.62, title, ha='left', va='center', fontsize=11.0, fontweight='bold', color=col)
        ax.text(x + 0.16, y + 0.24, desc, ha='left', va='center', fontsize=10.0, color=SLATE, fontweight='bold')
        if y < 4.3:
            ax.annotate('', xy=(x - 0.04, y + 0.44), xytext=(3.95, y + 0.44),
                        arrowprops=dict(facecolor=col, edgecolor=col, width=2.4, headwidth=7.5))

    plt.tight_layout(pad=0.15)
    save_path = OUTPUT_DIR / "info_kd_5loss_pipeline.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Generated: {save_path}")


# ------------------------------------------------------------------------------
# 8. Slide 28: Codebase, 9 Kaggle Notebooks & GitHub Repository Map
# ------------------------------------------------------------------------------
def generate_code_and_github_diagram():
    fig, ax = plt.subplots(figsize=(12.2, 5.2), facecolor='white')
    ax.set_xlim(0, 12.2)
    ax.set_ylim(0, 5.3)
    ax.axis('off')

    ax.text(6.1, 4.98, "Peta Kode, 9 Notebook Kaggle & Repositori GitHub (Reiii0-0/MNIST-ADM)",
            ha='center', va='center', fontsize=15.0, fontweight='bold', color=NAVY)

    pillars = [
        (0.18, 1.38, 3.72, 3.15, NAVY, "1. ADM Klasik & Praproses (src/)",
         "• multivariate_analysis.py\n  PCA, MANOVA, Box's M, LDA vs QDA\n"
         "• lecturer_exact_filter.py\n  Otsu + Crop Rasio + Blur σ=0,8\n"
         "• biomechanical_augmentation.py\n  Simard Elastic + Kuas 45°/135°"),
        (4.24, 1.38, 3.72, 3.15, ROYAL, "2. Arsitektur & Distilasi 5-Loss",
         "• model_densenet_cbam.py\n  DenseNet-121 CBAM + MixStyle + ArcFace\n"
         "• distillation_loss.py\n  Task + KD(T=3,5) + CRD + RKD D/A\n"
         "• test_time_prior_adaptation.py\n  Bayesian EM TTPA (α=1,05, 5,51 ms)"),
        (8.30, 1.38, 3.72, 3.15, EMERALD, "3. Kaggle Notebooks & Inferensi",
         "• notebooks/01..09 (.ipynb)\n  9 Kernel (ResNet18 -> Distilasi Final)\n"
         "• synthesize_sub30_...fusion.py\n  Joint MILP HiGHS (4.993 eqs, 99,76%)\n"
         "• predict_phase2_distillation.py\n  1× Pass FP16 (5,44 dtk -> 98,62%)"),
    ]

    for x, y, w, h, col, title, body in pillars:
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", facecolor='#FFFFFF', edgecolor=col, lw=2.4)
        ax.add_patch(box)
        hdr = patches.FancyBboxPatch((x, y + h - 0.58), w, 0.58, boxstyle="round,pad=0.03", facecolor=col, edgecolor=col, lw=1)
        ax.add_patch(hdr)
        ax.text(x + w / 2, y + h - 0.29, title, ha='center', va='center', fontsize=11.4, fontweight='bold', color='white')
        ax.text(x + 0.16, y + (h - 0.58) / 2, body, ha='left', va='center', fontsize=10.2, color=SLATE, fontweight='bold', linespacing=1.42)

    footer = patches.FancyBboxPatch((0.18, 0.20), 11.84, 0.88, boxstyle="round,pad=0.05", facecolor=ICE, edgecolor=ROYAL, lw=2.0)
    ax.add_patch(footer)
    ax.text(6.1, 0.64,
            "GitHub: https://github.com/Reiii0-0/MNIST-ADM  (notebooks/01–09 .ipynb, src/, dokumentasi/)\n"
            "Reproduksi Bersih: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt  |  Zero Checkpoints Bloat",
            ha='center', va='center', fontsize=10.8, fontweight='bold', color=NAVY, linespacing=1.4)

    plt.tight_layout(pad=0.15)
    save_path = OUTPUT_DIR / "info_code_and_github_repo.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Generated: {save_path}")


if __name__ == "__main__":
    generate_15_classes_gallery()
    generate_domain_shift_comparison()
    generate_dataset_architecture_table()
    generate_adm_to_dl_bridge()
    generate_student_architecture_diagram()
    generate_biomechanical_augmentation_visual()
    generate_kd_5loss_diagram()
    generate_code_and_github_diagram()
    print("\nAll 8 Keynote-Scale infographics generated successfully!")
