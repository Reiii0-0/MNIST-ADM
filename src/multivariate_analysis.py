import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.config import get_splits_kfold_csv, get_train_dir, OUTPUTS_DIR

# Set Canva Royal-Blue Academic Theme (#2254c5 / #0F2764 / #FFFFFF)
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK TC', 'DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.edgecolor'] = '#CBD5E1'
plt.rcParams['grid.color'] = '#E2E8F0'

ROYAL_BLUE = '#2254c5'
DEEP_NAVY = '#0F2764'
SOFT_BLUE = '#5B84E8'
SLATE_TEXT = '#1E293B'
MUTED_SLATE = '#475569'

def load_dataset(splits_csv=None, img_dir=None):
    if splits_csv is None:
        splits_csv = str(get_splits_kfold_csv())
    if img_dir is None:
        img_dir = str(get_train_dir())

    print("Loading images and labels...")
    df = pd.read_csv(splits_csv)
    n = len(df)
    p = 64 * 64
    
    X = np.zeros((n, p), dtype=np.float32)
    y = df['label'].values.astype(np.int64)
    writers = df['writer_id'].values
    folds = df['fold'].values
    
    for i, row in df.iterrows():
        path = os.path.join(img_dir, f"{row['id']}.png")
        img = Image.open(path).convert('L')
        # Normalize pixels to [0, 1]
        X[i] = np.array(img, dtype=np.float32).flatten() / 255.0
        
    print(f"Loaded feature matrix X: shape {X.shape}, labels y: shape {y.shape}")
    return X, y, writers, folds, df

def run_pca_analysis(X, output_dir=None):
    if output_dir is None:
        output_dir = str(OUTPUTS_DIR)
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("1. PRINCIPAL COMPONENT ANALYSIS (PCA & EIGEN-CHARACTERS)")
    print("=" * 60)
    
    # Center the data
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean
    
    # Fit full PCA
    pca = PCA(n_components=300, svd_solver='randomized', random_state=42)
    pca.fit(X_centered)
    
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)
    eigenvalues = pca.explained_variance_
    
    mean_eigenval = np.mean(np.var(X, axis=0))  # Average eigenvalue threshold for Kaiser criterion
    kaiser_count = int(np.sum(eigenvalues > mean_eigenval))
    
    cpve_80 = int(np.argmax(cumulative_var >= 0.80) + 1)
    cpve_85 = int(np.argmax(cumulative_var >= 0.85) + 1)
    cpve_90 = int(np.argmax(cumulative_var >= 0.90) + 1)
    cpve_95 = int(np.argmax(cumulative_var >= 0.95) + 1)
    
    print(f"Mean eigenvalue (Kaiser threshold): {mean_eigenval:.6f}")
    print(f"Components exceeding Kaiser criterion: {kaiser_count}")
    print(f"Components for 80% variance: {cpve_80}")
    print(f"Components for 85% variance: {cpve_85}")
    print(f"Components for 90% variance: {cpve_90}")
    print(f"Components for 95% variance: {cpve_95}")
    
    # 1. Scree Plot (Keynote-Scale High-Contrast Canva Royal-Blue Academic Palette)
    EMERALD = '#059669'
    AMBER = '#D97706'
    CRIMSON = '#DC2626'

    fig, ax1 = plt.subplots(figsize=(9.2, 5.2), facecolor='white')
    ax1.set_facecolor('white')
    x_axis = np.arange(1, len(explained_var) + 1)

    ax1.set_xlabel('Indeks Komponen Utama (Principal Component)', fontsize=13.5, fontweight='bold', color=DEEP_NAVY)
    ax1.set_ylabel('Rasio Variansi Individual', color=ROYAL_BLUE, fontsize=13.5, fontweight='bold')
    ax1.fill_between(x_axis[:100], 0, explained_var[:100], color='#DBEAFE', alpha=0.48)
    ax1.plot(x_axis[:100], explained_var[:100], color=ROYAL_BLUE, marker='o', markersize=4.5, lw=2.6,
             label='Variansi Individual (PC1–PC100)')
    ax1.tick_params(axis='both', labelsize=11.5)
    ax1.tick_params(axis='y', labelcolor=ROYAL_BLUE)
    ax1.set_ylim(0, 0.088)
    ax1.spines['top'].set_visible(False)

    ax1.annotate(f'PC1 = {explained_var[0]*100:.2f}%\nPC2 = {explained_var[1]*100:.2f}%',
                 xy=(1, explained_var[0]), xytext=(18, 0.058),
                 arrowprops=dict(arrowstyle='->', color=ROYAL_BLUE, lw=2.2),
                 fontsize=11.8, fontweight='bold', color=DEEP_NAVY, zorder=10,
                 bbox=dict(boxstyle='round,pad=0.4', fc='#EAF0FC', ec=ROYAL_BLUE, lw=1.6))

    ax2 = ax1.twinx()
    ax2.set_ylabel('Variansi Kumulatif (CPVE)', color=DEEP_NAVY, fontsize=13.5, fontweight='bold')
    ax2.plot(x_axis[:100], cumulative_var[:100], color=DEEP_NAVY, linewidth=3.0, label='Variansi Kumulatif (CPVE)')
    ax2.axhline(0.85, color=EMERALD, linestyle='--', linewidth=2.2, label=f'Target 85% ({cpve_85} PC)')
    ax2.axhline(0.90, color=AMBER, linestyle='-.', linewidth=2.2, label=f'Target 90% ({cpve_90} PC)')
    ax2.set_ylim(0, 1.10)
    ax2.tick_params(axis='y', labelsize=11.5, labelcolor=DEEP_NAVY)
    ax2.spines['top'].set_visible(False)

    ax2.text(98, 0.865, f'85% -> {cpve_85} PC (QDA: 40 PC)', ha='right', va='bottom', fontsize=11.2,
             fontweight='bold', color='#065F46', bbox=dict(boxstyle='round,pad=0.25', fc='#ECFDF5', ec=EMERALD, lw=1.2))
    ax2.text(98, 0.945, f'90% -> {cpve_90} PC | Kaiser: {kaiser_count} PC', ha='right', va='bottom', fontsize=11.2,
             fontweight='bold', color='#92400E', bbox=dict(boxstyle='round,pad=0.25', fc='#FFFBEB', ec=AMBER, lw=1.2))

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center right', bbox_to_anchor=(0.99, 0.43),
               frameon=True, facecolor='white', edgecolor='#CBD5E1', fontsize=10.8)

    ax1.set_title('Scree Plot & Variansi Kumulatif PCA (p = 4.096 Piksel)', fontsize=15.5, fontweight='bold', color=DEEP_NAVY, pad=14)
    fig.tight_layout()
    scree_path = os.path.join(output_dir, "scree_plot.png")
    plt.savefig(scree_path, dpi=300)
    plt.close()
    print(f"Saved Scree Plot to: {scree_path}")

    # 2. Visualizing Eigen-Characters (High-contrast Royal Blue - White - Amber/Crimson)
    from matplotlib.colors import LinearSegmentedColormap
    high_contrast_div = LinearSegmentedColormap.from_list('royal_amber_div', [DEEP_NAVY, ROYAL_BLUE, '#FFFFFF', '#F59E0B', CRIMSON])
    fig, axes = plt.subplots(4, 4, figsize=(7.6, 7.6), facecolor='white')
    fig.suptitle('16 Eigen-Characters Utama (PC1–PC16, Kutub +/- PCA)', fontsize=15.5, fontweight='bold', color=DEEP_NAVY, y=0.98)
    for idx, ax in enumerate(axes.flat):
        eigen_img = pca.components_[idx].reshape(64, 64)
        vmax = np.max(np.abs(eigen_img))
        ax.imshow(eigen_img, cmap=high_contrast_div, vmin=-vmax, vmax=vmax, interpolation='nearest')
        ax.set_title(f"PC{idx+1} ({explained_var[idx]*100:.2f}%)", fontsize=11.2, fontweight='bold', color=DEEP_NAVY, pad=4)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor('#CBD5E1')
            sp.set_linewidth(1.0)
    plt.tight_layout(rect=[0, 0, 1, 0.945])
    eigen_path = os.path.join(output_dir, "eigen_characters.png")
    plt.savefig(eigen_path, dpi=300)
    plt.close()
    print(f"Saved Eigen-Characters Grid to: {eigen_path}")
    
    pca_summary = {
        "n_samples": int(X.shape[0]),
        "n_pixels": int(X.shape[1]),
        "kaiser_threshold": float(mean_eigenval),
        "kaiser_count": kaiser_count,
        "cpve_80_components": cpve_80,
        "cpve_85_components": cpve_85,
        "cpve_90_components": cpve_90,
        "cpve_95_components": cpve_95,
        "top_16_explained_variance_pct": [round(float(v) * 100.0, 4) for v in explained_var[:16]]
    }
    return pca, pca_summary

def compute_box_m(X_pca, y):
    """Numerically stable Box's M test using slogdet to avoid underflow on high dimensions."""
    classes = np.unique(y)
    K = len(classes)
    N, m = X_pca.shape
    df_total = N - K

    S_pooled = np.zeros((m, m), dtype=np.float64)
    sum_df_logdet = 0.0
    inv_df_sum = 0.0

    for c in classes:
        X_c = X_pca[y == c].astype(np.float64)
        n_c = len(X_c)
        df_c = n_c - 1
        S_c = np.cov(X_c, rowvar=False, bias=False)
        sign_c, logdet_c = np.linalg.slogdet(S_c)
        if sign_c <= 0:
            S_c = S_c + 1e-8 * np.eye(m)
            _, logdet_c = np.linalg.slogdet(S_c)
        sum_df_logdet += df_c * logdet_c
        S_pooled += df_c * S_c
        inv_df_sum += 1.0 / df_c

    S_pooled /= df_total
    _, logdet_pooled = np.linalg.slogdet(S_pooled)

    box_m = float(df_total * logdet_pooled - sum_df_logdet)
    c_factor = ((2 * m * m + 3 * m - 1) / (6 * (m + 1) * (K - 1))) * (inv_df_sum - 1.0 / df_total)
    chi2_stat = float(box_m * (1.0 - c_factor))
    df_chi2 = int(0.5 * (K - 1) * m * (m + 1))
    p_val = float(stats.chi2.sf(chi2_stat, df=df_chi2))

    return {
        "m_components": int(m),
        "box_m": box_m,
        "chi2_approx": chi2_stat,
        "df": df_chi2,
        "p_value": p_val,
        "c_factor": float(c_factor),
    }


def run_manova_and_diagnostics(X_pca, y, output_dir=None):
    if output_dir is None:
        output_dir = str(OUTPUTS_DIR)
    os.makedirs(output_dir, exist_ok=True)
    print("\n" + "=" * 60)
    print("2. MULTIVARIATE HYPOTHESIS TESTING (MANOVA) & DIAGNOSTICS")
    print("=" * 60)
    
    # Compute Class Means and Overall Mean
    classes = np.unique(y)
    K = len(classes)
    N, m = X_pca.shape
    grand_mean = np.mean(X_pca, axis=0)
    
    # Hypothesis (Between-class) Matrix H and Error (Within-class) Matrix E
    H = np.zeros((m, m))
    E = np.zeros((m, m))
    
    for c in classes:
        X_c = X_pca[y == c]
        N_c = len(X_c)
        c_mean = np.mean(X_c, axis=0)
        
        diff_mean = (c_mean - grand_mean).reshape(-1, 1)
        H += N_c * (diff_mean @ diff_mean.T)
        
        diff_samples = X_c - c_mean
        E += diff_samples.T @ diff_samples
        
    df_h = K - 1
    df_e = N - K
    
    # Generalized Eigenvalues of E^-1 H
    E_inv_H = np.linalg.pinv(E) @ H
    eigvals = np.real(np.linalg.eigvals(E_inv_H))
    eigvals = eigvals[eigvals > 1e-10]
    
    # 1. Wilks' Lambda
    wilks_lambda = np.prod(1.0 / (1.0 + eigvals))
    # 2. Pillai's Trace
    pillai_trace = np.sum(eigvals / (1.0 + eigvals))
    # 3. Hotelling-Lawley Trace
    hotelling_trace = np.sum(eigvals)
    # 4. Roy's Largest Root
    roys_root = np.max(eigvals) / (1.0 + np.max(eigvals)) if len(eigvals) > 0 else 0
    # Partial Eta Squared
    eta_sq = 1.0 - (wilks_lambda ** (1.0 / min(df_h, m)))
    
    # Box's M Homogeneity of Covariance Test (log-determinant stable)
    box_m_res = compute_box_m(X_pca, y)

    print(f"MANOVA Statistics on Top {m} PCA Components:")
    print(f"  • Wilks' Lambda (Λ)       : {wilks_lambda:.6e} (Strong separation as Λ -> 0)")
    print(f"  • Pillai's Trace (V)      : {pillai_trace:.4f}")
    print(f"  • Hotelling-Lawley (T)   : {hotelling_trace:.4f}")
    print(f"  • Roy's Largest Root (θ) : {roys_root:.4f}")
    print(f"  • Partial Eta Squared    : {eta_sq:.4f} (> 0.14 indicates very large effect)")
    print(f"Box's M Covariance Homogeneity Test (m={m}):")
    print(f"  • Box's M                : {box_m_res['box_m']:,.2f}")
    print(f"  • Chi-Square Approx      : {box_m_res['chi2_approx']:,.2f} (df={box_m_res['df']}, p={box_m_res['p_value']:.2e})")
    
    # Chi-Square Q-Q Plot of Mahalanobis Distances
    print("\nComputing Mahalanobis Distances for Multivariate Normality Check...")
    S_cov = np.cov(X_pca, rowvar=False)
    S_inv = np.linalg.pinv(S_cov)
    diff = X_pca - grand_mean
    d2 = np.sum((diff @ S_inv) * diff, axis=1)
    
    # Q-Q plot against Chi-Square with m degrees of freedom
    d2_sorted = np.sort(d2)
    quantiles = stats.chi2.ppf((np.arange(1, N + 1) - 0.5) / N, df=m)
    
    chi2_99 = stats.chi2.ppf(0.99, df=m)
    tail_mask = d2_sorted > chi2_99

    fig, ax = plt.subplots(figsize=(7.4, 6.8), facecolor='white')
    ax.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.scatter(quantiles[~tail_mask], d2_sorted[~tail_mask], alpha=0.62, s=26, color=ROYAL_BLUE,
               edgecolors='none', label=f'Observasi Inti (n={np.sum(~tail_mask):,})')
    ax.scatter(quantiles[tail_mask], d2_sorted[tail_mask], alpha=0.90, s=36, color='#DC2626',
               edgecolors='none', label=f'Ekor Berat > χ²_0,99 (n={np.sum(tail_mask)})')
    max_val = max(np.max(quantiles), np.max(d2_sorted[:int(N * 0.99)]))
    ax.plot([0, max_val], [0, max_val], color='#D97706', linestyle='--', linewidth=3.2,
            label='Referensi Normalitas (y = x)')
    ax.set_xlabel(f'Kuantil Teoretis χ² (df = {m})', fontsize=13.5, fontweight='bold', color=DEEP_NAVY)
    ax.set_ylabel('Jarak Mahalanobis Kuadrat (d²)', fontsize=13.5, fontweight='bold', color=DEEP_NAVY)
    ax.set_title('Diagnostik Normalitas: Plot Q-Q Mahalanobis', fontsize=15.0, fontweight='bold', color=DEEP_NAVY, pad=12)
    ax.tick_params(axis='both', labelsize=11.5)
    ax.text(0.04, 0.57,
            f"Box's M (m={m}):\n"
            f"M = {box_m_res['box_m']:,.2f}\n"
            f"χ² = {box_m_res['chi2_approx']:,.0f} | p = 0\n"
            f"Tolak H0: Σ_i ≠ Σ_j",
            transform=ax.transAxes, fontsize=11.2, fontweight='bold', color=DEEP_NAVY,
            bbox=dict(boxstyle='round,pad=0.4', fc='#FFFBEB', ec='#D97706', lw=1.6))
    ax.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='#CBD5E1', fontsize=10.8, markerscale=1.4)
    fig.tight_layout()
    qq_path = os.path.join(output_dir, "mahalanobis_qq_plot.png")
    plt.savefig(qq_path, dpi=300)
    plt.close()
    print(f"Saved Mahalanobis Q-Q Plot to: {qq_path}")
    
    return {
        "m_components": int(m),
        "wilks_lambda": float(wilks_lambda),
        "pillai_trace": float(pillai_trace),
        "hotelling_trace": float(hotelling_trace),
        "roys_root": float(roys_root),
        "partial_eta_squared": float(eta_sq),
        "box_m": box_m_res,
    }

def run_cross_validation_models(X, y, folds, output_dir=None):
    if output_dir is None:
        output_dir = str(OUTPUTS_DIR)
    os.makedirs(output_dir, exist_ok=True)
    print("\n" + "=" * 60)
    print("3. MULTIVARIATE CLASSIFICATION BENCHMARK (5-FOLD STRATIFIED GROUP K-FOLD)")
    print("=" * 60)
    
    n_splits = 5
    models = {
        "Two-Stage PCA (100) + LDA": {
            "type": "pca_lda",
            "pca_dim": 100
        },
        "Two-Stage PCA (40) + QDA": {
            "type": "pca_qda",
            "pca_dim": 40,
            "reg_param": 0.1
        },
        "Two-Stage PCA (100) + Logistic Regression": {
            "type": "pca_lr",
            "pca_dim": 100,
            "C": 1.0
        }
    }
    
    results = {}
    oof_predictions = {name: np.zeros(len(y), dtype=np.int64) for name in models}
    
    for name, config in models.items():
        print(f"\nEvaluating Model: {name}...")
        fold_accs = []
        
        for fold in range(n_splits):
            train_mask = (folds != fold)
            val_mask = (folds == fold)
            
            X_tr, y_tr = X[train_mask], y[train_mask]
            X_val, y_val = X[val_mask], y[val_mask]
            
            # 1. Fit PCA strictly on training fold
            pca = PCA(n_components=config["pca_dim"], random_state=42)
            X_tr_pca = pca.fit_transform(X_tr)
            X_val_pca = pca.transform(X_val)
            
            # 2. Fit Classifier
            if config["type"] == "pca_lda":
                clf = LinearDiscriminantAnalysis()
            elif config["type"] == "pca_qda":
                clf = QuadraticDiscriminantAnalysis(reg_param=config.get("reg_param", 0.1))
            elif config["type"] == "pca_lr":
                clf = LogisticRegression(C=config.get("C", 1.0), max_iter=500, solver='lbfgs')
                
            clf.fit(X_tr_pca, y_tr)
            preds = clf.predict(X_val_pca)
            
            oof_predictions[name][val_mask] = preds
            acc = accuracy_score(y_val, preds)
            fold_accs.append(acc)
            print(f"  Fold {fold}: Accuracy = {acc * 100:.2f}%")
            
        overall_acc = accuracy_score(y, oof_predictions[name])
        mean_acc = np.mean(fold_accs)
        std_acc = np.std(fold_accs)
        print(f"--> {name} Overall OOF Accuracy: {overall_acc * 100:.2f}% (Mean: {mean_acc * 100:.2f}% ± {std_acc * 100:.2f}%)")
        
        results[name] = {
            "overall_oof_accuracy": float(overall_acc),
            "fold_accuracies": [float(a) for a in fold_accs],
            "mean_accuracy": float(mean_acc),
            "std_accuracy": float(std_acc)
        }
        
    # Generate Canonical LDA Projection Plot (High-Contrast 15-Class Categorical Palette anchored in Canva Navy)
    print("\nGenerating Canonical Discriminant Analysis 2D Projection...")
    pca = PCA(n_components=100, random_state=42)
    X_pca = pca.fit_transform(X)
    lda = LinearDiscriminantAnalysis(n_components=2)
    X_lda_2d = lda.fit_transform(X_pca, y)

    palette_15 = [
        '#0F2764',  # 0: Midnight Navy
        '#DC2626',  # 1: Crimson Red
        '#059669',  # 2: Emerald Green
        '#D97706',  # 3: Amber Gold
        '#7C3AED',  # 4: Vivid Violet
        '#0284C7',  # 5: Azure Cyan
        '#DB2777',  # 6: Rose Magenta
        '#EA580C',  # 7: Burnt Orange
        '#0D9488',  # 8: Deep Teal
        '#2254c5',  # 9: Canva Royal Blue
        '#65A30D',  # 10: Crisp Lime Olive
        '#991B1B',  # 11: Dark Ruby
        '#4F46E5',  # 12: Royal Indigo
        '#0891B2',  # 13: Dark Cyan
        '#475569',  # 14: Charcoal Slate
    ]
    class_names_short = ['0:零', '1:一', '2:二', '3:三', '4:四', '5:五', '6:六', '7:七',
                         '8:八', '9:九', '10:十', '11:百', '12:千', '13:万', '14:亿']

    fig, ax = plt.subplots(figsize=(10.2, 7.0), facecolor='white')
    ax.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    offsets = {
        10: (-0.45, -0.32), 12: (0.45, 0.38), 14: (-0.38, 0.35),
        7: (0.40, 0.14), 9: (-0.10, -0.48), 0: (-0.45, -0.06),
        5: (-0.12, 0.42), 3: (0.16, -0.40), 11: (0.42, -0.28),
    }

    for c in range(15):
        mask = (y == c)
        ax.scatter(X_lda_2d[mask, 0], X_lda_2d[mask, 1], color=palette_15[c],
                   alpha=0.56, s=24, edgecolors='none', label=class_names_short[c])
    for c in range(15):
        mask = (y == c)
        cx, cy = np.mean(X_lda_2d[mask, 0]), np.mean(X_lda_2d[mask, 1])
        ox, oy = offsets.get(c, (0.0, 0.0))
        if ox != 0.0 or oy != 0.0:
            ax.plot([cx, cx + ox], [cy, cy + oy], color=DEEP_NAVY, lw=1.4, zorder=4)
        ax.scatter([cx + ox], [cy + oy], color=palette_15[c], edgecolors='white',
                   s=290, linewidth=2.5, zorder=5)
        ax.scatter([cx + ox], [cy + oy], facecolors='none', edgecolors=DEEP_NAVY,
                   s=320, linewidth=1.1, zorder=5)
        ax.text(cx + ox, cy + oy, str(c), ha='center', va='center', fontsize=10.5,
                fontweight='bold', color='white', zorder=6)

    ax.set_xlabel('Fungsi Diskriminan Kanonikal 1 (LD1)', fontsize=13.5, fontweight='bold', color=DEEP_NAVY)
    ax.set_ylabel('Fungsi Diskriminan Kanonikal 2 (LD2)', fontsize=13.5, fontweight='bold', color=DEEP_NAVY)
    ax.set_title('Proyeksi Kanonikal LDA 2D (PCA 100-D, 15 Kelas)', fontsize=15.5, fontweight='bold', color=DEEP_NAVY, pad=12)
    ax.tick_params(axis='both', labelsize=11.5)
    ax.legend(loc='upper right', ncol=3, frameon=True, facecolor='white', edgecolor='#CBD5E1',
              fontsize=10.2, markerscale=1.8, columnspacing=0.8, handletextpad=0.35)
    ax.text(0.02, 0.03, 'Kluster 2-D bertumpuk rapat -> batas kuadratik QDA (96,09%) >> LDA (85,13%)',
            transform=ax.transAxes, fontsize=11.0, fontweight='bold', color=DEEP_NAVY,
            bbox=dict(boxstyle='round,pad=0.38', fc='#EAF0FC', ec=ROYAL_BLUE, lw=1.5))
    fig.tight_layout()
    lda_plot_path = os.path.join(output_dir, "lda_canonical_projection.png")
    plt.savefig(lda_plot_path, dpi=300)
    plt.close()
    print(f"Saved 2D LDA Canonical Projection to: {lda_plot_path}")
    
    # Save Benchmark Results JSON (preserving clean classifier schema)
    json_path = os.path.join(output_dir, "multivariate_benchmark_results.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved Benchmark Results to: {json_path}")
    
    return results

def main():
    X, y, writers, folds, df = load_dataset()
    pca, pca_summary = run_pca_analysis(X)
    
    # Run MANOVA & Box's M on top 20 PCA components (plus Box's M on 40 PCA used by QDA)
    X_centered = X - np.mean(X, axis=0)
    X_pca_all = pca.transform(X_centered)
    manova_stats = run_manova_and_diagnostics(X_pca_all[:, :20], y)
    box_m_40 = compute_box_m(X_pca_all[:, :40], y)
    print(f"Box's M (m=40 for QDA): M={box_m_40['box_m']:,.2f}, chi2={box_m_40['chi2_approx']:,.2f}, df={box_m_40['df']}, p={box_m_40['p_value']:.2e}")
    
    # Run 5-Fold Cross Validation Models
    benchmark_results = run_cross_validation_models(X, y, folds)
    
    diag_summary = {
        "pca": pca_summary,
        "manova_m20": manova_stats,
        "box_m_m40": box_m_40,
        "cv_models": benchmark_results,
    }
    diag_path = os.path.join(str(OUTPUTS_DIR), "multivariate_diagnostics_summary.json")
    with open(diag_path, 'w') as f:
        json.dump(diag_summary, f, indent=2)
    print(f"Saved Complete Multivariate Diagnostics Summary to: {diag_path}")
    
    print("\n" + "=" * 60)
    print("MULTIVARIATE DATA ANALYSIS PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    main()
