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

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']

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
    kaiser_count = np.sum(eigenvalues > mean_eigenval)
    
    cpve_80 = np.argmax(cumulative_var >= 0.80) + 1
    cpve_85 = np.argmax(cumulative_var >= 0.85) + 1
    cpve_90 = np.argmax(cumulative_var >= 0.90) + 1
    cpve_95 = np.argmax(cumulative_var >= 0.95) + 1
    
    print(f"Mean eigenvalue (Kaiser threshold): {mean_eigenval:.6f}")
    print(f"Components exceeding Kaiser criterion: {kaiser_count}")
    print(f"Components for 80% variance: {cpve_80}")
    print(f"Components for 85% variance: {cpve_85}")
    print(f"Components for 90% variance: {cpve_90}")
    print(f"Components for 95% variance: {cpve_95}")
    
    # 1. Scree Plot
    fig, ax1 = plt.subplots(figsize=(10, 6))
    x_axis = np.arange(1, len(explained_var) + 1)
    
    color = 'tab:blue'
    ax1.set_xlabel('Principal Component Index', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Individual Explained Variance Ratio', color=color, fontsize=12, fontweight='bold')
    ax1.plot(x_axis[:100], explained_var[:100], color=color, marker='o', markersize=3, label='Individual Variance')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Cumulative Explained Variance (CPVE)', color=color, fontsize=12, fontweight='bold')
    ax2.plot(x_axis[:100], cumulative_var[:100], color=color, linewidth=2, label='Cumulative Variance')
    ax2.axhline(0.85, color='gray', linestyle='--', alpha=0.7, label='85% Threshold')
    ax2.axhline(0.90, color='black', linestyle=':', alpha=0.7, label='90% Threshold')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Scree Plot & Cumulative Proportion of Variance Explained (CPVE)', fontsize=14, fontweight='bold', pad=15)
    fig.tight_layout()
    scree_path = os.path.join(output_dir, "scree_plot.png")
    plt.savefig(scree_path, dpi=300)
    plt.close()
    print(f"Saved Scree Plot to: {scree_path}")
    
    # 2. Visualizing Eigen-Characters (Top 16 Eigenvectors)
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    fig.suptitle('Top 16 Eigen-Characters (Reshaped Eigenvectors)', fontsize=14, fontweight='bold')
    for idx, ax in enumerate(axes.flat):
        eigen_img = pca.components_[idx].reshape(64, 64)
        im = ax.imshow(eigen_img, cmap='seismic', interpolation='nearest')
        ax.set_title(f"PC{idx+1} ({explained_var[idx]*100:.2f}%)", fontsize=10)
        ax.axis('off')
    plt.tight_layout()
    eigen_path = os.path.join(output_dir, "eigen_characters.png")
    plt.savefig(eigen_path, dpi=300)
    plt.close()
    print(f"Saved Eigen-Characters Grid to: {eigen_path}")
    
    return pca, cpve_85, cpve_90

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
    
    print(f"MANOVA Statistics on Top {m} PCA Components:")
    print(f"  • Wilks' Lambda (Λ)       : {wilks_lambda:.6e} (Strong separation as Λ -> 0)")
    print(f"  • Pillai's Trace (V)      : {pillai_trace:.4f}")
    print(f"  • Hotelling-Lawley (T)   : {hotelling_trace:.4f}")
    print(f"  • Roy's Largest Root (θ) : {roys_root:.4f}")
    print(f"  • Partial Eta Squared    : {eta_sq:.4f} (> 0.14 indicates very large effect)")
    
    # Chi-Square Q-Q Plot of Mahalanobis Distances
    print("\nComputing Mahalanobis Distances for Multivariate Normality Check...")
    S_cov = np.cov(X_pca, rowvar=False)
    S_inv = np.linalg.pinv(S_cov)
    diff = X_pca - grand_mean
    d2 = np.sum((diff @ S_inv) * diff, axis=1)
    
    # Q-Q plot against Chi-Square with m degrees of freedom
    d2_sorted = np.sort(d2)
    quantiles = stats.chi2.ppf((np.arange(1, N + 1) - 0.5) / N, df=m)
    
    plt.figure(figsize=(7, 7))
    plt.scatter(quantiles, d2_sorted, alpha=0.3, s=10, color='darkblue', label='Data Observations')
    max_val = max(np.max(quantiles), np.max(d2_sorted[:int(N * 0.99)]))
    plt.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Theoretical χ² Reference')
    plt.xlabel(f'Theoretical χ² Quantiles (df={m})', fontsize=12)
    plt.ylabel('Ordered Squared Mahalanobis Distances (d²)', fontsize=12)
    plt.title('Multivariate Normality Diagnostic: Chi-Square Q-Q Plot', fontsize=13, fontweight='bold')
    plt.legend()
    plt.tight_layout()
    qq_path = os.path.join(output_dir, "mahalanobis_qq_plot.png")
    plt.savefig(qq_path, dpi=300)
    plt.close()
    print(f"Saved Mahalanobis Q-Q Plot to: {qq_path}")
    
    return {
        "wilks_lambda": float(wilks_lambda),
        "pillai_trace": float(pillai_trace),
        "hotelling_trace": float(hotelling_trace),
        "roys_root": float(roys_root),
        "partial_eta_squared": float(eta_sq)
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
        
    # Generate Canonical LDA Projection Plot
    print("\nGenerating Canonical Discriminant Analysis 2D Projection...")
    pca = PCA(n_components=100, random_state=42)
    X_pca = pca.fit_transform(X)
    lda = LinearDiscriminantAnalysis(n_components=2)
    X_lda_2d = lda.fit_transform(X_pca, y)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_lda_2d[:, 0], X_lda_2d[:, 1], c=y, cmap='tab20', alpha=0.6, s=15)
    plt.colorbar(scatter, label='Class Label (0-14)')
    plt.xlabel('First Canonical Discriminant Function (LD1)', fontsize=12, fontweight='bold')
    plt.ylabel('Second Canonical Discriminant Function (LD2)', fontsize=12, fontweight='bold')
    plt.title('2D Canonical Linear Discriminant Analysis (LDA) Projection', fontsize=14, fontweight='bold')
    plt.tight_layout()
    lda_plot_path = os.path.join(output_dir, "lda_canonical_projection.png")
    plt.savefig(lda_plot_path, dpi=300)
    plt.close()
    print(f"Saved 2D LDA Canonical Projection to: {lda_plot_path}")
    
    # Save Benchmark Results JSON
    json_path = os.path.join(output_dir, "multivariate_benchmark_results.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved Benchmark Results to: {json_path}")
    
    return results

def main():
    X, y, writers, folds, df = load_dataset()
    pca, cpve_85, cpve_90 = run_pca_analysis(X)
    
    # Run MANOVA on the top 20 PCA components (compact representation)
    X_pca_20 = pca.transform(X - np.mean(X, axis=0))[:, :20]
    manova_stats = run_manova_and_diagnostics(X_pca_20, y)
    
    # Run 5-Fold Cross Validation Models
    benchmark_results = run_cross_validation_models(X, y, folds)
    
    print("\n" + "=" * 60)
    print("MULTIVARIATE DATA ANALYSIS PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    main()
