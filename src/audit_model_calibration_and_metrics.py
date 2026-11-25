#!/usr/bin/env python3
"""
================================================================================
Academic Rigor Audit: Confidence Calibration, ECE, Macro-F1, and Wilson CI
================================================================================
Evaluates the Distilled Student Model (checkpoint_student_fp16.pt) on Phase 1 Validation:
1. Multi-Class Metrics: Top-1 Accuracy, Macro-F1, Per-Class Precision/Recall
2. Sub-Domain Disaggregation: Chinese MNIST (Kaishu) vs Gaya Baru (Caoshu)
3. Confidence Calibration: Expected Calibration Error (ECE, 10 bins), Brier Score, NLL
4. Statistical Uncertainty: Wilson Score 95% Confidence Intervals
5. Worst-Group Accuracy & Confusion Topology
================================================================================
"""

import math
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix, log_loss

from src.evaluate_student_vs_ensemble import (
    StudentDenseNet121_CBAM,
    StudentSubCenterArcFace,
    BenchmarkDataset,
    DEVICE
)

def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10):
    """
    Computes Expected Calibration Error (ECE) (Guo et al., ICML 2017).
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_details = []

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper) if i > 0 else (confidences >= bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            bin_details.append({
                'bin': f"[{bin_lower:.1f}, {bin_upper:.1f}]",
                'count': int(np.sum(in_bin)),
                'acc': float(accuracy_in_bin),
                'conf': float(avg_confidence_in_bin)
            })

    return ece, bin_details

def wilson_score_interval(successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    p = successes / total
    z = 1.95996  # 95% confidence
    denominator = 1 + z**2 / total
    centre_adjusted_probability = (p + z**2 / (2 * total)) / denominator
    adjusted_standard_error = z * math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denominator
    lower = max(0.0, centre_adjusted_probability - adjusted_standard_error)
    upper = min(1.0, centre_adjusted_probability + adjusted_standard_error)
    return lower * 100.0, upper * 100.0

def main():
    manifest_csv = "FinalProject/data/metadata/super_dataset_15918.csv"
    df_all = pd.read_csv(manifest_csv)
    df_val = df_all[df_all['source'] == 'validation_phase1'].reset_index(drop=True)

    print(f"[*] Auditing Distilled Student on {len(df_val)} Validation Samples...")

    loader = DataLoader(BenchmarkDataset(df_val), batch_size=64, shuffle=False)

    ckpt = torch.load("FinalProject/models/densenet121_cbam_student_fp16.pt", map_location=DEVICE, weights_only=False)
    m = StudentDenseNet121_CBAM(num_classes=15).half().to(DEVICE).eval()
    h = StudentSubCenterArcFace(in_features=256, num_classes=15, sub_centers=2).half().to(DEVICE).eval()

    m.load_state_dict(ckpt['model_state_dict'])
    h.load_state_dict(ckpt['metric_head_state_dict'])

    all_probs = []
    with torch.no_grad():
        for x, _, _ in loader:
            x = x.half().to(DEVICE)
            lin, f = m(x)
            arc = h(f)
            p = F.softmax(0.5 * lin + 0.5 * arc, dim=1).cpu().numpy()
            all_probs.append(p)

    probs = np.concatenate(all_probs, axis=0)
    preds = np.argmax(probs, axis=1)
    labels = df_val['label'].values
    strata = df_val['stratum'].values

    # 1. Global Metrics
    correct = np.sum(preds == labels)
    n_samples = len(labels)
    acc = correct / n_samples * 100.0
    macro_f1 = f1_score(labels, preds, average='macro') * 100.0
    weighted_f1 = f1_score(labels, preds, average='weighted') * 100.0

    lower_ci, upper_ci = wilson_score_interval(correct, n_samples)

    # 2. Calibration Metrics
    ece, bin_details = compute_ece(probs, labels, n_bins=10)
    one_hot = np.eye(15)[labels]
    brier_score = np.mean(np.sum((probs - one_hot) ** 2, axis=1))
    nll = log_loss(labels, probs, labels=list(range(15)))

    # 3. Disaggregated Metrics
    mask_cm = (strata == 'val_chinese_mnist')
    mask_gb = (strata == 'val_gaya_baru')

    acc_cm = np.mean(preds[mask_cm] == labels[mask_cm]) * 100.0
    acc_gb = np.mean(preds[mask_gb] == labels[mask_gb]) * 100.0
    f1_cm = f1_score(labels[mask_cm], preds[mask_cm], average='macro') * 100.0
    f1_gb = f1_score(labels[mask_gb], preds[mask_gb], average='macro') * 100.0

    # 4. Per-Class Analysis & Worst-Group
    precision, recall, f1, support = precision_recall_fscore_support(labels, preds, labels=list(range(15)), zero_division=0)
    class_acc = []
    for c in range(15):
        m_c = (labels == c)
        c_acc = np.mean(preds[m_c] == labels[m_c]) * 100.0 if np.sum(m_c) > 0 else 100.0
        class_acc.append(c_acc)

    worst_class_idx = int(np.argmin(class_acc))
    worst_class_acc = class_acc[worst_class_idx]

    # Print Formal Academic Report
    print("\n" + "=" * 80)
    print("🎓 FORMAL ACADEMIC AUDIT: DISTILLED STUDENT MODEL RIGOR REPORT")
    print("=" * 80)
    print(f"1. OVERALL ACCURACY & UNCERTAINTY BOUNDS:")
    print(f"   • Top-1 Accuracy:            {acc:.4f}% ({correct} / {n_samples})")
    print(f"   • Wilson 95% Confidence Int: [{lower_ci:.2f}%, {upper_ci:.2f}%]")
    print(f"   • Macro-F1 Score:            {macro_f1:.4f}%")
    print(f"   • Weighted-F1 Score:         {weighted_f1:.4f}%")
    print("-" * 80)
    print(f"2. DISAGGREGATED DOMAIN PERFORMANCE:")
    print(f"   • Chinese MNIST Formal (1500): Accuracy = {acc_cm:.2f}% | Macro-F1 = {f1_cm:.2f}%")
    print(f"   • Gaya Baru Cursive (966):    Accuracy = {acc_gb:.2f}% | Macro-F1 = {f1_gb:.2f}%")
    print("-" * 80)
    print(f"3. CONFIDENCE CALIBRATION & PROBABILISTIC RELIABILITY:")
    print(f"   • Expected Calibration Error: {ece * 100:.3f}% (ECE < 1% indicates near-perfect calibration)")
    print(f"   • Brier Multi-Class Score:   {brier_score:.6f} (Optimal -> 0.000)")
    print(f"   • Negative Log-Likelihood:   {nll:.6f}")
    print("-" * 80)
    print(f"4. WORST-GROUP SENSITIVITY AUDIT:")
    print(f"   • Lowest Class Accuracy:     Class {worst_class_idx} ({worst_class_acc:.2f}%)")
    print(f"   • Mean Class Accuracy:       {np.mean(class_acc):.2f}%")
    print("=" * 80)

if __name__ == "__main__":
    main()
