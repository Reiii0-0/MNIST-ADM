#!/usr/bin/env python3
"""
Master Training Orchestrator for Tri-Orthogonal SOTA v2 Ensemble (Taktik A).
Trains 5-Fold Stratified Group K-Fold models with the expanded 15,100 sample dataset
(authentic Kaishu + 7 Google Fonts Caoshu/Xingshu/Maobi + Targeted Biomechanical Perturbations).

Models:
1. ResNet-18 CBAM (Batch size: 64, Epochs: 15)
2. ConvNeXt-Femto (Batch size: 64, Epochs: 15)
3. DenseNet-121 CBAM (Batch size: 32, Epochs: 15)
"""

import sys
import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.train_biomechanical_model import train_biomechanical_fold
from src.config import get_splits_kfold_csv, MODELS_DIR

def run_training_pipeline(models=None, folds=None, epochs=15):
    if models is None:
        models = ["resnet18_cbam", "convnext_femto", "densenet121_cbam"]
    if folds is None:
        folds = list(range(5))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    df_splits = pd.read_csv(get_splits_kfold_csv())

    print("==================================================================")
    print("🏆 MASTER TRAINING PIPELINE: SOTA V2 TRI-ORTHOGONAL ENSEMBLE")
    print(f"Target Models: {models}")
    print(f"Folds: {folds}")
    print(f"Epochs per fold: {epochs}")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("==================================================================")

    results = {}

    for model_name in models:
        batch_size = 32 if "densenet" in model_name else 64
        lr = 5e-4
        suffix = "_v2"
        oof_path = MODELS_DIR / f"oof_probabilities_{model_name}{suffix}.npy"
        
        if oof_path.exists():
            oof_probs = np.load(oof_path)
        else:
            oof_probs = np.zeros((len(df_splits), 15), dtype=np.float32)

        model_scores = []
        print(f"\n▶ Starting Model: {model_name}{suffix} (Batch Size: {batch_size}, LR: {lr})")

        for f in folds:
            acc, val_probs, val_indices = train_biomechanical_fold(
                fold=f,
                model_type=model_name,
                suffix=suffix,
                use_synthetic=True,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                device=device
            )
            model_scores.append(acc)
            oof_probs[val_indices] = val_probs
            np.save(oof_path, oof_probs)

        mean_acc = np.mean(model_scores)
        results[model_name] = {
            "scores": model_scores,
            "mean_acc": mean_acc,
            "oof_path": str(oof_path)
        }
        print(f"✅ {model_name}{suffix} Complete! Mean Val Acc across folds {folds}: {mean_acc*100:.2f}%")

    print("\n==================================================================")
    print("📊 ALL TRAINING RUNS COMPLETE SUMMARY:")
    for m, res in results.items():
        print(f"  • {m:20s}: {res['mean_acc']*100:.2f}% (Folds: {[round(s*100, 2) for s in res['scores']]})")
    print("==================================================================")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["resnet18_cbam", "convnext_femto", "densenet121_cbam"],
                        help="List of models to train")
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4],
                        help="List of folds to train")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs per fold")
    args = parser.parse_args()

    run_training_pipeline(models=args.models, folds=args.folds, epochs=args.epochs)
