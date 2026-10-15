import os
import glob
import zipfile
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from FinalProject.src.models import ResNet18_CBAM, CompactCNN

def load_and_preprocess_img(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        img = np.zeros((64, 64), dtype=np.uint8)
    if img.shape != (64, 64):
        img = cv2.resize(img, (64, 64))
    tensor = torch.from_numpy(img).float() / 255.0
    tensor = (tensor - 0.5) / 0.5
    return tensor.unsqueeze(0).unsqueeze(0)  # Shape (1, 1, 64, 64)

def tta_variants(tensor):
    """
    Generate 5 safe TTA variants:
    1. Original
    2. Rotation +4 deg
    3. Rotation -4 deg
    4. Translation dx=+2
    5. Translation dx=-2
    """
    variants = [tensor]
    b, c, h, w = tensor.shape
    device = tensor.device

    # Rot +4 deg
    theta_pos = torch.tensor([[np.cos(np.radians(4)), -np.sin(np.radians(4)), 0],
                              [np.sin(np.radians(4)),  np.cos(np.radians(4)), 0]], dtype=torch.float, device=device).unsqueeze(0)
    grid_pos = F.affine_grid(theta_pos, tensor.size(), align_corners=False)
    variants.append(F.grid_sample(tensor, grid_pos, align_corners=False))

    # Rot -4 deg
    theta_neg = torch.tensor([[np.cos(np.radians(-4)), -np.sin(np.radians(-4)), 0],
                              [np.sin(np.radians(-4)),  np.cos(np.radians(-4)), 0]], dtype=torch.float, device=device).unsqueeze(0)
    grid_neg = F.affine_grid(theta_neg, tensor.size(), align_corners=False)
    variants.append(F.grid_sample(tensor, grid_neg, align_corners=False))

    # Shift dx=+2 (approx 2/32 = 0.0625)
    theta_shift_r = torch.tensor([[1, 0, 0.06], [0, 1, 0]], dtype=torch.float, device=device).unsqueeze(0)
    grid_sr = F.affine_grid(theta_shift_r, tensor.size(), align_corners=False)
    variants.append(F.grid_sample(tensor, grid_sr, align_corners=False))

    # Shift dx=-2
    theta_shift_l = torch.tensor([[1, 0, -0.06], [0, 1, 0]], dtype=torch.float, device=device).unsqueeze(0)
    grid_sl = F.affine_grid(theta_shift_l, tensor.size(), align_corners=False)
    variants.append(F.grid_sample(tensor, grid_sl, align_corners=False))

    return variants

def apply_adabn(model, img_files, device, batch_size=64):
    """
    Adaptive Batch Normalization (AdaBN, Li et al., Pattern Recognition 2018).
    Updates running mean and running variance on unlabeled test set without modifying weights.
    """
    print(f"[*] Applying AdaBN Domain Adaptation on {len(img_files)} test images...")
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.reset_running_stats()
            m.training = True
            m.momentum = None  # Exact cumulative empirical dataset mean & variance

    with torch.no_grad():
        for i in range(0, len(img_files), batch_size):
            batch_paths = img_files[i:i + batch_size]
            tensors = [load_and_preprocess_img(p) for p in batch_paths]
            batch_tensor = torch.cat(tensors, dim=0).to(device)
            _ = model(batch_tensor)

    model.eval()
    print("[+] AdaBN adaptation complete.")

def generate_predictions(
    test_dir,
    ckpt_paths,
    output_csv="FinalProject/predictions.csv",
    output_zip="FinalProject/predictions.zip",
    use_tta=True,
    use_adabn=True,
    power_p=1.8,
    device="cuda",
    save_probs_path=None
):
    print("=" * 70)
    print("GENERATING SOTA SUBMISSION PREDICTIONS FOR CODABENCH")
    print(f"Test Directory : {test_dir}")
    print(f"Models Count   : {len(ckpt_paths)}")
    print(f"AdaBN Enabled  : {use_adabn}")
    print(f"TTA Enabled    : {use_tta} (Power-Mean p={power_p})")
    print(f"Device         : {device}")
    print("=" * 70)

    # Gather test images (support .png, .jpg, .jpeg)
    img_files = []
    for ext in ["*.png", "*.PNG", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG"]:
        img_files.extend(glob.glob(os.path.join(test_dir, ext)))
        img_files.extend(glob.glob(os.path.join(test_dir, "**", ext), recursive=True))
    img_files = sorted(list(set(img_files)))

    if not img_files:
        raise ValueError(f"No image files found in {test_dir}!")
    print(f"Found {len(img_files)} test images in target directory.")

    # Load and adapt models
    models = []
    for ckpt_p in ckpt_paths:
        ckpt = torch.load(ckpt_p, map_location=device)
        model_type = ckpt.get("model_type", "resnet18_cbam")
        if model_type == "resnet18_cbam":
            m = ResNet18_CBAM(in_channels=1, num_classes=15, feature_dim=256, use_mixstyle=False)
        else:
            m = CompactCNN(in_channels=1, num_classes=15)
        m.load_state_dict(ckpt["model_state_dict"])
        m.to(device)
        m.eval()

        if use_adabn:
            apply_adabn(m, img_files, device=device)

        models.append(m)
        val_acc = ckpt.get('val_acc', 0.0) * 100
        print(f"Loaded & Ready: {os.path.basename(ckpt_p)} (OOF Val Acc: {val_acc:.2f}%)")

    # Weights for TTA variants: [Original, Rot+4, Rot-4, Shift+2, Shift-2]
    tta_weights = [1.0, 0.8, 0.8, 0.7, 0.7]

    results = []
    all_final_probs = []
    print("\nRunning Ensemble Inference across test set...")
    with torch.no_grad():
        for i, fpath in enumerate(img_files):
            basename = os.path.splitext(os.path.basename(fpath))[0]
            inp = load_and_preprocess_img(fpath).to(device)

            if use_tta:
                variants = tta_variants(inp)
            else:
                variants = [inp]
                tta_weights = [1.0]

            # Accumulate across models and TTA variants
            model_probs_list = []
            for m in models:
                var_probs_list = []
                for v_idx, v in enumerate(variants):
                    logits, _ = m(v)
                    p = F.softmax(logits, dim=1).squeeze(0)  # (15,)
                    var_probs_list.append(p * tta_weights[v_idx])
                
                # Temperature Power Mean across TTA variants for this model
                var_probs_stack = torch.stack(var_probs_list, dim=0)  # (K, 15)
                model_power_mean = torch.pow(torch.sum(torch.pow(var_probs_stack, power_p), dim=0), 1.0 / power_p)
                model_power_mean = model_power_mean / torch.sum(model_power_mean)
                model_probs_list.append(model_power_mean)

            # Consensus across 5 models
            final_prob = torch.mean(torch.stack(model_probs_list, dim=0), dim=0)
            pred_label = int(torch.argmax(final_prob).item())
            results.append({"id": basename, "label": pred_label})
            all_final_probs.append(final_prob.cpu().numpy())

            if (i + 1) % 500 == 0 or (i + 1) == len(img_files):
                print(f"  Processed {i + 1}/{len(img_files)} images ({((i + 1)/len(img_files))*100:.1f}%)...")

    df_preds = pd.DataFrame(results)
    probs_arr = np.array(all_final_probs)
    if save_probs_path:
        os.makedirs(os.path.dirname(save_probs_path) or ".", exist_ok=True)
        np.save(save_probs_path, probs_arr)
        print(f"[+] Saved prediction probability array to: {save_probs_path}")

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    df_preds.to_csv(output_csv, index=False)
    print(f"\n[+] Saved predictions to: {output_csv}")

    # ==============================================================================
    # PRE-FLIGHT SANITY CHECKS FOR CODABENCH COMPLIANCE
    # ==============================================================================
    print("\n" + "=" * 70)
    print("RUNNING PRE-FLIGHT VERIFICATION CHECKS (CODABENCH AUDIT)")
    print("=" * 70)

    # 1. Header check
    assert list(df_preds.columns) == ["id", "label"], f"Invalid columns! Found: {df_preds.columns}"
    print("[PASS] Columns match required schema: ['id', 'label']")

    # 2. Null check
    assert df_preds["label"].isnull().sum() == 0, "CRITICAL: Found null values in predictions!"
    print("[PASS] Zero null / NaN values found.")

    # 3. Label range check
    assert df_preds["label"].min() >= 0 and df_preds["label"].max() <= 14, "CRITICAL: Label out of [0, 14]!"
    assert df_preds["label"].dtype in [int, np.int64, np.int32], "CRITICAL: Label is not integer!"
    print(f"[PASS] Integer data type verified. Range: [{df_preds['label'].min()}, {df_preds['label'].max()}].")

    # 4. Count check
    assert len(df_preds) == len(img_files), f"Count mismatch! {len(df_preds)} vs {len(img_files)}"
    print(f"[PASS] Row count matches image count: {len(df_preds)} samples.")

    # 5. Class distribution sanity profiling
    print("\n[PROFILING] Predicted Class Distribution:")
    val_counts = df_preds["label"].value_counts().sort_index()
    for cls in range(15):
        cnt = val_counts.get(cls, 0)
        pct = (cnt / len(df_preds)) * 100
        print(f"  • Class {cls:2d}: {cnt:4d} samples ({pct:5.2f}%)")

    # Warn if any class is 0 (referencing committee note)
    absent_classes = [c for c in range(15) if val_counts.get(c, 0) == 0]
    if absent_classes:
        print(f"[NOTE] Classes absent in test predictions: {absent_classes} (Consistent with committee note on unbalanced test set).")
    else:
        print("[NOTE] All 15 classes appear in test predictions.")

    # 6. Create flat root ZIP archive (CodaBench requires exactly 'predictions.csv' at zip root)
    os.makedirs(os.path.dirname(output_zip) or ".", exist_ok=True)
    with zipfile.ZipFile(output_zip, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(output_csv, arcname="predictions.csv")
    print(f"\n[+] Created flat root ZIP archive: {output_zip} ({os.path.getsize(output_zip):,} bytes)")

    # 7. Verify ZIP file internal hierarchy
    with zipfile.ZipFile(output_zip, 'r') as z:
        files = z.namelist()
        assert "predictions.csv" in files, "CRITICAL: 'predictions.csv' missing at zip root!"
        assert len(files) == 1, f"CRITICAL: Extraneous files found in zip: {files}"
    print("[PASS] Verified ZIP archive: 'predictions.csv' is located directly at root without subfolders.")
    print("=" * 70)
    print("SUBMISSION FILE 100% AUDITED, VERIFIED, AND READY FOR CODABENCH!")
    print("=" * 70)
    return df_preds, probs_arr

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate CodaBench Submission with AdaBN & 5-Way TTA")
    default_ckpts = sorted(glob.glob("FinalProject/models/kaggle_outputs/models/*.pt"))
    if not default_ckpts:
        default_ckpts = sorted(glob.glob("FinalProject/models/*.pt"))

    parser.add_argument("--test_dir", type=str, required=True, help="Path to test images directory")
    parser.add_argument("--ckpts", nargs="+", default=default_ckpts, help="List of checkpoint paths")
    parser.add_argument("--output_csv", type=str, default="FinalProject/predictions.csv", help="Path to output CSV")
    parser.add_argument("--output_zip", type=str, default="FinalProject/predictions.zip", help="Path to output ZIP")
    parser.add_argument("--no_tta", action="store_true", help="Disable Test-Time Augmentation")
    parser.add_argument("--no_adabn", action="store_true", help="Disable AdaBN Domain Adaptation")
    parser.add_argument("--power_p", type=float, default=1.8, help="Temperature power p for TTA consensus")

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    generate_predictions(
        test_dir=args.test_dir,
        ckpt_paths=args.ckpts,
        output_csv=args.output_csv,
        output_zip=args.output_zip,
        use_tta=not args.no_tta,
        use_adabn=not args.no_adabn,
        power_p=args.power_p,
        device=device
    )
