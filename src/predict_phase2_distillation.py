#!/usr/bin/env python3
"""
================================================================================
Phase 2 / Unseen Test Inference Pipeline: Distilled Student Model (Sub 32)
================================================================================
Runs pure inductive inference using the Kaggle-distilled Student checkpoint
(DenseNet-121 CBAM + SubCenterArcFace K=2 + MixStyle + SWA, 14.75 MB FP16)
and applies Unsupervised Test-Time Prior Adaptation (TTPA via EM, Park ICCV 2023).

Supports:
- Direct directory input (--input-dir containing test/ & sample_submission.csv)
- Zip archive input (--input-zip such as data_test.zip)
- Exact CodaBench schema verification (header ['id', 'label'], ID stem/format
  matched to sample_submission.csv, zero NaNs, flat zip root predictions.csv).
================================================================================
"""

import argparse
import os
import sys
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.evaluate_student_vs_ensemble import (
    DEVICE,
    StudentDenseNet121_CBAM,
    StudentSubCenterArcFace,
)
from src.test_time_prior_adaptation import TestTimePriorAdapter


DEFAULT_CHECKPOINT = BASE_DIR / "models" / "densenet121_cbam_student_fp16.pt"
CLASS_NAMES = [
    "0 零", "1 一", "2 二", "3 三", "4 四", "5 五", "6 六", "7 七",
    "8 八", "9 九", "10 十", "11 百", "12 千", "13 万", "14 亿",
]


class UnlabeledImageDataset(Dataset):
    """Loads 64x64 grayscale images with exact training normalization [-1, 1]."""

    def __init__(self, image_paths: List[Path]):
        self.image_paths = image_paths

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = cv2.imread(str(self.image_paths[idx]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {self.image_paths[idx]}")
        if img.shape != (64, 64):
            img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(img.astype(np.float32) / 127.5 - 1.0).unsqueeze(0)


def resolve_test_inputs(
    input_dir: Optional[Path],
    input_zip: Optional[Path],
    extract_root: Path,
) -> Tuple[List[str], List[Path], bool]:
    """Resolves ordered sample IDs, image paths, and whether sample_submission was used."""
    work_dir = input_dir
    if input_zip is not None:
        if not input_zip.exists():
            raise FileNotFoundError(f"Test zip not found: {input_zip}")
        work_dir = extract_root / input_zip.stem
        if not work_dir.exists() or not any(work_dir.rglob("*.png")):
            work_dir.mkdir(parents=True, exist_ok=True)
            print(f"[*] Extracting {input_zip} -> {work_dir}...")
            with zipfile.ZipFile(input_zip, "r") as zf:
                zf.extractall(work_dir)

    if work_dir is None or not work_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {work_dir}")

    # Index all images recursively by basename and stem for exact lookup
    by_name: Dict[str, Path] = {}
    by_stem: Dict[str, Path] = {}
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        for p in sorted(work_dir.rglob(ext)):
            by_name[p.name] = p
            by_stem[p.stem] = p

    if not by_name:
        raise FileNotFoundError(f"No image files (.png/.jpg) found under {work_dir}")

    sample_csvs = sorted(work_dir.rglob("sample_submission.csv"))
    if sample_csvs:
        df_sample = pd.read_csv(sample_csvs[0])
        if "id" not in df_sample.columns:
            raise ValueError(f"Missing 'id' column in {sample_csvs[0]}")
        ids = [str(x) for x in df_sample["id"].tolist()]
        paths: List[Path] = []
        for raw_id in ids:
            img_p = by_name.get(raw_id) or by_stem.get(Path(raw_id).stem)
            if img_p is None:
                raise FileNotFoundError(
                    f"Image for sample_submission id '{raw_id}' missing under {work_dir}"
                )
            paths.append(img_p)
        print(f"[+] Matched {len(paths)} images strictly against {sample_csvs[0]}")
        return ids, paths, True

    # Fallback: sorted image stems when sample_submission.csv is absent
    sorted_paths = sorted(by_name.values(), key=lambda p: p.name)
    ids = [p.stem for p in sorted_paths]
    print(f"[!] No sample_submission.csv found; using {len(ids)} sorted image stems.")
    return ids, sorted_paths, False


def load_student_model(
    ckpt_path: Path, device: torch.device, use_fp16: bool = False
) -> Tuple[StudentDenseNet121_CBAM, StudentSubCenterArcFace, np.ndarray]:
    """Loads the distilled Student model, metric head, and training class counts."""
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Student checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = StudentDenseNet121_CBAM(num_classes=15).to(device)
    head = StudentSubCenterArcFace(in_features=256, num_classes=15, sub_centers=2).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    head.load_state_dict(ckpt["metric_head_state_dict"])

    if use_fp16 and device.type == "cuda":
        model = model.half()
        head = head.half()
    else:
        model = model.float()
        head = head.float()

    model.eval()
    head.eval()

    source_counts = np.asarray(
        ckpt.get("source_counts", np.ones(15, dtype=np.float64)), dtype=np.float64
    )
    return model, head, source_counts


def predict_probabilities(
    model: StudentDenseNet121_CBAM,
    head: StudentSubCenterArcFace,
    loader: DataLoader,
    device: torch.device,
    use_fp16: bool = False,
) -> np.ndarray:
    """Computes pure inductive class probabilities across batches."""
    probs_batches: List[np.ndarray] = []
    with torch.no_grad():
        for x in loader:
            x = x.to(device)
            if use_fp16 and device.type == "cuda":
                x = x.half()
            s_lin, s_feats = model(x)
            s_arc = head(s_feats)
            logits = 0.5 * s_lin.float() + 0.5 * s_arc.float()
            probs = F.softmax(logits, dim=1).cpu().numpy()
            probs_batches.append(probs)
    return np.concatenate(probs_batches, axis=0)


def write_codabench_submission(
    ids: List[str], labels: np.ndarray, csv_path: Path, zip_path: Path
) -> None:
    """Exports CodaBench-compliant predictions.csv and flat-root zip archive."""
    assert len(ids) == len(labels), f"Length mismatch: {len(ids)} vs {len(labels)}"
    assert np.all((labels >= 0) & (labels <= 14)), "Predictions out of valid range [0, 14]"

    df_out = pd.DataFrame({"id": ids, "label": labels.astype(int)})
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    df_out.to_csv(csv_path, index=False)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path, arcname="predictions.csv")


def print_diagnostics(
    ids: List[str],
    probs_raw: np.ndarray,
    probs_ttpa: np.ndarray,
    adapter: TestTimePriorAdapter,
    elapsed_sec: float,
) -> None:
    """Prints class distribution, TTPA shift diagnostics, and low-confidence review list."""
    preds_raw = np.argmax(probs_raw, axis=1)
    preds_ttpa = np.argmax(probs_ttpa, axis=1)
    n_samples = len(ids)
    n_diff = int(np.sum(preds_raw != preds_ttpa))

    conf_raw = np.max(probs_raw, axis=1)
    sorted_p = np.sort(probs_raw, axis=1)
    margins = sorted_p[:, -1] - sorted_p[:, -2]

    print("\n" + "=" * 86)
    print(f"📊 STUDENT DISTILLATION INFERENCE SUMMARY ({n_samples} samples | {elapsed_sec:.2f}s)")
    print("=" * 86)
    print(f"• Mean Top-1 Confidence (Raw): {np.mean(conf_raw)*100:.2f}% | <0.80: {np.sum(conf_raw < 0.80)}")
    print(f"• Mean Top1-Top2 Margin (Raw): {np.mean(margins):.4f} | <0.50: {np.sum(margins < 0.50)}")
    print(f"• TTPA Converged: {adapter.converged_} ({adapter.n_iter_} iters) | Raw vs TTPA diffs: {n_diff}")
    print("-" * 86)
    print(f"{'Class':<8} | {'Raw Count':>9} ({'%':>6}) | {'TTPA Count':>10} ({'%':>6}) | {'Est. Prior':>10}")
    print("-" * 86)
    raw_counts = np.bincount(preds_raw, minlength=15)
    ttpa_counts = np.bincount(preds_ttpa, minlength=15)
    est_prior = adapter.target_prior_ if adapter.target_prior_ is not None else np.zeros(15)
    for c in range(15):
        print(
            f"{CLASS_NAMES[c]:<8} | {raw_counts[c]:9d} ({raw_counts[c]/n_samples*100:5.2f}%) | "
            f"{ttpa_counts[c]:10d} ({ttpa_counts[c]/n_samples*100:5.2f}%) | {est_prior[c]*100:9.2f}%"
        )
    print("=" * 86)

    low_idx = np.argsort(conf_raw)[:10]
    print("\n🔎 Top 10 Lowest-Confidence Samples (for pre-submission sanity check):")
    for idx in low_idx:
        top2 = np.argsort(probs_raw[idx])[::-1][:2]
        print(
            f"  • {ids[idx]:<15} | Top1={CLASS_NAMES[top2[0]]} ({probs_raw[idx, top2[0]]:.3f}) "
            f"| Top2={CLASS_NAMES[top2[1]]} ({probs_raw[idx, top2[1]]:.3f}) "
            f"| TTPA={CLASS_NAMES[preds_ttpa[idx]]}"
        )


def auto_find_phase2_source() -> Tuple[Optional[Path], Optional[Path]]:
    """Checks common local folders for Phase 2 test archives or directories."""
    dir_candidates = [
        BASE_DIR / "data" / "data_test",
        BASE_DIR / "data" / "test",
        BASE_DIR / "data" / "fase2",
        BASE_DIR / "data" / "phase2",
    ]
    for d in dir_candidates:
        if d.exists() and any(d.rglob("*.png")):
            return d, None

    zip_candidates = [
        BASE_DIR / "data" / "archives" / "data_test.zip",
        BASE_DIR / "data" / "data_test.zip",
        Path.home() / "Downloads" / "data_test.zip",
        Path.home() / "Downloads" / "Archives" / "data_test.zip",
    ]
    for z in zip_candidates:
        if z.exists():
            return None, z
    return None, None


def main():
    parser = argparse.ArgumentParser(description="Run Distilled Student (Sub 32) inference on test data.")
    parser.add_argument("--input-dir", type=Path, default=None, help="Directory with test images (& optional sample_submission.csv)")
    parser.add_argument("--input-zip", type=Path, default=None, help="Zip archive (e.g. data_test.zip)")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="Student .pt checkpoint")
    parser.add_argument("--out-zip", type=Path, default=BASE_DIR / "submissions" / "sub_phase2_student_distillation.zip")
    parser.add_argument("--out-probs", type=Path, default=BASE_DIR / "submissions" / "probabilities_phase2_student_distillation.npy")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--no-ttpa", action="store_true", help="Export raw Student predictions instead of TTPA-adapted")
    parser.add_argument("--fp16", action="store_true", help="Run GPU inference in FP16 instead of FP32")
    args = parser.parse_args()

    input_dir, input_zip = args.input_dir, args.input_zip
    if input_dir is None and input_zip is None:
        input_dir, input_zip = auto_find_phase2_source()

    if input_dir is None and input_zip is None:
        print(
            "[!] Phase 2 test dataset (data_test.zip / test images) not found locally.\n"
            "    Checked: FinalProject/data/{data_test,test,fase2,phase2,archives/data_test.zip} "
            "and ~/Downloads/{,Archives/}data_test.zip.\n"
            "    Download data_test from CodaBench/Google Drive and re-run with --input-dir or --input-zip.",
            file=sys.stderr,
        )
        sys.exit(2)

    ids, image_paths, _ = resolve_test_inputs(
        input_dir=input_dir,
        input_zip=input_zip,
        extract_root=BASE_DIR / "data" / "_extracted_test",
    )

    dataset = UnlabeledImageDataset(image_paths)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    print(f"[*] Loading Student Distillation checkpoint: {args.checkpoint}")
    model, head, source_counts = load_student_model(args.checkpoint, DEVICE, use_fp16=args.fp16)

    t0 = time.perf_counter()
    probs_raw = predict_probabilities(model, head, loader, DEVICE, use_fp16=args.fp16)
    elapsed = time.perf_counter() - t0

    source_priors = source_counts / np.sum(source_counts)
    adapter = TestTimePriorAdapter(alpha_dirichlet=1.05, source_priors=source_priors)
    probs_ttpa = adapter.fit_transform(probs_raw)

    print_diagnostics(ids, probs_raw, probs_ttpa, adapter, elapsed)

    # Save raw and TTPA CSVs alongside zip
    raw_csv = args.out_zip.with_name(f"{args.out_zip.stem}_raw.csv")
    ttpa_csv = args.out_zip.with_name(f"{args.out_zip.stem}_ttpa.csv")
    pd.DataFrame({"id": ids, "label": np.argmax(probs_raw, axis=1)}).to_csv(raw_csv, index=False)
    pd.DataFrame({"id": ids, "label": np.argmax(probs_ttpa, axis=1)}).to_csv(ttpa_csv, index=False)

    np.save(args.out_probs, probs_raw)
    final_preds = np.argmax(probs_raw if args.no_ttpa else probs_ttpa, axis=1)
    write_codabench_submission(ids, final_preds, args.out_zip.with_suffix(".csv"), args.out_zip)

    print(f"\n✅ Saved raw probabilities -> {args.out_probs}")
    print(f"✅ Saved raw & TTPA CSVs -> {raw_csv.name}, {ttpa_csv.name}")
    print(f"✅ Saved CodaBench zip    -> {args.out_zip} (mode={'RAW' if args.no_ttpa else 'TTPA'})")


if __name__ == "__main__":
    main()
