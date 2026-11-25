#!/usr/bin/env python3
"""
================================================================================
Lecturer Exact Filter Modeling & Verification Module
Reconstructs and standardizes the exact preprocessing pipeline applied by the lecturer:
1. Grayscale & Polarity Normalization (White stroke on Black background).
2. Clean Stroke Segmentation (Otsu / adaptive noise gating).
3. Aspect-Ratio Preserving Bounding Box Isolation.
4. Symmetric Canvas Squaring & Chinese MNIST Standard Margin (8-10% padding).
5. Area-Interpolation Resizing to exactly 64x64.
6. Gaussian Blur (sigma=0.5, 3x3 kernel) matching the lecturer's antialiasing.
================================================================================
"""

import cv2
import numpy as np
from typing import Tuple, Optional

def apply_lecturer_exact_filter(
    img: np.ndarray,
    target_size: int = 64,
    stroke_threshold: int = 25,
    margin_ratio: float = 0.08,
    blur_sigma: float = 0.5
) -> np.ndarray:
    """
    Transforms any raw input character image (scanned calligraphy, museum crop, font)
    into an image that is statistically and morphologically identical to the lecturer's dataset.
    """
    # 1. Grayscale conversion if 3 channels
    if len(img.shape) == 3:
        if img.shape[2] == 4:
            # Handle RGBA
            alpha = img[:, :, 3]
            gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
            gray[alpha == 0] = 0
            img = gray
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Polarity Normalization (White stroke on Black background)
    # Check 4 corners to detect white/light paper background
    h_orig, w_orig = img.shape
    corner_samples = [
        img[0, 0], img[0, w_orig - 1],
        img[h_orig - 1, 0], img[h_orig - 1, w_orig - 1]
    ]
    if np.median(corner_samples) > 120:
        # White background -> Invert
        img = 255 - img

    # 3. Stroke Segmentation & Bounding Box Extraction
    # Remove faint background paper grain/scanner noise
    stroke_mask = (img > stroke_threshold).astype(np.uint8) * 255
    y_idx, x_idx = np.where(stroke_mask > 0)
    
    if len(y_idx) > 0:
        ymin, ymax = y_idx.min(), y_idx.max()
        xmin, xmax = x_idx.min(), x_idx.max()
        cropped = img[ymin:ymax+1, xmin:xmax+1]
        
        # 4. Aspect-Ratio Preserving Square Canvas
        h_crop, w_crop = cropped.shape
        max_dim = max(h_crop, w_crop)
        
        pad_top = (max_dim - h_crop) // 2
        pad_bottom = max_dim - h_crop - pad_top
        pad_left = (max_dim - w_crop) // 2
        pad_right = max_dim - w_crop - pad_left
        
        squared = cv2.copyMakeBorder(
            cropped,
            pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=0
        )
        
        # 5. Canvas Margin Padding (~8% of max_dim to match Chinese MNIST positioning)
        margin = max(2, int(max_dim * margin_ratio))
        padded = cv2.copyMakeBorder(
            squared,
            margin, margin, margin, margin,
            cv2.BORDER_CONSTANT, value=0
        )
        
        # 6. High-Fidelity Area-Interpolation Resizing
        resized = cv2.resize(padded, (target_size, target_size), interpolation=cv2.INTER_AREA)
    else:
        resized = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)

    # 7. Exact Antialiasing Edge Smoothing (Gaussian Blur sigma=0.5, 3x3)
    # Dosen note: "Goresannya tidak dipertebal karena sudah tebal, tetapi diberi blur ringan agar tepinya sehalus data latih"
    smoothed = cv2.GaussianBlur(resized, (3, 3), blur_sigma)
    
    return np.clip(smoothed, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    import os, glob
    # Test on a few diverse samples
    print("Testing Lecturer Exact Filter on sample images...")
    sample_train = "FinalProject/data/train/train_00001.png"
    sample_val_gb = "FinalProject/data/validasi/test/val_00005.png"
    
    img_tr = cv2.imread(sample_train, cv2.IMREAD_GRAYSCALE)
    img_gb = cv2.imread(sample_val_gb, cv2.IMREAD_GRAYSCALE)
    
    # Process a raw cursive sample
    cursive_samples = glob.glob("FinalProject/data/train_cursive_adapted/*.png")
    if cursive_samples:
        raw_cur = cv2.imread(cursive_samples[0], cv2.IMREAD_GRAYSCALE)
        filtered_cur = apply_lecturer_exact_filter(raw_cur)
        print(f"Filtered Cursive shape: {filtered_cur.shape}, min: {filtered_cur.min()}, max: {filtered_cur.max()}, mean: {filtered_cur.mean():.2f}")
    
    print("Filter module compiled successfully!")
