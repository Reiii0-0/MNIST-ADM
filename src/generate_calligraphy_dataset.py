#!/usr/bin/env python3
"""
Synthetic Calligraphy Dataset Generator for Chinese MNIST.
Generates authentic Caoshu (Cursive), Xingshu (Semi-Cursive), and Maobi (Brush)
training samples using open-source Google Fonts:
1. LiuJianMaoCao-Regular.ttf (Caoshu - Wild Cursive)
2. ZhiMangXing-Regular.ttf (Xingshu - Semi-Cursive)
3. MaShanZheng-Regular.ttf (Maobi - Brush Calligraphy)

Calibrated strictly to CodaBench Chinese MNIST specifications:
- Canvas: 64x64 Grayscale, white stroke (255) on black background (0).
- Aspect-ratio-preserved bounding box scaling & centering.
- Mild calligraphic transforms: gentle elastic distortion, brush morphology, and light Gaussian smoothing.
"""

import os
import sys
import random
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
FONTS_DIR = DATA_DIR / "fonts"
OUTPUT_DIR = DATA_DIR / "train_synthetic_calligraphy"
METADATA_DIR = DATA_DIR / "metadata"

# 15 Classes with character variants (including traditional 萬 & 億)
CLASS_DEFINITIONS = {
    0:  {"chars": ["零"], "name": "ling"},
    1:  {"chars": ["一"], "name": "yi"},
    2:  {"chars": ["二"], "name": "er"},
    3:  {"chars": ["三"], "name": "san"},
    4:  {"chars": ["四"], "name": "si"},
    5:  {"chars": ["五"], "name": "wu"},
    6:  {"chars": ["六"], "name": "liu"},
    7:  {"chars": ["七"], "name": "qi"},
    8:  {"chars": ["八"], "name": "ba"},
    9:  {"chars": ["九"], "name": "jiu"},
    10: {"chars": ["十"], "name": "shi"},
    11: {"chars": ["百"], "name": "bai"},
    12: {"chars": ["千"], "name": "qian"},
    13: {"chars": ["万", "萬"], "name": "wan"},
    14: {"chars": ["亿", "億"], "name": "yi_hundred_million"}
}

FONT_PATHS = [
    FONTS_DIR / "LiuJianMaoCao-Regular.ttf",
    FONTS_DIR / "ZhiMangXing-Regular.ttf",
    FONTS_DIR / "MaShanZheng-Regular.ttf"
]


def elastic_distortion(img_arr: np.ndarray, alpha=5.5, sigma=4.0) -> np.ndarray:
    """
    Gentle elastic deformation (Simard et al. ICDAR 2003).
    Calibrated for 64x64 glyphs to avoid stroke collision.
    """
    H, W = img_arr.shape[:2]
    dx = gaussian_filter((np.random.rand(H, W) * 2 - 1), sigma, mode="constant", cval=0.0) * alpha
    dy = gaussian_filter((np.random.rand(H, W) * 2 - 1), sigma, mode="constant", cval=0.0) * alpha

    grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
    map_x = (grid_x + dx).astype(np.float32)
    map_y = (grid_y + dy).astype(np.float32)

    distorted = cv2.remap(
        img_arr,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    return distorted


def render_base_glyph(char: str, font_path: Path, font_size=48) -> np.ndarray:
    """
    Renders character into tight aspect-ratio-preserved 64x64 canvas.
    """
    font = ImageFont.truetype(str(font_path), font_size)
    
    # Render at larger canvas first to get clean rasterization
    canvas_size = 128
    img = Image.new("L", (canvas_size, canvas_size), color=0)
    draw = ImageDraw.Draw(img)
    draw.text((32, 24), char, fill=255, font=font)
    arr = np.array(img, dtype=np.uint8)

    # Find tight bounding box of character
    coords = cv2.findNonZero(arr)
    if coords is None:
        return np.zeros((64, 64), dtype=np.uint8)

    x, y, w, h = cv2.boundingRect(coords)
    crop = arr[y:y+h, x:x+w]

    # Target interior size in 64x64 canvas (approx 44-50 px)
    target_max = random.randint(44, 50)
    scale = target_max / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Center into 64x64 canvas with symmetric padding
    out = np.zeros((64, 64), dtype=np.uint8)
    pad_y = (64 - new_h) // 2
    pad_x = (64 - new_w) // 2
    out[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
    return out


def apply_calligraphic_transforms(img_arr: np.ndarray) -> np.ndarray:
    """
    Applies calligraphic physical transforms:
    - Mild random affine (rotation, scale, micro-shear)
    - Gentle elastic deformation
    - Directional brush pressure (dilation/erosion)
    - Gaussian smoothing matching CodaBench specifications
    """
    H, W = img_arr.shape[:2]

    # 1. Random Affine (Rotation, scale, shear)
    angle = random.uniform(-7.0, 7.0)
    scale = random.uniform(0.90, 1.04)
    tx = random.uniform(-1.5, 1.5)
    ty = random.uniform(-1.5, 1.5)
    shear = random.uniform(-3.5, 3.5)

    M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    M[0, 1] += np.tan(np.radians(shear))

    transformed = cv2.warpAffine(
        img_arr, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )

    # 2. Gentle Elastic Distortion (simulating hand muscle tremor)
    if random.random() < 0.80:
        alpha = random.uniform(4.0, 7.0)
        sigma = random.uniform(3.5, 4.5)
        transformed = elastic_distortion(transformed, alpha=alpha, sigma=sigma)

    # 3. Anisotropic Brush Pressure (Directional Dilation/Erosion)
    morph_choice = random.random()
    if morph_choice < 0.40:
        # Diagonal dilation (45 deg brush stroke)
        kernel = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.uint8)
        transformed = cv2.dilate(transformed, kernel, iterations=1)
    elif morph_choice < 0.65:
        # Cross dilation
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2))
        transformed = cv2.dilate(transformed, kernel, iterations=1)
    elif morph_choice < 0.85:
        # Mild erosion (dry brush effect / Feibai)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        transformed = cv2.erode(transformed, kernel, iterations=1)

    # 4. Mild Gaussian Blur (matching CodaBench: 'diberi blur ringan agar tepinya sehalus data latih')
    sigma_b = random.uniform(0.40, 0.70)
    transformed = cv2.GaussianBlur(transformed, (3, 3), sigmaX=sigma_b, sigmaY=sigma_b)

    # Contrast normalization
    transformed = np.clip(transformed, 0, 255).astype(np.uint8)
    return transformed


def generate_calligraphy_dataset(samples_per_class=100):
    """
    Generates synthetic calligraphy dataset across all 15 classes.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    total_imgs = 0

    print("==============================================================")
    print("🎨 Generating Synthetic Caoshu & Xingshu Calligraphy Dataset")
    print(f"Target: {samples_per_class} samples/class across 15 classes = {samples_per_class * 15} images")
    print("==============================================================")

    for label, info in CLASS_DEFINITIONS.items():
        chars = info["chars"]
        char_name = info["name"]
        print(f"Generating Class {label:2d} ({'/'.join(chars)}) ...", end=" ", flush=True)

        for i in range(samples_per_class):
            # Select character variant (e.g. 万 vs 萬)
            char = random.choice(chars)
            # Select font (LiuJianMaoCao / ZhiMangXing / MaShanZheng)
            font_path = random.choice(FONT_PATHS)
            font_tag = font_path.stem.split("-")[0].lower()

            # Render & augment
            base_glyph = render_base_glyph(char, font_path)
            augmented = apply_calligraphic_transforms(base_glyph)

            img_id = f"callig_{label:02d}_{i:04d}_{font_tag}"
            img_filename = f"{img_id}.png"
            img_path = OUTPUT_DIR / img_filename

            cv2.imwrite(str(img_path), augmented)

            records.append({
                "id": img_id,
                "label": label,
                "character": char,
                "class_name": char_name,
                "font_style": font_tag,
                "writer_id": 990 + (i % 10)  # Distinct synthetic pseudo-writer IDs (990-999)
            })
            total_imgs += 1

        print(f"Done ({samples_per_class} images)")

    df = pd.DataFrame(records)
    csv_path = METADATA_DIR / "train_synthetic_calligraphy.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Successfully generated {total_imgs} synthetic calligraphy images!")
    print(f"📁 Images saved to: {OUTPUT_DIR}")
    print(f"📄 Metadata saved to: {csv_path}")
    return df


if __name__ == "__main__":
    generate_calligraphy_dataset(samples_per_class=100)
