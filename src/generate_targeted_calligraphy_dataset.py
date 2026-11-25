#!/usr/bin/env python3
"""
Targeted Synthetic Calligraphy Dataset Generator v2 for Chinese MNIST (Taktik A).
Generates authentic Caoshu (Wild Cursive), Xingshu (Semi-Cursive), and Maobi (Brush)
training samples across 7 authentic Google Fonts:
1. LiuJianMaoCao-Regular.ttf (Caoshu - Wild Cursive)
2. LongCang-Regular.ttf (Caoshu - Expressive Flowing Cursive)
3. ZhiMangXing-Regular.ttf (Xingshu - Semi-Cursive)
4. MaShanZheng-Regular.ttf (Maobi - Traditional Brush Calligraphy)
5. ZCOOLXiaoWei-Regular.ttf (Slender Dynamic Artistic Script)
6. ZCOOLKuaiLe-Regular.ttf (Casual Dynamic Script)
7. ZCOOLQingKeHuangYou-Regular.ttf (Heavy Structural Calligraphy)

Includes targeted biomechanical stroke perturbations to eliminate the 3 critical confusion pairs:
- 十 (10) vs 千 (12): Vertical entry flick & top slash variation (22 errors in Sub 10)
- 五 (5) vs 六 (6): Caoshu continuous Z/S wave flow & loop preservation (15 errors in Sub 10)
- 四 (4) vs 亿 (14): Inner stroke wall-touching ligature vs two-component radical (12 errors in Sub 10)

Output format:
- Canvas: 64x64 Grayscale, white stroke (255) on black background (0).
- Standardized aspect-ratio preservation and centering.
- Saved to: FinalProject/data/train_synthetic_calligraphy_v2/
- Metadata: FinalProject/data/metadata/train_synthetic_calligraphy_v2.csv
"""

import os
import sys
import math
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
OUTPUT_DIR = DATA_DIR / "train_synthetic_calligraphy_v2"
METADATA_DIR = DATA_DIR / "metadata"

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

ALL_FONTS = [
    FONTS_DIR / "LiuJianMaoCao-Regular.ttf",
    FONTS_DIR / "LongCang-Regular.ttf",
    FONTS_DIR / "ZhiMangXing-Regular.ttf",
    FONTS_DIR / "MaShanZheng-Regular.ttf",
    FONTS_DIR / "ZCOOLXiaoWei-Regular.ttf",
    FONTS_DIR / "ZCOOLKuaiLe-Regular.ttf",
    FONTS_DIR / "ZCOOLQingKeHuangYou-Regular.ttf"
]


def elastic_distortion(img_arr: np.ndarray, alpha=6.0, sigma=4.0) -> np.ndarray:
    """
    Simard et al. (ICDAR 2003) elastic deformation calibrated for 64x64 glyphs.
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
    try:
        font = ImageFont.truetype(str(font_path), font_size)
    except Exception:
        font = ImageFont.load_default()

    canvas_size = 128
    img = Image.new("L", (canvas_size, canvas_size), color=0)
    draw = ImageDraw.Draw(img)
    draw.text((32, 24), char, fill=255, font=font)
    arr = np.array(img, dtype=np.uint8)

    coords = cv2.findNonZero(arr)
    if coords is None:
        return np.zeros((64, 64), dtype=np.uint8)

    x, y, w, h = cv2.boundingRect(coords)
    crop = arr[y:y+h, x:x+w]

    target_max = random.randint(44, 50)
    scale = target_max / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

    out = np.zeros((64, 64), dtype=np.uint8)
    pad_y = (64 - new_h) // 2
    pad_x = (64 - new_w) // 2
    out[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
    return out


def apply_targeted_perturbation(img: np.ndarray, label: int) -> np.ndarray:
    """
    Injects targeted domain perturbations specifically addressing the 3 major confusion pairs.
    """
    out = img.copy()
    H, W = out.shape[:2]

    # --- Confusion Pair 1: 十 (10) vs 千 (12) ---
    if label == 10:
        # In calligraphy, 十 often has an entry flick at the top-left of the vertical stroke.
        # We simulate this so the model learns that a small entry hook does NOT make it 千.
        if random.random() < 0.45:
            # Find the vertical stroke top point
            col_sums = np.sum(out[:24, :], axis=0)
            if np.max(col_sums) > 0:
                mid_x = int(np.argmax(col_sums))
                row_indices = np.where(out[:, mid_x] > 50)[0]
                if len(row_indices) > 0:
                    top_y = row_indices[0]
                    # Draw a slight entry hook / slant flick
                    hook_len = random.randint(3, 6)
                    pt1 = (max(0, mid_x - hook_len), max(0, top_y - random.randint(1, 3)))
                    pt2 = (mid_x, top_y)
                    cv2.line(out, pt1, pt2, color=200, thickness=random.choice([1, 2]))

    elif label == 12:  # 千
        #千 must have a clear top slant stroke.
        # We vary its intensity and angle slightly to distinguish it from a mere entry hook.
        if random.random() < 0.35:
            # Emphasize top stroke
            top_h = int(H * 0.25)
            out[:top_h, :] = np.clip(out[:top_h, :].astype(np.float32) * random.uniform(1.1, 1.3), 0, 255).astype(np.uint8)

    # --- Confusion Pair 2: 五 (5) vs 六 (6) ---
    elif label == 5:
        # Caoshu cursive 五: sinusoidal vertical wave / continuous brush flow
        if random.random() < 0.50:
            A = random.uniform(1.8, 3.2)
            period = random.uniform(28.0, 44.0)
            phase = random.uniform(0, 2 * math.pi)
            y_indices = np.arange(H).reshape(H, 1)
            dx = (A * np.sin(2 * np.pi * y_indices / period + phase)).astype(np.float32)
            dx_grid = np.repeat(dx, W, axis=1)

            grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
            map_x = (grid_x + dx_grid).astype(np.float32)
            map_y = grid_y.astype(np.float32)
            out = cv2.remap(out, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    elif label == 6:  # 六
        # In 六, preserve distinct top dot/slash and bottom diverging legs
        if random.random() < 0.30:
            # Slight diagonal brush lift
            kernel = np.array([[0, 1], [1, 0]], dtype=np.uint8)
            out = cv2.dilate(out, kernel, iterations=1)

    # --- Confusion Pair 3: 四 (4) vs 亿 (14) ---
    elif label == 4:
        # In cursive 四, inner strokes can touch the left border, but the box is closed.
        if random.random() < 0.45:
            # Morphological bridge inside the box
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            closed = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)
            alpha = random.uniform(0.4, 0.8)
            out = cv2.addWeighted(out, 1.0 - alpha, closed, alpha, 0)

    elif label == 14:  # 亿 / 億
        # Ensure two-component structure is maintained
        if random.random() < 0.30:
            # Slight vertical dilation
            kernel = np.array([[1], [1]], dtype=np.uint8)
            out = cv2.dilate(out, kernel, iterations=1)

    # --- Other important characters ---
    elif label == 11:  # 百
        # Cursive 百: lower box closing
        if random.random() < 0.40:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            closed = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)
            out = cv2.addWeighted(out, 0.4, closed, 0.6, 0)

    elif label == 13:  # 万 / 萬
        if random.random() < 0.35:
            # Diagonal brush sweep
            kernel = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.uint8)
            out = cv2.dilate(out, kernel, iterations=1)

    return out


def apply_calligraphic_transforms(img_arr: np.ndarray, label: int) -> np.ndarray:
    """
    Applies comprehensive calligraphic physical transforms:
    - Targeted perturbation for the character's confusion mode
    - Random affine (rotation, scale, micro-shear)
    - Gentle elastic deformation (biomechanical tremor)
    - Directional brush pressure (dilation/erosion)
    - Gaussian smoothing matching CodaBench specifications
    """
    H, W = img_arr.shape[:2]

    # 1. Targeted stroke perturbation
    transformed = apply_targeted_perturbation(img_arr, label)

    # 2. Random Affine (Rotation, scale, shear)
    angle = random.uniform(-7.0, 7.0)
    scale = random.uniform(0.92, 1.05)
    tx = random.uniform(-1.5, 1.5)
    ty = random.uniform(-1.5, 1.5)
    shear = random.uniform(-3.5, 3.5)

    M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    M[0, 1] += np.tan(np.radians(shear))

    transformed = cv2.warpAffine(
        transformed, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )

    # 3. Gentle Elastic Distortion
    if random.random() < 0.80:
        alpha = random.uniform(4.0, 7.5)
        sigma = random.uniform(3.5, 4.5)
        transformed = elastic_distortion(transformed, alpha=alpha, sigma=sigma)

    # 4. Anisotropic Brush Pressure (Directional Dilation/Erosion)
    morph_choice = random.random()
    if morph_choice < 0.35:
        # Diagonal dilation (45 deg brush stroke)
        kernel = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.uint8)
        transformed = cv2.dilate(transformed, kernel, iterations=1)
    elif morph_choice < 0.55:
        # Diagonal dilation (135 deg brush stroke)
        kernel = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.uint8)
        transformed = cv2.dilate(transformed, kernel, iterations=1)
    elif morph_choice < 0.75:
        # Cross dilation
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2))
        transformed = cv2.dilate(transformed, kernel, iterations=1)
    elif morph_choice < 0.90:
        # Mild erosion (dry brush effect / Feibai)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        transformed = cv2.erode(transformed, kernel, iterations=1)

    # 5. Mild Gaussian Blur (matching CodaBench: 'diberi blur ringan agar tepinya sehalus data latih')
    sigma_b = random.uniform(0.40, 0.70)
    transformed = cv2.GaussianBlur(transformed, (3, 3), sigmaX=sigma_b, sigmaY=sigma_b)

    # Contrast normalization
    transformed = np.clip(transformed, 0, 255).astype(np.uint8)
    return transformed


def generate_targeted_calligraphy_dataset(samples_per_class=400):
    """
    Generates synthetic calligraphy dataset across all 15 classes using 7 Google Fonts.
    400 samples/class x 15 classes = 6,000 images.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    total_imgs = 0

    print("==============================================================")
    print("🎨 Generating Targeted Synthetic Calligraphy Dataset v2 (Taktik A)")
    print(f"Fonts available: {len(ALL_FONTS)}")
    for f in ALL_FONTS:
        print(f"  - {f.name}")
    print(f"Target: {samples_per_class} samples/class across 15 classes = {samples_per_class * 15:,} images")
    print("==============================================================")

    for label, info in CLASS_DEFINITIONS.items():
        chars = info["chars"]
        char_name = info["name"]
        print(f"Generating Class {label:2d} ({'/'.join(chars)}) ...", end=" ", flush=True)

        for i in range(samples_per_class):
            char = random.choice(chars)

            # Filter fonts that support this character
            valid_fonts = []
            for f in ALL_FONTS:
                try:
                    font = ImageFont.truetype(str(f), 32)
                    bbox = font.getmask(char).getbbox()
                    if bbox is not None and bbox[2] > 0 and bbox[3] > 0:
                        valid_fonts.append(f)
                except Exception:
                    pass

            if not valid_fonts:
                valid_fonts = ALL_FONTS

            font_path = random.choice(valid_fonts)
            font_tag = font_path.stem.split("-")[0].lower()

            base_glyph = render_base_glyph(char, font_path)
            augmented = apply_calligraphic_transforms(base_glyph, label=label)

            img_id = f"callig_v2_{label:02d}_{i:04d}_{font_tag}"
            img_filename = f"{img_id}.png"
            img_path = OUTPUT_DIR / img_filename

            cv2.imwrite(str(img_path), augmented)

            records.append({
                "id": img_id,
                "label": label,
                "character": char,
                "class_name": char_name,
                "font_style": font_tag,
                "writer_id": 950 + (i % 50)  # Pseudo-writer IDs 950-999
            })
            total_imgs += 1

        print(f"Done ({samples_per_class} images)")

    df = pd.DataFrame(records)
    csv_path = METADATA_DIR / "train_synthetic_calligraphy_v2.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Successfully generated {total_imgs:,} synthetic calligraphy images!")
    print(f"📁 Images saved to: {OUTPUT_DIR}")
    print(f"📄 Metadata saved to: {csv_path}")
    return df


if __name__ == "__main__":
    generate_targeted_calligraphy_dataset(samples_per_class=400)
