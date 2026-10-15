import os
import random
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter

def elastic_transform(image, alpha=14, sigma=3):
    shape = image.shape
    dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)
    distorted = cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    return distorted

def render_character_base(char, font_path, font_size=46):
    img = Image.new("L", (64, 64), color=0)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        font = ImageFont.load_default()
    
    bbox = draw.textbbox((0, 0), char, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    
    x = (64 - w) // 2 - bbox[0]
    y = (64 - h) // 2 - bbox[1]
    draw.text((x, y), char, fill=255, font=font)
    return np.array(img, dtype=np.uint8)

def apply_handwriting_augmentations(img_arr):
    # 1. Random Affine (Rotation, Scale, Translation, Shear)
    h, w = img_arr.shape
    angle = random.uniform(-8.0, 8.0)
    scale = random.uniform(0.88, 1.05)
    tx = random.uniform(-2.5, 2.5)
    ty = random.uniform(-2.5, 2.5)
    shear = random.uniform(-5.0, 5.0)
    
    M_rot = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    M_rot[0, 2] += tx
    M_rot[1, 2] += ty
    # Add shear
    M_rot[0, 1] += np.tan(np.radians(shear))
    
    transformed = cv2.warpAffine(img_arr, M_rot, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    # 2. Elastic Deformation
    if random.random() < 0.85:
        alpha = random.uniform(8.0, 16.0)
        sigma = random.uniform(2.5, 3.5)
        transformed = elastic_transform(transformed, alpha=alpha, sigma=sigma)
        
    # 3. Morphological Dilation / Erosion (Stroke thickness)
    morph_op = random.random()
    if morph_op < 0.45:
        # Dilation (Thicker stroke)
        k_size = random.choice([2, 3])
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        transformed = cv2.dilate(transformed, kernel, iterations=1)
    elif morph_op < 0.70:
        # Erosion (Thinner stroke)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        transformed = cv2.erode(transformed, kernel, iterations=1)
        
    # 4. Mild Gaussian Blur (matching the smoothness of dataset pre-processing)
    if random.random() < 0.6:
        ksize = random.choice([3, 5])
        transformed = cv2.GaussianBlur(transformed, (ksize, ksize), sigmaX=random.uniform(0.4, 0.9))
        
    # 5. Threshold normalization to ensure clean white strokes on black background
    transformed = np.clip(transformed, 0, 255).astype(np.uint8)
    return transformed

def generate_synthetic_dataset(
    output_dir="FinalProject/train_synthetic",
    output_csv="FinalProject/train_synthetic_labels.csv",
    num_per_char=350
):
    os.makedirs(output_dir, exist_ok=True)
    
    font_candidates = [
        "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/noto-cjk/NotoSerifCJK-Regular.ttc",
        "/usr/share/fonts/noto-cjk/NotoSansCJK-Medium.ttc",
        "/usr/share/fonts/noto-cjk/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/noto-cjk/NotoSerifCJK-Bold.ttc",
        "/usr/share/fonts/noto-cjk/NotoSansCJK-Light.ttc"
    ]
    available_fonts = [f for f in font_candidates if os.path.exists(f)]
    if not available_fonts:
        available_fonts = ["/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc"]
    
    print(f"Using {len(available_fonts)} CJK system fonts for synthetic generation.")
    
    targets = [
        {"label": 13, "character": "萬", "value": 10000, "name": "wan_trad"},
        {"label": 14, "character": "億", "value": 100000000, "name": "yi_trad"}
    ]
    
    records = []
    sample_id = 1
    
    print(f"Generating {num_per_char} synthetic samples for each traditional character...")
    for item in targets:
        char = item["character"]
        label = item["label"]
        val = item["value"]
        
        for i in range(num_per_char):
            font_path = random.choice(available_fonts)
            font_size = random.randint(42, 50)
            base_img = render_character_base(char, font_path, font_size=font_size)
            aug_img = apply_handwriting_augmentations(base_img)
            
            img_id = f"synth_{sample_id:05d}"
            file_name = f"{img_id}.png"
            file_path = os.path.join(output_dir, file_name)
            
            # Save PNG
            cv2.imwrite(file_path, aug_img)
            
            records.append({
                "id": img_id,
                "label": label,
                "character": char,
                "value": val,
                "writer_id": 999  # Flag as synthetic
            })
            sample_id += 1
            
    df_synth = pd.DataFrame(records)
    df_synth.to_csv(output_csv, index=False)
    print(f"Successfully generated {len(df_synth)} synthetic images at: {output_dir}")
    print(f"Saved synthetic labels to: {output_csv}")
    print(df_synth.groupby(["label", "character"]).size())
    return df_synth

if __name__ == "__main__":
    generate_synthetic_dataset()
