#!/usr/bin/env python3
"""
Biomechanical Data Augmentation Module for Handwritten Chinese Character Recognition (HCCR).
Implements SOTA biomechanical transformations:
1. Calibrated Elastic Distortion (Simard et al. ICDAR 2003) for 64x64 Hanzi glyphs.
2. Anisotropic Brush Pressure & Ink Diffusion Morphology (Directional Dilation/Erosion).
3. Targeted Stroke Perturbation (DropStroke & Ligature Bridging) for failure-mode mitigation
   (五 vs 万/2/6, 四 vs 亿, 百 vs 万, 十 vs 千).
"""

import math
import random
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
import torch
import torch.nn as nn

class ElasticDistortion:
    """
    Elastic Deformation for Handwritten Characters based on Simard, Steinkraus, & Platt (ICDAR 2003).
    Calibrated specifically for 64x64 Chinese Character glyphs:
      - alpha: displacement magnitude scaling. For 64x64, alpha in [6.0, 10.0] ensures peak
               displacements of 2.0-3.5 pixels (matching inter-stroke distances).
      - sigma: Gaussian smoothing standard deviation. For 64x64, sigma in [3.5, 4.5] ensures
               spatial smoothness spans across full radicals without high-frequency tearing.
    """
    def __init__(self, alpha_range=(6.0, 10.0), sigma_range=(3.5, 4.5), p=0.5):
        self.alpha_range = alpha_range
        self.sigma_range = sigma_range
        self.p = p

    def __call__(self, img_np: np.ndarray) -> np.ndarray:
        """
        Args:
            img_np: Grayscale image as uint8 numpy array of shape (H, W), values in [0, 255].
        Returns:
            Distorted grayscale image as uint8 numpy array of shape (H, W).
        """
        if random.random() > self.p:
            return img_np

        H, W = img_np.shape[:2]
        alpha = random.uniform(*self.alpha_range)
        sigma = random.uniform(*self.sigma_range)

        # 1. Generate uniform random displacement fields in [-1, 1]
        noise_x = np.random.uniform(-1.0, 1.0, size=(H, W))
        noise_y = np.random.uniform(-1.0, 1.0, size=(H, W))

        # 2. Convolve with Gaussian filter to induce smooth biomechanical muscle tremor
        dx = gaussian_filter(noise_x, sigma, mode="constant", cval=0.0) * alpha
        dy = gaussian_filter(noise_y, sigma, mode="constant", cval=0.0) * alpha

        # 3. Construct spatial sampling grid
        grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
        map_x = (grid_x + dx).astype(np.float32)
        map_y = (grid_y + dy).astype(np.float32)

        # 4. Bilinear interpolation with constant border (background is black: 0)
        distorted = cv2.remap(
            img_np,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        return distorted


class DynamicBrushMorphology:
    """
    Simulates Calligraphic Brush Dynamics & Ink Diffusion (Yongbi / 用笔 mechanics):
    - Anisotropic Directional Kernels: Chinese brush calligraphy naturally slants at 45° or 135°.
    - Variable Brush Pressure: Inhomogeneous dilation / erosion across glyph axes.
    - Ink Bleed (Yunran / 晕染) vs Dry Brush Fading (Feibai / 飞白).
    """
    def __init__(self, p_dilate=0.35, p_erode=0.30, p_gradient=0.20):
        self.p_dilate = p_dilate
        self.p_erode = p_erode
        self.p_gradient = p_gradient

        # Anisotropic calligraphic structuring elements
        self.kernel_45 = np.array([
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0]
        ], dtype=np.uint8)

        self.kernel_135 = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ], dtype=np.uint8)

        self.kernel_cross = np.array([
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0]
        ], dtype=np.uint8)

        self.kernel_rect2 = np.ones((2, 2), dtype=np.uint8)

    def _get_random_kernel(self):
        choice = random.choice(["45", "135", "cross", "rect2"])
        if choice == "45":
            return self.kernel_45
        elif choice == "135":
            return self.kernel_135
        elif choice == "cross":
            return self.kernel_cross
        else:
            return self.kernel_rect2

    def __call__(self, img_np: np.ndarray) -> np.ndarray:
        out = img_np.copy()
        H, W = out.shape[:2]

        # 1. Random Brush Pressure Gradient (heterogeneous thickness across height)
        if random.random() < self.p_gradient:
            # Gradient mask: thicker strokes at bottom or top (simulating brush press down)
            grad_type = random.choice(["vertical", "diagonal"])
            if grad_type == "vertical":
                grad = np.linspace(0.85, 1.25, H)[:, None]
                if random.random() < 0.5:
                    grad = grad[::-1]
            else:
                gx, gy = np.meshgrid(np.linspace(0.9, 1.2, W), np.linspace(0.9, 1.2, H))
                grad = (gx + gy) / 2.0
            out = np.clip(out.astype(np.float32) * grad, 0, 255).astype(np.uint8)

        # 2. Dynamic Dilation (Wet ink / Heavy brush press)
        if random.random() < self.p_dilate:
            kernel = self._get_random_kernel()
            out = cv2.dilate(out, kernel, iterations=1)

        # 3. Dynamic Erosion (Dry ink / Fast brush lift)
        elif random.random() < self.p_erode:
            kernel = self._get_random_kernel()
            out = cv2.erode(out, kernel, iterations=1)

        return out


class TargetedStrokePerturbation:
    """
    Targeted Morphological Perturbations designed to resolve CodaBench's 4 major error modes:
    1. 十 (10) vs 千 (12): Random Stroke Fading (DropStroke) in top 25% region or top tick injection.
    2. 四 (4) vs 亿 (14): Inner stroke ligature bridging (closing inner vertical strokes into '亻' like curve).
    3. 五 (5): Cursive Z/S wave deformation & loop closure to mimic Caoshu ligature style.
    4. 百 (11) vs 万 (13): Lower bounding box closure and baseline bridging.
    """
    def __init__(self, p_perturb=0.40):
        self.p_perturb = p_perturb

    def apply_stroke_fade_top(self, img_np: np.ndarray) -> np.ndarray:
        """
        Simulates top slant stroke of '千' (12) fading away, or tests robustness of '十' (10).
        Fades pixel intensity in top 25% of image by factor [0.15, 0.50].
        """
        out = img_np.copy()
        H, W = out.shape[:2]
        top_h = int(H * 0.28)
        fade_factor = random.uniform(0.15, 0.55)
        out[:top_h, :] = (out[:top_h, :].astype(np.float32) * fade_factor).astype(np.uint8)
        return out

    def apply_stroke_bridge_inner(self, img_np: np.ndarray) -> np.ndarray:
        """
        Simulates cursive stroke bridging (Lianbi) by morphological closing along vertical/diagonal axes.
        Helps model learn that inner stroke fusion in '四' or '百' remains the same character class.
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        closed = cv2.morphologyEx(img_np, cv2.MORPH_CLOSE, kernel)
        # Blend original with closed
        alpha = random.uniform(0.5, 0.9)
        out = cv2.addWeighted(img_np, 1.0 - alpha, closed, alpha, 0)
        return out

    def apply_cursive_wave_warp(self, img_np: np.ndarray) -> np.ndarray:
        """
        Simulates cursive S/Z wave warping on '五' (5) by applying sinusoidal displacement:
        dx = A * sin(2 * pi * y / lambda)
        """
        H, W = img_np.shape[:2]
        A = random.uniform(1.8, 3.2)  # amplitude in pixels
        period = random.uniform(28.0, 48.0)  # wavelength
        phase = random.uniform(0, 2 * math.pi)

        y_indices = np.arange(H).reshape(H, 1)
        dx = (A * np.sin(2 * np.pi * y_indices / period + phase)).astype(np.float32)
        dx_grid = np.repeat(dx, W, axis=1)

        grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
        map_x = (grid_x + dx_grid).astype(np.float32)
        map_y = grid_y.astype(np.float32)

        warped = cv2.remap(
            img_np,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        return warped

    def __call__(self, img_np: np.ndarray, label: int = -1) -> np.ndarray:
        if random.random() > self.p_perturb:
            return img_np

        # If class-specific label is provided, target specifically
        if label == 12 or label == 10:  # 千 or 十
            if random.random() < 0.6:
                return self.apply_stroke_fade_top(img_np)
        elif label == 5:  # 五 (cursive Caoshu)
            if random.random() < 0.7:
                return self.apply_cursive_wave_warp(img_np)
        elif label == 4 or label == 11:  # 四 or 百
            if random.random() < 0.6:
                return self.apply_stroke_bridge_inner(img_np)

        # General case (random targeted strategy)
        strat = random.choice(["fade_top", "bridge_inner", "wave_warp"])
        if strat == "fade_top":
            return self.apply_stroke_fade_top(img_np)
        elif strat == "bridge_inner":
            return self.apply_stroke_bridge_inner(img_np)
        else:
            return self.apply_cursive_wave_warp(img_np)


class BiomechanicalAugmentationPipeline:
    """
    Production-grade Unified Biomechanical Augmentation Pipeline for 64x64 Hanzi HCCR.
    Combines:
    1. Affine Jitter (Subtle rotation ±7°, scale [0.93, 1.05], translation ±2px).
    2. Elastic Distortion (alpha=8.0, sigma=4.0).
    3. Dynamic Brush Morphology (Directional dilation / erosion / pressure gradient).
    4. Targeted Stroke Perturbation (DropStroke & Ligature bridging).
    """
    def __init__(
        self,
        elastic_p=0.45,
        morph_p=0.50,
        perturb_p=0.35,
        affine_p=0.85
    ):
        self.affine_p = affine_p
        self.elastic = ElasticDistortion(alpha_range=(6.0, 9.5), sigma_range=(3.6, 4.4), p=elastic_p)
        self.morph = DynamicBrushMorphology(p_dilate=0.30, p_erode=0.25, p_gradient=0.20)
        self.perturb = TargetedStrokePerturbation(p_perturb=perturb_p)

    def __call__(self, img_np: np.ndarray, label: int = -1) -> np.ndarray:
        """
        Processes single 64x64 uint8 numpy image.
        """
        out = img_np.copy()
        H, W = out.shape[:2]

        # Step 1: Subtle Affine Jitter
        if random.random() < self.affine_p:
            angle = random.uniform(-7.0, 7.0)
            scale = random.uniform(0.93, 1.05)
            tx = random.uniform(-2.0, 2.0)
            ty = random.uniform(-2.0, 2.0)
            M = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), angle, scale)
            M[0, 2] += tx
            M[1, 2] += ty
            out = cv2.warpAffine(
                out, M, (W, H),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )

        # Step 2: Targeted Stroke Perturbation (Address 十 vs 千, 四 vs 亿, etc.)
        out = self.perturb(out, label=label)

        # Step 3: Elastic Distortion (Simard Biomechanical Muscle Tremor)
        out = self.elastic(out)

        # Step 4: Dynamic Calligraphic Brush Morphology (Ink & Pressure)
        out = self.morph(out)

        return out
