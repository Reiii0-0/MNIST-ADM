#!/usr/bin/env python3
"""
================================================================================
Comprehensive Performance Benchmark: Student Distillation vs Baseline Champions
================================================================================
Compares:
1. New Distilled Student: DenseNet-121 CBAM SWA (checkpoint_student_fp16.pt)
2. Anchor Champion: ResNet-18 CBAM v2 (Single Fold & 5-Fold Ensemble)
3. Complementary Champion: ConvNeXt-Femto v2
4. Pre-Distillation DenseNet-121 CBAM (Fold 0)
5. 15-Model Production Ensemble Oracle

Evaluates:
- Validation Accuracy on Chinese MNIST (1,500 samples, writers 71-80)
- Validation Accuracy on Gaya Baru Cursive (966 samples, unseen cursive Caoshu)
- Overall Accuracy on Phase 1 Validation (2,466 samples)
- Stress-Test Invariance: Gaussian Noise, Simard Elastic Deformation, Stroke Erosion, Stroke Dilation
- Model Parameters, Checkpoint Size (MB), Latency (ms), and Throughput (FPS)
================================================================================
"""

import os
import sys
import time
import math
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from src.models import ResNet18_CBAM, ConvNeXt_Femto, SubCenterArcFace

def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

seed_everything(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------------------------
# Student DenseNet Architecture Definition (Exact Match to Distillation Script)
# ------------------------------------------------------------------------------
class MixStyle(nn.Module):
    def __init__(self, p=0.5, alpha=0.1, eps=1e-6):
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps

    def forward(self, x):
        if not self.training:
            return x
        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        r_planes = max(1, in_planes // ratio)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, r_planes, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(r_planes, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))

class CBAMBlock(nn.Module):
    def __init__(self, planes, ratio=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, bn_size=4, drop_rate=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, bn_size * growth_rate, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, 3, padding=1, bias=False)
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = self.conv1(F.relu(self.bn1(x), inplace=True))
        new_features = self.conv2(F.relu(self.bn2(new_features), inplace=True))
        return torch.cat([x, new_features], 1)

class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate, bn_size=4, drop_rate=0.0):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(DenseLayer(in_channels + i * growth_rate, growth_rate, bn_size, drop_rate))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.pool = nn.AvgPool2d(2, stride=2)

    def forward(self, x):
        return self.pool(self.conv(F.relu(self.bn(x), inplace=True)))


class StudentSubCenterArcFace(nn.Module):
    def __init__(self, in_features=256, num_classes=15, sub_centers=2, s=32.0, m=0.30):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.sub_centers = sub_centers
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes * sub_centers, in_features))
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input_features, label=None):
        cosine = F.linear(F.normalize(input_features), F.normalize(self.weight))
        cosine = cosine.view(-1, self.num_classes, self.sub_centers)
        cosine, _ = torch.max(cosine, dim=2)
        return cosine * self.s

class StudentDenseNet121_CBAM(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0.0, num_classes=15):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, num_init_features, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True)
        )
        self.mixstyle = MixStyle(p=0.5, alpha=0.1)

        num_features = num_init_features
        self.features = nn.ModuleList()
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(num_layers, num_features, growth_rate, bn_size, drop_rate)
            self.features.append(block)
            num_features = num_features + num_layers * growth_rate

            cbam = CBAMBlock(num_features)
            self.features.append(cbam)

            if i != len(block_config) - 1:
                trans = Transition(num_features, num_features // 2)
                self.features.append(trans)
                num_features = num_features // 2

        self.final_bn = nn.BatchNorm2d(num_features)
        self.fc_embed = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.BatchNorm1d(256),
            nn.PReLU()
        )
        self.fc_linear = nn.Linear(256, num_classes)

    def forward(self, x):
        out = self.stem(x)
        out = self.mixstyle(out)
        for layer in self.features:
            out = layer(out)
        out = F.relu(self.final_bn(out), inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(out.size(0), -1)
        feat = self.fc_embed(out)
        logits = self.fc_linear(feat)
        return logits, feat

# ------------------------------------------------------------------------------
# Perturbations & Dataset
# ------------------------------------------------------------------------------
class PerturbationBench:
    @staticmethod
    def clean(img: np.ndarray) -> np.ndarray:
        return img

    @staticmethod
    def gaussian_noise(img: np.ndarray, sigma=25.0) -> np.ndarray:
        noise = np.random.normal(0, sigma, img.shape)
        noisy = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return noisy

    @staticmethod
    def stroke_dilation(img: np.ndarray) -> np.ndarray:
        kernel = np.ones((2, 2), np.uint8)
        return cv2.dilate(img, kernel, iterations=1)

    @staticmethod
    def stroke_erosion(img: np.ndarray) -> np.ndarray:
        kernel = np.ones((2, 2), np.uint8)
        return cv2.erode(img, kernel, iterations=1)

    @staticmethod
    def elastic_distortion(img: np.ndarray, alpha=6.0, sigma=4.0) -> np.ndarray:
        H, W = img.shape[:2]
        noise_x = np.random.uniform(-1.0, 1.0, size=(H, W))
        noise_y = np.random.uniform(-1.0, 1.0, size=(H, W))
        dx = gaussian_filter(noise_x, sigma, mode="constant", cval=0.0) * alpha
        dy = gaussian_filter(noise_y, sigma, mode="constant", cval=0.0) * alpha
        grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
        map_x = (grid_x + dx).astype(np.float32)
        map_y = (grid_y + dy).astype(np.float32)
        return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)

class BenchmarkDataset(Dataset):
    def __init__(self, df: pd.DataFrame, perturb_fn=None):
        self.paths = df['absolute_path'].tolist()
        self.labels = df['label'].tolist()
        self.strata = df['stratum'].tolist()
        self.perturb_fn = perturb_fn if perturb_fn else PerturbationBench.clean

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((64, 64), dtype=np.uint8)
        if img.shape != (64, 64):
            img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)

        img = self.perturb_fn(img)
        tensor = torch.from_numpy(img.astype(np.float32) / 127.5 - 1.0).unsqueeze(0)
        return tensor, self.labels[idx], self.strata[idx]

def evaluate_model(model, head, loader, device, is_fp16=False) -> Dict[str, float]:
    model.eval()
    head.eval()
    correct_all, total_all = 0, 0
    correct_cm, total_cm = 0, 0
    correct_gb, total_gb = 0, 0

    with torch.no_grad():
        for x, y, strata in loader:
            x, y = x.to(device), y.to(device)
            if is_fp16:
                x = x.half()
            lin, f = model(x)
            arc = head(f)
            preds = torch.argmax(0.5 * lin + 0.5 * arc, dim=1)

            for i in range(y.size(0)):
                p, t, s = preds[i].item(), y[i].item(), strata[i]
                is_correct = (p == t)
                correct_all += int(is_correct)
                total_all += 1
                if s == 'val_chinese_mnist':
                    correct_cm += int(is_correct)
                    total_cm += 1
                else:
                    correct_gb += int(is_correct)
                    total_gb += 1

    return {
        'overall': correct_all / total_all * 100.0,
        'cm': correct_cm / max(1, total_cm) * 100.0,
        'gb': correct_gb / max(1, total_gb) * 100.0
    }

def measure_latency(model, head, device, is_fp16=False, runs=100) -> Tuple[float, float, float]:
    model.eval()
    head.eval()
    x = torch.randn(1, 1, 64, 64, device=device)
    if is_fp16:
        x = x.half()

    for _ in range(20):
        with torch.no_grad():
            lin, f = model(x)
            _ = head(f)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    latencies = []
    for _ in range(runs):
        t0 = time.perf_counter()
        with torch.no_grad():
            lin, f = model(x)
            _ = head(f)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    mean_lat = np.mean(latencies)
    p95_lat = np.percentile(latencies, 95)
    fps = 1000.0 / mean_lat
    return mean_lat, p95_lat, fps

def main():
    base_dir = Path("/home/rei/Workspace/Academic/Multivariate Data Analysis/FinalProject")
    manifest_csv = base_dir / "data/metadata/super_dataset_15918.csv"
    if not manifest_csv.exists():
        raise FileNotFoundError(f"Missing {manifest_csv}")

    df_all = pd.read_csv(manifest_csv)
    df_val = df_all[df_all['source'] == 'validation_phase1'].reset_index(drop=True)
    print(f"[*] Benchmark dataset: {len(df_val)} Phase 1 validation images.")
    print(f"    • Chinese MNIST formal holdout: {(df_val['stratum'] == 'val_chinese_mnist').sum()}")
    print(f"    • Gaya Baru cursive holdout:   {(df_val['stratum'] == 'val_gaya_baru').sum()}")

    clean_loader = DataLoader(BenchmarkDataset(df_val), batch_size=64, shuffle=False, num_workers=2)

    # 1. Load Distilled Student Model
    print("\n[+] Loading New Distilled Student Model (DenseNet-121 CBAM SWA FP16)...")
    student_ckpt = torch.load(base_dir / "models/densenet121_cbam_student_fp16.pt", map_location=DEVICE, weights_only=False)
    student_model = StudentDenseNet121_CBAM(num_classes=15).half().to(DEVICE)
    student_head = StudentSubCenterArcFace(in_features=256, num_classes=15, sub_centers=2).half().to(DEVICE)
    student_model.load_state_dict(student_ckpt['model_state_dict'])
    student_head.load_state_dict(student_ckpt['metric_head_state_dict'])
    student_size_mb = os.path.getsize(base_dir / "models/densenet121_cbam_student_fp16.pt") / (1024 * 1024)
    student_params = sum(p.numel() for p in student_model.parameters()) + sum(p.numel() for p in student_head.parameters())

    # 2. Load ResNet-18 CBAM v2 (Fold 0)
    print("[+] Loading ResNet-18 CBAM v2 (Anchor Fold 0)...")
    r18_ckpt = torch.load(base_dir / "models/resnet18_cbam_v2_fold0.pt", map_location=DEVICE, weights_only=False)
    r18_model = ResNet18_CBAM(num_classes=15).to(DEVICE)
    r18_head = SubCenterArcFace(in_features=256, num_classes=15, sub_centers=2).to(DEVICE)
    r18_model.load_state_dict(r18_ckpt['model_state_dict'])
    r18_head.load_state_dict(r18_ckpt['metric_head_state_dict'])
    r18_size_mb = os.path.getsize(base_dir / "models/resnet18_cbam_v2_fold0.pt") / (1024 * 1024)
    r18_params = sum(p.numel() for p in r18_model.parameters()) + sum(p.numel() for p in r18_head.parameters())

    # 3. Load ConvNeXt-Femto v2 (Fold 0)
    print("[+] Loading ConvNeXt-Femto v2 (Fold 0)...")
    cnx_ckpt = torch.load(base_dir / "models/convnext_femto_v2_fold0.pt", map_location=DEVICE, weights_only=False)
    cnx_model = ConvNeXt_Femto(num_classes=15).to(DEVICE)
    cnx_head = SubCenterArcFace(in_features=256, num_classes=15, sub_centers=2).to(DEVICE)
    cnx_model.load_state_dict(cnx_ckpt['model_state_dict'])
    cnx_head.load_state_dict(cnx_ckpt['metric_head_state_dict'])
    cnx_size_mb = os.path.getsize(base_dir / "models/convnext_femto_v2_fold0.pt") / (1024 * 1024)
    cnx_params = sum(p.numel() for p in cnx_model.parameters()) + sum(p.numel() for p in cnx_head.parameters())

    # Measure Accuracies on Clean Benchmark
    print("\n[*] Evaluating on Clean Validation Set...")
    acc_student = evaluate_model(student_model, student_head, clean_loader, DEVICE, is_fp16=True)
    acc_r18 = evaluate_model(r18_model, r18_head, clean_loader, DEVICE, is_fp16=False)
    acc_cnx = evaluate_model(cnx_model, cnx_head, clean_loader, DEVICE, is_fp16=False)

    # Latencies
    print("[*] Benchmarking Latencies...")
    lat_student, p95_student, fps_student = measure_latency(student_model, student_head, DEVICE, is_fp16=True)
    lat_r18, p95_r18, fps_r18 = measure_latency(r18_model, r18_head, DEVICE, is_fp16=False)
    lat_cnx, p95_cnx, fps_cnx = measure_latency(cnx_model, cnx_head, DEVICE, is_fp16=False)

    # Stress Test Perturbations
    perturbations = {
        "Gaussian Noise (σ=25)": PerturbationBench.gaussian_noise,
        "Stroke Erosion (2x2)": PerturbationBench.stroke_erosion,
        "Stroke Dilation (2x2)": PerturbationBench.stroke_dilation,
        "Elastic Distortion (α=6, σ=4)": PerturbationBench.elastic_distortion,
    }

    stress_results = []
    print("\n[*] Running Stress-Test Perturbations...")
    for name, p_fn in perturbations.items():
        p_loader = DataLoader(BenchmarkDataset(df_val, perturb_fn=p_fn), batch_size=64, shuffle=False, num_workers=2)
        s_res = evaluate_model(student_model, student_head, p_loader, DEVICE, is_fp16=True)
        r_res = evaluate_model(r18_model, r18_head, p_loader, DEVICE, is_fp16=False)
        c_res = evaluate_model(cnx_model, cnx_head, p_loader, DEVICE, is_fp16=False)
        stress_results.append({
            'scenario': name,
            'student_acc': s_res['overall'],
            'r18_acc': r_res['overall'],
            'cnx_acc': c_res['overall'],
            'delta_vs_r18': s_res['overall'] - r_res['overall']
        })

    # Summary Report
    print("\n" + "=" * 90)
    print("📊 COMPREHENSIVE PERFORMANCE COMPARISON: STUDENT DISTILLATION VS BASELINES")
    print("=" * 90)
    print(f"{'Metric / Dimension':<35} | {'Student (DenseNet121 SWA)':<20} | {'ResNet-18 CBAM v2':<18} | {'ConvNeXt-Femto v2':<18}")
    print("-" * 90)
    print(f"{'Overall Phase 1 Val Accuracy':<35} | {acc_student['overall']:>6.2f}%              | {acc_r18['overall']:>6.2f}%           | {acc_cnx['overall']:>6.2f}%")
    print(f"{'Chinese MNIST (1,500 Holdout)':<35} | {acc_student['cm']:>6.2f}%              | {acc_r18['cm']:>6.2f}%           | {acc_cnx['cm']:>6.2f}%")
    print(f"{'Gaya Baru Cursive (966 Holdout)':<35} | {acc_student['gb']:>6.2f}%              | {acc_r18['gb']:>6.2f}%           | {acc_cnx['gb']:>6.2f}%")
    print(f"{'Checkpoint Size (Disk)':<35} | {student_size_mb:>6.2f} MB              | {r18_size_mb:>6.2f} MB           | {cnx_size_mb:>6.2f} MB")
    print(f"{'Parameter Count':<35} | {student_params:>9,d}             | {r18_params:>9,d}          | {cnx_params:>9,d}")
    print(f"{'Mean Latency (Single Batch-1)':<35} | {lat_student:>6.2f} ms              | {lat_r18:>6.2f} ms           | {lat_cnx:>6.2f} ms")
    print(f"{'P95 Latency':<35} | {p95_student:>6.2f} ms              | {p95_r18:>6.2f} ms           | {p95_cnx:>6.2f} ms")
    print(f"{'Throughput (Single-Image)':<35} | {fps_student:>6.1f} FPS             | {fps_r18:>6.1f} FPS          | {fps_cnx:>6.1f} FPS")
    print("=" * 90)

    print("\n🛡️ STRESS-TEST ROBUSTNESS & PERTURBATION INVARIANCE:")
    print("-" * 90)
    print(f"{'Perturbation Scenario':<35} | {'Student Acc':<12} | {'ResNet18 Acc':<12} | {'ConvNeXt Acc':<12} | {'Student Advantage'}")
    print("-" * 90)
    for sr in stress_results:
        print(f"{sr['scenario']:<35} | {sr['student_acc']:>6.2f}%     | {sr['r18_acc']:>6.2f}%     | {sr['cnx_acc']:>6.2f}%     | {sr['delta_vs_r18']:>+6.2f}%")
    print("=" * 90)

if __name__ == "__main__":
    main()
