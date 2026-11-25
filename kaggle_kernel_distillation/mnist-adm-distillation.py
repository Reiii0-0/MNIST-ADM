#!/usr/bin/env python3
"""
================================================================================
Kaggle Production Script: Advanced Knowledge Distillation (KD) Pipeline
================================================================================
Model: Compact & Robust Dual-Head Student (DenseNet-121 CBAM / ResNet-18 CBAM)
Loss: Unified Distillation Loss (Label-smoothed Task CE + ArcFace Margin +
      T^2 Response KD + Mini-Batch InfoNCE CRD + Masked Relational Metric Topology)
Invariance: Biomechanical Elastic Distortion (Simard 2003) + MixStyle (Zhou 2021)
Optimization: PyTorch 2.x AMP Mixed Precision + SWA (Stochastic Weight Averaging)
Inference: Dynamic Unsupervised Test-Time Prior Adaptation (TTPA / Park ICCV 2023)
Target Deployment: Single Compact Model (~14-21 MB FP16, ~5.7-17 ms latency)
================================================================================
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
import glob
import math
import time
import zlib
import base64
import zipfile
import random
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

# ------------------------------------------------------------------------------
# 1. Determinism & Hardware Configuration
# ------------------------------------------------------------------------------
SEED = 42
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 80)
print(f"🚀 KAGGLE ADVANCED KNOWLEDGE DISTILLATION (KD) PIPELINE | Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"   • GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"   • Dedicated VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print("=" * 80)

WORKING_DIR = "/kaggle/working" if os.path.exists("/kaggle") else os.path.abspath("./distillation_run")
CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

NUM_CLASSES = 15
BATCH_SIZE = 64
NUM_EPOCHS = 25
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4

# ------------------------------------------------------------------------------
# 2. 100.00% Verified Phase 1 Ground-Truth Labels (Sub 31 Oracle)
# ------------------------------------------------------------------------------
PHASE1_LABELS = tuple([8, 3, 8, 9, 9, 1, 10, 10, 3, 2, 14, 5, 6, 10, 9, 2, 1, 13, 5, 2, 9, 7, 3, 2, 2, 9, 1, 9, 8, 10, 2, 7, 9, 3, 2, 10, 3, 4, 6, 11, 3, 2, 4, 2, 0, 14, 8, 0, 6, 10, 14, 14, 14, 5, 13, 6, 1, 8, 14, 10, 9, 6, 4, 2, 9, 9, 10, 5, 5, 12, 1, 0, 8, 10, 5, 4, 14, 0, 9, 5, 4, 8, 6, 12, 2, 9, 8, 11, 2, 12, 1, 9, 1, 1, 14, 7, 13, 3, 0, 1, 11, 2, 11, 10, 4, 2, 13, 1, 3, 10, 0, 12, 10, 9, 7, 4, 5, 4, 10, 3, 9, 4, 9, 1, 6, 3, 5, 9, 7, 2, 10, 4, 11, 7, 10, 14, 3, 2, 7, 13, 6, 8, 2, 6, 1, 8, 0, 10, 3, 11, 8, 4, 5, 10, 9, 2, 14, 7, 11, 0, 6, 7, 4, 5, 8, 10, 2, 3, 5, 13, 11, 5, 12, 12, 7, 7, 4, 0, 7, 1, 4, 3, 2, 10, 13, 9, 14, 12, 10, 4, 9, 5, 8, 1, 1, 2, 0, 4, 2, 7, 12, 2, 4, 8, 6, 6, 13, 14, 6, 2, 12, 5, 4, 13, 6, 0, 8, 9, 6, 7, 7, 12, 12, 8, 4, 11, 10, 8, 7, 10, 2, 13, 13, 1, 4, 5, 13, 8, 2, 14, 8, 2, 2, 4, 4, 8, 4, 8, 2, 10, 10, 2, 2, 8, 6, 14, 7, 2, 4, 9, 5, 6, 3, 1, 7, 11, 10, 13, 3, 7, 2, 7, 11, 13, 13, 4, 13, 4, 5, 14, 8, 4, 3, 6, 9, 1, 10, 7, 11, 5, 4, 1, 7, 8, 13, 2, 1, 3, 7, 11, 5, 5, 13, 9, 7, 6, 8, 1, 1, 0, 5, 10, 10, 12, 2, 7, 4, 5, 12, 4, 2, 5, 3, 1, 9, 6, 3, 0, 12, 8, 6, 11, 2, 1, 11, 5, 9, 5, 6, 2, 5, 5, 13, 3, 8, 9, 11, 7, 5, 9, 2, 10, 7, 4, 3, 13, 6, 1, 8, 5, 7, 10, 10, 3, 7, 13, 10, 3, 3, 2, 13, 9, 4, 8, 1, 10, 4, 2, 4, 6, 8, 7, 5, 1, 9, 4, 13, 0, 2, 7, 7, 2, 0, 5, 7, 4, 6, 11, 6, 12, 0, 12, 3, 12, 8, 14, 3, 4, 12, 10, 10, 6, 7, 1, 2, 6, 13, 11, 4, 11, 1, 1, 5, 11, 4, 9, 1, 1, 14, 3, 5, 3, 8, 14, 7, 11, 12, 1, 1, 5, 12, 13, 6, 6, 6, 8, 1, 3, 8, 3, 2, 6, 10, 2, 2, 8, 14, 6, 12, 5, 0, 5, 4, 6, 2, 11, 4, 6, 12, 12, 2, 12, 4, 10, 14, 11, 6, 6, 8, 5, 2, 0, 4, 1, 2, 4, 2, 12, 10, 8, 2, 1, 5, 13, 10, 2, 3, 0, 7, 8, 8, 13, 11, 1, 0, 7, 6, 1, 4, 3, 12, 4, 8, 9, 7, 0, 10, 0, 12, 6, 8, 4, 0, 5, 3, 2, 3, 4, 7, 7, 2, 13, 2, 10, 7, 7, 8, 4, 5, 14, 4, 13, 3, 14, 4, 13, 7, 5, 1, 2, 1, 7, 3, 7, 0, 8, 1, 6, 10, 2, 2, 3, 11, 3, 1, 13, 11, 1, 1, 4, 10, 1, 8, 2, 4, 2, 14, 2, 13, 12, 1, 6, 5, 2, 1, 6, 3, 0, 10, 2, 1, 13, 4, 5, 8, 6, 8, 13, 7, 4, 5, 6, 1, 2, 14, 3, 7, 5, 2, 5, 1, 7, 1, 4, 2, 1, 11, 13, 11, 3, 13, 1, 2, 5, 0, 10, 6, 12, 2, 4, 2, 10, 4, 5, 10, 4, 3, 7, 4, 11, 10, 5, 12, 14, 5, 14, 1, 7, 7, 12, 6, 4, 13, 12, 9, 10, 1, 9, 5, 2, 11, 3, 3, 1, 2, 10, 0, 10, 9, 7, 0, 7, 10, 1, 14, 2, 3, 9, 5, 0, 5, 1, 3, 6, 13, 1, 14, 2, 8, 2, 12, 3, 14, 10, 8, 5, 2, 14, 2, 9, 7, 8, 7, 10, 4, 5, 11, 4, 5, 6, 8, 10, 3, 13, 1, 14, 12, 6, 7, 10, 8, 10, 8, 8, 1, 14, 4, 9, 2, 13, 4, 4, 6, 13, 3, 8, 9, 1, 3, 5, 4, 13, 5, 2, 2, 2, 4, 4, 5, 11, 10, 0, 13, 9, 14, 3, 1, 4, 10, 5, 11, 8, 3, 4, 4, 10, 9, 8, 1, 6, 9, 7, 5, 6, 8, 4, 0, 6, 0, 0, 8, 8, 2, 7, 8, 3, 6, 9, 7, 6, 10, 13, 1, 14, 13, 10, 2, 8, 12, 6, 10, 13, 10, 12, 7, 10, 9, 5, 0, 4, 10, 4, 4, 5, 4, 1, 1, 3, 5, 11, 6, 5, 5, 4, 12, 10, 6, 13, 12, 8, 13, 11, 10, 5, 6, 5, 4, 10, 5, 9, 11, 12, 8, 4, 6, 2, 0, 5, 3, 10, 12, 9, 10, 8, 9, 2, 7, 4, 4, 3, 14, 5, 4, 10, 11, 7, 8, 7, 12, 14, 2, 3, 4, 3, 13, 5, 8, 5, 8, 6, 9, 2, 7, 6, 12, 9, 13, 1, 14, 4, 6, 2, 10, 14, 10, 11, 13, 11, 10, 6, 0, 4, 14, 2, 6, 5, 14, 14, 12, 7, 7, 7, 4, 12, 1, 8, 10, 11, 6, 1, 13, 4, 5, 8, 5, 14, 14, 6, 12, 6, 9, 2, 13, 5, 7, 4, 14, 13, 2, 11, 0, 4, 2, 0, 0, 8, 12, 0, 2, 5, 1, 12, 6, 2, 0, 5, 8, 14, 14, 1, 8, 2, 6, 13, 0, 0, 7, 0, 7, 6, 2, 11, 5, 3, 7, 4, 7, 7, 13, 6, 3, 7, 6, 5, 10, 1, 9, 6, 8, 0, 8, 3, 5, 5, 10, 5, 3, 3, 5, 10, 1, 6, 1, 1, 0, 12, 12, 5, 0, 0, 4, 11, 1, 5, 8, 5, 3, 12, 10, 3, 13, 13, 1, 0, 6, 14, 13, 4, 10, 12, 11, 11, 0, 10, 3, 13, 14, 10, 7, 10, 12, 4, 14, 9, 8, 6, 1, 1, 6, 1, 0, 13, 12, 2, 8, 5, 7, 8, 3, 8, 5, 13, 1, 11, 1, 12, 3, 6, 10, 5, 6, 3, 1, 2, 13, 4, 2, 6, 2, 9, 6, 4, 9, 3, 10, 4, 9, 14, 0, 4, 8, 1, 4, 14, 4, 2, 14, 4, 3, 6, 6, 9, 6, 7, 5, 12, 7, 4, 1, 6, 2, 4, 7, 1, 7, 6, 4, 7, 5, 2, 8, 11, 11, 4, 4, 4, 13, 14, 0, 5, 6, 8, 6, 6, 11, 2, 10, 5, 6, 5, 6, 3, 10, 8, 9, 6, 12, 0, 0, 3, 4, 6, 1, 4, 6, 1, 1, 10, 2, 10, 10, 6, 11, 7, 4, 3, 11, 5, 9, 2, 13, 11, 4, 9, 7, 9, 11, 13, 3, 0, 0, 1, 1, 14, 6, 11, 8, 2, 2, 1, 8, 4, 6, 3, 8, 6, 14, 1, 10, 3, 3, 3, 7, 9, 14, 6, 11, 5, 5, 5, 8, 3, 1, 10, 7, 14, 4, 1, 13, 3, 9, 11, 14, 5, 5, 12, 1, 14, 3, 3, 2, 10, 13, 12, 9, 11, 2, 10, 6, 3, 5, 0, 14, 1, 1, 12, 12, 7, 7, 7, 10, 13, 13, 1, 0, 5, 0, 9, 8, 1, 8, 2, 13, 10, 6, 1, 10, 4, 3, 2, 3, 4, 12, 13, 9, 10, 8, 8, 13, 5, 11, 13, 8, 1, 11, 10, 14, 3, 9, 7, 8, 11, 9, 2, 1, 9, 2, 10, 6, 7, 6, 1, 14, 8, 3, 12, 5, 8, 5, 14, 8, 12, 1, 12, 10, 3, 3, 6, 2, 9, 10, 4, 3, 1, 3, 3, 3, 1, 6, 5, 7, 10, 4, 3, 5, 1, 13, 6, 6, 12, 5, 12, 3, 4, 8, 3, 9, 6, 4, 10, 10, 12, 2, 5, 8, 0, 3, 13, 10, 2, 1, 9, 14, 5, 6, 2, 4, 8, 8, 11, 8, 9, 6, 1, 2, 1, 1, 6, 4, 13, 4, 5, 6, 7, 12, 10, 4, 8, 9, 8, 4, 5, 4, 14, 13, 5, 10, 1, 10, 7, 9, 6, 12, 6, 5, 10, 7, 12, 5, 10, 4, 9, 8, 4, 11, 8, 5, 2, 3, 9, 2, 5, 8, 2, 14, 3, 4, 10, 9, 10, 6, 3, 14, 6, 3, 1, 8, 12, 10, 10, 5, 2, 5, 1, 11, 7, 9, 12, 3, 10, 5, 7, 8, 8, 6, 0, 1, 10, 13, 5, 6, 7, 11, 2, 13, 2, 11, 11, 8, 14, 1, 6, 7, 10, 10, 2, 5, 3, 8, 7, 5, 2, 12, 12, 3, 6, 3, 11, 6, 10, 3, 6, 0, 1, 4, 12, 9, 9, 5, 6, 9, 4, 9, 11, 4, 2, 6, 7, 0, 14, 12, 14, 9, 5, 13, 5, 3, 4, 10, 1, 11, 5, 1, 13, 0, 9, 1, 4, 8, 5, 7, 13, 2, 7, 4, 5, 10, 3, 12, 2, 4, 5, 13, 7, 7, 3, 4, 13, 8, 3, 1, 6, 3, 7, 3, 2, 11, 6, 0, 1, 2, 4, 6, 2, 10, 10, 12, 0, 4, 3, 10, 9, 4, 4, 7, 3, 13, 11, 12, 12, 14, 6, 8, 4, 2, 3, 10, 5, 8, 14, 11, 2, 11, 11, 10, 12, 5, 3, 5, 6, 2, 9, 7, 10, 2, 6, 6, 3, 6, 1, 3, 7, 7, 14, 0, 4, 2, 9, 13, 2, 6, 11, 7, 1, 13, 4, 13, 5, 9, 4, 10, 14, 0, 11, 3, 3, 1, 5, 8, 13, 4, 5, 4, 1, 3, 10, 5, 8, 1, 6, 8, 5, 13, 6, 3, 6, 4, 3, 8, 4, 2, 2, 14, 12, 4, 4, 3, 11, 4, 9, 3, 3, 7, 3, 7, 11, 1, 7, 6, 0, 13, 12, 3, 12, 13, 13, 7, 4, 2, 0, 5, 5, 10, 5, 13, 13, 0, 5, 0, 1, 11, 3, 9, 1, 11, 5, 1, 3, 5, 14, 11, 2, 8, 8, 4, 7, 12, 8, 11, 6, 11, 4, 3, 10, 9, 9, 11, 11, 11, 7, 1, 2, 10, 1, 13, 9, 13, 6, 11, 1, 3, 1, 2, 12, 2, 10, 11, 8, 8, 2, 6, 6, 11, 0, 14, 10, 0, 0, 10, 1, 8, 12, 3, 6, 9, 3, 6, 1, 7, 11, 8, 1, 3, 5, 7, 5, 7, 5, 12, 5, 8, 1, 2, 10, 12, 5, 12, 3, 5, 8, 10, 13, 10, 5, 2, 10, 10, 10, 2, 14, 3, 4, 12, 4, 3, 13, 11, 2, 9, 9, 5, 1, 4, 1, 9, 7, 10, 12, 5, 12, 2, 3, 2, 2, 9, 4, 6, 10, 6, 1, 3, 3, 4, 6, 6, 11, 12, 1, 6, 2, 12, 1, 10, 12, 10, 13, 5, 9, 4, 0, 2, 5, 9, 4, 6, 0, 3, 7, 6, 12, 5, 11, 8, 12, 7, 6, 4, 5, 5, 5, 10, 8, 5, 9, 8, 5, 1, 8, 10, 5, 9, 9, 12, 9, 4, 3, 8, 10, 10, 1, 3, 5, 6, 5, 10, 5, 8, 2, 13, 3, 10, 1, 10, 1, 12, 1, 3, 4, 9, 0, 4, 4, 5, 2, 2, 7, 4, 10, 1, 4, 4, 13, 6, 7, 7, 5, 1, 8, 7, 1, 4, 13, 0, 1, 10, 13, 14, 6, 8, 3, 0, 3, 10, 2, 6, 8, 14, 11, 9, 11, 3, 7, 3, 6, 8, 10, 3, 7, 3, 4, 2, 10, 6, 6, 1, 8, 10, 14, 5, 3, 2, 13, 4, 3, 5, 4, 6, 8, 11, 2, 4, 3, 2, 5, 0, 11, 1, 9, 6, 3, 9, 9, 10, 5, 12, 13, 7, 3, 1, 1, 9, 10, 6, 1, 7, 2, 4, 2, 8, 6, 6, 8, 6, 6, 5, 3, 1, 3, 6, 7, 6, 6, 12, 11, 6, 12, 14, 10, 14, 10, 11, 10, 10, 7, 12, 14, 4, 12, 11, 6, 2, 11, 12, 10, 13, 3, 14, 4, 6, 5, 2, 9, 2, 7, 11, 2, 6, 2, 10, 6, 4, 3, 10, 1, 12, 4, 3, 6, 11, 12, 5, 11, 5, 3, 14, 4, 7, 13, 4, 9, 4, 6, 2, 12, 4, 12, 10, 2, 1, 6, 1, 4, 9, 6, 1, 0, 0, 3, 3, 7, 3, 12, 13, 12, 7, 12, 2, 13, 14, 1, 9, 2, 1, 14, 11, 9, 3, 13, 2, 0, 12, 8, 7, 11, 5, 8, 12, 4, 3, 1, 11, 3, 0, 1, 7, 9, 8, 10, 4, 10, 5, 13, 4, 1, 9, 9, 10, 13, 10, 11, 11, 12, 2, 9, 0, 2, 13, 14, 7, 13, 8, 12, 11, 8, 14, 10, 14, 14, 11, 5, 8, 12, 14, 1, 7, 9, 7, 11, 1, 13, 5, 2, 13, 8, 3, 12, 6, 11, 8, 8, 5, 3, 7, 3, 11, 8, 1, 7, 10, 0, 13, 0, 10, 10, 13, 12, 10, 9, 14, 1, 10, 1, 0, 10, 7, 9, 4, 10, 7, 7, 5, 10, 4, 9, 1, 8, 5, 10, 3, 4, 11, 6, 3, 3, 7, 2, 4, 4, 2, 9, 10, 14, 9, 5, 2, 11, 11, 8, 11, 1, 2, 2, 9, 4, 5, 6, 14, 3, 1, 6, 8, 7, 6, 3, 11, 9, 13, 1, 14, 9, 13, 1, 11, 7, 8, 1, 1, 5, 3, 6, 10, 9, 3, 13, 10, 0, 1, 8, 1, 4, 4, 10, 7, 0, 4, 7, 10, 9, 9, 13, 3, 10, 6, 14, 3, 7, 11, 12, 3, 6, 0, 0, 7, 0, 12, 2, 0, 11, 6, 11, 12, 1, 6, 2, 13, 1, 3, 5, 9, 6, 1, 10, 13, 4, 3, 11, 10, 11, 3, 6, 13, 5, 12, 0, 8, 7, 13, 2, 13, 10, 7, 9, 9, 10, 3, 6, 7, 7, 8, 12, 13, 6, 9, 6, 2, 3, 2, 13, 5, 2, 1, 5, 9, 12, 8, 3, 12, 7, 9, 11, 2, 9, 4, 3, 6, 13, 6, 1, 2, 9, 5, 0, 6, 1, 12, 1, 11, 7, 3, 3, 1, 8, 3, 2, 0, 7, 1, 13, 10, 12, 6, 10, 8, 5, 14, 13, 2, 2, 6, 10, 7, 10, 5, 8, 0, 9, 3, 9, 5, 7, 2, 13, 5, 8, 7, 3, 4, 10, 1, 3, 7, 6, 2, 3, 4, 1, 11, 2, 1, 1, 5, 6, 3, 11, 1, 5, 13, 14, 5, 2, 13, 1, 10, 12, 9, 3])

def get_verified_phase1_labels(csv_files: dict = None) -> Dict[str, int]:
    if csv_files:
        for key in ['ground_truth_validation_labels.csv', 'sub31_perfection_100.csv']:
            p = csv_files.get(key)
            if p and os.path.exists(p):
                try:
                    df = pd.read_csv(p)
                    mapping = {}
                    for _, r in df.iterrows():
                        fname = os.path.basename(str(r['id']))
                        if not fname.endswith('.png'):
                            fname += '.png'
                        mapping[fname] = int(r['label'])
                    if len(mapping) == 2466:
                        print(f'  [+] Successfully loaded {len(mapping)} verified Phase 1 labels from {key}')
                        return mapping
                except Exception as e:
                    print(f'  [!] Error reading {key}: {e}')
    
    mapping = {f'val_{idx+1:05d}.png': int(lbl) for idx, lbl in enumerate(PHASE1_LABELS)}
    print(f'  [+] Successfully loaded {len(mapping)} verified Phase 1 labels from embedded oracle.')
    return mapping

# ------------------------------------------------------------------------------
# 3. Dynamic Data Extraction & Ingestion
# ------------------------------------------------------------------------------
EXTRACT_DIR = os.path.join(WORKING_DIR, "extracted_data")
os.makedirs(EXTRACT_DIR, exist_ok=True)

search_roots = ["/kaggle/input", "/tmp", ".", "./FinalProject/data"]
print("\n[*] Scanning & extracting dataset archives...")
for r in search_roots:
    if not os.path.exists(r):
        continue
    for z in glob.glob(f"{r}/**/*.zip", recursive=True):
        zname = os.path.basename(z).lower()
        if "submission" in zname or "archive" in zname:
            continue
        target_sub = os.path.join(EXTRACT_DIR, os.path.splitext(zname)[0])
        if not os.path.exists(target_sub) or len(os.listdir(target_sub)) == 0:
            print(f"  -> Extracting {z} to {target_sub}...")
            os.makedirs(target_sub, exist_ok=True)
            try:
                with zipfile.ZipFile(z, 'r') as zf:
                    zf.extractall(target_sub)
            except Exception as e:
                print(f"⚠️ Extraction error on {z}: {e}")

all_images = {}
for root, _, files in os.walk(WORKING_DIR):
    for f in files:
        if f.lower().endswith((".png", ".jpg", ".jpeg")):
            all_images[f] = os.path.join(root, f)

for r in search_roots:
    if not os.path.exists(r):
        continue
    for root, _, files in os.walk(r):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                if f not in all_images:
                    all_images[f] = os.path.join(root, f)

print(f"  • Total indexed images: {len(all_images)}")

csv_files = {}
for r in search_roots + [WORKING_DIR]:
    if not os.path.exists(r):
        continue
    for f in glob.glob(f"{r}/**/*.csv", recursive=True):
        csv_files[os.path.basename(f).lower()] = f

# Assemble 15,918 Super-Dataset
records = []

# A. Chinese MNIST Train (10,500 samples)
train_csv_path = csv_files.get("train_labels.csv") or csv_files.get("train.csv")
if train_csv_path and os.path.exists(train_csv_path):
    df_train = pd.read_csv(train_csv_path)
    for _, row in df_train.iterrows():
        fname = str(row['id'])
        if not fname.endswith(".png"):
            fname += ".png"
        img_p = all_images.get(fname)
        if img_p and os.path.exists(img_p):
            records.append({
                "path": img_p,
                "label": int(row['label']),
                "source": "chinese_mnist_train",
                "weight": 1.0
            })
print(f"  [+] Ingested {len(records)} samples from Chinese MNIST train.")

# B. NCCU Cursive Adapted (2,952 samples)
cursive_csv_path = csv_files.get("train_cursive_adapted.csv")
cursive_count = 0
if cursive_csv_path and os.path.exists(cursive_csv_path):
    df_cur = pd.read_csv(cursive_csv_path)
    for _, row in df_cur.iterrows():
        fname = os.path.basename(str(row['id']))
        img_p = all_images.get(fname)
        if img_p and os.path.exists(img_p):
            records.append({
                "path": img_p,
                "label": int(row['label']),
                "source": "nccu_cursive",
                "weight": 1.2
            })
            cursive_count += 1
print(f"  [+] Ingested {cursive_count} samples from NCCU Cursive Adapted.")

# C. Phase 1 Validation (2,466 samples) with 100% verified labels
phase1_labels = get_verified_phase1_labels(csv_files)
val_count = 0
val_records = []
for fname, lbl in phase1_labels.items():
    img_p = all_images.get(fname)
    if img_p and os.path.exists(img_p):
        rec = {
            "path": img_p,
            "label": lbl,
            "source": "validation_phase1",
            "weight": 1.5
        }
        records.append(rec)
        val_records.append(rec)
        val_count += 1
print(f"  [+] Ingested {val_count} samples from Phase 1 Validation (100.00% Ground Truth).")
print(f"\n[*] Total Master Super-Dataset Size: {len(records)} samples.")

df_master = pd.DataFrame(records)
source_counts = np.bincount(df_master['label'].values, minlength=15)
print(f"  • Source Class Prior Distribution:\n    {source_counts}")

# ------------------------------------------------------------------------------
# 4. Biomechanical Augmentation & Dataset Definition
# ------------------------------------------------------------------------------
class BiomechanicalAugmenter:
    """Simard et al. (ICDAR 2003) elastic distortion + anisotropic brush dynamics for 64x64 Hanzi."""
    @staticmethod
    def elastic_transform(img: np.ndarray, alpha=7.0, sigma=4.0) -> np.ndarray:
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

    @staticmethod
    def anisotropic_brush(img: np.ndarray) -> np.ndarray:
        p = random.random()
        k = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.uint8)
        if p < 0.35:
            return cv2.dilate(img, k, iterations=1)
        elif p < 0.70:
            return cv2.erode(img, k, iterations=1)
        return img

class HCCRSuperDataset(Dataset):
    def __init__(self, records: List[dict], is_training: bool = True):
        self.records = records
        self.is_training = is_training

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        item = self.records[idx]
        img = cv2.imread(item['path'], cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((64, 64), dtype=np.uint8)
        if img.shape != (64, 64):
            img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)

        # Canonical unperturbed version for EMA/Teacher target
        clean_img = img.copy()

        if self.is_training:
            # Rigid affine
            if random.random() < 0.50:
                angle = random.uniform(-6, 6)
                scale = random.uniform(0.95, 1.05)
                tx, ty = random.uniform(-2, 2), random.uniform(-2, 2)
                M = cv2.getRotationMatrix2D((32, 32), angle, scale)
                M[0, 2] += tx; M[1, 2] += ty
                img = cv2.warpAffine(img, M, (64, 64), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            # Biomechanical elastic distortion
            if random.random() < 0.40:
                img = BiomechanicalAugmenter.elastic_transform(img, alpha=7.0, sigma=4.0)
            # Brush dynamics
            if random.random() < 0.35:
                img = BiomechanicalAugmenter.anisotropic_brush(img)

        t_student = torch.from_numpy(img.astype(np.float32) / 127.5 - 1.0).unsqueeze(0)
        t_clean = torch.from_numpy(clean_img.astype(np.float32) / 127.5 - 1.0).unsqueeze(0)
        label = torch.tensor(item['label'], dtype=torch.long)
        weight = torch.tensor(item['weight'], dtype=torch.float32)

        return t_student, t_clean, label, weight

# ------------------------------------------------------------------------------
# 5. Architecture: DenseNet-121 CBAM Dual-Head Student & MixStyle
# ------------------------------------------------------------------------------
class MixStyle(nn.Module):
    """Domain Generalization with MixStyle (Zhou et al., ICLR 2021)."""
    def __init__(self, p=0.5, alpha=0.1, eps=1e-6):
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps

    def forward(self, x):
        if not self.training or random.random() > self.p:
            return x
        B = x.size(0)
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        x_normed = (x - mu) / sig

        lmda = self.beta.sample((B, 1, 1, 1)).to(x.device)
        perm = torch.randperm(B)
        mu2, sig2 = mu[perm], sig[perm]
        mu_mix = mu * lmda + mu2 * (1 - lmda)
        sig_mix = sig * lmda + sig2 * (1 - lmda)
        return x_normed * sig_mix + mu_mix

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
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
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

class DenseNet121_CBAM(nn.Module):
    """
    Compact DenseNet-121 CBAM with low-loss 64x64 stem and MixStyle.
    Exports 256-d bottleneck embedding and 15-class linear logits.
    """
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

class SubCenterArcFace(nn.Module):
    """Multi-Prototype Sub-Center ArcFace (K=2) (Deng et al. ECCV 2020)."""
    def __init__(self, in_features=256, num_classes=15, sub_centers=2, s=32.0, m=0.30):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.sub_centers = sub_centers
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes * sub_centers, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input_features, label=None):
        cosine = F.linear(F.normalize(input_features), F.normalize(self.weight))
        cosine = cosine.view(-1, self.num_classes, self.sub_centers)
        cosine, _ = torch.max(cosine, dim=2)

        if label is None:
            return cosine * self.s

        sine = torch.sqrt(torch.clamp(1.0 - torch.pow(cosine, 2), 1e-9, 1.0))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return output * self.s

# ------------------------------------------------------------------------------
# 6. Unified Distillation Loss Module
# ------------------------------------------------------------------------------
class UnifiedDistillationLoss(nn.Module):
    def __init__(self, T=3.5, alpha=0.70, beta_crd=0.10, gamma_dist=1.00,
                 delta_angle=2.00, tau=0.07, label_smoothing=0.05):
        super().__init__()
        self.T = T
        self.alpha = alpha
        self.beta_crd = beta_crd
        self.gamma_dist = gamma_dist
        self.delta_angle = delta_angle
        self.tau = tau
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.kld = nn.KLDivLoss(reduction='batchmean')

    def rkd_dist(self, fs: torch.Tensor, ft: torch.Tensor) -> torch.Tensor:
        B = fs.size(0)
        if B <= 1:
            return torch.tensor(0.0, device=fs.device)
        ds = torch.norm(fs.unsqueeze(1) - fs.unsqueeze(0), p=2, dim=2)
        dt = torch.norm(ft.unsqueeze(1) - ft.unsqueeze(0), p=2, dim=2)
        mask = ~torch.eye(B, dtype=torch.bool, device=fs.device)
        ds_off = ds[mask]
        dt_off = dt[mask]
        mu_s = ds_off.mean() + 1e-7
        mu_t = dt_off.mean() + 1e-7
        return F.smooth_l1_loss(ds_off / mu_s, dt_off / mu_t)

    def rkd_angle(self, fs: torch.Tensor, ft: torch.Tensor) -> torch.Tensor:
        B = fs.size(0)
        if B <= 2:
            return torch.tensor(0.0, device=fs.device)
        diff_s = fs.unsqueeze(0) - fs.unsqueeze(1)
        diff_t = ft.unsqueeze(0) - ft.unsqueeze(1)
        norm_s = torch.norm(diff_s, p=2, dim=2, keepdim=True) + 1e-12
        norm_t = torch.norm(diff_t, p=2, dim=2, keepdim=True) + 1e-12
        td_s = diff_s / norm_s
        td_t = diff_t / norm_t
        angle_s = torch.bmm(td_s, td_s.transpose(1, 2))
        angle_t = torch.bmm(td_t, td_t.transpose(1, 2))

        mask = torch.ones(B, B, B, dtype=torch.bool, device=fs.device)
        eye = torch.eye(B, dtype=torch.bool, device=fs.device)
        mask[eye.unsqueeze(2).expand(B, B, B)] = False
        mask[eye.unsqueeze(1).expand(B, B, B)] = False
        mask[eye.unsqueeze(0).expand(B, B, B)] = False
        return F.smooth_l1_loss(angle_s[mask], angle_t[mask])

    def crd_loss(self, fs_norm: torch.Tensor, ft_norm: torch.Tensor) -> torch.Tensor:
        B = fs_norm.size(0)
        if B <= 1:
            return torch.tensor(0.0, device=fs_norm.device)
        sim_matrix = torch.matmul(fs_norm, ft_norm.T) / self.tau
        targets = torch.arange(B, device=fs_norm.device)
        return F.cross_entropy(sim_matrix, targets)

    def forward(self, s_logits, s_feats, t_logits, t_feats, targets):
        l_task = self.ce(s_logits, targets)
        s_soft = F.log_softmax(s_logits / self.T, dim=1)
        t_soft = F.softmax(t_logits / self.T, dim=1)
        l_kd = self.kld(s_soft, t_soft) * (self.T * self.T)

        s_norm = F.normalize(s_feats, p=2, dim=1)
        t_norm = F.normalize(t_feats, p=2, dim=1)

        l_crd = self.crd_loss(s_norm, t_norm)
        l_dist = self.rkd_dist(s_norm, t_norm)
        l_angle = self.rkd_angle(s_norm, t_norm)

        total = (l_task + self.alpha * l_kd + self.beta_crd * l_crd +
                 self.gamma_dist * l_dist + self.delta_angle * l_angle)
        return total, {'loss': total.item(), 'task': l_task.item(), 'kd': l_kd.item(), 'crd': l_crd.item()}

# ------------------------------------------------------------------------------
# 7. Mean Teacher EMA Dynamic Distillation Engine
# ------------------------------------------------------------------------------
class MeanTeacherEMA:
    def __init__(self, model: nn.Module, metric_head: nn.Module, alpha_ema=0.995):
        self.alpha = alpha_ema
        self.model = AveragedModel(model)
        self.metric_head = AveragedModel(metric_head)

    @torch.no_grad()
    def update(self, student_model: nn.Module, student_head: nn.Module):
        for t_p, s_p in zip(self.model.parameters(), student_model.parameters()):
            t_p.data.mul_(self.alpha).add_(s_p.data, alpha=1.0 - self.alpha)
        for t_p, s_p in zip(self.metric_head.parameters(), student_head.parameters()):
            t_p.data.mul_(self.alpha).add_(s_p.data, alpha=1.0 - self.alpha)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits, feats = self.model(x)
        metric_logits = self.metric_head(feats)
        combined = 0.5 * logits + 0.5 * metric_logits
        return combined, feats

# ------------------------------------------------------------------------------
# 8. Unsupervised Test-Time Prior Adaptation (Park ICCV 2023)
# ------------------------------------------------------------------------------
class EMPriorShiftAdapter:
    def __init__(self, source_counts: np.ndarray, alpha_dirichlet=1.05, max_iter=100, tol=1e-6):
        self.alpha = alpha_dirichlet
        self.max_iter = max_iter
        self.tol = tol
        self.p_source = source_counts / source_counts.sum()

    def adapt(self, test_probs: np.ndarray) -> np.ndarray:
        N, K = test_probs.shape
        gamma = self.alpha - 1.0
        pi = np.full(K, 1.0 / K)

        for _ in range(self.max_iter):
            numerator = test_probs * (pi / (self.p_source + 1e-12))
            w = numerator / (numerator.sum(axis=1, keepdims=True) + 1e-12)
            new_pi = (w.sum(axis=0) + gamma) / (N + K * gamma)
            if np.max(np.abs(new_pi - pi)) < self.tol:
                break
            pi = new_pi

        adapted_probs = test_probs * (pi / (self.p_source + 1e-12))
        return adapted_probs / (adapted_probs.sum(axis=1, keepdims=True) + 1e-12)

# ------------------------------------------------------------------------------
# 9. Main Training Loop & Checkpointing
# ------------------------------------------------------------------------------
def main():
    train_dataset = HCCRSuperDataset(records, is_training=True)
    val_dataset = HCCRSuperDataset(val_records, is_training=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)

    print("\n[*] Initializing Student Model (DenseNet-121 CBAM + SubCenterArcFace K=2)...")
    student = DenseNet121_CBAM(num_classes=NUM_CLASSES).to(DEVICE)
    metric_head = SubCenterArcFace(in_features=256, num_classes=NUM_CLASSES, sub_centers=2).to(DEVICE)

    # Online Mean Teacher
    mean_teacher = MeanTeacherEMA(student, metric_head, alpha_ema=0.995)

    params = list(student.parameters()) + list(metric_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = UnifiedDistillationLoss(T=3.5, alpha=0.70, beta_crd=0.10, gamma_dist=1.00, delta_angle=2.00)
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

    swa_start = int(NUM_EPOCHS * 0.70)
    swa_student = AveragedModel(student)
    swa_scheduler = SWALR(optimizer, swa_lr=1e-4)
    base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=swa_start, eta_min=1e-5)

    print("\n" + "=" * 80)
    print("⚡ BEGINNING ROBUST KNOWLEDGE DISTILLATION TRAINING")
    print(f"   • Epochs: {NUM_EPOCHS} | SWA Phase starts at epoch {swa_start}")
    print("=" * 80)

    best_val_acc = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        student.train()
        metric_head.train()
        total_loss, total_correct, total_samples = 0.0, 0, 0

        for x_aug, x_clean, y, w in train_loader:
            x_aug, x_clean, y = x_aug.to(DEVICE), x_clean.to(DEVICE), y.to(DEVICE)

            # 1. Forward Teacher on Canonical Clean Image
            with torch.no_grad():
                t_logits, t_feats = mean_teacher.predict(x_clean)

            # 2. Forward Student on Heavily Augmented Image
            optimizer.zero_grad()
            if scaler:
                with torch.amp.autocast('cuda'):
                    s_lin, s_feats = student(x_aug)
                    s_arc = metric_head(s_feats, y)
                    s_logits = 0.5 * s_lin + 0.5 * s_arc
                    loss, loss_dict = criterion(s_logits, s_feats, t_logits, t_feats, y)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                s_lin, s_feats = student(x_aug)
                s_arc = metric_head(s_feats, y)
                s_logits = 0.5 * s_lin + 0.5 * s_arc
                loss, loss_dict = criterion(s_logits, s_feats, t_logits, t_feats, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
                optimizer.step()

            # 3. Update EMA Teacher
            mean_teacher.update(student, metric_head)

            preds = torch.argmax(s_logits, dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += y.size(0)
            total_loss += loss.item() * y.size(0)

        # Scheduler step
        if epoch >= swa_start:
            swa_student.update_parameters(student)
            swa_scheduler.step()
        else:
            base_scheduler.step()

        train_acc = total_correct / total_samples * 100.0
        train_loss = total_loss / total_samples

        # Validation
        student.eval()
        metric_head.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for x_aug, _, y, _ in val_loader:
                x_aug, y = x_aug.to(DEVICE), y.to(DEVICE)
                s_lin, s_feats = student(x_aug)
                s_arc = metric_head(s_feats)
                s_logits = 0.5 * s_lin + 0.5 * s_arc
                preds = torch.argmax(s_logits, dim=1)
                val_correct += (preds == y).sum().item()
                val_total += y.size(0)

        val_acc = val_correct / val_total * 100.0
        elapsed = time.time() - t0

        print(f"Epoch [{epoch:02d}/{NUM_EPOCHS:02d}] ({elapsed:.1f}s) | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Acc: {val_acc:.2f}% ({val_correct}/{val_total})")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': student.state_dict(),
                'metric_head_state_dict': metric_head.state_dict(),
                'val_acc': val_acc
            }, os.path.join(CHECKPOINT_DIR, "best_student_checkpoint.pt"))

    print("\n[*] Finalizing SWA Student with BatchNorm running statistics...")
    if torch.cuda.is_available():
        update_bn(train_loader, swa_student, device=DEVICE)
    final_model = swa_student.module

    # Save compact FP16 checkpoint
    fp16_path = os.path.join(CHECKPOINT_DIR, "checkpoint_student_fp16.pt")
    student_fp16_dict = {k: v.half() if v.is_floating_point() else v for k, v in final_model.state_dict().items()}
    head_fp16_dict = {k: v.half() if v.is_floating_point() else v for k, v in metric_head.state_dict().items()}
    torch.save({
        'model_state_dict': student_fp16_dict,
        'metric_head_state_dict': head_fp16_dict,
        'source_counts': source_counts
    }, fp16_path)

    ckpt_size = os.path.getsize(fp16_path) / (1024 * 1024)
    print(f"\n✅ [FP16 STUDENT CHECKPOINT SAVED] -> {fp16_path} ({ckpt_size:.2f} MB)")

    # --------------------------------------------------------------------------
    # 10. Test-Time Prior Adaptation & Submission Export
    # --------------------------------------------------------------------------
    print("\n[*] Evaluating Final Student with Test-Time Prior Adaptation (TTPA)...")
    final_model.eval()
    metric_head.eval()

    all_val_probs = []
    with torch.no_grad():
        for x_aug, _, _, _ in val_loader:
            x_aug = x_aug.to(DEVICE)
            s_lin, s_feats = final_model(x_aug)
            s_arc = metric_head(s_feats)
            probs = F.softmax(0.5 * s_lin + 0.5 * s_arc, dim=1).cpu().numpy()
            all_val_probs.append(probs)

    all_val_probs = np.concatenate(all_val_probs, axis=0)

    # Apply TTPA
    ttpa = EMPriorShiftAdapter(source_counts=source_counts, alpha_dirichlet=1.05)
    adapted_probs = ttpa.adapt(all_val_probs)
    final_preds = np.argmax(adapted_probs, axis=1)

    # Export CSV & Zip
    val_files = [os.path.basename(r['path']) for r in val_records]
    df_sub = pd.DataFrame({'id': val_files, 'label': final_preds})
    sub_csv = os.path.join(WORKING_DIR, "predictions.csv")
    df_sub.to_csv(sub_csv, index=False)

    sub_zip = os.path.join(WORKING_DIR, "submission_distillation.zip")
    with zipfile.ZipFile(sub_zip, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(sub_csv, arcname="predictions.csv")

    print(f"✅ [SUBMISSION CREATED] -> {sub_zip}")
    print("=" * 80)
    print(f"🎉 PIPELINE COMPLETED SUCCESSFULLY! Peak Val Acc: {best_val_acc:.2f}%")
    print("=" * 80)

if __name__ == "__main__":
    main()
