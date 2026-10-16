#!/usr/bin/env python3
"""
Central Configuration & Path Management for Chinese MNIST ADM Project.
Provides unified, robust, and backward-compatible path resolution across all datasets,
metadata CSVs, outputs, models, and submission packages.
"""

import os
from pathlib import Path

# Project root: /path/to/FinalProject
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Base directories
DATA_DIR = PROJECT_ROOT / "data"
METADATA_DIR = DATA_DIR / "metadata"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
TOOLS_DIR = PROJECT_ROOT / "tools"
DOCS_DIR = PROJECT_ROOT / "dokumentasi"
REFS_DIR = PROJECT_ROOT / "references"

# Ensure essential output directories exist
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
os.makedirs(TOOLS_DIR, exist_ok=True)

# -------------------------------------------------------------------------
# Metadata Path Resolvers (Check data/metadata/ first, fallback to root)
# -------------------------------------------------------------------------
def get_classes_csv() -> Path:
    p = METADATA_DIR / "classes.csv"
    return p if p.exists() else PROJECT_ROOT / "classes.csv"

def get_train_labels_csv() -> Path:
    p = METADATA_DIR / "train_labels.csv"
    return p if p.exists() else PROJECT_ROOT / "train_labels.csv"

def get_splits_kfold_csv() -> Path:
    p = METADATA_DIR / "splits_kfold.csv"
    return p if p.exists() else PROJECT_ROOT / "splits_kfold.csv"

def get_train_synthetic_labels_csv() -> Path:
    p = METADATA_DIR / "train_synthetic_labels.csv"
    return p if p.exists() else PROJECT_ROOT / "train_synthetic_labels.csv"

# -------------------------------------------------------------------------
# Dataset Directory Resolvers (Check data/ first, fallback to root)
# -------------------------------------------------------------------------
def get_train_dir() -> Path:
    p = DATA_DIR / "train"
    return p if p.exists() else PROJECT_ROOT / "train"

def get_train_synthetic_dir() -> Path:
    p = DATA_DIR / "train_synthetic"
    return p if p.exists() else PROJECT_ROOT / "train_synthetic"

def get_validation_dir() -> Path:
    for base in [DATA_DIR / "validasi", PROJECT_ROOT / "data_validasi"]:
        if (base / "test").exists():
            return base / "test"
        if base.exists():
            return base
    return DATA_DIR / "validasi" / "test"

def get_vanilla_dir() -> Path:
    p = DATA_DIR / "vanilla"
    return p if p.exists() else PROJECT_ROOT / "data_vanilla"

def get_archives_dir() -> Path:
    p = DATA_DIR / "archives"
    return p if p.exists() else DATA_DIR

def get_archive_zip() -> Path:
    p = DATA_DIR / "archives" / "archive.zip"
    return p if p.exists() else PROJECT_ROOT / "archive.zip"
