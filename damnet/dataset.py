"""
Datasets for DAM-Net Training & Evaluation
===========================================
Two modes:

1. **Sen1Floods11Dataset** – single-image flood segmentation
   Loads Sentinel-1 (VV+VH) tiles and hand-labeled flood masks from the
   Sen1Floods11 CSV splits.  During training the *post-flood* branch receives
   the actual SAR image and the *pre-flood* branch receives a learned-zero
   baseline, so the model learns what flooding looks like in SAR.

2. **BitemporalFloodDataset** – true bitemporal pairs
   For datasets like SenForFlood CEMS that have both before- and during-event
   images.  Provides real pre/post pairs for full change-detection training
   or inference.

All loaders return dicts with keys:
    pre_image  : (2, H, W)  float32
    post_image : (2, H, W)  float32
    label      : (1, H, W)  float32 ∈ {0, 1}
"""

from __future__ import annotations

import csv
import os
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset


# =====================================================================
# Helpers
# =====================================================================

def _read_tif(path: str, bands: Optional[List[int]] = None) -> np.ndarray:
    """Read a GeoTIFF and return (C, H, W) float32 array."""
    with rasterio.open(path) as src:
        if bands is not None:
            # rasterio uses 1-based indexing
            data = src.read([b + 1 for b in bands]).astype(np.float32)
        else:
            data = src.read().astype(np.float32)
    return data


def _normalize_sar(img: np.ndarray) -> np.ndarray:
    """Normalize SAR backscatter to roughly [0, 1].

    Sentinel-1 GRD backscatter values are typically in dB.
    We clip to a sensible range and min-max normalize per-band.
    NaN / Inf values are replaced with 0 before processing.
    """
    out = np.empty_like(img)
    for c in range(img.shape[0]):
        band = img[c].copy()
        # Replace NaN and Inf with 0
        band = np.where(np.isfinite(band), band, 0.0)
        # Clip extreme values
        valid = band[band != 0]
        p1, p99 = np.percentile(valid, [1, 99]) if valid.size > 0 else (0, 1)
        band = np.clip(band, p1, p99)
        lo, hi = band.min(), band.max()
        if hi - lo < 1e-8:
            out[c] = np.zeros_like(band)
        else:
            out[c] = (band - lo) / (hi - lo)
    return out


def _random_crop(arrays: List[np.ndarray], size: int) -> List[np.ndarray]:
    """Random spatial crop of multiple (C, H, W) arrays to (C, size, size)."""
    _, H, W = arrays[0].shape
    if H <= size and W <= size:
        # Pad if needed
        results = []
        for a in arrays:
            padded = np.zeros((a.shape[0], size, size), dtype=a.dtype)
            padded[:, :H, :W] = a
            results.append(padded)
        return results
    h0 = random.randint(0, max(0, H - size))
    w0 = random.randint(0, max(0, W - size))
    return [a[:, h0:h0 + size, w0:w0 + size] for a in arrays]


def _random_flip(arrays: List[np.ndarray]) -> List[np.ndarray]:
    """Random horizontal and vertical flip."""
    if random.random() > 0.5:
        arrays = [np.flip(a, axis=-1).copy() for a in arrays]
    if random.random() > 0.5:
        arrays = [np.flip(a, axis=-2).copy() for a in arrays]
    return arrays


def _random_rotate90(arrays: List[np.ndarray]) -> List[np.ndarray]:
    """Random 90° rotation."""
    k = random.randint(0, 3)
    if k > 0:
        arrays = [np.rot90(a, k, axes=(-2, -1)).copy() for a in arrays]
    return arrays


# =====================================================================
# Sen1Floods11 Dataset  (single-image with labels)
# =====================================================================

class Sen1Floods11Dataset(Dataset):
    """Sen1Floods11 dataset for training DAM-Net.

    Each sample returns a pre/post pair where:
    - post_image = actual S1 SAR image (may contain flooding)
    - pre_image  = zeros (learned baseline) or a random non-flood tile
    - label      = hand-drawn flood mask

    This teaches the model to detect flood patterns in SAR backscatter.
    For true bitemporal training, use ``BitemporalFloodDataset``.

    Parameters
    ----------
    csv_path : str
        Path to split CSV (columns: s1_file, label_file).
    s1_dir : str
        Root directory containing *_S1Hand.tif files.
    label_dir : str
        Root directory containing *_LabelHand.tif files.
    img_size : int
        Crop / pad to this size.
    augment : bool
        Apply random flip/rotate during training.
    use_reference_pairs : bool
        If True, randomly pair each flood image with a non-flood image
        as the pre-flood reference (more realistic training).
    """

    def __init__(
        self,
        csv_path: str,
        s1_dir: str,
        label_dir: str,
        img_size: int = 512,
        augment: bool = True,
        use_reference_pairs: bool = False,
    ):
        super().__init__()
        self.s1_dir = s1_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.augment = augment
        self.use_reference_pairs = use_reference_pairs

        # Parse CSV
        self.samples: List[Tuple[str, str]] = []
        with open(csv_path) as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    s1_file = row[0].strip()
                    lbl_file = row[1].strip()
                    # Verify files exist
                    s1_path = os.path.join(s1_dir, s1_file)
                    lbl_path = os.path.join(label_dir, lbl_file)
                    if os.path.isfile(s1_path) and os.path.isfile(lbl_path):
                        self.samples.append((s1_path, lbl_path))

        if not self.samples:
            raise FileNotFoundError(
                f"No valid samples found. Check paths:\n"
                f"  CSV: {csv_path}\n  S1 dir: {s1_dir}\n  Label dir: {label_dir}")

        print(f"[Sen1Floods11Dataset] Loaded {len(self.samples)} samples from {csv_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s1_path, lbl_path = self.samples[idx]

        # Load post-flood image (S1 VV+VH = 2 bands)
        post_img = _read_tif(s1_path)                 # (C, H, W)
        if post_img.shape[0] > 2:
            post_img = post_img[:2]                    # keep VV + VH only
        post_img = _normalize_sar(post_img)

        # Load label
        label = _read_tif(lbl_path)                    # (1, H, W)
        if label.shape[0] > 1:
            label = label[:1]
        # Binarize: flood=1, everything else=0; treat -1/nodata as 0
        label = (label > 0).astype(np.float32)

        # Pre-flood reference
        if self.use_reference_pairs:
            # Pick a random other sample as reference (imperfect but effective)
            ref_idx = random.randint(0, len(self.samples) - 1)
            ref_path, _ = self.samples[ref_idx]
            pre_img = _read_tif(ref_path)
            if pre_img.shape[0] > 2:
                pre_img = pre_img[:2]
            pre_img = _normalize_sar(pre_img)
        else:
            # Zero baseline – model learns to detect absolute flood signals
            pre_img = np.zeros_like(post_img)

        # Augmentation
        if self.augment:
            pre_img, post_img, label = _random_crop(
                [pre_img, post_img, label], self.img_size)
            pre_img, post_img, label = _random_flip(
                [pre_img, post_img, label])
            pre_img, post_img, label = _random_rotate90(
                [pre_img, post_img, label])
        else:
            # Center-crop / pad
            pre_img, post_img, label = _random_crop(
                [pre_img, post_img, label], self.img_size)

        return {
            "pre_image": torch.from_numpy(pre_img.copy()),
            "post_image": torch.from_numpy(post_img.copy()),
            "label": torch.from_numpy(label.copy()),
        }


# =====================================================================
# Bitemporal Flood Dataset  (true before/after pairs)
# =====================================================================

class BitemporalFloodDataset(Dataset):
    """Bitemporal dataset with real pre-flood and post-flood SAR pairs.

    Expects a CSV with columns:
        pre_path, post_path, label_path

    Or a directory structure like SenForFlood CEMS:
        {root}/{event}/s1_before_flood/*.tif
        {root}/{event}/s1_during_flood/*.tif
        {root}/{event}/labels/*.tif
    """

    def __init__(
        self,
        pairs: List[Tuple[str, str, str]] | None = None,
        csv_path: str | None = None,
        img_size: int = 512,
        augment: bool = False,
        bands: List[int] | None = None,
    ):
        super().__init__()
        self.img_size = img_size
        self.augment = augment
        self.bands = bands  # which bands to read (None = all)

        if pairs is not None:
            self.samples = pairs
        elif csv_path is not None:
            self.samples = []
            with open(csv_path) as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 3:
                        self.samples.append(
                            (row[0].strip(), row[1].strip(), row[2].strip()))
        else:
            raise ValueError("Provide either `pairs` or `csv_path`.")

        print(f"[BitemporalFloodDataset] {len(self.samples)} triplets loaded.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pre_path, post_path, lbl_path = self.samples[idx]

        pre_img = _read_tif(pre_path, self.bands)
        post_img = _read_tif(post_path, self.bands)
        label = _read_tif(lbl_path)

        # Keep first 2 channels if multi-band
        if pre_img.shape[0] > 2:
            pre_img = pre_img[:2]
        if post_img.shape[0] > 2:
            post_img = post_img[:2]
        if label.shape[0] > 1:
            label = label[:1]

        pre_img = _normalize_sar(pre_img)
        post_img = _normalize_sar(post_img)
        label = (label > 0).astype(np.float32)

        if self.augment:
            pre_img, post_img, label = _random_crop(
                [pre_img, post_img, label], self.img_size)
            pre_img, post_img, label = _random_flip(
                [pre_img, post_img, label])
            pre_img, post_img, label = _random_rotate90(
                [pre_img, post_img, label])
        else:
            pre_img, post_img, label = _random_crop(
                [pre_img, post_img, label], self.img_size)

        return {
            "pre_image": torch.from_numpy(pre_img.copy()),
            "post_image": torch.from_numpy(post_img.copy()),
            "label": torch.from_numpy(label.copy()),
        }


# =====================================================================
# Convenience: build dataloaders from config
# =====================================================================

def build_dataloaders(cfg):
    """Build train / val DataLoaders from a DAMNetConfig."""
    from torch.utils.data import DataLoader

    train_ds = Sen1Floods11Dataset(
        csv_path=cfg.train_csv,
        s1_dir=cfg.s1_dir,
        label_dir=cfg.label_dir,
        img_size=cfg.img_size,
        augment=True,
    )
    val_ds = Sen1Floods11Dataset(
        csv_path=cfg.valid_csv,
        s1_dir=cfg.s1_dir,
        label_dir=cfg.label_dir,
        img_size=cfg.img_size,
        augment=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )
    return train_loader, val_loader
