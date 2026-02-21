"""
Finetune Prithvi-EO-2.0-300M-TL on WildfireSpreadTS for Multi-Day Fire Prediction
==================================================================================

Uses the larger Prithvi EO 2.0 model (300M params) with temporal+location awareness
to predict next-day wildfire spread from 4 consecutive days of input features.

Key improvements over Prithvi 100M finetuning:
  - 4 temporal frames as input (vs 1) -> leverages fire progression history
  - 1024-dim embeddings (vs 768) -> richer feature representations
  - 24 transformer layers (vs 12) -> deeper feature hierarchy
  - Temporal + location encodings -> calendar-time and geo-aware

Architecture:
  Input: 4 consecutive days of features (6 bands x 4 timesteps x 224 x 224)
  -> PrithviViT encoder (pretrained, 300M params)
  -> Progressive upsampling decoder (4096 -> 256 -> 2)
  -> Binary fire/no-fire prediction for day 5

Dependencies:
  pip install torch numpy rasterio einops timm matplotlib

Usage:
    # Quick test (1 epoch, small subset):
    python finetune_prithvi_v2_wildfire.py --epochs 1 --max_samples 20

    # Full training (freeze backbone first):
    python finetune_prithvi_v2_wildfire.py --epochs 30 --freeze_backbone --lr 1e-3

    # End-to-end finetuning (lower LR for backbone):
    python finetune_prithvi_v2_wildfire.py --epochs 50 --lr 1e-4

    # Use channel adapter (learn 22->6 projection):
    python finetune_prithvi_v2_wildfire.py --use_adapter --epochs 30
"""

import argparse
import gc
import importlib.util
import json
import math
import os
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ═══════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# HuggingFace model URLs
HF_MODEL_ID = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL"
HF_BASE_URL = f"https://huggingface.co/{HF_MODEL_ID}/resolve/main"
HF_CHECKPOINT_URL = f"{HF_BASE_URL}/Prithvi_EO_V2_300M_TL.pt"
HF_MAE_SOURCE_URL = f"{HF_BASE_URL}/prithvi_mae.py"
HF_CONFIG_URL = f"{HF_BASE_URL}/config.json"

# Default model config (from HuggingFace config.json)
DEFAULT_MODEL_CONFIG = {
    "img_size": 224,
    "num_frames": 4,
    "patch_size": [1, 16, 16],
    "in_chans": 6,
    "embed_dim": 1024,
    "depth": 24,
    "num_heads": 16,
    "decoder_embed_dim": 512,
    "decoder_depth": 8,
    "decoder_num_heads": 16,
    "mlp_ratio": 4.0,
    "coords_encoding": ["time", "location"],
    "coords_scale_learn": True,
    "mask_ratio": 0.75,
}

# WildfireSpreadTS band definitions (23 bands, 0-indexed)
BAND_NAMES = [
    "M11", "I2", "I1",                     # 0-2: VIIRS reflectance
    "NDVI_last", "EVI2_last",              # 3-4: Vegetation indices
    "total_precip", "wind_speed",          # 5-6: Weather
    "wind_dir", "min_temp", "max_temp",    # 7-9: Weather
    "energy_release", "specific_humidity", # 10-11: Fire weather
    "slope", "aspect", "elevation",        # 12-14: Topography
    "pdsi", "LC_Type1",                    # 15-16: Drought & land cover
    "total_precip_surface",                # 17: ERA5 precipitation
    "forecast_wind_speed",                 # 18: ERA5 wind
    "forecast_wind_dir",                   # 19: ERA5 wind direction
    "forecast_temp",                       # 20: ERA5 temperature
    "forecast_specific_humidity",          # 21: ERA5 humidity
    "active_fire",                         # 22: Label - active fire mask
]

# Most fire-relevant 6 bands to map to Prithvi's 6-channel input
SELECTED_BANDS = [3, 4, 6, 9, 10, 14]
SELECTED_BAND_NAMES = [BAND_NAMES[i] for i in SELECTED_BANDS]

FIRE_LABEL_BAND = 22  # active fire band index
NUM_TEMPORAL_FRAMES = 4  # Prithvi EO 2.0 native temporal frames


# ═══════════════════════════════════════════════════════════════
#  AUTO-DOWNLOAD UTILITIES
# ═══════════════════════════════════════════════════════════════

def download_file(url, dest_path, desc=None):
    """Download a file with progress indication."""
    desc = desc or os.path.basename(dest_path)
    print(f"    Downloading {desc} ...")
    print(f"    URL: {url}")

    tmp_path = dest_path + ".tmp"
    try:
        def _progress(block_count, block_size, total_size):
            if total_size > 0:
                pct = min(100.0, block_count * block_size * 100.0 / total_size)
                mb_done = block_count * block_size / 1e6
                mb_total = total_size / 1e6
                sys.stdout.write(
                    f"\r    {mb_done:.1f}/{mb_total:.1f} MB ({pct:.0f}%)"
                )
                sys.stdout.flush()

        urllib.request.urlretrieve(url, tmp_path, reporthook=_progress)
        print()  # newline after progress
        os.replace(tmp_path, dest_path)
        print(f"    Saved: {dest_path}")
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise RuntimeError(f"Download failed: {e}") from e

    return dest_path


def ensure_prithvi_mae_source(dest_dir=None):
    """Download prithvi_mae.py from HuggingFace if not present."""
    dest_dir = dest_dir or _BASE_DIR
    mae_path = os.path.join(dest_dir, "prithvi_mae.py")
    if not os.path.exists(mae_path):
        download_file(HF_MAE_SOURCE_URL, mae_path, "prithvi_mae.py (model source)")
    return mae_path


def ensure_checkpoint(dest_dir=None):
    """Download Prithvi EO V2 300M TL checkpoint if not present."""
    dest_dir = dest_dir or os.path.join(_BASE_DIR, "Model")
    os.makedirs(dest_dir, exist_ok=True)
    ckpt_path = os.path.join(dest_dir, "Prithvi_EO_V2_300M_TL.pt")
    if not os.path.exists(ckpt_path):
        download_file(HF_CHECKPOINT_URL, ckpt_path,
                       "Prithvi_EO_V2_300M_TL.pt (1.33 GB)")
    return ckpt_path


def load_prithvi_mae_module(mae_path):
    """Dynamically import PrithviMAE and PrithviViT from prithvi_mae.py."""
    spec = importlib.util.spec_from_file_location("prithvi_mae", mae_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ═══════════════════════════════════════════════════════════════
#  DATASET: 4-Day Temporal Windows
# ═══════════════════════════════════════════════════════════════

class WildfireSpreadTemporalDataset(Dataset):
    """
    Dataset for multi-day wildfire prediction with Prithvi EO 2.0.

    Creates sliding windows of N consecutive days as input (default N=4),
    with the (N+1)-th day's fire mask as the target label.

    Input shape per sample:  (6, 4, H, W) -> 6 bands x 4 timesteps
    Label shape per sample:  (H, W) -> binary fire mask
    """

    def __init__(self, data_dir, tile_size=224, selected_bands=None,
                 use_all_bands=False, augment=False, max_samples=None,
                 num_frames=NUM_TEMPORAL_FRAMES):
        super().__init__()
        self.data_dir = data_dir
        self.tile_size = tile_size
        self.use_all_bands = use_all_bands
        self.selected_bands = selected_bands or SELECTED_BANDS
        self.augment = augment
        self.num_frames = num_frames

        # Discover all valid temporal windows
        self.windows = []
        self._discover_windows()

        if max_samples and len(self.windows) > max_samples:
            np.random.seed(42)
            indices = np.random.choice(len(self.windows), max_samples, replace=False)
            self.windows = [self.windows[i] for i in sorted(indices)]

        print(f"  Dataset: {len(self.windows)} {num_frames}-day windows from {data_dir}")

    def _discover_windows(self):
        """Find all N-day sliding windows across fire events."""
        # Auto-detect: data_dir might contain year dirs directly,
        # or an extra subdirectory (e.g. 'WildfireSpreadTS/') wrapping them
        root = Path(self.data_dir)
        year_dirs = [d for d in sorted(root.iterdir()) if d.is_dir() and d.name.isdigit()]
        if not year_dirs:
            # Check one level deeper (extra wrapper dir)
            for sub in sorted(root.iterdir()):
                if sub.is_dir():
                    year_dirs.extend(
                        d for d in sorted(sub.iterdir())
                        if d.is_dir() and d.name.isdigit()
                    )
        for year_dir in year_dirs:
            for fire_dir in sorted(year_dir.iterdir()):
                if not fire_dir.is_dir():
                    continue
                tifs = sorted(fire_dir.glob("*.tif"))
                if len(tifs) < self.num_frames + 1:
                    continue
                # Sliding window: num_frames input days + 1 label day
                for i in range(len(tifs) - self.num_frames):
                    input_days = [str(tifs[i + j]) for j in range(self.num_frames)]
                    label_day = str(tifs[i + self.num_frames])
                    self.windows.append((input_days, label_day))

    def __len__(self):
        return len(self.windows)

    def _load_tif(self, path):
        """Load a GeoTIFF and return all bands as numpy array."""
        with rasterio.open(path) as ds:
            data = ds.read().astype(np.float32)  # (23, H, W)
        return data

    def _extract_features(self, data):
        """Extract feature bands from the 23-band data."""
        if self.use_all_bands:
            return data[:22]  # All bands except active_fire label
        return data[self.selected_bands]  # (6, H, W)

    def _extract_label(self, data):
        """Extract binary fire mask from active_fire band."""
        fire = data[FIRE_LABEL_BAND]  # (H, W)
        label = np.zeros_like(fire, dtype=np.float32)
        valid = ~np.isnan(fire)
        label[valid] = 1.0  # Any non-NaN = fire pixel
        return label

    def _pad_to_size(self, data, size):
        """Pad data to at least (size x size) in spatial dims."""
        if data.ndim == 4:  # (C, T, H, W)
            _, _, h, w = data.shape
            ph, pw = max(0, size - h), max(0, size - w)
            if ph > 0 or pw > 0:
                data = np.pad(data, ((0, 0), (0, 0), (0, ph), (0, pw)),
                              mode="reflect")
            return data[:, :, :size, :size]
        elif data.ndim == 2:  # (H, W)
            h, w = data.shape
            ph, pw = max(0, size - h), max(0, size - w)
            if ph > 0 or pw > 0:
                data = np.pad(data, ((0, ph), (0, pw)), mode="reflect")
            return data[:size, :size]
        raise ValueError(f"Unexpected ndim={data.ndim}")

    def _random_crop(self, features, label, size):
        """Random crop to (size x size), padding first if needed."""
        _, _, h, w = features.shape
        if h < size or w < size:
            features = self._pad_to_size(features, size)
            label = self._pad_to_size(label, size)
            _, _, h, w = features.shape
        top = np.random.randint(0, max(1, h - size + 1))
        left = np.random.randint(0, max(1, w - size + 1))
        features = features[:, :, top:top + size, left:left + size]
        label = label[top:top + size, left:left + size]
        return features, label

    def _center_crop(self, features, label, size):
        """Center crop to (size x size), padding first if needed."""
        _, _, h, w = features.shape
        if h < size or w < size:
            features = self._pad_to_size(features, size)
            label = self._pad_to_size(label, size)
            _, _, h, w = features.shape
        top = (h - size) // 2
        left = (w - size) // 2
        features = features[:, :, top:top + size, left:left + size]
        label = label[top:top + size, left:left + size]
        return features, label

    def _normalize(self, features):
        """Per-band normalization across all timesteps (shared stats)."""
        for c in range(features.shape[0]):
            band_data = features[c]  # (T, H, W)
            valid = ~np.isnan(band_data)
            if valid.any():
                mu = band_data[valid].mean()
                std = band_data[valid].std() + 1e-8
                features[c] = np.where(valid, (band_data - mu) / std, 0.0)
            else:
                features[c] = 0.0
        return features

    def __getitem__(self, idx):
        input_paths, label_path = self.windows[idx]

        # Load all input days and stack
        day_features = []
        for path in input_paths:
            data = self._load_tif(path)
            feats = self._extract_features(data)  # (6, H, W)
            day_features.append(feats)

        # Stack into (C, T, H, W)
        features = np.stack(day_features, axis=1)  # (6, 4, H, W)

        # Load label from the day after the input window
        label_data = self._load_tif(label_path)
        label = self._extract_label(label_data)  # (H, W)

        # Replace NaN
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalize per band (shared across all timesteps)
        features = self._normalize(features)

        # Spatial crop/pad to tile_size
        if self.augment:
            features, label = self._random_crop(features, label, self.tile_size)
            if np.random.random() > 0.5:
                features = np.flip(features, axis=3).copy()
                label = np.flip(label, axis=1).copy()
            if np.random.random() > 0.5:
                features = np.flip(features, axis=2).copy()
                label = np.flip(label, axis=0).copy()
        else:
            features, label = self._center_crop(features, label, self.tile_size)

        # To tensors: (C, T, H, W) = (6, 4, 224, 224)
        features = torch.from_numpy(features.copy()).float()
        label = torch.from_numpy(label.copy()).long()

        return features, label


# ═══════════════════════════════════════════════════════════════
#  CHANNEL ADAPTER (optional: 22 -> 6 bands)
# ═══════════════════════════════════════════════════════════════

class ChannelAdapter(nn.Module):
    """Learns a projection from N input channels to 6 (Prithvi's expected input)."""

    def __init__(self, in_channels, out_channels=6):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, x):
        # x: (B, C_in, T, H, W) -> apply per-frame
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        x = self.conv(x)  # (B*T, 6, H, W)
        x = x.reshape(B, T, 6, H, W).permute(0, 2, 1, 3, 4)
        return x  # (B, 6, T, H, W)


# ═══════════════════════════════════════════════════════════════
#  SEGMENTATION DECODER
# ═══════════════════════════════════════════════════════════════

class ProgressiveUpsampleDecoder(nn.Module):
    """
    Progressive upsampling decoder for ViT features.

    Takes (B, in_channels, 14, 14) and upsamples to (B, num_classes, 224, 224).
    Stages: 14 -> 28 -> 56 -> 112 -> 224 (4x ConvTranspose2d, each 2x)
    """

    def __init__(self, in_channels=4096, num_classes=2, dropout=0.1):
        super().__init__()

        # Channel reduction: 4096 -> 512
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, 512, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.GELU(),
        )

        # Progressive upsampling stages
        self.up1 = self._up_block(512, 256)   # 14 -> 28
        self.up2 = self._up_block(256, 128)   # 28 -> 56
        self.up3 = self._up_block(128, 64)    # 56 -> 112
        self.up4 = self._up_block(64, 32)     # 112 -> 224

        # Classification head
        self.head = nn.Sequential(
            nn.Dropout2d(dropout),
            nn.Conv2d(32, num_classes, 1),
        )

    def _up_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        x = self.reduce(x)  # (B, 512, 14, 14)
        x = self.up1(x)     # (B, 256, 28, 28)
        x = self.up2(x)     # (B, 128, 56, 56)
        x = self.up3(x)     # (B, 64, 112, 112)
        x = self.up4(x)     # (B, 32, 224, 224)
        x = self.head(x)    # (B, 2, 224, 224)
        return x


# ═══════════════════════════════════════════════════════════════
#  WILDFIRE MODEL
# ═══════════════════════════════════════════════════════════════

class PrithviV2WildfireModel(nn.Module):
    """
    Prithvi-EO-2.0-300M-TL + Progressive Decoder for wildfire prediction.

    Architecture:
        Input (B, 6, 4, 224, 224)
        -> PrithviViT encoder: list of 24 x (B, 785, 1024)
        -> prepare_features_for_image_model: list of 24 x (B, 4096, 14, 14)
        -> Select last layer -> (B, 4096, 14, 14)
        -> ProgressiveUpsampleDecoder -> (B, 2, 224, 224)
    """

    def __init__(self, encoder, use_adapter=False, in_channels=6,
                 feature_layer=-1, num_classes=2, dropout=0.1):
        super().__init__()
        self.encoder = encoder
        self.feature_layer = feature_layer
        self.use_adapter = use_adapter

        if use_adapter:
            self.adapter = ChannelAdapter(in_channels, 6)
        else:
            self.adapter = None

        # in_channels for decoder = embed_dim * num_temporal_patches
        feat_channels = encoder.out_channels[0]  # embed_dim * grid_size[0]

        self.decoder = ProgressiveUpsampleDecoder(
            in_channels=feat_channels,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(self, x, temporal_coords=None, location_coords=None):
        """
        x: (B, C, T, H, W) = (B, 6, 4, 224, 224)
        Returns: logits (B, num_classes, H, W)
        """
        if self.adapter is not None:
            x = self.adapter(x)

        # Get features from all transformer layers (no masking)
        features_list = self.encoder.forward_features(
            x, temporal_coords, location_coords
        )

        # Convert token sequences to spatial feature maps
        spatial_features = self.encoder.prepare_features_for_image_model(
            features_list
        )

        # Select feature layer (default: last)
        features = spatial_features[self.feature_layer]  # (B, 4096, 14, 14)

        # Decode to full resolution
        logits = self.decoder(features)  # (B, 2, 224, 224)

        return logits


# ═══════════════════════════════════════════════════════════════
#  MODEL LOADING
# ═══════════════════════════════════════════════════════════════

def load_prithvi_v2_encoder(checkpoint_path, config=None, device="cpu"):
    """
    Load pretrained Prithvi-EO-2.0-300M-TL encoder.

    Downloads:
      1. prithvi_mae.py (model architecture source) from HuggingFace
      2. Checkpoint weights (if not already present)

    Returns:
      PrithviViT encoder module with loaded pretrained weights.
    """
    # 1. Get model source code
    mae_path = ensure_prithvi_mae_source()
    prithvi_module = load_prithvi_mae_module(mae_path)

    # 2. Build config
    config = config or DEFAULT_MODEL_CONFIG.copy()

    # 3. Create full MAE model (encoder + decoder)
    print(f"    Creating PrithviMAE (embed_dim={config['embed_dim']}, "
          f"depth={config['depth']}, num_frames={config['num_frames']})")

    mae = prithvi_module.PrithviMAE(
        img_size=config["img_size"],
        patch_size=config["patch_size"],
        num_frames=config["num_frames"],
        in_chans=config["in_chans"],
        embed_dim=config["embed_dim"],
        depth=config["depth"],
        num_heads=config["num_heads"],
        decoder_embed_dim=config["decoder_embed_dim"],
        decoder_depth=config["decoder_depth"],
        decoder_num_heads=config["decoder_num_heads"],
        mlp_ratio=config["mlp_ratio"],
        coords_encoding=config.get("coords_encoding"),
        coords_scale_learn=config.get("coords_scale_learn", False),
    )

    # 4. Load checkpoint
    print(f"    Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device,
                            weights_only=True)

    # Discard fixed positional embeddings (recomputed from sincos)
    removed_keys = []
    for k in list(state_dict.keys()):
        if "pos_embed" in k:
            del state_dict[k]
            removed_keys.append(k)
    if removed_keys:
        print(f"    Removed {len(removed_keys)} pos_embed keys (recomputed)")

    missing, unexpected = mae.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"    Missing keys: {len(missing)} (pos_embed expected)")
    if unexpected:
        print(f"    Unexpected keys: {len(unexpected)}")

    # 5. Extract encoder only (discard MAE decoder)
    encoder = mae.encoder

    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"    Encoder loaded: {total_params:,} parameters")

    return encoder


# ═══════════════════════════════════════════════════════════════
#  LOSS FUNCTION
# ═══════════════════════════════════════════════════════════════

class DiceLoss(nn.Module):
    """Dice loss for binary segmentation."""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)[:, 1]  # fire probability
        targets_f = targets.float()
        intersection = (probs * targets_f).sum()
        return 1.0 - (2.0 * intersection + self.smooth) / (
            probs.sum() + targets_f.sum() + self.smooth
        )


class CombinedLoss(nn.Module):
    """Weighted CE + Dice loss for imbalanced fire detection."""

    def __init__(self, ce_weight=1.0, dice_weight=1.0, pos_weight=10.0):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce_loss = nn.CrossEntropyLoss(
            weight=torch.tensor([1.0, pos_weight])
        )
        self.dice_loss = DiceLoss()

    def to(self, device):
        super().to(device)
        self.ce_loss.weight = self.ce_loss.weight.to(device)
        return self

    def forward(self, logits, targets):
        ce = self.ce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        return self.ce_weight * ce + self.dice_weight * dice


# ═══════════════════════════════════════════════════════════════
#  METRICS
# ═══════════════════════════════════════════════════════════════

def compute_metrics(logits, targets):
    """Compute precision, recall, F1, IoU for fire class."""
    preds = torch.argmax(logits, dim=1)  # (B, H, W)
    tp = ((preds == 1) & (targets == 1)).sum().float()
    fp = ((preds == 1) & (targets == 0)).sum().float()
    fn = ((preds == 0) & (targets == 1)).sum().float()
    tn = ((preds == 0) & (targets == 0)).sum().float()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)

    return {
        "accuracy": accuracy.item(),
        "precision": precision.item(),
        "recall": recall.item(),
        "f1": f1.item(),
        "iou": iou.item(),
        "fire_pixels": (tp + fn).item(),
        "pred_fire_pixels": (tp + fp).item(),
    }


# ═══════════════════════════════════════════════════════════════
#  TRAINING / VALIDATION LOOPS
# ═══════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, criterion, optimizer, device, epoch,
                    grad_accum_steps=1):
    """Train for one epoch with optional gradient accumulation."""
    model.train()
    total_loss = 0.0
    total_metrics = defaultdict(float)
    n_batches = 0

    optimizer.zero_grad()

    for batch_idx, (features, labels) in enumerate(loader):
        features = features.to(device)  # (B, C, T, H, W)
        labels = labels.to(device)      # (B, H, W)

        logits = model(features)

        # Resize logits to match label size if needed
        if logits.shape[-2:] != labels.shape[-2:]:
            logits = F.interpolate(logits, size=labels.shape[-2:],
                                   mode="bilinear", align_corners=False)

        loss = criterion(logits, labels) / grad_accum_steps
        loss.backward()

        if (batch_idx + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum_steps
        metrics = compute_metrics(logits.detach(), labels)
        for k, v in metrics.items():
            total_metrics[k] += v
        n_batches += 1

        if (batch_idx + 1) % 5 == 0 or batch_idx == 0:
            sys.stdout.write(
                f"\r  Epoch {epoch} [{batch_idx+1}/{len(loader)}] "
                f"Loss={loss.item()*grad_accum_steps:.4f}  "
                f"F1={metrics['f1']:.3f}  IoU={metrics['iou']:.3f}"
            )
            sys.stdout.flush()

    # Handle remaining gradients
    if len(loader) % grad_accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

    avg_loss = total_loss / max(n_batches, 1)
    avg_metrics = {k: v / max(n_batches, 1) for k, v in total_metrics.items()}
    print(f"\r  Train {epoch} — Loss={avg_loss:.4f}  F1={avg_metrics['f1']:.3f}  "
          f"IoU={avg_metrics['iou']:.3f}  Prec={avg_metrics['precision']:.3f}  "
          f"Rec={avg_metrics['recall']:.3f}")
    return avg_loss, avg_metrics


@torch.no_grad()
def validate(model, loader, criterion, device, epoch):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    total_metrics = defaultdict(float)
    n_batches = 0

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)

        logits = model(features)
        if logits.shape[-2:] != labels.shape[-2:]:
            logits = F.interpolate(logits, size=labels.shape[-2:],
                                   mode="bilinear", align_corners=False)

        loss = criterion(logits, labels)
        total_loss += loss.item()
        metrics = compute_metrics(logits, labels)
        for k, v in metrics.items():
            total_metrics[k] += v
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    avg_metrics = {k: v / max(n_batches, 1) for k, v in total_metrics.items()}
    print(f"  Val   {epoch} — Loss={avg_loss:.4f}  F1={avg_metrics['f1']:.3f}  "
          f"IoU={avg_metrics['iou']:.3f}  Prec={avg_metrics['precision']:.3f}  "
          f"Rec={avg_metrics['recall']:.3f}")
    return avg_loss, avg_metrics


# ═══════════════════════════════════════════════════════════════
#  MAIN FINETUNING PIPELINE
# ═══════════════════════════════════════════════════════════════

def run_finetuning(args):
    """Main finetuning pipeline."""
    print("=" * 70)
    print("  Prithvi-EO-2.0-300M-TL -> WildfireSpreadTS Finetuning")
    print("  4-Day Temporal Input -> Next-Day Fire Prediction")
    print("=" * 70)

    device = torch.device(args.device)
    print(f"\n  Device:          {device}")
    print(f"  Checkpoint:      {args.checkpoint}")
    print(f"  Data dir:        {args.data_dir}")
    print(f"  Output dir:      {args.output_dir}")
    print(f"  Epochs:          {args.epochs}")
    print(f"  Batch size:      {args.batch_size}")
    print(f"  Grad accum:      {args.grad_accum}")
    print(f"  LR:              {args.lr}")
    print(f"  Freeze backbone: {args.freeze_backbone}")
    print(f"  Adapter:         {args.use_adapter}")
    print(f"  Temporal frames: {args.num_frames}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Step 1: Ensure checkpoint exists ──
    print(f"\n[1] Preparing model checkpoint ...")
    if not os.path.exists(args.checkpoint):
        print(f"    Checkpoint not found at {args.checkpoint}")
        args.checkpoint = ensure_checkpoint()

    # ── Step 2: Load pretrained encoder ──
    print(f"\n[2] Loading pretrained Prithvi-EO-2.0-300M-TL encoder ...")
    encoder = load_prithvi_v2_encoder(args.checkpoint, device="cpu")

    # ── Step 3: Build wildfire model ──
    print(f"\n[3] Building wildfire spread model ...")
    in_channels = 22 if args.use_adapter else 6
    model = PrithviV2WildfireModel(
        encoder=encoder,
        use_adapter=args.use_adapter,
        in_channels=in_channels,
        num_classes=2,
        dropout=args.dropout,
    )

    # Freeze backbone if requested
    if args.freeze_backbone:
        print("    Freezing encoder weights ...")
        for param in model.encoder.parameters():
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"    Total params:     {total:,}")
    print(f"    Trainable params: {trainable:,} ({trainable/total*100:.1f}%)")

    model.to(device)

    # ── Step 4: Prepare datasets ──
    print(f"\n[4] Preparing datasets ...")

    # Discover all fire events
    all_fire_dirs = []
    for year_dir in sorted(Path(args.data_dir).iterdir()):
        if year_dir.is_dir():
            for fire_dir in sorted(year_dir.iterdir()):
                if fire_dir.is_dir():
                    tifs = list(fire_dir.glob("*.tif"))
                    if len(tifs) >= args.num_frames + 1:
                        all_fire_dirs.append(str(fire_dir))

    print(f"    Found {len(all_fire_dirs)} fire events with >={args.num_frames+1} days")

    if len(all_fire_dirs) < 2:
        print("    WARNING: Very few fire events. Using same data for train/val.")
        train_dirs = all_fire_dirs
        val_dirs = all_fire_dirs
    else:
        np.random.seed(42)
        np.random.shuffle(all_fire_dirs)
        split_idx = max(1, int(len(all_fire_dirs) * 0.8))
        train_dirs = all_fire_dirs[:split_idx]
        val_dirs = all_fire_dirs[split_idx:]

    print(f"    Train fires: {len(train_dirs)}, Val fires: {len(val_dirs)}")

    train_dataset = WildfireSpreadTemporalDataset(
        data_dir=args.data_dir,
        tile_size=args.tile_size,
        use_all_bands=args.use_adapter,
        augment=True,
        max_samples=args.max_samples,
        num_frames=args.num_frames,
    )
    val_dataset = WildfireSpreadTemporalDataset(
        data_dir=args.data_dir,
        tile_size=args.tile_size,
        use_all_bands=args.use_adapter,
        augment=False,
        max_samples=args.max_samples // 4 if args.max_samples else None,
        num_frames=args.num_frames,
    )

    # Split windows by fire event
    if len(all_fire_dirs) >= 2:
        train_windows = [w for w in train_dataset.windows
                         if any(w[0][0].startswith(d) for d in train_dirs)]
        val_windows = [w for w in val_dataset.windows
                       if any(w[0][0].startswith(d) for d in val_dirs)]
        if train_windows:
            train_dataset.windows = train_windows
        if val_windows:
            val_dataset.windows = val_windows

    print(f"    Train samples: {len(train_dataset)}")
    print(f"    Val samples:   {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    # ── Step 5: Setup training ──
    print(f"\n[5] Setting up training ...")

    fire_ratio = 0.05
    pos_weight = (1.0 - fire_ratio) / fire_ratio
    print(f"    Pos weight: {pos_weight:.1f}")

    criterion = CombinedLoss(
        ce_weight=1.0, dice_weight=1.0, pos_weight=min(pos_weight, 20.0)
    ).to(device)

    # Parameter groups with different learning rates
    param_groups = []
    if args.use_adapter and model.adapter is not None:
        param_groups.append({
            "params": model.adapter.parameters(),
            "lr": args.lr, "name": "adapter"
        })
    param_groups.append({
        "params": model.decoder.parameters(),
        "lr": args.lr, "name": "decoder"
    })
    if not args.freeze_backbone:
        param_groups.append({
            "params": model.encoder.parameters(),
            "lr": args.lr * 0.1, "name": "encoder"
        })

    for pg in param_groups:
        n = sum(p.numel() for p in pg["params"])
        print(f"    {pg['name']:>10}: {n:>12,} params @ lr={pg['lr']:.2e}")

    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # ── Step 6: Training loop ──
    print(f"\n[6] Starting training ...")
    print(f"    {'─' * 60}")

    best_val_f1 = 0.0
    best_epoch = 0
    history = {"train_loss": [], "val_loss": [],
               "train_f1": [], "val_f1": []}

    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        t_ep = time.time()

        train_loss, train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            grad_accum_steps=args.grad_accum,
        )
        val_loss, val_metrics = validate(
            model, val_loader, criterion, device, epoch
        )

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_f1"].append(train_metrics["f1"])
        history["val_f1"].append(val_metrics["f1"])

        # Save best model
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_epoch = epoch
            save_path = os.path.join(args.output_dir, "best_wildfire_v2_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": best_val_f1,
                "val_loss": val_loss,
                "args": vars(args),
                "model_config": DEFAULT_MODEL_CONFIG,
            }, save_path)
            print(f"    ★ Best model saved (F1={best_val_f1:.4f})")

        ep_time = time.time() - t_ep
        print(f"    Epoch time: {ep_time:.1f}s  |  "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        print(f"    {'─' * 60}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_time = time.time() - t_start

    # ── Step 7: Save final model ──
    final_path = os.path.join(args.output_dir, "final_wildfire_v2_model.pth")
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "history": history,
        "args": vars(args),
        "model_config": DEFAULT_MODEL_CONFIG,
    }, final_path)

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print(f"  FINETUNING COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Model:         Prithvi-EO-2.0-300M-TL")
    print(f"  Total time:    {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"  Best epoch:    {best_epoch}")
    print(f"  Best val F1:   {best_val_f1:.4f}")
    print(f"  Best model:    {save_path}")
    print(f"  Final model:   {final_path}")

    # Plot training curves
    try:
        _plot_training_curves(history, args.output_dir)
        print(f"  Curves:        {os.path.join(args.output_dir, 'training_curves.jpg')}")
    except Exception as e:
        print(f"  (Could not plot curves: {e})")

    print(f"{'=' * 70}")
    return model, history


def _plot_training_curves(history, output_dir):
    """Plot training loss and F1 curves."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], "b-", label="Train")
    ax1.plot(epochs, history["val_loss"], "r-", label="Val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_f1"], "b-", label="Train F1")
    ax2.plot(epochs, history["val_f1"], "r-", label="Val F1")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("F1 Score")
    ax2.set_title("Training & Validation F1")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Prithvi-EO-2.0-300M-TL → WildfireSpreadTS Finetuning",
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curves.jpg"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    gc.collect()


# ═══════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Finetune Prithvi-EO-2.0-300M-TL on WildfireSpreadTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--checkpoint", type=str,
                   default=os.path.join(_BASE_DIR, "Model",
                                        "Prithvi_EO_V2_300M_TL.pt"),
                   help="Path to Prithvi EO V2 300M TL checkpoint "
                        "(auto-downloads if not found)")
    p.add_argument("--data_dir", type=str,
                   default=os.path.join(_BASE_DIR, "DATA", "WildfireSpreadTS"),
                   help="Path to WildfireSpreadTS data")
    p.add_argument("--output_dir", type=str,
                   default=os.path.join(_BASE_DIR, "OUTPUTS",
                                        "wildfire_finetune_v2"),
                   help="Output directory for checkpoints and logs")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=2,
                   help="Batch size (default 2; 300M model is memory-hungry)")
    p.add_argument("--grad_accum", type=int, default=2,
                   help="Gradient accumulation steps (effective batch = "
                        "batch_size * grad_accum)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--tile_size", type=int, default=224)
    p.add_argument("--num_frames", type=int, default=NUM_TEMPORAL_FRAMES,
                   help="Number of temporal input frames (default 4)")
    p.add_argument("--freeze_backbone", action="store_true",
                   help="Freeze the pretrained ViT encoder")
    p.add_argument("--use_adapter", action="store_true",
                   help="Use a learnable 22->6 channel adapter")
    p.add_argument("--max_samples", type=int, default=None,
                   help="Max training samples (for quick testing)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_finetuning(args)
