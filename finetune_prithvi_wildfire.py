"""
Finetune Prithvi 100M on WildfireSpreadTS for Next-Day Fire Spread Prediction
==============================================================================

This script finetunes the pretrained Prithvi 100M backbone (from the burn-scar
checkpoint) on the WildfireSpreadTS dataset for predicting next-day wildfire spread.

**What it does:**
  - Uses the Prithvi ViT encoder as a frozen/unfrozen backbone
  - Adds a new segmentation head for binary fire/no-fire prediction
  - Trains on WildfireSpreadTS daily GeoTIFFs: input = day T features → label = day T+1 active fire
  - Handles the 23-band WildfireSpreadTS format by projecting to 6 channels (Prithvi input)

**Key architecture decisions:**
  - The 6 most fire-relevant bands are selected from the 23 available bands
  - Alternatively, a learned 23→6 channel adapter can be used
  - Training uses next-day fire prediction: features(day_t) → fire_mask(day_t+1)

Usage:
    # Quick test (1 epoch, small subset):
    python finetune_prithvi_wildfire.py --epochs 1 --max_samples 50

    # Full training (freeze backbone first):
    python finetune_prithvi_wildfire.py --epochs 30 --freeze_backbone --lr 1e-3

    # End-to-end finetuning:
    python finetune_prithvi_wildfire.py --epochs 50 --lr 1e-4

    # Use channel adapter (learn 23→6 projection):
    python finetune_prithvi_wildfire.py --use_adapter --epochs 30
"""

import argparse
import gc
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Import model architecture from existing pipeline
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fire_inference_pipeline import (
    PrithviBurnScarModel,
    load_prithvi_burn_scar_model,
    TemporalViTEncoder,
    ConvTransformerTokensToEmbeddingNeck,
    FCNHead,
    Norm2d,
)

# ═══════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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
# Selected: NDVI, EVI2, wind_speed, max_temp, energy_release, elevation
SELECTED_BANDS = [3, 4, 6, 9, 10, 14]
SELECTED_BAND_NAMES = [BAND_NAMES[i] for i in SELECTED_BANDS]

FIRE_LABEL_BAND = 22  # active fire band index


# ═══════════════════════════════════════════════════════════════
#  DATASET
# ═══════════════════════════════════════════════════════════════

class WildfireSpreadDataset(Dataset):
    """
    PyTorch dataset for WildfireSpreadTS next-day fire prediction.

    For each sample, loads:
      - Input: features from day T (selected bands)
      - Label: active fire mask from day T+1 (binary: fire / no fire)

    The dataset creates consecutive-day pairs across all fire events.
    """

    def __init__(self, data_dir, tile_size=224, use_all_bands=False,
                 selected_bands=None, augment=False, max_samples=None):
        super().__init__()
        self.data_dir = data_dir
        self.tile_size = tile_size
        self.use_all_bands = use_all_bands
        self.selected_bands = selected_bands or SELECTED_BANDS
        self.augment = augment

        # Discover all day-pairs
        self.pairs = []
        self._discover_pairs()

        if max_samples and len(self.pairs) > max_samples:
            # Sample evenly across fires
            np.random.seed(42)
            indices = np.random.choice(len(self.pairs), max_samples, replace=False)
            self.pairs = [self.pairs[i] for i in sorted(indices)]

        print(f"  Dataset: {len(self.pairs)} day-pairs from {data_dir}")

    def _discover_pairs(self):
        """Find all consecutive day pairs across fire events."""
        for year_dir in sorted(Path(self.data_dir).iterdir()):
            if not year_dir.is_dir():
                continue
            for fire_dir in sorted(year_dir.iterdir()):
                if not fire_dir.is_dir():
                    continue
                tifs = sorted(fire_dir.glob("*.tif"))
                if len(tifs) < 2:
                    continue
                # Create consecutive pairs: (day_t, day_t+1)
                for i in range(len(tifs) - 1):
                    self.pairs.append((str(tifs[i]), str(tifs[i + 1])))

    def __len__(self):
        return len(self.pairs)

    def _load_tif(self, path):
        """Load a GeoTIFF and return all bands as numpy array."""
        with rasterio.open(path) as ds:
            data = ds.read().astype(np.float32)  # (23, H, W)
        return data

    def _extract_features(self, data):
        """Extract feature bands from the 23-band data."""
        if self.use_all_bands:
            # Use all bands except active_fire (band 22)
            features = data[:22]  # (22, H, W)
        else:
            features = data[self.selected_bands]  # (6, H, W)
        return features

    def _extract_label(self, data):
        """Extract binary fire mask from active_fire band."""
        fire = data[FIRE_LABEL_BAND]  # (H, W)
        # Active fire band: has values where fire detected, NaN elsewhere
        # Convert to binary: fire=1, no-fire=0
        label = np.zeros_like(fire, dtype=np.float32)
        valid = ~np.isnan(fire)
        label[valid] = 1.0  # Any non-NaN value = fire
        return label

    def _random_crop(self, features, label, size):
        """Random crop to (size x size), padding first if needed."""
        _, h, w = features.shape
        # Pad any dimension that is smaller than size
        if h < size or w < size:
            features = self._pad_to_size(features, size)
            label = self._pad_to_size(label[np.newaxis], size)[0]
            _, h, w = features.shape

        top = np.random.randint(0, max(1, h - size + 1))
        left = np.random.randint(0, max(1, w - size + 1))
        features = features[:, top:top + size, left:left + size]
        label = label[top:top + size, left:left + size]
        return features, label

    def _center_crop(self, features, label, size):
        """Center crop to (size x size), padding first if needed."""
        _, h, w = features.shape
        # Pad any dimension that is smaller than size
        if h < size or w < size:
            features = self._pad_to_size(features, size)
            label = self._pad_to_size(label[np.newaxis], size)[0]
            _, h, w = features.shape

        top = (h - size) // 2
        left = (w - size) // 2
        features = features[:, top:top + size, left:left + size]
        label = label[top:top + size, left:left + size]
        return features, label

    def _pad_to_size(self, data, size):
        """Pad data to at least (size x size)."""
        if data.ndim == 3:
            _, h, w = data.shape
            ph = max(0, size - h)
            pw = max(0, size - w)
            if ph > 0 or pw > 0:
                data = np.pad(data, ((0, 0), (0, ph), (0, pw)),
                              mode="reflect")
            return data[:, :size, :size]
        else:
            h, w = data.shape
            ph = max(0, size - h)
            pw = max(0, size - w)
            if ph > 0 or pw > 0:
                data = np.pad(data, ((0, ph), (0, pw)), mode="reflect")
            return data[:size, :size]

    def _normalize(self, features):
        """Per-band normalization: zero-mean, unit-std."""
        for i in range(features.shape[0]):
            band = features[i]
            valid = ~np.isnan(band)
            if valid.any():
                mu = band[valid].mean()
                std = band[valid].std() + 1e-8
                features[i] = np.where(valid, (band - mu) / std, 0.0)
            else:
                features[i] = 0.0
        return features

    def __getitem__(self, idx):
        path_t, path_t1 = self.pairs[idx]

        # Load day T features and day T+1 label
        data_t = self._load_tif(path_t)
        data_t1 = self._load_tif(path_t1)

        features = self._extract_features(data_t)
        label = self._extract_label(data_t1)

        # Replace NaN with 0 in features
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalize features
        features = self._normalize(features)

        # Crop/pad to tile_size
        if self.augment:
            features, label = self._random_crop(features, label, self.tile_size)
            # Random horizontal/vertical flip
            if np.random.random() > 0.5:
                features = np.flip(features, axis=2).copy()
                label = np.flip(label, axis=1).copy()
            if np.random.random() > 0.5:
                features = np.flip(features, axis=1).copy()
                label = np.flip(label, axis=0).copy()
        else:
            features, label = self._center_crop(features, label, self.tile_size)

        # Convert to tensors
        # features: (C, H, W) → (C, 1, H, W) for Prithvi temporal dim
        features = torch.from_numpy(features.copy()).float()
        features = features.unsqueeze(1)  # (C, 1, H, W)
        label = torch.from_numpy(label.copy()).long()

        return features, label


# ═══════════════════════════════════════════════════════════════
#  MODEL: Prithvi with optional channel adapter
# ═══════════════════════════════════════════════════════════════

class ChannelAdapter(nn.Module):
    """Learns a projection from N input channels to 6 (Prithvi's expected input)."""

    def __init__(self, in_channels, out_channels=6):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, x):
        # x: (B, C_in, T, H, W) → reshape → conv → reshape back
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        x = self.conv(x)  # (B*T, 6, H, W)
        x = x.reshape(B, T, 6, H, W).permute(0, 2, 1, 3, 4)
        return x


class PrithviWildfireModel(nn.Module):
    """
    Prithvi-based wildfire spread prediction model.

    Architecture:
      (optional) ChannelAdapter: 22/6 bands → 6 bands
      → Prithvi ViT Encoder (pretrained backbone)
      → ConvTransformerTokensToEmbeddingNeck (pretrained)
      → New FCN Head (randomly initialized, binary output)
    """

    def __init__(self, pretrained_model, use_adapter=False, in_channels=6):
        super().__init__()
        self.use_adapter = use_adapter

        if use_adapter:
            self.adapter = ChannelAdapter(in_channels, 6)
        else:
            self.adapter = None

        # Reuse pretrained backbone and neck
        self.backbone = pretrained_model.backbone
        self.neck = pretrained_model.neck

        # New decode head for binary fire prediction
        self.decode_head = FCNHead(
            in_channels=768,  # neck output channels
            channels=256,
            num_classes=2,    # fire / no-fire
            num_convs=2,      # deeper head for better features
            dropout_ratio=0.1,
        )

    def forward(self, x):
        """x: (B, C, T, H, W) → logits (B, 2, H, W)."""
        if self.adapter is not None:
            x = self.adapter(x)

        tokens = self.backbone(x)        # (B, 1+N, D)
        features = self.neck(tokens)      # (B, D_out, H, W)
        logits = self.decode_head(features)  # (B, 2, H, W)
        return logits


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
#  TRAINING LOOP
# ═══════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_metrics = defaultdict(float)
    n_batches = 0

    for batch_idx, (features, labels) in enumerate(loader):
        features = features.to(device)  # (B, C, T, H, W)
        labels = labels.to(device)      # (B, H, W)

        optimizer.zero_grad()
        logits = model(features)

        # Resize logits to match label size if needed
        if logits.shape[-2:] != labels.shape[-2:]:
            logits = F.interpolate(logits, size=labels.shape[-2:],
                                   mode="bilinear", align_corners=False)

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        metrics = compute_metrics(logits.detach(), labels)
        for k, v in metrics.items():
            total_metrics[k] += v
        n_batches += 1

        if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
            sys.stdout.write(
                f"\r  Epoch {epoch} [{batch_idx+1}/{len(loader)}] "
                f"Loss={loss.item():.4f}  F1={metrics['f1']:.3f}  "
                f"IoU={metrics['iou']:.3f}"
            )
            sys.stdout.flush()

    avg_loss = total_loss / max(n_batches, 1)
    avg_metrics = {k: v / max(n_batches, 1) for k, v in total_metrics.items()}
    print(f"\r  Epoch {epoch} — Loss={avg_loss:.4f}  F1={avg_metrics['f1']:.3f}  "
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
    print("  Prithvi 100M → WildfireSpreadTS Finetuning")
    print("=" * 70)

    device = torch.device(args.device)
    print(f"\n  Device:     {device}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Data dir:   {args.data_dir}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  LR:         {args.lr}")
    print(f"  Freeze BB:  {args.freeze_backbone}")
    print(f"  Adapter:    {args.use_adapter}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Step 1: Load pretrained Prithvi model ──
    print(f"\n[1] Loading pretrained Prithvi model ...")
    pretrained = load_prithvi_burn_scar_model(args.checkpoint, device="cpu")
    print(f"    Pretrained model loaded (6-band, 224×224)")

    # ── Step 2: Build wildfire model ──
    print(f"\n[2] Building wildfire spread model ...")
    in_channels = 22 if args.use_adapter else 6
    model = PrithviWildfireModel(
        pretrained_model=pretrained,
        use_adapter=args.use_adapter,
        in_channels=in_channels,
    )

    # Freeze backbone if requested
    if args.freeze_backbone:
        print("    Freezing backbone weights ...")
        for param in model.backbone.parameters():
            param.requires_grad = False
        for param in model.neck.parameters():
            param.requires_grad = False
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"    Trainable: {trainable:,} / {total:,} params "
              f"({trainable/total*100:.1f}%)")
    else:
        total = sum(p.numel() for p in model.parameters())
        print(f"    Total params: {total:,}")

    model.to(device)

    # ── Step 3: Prepare datasets ──
    print(f"\n[3] Preparing datasets ...")

    # Discover all fire events and split 80/20
    all_fire_dirs = []
    for year_dir in sorted(Path(args.data_dir).iterdir()):
        if year_dir.is_dir():
            for fire_dir in sorted(year_dir.iterdir()):
                if fire_dir.is_dir():
                    tifs = list(fire_dir.glob("*.tif"))
                    if len(tifs) >= 5:  # Need at least 5 days
                        all_fire_dirs.append(str(fire_dir))

    print(f"    Found {len(all_fire_dirs)} fire events with ≥5 days")

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

    # Create temp directories with symlinks for train/val
    train_dataset = WildfireSpreadDataset(
        data_dir=args.data_dir,
        tile_size=args.tile_size,
        use_all_bands=args.use_adapter,
        augment=True,
        max_samples=args.max_samples,
    )
    val_dataset = WildfireSpreadDataset(
        data_dir=args.data_dir,
        tile_size=args.tile_size,
        use_all_bands=args.use_adapter,
        augment=False,
        max_samples=args.max_samples // 4 if args.max_samples else None,
    )

    # Manual train/val split based on fire events
    if len(all_fire_dirs) >= 2:
        val_prefixes = set(val_dirs)
        train_pairs = [p for p in train_dataset.pairs
                       if any(p[0].startswith(d) for d in train_dirs)]
        val_pairs = [p for p in val_dataset.pairs
                     if any(p[0].startswith(d) for d in val_dirs)]
        if train_pairs:
            train_dataset.pairs = train_pairs
        if val_pairs:
            val_dataset.pairs = val_pairs

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

    # ── Step 4: Setup training ──
    print(f"\n[4] Setting up training ...")

    # Compute class weights based on fire pixel ratio
    fire_ratio = 0.05  # Approximate: ~5% fire pixels
    pos_weight = (1.0 - fire_ratio) / fire_ratio
    print(f"    Pos weight: {pos_weight:.1f} (fire ratio ~{fire_ratio*100:.0f}%)")

    criterion = CombinedLoss(
        ce_weight=1.0, dice_weight=1.0, pos_weight=min(pos_weight, 20.0)
    ).to(device)

    # Different LR for adapter, head, and backbone
    param_groups = []
    if args.use_adapter and model.adapter is not None:
        param_groups.append({
            "params": model.adapter.parameters(),
            "lr": args.lr
        })
    param_groups.append({
        "params": model.decode_head.parameters(),
        "lr": args.lr
    })
    if not args.freeze_backbone:
        param_groups.append({
            "params": model.backbone.parameters(),
            "lr": args.lr * 0.1  # Lower LR for pretrained backbone
        })
        param_groups.append({
            "params": model.neck.parameters(),
            "lr": args.lr * 0.1
        })

    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    # Cosine annealing scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # ── Step 5: Training loop ──
    print(f"\n[5] Starting training ...")
    print(f"    {'─' * 60}")

    best_val_f1 = 0.0
    best_epoch = 0
    history = {"train_loss": [], "val_loss": [],
               "train_f1": [], "val_f1": []}

    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        t_ep = time.time()

        train_loss, train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
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
            save_path = os.path.join(args.output_dir, "best_wildfire_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": best_val_f1,
                "val_loss": val_loss,
                "args": vars(args),
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

    # ── Step 6: Save final model ──
    final_path = os.path.join(args.output_dir, "final_wildfire_model.pth")
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "history": history,
        "args": vars(args),
    }, final_path)

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print(f"  FINETUNING COMPLETE")
    print(f"{'=' * 70}")
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

    fig.suptitle("Prithvi → WildfireSpreadTS Finetuning", fontsize=14)
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
        description="Finetune Prithvi 100M on WildfireSpreadTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--checkpoint", type=str,
                   default=os.path.join(_BASE_DIR, "Model",
                                        "burn_scars_Prithvi_100M.pth"),
                   help="Path to pretrained Prithvi burn-scar checkpoint")
    p.add_argument("--data_dir", type=str,
                   default=os.path.join(_BASE_DIR, "DATA", "WildfireSpreadTS"),
                   help="Path to WildfireSpreadTS data")
    p.add_argument("--output_dir", type=str,
                   default=os.path.join(_BASE_DIR, "OUTPUTS", "wildfire_finetune"),
                   help="Output directory for checkpoints and logs")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--tile_size", type=int, default=224)
    p.add_argument("--freeze_backbone", action="store_true",
                   help="Freeze the pretrained ViT backbone (only train head)")
    p.add_argument("--use_adapter", action="store_true",
                   help="Use a learnable 22→6 channel adapter (uses all bands)")
    p.add_argument("--max_samples", type=int, default=None,
                   help="Max training samples (for quick testing)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_finetuning(args)
