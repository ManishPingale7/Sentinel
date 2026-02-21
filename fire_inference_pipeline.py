"""
Burn-Scar / Fire Inference Pipeline — Prithvi 100M
====================================================
Scans DATA/Fire/training and DATA/Fire/validation for HLS burn-scar
tiles (*_merged.tif + *.mask.tif), runs the Prithvi-100M fine-tuned
burn-scar model with sliding-window inference, evaluates against
ground-truth masks, and saves:

  - GeoTIFF burn-scar predictions per tile
  - Per-tile and per-split metrics CSV
  - Summary report across all splits

Usage:
    python fire_inference_pipeline.py
    python fire_inference_pipeline.py --data_root "D:\\path\\to\\Fire"
    python fire_inference_pipeline.py --splits validation
    python fire_inference_pipeline.py --device cuda
"""

import argparse
import glob
import json
import math
import os
import pathlib
import re
import sys
import time
from collections import defaultdict

import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import binary_closing, binary_opening

# ── Fix PosixPath on Windows ──
pathlib.PosixPath = pathlib.WindowsPath


# ═══════════════════════════════════════════════════════════
#  MODEL ARCHITECTURE  (pure PyTorch, no mmseg needed)
# ═══════════════════════════════════════════════════════════

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = np.asarray(pos).reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def get_3d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    t_size, h_size, w_size = grid_size
    w_embed_dim = embed_dim // 16 * 6
    h_embed_dim = embed_dim // 16 * 6
    t_embed_dim = embed_dim // 16 * 4
    w_pos = get_1d_sincos_pos_embed_from_grid(w_embed_dim, np.arange(w_size))
    h_pos = get_1d_sincos_pos_embed_from_grid(h_embed_dim, np.arange(h_size))
    t_pos = get_1d_sincos_pos_embed_from_grid(t_embed_dim, np.arange(t_size))
    w_pos = np.tile(w_pos, (t_size * h_size, 1))
    h_pos = np.tile(np.repeat(h_pos, w_size, axis=0), (t_size, 1))
    t_pos = np.repeat(t_pos, h_size * w_size, axis=0)
    pos_embed = np.concatenate((w_pos, h_pos, t_pos), axis=1)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


class PatchEmbed3D(nn.Module):
    """3D patch embedding for temporal ViT."""

    def __init__(self, img_size=224, patch_size=16, num_frames=1,
                 tubelet_size=1, in_chans=6, embed_dim=768):
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.grid_size = (num_frames // tubelet_size,
                          img_size // patch_size,
                          img_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        self.proj = nn.Conv3d(
            in_chans, embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )

    def forward(self, x):
        # x: (B, C, T, H, W)
        x = self.proj(x)  # (B, embed_dim, t, h, w)
        x = x.flatten(2).transpose(1, 2)  # (B, N, embed_dim)
        return x


class Norm2d(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        return x.permute(0, 3, 1, 2).contiguous()


class ConvTransformerTokensToEmbeddingNeck(nn.Module):
    """Upsamples tokens from (B, N, D) → (B, D_out, H_out, W_out) via 4x ConvTranspose2d."""

    def __init__(self, embed_dim=768, output_embed_dim=768, Hp=14, Wp=14,
                 drop_cls_token=True):
        super().__init__()
        self.drop_cls_token = drop_cls_token
        self.Hp = Hp
        self.Wp = Wp
        self.embed_dim = embed_dim
        self.output_embed_dim = output_embed_dim

        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, output_embed_dim, kernel_size=2, stride=2),
            Norm2d(output_embed_dim),
            nn.GELU(),
            nn.ConvTranspose2d(output_embed_dim, output_embed_dim, kernel_size=2, stride=2),
        )
        self.fpn2 = nn.Sequential(
            nn.ConvTranspose2d(output_embed_dim, output_embed_dim, kernel_size=2, stride=2),
            Norm2d(output_embed_dim),
            nn.GELU(),
            nn.ConvTranspose2d(output_embed_dim, output_embed_dim, kernel_size=2, stride=2),
        )

    def forward(self, x):
        if self.drop_cls_token:
            x = x[:, 1:, :]
        x = x.permute(0, 2, 1).reshape(x.shape[0], -1, self.Hp, self.Wp)
        x = self.fpn1(x)
        x = self.fpn2(x)
        return x


class FCNHead(nn.Module):
    """Simple FCN decode head: Conv+BN+ReLU → Conv 1x1 → num_classes."""

    def __init__(self, in_channels, channels, num_classes, num_convs=1, dropout_ratio=0.1):
        super().__init__()
        convs = []
        for i in range(num_convs):
            c_in = in_channels if i == 0 else channels
            convs.append(nn.Conv2d(c_in, channels, 3, padding=1, bias=False))
            convs.append(nn.BatchNorm2d(channels))
            convs.append(nn.ReLU(inplace=True))
        self.convs = nn.Sequential(*convs)
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        self.conv_seg = nn.Conv2d(channels, num_classes, 1)

    def forward(self, x):
        x = self.convs(x)
        x = self.dropout(x)
        x = self.conv_seg(x)
        return x


class TemporalViTEncoder(nn.Module):
    """Pure-PyTorch re-implementation of the TemporalViTEncoder from geospatial_fm."""

    def __init__(self, img_size=224, patch_size=16, num_frames=1,
                 tubelet_size=1, in_chans=6, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4.0):
        super().__init__()
        from timm.models.vision_transformer import Block

        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed3D(
            img_size, patch_size, num_frames, tubelet_size, in_chans, embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False,
        )
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True,
                  norm_layer=nn.LayerNorm)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Init sincos pos embed
        pos_embed = get_3d_sincos_pos_embed(
            embed_dim, self.patch_embed.grid_size, cls_token=True,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def forward(self, x):
        # x: (B, C, T, H, W)
        x = self.patch_embed(x)           # (B, N, D)
        x = x + self.pos_embed[:, 1:, :]  # add pos embed (no cls)
        cls_tokens = (self.cls_token + self.pos_embed[:, :1, :]).expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, 1+N, D)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x  # (B, 1+N, D)


class PrithviBurnScarModel(nn.Module):
    """Complete Prithvi burn-scar segmentation model: ViT backbone + neck + FCN head."""

    def __init__(self, img_size=224, patch_size=16, num_frames=1,
                 in_chans=6, embed_dim=768, depth=12, num_heads=12,
                 num_classes=2):
        super().__init__()
        self.backbone = TemporalViTEncoder(
            img_size=img_size, patch_size=patch_size, num_frames=num_frames,
            tubelet_size=1, in_chans=in_chans, embed_dim=embed_dim,
            depth=depth, num_heads=num_heads,
        )
        Hp = img_size // patch_size  # 14
        Wp = img_size // patch_size  # 14
        output_embed_dim = num_frames * embed_dim  # 768

        self.neck = ConvTransformerTokensToEmbeddingNeck(
            embed_dim=embed_dim, output_embed_dim=output_embed_dim,
            Hp=Hp, Wp=Wp, drop_cls_token=True,
        )
        self.decode_head = FCNHead(
            in_channels=output_embed_dim, channels=256,
            num_classes=num_classes, num_convs=1, dropout_ratio=0.1,
        )

    def forward(self, x):
        """x: (B, C, T, H, W) → logits (B, num_classes, H, W)."""
        tokens = self.backbone(x)        # (B, 1+N, D)
        features = self.neck(tokens)      # (B, D_out, H, W)
        logits = self.decode_head(features)  # (B, num_classes, H, W)
        return logits


def load_prithvi_burn_scar_model(checkpoint_path, device):
    """Build model and load state_dict from mmseg-style checkpoint."""
    model = PrithviBurnScarModel(
        img_size=224, patch_size=16, num_frames=1,
        in_chans=6, embed_dim=768, depth=12, num_heads=12, num_classes=2,
    )

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]

    # Map mmseg state_dict keys to our model
    new_sd = {}
    for k, v in sd.items():
        # Skip auxiliary_head (not used in inference)
        if k.startswith("auxiliary_head"):
            continue
        # decode_head: mmseg stores convs as indexed modules, ours is Sequential
        # mmseg: decode_head.convs.0.conv.weight → our: decode_head.convs.0.weight
        #        decode_head.convs.0.bn.weight   → our: decode_head.convs.1.weight
        if k.startswith("decode_head.convs."):
            parts = k.split(".")
            # parts: ['decode_head', 'convs', idx, 'conv'|'bn', param]
            conv_idx = int(parts[2])
            sub_type = parts[3]  # 'conv' or 'bn'
            param_name = parts[4]
            if sub_type == "conv":
                seq_idx = conv_idx * 3  # conv is at 0, 3, 6, ...
            elif sub_type == "bn":
                seq_idx = conv_idx * 3 + 1  # bn is at 1, 4, 7, ...
            else:
                continue
            new_key = f"decode_head.convs.{seq_idx}.{param_name}"
            new_sd[new_key] = v
        else:
            new_sd[k] = v

    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    if missing:
        print(f"  [WARN] Missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  [WARN] Unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    model.to(device)
    model.eval()
    return model


# ═══════════════════════════════════════════════════════════
#  CONFIG / DEFAULTS
# ═══════════════════════════════════════════════════════════

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULTS = dict(
    data_root   = os.path.join(_BASE_DIR, "DATA", "Fire"),
    model_path  = os.path.join(_BASE_DIR, "Model", "burn_scars_Prithvi_100M.pth"),
    output_root = os.path.join(_BASE_DIR, "OUTPUTS", "prithvi_burn_scars"),
    tile_size   = 224,
    stride      = 112,   # sliding window stride (tile_size // 2)
    img_suffix  = "_merged.tif",
    mask_suffix = ".mask.tif",
    nodata      = -9999,
    nodata_replace = 0,
    # HLS band normalization (from burn_scars.py config)
    means = [0.033349706741586264, 0.05701185520536176, 0.05889748132001316,
             0.2323245113436119, 0.1972854853760658, 0.11944914225186566],
    stds  = [0.02269135568823774, 0.026807560223070237, 0.04004109844362779,
             0.07791732423672691, 0.08708738838140137, 0.07241979477437814],
    bands = [0, 1, 2, 3, 4, 5],
)

PIXEL_RES_M = 30.0  # HLS is ~30 m/pixel


# ═══════════════════════════════════════════════════════════
#  DATA I/O
# ═══════════════════════════════════════════════════════════

def read_hls_tile(path, cfg):
    """Read a HLS *_merged.tif → normalised (C, H, W) float32 array, crs, transform."""
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)  # (C, H, W)
        crs = src.crs
        transform = src.transform
        nodata_val = src.nodata if src.nodata is not None else cfg["nodata"]

    # Select bands
    bands = cfg["bands"]
    data = data[bands]  # (6, H, W)

    # Build nodata mask before normalisation
    nodata_mask = np.any(np.isclose(data, nodata_val) | np.isnan(data), axis=0)

    # Replace nodata
    data[np.isclose(data, nodata_val)] = cfg["nodata_replace"]
    data = np.nan_to_num(data, nan=cfg["nodata_replace"])

    # Normalise per band
    means = np.array(cfg["means"], dtype=np.float32).reshape(-1, 1, 1)
    stds  = np.array(cfg["stds"],  dtype=np.float32).reshape(-1, 1, 1)
    data = (data - means) / (stds + 1e-10)

    return data, nodata_mask, crs, transform


def read_gt_mask(path):
    """Read ground-truth mask → (burn_binary_uint8, raw_mask).
    Expects: 0 = no-burn, 1 = burn scar."""
    with rasterio.open(path) as src:
        raw = src.read(1)
    # Handle potential -1 ignore values
    burn_binary = (raw == 1).astype(np.uint8)
    return burn_binary, raw


def save_geotiff(array, path, crs, transform, dtype="float32", nodata=None):
    """Write single-band GeoTIFF."""
    arr = array.astype(np.float32 if dtype == "float32" else np.int16).copy()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    meta = dict(
        driver="GTiff", height=arr.shape[0], width=arr.shape[1],
        count=1, dtype=dtype, crs=crs, transform=transform,
        compress="lzw",
    )
    if nodata is not None:
        meta["nodata"] = nodata
    with rasterio.open(path, "w", **meta) as dst:
        dst.write(arr, 1)


# ═══════════════════════════════════════════════════════════
#  SLIDING-WINDOW INFERENCE
# ═══════════════════════════════════════════════════════════

def sliding_window_inference(model, image, nodata_mask, device, cfg):
    """
    Run sliding-window inference on a single HLS tile.

    Args:
        model: PrithviBurnScarModel
        image: (C, H, W) normalised float32 numpy array
        nodata_mask: (H, W) bool — True where nodata
        device: torch device
        cfg: config dict with tile_size, stride

    Returns:
        pred_mask: (H, W) uint8 — 0 = no-burn, 1 = burn scar
    """
    C, H, W = image.shape
    tile_size = cfg["tile_size"]
    stride = cfg["stride"]
    num_classes = 2

    # Pad image so sliding window covers the whole image
    pad_h = (math.ceil((H - tile_size) / stride) * stride + tile_size) - H if H > tile_size else tile_size - H
    pad_w = (math.ceil((W - tile_size) / stride) * stride + tile_size) - W if W > tile_size else tile_size - W

    padded = np.pad(image, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
    nodata_padded = np.pad(nodata_mask, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=True)
    _, pH, pW = padded.shape

    # Accumulation buffers
    logits_sum = np.zeros((num_classes, pH, pW), dtype=np.float32)
    count_map  = np.zeros((pH, pW), dtype=np.float32)

    with torch.no_grad():
        for y in range(0, pH - tile_size + 1, stride):
            for x in range(0, pW - tile_size + 1, stride):
                patch = padded[:, y:y+tile_size, x:x+tile_size]  # (C, ts, ts)
                # Reshape to (1, C, T=1, ts, ts) for Prithvi input
                inp = torch.from_numpy(patch[None, :, None, :, :]).float().to(device)
                out = model(inp)  # (1, num_classes, ts, ts)
                logits_sum[:, y:y+tile_size, x:x+tile_size] += out[0].cpu().numpy()
                count_map[y:y+tile_size, x:x+tile_size] += 1.0

    # Average overlapping predictions
    count_map[count_map == 0] = 1.0
    avg_logits = logits_sum / count_map[None, :, :]

    # Crop back to original size
    avg_logits = avg_logits[:, :H, :W]
    nodata_orig = nodata_mask

    # Argmax → class prediction
    pred = np.argmax(avg_logits, axis=0).astype(np.uint8)

    # Mark nodata regions as 0 (no-burn)
    pred[nodata_orig] = 0

    # Optional morphological cleanup
    pred = binary_opening(pred, structure=np.ones((3, 3))).astype(np.uint8)
    pred = binary_closing(pred, structure=np.ones((3, 3))).astype(np.uint8)

    return pred


# ═══════════════════════════════════════════════════════════
#  METRICS
# ═══════════════════════════════════════════════════════════

def compute_metrics(pred_mask, gt_binary):
    """Returns dict with accuracy, precision, recall, f1, iou, tp/fp/fn/tn."""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, jaccard_score, confusion_matrix,
    )
    p = pred_mask.flatten().astype(int)
    g = gt_binary.flatten().astype(int)
    m = dict(
        accuracy  = accuracy_score(g, p),
        precision = precision_score(g, p, zero_division=0),
        recall    = recall_score(g, p, zero_division=0),
        f1        = f1_score(g, p, zero_division=0),
        iou       = jaccard_score(g, p, zero_division=0),
    )
    cm = confusion_matrix(g, p, labels=[0, 1])
    m["tn"], m["fp"], m["fn"], m["tp"] = cm.ravel()
    return m


def pixels_to_km2(n):
    return n * (PIXEL_RES_M ** 2) / 1e6


# ═══════════════════════════════════════════════════════════
#  TILE / EVENT DISCOVERY
# ═══════════════════════════════════════════════════════════

def discover_tiles(data_root, split, cfg):
    """Find all image/mask pairs in a split folder.
    Returns list of dicts: {tile_id, image_path, mask_path}."""
    split_dir = os.path.join(data_root, split)
    if not os.path.isdir(split_dir):
        return []

    img_suffix = cfg["img_suffix"]
    mask_suffix = cfg["mask_suffix"]

    img_files = sorted(glob.glob(os.path.join(split_dir, f"*{img_suffix}")))
    tiles = []
    for img_path in img_files:
        fname = os.path.basename(img_path)
        # Derive tile_id by stripping the suffix
        tile_id = fname.replace(img_suffix, "")
        mask_name = tile_id + mask_suffix
        mask_path = os.path.join(split_dir, mask_name)
        tiles.append(dict(
            tile_id=tile_id,
            image_path=img_path,
            mask_path=mask_path if os.path.exists(mask_path) else None,
        ))
    return tiles


def parse_tile_info(tile_id):
    """Parse MGRS grid cell, year, DOY from an HLS tile ID.
    e.g. 'subsetted_512x512_HLS.S30.T10SEH.2018245.v1.4'
      → {'mgrs': 'T10SEH', 'year': 2018, 'doy': 245, 'event_id': 'FIRE_T10SEH_2018'}
    """
    m = re.search(r'HLS\.S30\.(T\w+)\.(\d{4})(\d{3})\.', tile_id)
    if m:
        mgrs = m.group(1)
        year = int(m.group(2))
        doy = int(m.group(3))
        return {
            'mgrs': mgrs,
            'year': year,
            'doy': doy,
            'event_id': f'FIRE_{mgrs}_{year}',
        }
    return None


def discover_events(data_root, splits, cfg):
    """Scan all splits and group tiles into fire events by MGRS cell + year.
    Returns OrderedDict: {event_id: [tile_info_dicts]} sorted by event_id.
    Each tile_info dict has: tile_id, image_path, mask_path, event_id, mgrs, year, doy, source_split.
    """
    events = defaultdict(list)
    for split in splits:
        tiles = discover_tiles(data_root, split, cfg)
        for t in tiles:
            info = parse_tile_info(t['tile_id'])
            if info:
                t['event_id'] = info['event_id']
                t['mgrs'] = info['mgrs']
                t['year'] = info['year']
                t['doy'] = info['doy']
                t['source_split'] = split
                events[info['event_id']].append(t)
            else:
                t['event_id'] = f'FIRE_UNKNOWN_{split}'
                t['source_split'] = split
                events[t['event_id']].append(t)

    # Sort tiles within each event by DOY
    for eid in events:
        events[eid].sort(key=lambda x: (x.get('year', 0), x.get('doy', 0)))

    return dict(sorted(events.items()))


# ═══════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ═══════════════════════════════════════════════════════════

def run_pipeline(cfg):
    print("=" * 72)
    print("  Prithvi Burn-Scar Inference Pipeline — HLS Fire Data (Event-Based)")
    print("=" * 72)

    splits = cfg.get("splits", ["training", "validation"])
    if isinstance(splits, str):
        splits = [splits]

    # ── Discover tiles and group by fire event ──
    all_event_data = discover_events(cfg["data_root"], splits, cfg)

    # Filter to requested events if specified
    requested_events = cfg.get("events")
    if requested_events:
        all_event_data = {k: v for k, v in all_event_data.items()
                          if k in requested_events}

    total_tiles = sum(len(t) for t in all_event_data.values())
    total_events = len(all_event_data)

    print(f"\n  Data root:   {cfg['data_root']}")
    print(f"  Splits:      {', '.join(splits)}")
    print(f"  Events:      {total_events}")
    for eid, tiles in all_event_data.items():
        print(f"    {eid}: {len(tiles)} tiles")
    print(f"  Total tiles: {total_tiles}")
    print(f"  Output:      {cfg['output_root']}")
    print(f"  Model:       {cfg['model_path']}")
    print(f"  Tile size:   {cfg['tile_size']}  Stride: {cfg['stride']}")

    # ── Load model ──
    print(f"\n[1] Loading Prithvi burn-scar model ...")
    device = torch.device(cfg.get("device", "cpu"))
    if cfg.get("device") == "cuda" and not torch.cuda.is_available():
        print("    CUDA not available, falling back to CPU")
        device = torch.device("cpu")
    model = load_prithvi_burn_scar_model(cfg["model_path"], device)
    print(f"    Device: {device}   OK\n")

    # ── Process each event ──
    print("[2] Running inference ...\n")
    all_event_results = []
    global_idx = 0
    t_global = time.time()

    for event_id, tiles in all_event_data.items():
        if not tiles:
            continue
        max_tiles = cfg.get("_max_tiles")
        if max_tiles is not None:
            tiles = tiles[:max_tiles]
        n = len(tiles)
        out_dir = os.path.join(cfg["output_root"], event_id)
        geo_dir = os.path.join(out_dir, "geotiffs")
        os.makedirs(geo_dir, exist_ok=True)

        print(f"  +-- {event_id}  ({n} tiles) " + "-" * 30)
        event_metrics = []
        t_event = time.time()

        for idx, tile_info in enumerate(tiles):
            global_idx += 1
            tid = tile_info["tile_id"]
            t0 = time.time()

            # Read image
            image, nodata_mask, crs, transform = read_hls_tile(tile_info["image_path"], cfg)

            # Inference
            pred_mask = sliding_window_inference(model, image, nodata_mask, device, cfg)
            n_pred = int(pred_mask.sum())
            elapsed = time.time() - t0

            # Save GeoTIFF prediction
            tif_name = f"{tid}_burn_pred.tif"
            save_geotiff(
                pred_mask.astype(np.int16), os.path.join(geo_dir, tif_name),
                crs, transform, dtype="int16", nodata=-1,
            )

            # Evaluate against ground-truth
            status = ""
            if tile_info["mask_path"] is not None:
                gt_bin, gt_raw = read_gt_mask(tile_info["mask_path"])
                m = compute_metrics(pred_mask, gt_bin)
                m.update(
                    event=event_id, tile_id=tid, n_pred=n_pred,
                    n_gt=int(gt_bin.sum()), time_s=round(elapsed, 2),
                    pred_km2=round(pixels_to_km2(n_pred), 4),
                    gt_km2=round(pixels_to_km2(gt_bin.sum()), 4),
                )
                event_metrics.append(m)
                status = (f"F1={m['f1']:.3f}  IoU={m['iou']:.3f}  "
                          f"Prec={m['precision']:.3f}  Rec={m['recall']:.3f}")
            else:
                status = f"Pred={n_pred} px ({pixels_to_km2(n_pred):.4f} km²)  [no GT]"

            print(f"  |  [{idx+1}/{n}] {tid[:40]}...  {status}  ({elapsed:.1f}s)")

        event_time = time.time() - t_event

        # ── Per-event CSV ──
        if event_metrics:
            csv_path = os.path.join(out_dir, f"{event_id}_metrics.csv")
            _write_metrics_csv(csv_path, event_metrics, event_id)

            avg_f1 = np.mean([m["f1"] for m in event_metrics])
            avg_iou = np.mean([m["iou"] for m in event_metrics])
            print(f"  |  Avg F1={avg_f1:.3f}  Avg IoU={avg_iou:.3f}")
            print(f"  |  Metrics -> {csv_path}")

        print(f"  +-- {event_id} done ({event_time:.1f}s)\n")
        all_event_results.append(dict(
            event=event_id, n_tiles=n, time_s=event_time,
            metrics=event_metrics,
        ))

    total_time = time.time() - t_global

    # ── Global summary CSV ──
    print("[3] Writing global summary ...\n")
    all_metrics = [m for er in all_event_results for m in er["metrics"]]
    if all_metrics:
        global_csv = os.path.join(cfg["output_root"], "all_events_metrics.csv")
        _write_metrics_csv(global_csv, all_metrics, "ALL")
        print(f"  Global CSV -> {global_csv}")

    # ── Write event index JSON ──
    event_index = {}
    for event_id, evtiles in all_event_data.items():
        capped = evtiles[:cfg.get("_max_tiles") or len(evtiles)]
        event_index[event_id] = {
            "tiles": [
                {
                    "tile_id": t["tile_id"],
                    "source_split": t.get("source_split", "unknown"),
                    "mgrs": t.get("mgrs", ""),
                    "year": t.get("year", 0),
                    "doy": t.get("doy", 0),
                }
                for t in capped
            ]
        }
    index_path = os.path.join(cfg["output_root"], "fire_events_index.json")
    with open(index_path, "w") as f:
        json.dump(event_index, f, indent=2)
    print(f"  Event index -> {index_path}")

    # ── Print summary table ──
    print(f"\n  {'-' * 80}")
    print(f"  {'Event':<24} | {'Tiles':>5} | {'Avg F1':>7} | {'Avg IoU':>7} | "
          f"{'Avg Prec':>8} | {'Avg Rec':>7} | {'Time':>6}")
    print(f"  {'-' * 80}")
    for er in all_event_results:
        ev = er["event"]
        n = er["n_tiles"]
        t = er["time_s"]
        if er["metrics"]:
            af1 = np.mean([m["f1"] for m in er["metrics"]])
            aio = np.mean([m["iou"] for m in er["metrics"]])
            apr = np.mean([m["precision"] for m in er["metrics"]])
            are = np.mean([m["recall"] for m in er["metrics"]])
            print(f"  {ev:<24} | {n:>5} | {af1:>7.4f} | {aio:>7.4f} | "
                  f"{apr:>8.4f} | {are:>7.4f} | {t:>5.1f}s")
        else:
            print(f"  {ev:<24} | {n:>5} | {'N/A':>7} | {'N/A':>7} | "
                  f"{'N/A':>8} | {'N/A':>7} | {t:>5.1f}s")
    print(f"  {'-' * 80}")

    if all_metrics:
        g_f1  = np.mean([m["f1"]  for m in all_metrics])
        g_iou = np.mean([m["iou"] for m in all_metrics])
        g_pr  = np.mean([m["precision"] for m in all_metrics])
        g_re  = np.mean([m["recall"] for m in all_metrics])
        print(f"  {'GLOBAL':<24} | {total_tiles:>5} | {g_f1:>7.4f} | {g_iou:>7.4f} | "
              f"{g_pr:>8.4f} | {g_re:>7.4f} |")

    print(f"\n  Total time: {total_time:.1f}s")

    # ── Done ──
    print("\n" + "=" * 72)
    print("  PIPELINE COMPLETE")
    print("=" * 72)
    print(f"""
  Outputs: {cfg['output_root']}
    +-- <event>/
    |   +-- geotiffs/            Burn-scar prediction GeoTIFFs per tile
    |   +-- <event>_metrics.csv
    +-- all_events_metrics.csv
    +-- fire_events_index.json
""")
    return all_event_results


def _write_metrics_csv(path, metrics_list, label):
    """Write a list of metric dicts to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cols = ["event", "tile_id", "accuracy", "precision", "recall", "f1", "iou",
            "tp", "fp", "fn", "tn", "n_pred", "n_gt", "pred_km2", "gt_km2", "time_s"]
    with open(path, "w") as f:
        f.write(",".join(cols) + "\n")
        for m in metrics_list:
            vals = [str(m.get(c, "")) for c in cols]
            f.write(",".join(vals) + "\n")
        # Average row
        avg_cols = ["accuracy", "precision", "recall", "f1", "iou"]
        avgs = {c: np.mean([m[c] for m in metrics_list]) for c in avg_cols}
        f.write(f"AVERAGE,,{avgs['accuracy']:.4f},{avgs['precision']:.4f},"
                f"{avgs['recall']:.4f},{avgs['f1']:.4f},{avgs['iou']:.4f}"
                + ",,,,,,,\n")


# ═══════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Prithvi burn-scar inference pipeline for HLS Fire data (event-based)",
    )
    p.add_argument("--data_root", default=DEFAULTS["data_root"],
                   help="Root directory of Fire data (contains training/ and validation/)")
    p.add_argument("--splits", nargs="*", default=["training", "validation"],
                   help="Which data splits to scan (e.g. training validation)")
    p.add_argument("--events", nargs="*", default=None,
                   help="Process only these fire events (e.g. FIRE_T10SEH_2018 FIRE_T10SEH_2020)")
    p.add_argument("--model_path", default=DEFAULTS["model_path"],
                   help="Path to burn_scars_Prithvi_100M.pth")
    p.add_argument("--output_root", default=DEFAULTS["output_root"],
                   help="Root output directory")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                   help="Inference device")
    p.add_argument("--tile_size", type=int, default=DEFAULTS["tile_size"],
                   help="Sliding window tile size (default: 224)")
    p.add_argument("--stride", type=int, default=DEFAULTS["stride"],
                   help="Sliding window stride (default: 112)")
    p.add_argument("--max_tiles", type=int, default=None,
                   help="Max tiles per event (for quick testing)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = dict(DEFAULTS)
    cfg.update(
        data_root   = args.data_root,
        splits      = args.splits,
        model_path  = args.model_path,
        output_root = args.output_root,
        device      = args.device,
        tile_size   = args.tile_size,
        stride      = args.stride,
    )

    # Optional: filter to specific events
    if args.events:
        cfg["events"] = args.events

    # Optional: limit tiles per event for quick testing
    if args.max_tiles is not None:
        cfg["_max_tiles"] = args.max_tiles

    run_pipeline(cfg)


if __name__ == "__main__":
    main()
