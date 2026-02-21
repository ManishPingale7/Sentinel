"""
Prithvi-EO-2.0-300M-TL — Direct Inference on WildfireSpreadTS
================================================================

Runs the pretrained Prithvi-EO-2.0-300M-TL model directly on wildfire
data WITHOUT finetuning. Uses the MAE self-supervised reconstruction
to detect fire anomalies:

  1. MAE Reconstruction — Mask 75% of patches, reconstruct, measure error.
     Fire regions produce higher reconstruction error = anomaly signal.

  2. Feature Extraction — Extract encoder features from 4 consecutive days,
     visualize latent space clusterings (PCA) to see fire vs non-fire.

  3. Temporal Anomaly — Compare reconstruction error across days to detect
     where the landscape changed most (fire progression).

No finetuning or labels needed. Just the pretrained self-supervised model.

Usage:
    python inference_prithvi_v2_wildfire.py
    python inference_prithvi_v2_wildfire.py --data_dir "D:\\path\\to\\WildfireSpreadTS"
    python inference_prithvi_v2_wildfire.py --fire_id fire_21889697 --n_masks 5
    python inference_prithvi_v2_wildfire.py --device cuda

Dependencies:
    pip install torch numpy rasterio einops timm matplotlib
"""

import argparse
import gc
import importlib.util
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn.functional as F

# ═══════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

HF_MODEL_ID = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL"
HF_BASE_URL = f"https://huggingface.co/{HF_MODEL_ID}/resolve/main"
HF_CHECKPOINT_URL = f"{HF_BASE_URL}/Prithvi_EO_V2_300M_TL.pt"
HF_MAE_SOURCE_URL = f"{HF_BASE_URL}/prithvi_mae.py"

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

# WildfireSpreadTS bands
BAND_NAMES = [
    "M11", "I2", "I1",
    "NDVI_last", "EVI2_last",
    "total_precip", "wind_speed",
    "wind_dir", "min_temp", "max_temp",
    "energy_release", "specific_humidity",
    "slope", "aspect", "elevation",
    "pdsi", "LC_Type1",
    "total_precip_surface",
    "forecast_wind_speed", "forecast_wind_dir",
    "forecast_temp", "forecast_specific_humidity",
    "active_fire",  # band 22 — ground truth label
]

SELECTED_BANDS = [3, 4, 6, 9, 10, 14]  # NDVI, EVI2, wind, temp, energy, elev
FIRE_LABEL_BAND = 22


# ═══════════════════════════════════════════════════════════════
#  DOWNLOAD UTILITIES
# ═══════════════════════════════════════════════════════════════

def download_file(url, dest_path, desc=None):
    """Download a file with progress."""
    desc = desc or os.path.basename(dest_path)
    print(f"  Downloading {desc} ...")
    tmp = dest_path + ".tmp"
    try:
        def _prog(bc, bs, ts):
            if ts > 0:
                pct = min(100.0, bc * bs * 100.0 / ts)
                sys.stdout.write(f"\r  {bc*bs/1e6:.1f}/{ts/1e6:.1f} MB ({pct:.0f}%)")
                sys.stdout.flush()
        urllib.request.urlretrieve(url, tmp, reporthook=_prog)
        print()
        os.replace(tmp, dest_path)
    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise RuntimeError(f"Download failed: {e}") from e
    return dest_path


def ensure_prithvi_mae_source(dest_dir=None):
    dest_dir = dest_dir or _BASE_DIR
    path = os.path.join(dest_dir, "prithvi_mae.py")
    if not os.path.exists(path):
        download_file(HF_MAE_SOURCE_URL, path, "prithvi_mae.py")
    return path


def ensure_checkpoint(dest_dir=None):
    dest_dir = dest_dir or os.path.join(_BASE_DIR, "Model")
    os.makedirs(dest_dir, exist_ok=True)
    path = os.path.join(dest_dir, "Prithvi_EO_V2_300M_TL.pt")
    if not os.path.exists(path):
        download_file(HF_CHECKPOINT_URL, path, "Prithvi_EO_V2_300M_TL.pt (1.33 GB)")
    return path


def load_prithvi_mae_module(mae_path):
    spec = importlib.util.spec_from_file_location("prithvi_mae", mae_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ═══════════════════════════════════════════════════════════════
#  LOAD FULL MAE MODEL (encoder + decoder for reconstruction)
# ═══════════════════════════════════════════════════════════════

def load_prithvi_v2_mae(checkpoint_path=None, device="cpu"):
    """
    Load the full PrithviMAE model (encoder + decoder).
    The decoder is needed for reconstruction-based anomaly detection.
    """
    mae_path = ensure_prithvi_mae_source()
    prithvi_module = load_prithvi_mae_module(mae_path)

    ckpt_path = checkpoint_path or ensure_checkpoint()

    cfg = DEFAULT_MODEL_CONFIG.copy()
    print(f"  Creating PrithviMAE (embed_dim={cfg['embed_dim']}, "
          f"depth={cfg['depth']}, frames={cfg['num_frames']})")

    mae = prithvi_module.PrithviMAE(
        img_size=cfg["img_size"],
        patch_size=cfg["patch_size"],
        num_frames=cfg["num_frames"],
        in_chans=cfg["in_chans"],
        embed_dim=cfg["embed_dim"],
        depth=cfg["depth"],
        num_heads=cfg["num_heads"],
        decoder_embed_dim=cfg["decoder_embed_dim"],
        decoder_depth=cfg["decoder_depth"],
        decoder_num_heads=cfg["decoder_num_heads"],
        mlp_ratio=cfg["mlp_ratio"],
        coords_encoding=cfg.get("coords_encoding"),
        coords_scale_learn=cfg.get("coords_scale_learn", False),
    )

    print(f"  Loading checkpoint: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)

    for k in list(state_dict.keys()):
        if "pos_embed" in k:
            del state_dict[k]

    missing, unexpected = mae.load_state_dict(state_dict, strict=False)
    print(f"  Missing keys: {len(missing)} (pos_embed — recomputed)")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")

    total = sum(p.numel() for p in mae.parameters())
    print(f"  Total MAE params: {total:,}")

    mae.eval()
    mae.to(device)
    return mae


# ═══════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════

def discover_fire_events(data_dir):
    """Find all fire events with their TIF files."""
    root = Path(data_dir)
    year_dirs = [d for d in sorted(root.iterdir())
                 if d.is_dir() and d.name.isdigit()]
    if not year_dirs:
        for sub in sorted(root.iterdir()):
            if sub.is_dir():
                year_dirs.extend(
                    d for d in sorted(sub.iterdir())
                    if d.is_dir() and d.name.isdigit()
                )

    events = {}
    for year_dir in year_dirs:
        for fire_dir in sorted(year_dir.iterdir()):
            if not fire_dir.is_dir():
                continue
            tifs = sorted(fire_dir.glob("*.tif"))
            if len(tifs) >= 2:
                events[fire_dir.name] = {
                    "path": str(fire_dir),
                    "tifs": [str(t) for t in tifs],
                    "n_days": len(tifs),
                }
    return events


def load_tif(path):
    """Load a GeoTIFF -> (23, H, W) float32."""
    with rasterio.open(path) as ds:
        return ds.read().astype(np.float32)


def prepare_input(tif_paths, num_frames=4, tile_size=224):
    """
    Load num_frames consecutive TIFs, extract bands, normalize,
    crop/pad to tile_size.

    Returns:
        features: (1, 6, num_frames, tile_size, tile_size)  tensor
        labels:   list of (H, W) numpy arrays (one per day, for visualization)
        raw_rgb:  list of (3, H, W) numpy arrays (VIIRS bands for display)
    """
    day_features = []
    labels = []
    raw_rgbs = []

    for p in tif_paths[:num_frames]:
        data = load_tif(p)

        # Extract feature bands
        feats = data[SELECTED_BANDS]  # (6, H, W)

        # Extract ground truth fire mask
        fire = data[FIRE_LABEL_BAND]
        label = np.where(~np.isnan(fire), 1.0, 0.0).astype(np.float32)

        # Extract pseudo-RGB (VIIRS I1, I2, M11 → bands 2, 1, 0)
        rgb = data[[2, 1, 0]]

        day_features.append(feats)
        labels.append(label)
        raw_rgbs.append(rgb)

    # Stack: (6, T, H, W)
    features = np.stack(day_features, axis=1)

    # Clean
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    # Per-band normalization across all timesteps
    for c in range(features.shape[0]):
        bd = features[c]
        valid = bd != 0
        if valid.any():
            mu = bd[valid].mean()
            std = bd[valid].std() + 1e-8
            features[c] = np.where(valid, (bd - mu) / std, 0.0)

    # Pad/crop to tile_size
    _, _, h, w = features.shape
    if h < tile_size or w < tile_size:
        ph = max(0, tile_size - h)
        pw = max(0, tile_size - w)
        features = np.pad(features, ((0, 0), (0, 0), (0, ph), (0, pw)),
                          mode="reflect")
    features = features[:, :, :tile_size, :tile_size]

    tensor = torch.from_numpy(features).float().unsqueeze(0)  # (1, 6, T, H, W)
    return tensor, labels, raw_rgbs


# ═══════════════════════════════════════════════════════════════
#  INFERENCE: MAE RECONSTRUCTION ANOMALY
# ═══════════════════════════════════════════════════════════════

def mae_reconstruction_anomaly(mae, input_tensor, n_masks=10, device="cpu"):
    """
    Run MAE with random masking multiple times, average reconstruction error.
    Fire regions produce higher reconstruction error = anomaly signal.

    Returns:
        anomaly_map: (H, W) averaged per-pixel reconstruction error
    """
    mae.eval()
    input_tensor = input_tensor.to(device)
    B, C, T, H, W = input_tensor.shape

    all_errors = []
    all_masks = []

    with torch.no_grad():
        for mask_idx in range(n_masks):
            # Forward through MAE (with masking)
            loss, pred, mask = mae(input_tensor)

            # pred: (B, num_patches, patch_dim) — in patch space
            # mask: (B, num_patches) — 1 = masked (reconstructed), 0 = visible

            # Unpatchify pred and input to pixel space for per-pixel comparison
            recon = mae.unpatchify(pred)  # (B, C, T, H, W) or (B, C*T, H, W)

            # Ensure shapes match
            target = input_tensor
            if recon.shape != target.shape:
                # unpatchify might merge time into channels
                recon = recon.reshape(target.shape)

            # Per-pixel MSE: (B, C, T, H, W) -> mean over C,T -> (B, H, W)
            pixel_error = ((recon - target) ** 2).mean(dim=(1, 2))  # (B, H, W)

            # We only trust error on masked patches — build a spatial mask
            # mask: (B, N) where N = t_grid * h_grid * w_grid
            patch_h, patch_w = H // 16, W // 16
            t_grid = T  # patch_size[0] = 1
            spatial_mask = mask.reshape(B, t_grid, patch_h, patch_w)
            spatial_mask = spatial_mask.mean(dim=1)  # avg across time -> (B, ph, pw)

            # Upsample mask to full resolution
            spatial_mask = F.interpolate(
                spatial_mask.unsqueeze(1), size=(H, W),
                mode="nearest"
            ).squeeze(1)  # (B, H, W)

            # Zero out error on visible (non-masked) patches
            masked_error = pixel_error * spatial_mask  # (B, H, W)

            all_errors.append(masked_error[0].cpu().numpy())
            all_masks.append(spatial_mask[0].cpu().numpy())

    # Stack and normalize by mask frequency
    errors = np.stack(all_errors, axis=0)   # (n_masks, H, W)
    masks = np.stack(all_masks, axis=0)     # (n_masks, H, W)

    mask_sum = masks.sum(axis=0)
    mask_sum = np.maximum(mask_sum, 1e-8)
    anomaly_map = errors.sum(axis=0) / mask_sum  # (H, W)

    return anomaly_map


def feature_extraction(mae, input_tensor, device="cpu"):
    """
    Extract encoder features for PCA visualization.

    Returns:
        spatial_features: (4096, 14, 14) — last layer features
        pca_map: (3, H, W) — first 3 PCA components mapped to RGB
    """
    mae.eval()
    encoder = mae.encoder
    input_tensor = input_tensor.to(device)
    B, C, T, H, W = input_tensor.shape

    with torch.no_grad():
        feat_list = encoder.forward_features(input_tensor)
        spatial = encoder.prepare_features_for_image_model(feat_list)
        last_feat = spatial[-1]  # (B, 4096, 14, 14)

    feat = last_feat[0].cpu().numpy()  # (4096, 14, 14)

    # PCA: reduce 4096 -> 3 components for RGB visualization
    feat_flat = feat.reshape(feat.shape[0], -1).T  # (196, 4096)
    mean = feat_flat.mean(axis=0, keepdims=True)
    feat_centered = feat_flat - mean

    # SVD-based PCA
    try:
        U, S, Vt = np.linalg.svd(feat_centered, full_matrices=False)
        pca_3 = U[:, :3] * S[:3]  # (196, 3)
    except np.linalg.LinAlgError:
        pca_3 = feat_centered[:, :3]

    # Reshape to spatial
    pca_map = pca_3.T.reshape(3, 14, 14)  # (3, 14, 14)

    # Normalize each component to [0, 1]
    for i in range(3):
        mn, mx = pca_map[i].min(), pca_map[i].max()
        if mx - mn > 1e-8:
            pca_map[i] = (pca_map[i] - mn) / (mx - mn)

    # Upsample to full resolution
    pca_tensor = torch.from_numpy(pca_map).float().unsqueeze(0)
    pca_up = F.interpolate(pca_tensor, size=(H, W), mode="bilinear",
                           align_corners=False)
    pca_map_full = pca_up[0].numpy()  # (3, H, W)

    return feat, pca_map_full


def temporal_anomaly(mae, fire_tifs, num_frames=4, tile_size=224,
                     n_masks=5, device="cpu"):
    """
    Compute per-day anomaly maps to track fire progression.
    Uses sliding windows of num_frames days.

    Returns:
        daily_anomalies: list of (H, W) anomaly maps per output day
        day_labels: list of date strings
    """
    n_days = len(fire_tifs)
    if n_days < num_frames:
        print(f"  Need >= {num_frames} days, got {n_days}")
        return [], []

    daily_anomalies = []
    day_labels = []

    for start in range(n_days - num_frames + 1):
        window = fire_tifs[start:start + num_frames]
        day_name = Path(window[-1]).stem  # date of last day in window

        tensor, _, _ = prepare_input(window, num_frames, tile_size)
        anomaly = mae_reconstruction_anomaly(mae, tensor, n_masks, device)

        daily_anomalies.append(anomaly[:tile_size, :tile_size])
        day_labels.append(day_name)

    return daily_anomalies, day_labels


# ═══════════════════════════════════════════════════════════════
#  VISUALIZATION
# ═══════════════════════════════════════════════════════════════

def normalize_for_display(arr, percentile=2):
    """Normalize array to [0, 1] clipping outliers."""
    lo = np.percentile(arr[arr != 0], percentile) if (arr != 0).any() else arr.min()
    hi = np.percentile(arr[arr != 0], 100 - percentile) if (arr != 0).any() else arr.max()
    if hi - lo < 1e-8:
        return np.zeros_like(arr)
    return np.clip((arr - lo) / (hi - lo), 0, 1)


def visualize_single_event(anomaly_map, labels, raw_rgbs, pca_map,
                           fire_id, output_dir, tif_dates):
    """Create visualization grid for a single fire event."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    n_days = len(labels)
    fig, axes = plt.subplots(3, max(n_days, 2), figsize=(5 * max(n_days, 2), 15))

    # Row 1: Pseudo-RGB per day
    for d in range(n_days):
        ax = axes[0, d] if n_days > 1 else axes[0]
        rgb = raw_rgbs[d][[0, 1, 2]]  # Already I1, I2, M11
        rgb_disp = np.stack([normalize_for_display(rgb[c]) for c in range(3)], axis=-1)
        h, w = min(224, rgb_disp.shape[0]), min(224, rgb_disp.shape[1])
        ax.imshow(rgb_disp[:h, :w])
        ax.set_title(f"Day {d+1}: {tif_dates[d]}", fontsize=10)
        ax.axis("off")

    # Row 2: Ground truth fire masks
    for d in range(n_days):
        ax = axes[1, d] if n_days > 1 else axes[1]
        lbl = labels[d][:224, :224]
        ax.imshow(lbl, cmap="Reds", vmin=0, vmax=1)
        fire_pix = (lbl > 0).sum()
        ax.set_title(f"GT Fire Mask ({fire_pix} px)", fontsize=10)
        ax.axis("off")

    # Row 3 left: Anomaly map
    ax = axes[2, 0]
    im = ax.imshow(anomaly_map[:224, :224], cmap="hot")
    ax.set_title("MAE Reconstruction\nAnomaly (fire=bright)", fontsize=10)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)

    # Row 3 right: PCA features
    ax = axes[2, 1]
    pca_disp = np.transpose(pca_map, (1, 2, 0))[:224, :224]
    ax.imshow(pca_disp)
    ax.set_title("Encoder PCA\n(RGB=PC1,PC2,PC3)", fontsize=10)
    ax.axis("off")

    # Hide extra subplots
    for d in range(2, n_days if n_days > 2 else 0):
        axes[2, d].axis("off")

    fig.suptitle(f"Prithvi-EO-2.0-300M-TL Direct Inference: {fire_id}",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"inference_v2_{fire_id}.jpg")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")
    return out_path


def visualize_temporal_progression(daily_anomalies, day_labels, labels_per_day,
                                   fire_id, output_dir):
    """Visualize fire progression via daily anomaly maps."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(daily_anomalies)
    if n == 0:
        return None

    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
    if n == 1:
        axes = axes.reshape(2, 1)

    vmax = max(a.max() for a in daily_anomalies) * 0.8

    for i in range(n):
        # Top: anomaly
        ax = axes[0, i]
        ax.imshow(daily_anomalies[i], cmap="hot", vmin=0,
                  vmax=max(vmax, 1e-8))
        ax.set_title(day_labels[i], fontsize=9)
        ax.axis("off")

        # Bottom: GT mask (if available)
        if i < len(labels_per_day):
            ax2 = axes[1, i]
            ax2.imshow(labels_per_day[i][:224, :224], cmap="Reds", vmin=0, vmax=1)
            fp = (labels_per_day[i][:224, :224] > 0).sum()
            ax2.set_title(f"GT ({fp} px)", fontsize=9)
            ax2.axis("off")

    axes[0, 0].set_ylabel("MAE Anomaly", fontsize=11)
    axes[1, 0].set_ylabel("Ground Truth", fontsize=11)

    fig.suptitle(f"Temporal Fire Progression: {fire_id}\n"
                 f"(Prithvi-EO-2.0-300M-TL MAE Reconstruction Error)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"temporal_v2_{fire_id}.jpg")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")
    return out_path


# ═══════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════

def run_inference(args):
    print("=" * 70)
    print("  Prithvi-EO-2.0-300M-TL — Direct Inference Pipeline")
    print("=" * 70)
    print(f"  Device:     {args.device}")
    print(f"  Data dir:   {args.data_dir}")
    print(f"  N masks:    {args.n_masks}")
    print(f"  Num frames: {args.num_frames}")
    print()

    # 1. Load model
    print("[1] Loading Prithvi-EO-2.0-300M-TL (full MAE) ...")
    t0 = time.time()
    mae = load_prithvi_v2_mae(args.checkpoint, args.device)
    print(f"  Loaded in {time.time()-t0:.1f}s\n")

    # 2. Discover fire events
    print("[2] Discovering fire events ...")
    events = discover_fire_events(args.data_dir)
    print(f"  Found {len(events)} fire events")
    for fid, info in events.items():
        print(f"    {fid}: {info['n_days']} days")

    if not events:
        print("\n  ERROR: No fire events found!")
        print(f"  Expected structure: {args.data_dir}/YYYY/fire_XXXXX/*.tif")
        return

    # Filter by fire_id if specified
    if args.fire_id:
        if args.fire_id in events:
            events = {args.fire_id: events[args.fire_id]}
        else:
            print(f"\n  Fire '{args.fire_id}' not found. Available: {list(events.keys())}")
            return

    os.makedirs(args.output_dir, exist_ok=True)
    results = []

    for fire_id, info in events.items():
        print(f"\n{'─' * 60}")
        print(f"[3] Processing: {fire_id} ({info['n_days']} days)")
        print(f"{'─' * 60}")

        tifs = info["tifs"]
        num_frames = min(args.num_frames, len(tifs))

        # Use the last num_frames days for the main inference
        selected_tifs = tifs[-num_frames:] if len(tifs) >= num_frames else tifs
        tif_dates = [Path(t).stem for t in selected_tifs]

        print(f"  Input days: {tif_dates}")

        # Prepare input
        print("  Preparing input ...")
        input_tensor, labels, raw_rgbs = prepare_input(
            selected_tifs, num_frames, args.tile_size
        )
        print(f"  Input shape: {input_tensor.shape}")

        # 3a. MAE Reconstruction Anomaly
        print(f"  Running MAE reconstruction ({args.n_masks} masks) ...")
        t1 = time.time()
        anomaly_map = mae_reconstruction_anomaly(
            mae, input_tensor, args.n_masks, args.device
        )
        print(f"  Anomaly map: shape={anomaly_map.shape}, "
              f"range=[{anomaly_map.min():.4f}, {anomaly_map.max():.4f}], "
              f"time={time.time()-t1:.1f}s")

        # 3b. Feature extraction + PCA
        print("  Extracting encoder features ...")
        feat_map, pca_map = feature_extraction(mae, input_tensor, args.device)
        print(f"  Features: {feat_map.shape}, PCA: {pca_map.shape}")

        # 3c. Visualize
        print("  Generating visualizations ...")
        out1 = visualize_single_event(
            anomaly_map, labels, raw_rgbs, pca_map,
            fire_id, args.output_dir, tif_dates
        )

        # 3d. Temporal progression (if enough days)
        out2 = None
        if len(tifs) >= num_frames + 1:
            print(f"  Computing temporal progression ({len(tifs)} days) ...")
            daily_anomalies, day_labels = temporal_anomaly(
                mae, tifs, num_frames, args.tile_size,
                min(args.n_masks, 3),  # fewer masks for speed
                args.device,
            )

            # Load all GT labels for comparison
            all_labels = []
            for t in tifs[num_frames - 1:]:
                d = load_tif(t)
                fire = d[FIRE_LABEL_BAND]
                all_labels.append(
                    np.where(~np.isnan(fire), 1.0, 0.0).astype(np.float32)
                )

            out2 = visualize_temporal_progression(
                daily_anomalies, day_labels, all_labels,
                fire_id, args.output_dir
            )

        # Compute simple metrics (anomaly vs GT)
        # Ensure both are exactly tile_size x tile_size
        ts = args.tile_size
        gt = labels[-1]
        h_gt, w_gt = gt.shape
        if h_gt < ts or w_gt < ts:
            gt = np.pad(gt, ((0, max(0, ts - h_gt)), (0, max(0, ts - w_gt))),
                        mode="constant", constant_values=0)
        gt_crop = gt[:ts, :ts]
        anom_crop = anomaly_map[:ts, :ts]
        gt_flat = gt_crop.flatten()
        anom_flat = anom_crop.flatten()

        # Threshold anomaly at various percentiles
        fire_pixels = (gt_flat > 0).sum()
        total_pixels = gt_flat.size
        print(f"\n  Ground truth: {fire_pixels}/{total_pixels} fire pixels "
              f"({100*fire_pixels/total_pixels:.2f}%)")

        if fire_pixels > 0 and anom_flat.max() > 0:
            # Mean anomaly score in fire vs non-fire regions
            fire_anom = anom_flat[gt_flat > 0].mean()
            bg_anom = anom_flat[gt_flat == 0].mean()
            ratio = fire_anom / (bg_anom + 1e-8)
            print(f"  Anomaly score — fire: {fire_anom:.4f}, "
                  f"background: {bg_anom:.4f}, ratio: {ratio:.2f}x")

            # AUC-like metric
            thresholds = np.percentile(anom_flat[anom_flat > 0],
                                       [50, 75, 90, 95])
            for pct, thr in zip([50, 75, 90, 95], thresholds):
                pred = (anom_flat >= thr).astype(float)
                tp = ((pred > 0) & (gt_flat > 0)).sum()
                fp = ((pred > 0) & (gt_flat == 0)).sum()
                fn = ((pred == 0) & (gt_flat > 0)).sum()
                prec = tp / (tp + fp + 1e-8)
                rec = tp / (tp + fn + 1e-8)
                f1 = 2 * prec * rec / (prec + rec + 1e-8)
                print(f"    @P{pct}: prec={prec:.3f} rec={rec:.3f} f1={f1:.3f}")
        else:
            print("  (No fire pixels or zero anomaly — skipping metrics)")

        results.append({
            "fire_id": fire_id,
            "n_days": info["n_days"],
            "fire_pixels": int(fire_pixels),
            "vis_path": out1,
            "temporal_path": out2,
        })

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  INFERENCE COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Events processed: {len(results)}")
    print(f"  Output dir:       {args.output_dir}")
    for r in results:
        print(f"    {r['fire_id']}: {r['n_days']} days, "
              f"{r['fire_pixels']} fire pixels")
        print(f"      → {r['vis_path']}")
        if r['temporal_path']:
            print(f"      → {r['temporal_path']}")
    print(f"{'=' * 70}")

    return results


# ═══════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Prithvi-EO-2.0-300M-TL direct inference on WildfireSpreadTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to checkpoint (auto-downloads if missing)")
    p.add_argument("--data_dir", type=str,
                   default=os.path.join(_BASE_DIR, "DATA", "WildfireSpreadTS"),
                   help="Path to WildfireSpreadTS root")
    p.add_argument("--output_dir", type=str,
                   default=os.path.join(_BASE_DIR, "OUTPUTS", "prithvi_v2_inference"),
                   help="Output directory")
    p.add_argument("--fire_id", type=str, default=None,
                   help="Specific fire event to process (e.g. fire_21889697)")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--n_masks", type=int, default=10,
                   help="Number of random masks for anomaly detection")
    p.add_argument("--num_frames", type=int, default=4,
                   help="Number of temporal frames per window")
    p.add_argument("--tile_size", type=int, default=224)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
