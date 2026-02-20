"""
Demo: ai4g-flood model (Microsoft, Nature Communications 2025)
================================================================
This script demonstrates how the ai4g-flood SAR flood detection model works
by using two Sen1Floods11 S1Hand tiles as a simulated pre/post pair.

What this model does:
  - Takes Sentinel-1 SAR imagery (VV + VH polarizations) from before & during a flood
  - Computes change-detection features (amplitude drop = new water)
  - Runs a U-Net (MobileNetV2 encoder) to produce binary flood masks
  - Works on 128x128 patches, CPU or GPU

How it differs from your Prithvi model:
  - ai4g-flood uses **SAR (radar)** data  → works through clouds/night
  - Prithvi uses **optical (Sentinel-2)** data → clearer imagery but blocked by clouds
  - ai4g-flood needs a pre+post pair (change detection approach)
  - Prithvi does single-image semantic segmentation
"""
import os, sys, time, pathlib
import numpy as np
import rasterio
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Fix: checkpoint saved on Linux contains PosixPath – patch for Windows
pathlib.PosixPath = pathlib.WindowsPath

# Add ai4g-flood src to path so we can import its utilities
AI4G_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ai4g-flood")
sys.path.insert(0, os.path.join(AI4G_ROOT, "src"))

from utils.image_processing import (
    db_scale, pad_to_nearest, create_patches,
    reconstruct_image_from_patches, apply_buffer,
)
from utils.model import load_model

# ── CONFIG ─────────────────────────────────────────────────
MODEL_PATH = os.path.join(AI4G_ROOT, "models", "ai4g_sar_model.ckpt")
DATA_DIR   = r"D:\Sentinel Final\Sentinel Backend\DATA\S1Hand"
OUTPUT_DIR = r"D:\Sentinel Final\Sentinel Backend\OUTPUTS\ai4g_demo"

# Pick two India tiles to simulate pre-event / post-event
PRE_TILE  = os.path.join(DATA_DIR, "India_103447_S1Hand.tif")   # "before" flood
POST_TILE = os.path.join(DATA_DIR, "India_979278_S1Hand.tif")   # "during" flood

INPUT_SIZE = 128
BUFFER_SIZE = 8   # pixels (~80m at 10m resolution)

# Thresholds for change detection (same defaults as the original script)
VV_THRESHOLD = 100
VH_THRESHOLD = 90
DELTA_AMPLITUDE = 10
VV_MIN_THRESHOLD = 75
VH_MIN_THRESHOLD = 70


def read_tile(path):
    """Read a 2-band S1Hand tile → returns VV, VH in dB-scale (0-255)."""
    with rasterio.open(path) as src:
        vv = db_scale(src.read(1))  # Band 1 = VV
        vh = db_scale(src.read(2))  # Band 2 = VH
        return vv, vh, src.crs, src.transform


def calculate_flood_change(vv_pre, vh_pre, vv_post, vh_post):
    """Compute 2-channel change map: where amplitude dropped = potential flood."""
    vv_change = (
        (vv_post < VV_THRESHOLD)
        & (vv_pre > VV_THRESHOLD)
        & ((vv_pre - vv_post) > DELTA_AMPLITUDE)
    ).astype(int)
    vh_change = (
        (vh_post < VH_THRESHOLD)
        & (vh_pre > VH_THRESHOLD)
        & ((vh_pre - vh_post) > DELTA_AMPLITUDE)
    ).astype(int)
    zero_idx = (
        (vv_post < VV_MIN_THRESHOLD) | (vv_pre < VV_MIN_THRESHOLD)
        | (vh_post < VH_MIN_THRESHOLD) | (vh_pre < VH_MIN_THRESHOLD)
    )
    vv_change[zero_idx] = 0
    vh_change[zero_idx] = 0
    return np.stack((vv_change, vh_change), axis=2)


def main():
    print("=" * 60)
    print("  ai4g-flood  Demo  –  SAR Change-Detection Flood Model")
    print("=" * 60)

    # ── 1. LOAD DATA ───────────────────────────────────────
    print(f"\n[1/4] Loading SAR tiles ...")
    print(f"  Pre-event  : {os.path.basename(PRE_TILE)}")
    print(f"  Post-event : {os.path.basename(POST_TILE)}")

    vv_pre, vh_pre, crs_pre, tf_pre = read_tile(PRE_TILE)
    vv_post, vh_post, crs_post, tf_post = read_tile(POST_TILE)

    print(f"  Tile shape : {vv_pre.shape}  (VV range {vv_pre.min():.0f}–{vv_pre.max():.0f})")
    print(f"  CRS        : {crs_post}")

    # ── 2. CHANGE DETECTION ────────────────────────────────
    print(f"\n[2/4] Computing SAR amplitude change map ...")
    flood_change = calculate_flood_change(vv_pre, vh_pre, vv_post, vh_post)
    n_change = (flood_change.sum(axis=2) > 0).sum()
    print(f"  Change pixels (potential flood): {n_change} / {vv_pre.size}  "
          f"({100*n_change/vv_pre.size:.1f}%)")

    # ── 3. MODEL INFERENCE ─────────────────────────────────
    print(f"\n[3/4] Running U-Net inference (MobileNetV2 encoder) ...")
    device = torch.device("cpu")
    model = load_model(MODEL_PATH, device, in_channels=2, n_classes=2)
    model.eval()

    target_shape = vv_post.shape
    flood_padded = pad_to_nearest(flood_change, INPUT_SIZE, [0, 1])
    patches = create_patches(flood_padded, (INPUT_SIZE, INPUT_SIZE), INPUT_SIZE)
    print(f"  Patches: {len(patches)}  ({INPUT_SIZE}x{INPUT_SIZE} each)")

    t0 = time.time()
    predictions = []
    with torch.no_grad():
        for i in range(0, len(patches), 1024):
            batch = patches[i:i+1024]
            tensor = torch.from_numpy(np.array(batch)).float().to(device)
            output = model(tensor)
            _, pred = torch.max(output, 1)
            pred = (pred * 255).to(torch.int)
            # Restrict to pixels that had change signal
            pred[(tensor[:, 0] == 0) & (tensor[:, 1] == 0)] = 0
            predictions.extend(pred.cpu().numpy())

    pred_image, _ = reconstruct_image_from_patches(
        predictions, flood_padded.shape[:2], (INPUT_SIZE, INPUT_SIZE), INPUT_SIZE
    )
    pred_image = pred_image[:target_shape[0], :target_shape[1]]

    if BUFFER_SIZE > 0:
        pred_image = apply_buffer(pred_image, BUFFER_SIZE)

    elapsed = time.time() - t0
    n_flood = (pred_image == 255).sum()
    print(f"  Inference time : {elapsed:.1f}s")
    print(f"  Flood pixels   : {n_flood} / {pred_image.size}  "
          f"({100*n_flood/pred_image.size:.2f}%)")

    # ── 4. SAVE & VISUALIZE ────────────────────────────────
    print(f"\n[4/4] Saving results to {OUTPUT_DIR} ...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save GeoTIFF
    tif_path = os.path.join(OUTPUT_DIR, "demo_flood_prediction.tif")
    save_pred = pred_image.copy().astype(np.float32)
    save_pred[save_pred == 0] = np.nan
    with rasterio.open(
        tif_path, "w", driver="GTiff",
        height=save_pred.shape[0], width=save_pred.shape[1],
        count=1, dtype="float32", crs=crs_post, transform=tf_post,
        compress="lzw", nodata=np.nan,
    ) as dst:
        dst.write(save_pred, 1)
    print(f"  GeoTIFF → {tif_path}")

    # Save visual comparison
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(vv_pre, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Pre-event VV")
    axes[1].imshow(vv_post, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("Post-event VV")
    axes[2].imshow(flood_change[:,:,0], cmap="Reds")
    axes[2].set_title("Change Detection\n(VV amplitude drop)")
    flood_vis = np.zeros((*pred_image.shape, 3), dtype=np.uint8)
    flood_vis[pred_image == 255] = [0, 120, 255]  # Blue = flood
    axes[3].imshow(vv_post, cmap="gray", vmin=0, vmax=255, alpha=0.7)
    axes[3].imshow(flood_vis, alpha=0.5)
    axes[3].set_title("Flood Prediction\n(blue overlay)")
    for ax in axes:
        ax.axis("off")
    plt.suptitle(
        "ai4g-flood Demo – SAR Change-Detection Flood Model\n"
        f"Pre: {os.path.basename(PRE_TILE)}  |  Post: {os.path.basename(POST_TILE)}",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    png_path = os.path.join(OUTPUT_DIR, "demo_comparison.png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Visual  → {png_path}")

    # ── SUMMARY ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  DEMO COMPLETE")
    print("=" * 60)
    print(f"""
What happened:
  1. Loaded two Sentinel-1 SAR tiles (VV+VH bands)
     - Pre-event  = baseline (no flood)
     - Post-event = during flood event
  2. Computed change detection (where radar backscatter dropped
     → water absorbs radar → low return signal)
  3. Fed the 2-channel change map into a U-Net (MobileNetV2)
     that classifies each 128x128 patch as flood/no-flood
  4. Applied an {BUFFER_SIZE}-pixel buffer around detections

Key differences from your Prithvi pipeline:
  ┌─────────────────┬──────────────────────┬─────────────────────┐
  │                  │  ai4g-flood (this)    │  Prithvi (yours)    │
  ├─────────────────┼──────────────────────┼─────────────────────┤
  │ Sensor           │  Sentinel-1 SAR       │  Sentinel-2 Optical │
  │ Cloud-proof?     │  YES (radar)          │  NO (blocked)       │
  │ Night imaging?   │  YES                  │  NO                 │
  │ Input            │  Pre+Post pair needed │  Single image       │
  │ Architecture     │  U-Net (MobileNetV2)  │  ViT (Prithvi-100M)│
  │ Resolution       │  10m (Sentinel-1)     │  10m (Sentinel-2)   │
  │ Training data    │  10 years global SAR  │  Sen1Floods11       │
  │ Published        │  Nature Comm. 2025    │  IBM/NASA 2023      │
  └─────────────────┴──────────────────────┴─────────────────────┘

When to use ai4g-flood over Prithvi:
  ✓ Monsoon/tropical regions with heavy cloud cover
  ✓ Rapid response (SAR available day & night)
  ✓ You have pre-event baseline imagery

When to stick with Prithvi:
  ✓ Clear-sky conditions (optical gives richer features)
  ✓ No pre-event image available
  ✓ Already integrated in your pipeline
""")


if __name__ == "__main__":
    main()
