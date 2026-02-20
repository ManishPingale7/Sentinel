"""
EMSR771 — Before vs During Flood Visualization (ai4g-flood + Prithvi)
======================================================================
Generates Prithvi-style 2×2 overlay images for every EMSR771 tile:

  Top-Left:     Before Flood (original image)
  Top-Right:    During Flood (original image)
  Bottom-Left:  Before + flood overlay (blue = detected water)
  Bottom-Right: During + overlay (blue = pre-existing water, red = NEW flood)

Also generates a comparison panel showing both models side by side.

Uses:
  - S2 optical RGB for backgrounds (Sentinel-2, bands 3/2/1)
  - S1 SAR grayscale (VV band) as alternate background
  - ai4g-flood predictions from run_emsr771_flood.py
  - Prithvi predictions from previous pipeline
  - Ground truth flood masks from SenForFlood
"""

import os
import sys
import glob
import pathlib
import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.ndimage import binary_opening

# ═══════════════════════════════════════════════════════════
#  PATHS
# ═══════════════════════════════════════════════════════════
DATA_ROOT    = r"D:\Sentinel Final\Sentinel Backend\DATA\CEMS\EMSR771"
AI4G_PRED    = r"D:\Sentinel Final\Sentinel Backend\OUTPUTS\EMSR771_ai4g\geotiffs"
PRITHVI_PRED = r"D:\Sentinel Final\Sentinel Backend\OUTPUTS\Prithvi\EMSR771"
GT_DIR       = os.path.join(DATA_ROOT, "flood_mask")
OUTPUT_DIR   = r"D:\Sentinel Final\Sentinel Backend\OUTPUTS\EMSR771_ai4g\overlays"

S1_BEFORE = os.path.join(DATA_ROOT, "s1_before_flood")
S1_DURING = os.path.join(DATA_ROOT, "s1_during_flood")
S2_BEFORE = os.path.join(DATA_ROOT, "s2_before_flood")
S2_DURING = os.path.join(DATA_ROOT, "s2_during_flood")

PIXEL_RES_M = 10.0  # 10m per pixel


# ═══════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════

def normalize_band(band):
    """Percentile-clip + min-max normalize to [0,1]."""
    valid = band[band > 0]
    if valid.size == 0:
        return np.zeros_like(band, dtype=np.float32)
    p2, p98 = np.percentile(valid, [2, 98])
    band = np.clip(band, p2, p98).astype(np.float32)
    lo, hi = band.min(), band.max()
    if hi - lo < 1e-6:
        return np.zeros_like(band, dtype=np.float32)
    return (band - lo) / (hi - lo)


def load_s2_rgb(path):
    """Load 8-band SenForFlood S2 .tif → RGB [H,W,3]."""
    with rasterio.open(path) as src:
        r = src.read(3).astype(np.float32)  # B4 Red
        g = src.read(2).astype(np.float32)  # B3 Green
        b = src.read(1).astype(np.float32)  # B2 Blue
    return np.stack([normalize_band(r), normalize_band(g), normalize_band(b)], axis=-1)


def load_s1_gray(path):
    """Load S1 tile → VV grayscale image [H,W,3] for display."""
    with rasterio.open(path) as src:
        vv = src.read(1).astype(np.float32)
    vv_scaled = vv * 2.0 + 135.0
    vv_scaled = np.clip(np.nan_to_num(vv_scaled, nan=0.0), 0, 255) / 255.0
    return np.stack([vv_scaled, vv_scaled, vv_scaled], axis=-1)


def load_ai4g_pred(tile_id):
    """Load ai4g-flood prediction → binary mask (True = flood)."""
    path = os.path.join(AI4G_PRED, f"tile_{tile_id}_flood_pred.tif")
    if not os.path.exists(path):
        return None
    with rasterio.open(path) as src:
        data = src.read(1)
    # ai4g saves flood as 255 and nodata as NaN
    return np.nan_to_num(data, nan=0.0) > 0


def load_prithvi_pred(tile_id, period):
    """Load Prithvi prediction → binary mask (True = flood)."""
    # Find matching file
    pattern = os.path.join(PRITHVI_PRED, period, f"{tile_id}_*_pred.tif")
    files = glob.glob(pattern)
    if not files:
        return None
    with rasterio.open(files[0]) as src:
        data = src.read(1)
    return data == 1


def load_gt_mask(tile_id):
    """Load ground truth flood mask. Returns (flood_binary, raw_mask)."""
    path = os.path.join(GT_DIR, f"{tile_id}_flood_mask.tif")
    if not os.path.exists(path):
        return None, None
    with rasterio.open(path) as src:
        raw = src.read(1)
    flood = (raw == 1)  # class 1 = flood, class 2 = permanent water
    return flood, raw


def pixels_to_km2(n_pixels):
    """Convert pixel count to km²."""
    return n_pixels * (PIXEL_RES_M ** 2) / 1e6


def clean_mask(mask, size=3):
    """Morphological opening to remove speckle."""
    struct = np.ones((size, size))
    return binary_opening(mask, structure=struct)


# ═══════════════════════════════════════════════════════════
#  VISUALIZATION 1: Before vs During (Prithvi-style 2×2)
# ═══════════════════════════════════════════════════════════

def make_before_during_overlay(tile_id, out_dir):
    """
    Generate the 2×2 Before vs During panel:
      TL: Before (S2 RGB)
      TR: During (S2 RGB)
      BL: Before + ai4g flood overlay (blue)
      BR: During + overlay (blue=permanent, red=NEW flood expansion)
    """
    # Load images
    s2_before_path = os.path.join(S2_BEFORE, f"{tile_id}_s2_before_flood.tif")
    s2_during_path = os.path.join(S2_DURING, f"{tile_id}_s2_during_flood.tif")

    if os.path.exists(s2_before_path) and os.path.exists(s2_during_path):
        rgb_before = load_s2_rgb(s2_before_path)
        rgb_during = load_s2_rgb(s2_during_path)
        img_type = "Sentinel-2 RGB"
    else:
        # Fall back to S1 grayscale
        rgb_before = load_s1_gray(os.path.join(S1_BEFORE, f"{tile_id}_s1_before_flood.tif"))
        rgb_during = load_s1_gray(os.path.join(S1_DURING, f"{tile_id}_s1_during_flood.tif"))
        img_type = "Sentinel-1 VV"

    # Load ai4g predictions (before = Prithvi before, during = ai4g)
    # ai4g-flood only produces a "during" prediction (change detection)
    ai4g_pred = load_ai4g_pred(tile_id)
    prithvi_before = load_prithvi_pred(tile_id, "before")
    prithvi_during = load_prithvi_pred(tile_id, "during")
    gt_flood, gt_raw = load_gt_mask(tile_id)

    if ai4g_pred is None:
        print(f"    Skipping {tile_id}: no ai4g prediction found")
        return None

    # Use Prithvi before as "pre-existing water" baseline, ai4g as "during" detection
    if prithvi_before is not None:
        before_water = clean_mask(prithvi_before)
    else:
        before_water = np.zeros_like(ai4g_pred, dtype=bool)

    during_flood = clean_mask(ai4g_pred)
    new_flood = during_flood & ~before_water
    permanent = during_flood & before_water

    # Stats
    before_km2 = pixels_to_km2(before_water.sum())
    during_km2 = pixels_to_km2(during_flood.sum())
    new_km2 = pixels_to_km2(new_flood.sum())
    if before_water.sum() > 0:
        pct_increase = 100.0 * new_flood.sum() / before_water.sum()
        pct_str = f"+{pct_increase:.1f}%"
    else:
        pct_str = "New detection"

    # ── Figure ──
    fig, axes = plt.subplots(2, 2, figsize=(18, 16))

    # TL: Original Before
    axes[0, 0].imshow(rgb_before)
    axes[0, 0].set_title(f"BEFORE Flood ({img_type})\nEMSR771 / tile {tile_id}", fontsize=12, fontweight="bold")
    axes[0, 0].axis("off")

    # TR: Original During
    axes[0, 1].imshow(rgb_during)
    axes[0, 1].set_title(f"DURING Flood ({img_type})\nEMSR771 / tile {tile_id}", fontsize=12, fontweight="bold")
    axes[0, 1].axis("off")

    # BL: Before + water overlay (blue)
    axes[1, 0].imshow(rgb_before)
    overlay_bl = np.zeros((*before_water.shape, 4), dtype=np.float32)
    overlay_bl[before_water] = [0.0, 0.4, 1.0, 0.55]
    axes[1, 0].imshow(overlay_bl)
    axes[1, 0].set_title(
        f"BEFORE + Water Overlay (Prithvi)\nWater area: {before_km2:.4f} km²",
        fontsize=12, fontweight="bold",
    )
    axes[1, 0].axis("off")
    handles_bl = [Patch(facecolor=[0.0, 0.4, 1.0, 0.55], edgecolor="k", label="Detected Water")]
    axes[1, 0].legend(handles=handles_bl, loc="lower left", fontsize=10)

    # BR: During + change overlay (blue=permanent, red=new flood)
    axes[1, 1].imshow(rgb_during)

    overlay_br = np.zeros((*during_flood.shape, 4), dtype=np.float32)
    overlay_br[permanent] = [0.0, 0.4, 1.0, 0.50]   # Blue = pre-existing
    overlay_br[new_flood] = [1.0, 0.0, 0.0, 0.65]    # Red = NEW flood

    axes[1, 1].imshow(overlay_br)
    axes[1, 1].set_title(
        f"DURING + Flood Change (ai4g-flood)\n"
        f"Total: {during_km2:.4f} km²  |  New: +{new_km2:.4f} km²  |  {pct_str}",
        fontsize=12, fontweight="bold",
    )
    axes[1, 1].axis("off")
    handles_br = [
        Patch(facecolor=[0.0, 0.4, 1.0, 0.50], edgecolor="k", label="Pre-existing Water"),
        Patch(facecolor=[1.0, 0.0, 0.0, 0.65], edgecolor="k", label="NEW Flood Expansion"),
    ]
    axes[1, 1].legend(handles=handles_br, loc="lower left", fontsize=10)

    fig.suptitle(
        f"Flood Change Detection — EMSR771 tile {tile_id}\n"
        f"Blue = existing water  |  Red = new flood expansion",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    path = os.path.join(out_dir, f"{tile_id}_before_vs_during.jpg")
    fig.savefig(path, dpi=120, bbox_inches="tight", format="jpg")
    plt.close(fig)
    return path


# ═══════════════════════════════════════════════════════════
#  VISUALIZATION 2: Model Comparison (ai4g vs Prithvi vs GT)
# ═══════════════════════════════════════════════════════════

def make_model_comparison(tile_id, out_dir):
    """
    6-panel comparison:
      Row 1: S2 During RGB, Ground Truth, ai4g Prediction
      Row 2: Prithvi Prediction, ai4g vs GT overlay, Prithvi vs GT overlay
    """
    s2_during_path = os.path.join(S2_DURING, f"{tile_id}_s2_during_flood.tif")
    if os.path.exists(s2_during_path):
        rgb = load_s2_rgb(s2_during_path)
    else:
        rgb = load_s1_gray(os.path.join(S1_DURING, f"{tile_id}_s1_during_flood.tif"))

    ai4g_pred = load_ai4g_pred(tile_id)
    prithvi_pred = load_prithvi_pred(tile_id, "during")
    gt_flood, gt_raw = load_gt_mask(tile_id)

    if ai4g_pred is None:
        return None

    fig, axes = plt.subplots(2, 3, figsize=(21, 14))

    # ── Row 1 ──

    # (1,1) During RGB
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("During Flood (S2 RGB)", fontsize=12, fontweight="bold")
    axes[0, 0].axis("off")

    # (1,2) Ground Truth
    if gt_raw is not None:
        gt_vis = np.zeros((*gt_raw.shape, 4), dtype=np.float32)
        gt_vis[gt_raw == 1] = [1.0, 0.2, 0.2, 0.7]   # Red = flood
        gt_vis[gt_raw == 2] = [0.5, 0.5, 0.5, 0.5]    # Gray = permanent water
        axes[0, 1].imshow(rgb)
        axes[0, 1].imshow(gt_vis)
        n_gt = (gt_raw == 1).sum()
        axes[0, 1].set_title(f"Ground Truth\nFlood: {pixels_to_km2(n_gt):.4f} km²", fontsize=12, fontweight="bold")
        handles_gt = [
            Patch(facecolor=[1.0, 0.2, 0.2, 0.7], edgecolor="k", label="Flood"),
            Patch(facecolor=[0.5, 0.5, 0.5, 0.5], edgecolor="k", label="Permanent Water"),
        ]
        axes[0, 1].legend(handles=handles_gt, loc="lower left", fontsize=9)
    else:
        axes[0, 1].text(0.5, 0.5, "No GT Available", ha="center", va="center", fontsize=14, transform=axes[0, 1].transAxes)
    axes[0, 1].axis("off")

    # (1,3) ai4g-flood prediction
    ai4g_vis = np.zeros((*ai4g_pred.shape, 4), dtype=np.float32)
    ai4g_vis[ai4g_pred] = [0.0, 0.47, 1.0, 0.7]
    axes[0, 2].imshow(rgb)
    axes[0, 2].imshow(ai4g_vis)
    n_ai4g = ai4g_pred.sum()
    axes[0, 2].set_title(f"ai4g-flood (SAR)\nFlood: {pixels_to_km2(n_ai4g):.4f} km²", fontsize=12, fontweight="bold")
    axes[0, 2].axis("off")

    # ── Row 2 ──

    # (2,1) Prithvi prediction
    if prithvi_pred is not None:
        pri_vis = np.zeros((*prithvi_pred.shape, 4), dtype=np.float32)
        pri_vis[prithvi_pred] = [0.93, 0.46, 0.0, 0.7]  # Orange
        axes[1, 0].imshow(rgb)
        axes[1, 0].imshow(pri_vis)
        n_pri = prithvi_pred.sum()
        axes[1, 0].set_title(f"Prithvi (Optical)\nFlood: {pixels_to_km2(n_pri):.4f} km²", fontsize=12, fontweight="bold")
    else:
        axes[1, 0].imshow(rgb)
        axes[1, 0].text(0.5, 0.5, "No Prithvi Pred", ha="center", va="center", fontsize=14,
                        transform=axes[1, 0].transAxes, color="white",
                        bbox=dict(facecolor="black", alpha=0.7))
    axes[1, 0].axis("off")

    # (2,2) ai4g vs GT overlay (TP/FP/FN)
    if gt_flood is not None:
        tp_a = gt_flood & ai4g_pred
        fp_a = ~gt_flood & ai4g_pred
        fn_a = gt_flood & ~ai4g_pred
        ov_a = np.zeros((*ai4g_pred.shape, 3), dtype=np.uint8)
        ov_a[tp_a] = [0, 200, 0]     # Green
        ov_a[fp_a] = [255, 0, 0]     # Red
        ov_a[fn_a] = [255, 165, 0]   # Orange
        axes[1, 1].imshow(rgb, alpha=0.35)
        axes[1, 1].imshow(ov_a, alpha=0.75)

        # Compute scores
        p_a = ai4g_pred.flatten().astype(int)
        g_a = gt_flood.flatten().astype(int)
        from sklearn.metrics import f1_score, jaccard_score
        f1_a = f1_score(g_a, p_a, zero_division=0)
        iou_a = jaccard_score(g_a, p_a, zero_division=0)
        axes[1, 1].set_title(f"ai4g vs GT\nF1={f1_a:.3f}  IoU={iou_a:.3f}", fontsize=12, fontweight="bold")
        handles_ov = [
            Patch(facecolor=[0, 0.78, 0], label="True Positive"),
            Patch(facecolor=[1, 0, 0], label="False Positive"),
            Patch(facecolor=[1, 0.65, 0], label="False Negative"),
        ]
        axes[1, 1].legend(handles=handles_ov, loc="lower left", fontsize=9)
    else:
        axes[1, 1].text(0.5, 0.5, "No GT", ha="center", va="center", fontsize=14, transform=axes[1, 1].transAxes)
    axes[1, 1].axis("off")

    # (2,3) Prithvi vs GT overlay
    if prithvi_pred is not None and gt_flood is not None:
        tp_p = gt_flood & prithvi_pred
        fp_p = ~gt_flood & prithvi_pred
        fn_p = gt_flood & ~prithvi_pred
        ov_p = np.zeros((*prithvi_pred.shape, 3), dtype=np.uint8)
        ov_p[tp_p] = [0, 200, 0]
        ov_p[fp_p] = [255, 0, 0]
        ov_p[fn_p] = [255, 165, 0]
        axes[1, 2].imshow(rgb, alpha=0.35)
        axes[1, 2].imshow(ov_p, alpha=0.75)

        p_p = prithvi_pred.flatten().astype(int)
        g_p = gt_flood.flatten().astype(int)
        f1_p = f1_score(g_p, p_p, zero_division=0)
        iou_p = jaccard_score(g_p, p_p, zero_division=0)
        axes[1, 2].set_title(f"Prithvi vs GT\nF1={f1_p:.3f}  IoU={iou_p:.3f}", fontsize=12, fontweight="bold")
        axes[1, 2].legend(handles=handles_ov, loc="lower left", fontsize=9)
    else:
        axes[1, 2].text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=14, transform=axes[1, 2].transAxes)
    axes[1, 2].axis("off")

    fig.suptitle(
        f"EMSR771 tile {tile_id} — Model Comparison: ai4g-flood (SAR) vs Prithvi (Optical) vs Ground Truth",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    path = os.path.join(out_dir, f"{tile_id}_model_comparison.jpg")
    fig.savefig(path, dpi=120, bbox_inches="tight", format="jpg")
    plt.close(fig)
    return path


# ═══════════════════════════════════════════════════════════
#  VISUALIZATION 3: Grand Summary (aggregate comparison)
# ═══════════════════════════════════════════════════════════

def make_summary(tile_ids, out_dir):
    """Create a summary bar chart comparing both models across all tiles."""
    from sklearn.metrics import f1_score, jaccard_score

    ai4g_f1s = []
    prithvi_f1s = []
    labels = []

    for tid in tile_ids:
        gt_flood, gt_raw = load_gt_mask(tid)
        if gt_flood is None:
            continue

        ai4g_pred = load_ai4g_pred(tid)
        prithvi_pred = load_prithvi_pred(tid, "during")

        g = gt_flood.flatten().astype(int)

        ai4g_f1 = 0
        if ai4g_pred is not None:
            ai4g_f1 = f1_score(g, ai4g_pred.flatten().astype(int), zero_division=0)

        pri_f1 = 0
        if prithvi_pred is not None:
            pri_f1 = f1_score(g, prithvi_pred.flatten().astype(int), zero_division=0)

        ai4g_f1s.append(ai4g_f1)
        prithvi_f1s.append(pri_f1)
        labels.append(tid)

    if not labels:
        return None

    x = np.arange(len(labels))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Bar chart
    bars1 = axes[0].bar(x - width/2, ai4g_f1s, width, label="ai4g-flood (SAR)", color="#3498db", edgecolor="k")
    bars2 = axes[0].bar(x + width/2, prithvi_f1s, width, label="Prithvi (Optical)", color="#e67e22", edgecolor="k")
    axes[0].set_xlabel("Tile ID", fontsize=12)
    axes[0].set_ylabel("F1 Score", fontsize=12)
    axes[0].set_title("F1 Score Comparison: ai4g-flood vs Prithvi", fontsize=13, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=45, fontsize=9)
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(fontsize=11)
    axes[0].axhline(np.mean(ai4g_f1s), color="#3498db", linestyle="--", alpha=0.7, label=f"ai4g avg={np.mean(ai4g_f1s):.3f}")
    axes[0].axhline(np.mean(prithvi_f1s), color="#e67e22", linestyle="--", alpha=0.7, label=f"Prithvi avg={np.mean(prithvi_f1s):.3f}")
    axes[0].legend(fontsize=10)
    axes[0].grid(axis="y", alpha=0.3)

    # Summary comparison table
    axes[1].axis("off")
    avg_ai4g = np.mean(ai4g_f1s)
    avg_prithvi = np.mean(prithvi_f1s)
    winner = "ai4g-flood (SAR)" if avg_ai4g > avg_prithvi else "Prithvi (Optical)"
    tiles_ai4g_better = sum(1 for a, p in zip(ai4g_f1s, prithvi_f1s) if a > p)
    tiles_prithvi_better = sum(1 for a, p in zip(ai4g_f1s, prithvi_f1s) if p > a)

    summary = (
        f"MODEL COMPARISON SUMMARY — EMSR771\n"
        f"{'═' * 50}\n\n"
        f"  {'Metric':<25} {'ai4g-flood':>12} {'Prithvi':>12}\n"
        f"  {'─' * 50}\n"
        f"  {'Avg F1 Score':<25} {avg_ai4g:>12.4f} {avg_prithvi:>12.4f}\n"
        f"  {'Best F1':<25} {max(ai4g_f1s):>12.4f} {max(prithvi_f1s):>12.4f}\n"
        f"  {'Worst F1':<25} {min(ai4g_f1s):>12.4f} {min(prithvi_f1s):>12.4f}\n"
        f"  {'Tiles won':<25} {tiles_ai4g_better:>12} {tiles_prithvi_better:>12}\n"
        f"  {'─' * 50}\n\n"
        f"  Sensor:        SAR (Sentinel-1)     Optical (Sentinel-2)\n"
        f"  Cloud-proof:   YES                  NO\n"
        f"  Night imaging: YES                  NO\n"
        f"  Approach:      Change Detection     Single-Image Seg.\n"
        f"  Architecture:  U-Net/MobileNetV2    ViT/Prithvi-100M\n"
        f"  Input:         Pre+Post pair        Single image\n\n"
        f"  OVERALL WINNER: {winner}\n\n"
        f"  RECOMMENDATION:\n"
        f"    • Use ai4g-flood when clouds block optical imagery\n"
        f"    • Use ai4g-flood for rapid response (SAR day+night)\n"
        f"    • Use Prithvi when clear-sky imagery available\n"
        f"    • Best strategy: ENSEMBLE both for robustness"
    )
    axes[1].text(0.02, 0.95, summary, transform=axes[1].transAxes,
                 fontsize=11, verticalalignment="top", fontfamily="monospace",
                 bbox=dict(facecolor="lightyellow", alpha=0.9, edgecolor="gray"))

    plt.suptitle("EMSR771 — ai4g-flood vs Prithvi: Which Model to Use?",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "model_comparison_summary.jpg")
    fig.savefig(path, dpi=120, bbox_inches="tight", format="jpg")
    plt.close(fig)
    return path


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  EMSR771 — Before vs During Flood Overlays + Model Comparison")
    print("=" * 70)

    overlay_dir = os.path.join(OUTPUT_DIR, "before_vs_during")
    compare_dir = os.path.join(OUTPUT_DIR, "model_comparison")
    os.makedirs(overlay_dir, exist_ok=True)
    os.makedirs(compare_dir, exist_ok=True)

    # Discover tile IDs
    before_files = sorted(glob.glob(os.path.join(S1_BEFORE, "*_s1_before_flood.tif")))
    tile_ids = [os.path.basename(f).split("_")[0] for f in before_files]
    n = len(tile_ids)
    print(f"\n  Tiles found: {n}")
    print(f"  Output:      {OUTPUT_DIR}\n")

    # ── 1. Before vs During overlays ──
    print("[1/3] Generating Before vs During overlays (Prithvi-style) ...\n")
    for i, tid in enumerate(tile_ids):
        print(f"  [{i+1}/{n}] Tile {tid} ...", end=" ", flush=True)
        path = make_before_during_overlay(tid, overlay_dir)
        if path:
            print(f"OK → {os.path.basename(path)}")
        else:
            print("SKIP")

    # ── 2. Model comparison panels ──
    print(f"\n[2/3] Generating ai4g vs Prithvi comparison panels ...\n")
    for i, tid in enumerate(tile_ids):
        print(f"  [{i+1}/{n}] Tile {tid} ...", end=" ", flush=True)
        path = make_model_comparison(tid, compare_dir)
        if path:
            print(f"OK → {os.path.basename(path)}")
        else:
            print("SKIP")

    # ── 3. Summary ──
    print(f"\n[3/3] Generating grand summary comparison ...")
    summary_path = make_summary(tile_ids, OUTPUT_DIR)
    if summary_path:
        print(f"  Summary → {summary_path}")

    print("\n" + "=" * 70)
    print("  COMPLETE")
    print("=" * 70)
    print(f"""
  Output: {OUTPUT_DIR}
    ├── before_vs_during/    → 2×2 Before/During overlays (Prithvi-style)
    ├── model_comparison/    → 6-panel ai4g vs Prithvi vs GT
    └── model_comparison_summary.jpg
""")


if __name__ == "__main__":
    main()
