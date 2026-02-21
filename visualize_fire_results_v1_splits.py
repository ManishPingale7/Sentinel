"""
Burn-Scar Visualization — Prithvi HLS Fire Predictions
=======================================================
Reads Prithvi burn-scar predictions + original HLS imagery and generates
per-tile overlay images and summary dashboards.

Per-tile outputs (2×2 Prithvi-style):

    Top-Left:     HLS True-Color RGB
    Top-Right:    HLS False-Color (SWIR2, NIR, Red) — highlights burn scars
    Bottom-Left:  RGB + Prediction overlay (orange = predicted burn scar)
    Bottom-Right: RGB + Ground Truth overlay + TP / FP / FN accuracy map

Also generates:
    - NBR (Normalized Burn Ratio) heatmap per tile
    - Per-split F1 bar chart summary dashboard

Usage:
    python visualize_fire_results.py
    python visualize_fire_results.py --splits validation
    python visualize_fire_results.py --splits validation --max_tiles 5
    python visualize_fire_results.py --pred_root "path/to/OUTPUTS/prithvi_burn_scars"
"""

import argparse
import csv
import glob
import math
import os
import sys
import time

import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import binary_opening, binary_closing

PIXEL_RES_M = 30.0  # HLS ~ 30 m/pixel

# HLS band indices (0-based) in the 6-band merged tile:
#   0=Blue(B02), 1=Green(B03), 2=Red(B04), 3=NarrowNIR(B8A), 4=SWIR1(B11), 5=SWIR2(B12)
B_BLUE, B_GREEN, B_RED, B_NIR, B_SWIR1, B_SWIR2 = 0, 1, 2, 3, 4, 5


# ═══════════════════════════════════════════════════════════
#  DEFAULTS
# ═══════════════════════════════════════════════════════════
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULTS = dict(
    data_root   = os.path.join(_BASE_DIR, "DATA", "Fire"),
    pred_root   = os.path.join(_BASE_DIR, "OUTPUTS", "prithvi_burn_scars"),
    output_root = os.path.join(_BASE_DIR, "OUTPUTS", "prithvi_burn_scars"),
    img_suffix  = "_merged.tif",
    mask_suffix = ".mask.tif",
)


# ═══════════════════════════════════════════════════════════
#  IMAGE LOADERS
# ═══════════════════════════════════════════════════════════

def _normalize(band):
    """Percentile-clip + min-max normalize → [0,1]."""
    valid = band[~np.isnan(band) & (band > 0)]
    if valid.size == 0:
        return np.zeros_like(band, dtype=np.float32)
    p2, p98 = np.percentile(valid, [2, 98])
    band = np.clip(band, p2, p98).astype(np.float32)
    lo, hi = band.min(), band.max()
    return np.zeros_like(band, dtype=np.float32) if (hi - lo) < 1e-6 else (band - lo) / (hi - lo)


def load_hls_rgb(path):
    """Load 6-band HLS tile → True-Color RGB [H,W,3] float [0,1]."""
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)
    r = _normalize(data[B_RED])
    g = _normalize(data[B_GREEN])
    b = _normalize(data[B_BLUE])
    return np.stack([r, g, b], axis=-1)


def load_hls_false_color(path):
    """Load 6-band HLS → False-Color composite [H,W,3] (SWIR2, NIR, Red).
    Burn scars appear bright magenta/red in this combo."""
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)
    r = _normalize(data[B_SWIR2])
    g = _normalize(data[B_NIR])
    b = _normalize(data[B_RED])
    return np.stack([r, g, b], axis=-1)


def compute_nbr(path):
    """Compute Normalized Burn Ratio: NBR = (NIR - SWIR2) / (NIR + SWIR2).
    Low/negative values → burn scars.  Returns float32 array in [-1, 1]."""
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)
    nir   = data[B_NIR]
    swir2 = data[B_SWIR2]
    denom = nir + swir2
    denom[denom == 0] = 1e-10
    nbr = (nir - swir2) / denom
    return np.clip(nbr, -1, 1)


def load_pred_mask(path):
    """Load prediction GeoTIFF → boolean mask (burn = True)."""
    with rasterio.open(path) as src:
        data = src.read(1)
    return data > 0


def load_gt_mask(path):
    """Load ground truth mask → (burn_bool, raw_mask).
    Expects: 0 = unburnt, 1 = burn scar."""
    with rasterio.open(path) as src:
        raw = src.read(1)
    return (raw == 1), raw


def clean(mask, k=3):
    return binary_opening(mask, structure=np.ones((k, k)))


def px_to_km2(n):
    return n * (PIXEL_RES_M ** 2) / 1e6


# ═══════════════════════════════════════════════════════════
#  VISUALIZATION 1:  Per-Tile 2×2  (RGB, False-Color, Pred, Accuracy)
# ═══════════════════════════════════════════════════════════

def make_tile_overlay(tile_id, split_name, rgb, false_color, pred_mask, gt_mask, gt_raw,
                      out_dir, metrics=None):
    """
    2×2 figure:
      TL: True-Color RGB
      TR: False-Color (SWIR2, NIR, Red) — burn scars pop
      BL: RGB + Prediction overlay (orange)
      BR: TP (green) / FP (red) / FN (orange) accuracy map
    """
    from sklearn.metrics import f1_score, jaccard_score

    has_gt = gt_mask is not None

    pred_clean = clean(pred_mask)
    pred_area  = px_to_km2(pred_clean.sum())

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # ── TL: True-Color RGB ──
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title(
        f"HLS True-Color RGB\n{tile_id[:50]}",
        fontsize=11, fontweight="bold",
    )

    # ── TR: False-Color (SWIR2, NIR, Red) ──
    axes[0, 1].imshow(false_color)
    axes[0, 1].set_title(
        f"False-Color (SWIR2 / NIR / Red)\nBurn scars → bright magenta",
        fontsize=11, fontweight="bold",
    )

    # ── BL: RGB + Prediction overlay (orange) ──
    axes[1, 0].imshow(rgb)
    ov_pred = np.zeros((*pred_clean.shape, 4), dtype=np.float32)
    ov_pred[pred_clean] = [1.0, 0.4, 0.0, 0.6]   # orange
    axes[1, 0].imshow(ov_pred)
    axes[1, 0].set_title(
        f"Prediction Overlay\nBurn area: {pred_area:.4f} km²  ({int(pred_clean.sum())} px)",
        fontsize=11, fontweight="bold",
    )
    axes[1, 0].legend(
        handles=[Patch(facecolor=(1, 0.4, 0, 0.6), edgecolor="k", label="Predicted Burn Scar")],
        loc="lower left", fontsize=9,
    )

    # ── BR: Accuracy overlay (TP/FP/FN) or GT overlay ──
    if has_gt:
        gt_clean = clean(gt_mask)
        gt_area  = px_to_km2(gt_clean.sum())

        tp = gt_clean & pred_clean
        fp = ~gt_clean & pred_clean
        fn = gt_clean & ~pred_clean

        g = gt_clean.flatten().astype(int)
        p = pred_clean.flatten().astype(int)
        f1  = f1_score(g, p, zero_division=0)
        iou = jaccard_score(g, p, zero_division=0)

        # Accuracy overlay
        ov3 = np.zeros((*pred_clean.shape, 3), dtype=np.uint8)
        ov3[tp] = [0, 200, 0]      # green
        ov3[fp] = [255, 0, 0]      # red
        ov3[fn] = [255, 165, 0]    # orange
        axes[1, 1].imshow(rgb, alpha=0.35)
        axes[1, 1].imshow(ov3, alpha=0.75)
        axes[1, 1].set_title(
            f"Prediction vs Ground Truth\n"
            f"F1={f1:.3f}  IoU={iou:.3f}  |  GT area: {gt_area:.4f} km²",
            fontsize=11, fontweight="bold",
        )
        axes[1, 1].legend(handles=[
            Patch(facecolor=(0, 0.78, 0), label="True Positive"),
            Patch(facecolor=(1, 0, 0), label="False Positive"),
            Patch(facecolor=(1, 0.65, 0), label="False Negative"),
        ], loc="lower left", fontsize=9)
    else:
        # No GT — show prediction on false-color
        axes[1, 1].imshow(false_color)
        axes[1, 1].imshow(ov_pred)
        axes[1, 1].set_title(
            f"Prediction on False-Color\nBurn area: {pred_area:.4f} km²  [no GT]",
            fontsize=11, fontweight="bold",
        )
        f1, iou = None, None

    for ax in axes.flat:
        ax.axis("off")

    fig.suptitle(
        f"Burn-Scar Detection — {split_name}\n{tile_id}",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    path = os.path.join(out_dir, f"{tile_id}_overlay.jpg")
    fig.savefig(path, dpi=120, bbox_inches="tight", format="jpg")
    plt.close(fig)
    return path, f1, iou


# ═══════════════════════════════════════════════════════════
#  VISUALIZATION 2:  NBR Heatmap
# ═══════════════════════════════════════════════════════════

def make_nbr_figure(tile_id, split_name, nbr, pred_mask, gt_mask, out_dir):
    """
    Left:  NBR heatmap (red = low NBR = burn scar)
    Right: NBR + Prediction contour overlay
    """
    pred_clean = clean(pred_mask)

    # Custom colormap: red (burn) → yellow → green (healthy)
    cmap_nbr = LinearSegmentedColormap.from_list(
        "nbr", [(0.7, 0, 0), (1, 0.6, 0), (1, 1, 0.3), (0.2, 0.8, 0.2), (0, 0.5, 0)],
        N=256,
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: NBR heatmap
    im = axes[0].imshow(nbr, cmap=cmap_nbr, vmin=-0.5, vmax=0.5)
    axes[0].set_title(
        f"Normalized Burn Ratio (NBR)\n(NIR − SWIR2) / (NIR + SWIR2)",
        fontsize=11, fontweight="bold",
    )
    plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04,
                 label="NBR  (low/negative → burn scar)")

    # Right: NBR + prediction contour
    axes[1].imshow(nbr, cmap=cmap_nbr, vmin=-0.5, vmax=0.5)
    # Overlay prediction boundary
    from scipy.ndimage import binary_dilation
    boundary = binary_dilation(pred_clean, structure=np.ones((3, 3))) & ~pred_clean
    ov_b = np.zeros((*boundary.shape, 4), dtype=np.float32)
    ov_b[boundary] = [1.0, 0.0, 0.0, 1.0]
    axes[1].imshow(ov_b)

    # If GT exists, overlay GT boundary in cyan
    if gt_mask is not None:
        gt_clean = clean(gt_mask)
        gt_bnd = binary_dilation(gt_clean, structure=np.ones((3, 3))) & ~gt_clean
        ov_gt = np.zeros((*gt_bnd.shape, 4), dtype=np.float32)
        ov_gt[gt_bnd] = [0.0, 1.0, 1.0, 1.0]
        axes[1].imshow(ov_gt)
        axes[1].legend(handles=[
            Patch(facecolor=(1, 0, 0, 1.0), edgecolor="k", label="Prediction boundary"),
            Patch(facecolor=(0, 1, 1, 1.0), edgecolor="k", label="Ground Truth boundary"),
        ], loc="lower left", fontsize=9)
        axes[1].set_title(
            f"NBR + Boundaries\nRed = Prediction  |  Cyan = Ground Truth",
            fontsize=11, fontweight="bold",
        )
    else:
        axes[1].legend(handles=[
            Patch(facecolor=(1, 0, 0, 1.0), edgecolor="k", label="Prediction boundary"),
        ], loc="lower left", fontsize=9)
        axes[1].set_title(
            f"NBR + Prediction Boundary",
            fontsize=11, fontweight="bold",
        )

    for ax in axes:
        ax.axis("off")

    fig.suptitle(
        f"NBR Analysis — {split_name}\n{tile_id}",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    path = os.path.join(out_dir, f"{tile_id}_nbr.jpg")
    fig.savefig(path, dpi=120, bbox_inches="tight", format="jpg")
    plt.close(fig)
    return path


# ═══════════════════════════════════════════════════════════
#  VISUALIZATION 3:  Per-Split Summary Dashboard
# ═══════════════════════════════════════════════════════════

def make_split_summary(split_name, tile_results, out_dir):
    """Bar chart of F1 per tile + summary statistics."""
    if not tile_results:
        return None

    tiles = [t["tile_id"][:35] for t in tile_results]
    f1s   = [t["f1"] for t in tile_results]
    ious  = [t["iou"] for t in tile_results]
    pred_areas = [t.get("pred_km2", 0) for t in tile_results]
    gt_areas   = [t.get("gt_km2", 0) for t in tile_results]
    avg_f1  = np.mean(f1s)
    avg_iou = np.mean(ious)

    # Determine figure height based on number of tiles
    n = len(tiles)
    fig_h = max(6, min(n * 0.35 + 2, 30))

    fig, axes = plt.subplots(1, 2, figsize=(18, fig_h),
                              gridspec_kw={"width_ratios": [2, 1]})

    # Left — F1 bar chart
    colors = ["#2ecc71" if f >= 0.5 else "#e67e22" if f >= 0.2 else "#e74c3c" for f in f1s]
    y_pos = np.arange(n)
    axes[0].barh(y_pos, f1s, color=colors, edgecolor="k", linewidth=0.4, height=0.7)
    axes[0].axvline(avg_f1, color="navy", ls="--", lw=1.5, label=f"Avg F1 = {avg_f1:.3f}")
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(tiles, fontsize=7)
    axes[0].set_xlabel("F1 Score", fontsize=11)
    axes[0].set_title(f"{split_name} — F1 per Tile", fontsize=13, fontweight="bold")
    axes[0].set_xlim(0, 1.05)
    axes[0].legend(fontsize=10)
    axes[0].grid(axis="x", alpha=0.3)
    axes[0].invert_yaxis()

    # Right — Summary text
    axes[1].axis("off")
    best_idx  = np.argmax(f1s)
    worst_idx = np.argmin(f1s)
    total_pred = sum(pred_areas)
    total_gt   = sum(gt_areas)

    txt = (
        f"Split: {split_name}\n"
        f"{'─' * 40}\n"
        f"Tiles processed:   {n}\n\n"
        f"Avg F1:            {avg_f1:.4f}\n"
        f"Avg IoU:           {avg_iou:.4f}\n"
        f"Median F1:         {np.median(f1s):.4f}\n"
        f"Std F1:            {np.std(f1s):.4f}\n\n"
        f"Best tile:         {tiles[best_idx]}\n"
        f"  (F1={f1s[best_idx]:.3f})\n"
        f"Worst tile:        {tiles[worst_idx]}\n"
        f"  (F1={f1s[worst_idx]:.3f})\n\n"
        f"Total pred area:   {total_pred:.4f} km²\n"
        f"Total GT area:     {total_gt:.4f} km²\n\n"
        f"Model:   Prithvi 100M (burn scars)\n"
        f"Arch:    ViT-Base / FCN head\n"
        f"Input:   HLS 6-band (30 m/px)\n"
        f"Tile:    224×224 sliding window\n"
    )
    axes[1].text(0.05, 0.95, txt, transform=axes[1].transAxes,
                 fontsize=11, verticalalignment="top", fontfamily="monospace",
                 bbox=dict(facecolor="lightyellow", alpha=0.9, edgecolor="gray"))

    plt.suptitle(f"Prithvi Burn-Scar — {split_name} Results Summary",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, f"{split_name}_summary.jpg")
    fig.savefig(path, dpi=120, bbox_inches="tight", format="jpg")
    plt.close(fig)
    return path


# ═══════════════════════════════════════════════════════════
#  TILE / SPLIT DISCOVERY
# ═══════════════════════════════════════════════════════════

def discover_tiles(data_root, pred_root, split, cfg):
    """Find tile image + mask + prediction triples."""
    data_dir = os.path.join(data_root, split)
    pred_geo_dir = os.path.join(pred_root, split, "geotiffs")

    if not os.path.isdir(data_dir):
        print(f"    [WARN] Data dir not found: {data_dir}")
        return []
    if not os.path.isdir(pred_geo_dir):
        print(f"    [WARN] Prediction dir not found: {pred_geo_dir}")
        return []

    img_suffix = cfg["img_suffix"]
    mask_suffix = cfg["mask_suffix"]

    img_files = sorted(glob.glob(os.path.join(data_dir, f"*{img_suffix}")))
    tiles = []
    for img_path in img_files:
        fname = os.path.basename(img_path)
        tile_id = fname.replace(img_suffix, "")
        mask_path = os.path.join(data_dir, tile_id + mask_suffix)
        pred_path = os.path.join(pred_geo_dir, f"{tile_id}_burn_pred.tif")

        if not os.path.exists(pred_path):
            continue  # skip tiles without predictions

        tiles.append(dict(
            tile_id=tile_id,
            image_path=img_path,
            mask_path=mask_path if os.path.exists(mask_path) else None,
            pred_path=pred_path,
        ))
    return tiles


# ═══════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ═══════════════════════════════════════════════════════════

def run_visualization(cfg):
    print("=" * 72)
    print("  Burn-Scar Visualization — Prithvi HLS Fire Predictions")
    print("=" * 72)

    splits = cfg.get("splits", ["training", "validation"])
    if isinstance(splits, str):
        splits = [splits]

    # Discover tiles per split
    all_split_data = {}
    total_tiles = 0
    for sp in splits:
        tiles = discover_tiles(cfg["data_root"], cfg["pred_root"], sp, cfg)
        all_split_data[sp] = tiles
        total_tiles += len(tiles)

    print(f"\n  Data root:   {cfg['data_root']}")
    print(f"  Pred root:   {cfg['pred_root']}")
    print(f"  Output:      {cfg['output_root']}")
    for sp, tiles in all_split_data.items():
        print(f"    {sp}: {len(tiles)} tiles with predictions")
    print(f"  Total:       {total_tiles}\n")

    if total_tiles == 0:
        print("  No tiles with predictions found. Run fire_inference_pipeline.py first.")
        return

    t_global = time.time()

    for split_name, tiles in all_split_data.items():
        if not tiles:
            continue

        # Apply max_tiles limit
        max_tiles = cfg.get("_max_tiles")
        if max_tiles is not None:
            tiles = tiles[:max_tiles]

        n = len(tiles)
        out_overlay = os.path.join(cfg["output_root"], split_name, "overlays")
        out_nbr     = os.path.join(cfg["output_root"], split_name, "nbr")
        os.makedirs(out_overlay, exist_ok=True)
        os.makedirs(out_nbr, exist_ok=True)

        print(f"  +-- {split_name}  ({n} tiles) " + "-" * 30)
        tile_results = []
        t_split = time.time()

        for idx, tile_info in enumerate(tiles):
            tid = tile_info["tile_id"]
            t0 = time.time()

            # Load imagery
            img_path = tile_info["image_path"]
            rgb = load_hls_rgb(img_path)
            fc  = load_hls_false_color(img_path)
            nbr = compute_nbr(img_path)

            # Load prediction
            pred_mask = load_pred_mask(tile_info["pred_path"])

            # Load GT if available
            gt_mask, gt_raw = None, None
            if tile_info["mask_path"] is not None:
                gt_mask, gt_raw = load_gt_mask(tile_info["mask_path"])

            # 1) 2×2 overlay
            ov_path, f1_val, iou_val = make_tile_overlay(
                tid, split_name, rgb, fc, pred_mask, gt_mask, gt_raw, out_overlay,
            )

            # 2) NBR figure
            nbr_path = make_nbr_figure(tid, split_name, nbr, pred_mask, gt_mask, out_nbr)

            elapsed = time.time() - t0

            pred_area = px_to_km2(clean(pred_mask).sum())
            gt_area   = px_to_km2(clean(gt_mask).sum()) if gt_mask is not None else 0

            if f1_val is not None:
                tile_results.append(dict(
                    tile_id=tid, f1=f1_val, iou=iou_val,
                    pred_km2=pred_area, gt_km2=gt_area,
                ))
                print(f"  |  [{idx+1}/{n}] {tid[:45]}...  "
                      f"F1={f1_val:.3f}  IoU={iou_val:.3f}  ({elapsed:.1f}s)")
            else:
                print(f"  |  [{idx+1}/{n}] {tid[:45]}...  "
                      f"Pred={pred_area:.4f} km²  [no GT]  ({elapsed:.1f}s)")

        split_time = time.time() - t_split

        # 3) Summary dashboard
        if tile_results:
            summary_path = make_split_summary(
                split_name, tile_results,
                os.path.join(cfg["output_root"], split_name),
            )
            avg_f1  = np.mean([t["f1"] for t in tile_results])
            avg_iou = np.mean([t["iou"] for t in tile_results])
            print(f"  |  Avg F1={avg_f1:.3f}  Avg IoU={avg_iou:.3f}")
            if summary_path:
                print(f"  |  Summary → {summary_path}")

        print(f"  +-- {split_name} done ({split_time:.1f}s)\n")

    total_time = time.time() - t_global

    print("=" * 72)
    print("  VISUALIZATION COMPLETE")
    print("=" * 72)
    print(f"""
  Outputs ({cfg['output_root']}/):
    +-- <split>/
    |   +-- overlays/   2×2 RGB + False-Color + Pred + Accuracy JPGs
    |   +-- nbr/        NBR heatmap + boundary overlay JPGs
    |   +-- <split>_summary.jpg  (F1 bar chart + stats)

  Total time: {total_time:.1f}s
""")


# ═══════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Visualize Prithvi burn-scar predictions on HLS Fire data",
    )
    p.add_argument("--data_root", default=DEFAULTS["data_root"],
                   help="Root of Fire data (contains training/ and validation/)")
    p.add_argument("--pred_root", default=DEFAULTS["pred_root"],
                   help="Root of inference outputs (contains <split>/geotiffs/)")
    p.add_argument("--output_root", default=DEFAULTS["output_root"],
                   help="Root output directory for visualizations")
    p.add_argument("--splits", nargs="*", default=["training", "validation"],
                   help="Which splits to visualize")
    p.add_argument("--max_tiles", type=int, default=None,
                   help="Max tiles per split (for quick testing)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = dict(DEFAULTS)
    cfg.update(
        data_root   = args.data_root,
        pred_root   = args.pred_root,
        output_root = args.output_root,
        splits      = args.splits,
    )
    if args.max_tiles is not None:
        cfg["_max_tiles"] = args.max_tiles

    run_visualization(cfg)


if __name__ == "__main__":
    main()
