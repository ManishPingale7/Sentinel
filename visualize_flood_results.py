"""
Flood Visualization — Before vs During + Overlays (all CEMS events)
====================================================================
Reads ai4g-flood predictions + original S1/S2 imagery and generates
per-tile 2×2 overlay images (Prithvi-style):

    Top-Left:     Before flood (S2 RGB or S1 grayscale)
    Top-Right:    During flood (S2 RGB or S1 grayscale)
    Bottom-Left:  Before + flood overlay (blue water)
    Bottom-Right: During + change overlay (blue=permanent, red=NEW flood)

Also generates:
    - Per-tile TP/FP/FN accuracy overlay vs ground truth
    - Per-event summary dashboard

Usage:
    python visualize_flood_results.py
    python visualize_flood_results.py --events EMSR771
    python visualize_flood_results.py --pred_root "G:\\path\\to\\ai4g_flood"
"""

import argparse
import glob
import os
import sys
import time

import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.ndimage import binary_opening, binary_closing, gaussian_filter

PIXEL_RES_M = 10.0


# ═══════════════════════════════════════════════════════════
#  DEFAULTS
# ═══════════════════════════════════════════════════════════
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULTS = dict(
    cems_root  = os.path.join(_BASE_DIR, "DATA", "CEMS"),
    pred_root  = os.path.join(_BASE_DIR, "OUTPUTS", "ai4g_flood"),
    output_root= os.path.join(_BASE_DIR, "OUTPUTS", "ai4g_flood"),
)


# ═══════════════════════════════════════════════════════════
#  DATA LOADERS
# ═══════════════════════════════════════════════════════════

def _normalize(band):
    """Percentile-clip + normalize to [0,1]."""
    valid = band[band > 0]
    if valid.size == 0:
        return np.zeros_like(band, dtype=np.float32)
    p2, p98 = np.percentile(valid, [2, 98])
    band = np.clip(band, p2, p98).astype(np.float32)
    lo, hi = band.min(), band.max()
    return np.zeros_like(band, dtype=np.float32) if (hi - lo) < 1e-6 else (band - lo) / (hi - lo)


def load_s2_rgb(path):
    """Load 8-band S2 tile → RGB [H,W,3] float [0,1]."""
    with rasterio.open(path) as src:
        r = src.read(3).astype(np.float32)
        g = src.read(2).astype(np.float32)
        b = src.read(1).astype(np.float32)
    return np.stack([_normalize(r), _normalize(g), _normalize(b)], axis=-1)


def load_s1_vv(path):
    """Load S1 tile → VV grayscale [H,W,3] for display."""
    with rasterio.open(path) as src:
        vv = src.read(1).astype(np.float32)
    vv = np.clip(np.nan_to_num(vv * 2.0 + 135.0, nan=0), 0, 255) / 255.0
    return np.stack([vv, vv, vv], axis=-1)


def load_background(s2_path, s1_path):
    """Load S2 RGB if available, else S1 grayscale. Returns (image, label)."""
    if s2_path and os.path.exists(s2_path):
        return load_s2_rgb(s2_path), "Sentinel-2 RGB"
    elif s1_path and os.path.exists(s1_path):
        return load_s1_vv(s1_path), "Sentinel-1 VV"
    else:
        return None, None


def load_pred_mask(path):
    """Load ai4g prediction → boolean mask."""
    with rasterio.open(path) as src:
        data = src.read(1)
    return np.nan_to_num(data, nan=0.0) > 0


def load_gt_mask(path):
    """Load ground truth. Returns (flood_bool, raw)."""
    with rasterio.open(path) as src:
        raw = src.read(1)
    return (raw == 1), raw


def clean(mask, k=3):
    return binary_opening(mask, structure=np.ones((k, k)))


def px_to_km2(n):
    return n * (PIXEL_RES_M ** 2) / 1e6


# ═══════════════════════════════════════════════════════════
#  WATER DETECTION  (S2 NDWI + SAR fallback)
# ═══════════════════════════════════════════════════════════

def detect_water_s2(s2_path, ndwi_thresh=0.0):
    """Detect water using NDWI from Sentinel-2 optical data.

    NDWI = (Green - NIR) / (Green + NIR)  →  positive for water.
    SenForFlood S2 bands: 1=B2, 2=B3(Green), 3=B4(Red), 4=B8A(NIR).
    """
    if not s2_path or not os.path.exists(s2_path):
        return None
    try:
        with rasterio.open(s2_path) as src:
            green = src.read(2).astype(np.float32)  # B3 Green
            nir   = src.read(4).astype(np.float32)  # B8A NIR
        denom = green + nir
        denom[denom == 0] = 1e-10
        ndwi = (green - nir) / denom

        # Smooth to reduce noise, then threshold
        ndwi_smooth = gaussian_filter(ndwi, sigma=1.5)
        water = ndwi_smooth > ndwi_thresh

        # Morphological closing to connect water bodies
        water = binary_closing(water, structure=np.ones((5, 5)), iterations=2)
        # Opening to remove small speckle
        water = binary_opening(water, structure=np.ones((3, 3)))
        return water
    except Exception:
        return None


def detect_water_sar(s1_path):
    """Detect water from SAR backscatter with speckle filtering.

    Water surfaces produce very low backscatter (specular reflection).
    Uses Gaussian smoothing + adaptive thresholding + morphological
    closing for continuous water bodies.
    """
    if not s1_path or not os.path.exists(s1_path):
        return None
    try:
        with rasterio.open(s1_path) as src:
            vv_db = src.read(1).astype(np.float32)
            vh_db = src.read(2).astype(np.float32)

        # Scale to 0-255 (same as ai4g db_scale)
        vv = np.clip(np.nan_to_num(vv_db * 2.0 + 135.0, nan=0), 0, 255)
        vh = np.clip(np.nan_to_num(vh_db * 2.0 + 135.0, nan=0), 0, 255)

        # Gaussian smoothing to suppress SAR speckle
        vv_sm = gaussian_filter(vv, sigma=2.0)
        vh_sm = gaussian_filter(vh, sigma=2.0)

        # Adaptive threshold: use Otsu if available, else percentile
        valid = (vv > 0) & (vh > 0)
        if valid.sum() == 0:
            return np.zeros(vv.shape, dtype=bool)

        try:
            from skimage.filters import threshold_otsu
            vv_thresh = threshold_otsu(vv_sm[valid])
            vh_thresh = threshold_otsu(vh_sm[valid])
        except Exception:
            vv_thresh = np.percentile(vv_sm[valid], 30)
            vh_thresh = np.percentile(vh_sm[valid], 30)

        # Water is dark in BOTH VV and VH
        water = (vv_sm < vv_thresh) & (vh_sm < vh_thresh)

        # Morphological closing to fill gaps → continuous water bodies
        water = binary_closing(water, structure=np.ones((7, 7)), iterations=3)
        # Opening to remove remaining speckle
        water = binary_opening(water, structure=np.ones((3, 3)))
        return water
    except Exception:
        return None


def detect_water(s2_path, s1_path):
    """Detect water: prefer S2 NDWI (cleaner), fall back to SAR."""
    mask = detect_water_s2(s2_path)
    if mask is not None:
        return mask
    mask = detect_water_sar(s1_path)
    if mask is not None:
        return mask
    return None


# ═══════════════════════════════════════════════════════════
#  VISUALIZATION 1:  Before vs During  (2×2 Prithvi-style)
# ═══════════════════════════════════════════════════════════

def make_before_during(event_id, tile_id, before_img, during_img, img_label,
                       before_water, during_water, ai4g_change, out_dir):
    """
    TL: Before (original)       TR: During (original)
    BL: Before + water (blue)   BR: During + change (blue=perm, red=new)

    before_water:  water detected in the BEFORE image (S2 NDWI or SAR)
    during_water:  water detected in the DURING image (S2 NDWI or SAR)
    ai4g_change:   ai4g model change-detection output (new flooding)
    """
    # Combine during_water + ai4g change → total water during flood
    during_total = during_water | ai4g_change

    # Compute change categories (like Prithvi pipeline)
    permanent  = clean(before_water & during_total)   # water in both periods
    new_flood  = clean(during_total & ~before_water)  # water only during
    receded    = clean(before_water & ~during_total)   # water only before
    before_clean = clean(before_water)

    bef_km2 = px_to_km2(before_clean.sum())
    dur_km2 = px_to_km2(during_total.sum())
    new_km2 = px_to_km2(new_flood.sum())
    rec_km2 = px_to_km2(receded.sum())
    pct = f"+{100 * new_flood.sum() / before_clean.sum():.1f}%" if before_clean.sum() > 0 else "New"

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # TL — Before original
    axes[0, 0].imshow(before_img)
    axes[0, 0].set_title(f"BEFORE Flood ({img_label})\n{event_id} / tile {tile_id}",
                         fontsize=11, fontweight="bold")

    # TR — During original
    axes[0, 1].imshow(during_img)
    axes[0, 1].set_title(f"DURING Flood ({img_label})\n{event_id} / tile {tile_id}",
                         fontsize=11, fontweight="bold")

    # BL — Before + water overlay
    axes[1, 0].imshow(before_img)
    ov = np.zeros((*before_clean.shape, 4), dtype=np.float32)
    ov[before_clean] = [0.0, 0.4, 1.0, 0.55]
    axes[1, 0].imshow(ov)
    axes[1, 0].set_title(f"BEFORE Overlay — Flood Mask\nFlood area: {bef_km2:.4f} km²",
                         fontsize=11, fontweight="bold")
    axes[1, 0].legend(
        handles=[Patch(facecolor=(0, 0.4, 1, 0.55), edgecolor="k", label="Detected Water")],
        loc="lower left", fontsize=9,
    )

    # BR — During + change overlay
    axes[1, 1].imshow(during_img)
    ov2 = np.zeros((*during_total.shape, 4), dtype=np.float32)
    ov2[permanent]  = [0.0, 0.4, 1.0, 0.50]
    ov2[new_flood]  = [1.0, 0.0, 0.0, 0.65]
    axes[1, 1].imshow(ov2)
    axes[1, 1].set_title(
        f"DURING Overlay — Change Detection\n"
        f"Total: {dur_km2:.4f} km²  |  New: +{new_km2:.4f} km²  |  Increase: {pct}",
        fontsize=11, fontweight="bold",
    )
    axes[1, 1].legend(handles=[
        Patch(facecolor=(0, 0.4, 1, 0.50), edgecolor="k", label="Pre-existing Water"),
        Patch(facecolor=(1, 0, 0, 0.65), edgecolor="k", label="NEW Flood"),
    ], loc="lower left", fontsize=9)

    for ax in axes.flat:
        ax.axis("off")

    fig.suptitle(
        f"Flood Change Detection — {event_id}  tile {tile_id}\n"
        f"Blue = existing water  |  Red = new flood expansion  |  "
        f"Receded: {rec_km2:.4f} km²",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    path = os.path.join(out_dir, f"{tile_id}_before_vs_during.jpg")
    fig.savefig(path, dpi=120, bbox_inches="tight", format="jpg")
    plt.close(fig)
    return path


# ═══════════════════════════════════════════════════════════
#  VISUALIZATION 2:  Accuracy Overlay  (Pred vs GT)
# ═══════════════════════════════════════════════════════════

def make_accuracy_overlay(event_id, tile_id, during_img, pred_flood, gt_flood, gt_raw, out_dir):
    """
    Left:  Prediction overlay (blue) on image
    Center: Ground truth overlay (red=flood, gray=perm water)
    Right: TP (green) / FP (red) / FN (orange) overlay
    """
    from sklearn.metrics import f1_score, jaccard_score

    tp = gt_flood & pred_flood
    fp = ~gt_flood & pred_flood
    fn = gt_flood & ~pred_flood

    g = gt_flood.flatten().astype(int)
    p = pred_flood.flatten().astype(int)
    f1  = f1_score(g, p, zero_division=0)
    iou = jaccard_score(g, p, zero_division=0)

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))

    # Left — Prediction
    axes[0].imshow(during_img)
    ov = np.zeros((*pred_flood.shape, 4), dtype=np.float32)
    ov[pred_flood] = [0, 0.47, 1.0, 0.65]
    axes[0].imshow(ov)
    axes[0].set_title(f"Prediction (ai4g-flood)\nFlood: {px_to_km2(pred_flood.sum()):.4f} km²",
                      fontsize=12, fontweight="bold")

    # Center — Ground truth
    axes[1].imshow(during_img)
    gt_vis = np.zeros((*gt_raw.shape, 4), dtype=np.float32)
    gt_vis[gt_raw == 1] = [1, 0.2, 0.2, 0.7]
    gt_vis[gt_raw == 2] = [0.5, 0.5, 0.5, 0.5]
    axes[1].imshow(gt_vis)
    axes[1].set_title(f"Ground Truth\nFlood: {px_to_km2(gt_flood.sum()):.4f} km²",
                      fontsize=12, fontweight="bold")
    axes[1].legend(handles=[
        Patch(facecolor=(1, 0.2, 0.2, 0.7), edgecolor="k", label="Flood"),
        Patch(facecolor=(0.5, 0.5, 0.5, 0.5), edgecolor="k", label="Perm. Water"),
    ], loc="lower left", fontsize=9)

    # Right — TP/FP/FN
    ov3 = np.zeros((*pred_flood.shape, 3), dtype=np.uint8)
    ov3[tp] = [0, 200, 0]
    ov3[fp] = [255, 0, 0]
    ov3[fn] = [255, 165, 0]
    axes[2].imshow(during_img, alpha=0.35)
    axes[2].imshow(ov3, alpha=0.75)
    axes[2].set_title(f"Prediction vs Ground Truth\nF1={f1:.3f}  IoU={iou:.3f}",
                      fontsize=12, fontweight="bold")
    axes[2].legend(handles=[
        Patch(facecolor=(0, 0.78, 0), label="True Positive"),
        Patch(facecolor=(1, 0, 0), label="False Positive"),
        Patch(facecolor=(1, 0.65, 0), label="False Negative"),
    ], loc="lower left", fontsize=9)

    for ax in axes:
        ax.axis("off")

    fig.suptitle(f"{event_id} tile {tile_id} — Prediction Accuracy",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, f"{tile_id}_accuracy.jpg")
    fig.savefig(path, dpi=120, bbox_inches="tight", format="jpg")
    plt.close(fig)
    return path, f1, iou


# ═══════════════════════════════════════════════════════════
#  VISUALIZATION 3:  Per-Event Summary Dashboard
# ═══════════════════════════════════════════════════════════

def make_event_summary(event_id, tile_results, out_dir):
    """Bar chart + confusion stats for one event."""
    if not tile_results:
        return None

    tiles  = [t["tile_id"] for t in tile_results]
    f1s    = [t["f1"] for t in tile_results]
    ious   = [t["iou"] for t in tile_results]
    avg_f1 = np.mean(f1s)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Bar chart
    x = np.arange(len(tiles))
    colors = ["#2ecc71" if f >= 0.3 else "#e74c3c" for f in f1s]
    axes[0].barh(tiles, f1s, color=colors, edgecolor="k", linewidth=0.5)
    axes[0].axvline(avg_f1, color="navy", ls="--", label=f"Avg F1 = {avg_f1:.3f}")
    axes[0].set_xlabel("F1 Score", fontsize=11)
    axes[0].set_title(f"{event_id} — F1 per Tile", fontsize=13, fontweight="bold")
    axes[0].set_xlim(0, 1.05)
    axes[0].legend(fontsize=10)
    axes[0].grid(axis="x", alpha=0.3)

    # Summary text
    axes[1].axis("off")
    avg_iou = np.mean(ious)
    best_tile = tiles[np.argmax(f1s)]
    worst_tile = tiles[np.argmin(f1s)]
    txt = (
        f"Event: {event_id}\n"
        f"{'-' * 40}\n"
        f"Tiles processed:  {len(tiles)}\n\n"
        f"Avg F1:     {avg_f1:.4f}\n"
        f"Avg IoU:    {avg_iou:.4f}\n"
        f"Best tile:  {best_tile} (F1={max(f1s):.3f})\n"
        f"Worst tile: {worst_tile} (F1={min(f1s):.3f})\n\n"
        f"Model:  ai4g-flood (SAR change detection)\n"
        f"Arch:   U-Net / MobileNetV2\n"
        f"Input:  Sentinel-1 VV+VH (pre+post pair)\n"
    )
    axes[1].text(0.05, 0.95, txt, transform=axes[1].transAxes,
                 fontsize=12, verticalalignment="top", fontfamily="monospace",
                 bbox=dict(facecolor="lightyellow", alpha=0.9, edgecolor="gray"))

    plt.suptitle(f"{event_id} — ai4g-flood Results Summary",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, f"{event_id}_summary.jpg")
    fig.savefig(path, dpi=120, bbox_inches="tight", format="jpg")
    plt.close(fig)
    return path


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════

def run_visualization(cfg):
    print("=" * 72)
    print("  Flood Visualization — Before vs During + Overlays")
    print("=" * 72)

    cems_root = cfg["cems_root"]
    pred_root = cfg["pred_root"]
    out_root  = cfg["output_root"]

    # Discover events that have predictions
    events = []
    for name in sorted(os.listdir(cems_root)):
        event_dir = os.path.join(cems_root, name)
        pred_dir  = os.path.join(pred_root, name, "geotiffs")
        if not os.path.isdir(event_dir) or not os.path.isdir(pred_dir):
            continue
        if cfg.get("events") and name not in cfg["events"]:
            continue
        events.append(name)

    print(f"\n  CEMS root:  {cems_root}")
    print(f"  Pred root:  {pred_root}")
    print(f"  Events:     {len(events)}  ({', '.join(events)})\n")

    for event_id in events:
        event_data_dir = os.path.join(cems_root, event_id)
        pred_geo_dir   = os.path.join(pred_root, event_id, "geotiffs")
        out_overlay    = os.path.join(out_root, event_id, "overlays")
        out_accuracy   = os.path.join(out_root, event_id, "accuracy")
        os.makedirs(out_overlay, exist_ok=True)
        os.makedirs(out_accuracy, exist_ok=True)

        # Get tile IDs from predictions
        pred_files = sorted(glob.glob(os.path.join(pred_geo_dir, "*_flood_pred.tif")))
        tile_ids = [os.path.basename(f).split("_")[0] for f in pred_files]
        n = len(tile_ids)

        print(f"  +-- {event_id}  ({n} tiles)")

        tile_results = []

        for idx, tid in enumerate(tile_ids):
            print(f"  |  [{idx+1}/{n}] {tid} ...", end=" ", flush=True)

            # Paths
            s1_before = os.path.join(event_data_dir, "s1_before_flood", f"{tid}_s1_before_flood.tif")
            s1_during = os.path.join(event_data_dir, "s1_during_flood", f"{tid}_s1_during_flood.tif")
            s2_before = os.path.join(event_data_dir, "s2_before_flood", f"{tid}_s2_before_flood.tif")
            s2_during = os.path.join(event_data_dir, "s2_during_flood", f"{tid}_s2_during_flood.tif")
            pred_path = os.path.join(pred_geo_dir, f"{tid}_flood_pred.tif")
            gt_path   = os.path.join(event_data_dir, "flood_mask", f"{tid}_flood_mask.tif")

            # Load background images
            before_img, lbl = load_background(s2_before, s1_before)
            during_img, _   = load_background(s2_during, s1_during)
            if before_img is None or during_img is None:
                print("SKIP (no image)")
                continue

            # Load ai4g change-detection prediction
            ai4g_change = load_pred_mask(pred_path)

            # Try loading pre-computed water masks (from inference pipeline),
            # fall back to on-the-fly detection (S2 NDWI → SAR fallback)
            bef_water_path = os.path.join(pred_geo_dir, f"{tid}_before_water.tif")
            dur_water_path = os.path.join(pred_geo_dir, f"{tid}_during_water.tif")

            if os.path.exists(bef_water_path):
                with rasterio.open(bef_water_path) as src:
                    before_water = src.read(1).astype(bool)
            else:
                before_water = detect_water(s2_before, s1_before)

            if os.path.exists(dur_water_path):
                with rasterio.open(dur_water_path) as src:
                    during_water = src.read(1).astype(bool)
            else:
                during_water = detect_water(s2_during, s1_during)

            if before_water is None:
                before_water = np.zeros(ai4g_change.shape, dtype=bool)
            if during_water is None:
                during_water = np.zeros(ai4g_change.shape, dtype=bool)

            # ── 1) Before vs During overlay ──
            make_before_during(
                event_id, tid, before_img, during_img, lbl,
                before_water, during_water, ai4g_change, out_overlay,
            )

            # ── 2) Accuracy overlay (if GT exists) ──
            f1_val, iou_val = 0, 0
            if os.path.exists(gt_path):
                gt_flood, gt_raw = load_gt_mask(gt_path)
                _, f1_val, iou_val = make_accuracy_overlay(
                    event_id, tid, during_img, ai4g_change, gt_flood, gt_raw, out_accuracy,
                )
                tile_results.append(dict(tile_id=tid, f1=f1_val, iou=iou_val))
                print(f"F1={f1_val:.3f}  IoU={iou_val:.3f}")
            else:
                print("OK [no GT]")

        # ── 3) Event summary ──
        summary_path = make_event_summary(event_id, tile_results,
                                          os.path.join(out_root, event_id))
        if summary_path:
            print(f"  |  Summary -> {os.path.basename(summary_path)}")
        print(f"  +-- {event_id} done\n")

    print("=" * 72)
    print("  VISUALIZATION COMPLETE")
    print("=" * 72)
    print(f"""
  Outputs per event ({out_root}/<EVENT>/):
    +-- overlays/     2x2 Before vs During overlay JPGs
    +-- accuracy/     Prediction vs Ground Truth (TP/FP/FN) JPGs
    +-- <EVENT>_summary.jpg
""")


# ═══════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Visualize ai4g-flood predictions")
    p.add_argument("--cems_root", default=DEFAULTS["cems_root"])
    p.add_argument("--pred_root", default=DEFAULTS["pred_root"],
                   help="Root of ai4g predictions (contains <EVENT>/geotiffs/)")
    p.add_argument("--output_root", default=DEFAULTS["output_root"])
    p.add_argument("--events", nargs="*", default=None,
                   help="Only visualize these events (e.g. EMSR771)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = dict(DEFAULTS)
    cfg.update(
        cems_root   = args.cems_root,
        pred_root   = args.pred_root,
        output_root = args.output_root,
        events      = args.events,
    )
    run_visualization(cfg)


if __name__ == "__main__":
    main()
