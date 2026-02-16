"""
Flood Visualization – Before vs During (side-by-side)
=====================================================
Reads Prithvi prediction masks + original SenForFlood images and generates
publication-ready side-by-side figures:

  Left  → Before-flood RGB with flood mask overlay (blue)
  Right → During-flood RGB with overlay:
            Blue  = pre-existing / permanent water
            Red   = NEW flood expansion
  Title → CEMS ID, tile, area stats, % increase

Usage:
    python visualize_flood.py                     # interactive – shows plots
    python visualize_flood.py --save              # save PNGs, no display
    python visualize_flood.py --save --cems EMSN194
    python visualize_flood.py --save -n 3         # first 3 tiles only
"""

import argparse
import csv
import glob
import os
import sys

import numpy as np
import rasterio
import matplotlib
import matplotlib.pyplot as plt
from scipy.ndimage import binary_opening

# ── Default paths (edit if needed) ──────────────────────────────────────
SENFORFLOOD_ROOT = r"G:\Sentinel\DATA\SenForFlood\CEMS"
PRED_ROOT        = r"G:\Sentinel\OUTPUTS\Prithvi"
REPORT_CSV       = r"G:\Sentinel\OUTPUTS\change_detection_report.csv"
SAVE_DIR         = r"G:\Sentinel\OUTPUTS\Visualizations"
PIXEL_RES_M      = 10.0


# ── Helpers ─────────────────────────────────────────────────────────────

def _normalize_band(band):
    """Percentile-clip + min-max normalize to [0,1]."""
    valid = band[band > 0]
    if valid.size == 0:
        return np.zeros_like(band, dtype=np.float32)
    p2  = np.percentile(valid, 2)
    p98 = np.percentile(valid, 98)
    band = np.clip(band, p2, p98).astype(np.float32)
    lo, hi = band.min(), band.max()
    if hi - lo < 1e-6:
        return np.zeros_like(band, dtype=np.float32)
    return (band - lo) / (hi - lo)


def load_rgb(tif_path):
    """Load 8-band SenForFlood .tif → RGB array [H, W, 3]."""
    with rasterio.open(tif_path) as src:
        red   = src.read(3).astype(np.float32)   # B4
        green = src.read(2).astype(np.float32)   # B3
        blue  = src.read(1).astype(np.float32)   # B2
    return np.stack([_normalize_band(red),
                     _normalize_band(green),
                     _normalize_band(blue)], axis=-1)


def load_mask(pred_path):
    """Load single-band prediction .tif → 2-D int array."""
    with rasterio.open(pred_path) as src:
        return src.read(1)


def load_report(csv_path):
    """Return dict  (cems_id, tile) → row-dict."""
    lut = {}
    if not os.path.exists(csv_path):
        return lut
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            lut[(row["cems_id"], row["tile"])] = row
    return lut


def discover_pairs(pred_root):
    """Scan OUTPUTS/Prithvi and return list of dicts with all paths.

    Each entry:
        cems_id, tile,
        before_pred, during_pred,
        before_src, during_src   (original SenForFlood images)
    """
    entries = []
    for cems_id in sorted(os.listdir(pred_root)):
        before_dir = os.path.join(pred_root, cems_id, "before")
        during_dir = os.path.join(pred_root, cems_id, "during")
        if not os.path.isdir(before_dir) or not os.path.isdir(during_dir):
            continue

        before_preds = {
            os.path.basename(f).split("_")[0]: f
            for f in sorted(glob.glob(os.path.join(before_dir, "*_pred.tif")))
        }
        during_preds = {
            os.path.basename(f).split("_")[0]: f
            for f in sorted(glob.glob(os.path.join(during_dir, "*_pred.tif")))
        }

        for tile in sorted(set(before_preds) & set(during_preds)):
            # Locate original images for RGB
            before_src_dir = os.path.join(SENFORFLOOD_ROOT, cems_id, "s2_before_flood")
            during_src_dir = os.path.join(SENFORFLOOD_ROOT, cems_id, "s2_during_flood")
            before_src = glob.glob(os.path.join(before_src_dir, f"{tile}_*.tif"))
            during_src = glob.glob(os.path.join(during_src_dir, f"{tile}_*.tif"))
            if not before_src or not during_src:
                continue

            entries.append({
                "cems_id": cems_id,
                "tile": tile,
                "before_pred": before_preds[tile],
                "during_pred": during_preds[tile],
                "before_src": before_src[0],
                "during_src": during_src[0],
            })
    return entries


# ── Main visualisation ──────────────────────────────────────────────────

def visualize_pair(entry, metrics, save_path=None, show=True):
    """Generate a 2×2 quadrant figure for a before/during pair.

    Top-left:     Original Before-flood RGB
    Top-right:    Original During-flood RGB
    Bottom-left:  Before-flood RGB + flood overlay (blue)
    Bottom-right: During-flood RGB + overlay (blue=permanent, red=new expansion)
    """

    # Load data
    rgb_before = load_rgb(entry["before_src"])
    rgb_during = load_rgb(entry["during_src"])
    mask_before = load_mask(entry["before_pred"])
    mask_during = load_mask(entry["during_pred"])

    before_flood = (mask_before == 1)
    during_flood = (mask_during == 1)
    new_flood    = during_flood & ~before_flood
    permanent    = before_flood & during_flood

    # Morphological cleanup
    struct = np.ones((3, 3))
    permanent_clean = binary_opening(permanent, structure=struct)
    new_clean       = binary_opening(new_flood, structure=struct)
    before_clean    = binary_opening(before_flood, structure=struct)
    during_clean    = binary_opening(during_flood, structure=struct)

    # Metrics
    cems_id = entry["cems_id"]
    tile    = entry["tile"]
    m = metrics.get((cems_id, tile), {})
    before_km2    = float(m.get("before_km2", 0))
    during_km2    = float(m.get("during_km2", 0))
    new_km2       = float(m.get("new_flood_km2", 0))
    receded_km2   = float(m.get("receded_km2", 0))
    pct_raw       = m.get("pct_increase", "0")
    try:
        pct = float(pct_raw)
    except ValueError:
        pct = 0.0

    pct_str = f"{pct:.1f}%" if pct != float("inf") else "N/A (no prior flood)"

    # ── 2×2 Figure ──────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(18, 16))

    # ── TOP-LEFT: Original Before RGB ───────────────────────────────
    axes[0, 0].imshow(rgb_before)
    axes[0, 0].set_title(
        f"Original BEFORE Flood\n{cems_id} / tile {tile}",
        fontsize=12, fontweight="bold",
    )
    axes[0, 0].axis("off")

    # ── TOP-RIGHT: Original During RGB ──────────────────────────────
    axes[0, 1].imshow(rgb_during)
    axes[0, 1].set_title(
        f"Original DURING Flood\n{cems_id} / tile {tile}",
        fontsize=12, fontweight="bold",
    )
    axes[0, 1].axis("off")

    # ── BOTTOM-LEFT: Before + Overlay (blue = detected flood) ──────
    axes[1, 0].imshow(rgb_before)
    blue_layer = np.zeros((*before_clean.shape, 4), dtype=np.float32)
    blue_layer[before_clean] = [0.0, 0.4, 1.0, 0.5]
    axes[1, 0].imshow(blue_layer)
    axes[1, 0].set_title(
        f"BEFORE Overlay – Flood Mask\n"
        f"Flood area: {before_km2:.4f} km²",
        fontsize=12, fontweight="bold",
    )
    axes[1, 0].axis("off")

    # ── BOTTOM-RIGHT: During + Overlay (blue=permanent, red=new) ───
    axes[1, 1].imshow(rgb_during)

    # Blue → permanent / pre-existing water
    blue_layer2 = np.zeros((*permanent_clean.shape, 4), dtype=np.float32)
    blue_layer2[permanent_clean] = [0.0, 0.4, 1.0, 0.5]
    axes[1, 1].imshow(blue_layer2)

    # Red → new flood expansion
    red_layer = np.zeros((*new_clean.shape, 4), dtype=np.float32)
    red_layer[new_clean] = [1.0, 0.0, 0.0, 0.6]
    axes[1, 1].imshow(red_layer)

    axes[1, 1].set_title(
        f"DURING Overlay – Change Detection\n"
        f"Total: {during_km2:.4f} km²  |  "
        f"New: +{new_km2:.4f} km²  |  "
        f"Increase: {pct_str}",
        fontsize=12, fontweight="bold",
    )
    axes[1, 1].axis("off")

    # ── Suptitle ────────────────────────────────────────────────────
    fig.suptitle(
        f"Flood Change Detection — {cems_id}  tile {tile}\n"
        f"Blue = existing water  |  Red = new flood expansion  |  "
        f"Receded: {receded_km2:.4f} km²",
        fontsize=13, fontweight="bold", y=1.01,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.98])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


# ── CLI ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Visualize flood before/during side-by-side from Prithvi outputs"
    )
    p.add_argument("--save", action="store_true",
                   help="Save PNGs instead of displaying interactively")
    p.add_argument("--save-dir", default=SAVE_DIR,
                   help="Directory for saved PNGs")
    p.add_argument("--cems", default=None,
                   help="Only visualize a specific CEMS folder (e.g. EMSN194)")
    p.add_argument("-n", "--num-images", type=int, default=None,
                   help="Number of tile pairs to visualize (default: all)")
    p.add_argument("--pred-root", default=PRED_ROOT)
    p.add_argument("--report", default=REPORT_CSV)
    return p.parse_args()


def main():
    args = parse_args()

    if not args.save:
        matplotlib.use("TkAgg")   # switch to interactive backend
        import importlib
        importlib.reload(plt)

    print("Scanning prediction outputs …")
    entries = discover_pairs(args.pred_root)
    if not entries:
        print("No prediction pairs found. Run flood_pipeline.py first.")
        return

    # Filter by CEMS folder if requested
    if args.cems:
        entries = [e for e in entries if e["cems_id"] == args.cems]

    print(f"Found {len(entries)} before/during pairs.")

    # Interactive limit
    if args.num_images is None and not args.save:
        print(f"How many pairs to visualize? (1-{len(entries)}, Enter for all)")
        try:
            user = input("▸ ").strip()
            if user:
                n = int(user)
                if 1 <= n <= len(entries):
                    entries = entries[:n]
                    print(f"→ Showing first {n} pair(s).")
                else:
                    print(f"Invalid. Showing all {len(entries)}.")
        except (ValueError, EOFError):
            pass
    elif args.num_images is not None:
        entries = entries[:args.num_images]
        print(f"→ Visualizing first {len(entries)} pair(s).")

    # Load metrics
    metrics = load_report(args.report)

    # Generate visualizations
    for i, entry in enumerate(entries):
        cid, tile = entry["cems_id"], entry["tile"]
        print(f"\n[{i+1}/{len(entries)}] {cid} / tile {tile}")

        save_path = None
        if args.save:
            save_path = os.path.join(args.save_dir, cid, f"{tile}_before_vs_during.png")

        visualize_pair(entry, metrics, save_path=save_path, show=not args.save)

    if args.save:
        print(f"\nAll visualizations saved to: {args.save_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
