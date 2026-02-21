"""
Temporal Fire Detection — Before / During / After
===================================================
Uses multi-date HLS imagery of the same MGRS location to show fire
progression over time, combining Prithvi burn-scar model predictions
with spectral analysis (NBR, dNBR).

For each MGRS location with ≥2 dates, it finds the best temporal
triplet: BEFORE (lowest burn) → DURING (highest burn) → AFTER (recovery
or post-fire, if a 3rd date exists).

Per-location 6-panel output:
    Top-Left:     BEFORE — RGB
    Top-Center:   DURING — RGB
    Top-Right:    AFTER — RGB  (or DURING False-Color if only 2 dates)
    Bottom-Left:  BEFORE NBR heatmap
    Bottom-Center: dNBR (BEFORE→DURING) severity map
    Bottom-Right:  Burn change overlay (green=unburnt, orange=new burn,
                   red=severe burn, blue=recovered)

Also produces:
    - Per-location before/during/after accuracy comparison
    - Summary dashboard across all locations

Usage:
    python fire_temporal_detection.py
    python fire_temporal_detection.py --splits validation
    python fire_temporal_detection.py --max_locations 5
    python fire_temporal_detection.py --device cuda
"""

import argparse
import collections
import gc
import glob
import math
import os
import pathlib
import re
import sys
import time

import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
from scipy.ndimage import binary_opening, binary_closing

# Fix PosixPath on Windows
pathlib.PosixPath = pathlib.WindowsPath


# ═══════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════
PIXEL_RES_M = 30.0
B_BLUE, B_GREEN, B_RED, B_NIR, B_SWIR1, B_SWIR2 = 0, 1, 2, 3, 4, 5

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULTS = dict(
    data_root   = os.path.join(_BASE_DIR, "DATA", "Fire"),
    model_path  = os.path.join(_BASE_DIR, "Model", "burn_scars_Prithvi_100M.pth"),
    output_root = os.path.join(_BASE_DIR, "OUTPUTS", "fire_temporal"),
    tile_size   = 224,
    stride      = 112,
    img_suffix  = "_merged.tif",
    mask_suffix = ".mask.tif",
    nodata      = -9999,
    nodata_replace = 0,
    means = [0.033349706741586264, 0.05701185520536176, 0.05889748132001316,
             0.2323245113436119, 0.1972854853760658, 0.11944914225186566],
    stds  = [0.02269135568823774, 0.026807560223070237, 0.04004109844362779,
             0.07791732423672691, 0.08708738838140137, 0.07241979477437814],
    bands = [0, 1, 2, 3, 4, 5],
)


# ═══════════════════════════════════════════════════════════
#  MODEL  (import from fire_inference_pipeline)
# ═══════════════════════════════════════════════════════════

def _load_model(model_path, device):
    """Load the Prithvi burn-scar model."""
    # Import from sibling file
    sys.path.insert(0, _BASE_DIR)
    from fire_inference_pipeline import load_prithvi_burn_scar_model
    return load_prithvi_burn_scar_model(model_path, device)


def _sliding_window_inference(model, image, nodata_mask, device, cfg):
    """Run sliding window inference on a single normalized tile."""
    sys.path.insert(0, _BASE_DIR)
    from fire_inference_pipeline import sliding_window_inference
    return sliding_window_inference(model, image, nodata_mask, device, cfg)


# ═══════════════════════════════════════════════════════════
#  DATA I/O
# ═══════════════════════════════════════════════════════════

def read_hls_normalized(path, cfg):
    """Read HLS tile → normalized (C,H,W), nodata_mask, crs, transform."""
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)
        crs, transform = src.crs, src.transform
        nodata_val = src.nodata if src.nodata is not None else cfg["nodata"]

    bands = cfg["bands"]
    data = data[bands]
    nodata_mask = np.any(np.isclose(data, nodata_val) | np.isnan(data), axis=0)
    data[np.isclose(data, nodata_val)] = cfg["nodata_replace"]
    data = np.nan_to_num(data, nan=cfg["nodata_replace"])

    means = np.array(cfg["means"], dtype=np.float32).reshape(-1, 1, 1)
    stds  = np.array(cfg["stds"],  dtype=np.float32).reshape(-1, 1, 1)
    data = (data - means) / (stds + 1e-10)
    return data, nodata_mask, crs, transform


def read_hls_raw(path):
    """Read raw HLS bands (no normalization) for spectral indices."""
    with rasterio.open(path) as src:
        return src.read().astype(np.float32)


def _normalize_band(band):
    """Percentile-clip + min-max → [0,1] for visualization."""
    valid = band[~np.isnan(band) & (band > 0)]
    if valid.size == 0:
        return np.zeros_like(band, dtype=np.float32)
    p2, p98 = np.percentile(valid, [2, 98])
    band = np.clip(band, p2, p98).astype(np.float32)
    lo, hi = band.min(), band.max()
    return np.zeros_like(band, dtype=np.float32) if (hi - lo) < 1e-6 else (band - lo) / (hi - lo)


def make_rgb(raw_data):
    """Raw 6-band → RGB [H,W,3]."""
    return np.stack([_normalize_band(raw_data[B_RED]),
                     _normalize_band(raw_data[B_GREEN]),
                     _normalize_band(raw_data[B_BLUE])], axis=-1)


def make_false_color(raw_data):
    """Raw 6-band → False-color [H,W,3] (SWIR2/NIR/Red)."""
    return np.stack([_normalize_band(raw_data[B_SWIR2]),
                     _normalize_band(raw_data[B_NIR]),
                     _normalize_band(raw_data[B_RED])], axis=-1)


def compute_nbr(raw_data):
    """NBR = (NIR - SWIR2) / (NIR + SWIR2). Low → burn scar."""
    nir   = raw_data[B_NIR]
    swir2 = raw_data[B_SWIR2]
    denom = nir + swir2
    denom[denom == 0] = 1e-10
    return np.clip((nir - swir2) / denom, -1, 1)


def compute_dnbr(nbr_before, nbr_after):
    """dNBR = NBR_before - NBR_after. Positive → burn damage."""
    return nbr_before - nbr_after


def classify_burn_severity(dnbr):
    """USGS dNBR severity classes."""
    severity = np.full(dnbr.shape, 0, dtype=np.int8)
    severity[dnbr < -0.25] = -2  # High post-fire regrowth
    severity[(dnbr >= -0.25) & (dnbr < -0.1)] = -1  # Low regrowth
    severity[(dnbr >= -0.1) & (dnbr < 0.1)]   =  0  # Unburned
    severity[(dnbr >= 0.1)  & (dnbr < 0.27)]  =  1  # Low severity
    severity[(dnbr >= 0.27) & (dnbr < 0.44)]  =  2  # Moderate-low
    severity[(dnbr >= 0.44) & (dnbr < 0.66)]  =  3  # Moderate-high
    severity[dnbr >= 0.66]                     =  4  # High severity
    return severity


def load_gt_mask(path):
    """Load GT → (burn_bool, raw)."""
    with rasterio.open(path) as src:
        raw = src.read(1)
    return (raw == 1), raw


def clean(mask, k=3):
    return binary_opening(mask, structure=np.ones((k, k)))


def px_to_km2(n):
    return n * (PIXEL_RES_M ** 2) / 1e6


def doy_to_date_str(year, doy):
    """Convert year + day-of-year to readable date string."""
    from datetime import datetime, timedelta
    dt = datetime(year, 1, 1) + timedelta(days=doy - 1)
    return dt.strftime("%Y-%m-%d")


# ═══════════════════════════════════════════════════════════
#  TEMPORAL TRIPLET DISCOVERY
# ═══════════════════════════════════════════════════════════

def discover_temporal_locations(data_root, splits, cfg):
    """
    Scan Fire data for MGRS locations with multiple dates.
    For each, find the best BEFORE / DURING / AFTER triplet.

    Returns list of dicts:
        mgrs, before, during, after (each is a tile-info dict or None)
    """
    pattern = re.compile(
        r"(subsetted_512x512_HLS\.S30\.(T\w+)\.(\d{4})(\d{3})\.\S+?)"
        + re.escape(cfg["img_suffix"])
    )

    tiles = collections.defaultdict(list)
    for split in splits:
        split_dir = os.path.join(data_root, split)
        if not os.path.isdir(split_dir):
            continue
        for f in sorted(os.listdir(split_dir)):
            m = pattern.search(f)
            if m:
                tile_id = m.group(1)
                mgrs    = m.group(2)
                year    = int(m.group(3))
                doy     = int(m.group(4))
                mask_f  = tile_id + cfg["mask_suffix"]
                mask_path = os.path.join(split_dir, mask_f)

                burn_pct = 0.0
                if os.path.exists(mask_path):
                    with rasterio.open(mask_path) as src:
                        gt = src.read(1)
                    burn_pct = (gt == 1).sum() / gt.size * 100

                tiles[mgrs].append(dict(
                    mgrs=mgrs, year=year, doy=doy, split=split,
                    tile_id=tile_id, burn_pct=round(burn_pct, 1),
                    img_path=os.path.join(split_dir, f),
                    mask_path=mask_path if os.path.exists(mask_path) else None,
                    date_str=doy_to_date_str(year, doy),
                ))

    # Build the best triplet for each MGRS with ≥2 dates
    locations = []
    for mgrs in sorted(tiles):
        entries = sorted(tiles[mgrs], key=lambda x: (x["year"], x["doy"]))
        if len(entries) < 2:
            continue

        # "DURING" = tile with highest burn percentage
        during = max(entries, key=lambda x: x["burn_pct"])
        if during["burn_pct"] < 5:
            continue  # skip locations with no significant burn

        # "BEFORE" = tile with lowest burn % that is chronologically before DURING
        before_candidates = [
            e for e in entries
            if (e["year"], e["doy"]) < (during["year"], during["doy"])
        ]
        # Also consider tiles from AFTER with low burn (fire may recur)
        if not before_candidates:
            # Use lowest-burn tile that isn't DURING
            before_candidates = [e for e in entries if e is not during]
            if not before_candidates:
                continue

        before = min(before_candidates, key=lambda x: x["burn_pct"])

        # "AFTER" = tile chronologically after DURING with lower burn
        after_candidates = [
            e for e in entries
            if (e["year"], e["doy"]) > (during["year"], during["doy"])
               and e is not before
        ]
        after = None
        if after_candidates:
            after = min(after_candidates, key=lambda x: x["burn_pct"])

        locations.append(dict(
            mgrs=mgrs,
            before=before,
            during=during,
            after=after,
            n_dates=len(entries),
            burn_delta=during["burn_pct"] - before["burn_pct"],
        ))

    # Sort by burn_delta (most dramatic fire first)
    locations.sort(key=lambda x: -x["burn_delta"])
    return locations


# ═══════════════════════════════════════════════════════════
#  VISUALIZATION:  6-Panel Temporal Fire Analysis
# ═══════════════════════════════════════════════════════════

# Custom colormaps
_NBR_CMAP = LinearSegmentedColormap.from_list(
    "nbr", [(0.7, 0, 0), (1, 0.6, 0), (1, 1, 0.3), (0.2, 0.8, 0.2), (0, 0.5, 0)], N=256,
)
_SEVERITY_COLORS = [
    "#1a9850",   # -2: High regrowth (green)
    "#91cf60",   # -1: Low regrowth
    "#d9ef8b",   #  0: Unburned
    "#fee08b",   #  1: Low severity
    "#fc8d59",   #  2: Moderate-low
    "#d73027",   #  3: Moderate-high
    "#a50026",   #  4: High severity (dark red)
]
_SEVERITY_CMAP = LinearSegmentedColormap.from_list("severity", _SEVERITY_COLORS, N=7)
_SEVERITY_LABELS = [
    "High Regrowth", "Low Regrowth", "Unburned",
    "Low Severity", "Moderate-Low", "Moderate-High", "High Severity"
]


def make_6panel(loc, before_data, during_data, after_data,
                before_pred, during_pred, after_pred, out_dir):
    """
    6-panel figure:
      Top:    BEFORE RGB  |  DURING RGB  |  AFTER RGB (or DURING false-color)
      Bottom: BEFORE NBR  |  dNBR severity (Before→During)  |  Burn change overlay
    """
    mgrs = loc["mgrs"]
    bef  = loc["before"]
    dur  = loc["during"]
    aft  = loc["after"]

    # Compute images
    before_rgb = make_rgb(before_data)
    during_rgb = make_rgb(during_data)
    before_nbr = compute_nbr(before_data)
    during_nbr = compute_nbr(during_data)
    dnbr = compute_dnbr(before_nbr, during_nbr)
    severity = classify_burn_severity(dnbr)

    has_after = after_data is not None
    if has_after:
        after_rgb = make_rgb(after_data)
        after_nbr = compute_nbr(after_data)
        dnbr_recovery = compute_dnbr(during_nbr, after_nbr)  # negative = recovery
    else:
        after_rgb = make_false_color(during_data)

    # Build figure
    fig, axes = plt.subplots(2, 3, figsize=(24, 14))

    # ── TOP LEFT: Before RGB ──
    axes[0, 0].imshow(before_rgb)
    bef_burn_str = f"Burn: {bef['burn_pct']:.1f}%"
    axes[0, 0].set_title(
        f"BEFORE FIRE — {bef['date_str']}\n{bef_burn_str}  |  {bef['split']}",
        fontsize=12, fontweight="bold", color="#2e7d32",
    )

    # ── TOP CENTER: During RGB + burn overlay ──
    axes[0, 1].imshow(during_rgb)
    ov = np.zeros((*during_pred.shape, 4), dtype=np.float32)
    ov[during_pred > 0] = [1.0, 0.3, 0.0, 0.45]
    axes[0, 1].imshow(ov)
    dur_km2 = px_to_km2((during_pred > 0).sum())
    axes[0, 1].set_title(
        f"DURING FIRE — {dur['date_str']}\n"
        f"Burn: {dur['burn_pct']:.1f}%  |  Pred: {dur_km2:.2f} km²",
        fontsize=12, fontweight="bold", color="#c62828",
    )

    # ── TOP RIGHT: After RGB (or false-color) ──
    axes[0, 2].imshow(after_rgb)
    if has_after:
        aft_burn_str = f"Burn: {aft['burn_pct']:.1f}%"
        # Overlay recovered areas in green
        if after_pred is not None:
            recovered = (during_pred > 0) & (after_pred == 0)
            still_burn = (during_pred > 0) & (after_pred > 0)
            ov_a = np.zeros((*after_pred.shape, 4), dtype=np.float32)
            ov_a[recovered]  = [0.0, 0.8, 0.2, 0.4]   # green = recovered
            ov_a[still_burn] = [1.0, 0.3, 0.0, 0.35]   # orange = still burnt
            axes[0, 2].imshow(ov_a)
        axes[0, 2].set_title(
            f"AFTER FIRE — {aft['date_str']}\n{aft_burn_str}  |  {aft['split']}",
            fontsize=12, fontweight="bold", color="#1565c0",
        )
    else:
        axes[0, 2].set_title(
            f"DURING — False-Color (SWIR2/NIR/Red)\nBurn scars → bright magenta",
            fontsize=12, fontweight="bold", color="#c62828",
        )

    # ── BOTTOM LEFT: Before NBR heatmap ──
    im_nbr = axes[1, 0].imshow(before_nbr, cmap=_NBR_CMAP, vmin=-0.5, vmax=0.5)
    axes[1, 0].set_title(
        f"BEFORE — NBR (Normalized Burn Ratio)\nGreen = healthy vegetation  |  Red = burn scar",
        fontsize=11, fontweight="bold",
    )
    plt.colorbar(im_nbr, ax=axes[1, 0], fraction=0.046, pad=0.04, label="NBR")

    # ── BOTTOM CENTER: dNBR severity map ──
    bounds = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm = BoundaryNorm(bounds, _SEVERITY_CMAP.N)
    im_sev = axes[1, 1].imshow(severity, cmap=_SEVERITY_CMAP, norm=norm)
    axes[1, 1].set_title(
        f"dNBR Burn Severity (Before → During)\n"
        f"dNBR = NBR_before − NBR_during  |  USGS classification",
        fontsize=11, fontweight="bold",
    )
    # Add severity legend
    legend_patches = [
        Patch(facecolor=_SEVERITY_COLORS[i], edgecolor="k", linewidth=0.5,
              label=_SEVERITY_LABELS[i])
        for i in range(7)
    ]
    axes[1, 1].legend(handles=legend_patches, loc="lower left", fontsize=7,
                      ncol=2, framealpha=0.9)

    # ── BOTTOM RIGHT: Burn change overlay on DURING RGB ──
    axes[1, 2].imshow(during_rgb, alpha=0.4)
    change_ov = np.zeros((*during_pred.shape, 3), dtype=np.uint8)

    # Categories
    before_burn = before_pred > 0
    during_burn = during_pred > 0

    unchanged_noburn = ~before_burn & ~during_burn  # no fire either time
    pre_existing     = before_burn & during_burn     # burned in both
    new_burn         = ~before_burn & during_burn    # NEW fire!
    recovered_area   = before_burn & ~during_burn    # fire receded

    change_ov[unchanged_noburn] = [34, 139, 34]    # forest green
    change_ov[pre_existing]     = [255, 165, 0]    # orange
    change_ov[new_burn]         = [220, 20, 20]    # red
    change_ov[recovered_area]   = [30, 144, 255]   # blue

    axes[1, 2].imshow(change_ov, alpha=0.75)

    new_km2  = px_to_km2(new_burn.sum())
    pre_km2  = px_to_km2(pre_existing.sum())
    rec_km2  = px_to_km2(recovered_area.sum())
    total_km2 = px_to_km2(during_burn.sum())

    axes[1, 2].set_title(
        f"Burn Change Detection\n"
        f"New: {new_km2:.2f} km²  |  Pre-existing: {pre_km2:.2f} km²  |  "
        f"Recovered: {rec_km2:.2f} km²",
        fontsize=11, fontweight="bold",
    )
    axes[1, 2].legend(handles=[
        Patch(facecolor="#228B22", edgecolor="k", label="Unburned"),
        Patch(facecolor="#FFA500", edgecolor="k", label="Pre-existing burn"),
        Patch(facecolor="#DC1414", edgecolor="k", label="NEW fire"),
        Patch(facecolor="#1E90FF", edgecolor="k", label="Recovered"),
    ], loc="lower left", fontsize=8, framealpha=0.9)

    for ax in axes.flat:
        ax.axis("off")

    # Suptitle
    title = f"Temporal Fire Analysis — MGRS {mgrs}"
    if has_after:
        title += f"\n{bef['date_str']}  →  {dur['date_str']}  →  {aft['date_str']}"
    else:
        title += f"\n{bef['date_str']}  →  {dur['date_str']}"
    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.02)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    path = os.path.join(out_dir, f"{mgrs}_6panel.jpg")
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", format="jpg")
    plt.close(fig)
    gc.collect()
    return path


# ═══════════════════════════════════════════════════════════
#  ACCURACY COMPARISON:  Before vs During vs After
# ═══════════════════════════════════════════════════════════

def make_accuracy_comparison(loc, preds, gts, out_dir):
    """
    3-column accuracy overlay:
      Left:   BEFORE — TP/FP/FN
      Center: DURING — TP/FP/FN
      Right:  AFTER — TP/FP/FN (if available)
    """
    from sklearn.metrics import f1_score, jaccard_score

    phases = [("BEFORE", loc["before"]), ("DURING", loc["during"])]
    if loc["after"] is not None:
        phases.append(("AFTER", loc["after"]))

    n_cols = len(phases)
    fig, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 7))
    if n_cols == 1:
        axes = [axes]

    results = {}

    for col, (phase_name, tile_info) in enumerate(phases):
        pred = preds.get(phase_name)
        gt   = gts.get(phase_name)

        if pred is None or gt is None:
            axes[col].text(0.5, 0.5, f"{phase_name}\nNo data",
                           ha="center", va="center", fontsize=14)
            axes[col].axis("off")
            continue

        pred_b = pred > 0
        gt_b   = gt

        tp = gt_b & pred_b
        fp = ~gt_b & pred_b
        fn = gt_b & ~pred_b

        g = gt_b.flatten().astype(int)
        p = pred_b.flatten().astype(int)
        f1  = f1_score(g, p, zero_division=0)
        iou = jaccard_score(g, p, zero_division=0)

        ov = np.zeros((*pred_b.shape, 3), dtype=np.uint8)
        ov[tp] = [0, 200, 0]
        ov[fp] = [255, 0, 0]
        ov[fn] = [255, 165, 0]

        axes[col].imshow(ov)
        gt_km2   = px_to_km2(gt_b.sum())
        pred_km2 = px_to_km2(pred_b.sum())

        color = {"BEFORE": "#2e7d32", "DURING": "#c62828", "AFTER": "#1565c0"}[phase_name]
        axes[col].set_title(
            f"{phase_name} — {tile_info['date_str']}\n"
            f"F1={f1:.3f}  IoU={iou:.3f}\n"
            f"GT: {gt_km2:.2f} km²  Pred: {pred_km2:.2f} km²",
            fontsize=11, fontweight="bold", color=color,
        )
        axes[col].axis("off")
        results[phase_name] = dict(f1=f1, iou=iou, gt_km2=gt_km2, pred_km2=pred_km2)

    # Legend on last panel
    axes[-1].legend(handles=[
        Patch(facecolor=(0, 0.78, 0), label="True Positive"),
        Patch(facecolor=(1, 0, 0), label="False Positive"),
        Patch(facecolor=(1, 0.65, 0), label="False Negative"),
    ], loc="lower left", fontsize=9, framealpha=0.9)

    fig.suptitle(f"Prediction Accuracy — MGRS {loc['mgrs']}",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, f"{loc['mgrs']}_accuracy.jpg")
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight", format="jpg")
    plt.close(fig)
    gc.collect()
    return path, results


# ═══════════════════════════════════════════════════════════
#  SUMMARY DASHBOARD
# ═══════════════════════════════════════════════════════════

def make_summary_dashboard(all_results, out_dir):
    """Bar chart of F1 per location + summary stats."""
    if not all_results:
        return None

    mgrs_ids = [r["mgrs"] for r in all_results]
    before_f1 = [r.get("BEFORE", {}).get("f1", 0) for r in all_results]
    during_f1 = [r.get("DURING", {}).get("f1", 0) for r in all_results]
    after_f1  = [r.get("AFTER", {}).get("f1", 0) for r in all_results]
    deltas    = [r["burn_delta"] for r in all_results]

    n = len(mgrs_ids)
    fig_h = max(7, min(n * 0.5 + 3, 35))
    fig, axes = plt.subplots(1, 2, figsize=(20, fig_h),
                              gridspec_kw={"width_ratios": [2.5, 1]})

    # Left — grouped bar chart
    y_pos = np.arange(n)
    bar_h = 0.25
    axes[0].barh(y_pos - bar_h, before_f1, bar_h, color="#4caf50", label="Before", edgecolor="k", linewidth=0.3)
    axes[0].barh(y_pos,         during_f1, bar_h, color="#f44336", label="During", edgecolor="k", linewidth=0.3)
    axes[0].barh(y_pos + bar_h, after_f1,  bar_h, color="#2196f3", label="After",  edgecolor="k", linewidth=0.3)

    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(mgrs_ids, fontsize=8)
    axes[0].set_xlabel("F1 Score", fontsize=11)
    axes[0].set_title("F1 Score per Phase per Location", fontsize=13, fontweight="bold")
    axes[0].set_xlim(0, 1.05)
    axes[0].legend(fontsize=10, loc="lower right")
    axes[0].grid(axis="x", alpha=0.3)
    axes[0].invert_yaxis()

    # Right — summary text
    axes[1].axis("off")
    avg_bef = np.mean(before_f1) if before_f1 else 0
    avg_dur = np.mean(during_f1) if during_f1 else 0
    avg_aft = np.mean([f for f in after_f1 if f > 0]) if any(f > 0 for f in after_f1) else 0
    n_with_after = sum(1 for f in after_f1 if f > 0)

    txt = (
        f"Temporal Fire Detection Summary\n"
        f"{'─' * 40}\n"
        f"Locations processed:    {n}\n"
        f"Locations with 3 dates: {n_with_after}\n\n"
        f"Avg F1 (Before):  {avg_bef:.4f}\n"
        f"Avg F1 (During):  {avg_dur:.4f}\n"
        f"Avg F1 (After):   {avg_aft:.4f}\n\n"
        f"Avg burn delta:   {np.mean(deltas):.1f}%\n"
        f"Max burn delta:   {max(deltas):.1f}%\n\n"
        f"Model:  Prithvi 100M\n"
        f"Input:  HLS 6-band (30 m/px)\n"
        f"Method: Temporal dNBR + ViT\n"
    )
    axes[1].text(0.05, 0.95, txt, transform=axes[1].transAxes,
                 fontsize=11, verticalalignment="top", fontfamily="monospace",
                 bbox=dict(facecolor="lightyellow", alpha=0.9, edgecolor="gray"))

    plt.suptitle("Temporal Fire Detection — All Locations",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "temporal_fire_summary.jpg")
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight", format="jpg")
    plt.close(fig)
    gc.collect()
    return path


# ═══════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ═══════════════════════════════════════════════════════════

def run_pipeline(cfg):
    _flush = lambda: sys.stdout.flush()
    print("=" * 72)
    print("  Temporal Fire Detection — Before / During / After")
    print("=" * 72); _flush()

    splits = cfg.get("splits", ["training", "validation"])
    if isinstance(splits, str):
        splits = [splits]

    # ── Discover temporal locations ──
    print("\n[1] Discovering multi-date locations ...")
    locations = discover_temporal_locations(cfg["data_root"], splits, cfg)
    max_loc = cfg.get("_max_locations")
    if max_loc is not None:
        locations = locations[:max_loc]

    n_loc = len(locations)
    n_3date = sum(1 for l in locations if l["after"] is not None)
    print(f"    Found {n_loc} locations ({n_3date} with 3+ dates)")
    if n_loc == 0:
        print("    No suitable multi-date locations found.")
        return

    for loc in locations[:5]:
        a_str = f" → {loc['after']['date_str']} ({loc['after']['burn_pct']}%)" if loc["after"] else ""
        print(f"    {loc['mgrs']}: {loc['before']['date_str']} ({loc['before']['burn_pct']}%) "
              f"→ {loc['during']['date_str']} ({loc['during']['burn_pct']}%){a_str}")
    if n_loc > 5:
        print(f"    ... and {n_loc - 5} more")

    # ── Load model ──
    print(f"\n[2] Loading Prithvi burn-scar model ...")
    device = torch.device(cfg.get("device", "cpu"))
    if cfg.get("device") == "cuda" and not torch.cuda.is_available():
        print("    CUDA not available, falling back to CPU")
        device = torch.device("cpu")
    model = _load_model(cfg["model_path"], device)
    print(f"    Device: {device}   OK\n")

    # ── Process each location ──
    print("[3] Processing locations ...\n")
    out_6panel  = os.path.join(cfg["output_root"], "6panel")
    out_acc     = os.path.join(cfg["output_root"], "accuracy")
    all_results = []
    t_global = time.time()

    for idx, loc in enumerate(locations):
        mgrs = loc["mgrs"]
        t0 = time.time()
        print(f"  +-- [{idx+1}/{n_loc}] MGRS {mgrs}  (delta={loc['burn_delta']:.1f}%)")

        phases = {"BEFORE": loc["before"], "DURING": loc["during"]}
        if loc["after"]:
            phases["AFTER"] = loc["after"]

        raw_data = {}
        preds    = {}
        gts      = {}

        for phase, tinfo in phases.items():
            # Read raw data for visualization
            raw = read_hls_raw(tinfo["img_path"])
            raw_data[phase] = raw

            # Run model inference
            norm_data, nodata_mask, crs, transform = read_hls_normalized(tinfo["img_path"], cfg)
            pred = _sliding_window_inference(model, norm_data, nodata_mask, device, cfg)
            preds[phase] = pred

            # Load GT
            if tinfo["mask_path"] and os.path.exists(tinfo["mask_path"]):
                gt_b, _ = load_gt_mask(tinfo["mask_path"])
                gts[phase] = gt_b

            pred_km2 = px_to_km2((pred > 0).sum())
            print(f"  |    {phase:7s}: {tinfo['date_str']}  "
                  f"GT={tinfo['burn_pct']:5.1f}%  Pred={pred_km2:.2f} km²")

        # ── Generate 6-panel ──
        panel_path = make_6panel(
            loc,
            raw_data["BEFORE"], raw_data["DURING"],
            raw_data.get("AFTER"), preds["BEFORE"], preds["DURING"],
            preds.get("AFTER"), out_6panel,
        )
        print(f"  |    6-panel  → {os.path.basename(panel_path)}"); _flush()

        # ── Generate accuracy comparison ──
        acc_path, acc_results = make_accuracy_comparison(loc, preds, gts, out_acc)
        print(f"  |    Accuracy → {os.path.basename(acc_path)}")

        elapsed = time.time() - t0
        print(f"  +-- {mgrs} done ({elapsed:.1f}s)\n"); _flush()

        result = dict(mgrs=mgrs, burn_delta=loc["burn_delta"])
        result.update(acc_results)
        all_results.append(result)

    total_time = time.time() - t_global

    # ── Summary dashboard ──
    print("[4] Generating summary dashboard ...")
    summary_path = make_summary_dashboard(all_results, cfg["output_root"])
    if summary_path:
        print(f"    Summary → {summary_path}")

    # ── Print results table ──
    print(f"\n  {'─' * 80}")
    print(f"  {'MGRS':<10} | {'Delta':>5} | {'Bef F1':>6} | {'Dur F1':>6} | "
          f"{'Aft F1':>6} | {'Dur IoU':>7}")
    print(f"  {'─' * 80}")
    for r in all_results:
        bef_f1 = r.get("BEFORE", {}).get("f1", 0)
        dur_f1 = r.get("DURING", {}).get("f1", 0)
        aft_f1 = r.get("AFTER", {}).get("f1", 0)
        dur_iou = r.get("DURING", {}).get("iou", 0)
        aft_str = f"{aft_f1:>6.3f}" if aft_f1 > 0 else "   N/A"
        print(f"  {r['mgrs']:<10} | {r['burn_delta']:>4.1f}% | {bef_f1:>6.3f} | "
              f"{dur_f1:>6.3f} | {aft_str} | {dur_iou:>7.3f}")
    print(f"  {'─' * 80}")

    print(f"\n  Total time: {total_time:.1f}s")
    print("\n" + "=" * 72)
    print("  PIPELINE COMPLETE")
    print("=" * 72)
    print(f"""
  Outputs ({cfg['output_root']}/):
    +-- 6panel/       6-panel Before/During/After temporal analysis JPGs
    +-- accuracy/     TP/FP/FN accuracy comparison per phase
    +-- temporal_fire_summary.jpg  (grouped bar chart + stats)
""")
    return all_results


# ═══════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Temporal fire detection: Before / During / After analysis",
    )
    p.add_argument("--data_root", default=DEFAULTS["data_root"])
    p.add_argument("--model_path", default=DEFAULTS["model_path"])
    p.add_argument("--output_root", default=DEFAULTS["output_root"])
    p.add_argument("--splits", nargs="*", default=["training", "validation"])
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--tile_size", type=int, default=DEFAULTS["tile_size"])
    p.add_argument("--stride", type=int, default=DEFAULTS["stride"])
    p.add_argument("--max_locations", type=int, default=None,
                   help="Max locations to process (for quick testing)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = dict(DEFAULTS)
    cfg.update(
        data_root    = args.data_root,
        model_path   = args.model_path,
        output_root  = args.output_root,
        splits       = args.splits,
        device       = args.device,
        tile_size    = args.tile_size,
        stride       = args.stride,
    )
    if args.max_locations is not None:
        cfg["_max_locations"] = args.max_locations
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
