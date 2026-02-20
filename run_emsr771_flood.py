"""
EMSR771 Flood Detection using ai4g-flood model (Microsoft, Nature Comm. 2025)
==============================================================================
Runs the ai4g SAR change-detection flood model on SenForFlood EMSR771 tiles:
  - Sentinel-1 before-flood vs during-flood (VV + VH bands)
  - U-Net (MobileNetV2) inference on change-detection features
  - Compares predictions against ground-truth flood masks
  - Generates per-tile overlays + full mosaic summary

Data Layout (SenForFlood EMSR771):
  Band 1 = VV  (already in dB)
  Band 2 = VH  (already in dB)
  Band 3 = VV/VH ratio
  Band 4 = orbit/timestamp constant
"""

import os
import sys
import time
import pathlib
import glob
import numpy as np
import rasterio
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, jaccard_score,
)

# Fix: checkpoint saved on Linux with PosixPath – patch for Windows
pathlib.PosixPath = pathlib.WindowsPath

# ── Add ai4g-flood source to path ──
AI4G_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ai4g-flood")
sys.path.insert(0, os.path.join(AI4G_ROOT, "src"))

from utils.image_processing import (
    pad_to_nearest, create_patches,
    reconstruct_image_from_patches, apply_buffer,
)
from utils.model import load_model

# ═══════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════
_BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(AI4G_ROOT, "models", "ai4g_sar_model.ckpt")
DATA_ROOT   = os.path.join(_BASE_DIR, "DATA", "CEMS", "EMSR771")
OUTPUT_DIR  = os.path.join(_BASE_DIR, "OUTPUTS", "EMSR771_ai4g")

BEFORE_DIR  = os.path.join(DATA_ROOT, "s1_before_flood")
DURING_DIR  = os.path.join(DATA_ROOT, "s1_during_flood")
MASK_DIR    = os.path.join(DATA_ROOT, "flood_mask")

INPUT_SIZE   = 128
BUFFER_SIZE  = 4       # pixels dilation around flood detections

# Change-detection thresholds (tuned for dB-scaled 0-255 range)
VV_THRESHOLD     = 100
VH_THRESHOLD     = 90
DELTA_AMPLITUDE  = 10
VV_MIN_THRESHOLD = 75
VH_MIN_THRESHOLD = 70


# ═══════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════

def db_to_scaled(db_values):
    """
    Convert dB values to the 0-255 scale that ai4g-flood expects.
    The model's db_scale() does: 10*log10(linear) * 2 + 135, clipped 0-255.
    Since SenForFlood data is already in dB, we just apply: dB * 2 + 135.
    """
    scaled = db_values * 2.0 + 135.0
    return np.clip(np.nan_to_num(scaled, nan=0.0), 0, 255)


def read_s1_tile(path):
    """Read SenForFlood S1 tile → VV_scaled, VH_scaled, crs, transform."""
    with rasterio.open(path) as src:
        vv_db = src.read(1)   # Band 1 = VV in dB
        vh_db = src.read(2)   # Band 2 = VH in dB
        crs = src.crs
        transform = src.transform
    vv = db_to_scaled(vv_db)
    vh = db_to_scaled(vh_db)
    return vv, vh, crs, transform


def calculate_flood_change(vv_pre, vh_pre, vv_post, vh_post):
    """Compute 2-channel change map: amplitude drop = potential flood."""
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


def run_inference(model, flood_change, device, target_shape):
    """Run U-Net inference on change-detection patches → binary flood mask."""
    flood_padded = pad_to_nearest(flood_change, INPUT_SIZE, [0, 1])
    patches = create_patches(flood_padded, (INPUT_SIZE, INPUT_SIZE), INPUT_SIZE)

    predictions = []
    with torch.no_grad():
        for i in range(0, len(patches), 512):
            batch = patches[i : i + 512]
            if not batch:
                continue
            tensor = torch.from_numpy(np.array(batch)).float().to(device)
            output = model(tensor)
            _, pred = torch.max(output, 1)
            pred = (pred * 255).to(torch.int)
            # Zero out predictions where there was no change signal
            pred[(tensor[:, 0] == 0) & (tensor[:, 1] == 0)] = 0
            predictions.extend(pred.cpu().numpy())

    pred_image, _ = reconstruct_image_from_patches(
        predictions, flood_padded.shape[:2], (INPUT_SIZE, INPUT_SIZE), INPUT_SIZE
    )
    pred_image = pred_image[: target_shape[0], : target_shape[1]]

    if BUFFER_SIZE > 0:
        pred_image = apply_buffer(pred_image, BUFFER_SIZE)

    return pred_image


def read_ground_truth(path):
    """Read flood mask: 0=no flood, 1=flood, 2=permanent water."""
    with rasterio.open(path) as src:
        mask = src.read(1)
    return mask


def compute_metrics(pred_binary, gt_binary):
    """Compute classification metrics between prediction and ground truth."""
    p = pred_binary.flatten()
    g = gt_binary.flatten()
    # Only evaluate where gt is valid (not permanent water = 2 in original)
    metrics = {
        "accuracy":  accuracy_score(g, p),
        "precision": precision_score(g, p, zero_division=0),
        "recall":    recall_score(g, p, zero_division=0),
        "f1":        f1_score(g, p, zero_division=0),
        "iou":       jaccard_score(g, p, zero_division=0),
    }
    cm = confusion_matrix(g, p, labels=[0, 1])
    metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"] = cm.ravel()
    return metrics


def save_tile_overlay(vv_post, pred_mask, gt_mask, flood_change, tile_id, out_dir):
    """Save a 4-panel visualization for a single tile."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1) Post-event VV
    axes[0, 0].imshow(vv_post, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title("During-Flood VV (Sentinel-1)", fontsize=11)

    # 2) Change detection map
    change_vis = np.zeros((*flood_change.shape[:2], 3), dtype=np.uint8)
    change_vis[flood_change[:, :, 0] > 0] = [255, 100, 0]   # Orange = VV change
    change_vis[flood_change[:, :, 1] > 0] = [255, 200, 0]   # Yellow = VH change
    both = (flood_change[:, :, 0] > 0) & (flood_change[:, :, 1] > 0)
    change_vis[both] = [255, 0, 0]                           # Red = both
    axes[0, 1].imshow(vv_post, cmap="gray", vmin=0, vmax=255, alpha=0.5)
    axes[0, 1].imshow(change_vis, alpha=0.6)
    axes[0, 1].set_title("Change Detection Map", fontsize=11)

    # 3) Model prediction
    pred_vis = np.zeros((*pred_mask.shape, 4), dtype=np.float32)
    pred_vis[pred_mask == 255] = [0, 0.47, 1.0, 0.8]  # Blue
    axes[0, 2].imshow(vv_post, cmap="gray", vmin=0, vmax=255)
    axes[0, 2].imshow(pred_vis)
    axes[0, 2].set_title("Model Prediction (Blue = Flood)", fontsize=11)

    # 4) Ground truth
    gt_vis = np.zeros((*gt_mask.shape, 4), dtype=np.float32)
    gt_vis[gt_mask == 1] = [1.0, 0.2, 0.2, 0.8]   # Red = flood
    gt_vis[gt_mask == 2] = [0.5, 0.5, 0.5, 0.5]   # Gray = permanent water
    axes[1, 0].imshow(vv_post, cmap="gray", vmin=0, vmax=255)
    axes[1, 0].imshow(gt_vis)
    axes[1, 0].set_title("Ground Truth (Red=Flood, Gray=Perm. Water)", fontsize=11)

    # 5) Overlay: Prediction vs Ground Truth
    overlay = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
    gt_flood = (gt_mask == 1)
    pred_flood = (pred_mask == 255)
    tp = gt_flood & pred_flood
    fp = ~gt_flood & pred_flood
    fn = gt_flood & ~pred_flood
    overlay[tp] = [0, 200, 0]     # Green  = True Positive
    overlay[fp] = [255, 0, 0]     # Red    = False Positive
    overlay[fn] = [255, 165, 0]   # Orange = False Negative
    axes[1, 1].imshow(vv_post, cmap="gray", vmin=0, vmax=255, alpha=0.4)
    axes[1, 1].imshow(overlay, alpha=0.7)
    legend_patches = [
        Patch(facecolor=[0, 0.78, 0], label="True Positive"),
        Patch(facecolor=[1, 0, 0], label="False Positive"),
        Patch(facecolor=[1, 0.65, 0], label="False Negative"),
    ]
    axes[1, 1].legend(handles=legend_patches, loc="lower right", fontsize=8)
    axes[1, 1].set_title("Prediction vs Ground Truth", fontsize=11)

    # 6) Side-by-side flood extents
    combined = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
    combined[gt_flood] = [255, 80, 80]     # Red = GT flood
    combined[pred_flood] = [80, 80, 255]   # Blue = predicted flood
    combined[tp] = [180, 0, 255]           # Purple = overlap
    axes[1, 2].imshow(combined)
    legend2 = [
        Patch(facecolor=[1, 0.31, 0.31], label="GT Flood Only"),
        Patch(facecolor=[0.31, 0.31, 1], label="Predicted Only"),
        Patch(facecolor=[0.71, 0, 1], label="Both (Overlap)"),
    ]
    axes[1, 2].legend(handles=legend2, loc="lower right", fontsize=8)
    axes[1, 2].set_title("Flood Extent Comparison", fontsize=11)

    for ax in axes.flat:
        ax.axis("off")

    plt.suptitle(f"EMSR771 Tile {tile_id} — ai4g-flood SAR Flood Detection", fontsize=14, y=1.01)
    plt.tight_layout()
    path = os.path.join(out_dir, f"tile_{tile_id}_analysis.jpg")
    fig.savefig(path, dpi=120, bbox_inches="tight", format="jpg")
    plt.close(fig)
    return path


def save_prediction_geotiff(pred_mask, crs, transform, out_path):
    """Save flood prediction as GeoTIFF."""
    save_arr = pred_mask.copy().astype(np.float32)
    save_arr[save_arr == 0] = np.nan
    with rasterio.open(
        out_path, "w", driver="GTiff",
        height=save_arr.shape[0], width=save_arr.shape[1],
        count=1, dtype="float32", crs=crs, transform=transform,
        compress="lzw", nodata=np.nan,
    ) as dst:
        dst.write(save_arr, 1)


# ═══════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  EMSR771 Flood Detection — ai4g-flood (SAR Change-Detection Model)")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tiles_dir = os.path.join(OUTPUT_DIR, "tiles")
    geotiff_dir = os.path.join(OUTPUT_DIR, "geotiffs")
    os.makedirs(tiles_dir, exist_ok=True)
    os.makedirs(geotiff_dir, exist_ok=True)

    # ── Discover tiles ──
    before_files = sorted(glob.glob(os.path.join(BEFORE_DIR, "*_s1_before_flood.tif")))
    during_files = sorted(glob.glob(os.path.join(DURING_DIR, "*_s1_during_flood.tif")))
    mask_files   = sorted(glob.glob(os.path.join(MASK_DIR, "*_flood_mask.tif")))

    tile_ids = [os.path.basename(f).split("_")[0] for f in before_files]
    n_tiles = len(tile_ids)
    print(f"\n  Found {n_tiles} tile pairs in EMSR771")
    print(f"  Model : {MODEL_PATH}")
    print(f"  Output: {OUTPUT_DIR}\n")

    # ── Load model ──
    print("[1/3] Loading ai4g-flood model (U-Net / MobileNetV2) ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    model = load_model(MODEL_PATH, device, in_channels=2, n_classes=2)
    model.eval()
    print("  Model loaded ✓\n")

    # ── Process each tile ──
    print("[2/3] Running flood detection on all tiles ...\n")
    all_metrics = []
    all_pred_pixels = 0
    all_gt_pixels = 0
    t_start = time.time()

    for idx, tile_id in enumerate(tile_ids):
        before_path = before_files[idx]
        during_path = during_files[idx]
        mask_path   = mask_files[idx] if idx < len(mask_files) else None

        print(f"  [{idx+1}/{n_tiles}] Tile {tile_id} ...", end=" ", flush=True)
        t0 = time.time()

        # Read S1 data
        vv_pre, vh_pre, _, _ = read_s1_tile(before_path)
        vv_post, vh_post, crs, transform = read_s1_tile(during_path)

        # Change detection
        flood_change = calculate_flood_change(vv_pre, vh_pre, vv_post, vh_post)

        # Model inference
        pred_mask = run_inference(model, flood_change, device, vv_post.shape)

        n_pred = (pred_mask == 255).sum()
        all_pred_pixels += n_pred
        elapsed = time.time() - t0

        # Save GeoTIFF prediction
        tif_path = os.path.join(geotiff_dir, f"tile_{tile_id}_flood_pred.tif")
        save_prediction_geotiff(pred_mask, crs, transform, tif_path)

        # Evaluate against ground truth if available
        if mask_path and os.path.exists(mask_path):
            gt_mask = read_ground_truth(mask_path)
            # Binarize: flood only (class 1); ignore permanent water (class 2)
            gt_binary = (gt_mask == 1).astype(int)
            pred_binary = (pred_mask == 255).astype(int)

            # Also compute including permanent water as "flood"
            gt_binary_with_pw = ((gt_mask == 1) | (gt_mask == 2)).astype(int)

            n_gt = gt_binary.sum()
            all_gt_pixels += n_gt

            metrics = compute_metrics(pred_binary, gt_binary)
            metrics["tile_id"] = tile_id
            metrics["n_pred"] = int(n_pred)
            metrics["n_gt"] = int(n_gt)
            metrics["time_s"] = elapsed
            all_metrics.append(metrics)

            # Save overlay
            overlay_path = save_tile_overlay(
                vv_post, pred_mask, gt_mask, flood_change, tile_id, tiles_dir
            )

            print(f"F1={metrics['f1']:.3f}  IoU={metrics['iou']:.3f}  "
                  f"Prec={metrics['precision']:.3f}  Rec={metrics['recall']:.3f}  "
                  f"Pred={n_pred}  GT={n_gt}  ({elapsed:.1f}s)")
        else:
            print(f"Pred={n_pred} pixels  ({elapsed:.1f}s)  [no GT]")

    total_time = time.time() - t_start

    # ── Summary ──
    print(f"\n[3/3] Generating summary ...\n")

    if all_metrics:
        # Aggregate metrics
        avg = {k: np.mean([m[k] for m in all_metrics])
               for k in ["accuracy", "precision", "recall", "f1", "iou"]}

        # Per-tile metrics table
        print("  " + "─" * 90)
        print(f"  {'Tile':>6} │ {'Acc':>6} │ {'Prec':>6} │ {'Recall':>6} │ "
              f"{'F1':>6} │ {'IoU':>6} │ {'TP':>6} │ {'FP':>6} │ {'FN':>6} │ {'Pred':>6} │ {'GT':>6}")
        print("  " + "─" * 90)
        for m in all_metrics:
            print(f"  {m['tile_id']:>6} │ {m['accuracy']:.4f} │ {m['precision']:.4f} │ "
                  f"{m['recall']:.4f} │ {m['f1']:.4f} │ {m['iou']:.4f} │ "
                  f"{m['tp']:>6} │ {m['fp']:>6} │ {m['fn']:>6} │ "
                  f"{m['n_pred']:>6} │ {m['n_gt']:>6}")
        print("  " + "─" * 90)
        print(f"  {'AVG':>6} │ {avg['accuracy']:.4f} │ {avg['precision']:.4f} │ "
              f"{avg['recall']:.4f} │ {avg['f1']:.4f} │ {avg['iou']:.4f} │")
        print("  " + "─" * 90)

        # Save summary metrics to CSV
        csv_path = os.path.join(OUTPUT_DIR, "metrics_summary.csv")
        with open(csv_path, "w") as f:
            header = "tile_id,accuracy,precision,recall,f1,iou,tp,fp,fn,tn,n_pred,n_gt,time_s"
            f.write(header + "\n")
            for m in all_metrics:
                f.write(f"{m['tile_id']},{m['accuracy']:.4f},{m['precision']:.4f},"
                        f"{m['recall']:.4f},{m['f1']:.4f},{m['iou']:.4f},"
                        f"{m['tp']},{m['fp']},{m['fn']},{m['tn']},"
                        f"{m['n_pred']},{m['n_gt']},{m['time_s']:.2f}\n")
            # Average row
            f.write(f"AVERAGE,{avg['accuracy']:.4f},{avg['precision']:.4f},"
                    f"{avg['recall']:.4f},{avg['f1']:.4f},{avg['iou']:.4f},,,,,,\n")
        print(f"\n  Metrics CSV → {csv_path}")

        # ── Full mosaic summary plot ──
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Bar chart of F1 per tile
        tile_labels = [m["tile_id"] for m in all_metrics]
        f1_vals = [m["f1"] for m in all_metrics]
        colors = ["#2ecc71" if f >= 0.3 else "#e74c3c" for f in f1_vals]
        axes[0, 0].barh(tile_labels, f1_vals, color=colors)
        axes[0, 0].axvline(avg["f1"], color="navy", linestyle="--", label=f'Avg F1={avg["f1"]:.3f}')
        axes[0, 0].set_xlabel("F1 Score")
        axes[0, 0].set_title("F1 Score per Tile")
        axes[0, 0].legend()
        axes[0, 0].set_xlim(0, 1)

        # Precision-Recall scatter
        precs = [m["precision"] for m in all_metrics]
        recs = [m["recall"] for m in all_metrics]
        axes[0, 1].scatter(recs, precs, c=f1_vals, cmap="RdYlGn", s=80, edgecolors="k", vmin=0, vmax=1)
        axes[0, 1].set_xlabel("Recall")
        axes[0, 1].set_ylabel("Precision")
        axes[0, 1].set_title("Precision vs Recall (color=F1)")
        axes[0, 1].set_xlim(-0.05, 1.05)
        axes[0, 1].set_ylim(-0.05, 1.05)
        axes[0, 1].plot([0, 1], [0, 1], "k--", alpha=0.2)

        # Aggregated confusion matrix
        total_cm = np.array([[sum(m["tn"] for m in all_metrics), sum(m["fp"] for m in all_metrics)],
                             [sum(m["fn"] for m in all_metrics), sum(m["tp"] for m in all_metrics)]])
        im = axes[1, 0].imshow(total_cm, cmap="Blues")
        for (i, j), val in np.ndenumerate(total_cm):
            axes[1, 0].text(j, i, f"{val:,}", ha="center", va="center",
                            fontsize=14, color="white" if val > total_cm.max()/2 else "black")
        axes[1, 0].set_xticks([0, 1]); axes[1, 0].set_xticklabels(["No Flood", "Flood"])
        axes[1, 0].set_yticks([0, 1]); axes[1, 0].set_yticklabels(["No Flood", "Flood"])
        axes[1, 0].set_xlabel("Predicted")
        axes[1, 0].set_ylabel("Actual")
        axes[1, 0].set_title("Aggregate Confusion Matrix")

        # Summary text
        axes[1, 1].axis("off")
        summary_text = (
            f"EMSR771 Flood Detection Summary\n"
            f"{'─' * 40}\n"
            f"Model: ai4g-flood (SAR Change-Detection)\n"
            f"Architecture: U-Net (MobileNetV2 encoder)\n"
            f"Input: Sentinel-1 VV+VH (before vs during)\n\n"
            f"Tiles processed:  {n_tiles}\n"
            f"Total time:       {total_time:.1f}s\n"
            f"Time/tile:        {total_time/n_tiles:.1f}s\n\n"
            f"Avg Accuracy:     {avg['accuracy']:.4f}\n"
            f"Avg Precision:    {avg['precision']:.4f}\n"
            f"Avg Recall:       {avg['recall']:.4f}\n"
            f"Avg F1 Score:     {avg['f1']:.4f}\n"
            f"Avg IoU:          {avg['iou']:.4f}\n\n"
            f"Total Pred Flood: {all_pred_pixels:,} px\n"
            f"Total GT Flood:   {all_gt_pixels:,} px\n"
            f"Buffer size:      {BUFFER_SIZE} px\n"
            f"Device:           {device}"
        )
        axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                        fontsize=11, verticalalignment="top", fontfamily="monospace",
                        bbox=dict(facecolor="lightyellow", alpha=0.8, edgecolor="gray"))

        plt.suptitle("EMSR771 — ai4g-flood Flood Detection Results", fontsize=15, fontweight="bold")
        plt.tight_layout()
        summary_path = os.path.join(OUTPUT_DIR, "EMSR771_summary.jpg")
        fig.savefig(summary_path, dpi=120, bbox_inches="tight", format="jpg")
        plt.close(fig)
        print(f"  Summary plot → {summary_path}")

    # ── Final report ──
    print("\n" + "=" * 70)
    print("  COMPLETE")
    print("=" * 70)
    print(f"""
  Output directory: {OUTPUT_DIR}
    ├── tiles/           → Per-tile 6-panel analysis JPGs
    ├── geotiffs/        → Flood prediction GeoTIFFs
    ├── metrics_summary.csv
    └── EMSR771_summary.jpg

  Total tiles:  {n_tiles}
  Total time:   {total_time:.1f}s
  Flood pixels: {all_pred_pixels:,}
""")


if __name__ == "__main__":
    main()
