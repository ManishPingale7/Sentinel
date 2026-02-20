"""
Automated Flood Segmentation & Change-Detection Pipeline
=========================================================
Input : SenForFlood CEMS folders (8-band Sentinel-2 before/during images)
        OR Sentinel-1 SAR bitemporal pairs (for DAM-Net)
Model :
    --model prithvi  : Prithvi sen1floods11_Prithvi_100M  (optical, default)
    --model damnet   : Custom DAM-Net Siamese ViT         (SAR, change detection)
Output:
    G:/Sentinel/OUTPUTS/Prithvi/{CEMS_ID}/before/    – segmentation masks
    G:/Sentinel/OUTPUTS/Prithvi/{CEMS_ID}/during/    – segmentation masks
    G:/Sentinel/OUTPUTS/DAMNet/{CEMS_ID}/            – DAM-Net flood masks
    G:/Sentinel/OUTPUTS/Overlays/{CEMS_ID}/          – overlay PNGs
    G:/Sentinel/OUTPUTS/change_detection_report.csv  – area metrics

Usage:
    python flood_pipeline.py                               # Prithvi (default)
    python flood_pipeline.py --model damnet                # DAM-Net SAR model
    python flood_pipeline.py --model damnet --device cpu   # force CPU
    python flood_pipeline.py --skip-inference               # reuse existing masks
"""

import argparse
import csv
import glob
import os
import sys
import time
import traceback

import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving PNGs
import matplotlib.pyplot as plt
from scipy.ndimage import binary_opening

# ── Paths (relative to project root) ────────────────────────────────────
_BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
SENFORFLOOD_ROOT = os.path.join(_BASE_DIR, "DATA", "CEMS")
CONFIG_PATH      = os.path.join(_BASE_DIR, "hls-foundation-os", "configs", "sen1floods11_config.py")
CKPT_PATH        = os.path.join(_BASE_DIR, "model", "sen1floods11_Prithvi_100M.pth")
DAMNET_CKPT      = os.path.join(_BASE_DIR, "OUTPUTS", "DAMNet", "best.pt")
OUTPUT_ROOT      = os.path.join(_BASE_DIR, "OUTPUTS")

# Band mapping: SenForFlood 8-band → first 6 = B2,B3,B4,B8A,B11,B12
SENFORFLOOD_BANDS = [0, 1, 2, 3, 4, 5]

# Pixel resolution (metres) – SenForFlood tiles are 10 m
PIXEL_RES_M = 10.0


# =====================================================================
# 1. DISCOVERY – scan CEMS folders
# =====================================================================

def discover_cems_folders(root):
    """Return list of CEMS sub-folder names that contain S2 before & during."""
    cems_ids = []
    for name in sorted(os.listdir(root)):
        folder = os.path.join(root, name)
        if not os.path.isdir(folder):
            continue
        before = os.path.join(folder, "s2_before_flood")
        during = os.path.join(folder, "s2_during_flood")
        if os.path.isdir(before) and os.path.isdir(during):
            cems_ids.append(name)
        else:
            print(f"  [SKIP] {name}: missing s2_before_flood and/or s2_during_flood")
    return cems_ids


def collect_image_pairs(root, cems_id, ext="tif"):
    """Return list of (before_path, during_path) matched by numeric prefix."""
    before_dir = os.path.join(root, cems_id, "s2_before_flood")
    during_dir = os.path.join(root, cems_id, "s2_during_flood")

    before_files = {
        os.path.basename(f).split("_")[0]: f
        for f in sorted(glob.glob(os.path.join(before_dir, f"*.{ext}")))
    }
    during_files = {
        os.path.basename(f).split("_")[0]: f
        for f in sorted(glob.glob(os.path.join(during_dir, f"*.{ext}")))
    }

    common = sorted(set(before_files) & set(during_files))
    pairs = [(before_files[k], during_files[k]) for k in common]
    return pairs


# =====================================================================
# 2. INFERENCE – run Prithvi segmentation
# =====================================================================

def run_inference(cems_ids, device="cuda", skip=False, max_images=None):
    """Run Prithvi inference on every before/during image.

    Args:
        max_images: If set, only process up to this many tiles per CEMS folder.

    Returns dict  cems_id → {before: {prefix: pred_path}, during: {…}}
    """
    # Lazy-import so the rest of the script can still be tested w/o torch
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "hls-foundation-os"))
    from model_inference import load_model, inference_on_file_list, SENFORFLOOD_BAND_INDICES

    pred_index = {}  # cems_id -> {"before": {prefix: path}, "during": {prefix: path}}

    if not skip:
        print("\n══════════════════════════════════════")
        print("  Loading Prithvi model …")
        print("══════════════════════════════════════")
        model = load_model(CONFIG_PATH, CKPT_PATH, device)

    for cems_id in cems_ids:
        pred_index[cems_id] = {"before": {}, "during": {}}
        for phase in ("before", "during"):
            src_dir = os.path.join(SENFORFLOOD_ROOT, cems_id, f"s2_{phase}_flood")
            out_dir = os.path.join(OUTPUT_ROOT, "Prithvi", cems_id, phase)
            os.makedirs(out_dir, exist_ok=True)

            images = sorted(glob.glob(os.path.join(src_dir, "*.tif")))
            # Limit to max_images if set
            if max_images is not None:
                images = images[:max_images]

            pairs = []
            for img_path in images:
                prefix = os.path.basename(img_path).split("_")[0]
                pred_path = os.path.join(
                    out_dir,
                    os.path.basename(img_path).replace(".tif", "_pred.tif"),
                )
                pred_index[cems_id][phase][prefix] = pred_path
                if skip and os.path.exists(pred_path):
                    continue
                pairs.append((img_path, pred_path))

            if pairs and not skip:
                print(f"\n── {cems_id} / {phase} ({len(pairs)} images) ──")
                inference_on_file_list(
                    model, pairs, bands=SENFORFLOOD_BANDS
                )
            else:
                existing = len(images) - len(pairs)
                print(f"  {cems_id}/{phase}: {existing} existing predictions, "
                      f"{len(pairs)} to run (skip={skip})")

    return pred_index


# =====================================================================
# 3. CHANGE DETECTION
# =====================================================================

def compute_change_detection(pred_index):
    """For each matched before/during pair compute flood expansion.

    Returns a list of dicts with per-tile metrics.
    """
    results = []

    for cems_id, phases in pred_index.items():
        before_preds = phases["before"]
        during_preds = phases["during"]
        common = sorted(set(before_preds) & set(during_preds))

        print(f"\n── Change detection: {cems_id} ({len(common)} pairs) ──")

        for prefix in common:
            before_path = before_preds[prefix]
            during_path = during_preds[prefix]

            if not os.path.exists(before_path) or not os.path.exists(during_path):
                print(f"  [{prefix}] SKIP – prediction file missing")
                continue

            try:
                with rasterio.open(before_path) as src:
                    before_mask = src.read(1)
                with rasterio.open(during_path) as src:
                    during_mask = src.read(1)

                # Treat nodata (-1) as not-flood
                before_flood = (before_mask == 1)
                during_flood = (during_mask == 1)

                # New flood = flooded during but NOT before
                new_flood = during_flood & ~before_flood
                # Receded = was flooded before but not during
                receded = before_flood & ~during_flood

                before_px = int(before_flood.sum())
                during_px = int(during_flood.sum())
                new_px = int(new_flood.sum())
                receded_px = int(receded.sum())

                pixel_area_km2 = (PIXEL_RES_M ** 2) / 1e6  # 0.0001 km²

                before_km2 = before_px * pixel_area_km2
                during_km2 = during_px * pixel_area_km2
                new_km2 = new_px * pixel_area_km2
                receded_km2 = receded_px * pixel_area_km2

                pct_increase = (
                    (new_px / before_px * 100) if before_px > 0 else float("inf")
                )

                row = {
                    "cems_id": cems_id,
                    "tile": prefix,
                    "before_px": before_px,
                    "during_px": during_px,
                    "new_flood_px": new_px,
                    "receded_px": receded_px,
                    "before_km2": round(before_km2, 4),
                    "during_km2": round(during_km2, 4),
                    "new_flood_km2": round(new_km2, 4),
                    "receded_km2": round(receded_km2, 4),
                    "pct_increase": round(pct_increase, 2),
                }
                results.append(row)

                print(f"  [{prefix}] before={before_km2:.4f} km²  during={during_km2:.4f} km²  "
                      f"new={new_km2:.4f} km²  +{pct_increase:.1f}%")

            except Exception as e:
                print(f"  [{prefix}] ERROR: {e}")
                traceback.print_exc()

    return results


def save_report(results, output_path):
    """Write change-detection metrics to CSV."""
    if not results:
        print("No change-detection results to save.")
        return
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = list(results[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nReport saved → {output_path}")


# =====================================================================
# 4. OVERLAY VISUALIZATION
# =====================================================================

def _normalize_band(band):
    """Percentile-clip + min-max normalize a single band to [0,1]."""
    p2 = np.percentile(band[band > 0], 2) if (band > 0).any() else 0
    p98 = np.percentile(band[band > 0], 98) if (band > 0).any() else 1
    band = np.clip(band, p2, p98).astype(np.float32)
    lo, hi = band.min(), band.max()
    if hi - lo < 1e-6:
        return np.zeros_like(band)
    return (band - lo) / (hi - lo)


def _load_rgb(tif_path):
    """Load an 8-band SenForFlood image and return an RGB array [H,W,3].

    Band order (0-indexed): 0=B2(Blue) 1=B3(Green) 2=B4(Red).
    """
    with rasterio.open(tif_path) as src:
        red   = src.read(3).astype(np.float32)  # band index 2, 1-based=3
        green = src.read(2).astype(np.float32)  # band index 1, 1-based=2
        blue  = src.read(1).astype(np.float32)  # band index 0, 1-based=1
    red   = _normalize_band(red)
    green = _normalize_band(green)
    blue  = _normalize_band(blue)
    return np.stack([red, green, blue], axis=-1)


def generate_overlays(pred_index, cd_results):
    """Create side-by-side overlay PNGs for every matched before/during pair."""

    # Build lookup: (cems_id, tile) → metrics row
    metrics_lut = {(r["cems_id"], r["tile"]): r for r in cd_results}

    for cems_id, phases in pred_index.items():
        before_preds = phases["before"]
        during_preds = phases["during"]
        common = sorted(set(before_preds) & set(during_preds))

        overlay_dir = os.path.join(OUTPUT_ROOT, "Overlays", cems_id)
        os.makedirs(overlay_dir, exist_ok=True)

        print(f"\n── Overlays: {cems_id} ({len(common)} pairs) ──")

        for prefix in common:
            before_pred_path = before_preds[prefix]
            during_pred_path = during_preds[prefix]

            # Locate original during-flood image for RGB backdrop
            during_src_dir = os.path.join(
                SENFORFLOOD_ROOT, cems_id, "s2_during_flood"
            )
            during_src_candidates = glob.glob(
                os.path.join(during_src_dir, f"{prefix}_*.tif")
            )
            if not during_src_candidates:
                print(f"  [{prefix}] SKIP – original during image not found")
                continue
            during_src_path = during_src_candidates[0]

            if not os.path.exists(before_pred_path) or not os.path.exists(during_pred_path):
                print(f"  [{prefix}] SKIP – prediction mask missing")
                continue

            try:
                # Load masks
                with rasterio.open(before_pred_path) as src:
                    before_mask = src.read(1)
                with rasterio.open(during_pred_path) as src:
                    during_mask = src.read(1)

                before_flood = (before_mask == 1)
                during_flood = (during_mask == 1)
                new_flood = during_flood & ~before_flood

                # Clean with morphological opening
                permanent = binary_opening(before_flood & during_flood, structure=np.ones((3, 3)))
                new_clean = binary_opening(new_flood, structure=np.ones((3, 3)))

                # Load RGB
                rgb = _load_rgb(during_src_path)

                # Metrics for title
                m = metrics_lut.get((cems_id, prefix), {})
                new_km2 = m.get("new_flood_km2", 0)
                pct = m.get("pct_increase", 0)

                # ── Build figure ────────────────────────────────
                fig, axes = plt.subplots(1, 2, figsize=(16, 7))

                # Left: original during-flood RGB
                axes[0].imshow(rgb)
                axes[0].set_title(f"During-Flood RGB\n{cems_id} / tile {prefix}", fontsize=11)
                axes[0].axis("off")

                # Right: overlay
                axes[1].imshow(rgb)

                # Blue overlay → permanent / pre-existing water
                blue_layer = np.zeros((*permanent.shape, 4), dtype=np.float32)
                blue_layer[permanent] = [0.0, 0.3, 1.0, 0.55]
                axes[1].imshow(blue_layer)

                # Red overlay → new flood expansion
                red_layer = np.zeros((*new_clean.shape, 4), dtype=np.float32)
                red_layer[new_clean] = [1.0, 0.0, 0.0, 0.6]
                axes[1].imshow(red_layer)

                pct_str = f"{pct:.1f}%" if pct != float("inf") else "N/A (no prior flood)"
                axes[1].set_title(
                    f"Flood Detection – {cems_id}\n"
                    f"Blue = permanent water | Red = new expansion\n"
                    f"Flood increase: {new_km2:.4f} km² | {pct_str}",
                    fontsize=10,
                )
                axes[1].axis("off")

                plt.tight_layout()
                out_png = os.path.join(overlay_dir, f"{prefix}_overlay.png")
                fig.savefig(out_png, dpi=150, bbox_inches="tight")
                plt.close(fig)
                print(f"  [{prefix}] saved → {out_png}")

            except Exception as e:
                print(f"  [{prefix}] ERROR: {e}")
                traceback.print_exc()


# =====================================================================
# 5. MAIN PIPELINE
# =====================================================================

def parse_pipeline_args():
    p = argparse.ArgumentParser(description="Automated flood pipeline")
    p.add_argument("--device", default="cuda", help="cuda or cpu")
    p.add_argument(
        "--skip-inference", action="store_true",
        help="Skip inference; reuse existing prediction masks",
    )
    p.add_argument(
        "--cems-root", default=SENFORFLOOD_ROOT,
        help="Root of CEMS folders",
    )
    p.add_argument(
        "--output-root", default=OUTPUT_ROOT,
        help="Root output directory",
    )
    p.add_argument(
        "-n", "--num-images", type=int, default=None,
        help="Max number of image pairs to process per CEMS folder. "
             "If omitted, you will be prompted interactively.",
    )
    p.add_argument(
        "--model", choices=["prithvi", "damnet"], default="prithvi",
        help="Which model backend to use: 'prithvi' (optical) or 'damnet' (SAR ViT)",
    )
    p.add_argument(
        "--damnet-ckpt", default=DAMNET_CKPT,
        help="Path to trained DAM-Net checkpoint (.pt)",
    )
    p.add_argument(
        "--damnet-preset", choices=["tiny", "small", "base"], default="small",
        help="DAM-Net model size (must match checkpoint)",
    )
    p.add_argument(
        "--threshold", type=float, default=0.5,
        help="Flood probability threshold for DAM-Net (0-1)",
    )
    return p.parse_args()


# =====================================================================
# 6. DAM-Net INTEGRATION – SAR bitemporal change detection
# =====================================================================

def run_damnet_pipeline(cems_ids, args):
    """Run the DAM-Net Siamese ViT pipeline on bitemporal SAR pairs.

    DAM-Net directly outputs a flood-change mask from pre/post pairs,
    so no separate change detection step is needed.
    """
    from damnet.inference import load_damnet, predict_tiled, compute_flood_metrics
    from damnet.inference import save_mask_geotiff, generate_overlay
    from damnet.dataset import _read_tif, _normalize_sar

    print("\n══════════════════════════════════════")
    print("  Loading DAM-Net model …")
    print("══════════════════════════════════════")
    model = load_damnet(args.damnet_ckpt, device=args.device)

    all_results = []

    for cems_id in cems_ids:
        pairs = collect_image_pairs(SENFORFLOOD_ROOT, cems_id)
        if args.num_images is not None:
            pairs = pairs[:args.num_images]

        out_dir = os.path.join(OUTPUT_ROOT, "DAMNet", cems_id)
        overlay_dir = os.path.join(OUTPUT_ROOT, "Overlays", cems_id)
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(overlay_dir, exist_ok=True)

        print(f"\n── DAM-Net: {cems_id} ({len(pairs)} pairs) ──")

        for before_path, during_path in pairs:
            prefix = os.path.basename(during_path).split("_")[0]
            print(f"  [{prefix}] Processing …")

            try:
                # Load & normalize SAR images
                pre_img = _read_tif(before_path)
                post_img = _read_tif(during_path)
                if pre_img.shape[0] > 2:
                    pre_img = pre_img[:2]
                if post_img.shape[0] > 2:
                    post_img = post_img[:2]
                pre_img = _normalize_sar(pre_img)
                post_img = _normalize_sar(post_img)

                # DAM-Net prediction (directly outputs flood-change mask)
                mask = predict_tiled(
                    model, pre_img, post_img,
                    tile_size=512, device=args.device,
                    threshold=args.threshold)

                # Metrics
                metrics = compute_flood_metrics(mask, PIXEL_RES_M)

                row = {
                    "cems_id": cems_id,
                    "tile": prefix,
                    "flood_pixels": metrics["flood_pixels"],
                    "inundated_km2": metrics["inundated_area_km2"],
                    "flood_fraction": round(metrics["flood_fraction"] * 100, 2),
                    "inundated_m2": metrics["inundated_area_m2"],
                }
                all_results.append(row)

                # Save mask
                mask_path = os.path.join(out_dir, f"{prefix}_flood_mask.tif")
                save_mask_geotiff(mask, mask_path, reference_tif=during_path)

                # Save overlay
                overlay_path = os.path.join(
                    overlay_dir, f"{prefix}_damnet_overlay.png")
                generate_overlay(post_img, mask, overlay_path, metrics,
                                 title=f"DAM-Net Flood Detection – {cems_id}/{prefix}")

                print(f"  [{prefix}] Inundated: {metrics['inundated_area_km2']:.4f} km² "
                      f"({metrics['flood_fraction']*100:.1f}%)")

            except Exception as e:
                print(f"  [{prefix}] ERROR: {e}")
                traceback.print_exc()

    return all_results


def main():
    args = parse_pipeline_args()

    global SENFORFLOOD_ROOT, OUTPUT_ROOT
    SENFORFLOOD_ROOT = args.cems_root
    OUTPUT_ROOT = args.output_root

    model_name = args.model.upper()
    print("╔══════════════════════════════════════════════╗")
    print(f"║  Flood Segmentation & Change-Detection       ║")
    print(f"║  Pipeline  ({model_name} + SenForFlood)     ║")
    print("╚══════════════════════════════════════════════╝")
    print(f"\nCEMS root   : {SENFORFLOOD_ROOT}")
    print(f"Output root : {OUTPUT_ROOT}")
    print(f"Device      : {args.device}")
    print(f"Model       : {model_name}")
    t0 = time.time()

    # ── Step 1: Discover CEMS folders ───────────────────────────────
    print("\n▶ Step 1 – Scanning CEMS folders …")
    cems_ids = discover_cems_folders(SENFORFLOOD_ROOT)
    if not cems_ids:
        print("No valid CEMS folders found. Exiting.")
        return
    print(f"  Found {len(cems_ids)} folder(s): {cems_ids}")

    # Show image pair counts and compute totals
    pair_counts = {}
    for cid in cems_ids:
        pairs = collect_image_pairs(SENFORFLOOD_ROOT, cid)
        pair_counts[cid] = len(pairs)
        print(f"  {cid}: {len(pairs)} matched before/during pairs")

    max_per_folder = max(pair_counts.values()) if pair_counts else 0
    total_pairs = sum(pair_counts.values())
    print(f"\n  Total image pairs across all folders: {total_pairs}")
    print(f"  Max per single folder: {max_per_folder}")

    # ── Determine how many images to process ────────────────────────
    num_images = args.num_images
    if num_images is None:
        # Interactive prompt
        print(f"\n  How many images per folder do you want to process?")
        print(f"  (Enter a number 1-{max_per_folder}, or press Enter for all)")
        try:
            user_input = input("  ▸ ").strip()
            if user_input == "":
                num_images = None  # process all
                print("  → Processing ALL images.")
            else:
                num_images = int(user_input)
                if num_images < 1 or num_images > max_per_folder:
                    print(f"  ⚠ Invalid. Must be 1-{max_per_folder}. Using all.")
                    num_images = None
                else:
                    effective = sum(min(n, num_images) for n in pair_counts.values())
                    print(f"  → Processing {num_images} per folder ({effective} pairs total).")
        except (ValueError, EOFError):
            print("  → Non-numeric input. Processing ALL images.")
            num_images = None
    else:
        num_images = min(num_images, max_per_folder) if num_images > 0 else None
        if num_images:
            effective = sum(min(n, num_images) for n in pair_counts.values())
            print(f"\n  → Processing {num_images} per folder ({effective} pairs total) via --num-images.")
    args.num_images = num_images

    # ── Branch based on model selection ─────────────────────────────
    if args.model == "damnet":
        # ── DAM-Net path: direct bitemporal change detection ────────
        print("\n▶ Step 2 – Running DAM-Net bitemporal flood detection …")
        cd_results = run_damnet_pipeline(cems_ids, args)
        report_path = os.path.join(OUTPUT_ROOT, "damnet_flood_report.csv")
        save_report(cd_results, report_path)

        elapsed = time.time() - t0
        print(f"\n{'='*50}")
        print(f"DAM-Net pipeline complete in {elapsed:.1f}s")
        print(f"  Masks     → {os.path.join(OUTPUT_ROOT, 'DAMNet')}")
        print(f"  Overlays  → {os.path.join(OUTPUT_ROOT, 'Overlays')}")
        print(f"  Report    → {report_path}")
        print(f"{'='*50}")
    else:
        # ── Prithvi path: existing pipeline ─────────────────────────
        print("\n▶ Step 2 – Running Prithvi inference …")
        pred_index = run_inference(cems_ids, device=args.device,
                                   skip=args.skip_inference,
                                   max_images=num_images)

        print("\n▶ Step 3 – Computing change detection …")
        cd_results = compute_change_detection(pred_index)
        report_path = os.path.join(OUTPUT_ROOT, "change_detection_report.csv")
        save_report(cd_results, report_path)

        print("\n▶ Step 4 – Generating overlay visualizations …")
        generate_overlays(pred_index, cd_results)

        elapsed = time.time() - t0
        print(f"\n{'='*50}")
        print(f"Prithvi pipeline complete in {elapsed:.1f}s")
        print(f"  Masks     → {os.path.join(OUTPUT_ROOT, 'Prithvi')}")
        print(f"  Overlays  → {os.path.join(OUTPUT_ROOT, 'Overlays')}")
        print(f"  Report    → {report_path}")
        print(f"{'='*50}")


if __name__ == "__main__":
    main()
