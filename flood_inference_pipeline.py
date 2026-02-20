"""
ai4g-flood Inference Pipeline — All CEMS Events
=================================================
Scans G:\\Sentinel\\DATA\\SenForFlood\\CEMS for all event folders (EMSR*),
reads paired S1 before/during flood tiles, runs change-detection + U-Netthe blue part isnt available in bottom right and even in the bottom left it is scattered and not continuous like it is in image 3
i basically want to do eveyrthing that we did with prithvi model but model should be ai4good and visuals should be like we have right now 

inference, evaluates against ground-truth masks, and saves:

  - GeoTIFF flood predictions per tile
  - Per-tile and per-event metrics CSV
  - Summary report across all events

Usage:
    python flood_inference_pipeline.py
    python flood_inference_pipeline.py --cems_root "G:\\path\\to\\CEMS"
    python flood_inference_pipeline.py --events EMSR771 EMSR773
    python flood_inference_pipeline.py --buffer 8 --device cuda
"""

import argparse
import glob
import os
import pathlib
import sys
import time

import numpy as np
import rasterio
import torch
from scipy.ndimage import binary_closing, binary_opening, gaussian_filter

# Fix: checkpoint saved on Linux with PosixPath – needed on Windows
pathlib.PosixPath = pathlib.WindowsPath

# ── Add ai4g-flood utilities to path ──
AI4G_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ai4g-flood")
sys.path.insert(0, os.path.join(AI4G_ROOT, "src"))

from utils.image_processing import (
    pad_to_nearest,
    create_patches,
    reconstruct_image_from_patches,
    apply_buffer,
)
from utils.model import load_model


# ═══════════════════════════════════════════════════════════
#  DEFAULTS
# ═══════════════════════════════════════════════════════════
DEFAULTS = dict(
    cems_root   = r"D:\Sentinel Final\Sentinel Backend\DATA\CEMS",
    model_path  = os.path.join(AI4G_ROOT, "models", "ai4g_sar_model.ckpt"),
    output_root = r"D:\Sentinel Final\Sentinel Backend\OUTPUTS\ai4g_flood",
    input_size  = 128,
    buffer_size = 4,
    batch_size  = 512,
    # Change-detection thresholds (0-255 dB-scaled space)
    vv_threshold     = 100,
    vh_threshold     = 90,
    delta_amplitude  = 10,
    vv_min_threshold = 75,
    vh_min_threshold = 70,
)

PIXEL_RES_M = 10.0  # SenForFlood is ~10 m/pixel


# ═══════════════════════════════════════════════════════════
#  DATA I/O
# ═══════════════════════════════════════════════════════════

def db_to_scaled(db_array):
    """SenForFlood S1 bands are already in dB.  Map to 0-255 like ai4g expects.
    Model's own db_scale(): 10*log10(linear)*2 + 135  →  dB*2 + 135."""
    scaled = db_array * 2.0 + 135.0
    return np.clip(np.nan_to_num(scaled, nan=0.0), 0, 255)


def read_s1_tile(path):
    """Read a SenForFlood S1 .tif (4 bands) → VV_scaled, VH_scaled, crs, transform."""
    with rasterio.open(path) as src:
        vv_db = src.read(1).astype(np.float32)
        vh_db = src.read(2).astype(np.float32)
        crs = src.crs
        transform = src.transform
    return db_to_scaled(vv_db), db_to_scaled(vh_db), crs, transform


def read_gt_mask(path):
    """Read ground-truth mask  → (flood_binary, raw_mask).
    raw: 0=no-flood, 1=flood, 2=permanent-water."""
    with rasterio.open(path) as src:
        raw = src.read(1)
    return (raw == 1).astype(np.uint8), raw


def save_geotiff(array, path, crs, transform, nodata=np.nan):
    """Write single-band float32 GeoTIFF."""
    arr = array.astype(np.float32).copy()
    arr[arr == 0] = nodata
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff",
        height=arr.shape[0], width=arr.shape[1],
        count=1, dtype="float32",
        crs=crs, transform=transform,
        compress="lzw", nodata=nodata,
    ) as dst:
        dst.write(arr, 1)


def save_water_mask(mask, path, crs, transform):
    """Write a boolean water mask as uint8 GeoTIFF (1=water, 0=no-water)."""
    arr = mask.astype(np.uint8)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff",
        height=arr.shape[0], width=arr.shape[1],
        count=1, dtype="uint8",
        crs=crs, transform=transform,
        compress="lzw",
    ) as dst:
        dst.write(arr, 1)


# ═══════════════════════════════════════════════════════════
#  WATER DETECTION  (S2 NDWI + SAR fallback)
# ═══════════════════════════════════════════════════════════

def detect_water_s2(s2_path, ndwi_thresh=0.0):
    """Detect water using NDWI from Sentinel-2 optical data.
    NDWI = (Green - NIR) / (Green + NIR) → positive for water.
    SenForFlood S2 bands: 1=B2, 2=B3(Green), 3=B4(Red), 4=B8A(NIR)."""
    if not os.path.exists(s2_path):
        return None
    try:
        with rasterio.open(s2_path) as src:
            green = src.read(2).astype(np.float32)
            nir   = src.read(4).astype(np.float32)
        denom = green + nir
        denom[denom == 0] = 1e-10
        ndwi = (green - nir) / denom
        ndwi_smooth = gaussian_filter(ndwi, sigma=1.5)
        water = ndwi_smooth > ndwi_thresh
        water = binary_closing(water, structure=np.ones((5, 5)), iterations=2)
        water = binary_opening(water, structure=np.ones((3, 3)))
        return water
    except Exception:
        return None


def detect_water_sar(vv_scaled, vh_scaled):
    """Detect water from pre-computed scaled SAR arrays (0-255).
    Uses Gaussian smoothing + adaptive threshold + morphological closing."""
    vv_sm = gaussian_filter(vv_scaled, sigma=2.0)
    vh_sm = gaussian_filter(vh_scaled, sigma=2.0)
    valid = (vv_scaled > 0) & (vh_scaled > 0)
    if valid.sum() == 0:
        return np.zeros(vv_scaled.shape, dtype=bool)
    try:
        from skimage.filters import threshold_otsu
        vv_thresh = threshold_otsu(vv_sm[valid])
        vh_thresh = threshold_otsu(vh_sm[valid])
    except Exception:
        vv_thresh = np.percentile(vv_sm[valid], 30)
        vh_thresh = np.percentile(vh_sm[valid], 30)
    water = (vv_sm < vv_thresh) & (vh_sm < vh_thresh)
    water = binary_closing(water, structure=np.ones((7, 7)), iterations=3)
    water = binary_opening(water, structure=np.ones((3, 3)))
    return water


# ═══════════════════════════════════════════════════════════
#  CHANGE DETECTION + INFERENCE
# ═══════════════════════════════════════════════════════════

def calculate_flood_change(vv_pre, vh_pre, vv_post, vh_post, cfg):
    """Rule-based SAR amplitude-drop change map  →  (H, W, 2) int array."""
    vv_change = (
        (vv_post < cfg["vv_threshold"])
        & (vv_pre  > cfg["vv_threshold"])
        & ((vv_pre - vv_post) > cfg["delta_amplitude"])
    ).astype(int)
    vh_change = (
        (vh_post < cfg["vh_threshold"])
        & (vh_pre  > cfg["vh_threshold"])
        & ((vh_pre - vh_post) > cfg["delta_amplitude"])
    ).astype(int)
    zero_idx = (
        (vv_post < cfg["vv_min_threshold"]) | (vv_pre < cfg["vv_min_threshold"])
        | (vh_post < cfg["vh_min_threshold"]) | (vh_pre < cfg["vh_min_threshold"])
    )
    vv_change[zero_idx] = 0
    vh_change[zero_idx] = 0
    return np.stack((vv_change, vh_change), axis=2)


def run_inference(model, flood_change, device, target_shape, cfg):
    """Patch-based U-Net inference  →  binary flood mask (0 or 255)."""
    inp = cfg["input_size"]
    padded = pad_to_nearest(flood_change, inp, [0, 1])
    patches = create_patches(padded, (inp, inp), inp)

    predictions = []
    with torch.no_grad():
        for i in range(0, len(patches), cfg["batch_size"]):
            batch = patches[i : i + cfg["batch_size"]]
            if not batch:
                continue
            tensor = torch.from_numpy(np.array(batch)).float().to(device)
            output = model(tensor)
            _, pred = torch.max(output, 1)
            pred = (pred * 255).to(torch.int)
            pred[(tensor[:, 0] == 0) & (tensor[:, 1] == 0)] = 0
            predictions.extend(pred.cpu().numpy())

    pred_img, _ = reconstruct_image_from_patches(
        predictions, padded.shape[:2], (inp, inp), inp,
    )
    pred_img = pred_img[: target_shape[0], : target_shape[1]]

    if cfg["buffer_size"] > 0:
        pred_img = apply_buffer(pred_img, cfg["buffer_size"])

    return pred_img


# ═══════════════════════════════════════════════════════════
#  METRICS
# ═══════════════════════════════════════════════════════════

def compute_metrics(pred_mask, gt_binary):
    """Returns dict with accuracy, precision, recall, f1, iou, tp/fp/fn/tn."""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, jaccard_score, confusion_matrix,
    )
    p = (pred_mask == 255).astype(int).flatten()
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
#  EVENT DISCOVERY
# ═══════════════════════════════════════════════════════════

def discover_events(cems_root, event_filter=None):
    """Scan CEMS root and return list of event dicts.
    Each dict: {event_id, before_files, during_files, mask_files, tile_ids}."""
    events = []
    for name in sorted(os.listdir(cems_root)):
        event_dir = os.path.join(cems_root, name)
        if not os.path.isdir(event_dir):
            continue
        if event_filter and name not in event_filter:
            continue

        before_dir = os.path.join(event_dir, "s1_before_flood")
        during_dir = os.path.join(event_dir, "s1_during_flood")
        mask_dir   = os.path.join(event_dir, "flood_mask")

        if not os.path.isdir(before_dir) or not os.path.isdir(during_dir):
            continue

        before_files = sorted(glob.glob(os.path.join(before_dir, "*_s1_before_flood.tif")))
        during_files = sorted(glob.glob(os.path.join(during_dir, "*_s1_during_flood.tif")))
        mask_files   = sorted(glob.glob(os.path.join(mask_dir,   "*_flood_mask.tif")))

        # Match tiles that exist in both before & during
        before_ids = {os.path.basename(f).split("_")[0] for f in before_files}
        during_ids = {os.path.basename(f).split("_")[0] for f in during_files}
        common = sorted(before_ids & during_ids)

        if not common:
            continue

        mask_ids = {os.path.basename(f).split("_")[0] for f in mask_files}

        events.append(dict(
            event_id     = name,
            event_dir    = event_dir,
            before_dir   = before_dir,
            during_dir   = during_dir,
            mask_dir     = mask_dir,
            tile_ids     = common,
            has_masks    = bool(mask_ids & set(common)),
            mask_ids     = mask_ids,
        ))

    return events


# ═══════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ═══════════════════════════════════════════════════════════

def run_pipeline(cfg):
    print("=" * 72)
    print("  ai4g-flood Inference Pipeline — SenForFlood CEMS")
    print("=" * 72)

    # ── Discover events ──
    events = discover_events(cfg["cems_root"], cfg.get("events"))
    total_tiles = sum(len(e["tile_ids"]) for e in events)
    print(f"\n  CEMS root:   {cfg['cems_root']}")
    print(f"  Events:      {len(events)}  ({', '.join(e['event_id'] for e in events)})")
    print(f"  Total tiles: {total_tiles}")
    print(f"  Output:      {cfg['output_root']}")
    print(f"  Model:       {cfg['model_path']}")

    # ── Load model ──
    print(f"\n[1] Loading ai4g-flood model ...")
    device = torch.device(cfg.get("device", "cpu"))
    if cfg.get("device") == "cuda" and not torch.cuda.is_available():
        print("    CUDA not available, falling back to CPU")
        device = torch.device("cpu")
    model = load_model(cfg["model_path"], device, in_channels=2, n_classes=2)
    model.eval()
    print(f"    Device: {device}   OK\n")

    # ── Process each event ──
    print("[2] Running inference ...\n")
    all_event_results = []
    global_idx = 0
    t_global = time.time()

    for event in events:
        eid = event["event_id"]
        tile_ids = event["tile_ids"]
        n = len(tile_ids)
        out_dir = os.path.join(cfg["output_root"], eid)
        geo_dir = os.path.join(out_dir, "geotiffs")
        os.makedirs(geo_dir, exist_ok=True)

        print(f"  +-- {eid}  ({n} tiles) " + "-" * 30)
        event_metrics = []
        t_event = time.time()

        for idx, tid in enumerate(tile_ids):
            global_idx += 1
            before_path = os.path.join(event["before_dir"], f"{tid}_s1_before_flood.tif")
            during_path = os.path.join(event["during_dir"], f"{tid}_s1_during_flood.tif")
            mask_path   = os.path.join(event["mask_dir"],   f"{tid}_flood_mask.tif")

            t0 = time.time()

            # Read
            vv_pre, vh_pre, _, _ = read_s1_tile(before_path)
            vv_post, vh_post, crs, transform = read_s1_tile(during_path)

            # Change detection
            flood_change = calculate_flood_change(vv_pre, vh_pre, vv_post, vh_post, cfg)

            # Inference
            pred_mask = run_inference(model, flood_change, device, vv_post.shape, cfg)
            n_pred = int((pred_mask == 255).sum())
            elapsed = time.time() - t0

            # Save GeoTIFF
            tif_name = f"{tid}_flood_pred.tif"
            save_geotiff(pred_mask, os.path.join(geo_dir, tif_name), crs, transform)

            # ── Detect water in before & during images ──
            s2_before = os.path.join(event["event_dir"], "s2_before_flood",
                                     f"{tid}_s2_before_flood.tif")
            s2_during = os.path.join(event["event_dir"], "s2_during_flood",
                                     f"{tid}_s2_during_flood.tif")

            bef_water = detect_water_s2(s2_before)
            dur_water = detect_water_s2(s2_during)
            if bef_water is None:
                bef_water = detect_water_sar(vv_pre, vh_pre)
            if dur_water is None:
                dur_water = detect_water_sar(vv_post, vh_post)

            save_water_mask(bef_water, os.path.join(geo_dir, f"{tid}_before_water.tif"),
                            crs, transform)
            save_water_mask(dur_water, os.path.join(geo_dir, f"{tid}_during_water.tif"),
                            crs, transform)

            # Evaluate
            status = ""
            if tid in event["mask_ids"] and os.path.exists(mask_path):
                gt_bin, gt_raw = read_gt_mask(mask_path)
                m = compute_metrics(pred_mask, gt_bin)
                m.update(event_id=eid, tile_id=tid, n_pred=n_pred,
                         n_gt=int(gt_bin.sum()), time_s=round(elapsed, 2),
                         pred_km2=round(pixels_to_km2(n_pred), 4),
                         gt_km2=round(pixels_to_km2(gt_bin.sum()), 4))
                event_metrics.append(m)
                status = (f"F1={m['f1']:.3f}  IoU={m['iou']:.3f}  "
                          f"Prec={m['precision']:.3f}  Rec={m['recall']:.3f}")
            else:
                status = f"Pred={n_pred} px ({pixels_to_km2(n_pred):.4f} km²)  [no GT]"

            print(f"  |  [{idx+1}/{n}] {tid}  {status}  ({elapsed:.1f}s)")

        event_time = time.time() - t_event

        # ── Per-event CSV ──
        if event_metrics:
            csv_path = os.path.join(out_dir, f"{eid}_metrics.csv")
            _write_metrics_csv(csv_path, event_metrics, eid)

            avg_f1 = np.mean([m["f1"] for m in event_metrics])
            avg_iou = np.mean([m["iou"] for m in event_metrics])
            print(f"  |  Avg F1={avg_f1:.3f}  Avg IoU={avg_iou:.3f}")
            print(f"  |  Metrics -> {csv_path}")

        print(f"  +-- {eid} done ({event_time:.1f}s)\n")
        all_event_results.append(dict(
            event_id=eid, n_tiles=n, time_s=event_time,
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

    # ── Print summary table ──
    print(f"  {'-' * 68}")
    print(f"  {'Event':<10} | {'Tiles':>5} | {'Avg F1':>7} | {'Avg IoU':>7} | "
          f"{'Avg Prec':>8} | {'Avg Rec':>7} | {'Time':>6}")
    print(f"  {'-' * 68}")
    for er in all_event_results:
        eid = er["event_id"]
        n = er["n_tiles"]
        t = er["time_s"]
        if er["metrics"]:
            af1 = np.mean([m["f1"] for m in er["metrics"]])
            aio = np.mean([m["iou"] for m in er["metrics"]])
            apr = np.mean([m["precision"] for m in er["metrics"]])
            are = np.mean([m["recall"] for m in er["metrics"]])
            print(f"  {eid:<10} | {n:>5} | {af1:>7.4f} | {aio:>7.4f} | "
                  f"{apr:>8.4f} | {are:>7.4f} | {t:>5.1f}s")
        else:
            print(f"  {eid:<10} | {n:>5} | {'N/A':>7} | {'N/A':>7} | "
                  f"{'N/A':>8} | {'N/A':>7} | {t:>5.1f}s")
    print(f"  {'-' * 68}")

    if all_metrics:
        g_f1  = np.mean([m["f1"]  for m in all_metrics])
        g_iou = np.mean([m["iou"] for m in all_metrics])
        print(f"  {'GLOBAL':<10} | {total_tiles:>5} | {g_f1:>7.4f} | {g_iou:>7.4f} |")

    print(f"\n  Total time: {total_time:.1f}s")

    # ── Done ──
    print("\n" + "=" * 72)
    print("  PIPELINE COMPLETE")
    print("=" * 72)
    print(f"""
  Outputs: {cfg['output_root']}
    +-- <EVENT_ID>/
    |   +-- geotiffs/          Flood prediction GeoTIFFs per tile
    |   +-- <EVENT>_metrics.csv
    +-- all_events_metrics.csv

  Run visualize_flood_results.py to generate overlay images.
""")
    return all_event_results


def _write_metrics_csv(path, metrics_list, label):
    """Write a list of metric dicts to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cols = ["event_id", "tile_id", "accuracy", "precision", "recall", "f1", "iou",
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
        description="ai4g-flood inference pipeline for SenForFlood CEMS data",
    )
    p.add_argument("--cems_root", default=DEFAULTS["cems_root"],
                   help="Root directory containing CEMS event folders")
    p.add_argument("--events", nargs="*", default=None,
                   help="Process only these events (e.g. EMSR771 EMSR773)")
    p.add_argument("--model_path", default=DEFAULTS["model_path"],
                   help="Path to ai4g_sar_model.ckpt")
    p.add_argument("--output_root", default=DEFAULTS["output_root"],
                   help="Root output directory")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                   help="Inference device")
    p.add_argument("--buffer", type=int, default=DEFAULTS["buffer_size"],
                   help="Dilation buffer in pixels (0 to disable)")
    p.add_argument("--batch_size", type=int, default=DEFAULTS["batch_size"])
    p.add_argument("--vv_threshold", type=int, default=DEFAULTS["vv_threshold"])
    p.add_argument("--vh_threshold", type=int, default=DEFAULTS["vh_threshold"])
    p.add_argument("--delta_amplitude", type=int, default=DEFAULTS["delta_amplitude"])
    return p.parse_args()


def main():
    args = parse_args()
    cfg = dict(DEFAULTS)
    cfg.update(
        cems_root   = args.cems_root,
        events      = args.events,
        model_path  = args.model_path,
        output_root = args.output_root,
        device      = args.device,
        buffer_size = args.buffer,
        batch_size  = args.batch_size,
        vv_threshold = args.vv_threshold,
        vh_threshold = args.vh_threshold,
        delta_amplitude = args.delta_amplitude,
    )
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
