"""
Satellite Fire / Burn-Scar Detection API — Prithvi 100M Model
===============================================================
Serves burn-scar prediction data from the Prithvi-100M fine-tuned model.
Reads GeoTIFFs from OUTPUTS/prithvi_burn_scars/<split>/geotiffs/ and
original HLS tiles from DATA/Fire/<split>/, and serves:

  - Split list with geo-coordinates, stats, metadata
  - GeoJSON burn-scar polygons per tile/split
  - HLS True-Color RGB PNGs
  - HLS False-Color (SWIR2, NIR, Red) PNGs  — highlights burn scars
  - NBR (Normalized Burn Ratio) heatmap PNGs
  - Burn-scar mask overlays
  - Per-tile and per-split statistics
  - Tile bounds for imagery draping on CesiumJS globe

Run:
    uvicorn backend.fire_satellite_api:app --reload --port 8002

Endpoints:
    GET  /api/summary                                          → global stats
    GET  /api/splits                                           → list splits
    GET  /api/splits/{split}                                   → split detail + tiles
    GET  /api/splits/{split}/geojson?layer=burn                → merged GeoJSON
    GET  /api/splits/{split}/tiles/{tile}/geojson               → per-tile GeoJSON
    GET  /api/splits/{split}/tiles/{tile}/rgb                   → HLS True-Color PNG
    GET  /api/splits/{split}/tiles/{tile}/false_color            → HLS False-Color PNG
    GET  /api/splits/{split}/tiles/{tile}/nbr                   → NBR heatmap PNG
    GET  /api/splits/{split}/tiles/{tile}/mask?overlay=false     → burn mask PNG
    GET  /api/splits/{split}/tiles/{tile}/overlay               → composite overlay PNG
    GET  /api/splits/{split}/tiles/{tile}/stats                 → per-tile stats
    GET  /api/splits/{split}/tiles/{tile}/bounds                → [w, s, e, n]
"""

import csv
import glob
import io
import os
from functools import lru_cache
from typing import Optional

import numpy as np
import rasterio
from rasterio.features import shapes as rio_shapes
from rasterio.warp import transform_bounds
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image
from scipy.ndimage import binary_opening, binary_closing

# ── Configuration ───────────────────────────────────────────────────────

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FIRE_DATA_ROOT = os.environ.get(
    "FIRE_DATA_ROOT",
    os.path.join(_BASE_DIR, "DATA", "Fire"),
)
FIRE_OUTPUT_ROOT = os.environ.get(
    "FIRE_OUTPUT_ROOT",
    os.path.join(_BASE_DIR, "OUTPUTS", "prithvi_burn_scars"),
)
PIXEL_RES_M = 30.0  # HLS ~30 m/pixel

# HLS 6-band indices: 0=Blue, 1=Green, 2=Red, 3=NarrowNIR, 4=SWIR1, 5=SWIR2
B_BLUE, B_GREEN, B_RED, B_NIR, B_SWIR1, B_SWIR2 = 0, 1, 2, 3, 4, 5

IMG_SUFFIX = "_merged.tif"
MASK_SUFFIX = ".mask.tif"
PRED_SUFFIX = "_burn_pred.tif"

# Split metadata
SPLIT_META = {
    "training": {
        "title": "HLS Burn-Scar Training Set",
        "description": "Training tiles from the NASA HLS burn-scar dataset",
        "event_type": "Wildfire / Burn Scar",
        "status": "complete",
    },
    "validation": {
        "title": "HLS Burn-Scar Validation Set",
        "description": "Validation tiles from the NASA HLS burn-scar dataset",
        "event_type": "Wildfire / Burn Scar",
        "status": "complete",
    },
}

# ── App ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Sentinel Satellite Fire / Burn-Scar Detection API",
    description="Serves Prithvi-100M burn-scar model predictions, GeoJSON polygons, "
                "HLS satellite imagery, and detection statistics.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ─────────────────────────────────────────────────────────────

def _get_splits():
    """Return sorted list of splits that have predictions."""
    if not os.path.isdir(FIRE_OUTPUT_ROOT):
        return []
    return sorted(
        d for d in os.listdir(FIRE_OUTPUT_ROOT)
        if os.path.isdir(os.path.join(FIRE_OUTPUT_ROOT, d, "geotiffs"))
    )


def _get_tiles(split: str):
    """Return sorted list of tile IDs for a split."""
    geo_dir = os.path.join(FIRE_OUTPUT_ROOT, split, "geotiffs")
    if not os.path.isdir(geo_dir):
        return []
    pred_files = glob.glob(os.path.join(geo_dir, f"*{PRED_SUFFIX}"))
    tiles = []
    for f in pred_files:
        fname = os.path.basename(f)
        # tile_id = filename without PRED_SUFFIX
        tile_id = fname[: -len(PRED_SUFFIX)]
        tiles.append(tile_id)
    return sorted(tiles)


def _pred_path(split: str, tile: str) -> str:
    """Path to burn-scar prediction GeoTIFF."""
    return os.path.join(FIRE_OUTPUT_ROOT, split, "geotiffs",
                        f"{tile}{PRED_SUFFIX}")


def _img_path(split: str, tile: str) -> str:
    """Path to original HLS merged tile (6-band)."""
    p = os.path.join(FIRE_DATA_ROOT, split, f"{tile}{IMG_SUFFIX}")
    return p if os.path.exists(p) else ""


def _gt_mask_path(split: str, tile: str) -> str:
    """Path to ground-truth burn-scar mask."""
    p = os.path.join(FIRE_DATA_ROOT, split, f"{tile}{MASK_SUFFIX}")
    return p if os.path.exists(p) else ""


def _overlay_path(split: str, tile: str) -> str:
    """Path to pre-rendered overlay image."""
    p = os.path.join(FIRE_OUTPUT_ROOT, split, "overlays", f"{tile}_overlay.jpg")
    return p if os.path.exists(p) else ""


def _nbr_img_path(split: str, tile: str) -> str:
    """Path to pre-rendered NBR heatmap image."""
    p = os.path.join(FIRE_OUTPUT_ROOT, split, "nbr", f"{tile}_nbr.jpg")
    return p if os.path.exists(p) else ""


def _tile_bounds_4326(tif_path: str):
    """Return [west, south, east, north] in EPSG:4326."""
    with rasterio.open(tif_path) as src:
        return list(transform_bounds(src.crs, "EPSG:4326", *src.bounds))


def _tile_center_4326(tif_path: str):
    b = _tile_bounds_4326(tif_path)
    return [(b[1] + b[3]) / 2, (b[0] + b[2]) / 2]  # [lat, lng]


def _normalize_band(band):
    """Percentile-clip + min-max normalize → [0,1]."""
    valid = band[~np.isnan(band) & (band > 0)]
    if valid.size == 0:
        return np.zeros_like(band, dtype=np.float32)
    p2, p98 = np.percentile(valid, [2, 98])
    band = np.clip(band, p2, p98).astype(np.float32)
    lo, hi = band.min(), band.max()
    if hi - lo < 1e-6:
        return np.zeros_like(band, dtype=np.float32)
    return (band - lo) / (hi - lo)


def _load_hls_rgb(tif_path: str) -> np.ndarray:
    """Return uint8 True-Color RGB [H,W,3] from 6-band HLS tile."""
    with rasterio.open(tif_path) as src:
        data = src.read().astype(np.float32)
    r = _normalize_band(data[B_RED])
    g = _normalize_band(data[B_GREEN])
    b = _normalize_band(data[B_BLUE])
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)


def _load_hls_false_color(tif_path: str) -> np.ndarray:
    """Return uint8 False-Color [H,W,3] (SWIR2, NIR, Red) — burn scars appear bright."""
    with rasterio.open(tif_path) as src:
        data = src.read().astype(np.float32)
    r = _normalize_band(data[B_SWIR2])
    g = _normalize_band(data[B_NIR])
    b = _normalize_band(data[B_RED])
    fc = np.stack([r, g, b], axis=-1)
    return (fc * 255).astype(np.uint8)


def _compute_nbr(tif_path: str) -> np.ndarray:
    """Compute NBR = (NIR - SWIR2) / (NIR + SWIR2). Returns float32 in [-1, 1]."""
    with rasterio.open(tif_path) as src:
        data = src.read().astype(np.float32)
    nir = data[B_NIR]
    swir2 = data[B_SWIR2]
    denom = nir + swir2
    denom[denom == 0] = 1e-10
    nbr = (nir - swir2) / denom
    return np.clip(nbr, -1, 1)


def _nbr_to_rgb(nbr: np.ndarray) -> np.ndarray:
    """Convert NBR float array to a colored heatmap uint8 [H,W,3].
    Low/negative = red/orange (burn), high = green (healthy)."""
    # Normalize from [-1, 1] to [0, 1]
    norm = (nbr + 1.0) / 2.0
    norm = np.clip(norm, 0, 1)
    # Simple colormap: red(0) → yellow(0.3) → green(0.6) → dark-green(1.0)
    rgb = np.zeros((*nbr.shape, 3), dtype=np.float32)
    # Red channel: high for low NBR
    rgb[:, :, 0] = np.clip(1.0 - norm * 2.0, 0, 1)
    # Green channel: high for high NBR
    rgb[:, :, 1] = np.clip(norm * 1.5 - 0.2, 0, 1)
    # Blue: low everywhere
    rgb[:, :, 2] = np.clip(0.3 - np.abs(norm - 0.5), 0, 1) * 0.5
    return (rgb * 255).astype(np.uint8)


def _load_burn_pred(tif_path: str) -> np.ndarray:
    """Load burn-scar prediction → binary mask (True=burn)."""
    with rasterio.open(tif_path) as src:
        d = src.read(1)
    return np.nan_to_num(d, nan=0.0) > 0


def _load_gt_mask(tif_path: str) -> Optional[np.ndarray]:
    """Load ground-truth mask → binary (True=burn). Returns None if path empty."""
    if not tif_path or not os.path.exists(tif_path):
        return None
    with rasterio.open(tif_path) as src:
        raw = src.read(1)
    return (raw == 1).astype(bool)


def _mask_to_geojson(binary_mask, crs, transform, simplify_tolerance=15.0):
    """Vectorize a binary mask into GeoJSON FeatureCollection in EPSG:4326."""
    from shapely.geometry import shape, mapping
    from shapely.ops import unary_union
    from pyproj import Transformer

    mask_u8 = binary_mask.astype(np.uint8)
    mask_u8 = binary_opening(mask_u8, structure=np.ones((3, 3))).astype(np.uint8)

    features = []
    for geom, val in rio_shapes(mask_u8, mask=mask_u8 == 1, transform=transform):
        if val == 1:
            features.append(shape(geom))

    if not features:
        return {"type": "FeatureCollection", "features": []}

    merged = unary_union(features)
    if simplify_tolerance > 0:
        merged = merged.simplify(simplify_tolerance, preserve_topology=True)

    transformer = Transformer.from_crs(str(crs), "EPSG:4326", always_xy=True)

    def _reproject(geom_map):
        if geom_map["type"] == "Polygon":
            geom_map["coordinates"] = [
                [list(transformer.transform(x, y)) for x, y in ring]
                for ring in geom_map["coordinates"]
            ]
        elif geom_map["type"] == "MultiPolygon":
            geom_map["coordinates"] = [
                [[list(transformer.transform(x, y)) for x, y in ring] for ring in poly]
                for poly in geom_map["coordinates"]
            ]
        return geom_map

    geom_dict = _reproject(mapping(merged))

    return {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "geometry": geom_dict,
            "properties": {},
        }],
    }


def _compute_tile_stats(split: str, tile: str):
    """Compute burn-scar stats for a tile."""
    pred_p = _pred_path(split, tile)
    gt_p = _gt_mask_path(split, tile)

    burn_pred = _load_burn_pred(pred_p)
    gt_mask = _load_gt_mask(gt_p)

    n_pred = int(burn_pred.sum())
    total_pixels = burn_pred.size
    n_gt = int(gt_mask.sum()) if gt_mask is not None else None

    px_to_km2 = lambda n: round(n * (PIXEL_RES_M ** 2) / 1e6, 6)

    pred_km2 = px_to_km2(n_pred)
    gt_km2 = px_to_km2(n_gt) if n_gt is not None else None

    # Accuracy metrics (if GT available)
    accuracy = None
    precision = None
    recall = None
    f1 = None
    iou = None
    if gt_mask is not None:
        p = burn_pred.flatten().astype(int)
        g = gt_mask.flatten().astype(int)
        tp = int(np.sum((p == 1) & (g == 1)))
        fp = int(np.sum((p == 1) & (g == 0)))
        fn = int(np.sum((p == 0) & (g == 1)))
        tn = int(np.sum((p == 0) & (g == 0)))
        accuracy = round((tp + tn) / max(tp + tn + fp + fn, 1), 6)
        precision = round(tp / max(tp + fp, 1), 6)
        recall = round(tp / max(tp + fn, 1), 6)
        f1 = round(2 * tp / max(2 * tp + fp + fn, 1), 6)
        iou = round(tp / max(tp + fp + fn, 1), 6)

    return {
        "pred_burn_pixels": n_pred,
        "pred_burn_km2": pred_km2,
        "gt_burn_pixels": n_gt,
        "gt_burn_km2": gt_km2,
        "total_pixels": total_pixels,
        "burn_fraction": round(n_pred / max(total_pixels, 1), 6),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
    }


def _load_metrics_csv(split: str):
    """Load per-tile metrics from the inference pipeline CSV."""
    csv_path = os.path.join(FIRE_OUTPUT_ROOT, split, f"{split}_metrics.csv")
    if not os.path.exists(csv_path):
        return {}
    lut = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            tid = row.get("tile_id", "")
            if tid and tid != "AVERAGE":
                lut[tid] = {k: _try_float(v) for k, v in row.items()}
    return lut


def _try_float(v):
    try:
        return float(v)
    except (ValueError, TypeError):
        return v


# ── Endpoints ───────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "status": "ok",
        "api": "Sentinel Satellite Fire / Burn-Scar Detection (Prithvi-100M)",
        "version": "1.0.0",
    }


@app.get("/api/summary")
def global_summary():
    """Dashboard-level summary stats across all splits."""
    splits = _get_splits()
    total_tiles = 0
    total_pred_km2 = 0.0
    total_gt_km2 = 0.0
    all_f1 = []

    for sp in splits:
        tiles = _get_tiles(sp)
        total_tiles += len(tiles)
        for t in tiles:
            try:
                s = _compute_tile_stats(sp, t)
                total_pred_km2 += s["pred_burn_km2"]
                if s["gt_burn_km2"] is not None:
                    total_gt_km2 += s["gt_burn_km2"]
                if s["f1"] is not None:
                    all_f1.append(s["f1"])
            except Exception:
                continue

    avg_f1 = round(sum(all_f1) / len(all_f1), 4) if all_f1 else None

    return {
        "total_splits": len(splits),
        "total_tiles": total_tiles,
        "total_predicted_burn_km2": round(total_pred_km2, 4),
        "total_gt_burn_km2": round(total_gt_km2, 4),
        "avg_f1": avg_f1,
        "splits": splits,
        "model": "Prithvi-100M (HLS Burn-Scar Fine-Tuned)",
        "pixel_resolution_m": PIXEL_RES_M,
    }


@app.get("/api/splits")
def list_splits():
    """List all splits with coordinates and aggregate stats."""
    result = []
    for sp in _get_splits():
        tiles = _get_tiles(sp)
        if not tiles:
            continue

        # Get geo-info from first tile prediction
        sample = _pred_path(sp, tiles[0])
        center = _tile_center_4326(sample) if os.path.exists(sample) else [0, 0]

        # Bbox from all tiles
        all_bounds = []
        for t in tiles:
            p = _pred_path(sp, t)
            if os.path.exists(p):
                all_bounds.append(_tile_bounds_4326(p))

        bbox = [
            min(b[0] for b in all_bounds),
            min(b[1] for b in all_bounds),
            max(b[2] for b in all_bounds),
            max(b[3] for b in all_bounds),
        ] if all_bounds else [0, 0, 0, 0]

        # Aggregate stats
        total_pred_km2 = 0.0
        total_gt_km2 = 0.0
        f1_vals = []
        for t in tiles:
            try:
                s = _compute_tile_stats(sp, t)
                total_pred_km2 += s["pred_burn_km2"]
                if s["gt_burn_km2"] is not None:
                    total_gt_km2 += s["gt_burn_km2"]
                if s["f1"] is not None:
                    f1_vals.append(s["f1"])
            except Exception:
                continue

        # Load CSV metrics for faster aggregate
        metrics = _load_metrics_csv(sp)
        if not f1_vals and metrics:
            f1_vals = [
                m.get("f1", 0)
                for m in metrics.values()
                if isinstance(m.get("f1"), (int, float))
            ]

        avg_f1 = round(sum(f1_vals) / len(f1_vals), 4) if f1_vals else None

        meta = SPLIT_META.get(sp, {})

        result.append({
            "split": sp,
            "title": meta.get("title", f"Burn-Scar {sp.title()}"),
            "description": meta.get("description", ""),
            "event_type": meta.get("event_type", "Wildfire / Burn Scar"),
            "status": meta.get("status", "complete"),
            "latitude": center[0],
            "longitude": center[1],
            "tile_count": len(tiles),
            "center": center,
            "bbox": bbox,
            "total_predicted_burn_km2": round(total_pred_km2, 4),
            "total_gt_burn_km2": round(total_gt_km2, 4),
            "avg_f1": avg_f1,
        })

    return result


@app.get("/api/splits/{split}")
def split_detail(split: str):
    """Detailed info for a single split with per-tile data."""
    tiles = _get_tiles(split)
    if not tiles:
        raise HTTPException(404, f"Split '{split}' not found or has no predictions")

    metrics = _load_metrics_csv(split)
    tile_list = []
    all_bounds = []

    for t in tiles:
        p = _pred_path(split, t)
        bounds = _tile_bounds_4326(p) if os.path.exists(p) else None
        if bounds:
            all_bounds.append(bounds)
        try:
            stats = _compute_tile_stats(split, t)
        except Exception:
            stats = {}
        tile_metrics = metrics.get(t, {})

        tile_list.append({
            "tile": t,
            "bounds": bounds,
            "center": [(bounds[1] + bounds[3]) / 2,
                        (bounds[0] + bounds[2]) / 2] if bounds else None,
            "stats": stats,
            "metrics": {
                "f1": tile_metrics.get("f1", None),
                "iou": tile_metrics.get("iou", None),
                "precision": tile_metrics.get("precision", None),
                "recall": tile_metrics.get("recall", None),
                "accuracy": tile_metrics.get("accuracy", None),
            },
            "has_rgb": bool(_img_path(split, t)),
            "has_gt": bool(_gt_mask_path(split, t)),
            "has_overlay": bool(_overlay_path(split, t)),
            "has_nbr": bool(_nbr_img_path(split, t)),
        })

    bbox = [
        min(b[0] for b in all_bounds),
        min(b[1] for b in all_bounds),
        max(b[2] for b in all_bounds),
        max(b[3] for b in all_bounds),
    ] if all_bounds else None

    meta = SPLIT_META.get(split, {})

    return {
        "split": split,
        "title": meta.get("title", f"Burn-Scar {split.title()}"),
        "description": meta.get("description", ""),
        "event_type": meta.get("event_type", "Wildfire / Burn Scar"),
        "tile_count": len(tiles),
        "bbox": bbox,
        "tiles": tile_list,
    }


@app.get("/api/splits/{split}/tiles/{tile}/bounds")
def tile_bounds(split: str, tile: str):
    """Return tile bounds in EPSG:4326 [west, south, east, north]."""
    p = _pred_path(split, tile)
    if not os.path.exists(p):
        raise HTTPException(404, f"Tile '{tile}' not found in split '{split}'")
    return {"tile": tile, "split": split, "bounds": _tile_bounds_4326(p)}


@app.get("/api/splits/{split}/tiles/{tile}/geojson")
def tile_geojson(
    split: str,
    tile: str,
    layer: str = Query("burn", pattern="^(burn|gt|accuracy)$"),
):
    """GeoJSON polygons for a tile's burn-scar detection.

    Layers:
        burn      → model prediction (burn scar)
        gt        → ground-truth burn-scar mask
        accuracy  → combined: TP (green), FP (red), FN (orange)
    """
    pred_p = _pred_path(split, tile)
    if not os.path.exists(pred_p):
        raise HTTPException(404, f"Tile '{tile}' not found in split '{split}'")

    with rasterio.open(pred_p) as src:
        crs = src.crs
        transform_ = src.transform

    burn_pred = _load_burn_pred(pred_p)
    gt_mask = _load_gt_mask(_gt_mask_path(split, tile))

    if layer == "burn":
        gj = _mask_to_geojson(burn_pred, crs, transform_)
        for f in gj["features"]:
            f["properties"]["layer"] = "burn_scar"
            f["properties"]["class"] = "burn_scar"
            f["properties"]["color"] = "#FF6600"
        return gj

    elif layer == "gt":
        if gt_mask is None:
            return {"type": "FeatureCollection", "features": []}
        gj = _mask_to_geojson(gt_mask, crs, transform_)
        for f in gj["features"]:
            f["properties"]["layer"] = "ground_truth"
            f["properties"]["class"] = "ground_truth"
            f["properties"]["color"] = "#FF0000"
        return gj

    elif layer == "accuracy":
        features = []
        if gt_mask is not None:
            tp = burn_pred & gt_mask
            fp = burn_pred & ~gt_mask
            fn = ~burn_pred & gt_mask

            # True positives — green
            if tp.any():
                gj_tp = _mask_to_geojson(tp, crs, transform_)
                for f in gj_tp["features"]:
                    f["properties"]["layer"] = "true_positive"
                    f["properties"]["class"] = "true_positive"
                    f["properties"]["color"] = "#00CC44"
                features.extend(gj_tp["features"])

            # False positives — red
            if fp.any():
                gj_fp = _mask_to_geojson(fp, crs, transform_)
                for f in gj_fp["features"]:
                    f["properties"]["layer"] = "false_positive"
                    f["properties"]["class"] = "false_positive"
                    f["properties"]["color"] = "#FF004D"
                features.extend(gj_fp["features"])

            # False negatives — orange
            if fn.any():
                gj_fn = _mask_to_geojson(fn, crs, transform_)
                for f in gj_fn["features"]:
                    f["properties"]["layer"] = "false_negative"
                    f["properties"]["class"] = "false_negative"
                    f["properties"]["color"] = "#FFA500"
                features.extend(gj_fn["features"])
        else:
            # No GT: just show prediction
            gj = _mask_to_geojson(burn_pred, crs, transform_)
            for f in gj["features"]:
                f["properties"]["layer"] = "burn_scar"
                f["properties"]["class"] = "burn_scar"
                f["properties"]["color"] = "#FF6600"
            features.extend(gj["features"])

        return {"type": "FeatureCollection", "features": features}


@app.get("/api/splits/{split}/geojson")
def split_geojson(
    split: str,
    layer: str = Query("burn", pattern="^(burn|gt|accuracy)$"),
):
    """Merged GeoJSON for ALL tiles in a split."""
    tiles = _get_tiles(split)
    if not tiles:
        raise HTTPException(404, f"Split '{split}' not found")

    all_features = []
    for t in tiles:
        try:
            gj = tile_geojson(split, t, layer=layer)
            stats = _compute_tile_stats(split, t)
            for f in gj["features"]:
                f["properties"]["tile"] = t
                f["properties"]["pred_burn_km2"] = stats.get("pred_burn_km2", 0)
                f["properties"]["f1"] = stats.get("f1")
            all_features.extend(gj["features"])
        except Exception:
            continue

    return {"type": "FeatureCollection", "features": all_features}


@app.get("/api/splits/{split}/tiles/{tile}/rgb")
def tile_rgb(split: str, tile: str):
    """Serve HLS True-Color RGB as PNG."""
    img_p = _img_path(split, tile)
    if not img_p:
        raise HTTPException(404, f"HLS imagery not found for {split}/{tile}")

    rgb = _load_hls_rgb(img_p)

    # Include bounds in header
    pred_p = _pred_path(split, tile)
    bounds = _tile_bounds_4326(pred_p) if os.path.exists(pred_p) else None

    img = Image.fromarray(rgb)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    headers = {"Cache-Control": "public, max-age=86400"}
    if bounds:
        headers["X-Tile-Bounds"] = ",".join(str(b) for b in bounds)
    return StreamingResponse(buf, media_type="image/png", headers=headers)


@app.get("/api/splits/{split}/tiles/{tile}/false_color")
def tile_false_color(split: str, tile: str):
    """Serve HLS False-Color (SWIR2, NIR, Red) as PNG — burn scars appear bright."""
    img_p = _img_path(split, tile)
    if not img_p:
        raise HTTPException(404, f"HLS imagery not found for {split}/{tile}")

    fc = _load_hls_false_color(img_p)

    pred_p = _pred_path(split, tile)
    bounds = _tile_bounds_4326(pred_p) if os.path.exists(pred_p) else None

    img = Image.fromarray(fc)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    headers = {"Cache-Control": "public, max-age=86400"}
    if bounds:
        headers["X-Tile-Bounds"] = ",".join(str(b) for b in bounds)
    return StreamingResponse(buf, media_type="image/png", headers=headers)


@app.get("/api/splits/{split}/tiles/{tile}/nbr")
def tile_nbr(split: str, tile: str):
    """Serve NBR (Normalized Burn Ratio) heatmap as PNG.
    Low/negative values indicate burn scars."""
    # Try pre-rendered image first
    nbr_p = _nbr_img_path(split, tile)
    if nbr_p:
        with open(nbr_p, "rb") as f:
            buf = io.BytesIO(f.read())
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/jpeg",
                                 headers={"Cache-Control": "public, max-age=86400"})

    # Compute on the fly from HLS tile
    img_p = _img_path(split, tile)
    if not img_p:
        raise HTTPException(404, f"HLS imagery not found for {split}/{tile}")

    nbr = _compute_nbr(img_p)
    nbr_rgb = _nbr_to_rgb(nbr)

    img = Image.fromarray(nbr_rgb)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png",
                             headers={"Cache-Control": "public, max-age=86400"})


@app.get("/api/splits/{split}/tiles/{tile}/mask")
def tile_mask_png(
    split: str, tile: str,
    overlay: bool = Query(False),
):
    """Serve burn-scar mask as colored transparent PNG.
    If overlay=true, composite onto HLS RGB."""
    pred_p = _pred_path(split, tile)
    if not os.path.exists(pred_p):
        raise HTTPException(404, f"Tile '{tile}' not found in split '{split}'")

    burn = _load_burn_pred(pred_p)

    if overlay:
        img_p = _img_path(split, tile)
        if img_p:
            rgb = _load_hls_rgb(img_p).astype(np.float32)
        else:
            rgb = np.zeros((*burn.shape, 3), dtype=np.float32)

        # Orange overlay for burn scar
        color = np.array([255, 102, 0], dtype=np.float32)
        alpha = 0.50
        for c in range(3):
            rgb[:, :, c] = np.where(
                burn,
                rgb[:, :, c] * (1 - alpha) + color[c] * alpha,
                rgb[:, :, c],
            )
        img = Image.fromarray(rgb.clip(0, 255).astype(np.uint8))
    else:
        # Transparent RGBA
        rgba = np.zeros((*burn.shape, 4), dtype=np.uint8)
        rgba[burn] = [255, 102, 0, 180]  # orange semi-transparent
        img = Image.fromarray(rgba)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


@app.get("/api/splits/{split}/tiles/{tile}/overlay")
def tile_overlay(split: str, tile: str):
    """Composite PNG: HLS RGB + orange(predicted burn) + red(GT burn) + accuracy map.
    Serves pre-rendered overlay if available, otherwise generates on the fly."""
    # Try pre-rendered
    ov_p = _overlay_path(split, tile)
    if ov_p:
        with open(ov_p, "rb") as f:
            buf = io.BytesIO(f.read())
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/jpeg",
                                 headers={"Cache-Control": "public, max-age=86400"})

    # Generate on the fly
    pred_p = _pred_path(split, tile)
    if not os.path.exists(pred_p):
        raise HTTPException(404, f"Tile '{tile}' not found")

    burn_pred = _load_burn_pred(pred_p)
    gt_mask = _load_gt_mask(_gt_mask_path(split, tile))

    # Get background image
    img_p = _img_path(split, tile)
    if img_p:
        rgb = _load_hls_rgb(img_p).astype(np.float32)
    else:
        raise HTTPException(404, "No HLS imagery available")

    # Clean up prediction
    burn_clean = binary_opening(burn_pred, structure=np.ones((3, 3)))

    if gt_mask is not None:
        gt_clean = binary_opening(gt_mask, structure=np.ones((3, 3)))
        tp = burn_clean & gt_clean
        fp = burn_clean & ~gt_clean
        fn = ~burn_clean & gt_clean

        # Green for TP
        for c, v in enumerate([0, 204, 68]):
            rgb[:, :, c] = np.where(tp, rgb[:, :, c] * 0.4 + v * 0.6, rgb[:, :, c])
        # Red for FP
        for c, v in enumerate([255, 0, 77]):
            rgb[:, :, c] = np.where(fp, rgb[:, :, c] * 0.4 + v * 0.6, rgb[:, :, c])
        # Orange for FN
        for c, v in enumerate([255, 165, 0]):
            rgb[:, :, c] = np.where(fn, rgb[:, :, c] * 0.4 + v * 0.6, rgb[:, :, c])
    else:
        # No GT: just overlay prediction in orange
        for c, v in enumerate([255, 102, 0]):
            rgb[:, :, c] = np.where(
                burn_clean,
                rgb[:, :, c] * 0.45 + v * 0.55,
                rgb[:, :, c],
            )

    img = Image.fromarray(rgb.clip(0, 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


@app.get("/api/splits/{split}/tiles/{tile}/gt_mask")
def tile_gt_mask_png(
    split: str, tile: str,
    overlay: bool = Query(False),
):
    """Serve ground-truth burn-scar mask as colored PNG."""
    gt_p = _gt_mask_path(split, tile)
    gt_mask = _load_gt_mask(gt_p)
    if gt_mask is None:
        raise HTTPException(404, f"Ground truth mask not found for {split}/{tile}")

    if overlay:
        img_p = _img_path(split, tile)
        if img_p:
            rgb = _load_hls_rgb(img_p).astype(np.float32)
        else:
            rgb = np.zeros((*gt_mask.shape, 3), dtype=np.float32)

        color = np.array([255, 0, 0], dtype=np.float32)
        alpha = 0.50
        for c in range(3):
            rgb[:, :, c] = np.where(
                gt_mask,
                rgb[:, :, c] * (1 - alpha) + color[c] * alpha,
                rgb[:, :, c],
            )
        img = Image.fromarray(rgb.clip(0, 255).astype(np.uint8))
    else:
        rgba = np.zeros((*gt_mask.shape, 4), dtype=np.uint8)
        rgba[gt_mask] = [255, 0, 0, 180]
        img = Image.fromarray(rgba)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


@app.get("/api/splits/{split}/tiles/{tile}/stats")
def tile_stats(split: str, tile: str):
    """Per-tile burn-scar stats + AI model metrics."""
    pred_p = _pred_path(split, tile)
    if not os.path.exists(pred_p):
        raise HTTPException(404, f"Tile '{tile}' not found in split '{split}'")

    stats = _compute_tile_stats(split, tile)
    bounds = _tile_bounds_4326(pred_p)
    stats["bounds"] = bounds
    stats["center"] = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
    stats["tile"] = tile
    stats["split"] = split

    # Add CSV metrics if available
    metrics = _load_metrics_csv(split)
    m = metrics.get(tile, {})
    if m:
        stats["csv_f1"] = m.get("f1")
        stats["csv_iou"] = m.get("iou")
        stats["csv_precision"] = m.get("precision")
        stats["csv_recall"] = m.get("recall")
        stats["csv_accuracy"] = m.get("accuracy")
        stats["pred_km2_csv"] = m.get("pred_km2")
        stats["gt_km2_csv"] = m.get("gt_km2")
        stats["time_s"] = m.get("time_s")

    return stats
