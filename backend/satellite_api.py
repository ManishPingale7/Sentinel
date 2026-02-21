"""
Satellite Flood Detection API — ai4g-flood Model
==================================================
Serves flood prediction data from the ai4g-flood SAR change-detection model.
Reads GeoTIFFs from OUTPUTS/ai4g_flood/<EVENT>/geotiffs/ and serves:

  - Event list with geo-coordinates, stats, metadata
  - GeoJSON flood polygons (existing water + new flood) per tile/event
  - Sentinel-2 RGB PNGs (before/during)
  - Sentinel-1 SAR grayscale PNGs
  - Flood mask overlays
  - Per-tile and per-event statistics
  - Tile bounds for imagery draping on CesiumJS globe

Run:
    uvicorn backend.satellite_api:app --reload --port 8001

Endpoints:
    GET  /api/summary                                      → global stats
    GET  /api/events                                       → list events
    GET  /api/events/{cems_id}                             → event detail + tiles
    GET  /api/events/{cems_id}/geojson?layer=change        → merged GeoJSON
    GET  /api/events/{cems_id}/tiles/{tile}/geojson        → per-tile GeoJSON
    GET  /api/events/{cems_id}/tiles/{tile}/rgb/{phase}    → S2 RGB PNG
    GET  /api/events/{cems_id}/tiles/{tile}/sar/{phase}    → S1 VV grayscale PNG
    GET  /api/events/{cems_id}/tiles/{tile}/mask/{phase}   → colored mask PNG
    GET  /api/events/{cems_id}/tiles/{tile}/change_overlay → composite PNG
    GET  /api/events/{cems_id}/tiles/{tile}/stats          → per-tile stats
    GET  /api/events/{cems_id}/tiles/{tile}/bounds         → [w, s, e, n]
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

CEMS_DATA_ROOT = os.environ.get(
    "CEMS_DATA_ROOT",
    os.path.join(_BASE_DIR, "DATA", "CEMS"),
)
AI4G_OUTPUT_ROOT = os.environ.get(
    "AI4G_OUTPUT_ROOT",
    os.path.join(_BASE_DIR, "OUTPUTS", "ai4g_flood"),
)
PIXEL_RES_M = 10.0

# Event metadata (CEMS activation info)
EVENT_META = {
    "EMSR766": {
        "title": "Floods in Croatia & Serbia",
        "description": "Severe flooding along Sava and Danube rivers caused by persistent heavy rainfall",
        "country": "Croatia / Serbia",
        "event_type": "Flood",
        "activation_time": "2024-09-22T00:00:00Z",
        "status": "active",
    },
    "EMSR771": {
        "title": "Flash Floods in Emilia-Romagna, Italy",
        "description": "Intense rainfall caused flash flooding across the Emilia-Romagna region",
        "country": "Italy",
        "event_type": "Flood",
        "activation_time": "2024-10-19T00:00:00Z",
        "status": "active",
    },
    "EMSR773": {
        "title": "Floods in Valencia Region, Spain",
        "description": "Catastrophic DANA storm caused devastating flash floods across Valencia, Castilla-La Mancha",
        "country": "Spain",
        "event_type": "Flood",
        "activation_time": "2024-10-29T00:00:00Z",
        "status": "active",
    },
}

# ── App ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Sentinel Satellite Flood Detection API",
    description="Serves ai4g-flood SAR model predictions, GeoJSON polygons, "
                "satellite imagery, and change-detection statistics.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ─────────────────────────────────────────────────────────────

def _get_cems_ids():
    """Return sorted list of CEMS event IDs that have ai4g predictions."""
    if not os.path.isdir(AI4G_OUTPUT_ROOT):
        return []
    return sorted(
        d for d in os.listdir(AI4G_OUTPUT_ROOT)
        if os.path.isdir(os.path.join(AI4G_OUTPUT_ROOT, d, "geotiffs"))
    )


def _get_tiles(cems_id: str):
    """Return sorted list of tile IDs for a CEMS event."""
    geo_dir = os.path.join(AI4G_OUTPUT_ROOT, cems_id, "geotiffs")
    if not os.path.isdir(geo_dir):
        return []
    pred_files = glob.glob(os.path.join(geo_dir, "*_flood_pred.tif"))
    return sorted({os.path.basename(f).split("_")[0] for f in pred_files})


def _pred_path(cems_id: str, tile: str) -> str:
    """Path to flood prediction GeoTIFF."""
    return os.path.join(AI4G_OUTPUT_ROOT, cems_id, "geotiffs",
                        f"{tile}_flood_pred.tif")


def _water_path(cems_id: str, tile: str, phase: str) -> str:
    """Path to before/during water mask."""
    return os.path.join(AI4G_OUTPUT_ROOT, cems_id, "geotiffs",
                        f"{tile}_{phase}_water.tif")


def _s2_path(cems_id: str, tile: str, phase: str) -> str:
    """Path to original Sentinel-2 tile."""
    d = os.path.join(CEMS_DATA_ROOT, cems_id, f"s2_{phase}_flood")
    p = os.path.join(d, f"{tile}_s2_{phase}_flood.tif")
    return p if os.path.exists(p) else ""


def _s1_path(cems_id: str, tile: str, phase: str) -> str:
    """Path to original Sentinel-1 tile."""
    d = os.path.join(CEMS_DATA_ROOT, cems_id, f"s1_{phase}_flood")
    p = os.path.join(d, f"{tile}_s1_{phase}_flood.tif")
    return p if os.path.exists(p) else ""


def _tile_bounds_4326(tif_path: str):
    """Return [west, south, east, north] in EPSG:4326."""
    with rasterio.open(tif_path) as src:
        return list(transform_bounds(src.crs, "EPSG:4326", *src.bounds))


def _tile_center_4326(tif_path: str):
    b = _tile_bounds_4326(tif_path)
    return [(b[1] + b[3]) / 2, (b[0] + b[2]) / 2]  # [lat, lng]


def _normalize_band(band):
    valid = band[band > 0]
    if valid.size == 0:
        return np.zeros_like(band, dtype=np.float32)
    p2, p98 = np.percentile(valid, [2, 98])
    band = np.clip(band, p2, p98).astype(np.float32)
    lo, hi = band.min(), band.max()
    if hi - lo < 1e-6:
        return np.zeros_like(band, dtype=np.float32)
    return (band - lo) / (hi - lo)


def _load_s2_rgb(tif_path: str) -> np.ndarray:
    """Return uint8 RGB [H,W,3] from 8-band Sentinel-2."""
    with rasterio.open(tif_path) as src:
        red   = src.read(3).astype(np.float32)
        green = src.read(2).astype(np.float32)
        blue  = src.read(1).astype(np.float32)
    rgb = np.stack([_normalize_band(red),
                    _normalize_band(green),
                    _normalize_band(blue)], axis=-1)
    return (rgb * 255).astype(np.uint8)


def _load_s1_gray(tif_path: str) -> np.ndarray:
    """Return uint8 grayscale [H,W,3] from Sentinel-1 VV band."""
    with rasterio.open(tif_path) as src:
        vv = src.read(1).astype(np.float32)
    vv_scaled = np.clip(np.nan_to_num(vv * 2.0 + 135.0, nan=0), 0, 255) / 255.0
    vv_u8 = (vv_scaled * 255).astype(np.uint8)
    return np.stack([vv_u8, vv_u8, vv_u8], axis=-1)


def _load_flood_pred(tif_path: str) -> np.ndarray:
    """Load flood prediction → binary mask (True=flood)."""
    with rasterio.open(tif_path) as src:
        d = src.read(1)
    return np.nan_to_num(d, nan=0.0) > 0


def _load_water_mask(tif_path: str) -> np.ndarray:
    """Load water mask → binary (True=water)."""
    if not os.path.exists(tif_path):
        return None
    with rasterio.open(tif_path) as src:
        return src.read(1).astype(bool)


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


def _compute_tile_stats(cems_id: str, tile: str):
    """Compute flood stats for a tile from the ai4g predictions."""
    pred_p = _pred_path(cems_id, tile)
    bef_p = _water_path(cems_id, tile, "before")
    dur_p = _water_path(cems_id, tile, "during")

    flood_pred = _load_flood_pred(pred_p)
    before_water = _load_water_mask(bef_p)
    during_water = _load_water_mask(dur_p)

    n_flood = int(flood_pred.sum())
    n_before = int(before_water.sum()) if before_water is not None else 0
    n_during = int(during_water.sum()) if during_water is not None else 0

    # Combine: total during = during_water OR flood_pred
    if during_water is not None:
        total_during = during_water | flood_pred
    else:
        total_during = flood_pred

    n_total_during = int(total_during.sum())

    # New flood = in total_during but NOT in before
    if before_water is not None:
        new_flood = total_during & ~before_water
    else:
        new_flood = total_during
    n_new = int(new_flood.sum())

    px_to_km2 = lambda n: round(n * (PIXEL_RES_M ** 2) / 1e6, 6)

    before_km2 = px_to_km2(n_before)
    during_km2 = px_to_km2(n_total_during)
    new_flood_km2 = px_to_km2(n_new)
    pct_increase = round(100.0 * n_new / n_before, 1) if n_before > 0 else 0.0

    return {
        "before_km2": before_km2,
        "during_km2": during_km2,
        "new_flood_km2": new_flood_km2,
        "pct_increase": pct_increase,
        "flood_pred_pixels": n_flood,
        "ai4g_pred_km2": px_to_km2(n_flood),
    }


def _load_metrics_csv(cems_id: str):
    """Load per-tile metrics from the inference pipeline CSV."""
    csv_path = os.path.join(AI4G_OUTPUT_ROOT, cems_id, f"{cems_id}_metrics.csv")
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
    return {"status": "ok", "api": "Sentinel Satellite Flood Detection (ai4g)", "version": "2.0.0"}


@app.get("/api/summary")
def global_summary():
    """Dashboard-level summary stats."""
    events = _get_cems_ids()
    total_tiles = 0
    total_new_km2 = 0.0
    total_during_km2 = 0.0
    all_f1 = []

    for cems_id in events:
        tiles = _get_tiles(cems_id)
        total_tiles += len(tiles)
        metrics = _load_metrics_csv(cems_id)
        for t in tiles:
            try:
                s = _compute_tile_stats(cems_id, t)
                total_new_km2 += s["new_flood_km2"]
                total_during_km2 += s["during_km2"]
            except Exception:
                continue
            m = metrics.get(t, {})
            if isinstance(m.get("f1"), (int, float)):
                all_f1.append(m["f1"])

    avg_f1 = round(sum(all_f1) / len(all_f1), 4) if all_f1 else None

    return {
        "total_events": len(events),
        "total_tiles": total_tiles,
        "total_new_flood_km2": round(total_new_km2, 4),
        "total_flooded_km2": round(total_during_km2, 4),
        "avg_f1": avg_f1,
        "events": events,
        "model": "ai4g-flood (SAR U-Net, MobileNetV2)",
        "pixel_resolution_m": PIXEL_RES_M,
    }


@app.get("/api/events")
def list_events():
    """List all CEMS flood events with coordinates and stats."""
    events = []
    for cems_id in _get_cems_ids():
        tiles = _get_tiles(cems_id)
        if not tiles:
            continue

        # Get geo-info from first tile
        sample = _pred_path(cems_id, tiles[0])
        center = _tile_center_4326(sample) if os.path.exists(sample) else [0, 0]

        # Bbox from all tiles
        all_bounds = []
        for t in tiles:
            p = _pred_path(cems_id, t)
            if os.path.exists(p):
                all_bounds.append(_tile_bounds_4326(p))

        bbox = [
            min(b[0] for b in all_bounds),
            min(b[1] for b in all_bounds),
            max(b[2] for b in all_bounds),
            max(b[3] for b in all_bounds),
        ] if all_bounds else [0, 0, 0, 0]

        # Aggregate stats
        total_new = 0.0
        total_during = 0.0
        for t in tiles:
            try:
                s = _compute_tile_stats(cems_id, t)
                total_new += s["new_flood_km2"]
                total_during += s["during_km2"]
            except Exception:
                continue

        # Load metrics
        metrics = _load_metrics_csv(cems_id)
        avg_f1 = 0.0
        if metrics:
            f1_vals = [m.get("f1", 0) for m in metrics.values() if isinstance(m.get("f1"), (int, float))]
            avg_f1 = round(sum(f1_vals) / len(f1_vals), 4) if f1_vals else 0.0

        meta = EVENT_META.get(cems_id, {})

        events.append({
            "event_id": cems_id,
            "cems_id": cems_id,
            "title": meta.get("title", f"Flood Event {cems_id}"),
            "description": meta.get("description", ""),
            "country": meta.get("country", "Unknown"),
            "event_type": meta.get("event_type", "Flood"),
            "activation_time": meta.get("activation_time", ""),
            "status": meta.get("status", "active"),
            "latitude": center[0],
            "longitude": center[1],
            "tile_count": len(tiles),
            "center": center,
            "bbox": bbox,
            "total_new_flood_km2": round(total_new, 4),
            "total_flooded_km2": round(total_during, 4),
            "avg_f1": avg_f1,
        })

    return events


@app.get("/api/events/{cems_id}")
def event_detail(cems_id: str):
    """Detailed info for a single event with per-tile data."""
    tiles = _get_tiles(cems_id)
    if not tiles:
        raise HTTPException(404, f"Event {cems_id} not found")

    metrics = _load_metrics_csv(cems_id)
    tile_list = []
    all_bounds = []

    for t in tiles:
        p = _pred_path(cems_id, t)
        bounds = _tile_bounds_4326(p) if os.path.exists(p) else None
        if bounds:
            all_bounds.append(bounds)
        try:
            stats = _compute_tile_stats(cems_id, t)
        except Exception:
            stats = {}
        tile_metrics = metrics.get(t, {})

        tile_list.append({
            "tile": t,
            "bounds": bounds,
            "center": [(bounds[1]+bounds[3])/2, (bounds[0]+bounds[2])/2] if bounds else None,
            "stats": stats,
            "metrics": {
                "f1": tile_metrics.get("f1", None),
                "iou": tile_metrics.get("iou", None),
                "precision": tile_metrics.get("precision", None),
                "recall": tile_metrics.get("recall", None),
                "accuracy": tile_metrics.get("accuracy", None),
            },
            "has_s2": bool(_s2_path(cems_id, t, "during")),
            "has_s1": bool(_s1_path(cems_id, t, "during")),
        })

    bbox = [
        min(b[0] for b in all_bounds),
        min(b[1] for b in all_bounds),
        max(b[2] for b in all_bounds),
        max(b[3] for b in all_bounds),
    ] if all_bounds else None

    meta = EVENT_META.get(cems_id, {})

    return {
        "event_id": cems_id,
        "cems_id": cems_id,
        "title": meta.get("title", f"Flood Event {cems_id}"),
        "description": meta.get("description", ""),
        "country": meta.get("country", "Unknown"),
        "event_type": meta.get("event_type", "Flood"),
        "tile_count": len(tiles),
        "bbox": bbox,
        "tiles": tile_list,
    }


@app.get("/api/events/{cems_id}/tiles/{tile}/bounds")
def tile_bounds(cems_id: str, tile: str):
    """Return tile bounds in EPSG:4326 [west, south, east, north]."""
    p = _pred_path(cems_id, tile)
    if not os.path.exists(p):
        raise HTTPException(404, f"Tile {tile} not found")
    return {"event_id": cems_id, "tile": tile, "bounds": _tile_bounds_4326(p)}


@app.get("/api/events/{cems_id}/tiles/{tile}/geojson")
def tile_geojson(
    cems_id: str,
    tile: str,
    layer: str = Query("change", pattern="^(flood|existing|new_flood|change)$"),
):
    """GeoJSON polygons for a tile's flood detection.

    Layers:
        flood      → ai4g model prediction (flood=255)
        existing   → water in before image
        new_flood  → flood_pred AND NOT before_water
        change     → combined: existing_water + new_flood as separate features
    """
    pred_p = _pred_path(cems_id, tile)
    if not os.path.exists(pred_p):
        raise HTTPException(404, f"Tile {tile} not found for {cems_id}")

    with rasterio.open(pred_p) as src:
        crs = src.crs
        transform = src.transform

    flood_pred = _load_flood_pred(pred_p)
    before_water = _load_water_mask(_water_path(cems_id, tile, "before"))
    during_water = _load_water_mask(_water_path(cems_id, tile, "during"))

    if layer == "flood":
        gj = _mask_to_geojson(flood_pred, crs, transform)
        for f in gj["features"]:
            f["properties"]["layer"] = "flood"
            f["properties"]["class"] = "flood"
        return gj

    elif layer == "existing":
        if before_water is None:
            return {"type": "FeatureCollection", "features": []}
        gj = _mask_to_geojson(before_water, crs, transform)
        for f in gj["features"]:
            f["properties"]["layer"] = "existing_water"
            f["properties"]["class"] = "existing_water"
        return gj

    elif layer == "new_flood":
        if before_water is not None:
            new = flood_pred & ~before_water
        else:
            new = flood_pred
        gj = _mask_to_geojson(new, crs, transform)
        for f in gj["features"]:
            f["properties"]["layer"] = "new_flood"
            f["properties"]["class"] = "new_flood"
        return gj

    elif layer == "change":
        features = []

        # Existing water
        if before_water is not None:
            if during_water is not None:
                existing = before_water & (during_water | flood_pred)
            else:
                existing = before_water & flood_pred
            gj_existing = _mask_to_geojson(existing, crs, transform)
            for f in gj_existing["features"]:
                f["properties"]["layer"] = "existing_water"
                f["properties"]["class"] = "existing_water"
                f["properties"]["color"] = "#00BFFF"
            features.extend(gj_existing["features"])

        # New flood
        if before_water is not None:
            new = flood_pred & ~before_water
        else:
            new = flood_pred
        gj_new = _mask_to_geojson(new, crs, transform)
        for f in gj_new["features"]:
            f["properties"]["layer"] = "new_flood"
            f["properties"]["class"] = "new_flood"
            f["properties"]["color"] = "#FF004D"
        features.extend(gj_new["features"])

        return {"type": "FeatureCollection", "features": features}


@app.get("/api/events/{cems_id}/geojson")
def event_geojson(
    cems_id: str,
    layer: str = Query("change", pattern="^(flood|existing|new_flood|change)$"),
):
    """Merged GeoJSON for ALL tiles in an event."""
    tiles = _get_tiles(cems_id)
    if not tiles:
        raise HTTPException(404, f"Event {cems_id} not found")

    all_features = []
    for t in tiles:
        try:
            gj = tile_geojson(cems_id, t, layer=layer)
            stats = _compute_tile_stats(cems_id, t)
            for f in gj["features"]:
                f["properties"]["tile"] = t
                f["properties"]["new_flood_km2"] = stats.get("new_flood_km2", 0)
                f["properties"]["pct_increase"] = stats.get("pct_increase", 0)
                f["properties"]["area_km2"] = stats.get("ai4g_pred_km2", 0)
            all_features.extend(gj["features"])
        except Exception:
            continue

    return {"type": "FeatureCollection", "features": all_features}


@app.get("/api/events/{cems_id}/tiles/{tile}/rgb/{phase}")
def tile_rgb(cems_id: str, tile: str, phase: str):
    """Serve Sentinel-2 RGB as PNG. Falls back to S1 grayscale."""
    if phase not in ("before", "during"):
        raise HTTPException(400, "Phase must be 'before' or 'during'")

    s2 = _s2_path(cems_id, tile, phase)
    if s2:
        rgb = _load_s2_rgb(s2)
    else:
        s1 = _s1_path(cems_id, tile, phase)
        if not s1:
            raise HTTPException(404, f"No imagery for {cems_id}/{tile}/{phase}")
        rgb = _load_s1_gray(s1)

    # Also return bounds in header
    pred_p = _pred_path(cems_id, tile)
    bounds = _tile_bounds_4326(pred_p) if os.path.exists(pred_p) else None

    img = Image.fromarray(rgb)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    headers = {"Cache-Control": "public, max-age=86400"}
    if bounds:
        headers["X-Tile-Bounds"] = ",".join(str(b) for b in bounds)
    return StreamingResponse(buf, media_type="image/png", headers=headers)


@app.get("/api/events/{cems_id}/tiles/{tile}/sar/{phase}")
def tile_sar(cems_id: str, tile: str, phase: str):
    """Serve Sentinel-1 SAR VV as grayscale PNG."""
    if phase not in ("before", "during"):
        raise HTTPException(400, "Phase must be 'before' or 'during'")
    s1 = _s1_path(cems_id, tile, phase)
    if not s1:
        raise HTTPException(404, f"S1 data not found for {cems_id}/{tile}/{phase}")
    gray = _load_s1_gray(s1)
    img = Image.fromarray(gray)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png",
                             headers={"Cache-Control": "public, max-age=86400"})


@app.get("/api/events/{cems_id}/tiles/{tile}/mask/{phase}")
def tile_mask_png(
    cems_id: str, tile: str, phase: str,
    overlay: bool = Query(False),
):
    """Serve flood/water mask as colored transparent PNG."""
    if phase not in ("before", "during"):
        raise HTTPException(400, "Phase must be 'before' or 'during'")

    if phase == "during":
        flood = _load_flood_pred(_pred_path(cems_id, tile))
    else:
        mask = _load_water_mask(_water_path(cems_id, tile, "before"))
        flood = mask if mask is not None else np.zeros((512, 512), dtype=bool)

    if overlay:
        # Composite onto satellite image
        s2 = _s2_path(cems_id, tile, phase)
        s1 = _s1_path(cems_id, tile, phase)
        if s2:
            rgb = _load_s2_rgb(s2).astype(np.float32)
        elif s1:
            rgb = _load_s1_gray(s1).astype(np.float32)
        else:
            rgb = np.zeros((*flood.shape, 3), dtype=np.float32)

        color = np.array([0, 100, 255], dtype=np.float32)
        alpha = 0.45
        for c in range(3):
            rgb[:, :, c] = np.where(flood, rgb[:, :, c] * (1 - alpha) + color[c] * alpha, rgb[:, :, c])
        img = Image.fromarray(rgb.clip(0, 255).astype(np.uint8))
    else:
        rgba = np.zeros((*flood.shape, 4), dtype=np.uint8)
        rgba[flood] = [0, 100, 255, 160]
        img = Image.fromarray(rgba)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


@app.get("/api/events/{cems_id}/tiles/{tile}/change_overlay")
def tile_change_overlay(cems_id: str, tile: str):
    """Composite PNG: satellite + blue(existing) + red(new flood)."""
    flood_pred = _load_flood_pred(_pred_path(cems_id, tile))
    before_water = _load_water_mask(_water_path(cems_id, tile, "before"))
    during_water = _load_water_mask(_water_path(cems_id, tile, "during"))

    # Get background image
    s2 = _s2_path(cems_id, tile, "during")
    s1 = _s1_path(cems_id, tile, "during")
    if s2:
        rgb = _load_s2_rgb(s2).astype(np.float32)
    elif s1:
        rgb = _load_s1_gray(s1).astype(np.float32)
    else:
        raise HTTPException(404, "No imagery available")

    if before_water is not None:
        existing = before_water & flood_pred
        new_flood = flood_pred & ~before_water
    else:
        existing = np.zeros_like(flood_pred)
        new_flood = flood_pred

    existing = binary_opening(existing, structure=np.ones((3, 3)))
    new_flood = binary_opening(new_flood, structure=np.ones((3, 3)))

    # Blue for existing water
    for c, v in enumerate([0, 191, 255]):
        rgb[:, :, c] = np.where(existing, rgb[:, :, c] * 0.5 + v * 0.5, rgb[:, :, c])
    # Red/magenta for new flood
    for c, v in enumerate([255, 0, 77]):
        rgb[:, :, c] = np.where(new_flood, rgb[:, :, c] * 0.4 + v * 0.6, rgb[:, :, c])

    img = Image.fromarray(rgb.clip(0, 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


@app.get("/api/events/{cems_id}/tiles/{tile}/stats")
def tile_stats(cems_id: str, tile: str):
    """Per-tile flood stats + AI model metrics."""
    pred_p = _pred_path(cems_id, tile)
    if not os.path.exists(pred_p):
        raise HTTPException(404, f"Tile {tile} not found for {cems_id}")

    stats = _compute_tile_stats(cems_id, tile)
    bounds = _tile_bounds_4326(pred_p)
    stats["bounds"] = bounds
    stats["center"] = [(bounds[1]+bounds[3])/2, (bounds[0]+bounds[2])/2]
    stats["tile"] = tile
    stats["event_id"] = cems_id

    # Add model metrics if available
    metrics = _load_metrics_csv(cems_id)
    m = metrics.get(tile, {})
    stats["f1"] = m.get("f1")
    stats["iou"] = m.get("iou")
    stats["precision"] = m.get("precision")
    stats["recall"] = m.get("recall")
    stats["accuracy"] = m.get("accuracy")

    return stats
