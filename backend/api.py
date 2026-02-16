"""
Flood Detection API – FastAPI Backend
======================================
Serves flood prediction data, GeoJSON polygons, RGB tiles, and statistics
for a Next.js map-based frontend.

Run:
    cd G:\Sentinel
    uvicorn backend.api:app --reload --port 8000

Endpoints:
    GET  /api/events                              → list CEMS events
    GET  /api/events/{cems_id}                    → event summary + bounds
    GET  /api/events/{cems_id}/tiles              → list tiles with bounds
    GET  /api/events/{cems_id}/tiles/{tile}/geojson?phase=during&layer=flood
    GET  /api/events/{cems_id}/tiles/{tile}/stats
    GET  /api/events/{cems_id}/tiles/{tile}/rgb/{phase}   → PNG image
    GET  /api/events/{cems_id}/tiles/{tile}/mask/{phase}  → coloured PNG mask
    GET  /api/events/{cems_id}/geojson            → ALL tiles merged GeoJSON
    GET  /api/summary                             → global dashboard stats
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
from scipy.ndimage import binary_opening

# ── Configuration ───────────────────────────────────────────────────────

SENFORFLOOD_ROOT = r"G:\Sentinel\DATA\SenForFlood\CEMS"
PRED_ROOT        = r"G:\Sentinel\OUTPUTS\Prithvi"
REPORT_CSV       = r"G:\Sentinel\OUTPUTS\change_detection_report.csv"
PIXEL_RES_M      = 10.0

# ── App ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Sentinel Flood Detection API",
    description="Serves flood segmentation masks, GeoJSON polygons, "
                "RGB imagery and change-detection statistics.",
    version="1.0.0",
)

# Allow Next.js dev server (localhost:3000) + any origin for hackathon
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ─────────────────────────────────────────────────────────────

def _get_cems_ids():
    """Return sorted list of CEMS event IDs that have predictions."""
    if not os.path.isdir(PRED_ROOT):
        return []
    return sorted(
        d for d in os.listdir(PRED_ROOT)
        if os.path.isdir(os.path.join(PRED_ROOT, d))
    )


def _get_tiles(cems_id: str):
    """Return sorted list of tile IDs for a CEMS event."""
    before_dir = os.path.join(PRED_ROOT, cems_id, "before")
    during_dir = os.path.join(PRED_ROOT, cems_id, "during")
    before = {os.path.basename(f).split("_")[0]
              for f in glob.glob(os.path.join(before_dir, "*_pred.tif"))}
    during = {os.path.basename(f).split("_")[0]
              for f in glob.glob(os.path.join(during_dir, "*_pred.tif"))}
    return sorted(before & during)


def _pred_path(cems_id: str, tile: str, phase: str) -> str:
    return os.path.join(PRED_ROOT, cems_id, phase,
                        f"{tile}_s2_{phase}_flood_pred.tif")


def _src_path(cems_id: str, tile: str, phase: str) -> str:
    d = os.path.join(SENFORFLOOD_ROOT, cems_id, f"s2_{phase}_flood")
    candidates = glob.glob(os.path.join(d, f"{tile}_*.tif"))
    return candidates[0] if candidates else ""


def _tile_bounds_4326(tif_path: str):
    """Return [west, south, east, north] in EPSG:4326."""
    with rasterio.open(tif_path) as src:
        return list(transform_bounds(src.crs, "EPSG:4326", *src.bounds))


def _tile_center_4326(tif_path: str):
    b = _tile_bounds_4326(tif_path)
    return [(b[1] + b[3]) / 2, (b[0] + b[2]) / 2]  # [lat, lng]


@lru_cache(maxsize=1)
def _load_report():
    """Load CSV report into a dict keyed by (cems_id, tile)."""
    lut = {}
    if not os.path.exists(REPORT_CSV):
        return lut
    with open(REPORT_CSV, newline="") as f:
        for row in csv.DictReader(f):
            lut[(row["cems_id"], row["tile"])] = {
                k: _try_float(v) for k, v in row.items()
            }
    return lut


def _try_float(v):
    try:
        return float(v)
    except (ValueError, TypeError):
        return v


def _normalize_band(band):
    valid = band[band > 0]
    if valid.size == 0:
        return np.zeros_like(band, dtype=np.float32)
    p2 = np.percentile(valid, 2)
    p98 = np.percentile(valid, 98)
    band = np.clip(band, p2, p98).astype(np.float32)
    lo, hi = band.min(), band.max()
    if hi - lo < 1e-6:
        return np.zeros_like(band, dtype=np.float32)
    return (band - lo) / (hi - lo)


def _load_rgb_array(tif_path: str) -> np.ndarray:
    """Return uint8 RGB [H,W,3]."""
    with rasterio.open(tif_path) as src:
        red   = src.read(3).astype(np.float32)
        green = src.read(2).astype(np.float32)
        blue  = src.read(1).astype(np.float32)
    rgb = np.stack([_normalize_band(red),
                    _normalize_band(green),
                    _normalize_band(blue)], axis=-1)
    return (rgb * 255).astype(np.uint8)


def _mask_to_geojson(tif_path: str, flood_value: int = 1,
                     simplify_tolerance: float = 15.0):
    """Vectorize a raster mask into GeoJSON Feature collection (EPSG:4326)."""
    from shapely.geometry import shape, mapping
    from shapely.ops import unary_union
    from pyproj import Transformer

    with rasterio.open(tif_path) as src:
        mask_data = src.read(1)
        transform = src.transform
        src_crs = src.crs

    binary = (mask_data == flood_value).astype(np.uint8)
    # Morphological cleanup to reduce noise
    binary = binary_opening(binary, structure=np.ones((3, 3))).astype(np.uint8)

    features = []
    for geom, val in rio_shapes(binary, mask=binary == 1, transform=transform):
        if val == 1:
            features.append(shape(geom))

    if not features:
        return {"type": "FeatureCollection", "features": []}

    # Merge into a single multipolygon, simplify, reproject to 4326
    merged = unary_union(features)
    if simplify_tolerance > 0:
        merged = merged.simplify(simplify_tolerance, preserve_topology=True)

    transformer = Transformer.from_crs(str(src_crs), "EPSG:4326", always_xy=True)

    def _reproject_coords(geom_mapping):
        """Recursively reproject coordinates."""
        if geom_mapping["type"] == "Polygon":
            new_coords = []
            for ring in geom_mapping["coordinates"]:
                new_coords.append(
                    [list(transformer.transform(x, y)) for x, y in ring]
                )
            geom_mapping["coordinates"] = new_coords
        elif geom_mapping["type"] == "MultiPolygon":
            new_polys = []
            for poly in geom_mapping["coordinates"]:
                new_rings = []
                for ring in poly:
                    new_rings.append(
                        [list(transformer.transform(x, y)) for x, y in ring]
                    )
                new_polys.append(new_rings)
            geom_mapping["coordinates"] = new_polys
        return geom_mapping

    geom_dict = _reproject_coords(mapping(merged))

    return {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "geometry": geom_dict,
            "properties": {},
        }],
    }


# ── Endpoints ───────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "api": "Sentinel Flood Detection", "version": "1.0.0"}


@app.get("/api/summary")
def global_summary():
    """Dashboard-level summary stats across all events."""
    report = _load_report()
    events = _get_cems_ids()
    total_tiles = sum(len(_get_tiles(c)) for c in events)
    total_new_km2 = sum(r.get("new_flood_km2", 0) for r in report.values())
    total_during_km2 = sum(r.get("during_km2", 0) for r in report.values())

    return {
        "total_events": len(events),
        "total_tiles": total_tiles,
        "total_new_flood_km2": round(total_new_km2, 2),
        "total_flooded_km2": round(total_during_km2, 2),
        "events": events,
    }


@app.get("/api/events")
def list_events():
    """List all CEMS flood events with center coordinates + summary."""
    report = _load_report()
    events = []

    for cems_id in _get_cems_ids():
        tiles = _get_tiles(cems_id)
        if not tiles:
            continue

        # Get center from first tile
        sample_pred = _pred_path(cems_id, tiles[0], "during")
        center = _tile_center_4326(sample_pred) if os.path.exists(sample_pred) else [0, 0]

        # Get bounds encompassing all tiles
        all_bounds = []
        for t in tiles:
            p = _pred_path(cems_id, t, "during")
            if os.path.exists(p):
                all_bounds.append(_tile_bounds_4326(p))

        if all_bounds:
            bbox = [
                min(b[0] for b in all_bounds),
                min(b[1] for b in all_bounds),
                max(b[2] for b in all_bounds),
                max(b[3] for b in all_bounds),
            ]
        else:
            bbox = [0, 0, 0, 0]

        # Aggregate metrics
        event_rows = [v for (c, t), v in report.items() if c == cems_id]
        total_new = sum(r.get("new_flood_km2", 0) for r in event_rows)
        total_during = sum(r.get("during_km2", 0) for r in event_rows)

        events.append({
            "cems_id": cems_id,
            "tile_count": len(tiles),
            "center": center,
            "bbox": bbox,
            "total_new_flood_km2": round(total_new, 2),
            "total_flooded_km2": round(total_during, 2),
        })

    return events


@app.get("/api/events/{cems_id}")
def event_detail(cems_id: str):
    """Detailed info for a single event."""
    tiles = _get_tiles(cems_id)
    if not tiles:
        raise HTTPException(404, f"Event {cems_id} not found")

    report = _load_report()
    tile_list = []
    all_bounds = []

    for t in tiles:
        p = _pred_path(cems_id, t, "during")
        bounds = _tile_bounds_4326(p) if os.path.exists(p) else None
        if bounds:
            all_bounds.append(bounds)
        stats = report.get((cems_id, t), {})
        tile_list.append({
            "tile": t,
            "bounds": bounds,
            "center": [(bounds[1]+bounds[3])/2, (bounds[0]+bounds[2])/2] if bounds else None,
            "stats": stats,
        })

    bbox = [
        min(b[0] for b in all_bounds),
        min(b[1] for b in all_bounds),
        max(b[2] for b in all_bounds),
        max(b[3] for b in all_bounds),
    ] if all_bounds else None

    return {
        "cems_id": cems_id,
        "tile_count": len(tiles),
        "bbox": bbox,
        "tiles": tile_list,
    }


@app.get("/api/events/{cems_id}/tiles")
def list_tiles(cems_id: str):
    """List all tiles for an event with their geo-bounds."""
    tiles = _get_tiles(cems_id)
    if not tiles:
        raise HTTPException(404, f"Event {cems_id} not found")

    report = _load_report()
    result = []
    for t in tiles:
        p = _pred_path(cems_id, t, "during")
        bounds = _tile_bounds_4326(p) if os.path.exists(p) else None
        stats = report.get((cems_id, t), {})
        result.append({
            "tile": t,
            "bounds": bounds,
            "new_flood_km2": stats.get("new_flood_km2", 0),
            "pct_increase": stats.get("pct_increase", 0),
        })
    return result


@app.get("/api/events/{cems_id}/tiles/{tile}/geojson")
def tile_geojson(
    cems_id: str,
    tile: str,
    phase: str = Query("during", pattern="^(before|during)$"),
    layer: str = Query("flood", pattern="^(flood|new_flood|change)$"),
):
    """Return GeoJSON polygons for a tile's flood mask.

    Layers:
        flood     → all pixels == 1 in the selected phase
        new_flood → pixels flooded in "during" but NOT in "before"
        change    → combined: existing water + new flood as separate features
    """
    pred_during = _pred_path(cems_id, tile, "during")
    pred_before = _pred_path(cems_id, tile, "before")

    if not os.path.exists(pred_during):
        raise HTTPException(404, f"Tile {tile} not found for {cems_id}")

    if layer == "flood":
        target = pred_during if phase == "during" else pred_before
        if not os.path.exists(target):
            raise HTTPException(404, f"Phase '{phase}' mask not found")
        geojson = _mask_to_geojson(target, flood_value=1)
        for f in geojson["features"]:
            f["properties"]["layer"] = "flood"
            f["properties"]["phase"] = phase
        return geojson

    elif layer == "new_flood":
        if not os.path.exists(pred_before):
            raise HTTPException(404, "Before mask not found")

        # Create a temporary difference mask
        with rasterio.open(pred_before) as src:
            before_mask = src.read(1)
        with rasterio.open(pred_during) as src:
            during_mask = src.read(1)
            transform = src.transform
            crs = src.crs
            meta = src.meta.copy()

        new_flood = ((during_mask == 1) & (before_mask != 1)).astype(np.int16)

        # Write to a temp file and vectorize
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
        tmp_path = tmp.name
        tmp.close()
        try:
            meta.update(count=1, dtype="int16")
            with rasterio.open(tmp_path, "w", **meta) as dst:
                dst.write(new_flood, 1)
            geojson = _mask_to_geojson(tmp_path, flood_value=1)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        for f in geojson["features"]:
            f["properties"]["layer"] = "new_flood"
        return geojson

    elif layer == "change":
        # Return both existing water and new flood as separate feature groups
        if not os.path.exists(pred_before):
            raise HTTPException(404, "Before mask not found")

        with rasterio.open(pred_before) as src:
            before_mask = src.read(1)
        with rasterio.open(pred_during) as src:
            during_mask = src.read(1)
            meta = src.meta.copy()

        existing = ((during_mask == 1) & (before_mask == 1)).astype(np.int16)
        new_flood = ((during_mask == 1) & (before_mask != 1)).astype(np.int16)

        import tempfile
        features = []

        for label, arr, color in [
            ("existing_water", existing, "#0066ff"),
            ("new_flood", new_flood, "#ff0000"),
        ]:
            tmp = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
            tmp_path = tmp.name
            tmp.close()
            try:
                meta.update(count=1, dtype="int16")
                with rasterio.open(tmp_path, "w", **meta) as dst:
                    dst.write(arr, 1)
                gj = _mask_to_geojson(tmp_path, flood_value=1)
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
            for f in gj["features"]:
                f["properties"]["layer"] = label
                f["properties"]["color"] = color
            features.extend(gj["features"])

        return {"type": "FeatureCollection", "features": features}


@app.get("/api/events/{cems_id}/tiles/{tile}/stats")
def tile_stats(cems_id: str, tile: str):
    """Return change-detection metrics for a tile."""
    report = _load_report()
    key = (cems_id, tile)
    if key not in report:
        raise HTTPException(404, f"No stats for {cems_id}/{tile}")

    stats = report[key]
    # Add bounds
    pred = _pred_path(cems_id, tile, "during")
    bounds = _tile_bounds_4326(pred) if os.path.exists(pred) else None
    stats["bounds"] = bounds
    return stats


@app.get("/api/events/{cems_id}/tiles/{tile}/rgb/{phase}")
def tile_rgb(cems_id: str, tile: str, phase: str):
    """Serve the original Sentinel-2 image as a PNG."""
    if phase not in ("before", "during"):
        raise HTTPException(400, "Phase must be 'before' or 'during'")

    src_path = _src_path(cems_id, tile, phase)
    if not src_path or not os.path.exists(src_path):
        raise HTTPException(404, f"Source image not found for {cems_id}/{tile}/{phase}")

    rgb = _load_rgb_array(src_path)
    img = Image.fromarray(rgb)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png",
                             headers={"Cache-Control": "public, max-age=86400"})


@app.get("/api/events/{cems_id}/tiles/{tile}/mask/{phase}")
def tile_mask_png(
    cems_id: str, tile: str, phase: str,
    overlay: bool = Query(False, description="Overlay mask on RGB"),
):
    """Serve the flood mask as a coloured transparent PNG.

    If overlay=true, composites the mask onto the RGB image.
    """
    if phase not in ("before", "during"):
        raise HTTPException(400, "Phase must be 'before' or 'during'")

    pred = _pred_path(cems_id, tile, phase)
    if not os.path.exists(pred):
        raise HTTPException(404, f"Mask not found for {cems_id}/{tile}/{phase}")

    with rasterio.open(pred) as src:
        mask_data = src.read(1)

    flood = (mask_data == 1)
    flood_clean = binary_opening(flood, structure=np.ones((3, 3)))

    if overlay:
        src_path = _src_path(cems_id, tile, phase)
        if src_path and os.path.exists(src_path):
            rgb = _load_rgb_array(src_path).astype(np.float32)
        else:
            rgb = np.zeros((*mask_data.shape, 3), dtype=np.float32)

        # Blue overlay for flood pixels
        overlay_color = np.array([0, 100, 255], dtype=np.float32)
        alpha = 0.45
        for c in range(3):
            rgb[:, :, c] = np.where(
                flood_clean,
                rgb[:, :, c] * (1 - alpha) + overlay_color[c] * alpha,
                rgb[:, :, c],
            )
        img = Image.fromarray(rgb.clip(0, 255).astype(np.uint8))
    else:
        # RGBA transparent PNG: flood=blue, rest=transparent
        rgba = np.zeros((*mask_data.shape, 4), dtype=np.uint8)
        rgba[flood_clean] = [0, 100, 255, 160]
        img = Image.fromarray(rgba)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png",
                             headers={"Cache-Control": "public, max-age=86400"})


@app.get("/api/events/{cems_id}/tiles/{tile}/change_overlay")
def tile_change_overlay(cems_id: str, tile: str):
    """Serve a composite PNG: RGB + blue(existing water) + red(new flood)."""
    pred_before = _pred_path(cems_id, tile, "before")
    pred_during = _pred_path(cems_id, tile, "during")

    if not os.path.exists(pred_during) or not os.path.exists(pred_before):
        raise HTTPException(404, "Prediction masks not found")

    src_path = _src_path(cems_id, tile, "during")
    if not src_path or not os.path.exists(src_path):
        raise HTTPException(404, "Source image not found")

    with rasterio.open(pred_before) as s:
        before_mask = s.read(1)
    with rasterio.open(pred_during) as s:
        during_mask = s.read(1)

    existing = (before_mask == 1) & (during_mask == 1)
    new_flood = (during_mask == 1) & (before_mask != 1)
    existing = binary_opening(existing, structure=np.ones((3, 3)))
    new_flood = binary_opening(new_flood, structure=np.ones((3, 3)))

    rgb = _load_rgb_array(src_path).astype(np.float32)

    # Blue for existing water
    for c, v in enumerate([0, 100, 255]):
        rgb[:, :, c] = np.where(existing, rgb[:, :, c] * 0.5 + v * 0.5, rgb[:, :, c])
    # Red for new flood
    for c, v in enumerate([255, 40, 40]):
        rgb[:, :, c] = np.where(new_flood, rgb[:, :, c] * 0.4 + v * 0.6, rgb[:, :, c])

    img = Image.fromarray(rgb.clip(0, 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


@app.get("/api/events/{cems_id}/geojson")
def event_geojson(
    cems_id: str,
    layer: str = Query("change", pattern="^(flood|new_flood|change)$"),
):
    """Merged GeoJSON for ALL tiles in an event — for rendering the full event on a map."""
    tiles = _get_tiles(cems_id)
    if not tiles:
        raise HTTPException(404, f"Event {cems_id} not found")

    report = _load_report()
    all_features = []

    for t in tiles:
        try:
            gj = tile_geojson(cems_id, t, phase="during", layer=layer)
            stats = report.get((cems_id, t), {})
            for f in gj["features"]:
                f["properties"]["tile"] = t
                f["properties"]["new_flood_km2"] = stats.get("new_flood_km2", 0)
                f["properties"]["pct_increase"] = stats.get("pct_increase", 0)
            all_features.extend(gj["features"])
        except Exception:
            continue

    return {"type": "FeatureCollection", "features": all_features}
