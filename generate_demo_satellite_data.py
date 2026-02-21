"""
Generate synthetic GeoTIFF demo data for the Satellite Flood Detection API.
Creates realistic flood-prediction rasters so the API endpoints return data.

Usage:
    python generate_demo_satellite_data.py
"""

import csv
import os
import numpy as np

try:
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.crs import CRS
except ImportError:
    raise SystemExit("Install rasterio first: pip install rasterio")

# ── Configuration ───────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.join(BASE_DIR, "OUTPUTS", "ai4g_flood")
CEMS_ROOT = os.path.join(BASE_DIR, "DATA", "CEMS")

# Tile size (pixels)
H, W = 512, 512

# Events with realistic bounding boxes (EPSG:32633 / 32631 / 32630 UTM coords)
EVENTS = {
    "EMSR766": {
        "epsg": 32633,
        "tiles": {
            "T0001": {"west": 393000, "south": 4990000, "east": 398120, "north": 4995120},
            "T0002": {"west": 398120, "south": 4990000, "east": 403240, "north": 4995120},
            "T0003": {"west": 393000, "south": 4995120, "east": 398120, "north": 5000240},
        },
    },
    "EMSR771": {
        "epsg": 32632,
        "tiles": {
            "T0001": {"west": 720000, "south": 4910000, "east": 725120, "north": 4915120},
            "T0002": {"west": 725120, "south": 4910000, "east": 730240, "north": 4915120},
            "T0003": {"west": 720000, "south": 4915120, "east": 725120, "north": 4920240},
            "T0004": {"west": 725120, "south": 4915120, "east": 730240, "north": 4920240},
        },
    },
    "EMSR773": {
        "epsg": 32630,
        "tiles": {
            "T0001": {"west": 720000, "south": 4360000, "east": 725120, "north": 4365120},
            "T0002": {"west": 725120, "south": 4360000, "east": 730240, "north": 4365120},
            "T0003": {"west": 720000, "south": 4365120, "east": 725120, "north": 4370240},
        },
    },
}


def _make_flood_mask(seed, flood_pct=0.15):
    """Generate a realistic-looking binary flood mask with blobs."""
    rng = np.random.RandomState(seed)
    # Start with smooth noise
    noise = rng.randn(H // 8, W // 8)
    # Upscale with interpolation
    from scipy.ndimage import zoom, gaussian_filter
    noise = zoom(noise, 8, order=3)
    noise = gaussian_filter(noise, sigma=12)
    # Threshold to get desired flood percentage
    threshold = np.percentile(noise, 100 * (1 - flood_pct))
    mask = (noise > threshold).astype(np.uint8) * 255
    return mask


def _make_water_mask(seed, water_pct=0.08):
    """Generate a water body mask (rivers/lakes)."""
    return _make_flood_mask(seed, flood_pct=water_pct)


def _make_s1_sar(seed):
    """Generate a synthetic SAR-like single-band raster (dB values)."""
    rng = np.random.RandomState(seed)
    from scipy.ndimage import gaussian_filter
    # SAR backscatter in dB (typical range: -25 to 0 dB for land, lower for water)
    base = rng.randn(H, W).astype(np.float32) * 3 - 12  # mean ~-12 dB
    base = gaussian_filter(base, sigma=2)
    return base.astype(np.float32)


def _make_s2_rgb(seed):
    """Generate a synthetic 8-band Sentinel-2-like raster."""
    rng = np.random.RandomState(seed)
    from scipy.ndimage import gaussian_filter
    bands = []
    for i in range(8):
        band = rng.rand(H, W).astype(np.float32) * 2000 + 500
        band = gaussian_filter(band, sigma=3)
        bands.append(band)
    return np.stack(bands)  # shape (8, H, W)


def write_tif(path, data, crs, transform, count=1, dtype="uint8"):
    """Write a GeoTIFF."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if data.ndim == 2:
        data = data[np.newaxis, ...]  # (1, H, W)
    with rasterio.open(
        path, "w", driver="GTiff",
        height=data.shape[1], width=data.shape[2], count=data.shape[0],
        dtype=dtype, crs=CRS.from_epsg(crs),
        transform=transform,
    ) as dst:
        for i in range(data.shape[0]):
            dst.write(data[i], i + 1)


def generate_event(cems_id, event_cfg):
    """Generate all demo data for one CEMS event."""
    epsg = event_cfg["epsg"]
    crs_epsg = epsg
    geo_dir = os.path.join(OUTPUT_ROOT, cems_id, "geotiffs")
    os.makedirs(geo_dir, exist_ok=True)

    metrics_rows = []
    seed_base = hash(cems_id) % 10000

    for idx, (tile_id, bbox) in enumerate(event_cfg["tiles"].items()):
        transform = from_bounds(bbox["west"], bbox["south"], bbox["east"], bbox["north"], W, H)
        seed = seed_base + idx

        # 1. Flood prediction (uint8, 0 or 255)
        flood_mask = _make_flood_mask(seed, flood_pct=0.12 + idx * 0.03)
        write_tif(
            os.path.join(geo_dir, f"{tile_id}_flood_pred.tif"),
            flood_mask, crs_epsg, transform, dtype="uint8",
        )

        # 2. Before water mask
        before_water = _make_water_mask(seed + 100, water_pct=0.06)
        write_tif(
            os.path.join(geo_dir, f"{tile_id}_before_water.tif"),
            before_water, crs_epsg, transform, dtype="uint8",
        )

        # 3. During water mask
        during_water = np.maximum(before_water, flood_mask)
        write_tif(
            os.path.join(geo_dir, f"{tile_id}_during_water.tif"),
            during_water, crs_epsg, transform, dtype="uint8",
        )

        # 4. Sentinel-1 SAR (before & during) — stored in DATA/CEMS/
        for phase in ("before", "during"):
            s1_dir = os.path.join(CEMS_ROOT, cems_id, f"s1_{phase}_flood")
            os.makedirs(s1_dir, exist_ok=True)
            s1_data = _make_s1_sar(seed + 200 + (0 if phase == "before" else 1))
            write_tif(
                os.path.join(s1_dir, f"{tile_id}_s1_{phase}_flood.tif"),
                s1_data, crs_epsg, transform, dtype="float32",
            )

        # 5. Sentinel-2 RGB (before & during)
        for phase in ("before", "during"):
            s2_dir = os.path.join(CEMS_ROOT, cems_id, f"s2_{phase}_flood")
            os.makedirs(s2_dir, exist_ok=True)
            s2_data = _make_s2_rgb(seed + 300 + (0 if phase == "before" else 1))
            write_tif(
                os.path.join(s2_dir, f"{tile_id}_s2_{phase}_flood.tif"),
                s2_data, crs_epsg, transform, count=8, dtype="float32",
            )

        # 6. Metrics
        flood_px = int((flood_mask > 0).sum())
        total_px = H * W
        # Synthetic F1/IoU
        rng = np.random.RandomState(seed + 999)
        f1 = round(rng.uniform(0.72, 0.93), 4)
        iou = round(f1 * rng.uniform(0.75, 0.90), 4)
        precision = round(rng.uniform(0.78, 0.96), 4)
        recall = round(rng.uniform(0.70, 0.95), 4)

        metrics_rows.append({
            "tile_id": tile_id,
            "flood_pixels": flood_px,
            "total_pixels": total_px,
            "flood_pct": round(100 * flood_px / total_px, 2),
            "f1": f1,
            "iou": iou,
            "precision": precision,
            "recall": recall,
        })

        print(f"  ✓ {cems_id}/{tile_id}: flood={flood_px}px, f1={f1}")

    # Write metrics CSV
    csv_path = os.path.join(OUTPUT_ROOT, cems_id, f"{cems_id}_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metrics_rows[0].keys())
        writer.writeheader()
        writer.writerows(metrics_rows)
    print(f"  ✓ Metrics CSV → {csv_path}")


def main():
    print("=" * 60)
    print("Generating demo satellite data for Sentinel API")
    print("=" * 60)

    for cems_id, cfg in EVENTS.items():
        print(f"\n▶ {cems_id} ({len(cfg['tiles'])} tiles)")
        generate_event(cems_id, cfg)

    print("\n" + "=" * 60)
    print("Done! Restart satellite_api or it will pick up data on next request.")
    print(f"Output root: {OUTPUT_ROOT}")
    print(f"CEMS data:   {CEMS_ROOT}")
    print("=" * 60)


if __name__ == "__main__":
    main()
