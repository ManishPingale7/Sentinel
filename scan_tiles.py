"""
Quick scan: for every SenForFlood tile, compute cloud % and flood change.
Helps find the best cloud-free before/during pairs for demo.
"""
import glob, os, numpy as np, rasterio, csv

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CEMS_ROOT = os.path.join(_BASE_DIR, "DATA", "SenForFloods", "SenForFlood", "CEMS")
PRED_ROOT = os.path.join(_BASE_DIR, "OUTPUTS", "Prithvi")

rows = []

for cems_id in sorted(os.listdir(CEMS_ROOT)):
    before_dir = os.path.join(CEMS_ROOT, cems_id, "s2_before_flood")
    during_dir = os.path.join(CEMS_ROOT, cems_id, "s2_during_flood")
    if not os.path.isdir(before_dir) or not os.path.isdir(during_dir):
        continue

    before_files = {os.path.basename(f).split("_")[0]: f
                    for f in sorted(glob.glob(os.path.join(before_dir, "*.tif")))}
    during_files = {os.path.basename(f).split("_")[0]: f
                    for f in sorted(glob.glob(os.path.join(during_dir, "*.tif")))}

    for tile in sorted(set(before_files) & set(during_files)):
        try:
            with rasterio.open(before_files[tile]) as src:
                bdata = src.read()
            with rasterio.open(during_files[tile]) as src:
                ddata = src.read()

            npix = bdata.shape[1] * bdata.shape[2]

            # Cloud proxy: pixels where ALL visible bands (B2,B3,B4 = idx 0,1,2) > 2500
            b_cloud = np.all(bdata[:3] > 2500, axis=0).sum()
            d_cloud = np.all(ddata[:3] > 2500, axis=0).sum()
            b_cloud_pct = b_cloud / npix * 100
            d_cloud_pct = d_cloud / npix * 100

            # NDWI from before and during: (Green - NIR) / (Green + NIR)
            # Band 1=Green, Band 3=NIR(B8A)
            def ndwi(data):
                g = data[1].astype(np.float32)
                n = data[3].astype(np.float32)
                idx = np.where((g + n) > 0, (g - n) / (g + n + 1e-6), 0)
                return idx

            b_ndwi = ndwi(bdata)
            d_ndwi = ndwi(ddata)
            # Water pixels: NDWI > 0.1
            b_water_pct = (b_ndwi > 0.1).sum() / npix * 100
            d_water_pct = (d_ndwi > 0.1).sum() / npix * 100
            water_increase = d_water_pct - b_water_pct

            # Check if prediction masks exist
            bp = os.path.join(PRED_ROOT, cems_id, "before", f"{os.path.basename(before_files[tile]).replace('.tif','_pred.tif')}")
            dp = os.path.join(PRED_ROOT, cems_id, "during", f"{os.path.basename(during_files[tile]).replace('.tif','_pred.tif')}")
            pred_change = ""
            if os.path.exists(bp) and os.path.exists(dp):
                with rasterio.open(bp) as s: bm = s.read(1)
                with rasterio.open(dp) as s: dm = s.read(1)
                pred_before_pct = (bm == 1).sum() / npix * 100
                pred_during_pct = (dm == 1).sum() / npix * 100
                pred_change = f"{pred_during_pct - pred_before_pct:.1f}%"

            row = {
                "cems_id": cems_id,
                "tile": tile,
                "b_cloud%": round(b_cloud_pct, 1),
                "d_cloud%": round(d_cloud_pct, 1),
                "b_water%": round(b_water_pct, 1),
                "d_water%": round(d_water_pct, 1),
                "water_incr": round(water_increase, 1),
                "pred_change": pred_change,
            }
            rows.append(row)
            status = "***GOOD***" if b_cloud_pct < 10 and d_cloud_pct < 10 and water_increase > 3 else ""
            print(f"  {cems_id}/{tile}  cloud: {b_cloud_pct:.0f}%/{d_cloud_pct:.0f}%  "
                  f"water: {b_water_pct:.1f}% -> {d_water_pct:.1f}% (+{water_increase:.1f}%)  {status}")
        except Exception as e:
            print(f"  {cems_id}/{tile} ERROR: {e}")

print(f"\n{'='*60}")
print("BEST CANDIDATES (low cloud + high water increase):")
print(f"{'='*60}")
good = [r for r in rows if r["b_cloud%"] < 15 and r["d_cloud%"] < 15 and r["water_incr"] > 2]
good.sort(key=lambda x: -x["water_incr"])
if good:
    for r in good:
        print(f"  {r['cems_id']}/{r['tile']}  cloud:{r['b_cloud%']}%/{r['d_cloud%']}%  "
              f"water:{r['b_water%']}% -> {r['d_water%']}%  increase:{r['water_incr']}%")
else:
    print("  No great cloud-free pairs found with significant water increase.")
    print("\n  Top tiles by water increase (any cloud level):")
    rows.sort(key=lambda x: -x["water_incr"])
    for r in rows[:10]:
        print(f"  {r['cems_id']}/{r['tile']}  cloud:{r['b_cloud%']}%/{r['d_cloud%']}%  "
              f"water:{r['b_water%']}% -> {r['d_water%']}%  increase:{r['water_incr']}%")
