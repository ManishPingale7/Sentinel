"""Find 3+ date tiles with clear fire progression."""
import os, re, collections, rasterio

data_dirs = {"training": r"D:\sih\Sentinel\DATA\Fire\training",
             "validation": r"D:\sih\Sentinel\DATA\Fire\validation"}
pat = re.compile(r"(subsetted_512x512_HLS\.S30\.(T\w+)\.(\d{4})(\d{3})\.\S+?)_merged\.tif")
tiles = collections.defaultdict(list)

for sp, d in data_dirs.items():
    for f in sorted(os.listdir(d)):
        m = pat.search(f)
        if m:
            tid, mgrs, yr, doy = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
            mp = os.path.join(d, tid + ".mask.tif")
            bp = 0
            if os.path.exists(mp):
                with rasterio.open(mp) as s:
                    mask = s.read(1)
                    bp = round((mask == 1).sum() / mask.size * 100, 1)
            tiles[mgrs].append(dict(
                mgrs=mgrs, year=yr, doy=doy, split=sp, tile_id=tid,
                burn_pct=bp, img_path=os.path.join(d, f), mask_path=mp,
            ))

print("Tiles with 3+ dates and clear fire progression (min<10%, max>30%):\n")
for mgrs in sorted(tiles):
    e = sorted(tiles[mgrs], key=lambda x: (x["year"], x["doy"]))
    if len(e) >= 3:
        burns = [x["burn_pct"] for x in e]
        mn, mx = min(burns), max(burns)
        if mx > 30 and mn < 10:
            parts = [f"  {x['year']}-{x['doy']:03d} burn={x['burn_pct']:5.1f}%" for x in e]
            print(f"{mgrs} [{len(e)} dates]:")
            for p in parts:
                print(p)
            print()
