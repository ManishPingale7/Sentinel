"""Scan Fire dataset for multi-date MGRS pairs and rank by burn-change delta."""
import os, re, collections
import numpy as np
import rasterio

data_dirs = {
    "training": r"D:\sih\Sentinel\DATA\Fire\training",
    "validation": r"D:\sih\Sentinel\DATA\Fire\validation",
}
pattern = re.compile(r"(subsetted_512x512_HLS\.S30\.(T\w+)\.(\d{4})(\d{3})\.\S+?)_merged\.tif")

tiles = collections.defaultdict(list)
for split, d in data_dirs.items():
    for f in sorted(os.listdir(d)):
        m = pattern.search(f)
        if m:
            tile_id, mgrs, year, doy = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
            mask_f = tile_id + ".mask.tif"
            mask_path = os.path.join(d, mask_f)
            burn_pct = 0.0
            if os.path.exists(mask_path):
                with rasterio.open(mask_path) as src:
                    mask = src.read(1)
                burn_pct = (mask == 1).sum() / mask.size * 100
            tiles[mgrs].append(dict(
                mgrs=mgrs, year=year, doy=doy, split=split,
                tile_id=tile_id, burn_pct=round(burn_pct, 1),
                img_path=os.path.join(d, f),
                mask_path=mask_path,
            ))

# Find temporal pairs: earlier image → later image
good_pairs = []
for mgrs in sorted(tiles):
    entries = sorted(tiles[mgrs], key=lambda x: (x["year"], x["doy"]))
    if len(entries) < 2:
        continue
    for i in range(len(entries)):
        for j in range(i + 1, len(entries)):
            e1, e2 = entries[i], entries[j]
            # Good pair: later has more burn than earlier
            if e2["burn_pct"] > e1["burn_pct"] and e2["burn_pct"] > 5:
                good_pairs.append((e1, e2, e2["burn_pct"] - e1["burn_pct"]))

good_pairs.sort(key=lambda x: -x[2])
print(f"Good before->after pairs: {len(good_pairs)}")
print(f"\nTop 15 pairs (biggest burn change):")
for e1, e2, delta in good_pairs[:15]:
    d1 = f"{e1['year']}-{e1['doy']:03d}"
    d2 = f"{e2['year']}-{e2['doy']:03d}"
    print(f"  {e1['mgrs']}  {d1} ({e1['burn_pct']}%) -> {d2} ({e2['burn_pct']}%)  delta={delta:.1f}%")

# Also show pairs where BOTH have burn but different amounts (during vs after)
print(f"\nPairs where both have burn (different severity):")
both_burn = [(e1, e2, d) for e1, e2, d in good_pairs if e1["burn_pct"] > 2 and e2["burn_pct"] > 10]
for e1, e2, delta in both_burn[:10]:
    d1 = f"{e1['year']}-{e1['doy']:03d}"
    d2 = f"{e2['year']}-{e2['doy']:03d}"
    print(f"  {e1['mgrs']}  {d1} ({e1['burn_pct']}%) -> {d2} ({e2['burn_pct']}%)  delta={delta:.1f}%")

# Show pairs where earlier has LOW burn (good "before" reference)
print(f"\nBest 'before' (low burn) -> 'after' (high burn) pairs:")
clean_before = [(e1, e2, d) for e1, e2, d in good_pairs if e1["burn_pct"] < 5]
for e1, e2, delta in clean_before[:10]:
    d1 = f"{e1['year']}-{e1['doy']:03d}"
    d2 = f"{e2['year']}-{e2['doy']:03d}"
    print(f"  {e1['mgrs']}  {d1} ({e1['burn_pct']}%) -> {d2} ({e2['burn_pct']}%)  delta={delta:.1f}%")
