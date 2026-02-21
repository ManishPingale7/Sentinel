"""Quick script to analyze fire tile groupings by MGRS grid cell."""
import glob, os, re
from collections import defaultdict

tiles = defaultdict(list)
for split in ['training', 'validation']:
    for f in glob.glob(os.path.join('D:/sih/Sentinel/DATA/Fire', split, '*_merged.tif')):
        bn = os.path.basename(f)
        m = re.search(r'HLS\.S30\.(T\w+)\.(\d{4})(\d{3})\.', bn)
        if m:
            mgrs, year, doy = m.group(1), m.group(2), m.group(3)
            tiles[mgrs].append(dict(year=year, doy=doy, split=split))

print(f'Unique MGRS cells: {len(tiles)}')
for mgrs in sorted(tiles):
    entries = tiles[mgrs]
    dates = []
    for e in entries:
        dates.append(f"{e['year']}-{e['doy']}({e['split'][0]})")
    print(f'  {mgrs}: {len(entries)} tiles  [{", ".join(dates)}]')
