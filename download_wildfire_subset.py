"""
Download a subset of the WildfireSpreadTS dataset from Zenodo.
==============================================================
The full dataset is 48.4 GB — this script downloads only a few fire events 
using HTTP range requests, so you only download what you need (~1-3 GB for 5-10 fires).

Usage:
    # List available fire events (no download):
    python download_wildfire_subset.py --list

    # Download 5 fire events (default):
    python download_wildfire_subset.py

    # Download 10 specific fires or N random ones:
    python download_wildfire_subset.py --n_fires 10

    # Download specific fire events by ID:
    python download_wildfire_subset.py --fire_ids fire_21890058 fire_21889719

    # Download fires from a specific year:
    python download_wildfire_subset.py --year 2020 --n_fires 5

    # Set output directory:
    python download_wildfire_subset.py --output_dir D:/sih/Sentinel/DATA/WildfireSpreadTS
"""

import argparse
import io
import os
import struct
import sys
import time
from collections import defaultdict
from pathlib import Path

try:
    import requests
except ImportError:
    print("Installing requests...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
    import requests


# ═══════════════════════════════════════════════════════════════════
#  Zenodo file URL for WildfireSpreadTS.zip
# ═══════════════════════════════════════════════════════════════════
ZENODO_URL = "https://zenodo.org/records/8006177/files/WildfireSpreadTS.zip"

# Also download the documentation PDF (small, 7.8 MB)
DOC_URL = "https://zenodo.org/records/8006177/files/WildfireSpreadTS_Documentation.pdf"


# ═══════════════════════════════════════════════════════════════════
#  Remote ZIP reader using HTTP Range requests
# ═══════════════════════════════════════════════════════════════════

class RemoteZipReader:
    """
    Read individual files from a remote ZIP archive using HTTP Range requests.
    Only downloads the central directory + requested files (not the whole ZIP).
    """

    def __init__(self, url, session=None):
        self.url = url
        self.session = session or requests.Session()
        self.entries = []
        self._file_size = None

    def _get_file_size(self):
        """Get total ZIP file size via HEAD request."""
        if self._file_size is None:
            resp = self.session.head(self.url, allow_redirects=True)
            resp.raise_for_status()
            self._file_size = int(resp.headers["Content-Length"])
        return self._file_size

    def _read_range(self, start, end, retries=3):
        """Read byte range from remote file with retry."""
        headers = {"Range": f"bytes={start}-{end}"}
        for attempt in range(retries):
            try:
                resp = self.session.get(
                    self.url, headers=headers, allow_redirects=True, timeout=120
                )
                if resp.status_code not in (200, 206):
                    raise RuntimeError(f"HTTP {resp.status_code} for range {start}-{end}")
                return resp.content
            except (requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout,
                    requests.exceptions.ChunkedEncodingError) as e:
                if attempt < retries - 1:
                    wait = 2 ** attempt * 5
                    print(f"\n    Retry {attempt+1}/{retries} after {wait}s: {e}")
                    time.sleep(wait)
                else:
                    raise

    def read_central_directory(self):
        """
        Parse the ZIP central directory from the end of the file.
        This tells us what files are in the ZIP and where they are located.
        """
        file_size = self._get_file_size()
        print(f"  Remote ZIP size: {file_size / 1e9:.2f} GB")

        # Read the End of Central Directory record (last 64KB should be enough)
        eocd_search_size = min(65536, file_size)
        eocd_data = self._read_range(file_size - eocd_search_size, file_size - 1)

        # Find EOCD signature (0x06054b50)
        eocd_sig = b"\x50\x4b\x05\x06"
        eocd_pos = eocd_data.rfind(eocd_sig)
        if eocd_pos == -1:
            # Try ZIP64 EOCD
            return self._read_central_directory_zip64(file_size, eocd_data)

        eocd = eocd_data[eocd_pos:]
        (_, _, _, _, total_entries, cd_size, cd_offset) = struct.unpack_from(
            "<IHHHHII", eocd, 0
        )

        # Check for ZIP64 (values are 0xFFFF or 0xFFFFFFFF)
        if total_entries == 0xFFFF or cd_offset == 0xFFFFFFFF or cd_size == 0xFFFFFFFF:
            return self._read_central_directory_zip64(file_size, eocd_data)

        print(f"  Total entries in ZIP: {total_entries}")
        print(f"  Central directory: {cd_size / 1024:.0f} KB at offset {cd_offset}")

        # Read the central directory
        cd_data = self._read_range(cd_offset, cd_offset + cd_size - 1)
        self._parse_cd_entries(cd_data, total_entries)
        return self.entries

    def _read_central_directory_zip64(self, file_size, eocd_data):
        """Handle ZIP64 format."""
        # Find ZIP64 EOCD Locator (0x07064b50)
        loc_sig = b"\x50\x4b\x06\x07"
        loc_pos = eocd_data.rfind(loc_sig)
        if loc_pos == -1:
            # Try reading more from the end
            search_size = min(1 * 1024 * 1024, file_size)
            eocd_data = self._read_range(file_size - search_size, file_size - 1)
            loc_pos = eocd_data.rfind(loc_sig)
            if loc_pos == -1:
                raise RuntimeError("Cannot find ZIP64 EOCD locator")

        # Parse locator: sig(4) + disk(4) + eocd64_offset(8) + total_disks(4)
        eocd64_offset = struct.unpack_from("<Q", eocd_data, loc_pos + 8)[0]

        # Read ZIP64 EOCD record
        eocd64_data = self._read_range(eocd64_offset, eocd64_offset + 56 - 1)
        sig = struct.unpack_from("<I", eocd64_data, 0)[0]
        if sig != 0x06064b50:
            raise RuntimeError(f"Invalid ZIP64 EOCD signature: {sig:#x}")

        # Parse ZIP64 EOCD: after sig(4)+size(8)+version(2)+version(2)+disk(4)+disk_cd(4):
        # total_entries_disk(8) + total_entries(8) + cd_size(8) + cd_offset(8)
        total_entries = struct.unpack_from("<Q", eocd64_data, 32)[0]
        cd_size = struct.unpack_from("<Q", eocd64_data, 40)[0]
        cd_offset = struct.unpack_from("<Q", eocd64_data, 48)[0]

        print(f"  ZIP64 format detected")
        print(f"  Total entries in ZIP: {total_entries}")
        print(f"  Central directory: {cd_size / 1024 / 1024:.1f} MB at offset {cd_offset}")

        # Read central directory (may need multiple chunks for large CDs)
        cd_data = b""
        chunk_size = 50 * 1024 * 1024  # 50 MB chunks
        offset = cd_offset
        remaining = cd_size
        while remaining > 0:
            read_size = min(chunk_size, remaining)
            chunk = self._read_range(offset, offset + read_size - 1)
            cd_data += chunk
            offset += read_size
            remaining -= read_size
            if remaining > 0:
                print(f"    Reading central directory: {len(cd_data) / 1024 / 1024:.1f} / {cd_size / 1024 / 1024:.1f} MB")

        self._parse_cd_entries(cd_data, total_entries)
        return self.entries

    def _parse_cd_entries(self, cd_data, expected_count):
        """Parse central directory file headers."""
        self.entries = []
        offset = 0
        cd_sig = 0x02014b50

        while offset < len(cd_data) and len(self.entries) < expected_count:
            sig = struct.unpack_from("<I", cd_data, offset)[0]
            if sig != cd_sig:
                break

            (_, version, version_needed, flags, compression, mod_time, mod_date,
             crc32, compressed_size, uncompressed_size, name_len, extra_len,
             comment_len, disk_start, internal_attr, external_attr,
             local_header_offset) = struct.unpack_from(
                "<IHHHHHHIIIHHHHHII", cd_data, offset
            )

            name_start = offset + 46
            name = cd_data[name_start:name_start + name_len].decode("utf-8", errors="replace")

            # Parse extra field for ZIP64 sizes
            extra_start = name_start + name_len
            extra_data = cd_data[extra_start:extra_start + extra_len]
            if compressed_size == 0xFFFFFFFF or uncompressed_size == 0xFFFFFFFF or local_header_offset == 0xFFFFFFFF:
                # Parse ZIP64 extra field
                ep = 0
                while ep < len(extra_data) - 4:
                    tag, sz = struct.unpack_from("<HH", extra_data, ep)
                    if tag == 0x0001:  # ZIP64
                        vals = []
                        vp = ep + 4
                        for _ in range(sz // 8):
                            vals.append(struct.unpack_from("<Q", extra_data, vp)[0])
                            vp += 8
                        vi = 0
                        if uncompressed_size == 0xFFFFFFFF and vi < len(vals):
                            uncompressed_size = vals[vi]; vi += 1
                        if compressed_size == 0xFFFFFFFF and vi < len(vals):
                            compressed_size = vals[vi]; vi += 1
                        if local_header_offset == 0xFFFFFFFF and vi < len(vals):
                            local_header_offset = vals[vi]; vi += 1
                        break
                    ep += 4 + sz

            self.entries.append({
                "name": name,
                "compressed_size": compressed_size,
                "uncompressed_size": uncompressed_size,
                "compression": compression,
                "offset": local_header_offset,
                "crc32": crc32,
            })

            offset = extra_start + extra_len + comment_len

        print(f"  Parsed {len(self.entries)} entries")

    def download_file(self, entry, output_dir):
        """Download a single file from the ZIP."""
        name = entry["name"]
        if name.endswith("/"):
            # Directory entry, just create it
            dir_path = os.path.join(output_dir, name)
            os.makedirs(dir_path, exist_ok=True)
            return dir_path

        # Read local file header first (30 bytes + name + extra)
        header_data = self._read_range(entry["offset"], entry["offset"] + 29)
        sig = struct.unpack_from("<I", header_data, 0)[0]
        if sig != 0x04034b50:
            raise RuntimeError(f"Invalid local header signature for {name}")

        name_len = struct.unpack_from("<H", header_data, 26)[0]
        extra_len = struct.unpack_from("<H", header_data, 28)[0]
        data_offset = entry["offset"] + 30 + name_len + extra_len

        # Download the compressed data
        compressed_data = self._read_range(
            data_offset, data_offset + entry["compressed_size"] - 1
        )

        # Decompress if needed
        if entry["compression"] == 0:
            file_data = compressed_data
        elif entry["compression"] == 8:
            import zlib
            file_data = zlib.decompress(compressed_data, -15)
        else:
            raise RuntimeError(f"Unsupported compression method: {entry['compression']}")

        # Write to disk
        out_path = os.path.join(output_dir, name)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(file_data)

        return out_path


# ═══════════════════════════════════════════════════════════════════
#  Dataset discovery & selection
# ═══════════════════════════════════════════════════════════════════

def categorize_entries(entries):
    """
    Group ZIP entries by year and fire event.
    Returns: {year: {fire_id: [entries]}}
    """
    fires = defaultdict(lambda: defaultdict(list))
    for entry in entries:
        parts = entry["name"].split("/")
        # Expected: WildfireSpreadTS/YEAR/fire_ID/DATE.tif  OR  YEAR/fire_ID/DATE.tif
        # Find the year and fire_id parts
        for i, part in enumerate(parts):
            if part.isdigit() and 2015 <= int(part) <= 2025:
                year = part
                if i + 1 < len(parts):
                    fire_id = parts[i + 1]
                    if fire_id.startswith("fire_"):
                        fires[year][fire_id].append(entry)
                break
    return fires


def select_fire_events(fires, args):
    """Select which fire events to download based on args."""
    selected = []

    if args.fire_ids:
        # User specified exact fire IDs
        for year in fires:
            for fire_id in fires[year]:
                if fire_id in args.fire_ids:
                    selected.append((year, fire_id, fires[year][fire_id]))
    elif args.year:
        # Filter by year
        year = str(args.year)
        if year in fires:
            fire_ids = sorted(fires[year].keys())
            for fid in fire_ids[:args.n_fires]:
                selected.append((year, fid, fires[year][fid]))
        else:
            print(f"  Year {year} not found. Available: {sorted(fires.keys())}")
    else:
        # Pick fires with the most days (most interesting for temporal analysis)
        all_fires = []
        for year in fires:
            for fire_id in fires[year]:
                n_days = len([e for e in fires[year][fire_id]
                             if e["name"].endswith(".tif")])
                total_size = sum(e["compressed_size"] for e in fires[year][fire_id])
                all_fires.append((year, fire_id, n_days, total_size, fires[year][fire_id]))

        # Sort by number of days (more days = longer fire = more interesting)
        all_fires.sort(key=lambda x: -x[2])

        for year, fire_id, n_days, total_size, entries in all_fires[:args.n_fires]:
            selected.append((year, fire_id, entries))

    return selected


# ═══════════════════════════════════════════════════════════════════
#  Main pipeline
# ═══════════════════════════════════════════════════════════════════

def list_fires(fires):
    """Print all available fire events."""
    print("\n" + "=" * 70)
    print("  Available Fire Events in WildfireSpreadTS")
    print("=" * 70)

    total_fires = 0
    total_days = 0
    for year in sorted(fires.keys()):
        n_fires_year = len(fires[year])
        total_fires += n_fires_year
        print(f"\n  Year {year}: {n_fires_year} fire events")
        print(f"  {'Fire ID':<20} {'Days':>5}  {'Size':>10}")
        print(f"  {'─' * 40}")

        fire_list = []
        for fire_id in sorted(fires[year].keys()):
            entries = fires[year][fire_id]
            n_days = len([e for e in entries if e["name"].endswith(".tif")])
            total_size = sum(e["compressed_size"] for e in entries)
            fire_list.append((fire_id, n_days, total_size))
            total_days += n_days

        fire_list.sort(key=lambda x: -x[1])
        for fire_id, n_days, total_size in fire_list:
            print(f"  {fire_id:<20} {n_days:>5}  {total_size / 1e6:>8.1f} MB")

    print(f"\n  Total: {total_fires} fires, {total_days} daily observations")
    print("=" * 70)


def download_subset(args):
    """Main download function."""
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("  WildfireSpreadTS — Partial Download")
    print("=" * 70)

    session = requests.Session()
    # Set reasonable timeout and retries
    adapter = requests.adapters.HTTPAdapter(max_retries=3)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    reader = RemoteZipReader(ZENODO_URL, session)

    # Step 1: Read central directory
    print("\n[1] Reading ZIP central directory (remote) ...")
    t0 = time.time()
    try:
        entries = reader.read_central_directory()
    except Exception as e:
        print(f"\n  ERROR reading central directory: {e}")
        print("  The Zenodo server may not support Range requests.")
        print("  Alternative: Use the streaming fallback below.")
        print(f"\n  You can also manually download a subset using:")
        print(f"    curl -L -r 0-1073741823 -o partial.zip {ZENODO_URL}")
        return
    t_cd = time.time() - t0
    print(f"  Central directory read in {t_cd:.1f}s")

    # Step 2: Categorize entries
    fires = categorize_entries(entries)
    total_fires = sum(len(v) for v in fires.values())
    print(f"  Found {total_fires} fire events across years: {sorted(fires.keys())}")

    if args.list:
        list_fires(fires)
        return

    # Step 3: Select fires to download
    print(f"\n[2] Selecting fire events ...")
    selected = select_fire_events(fires, args)
    if not selected:
        print("  No fires matched your criteria.")
        print("  Try: --list to see available fires, or --year 2020 --n_fires 5")
        return

    # Calculate total download size
    total_download = 0
    print(f"\n  Selected {len(selected)} fire events:")
    print(f"  {'Year':<6} {'Fire ID':<20} {'Days':>5}  {'Size':>10}")
    print(f"  {'─' * 45}")
    for year, fire_id, fire_entries in selected:
        n_days = len([e for e in fire_entries if e["name"].endswith(".tif")])
        size = sum(e["compressed_size"] for e in fire_entries)
        total_download += size
        print(f"  {year:<6} {fire_id:<20} {n_days:>5}  {size / 1e6:>8.1f} MB")
    print(f"  {'─' * 45}")
    print(f"  Total download: {total_download / 1e9:.2f} GB")

    # Step 4: Download
    print(f"\n[3] Downloading to {output_dir} ...")
    t_start = time.time()
    total_files = 0
    downloaded_bytes = 0

    for fi, (year, fire_id, fire_entries) in enumerate(selected):
        tif_entries = [e for e in fire_entries if e["name"].endswith(".tif")]
        n_days = len(tif_entries)
        fire_size = sum(e["compressed_size"] for e in fire_entries)

        print(f"\n  [{fi+1}/{len(selected)}] {fire_id} ({year}) — {n_days} days, "
              f"{fire_size / 1e6:.1f} MB")

        for ei, entry in enumerate(fire_entries):
            if entry["name"].endswith("/"):
                continue  # skip directory entries

            # Skip already-downloaded files (resume support)
            out_path_check = os.path.join(output_dir, entry["name"])
            if os.path.exists(out_path_check):
                existing_size = os.path.getsize(out_path_check)
                if existing_size == entry["uncompressed_size"]:
                    downloaded_bytes += entry["compressed_size"]
                    total_files += 1
                    continue  # already downloaded

            try:
                out_path = reader.download_file(entry, output_dir)
                downloaded_bytes += entry["compressed_size"]
                total_files += 1

                # Progress
                if (ei + 1) % 5 == 0 or ei == len(fire_entries) - 1:
                    pct = downloaded_bytes / total_download * 100
                    elapsed = time.time() - t_start
                    speed = downloaded_bytes / elapsed / 1e6 if elapsed > 0 else 0
                    sys.stdout.write(
                        f"\r    Files: {ei+1}/{len(fire_entries)}  "
                        f"| Total: {downloaded_bytes / 1e6:.0f} / {total_download / 1e6:.0f} MB "
                        f"({pct:.1f}%)  | {speed:.1f} MB/s"
                    )
                    sys.stdout.flush()
            except Exception as e:
                print(f"\n    WARNING: Failed to download {entry['name']}: {e}")

        print()  # newline after progress

    elapsed = time.time() - t_start

    # Step 5: Download documentation PDF
    print(f"\n[4] Downloading documentation PDF ...")
    try:
        doc_resp = session.get(DOC_URL, allow_redirects=True)
        doc_path = os.path.join(output_dir, "WildfireSpreadTS_Documentation.pdf")
        with open(doc_path, "wb") as f:
            f.write(doc_resp.content)
        print(f"    Saved: {doc_path} ({len(doc_resp.content) / 1e6:.1f} MB)")
    except Exception as e:
        print(f"    WARNING: Could not download docs: {e}")

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  DOWNLOAD COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Files downloaded:  {total_files}")
    print(f"  Total size:        {downloaded_bytes / 1e9:.2f} GB")
    print(f"  Time:              {elapsed:.0f}s ({downloaded_bytes / elapsed / 1e6:.1f} MB/s)")
    print(f"  Output directory:  {output_dir}")
    print(f"\n  Dataset structure:")
    print(f"    {output_dir}/")
    for year, fire_id, _ in selected[:5]:
        print(f"      {year}/{fire_id}/  (daily .tif files)")
    if len(selected) > 5:
        print(f"      ... and {len(selected) - 5} more fire events")

    # Space saved
    full_size_gb = 48.4
    saved_gb = full_size_gb - downloaded_bytes / 1e9
    print(f"\n  Space saved: ~{saved_gb:.1f} GB "
          f"(downloaded {downloaded_bytes / 1e9:.2f} GB instead of {full_size_gb} GB)")
    print(f"{'=' * 70}")


# ═══════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Download a subset of WildfireSpreadTS from Zenodo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available fire events (no download):
  python download_wildfire_subset.py --list

  # Download 5 fires with the longest time series:
  python download_wildfire_subset.py --n_fires 5

  # Download 3 fires from 2020:
  python download_wildfire_subset.py --year 2020 --n_fires 3

  # Download specific fire events:
  python download_wildfire_subset.py --fire_ids fire_21890058 fire_21889719

Storage estimates:
  5 fires  ≈ 0.5-1.5 GB
  10 fires ≈ 1-3 GB
  20 fires ≈ 2-5 GB
  Full     = 48.4 GB
"""
    )
    p.add_argument("--list", action="store_true",
                   help="List all available fire events (no download)")
    p.add_argument("--n_fires", type=int, default=5,
                   help="Number of fire events to download (default: 5)")
    p.add_argument("--fire_ids", nargs="*", default=None,
                   help="Specific fire event IDs to download")
    p.add_argument("--year", type=int, default=None,
                   help="Only download fires from this year")
    p.add_argument("--output_dir", type=str,
                   default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "DATA", "WildfireSpreadTS"),
                   help="Output directory")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    download_subset(args)
