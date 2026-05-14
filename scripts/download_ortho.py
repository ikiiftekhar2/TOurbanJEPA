#!/usr/bin/env python3
"""
Download Toronto ortho imagery using the ArcGIS /export endpoint.
Based on the H3MRL spiral scraper — downloads 4096x4096 tiles at ~15cm/pixel
in a spiral pattern from the center of Toronto outward. Resumable via manifest.

Usage:
    # Start/resume download at default settings (L20, ~15cm/px)
    python scripts/download_ortho.py

    # Different zoom level
    python scripts/download_ortho.py --level 19   # ~30cm/px
    python scripts/download_ortho.py --level 21   # ~7.5cm/px

    # Custom output dir and workers
    python scripts/download_ortho.py --output data/ortho/tiles --workers 16

Resolution reference:
    L19 = 0.29858214164761667 m/px  (~30 cm/px, 1222m tile width)
    L20 = 0.14929107082380833 m/px  (~15 cm/px,  611m tile width)
    L21 = 0.07464553541190417 m/px  (~7.5 cm/px, 306m tile width)
"""

import os
import sys
import json
import math
import time
import signal
import urllib.parse
from datetime import datetime
from pathlib import Path

import requests
from tqdm import tqdm

# ===================== CONFIG =====================
SERVICE = "https://gis.toronto.ca/arcgis/rest/services/basemap/cot_ortho/MapServer/export"

# Full Toronto ortho extent (EPSG:3857 meters)
TORONTO_3857 = {
    "xmin": -8869931.9307,
    "ymin": 5395816.8838,
    "xmax": -8802055.8621,
    "ymax": 5451532.5866,
}

# ROM center (EPSG:3857)
CENTER_X = -8836061.0
CENTER_Y = 5409486.0

# Resolution presets for each zoom level (m/px)
LEVEL_RESOLUTIONS = {
    18: 0.5971642832952333,
    19: 0.29858214164761667,
    20: 0.14929107082380833,
    21: 0.07464553541190417,
}

PIX_MAX = 4096  # pixels per tile edge
EDGE_TRIM = 1.0  # meter trim on bounds

# Networking
SLEEP_BETWEEN = 0.10
REQUEST_TIMEOUT = 180
MAX_RETRIES = 6
BACKOFF_BASE = 1.6
USER_AGENT = "UrbanJEPA-ortho-spiral/1.0 (+local)"

# Blank detection & stopping
MIN_FILE_BYTES = 1000
RING_STOP_RATIO = 0.95

# Checkpoint cadence
CHECKPOINT_EVERY_N = 25
CHECKPOINT_EVERY_SEC = 20

# =================== END CONFIG ===================


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def write_jgw(jgw_path, res, ulx, uly):
    with open(jgw_path, "w") as f:
        f.write(f"{res:.15f}\n")
        f.write("0.0\n")
        f.write("0.0\n")
        f.write(f"{-res:.15f}\n")
        f.write(f"{ulx + res / 2:.15f}\n")
        f.write(f"{uly - res / 2:.15f}\n")


def jpg_export(session, bbox, size_xy, out_path):
    xmin, ymin, xmax, ymax = bbox
    width_px, height_px = size_xy
    params = {
        "f": "image",
        "format": "jpg",
        "transparent": "false",
        "bboxSR": "102100",
        "imageSR": "102100",
        "bbox": f"{xmin},{ymin},{xmax},{ymax}",
        "size": f"{width_px},{height_px}",
    }
    url = SERVICE + "?" + urllib.parse.urlencode(params)
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.get(url, timeout=REQUEST_TIMEOUT)
            if r.status_code == 429:
                ra = r.headers.get("Retry-After")
                wait = float(ra) if (ra and ra.isdigit()) else (BACKOFF_BASE**attempt)
                time.sleep(wait)
                continue
            r.raise_for_status()
            out_path.write_bytes(r.content)
            return len(r.content)
        except Exception as e:
            last_err = e
            time.sleep(BACKOFF_BASE**attempt)
    raise last_err


def spiral_offsets():
    """Generate spiral ring offsets: (0,0), (1,0),(1,1),(0,1),(-1,1),(-1,0),..."""
    x = y = 0
    dx, dy = 1, 0
    step_len = 1
    yield (0, 0)
    while True:
        for _ in range(2):
            for _ in range(step_len):
                x, y = x + dx, y + dy
                yield (x, y)
            dx, dy = -dy, dx
        step_len += 1


def chunk_index_from_xy(x, y, origin_x, origin_y, stride_m):
    col = int(math.floor((x - origin_x) / stride_m))
    row = int(math.floor((y - origin_y) / stride_m))
    return col, row


def chunk_bounds(col, row, origin_x, origin_y, stride_m, xmin_b, ymin_b, xmax_b, ymax_b):
    xmin = origin_x + col * stride_m
    ymin = origin_y + row * stride_m
    return (
        clamp(xmin, xmin_b, xmax_b),
        clamp(ymin, ymin_b, ymax_b),
        clamp(xmin + stride_m, xmin_b, xmax_b),
        clamp(ymin + stride_m, ymin_b, ymax_b),
    )


def manifest_init(path, out_dir, level, res):
    return {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "last_updated": None,
        "config": {
            "service": SERVICE,
            "level": level,
            "res": res,
            "pix_max": PIX_MAX,
            "bounds": [
                TORONTO_3857["xmin"] + EDGE_TRIM,
                TORONTO_3857["ymin"] + EDGE_TRIM,
                TORONTO_3857["xmax"] - EDGE_TRIM,
                TORONTO_3857["ymax"] - EDGE_TRIM,
            ],
            "center": [CENTER_X, CENTER_Y],
            "out_dir": str(out_dir),
            "ring_stop_ratio": RING_STOP_RATIO,
            "min_file_bytes": MIN_FILE_BYTES,
        },
        "stats": {
            "tiles_total_est": 0,
            "tiles_done": 0,
            "tiles_blank": 0,
            "tiles_error": 0,
            "bytes_downloaded": 0,
        },
        "tiles": {},
        "stopped_reason": None,
    }


def manifest_load(path, out_dir, level, res):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            m = json.load(f)
        m["config"]["out_dir"] = str(out_dir)
        return m
    return manifest_init(path, out_dir, level, res)


def manifest_save(path, m):
    m["last_updated"] = datetime.utcnow().isoformat() + "Z"
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


# Graceful interrupt handler
_interrupted = False


def _sigint_handler(signum, frame):
    global _interrupted
    _interrupted = True


signal.signal(signal.SIGINT, _sigint_handler)


def download_full_extent(level=20, output_dir=None, max_workers=None):
    """Spiral-download all Toronto ortho tiles at the given zoom level."""
    output_dir = Path(output_dir) if output_dir else Path("data/ortho/tiles")
    output_dir.mkdir(parents=True, exist_ok=True)

    res = LEVEL_RESOLUTIONS.get(level)
    if res is None:
        print(f"ERROR: Unknown level {level}. Choose from {list(LEVEL_RESOLUTIONS.keys())}")
        raise SystemExit(1)

    stride_m = res * PIX_MAX
    xmin_b = TORONTO_3857["xmin"] + EDGE_TRIM
    ymin_b = TORONTO_3857["ymin"] + EDGE_TRIM
    xmax_b = TORONTO_3857["xmax"] - EDGE_TRIM
    ymax_b = TORONTO_3857["ymax"] - EDGE_TRIM

    level_prefix = f"L{level}"

    manifest_path = output_dir / f"manifest_{level_prefix}.json"
    input_list_path = output_dir / f"input_files_{level_prefix}.txt"

    # Center chunk and grid extents
    cc, cr = chunk_index_from_xy(CENTER_X, CENTER_Y, xmin_b, ymin_b, stride_m)
    c_min, r_min = chunk_index_from_xy(xmin_b, ymin_b, xmin_b, ymin_b, stride_m)
    c_max, r_max = chunk_index_from_xy(xmax_b - 1e-6, ymax_b - 1e-6, xmin_b, ymin_b, stride_m)
    max_radius = max(abs(cc - c_min), abs(cc - c_max), abs(cr - r_min), abs(cr - r_max))

    # Load or create manifest
    m = manifest_load(manifest_path, output_dir, level, res)
    m["stats"]["tiles_total_est"] = (2 * max_radius + 1) ** 2

    # Resume: scan existing files on disk
    for name in os.listdir(output_dir):
        if not name.lower().endswith(".jpg"):
            continue
        base = os.path.splitext(name)[0]
        parts = base.split("_")
        try:
            r = int([p for p in parts if p.startswith("r")][0][1:])
            c = int([p for p in parts if p.startswith("c")][0][1:])
        except Exception:
            continue
        key = f"r{r}_c{c}"
        img_path = output_dir / name
        jgw_path = output_dir / (base + ".jgw")
        sz = img_path.stat().st_size
        rec = m["tiles"].get(
            key,
            {
                "row": r,
                "col": c,
                "ring": None,
                "status": "pending",
                "file": str(img_path),
                "worldfile": str(jgw_path),
                "bytes": 0,
                "tries": 0,
            },
        )
        rec["file"] = str(img_path)
        rec["worldfile"] = str(jgw_path)
        if sz >= MIN_FILE_BYTES:
            if rec.get("status") != "done":
                rec["status"] = "done"
                m["stats"]["tiles_done"] += 1
                m["stats"]["bytes_downloaded"] += sz
        else:
            if rec.get("status") not in ("blank", "error"):
                rec["status"] = "blank"
                m["stats"]["tiles_blank"] += 1
        m["tiles"][key] = rec

    # Print status
    print("=" * 60)
    print(f"UrbanJEPA — Toronto Ortho Download (L{level}, ~{res*100:.0f}cm/px)")
    print("=" * 60)
    print(f"  Res:        {res:.6f} m/px (~{res*100:.0f} cm/px)")
    print(f"  Tile size:  {PIX_MAX}x{PIX_MAX} px ({stride_m:.0f}m x {stride_m:.0f}m)")
    print(f"  Grid:       cols {c_min}..{c_max}, rows {r_min}..{r_max}")
    print(f"  Max radius: {max_radius} rings")
    print(f"  Output:     {output_dir}")
    print(f"  Existing:   {m['stats']['tiles_done']} done, "
          f"{m['stats']['tiles_blank']} blank ({m['stats']['bytes_downloaded']/1e6:.0f} MB)")
    print()

    # HTTP session
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    # Spiral bookkeeping
    visited = set((t["row"], t["col"]) for t in m["tiles"].values())
    blank_counts_by_ring = {}
    tiles_in_ring = {}

    completed = 0
    downloaded_bytes = 0
    last_checkpoint = time.time()

    pbar = tqdm(total=m["stats"]["tiles_total_est"], desc=f"L{level} spiral", unit="tile")

    try:
        for dx, dy in spiral_offsets():
            if _interrupted:
                m["stopped_reason"] = "Interrupted (Ctrl-C)"
                break

            col, row = cc + dx, cr + dy
            ring = max(abs(dx), abs(dy))
            if ring > max_radius:
                m["stopped_reason"] = "Reached max radius"
                break

            # Outside bounds
            if col < c_min or col > c_max or row < r_min or row > r_max:
                tiles_in_ring[ring] = tiles_in_ring.get(ring, 0) + 1
                blank_counts_by_ring[ring] = blank_counts_by_ring.get(ring, 0) + 1
                if ring >= 2:
                    prev_total = tiles_in_ring.get(ring - 1, 0)
                    prev_blank = blank_counts_by_ring.get(ring - 1, 0)
                    if prev_total and prev_blank >= int(RING_STOP_RATIO * prev_total):
                        total = tiles_in_ring.get(ring, 0)
                        blanks = blank_counts_by_ring.get(ring, 0)
                        if total and blanks >= int(RING_STOP_RATIO * total):
                            m["stopped_reason"] = "Two mostly-blank rings at edge"
                            break
                pbar.update(1)
                continue

            if (col, row) in visited:
                pbar.update(1)
                continue
            visited.add((col, row))

            # Compute tile bounds
            xmin, ymin, xmax, ymax = chunk_bounds(
                col, row, xmin_b, ymin_b, stride_m, xmin_b, ymin_b, xmax_b, ymax_b
            )
            w_px = PIX_MAX if (xmax - xmin) >= stride_m * 0.999 else int(round((xmax - xmin) / res))
            h_px = PIX_MAX if (ymax - ymin) >= stride_m * 0.999 else int(round((ymax - ymin) / res))

            key = f"r{row}_c{col}"
            base = f"tile_{level_prefix}_r{row}_c{col}"
            img_path = output_dir / (base + ".jpg")
            jgw_path = output_dir / (base + ".jgw")

            rec = m["tiles"].get(
                key,
                {
                    "row": row,
                    "col": col,
                    "ring": ring,
                    "status": "pending",
                    "file": str(img_path),
                    "worldfile": str(jgw_path),
                    "bytes": 0,
                    "tries": 0,
                    "bbox": [xmin, ymin, xmax, ymax],
                    "size": [w_px, h_px],
                },
            )
            rec.update(
                {
                    "ring": ring,
                    "file": str(img_path),
                    "worldfile": str(jgw_path),
                    "bbox": [xmin, ymin, xmax, ymax],
                    "size": [w_px, h_px],
                }
            )
            m["tiles"][key] = rec

            # Skip already-done tiles
            if (
                img_path.exists()
                and img_path.stat().st_size >= MIN_FILE_BYTES
                and rec.get("status") == "done"
            ):
                tiles_in_ring[ring] = tiles_in_ring.get(ring, 0) + 1
                pbar.update(1)
                continue

            # Download
            try:
                nbytes = jpg_export(session, (xmin, ymin, xmax, ymax), (w_px, h_px), img_path)
                rec["tries"] += 1
                file_bytes = img_path.stat().st_size
                if file_bytes < MIN_FILE_BYTES:
                    rec["status"] = "blank"
                    rec["bytes"] = file_bytes
                    m["stats"]["tiles_blank"] += 1
                    blank_counts_by_ring[ring] = blank_counts_by_ring.get(ring, 0) + 1
                else:
                    write_jgw(jgw_path, res, xmin, ymax)
                    rec["status"] = "done"
                    rec["bytes"] = file_bytes
                    m["stats"]["tiles_done"] += 1
                    m["stats"]["bytes_downloaded"] += file_bytes
                    downloaded_bytes += file_bytes
                    with open(input_list_path, "a", encoding="utf-8") as f:
                        f.write(str(img_path) + "\n")
                    completed += 1
            except Exception as e:
                rec["tries"] += 1
                rec["status"] = "error"
                rec["last_error"] = str(e)
                m["stats"]["tiles_error"] += 1
                blank_counts_by_ring[ring] = blank_counts_by_ring.get(ring, 0) + 1

            m["tiles"][key] = rec
            tiles_in_ring[ring] = tiles_in_ring.get(ring, 0) + 1

            # Ring stopping check
            if ring >= 2:
                prev_total = tiles_in_ring.get(ring - 1, 0)
                prev_blank = blank_counts_by_ring.get(ring - 1, 0)
                if prev_total and prev_blank >= int(RING_STOP_RATIO * prev_total):
                    total = tiles_in_ring.get(ring, 0)
                    blanks = blank_counts_by_ring.get(ring, 0)
                    if total and blanks >= int(RING_STOP_RATIO * total):
                        m["stopped_reason"] = "Two mostly-blank rings at edge"
                        manifest_save(manifest_path, m)
                        break

            # Periodic checkpoint
            now = time.time()
            if (completed % CHECKPOINT_EVERY_N == 0 and completed > 0) or (
                now - last_checkpoint
            ) >= CHECKPOINT_EVERY_SEC:
                manifest_save(manifest_path, m)
                last_checkpoint = now

            pbar.update(1)
            time.sleep(SLEEP_BETWEEN)

    finally:
        pbar.close()
        manifest_save(manifest_path, m)

    # Summary
    elapsed = time.time() - last_checkpoint + 0.01  # rough, but fine for summary
    print(f"\nDone. Stopped reason: {m.get('stopped_reason', 'completed')}")
    print(f"  Downloaded this run: {completed} tiles, {downloaded_bytes/1e6:.0f} MB")
    print(f"  Totals: done={m['stats']['tiles_done']}, "
          f"blank={m['stats']['tiles_blank']}, "
          f"error={m['stats']['tiles_error']}, "
          f"bytes={m['stats']['bytes_downloaded']/1e6:.0f} MB")
    print(f"  Manifest: {manifest_path}")
    print(f"  Resume with same command to continue.")

    return m["stats"]["tiles_done"], m["stats"]["tiles_blank"]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download Toronto ortho tiles via spiral /export (resumable)"
    )
    parser.add_argument(
        "--level", type=int, default=20,
        help=f"Zoom/resolution level ({', '.join(str(k) + '=' + str(int(v*100)) + 'cm' for k, v in LEVEL_RESOLUTIONS.items())})"
    )
    parser.add_argument(
        "--output", type=str,
        default="data/ortho/tiles",
        help="Output directory for tiles"
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Ignored (single-threaded /export is required). Accepted for compat."
    )
    args = parser.parse_args()

    download_full_extent(level=args.level, output_dir=args.output)
