#!/usr/bin/env python3
"""
Import H3MRL's pre-downloaded Toronto ortho tiles into UrbanJEPA's data directory.

What this does:
  1. Reads manifest_nonwhite.json to find all valid (non-white) tiles
  2. Copies 4096x4096 tiles + JGW world files to data/ortho/tiles/
  3. Pre-extracts 256x256 patches from each tile into data/ortho/patches/
  4. Creates patch_index.csv mapping every patch back to its parent tile
  5. Copies metadata files (manifest, bounds, filter report)

Tile source:  /mnt/eskeetit/Code-server/H3MRL/datafetch/rom_L20_spiral_jpg
Tile dest:    data/ortho/tiles/
Patch dest:   data/ortho/patches/

Tile naming:  tile_L20_r{row}_c{col}.jpg  (4096x4096, ~15cm/px)
Patch naming: tile_L20_r{row}_c{col}_pr{py}_pc{px}.jpg  (256x256)

The patch index allows reconstructing which patches belong to which tile,
so you can reassemble full 4096x4096 images for visualization/evaluation.
"""

import json
import shutil
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from PIL import Image
from tqdm import tqdm

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
H3MRL_DIR = Path("/mnt/eskeetit/Code-server/H3MRL/datafetch/rom_L20_spiral_jpg")
TILES_OUT = PROJECT_ROOT / "data" / "ortho" / "tiles"
PATCHES_OUT = PROJECT_ROOT / "data" / "ortho" / "patches"

PATCH_SIZE = 256
TILE_SIZE = 4096
PATCHES_PER_SIDE = TILE_SIZE // PATCH_SIZE  # 16
PATCHES_PER_TILE = PATCHES_PER_SIDE * PATCHES_PER_SIDE  # 256

# --- Load manifest ---
manifest_path = H3MRL_DIR / "manifest_nonwhite.json"
if not manifest_path.exists():
    print(f"ERROR: {manifest_path} not found. Run file_cleaner.ipynb first?")
    raise SystemExit(1)

with open(manifest_path) as f:
    manifest = json.load(f)

tiles_meta = manifest.get("tiles", {})
print(f"Non-white tiles in manifest: {len(tiles_meta):,}")

# --- Resolve tile paths ---
out_dir_cfg = Path(manifest.get("config", {}).get("out_dir", "."))
valid_tiles = []  # (key, img_path, jgw_path)

for key, meta in tiles_meta.items():
    img = Path(meta.get("file", ""))
    jgw = Path(meta.get("worldfile", ""))
    if not img.is_absolute():
        img = (out_dir_cfg / img).resolve()
    if not jgw.is_absolute():
        jgw = (out_dir_cfg / jgw).resolve()
    if img.exists():
        valid_tiles.append((key, img, jgw))

print(f"Tiles with valid files on disk: {len(valid_tiles):,}")


def copy_one_tile(args):
    """Copy one tile (JPG + JGW) from H3MRL to UrbanJEPA."""
    key, src_img, src_jgw = args
    dest_img = TILES_OUT / src_img.name
    dest_jgw = TILES_OUT / src_jgw.name

    if dest_img.exists() and dest_jgw.exists():
        return "skip", key

    try:
        shutil.copy2(src_img, dest_img)
        if src_jgw.exists():
            shutil.copy2(src_jgw, dest_jgw)
        return "ok", key
    except Exception as e:
        return "fail", (key, str(e))


def extract_patches(args):
    """Extract 256x256 patches from a single 4096x4096 tile."""
    key, src_img, src_jgw = args

    # Derive row/col from tile name: tile_L20_r{row}_c{col}.jpg
    stem = src_img.stem  # tile_L20_r22_c55
    parts = stem.split("_")
    row_str = next((p[1:] for p in parts if p.startswith("r")), None)
    col_str = next((p[1:] for p in parts if p.startswith("c")), None)
    if row_str is None or col_str is None:
        return "bad_name", key

    # Check if patches already exist
    first_patch = PATCHES_OUT / f"{stem}_pr00_pc00.jpg"
    if first_patch.exists():
        return "skip", key

    try:
        im = Image.open(src_img)
        im = im.convert("RGB")
    except Exception as e:
        return "fail_img", (key, str(e))

    patches_written = 0
    try:
        for py in range(PATCHES_PER_SIDE):
            y = py * PATCH_SIZE
            for px in range(PATCHES_PER_SIDE):
                x = px * PATCH_SIZE
                patch = im.crop((x, y, x + PATCH_SIZE, y + PATCH_SIZE))
                patch_name = f"{stem}_pr{py:02d}_pc{px:02d}.jpg"
                patch_path = PATCHES_OUT / patch_name
                patch.save(patch_path, "JPEG", quality=92)
                patches_written += 1
    except Exception as e:
        return "fail_patches", (key, str(e))

    return "ok", (key, patches_written)


def build_patch_index():
    """Build patch_index.csv mapping every patch to its parent tile and grid position."""
    import csv

    index_path = PROJECT_ROOT / "data" / "ortho" / "patch_index.csv"
    print(f"\nBuilding patch index: {index_path}")

    all_patches = sorted(PATCHES_OUT.glob("*.jpg"))
    print(f"  Found {len(all_patches):,} patch files")

    with open(index_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "patch_file", "parent_tile", "grid_row", "grid_col",
            "pixel_x", "pixel_y", "tile_row", "tile_col"
        ])
        for pp in tqdm(all_patches, desc="Indexing patches", unit="patch"):
            stem = pp.stem
            # Parse: tile_L20_r{row}_c{col}_pr{py}_pc{px}
            parts = stem.split("_")
            tile_row = None
            tile_col = None
            patch_row = None
            patch_col = None
            for p in parts:
                if p.startswith("r") and not p.startswith("pr"):
                    try:
                        tile_row = int(p[1:])
                    except ValueError:
                        pass
                elif p.startswith("c") and not p.startswith("pc"):
                    try:
                        tile_col = int(p[1:])
                    except ValueError:
                        pass
                elif p.startswith("pr"):
                    try:
                        patch_row = int(p[2:])
                    except ValueError:
                        pass
                elif p.startswith("pc"):
                    try:
                        patch_col = int(p[2:])
                    except ValueError:
                        pass

            parent_tile = f"tile_L20_r{tile_row}_c{tile_col}.jpg"
            pixel_x = (patch_col or 0) * PATCH_SIZE
            pixel_y = (patch_row or 0) * PATCH_SIZE

            w.writerow([
                pp.name, parent_tile, patch_row, patch_col,
                pixel_x, pixel_y, tile_row, tile_col
            ])

    print(f"  Wrote {len(all_patches):,} rows to {index_path}")


def copy_metadata():
    """Copy metadata files from H3MRL that are useful for UrbanJEPA."""
    files_to_copy = [
        "manifest_nonwhite.json",
        "manifest.json",
        "filter_report.csv",
        "input_files_nonwhite.txt",
        "rom_L20_silhouette.wkt",
    ]
    meta_dir = PROJECT_ROOT / "data" / "ortho" / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)

    for fname in files_to_copy:
        src = H3MRL_DIR / fname
        if src.exists():
            shutil.copy2(src, meta_dir / fname)
            print(f"  Copied: {fname}")


def main():
    print("=" * 60)
    print("UrbanJEPA — H3MRL Tile Import")
    print("=" * 60)
    print(f"  Source:      {H3MRL_DIR}")
    print(f"  Tiles dest:  {TILES_OUT}")
    print(f"  Patches dest:{PATCHES_OUT}")
    print(f"  Valid tiles: {len(valid_tiles):,}")
    print(f"  Patches/tile:{PATCHES_PER_TILE} ({PATCHES_PER_SIDE}x{PATCHES_PER_SIDE})")
    print(f"  Total patches:{len(valid_tiles) * PATCHES_PER_TILE:,} (estimated)")
    print()

    TILES_OUT.mkdir(parents=True, exist_ok=True)
    PATCHES_OUT.mkdir(parents=True, exist_ok=True)

    # Stage 1: Copy tiles
    print("[1/3] Copying tiles...")
    t0 = time.time()
    copied, skipped, failed = 0, 0, 0
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = {pool.submit(copy_one_tile, args): args for args in valid_tiles}
        for f in tqdm(as_completed(futures), total=len(futures), desc="Copy tiles", unit="tile"):
            status, info = f.result()
            if status == "ok":
                copied += 1
            elif status == "skip":
                skipped += 1
            else:
                failed += 1
    print(f"  Copied: {copied}, Skipped: {skipped}, Failed: {failed} "
          f"({time.time() - t0:.1f}s)")

    # Stage 2: Extract patches
    print("\n[2/3] Extracting 256x256 patches...")
    t0 = time.time()
    patches_ok, patches_skip, patches_fail = 0, 0, 0
    total_patches = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(extract_patches, args): args for args in valid_tiles}
        for f in tqdm(as_completed(futures), total=len(futures), desc="Extract patches", unit="tile"):
            status, info = f.result()
            if status == "ok":
                patches_ok += 1
                total_patches += info[1]
            elif status == "skip":
                patches_skip += 1
                total_patches += PATCHES_PER_TILE
            else:
                patches_fail += 1
    print(f"  Done: {patches_ok}, Skipped: {patches_skip}, Failed: {patches_fail} "
          f"({time.time() - t0:.1f}s)")
    print(f"  Total patches: {total_patches:,}")

    # Stage 3: Build patch index
    print("\n[3/3] Building patch index...")
    build_patch_index()
    copy_metadata()

    # Summary
    print("\n" + "=" * 60)
    print("IMPORT COMPLETE")
    print(f"  Tiles copied:  {copied}")
    print(f"  Patches total: {total_patches:,}")
    print(f"  Index:         data/ortho/patch_index.csv")
    print(f"  Metadata:      data/ortho/metadata/")
    print(f"\nDisk usage:")
    for d in [TILES_OUT, PATCHES_OUT]:
        total = sum(f.stat().st_size for f in d.rglob("*.jpg"))
        print(f"  {d.relative_to(PROJECT_ROOT)}: {total / 1e9:.1f} GB")
    print("=" * 60)


if __name__ == "__main__":
    main()
