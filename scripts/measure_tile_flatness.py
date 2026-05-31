"""Measure flatness/texture distribution across all train tiles.

For each tile, computes:
  - global_std: std of grayscale intensities over the full tile
  - mean_patch_std: per-256x256-patch std, averaged (matches training crop size)
  - flat_patch_frac: fraction of 256x256 patches with std < 0.03 (essentially flat)
  - edge_density: Sobel-magnitude mean (Sobel of grayscale, threshold 0.05)

Outputs:
  experiments/v4_phase1_5/tile_flatness.csv (sortable)
  printed histogram + percentile summary
"""
import csv
import glob
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Downsample to 1024x1024 before measuring — flatness/edge statistics are
# scale-invariant in the regimes we care about (a parking lot is flat at any
# resolution). Cuts per-tile work ~16x vs full 4096².
ANALYZE = 1024
PATCH = 64       # 1024 / 16  (preserves 16x16 patch grid analysis)
GRID = 16
SOBEL_X = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
SOBEL_Y = SOBEL_X.t()
FLAT_STD_THRESHOLD = 0.03  # 3% std on [0,1] = essentially flat
EDGE_MAG_THRESHOLD = 0.05


def measure_one(path: str):
    img = Image.open(path).convert("RGB")
    # Downsample at decode time for speed (PIL.thumbnail is in-place + fast).
    img.thumbnail((ANALYZE, ANALYZE), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0   # (~1024, ~1024, 3)
    if arr.shape[0] < ANALYZE or arr.shape[1] < ANALYZE:
        return None
    arr = arr[:ANALYZE, :ANALYZE]
    gray = arr.mean(axis=2)                            # (ANALYZE, ANALYZE)
    global_std = float(gray.std())

    # 16x16 grid of 256x256 patches
    patches = gray.reshape(GRID, PATCH, GRID, PATCH).transpose(0, 2, 1, 3)
    patches = patches.reshape(GRID * GRID, PATCH, PATCH)
    patch_stds = patches.std(axis=(1, 2))              # (256,)
    mean_patch_std = float(patch_stds.mean())
    flat_patch_frac = float((patch_stds < FLAT_STD_THRESHOLD).mean())

    # Sobel edge magnitude
    g = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0)
    gx = F.conv2d(g, SOBEL_X.unsqueeze(0).unsqueeze(0), padding=1)
    gy = F.conv2d(g, SOBEL_Y.unsqueeze(0).unsqueeze(0), padding=1)
    mag = (gx ** 2 + gy ** 2).sqrt().squeeze()
    edge_density = float((mag > EDGE_MAG_THRESHOLD).float().mean())

    return {
        "path": path,
        "global_std": global_std,
        "mean_patch_std": mean_patch_std,
        "flat_patch_frac": flat_patch_frac,
        "edge_density": edge_density,
    }


def main():
    tile_dir = "data/ortho/tiles"
    tiles = sorted(glob.glob(os.path.join(tile_dir, "*.jpg")))
    if not tiles:
        raise SystemExit(f"no tiles in {tile_dir}")
    print(f"measuring {len(tiles)} tiles...", flush=True)

    rows = []
    for i, p in enumerate(tiles):
        try:
            r = measure_one(p)
            if r is None:
                continue
            rows.append(r)
        except Exception as e:
            print(f"  skipped {os.path.basename(p)}: {e}", flush=True)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(tiles)}", flush=True)

    # Save CSV
    out_dir = "experiments/v4_phase1_5"
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "tile_flatness.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {csv_path}\n", flush=True)

    # Distributions
    def pcts(name, arr, lower_better=False):
        a = np.array(arr)
        print(f"\n{name}: N={len(a)}, min={a.min():.4f}, mean={a.mean():.4f}, max={a.max():.4f}")
        for q in [10, 25, 50, 75, 90, 95, 99]:
            print(f"   p{q:<3} = {np.percentile(a, q):.4f}")
        if lower_better:
            for cut in [0.20, 0.30, 0.40, 0.50]:
                threshold = np.percentile(a, cut * 100)
                n_drop = (a <= threshold).sum()
                print(f"   drop bottom {int(cut*100)}% (cutoff={threshold:.4f}): "
                      f"{n_drop} tiles dropped, {len(a)-n_drop} remain")

    pcts("global_std (LOW = flat)", [r["global_std"] for r in rows], lower_better=True)
    pcts("mean_patch_std (LOW = flat patches)",
         [r["mean_patch_std"] for r in rows], lower_better=True)
    pcts("flat_patch_frac (HIGH = many flat patches per tile)",
         [r["flat_patch_frac"] for r in rows])
    pcts("edge_density (LOW = few edges)",
         [r["edge_density"] for r in rows], lower_better=True)

    # Combined drop suggestion
    print("\n=== combined filter suggestions (drop a tile if ANY criterion met) ===")
    for cut in [0.20, 0.30, 0.40]:
        gs_cut = np.percentile([r["global_std"] for r in rows], cut * 100)
        ed_cut = np.percentile([r["edge_density"] for r in rows], cut * 100)
        flat_cut = np.percentile([r["flat_patch_frac"] for r in rows], (1 - cut) * 100)
        drop = [r for r in rows
                if r["global_std"] <= gs_cut
                or r["edge_density"] <= ed_cut
                or r["flat_patch_frac"] >= flat_cut]
        print(f"  union-of-bottom-{int(cut*100)}%: drop {len(drop)} "
              f"({100*len(drop)/len(rows):.1f}%), keep {len(rows)-len(drop)}")


if __name__ == "__main__":
    main()
