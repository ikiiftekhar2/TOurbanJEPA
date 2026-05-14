#!/usr/bin/env python3
"""
Smoke test: verify the data pipeline works with the H3MRL 4096x4096 tiles.
Tests OrthoDataset (random crops + downsampling), dataloaders, and visualizes pairs.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.ortho_dataset import OrthoDataset, create_dataloaders, PATCHES_PER_TILE


def visualize_pairs(pairs, output_path):
    """Create a visualization grid of (low_res, high_res) pairs."""
    n = min(len(pairs), 8)
    fig, axes = plt.subplots(n, 2, figsize=(8, 2.2 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    for i in range(n):
        pair = pairs[i]
        low = pair["low_res"].permute(1, 2, 0).clamp(0, 1)
        high = pair["high_res"].permute(1, 2, 0).clamp(0, 1)

        axes[i, 0].imshow(low)
        axes[i, 0].set_title(f"Low-res (scale {pair['scale']}x)", fontsize=9)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(high)
        axes[i, 1].set_title(
            f"High-res (tile {Path(pair['tile_path']).stem}, "
            f"crop {pair['crop_row']},{pair['crop_col']})",
            fontsize=8,
        )
        axes[i, 1].axis("off")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Visualization saved to {output_path}")


def run_smoke_test():
    project_root = Path(__file__).resolve().parents[1]
    ortho_dir = project_root / "data" / "ortho"
    viz_dir = project_root / "notebooks" / "smoke_test_output"
    viz_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("UrbanJEPA Smoke Test (H3MRL 4096x4096 tiles)")
    print("=" * 60)

    # Step 1: Check tiles exist
    print("\n[1/5] Checking tiles on disk...")
    tiles_dir = ortho_dir / "tiles"
    jpgs = sorted(tiles_dir.glob("*.jpg"))
    jgws = sorted(tiles_dir.glob("*.jgw"))
    print(f"  JPGs: {len(jpgs):,}")
    print(f"  JGWs: {len(jgws):,}")
    if len(jpgs) < 10:
        print("  FAILED: Not enough tiles. Run scripts/import_h3mrl_tiles.py first.")
        return False
    # Check tile size
    from PIL import Image
    sample = Image.open(jpgs[0])
    print(f"  Tile size: {sample.size} (expected 4096x4096)")
    print(f"  OK: {len(jpgs):,} tiles available")

    # Show some tile names to verify we have diverse coverage
    print(f"  Sample tiles: {[t.name for t in jpgs[:5]]}")

    # Step 2: Test OrthoDataset
    print("\n[2/5] Testing OrthoDataset...")
    train_ds = OrthoDataset(str(ortho_dir), split="train", train_ratio=0.9, augment=False)
    val_ds = OrthoDataset(str(ortho_dir), split="val", train_ratio=0.9, augment=False)

    estimates = len(train_ds.tiles) * PATCHES_PER_TILE
    estimates_val = len(val_ds.tiles) * PATCHES_PER_TILE
    print(f"  OK: {len(train_ds.tiles):,} train tiles (~{estimates:,} possible patches)")
    print(f"  OK: {len(val_ds.tiles):,} val tiles (~{estimates_val:,} deterministic patches)")

    if len(train_ds.tiles) < 2:
        print(f"  FAILED: Only {len(train_ds.tiles)} train tiles")
        return False

    # Step 3: Generate sample pairs (random crops for train, grid for val)
    print("\n[3/5] Generating sample pairs...")
    sample_pairs = []
    for i in range(min(16, len(train_ds))):
        pair = train_ds[i]
        sample_pairs.append(pair)

    # Also get a val pair (deterministic grid)
    val_pair = val_ds[0]
    sample_pairs.append(val_pair)

    for i, pair in enumerate(sample_pairs):
        tile_name = Path(pair["tile_path"]).stem
        print(
            f"  Pair {i}: low={list(pair['low_res'].shape)}, "
            f"high={list(pair['high_res'].shape)}, "
            f"scale={pair['scale']}x, "
            f"crop=({pair['crop_row']},{pair['crop_col']}), "
            f"tile={tile_name}"
        )

    # Step 4: Check statistics
    print("\n[4/5] Checking data statistics...")
    all_low = torch.stack([p["low_res"] for p in sample_pairs[:8]])
    all_high = torch.stack([p["high_res"] for p in sample_pairs[:8]])

    print(f"  Low-res:  mean={all_low.mean():.3f}, std={all_low.std():.3f}, "
          f"min={all_low.min():.3f}, max={all_low.max():.3f}")
    print(f"  High-res: mean={all_high.mean():.3f}, std={all_high.std():.3f}, "
          f"min={all_high.min():.3f}, max={all_high.max():.3f}")

    # Spatial frequency check
    diff_low = torch.abs(all_low[:, :, 1:, :] - all_low[:, :, :-1, :]).mean()
    diff_high = torch.abs(all_high[:, :, 1:, :] - all_high[:, :, :-1, :]).mean()
    print(f"  Horizontal gradient (low-res): {diff_low:.4f}")
    print(f"  Horizontal gradient (high-res): {diff_high:.4f}")
    if diff_low >= diff_high:
        print("  WARNING: Low-res not smoother than high-res — check downsampling.")
    else:
        print("  OK: Low-res is smoother than high-res")

    # Check that val patches are deterministic
    pair_a = val_ds[0]
    pair_b = val_ds[0]
    same = torch.equal(pair_a["high_res"], pair_b["high_res"])
    print(f"  Val determinism: {'OK' if same else 'WARNING: not deterministic'}")

    # Step 5: Test DataLoader
    print("\n[5/5] Testing DataLoader...")
    train_loader, val_loader = create_dataloaders(
        str(ortho_dir), batch_size=4, num_workers=2, train_ratio=0.9
    )
    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))
    print(f"  Train batch: low={list(train_batch['low_res'].shape)}, "
          f"high={list(train_batch['high_res'].shape)}")
    print(f"  Val batch:   low={list(val_batch['low_res'].shape)}, "
          f"high={list(val_batch['high_res'].shape)}")
    print("  OK: DataLoader works")

    # Visualize
    print("\n[Visualizing]...")
    viz_path = viz_dir / "smoke_test_pairs.png"
    visualize_pairs(sample_pairs, viz_path)

    # Summary
    print("\n" + "=" * 60)
    print("SMOKE TEST PASSED")
    print(f"  Train tiles: {len(train_ds.tiles):,} -> ~{estimates:,} possible patches")
    print(f"  Val tiles:   {len(val_ds.tiles):,} -> {estimates_val:,} grid patches")
    print(f"  Tile size:   {sample.size[0]}x{sample.size[1]}")
    print(f"  Crop size:   256x256 (random for train, grid for val)")
    print(f"  Visualization: {viz_path}")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)
