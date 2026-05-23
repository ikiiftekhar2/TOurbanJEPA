"""
Toronto Ortho Dataset — self-supervised (low_res, high_res) training pairs.

Tiles are 4096x4096 JPEGs at ~15cm/pixel (L20), downloaded from the City of
Toronto ArcGIS MapServer via the H3MRL project's spiral scraper.

High-res = a random 256x256 crop from the tile.
Low-res = area-averaged downsampled version simulating coarser satellite inputs.

For validation, patches are cropped from a deterministic grid (16x16 per tile)
to ensure reproducibility. The patch index CSV maps each patch back to its
parent tile and grid position for reconstruction.
"""

import csv
import random
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image

# Resolution at L20: ~14.93 cm/pixel
# Inference target is ALWAYS PlanetScope 3m/pixel — so training scales
# must cluster tightly around that. Scale 20 = 2.99m (exact match).
# Scales 18-22 cover ±0.3m for real Planet variability (haze, angle, season).
DOWNSAMPLE_SCALES = [18, 20, 22]

PATCH_SIZE = 256
TILE_SIZE = 4096
PATCHES_PER_SIDE = TILE_SIZE // PATCH_SIZE  # 16
PATCHES_PER_TILE = PATCHES_PER_SIDE * PATCHES_PER_SIDE  # 256


class OrthoDataset(Dataset):
    """
    Self-supervised dataset that creates (low_res, high_res) pairs from ortho tiles.

    Training: random 256x256 crops from random 4096x4096 tiles.
    Validation: deterministic grid of 256x256 crops from validation tiles.
    """

    def __init__(
        self,
        ortho_dir: str,
        split: str = "train",
        train_ratio: float = 0.9,
        augment: bool = True,
        seed: int = 42,
        patches_per_epoch: int = 4,
        val_patches_per_tile: int = 4,
    ):
        self.ortho_dir = Path(ortho_dir)
        self.split = split
        self.augment = augment and split == "train"
        self.patch_size = PATCH_SIZE

        # Collect tile files
        tiles_dir = self.ortho_dir / "tiles"
        self.tiles: List[Path] = sorted(tiles_dir.glob("*.jpg"))

        if not self.tiles:
            raise RuntimeError(
                f"No tiles found in {tiles_dir}. Run scripts/import_h3mrl_tiles.py first."
            )

        # Shuffle and split at TILE level (prevents patch-level leakage)
        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(self.tiles))
        split_idx = int(len(indices) * train_ratio)
        if split == "train":
            indices = indices[:split_idx]
        else:
            indices = indices[split_idx:]
        self.tiles = [self.tiles[i] for i in indices]

        # For validation: deterministic random subset of grid patches per tile.
        # Using a fixed seed per tile ensures reproducibility while sampling
        # spatially diverse positions (not just top-left corner).
        # Default 4 patches/tile × 416 tiles = 1,664 samples gives SE ≈ σ/40,
        # sufficient for early stopping (Goodfellow et al. 2016, Bengio 2012).
        if split == "val":
            self._grid: List[Tuple[int, int, int]] = []
            tile_rng = np.random.RandomState(seed + 1)  # offset seed from split seed
            for tile_idx in range(len(self.tiles)):
                all_positions = [(pr, pc) for pr in range(PATCHES_PER_SIDE)
                                 for pc in range(PATCHES_PER_SIDE)]
                tile_rng.shuffle(all_positions)
                for pr, pc in all_positions[:val_patches_per_tile]:
                    self._grid.append((tile_idx, pr, pc))
        else:
            self._grid = None

        # Train epoch = N random crops per tile (avoids 256 crops/tile being too large)
        self.patches_per_epoch = patches_per_epoch

        self.downsample_scales = DOWNSAMPLE_SCALES

        if split == "train":
            epoch_size = len(self.tiles) * patches_per_epoch
        else:
            epoch_size = len(self._grid)
        print(
            f"OrthoDataset [{split}]: {len(self.tiles):,} tiles, "
            f"{epoch_size:,} samples/epoch, "
            f"~15cm/px, scales={self.downsample_scales}"
        )

    def __len__(self) -> int:
        if self._grid is not None:
            return len(self._grid)
        return len(self.tiles) * self.patches_per_epoch

    def _load_tile(self, tile_idx: int) -> np.ndarray:
        """Load a tile and return as HxWxC uint8 numpy array."""
        img = Image.open(self.tiles[tile_idx])
        img = img.convert("RGB")
        return np.array(img, dtype=np.uint8)

    def _crop_patch(self, tile: np.ndarray, r: int, c: int) -> np.ndarray:
        """Crop a 256x256 patch at position (r, c) in pixel coordinates."""
        h, w = tile.shape[:2]
        r = max(0, min(r, h - self.patch_size))
        c = max(0, min(c, w - self.patch_size))
        patch = tile[r : r + self.patch_size, c : c + self.patch_size]
        patch = patch.astype(np.float32) / 255.0  # normalize to [0, 1]
        return np.transpose(patch, (2, 0, 1))  # HWC -> CHW

    def _downsample(self, high_res: np.ndarray, scale: int) -> np.ndarray:
        """Area-average downsample then bilinear upsample back to 256x256."""
        t = torch.from_numpy(high_res).unsqueeze(0)  # (1, 3, 256, 256)
        small = F.avg_pool2d(t, kernel_size=scale, stride=scale)
        upsampled = F.interpolate(
            small, size=(256, 256), mode="bilinear", align_corners=False
        )
        return upsampled.squeeze(0).numpy()

    def _augment(self, high_res: np.ndarray, low_res: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Synchronized augmentations. Low-res gets extra sensor simulation."""
        import torchvision.transforms.functional as TF

        hr_t = torch.from_numpy(high_res)
        lr_t = torch.from_numpy(low_res)

        # Geometric (identical for both)
        if random.random() > 0.5:
            hr_t = TF.hflip(hr_t)
            lr_t = TF.hflip(lr_t)
        if random.random() > 0.5:
            hr_t = TF.vflip(hr_t)
            lr_t = TF.vflip(lr_t)
        k = random.randint(0, 3)
        if k > 0:
            hr_t = torch.rot90(hr_t, k, dims=[1, 2])
            lr_t = torch.rot90(lr_t, k, dims=[1, 2])

        # Photometric and sensor noise on low-res only
        brightness = random.uniform(0.8, 1.2)
        contrast = random.uniform(0.8, 1.2)
        lr_t = TF.adjust_brightness(lr_t, brightness)
        lr_t = TF.adjust_contrast(lr_t, contrast)

        noise = torch.randn_like(lr_t) * 0.02
        lr_t = (lr_t + noise).clamp(0, 1)

        # Random cloud mask (20% chance)
        if random.random() < 0.2:
            _, h, w = lr_t.shape
            cx = random.randint(w // 4, 3 * w // 4)
            cy = random.randint(h // 4, 3 * h // 4)
            radius = random.randint(20, 80)
            y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
            mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2
            cloud_val = 0.95 + torch.randn(1).item() * 0.05
            lr_t[:, mask] = cloud_val

        return hr_t.numpy(), lr_t.numpy()

    def __getitem__(self, idx: int) -> Dict:
        if self._grid is not None:
            # Validation: deterministic grid position
            tile_idx, pr, pc = self._grid[idx]
            r = pr * self.patch_size
            c = pc * self.patch_size
            tile_arr = self._load_tile(tile_idx)
        else:
            # Training: random tile and random crop position
            tile_idx = random.randrange(len(self.tiles))
            tile = Image.open(self.tiles[tile_idx]).convert("RGB")
            w, h = tile.size
            tile_arr = np.array(tile, dtype=np.uint8)
            max_r = max(0, h - self.patch_size)
            max_c = max(0, w - self.patch_size)
            r = random.randint(0, max_r) if max_r > 0 else 0
            c = random.randint(0, max_c) if max_c > 0 else 0

        tile_path = self.tiles[tile_idx]

        # Crop patch from already-loaded tile
        high_res = self._crop_patch(tile_arr, r, c)

        # Downsample
        scale = np.random.choice(self.downsample_scales)
        low_res = self._downsample(high_res, scale)

        # Augment (train only)
        if self.augment:
            high_res, low_res = self._augment(high_res, low_res)

        return {
            "low_res": torch.from_numpy(low_res).float(),
            "high_res": torch.from_numpy(high_res).float(),
            "scale": scale,
            "tile_path": str(tile_path),
            "crop_row": r,
            "crop_col": c,
            "tile_idx": tile_idx,
        }


def create_dataloaders(
    ortho_dir: str,
    batch_size: int = 8,
    num_workers: int = 4,
    train_ratio: float = 0.9,
    patches_per_epoch: int = 4,
    val_patches_per_tile: int = 4,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train and validation dataloaders."""
    train_ds = OrthoDataset(
        ortho_dir, split="train", train_ratio=train_ratio, augment=True,
        patches_per_epoch=patches_per_epoch,
    )
    val_ds = OrthoDataset(
        ortho_dir, split="val", train_ratio=train_ratio, augment=False,
        val_patches_per_tile=val_patches_per_tile,
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader


def build_patch_index(ortho_dir: str, output_path: Optional[str] = None):
    """
    Build a CSV mapping every grid-aligned patch to its parent tile.
    This is for reconstructing full images during evaluation.
    """
    tiles_dir = Path(ortho_dir) / "tiles"
    tiles = sorted(tiles_dir.glob("*.jpg"))

    if output_path is None:
        output_path = str(Path(ortho_dir) / "patch_index.csv")

    with open(output_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "patch_id", "parent_tile", "grid_row", "grid_col",
            "pixel_y", "pixel_x", "tile_idx"
        ])
        patch_id = 0
        for tile_idx, tile_path in enumerate(tiles):
            for pr in range(PATCHES_PER_SIDE):
                for pc in range(PATCHES_PER_SIDE):
                    py = pr * PATCH_SIZE
                    px = pc * PATCH_SIZE
                    w.writerow([
                        patch_id, tile_path.name, pr, pc, py, px, tile_idx
                    ])
                    patch_id += 1

    print(f"Patch index: {patch_id:,} patches -> {output_path}")
    return output_path
