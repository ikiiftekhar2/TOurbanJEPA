"""
Toronto Ortho Dataset — self-supervised (low_res, high_res) training pairs.

Tiles are 4096x4096 JPEGs at ~15cm/pixel (L20), downloaded from the City of
Toronto ArcGIS MapServer via the H3MRL project's spiral scraper.

High-res = a 256x256 crop from the tile (rejected if white/empty).
Low-res = realistic satellite degradation (Gaussian PSF + downsample + optional JPEG).

v3 upgrades over v2:
  - In-process tile LRU cache (per worker)
  - Gaussian PSF blur + optional JPEG compression for degradation
  - Perlin-style cloud/haze augmentation
  - Continuous scale range [16, 24] during training
  - Synchronized color jitter (HR + LR) + extra LR-only jitter
  - Round-robin training index so every tile is visited each epoch
  - White / low-variance patch rejection
"""

import csv
import io
import random
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

# L20 native resolution ≈ 14.93 cm/px. PlanetScope effective resolution varies
# with atmospherics and off-nadir angle; train across a wide range, validate at
# the central value.
TRAIN_SCALE_MIN = 16.0
TRAIN_SCALE_MAX = 24.0
VAL_SCALE = 20

PATCH_SIZE = 256
TILE_SIZE = 4096
PATCHES_PER_SIDE = TILE_SIZE // PATCH_SIZE  # 16
PATCHES_PER_TILE = PATCHES_PER_SIDE * PATCHES_PER_SIDE  # 256


def _gaussian_blur(t: torch.Tensor, sigma: float) -> torch.Tensor:
    """Apply Gaussian blur with kernel size chosen from sigma."""
    k = int(6 * sigma) | 1
    k = max(3, min(k, 11))
    return TF.gaussian_blur(t, kernel_size=k, sigma=sigma)


def _jpeg_recompress(img_t: torch.Tensor, quality: int) -> torch.Tensor:
    """Round-trip a (3, H, W) tensor in [0, 1] through JPEG at given quality."""
    arr = (img_t.clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    arr2 = np.array(Image.open(buf), dtype=np.float32) / 255.0
    return torch.from_numpy(arr2.transpose(2, 0, 1))


class OrthoDataset(Dataset):
    """Self-supervised (low_res, high_res) dataset over Toronto L20 ortho tiles."""

    def __init__(
        self,
        ortho_dir: str,
        split: str = "train",
        train_ratio: float = 0.9,
        augment: bool = True,
        seed: int = 42,
        patches_per_epoch: int = 128,
        val_patches_per_tile: int = 4,
        tile_cache_size: int = 32,
        white_threshold: float = 0.95,
        std_threshold: float = 0.02,
        max_reject_retries: int = 5,
        # v4 Phase 1: when True + split='val', apply the same realistic
        # LR degradation pipeline as train (random sigma + 30% JPEG), seeded
        # deterministically per sample so val stays reproducible. Geometric
        # augmentations (flip/rotation/color jitter) are NOT applied to val
        # — they would change the HR target relative to its grid position.
        match_train_aug_in_val: bool = False,
        aug_match_seed: int = 1337,
        # Cross-sensor LR-only photometric simulation (brightness/contrast jitter,
        # sensor noise, atmospheric haze/clouds). These break LR<->HR photometric
        # alignment to mimic the PlanetScope<->ortho sensor gap. OFF by default:
        # for clean-SR training they create a train/val distribution mismatch
        # (the model learns colour corrections that misfire on the un-augmented
        # val set). Re-enable for the PlanetScope finetune phase.
        cross_sensor_aug: bool = False,
        # Optional path to a tile-basename allowlist (one filename per line).
        # When set, only tiles whose basename appears in the file are used.
        # Used to filter out flat/poison tiles that teach the decoder a
        # content-independent degenerate solution. The 90/10 train/val split
        # then operates over the filtered set, not the raw 4154 tile glob.
        tile_manifest: Optional[str] = None,
    ):
        self.ortho_dir = Path(ortho_dir)
        self.split = split
        self.augment = augment and split == "train"
        self.match_train_aug_in_val = match_train_aug_in_val and split == "val"
        self.cross_sensor_aug = cross_sensor_aug
        # Whether to use realistic randomized LR degradation (vs fixed sigma=1.0).
        self._realistic_degradation = self.augment or self.match_train_aug_in_val
        self._aug_seed = int(aug_match_seed)
        self.patch_size = PATCH_SIZE
        self.tile_cache_size = tile_cache_size
        self.white_threshold = white_threshold
        self.std_threshold = std_threshold
        self.max_reject_retries = max_reject_retries

        tiles_dir = self.ortho_dir / "tiles"
        self.tiles: List[Path] = sorted(tiles_dir.glob("*.jpg"))
        if not self.tiles:
            raise RuntimeError(
                f"No tiles found in {tiles_dir}. Run scripts/import_h3mrl_tiles.py first."
            )
        if tile_manifest is not None:
            manifest_path = Path(tile_manifest)
            with open(manifest_path) as _f:
                allow = {line.strip() for line in _f if line.strip()}
            pre_n = len(self.tiles)
            self.tiles = [t for t in self.tiles if t.name in allow]
            print(f"OrthoDataset [{split}]: tile_manifest={manifest_path.name} "
                  f"filtered {pre_n} -> {len(self.tiles)} tiles", flush=True)

        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(self.tiles))
        split_idx = int(len(indices) * train_ratio)
        keep = indices[:split_idx] if split == "train" else indices[split_idx:]
        self.tiles = [self.tiles[i] for i in keep]

        if split == "val":
            self._grid: Optional[List[Tuple[int, int, int]]] = []
            tile_rng = np.random.RandomState(seed + 1)
            for tile_idx in range(len(self.tiles)):
                positions = [(pr, pc) for pr in range(PATCHES_PER_SIDE)
                             for pc in range(PATCHES_PER_SIDE)]
                tile_rng.shuffle(positions)
                for pr, pc in positions[:val_patches_per_tile]:
                    self._grid.append((tile_idx, pr, pc))
        else:
            self._grid = None

        self.patches_per_epoch = patches_per_epoch

        # Per-worker LRU cache. DataLoader workers each get their own copy
        # (no cross-worker sharing); memory budget = num_workers * tile_cache_size * 50 MB.
        self._tile_cache: "OrderedDict[int, np.ndarray]" = OrderedDict()

        if split == "train":
            epoch_size = len(self.tiles) * patches_per_epoch
        else:
            epoch_size = len(self._grid)
        print(
            f"OrthoDataset [{split}]: {len(self.tiles):,} tiles, "
            f"{epoch_size:,} samples/epoch, ~15cm/px, "
            f"scale={'continuous [%g, %g]' % (TRAIN_SCALE_MIN, TRAIN_SCALE_MAX) if self.augment else f'fixed {VAL_SCALE}'}",
            flush=True,
        )

    def __len__(self) -> int:
        if self._grid is not None:
            return len(self._grid)
        return len(self.tiles) * self.patches_per_epoch

    def _load_tile_cached(self, tile_idx: int) -> np.ndarray:
        """LRU-cached tile load. Returns HxWx3 uint8 array."""
        cache = self._tile_cache
        if tile_idx in cache:
            cache.move_to_end(tile_idx)
            return cache[tile_idx]
        img = Image.open(self.tiles[tile_idx]).convert("RGB")
        arr = np.array(img, dtype=np.uint8)
        cache[tile_idx] = arr
        if len(cache) > self.tile_cache_size:
            cache.popitem(last=False)
        return arr

    def _crop_patch(self, tile: np.ndarray, r: int, c: int) -> np.ndarray:
        h, w = tile.shape[:2]
        r = max(0, min(r, h - self.patch_size))
        c = max(0, min(c, w - self.patch_size))
        patch = tile[r:r + self.patch_size, c:c + self.patch_size]
        patch = patch.astype(np.float32) / 255.0
        return np.transpose(patch, (2, 0, 1))

    def _is_valid_patch(self, patch: np.ndarray) -> bool:
        return bool(patch.mean() < self.white_threshold and patch.std() > self.std_threshold)

    def _degrade(self, high_res: np.ndarray, scale: float) -> np.ndarray:
        """Realistic satellite degradation: PSF blur → downsample → upsample → optional JPEG."""
        t = torch.from_numpy(high_res).unsqueeze(0)  # (1, 3, 256, 256)
        if self._realistic_degradation:
            sigma = random.uniform(0.8, 1.5)
        else:
            sigma = 1.0
        t = _gaussian_blur(t, sigma)

        if abs(scale - round(scale)) < 1e-6:
            small = F.avg_pool2d(t, kernel_size=int(round(scale)), stride=int(round(scale)))
        else:
            target = max(1, int(round(self.patch_size / scale)))
            small = F.interpolate(t, size=(target, target), mode="area")
        up = F.interpolate(small, size=(self.patch_size, self.patch_size),
                           mode="bilinear", align_corners=False)

        if self._realistic_degradation and random.random() < 0.3:
            quality = random.randint(60, 95)
            up = _jpeg_recompress(up.squeeze(0), quality).unsqueeze(0)
        return up.squeeze(0).numpy()

    def _add_atmospheric(self, lr_t: torch.Tensor, cloud_prob: float = 0.2,
                         haze_prob: float = 0.3) -> torch.Tensor:
        _, h, w = lr_t.shape
        if random.random() < haze_prob:
            strength = random.uniform(0.05, 0.25)
            lr_t = lr_t * (1 - strength) + strength
        if random.random() < cloud_prob:
            noise = torch.randn(1, 1, max(1, h // 16), max(1, w // 16))
            mask = F.interpolate(noise, size=(h, w), mode="bilinear", align_corners=False)
            mask = torch.sigmoid(mask * 3 - 1.5).squeeze(0)
            cloud_val = random.uniform(0.85, 1.0)
            lr_t = lr_t * (1 - mask) + cloud_val * mask
        return lr_t.clamp(0, 1)

    def _augment(self, high_res: np.ndarray, low_res: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        hr_t = torch.from_numpy(high_res)
        lr_t = torch.from_numpy(low_res)

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

        # Synchronized mild color jitter — represents lighting variability across tiles.
        if random.random() < 0.5:
            b = random.uniform(0.95, 1.05)
            c = random.uniform(0.95, 1.05)
            hr_t = TF.adjust_brightness(hr_t, b)
            hr_t = TF.adjust_contrast(hr_t, c)
            lr_t = TF.adjust_brightness(lr_t, b)
            lr_t = TF.adjust_contrast(lr_t, c)

        # Cross-sensor LR-only distortions (PlanetScope<->ortho gap). OFF for
        # clean-SR training — these break LR<->HR alignment and create a
        # train/val mismatch. Re-enabled (--cross_sensor_aug) for Planet finetune.
        if self.cross_sensor_aug:
            # Extra LR-only sensor jitter.
            lr_t = TF.adjust_brightness(lr_t, random.uniform(0.85, 1.15))
            lr_t = TF.adjust_contrast(lr_t, random.uniform(0.85, 1.15))

            # Sensor noise.
            lr_t = (lr_t + torch.randn_like(lr_t) * 0.02).clamp(0, 1)

            # Atmospheric effects (haze + smooth clouds).
            lr_t = self._add_atmospheric(lr_t)

        return hr_t.numpy(), lr_t.numpy()

    def _sample_train_position(self, tile_idx: int) -> Tuple[np.ndarray, int, int]:
        """Round-robin tile selection (Option B), random valid crop within."""
        tile_arr = self._load_tile_cached(tile_idx)
        h, w = tile_arr.shape[:2]
        max_r = max(0, h - self.patch_size)
        max_c = max(0, w - self.patch_size)
        for _ in range(self.max_reject_retries):
            r = random.randint(0, max_r) if max_r > 0 else 0
            c = random.randint(0, max_c) if max_c > 0 else 0
            patch = self._crop_patch(tile_arr, r, c)
            if self._is_valid_patch(patch):
                return tile_arr, r, c
        return tile_arr, r, c  # accept last attempt if all retries rejected

    def __getitem__(self, idx: int) -> Dict:
        # Per-sample deterministic seeding for val+aug-match (Phase 1+).
        # We re-seed the global random / numpy modules before any stochastic
        # call inside _degrade so val degradation is varied across samples
        # but identical between runs / workers / epochs.
        if self.match_train_aug_in_val:
            random.seed(self._aug_seed + idx)
            np.random.seed(self._aug_seed + idx)

        if self._grid is not None:
            tile_idx, pr, pc = self._grid[idx]
            tile_arr = self._load_tile_cached(tile_idx)
            r = pr * self.patch_size
            c = pc * self.patch_size
        else:
            tile_idx = idx % len(self.tiles)
            tile_arr, r, c = self._sample_train_position(tile_idx)

        high_res = self._crop_patch(tile_arr, r, c)

        if self._grid is not None:
            scale: float = float(VAL_SCALE)
        elif self.augment:
            scale = random.uniform(TRAIN_SCALE_MIN, TRAIN_SCALE_MAX)
        else:
            scale = float(VAL_SCALE)
        low_res = self._degrade(high_res, scale)

        if self.augment:
            high_res, low_res = self._augment(high_res, low_res)

        return {
            "low_res": torch.from_numpy(low_res).float(),
            "high_res": torch.from_numpy(high_res).float(),
            "scale": float(scale),
            "tile_path": str(self.tiles[tile_idx]),
            "crop_row": r,
            "crop_col": c,
            "tile_idx": tile_idx,
        }


def create_dataloaders(
    ortho_dir: str,
    batch_size: int = 8,
    num_workers: int = 4,
    train_ratio: float = 0.9,
    patches_per_epoch: int = 128,
    val_patches_per_tile: int = 4,
    persistent_workers: bool = True,
    tile_cache_size: int = 32,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    train_ds = OrthoDataset(
        ortho_dir, split="train", train_ratio=train_ratio, augment=True,
        patches_per_epoch=patches_per_epoch, tile_cache_size=tile_cache_size,
    )
    val_ds = OrthoDataset(
        ortho_dir, split="val", train_ratio=train_ratio, augment=False,
        val_patches_per_tile=val_patches_per_tile, tile_cache_size=tile_cache_size,
    )
    common = dict(
        num_workers=num_workers, pin_memory=True,
        persistent_workers=persistent_workers and num_workers > 0,
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True, **common,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=False, **common,
    )
    return train_loader, val_loader


def build_patch_index(ortho_dir: str, output_path: Optional[str] = None):
    tiles_dir = Path(ortho_dir) / "tiles"
    tiles = sorted(tiles_dir.glob("*.jpg"))
    if output_path is None:
        output_path = str(Path(ortho_dir) / "patch_index.csv")
    with open(output_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patch_id", "parent_tile", "grid_row", "grid_col",
                    "pixel_y", "pixel_x", "tile_idx"])
        patch_id = 0
        for tile_idx, tile_path in enumerate(tiles):
            for pr in range(PATCHES_PER_SIDE):
                for pc in range(PATCHES_PER_SIDE):
                    py = pr * PATCH_SIZE
                    px = pc * PATCH_SIZE
                    w.writerow([patch_id, tile_path.name, pr, pc, py, px, tile_idx])
                    patch_id += 1
    print(f"Patch index: {patch_id:,} patches -> {output_path}")
    return output_path
