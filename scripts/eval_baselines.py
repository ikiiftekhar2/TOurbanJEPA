"""
Baseline PSNR/SSIM evaluation on v3 val tiles.

Runs entirely on CPU so it can coexist with GPU training. For each combination of
(LR degradation mode × upsampler), reports mean PSNR / L1 / SSIM over N val
patches.

Three LR degradation modes:
  - v3_realistic    : current OrthoDataset val pipeline (PSF σ=1.0 + downsample
                      + bilinear up). What v3 training sees on val.
  - v1_plain        : avg_pool2d(scale) + bilinear up. What v1/v2 evaluated on —
                      this is where the "28 dB bilinear baseline" came from.
  - clean_psf_only  : PSF σ=1.0 + downsample, no JPEG / clouds / noise / jitter.
                      Same blur as v3 val but without the rest.

Four upsamplers re-applied to the "small" intermediate (so the upsampler is what
varies, not the degradation): nearest, bilinear, bicubic, lanczos.

Usage:
    PYTHONPATH=. python scripts/eval_baselines.py --n_patches 128
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.ortho_dataset import OrthoDataset


VAL_SCALE = 20
PATCH = 256


# ---------------- degradation modes (all return 256x256 LR + small intermediate) ----------------

def _gaussian_kernel1d(sigma: float, ksize: int = 9) -> torch.Tensor:
    x = torch.arange(ksize, dtype=torch.float32) - (ksize - 1) / 2
    k = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    return (k / k.sum())


def _gauss_blur(t: torch.Tensor, sigma: float) -> torch.Tensor:
    k1 = _gaussian_kernel1d(sigma).to(t.device)
    k = (k1[:, None] * k1[None, :])[None, None]  # (1,1,K,K)
    pad = k.shape[-1] // 2
    out = []
    for c in range(t.shape[1]):
        x = t[:, c:c + 1]
        x = F.pad(x, [pad] * 4, mode="reflect")
        x = F.conv2d(x, k)
        out.append(x)
    return torch.cat(out, dim=1)


def degrade_v3_realistic(hr: torch.Tensor, scale: int) -> torch.Tensor:
    """PSF blur (σ=1.0, val-mode) + downsample (bilinear area) + bilinear up."""
    t = _gauss_blur(hr, sigma=1.0)
    target = max(1, int(round(PATCH / scale)))
    small = F.interpolate(t, size=(target, target), mode="area")
    return small


def degrade_v1_plain(hr: torch.Tensor, scale: int) -> torch.Tensor:
    """v1/v2 style: pure avg_pool2d. No blur, no JPEG, no nothing."""
    small = F.avg_pool2d(hr, kernel_size=scale, stride=scale)
    return small


def degrade_clean_psf(hr: torch.Tensor, scale: int) -> torch.Tensor:
    """PSF + downsample only — same blur as v3 val but no JPEG/clouds/noise/jitter."""
    return degrade_v3_realistic(hr, scale)


# ---------------- upsamplers (all take "small" → 256x256) ----------------

def up_bilinear(small):
    return F.interpolate(small, size=(PATCH, PATCH), mode="bilinear", align_corners=False)


def up_nearest(small):
    return F.interpolate(small, size=(PATCH, PATCH), mode="nearest")


def up_bicubic(small):
    return F.interpolate(small, size=(PATCH, PATCH), mode="bicubic", align_corners=False).clamp(0, 1)


def up_lanczos(small):
    # torch has no built-in lanczos; use PIL per-channel.
    from PIL import Image
    arr = (small.clamp(0, 1) * 255).byte().cpu().numpy()  # (B, C, h, w)
    out = np.empty((arr.shape[0], arr.shape[1], PATCH, PATCH), dtype=np.uint8)
    for b in range(arr.shape[0]):
        for c in range(arr.shape[1]):
            img = Image.fromarray(arr[b, c])
            out[b, c] = np.asarray(img.resize((PATCH, PATCH), Image.LANCZOS))
    return torch.from_numpy(out).float() / 255.0


UPSAMPLERS = {
    "nearest": up_nearest,
    "bilinear": up_bilinear,
    "bicubic": up_bicubic,
    "lanczos": up_lanczos,
}

DEGRADATIONS = {
    "v3_realistic": degrade_v3_realistic,
    "v1_plain": degrade_v1_plain,
    "clean_psf": degrade_clean_psf,
}


# ---------------- metrics ----------------

def psnr(pred, target, max_val=1.0):
    mse = F.mse_loss(pred, target)
    if mse.item() < 1e-12:
        return torch.tensor(100.0)
    return 10.0 * torch.log10(torch.tensor(max_val ** 2) / mse)


def ssim_simple(pred, target):
    # Simple SSIM via pytorch_msssim if available, otherwise mean L1 fallback.
    try:
        from pytorch_msssim import ssim as _ssim
        return _ssim(pred, target, data_range=1.0, size_average=True)
    except Exception:
        return torch.tensor(float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/ortho")
    ap.add_argument("--n_patches", type=int, default=128,
                    help="Number of val patches to evaluate.")
    ap.add_argument("--scale", type=int, default=VAL_SCALE,
                    help="Downsample scale used by all degradation modes.")
    args = ap.parse_args()

    print(f"Loading val tiles (scale={args.scale}, target n_patches={args.n_patches})")
    # augment=False so OrthoDataset.__getitem__ returns the deterministic
    # val LR/HR pair; we only use the HR though.
    ds = OrthoDataset(args.data_dir, split="val", augment=False, seed=42)
    n = min(args.n_patches, len(ds))

    # Pre-collect HR patches into a single tensor for stable iteration.
    hr_list = []
    for i in range(n):
        s = ds[i]
        hr_list.append(s["high_res"])
    hr = torch.stack(hr_list, dim=0)  # (n, 3, 256, 256)
    print(f"Collected {hr.shape[0]} HR patches, shape={tuple(hr.shape)}\n")

    results = []
    for deg_name, deg_fn in DEGRADATIONS.items():
        small = deg_fn(hr, args.scale)
        for up_name, up_fn in UPSAMPLERS.items():
            up = up_fn(small).clamp(0, 1)
            ps = psnr(up, hr).item()
            l1 = F.l1_loss(up, hr).item()
            try:
                ss = ssim_simple(up, hr).item()
            except Exception:
                ss = float("nan")
            results.append((deg_name, up_name, ps, l1, ss))

    print(f"\n{'degradation':<14} {'upsampler':<10} {'PSNR':>8} {'L1':>8} {'SSIM':>8}")
    print("-" * 54)
    last_deg = None
    for deg, up, ps, l1, ss in results:
        if last_deg and last_deg != deg:
            print()
        print(f"{deg:<14} {up:<10} {ps:>6.2f}dB {l1:>8.4f} {ss:>8.4f}")
        last_deg = deg


if __name__ == "__main__":
    main()
