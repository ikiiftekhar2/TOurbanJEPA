"""
Phase 0 smoke test: instantiate RRDBNet, load RealESRGAN_x4plus weights,
run a forward pass on a real Toronto ortho tile, sanity-check the output.

Pass criteria:
  - weights load strict, no missing/unexpected keys
  - forward pass produces tensor of shape (1, 3, 4H, 4W)
  - output range plausibly in [0, 1] (after clamp) and not constant/NaN
  - bilinear-vs-ESRGAN PSNR delta logged for awareness
  - side-by-side PNG written to /tmp for eyeball check
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.models.esrgan import build_pretrained_x4plus


def load_tile(path: Path, crop_size: int = 256) -> torch.Tensor:
    """Load a tile and return a centre crop as (1, 3, crop, crop) float in [0,1]."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    left = (w - crop_size) // 2
    top = (h - crop_size) // 2
    img = img.crop((left, top, left + crop_size, top + crop_size))
    arr = np.asarray(img, dtype=np.float32) / 255.0  # HWC
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()
    return t


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a.clamp(0, 1), b.clamp(0, 1)).item()
    if mse <= 0:
        return float("inf")
    return -10.0 * np.log10(mse)


def save_comparison(lr: torch.Tensor, bil: torch.Tensor, esr: torch.Tensor, out_path: Path):
    def to_img(t):
        t = t.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
        return (t * 255).astype(np.uint8)

    lr_img = to_img(lr)
    bil_img = to_img(bil)
    esr_img = to_img(esr)
    H_target = esr_img.shape[0]
    lr_up = np.asarray(Image.fromarray(lr_img).resize((H_target, H_target), Image.NEAREST))
    panel = np.concatenate([lr_up, bil_img, esr_img], axis=1)
    Image.fromarray(panel).save(out_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tile", type=str,
                   default="data/ortho/tiles/tile_L20_r53_c16.jpg")
    p.add_argument("--weights", type=str,
                   default="models/pretrained/RealESRGAN_x4plus.pth")
    p.add_argument("--crop", type=int, default=256,
                   help="Centre-crop size of the HR tile.")
    p.add_argument("--lr_scale", type=int, default=4,
                   help="Downsample factor to create the LR input (4 = native ESRGAN).")
    p.add_argument("--out", type=str, default="/tmp/v5_phase0_smoketest.png")
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    print(f"[weights] loading {args.weights}")
    model = build_pretrained_x4plus(args.weights, device=device, eval_mode=True)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model]   RRDBNet x4 loaded, {n_params/1e6:.2f}M params")

    tile_path = Path(args.tile)
    print(f"[input]   {tile_path}")
    hr = load_tile(tile_path, crop_size=args.crop).to(device)
    print(f"[hr]      shape={tuple(hr.shape)} range=[{hr.min():.3f},{hr.max():.3f}]")

    lr_size = args.crop // args.lr_scale
    lr = F.interpolate(hr, size=(lr_size, lr_size), mode="bicubic", antialias=True).clamp(0, 1)
    print(f"[lr]      shape={tuple(lr.shape)} range=[{lr.min():.3f},{lr.max():.3f}]")

    with torch.no_grad():
        esr = model(lr).clamp(0, 1)
    print(f"[esr]     shape={tuple(esr.shape)} range=[{esr.min():.3f},{esr.max():.3f}] "
          f"mean={esr.mean():.3f} std={esr.std():.3f}")

    bil = F.interpolate(lr, size=(args.crop, args.crop), mode="bilinear", align_corners=False).clamp(0, 1)
    psnr_bil = psnr(bil, hr)
    psnr_esr = psnr(esr, hr)
    print(f"[psnr]    bilinear: {psnr_bil:.2f} dB  |  ESRGAN: {psnr_esr:.2f} dB  "
          f"|  delta: {psnr_esr - psnr_bil:+.2f} dB")

    assert torch.isfinite(esr).all(), "ESRGAN output has NaN/Inf"
    assert esr.shape == hr.shape, f"shape mismatch {esr.shape} vs {hr.shape}"
    assert esr.std().item() > 1e-3, "ESRGAN output is near-constant (suspicious)"

    out_path = Path(args.out)
    save_comparison(lr, bil, esr, out_path)
    print(f"[saved]   {out_path}  (LR-nearest | bilinear | ESRGAN)")

    print("[ok]      Phase 0 smoke test passed.")


if __name__ == "__main__":
    main()
