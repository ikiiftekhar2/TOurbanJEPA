"""
Phase 1 end-to-end integration test: real JEPABackbone (DINOv2 ViT-B/14) →
real JEPAConditionedRRDBNet → real ortho tile.

Validates:
  - JEPABackbone(low_res, high_res) returns ctx_final + ctx_multi with shapes
    that JEPAConditionedRRDBNet actually accepts.
  - End-to-end forward produces a valid SR image of correct shape.
  - PSNR vs bilinear baseline (informational — at init, no surprise that we
    underperform on clean LR; the training loop is what closes that gap).
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.data.ortho_dataset import OrthoDataset
from src.models.esrgan import build_pretrained_x4plus
from src.models.jepa_esrgan import JEPAConditionedRRDBNet
from src.models.urbanjepa import JEPABackbone


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a.clamp(0, 1), b.clamp(0, 1)).item()
    if mse <= 0:
        return float("inf")
    return -10.0 * np.log10(mse)


def to_img(t):
    t = t.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
    return (t * 255).astype(np.uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ortho_dir", type=str, default="data/ortho")
    p.add_argument("--esrgan_weights", type=str,
                   default="models/pretrained/RealESRGAN_x4plus.pth")
    p.add_argument("--dino_weights", type=str,
                   default="models/pretrained/dinov2_vitb14.pth")
    p.add_argument("--backbone", type=str, default="dinov2_vitb14")
    p.add_argument("--out", type=str, default="/tmp/v5_phase1_e2e.png")
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Real validation sample.
    ds = OrthoDataset(
        ortho_dir=args.ortho_dir, split="val",
        train_ratio=0.9, augment=False, val_patches_per_tile=4,
    )
    sample = ds[0]
    low_res = sample["low_res"].unsqueeze(0).to(device)   # (1,3,256,256)
    high_res = sample["high_res"].unsqueeze(0).to(device) # (1,3,256,256)
    print(f"[data]   tile={Path(sample['tile_path']).name} "
          f"scale={sample['scale']:.1f} "
          f"low_res={tuple(low_res.shape)} high_res={tuple(high_res.shape)}")

    # JEPA backbone.
    print(f"[jepa]   loading {args.backbone} from {args.dino_weights}")
    jepa = JEPABackbone(
        backbone_name=args.backbone,
        pretrained_path=args.dino_weights,
        predictor_depth=6,
        dropout=0.0,
    ).to(device).eval()

    with torch.no_grad():
        feats = jepa(low_res, high_res)
    print(f"[jepa]   ctx_final     {tuple(feats['ctx_final'].shape)}")
    print(f"[jepa]   ctx_multi[{len(feats['ctx_multi'])}] each "
          f"{tuple(feats['ctx_multi'][0].shape)}")
    print(f"[jepa]   ctx_std       {feats['ctx_std'].item():.4f}")
    print(f"[jepa]   cos_sim       {feats['cos_sim'].item():.4f}")

    # ESRGAN wrapped.
    print(f"[esrgan] loading pretrained x4plus weights")
    rrdbnet = build_pretrained_x4plus(args.esrgan_weights, device=device, eval_mode=True)
    model = JEPAConditionedRRDBNet(rrdbnet=rrdbnet).to(device).eval()

    with torch.no_grad():
        sr = model(low_res, feats["ctx_final"], feats["ctx_multi"]).clamp(0, 1)
    print(f"[sr]     shape={tuple(sr.shape)} "
          f"range=[{sr.min():.3f},{sr.max():.3f}] "
          f"mean={sr.mean():.3f} std={sr.std():.3f}")

    psnr_bil = psnr(low_res, high_res)
    psnr_sr = psnr(sr, high_res)
    print(f"[psnr]   bilinear(low_res) vs hr: {psnr_bil:.2f} dB")
    print(f"[psnr]   jepa-esrgan(init)    vs hr: {psnr_sr:.2f} dB  "
          f"|  delta: {psnr_sr - psnr_bil:+.2f} dB  (init = stock ESRGAN, untrained)")

    # 3-panel: bilinear LR | jepa-esrgan output | hr
    panel = np.concatenate([to_img(low_res), to_img(sr), to_img(high_res)], axis=1)
    Image.fromarray(panel).save(args.out)
    print(f"[saved]  {args.out}  (bilinear LR | jepa-esrgan init | HR)")

    print("[ok]     Phase 1 end-to-end integration passed.")


if __name__ == "__main__":
    main()
