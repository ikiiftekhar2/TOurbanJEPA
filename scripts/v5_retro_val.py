"""
Retroactive val on a v5 checkpoint using LIVE model weights (not EMA shadow).

The training run had a corrupted EMA shadow (10 NaN scalars in 3 RRDBNet
biases) that caused all val outputs to NaN from step 2000 onward. Live weights
are clean. This script bypasses EMA-apply and runs validate() on whatever
weights the checkpoint's "model" key stores — giving us the val numbers we
would have seen if EMA had not been corrupted.

Also runs the EMA-applied path for comparison; expect NaN unless --skip_ema.
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.ortho_dataset import OrthoDataset
from src.evaluation.metrics import psnr as psnr_metric, ssim_metric
from src.models.v5_model import build_v5_model
from src.training.ema import ModelEMA
from src.training.losses import LPIPSLoss, l1_pixel_loss


@torch.no_grad()
def run_val(model, loader, device, lpips_loss, max_batches=0):
    model.eval()
    sums = {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0, "l1": 0.0,
            "psnr_bil": 0.0, "lpips_bil": 0.0}
    n = 0
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        lr = batch["low_res"].to(device, non_blocking=True)
        hr = batch["high_res"].to(device, non_blocking=True)
        out = model(lr, hr)
        sr = out["sr"].clamp(0, 1)
        sums["psnr"] += psnr_metric(sr, hr).item()
        sums["ssim"] += ssim_metric(sr, hr).item()
        sums["l1"] += l1_pixel_loss(sr, hr).item()
        sums["psnr_bil"] += psnr_metric(lr, hr).item()
        sums["lpips"] += lpips_loss(sr, hr).item()
        sums["lpips_bil"] += lpips_loss(lr, hr).item()
        n += 1
    return {k: v / max(1, n) for k, v in sums.items()}, n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--ortho_dir", default="data/ortho")
    p.add_argument("--tile_manifest", default="data/ortho/metadata/train_textured.txt")
    p.add_argument("--esrgan_weights", default="models/pretrained/RealESRGAN_x4plus.pth")
    p.add_argument("--jepa_pretrained", default="models/pretrained/dinov2_vitb14.pth")
    p.add_argument("--backbone", default="dinov2_vitb14")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--max_batches", type=int, default=0)
    p.add_argument("--also_apply_ema", action="store_true",
                   help="Also evaluate with EMA shadow applied (expect NaN if corrupt).")
    args = p.parse_args()

    device = torch.device("cuda")
    print(f"[device] {device}, free={torch.cuda.mem_get_info()[0]/1024**3:.2f} GiB",
          flush=True)

    # Val loader (same construction as train.py).
    val_ds = OrthoDataset(
        ortho_dir=args.ortho_dir, split="val", augment=False,
        val_patches_per_tile=1, tile_manifest=args.tile_manifest,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    print(f"[data] val_ds={len(val_ds)} batches={len(val_loader)}", flush=True)

    # Build the v5 model (this loads the pretrained ESRGAN+JEPA weights — they
    # will be overwritten by the checkpoint state in the next step).
    print(f"[model] building V5Model", flush=True)
    model = build_v5_model(
        backbone_name=args.backbone,
        jepa_pretrained_path=args.jepa_pretrained,
        esrgan_weights_path=args.esrgan_weights,
    ).to(device)

    # Load ckpt.
    print(f"[ckpt] loading {args.ckpt}", flush=True)
    st = torch.load(args.ckpt, map_location=device, weights_only=False)
    print(f"[ckpt] global_step={st['global_step']} epoch={st['epoch']}", flush=True)
    model.load_checkpoint_state(st["model"], strict=False)

    lpips_loss = LPIPSLoss(net="vgg").to(device).eval()

    # ---- (1) Live-weights val: bypass EMA entirely ----
    print("\n[live] running val with LIVE weights (no EMA applied)…", flush=True)
    t0 = time.time()
    m_live, n = run_val(model, val_loader, device, lpips_loss, args.max_batches)
    print(f"[live] done {n} batches in {time.time()-t0:.1f}s")
    print(f"[live] PSNR={m_live['psnr']:.3f}  (bil={m_live['psnr_bil']:.3f}, "
          f"d={m_live['psnr']-m_live['psnr_bil']:+.3f})")
    print(f"[live] LPIPS={m_live['lpips']:.4f}  (bil={m_live['lpips_bil']:.4f}, "
          f"d={m_live['lpips']-m_live['lpips_bil']:+.4f})")
    print(f"[live] SSIM={m_live['ssim']:.4f}  L1={m_live['l1']:.4f}")

    # ---- (2) Optional: EMA-applied val (will NaN if shadow is dirty) ----
    if args.also_apply_ema and "ema" in st:
        ema = ModelEMA(model, decay_start=0.995, decay_end=0.9999)
        ema.load_state_dict(st["ema"])
        backup = ema.apply_to(model)
        print("\n[ema] running val with EMA shadow applied…", flush=True)
        m_ema, _ = run_val(model, val_loader, device, lpips_loss, args.max_batches)
        ema.restore(model, backup)
        print(f"[ema] PSNR={m_ema['psnr']:.3f}  LPIPS={m_ema['lpips']:.4f} "
              f"(NaN = EMA corruption confirmed)")


if __name__ == "__main__":
    main()
