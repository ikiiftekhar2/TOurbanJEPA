"""
Bare Real-ESRGAN baseline for v5.

Question we want to answer: does JEPA conditioning earn its weight, or would
pretrained RRDBNet alone (no JEPA) already match/beat bilinear on Toronto
ortho 20× SR?

Setup:
  - Build a fresh JEPAConditionedRRDBNet from pretrained Real-ESRGAN x4plus.
  - The JEPA FeatureInjection.fuse layers are ZERO-INIT — so at this fresh
    state, the model is bit-exact bare RRDBNet (identity-at-init invariant,
    verified by scripts/v5_phase1_smoketest.py).
  - Run inference over the SAME val split that training uses.
  - Compute PSNR + LPIPS for bilinear, bare RRDBNet, and the (zero-init) JEPA
    path as a sanity check (must equal bare RRDBNet to 6 decimal places).

GPU-frugal: bs=2, no_grad, no LPIPS-on-train, single forward — peak ~1 GiB.
Safe to run concurrently with active training (only ~4 GiB free).
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.ortho_dataset import OrthoDataset
from src.models.esrgan.weight_loader import build_pretrained_x4plus
from src.models.jepa_esrgan import JEPAConditionedRRDBNet
from src.evaluation.metrics import psnr as psnr_metric, ssim_metric
from src.training.losses import LPIPSLoss


def fmt(x):
    return f"{x:7.4f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ortho_dir", default="data/ortho")
    p.add_argument("--tile_manifest", default="data/ortho/metadata/train_textured.txt")
    p.add_argument("--esrgan_weights", default="models/pretrained/RealESRGAN_x4plus.pth")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--val_patches_per_tile", type=int, default=1)
    p.add_argument("--max_batches", type=int, default=0,
                   help="0 = run full val set")
    p.add_argument("--lpips_net", default="vgg")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}, free={torch.cuda.mem_get_info()[0]/1024**3:.2f} GiB",
          flush=True)

    # ---------------- val dataset ----------------
    val_ds = OrthoDataset(
        ortho_dir=args.ortho_dir, split="val", augment=False,
        val_patches_per_tile=args.val_patches_per_tile,
        tile_manifest=args.tile_manifest,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    print(f"[data] val_ds size={len(val_ds)} batches={len(val_loader)}", flush=True)

    # ---------------- bare RRDBNet (pretrained x4plus) ----------------
    print(f"[model] loading pretrained RRDBNet from {args.esrgan_weights}", flush=True)
    rrdbnet = build_pretrained_x4plus(args.esrgan_weights).to(device).eval()

    # JEPAConditionedRRDBNet at fresh init = bit-exact bare RRDBNet
    # (FeatureInjection.fuse weights are zero-init; pred==RRDBNet(avg_pool(lr))).
    jepa_wrap = JEPAConditionedRRDBNet(rrdbnet=rrdbnet).to(device).eval()

    # Dummy JEPA tokens for the wrap path (they get zero-multiplied by fuse).
    def dummy_jepa(B):
        ctx_final = torch.zeros(B, 256, 768, device=device)
        ctx_multi = [torch.zeros(B, 256, 768, device=device) for _ in range(4)]
        return ctx_final, ctx_multi

    lpips_loss = LPIPSLoss(net=args.lpips_net).to(device).eval()

    sums = {
        "psnr_bil": 0.0, "lpips_bil": 0.0, "ssim_bil": 0.0,
        "psnr_rrdb": 0.0, "lpips_rrdb": 0.0, "ssim_rrdb": 0.0,
        "psnr_wrap": 0.0, "lpips_wrap": 0.0,
        "wrap_vs_rrdb_maxabs": 0.0,
    }
    n = 0
    t0 = time.time()
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if args.max_batches and i >= args.max_batches:
                break
            lr = batch["low_res"].to(device, non_blocking=True)   # (B,3,256,256)
            hr = batch["high_res"].to(device, non_blocking=True)  # (B,3,256,256)

            # ----- bilinear (lr already is bilinear-upsampled-to-256) -----
            sr_bil = lr.clamp(0, 1)
            sums["psnr_bil"] += psnr_metric(sr_bil, hr).item()
            sums["ssim_bil"] += ssim_metric(sr_bil, hr).item()
            sums["lpips_bil"] += lpips_loss(sr_bil, hr).item()

            # ----- bare RRDBNet: avg_pool 256→64, then native 4× -----
            lr_small = F.avg_pool2d(lr, kernel_size=4, stride=4)  # (B,3,64,64)
            sr_rrdb = rrdbnet(lr_small).clamp(0, 1)               # (B,3,256,256)
            sums["psnr_rrdb"] += psnr_metric(sr_rrdb, hr).item()
            sums["ssim_rrdb"] += ssim_metric(sr_rrdb, hr).item()
            sums["lpips_rrdb"] += lpips_loss(sr_rrdb, hr).item()

            # ----- JEPA-wrap at zero-init (must equal bare RRDBNet) -----
            ctx_final, ctx_multi = dummy_jepa(lr.shape[0])
            sr_wrap = jepa_wrap(lr, ctx_final, ctx_multi).clamp(0, 1)
            sums["psnr_wrap"] += psnr_metric(sr_wrap, hr).item()
            sums["lpips_wrap"] += lpips_loss(sr_wrap, hr).item()
            sums["wrap_vs_rrdb_maxabs"] = max(
                sums["wrap_vs_rrdb_maxabs"],
                (sr_wrap - sr_rrdb).abs().max().item(),
            )

            n += 1
            if i % 25 == 0:
                print(f"  [batch {i:4d}/{len(val_loader)}] "
                      f"bil_psnr={sums['psnr_bil']/n:.3f} "
                      f"rrdb_psnr={sums['psnr_rrdb']/n:.3f} "
                      f"wrap-vs-rrdb_max={sums['wrap_vs_rrdb_maxabs']:.2e}",
                      flush=True)

    elapsed = time.time() - t0
    m = {k: v / max(1, n) for k, v in sums.items()}
    m["wrap_vs_rrdb_maxabs"] = sums["wrap_vs_rrdb_maxabs"]
    m["n_batches"] = n

    print(f"\n[done] {n} batches in {elapsed:.1f}s\n")
    print("=" * 70)
    print(f"  bilinear          PSNR={fmt(m['psnr_bil'])}  "
          f"SSIM={fmt(m['ssim_bil'])}  LPIPS={fmt(m['lpips_bil'])}")
    print(f"  bare RRDBNet      PSNR={fmt(m['psnr_rrdb'])}  "
          f"SSIM={fmt(m['ssim_rrdb'])}  LPIPS={fmt(m['lpips_rrdb'])}")
    print(f"  JEPA-wrap (z-init) PSNR={fmt(m['psnr_wrap'])}  "
          f"                    LPIPS={fmt(m['lpips_wrap'])}")
    print("=" * 70)
    print(f"  identity-at-init check: max|wrap - rrdb| = "
          f"{m['wrap_vs_rrdb_maxabs']:.2e}  "
          f"({'OK' if m['wrap_vs_rrdb_maxabs'] < 1e-4 else 'FAIL'})")
    print("\n  Δ vs bilinear:")
    print(f"     bare RRDBNet:  PSNR {m['psnr_rrdb'] - m['psnr_bil']:+.3f} dB   "
          f"LPIPS {m['lpips_rrdb'] - m['lpips_bil']:+.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
