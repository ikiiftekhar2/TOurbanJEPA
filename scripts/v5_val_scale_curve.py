"""
Apples-to-apples val measurement on a v5 checkpoint.

Training samples scale ∈ continuous [16, 24] with realistic LR degradation
(random Gaussian blur sigma + 30% JPEG); val uses scale=20 fixed and clean
degradation (sigma=1.0, no JPEG). To make train↔val PSNR comparable, this
script sweeps the val set at multiple scales with `match_train_aug_in_val`
enabled (same degradation pipeline as train, deterministic per-sample seed).

Reports a scale → metrics curve so we can see *where* the model is strongest
across the trained range, not just at the single 20× point.
"""

import argparse, copy, time
import torch
from torch.utils.data import DataLoader

import src.data.ortho_dataset as od
from src.evaluation.metrics import psnr as psnr_metric, ssim_metric
from src.models.v5_model import build_v5_model
from src.training.losses import LPIPSLoss, l1_pixel_loss


@torch.no_grad()
def run_val(model, loader, device, lpips_loss):
    model.eval()
    s = {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0, "l1": 0.0,
         "psnr_bil": 0.0, "lpips_bil": 0.0}
    n = 0
    for batch in loader:
        lr = batch["low_res"].to(device, non_blocking=True)
        hr = batch["high_res"].to(device, non_blocking=True)
        out = model(lr, hr)
        sr = out["sr"].clamp(0, 1)
        s["psnr"]      += psnr_metric(sr, hr).item()
        s["ssim"]      += ssim_metric(sr, hr).item()
        s["l1"]        += l1_pixel_loss(sr, hr).item()
        s["lpips"]     += lpips_loss(sr, hr).item()
        s["psnr_bil"]  += psnr_metric(lr, hr).item()
        s["lpips_bil"] += lpips_loss(lr, hr).item()
        n += 1
    return {k: v/n for k, v in s.items()}, n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", help="V5 checkpoint .pt")
    p.add_argument("--scales", type=float, nargs="+",
                   default=[16, 18, 20, 22, 24])
    p.add_argument("--match_aug", action="store_true", default=True,
                   help="Use train-matched LR degradation (random sigma + JPEG)")
    p.add_argument("--no_match_aug", action="store_false", dest="match_aug")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=2)
    args = p.parse_args()

    device = torch.device("cuda")
    print(f"[device] free={torch.cuda.mem_get_info()[0]/1024**3:.2f} GiB", flush=True)
    print(f"[config] scales={args.scales}  match_aug={args.match_aug}", flush=True)

    # Build model + load ckpt (LIVE weights, bypass EMA — the v5 EMA had a NaN
    # shadow until step 6200; bypassing is the right baseline anyway).
    print("[model] building V5Model", flush=True)
    model = build_v5_model(
        backbone_name="dinov2_vitb14",
        jepa_pretrained_path="models/pretrained/dinov2_vitb14.pth",
        esrgan_weights_path="models/pretrained/RealESRGAN_x4plus.pth",
    ).to(device)
    print(f"[ckpt] loading {args.ckpt}", flush=True)
    st = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_checkpoint_state(st["model"], strict=False)
    print(f"[ckpt] global_step={st['global_step']}", flush=True)

    lpips = LPIPSLoss(net="vgg").to(device).eval()

    # Sweep scale by temporarily overriding the module-level VAL_SCALE.
    original_val_scale = od.VAL_SCALE
    rows = []
    for s in args.scales:
        od.VAL_SCALE = float(s)
        val_ds = od.OrthoDataset(
            ortho_dir="data/ortho", split="val", augment=False,
            val_patches_per_tile=1,
            tile_manifest="data/ortho/metadata/train_textured.txt",
            match_train_aug_in_val=args.match_aug,
        )
        val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.num_workers,
                                pin_memory=True)
        t0 = time.time()
        m, n = run_val(model, val_loader, device, lpips)
        dt = time.time() - t0
        rows.append((s, m, n, dt))
        print(f"  scale={s:4.1f}x  PSNR={m['psnr']:.3f} (bil={m['psnr_bil']:.3f}, "
              f"d={m['psnr']-m['psnr_bil']:+.3f})  "
              f"LPIPS={m['lpips']:.4f} (bil={m['lpips_bil']:.4f}, "
              f"d={m['lpips']-m['lpips_bil']:+.4f})  {n} batches in {dt:.1f}s",
              flush=True)
    od.VAL_SCALE = original_val_scale

    print("\n" + "=" * 90)
    print(f"  Val-distribution-matched curve "
          f"(LR degradation = {'train-style randomized' if args.match_aug else 'clean fixed'})")
    print("-" * 90)
    print(f"{'scale':>6}  {'PSNR':>6}  {'bil':>6}  {'Δbil':>7}  "
          f"{'LPIPS':>6}  {'bil':>6}  {'Δbil':>7}  {'SSIM':>6}  {'L1':>6}")
    print("-" * 90)
    sum_psnr = sum_lpips = sum_dpsnr = sum_dlpips = 0.0
    for s, m, _, _ in rows:
        print(f"{s:6.1f}  {m['psnr']:6.3f}  {m['psnr_bil']:6.3f}  "
              f"{m['psnr']-m['psnr_bil']:+7.3f}  "
              f"{m['lpips']:6.4f}  {m['lpips_bil']:6.4f}  "
              f"{m['lpips']-m['lpips_bil']:+7.4f}  "
              f"{m['ssim']:6.4f}  {m['l1']:6.4f}")
        sum_psnr += m["psnr"]; sum_lpips += m["lpips"]
        sum_dpsnr += m["psnr"] - m["psnr_bil"]
        sum_dlpips += m["lpips"] - m["lpips_bil"]
    n = len(rows)
    print("-" * 90)
    print(f"{'avg':>6}  {sum_psnr/n:6.3f}  {'':6s}  {sum_dpsnr/n:+7.3f}  "
          f"{sum_lpips/n:6.4f}  {'':6s}  {sum_dlpips/n:+7.4f}")
    print("=" * 90)


if __name__ == "__main__":
    main()
