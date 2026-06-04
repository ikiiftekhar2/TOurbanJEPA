"""
Three-way comparison at apples-to-apples val (train-matched scale + degradation):
  (1) Bilinear (the LR-upsampled baseline)
  (2) Bare RRDBNet (pretrained Real-ESRGAN x4plus, no JEPA, no fine-tune)
  (3) JEPA-Conditioned RRDBNet at the given checkpoint

Per-batch, per-scale. The bare RRDBNet output is `rrdbnet(avg_pool_4x(lr))`.
For (3) we run the full V5Model forward with whatever weights the ckpt holds.

Lets us answer directly: is JEPA conditioning earning its weight over the
"free" pretrained-ESRGAN baseline?
"""

import argparse, time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import src.data.ortho_dataset as od
from src.evaluation.metrics import psnr as psnr_metric, ssim_metric
from src.models.esrgan.weight_loader import build_pretrained_x4plus
from src.models.v5_model import build_v5_model
from src.training.losses import LPIPSLoss


@torch.no_grad()
def eval_three(jepa_model, rrdbnet, loader, device, lpips_loss):
    jepa_model.eval()
    rrdbnet.eval()
    keys = ["psnr_bil", "lpips_bil", "psnr_rrdb", "lpips_rrdb",
            "psnr_jepa", "lpips_jepa", "ssim_jepa", "l1_jepa"]
    s = {k: 0.0 for k in keys}
    n = 0
    for batch in loader:
        lr = batch["low_res"].to(device, non_blocking=True)
        hr = batch["high_res"].to(device, non_blocking=True)

        # 1. bilinear
        s["psnr_bil"]  += psnr_metric(lr.clamp(0, 1), hr).item()
        s["lpips_bil"] += lpips_loss(lr.clamp(0, 1), hr).item()

        # 2. bare RRDBNet
        sr_rrdb = rrdbnet(F.avg_pool2d(lr, 4)).clamp(0, 1)
        s["psnr_rrdb"]  += psnr_metric(sr_rrdb, hr).item()
        s["lpips_rrdb"] += lpips_loss(sr_rrdb, hr).item()

        # 3. JEPA-Conditioned RRDBNet
        out = jepa_model(lr, hr)
        sr_j = out["sr"].clamp(0, 1)
        s["psnr_jepa"]  += psnr_metric(sr_j, hr).item()
        s["lpips_jepa"] += lpips_loss(sr_j, hr).item()
        s["ssim_jepa"]  += ssim_metric(sr_j, hr).item()
        s["l1_jepa"]    += (sr_j - hr).abs().mean().item()
        n += 1
    return {k: v/n for k, v in s.items()}, n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--scales", type=float, nargs="+",
                   default=[16, 18, 20, 22, 24])
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--match_aug", action="store_true", default=True)
    p.add_argument("--no_match_aug", action="store_false", dest="match_aug")
    args = p.parse_args()

    device = torch.device("cuda")
    print(f"[device] free={torch.cuda.mem_get_info()[0]/1024**3:.2f} GiB", flush=True)

    print("[models] building V5 + standalone RRDBNet", flush=True)
    jepa_model = build_v5_model(
        backbone_name="dinov2_vitb14",
        jepa_pretrained_path="models/pretrained/dinov2_vitb14.pth",
        esrgan_weights_path="models/pretrained/RealESRGAN_x4plus.pth",
    ).to(device)
    rrdbnet = build_pretrained_x4plus(
        "models/pretrained/RealESRGAN_x4plus.pth").to(device).eval()

    print(f"[ckpt] loading {args.ckpt}", flush=True)
    st = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    jepa_model.load_checkpoint_state(st["model"], strict=False)
    print(f"[ckpt] global_step={st['global_step']}", flush=True)

    lpips = LPIPSLoss(net="vgg").to(device).eval()

    orig = od.VAL_SCALE
    rows = []
    for s in args.scales:
        od.VAL_SCALE = float(s)
        val_ds = od.OrthoDataset(
            ortho_dir="data/ortho", split="val", augment=False,
            val_patches_per_tile=1,
            tile_manifest="data/ortho/metadata/train_textured.txt",
            match_train_aug_in_val=args.match_aug,
        )
        loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
        t0 = time.time()
        m, n = eval_three(jepa_model, rrdbnet, loader, device, lpips)
        dt = time.time() - t0
        rows.append((s, m))
        print(f"  scale={s:4.1f}x  "
              f"bil={m['psnr_bil']:.3f}/{m['lpips_bil']:.4f}  "
              f"rrdb={m['psnr_rrdb']:.3f}/{m['lpips_rrdb']:.4f}  "
              f"jepa={m['psnr_jepa']:.3f}/{m['lpips_jepa']:.4f}  "
              f"({dt:.1f}s)", flush=True)
    od.VAL_SCALE = orig

    # -------- summary --------
    print("\n" + "=" * 110)
    print("Three-way val comparison: bilinear vs bare RRDBNet vs JEPA-Conditioned "
          f"(LR aug = {'matched' if args.match_aug else 'clean'})")
    print("-" * 110)
    print(f"{'scale':>6}  | "
          f"{'PSNR bil':>9}  {'PSNR rrdb':>9}  {'Δ_bil':>7}  | "
          f"{'PSNR jepa':>9}  {'Δ_bil':>7}  {'Δ_rrdb':>7}  | "
          f"{'LPIPS rrdb':>10}  {'LPIPS jepa':>10}  {'Δ_rrdb':>7}")
    print("-" * 110)
    for s, m in rows:
        d_rrdb_psnr  = m["psnr_rrdb"] - m["psnr_bil"]
        d_jepa_psnr_bil  = m["psnr_jepa"] - m["psnr_bil"]
        d_jepa_psnr_rrdb = m["psnr_jepa"] - m["psnr_rrdb"]
        d_jepa_lpips_rrdb = m["lpips_jepa"] - m["lpips_rrdb"]
        print(f"{s:6.1f}  | "
              f"{m['psnr_bil']:9.3f}  {m['psnr_rrdb']:9.3f}  {d_rrdb_psnr:+7.3f}  | "
              f"{m['psnr_jepa']:9.3f}  {d_jepa_psnr_bil:+7.3f}  {d_jepa_psnr_rrdb:+7.3f}  | "
              f"{m['lpips_rrdb']:10.4f}  {m['lpips_jepa']:10.4f}  {d_jepa_lpips_rrdb:+7.4f}")
    # averages
    def avg(k): return sum(m[k] for _, m in rows) / len(rows)
    print("-" * 110)
    pb, pr, pj = avg("psnr_bil"), avg("psnr_rrdb"), avg("psnr_jepa")
    lb, lr_, lj = avg("lpips_bil"), avg("lpips_rrdb"), avg("lpips_jepa")
    print(f"{'avg':>6}  | "
          f"{pb:9.3f}  {pr:9.3f}  {pr-pb:+7.3f}  | "
          f"{pj:9.3f}  {pj-pb:+7.3f}  {pj-pr:+7.3f}  | "
          f"{lr_:10.4f}  {lj:10.4f}  {lj-lr_:+7.4f}")
    print("=" * 110)
    print(f"\nVerdict (averaged across scales {[float(s) for s,_ in rows]}):")
    print(f"  Bare RRDBNet vs bilinear:  PSNR {pr-pb:+.3f} dB,  LPIPS {lr_-lb:+.4f}")
    print(f"  JEPA vs bilinear:          PSNR {pj-pb:+.3f} dB,  LPIPS {lj-lb:+.4f}")
    print(f"  JEPA vs bare RRDBNet:      PSNR {pj-pr:+.3f} dB,  LPIPS {lj-lr_:+.4f}")
    print(f"  >>> JEPA contribution beyond bare RRDBNet: "
          f"PSNR={pj-pr:+.3f} dB, LPIPS={lj-lr_:+.4f}")


if __name__ == "__main__":
    main()
