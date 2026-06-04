"""
Four-way comparison at apples-to-apples val (train-matched scale + degradation):
  (1) Bilinear
  (2) Bare RRDBNet pretrained (off-the-shelf, no fine-tune)
  (3) Bare RRDBNet fine-tuned (the v5 attribution control, best.pt)
  (4) JEPA-Conditioned RRDBNet fine-tuned (v5 best.pt)

Uses the SAME per-scale loop pattern as v5_three_way_comparison.py: mutate
od.VAL_SCALE before constructing each per-scale OrthoDataset (since OrthoDataset
reads VAL_SCALE LIVE at __getitem__ time, not at construct time).

Answers: did JEPA conditioning beat what plain Toronto fine-tuning of the same
pretrained RRDBNet could achieve at the same step budget?
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
def eval_all(v5, rrdb_pre, rrdb_ft, loader, device, lpips):
    v5.eval(); rrdb_pre.eval(); rrdb_ft.eval()
    keys = ["psnr_bil", "lpips_bil",
            "psnr_rpre", "lpips_rpre",
            "psnr_rft",  "lpips_rft",
            "psnr_v5",   "lpips_v5"]
    s = {k: 0.0 for k in keys}
    n = 0
    for batch in loader:
        lr = batch["low_res"].to(device, non_blocking=True)
        hr = batch["high_res"].to(device, non_blocking=True)

        s["psnr_bil"]  += psnr_metric(lr.clamp(0, 1), hr).item()
        s["lpips_bil"] += lpips(lr.clamp(0, 1), hr).item()

        lr_small = F.avg_pool2d(lr, 4)
        sr_pre = rrdb_pre(lr_small).clamp(0, 1)
        s["psnr_rpre"]  += psnr_metric(sr_pre, hr).item()
        s["lpips_rpre"] += lpips(sr_pre, hr).item()

        sr_ft = rrdb_ft(lr_small).clamp(0, 1)
        s["psnr_rft"]  += psnr_metric(sr_ft, hr).item()
        s["lpips_rft"] += lpips(sr_ft, hr).item()

        sr_v5 = v5(lr, hr)["sr"].clamp(0, 1)
        s["psnr_v5"]  += psnr_metric(sr_v5, hr).item()
        s["lpips_v5"] += lpips(sr_v5, hr).item()
        n += 1
    return {k: v / n for k, v in s.items()}, n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--v5_ckpt", required=True)
    p.add_argument("--ctrl_ckpt", required=True)
    p.add_argument("--scales", type=float, nargs="+",
                   default=[16, 18, 20, 22, 24])
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--match_aug", action="store_true", default=True)
    p.add_argument("--no_match_aug", action="store_false", dest="match_aug")
    args = p.parse_args()

    device = torch.device("cuda")
    print(f"[device] free={torch.cuda.mem_get_info()[0]/1024**3:.2f} GiB", flush=True)

    print("[models] building V5 + bare RRDBNet (pretrained) + bare RRDBNet (control)", flush=True)
    v5 = build_v5_model(
        backbone_name="dinov2_vitb14",
        jepa_pretrained_path="models/pretrained/dinov2_vitb14.pth",
        esrgan_weights_path="models/pretrained/RealESRGAN_x4plus.pth",
    ).to(device)
    rrdb_pre = build_pretrained_x4plus("models/pretrained/RealESRGAN_x4plus.pth").to(device).eval()
    rrdb_ft  = build_pretrained_x4plus("models/pretrained/RealESRGAN_x4plus.pth").to(device).eval()

    print(f"[ckpt] v5     {args.v5_ckpt}", flush=True)
    st_v5 = torch.load(args.v5_ckpt, map_location="cpu", weights_only=False)
    v5.load_checkpoint_state(st_v5["model"], strict=False)
    print(f"[ckpt] v5 global_step={st_v5.get('global_step')}", flush=True)

    print(f"[ckpt] ctrl   {args.ctrl_ckpt}", flush=True)
    st_ct = torch.load(args.ctrl_ckpt, map_location="cpu", weights_only=False)
    rrdb_ft.load_state_dict(st_ct["model"])
    print(f"[ckpt] ctrl global_step={st_ct.get('global_step')}", flush=True)

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
        m, n = eval_all(v5, rrdb_pre, rrdb_ft, loader, device, lpips)
        dt = time.time() - t0
        rows.append((s, m))
        print(f"  scale={s:4.1f}x  "
              f"bil={m['psnr_bil']:.3f}/{m['lpips_bil']:.4f}  "
              f"rpre={m['psnr_rpre']:.3f}/{m['lpips_rpre']:.4f}  "
              f"rft={m['psnr_rft']:.3f}/{m['lpips_rft']:.4f}  "
              f"v5={m['psnr_v5']:.3f}/{m['lpips_v5']:.4f}  "
              f"(n={n}, {dt:.1f}s)", flush=True)
    od.VAL_SCALE = orig

    print("\n" + "=" * 130)
    print("Four-way val comparison: bilinear / bare-RRDB-pretrained / bare-RRDB-finetuned / v5-JEPA "
          f"(LR aug = {'matched' if args.match_aug else 'clean'})")
    print("-" * 130)
    print(f"{'scale':>6}  | {'PSNR bil':>8}  {'PSNR Rpre':>9}  {'PSNR Rft':>8}  {'PSNR v5':>8}  | "
          f"{'LPIPS bil':>9}  {'LPIPS Rpre':>10}  {'LPIPS Rft':>9}  {'LPIPS v5':>9}")
    print("-" * 130)
    for s, m in rows:
        print(f"{s:6.1f}  | "
              f"{m['psnr_bil']:8.3f}  {m['psnr_rpre']:9.3f}  {m['psnr_rft']:8.3f}  {m['psnr_v5']:8.3f}  | "
              f"{m['lpips_bil']:9.4f}  {m['lpips_rpre']:10.4f}  {m['lpips_rft']:9.4f}  {m['lpips_v5']:9.4f}")

    def avg(k): return sum(m[k] for _, m in rows) / len(rows)
    pb, prp, prf, pv5 = avg("psnr_bil"), avg("psnr_rpre"), avg("psnr_rft"), avg("psnr_v5")
    lb, lrp, lrf, lv5 = avg("lpips_bil"), avg("lpips_rpre"), avg("lpips_rft"), avg("lpips_v5")
    print("-" * 130)
    print(f"{'avg':>6}  | "
          f"{pb:8.3f}  {prp:9.3f}  {prf:8.3f}  {pv5:8.3f}  | "
          f"{lb:9.4f}  {lrp:10.4f}  {lrf:9.4f}  {lv5:9.4f}")
    print("=" * 130)

    print(f"\nAveraged across scales {[float(s) for s,_ in rows]}:")
    print(f"  Bare RRDB (pretrained)    vs bilinear:   PSNR {prp-pb:+.3f} dB,  LPIPS {lrp-lb:+.4f}")
    print(f"  Bare RRDB (fine-tuned)    vs bilinear:   PSNR {prf-pb:+.3f} dB,  LPIPS {lrf-lb:+.4f}")
    print(f"  v5 JEPA                   vs bilinear:   PSNR {pv5-pb:+.3f} dB,  LPIPS {lv5-lb:+.4f}")
    print(f"  Bare RRDB (fine-tuned)    vs pretrained: PSNR {prf-prp:+.3f} dB,  LPIPS {lrf-lrp:+.4f}")
    print(f"  v5 JEPA                   vs pretrained: PSNR {pv5-prp:+.3f} dB,  LPIPS {lv5-lrp:+.4f}")
    print(f"\n  >>> JEPA contribution beyond fine-tuned RRDB control: "
          f"PSNR={pv5-prf:+.3f} dB, LPIPS={lv5-lrf:+.4f}  <<<")


if __name__ == "__main__":
    main()
