"""
Quick eval of multiple dinov2 checkpoints to pick the best one for v4 baseline.
Reuses the metric stack from eval_all.py.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from pytorch_msssim import ms_ssim as ms_ssim_fn
from pytorch_msssim import ssim as ssim_fn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.ortho_dataset import OrthoDataset  # noqa: E402
from src.models.urbanjepa import UrbanJEPA  # noqa: E402

PRETRAINED = "models/pretrained/dinov2_vitb14.pth"

CANDIDATES = [
    ("dinov2-E0 (best.pt)",          "checkpoints/backbone_dinov2/best.pt"),
    ("dinov2-E1 (epoch_1.pt)",       "checkpoints/backbone_dinov2/epoch_1.pt"),
    ("dinov2-E2 full (epoch_2.pt)",  "checkpoints/backbone_dinov2/epoch_2.pt"),
    ("dinov2-E2 slot_0 (b20793)",    "checkpoints/backbone_dinov2/step_slot_0.pt"),
    ("dinov2-E2 slot_1 (b20543)",    "checkpoints/backbone_dinov2/step_slot_1.pt"),
]


def _psnr(pred, target):
    mse = F.mse_loss(pred.clamp(0, 1), target.clamp(0, 1), reduction="none")
    mse = mse.mean(dim=[1, 2, 3])
    return torch.where(mse < 1e-12, torch.full_like(mse, 100.0),
                       10 * torch.log10(1.0 / mse))


def per_image_metrics(pred, hr, lpips_alex, lpips_vgg):
    pred = pred.clamp(0, 1)
    hr = hr.clamp(0, 1)
    psnr_v = _psnr(pred, hr)
    l1 = (pred - hr).abs().mean(dim=[1, 2, 3])
    rmse = ((pred - hr) ** 2).mean(dim=[1, 2, 3]).sqrt()
    ssim_v = torch.stack([ssim_fn(pred[i:i + 1], hr[i:i + 1], data_range=1.0,
                                  size_average=True) for i in range(pred.shape[0])])
    msssim_v = torch.stack([ms_ssim_fn(pred[i:i + 1], hr[i:i + 1], data_range=1.0,
                                       size_average=True) for i in range(pred.shape[0])])
    alex_v = lpips_alex(pred * 2 - 1, hr * 2 - 1).view(-1)
    vgg_v = lpips_vgg(pred * 2 - 1, hr * 2 - 1).view(-1)
    return {"psnr": psnr_v.cpu(), "ssim": ssim_v.cpu(), "msssim": msssim_v.cpu(),
            "l1": l1.cpu(), "rmse": rmse.cpu(),
            "lpips_alex": alex_v.cpu(), "lpips_vgg": vgg_v.cpu()}


METRIC_KEYS = ["psnr", "ssim", "msssim", "l1", "rmse", "lpips_alex", "lpips_vgg"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/ortho")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--out", default="experiments/eval_all/dinov2_picks.csv")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"Device: {device}, amp: {amp_dtype}")

    import lpips
    print("Loading LPIPS-Alex + LPIPS-VGG...")
    lpips_alex = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    lpips_vgg = lpips.LPIPS(net="vgg", verbose=False).to(device).eval()
    for p in lpips_alex.parameters():
        p.requires_grad = False
    for p in lpips_vgg.parameters():
        p.requires_grad = False

    val_ds = OrthoDataset(args.data_dir, split="val", augment=False, seed=42)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    rows = []
    for label, ckpt in CANDIDATES:
        ck = Path(ckpt)
        if not ck.exists():
            print(f"[skip] {label}: {ckpt} not found")
            continue
        print(f"\n=== {label} ===")
        t0 = time.time()
        model = UrbanJEPA(backbone_name="dinov2_vitb14",
                          pretrained_path=PRETRAINED).to(device)
        sd = torch.load(str(ck), map_location="cpu", weights_only=False)
        model.load_checkpoint_state(sd["model"])
        model.eval()

        agg = {k: [] for k in METRIC_KEYS}
        with torch.no_grad():
            for bi, batch in enumerate(val_loader):
                lr = batch["low_res"].to(device, non_blocking=True)
                hr = batch["high_res"].to(device, non_blocking=True)
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    out = model(lr, hr)
                pred = out["pred_image"].float().clamp(0, 1)
                m = per_image_metrics(pred, hr, lpips_alex, lpips_vgg)
                for k in METRIC_KEYS:
                    agg[k].append(m[k])
                if bi % 20 == 0:
                    print(f"  batch {bi}/{len(val_loader)} ({time.time()-t0:.0f}s)")
        means = {k: float(torch.cat(agg[k]).mean()) for k in METRIC_KEYS}
        print(f"  PSNR={means['psnr']:.2f}  SSIM={means['ssim']:.4f}  "
              f"L1={means['l1']:.4f}  LPIPS-A={means['lpips_alex']:.4f}  "
              f"LPIPS-V={means['lpips_vgg']:.4f}")
        rows.append({"label": label, "ckpt": ckpt,
                     "step": sd.get("global_step"), **means})
        del model
        torch.cuda.empty_cache()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "ckpt", "step"] + METRIC_KEYS)
        for r in rows:
            w.writerow([r["label"], r["ckpt"], r["step"]]
                       + [f"{r[k]:.6f}" for k in METRIC_KEYS])

    print("\n" + "=" * 100)
    print(f"{'ckpt':<30} {'step':>7} {'PSNR':>7} {'SSIM':>7} {'L1':>7} "
          f"{'LPIPS-A':>9} {'LPIPS-V':>9}")
    print("-" * 100)
    for r in rows:
        print(f"{r['label']:<30} {r['step']:>7} "
              f"{r['psnr']:>5.2f}dB {r['ssim']:>7.4f} {r['l1']:>7.4f} "
              f"{r['lpips_alex']:>9.4f} {r['lpips_vgg']:>9.4f}")
    print("=" * 100)
    print(f"\nWritten to: {out}")


if __name__ == "__main__":
    main()
