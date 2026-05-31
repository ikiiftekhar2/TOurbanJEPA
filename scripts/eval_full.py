"""
Full evaluation: PSNR + SSIM + MS-SSIM + LPIPS + L1 on a backbone checkpoint.

PSNR rewards per-pixel match (and hates plausible-detail-in-wrong-position).
LPIPS rewards "looks real to a deep classifier" — much closer to human judgment.
A model that loses on PSNR but wins on LPIPS is usually preferred by humans.

Runs on CPU by default so it doesn't contend with GPU training. Reuses the v3
val OrthoDataset exactly (PSF σ=1.0 + downsample + bilinear up).

Usage:
    PYTHONPATH=. python scripts/eval_full.py \
        --checkpoint checkpoints/backbone_imagenet/epoch_2.pt \
        --backbone imagenet_vitb16 \
        --pretrained models/pretrained/imagenet_vitb16.pt \
        --n_patches 256

    # Or auto-discover the latest checkpoint per backbone:
    PYTHONPATH=. python scripts/eval_full.py --auto-all
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.ortho_dataset import OrthoDataset
from src.models.urbanjepa import UrbanJEPA


BACKBONE_DEFAULTS = {
    "imagenet_vitb16": "models/pretrained/imagenet_vitb16.pt",
    "dinov2_vitb14": "models/pretrained/dinov2_vitb14.pth",
    "explora_vitb14": "models/pretrained/explora_vitb14.pth",
}


def _psnr(pred, target):
    mse = F.mse_loss(pred.clamp(0, 1), target.clamp(0, 1))
    if mse.item() < 1e-12:
        return float("inf")
    return (10 * torch.log10(1.0 / mse)).item()


def _bilinear_lr(low_res):
    # OrthoDataset returns the LR already upsampled to 256 via bilinear; this is
    # the same input the model trains on. As a fair baseline we just compute
    # metrics against that.
    return low_res.clamp(0, 1)


@torch.no_grad()
def evaluate(model, loader, device, ssim_fn, ms_ssim_fn, lpips_fn,
             max_batches: int = None):
    model.eval()
    n = 0
    metrics = {k: 0.0 for k in ("pred_psnr", "pred_l1", "pred_ssim", "pred_msssim",
                                "pred_lpips", "bilinear_psnr", "bilinear_lpips",
                                "bilinear_ssim", "bilinear_msssim")}
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        lr = batch["low_res"].to(device)
        hr = batch["high_res"].to(device)
        pred = model(lr, hr)["pred_image"].float().clamp(0, 1)
        bil = _bilinear_lr(lr)
        # PSNR
        metrics["pred_psnr"] += _psnr(pred, hr)
        metrics["bilinear_psnr"] += _psnr(bil, hr)
        # L1
        metrics["pred_l1"] += F.l1_loss(pred, hr).item()
        # SSIM / MS-SSIM (need data_range=1.0)
        metrics["pred_ssim"] += ssim_fn(pred, hr, data_range=1.0, size_average=True).item()
        metrics["bilinear_ssim"] += ssim_fn(bil, hr, data_range=1.0, size_average=True).item()
        metrics["pred_msssim"] += ms_ssim_fn(pred, hr, data_range=1.0, size_average=True).item()
        metrics["bilinear_msssim"] += ms_ssim_fn(bil, hr, data_range=1.0, size_average=True).item()
        # LPIPS (expects [-1, 1] inputs)
        metrics["pred_lpips"] += lpips_fn(pred * 2 - 1, hr * 2 - 1).mean().item()
        metrics["bilinear_lpips"] += lpips_fn(bil * 2 - 1, hr * 2 - 1).mean().item()
        n += 1
    return {k: v / max(1, n) for k, v in metrics.items()}, n


def _print_table(label: str, m: dict, n: int):
    print(f"\n--- {label} (n={n} batches) ---")
    print(f"{'metric':<10} {'pred':>10} {'bilinear':>12} {'pred wins?':>14}")
    print("-" * 50)
    for short, pred_k, bil_k, higher_better in [
        ("PSNR", "pred_psnr", "bilinear_psnr", True),
        ("SSIM", "pred_ssim", "bilinear_ssim", True),
        ("MS-SSIM", "pred_msssim", "bilinear_msssim", True),
        ("LPIPS", "pred_lpips", "bilinear_lpips", False),  # lower better
    ]:
        a, b = m[pred_k], m[bil_k]
        win = ("YES" if (a > b) == higher_better else "no")
        fmt = "{:>10.4f}"
        if short == "PSNR":
            fmt = "{:>8.2f}dB"
        print(f"{short:<10} {fmt.format(a)} {fmt.format(b)} {win:>14}")
    print(f"L1         {m['pred_l1']:>10.4f}")


def run_one(args, backbone: str, ckpt_path: Path, pretrained: Path):
    print("\n" + "=" * 70)
    print(f"Backbone: {backbone}   ckpt: {ckpt_path}")
    print("=" * 70)

    device = torch.device("cpu" if args.cpu else
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    from pytorch_msssim import ssim as ssim_fn, ms_ssim as ms_ssim_fn
    import lpips
    print("Loading LPIPS (alex)...")
    lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(device)
    lpips_fn.eval()

    print("Loading model + checkpoint...")
    model = UrbanJEPA(backbone_name=backbone, pretrained_path=str(pretrained)).to(device)
    sd = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    model.load_checkpoint_state(sd["model"])

    val_ds = OrthoDataset(args.data_dir, split="val", augment=False, seed=42)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=False)
    max_batches = (args.n_patches + args.batch_size - 1) // args.batch_size
    print(f"Evaluating {min(args.n_patches, len(val_ds))} patches in "
          f"{max_batches} batches of {args.batch_size}...")

    metrics, n = evaluate(model, val_loader, device, ssim_fn, ms_ssim_fn,
                          lpips_fn, max_batches=max_batches)
    _print_table(backbone, metrics, n)
    return {"backbone": backbone, "checkpoint": str(ckpt_path),
            "global_step": sd.get("global_step"),
            "epoch": sd.get("epoch"), "metrics": metrics, "n_batches": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", help="Single checkpoint to evaluate")
    ap.add_argument("--backbone", help="Backbone name (required with --checkpoint)")
    ap.add_argument("--pretrained", help="Pretrained path (defaults to BACKBONE_DEFAULTS)")
    ap.add_argument("--auto-all", action="store_true",
                    help="Auto-discover latest checkpoint per backbone in checkpoints/")
    ap.add_argument("--data_dir", default="data/ortho")
    ap.add_argument("--n_patches", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--cpu", action="store_true", default=True,
                    help="Force CPU (default true to avoid GPU contention)")
    ap.add_argument("--gpu", dest="cpu", action="store_false")
    ap.add_argument("--out", default="runs/eval_full_results.json")
    args = ap.parse_args()

    results = []
    if args.auto_all:
        for name, default_pre in BACKBONE_DEFAULTS.items():
            exp_dir = Path("checkpoints") / f"backbone_{name.split('_')[0]}"
            ckpts = sorted(exp_dir.glob("epoch_*.pt"))
            if not ckpts:
                ckpts = sorted(exp_dir.glob("step_slot_*.pt"))
            if not ckpts:
                print(f"[skip] {name}: no checkpoints in {exp_dir}")
                continue
            results.append(run_one(args, name, ckpts[-1], Path(default_pre)))
    else:
        if not args.checkpoint or not args.backbone:
            ap.error("--checkpoint and --backbone required (or use --auto-all)")
        pretrained = args.pretrained or BACKBONE_DEFAULTS[args.backbone]
        results.append(run_one(args, args.backbone, Path(args.checkpoint),
                               Path(pretrained)))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {args.out}")


if __name__ == "__main__":
    main()
