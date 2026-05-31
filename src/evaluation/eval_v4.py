"""
v4 evaluation: PSNR / SSIM / MS-SSIM / L1 / LPIPS-Alex / LPIPS-VGG with
optional 8x TTA + sample visual grid.

Per V4_PLAN §10.2 the v4 eval pipeline replaces scripts/eval_full.py. It also
supports v4-trained checkpoints (use_v4_decoder / use_v4_predictor /
hierarchical_jepa) by passing through the same flags used at training time.

NIQE (no-reference IQA) is mentioned in V4_PLAN but requires the `pyiqa`
package which isn't installed in this env. Skipped here; can be added with
`pip install pyiqa` and a small additional branch.

Usage:
    PYTHONPATH=. python -m src.evaluation.eval_v4 \
        --checkpoint checkpoints/backbone_dinov2/epoch_1.pt \
        --backbone dinov2_vitb14 \
        --pretrained models/pretrained/dinov2_vitb14.pth \
        --n_patches 256 \
        --tta 8

    # smoke (16 patches, no TTA)
    PYTHONPATH=. python -m src.evaluation.eval_v4 \
        --checkpoint checkpoints/backbone_dinov2/epoch_1.pt \
        --backbone dinov2_vitb14 \
        --pretrained models/pretrained/dinov2_vitb14.pth \
        --n_patches 16 --smoke
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from pytorch_msssim import ms_ssim as ms_ssim_fn
from pytorch_msssim import ssim as ssim_fn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.ortho_dataset import OrthoDataset  # noqa: E402
from src.evaluation.tta import tta_predict  # noqa: E402
from src.models.urbanjepa import UrbanJEPA  # noqa: E402


METRIC_KEYS = ["psnr", "ssim", "msssim", "l1", "rmse", "lpips_alex", "lpips_vgg"]


def _psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = F.mse_loss(pred.clamp(0, 1), target.clamp(0, 1), reduction="none")
    mse = mse.mean(dim=[1, 2, 3])
    return torch.where(mse < 1e-12, torch.full_like(mse, 100.0),
                       10 * torch.log10(1.0 / mse))


def per_image_metrics(pred: torch.Tensor, hr: torch.Tensor,
                      lpips_alex, lpips_vgg) -> dict:
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


def to_pil(t: torch.Tensor) -> Image.Image:
    arr = (t.clamp(0, 1) * 255).byte().cpu().numpy().transpose(1, 2, 0)
    return Image.fromarray(arr)


def render_grid(samples: dict, save_path: Path, max_rows: int = 8) -> None:
    """samples: {row_idx: {col_name: tensor (3,H,W)}}.
    First col label is shown above; row label is the index."""
    rows = sorted(samples.keys())[:max_rows]
    if not rows:
        return
    col_names = list(samples[rows[0]].keys())
    cell = 192
    pad = 8
    header = 24
    label_w = 60
    W = label_w + len(col_names) * cell + (len(col_names) + 1) * pad
    H = header + len(rows) * cell + (len(rows) + 1) * pad
    img = Image.new("RGB", (W, H), (245, 245, 245))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 11)
    except Exception:
        font = ImageFont.load_default()
    for ci, name in enumerate(col_names):
        x = label_w + pad + ci * (cell + pad)
        draw.text((x + 4, 6), name, fill=(0, 0, 0), font=font)
    for ri, r in enumerate(rows):
        y = header + pad + ri * (cell + pad)
        draw.text((4, y + cell // 2 - 6), f"#{r}", fill=(0, 0, 0), font=font)
        for ci, col in enumerate(col_names):
            x = label_w + pad + ci * (cell + pad)
            im = to_pil(samples[r][col])
            if im.size != (cell, cell):
                im = im.resize((cell, cell), Image.BICUBIC)
            img.paste(im, (x, y))
    img.save(save_path)


def build_model(args, device) -> UrbanJEPA:
    model = UrbanJEPA(
        backbone_name=args.backbone,
        pretrained_path=args.pretrained,
        predictor_depth=args.predictor_depth,
        decoder_attn_blocks=args.decoder_attn_blocks,
        decoder_base_dim=args.decoder_base_dim,
        dropout=args.dropout,
        use_v4_predictor=args.use_v4_predictor,
        use_v4_decoder=args.use_v4_decoder,
        hierarchical_jepa=args.hierarchical_jepa,
    ).to(device)
    sd = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = sd["model"] if isinstance(sd, dict) and "model" in sd else sd
    model.load_checkpoint_state(state)
    model.eval()
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--pretrained", required=True)
    ap.add_argument("--data_dir", default="data/ortho")
    ap.add_argument("--n_patches", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--tta", type=int, default=1, choices=[1, 4, 8],
                    help="1 = no TTA, 4 = rotations only, 8 = full dihedral")
    ap.add_argument("--out_dir", default="experiments/eval_v4")
    ap.add_argument("--smoke", action="store_true",
                    help="quick smoke pass: small batch, no grid, no TTA")
    ap.add_argument("--n_grid_samples", type=int, default=8)
    ap.add_argument("--cpu", action="store_true")
    # v3/v4 architecture flags (must match training to load weights cleanly)
    ap.add_argument("--predictor_depth", type=int, default=6)
    ap.add_argument("--decoder_attn_blocks", type=int, default=4)
    ap.add_argument("--decoder_base_dim", type=int, default=768)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--use_v4_predictor", action="store_true")
    ap.add_argument("--use_v4_decoder", action="store_true")
    ap.add_argument("--hierarchical_jepa", action="store_true")
    ap.add_argument("--match_train_aug_in_val", action="store_true",
                    help="apply same realistic LR degradation to val")
    args = ap.parse_args()

    device = torch.device("cpu" if args.cpu else
                          "cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        args.tta = 1  # smoke shouldn't do 8x extra work
        args.n_patches = min(args.n_patches, 16)
        print("[smoke] capped n_patches=16, tta=1")

    print(f"Device: {device}, amp: {amp_dtype}, TTA={args.tta}x")

    import lpips
    print("Loading LPIPS-Alex + LPIPS-VGG...")
    lpips_alex = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    lpips_vgg = lpips.LPIPS(net="vgg", verbose=False).to(device).eval()
    for p in lpips_alex.parameters():
        p.requires_grad = False
    for p in lpips_vgg.parameters():
        p.requires_grad = False

    print("Building model + loading checkpoint...")
    model = build_model(args, device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  model: {n_params:.1f} M params")

    val_ds = OrthoDataset(args.data_dir, split="val", augment=False, seed=42,
                          match_train_aug_in_val=args.match_train_aug_in_val)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=device.type == "cuda")
    n_total = len(val_ds)
    max_batches = (args.n_patches + args.batch_size - 1) // args.batch_size
    n_to_eval = min(args.n_patches, n_total)
    print(f"Val: {n_total} total, evaluating {n_to_eval} patches "
          f"in {max_batches} batches of {args.batch_size}")

    # TTA wrapper around model: we need a callable that takes (low_res) and
    # returns (pred_image). The model's forward signature is (lr, hr); we use
    # the same hr inside (it's only used for the JEPA target encoder which is
    # discarded under no_grad eval).
    def model_pred(lr_batch, hr_batch):
        with torch.amp.autocast("cuda", dtype=amp_dtype,
                                enabled=device.type == "cuda"):
            out = model(lr_batch, hr_batch)
        pred = out["pred_image"].float().clamp(0, 1)
        extra = {}
        if "lr_clamp_rate" in out:
            extra["lr_clamp_rate"] = float(out["lr_clamp_rate"].item())
        return pred, extra

    agg = {k: [] for k in METRIC_KEYS}
    sample_panels: dict = {}
    sample_count = 0
    extras = []
    t0 = time.time()
    with torch.no_grad():
        for bi, batch in enumerate(val_loader):
            if bi >= max_batches:
                break
            lr = batch["low_res"].to(device, non_blocking=True)
            hr = batch["high_res"].to(device, non_blocking=True)
            if args.tta == 1:
                pred, ex = model_pred(lr, hr)
            else:
                # For TTA, run the same hr through (the target encoder pass is
                # unaffected by geometric augmentation of lr).
                def fn(z):
                    p, _ = model_pred(z, hr)
                    return p
                pred = tta_predict(fn, lr, n_transforms=args.tta).clamp(0, 1)
                ex = {}
            if ex:
                extras.append(ex)
            m = per_image_metrics(pred, hr, lpips_alex, lpips_vgg)
            for k in METRIC_KEYS:
                agg[k].append(m[k])
            # Collect a few visual samples
            if sample_count < args.n_grid_samples and not args.smoke:
                for i in range(pred.shape[0]):
                    if sample_count >= args.n_grid_samples:
                        break
                    sample_panels[sample_count] = {
                        "HR": hr[i].cpu(),
                        "LR": lr[i].cpu(),
                        "Pred": pred[i].cpu(),
                    }
                    sample_count += 1
            if bi % 10 == 0:
                print(f"  batch {bi}/{max_batches} ({time.time()-t0:.0f}s)")

    means = {k: float(torch.cat(agg[k]).mean()) for k in METRIC_KEYS}
    stds = {k: float(torch.cat(agg[k]).std()) for k in METRIC_KEYS}
    n_seen = sum(t.numel() for t in agg["psnr"])

    print("\n" + "=" * 80)
    print(f"{'metric':<15} {'mean':>12} {'std':>12}")
    print("-" * 80)
    for k in METRIC_KEYS:
        print(f"{k:<15} {means[k]:>12.4f} {stds[k]:>12.4f}")
    print(f"{'n_patches':<15} {n_seen:>12}")
    if extras:
        avg_cr = sum(e.get("lr_clamp_rate", 0.0) for e in extras) / len(extras)
        print(f"{'lr_clamp_rate':<15} {avg_cr:>12.4e}  (over {len(extras)} batches)")
    print("=" * 80)

    # Outputs
    name = Path(args.checkpoint).parent.name + "__" + Path(args.checkpoint).stem
    out_json = out_dir / f"{name}.json"
    out_json.write_text(json.dumps(
        {"checkpoint": str(args.checkpoint), "backbone": args.backbone,
         "tta": args.tta, "n_patches": n_seen,
         "means": means, "stds": stds,
         "smoke": args.smoke}, indent=2))
    print(f"\nWrote: {out_json}")

    if sample_panels and not args.smoke:
        grid_path = out_dir / f"{name}_grid.png"
        render_grid(sample_panels, grid_path, max_rows=args.n_grid_samples)
        print(f"Wrote: {grid_path}")

    if args.smoke:
        # Smoke assertions
        for k in METRIC_KEYS:
            assert torch.cat(agg[k]).isfinite().all(), f"NaN/Inf in {k}"
        print("eval_v4 smoke PASSED")


if __name__ == "__main__":
    main()
