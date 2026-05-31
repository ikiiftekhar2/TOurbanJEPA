"""
Comprehensive evaluation: classical baselines + all trained backbones + oracle-picker.

For each method we compute on the deterministic val split:
  PSNR, SSIM, MS-SSIM, L1, L2/RMSE, LPIPS-Alex, LPIPS-VGG.

Methods:
  Classical (no learning):    nearest, bilinear, bicubic, lanczos
  Trained best.pt:            imagenet, dinov2, explora    (best val PSNR ckpt)
  Trained epoch_2.pt:         imagenet, dinov2, explora    (final ckpt)
  Oracle picker:              per-image best PSNR over the 6 trained variants

Outputs:
  experiments/eval_all/full_val_stats.csv      aggregate over full val set
  experiments/eval_all/per_sample_stats.csv    per-sample over 20 visual samples
  experiments/eval_all/eval_all_results.json   everything machine-readable
  experiments/eval_all/grid_full.png           20 rows x 12 cols composite
  experiments/eval_all/grid_classical.png      20 rows x 6 cols  (HR LR + 4 classical)
  experiments/eval_all/grid_trained_best.png   20 rows x 6 cols  (HR LR + 3 best + oracle)
  experiments/eval_all/grid_trained_last.png   20 rows x 6 cols  (HR LR + 3 last + oracle)
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from pytorch_msssim import ms_ssim as ms_ssim_fn
from pytorch_msssim import ssim as ssim_fn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.ortho_dataset import OrthoDataset  # noqa: E402
from src.models.urbanjepa import UrbanJEPA  # noqa: E402

PATCH = 256

BACKBONES = [
    ("imagenet", "imagenet_vitb16", "models/pretrained/imagenet_vitb16.pt"),
    ("dinov2", "dinov2_vitb14", "models/pretrained/dinov2_vitb14.pth"),
    ("explora", "explora_vitb14", "models/pretrained/explora_vitb14.pth"),
]

# (label, backbone-id, pretrained, ckpt-path)
MODEL_VARIANTS = []
for short, backbone, pretrained in BACKBONES:
    for tag in ("best", "epoch_2"):
        MODEL_VARIANTS.append((
            f"{short}-{tag}",
            backbone,
            pretrained,
            f"checkpoints/backbone_{short}/{tag}.pt",
        ))

CLASSICAL_NAMES = ["nearest", "bilinear", "bicubic", "lanczos"]


# ---------- classical upsamplers (apply to the "small" intermediate) ----------

def reverse_to_small(lr_up: torch.Tensor, scale: int) -> torch.Tensor:
    """The val LR is already bilinear-upsampled to 256. To compare other
    upsamplers we need to recover the small intermediate. The dataset path is
      hr -> psf -> avg_pool(scale) -> bilinear_up = lr_up
    so we can re-downsample lr_up with area to get an estimate. But that
    estimate is double-blurred. The clean route is to redo the degradation on
    HR with the same recipe (the dataset uses sigma=1.0, val_mode). We do that
    here for parity, instead of trying to invert."""
    raise NotImplementedError  # not used — we redo degradation from HR below


def _gauss_kernel(sigma: float, ksize: int = 9) -> torch.Tensor:
    x = torch.arange(ksize, dtype=torch.float32) - (ksize - 1) / 2
    k = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def _gauss_blur(t: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    k1 = _gauss_kernel(sigma).to(t.device)
    k = (k1[:, None] * k1[None, :])[None, None]
    pad = k.shape[-1] // 2
    out = []
    for c in range(t.shape[1]):
        x = F.pad(t[:, c:c + 1], [pad] * 4, mode="reflect")
        out.append(F.conv2d(x, k))
    return torch.cat(out, dim=1)


def degrade_to_small(hr: torch.Tensor, scale: int) -> torch.Tensor:
    """Exact val degradation (PSF sigma=1.0 + avg_pool when integer scale)."""
    t = _gauss_blur(hr, sigma=1.0)
    if abs(scale - round(scale)) < 1e-6:
        s = int(round(scale))
        small = F.avg_pool2d(t, kernel_size=s, stride=s)
    else:
        target = max(1, int(round(PATCH / scale)))
        small = F.interpolate(t, size=(target, target), mode="area")
    return small


def up_nearest(small):
    return F.interpolate(small, size=(PATCH, PATCH), mode="nearest").clamp(0, 1)


def up_bilinear(small):
    return F.interpolate(small, size=(PATCH, PATCH), mode="bilinear",
                         align_corners=False).clamp(0, 1)


def up_bicubic(small):
    return F.interpolate(small, size=(PATCH, PATCH), mode="bicubic",
                         align_corners=False).clamp(0, 1)


def up_lanczos(small):
    arr = (small.clamp(0, 1) * 255).byte().cpu().numpy()
    out = np.empty((arr.shape[0], arr.shape[1], PATCH, PATCH), dtype=np.uint8)
    for b in range(arr.shape[0]):
        for c in range(arr.shape[1]):
            img = Image.fromarray(arr[b, c])
            out[b, c] = np.asarray(img.resize((PATCH, PATCH), Image.LANCZOS))
    return (torch.from_numpy(out).float() / 255.0).to(small.device)


CLASSICAL_FNS = {
    "nearest": up_nearest,
    "bilinear": up_bilinear,
    "bicubic": up_bicubic,
    "lanczos": up_lanczos,
}


# ---------- metrics ----------

def _psnr(pred, target):
    mse = F.mse_loss(pred.clamp(0, 1), target.clamp(0, 1), reduction="none")
    mse = mse.mean(dim=[1, 2, 3])
    return torch.where(mse < 1e-12, torch.full_like(mse, 100.0),
                       10 * torch.log10(1.0 / mse))


def per_image_metrics(pred, hr, lpips_alex, lpips_vgg):
    """Returns dict of per-image-tensor metrics, all shape (B,)."""
    pred = pred.clamp(0, 1)
    hr = hr.clamp(0, 1)
    psnr_v = _psnr(pred, hr)
    l1 = (pred - hr).abs().mean(dim=[1, 2, 3])
    l2 = ((pred - hr) ** 2).mean(dim=[1, 2, 3])
    rmse = l2.sqrt()
    # SSIM/MS-SSIM are global (per-image)
    ssim_v = torch.stack([ssim_fn(pred[i:i + 1], hr[i:i + 1],
                                  data_range=1.0, size_average=True)
                          for i in range(pred.shape[0])])
    msssim_v = torch.stack([ms_ssim_fn(pred[i:i + 1], hr[i:i + 1],
                                       data_range=1.0, size_average=True)
                            for i in range(pred.shape[0])])
    # LPIPS expects [-1, 1]
    alex_v = lpips_alex(pred * 2 - 1, hr * 2 - 1).view(-1)
    vgg_v = lpips_vgg(pred * 2 - 1, hr * 2 - 1).view(-1)
    return {
        "psnr": psnr_v.cpu(),
        "ssim": ssim_v.cpu(),
        "msssim": msssim_v.cpu(),
        "l1": l1.cpu(),
        "rmse": rmse.cpu(),
        "lpips_alex": alex_v.cpu(),
        "lpips_vgg": vgg_v.cpu(),
    }


METRIC_KEYS = ["psnr", "ssim", "msssim", "l1", "rmse", "lpips_alex", "lpips_vgg"]
HIGHER_IS_BETTER = {"psnr": True, "ssim": True, "msssim": True,
                    "l1": False, "rmse": False,
                    "lpips_alex": False, "lpips_vgg": False}


# ---------- visualization ----------

def to_pil(t):
    arr = (t.clamp(0, 1) * 255).byte().cpu().numpy().transpose(1, 2, 0)
    return Image.fromarray(arr)


def make_grid(rows, col_labels, row_labels, save_path,
              cell=192, pad=8, header_h=28, label_w=80,
              per_cell_text=None):
    """rows[r][c] = PIL Image (or None). per_cell_text[r][c] = str (optional)."""
    n_rows = len(rows)
    n_cols = len(rows[0])
    text_h = 32 if per_cell_text is not None else 0
    cell_total_h = cell + text_h
    W = label_w + n_cols * cell + (n_cols + 1) * pad
    H = header_h + n_rows * cell_total_h + (n_rows + 1) * pad
    img = Image.new("RGB", (W, H), (245, 245, 245))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 11)
        font_b = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 11)
    except Exception:
        font = font_b = ImageFont.load_default()
    for ci, name in enumerate(col_labels):
        x = label_w + pad + ci * (cell + pad)
        draw.text((x + 4, 8), name, fill=(0, 0, 0), font=font_b)
    for ri, row in enumerate(rows):
        y = header_h + pad + ri * (cell_total_h + pad)
        draw.text((4, y + cell // 2 - 6), row_labels[ri], fill=(0, 0, 0), font=font)
        for ci, im in enumerate(row):
            x = label_w + pad + ci * (cell + pad)
            if im is not None:
                if im.size != (cell, cell):
                    im = im.resize((cell, cell), Image.BICUBIC)
                img.paste(im, (x, y))
            if per_cell_text is not None and per_cell_text[ri][ci]:
                draw.text((x + 2, y + cell + 2), per_cell_text[ri][ci],
                          fill=(0, 0, 0), font=font)
    img.save(save_path)


# ---------- helpers ----------

def load_model(backbone, pretrained, ckpt_path, device):
    print(f"  loading model: backbone={backbone} ckpt={ckpt_path}")
    m = UrbanJEPA(backbone_name=backbone, pretrained_path=pretrained).to(device)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    res = m.load_checkpoint_state(sd["model"])
    if getattr(res, "missing_keys", None):
        print(f"    missing keys ({len(res.missing_keys)}): "
              f"{res.missing_keys[:3]}...")
    if getattr(res, "unexpected_keys", None):
        print(f"    unexpected keys ({len(res.unexpected_keys)}): "
              f"{res.unexpected_keys[:3]}...")
    m.eval()
    return m, sd


@torch.no_grad()
def model_predict_batch(model, lr, hr, device, amp_dtype):
    lr_d = lr.to(device, non_blocking=True)
    hr_d = hr.to(device, non_blocking=True)
    with torch.amp.autocast("cuda", dtype=amp_dtype,
                            enabled=device.type == "cuda"):
        out = model(lr_d, hr_d)
    return out["pred_image"].float().clamp(0, 1)


def aggregate(per_image_dicts):
    """List of {metric: tensor(B,)} -> dict of {metric: (mean, std)}."""
    out = {}
    for k in METRIC_KEYS:
        cat = torch.cat([d[k] for d in per_image_dicts]).numpy()
        out[k] = {"mean": float(np.mean(cat)), "std": float(np.std(cat)),
                  "n": int(cat.size)}
    return out


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/ortho")
    ap.add_argument("--out_dir", default="experiments/eval_all")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--n_samples", type=int, default=20,
                    help="how many val patches to render as visual strips")
    ap.add_argument("--max_val_batches", type=int, default=None,
                    help="cap for full-val stats (None = all 73 batches)")
    ap.add_argument("--scale", type=int, default=20)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"Device: {device}, amp_dtype: {amp_dtype}")

    # Lazy import LPIPS (loads weights).
    import lpips
    print("Loading LPIPS-Alex + LPIPS-VGG...")
    lpips_alex = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    lpips_vgg = lpips.LPIPS(net="vgg", verbose=False).to(device).eval()
    for p in lpips_alex.parameters():
        p.requires_grad = False
    for p in lpips_vgg.parameters():
        p.requires_grad = False

    # Deterministic val dataset.
    val_ds = OrthoDataset(args.data_dir, split="val", augment=False, seed=42)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=device.type == "cuda")
    n_total = len(val_ds)
    print(f"Val: {n_total} samples in {len(val_loader)} batches")

    # Pre-collect the 20 visual sample tensors (HR + LR).
    print(f"Collecting first {args.n_samples} val samples for visuals...")
    sample_hr = []
    sample_lr = []
    sample_tiles = []
    sample_crops = []
    for i in range(args.n_samples):
        s = val_ds[i]
        sample_hr.append(s["high_res"])
        sample_lr.append(s["low_res"])
        sample_tiles.append(Path(s["tile_path"]).stem)
        sample_crops.append((s["crop_row"], s["crop_col"]))
    sample_hr = torch.stack(sample_hr).to(device)
    sample_lr = torch.stack(sample_lr).to(device)

    # Will hold the 20-sample predictions per method, plus per-image metrics.
    sample_preds = {}        # method -> tensor (n_samples, 3, 256, 256)
    sample_metrics = {}      # method -> per_image_metrics dict
    full_val_agg = {}        # method -> aggregate
    method_meta = {}         # method -> {ckpt_epoch, ckpt_step}

    # ---- classical baselines (run on 20-sample tensor and full val) ----
    print("\n=== classical baselines ===")
    # Per-sample: re-degrade HR -> small -> upsamplers.
    sample_small = degrade_to_small(sample_hr, args.scale)
    for name in CLASSICAL_NAMES:
        pred = CLASSICAL_FNS[name](sample_small)
        sample_preds[name] = pred
        sample_metrics[name] = per_image_metrics(pred, sample_hr,
                                                 lpips_alex, lpips_vgg)
        print(f"  [{name}] sample mean PSNR={sample_metrics[name]['psnr'].mean():.2f} dB")

    # Full-val classical metrics: stream batches.
    print("  full-val pass for classical baselines...")
    cls_per_batch = {n: [] for n in CLASSICAL_NAMES}
    t0 = time.time()
    for bi, batch in enumerate(val_loader):
        if args.max_val_batches is not None and bi >= args.max_val_batches:
            break
        hr = batch["high_res"].to(device, non_blocking=True)
        small = degrade_to_small(hr, args.scale)
        for name in CLASSICAL_NAMES:
            up = CLASSICAL_FNS[name](small).to(device)
            cls_per_batch[name].append(
                per_image_metrics(up, hr, lpips_alex, lpips_vgg))
        if bi % 10 == 0:
            print(f"    batch {bi}/{len(val_loader)} ({time.time()-t0:.0f}s)")
    for name in CLASSICAL_NAMES:
        full_val_agg[name] = aggregate(cls_per_batch[name])
        print(f"  [{name}] full-val PSNR={full_val_agg[name]['psnr']['mean']:.2f} dB "
              f"LPIPS-A={full_val_agg[name]['lpips_alex']['mean']:.4f}")

    # ---- trained models ----
    for label, backbone, pretrained, ckpt_path in MODEL_VARIANTS:
        if not Path(ckpt_path).exists():
            print(f"\n[skip] {label}: {ckpt_path} not found")
            continue
        print(f"\n=== model: {label} ===")
        model, sd = load_model(backbone, pretrained, ckpt_path, device)
        method_meta[label] = {"ckpt_epoch": sd.get("epoch"),
                              "ckpt_step": sd.get("global_step"),
                              "ckpt_path": str(ckpt_path)}

        # 20 samples in one batch (gpu has plenty of room).
        try:
            pred_samples = model_predict_batch(model, sample_lr, sample_hr,
                                               device, amp_dtype)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                pred_samples = []
                for i in range(0, sample_lr.shape[0], 4):
                    p = model_predict_batch(model,
                                            sample_lr[i:i + 4], sample_hr[i:i + 4],
                                            device, amp_dtype)
                    pred_samples.append(p)
                pred_samples = torch.cat(pred_samples)
            else:
                raise
        sample_preds[label] = pred_samples
        sample_metrics[label] = per_image_metrics(pred_samples, sample_hr,
                                                  lpips_alex, lpips_vgg)
        print(f"  sample mean PSNR={sample_metrics[label]['psnr'].mean():.2f} dB "
              f"LPIPS-A={sample_metrics[label]['lpips_alex'].mean():.4f}")

        # Full val stats.
        print(f"  full-val pass ({len(val_loader)} batches)...")
        per_batch = []
        t0 = time.time()
        for bi, batch in enumerate(val_loader):
            if args.max_val_batches is not None and bi >= args.max_val_batches:
                break
            pred = model_predict_batch(model, batch["low_res"], batch["high_res"],
                                       device, amp_dtype)
            per_batch.append(per_image_metrics(pred,
                                               batch["high_res"].to(device),
                                               lpips_alex, lpips_vgg))
            if bi % 10 == 0:
                print(f"    batch {bi}/{len(val_loader)} ({time.time()-t0:.0f}s)")
        full_val_agg[label] = aggregate(per_batch)
        print(f"  [{label}] full-val PSNR={full_val_agg[label]['psnr']['mean']:.2f} dB "
              f"LPIPS-A={full_val_agg[label]['lpips_alex']['mean']:.4f}")

        del model
        torch.cuda.empty_cache()

    # ---- oracle picker: per-image best model variant ----
    print("\n=== oracle picker (per-image best PSNR of 6 trained variants) ===")
    trained_labels = [v[0] for v in MODEL_VARIANTS if v[0] in sample_preds]
    if trained_labels:
        psnr_stack = torch.stack([sample_metrics[l]["psnr"] for l in trained_labels])
        best_idx = psnr_stack.argmax(dim=0)  # (n_samples,)
        oracle_pred = torch.stack(
            [sample_preds[trained_labels[best_idx[i].item()]][i]
             for i in range(sample_hr.shape[0])])
        oracle_meta = [trained_labels[best_idx[i].item()] for i in range(sample_hr.shape[0])]
        sample_preds["oracle"] = oracle_pred
        sample_metrics["oracle"] = per_image_metrics(oracle_pred, sample_hr,
                                                     lpips_alex, lpips_vgg)
        print(f"  oracle sample PSNR={sample_metrics['oracle']['psnr'].mean():.2f} dB")
    else:
        oracle_meta = []

    # Full-val oracle: requires keeping all model preds in memory or replaying
    # passes. Skip the full-val oracle (would 6x memory). Per-sample oracle is
    # enough to show the upper bound.

    # ---- write CSVs ----
    print("\nWriting CSVs / JSON...")

    # Full-val aggregate stats
    full_csv = out_dir / "full_val_stats.csv"
    with full_csv.open("w", newline="") as f:
        w = csv.writer(f)
        header = ["method"] + [f"{k}_mean" for k in METRIC_KEYS] \
                 + [f"{k}_std" for k in METRIC_KEYS] + ["n"]
        w.writerow(header)
        for method, agg in full_val_agg.items():
            row = [method]
            for k in METRIC_KEYS:
                row.append(f"{agg[k]['mean']:.6f}")
            for k in METRIC_KEYS:
                row.append(f"{agg[k]['std']:.6f}")
            row.append(agg[METRIC_KEYS[0]]["n"])
            w.writerow(row)

    # Per-sample stats over the 20 visuals
    persamp_csv = out_dir / "per_sample_stats.csv"
    sample_methods = list(sample_metrics.keys())
    with persamp_csv.open("w", newline="") as f:
        w = csv.writer(f)
        header = ["sample_idx", "tile", "crop_row", "crop_col", "method"] + METRIC_KEYS
        w.writerow(header)
        for i in range(sample_hr.shape[0]):
            for method in sample_methods:
                row = [i, sample_tiles[i], sample_crops[i][0], sample_crops[i][1], method]
                for k in METRIC_KEYS:
                    row.append(f"{sample_metrics[method][k][i].item():.6f}")
                w.writerow(row)

    # JSON dump
    json_out = {
        "full_val_aggregate": full_val_agg,
        "sample_per_image": {
            method: {k: sample_metrics[method][k].tolist() for k in METRIC_KEYS}
            for method in sample_metrics
        },
        "sample_oracle_winner": oracle_meta,
        "sample_tiles": sample_tiles,
        "sample_crops": sample_crops,
        "model_meta": method_meta,
        "config": vars(args),
    }
    (out_dir / "eval_all_results.json").write_text(json.dumps(json_out, indent=2,
                                                              default=str))
    print(f"  wrote {full_csv}")
    print(f"  wrote {persamp_csv}")
    print(f"  wrote {out_dir / 'eval_all_results.json'}")

    # ---- visual grids ----
    print("Rendering visual grids...")

    def fmt_cell(method, idx):
        if method not in sample_metrics:
            return ""
        m = sample_metrics[method]
        return (f"PSNR {m['psnr'][idx].item():5.2f}\n"
                f"LPIPS {m['lpips_alex'][idx].item():.3f}")

    def pil_row(methods, idx):
        out = []
        for method in methods:
            if method == "HR":
                out.append(to_pil(sample_hr[idx]))
            elif method == "LR":
                out.append(to_pil(sample_lr[idx]))
            elif method in sample_preds:
                out.append(to_pil(sample_preds[method][idx]))
            else:
                out.append(None)
        return out

    def text_row(methods, idx):
        return [("HR" if m == "HR" else
                 "LR-bilin" if m == "LR" else
                 fmt_cell(m, idx)) for m in methods]

    # Big composite (12 cols)
    big_cols = ["HR", "LR", "nearest", "bicubic", "lanczos",
                "imagenet-best", "dinov2-best", "explora-best",
                "imagenet-epoch_2", "dinov2-epoch_2", "explora-epoch_2", "oracle"]
    big_rows = []
    big_text = []
    big_labels = []
    for i in range(sample_hr.shape[0]):
        big_rows.append(pil_row(big_cols, i))
        big_text.append(text_row(big_cols, i))
        winner = oracle_meta[i] if oracle_meta else "?"
        big_labels.append(f"#{i}\n{sample_tiles[i][:6]}\nwin:{winner.split('-')[0]}")
    make_grid(big_rows, big_cols, big_labels,
              out_dir / "grid_full.png", per_cell_text=big_text)
    print(f"  wrote {out_dir / 'grid_full.png'}")

    # Classical-only grid
    cls_cols = ["HR", "LR", "nearest", "bilinear", "bicubic", "lanczos"]
    cls_rows = [pil_row(cls_cols, i) for i in range(sample_hr.shape[0])]
    cls_text = [text_row(cls_cols, i) for i in range(sample_hr.shape[0])]
    cls_labels = [f"#{i}" for i in range(sample_hr.shape[0])]
    make_grid(cls_rows, cls_cols, cls_labels,
              out_dir / "grid_classical.png", per_cell_text=cls_text)
    print(f"  wrote {out_dir / 'grid_classical.png'}")

    # Trained-best grid
    tb_cols = ["HR", "LR", "imagenet-best", "dinov2-best", "explora-best", "oracle"]
    tb_rows = [pil_row(tb_cols, i) for i in range(sample_hr.shape[0])]
    tb_text = [text_row(tb_cols, i) for i in range(sample_hr.shape[0])]
    make_grid(tb_rows, tb_cols, cls_labels,
              out_dir / "grid_trained_best.png", per_cell_text=tb_text)
    print(f"  wrote {out_dir / 'grid_trained_best.png'}")

    # Trained-last grid
    tl_cols = ["HR", "LR", "imagenet-epoch_2", "dinov2-epoch_2", "explora-epoch_2", "oracle"]
    tl_rows = [pil_row(tl_cols, i) for i in range(sample_hr.shape[0])]
    tl_text = [text_row(tl_cols, i) for i in range(sample_hr.shape[0])]
    make_grid(tl_rows, tl_cols, cls_labels,
              out_dir / "grid_trained_last.png", per_cell_text=tl_text)
    print(f"  wrote {out_dir / 'grid_trained_last.png'}")

    # ---- summary table to stdout ----
    print("\n" + "=" * 90)
    print(f"{'method':<25} {'PSNR':>8} {'SSIM':>7} {'MS-SSIM':>8} "
          f"{'L1':>7} {'RMSE':>7} {'LPIPS-A':>9} {'LPIPS-V':>9}")
    print("-" * 90)
    for method, agg in full_val_agg.items():
        print(f"{method:<25} "
              f"{agg['psnr']['mean']:>6.2f}dB "
              f"{agg['ssim']['mean']:>7.4f} "
              f"{agg['msssim']['mean']:>8.4f} "
              f"{agg['l1']['mean']:>7.4f} "
              f"{agg['rmse']['mean']:>7.4f} "
              f"{agg['lpips_alex']['mean']:>9.4f} "
              f"{agg['lpips_vgg']['mean']:>9.4f}")
    print("=" * 90)
    print(f"\nAll outputs in: {out_dir}")


if __name__ == "__main__":
    main()
