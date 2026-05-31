"""
Side-by-side image grid: HR ground truth vs LR vs classical upsamplers vs the
"oracle" (decoder fed with HR-encoded features → upper bound for the current
v3 decoder architecture).

Runs on CPU so it does not contend with the GPU training process.

Oracle = encoder(HR) → context features → decoder(low_res, ctx_HR, ctx_HR, multi_HR).
Skips the predictor entirely (feeds ctx_final twice) so the decoder sees the
best possible feature inputs. This isolates "how good is the decoder when its
features are perfect?" — the structural ceiling for *this architecture* on
*this task*.

Usage:
    PYTHONPATH=. python scripts/sample_baselines.py \
        --checkpoint checkpoints/backbone_imagenet/step_slot_X.pt \
        --n_patches 6 --out_dir runs/baseline_samples
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.ortho_dataset import OrthoDataset
from src.models.urbanjepa import UrbanJEPA


# --------------- degradation (matches eval_baselines.py v3_realistic mode) ---------------

def _gauss_kernel(sigma: float, ksize: int = 9):
    x = torch.arange(ksize, dtype=torch.float32) - (ksize - 1) / 2
    k = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    return (k / k.sum())


def _gauss_blur(t, sigma):
    k1 = _gauss_kernel(sigma)
    k = (k1[:, None] * k1[None, :])[None, None]
    pad = k.shape[-1] // 2
    out = []
    for c in range(t.shape[1]):
        x = F.pad(t[:, c:c + 1], [pad] * 4, mode="reflect")
        out.append(F.conv2d(x, k))
    return torch.cat(out, dim=1)


def degrade(hr, scale, patch=256):
    blurred = _gauss_blur(hr, sigma=1.0)
    target = max(1, int(round(patch / scale)))
    small = F.interpolate(blurred, size=(target, target), mode="area")
    return small


def up_bilinear(small, p=256):
    return F.interpolate(small, size=(p, p), mode="bilinear", align_corners=False)


def up_bicubic(small, p=256):
    return F.interpolate(small, size=(p, p), mode="bicubic", align_corners=False).clamp(0, 1)


def up_lanczos(small, p=256):
    arr = (small.clamp(0, 1) * 255).byte().cpu().numpy()
    out = np.empty((arr.shape[0], arr.shape[1], p, p), dtype=np.uint8)
    for b in range(arr.shape[0]):
        for c in range(arr.shape[1]):
            img = Image.fromarray(arr[b, c])
            out[b, c] = np.asarray(img.resize((p, p), Image.LANCZOS))
    return torch.from_numpy(out).float() / 255.0


# --------------- metrics ---------------

def psnr(pred, target):
    mse = F.mse_loss(pred.clamp(0, 1), target.clamp(0, 1))
    return float("inf") if mse.item() < 1e-12 else 10 * torch.log10(1.0 / mse).item()


# --------------- oracle ---------------

@torch.no_grad()
def oracle_predict(model, low_res, hr):
    """Decoder fed with HR-encoded features. Upper bound for this architecture."""
    ctx_hr, multi_hr = model.context_encoder(hr)
    # pred_features = ctx_hr — best case, skips the predictor entirely
    pred = model.decoder(low_res, ctx_hr, ctx_hr, multi_hr)
    return pred.clamp(0, 1)


@torch.no_grad()
def trained_predict(model, low_res, hr):
    """What the model would actually do at this checkpoint (LR-only inputs)."""
    result = model(low_res, hr)
    return result["pred_image"].clamp(0, 1)


# --------------- grid builder ---------------

def to_pil(t):
    arr = (t.clamp(0, 1) * 255).byte().cpu().numpy().transpose(1, 2, 0)
    return Image.fromarray(arr)


def make_grid(rows, col_labels, row_labels, save_path, cell_size=256, pad=10,
              label_h=20):
    cols = len(rows[0])
    W = cols * cell_size + (cols + 1) * pad + 80  # +80 for row labels
    H = len(rows) * cell_size + (len(rows) + 1) * pad + label_h
    grid = Image.new("RGB", (W, H), (240, 240, 240))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(grid)
    # column headers
    for ci, name in enumerate(col_labels):
        x = 80 + pad + ci * (cell_size + pad)
        draw.text((x + 5, 2), name, fill=(0, 0, 0))
    for ri, row in enumerate(rows):
        y = label_h + pad + ri * (cell_size + pad)
        draw.text((5, y + cell_size // 2), row_labels[ri], fill=(0, 0, 0))
        for ci, img in enumerate(row):
            x = 80 + pad + ci * (cell_size + pad)
            grid.paste(img, (x, y))
    grid.save(save_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--backbone", default="imagenet_vitb16")
    ap.add_argument("--pretrained", default="models/pretrained/imagenet_vitb16.pt")
    ap.add_argument("--data_dir", default="data/ortho")
    ap.add_argument("--n_patches", type=int, default=6)
    ap.add_argument("--scale", type=int, default=20)
    ap.add_argument("--out_dir", default="runs/baseline_samples")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading val dataset...")
    ds = OrthoDataset(args.data_dir, split="val", augment=False, seed=42)
    n = min(args.n_patches, len(ds))
    hr = torch.stack([ds[i]["high_res"] for i in range(n)], dim=0)

    print("Loading model on CPU (training still owns the GPU)...")
    device = torch.device("cpu")
    model = UrbanJEPA(backbone_name=args.backbone,
                      pretrained_path=args.pretrained).to(device)
    sd = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_checkpoint_state(sd["model"])
    model.eval()
    print(f"  loaded {args.checkpoint} (step={sd.get('global_step')})")

    # Re-degrade HR exactly the same way v3 val does
    small = degrade(hr, args.scale)
    lr_bilinear = up_bilinear(small)
    lr_bicubic = up_bicubic(small)
    lr_lanczos = up_lanczos(small)

    print("Running oracle (HR → encoder → decoder) ...")
    oracle = oracle_predict(model, lr_bilinear, hr)
    print("Running actual trained model (LR → encoder → predictor → decoder) ...")
    trained = trained_predict(model, lr_bilinear, hr)

    # Print per-patch metrics + averages
    print(f"\n{'patch':<8} {'bilinear':>10} {'bicubic':>10} {'lanczos':>10} "
          f"{'trained':>10} {'oracle':>10}")
    aggregates = {k: 0.0 for k in ("bilinear", "bicubic", "lanczos", "trained", "oracle")}
    for i in range(n):
        p_bil = psnr(lr_bilinear[i:i+1], hr[i:i+1])
        p_bic = psnr(lr_bicubic[i:i+1], hr[i:i+1])
        p_lan = psnr(lr_lanczos[i:i+1], hr[i:i+1])
        p_tra = psnr(trained[i:i+1], hr[i:i+1])
        p_ora = psnr(oracle[i:i+1], hr[i:i+1])
        aggregates["bilinear"] += p_bil
        aggregates["bicubic"] += p_bic
        aggregates["lanczos"] += p_lan
        aggregates["trained"] += p_tra
        aggregates["oracle"] += p_ora
        print(f"{i:<8} {p_bil:>8.2f}dB {p_bic:>8.2f}dB {p_lan:>8.2f}dB "
              f"{p_tra:>8.2f}dB {p_ora:>8.2f}dB")
    print("-" * 70)
    print(f"{'mean':<8} " + " ".join(f"{aggregates[k]/n:>8.2f}dB"
                                       for k in ("bilinear", "bicubic", "lanczos",
                                                 "trained", "oracle")))

    rows = []
    row_labels = []
    for i in range(n):
        rows.append([to_pil(hr[i]),
                     to_pil(lr_bilinear[i]),
                     to_pil(lr_bicubic[i]),
                     to_pil(lr_lanczos[i]),
                     to_pil(trained[i]),
                     to_pil(oracle[i])])
        row_labels.append(f"#{i}")
    grid_path = out_dir / f"grid_scale{args.scale}_n{n}.png"
    make_grid(rows, col_labels=["HR (truth)", "bilinear", "bicubic", "lanczos",
                                  "trained", "oracle"],
              row_labels=row_labels, save_path=str(grid_path))
    print(f"\nWrote grid: {grid_path}")


if __name__ == "__main__":
    main()
