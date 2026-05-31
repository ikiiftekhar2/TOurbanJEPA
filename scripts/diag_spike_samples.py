"""Find which val samples blow up under a pre_nan checkpoint.

Loads the latest pre_nan_step_*.pt, runs forward on the deterministic val set,
records per-sample L1, and writes:
  - diag_spike_samples_l1.csv  (all samples, sortable in a spreadsheet)
  - diag_spike_top_<N>.png     (worst N triptychs: LR_up / pred / HR + L1)
  - diag_spike_median.png      (5 samples near the median for control)

Run:
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    /home/ubuntu/urbanjepa-venv/bin/python scripts/diag_spike_samples.py \
      --ckpt checkpoints/v4_p1_5_stageA/pre_nan_step_22441.pt --top 20
"""
import argparse, csv, glob, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image

from src.models.urbanjepa import UrbanJEPA
from src.data.ortho_dataset import OrthoDataset


def _latest_pre_nan():
    pats = sorted(glob.glob("checkpoints/v4_p1_5_stageA/pre_nan_step_*.pt"),
                  key=os.path.getmtime, reverse=True)
    return pats[0] if pats else None


def _save_triptych(path, lr_up, pred, hr, title):
    """Stack LR (bilinear-up) | pred | HR side-by-side as one PNG.

    Each tensor: (3,H,W) float in [0,1]. HR is the target."""
    imgs = [lr_up, pred.clamp(0, 1), hr]
    row = torch.cat([t.cpu() for t in imgs], dim=2)  # (3,H,3W)
    arr = (row.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    im = Image.fromarray(arr)
    # Crude caption: bar at top with text-via-numpy-blit would need PIL.ImageDraw
    from PIL import ImageDraw, ImageFont
    canvas = Image.new("RGB", (im.width, im.height + 28), (0, 0, 0))
    canvas.paste(im, (0, 28))
    d = ImageDraw.Draw(canvas)
    d.text((4, 6), title, fill=(255, 255, 0))
    canvas.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--data_dir", default="data/ortho")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--out_dir", default="experiments/v4_phase1_5/diag_spike")
    args = ap.parse_args()

    ckpt = args.ckpt or _latest_pre_nan()
    if ckpt is None:
        raise SystemExit("no pre_nan_*.pt found")
    os.makedirs(args.out_dir, exist_ok=True)
    dev = torch.device("cuda")
    print(f"[diag] ckpt={ckpt}", flush=True)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    gstep = ck.get("global_step", -1)
    print(f"[diag] global_step={gstep} epoch={ck.get('epoch', -1)} "
          f"batch={ck.get('batch_idx', -1)}", flush=True)

    model = UrbanJEPA(
        backbone_name=cfg["backbone"], pretrained_path=cfg["pretrained_path"],
        predictor_depth=cfg["predictor_depth"],
        decoder_attn_blocks=cfg["decoder_attn_blocks"],
        decoder_base_dim=cfg["decoder_base_dim"], dropout=cfg["dropout"],
        use_v4_predictor=cfg.get("use_v4_predictor", False),
        use_v4_decoder=cfg.get("use_v4_decoder", False),
        use_v5_decoder=cfg.get("use_v5_decoder", False),
        hierarchical_jepa=cfg.get("hierarchical_jepa", False),
        use_grad_checkpoint=False,
    ).to(dev)
    model.load_checkpoint_state(ck["model"])
    model.eval()
    print("[diag] model loaded", flush=True)

    val_ds = OrthoDataset(
        args.data_dir, split="val", augment=False,
        val_patches_per_tile=cfg.get("val_patches_per_tile", 4),
        seed=cfg.get("seed", 42), tile_cache_size=cfg.get("tile_cache_size", 32),
        match_train_aug_in_val=cfg.get("match_train_aug_in_val", False),
    )
    loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        drop_last=False, num_workers=0)
    print(f"[diag] val {len(val_ds)} samples, {len(loader)} batches", flush=True)

    rows = []  # (l1, tile_path, crop_row, crop_col, sample_idx)
    saved_examples = []  # (l1, tile_path, lr_up, pred, hr) for later triptych
    sample_idx = 0
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            lr = batch["low_res"].to(dev, non_blocking=True)
            hr = batch["high_res"].to(dev, non_blocking=True)
            scales = batch["scale"]  # float tensor
            tile_paths = batch["tile_path"]
            crop_rows = batch["crop_row"]
            crop_cols = batch["crop_col"]

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                out = model(lr, hr)
            pred = out["pred_image"].float().clamp(0, 1)

            # bilinear-up LR for visualization
            lr_up = F.interpolate(lr, size=hr.shape[-2:], mode="bilinear",
                                  align_corners=False)
            per_l1 = (pred - hr).abs().mean(dim=[1, 2, 3])  # (B,)

            for j in range(per_l1.shape[0]):
                l1 = float(per_l1[j].item())
                rows.append((l1, tile_paths[j], int(crop_rows[j]), int(crop_cols[j]),
                             sample_idx + j))
                saved_examples.append((l1, tile_paths[j], int(crop_rows[j]),
                                       int(crop_cols[j]),
                                       lr_up[j].cpu(), pred[j].cpu(), hr[j].cpu()))
            sample_idx += per_l1.shape[0]
            if bi % 25 == 0:
                print(f"[diag] batch {bi}/{len(loader)}  "
                      f"running max L1={max(r[0] for r in rows):.4f}", flush=True)

    rows.sort(key=lambda r: -r[0])  # highest L1 first
    csv_path = os.path.join(args.out_dir, "diag_spike_samples_l1.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["l1", "tile_path", "crop_row", "crop_col", "sample_idx"])
        for r in rows:
            w.writerow(r)
    print(f"[diag] wrote {csv_path}", flush=True)

    # Headline distribution
    l1s = np.array([r[0] for r in rows])
    print(f"\n[diag] val per-sample L1 distribution (N={len(l1s)}):")
    for q in [50, 75, 90, 95, 99, 99.5, 100]:
        print(f"   p{q:<4} = {np.percentile(l1s, q):.4f}")
    print(f"   mean  = {l1s.mean():.4f}")
    over_15 = (l1s > 0.15).sum()
    over_18 = (l1s > 0.18).sum()
    over_19 = (l1s > 0.19).sum()
    print(f"   #L1>0.15 = {over_15}/{len(l1s)} ({100*over_15/len(l1s):.2f}%)")
    print(f"   #L1>0.18 = {over_18}/{len(l1s)} ({100*over_18/len(l1s):.2f}%)")
    print(f"   #L1>0.19 = {over_19}/{len(l1s)} ({100*over_19/len(l1s):.2f}%)")

    # Sort saved_examples by L1 desc; save top-N
    saved_examples.sort(key=lambda r: -r[0])
    top = saved_examples[:args.top]
    print(f"\n[diag] top {len(top)} worst samples:")
    for i, (l1, tp, cr, cc, lr_up, pred, hr) in enumerate(top):
        title = f"#{i:02d} L1={l1:.4f} | {os.path.basename(tp)} crop=({cr},{cc})"
        out = os.path.join(args.out_dir, f"top_{i:02d}_l1_{l1:.4f}.png")
        _save_triptych(out, lr_up, pred, hr, title)
        print(f"   {title} -> {out}")

    # Median controls
    n = len(saved_examples)
    if n >= 5:
        mid_idx = n // 2
        median_band = saved_examples[mid_idx - 2 : mid_idx + 3]
        for i, (l1, tp, cr, cc, lr_up, pred, hr) in enumerate(median_band):
            title = f"MED#{i} L1={l1:.4f} | {os.path.basename(tp)} crop=({cr},{cc})"
            out = os.path.join(args.out_dir, f"median_{i}_l1_{l1:.4f}.png")
            _save_triptych(out, lr_up, pred, hr, title)
            print(f"   {title} -> {out}")

    # Tile-level aggregate: per-tile max L1
    from collections import defaultdict
    per_tile = defaultdict(list)
    for r in rows:
        per_tile[r[1]].append(r[0])
    tile_max = [(max(v), len(v), tp) for tp, v in per_tile.items()]
    tile_max.sort(key=lambda x: -x[0])
    print(f"\n[diag] top 15 tiles by max per-sample L1:")
    for mx, n, tp in tile_max[:15]:
        print(f"   max_L1={mx:.4f}  ({n} samples)  {os.path.basename(tp)}")


if __name__ == "__main__":
    main()
