"""Standalone val-set evaluation of a checkpoint.

Safe to run alongside live training: small batch + no_grad + expandable_segments
keep the GPU footprint small. Reconstructs the exact model from the checkpoint's
saved config so the numbers match training's own validation. Prints
PSNR/L1/SSIM/LPIPS-VGG and the Stage-A gate verdict.

Usage:
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python scripts/eval_checkpoint.py --ckpt /tmp/eval_best.pt
"""
import argparse
import glob
import os
import sys

# Allow running as a plain script (python scripts/eval_checkpoint.py): put the
# repo root on sys.path so `src` is importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from src.models.urbanjepa import UrbanJEPA
from src.data.ortho_dataset import OrthoDataset
from src.training.train import validate
from src.training.losses import VGGPerceptualLoss


def _pick_ckpt():
    best = glob.glob("checkpoints/v4_p1_5_stageA/best.pt")
    if best:
        return best[0]
    slots = sorted(glob.glob("checkpoints/v4_p1_5_stageA/step_slot_*.pt"),
                   key=os.path.getmtime, reverse=True)
    return slots[0] if slots else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None,
                    help="Checkpoint to eval (default: best.pt or latest step_slot)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=4,
                    help="Small by default so it fits alongside live training.")
    ap.add_argument("--data_dir", default="data/ortho")
    args = ap.parse_args()

    ckpt = args.ckpt or _pick_ckpt()
    if ckpt is None:
        raise SystemExit("no checkpoint found")
    dev = torch.device(args.device)
    print(f"[eval] loading {ckpt}", flush=True)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    gstep = ck.get("global_step", 0)
    print(f"[eval] checkpoint global_step={gstep}", flush=True)

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
    print("[eval] model loaded", flush=True)

    val_ds = OrthoDataset(
        args.data_dir, split="val", augment=False,
        val_patches_per_tile=cfg.get("val_patches_per_tile", 4),
        seed=cfg.get("seed", 42), tile_cache_size=cfg.get("tile_cache_size", 32),
        match_train_aug_in_val=cfg.get("match_train_aug_in_val", False),
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            drop_last=False, num_workers=0)
    vgg = VGGPerceptualLoss().to(dev)

    print(f"[eval] validating {len(val_ds)} samples "
          f"({len(val_loader)} batches @ bs{args.batch_size})...", flush=True)
    with torch.no_grad():
        m = validate(model, val_loader, dev, vgg, global_step=gstep,
                     amp_dtype=torch.bfloat16, autocast_enabled=True)

    print(f"\n=== VAL RESULTS (ckpt step {gstep}) ===", flush=True)
    for k in sorted(m.keys()):
        print(f"  {k:16} = {m[k]:.4f}", flush=True)
    print("\n=== MODEL vs BILINEAR baseline ===", flush=True)
    print(f"  PSNR:      model {m['psnr']:.2f}  vs  bilinear {m.get('bilinear_psnr', float('nan')):.2f} dB  "
          f"(model {'BEATS' if m['psnr'] > m.get('bilinear_psnr', 1e9) else 'WORSE than'} baseline)", flush=True)
    if "bilinear_lpips" in m:
        print(f"  LPIPS:     model {m['lpips']:.4f}  vs  bilinear {m['bilinear_lpips']:.4f}  "
              f"(lower=better; model {'BEATS' if m['lpips'] < m['bilinear_lpips'] else 'WORSE than'} baseline)", flush=True)
    beats_psnr = m["psnr"] > m.get("bilinear_psnr", 1e9) + 0.5
    beats_lpips = m.get("lpips", 1e9) < m.get("bilinear_lpips", -1.0)
    gate = beats_psnr and beats_lpips and m["psnr"] >= 23.0
    print(f"\n  STAGE-A GATE (beat bilinear on BOTH PSNR & LPIPS, psnr>=23): "
          f"{'*** PASS ***' if gate else 'FAIL'}", flush=True)
    print(f"  -> psnr {m['psnr']:.2f} vs bilinear {m.get('bilinear_psnr', float('nan')):.2f} "
          f"| lpips(alex) {m.get('lpips', float('nan')):.4f} vs bilinear "
          f"{m.get('bilinear_lpips', float('nan')):.4f}", flush=True)


if __name__ == "__main__":
    main()
