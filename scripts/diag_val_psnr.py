"""
Diagnostic: load a step checkpoint, run a few val batches in fp16, bf16, and
fp32. Compare PSNR against the bilinear baseline (just low_res). The goal is
to figure out whether the 12 dB val PSNR seen in training is a real model
issue or a measurement artifact (fp16 overflow when running bf16-trained
weights through the default fp16 autocast).
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.ortho_dataset import OrthoDataset
from src.evaluation.metrics import psnr as psnr_metric
from src.models.urbanjepa import UrbanJEPA


def run_val(model, loader, device, dtype, n_batches: int):
    model.eval()
    psnr_pred = 0.0
    psnr_bilinear = 0.0
    l1_pred = 0.0
    n = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
            low = batch["low_res"].to(device)
            hr = batch["high_res"].to(device)
            if dtype == torch.float32:
                result = model(low, hr)
            else:
                with torch.amp.autocast("cuda", dtype=dtype):
                    result = model(low, hr)
            pred = result["pred_image"].float().clamp(0, 1)
            psnr_pred += psnr_metric(pred, hr.float()).item()
            psnr_bilinear += psnr_metric(low.float().clamp(0, 1), hr.float()).item()
            l1_pred += F.l1_loss(pred, hr.float()).item()
            n += 1
    return {
        "psnr_pred": psnr_pred / max(1, n),
        "psnr_bilinear": psnr_bilinear / max(1, n),
        "l1_pred": l1_pred / max(1, n),
        "batches": n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--backbone", default="imagenet_vitb16")
    ap.add_argument("--pretrained", default="models/pretrained/imagenet_vitb16.pt")
    ap.add_argument("--data_dir", default="data/ortho")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--n_batches", type=int, default=8)
    args = ap.parse_args()

    device = torch.device("cuda")
    model = UrbanJEPA(backbone_name=args.backbone,
                      pretrained_path=args.pretrained).to(device)
    sd = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_checkpoint_state(sd["model"])
    print(f"Loaded {args.checkpoint} — step {sd.get('global_step')}, "
          f"epoch {sd.get('epoch')}, best_val={sd.get('best_val_psnr')}")

    val_ds = OrthoDataset(args.data_dir, split="val", augment=False, seed=42)
    loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)

    print(f"\n{'dtype':<8} {'pred PSNR':>12} {'bilinear':>12} {'pred L1':>10}")
    for label, dtype in [("fp16", torch.float16),
                         ("bf16", torch.bfloat16),
                         ("fp32", torch.float32)]:
        r = run_val(model, loader, device, dtype, args.n_batches)
        print(f"{label:<8} {r['psnr_pred']:>10.2f} dB "
              f"{r['psnr_bilinear']:>10.2f} dB "
              f"{r['l1_pred']:>10.4f}")


if __name__ == "__main__":
    main()
