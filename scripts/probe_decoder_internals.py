"""Probe the decoder's internal activations across checkpoints.

For each checkpoint passed, runs N val batches and captures:
  - raw_residual (pre-tanh output) stats: mean, std, abs_max
  - final_conv output magnitudes
  - whether tanh is saturating (|residual| > 2 means tanh > 0.96)
  - fraction of output pixels at tanh saturation

Goal: confirm whether the decoder's residual is RUNAWAY GROWING into tanh
saturation, locking in degenerate outputs.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from src.models.urbanjepa import UrbanJEPA
from src.data.ortho_dataset import OrthoDataset


def probe(ckpt_path: str, n_batches: int = 5):
    print(f"\n=== {ckpt_path} ===", flush=True)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    step = ck.get("global_step", -1)
    print(f"step = {step}", flush=True)

    dev = torch.device("cuda")
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

    val_ds = OrthoDataset(
        "data/ortho", split="val", augment=False,
        val_patches_per_tile=cfg.get("val_patches_per_tile", 4),
        seed=cfg.get("seed", 42),
        tile_manifest=cfg.get("tile_manifest"),
        match_train_aug_in_val=cfg.get("match_train_aug_in_val", False),
    )
    loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)

    # Hook the final_conv output (raw residual before tanh)
    raw_residuals = []
    def hook(_module, _inp, out):
        raw_residuals.append(out.detach().float())

    # PixelDecoderWithSkips: final_conv is the last conv before tanh*0.5 + LR.
    if not hasattr(model.decoder, "final_conv"):
        print("  ERROR: model.decoder has no final_conv attribute")
        return
    handle = model.decoder.final_conv.register_forward_hook(hook)

    out_stats = []
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if bi >= n_batches:
                break
            lr = batch["low_res"].to(dev)
            hr = batch["high_res"].to(dev)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(lr, hr)
            pred = out["pred_image"].float()
            l1 = (pred.clamp(0, 1) - hr).abs().mean().item()
            out_stats.append(l1)
    handle.remove()

    if not raw_residuals:
        print("  no residuals captured")
        return

    all_r = torch.cat(raw_residuals, dim=0)   # (N, 3, 256, 256)
    abs_r = all_r.abs()
    print(f"  raw_residual (= final_conv output, pre-tanh): "
          f"shape={tuple(all_r.shape)}", flush=True)
    print(f"    mean={all_r.mean():.5f}  std={all_r.std():.5f}  "
          f"abs_mean={abs_r.mean():.5f}  abs_max={abs_r.max():.5f}",
          flush=True)
    # tanh saturates: tanh(2) = 0.964, tanh(3) = 0.995
    sat2 = (abs_r > 2.0).float().mean().item()
    sat3 = (abs_r > 3.0).float().mean().item()
    sat5 = (abs_r > 5.0).float().mean().item()
    print(f"    frac |r|>2.0 (tanh>0.964): {sat2:.4%}", flush=True)
    print(f"    frac |r|>3.0 (tanh>0.995): {sat3:.4%}", flush=True)
    print(f"    frac |r|>5.0 (tanh>0.9999): {sat5:.4%}", flush=True)
    # After tanh*0.5: bounded value
    bounded = torch.tanh(all_r) * 0.5
    print(f"    after tanh*0.5: abs_mean={bounded.abs().mean():.5f}  "
          f"abs_max={bounded.abs().max():.5f}", flush=True)
    print(f"  val L1 (over {n_batches} batches): mean={sum(out_stats)/len(out_stats):.4f}",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--n_batches", type=int, default=5)
    args = ap.parse_args()
    for c in args.ckpts:
        probe(c, args.n_batches)


if __name__ == "__main__":
    main()
