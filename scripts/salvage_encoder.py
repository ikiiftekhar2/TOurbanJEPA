"""
Salvage encoder/predictor/projection from the collapsed imagenet run.

The v3 imagenet training collapsed at step ~7800 when the unbounded residual
decoder leaked past the hard clamp(0,1) and got stuck in a gradient dead zone.
JEPA/cos_sim metrics show the context_encoder + predictor + target_encoder
were still healthy through the collapse, so we keep those weights and drop
only the decoder + optimizer/scheduler state.

Output: a 'pretrained_path'-style checkpoint that the new training loads as if
it were a fancier initial backbone. Training starts at step 0 with a fresh
decoder (zero-init final_conv, tanh-bounded residual) and the salvaged
encoder/predictor as warm start.
"""

import argparse
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True,
                   help="Path to a step_slot_*.pt from the collapsed run.")
    p.add_argument("--dst", required=True,
                   help="Output path for the salvaged model state dict.")
    p.add_argument("--keep_prefixes", nargs="+",
                   default=["context_encoder.", "predictor.", "projection_head."],
                   help="State-dict key prefixes to retain. Everything else is dropped.")
    args = p.parse_args()

    src_path = Path(args.src)
    dst_path = Path(args.dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {src_path} ...")
    sd = torch.load(src_path, map_location="cpu", weights_only=False)
    model_sd = sd["model"]

    print(f"Source: {len(model_sd)} keys, step={sd.get('global_step')}, epoch={sd.get('epoch')}")

    kept = {k: v for k, v in model_sd.items()
            if any(k.startswith(pref) for pref in args.keep_prefixes)}
    dropped = sorted({k.split('.')[0] for k in model_sd if k not in kept})

    print(f"Kept   prefixes: {args.keep_prefixes}")
    print(f"  → {len(kept)} keys retained")
    print(f"Dropped roots: {dropped}")

    out = {
        "model": kept,
        "salvaged_from": str(src_path),
        "salvaged_step": sd.get("global_step"),
        "salvaged_epoch": sd.get("epoch"),
        "backbone_name": sd.get("backbone_name"),
        "config": sd.get("config"),
    }
    torch.save(out, dst_path)
    size_mb = dst_path.stat().st_size / 1e6
    print(f"Wrote {dst_path} ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
