"""Inspect a v5 checkpoint for NaN in model and EMA state."""
import argparse, sys
import torch


def scan(name, sd):
    bad = []
    for k, v in sd.items():
        if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
            n_nan = torch.isnan(v).sum().item()
            n_inf = torch.isinf(v).sum().item()
            if n_nan or n_inf:
                bad.append((k, n_nan, n_inf, tuple(v.shape)))
    print(f"\n=== {name}: {len(sd)} tensors, {len(bad)} with NaN/Inf ===")
    for k, n, ni, shp in bad[:30]:
        print(f"  {k:60s} shape={shp} nan={n} inf={ni}")
    if len(bad) > 30:
        print(f"  ... and {len(bad)-30} more")
    return bad


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    args = p.parse_args()

    print(f"Loading {args.ckpt}")
    st = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    print(f"Keys: {sorted(st.keys())}")
    print(f"global_step={st.get('global_step')} epoch={st.get('epoch')}")

    bad_m = scan("model", st["model"])
    if "ema" in st and "shadow" in st["ema"]:
        bad_e = scan("ema.shadow", st["ema"]["shadow"])
    else:
        print("\n[!] No EMA state in checkpoint.")
        bad_e = []

    print(f"\n>>> model dirty params: {len(bad_m)}")
    print(f">>> ema   dirty params: {len(bad_e)}")
    sys.exit(0 if not bad_e else 1)


if __name__ == "__main__":
    main()
