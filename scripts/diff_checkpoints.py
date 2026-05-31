"""Forensic weight-diff between two checkpoints to find what BROKE.

For each named parameter, computes:
  - abs delta L2: ||W_b - W_a||
  - rel delta:    ||W_b - W_a|| / ||W_a||
  - mean abs delta per element
  - max abs delta
  - Adam second-moment (exp_avg_sq) max + mean

Outputs ranked tables — top-N parameters by relative change.

Usage:
  python scripts/diff_checkpoints.py --a step_slot_X.pt --b step_slot_Y.pt
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="healthy checkpoint")
    ap.add_argument("--b", required=True, help="crashed checkpoint")
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    print(f"a = {args.a}")
    print(f"b = {args.b}")
    ca = torch.load(args.a, map_location="cpu", weights_only=False)
    cb = torch.load(args.b, map_location="cpu", weights_only=False)
    sa, sb = ca["model"], cb["model"]
    print(f"step_a = {ca.get('global_step')}  step_b = {cb.get('global_step')}")

    common = sorted(set(sa.keys()) & set(sb.keys()))
    print(f"common params: {len(common)}")

    rows = []
    for name in common:
        wa, wb = sa[name], sb[name]
        if wa.shape != wb.shape:
            continue
        if not torch.is_floating_point(wa):
            continue
        delta = (wb - wa).float()
        a_norm = wa.float().norm().item() + 1e-12
        rows.append({
            "name": name,
            "shape": tuple(wa.shape),
            "numel": wa.numel(),
            "a_norm": a_norm,
            "delta_norm": delta.norm().item(),
            "rel_change": delta.norm().item() / a_norm,
            "mean_abs_delta": delta.abs().mean().item(),
            "max_abs_delta": delta.abs().max().item(),
            "a_mean_abs": wa.float().abs().mean().item(),
        })

    # Filter to decoder-only (encoder/predictor were FROZEN so should have 0 change)
    print("\n=== sanity check: encoder/predictor should be unchanged (frozen) ===")
    enc_changed = [r for r in rows if r["name"].startswith(("context_encoder.", "predictor.", "projection_head."))
                   and r["delta_norm"] > 1e-7]
    print(f"frozen-side params with nonzero delta: {len(enc_changed)}/{sum(1 for r in rows if r['name'].startswith(('context_encoder.', 'predictor.', 'projection_head.')))}")
    if enc_changed:
        print("  *** WARNING: frozen side has nonzero delta! ***")
        for r in enc_changed[:5]:
            print(f"    {r['name']} delta_norm={r['delta_norm']:.2e}")

    # Top changes — decoder side
    dec = [r for r in rows if r["name"].startswith("decoder.")]
    print(f"\n=== decoder params: {len(dec)} ===")
    print(f"total decoder weight movement (sum of delta_norm^2): {sum(r['delta_norm']**2 for r in dec):.4e}")
    print(f"total decoder param magnitude (sum of a_norm^2):     {sum(r['a_norm']**2 for r in dec):.4e}")

    # Top by relative change
    dec_sorted = sorted(dec, key=lambda r: -r["rel_change"])
    print(f"\n=== top {args.top} decoder params by RELATIVE change ===")
    print(f"{'name':<70} {'shape':<20} {'rel_chg':>10} {'a_norm':>10} {'mean|d|':>10} {'max|d|':>10}")
    for r in dec_sorted[:args.top]:
        print(f"{r['name']:<70} {str(r['shape']):<20} {r['rel_change']:>10.4f} {r['a_norm']:>10.3e} {r['mean_abs_delta']:>10.3e} {r['max_abs_delta']:>10.3e}")

    # Top by absolute change
    dec_sorted_abs = sorted(dec, key=lambda r: -r["delta_norm"])
    print(f"\n=== top {args.top} decoder params by ABSOLUTE delta norm ===")
    print(f"{'name':<70} {'shape':<20} {'delta_norm':>12} {'a_norm':>10} {'rel_chg':>10}")
    for r in dec_sorted_abs[:args.top]:
        print(f"{r['name']:<70} {str(r['shape']):<20} {r['delta_norm']:>12.4e} {r['a_norm']:>10.3e} {r['rel_change']:>10.4f}")

    # Aggregate by top-level module
    print(f"\n=== decoder change aggregated by top-level submodule ===")
    from collections import defaultdict
    agg = defaultdict(lambda: {"delta_sq": 0.0, "a_sq": 0.0, "n": 0, "max_d": 0.0})
    for r in dec:
        # decoder.X.Y.Z -> "decoder.X"
        parts = r["name"].split(".")
        sub = ".".join(parts[:2])
        agg[sub]["delta_sq"] += r["delta_norm"] ** 2
        agg[sub]["a_sq"] += r["a_norm"] ** 2
        agg[sub]["n"] += 1
        agg[sub]["max_d"] = max(agg[sub]["max_d"], r["max_abs_delta"])
    print(f"{'submodule':<40} {'n_params':>9} {'delta_norm':>12} {'a_norm':>12} {'rel_chg':>10} {'max|d|':>12}")
    for sub in sorted(agg, key=lambda s: -agg[s]["delta_sq"]):
        d = agg[sub]
        rel = (d["delta_sq"] ** 0.5) / (d["a_sq"] ** 0.5 + 1e-12)
        print(f"{sub:<40} {d['n']:>9} {d['delta_sq']**0.5:>12.4e} {d['a_sq']**0.5:>12.4e} {rel:>10.4f} {d['max_d']:>12.3e}")

    # Optimizer state
    if "optimizer" in cb:
        opt = cb["optimizer"]
        print(f"\n=== optimizer (b) state ===")
        # AdamW saves state as dict keyed by param-id ints
        st = opt.get("state", {})
        print(f"  {len(st)} param state entries")
        all_v_max = []
        all_v_mean = []
        for pid, ps in st.items():
            if "exp_avg_sq" in ps:
                v = ps["exp_avg_sq"]
                all_v_max.append(v.max().item())
                all_v_mean.append(v.mean().item())
        if all_v_max:
            print(f"  Adam second-moment (exp_avg_sq) max:  min={min(all_v_max):.3e}  median={sorted(all_v_max)[len(all_v_max)//2]:.3e}  max={max(all_v_max):.3e}")
            print(f"  Adam second-moment mean:              min={min(all_v_mean):.3e}  median={sorted(all_v_mean)[len(all_v_mean)//2]:.3e}  max={max(all_v_mean):.3e}")


if __name__ == "__main__":
    main()
