"""
Find the max batch size that fits in GPU memory for v5 training, both Stage A
(no GAN) and Stage B (+ U-Net discriminator).

Strategy: bisection. For each stage, try increasing batch sizes; mark each
attempt OK or OOM. Report the largest OK batch size and the corresponding
peak memory.

We measure peak via `torch.cuda.max_memory_allocated()` after running a few
forward+backward iterations — this is the true HW peak across the run, not
just the snapshot at exit.

Usage:
  source /home/ubuntu/urbanjepa-venv/bin/activate
  PYTHONPATH=. python scripts/v5_gpu_probe.py --candidates 4 8 12 16 20 24 28 32
"""

import argparse
import gc
import sys

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from src.models.discriminator import UNetDiscriminatorSN
from src.models.v5_model import build_v5_model
from src.training.losses import LPIPSLoss, l1_pixel_loss


def fmt_gib(bytes_: int) -> str:
    return f"{bytes_ / 1024**3:.2f} GiB"


def try_batch(batch_size: int, use_gan: bool, n_iters: int = 4,
              with_lpips: bool = True, freeze_rrdb: bool = False) -> dict:
    """Builds the model fresh, runs n_iters of forward+backward+step.
    Returns {ok, peak_alloc, err}.

    `freeze_rrdb=True` simulates the warmup phase (RRDBNet has requires_grad=False
    so its activations are not retained for backward — much lower peak memory).
    `freeze_rrdb=False` simulates the steady-state post-warmup phase.
    """
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()

    device = torch.device("cuda")
    try:
        model = build_v5_model().to(device).train()
        if freeze_rrdb:
            model.freeze_rrdbnet()
        optimizer = AdamW(
            model.trainable_param_groups(
                jepa_encoder_lr=1e-5, jepa_pred_proj_lr=1e-4,
                injection_lr=1e-4, rrdbnet_lr=1e-5,
            ), weight_decay=1e-4, betas=(0.9, 0.95),
        )
        lpips = LPIPSLoss(net="vgg").to(device) if with_lpips else None

        D = None
        d_optimizer = None
        bce = None
        if use_gan:
            D = UNetDiscriminatorSN(num_in_ch=3, num_feat=64).to(device).train()
            d_optimizer = AdamW(D.parameters(), lr=1e-4, betas=(0.9, 0.99))
            bce = torch.nn.BCEWithLogitsLoss()

        for _ in range(n_iters):
            lr = torch.rand(batch_size, 3, 256, 256, device=device)
            hr = torch.rand(batch_size, 3, 256, 256, device=device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(lr, hr)
                sr = out["sr"].clamp(0, 1)
                loss = l1_pixel_loss(sr, hr)
                if lpips is not None:
                    loss = loss + 0.1 * lpips(sr, hr)
                if use_gan:
                    d_real = D(hr)
                    d_fake_d = D(sr.detach())
                    l_d = 0.5 * (bce(d_real, torch.ones_like(d_real))
                                 + bce(d_fake_d, torch.zeros_like(d_fake_d)))
                    d_optimizer.zero_grad(set_to_none=True)
                    l_d.backward()
                    d_optimizer.step()

                    d_fake_g = D(sr)
                    l_g_adv = bce(d_fake_g, torch.ones_like(d_fake_g))
                    loss = loss + 0.1 * l_g_adv
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        return {"ok": True, "peak_alloc": peak, "err": None}
    except RuntimeError as e:
        return {"ok": False, "peak_alloc": torch.cuda.max_memory_allocated(),
                "err": str(e)[:200]}
    finally:
        try:
            del model, optimizer, lpips, D, d_optimizer
        except Exception:
            pass
        torch.cuda.empty_cache()
        gc.collect()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--candidates", type=int, nargs="+",
                   default=[4, 6, 8, 10, 12, 14, 16, 20, 24])
    p.add_argument("--n_iters", type=int, default=4)
    p.add_argument("--stages", type=str, nargs="+", default=["A", "B"])
    p.add_argument("--phases", type=str, nargs="+", default=["warmup", "steady"],
                   help="warmup = RRDB frozen, steady = RRDB trainable")
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available — aborting.")
        sys.exit(1)
    total = torch.cuda.get_device_properties(0).total_memory
    print(f"[gpu] {torch.cuda.get_device_name(0)}, {fmt_gib(total)} total")
    print(f"[probe] iters per attempt = {args.n_iters}")

    for stage in args.stages:
        use_gan = (stage == "B")
        for phase in args.phases:
            freeze = (phase == "warmup")
            print(f"\n========= Stage {stage} | phase={phase} "
                  f"(use_gan={use_gan}, freeze_rrdb={freeze}) =========")
            best_ok = None
            for bs in args.candidates:
                r = try_batch(bs, use_gan=use_gan, n_iters=args.n_iters,
                              freeze_rrdb=freeze)
                tag = "OK " if r["ok"] else "OOM"
                print(f"  batch={bs:>3}  {tag}  peak={fmt_gib(r['peak_alloc'])}"
                      + (f"   ({r['err'][:80]})" if not r["ok"] else ""))
                if r["ok"]:
                    best_ok = (bs, r["peak_alloc"])
                else:
                    break
            if best_ok:
                print(f"  >>> Stage {stage} {phase} max batch = {best_ok[0]}  "
                      f"(peak {fmt_gib(best_ok[1])})")
            else:
                print(f"  >>> Stage {stage} {phase} FAILED at every probe batch size")


if __name__ == "__main__":
    main()
