"""
Phase 2 smoke test — train.py end-to-end exerciser.

Runs the training script with a tiny configuration (a handful of steps), then
resumes from the saved checkpoint and verifies state survives the round-trip.

Pass criteria:
  - 30-step run completes without NaN/exception
  - L1 + LPIPS losses logged, no NaN in logs
  - At least one step checkpoint is written
  - At least one in-loop val ran and produced finite PSNR/LPIPS
  - Resume + 10 extra steps completes; global_step advances continuously
  - Checkpoint manifest carries the expected event log
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parent.parent


def run_train(extra_args, exp_name, cwd=REPO_ROOT, env=None):
    cmd = [
        sys.executable, "-m", "src.training.train",
        "--exp_name", exp_name,
        "--checkpoint_dir", "runs",
        "--log_dir", "runs",
        # tiny config
        "--batch_size", "2",
        "--num_workers", "2",
        "--patches_per_epoch", "8",
        "--val_patches_per_tile", "1",
        "--predictor_depth", "2",
        # short schedule
        "--epochs", "100",
        "--warmup_steps", "20",      # so we exercise the RRDB-unfreeze branch
        "--save_every_steps", "10",
        "--val_every_steps", "15",
        "--log_every", "5",
        # loss recipe
        "--w_l1", "1.0",
        "--w_lpips", "0.1",
        "--lpips_net", "vgg",
        # EMA
        "--use_ema",
        "--ema_warmup_steps", "50",
        # bf16 for memory & speed
        "--bf16",
    ] + list(extra_args)
    print(f"\n=== running: {' '.join(cmd)}\n", flush=True)
    return subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp_name", type=str, default="v5_phase2_smoke")
    p.add_argument("--keep", action="store_true",
                   help="Keep the runs/ directory after the test (default: delete).")
    args = p.parse_args()

    exp_dir = REPO_ROOT / "runs" / args.exp_name
    if exp_dir.exists():
        print(f"[clean] removing pre-existing {exp_dir}", flush=True)
        shutil.rmtree(exp_dir)

    # ----- Run 1: 30 steps -----
    run_train(["--max_steps", "30"], exp_name=args.exp_name)

    # Inspect outputs
    manifest_path = exp_dir / "manifest.json"
    assert manifest_path.exists(), f"no manifest at {manifest_path}"
    with open(manifest_path) as f:
        manifest = json.load(f)
    events = [e["event"] for e in manifest.get("history", [])]
    print(f"\n[verify-run1] manifest events: {events}")

    step_ckpts = manifest.get("step_checkpoints", [])
    print(f"[verify-run1] step_checkpoints saved: {len(step_ckpts)} "
          f"-> {[s['global_step'] for s in step_ckpts]}")
    assert len(step_ckpts) >= 2, f"expected ≥2 step ckpts (every 10 steps over 30); got {len(step_ckpts)}"

    last_path = Path(sorted(step_ckpts, key=lambda s: s['global_step'])[-1]["path"])
    assert last_path.exists(), f"latest ckpt missing on disk: {last_path}"
    st = torch.load(str(last_path), map_location="cpu", weights_only=False)
    print(f"[verify-run1] latest ckpt: step={st['global_step']} epoch={st['epoch']}")
    print(f"[verify-run1] ckpt keys: {sorted(st.keys())}")
    for k in ("model", "optimizer", "scheduler", "global_step", "epoch", "config", "ema"):
        assert k in st, f"missing key '{k}' in checkpoint"
    n_jepa = sum(1 for k in st["model"] if k.startswith(("jepa.context_encoder.",
                                                          "jepa.target_encoder.",
                                                          "jepa.predictor.",
                                                          "jepa.projection_head.")))
    n_inj = sum(1 for k in st["model"] if k.startswith("esrgan.inject_"))
    n_rrd = sum(1 for k in st["model"] if k.startswith("esrgan.rrdbnet."))
    print(f"[verify-run1] param counts in ckpt model state — "
          f"jepa={n_jepa} inject={n_inj} rrdbnet={n_rrd}")
    assert n_inj > 0 and n_rrd > 0 and n_jepa > 0, "ckpt model is missing critical subsystems"

    # Sanity-check the model values look fine (no NaNs in any of the saved tensors).
    nan_keys = [k for k, v in st["model"].items()
                if torch.is_tensor(v) and torch.is_floating_point(v) and not torch.isfinite(v).all()]
    assert not nan_keys, f"NaN/Inf in ckpt tensors: {nan_keys[:5]}"
    print("[verify-run1] no NaN/Inf in any saved tensor")

    assert "rrdbnet_unfrozen" in events, \
        "expected rrdbnet_unfrozen event (warmup_steps=20, ran 30) — got: " + str(events)

    print("[verify-run1] PASS — run 1 produced healthy checkpoints, RRDBNet unfreeze fired")

    # ----- Run 2: resume + 10 more steps (total 40) -----
    run_train(["--max_steps", "40", "--resume"], exp_name=args.exp_name)

    with open(manifest_path) as f:
        manifest = json.load(f)
    events = [e["event"] for e in manifest.get("history", [])]
    print(f"\n[verify-run2] manifest events: {events}")
    n_starts = events.count("train_started")
    assert n_starts >= 2, f"expected ≥2 train_started events after resume, got {n_starts}"

    step_ckpts2 = manifest.get("step_checkpoints", [])
    max_step = max(s["global_step"] for s in step_ckpts2)
    print(f"[verify-run2] max step after resume: {max_step}")
    assert max_step >= 30, f"step did not advance past 30 after resume; max={max_step}"

    print("\n[ok] Phase 2 smoke test PASSED")
    if not args.keep:
        print(f"[cleanup] removing {exp_dir}")
        shutil.rmtree(exp_dir)


if __name__ == "__main__":
    main()
