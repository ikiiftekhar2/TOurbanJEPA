"""Stage-gate evaluator for v5 JEPA-Conditioned ESRGAN.

Reads the latest epoch checkpoint of the given experiment and checks whether
it meets the GO criteria for advancing to the next stage. Exit code 0 = pass
(advance), exit code 1 = fail (retry same stage).

Stage A — L1 + LPIPS, no GAN:
    val_psnr > bilinear + 0.5 dB    (clearly above the do-nothing floor)
    val_lpips < bilinear            (perceptually better than bilinear)
    val_psnr >= 19.5                (absolute floor; below this = something broke)
    no collapse signature           (val_psnr < 15 → collapse)

Stage B — + GAN adversarial:
    val_psnr > bilinear             (PSNR cost from GAN is OK but must beat bilinear)
    val_lpips < Stage-A_best - 0.05 (perceptual improvement is the whole point)
    val_psnr >= 19.0                (absolute floor; lower than Stage A because GAN
                                     trades fidelity for realism)
    no collapse signature
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


STAGE_GATES = {
    "A": {
        "description": "Stage A: L1 + LPIPS-VGG, no GAN — beat bilinear on BOTH axes",
        "val_psnr_min_abs": 19.5,
        "beat_bilinear_psnr_margin": 0.5,
        "beat_bilinear_lpips": True,
        "stage_b_warm_target_lpips_drop": None,
    },
    "B": {
        "description": "Stage B: + GAN — keep beating bilinear PSNR, drop LPIPS further",
        "val_psnr_min_abs": 19.0,
        "beat_bilinear_psnr_margin": 0.0,
        "beat_bilinear_lpips": True,
        "stage_b_warm_target_lpips_drop": 0.05,
    },
}


def find_best_checkpoint(exp_dir: Path) -> Path | None:
    best = exp_dir / "best.pt"
    if best.exists():
        return best
    epochs = sorted(exp_dir.glob("epoch_*.pt"),
                    key=lambda p: int(p.stem.split("_")[1]))
    return epochs[-1] if epochs else None


def stage_a_best_lpips(checkpoints_root: Path) -> float | None:
    """Return val_lpips of Stage A's best.pt (if it exists), for the
    Stage B improvement gate."""
    p = checkpoints_root / "v5_jepa_esrgan_stageA" / "best.pt"
    if not p.exists():
        return None
    sd = torch.load(p, map_location="cpu", weights_only=False)
    m = sd.get("val_metrics", {})
    return m.get("lpips", None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=list(STAGE_GATES))
    ap.add_argument("--exp", required=True,
                    help="Experiment name (e.g. v5_jepa_esrgan_stageA).")
    ap.add_argument("--checkpoints_root", default="checkpoints")
    args = ap.parse_args()

    gate = STAGE_GATES[args.stage]
    exp_dir = Path(args.checkpoints_root) / args.exp
    ckpt_path = find_best_checkpoint(exp_dir)
    if ckpt_path is None:
        print(f"[gate] FAIL: no checkpoint at {exp_dir}")
        sys.exit(1)

    print(f"[gate] Stage {args.stage}: {gate['description']}")
    print(f"[gate] Evaluating {ckpt_path}")

    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    metrics = sd.get("val_metrics", {})
    val_psnr = sd.get("val_psnr", metrics.get("psnr", float("nan")))
    val_lpips = metrics.get("lpips", float("nan"))
    val_ssim = metrics.get("ssim", float("nan"))
    bil_psnr = metrics.get("psnr_bil", float("nan"))
    bil_lpips = metrics.get("lpips_bil", float("nan"))

    print(f"[gate] val_psnr={val_psnr:.3f}  val_lpips={val_lpips:.4f}  "
          f"val_ssim={val_ssim:.4f}")
    print(f"[gate] bilinear_psnr={bil_psnr:.3f}  bilinear_lpips={bil_lpips:.4f}")

    failures = []
    if val_psnr < 15.0:
        failures.append(f"val_psnr {val_psnr:.3f} < 15.0 (collapse)")
    if val_psnr < gate["val_psnr_min_abs"]:
        failures.append(f"val_psnr {val_psnr:.3f} < absolute floor {gate['val_psnr_min_abs']}")

    margin = gate["beat_bilinear_psnr_margin"]
    if not (val_psnr > bil_psnr + margin):
        failures.append(f"val_psnr {val_psnr:.3f} <= bilinear {bil_psnr:.3f} + {margin}")
    if gate["beat_bilinear_lpips"] and not (val_lpips < bil_lpips):
        failures.append(f"val_lpips {val_lpips:.4f} >= bilinear {bil_lpips:.4f}")

    if args.stage == "B":
        a_lpips = stage_a_best_lpips(Path(args.checkpoints_root))
        if a_lpips is not None:
            drop = gate["stage_b_warm_target_lpips_drop"]
            print(f"[gate] Stage A best lpips={a_lpips:.4f}, "
                  f"required drop {drop:+.3f}")
            if not (val_lpips < a_lpips - drop):
                failures.append(
                    f"val_lpips {val_lpips:.4f} >= Stage-A best {a_lpips:.4f} - {drop}"
                )

    if failures:
        print("[gate] FAIL:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("[gate] PASS — all criteria met")
    sys.exit(0)


if __name__ == "__main__":
    main()
