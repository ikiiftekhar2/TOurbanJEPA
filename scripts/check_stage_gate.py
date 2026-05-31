"""Stage-gate evaluator for v4 Phase 1.5 staged training.

Reads the latest epoch checkpoint of the given experiment and checks
whether it meets the stage's GO criteria. Exit code 0 = pass (advance to
next stage), exit code 1 = fail (retry same stage).

Stage A (pure L1+SSIM+JEPA, no perceptual):
    val_psnr >= 21.0 dB
    val_l1 <= 0.10
    no collapse signature
Goal: stable pixel baseline before introducing perceptual loss.

Stage B (LPIPS-VGG @ w=0.5):
    val_lpips_vgg <= 0.50
    val_psnr >= 19.5
    val_ssim >= 0.40
Goal: perceptual gradient engaged, lpips dropping past v3 baseline (0.612).

Stage C (full LPIPS @ w=1.0, slow LR):
    val_lpips_vgg <= 0.45
    val_psnr >= 19.5
    val_ssim >= 0.55
Goal: final Phase 1 production target.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


STAGE_GATES = {
    "A": {
        "val_psnr_min": 23.0,        # absolute fidelity floor (~bilinear is 23.9)
        "val_l1_max": None,
        "val_lpips_max": None,
        "val_ssim_min": None,
        "beat_bilinear": True,       # must beat bilinear on BOTH PSNR and LPIPS
        "psnr_margin": 0.5,          # clearly beat, not tie
        "description": "Fidelity-first: beat bilinear on BOTH PSNR and LPIPS.",
    },
    "B": {
        "val_psnr_min": 19.5,
        "val_l1_max": 0.12,
        "val_lpips_max": 0.50,
        "val_ssim_min": 0.40,
        "description": "Perceptual engaged (LPIPS-VGG w=0.5).",
    },
    "C": {
        "val_psnr_min": 19.5,
        "val_l1_max": 0.12,
        "val_lpips_max": 0.45,
        "val_ssim_min": 0.55,
        "description": "Full Phase 1.5 target (LPIPS-VGG w=1.0).",
    },
}


def find_best_checkpoint(exp_dir: Path) -> Path | None:
    """Pick best.pt if it exists, else the latest epoch_N.pt."""
    best = exp_dir / "best.pt"
    if best.exists():
        return best
    epochs = sorted(exp_dir.glob("epoch_*.pt"),
                    key=lambda p: int(p.stem.split("_")[1]))
    return epochs[-1] if epochs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=list(STAGE_GATES.keys()),
                    help="Stage letter to evaluate.")
    ap.add_argument("--exp", required=True,
                    help="Experiment name (e.g. v4_p1_5_stageA). "
                         "Looks for checkpoints/<exp>/best.pt or epoch_N.pt.")
    args = ap.parse_args()

    gate = STAGE_GATES[args.stage]
    exp_dir = Path("checkpoints") / args.exp
    ckpt_path = find_best_checkpoint(exp_dir)
    if ckpt_path is None:
        print(f"[gate] FAIL: no checkpoint found in {exp_dir}")
        sys.exit(1)

    print(f"[gate] Stage {args.stage}: {gate['description']}")
    print(f"[gate] Evaluating {ckpt_path}")

    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    metrics = sd.get("val_metrics", {})
    val_psnr = sd.get("val_psnr", metrics.get("psnr", float("nan")))
    val_l1 = metrics.get("l1", float("nan"))
    val_ssim = metrics.get("ssim", float("nan"))
    val_lpips = metrics.get("lpips_vgg", metrics.get("lpips", float("nan")))

    print(f"[gate] val_psnr={val_psnr:.3f}  "
          f"val_l1={val_l1:.4f}  "
          f"val_ssim={val_ssim:.4f}  "
          f"val_lpips_vgg={val_lpips:.4f}")

    failures = []
    if gate["val_psnr_min"] is not None and val_psnr < gate["val_psnr_min"]:
        failures.append(f"val_psnr {val_psnr:.3f} < {gate['val_psnr_min']}")
    if gate["val_l1_max"] is not None and val_l1 > gate["val_l1_max"]:
        failures.append(f"val_l1 {val_l1:.4f} > {gate['val_l1_max']}")
    if gate["val_lpips_max"] is not None and val_lpips > gate["val_lpips_max"]:
        failures.append(f"val_lpips {val_lpips:.4f} > {gate['val_lpips_max']}")
    if gate["val_ssim_min"] is not None and val_ssim < gate["val_ssim_min"]:
        failures.append(f"val_ssim {val_ssim:.4f} < {gate['val_ssim_min']}")
    # Beat-bilinear gate (Stage A): must clearly beat the naive bilinear baseline
    # on BOTH PSNR and LPIPS, else the model isn't doing useful SR. Compare model
    # alex-LPIPS to bilinear alex-LPIPS (apples-to-apples; bilinear_lpips is alex).
    if gate.get("beat_bilinear"):
        bil_psnr = metrics.get("bilinear_psnr", float("nan"))
        bil_lpips = metrics.get("bilinear_lpips", float("nan"))
        model_lpips_alex = metrics.get("lpips", float("nan"))
        margin = gate.get("psnr_margin", 0.0)
        print(f"[gate] bilinear_psnr={bil_psnr:.3f}  bilinear_lpips={bil_lpips:.4f}  "
              f"model_lpips(alex)={model_lpips_alex:.4f}")
        if not (val_psnr > bil_psnr + margin):
            failures.append(f"val_psnr {val_psnr:.3f} <= bilinear {bil_psnr:.3f}+{margin}")
        if not (model_lpips_alex < bil_lpips):
            failures.append(f"lpips(alex) {model_lpips_alex:.4f} >= bilinear {bil_lpips:.4f}")
    # Collapse signature: PSNR way below bicubic floor.
    if val_psnr < 15.0:
        failures.append(f"val_psnr {val_psnr:.3f} < 15.0 (collapse signature)")

    if failures:
        print("[gate] FAIL:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("[gate] PASS — all criteria met")
        sys.exit(0)


if __name__ == "__main__":
    main()
