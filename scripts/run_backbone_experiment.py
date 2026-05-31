"""
Sequentially run 1-epoch training on each backbone, then print a comparison table.

Each backbone gets its own:
  - experiment subdir with manifest.json (resumable independently)
  - TensorBoard run dir
  - val PSNR / SSIM / L1 measurement

Usage:
    python scripts/run_backbone_experiment.py [--data_dir data/ortho]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


EXPERIMENTS = [
    {
        "name": "backbone_imagenet",
        "backbone": "imagenet_vitb16",
        "pretrained": "models/pretrained/imagenet_vitb16.pt",
        # Encoder/predictor salvaged from the collapsed run; decoder restarts fresh.
        "salvage": "checkpoints/salvaged/imagenet_encoder_step18850.pt",
    },
    {
        "name": "backbone_dinov2",
        "backbone": "dinov2_vitb14",
        "pretrained": "models/pretrained/dinov2_vitb14.pth",
    },
    {
        "name": "backbone_explora",
        "backbone": "explora_vitb14",
        "pretrained": "models/pretrained/explora_vitb14.pth",
    },
]


def manifest_says_done(exp_dir: Path, target_epoch: int) -> bool:
    m = exp_dir / "manifest.json"
    if not m.exists():
        return False
    data = json.loads(m.read_text())
    latest = data.get("latest_epoch")
    return bool(latest and latest.get("epoch", -1) >= target_epoch - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/ortho")
    ap.add_argument("--checkpoint_dir", default="checkpoints")
    ap.add_argument("--log_dir", default="runs")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=22)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--patches_per_epoch", type=int, default=128)
    ap.add_argument("--only", choices=[e["name"] for e in EXPERIMENTS], default=None,
                    help="Run a single experiment only")
    ap.add_argument("--skip_missing_weights", action="store_true",
                    help="Skip experiments whose pretrained file is missing")
    args = ap.parse_args()

    results = []
    runs = [e for e in EXPERIMENTS if (args.only is None or e["name"] == args.only)]
    for exp in runs:
        weights = Path(exp["pretrained"])
        if not weights.exists():
            msg = f"Pretrained weights missing: {weights}"
            if args.skip_missing_weights:
                print(f"[skip] {exp['name']} — {msg}")
                continue
            print(f"[error] {msg} — run scripts/download_weights.sh first")
            sys.exit(2)

        exp_dir = Path(args.checkpoint_dir) / exp["name"]
        if manifest_says_done(exp_dir, args.epochs):
            print(f"[skip] {exp['name']} — manifest reports epoch {args.epochs-1} complete")
        else:
            cmd = [
                args.python, "-m", "src.training.train",
                "--data_dir", args.data_dir,
                "--backbone", exp["backbone"],
                "--pretrained_path", str(weights),
                "--experiment_name", exp["name"],
                "--epochs", str(args.epochs),
                "--batch_size", str(args.batch_size),
                "--patches_per_epoch", str(args.patches_per_epoch),
                "--log_dir", args.log_dir,
                "--checkpoint_dir", args.checkpoint_dir,
                "--resume",
            ]
            # If --resume can't find a checkpoint, train.py also honors --salvage_from.
            salvage = exp.get("salvage")
            if salvage and Path(salvage).exists():
                cmd += ["--salvage_from", salvage]
            print("=" * 70)
            print(f"Starting experiment: {exp['name']}")
            print("=" * 70, flush=True)
            subprocess.run(cmd, check=True)

        m_path = exp_dir / "manifest.json"
        if m_path.exists():
            mdata = json.loads(m_path.read_text())
            best = mdata.get("best") or {}
            latest = mdata.get("latest_epoch") or {}
            results.append({
                "experiment": exp["name"],
                "backbone": exp["backbone"],
                "val_psnr": best.get("val_psnr"),
                "epoch": latest.get("epoch"),
                "global_step": latest.get("global_step"),
            })

    print("\n" + "=" * 70)
    print("Backbone comparison")
    print("=" * 70)
    print(f"{'experiment':<25} {'backbone':<22} {'val PSNR':>10}")
    for r in results:
        psnr = f"{r['val_psnr']:.2f} dB" if r.get("val_psnr") is not None else "—"
        print(f"{r['experiment']:<25} {r['backbone']:<22} {psnr:>10}")

    out = Path("experiments") / "backbone_comparison.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nResults: {out}")


if __name__ == "__main__":
    main()
