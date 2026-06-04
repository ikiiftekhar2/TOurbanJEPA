"""Bare-RRDBNet fine-tune control for v5 attribution.

Mirrors the v5 Stage A recipe EXACTLY minus the JEPA branch, so we can
distinguish "JEPA features helped" from "Toronto fine-tuning alone would
have closed the gap." Specifically:

  * Same data: OrthoDataset(train_textured.txt manifest), train_ratio=0.9.
  * Same losses: w_l1=1.0, w_lpips=0.1 (VGG).
  * Same schedule: 3 epochs, bs=20, AdamW(betas=(0.9,0.95), wd=1e-4),
    warmup-cosine LR over total_steps, min_lr_ratio=0.05.
  * Same LR for the pretrained backbone: rrdbnet_lr=1e-5.
  * bf16 autocast, grad_clip=1.0.
  * Scale-matched val on {16, 18, 20, 22, 24}× with match_train_aug_in_val=True
    every val_every_steps (default 1000) — this is the truth-of-record
    metric set we used to score v5 itself.
  * best.pt promoted by *scale-match LPIPS avg* (the v5 winning criterion),
    NOT fixed-20× PSNR.

No JEPA branch, no injection layers, no EMA (we proved it's stale/sticky
for this setup), no GAN.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import src.data.ortho_dataset as od
from src.data.ortho_dataset import OrthoDataset
from src.models.esrgan.weight_loader import build_pretrained_x4plus
from src.evaluation.metrics import psnr as psnr_metric, ssim_metric
from src.training.losses import LPIPSLoss, l1_pixel_loss


# ============================================================ args
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ortho_dir", default="data/ortho")
    p.add_argument("--tile_manifest",
                   default="data/ortho/metadata/train_textured.txt")
    p.add_argument("--esrgan_weights",
                   default="models/pretrained/RealESRGAN_x4plus.pth")
    p.add_argument("--exp_name", default="v5_bare_rrdb_control")
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--log_dir", default="runs")

    p.add_argument("--batch_size", type=int, default=20)
    p.add_argument("--val_batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--patches_per_epoch", type=int, default=32)
    p.add_argument("--val_patches_per_tile", type=int, default=2)

    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--rrdbnet_lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_steps", type=int, default=1000)
    p.add_argument("--min_lr_ratio", type=float, default=0.05)
    p.add_argument("--grad_clip", type=float, default=1.0)

    p.add_argument("--w_l1", type=float, default=1.0)
    p.add_argument("--w_lpips", type=float, default=0.1)
    p.add_argument("--lpips_net", default="vgg", choices=["alex", "vgg"])

    p.add_argument("--log_every", type=int, default=25)
    p.add_argument("--val_every_steps", type=int, default=1000)
    p.add_argument("--save_every_steps", type=int, default=500)
    p.add_argument("--scale_match_val_scales", type=float, nargs="+",
                   default=[16.0, 18.0, 20.0, 22.0, 24.0])

    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--empty_cache_every", type=int, default=200)

    return p.parse_args()


# ============================================================ data
def build_train_loader(args):
    ds = OrthoDataset(
        ortho_dir=args.ortho_dir, split="train", augment=True,
        patches_per_epoch=args.patches_per_epoch,
        tile_manifest=args.tile_manifest,
    )
    return DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )


def build_scale_match_val_loaders(args) -> List[Tuple[float, DataLoader]]:
    """Mirror train.py's scale-match val loaders: one DataLoader per scale,
    all with match_train_aug_in_val=True (same blur+JPEG distribution as
    train). num_workers=0 (val is IO-cheap, runs serially)."""
    orig = od.VAL_SCALE
    val_bs = args.val_batch_size if args.val_batch_size > 0 else args.batch_size
    loaders = []
    for s in args.scale_match_val_scales:
        od.VAL_SCALE = float(s)
        ds = OrthoDataset(
            ortho_dir=args.ortho_dir, split="val", augment=False,
            val_patches_per_tile=args.val_patches_per_tile,
            tile_manifest=args.tile_manifest,
            match_train_aug_in_val=True,
        )
        loaders.append((float(s), DataLoader(
            ds, batch_size=val_bs, shuffle=False, drop_last=False,
            num_workers=0, pin_memory=True)))
    od.VAL_SCALE = orig
    return loaders


# ============================================================ LR schedule
def warmup_cosine_lambda(warmup_steps: int, total_steps: int, min_ratio: float):
    def f(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        cos = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cos
    return f


# ============================================================ val
@torch.no_grad()
def validate_scale_matched(rrdbnet, sm_loaders, device, lpips_loss) -> Dict[str, float]:
    """Run scale-matched val across all (scale, loader) pairs. Returns
    {psnr_avg, ssim_avg, lpips_avg, psnr_bil_avg, lpips_bil_avg,
     per_scale: {scale -> {psnr,ssim,lpips,psnr_bil,lpips_bil}}}.

    Average is uniform across scales (each scale weighted equally), matching
    how v5 reported its final headline number."""
    rrdbnet.eval()
    per_scale = {}
    for scale, loader in sm_loaders:
        sums = {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0, "l1": 0.0,
                "psnr_bil": 0.0, "lpips_bil": 0.0}
        n = 0
        for batch in loader:
            lr = batch["low_res"].to(device, non_blocking=True)
            hr = batch["high_res"].to(device, non_blocking=True)
            # bare RRDBNet: input is 256×256 bilinear LR, the model expects
            # native 64×64 → 256 (4×). Match the baseline path: avg_pool 4×
            # then run RRDBNet.
            lr_small = F.avg_pool2d(lr, kernel_size=4, stride=4)
            sr = rrdbnet(lr_small).clamp(0, 1)
            sums["psnr"] += psnr_metric(sr, hr).item()
            sums["ssim"] += ssim_metric(sr, hr).item()
            sums["l1"] += l1_pixel_loss(sr, hr).item()
            sums["psnr_bil"] += psnr_metric(lr, hr).item()
            if lpips_loss is not None:
                sums["lpips"] += lpips_loss(sr, hr).item()
                sums["lpips_bil"] += lpips_loss(lr, hr).item()
            n += 1
        per_scale[scale] = {k: v / max(1, n) for k, v in sums.items()}

    # Uniform avg across scales
    keys = next(iter(per_scale.values())).keys()
    avg = {k: sum(per_scale[s][k] for s in per_scale) / len(per_scale)
           for k in keys}
    return {
        **{k + "_avg": v for k, v in avg.items()},
        "per_scale": per_scale,
    }


# ============================================================ checkpoint
class SimpleCkpt:
    def __init__(self, root: Path, exp: str, slots: int = 6):
        self.dir = Path(root) / exp
        self.dir.mkdir(parents=True, exist_ok=True)
        self.slots = slots
        self._slot_idx = 0
        self.manifest_path = self.dir / "manifest.json"
        if self.manifest_path.exists():
            self.manifest = json.loads(self.manifest_path.read_text())
        else:
            self.manifest = {"slots": {}, "best": None, "history": []}

    def _save(self, path: Path, payload: dict):
        tmp = path.with_suffix(".pt.tmp")
        torch.save(payload, tmp)
        tmp.replace(path)

    def save_step(self, model, optimizer, scheduler, epoch, global_step,
                  config, val_metrics=None):
        p = self.dir / f"step_slot_{self._slot_idx}.pt"
        payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "config": config,
            "val_metrics": val_metrics or {},
        }
        self._save(p, payload)
        self.manifest["slots"][str(self._slot_idx)] = {
            "path": p.name, "step": global_step,
        }
        self._slot_idx = (self._slot_idx + 1) % self.slots
        self.manifest_path.write_text(json.dumps(self.manifest, indent=2))

    def save_epoch(self, model, optimizer, scheduler, epoch, global_step,
                   config, val_metrics):
        p = self.dir / f"epoch_{epoch}.pt"
        payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "config": config,
            "val_metrics": val_metrics,
        }
        self._save(p, payload)

    def maybe_promote_best(self, model, global_step, val_metrics, key="lpips_avg"):
        v = val_metrics.get(key)
        if v is None or not math.isfinite(v):
            return False
        current_best = (self.manifest.get("best") or {}).get(key)
        # Lower LPIPS is better.
        if current_best is None or v < current_best:
            best_path = self.dir / "best.pt"
            payload = {
                "model": model.state_dict(),
                "global_step": global_step,
                "val_metrics": val_metrics,
                "val_psnr": val_metrics.get("psnr_avg", float("nan")),
            }
            self._save(best_path, payload)
            self.manifest["best"] = {
                "step": global_step, "path": "best.pt",
                key: v,
                "psnr_avg": val_metrics.get("psnr_avg"),
            }
            self.manifest_path.write_text(json.dumps(self.manifest, indent=2))
            return True
        return False

    def get_resume(self):
        """Return latest step slot's path + meta, or None."""
        if not self.manifest["slots"]:
            return None, None
        # Pick the slot with the LARGEST step.
        slots = self.manifest["slots"]
        best_slot = max(slots.values(), key=lambda v: v["step"])
        p = self.dir / best_slot["path"]
        if not p.exists():
            return None, None
        return p, best_slot


# ============================================================ main
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}", flush=True)
    torch.manual_seed(42)

    exp_dir = Path(args.log_dir) / args.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(exp_dir))

    # ---- data
    train_loader = build_train_loader(args)
    sm_loaders = build_scale_match_val_loaders(args)
    print(f"[data] scales={[s for s,_ in sm_loaders]}", flush=True)
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    print(f"[data] steps_per_epoch={steps_per_epoch} total_steps={total_steps}",
          flush=True)

    # ---- model: bare pretrained RRDBNet (no JEPA, no injection layers)
    print(f"[model] loading pretrained RRDBNet from {args.esrgan_weights}",
          flush=True)
    rrdbnet = build_pretrained_x4plus(args.esrgan_weights, eval_mode=False).to(device)
    rrdbnet.train()
    n_params = sum(p.numel() for p in rrdbnet.parameters())
    print(f"[model] {n_params/1e6:.2f}M params, all trainable from step 0",
          flush=True)

    # ---- losses
    lpips_loss = LPIPSLoss(net=args.lpips_net).to(device) if args.w_lpips > 0 else None

    # ---- optimizer + scheduler (single param group)
    optimizer = AdamW(rrdbnet.parameters(), lr=args.rrdbnet_lr,
                      weight_decay=args.weight_decay, betas=(0.9, 0.95))
    scheduler = LambdaLR(optimizer,
                         warmup_cosine_lambda(args.warmup_steps, total_steps,
                                              args.min_lr_ratio))

    # ---- ckpt
    cm = SimpleCkpt(Path(args.checkpoint_dir), args.exp_name)

    start_epoch, start_step = 0, 0
    if args.resume:
        path, meta = cm.get_resume()
        if path is not None:
            print(f"[resume] {path} step={meta['step']}", flush=True)
            st = torch.load(str(path), map_location=device, weights_only=False)
            rrdbnet.load_state_dict(st["model"])
            optimizer.load_state_dict(st["optimizer"])
            scheduler.load_state_dict(st["scheduler"])
            start_epoch = st["epoch"]
            start_step = st["global_step"]

    # ---- loop
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float32
    def autocast_ctx():
        return torch.autocast(device_type="cuda", dtype=amp_dtype)

    global_step = start_step
    t_start = time.time()
    print(f"[train] starting epoch={start_epoch} step={global_step}", flush=True)

    for epoch in range(start_epoch, args.epochs):
        rrdbnet.train()
        for batch_idx, batch in enumerate(train_loader):
            lr = batch["low_res"].to(device, non_blocking=True)
            hr = batch["high_res"].to(device, non_blocking=True)
            # Match the baseline: avg_pool 256→64, native 4× upsample.
            lr_small = F.avg_pool2d(lr, kernel_size=4, stride=4)

            with autocast_ctx():
                sr = rrdbnet(lr_small)
                sr_c = sr.clamp(0, 1)
                l_l1 = l1_pixel_loss(sr_c, hr)
                l_lpips = (lpips_loss(sr_c, hr) if lpips_loss is not None
                           else torch.zeros((), device=device))
                total = args.w_l1 * l_l1 + args.w_lpips * l_lpips

            if torch.isfinite(total):
                optimizer.zero_grad(set_to_none=True)
                total.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(rrdbnet.parameters(),
                                                    args.grad_clip)
                optimizer.step()
                scheduler.step()

            if global_step % args.log_every == 0:
                with torch.no_grad():
                    psnr_t = psnr_metric(sr_c, hr).item()
                writer.add_scalar("train/l1", float(l_l1.detach()), global_step)
                writer.add_scalar("train/lpips", float(l_lpips.detach()), global_step)
                writer.add_scalar("train/total", float(total.detach()), global_step)
                writer.add_scalar("train/psnr", psnr_t, global_step)
                writer.add_scalar("opt/lr", optimizer.param_groups[0]["lr"],
                                  global_step)
                elapsed = time.time() - t_start
                print(f"[step {global_step:>6}] ep={epoch} b={batch_idx} "
                      f"l1={float(l_l1):.4f} lpips={float(l_lpips):.4f} "
                      f"total={float(total):.4f} psnr={psnr_t:.2f} "
                      f"t={elapsed:.0f}s", flush=True)

            if global_step > 0 and global_step % args.save_every_steps == 0:
                cm.save_step(rrdbnet, optimizer, scheduler, epoch, global_step,
                             config=vars(args))

            if (args.empty_cache_every > 0
                    and global_step > 0
                    and global_step % args.empty_cache_every == 0
                    and torch.cuda.is_available()):
                torch.cuda.empty_cache()

            if global_step > 0 and global_step % args.val_every_steps == 0:
                metrics = validate_scale_matched(rrdbnet, sm_loaders, device,
                                                 lpips_loss)
                writer.add_scalar("val/psnr_avg", metrics["psnr_avg"], global_step)
                writer.add_scalar("val/lpips_avg", metrics["lpips_avg"], global_step)
                writer.add_scalar("val/ssim_avg", metrics["ssim_avg"], global_step)
                writer.add_scalar("val/psnr_bil_avg", metrics["psnr_bil_avg"],
                                  global_step)
                writer.add_scalar("val/lpips_bil_avg", metrics["lpips_bil_avg"],
                                  global_step)
                for s, m in metrics["per_scale"].items():
                    writer.add_scalar(f"val/psnr_{int(s)}x", m["psnr"], global_step)
                    writer.add_scalar(f"val/lpips_{int(s)}x", m["lpips"], global_step)
                promoted = cm.maybe_promote_best(rrdbnet, global_step, metrics)
                print(f"[step_val @{global_step}] psnr_avg={metrics['psnr_avg']:.3f} "
                      f"lpips_avg={metrics['lpips_avg']:.4f}  "
                      f"(bil psnr={metrics['psnr_bil_avg']:.3f} "
                      f"lpips={metrics['lpips_bil_avg']:.4f}) "
                      f"{'PROMOTED' if promoted else ''}", flush=True)
                rrdbnet.train()

            global_step += 1

        # Epoch end: save + scale-match val
        metrics = validate_scale_matched(rrdbnet, sm_loaders, device, lpips_loss)
        for s, m in metrics["per_scale"].items():
            writer.add_scalar(f"epoch_val/psnr_{int(s)}x", m["psnr"], global_step)
            writer.add_scalar(f"epoch_val/lpips_{int(s)}x", m["lpips"], global_step)
        writer.add_scalar("epoch_val/psnr_avg", metrics["psnr_avg"], global_step)
        writer.add_scalar("epoch_val/lpips_avg", metrics["lpips_avg"], global_step)
        cm.save_epoch(rrdbnet, optimizer, scheduler, epoch + 1, global_step,
                      config=vars(args), val_metrics=metrics)
        cm.maybe_promote_best(rrdbnet, global_step, metrics)
        print(f"[epoch_val ep={epoch+1}] psnr_avg={metrics['psnr_avg']:.3f} "
              f"lpips_avg={metrics['lpips_avg']:.4f}", flush=True)

    elapsed = time.time() - t_start
    print(f"[done] step={global_step} time={elapsed:.0f}s", flush=True)


if __name__ == "__main__":
    main()
