"""
v5 training loop — JEPA-Conditioned ESRGAN.

The v4 training loop (1348 lines of spike-watchdog + R-schedule + 7-term loss
combo + frozen-encoder scaffolding) is preserved on the `v4` branch. v5 is a
ground-up rewrite around a much cleaner recipe:

    Loss = w_l1 * L1(sr, hr) + w_lpips * LPIPS(sr, hr) [+ w_jepa * jepa_loss]

  Optimizer = AdamW with 4 param groups:
      jepa_encoder         (DINOv2)  — low LR (1e-5)
      jepa_pred_proj                 — medium LR (1e-4)
      jepa_injection                 — high LR (1e-4) — the NEW zero-init layers
      rrdbnet_pretrained             — very low LR (1e-5), FROZEN for warmup_steps

  Schedule: linear warmup over `warmup_steps` then cosine to `min_lr`.
  Spike guard: simple — if l1 > spike_threshold AND > 2× recent median, skip
               the batch (no grad step). We do NOT roll back; we trust EMA.
  EMA + checkpoints (rotating step slots + epoch best) carried from v4.

The training entrypoint stays `python -m src.training.train ...` for habit.
"""

from __future__ import annotations

import argparse
import math
import statistics
import time
from collections import deque
from pathlib import Path
from typing import Optional

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.data.ortho_dataset import OrthoDataset
from src.evaluation.metrics import psnr as psnr_metric, ssim_metric
from src.models.discriminator import UNetDiscriminatorSN
from src.models.v5_model import V5Model, build_v5_model
from src.training.checkpoint import CheckpointManager
from src.training.ema import ModelEMA
from src.training.losses import LPIPSLoss, l1_pixel_loss


# --------------------------------------------------------------------------- args
def parse_args():
    p = argparse.ArgumentParser()
    # Data
    p.add_argument("--ortho_dir", type=str, default="data/ortho")
    p.add_argument("--tile_manifest", type=str, default=None,
                   help="Optional path to a tile-basename allowlist.")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--val_batch_size", type=int, default=0,
                   help="Val batch size — defaults to train batch_size. With no "
                        "grads, val can usually run 2-3× the train batch size.")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--empty_cache_every", type=int, default=0,
                   help="Call torch.cuda.empty_cache() every N steps to cap the "
                        "reserved-pool creep that AdamW + EMA can cause. 0 = off.")
    p.add_argument("--patches_per_epoch", type=int, default=32)
    p.add_argument("--val_patches_per_tile", type=int, default=2)

    # Model
    p.add_argument("--backbone", type=str, default="dinov2_vitb14")
    p.add_argument("--jepa_pretrained", type=str,
                   default="models/pretrained/dinov2_vitb14.pth")
    p.add_argument("--esrgan_weights", type=str,
                   default="models/pretrained/RealESRGAN_x4plus.pth")
    p.add_argument("--resume_jepa_from", type=str, default=None,
                   help="Path to v4 best.pt — warm-start JEPABackbone only.")
    p.add_argument("--predictor_depth", type=int, default=6)
    p.add_argument("--injection_project_dim", type=int, default=64)
    p.add_argument("--use_grad_checkpoint", action="store_true")

    # Training
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--max_steps", type=int, default=0,
                   help="Stop after this many global steps (0 = unlimited).")
    p.add_argument("--rrdbnet_lr", type=float, default=1e-5)
    p.add_argument("--injection_lr", type=float, default=1e-4)
    p.add_argument("--jepa_encoder_lr", type=float, default=1e-5)
    p.add_argument("--jepa_pred_proj_lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_steps", type=int, default=1000,
                   help="LR linear warmup AND RRDBNet-frozen duration.")
    p.add_argument("--min_lr_ratio", type=float, default=0.05)
    p.add_argument("--grad_clip", type=float, default=1.0)

    # Loss weights
    p.add_argument("--w_l1", type=float, default=1.0)
    p.add_argument("--w_lpips", type=float, default=0.1)
    p.add_argument("--w_jepa", type=float, default=0.0,
                   help="Optional self-supervision term on JEPA features.")
    p.add_argument("--lpips_net", type=str, default="vgg", choices=["alex", "vgg"])

    # GAN stage (Stage B). Disabled by default.
    p.add_argument("--use_gan", action="store_true",
                   help="Enable adversarial training with U-Net discriminator (Stage B).")
    p.add_argument("--w_gan", type=float, default=0.1,
                   help="Weight on the generator-side adversarial loss.")
    p.add_argument("--d_lr", type=float, default=1e-4,
                   help="Discriminator learning rate (AdamW).")
    p.add_argument("--d_num_feat", type=int, default=64,
                   help="Base feature width of the U-Net discriminator.")
    p.add_argument("--gan_warmup_steps", type=int, default=0,
                   help="Steps to delay adversarial backprop after GAN mode starts. "
                        "Useful when resuming Stage A into Stage B — discriminator "
                        "trains alone for this many steps before G feels it.")
    p.add_argument("--resume_g_from", type=str, default=None,
                   help="Path to a Stage-A best.pt — warm-start G (full model) only.")

    # Spike guard
    p.add_argument("--l1_spike_abs", type=float, default=0.20)
    p.add_argument("--l1_spike_ratio", type=float, default=2.0)
    p.add_argument("--l1_window_size", type=int, default=50)

    # EMA
    p.add_argument("--use_ema", action="store_true")
    p.add_argument("--ema_decay_start", type=float, default=0.995)
    p.add_argument("--ema_decay_end", type=float, default=0.9999)
    p.add_argument("--ema_warmup_steps", type=int, default=2000)
    p.add_argument("--ema_start_step", type=int, default=0)

    # JEPA target EMA
    p.add_argument("--target_ema_decay_start", type=float, default=0.996)
    p.add_argument("--target_ema_decay_end", type=float, default=0.9999)

    # Logging / checkpoints
    p.add_argument("--exp_name", type=str, required=True)
    p.add_argument("--checkpoint_dir", type=str, default="runs")
    p.add_argument("--log_dir", type=str, default="runs")
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--val_every_steps", type=int, default=500)
    # Apples-to-apples val matching train's degradation + scale distribution.
    # Default ON; runs at epoch boundaries (cheap) + optionally at step_val.
    p.add_argument("--scale_match_val", action="store_true", default=True)
    p.add_argument("--no_scale_match_val", action="store_false",
                   dest="scale_match_val")
    p.add_argument("--scale_match_val_scales", type=float, nargs="+",
                   default=[16.0, 20.0, 24.0])
    p.add_argument("--scale_match_val_at_step_val", action="store_true",
                   default=False,
                   help="Run the scale-match val at every step_val too "
                        "(adds ~scales×30s per val). Off by default — "
                        "epoch_val only.")
    p.add_argument("--save_every_steps", type=int, default=200)
    p.add_argument("--recovery_slots", type=int, default=6)
    p.add_argument("--rollback_steps", type=int, default=200,
                   help="On resume, pick the latest ckpt with global_step "
                        "<= latest_step - rollback_steps. 0 = resume from "
                        "the most recent checkpoint exactly (no rollback).")
    p.add_argument("--resume", action="store_true",
                   help="Auto-resume from latest checkpoint in the experiment dir.")
    p.add_argument("--reset_ema_on_resume", action="store_true",
                   help="On resume, do NOT load EMA shadow from ckpt — "
                        "instead re-snapshot from current live model weights. "
                        "Use when EMA shadow is stale/corrupt and you want "
                        "val to reflect live model behaviour going forward.")
    p.add_argument("--bf16", action="store_true")

    return p.parse_args()


# ----------------------------------------------------------------- LR schedule
def warmup_cosine_lambda(warmup_steps: int, total_steps: int, min_ratio: float):
    def f(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        cos = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cos
    return f


def ema_target_decay(step: int, total_steps: int, start: float, end: float) -> float:
    if total_steps <= 0:
        return end
    progress = min(1.0, step / total_steps)
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))  # 1 → 0
    return end + (start - end) * cos


# ------------------------------------------------------------------ dataloaders
def build_dataloaders(args):
    train_ds = OrthoDataset(
        ortho_dir=args.ortho_dir, split="train", augment=True,
        patches_per_epoch=args.patches_per_epoch,
        tile_manifest=args.tile_manifest,
    )
    val_ds = OrthoDataset(
        ortho_dir=args.ortho_dir, split="val", augment=False,
        val_patches_per_tile=args.val_patches_per_tile,
        tile_manifest=args.tile_manifest,
    )
    common = dict(
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    val_bs = args.val_batch_size if args.val_batch_size > 0 else args.batch_size
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(
        val_ds, batch_size=val_bs, shuffle=False, drop_last=False, **common)
    return train_loader, val_loader


def build_scale_match_val_loaders(args):
    """One val DataLoader per scale, all with train-matched LR degradation.

    Returns: list of (scale_float, DataLoader). Uses num_workers=0 to keep
    process count bounded — the eval runs are short and val is IO-cheap.
    """
    # Local import to avoid circulars; the module-level VAL_SCALE is patched
    # before each dataset construction, then restored.
    import src.data.ortho_dataset as od
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


# ------------------------------------------------------------------------- val
@torch.no_grad()
def validate(model: V5Model, val_loader, device, lpips_loss, max_batches: int = 0):
    """Returns {psnr, ssim, lpips, l1, psnr_bil, lpips_bil, n_batches}."""
    model.eval()
    sums = {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0, "l1": 0.0,
            "psnr_bil": 0.0, "lpips_bil": 0.0}
    n = 0
    for i, batch in enumerate(val_loader):
        if max_batches and i >= max_batches:
            break
        lr = batch["low_res"].to(device, non_blocking=True)
        hr = batch["high_res"].to(device, non_blocking=True)
        out = model(lr, hr)
        sr = out["sr"].clamp(0, 1)
        sums["psnr"] += psnr_metric(sr, hr).item()
        sums["ssim"] += ssim_metric(sr, hr).item()
        sums["l1"] += l1_pixel_loss(sr, hr).item()
        sums["psnr_bil"] += psnr_metric(lr, hr).item()
        if lpips_loss is not None:
            sums["lpips"] += lpips_loss(sr, hr).item()
            sums["lpips_bil"] += lpips_loss(lr, hr).item()
        n += 1
    metrics = {k: v / max(1, n) for k, v in sums.items()}
    metrics["n_batches"] = n
    return metrics


# -------------------------------------------------------------------- main loop
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}", flush=True)
    torch.manual_seed(42)

    exp_dir = Path(args.log_dir) / args.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(exp_dir))

    train_loader, val_loader = build_dataloaders(args)
    sm_loaders = []
    if args.scale_match_val:
        sm_loaders = build_scale_match_val_loaders(args)
        print(f"[data] scale-match val enabled at scales="
              f"{[s for s, _ in sm_loaders]}", flush=True)
    steps_per_epoch = len(train_loader)
    total_steps = args.max_steps or steps_per_epoch * args.epochs
    print(f"[data] steps_per_epoch={steps_per_epoch} total_steps={total_steps}", flush=True)

    print(f"[model] building V5Model (backbone={args.backbone})", flush=True)
    model = build_v5_model(
        backbone_name=args.backbone,
        jepa_pretrained_path=args.jepa_pretrained,
        esrgan_weights_path=args.esrgan_weights,
        predictor_depth=args.predictor_depth,
        project_dim=args.injection_project_dim,
        use_grad_checkpoint=args.use_grad_checkpoint,
    ).to(device)
    if args.resume_jepa_from:
        print(f"[warm-start] loading JEPA from {args.resume_jepa_from}", flush=True)
        model.load_jepa_from_v4_checkpoint(args.resume_jepa_from, strict=False)

    # RRDBNet starts FROZEN. We thaw at warmup_steps (per v4 lesson #3).
    print(f"[freeze] RRDBNet frozen for warmup_steps={args.warmup_steps}", flush=True)
    model.freeze_rrdbnet()

    param_groups = model.trainable_param_groups(
        jepa_encoder_lr=args.jepa_encoder_lr,
        jepa_pred_proj_lr=args.jepa_pred_proj_lr,
        injection_lr=args.injection_lr,
        rrdbnet_lr=args.rrdbnet_lr,
    )
    optimizer = AdamW(param_groups, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    scheduler = LambdaLR(optimizer,
                         warmup_cosine_lambda(args.warmup_steps, total_steps,
                                              args.min_lr_ratio))

    lpips_loss = LPIPSLoss(net=args.lpips_net).to(device) if args.w_lpips > 0 else None

    # --------- Stage-B warm-start: load G (full V5Model) from a Stage-A best.pt
    if args.resume_g_from:
        print(f"[warm-start] loading G (V5Model state) from {args.resume_g_from}", flush=True)
        st_g = torch.load(args.resume_g_from, map_location=device, weights_only=False)
        model.load_checkpoint_state(st_g["model"], strict=False)

    # ---------------------------------------------------- Discriminator (Stage B)
    discriminator = None
    d_optimizer = None
    d_scheduler = None
    bce_logits = None
    if args.use_gan:
        print(f"[gan] building UNetDiscriminatorSN(num_feat={args.d_num_feat})", flush=True)
        discriminator = UNetDiscriminatorSN(
            num_in_ch=3, num_feat=args.d_num_feat,
        ).to(device)
        d_optimizer = AdamW(discriminator.parameters(), lr=args.d_lr,
                            betas=(0.9, 0.99), weight_decay=0.0)
        # Use the same warmup-cosine schedule shape as G; this keeps D's
        # effective LR sub-1 during the early steps so it doesn't lock down.
        d_scheduler = LambdaLR(d_optimizer,
                               warmup_cosine_lambda(args.warmup_steps, total_steps,
                                                    args.min_lr_ratio))
        bce_logits = torch.nn.BCEWithLogitsLoss()

    ema = None
    if args.use_ema:
        ema = ModelEMA(model,
                       decay_start=args.ema_decay_start,
                       decay_end=args.ema_decay_end,
                       warmup_steps=args.ema_warmup_steps,
                       start_step=args.ema_start_step,
                       device="cpu")

    cm = CheckpointManager(args.checkpoint_dir, args.exp_name,
                           recovery_every=args.save_every_steps,
                           recovery_slots=args.recovery_slots,
                           rollback_steps=args.rollback_steps)

    # ------------------------------------------------------- resume (optional)
    start_epoch, start_step, best_val_psnr = 0, 0, -float("inf")
    if args.resume:
        path, _meta = cm.get_resume_checkpoint()
        if path is not None:
            print(f"[resume] {path}", flush=True)
            st = torch.load(str(path), map_location=device, weights_only=False)
            model.load_checkpoint_state(st["model"], strict=False)
            optimizer.load_state_dict(st["optimizer"])
            if st.get("scheduler"):
                scheduler.load_state_dict(st["scheduler"])
            if ema is not None and "ema" in st and not args.reset_ema_on_resume:
                ema.load_state_dict(st["ema"])
            elif ema is not None and args.reset_ema_on_resume:
                print("[resume] --reset_ema_on_resume: re-snapshotting EMA "
                      "shadow from live model weights", flush=True)
                # Rebuild the shadow dict from the (just-loaded) live model.
                ema.shadow = {
                    name: p.detach().to(ema.device).clone()
                    for name, p in ema._trainable_named_params(model)
                }
            # Discriminator state (Stage B): resume D + d_optimizer + d_scheduler
            # from the same checkpoint slot. A Stage-A→Stage-B transition won't
            # have any of these keys, which is the correct behaviour (D starts
            # fresh on Stage B). A Stage-B → Stage-B restart restores everything.
            if discriminator is not None and "discriminator" in st:
                discriminator.load_state_dict(st["discriminator"])
                if "d_optimizer" in st:
                    d_optimizer.load_state_dict(st["d_optimizer"])
                if "d_scheduler" in st and d_scheduler is not None:
                    d_scheduler.load_state_dict(st["d_scheduler"])
                print("[resume] discriminator state restored", flush=True)
            start_epoch = st["epoch"]
            start_step = st["global_step"]
            best_val_psnr = st.get("best_val_psnr", -float("inf"))
            if start_step >= args.warmup_steps:
                model.unfreeze_rrdbnet()

    # ------------------------------------------------------------- train loop
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float32
    def autocast_ctx():
        return torch.autocast(device_type="cuda", dtype=amp_dtype)
    l1_window: deque = deque(maxlen=args.l1_window_size)

    global_step = start_step
    rrdbnet_unfrozen = global_step >= args.warmup_steps
    if rrdbnet_unfrozen:
        model.unfreeze_rrdbnet()

    print(f"[train] starting at epoch={start_epoch} global_step={global_step} "
          f"best_val_psnr={best_val_psnr:.3f}", flush=True)
    cm.log_event("train_started", global_step=global_step, epoch=start_epoch,
                 total_steps=total_steps)

    t_start = time.time()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            if args.max_steps and global_step >= args.max_steps:
                break

            # Thaw RRDBNet at the warmup boundary.
            if (not rrdbnet_unfrozen) and global_step >= args.warmup_steps:
                model.unfreeze_rrdbnet()
                rrdbnet_unfrozen = True
                cm.log_event("rrdbnet_unfrozen", global_step=global_step)
                print(f"[freeze] RRDBNet UNFROZEN at step {global_step}", flush=True)

            lr = batch["low_res"].to(device, non_blocking=True)
            hr = batch["high_res"].to(device, non_blocking=True)

            # ---- G forward (always) ----
            with autocast_ctx():
                out = model(lr, hr)
                sr_clamped = out["sr"].clamp(0, 1)
                l_l1 = l1_pixel_loss(sr_clamped, hr)
                l_lpips = (lpips_loss(sr_clamped, hr) if lpips_loss is not None
                           else torch.zeros((), device=device))
                l_jepa = out["jepa_loss"] if args.w_jepa > 0 else torch.zeros((), device=device)

            # ---- Spike guard on L1 (drop the batch entirely, no D step either) ----
            l1_val = float(l_l1.detach())
            window_med = statistics.median(l1_window) if l1_window else l1_val
            skip = (l1_val > args.l1_spike_abs and
                    l1_val > args.l1_spike_ratio * window_med)

            l_d = torch.zeros((), device=device)
            l_g_adv = torch.zeros((), device=device)
            gan_active = (discriminator is not None
                          and global_step >= args.warmup_steps + args.gan_warmup_steps)

            if not skip:
                # ---- Discriminator step (Stage B) ----
                if discriminator is not None:
                    sr_detached = sr_clamped.detach()
                    with autocast_ctx():
                        d_real = discriminator(hr)
                        d_fake = discriminator(sr_detached)
                        l_d_real = bce_logits(d_real, torch.ones_like(d_real))
                        l_d_fake = bce_logits(d_fake, torch.zeros_like(d_fake))
                        l_d = 0.5 * (l_d_real + l_d_fake)
                    if torch.isfinite(l_d):
                        d_optimizer.zero_grad(set_to_none=True)
                        l_d.backward()
                        if args.grad_clip > 0:
                            torch.nn.utils.clip_grad_norm_(discriminator.parameters(),
                                                            args.grad_clip)
                        d_optimizer.step()
                        if d_scheduler is not None:
                            d_scheduler.step()

                # ---- Generator step ----
                with autocast_ctx():
                    if gan_active:
                        d_fake_for_g = discriminator(sr_clamped)
                        l_g_adv = bce_logits(d_fake_for_g, torch.ones_like(d_fake_for_g))
                    total = (args.w_l1 * l_l1
                             + args.w_lpips * l_lpips
                             + args.w_jepa * l_jepa
                             + (args.w_gan * l_g_adv if gan_active else 0.0))

                if torch.isfinite(total):
                    optimizer.zero_grad(set_to_none=True)
                    total.backward()
                    if args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()
                    scheduler.step()
                    l1_window.append(l1_val)
                else:
                    cm.log_event("batch_skipped", global_step=global_step,
                                 l1=l1_val, window_median=window_med, skip_reason="nan_g")
            else:
                total = torch.zeros((), device=device)
                cm.log_event("batch_skipped", global_step=global_step,
                             l1=l1_val, window_median=window_med,
                             skip_reason="l1_spike")

            if ema is not None:
                ema.update(model, global_step)
            tgt_decay = ema_target_decay(global_step, total_steps,
                                         args.target_ema_decay_start,
                                         args.target_ema_decay_end)
            model.update_target_ema(tgt_decay)

            # Logging
            if global_step % args.log_every == 0:
                with torch.no_grad():
                    psnr_train = psnr_metric(sr_clamped, hr).item()
                writer.add_scalar("train/l1", l1_val, global_step)
                writer.add_scalar("train/lpips", float(l_lpips.detach()), global_step)
                writer.add_scalar("train/total", float(total.detach()), global_step)
                writer.add_scalar("train/psnr", psnr_train, global_step)
                writer.add_scalar("train/jepa_loss", float(out["jepa_loss"].detach()), global_step)
                writer.add_scalar("train/cos_sim", float(out["cos_sim"].detach()), global_step)
                writer.add_scalar("train/ctx_std", float(out["ctx_std"].detach()), global_step)
                writer.add_scalar("train/pred_feat_std", float(out["pred_feat_std"].detach()), global_step)
                writer.add_scalar("opt/lr_injection",
                                  optimizer.param_groups[2]["lr"], global_step)
                writer.add_scalar("opt/lr_rrdbnet",
                                  optimizer.param_groups[3]["lr"], global_step)
                if discriminator is not None:
                    writer.add_scalar("train/d_loss", float(l_d.detach()), global_step)
                    writer.add_scalar("train/g_adv", float(l_g_adv.detach()), global_step)
                    writer.add_scalar("opt/lr_d", d_optimizer.param_groups[0]["lr"], global_step)
                elapsed = time.time() - t_start
                gan_str = (f"d={float(l_d):.4f} g_adv={float(l_g_adv):.4f} "
                           if discriminator is not None else "")
                print(f"[step {global_step:>6}] ep={epoch} b={batch_idx} "
                      f"l1={l1_val:.4f} lpips={float(l_lpips):.4f} "
                      f"total={float(total):.4f} psnr={psnr_train:.2f} "
                      f"ctx_std={float(out['ctx_std']):.3f} "
                      f"{gan_str}t={elapsed:.0f}s "
                      f"{'[SKIP]' if skip else ''}", flush=True)

            # Step checkpoint
            if global_step > 0 and global_step % args.save_every_steps == 0:
                d_state = _discriminator_state(discriminator, d_optimizer, d_scheduler)
                cm.save_step(model, optimizer, scheduler, scaler=None,
                             epoch=epoch, batch_idx=batch_idx,
                             global_step=global_step, best_val_psnr=best_val_psnr,
                             config=vars(args), l1_window=l1_window, ema=ema,
                             extra_state=d_state)

            # Periodic reserved-pool flush — keeps peak from creeping.
            if (args.empty_cache_every > 0
                    and global_step > 0
                    and global_step % args.empty_cache_every == 0
                    and torch.cuda.is_available()):
                torch.cuda.empty_cache()

            # Periodic in-loop validation
            if global_step > 0 and global_step % args.val_every_steps == 0:
                _run_val_and_log(model, ema, val_loader, device, lpips_loss,
                                 writer, global_step, label="step_val")
                if sm_loaders and args.scale_match_val_at_step_val:
                    _run_scale_match_val_and_log(
                        model, ema, sm_loaders, device, lpips_loss,
                        writer, global_step, label="step_scalematch_val")
                model.train()

            global_step += 1

        if args.max_steps and global_step >= args.max_steps:
            break

        val_metrics = _run_val_and_log(model, ema, val_loader, device, lpips_loss,
                                       writer, global_step, label="epoch_val")
        if sm_loaders:
            _run_scale_match_val_and_log(
                model, ema, sm_loaders, device, lpips_loss,
                writer, global_step, label="epoch_scalematch_val")
        v_psnr = val_metrics["psnr"]
        if v_psnr > best_val_psnr:
            best_val_psnr = v_psnr
        d_state = _discriminator_state(discriminator, d_optimizer, d_scheduler)
        cm.save_epoch(model, optimizer, scheduler, scaler=None,
                      epoch=epoch + 1, global_step=global_step,
                      val_psnr=v_psnr, best_val_psnr=best_val_psnr,
                      config=vars(args), extra_metrics=val_metrics, ema=ema,
                      extra_state=d_state)

    cm.join_pending_saves()
    cm.log_event("train_finished", global_step=global_step, best_val_psnr=best_val_psnr)
    print(f"[done] global_step={global_step} best_val_psnr={best_val_psnr:.3f}", flush=True)


def _discriminator_state(discriminator, d_optimizer, d_scheduler) -> Optional[dict]:
    if discriminator is None:
        return None
    return {
        "discriminator": discriminator.state_dict(),
        "d_optimizer": d_optimizer.state_dict(),
        "d_scheduler": d_scheduler.state_dict() if d_scheduler is not None else None,
    }


def _run_val_and_log(model: V5Model, ema: Optional[ModelEMA], val_loader, device,
                     lpips_loss, writer: SummaryWriter, global_step: int,
                     label: str = "val") -> dict:
    """Validate with EMA weights applied (if EMA enabled), restore after."""
    backup = None
    if ema is not None:
        backup = ema.apply_to(model)
    try:
        metrics = validate(model, val_loader, device, lpips_loss)
    finally:
        if ema is not None and backup is not None:
            ema.restore(model, backup)

    for k, v in metrics.items():
        if k == "n_batches":
            continue
        writer.add_scalar(f"{label}/{k}", v, global_step)
    psnr_delta = metrics["psnr"] - metrics["psnr_bil"]
    lpips_delta = metrics["lpips"] - metrics["lpips_bil"] if lpips_loss is not None else 0.0
    writer.add_scalar(f"{label}/psnr_delta_vs_bilinear", psnr_delta, global_step)
    writer.add_scalar(f"{label}/lpips_delta_vs_bilinear", lpips_delta, global_step)
    print(f"[{label} @ step {global_step}] "
          f"psnr={metrics['psnr']:.3f}  (bil={metrics['psnr_bil']:.3f}, "
          f"d={psnr_delta:+.3f}) "
          f"lpips={metrics['lpips']:.3f}  (bil={metrics['lpips_bil']:.3f}, "
          f"d={lpips_delta:+.3f}) "
          f"ssim={metrics['ssim']:.3f} l1={metrics['l1']:.4f} "
          f"n={metrics['n_batches']}",
          flush=True)
    return metrics


def _run_scale_match_val_and_log(
        model: V5Model, ema: Optional[ModelEMA], sm_loaders, device,
        lpips_loss, writer: SummaryWriter, global_step: int,
        label: str = "scalematch_val") -> dict:
    """Run val on each (scale, loader); log per-scale and aggregate metrics.

    EMA is applied once around the whole sweep (not per-scale) to avoid 5×
    apply/restore cost. Returns the average metrics dict across scales.
    """
    if not sm_loaders:
        return {}
    backup = None
    if ema is not None:
        backup = ema.apply_to(model)
    per_scale = []
    try:
        for s, loader in sm_loaders:
            m = validate(model, loader, device, lpips_loss)
            per_scale.append((s, m))
    finally:
        if ema is not None and backup is not None:
            ema.restore(model, backup)

    keys = [k for k in per_scale[0][1] if k != "n_batches"]
    avg = {k: sum(m[k] for _, m in per_scale) / len(per_scale) for k in keys}

    for s, m in per_scale:
        tag = f"{label}_{s:g}x"
        for k in keys:
            writer.add_scalar(f"{tag}/{k}", m[k], global_step)
        writer.add_scalar(f"{tag}/psnr_delta_vs_bilinear",
                          m["psnr"] - m["psnr_bil"], global_step)
        writer.add_scalar(f"{tag}/lpips_delta_vs_bilinear",
                          m["lpips"] - m["lpips_bil"], global_step)
    for k in keys:
        writer.add_scalar(f"{label}/{k}_avg", avg[k], global_step)
    writer.add_scalar(f"{label}/psnr_delta_vs_bilinear_avg",
                      avg["psnr"] - avg["psnr_bil"], global_step)
    writer.add_scalar(f"{label}/lpips_delta_vs_bilinear_avg",
                      avg["lpips"] - avg["lpips_bil"], global_step)

    scales_str = ",".join(f"{s:g}" for s, _ in per_scale)
    per_scale_str = " ".join(
        f"{s:g}x:{m['psnr']:.2f}(d{m['psnr']-m['psnr_bil']:+.2f})"
        for s, m in per_scale)
    print(f"[{label} @ step {global_step}] scales=[{scales_str}]  "
          f"avg_psnr={avg['psnr']:.3f} (bil={avg['psnr_bil']:.3f}, "
          f"d={avg['psnr']-avg['psnr_bil']:+.3f})  "
          f"avg_lpips={avg['lpips']:.4f} (bil={avg['lpips_bil']:.4f}, "
          f"d={avg['lpips']-avg['lpips_bil']:+.4f})  | per-scale: {per_scale_str}",
          flush=True)
    return avg


if __name__ == "__main__":
    main()
