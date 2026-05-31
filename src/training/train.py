"""
Unified UrbanJEPA v3 training script.

End-to-end: context encoder (low LR) + predictor + decoder all trainable from
step 1. JEPA loss is a regularizer. Zero-init residual decoder gives identity
at step 0, so no freeze-first-epoch trick is needed.

Resume:
  --resume        auto-pick latest step checkpoint with rollback, fall back to epoch
  --resume_from   pick a specific checkpoint path
"""

import argparse
import json
import math
import os
import statistics
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.data.ortho_dataset import OrthoDataset
from src.evaluation.metrics import psnr as psnr_metric
from src.models.urbanjepa import UrbanJEPA
from src.training.checkpoint import CheckpointManager
from src.training.ema import ModelEMA
from src.training.losses import (
    LPIPSLoss,
    VGGPerceptualLoss,
    hierarchical_jepa_loss,
    high_frequency_loss,
    l1_pixel_loss,
    out_of_range_loss,
    ssim_loss,
    vicreg_loss,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--backbone", required=True,
                   choices=["imagenet_vitb16", "dinov2_vitb14", "explora_vitb14"])
    p.add_argument("--pretrained_path", default=None)

    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=22)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--encoder_lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--grad_clip", type=float, default=0.5,
                   help="Max grad norm. 0.5 is tighter than the usual 1.0 — helps with AMP edge cases.")
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16",
                   help="bf16 has fp32 exponent range so no overflow on Ampere/RTX 3090. "
                        "fp16 needs GradScaler and can overflow forward activations.")
    p.add_argument("--max_consecutive_nan", type=int, default=20,
                   help="Skip-on-NaN: if more than this many NaNs in a row, save and raise.")
    p.add_argument("--l1_spike_window", type=int, default=200,
                   help="Window of recent train L1 values for the spike watchdog.")
    p.add_argument("--l1_spike_ratio", type=float, default=3.0,
                   help="Abort if current train L1 > ratio * median(window). Triggers "
                        "systemd restart, which rolls back to a pre-spike checkpoint.")
    p.add_argument("--l1_spike_warmup", type=int, default=500,
                   help="Don't arm the spike watchdog until this many steps have passed.")
    p.add_argument("--per_sample_l1_max", type=float, default=0.30,
                   help="Per-sample L1 outlier threshold: samples with L1 > this "
                        "are masked OUT of pixel losses (L1/SSIM/VGG/HF/LPIPS/OOB) "
                        "for the current batch. JEPA loss stays on full batch. "
                        "If all samples masked, batch is fully dropped. Healthy "
                        "training samples have L1 ~0.05-0.15; >0.30 is genuine "
                        "outlier territory (text, fence grilles, signage).")
    p.add_argument("--hard_tile_log_path", type=str, default=None,
                   help="If set, write per-tile mean L1 statistics to this JSON "
                        "after each epoch. Used for offline analysis to identify "
                        "systematically hard tiles for blacklisting.")
    p.add_argument("--skip_high_l1", type=float, default=0.20,
                   help="Skip backward+optimizer.step for batches with L1 > this. "
                        "These batches likely have hard-content outputs that produce "
                        "noisy/unreliable gradients; updating from them poisons Adam "
                        "momentum. Skipping keeps optimizer state clean. Healthy "
                        "training L1 is ~0.07-0.10, so 0.20 is 2-3x normal — single "
                        "occasional skips are normal; sustained skipping is bad.")
    p.add_argument("--l1_spike_consecutive", type=int, default=2,
                   help="Require N consecutive spike-level batches before aborting. "
                        "Single high-L1 batches are usually hard-content false-positives. "
                        "Genuine collapse produces sustained high L1 over many batches.")
    p.add_argument("--l1_spike_absolute", type=float, default=0.25,
                   help="Absolute L1 threshold: abort if L1 > this regardless of window. "
                        "Active from step 1 (no warmup), so it catches collapse during "
                        "the first ~200 steps after resume when the window-based check "
                        "isn't filled yet. Healthy SR training has L1 ~0.07-0.10, so 0.25 "
                        "is a conservative 2.5-3.5x normal level.")
    p.add_argument("--save_step_l1_max", type=float, default=0.20,
                   help="Refuse to save a step_slot checkpoint if median L1 over the "
                        "recent window exceeds this. Prevents the rotating step_slot "
                        "buffer from overwriting healthy weights with collapsed ones — "
                        "the exact failure mode hit on 2026-05-27 ~19:50.")
    p.add_argument("--warmup_epochs", type=float, default=0.3,
                   help="Fractional epochs of linear warmup. Hard-capped at 10%% of total "
                        "steps so the LR actually plateaus and cosine-decays. The old "
                        "default of 3 produced pure linear ramp for 3-epoch runs.")
    p.add_argument("--warmup_cap_frac", type=float, default=0.10,
                   help="Maximum warmup fraction of total steps.")
    p.add_argument("--num_workers", type=int, default=8,
                   help="DataLoader workers (12-core box: 8 leaves 4 for main/GPU/system)")
    p.add_argument("--patches_per_epoch", type=int, default=128)
    p.add_argument("--val_patches_per_tile", type=int, default=4)
    p.add_argument("--persistent_workers", action="store_true", default=True)
    p.add_argument("--dataloader_spawn", action="store_true",
                   help="Use 'spawn' multiprocessing for DataLoader workers so they "
                        "don't inherit the parent's CUDA context (~600 MB each). "
                        "Lets us run many workers at zero GPU-memory cost.")
    p.add_argument("--prefetch_factor", type=int, default=4,
                   help="Batches prefetched per worker (only with num_workers>0). "
                        "Higher = more buffering against NAS read latency.")
    p.add_argument("--tile_cache_size", type=int, default=32,
                   help="Per-worker LRU cache size (~50 MB/tile)")

    p.add_argument("--predictor_depth", type=int, default=6)
    p.add_argument("--decoder_attn_blocks", type=int, default=4)
    p.add_argument("--decoder_base_dim", type=int, default=768)
    p.add_argument("--dropout", type=float, default=0.1)

    p.add_argument("--w_l1", type=float, default=2.0)
    p.add_argument("--w_ssim", type=float, default=0.5)
    p.add_argument("--w_vgg", type=float, default=0.5)
    p.add_argument("--w_hf", type=float, default=0.0,
                   help="high-frequency FFT loss weight. DEFAULT 0: at 20x SR "
                        "the target has ~no real high-freq, so this forces the "
                        "decoder to hallucinate detail that L1 punishes -> "
                        "contradictory gradients -> collapse (the destabilizer "
                        "that heavy LPIPS was masking). Re-enable deliberately.")
    p.add_argument("--w_jepa", type=float, default=0.1)
    p.add_argument("--w_vicreg", type=float, default=0.0,
                   help="VICReg variance+covariance anti-collapse weight on the "
                        "predictor's projected features. The JEPA otherwise has "
                        "no explicit force preventing representation collapse.")
    p.add_argument("--w_oob", type=float, default=0.05,
                   help="Out-of-range penalty on the unclamped decoder output. "
                        "Keeps pixels in [0,1] without using a gradient-killing clamp.")

    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--experiment_name", default=None)
    p.add_argument("--recovery_every", type=int, default=50)
    p.add_argument("--recovery_slots", type=int, default=6)
    p.add_argument("--rollback_steps", type=int, default=500,
                   help="On resume, roll back this many steps. Larger helps escape NaN regions.")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--resume_from", default=None)
    p.add_argument("--salvage_from", default=None,
                   help="Path to a salvage_encoder.py output (.pt). Loaded with "
                        "strict=False after backbone init — overlays trained "
                        "context_encoder/predictor/projection_head weights from a "
                        "prior collapsed run while leaving the (new) decoder fresh. "
                        "Only used when --resume_from / --resume don't apply.")

    p.add_argument("--log_dir", default="runs")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--val_log_images", type=int, default=4)
    p.add_argument("--val_every_steps", type=int, default=7000,
                   help="Run validation every N GLOBAL steps, independent of "
                        "epoch boundaries. Critical because collapses reset "
                        "batch_idx to 0, so epoch-end-only validation can be "
                        "starved forever; global_step survives restarts. "
                        "0 = epoch-end only.")

    p.add_argument("--ema_start", type=float, default=0.996)
    p.add_argument("--ema_end", type=float, default=1.0)

    # ---- v4 feature flags ----
    p.add_argument("--residual_range_start", type=float, default=0.05,
                   help="Initial RESIDUAL_RANGE for v3 PixelDecoderWithSkips. "
                        "Linearly ramps to --residual_range_end across "
                        "--residual_range_ramp_start..end. Default keeps it pinned at 0.05.")
    p.add_argument("--residual_range_end", type=float, default=0.05,
                   help="Final RESIDUAL_RANGE after the ramp completes.")
    p.add_argument("--residual_range_ramp_start", type=int, default=5000,
                   help="Step at which to begin ramping R from start -> end.")
    p.add_argument("--residual_range_ramp_end", type=int, default=15000,
                   help="Step at which R reaches --residual_range_end and holds.")
    p.add_argument("--use_v4_decoder", action="store_true",
                   help="cross-attn + sigmoid PixelDecoderV4 (V4_PLAN §2.3) — known unstable, prefer v5")
    p.add_argument("--use_v5_decoder", action="store_true",
                   help="PixelDecoderV5: additive residual + clamp output, "
                        "single bottleneck cross-attn + U-Net skips, GroupNorm(8). "
                        "Replaces v4 after the 2026-05-27 architectural cliff diagnosis.")
    p.add_argument("--use_grad_checkpoint", action="store_true",
                   help="Gradient checkpointing on the context encoder's transformer "
                        "blocks: recompute activations during backward instead of "
                        "storing them. ~40%% lower peak GPU memory, ~25%% slower. "
                        "Enables larger batch on memory-bound GPUs.")
    p.add_argument("--empty_cache_every", type=int, default=100,
                   help="Call torch.cuda.empty_cache() every N steps to cap "
                        "reserved-pool fragmentation growth on real-data training. "
                        "0 disables.")
    p.add_argument("--use_v4_predictor", action="store_true",
                   help="8-layer HierarchicalFeaturePredictor (V4_PLAN §2.2)")
    p.add_argument("--hierarchical_jepa", action="store_true",
                   help="use multi-layer JEPA loss; requires --use_v4_predictor")
    p.add_argument("--use_lpips", action="store_true",
                   help="add LPIPS perceptual loss (Phase 1+); set --w_lpips and --lpips_net")
    p.add_argument("--lpips_net", type=str, default="vgg", choices=["vgg", "alex"],
                   help="LPIPS backbone net. 'vgg' matches the production eval metric "
                        "(Zhang 2018 LPIPS-VGG). 'alex' is faster but less perceptually correlated.")
    p.add_argument("--l1_warmup_steps", type=int, default=1000,
                   help="Steps over which w_lpips ramps 0 -> args.w_lpips (linear). "
                        "Real-ESRGAN convention (Wang 2021): avoid perceptual gradient "
                        "spikes when the decoder is still outputting near-noise.")
    p.add_argument("--w_lpips", type=float, default=1.0,
                   help="LPIPS loss weight (Path A default 1.0)")
    p.add_argument("--match_train_aug_in_val", action="store_true",
                   help="apply train-style realistic LR degradation to val (Phase 1+)")
    p.add_argument("--cross_sensor_aug", action="store_true",
                   help="enable LR-only photometric aug (brightness/contrast jitter, "
                        "sensor noise, atmospheric haze/clouds) that simulates the "
                        "PlanetScope<->ortho sensor gap. OFF for clean-SR training "
                        "(it creates a train/val mismatch); ON for the Planet finetune.")
    p.add_argument("--use_ema", action="store_true",
                   help="maintain EMA shadow of trainable params (V4_PLAN §2.7)")
    p.add_argument("--freeze_encoder", action="store_true",
                   help="Freeze context_encoder + predictor + projection_head "
                        "(Phase-2-style: train decoder alone against fixed "
                        "pretrained features). Implies disabling JEPA/VICReg "
                        "loss contribution since the encoder side won't update.")
    p.add_argument("--ema_decay_start", type=float, default=0.995)
    p.add_argument("--ema_decay_end", type=float, default=0.9999)
    p.add_argument("--ema_start_step", type=int, default=0,
                   help="global step at which EMA begins updating (0 = from start)")
    p.add_argument("--ema_warmup_steps", type=int, default=0,
                   help="linear ramp of EMA decay over this many steps")
    p.add_argument("--lr_clamp_abort_rate", type=float, default=0.05,
                   help="abort if decoder lr_clamp_rate > this for "
                        "--lr_clamp_abort_window steps (Phase 1 watchdog)")
    p.add_argument("--lr_clamp_abort_window", type=int, default=100)
    p.add_argument("--feat_std_abort_min", type=float, default=0.01,
                   help="abort if mean per-dim encoder-feature std falls below "
                        "this for --feat_std_abort_window steps (JEPA "
                        "representation-collapse watchdog)")
    p.add_argument("--feat_std_abort_window", type=int, default=100)

    p.add_argument("--smoke_test", action="store_true",
                   help="Run 20 steps + val and exit (for sanity checks)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tile_manifest", type=str, default=None,
                   help="Optional path to a tile-basename allowlist (one filename "
                        "per line). When set, train+val operate only on these tiles. "
                        "Use to drop flat/poison tiles that train the decoder to a "
                        "content-independent degenerate solution.")
    return p.parse_args()


def build_dataloaders(args):
    tm = getattr(args, "tile_manifest", None)
    train_ds = OrthoDataset(
        args.data_dir, split="train", augment=True,
        patches_per_epoch=args.patches_per_epoch, seed=args.seed,
        tile_cache_size=args.tile_cache_size,
        cross_sensor_aug=getattr(args, "cross_sensor_aug", False),
        tile_manifest=tm,
    )
    val_ds = OrthoDataset(
        args.data_dir, split="val", augment=False,
        val_patches_per_tile=args.val_patches_per_tile, seed=args.seed,
        tile_cache_size=args.tile_cache_size,
        match_train_aug_in_val=getattr(args, "match_train_aug_in_val", False),
        tile_manifest=tm,
    )
    common = dict(
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.persistent_workers and args.num_workers > 0,
    )
    # 'spawn' start method for workers: forked workers inherit the parent's
    # CUDA context (~600 MB each on GPU) because the DataLoader iterator is
    # first created after model.cuda(). Spawned workers start as fresh
    # processes WITHOUT the CUDA context, so we can run many workers at zero
    # GPU-memory cost — fixes the data-starvation that left the GPU idle ~80%.
    if args.num_workers > 0 and args.dataloader_spawn:
        import multiprocessing as _mp
        common["multiprocessing_context"] = _mp.get_context("spawn")
        common["prefetch_factor"] = args.prefetch_factor
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, **common,
    )
    # Val loader runs in the main process (num_workers=0): val tiles are small
    # and only ~119 batches, but PERSISTENT spawn workers would otherwise stay
    # alive between validations holding ~2 GB each in RAM (~16 GB total for 8),
    # which evicted the train tile cache and starved the GPU.
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False,
        num_workers=0, pin_memory=True,
    )
    return train_loader, val_loader


def warmup_cosine_lambda(warmup_steps: int, total_steps: int, min_ratio: float = 0.01):
    def fn(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine
    return fn


def cosine_ema_decay(step: int, total_steps: int, start: float, end: float) -> float:
    progress = min(1.0, max(0.0, step / max(1, total_steps)))
    cosine = 0.5 * (1.0 - math.cos(math.pi * progress))
    return start + (end - start) * cosine


# Optional perceptual metrics — only loaded if lpips / pytorch_msssim ms_ssim
# are available. Older subprocesses keep working unchanged if the import fails.
# Both Alex and VGG variants of LPIPS are loaded: VGG matches the production
# eval metric (oracle pick used LPIPS-VGG); Alex kept for back-compat.
_LPIPS_FN_ALEX = None
_LPIPS_FN_VGG = None
_MS_SSIM_FN = None


def _maybe_load_perceptual(device):
    global _LPIPS_FN_ALEX, _LPIPS_FN_VGG, _MS_SSIM_FN
    if _LPIPS_FN_ALEX is None:
        try:
            import lpips
            _LPIPS_FN_ALEX = lpips.LPIPS(net="alex", verbose=False).to(device)
            _LPIPS_FN_ALEX.eval()
            for p in _LPIPS_FN_ALEX.parameters():
                p.requires_grad = False
        except ImportError:
            _LPIPS_FN_ALEX = False
    if _LPIPS_FN_VGG is None:
        try:
            import lpips
            _LPIPS_FN_VGG = lpips.LPIPS(net="vgg", verbose=False).to(device)
            _LPIPS_FN_VGG.eval()
            for p in _LPIPS_FN_VGG.parameters():
                p.requires_grad = False
        except ImportError:
            _LPIPS_FN_VGG = False
    if _MS_SSIM_FN is None:
        try:
            from pytorch_msssim import ms_ssim as ms_ssim_fn
            _MS_SSIM_FN = ms_ssim_fn
        except ImportError:
            _MS_SSIM_FN = False
    return (
        _LPIPS_FN_ALEX if _LPIPS_FN_ALEX is not False else None,
        _LPIPS_FN_VGG if _LPIPS_FN_VGG is not False else None,
        _MS_SSIM_FN if _MS_SSIM_FN is not False else None,
    )


@torch.no_grad()
def validate(model, val_loader, device, vgg_loss, writer=None, global_step=0,
             log_images: int = 0, amp_dtype=torch.bfloat16, autocast_enabled=True,
             ema=None):
    """Eval. If `ema` is provided (and has begun updating), the EMA-smoothed
    weights are swapped in for the duration of validation and restored
    afterward via try/finally. Otherwise live weights are used."""
    model.eval()
    use_ema = (ema is not None and getattr(ema, "start_step", 0) <= global_step)
    ema_backup = ema.apply_to(model) if use_ema else None
    try:
        return _validate_inner(model, val_loader, device, vgg_loss, writer,
                               global_step, log_images, amp_dtype, autocast_enabled)
    finally:
        if ema_backup is not None:
            ema.restore(model, ema_backup)


def _validate_inner(model, val_loader, device, vgg_loss, writer, global_step,
                    log_images, amp_dtype, autocast_enabled):
    lpips_alex_fn, lpips_vgg_fn, ms_ssim_fn = _maybe_load_perceptual(device)
    total = {"psnr": 0.0, "l1": 0.0, "ssim": 0.0, "vgg": 0.0, "jepa": 0.0,
             "lpips": 0.0, "lpips_vgg": 0.0, "msssim": 0.0,
             "bilinear_psnr": 0.0, "bilinear_lpips": 0.0}
    n = 0
    img_logged = 0
    for batch in val_loader:
        low_res = batch["low_res"].to(device, non_blocking=True)
        high_res = batch["high_res"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=autocast_enabled):
            result = model(low_res, high_res)
        # Decoder forward is unclamped; metrics + image logging clamp to [0,1].
        pred = result["pred_image"].float().clamp(0.0, 1.0)
        hr = high_res.float()
        bil = low_res.float().clamp(0.0, 1.0)
        l1 = F.l1_loss(pred, hr)
        ssim_val = 1.0 - ssim_loss(pred, hr)
        vgg = vgg_loss(pred, hr)
        ps = psnr_metric(pred, hr)
        total["psnr"] += ps.item()
        total["l1"] += l1.item()
        total["ssim"] += ssim_val.item()
        total["vgg"] += vgg.item()
        total["jepa"] += result["jepa_loss"].item()
        total["bilinear_psnr"] += psnr_metric(bil, hr).item()
        if lpips_alex_fn is not None:
            total["lpips"] += lpips_alex_fn(pred * 2 - 1, hr * 2 - 1).mean().item()
            total["bilinear_lpips"] += lpips_alex_fn(bil * 2 - 1, hr * 2 - 1).mean().item()
        if lpips_vgg_fn is not None:
            total["lpips_vgg"] += lpips_vgg_fn(pred * 2 - 1, hr * 2 - 1).mean().item()
        if ms_ssim_fn is not None:
            total["msssim"] += ms_ssim_fn(pred, hr, data_range=1.0,
                                          size_average=True).item()
        n += 1
        if writer is not None and img_logged < log_images:
            k = min(log_images - img_logged, pred.shape[0])
            writer.add_images("val/pred", pred[:k].clamp(0, 1), global_step)
            writer.add_images("val/low_res", low_res[:k].clamp(0, 1), global_step)
            writer.add_images("val/high_res", hr[:k].clamp(0, 1), global_step)
            img_logged += k
    return {k: v / max(1, n) for k, v in total.items()}


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    if args.experiment_name is None:
        args.experiment_name = f"backbone_{args.backbone}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Record the CUDA allocation history so that on the next OOM we can dump a
    # snapshot showing WHICH tensors (with stack traces) held the memory at the
    # moment of failure. The live process OOMs at ~22.9 GB while the offline
    # trace peaks at 19.5 GB — this captures the missing ~3 GB at the source.
    if torch.cuda.is_available() and os.environ.get("UJ_MEM_DEBUG", "1") == "1":
        try:
            torch.cuda.memory._record_memory_history(max_entries=200000)
            print("[mem-debug] recording CUDA allocation history", flush=True)
        except Exception as _e:
            print(f"[mem-debug] could not enable history: {_e}", flush=True)
    log_dir = Path(args.log_dir) / args.experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    train_loader, val_loader = build_dataloaders(args)
    print(f"Train: {len(train_loader.dataset)} samples, {len(train_loader)} batches/epoch")
    print(f"Val:   {len(val_loader.dataset)} samples, {len(val_loader)} batches")

    model = UrbanJEPA(
        backbone_name=args.backbone,
        pretrained_path=args.pretrained_path,
        predictor_depth=args.predictor_depth,
        decoder_attn_blocks=args.decoder_attn_blocks,
        decoder_base_dim=args.decoder_base_dim,
        dropout=args.dropout,
        use_v4_predictor=getattr(args, "use_v4_predictor", False),
        use_v4_decoder=getattr(args, "use_v4_decoder", False),
        use_v5_decoder=getattr(args, "use_v5_decoder", False),
        hierarchical_jepa=getattr(args, "hierarchical_jepa", False),
        use_grad_checkpoint=getattr(args, "use_grad_checkpoint", False),
    ).to(device)

    if getattr(args, "freeze_encoder", False):
        model.freeze_encoder_side()
        print("[freeze-encoder] context_encoder + predictor + projection_head "
              "frozen (decoder-only training against pretrained features)",
              flush=True)

    n_total = sum(p.numel() for p in model.parameters()) / 1e6
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Model: {n_total:.1f}M params, trainable {n_train:.1f}M")

    param_groups = model.trainable_param_groups(decoder_lr=args.lr, encoder_lr=args.encoder_lr)
    optimizer = AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay,
                      betas=(0.9, 0.95))

    steps_per_epoch = max(1, len(train_loader) // max(1, args.grad_accum))
    total_steps = max(1, steps_per_epoch * args.epochs)
    raw_warmup = int(steps_per_epoch * args.warmup_epochs)
    warmup_cap = max(1, int(total_steps * args.warmup_cap_frac))
    warmup_steps = max(1, min(raw_warmup, warmup_cap))
    print(f"Schedule: total_steps={total_steps} warmup_steps={warmup_steps} "
          f"(raw={raw_warmup}, capped at {warmup_cap})")
    scheduler = LambdaLR(optimizer, warmup_cosine_lambda(warmup_steps, total_steps))
    # Intended base LRs from THIS launch's args (decoder/predictor/proj at args.lr,
    # encoder at args.encoder_lr). LambdaLR captures these at construction. On
    # resume the checkpoint's scheduler+optimizer state would otherwise restore
    # the OLD base LRs (whatever --lr the saving run used) and silently override
    # any LR change — so we re-assert these after loading checkpoint state.
    intended_base_lrs = list(scheduler.base_lrs)
    scaler = torch.amp.GradScaler("cuda") if args.precision == "fp16" else None

    vgg_loss_fn = VGGPerceptualLoss().to(device)
    lpips_loss_fn = None
    if getattr(args, "use_lpips", False):
        lpips_loss_fn = LPIPSLoss(net=args.lpips_net).to(device)
        print(f"LPIPS-{args.lpips_net.upper()} training loss enabled "
              f"(w_lpips={args.w_lpips}, l1_warmup={args.l1_warmup_steps} steps)")

    ema = None
    if getattr(args, "use_ema", False):
        # EMA shadow on GPU (was 'cpu'). CPU shadow forced 174 GPU→CPU param
        # transfers + syncs every step once past ema_start_step=4000 — that was
        # the throughput collapse at step ~4000-5000 (311% CPU, stalled steps).
        # GPU shadow blends in-place with no transfer; costs ~1 GB, which we
        # now have (memory solved: model peak 15.4 GB without checkpointing).
        ema_device = "cuda" if torch.cuda.is_available() else "cpu"
        ema = ModelEMA(
            model,
            decay_start=args.ema_decay_start,
            decay_end=args.ema_decay_end,
            warmup_steps=args.ema_warmup_steps,
            start_step=args.ema_start_step,
            device=ema_device,
        )
        print(f"EMA enabled ({ema_device}): decay {args.ema_decay_start} -> "
              f"{args.ema_decay_end} over {args.ema_warmup_steps} steps "
              f"starting at step {args.ema_start_step}")

    precision_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    amp_dtype = precision_map[args.precision]
    use_scaler = args.precision == "fp16"
    print(f"Precision: {args.precision} (dtype={amp_dtype}, scaler={'on' if use_scaler else 'off'})")

    ckpt_mgr = CheckpointManager(
        args.checkpoint_dir, args.experiment_name,
        recovery_every=args.recovery_every,
        recovery_slots=args.recovery_slots,
        rollback_steps=args.rollback_steps,
    )

    # Count prior boots from manifest history (every "training_started" event = one boot).
    prior_boots = sum(1 for h in ckpt_mgr.manifest.get("history", [])
                      if h.get("event") == "training_started")
    boot_id = prior_boots + 1
    ckpt_mgr.log_event("training_started", config=vars(args), boot_id=boot_id)

    # Log startup info to TensorBoard.
    boot_wall = time.time()
    writer.add_text("startup/config", "```\n" + json.dumps(vars(args), indent=2) + "\n```", 0)
    writer.add_text("startup/backbone", args.backbone, 0)
    writer.add_text("startup/experiment", args.experiment_name, 0)
    writer.add_text("startup/host_invocation",
                    f"cwd={os.getcwd()}\nhostname={os.uname().nodename}\n"
                    f"python={sys.executable}\nargv={sys.argv}",
                    0)
    writer.add_scalar("startup/boot_id", boot_id, 0)
    writer.add_scalar("startup/boot_wall_clock", boot_wall, 0)
    writer.add_scalar("startup/total_params_M", n_total, 0)
    writer.add_scalar("startup/trainable_params_M", n_train, 0)

    start_epoch = 0
    skip_batches = 0
    global_step = 0
    best_val_psnr = -1e9

    resume_path = None
    restored_l1_window: Optional[list] = None
    if args.resume_from:
        resume_path = Path(args.resume_from)
    elif args.resume:
        p, meta = ckpt_mgr.get_resume_checkpoint()
        resume_path = p
        if meta:
            print(f"[resume] picked {p.name} (step={meta.get('global_step')}, epoch={meta.get('epoch')})")
    if resume_path is not None and resume_path.exists():
        sd = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_checkpoint_state(sd["model"])
        if sd.get("optimizer"):
            optimizer.load_state_dict(sd["optimizer"])
        if sd.get("scheduler"):
            scheduler.load_state_dict(sd["scheduler"])
        # Re-assert the intended base LRs from THIS launch (the checkpoint's
        # optimizer+scheduler state restored the OLD base LRs, which would
        # silently override any --lr change made since the checkpoint was saved).
        scheduler.base_lrs = list(intended_base_lrs)
        for g, b, lam in zip(optimizer.param_groups, intended_base_lrs,
                             scheduler.lr_lambdas):
            g["initial_lr"] = b
            g["lr"] = b * lam(scheduler.last_epoch)
        print(f"[resume] re-asserted base LRs {intended_base_lrs} "
              f"(now lr={[round(g['lr'], 8) for g in optimizer.param_groups]})",
              flush=True)
        if sd.get("scaler") and scaler is not None:
            try:
                scaler.load_state_dict(sd["scaler"])
            except Exception as e:
                print(f"[resume] could not restore scaler state ({e}); using fresh scaler")
        if sd.get("ema") is not None and ema is not None:
            try:
                ema.load_state_dict(sd["ema"])
                print(f"[resume] restored EMA shadow ({len(ema.shadow)} tensors)",
                      flush=True)
            except Exception as e:
                print(f"[resume] could not restore EMA ({e}); using fresh EMA shadow")
        elif ema is not None:
            print(f"[resume] no EMA state in checkpoint — using fresh shadow "
                  f"(legacy ckpt; EMA will warm up from live weights)",
                  flush=True)
        start_epoch = sd.get("epoch", 0)
        # Resume at the saved batch position within the epoch (2026-05-29:
        # explicit user preference — don't restart the epoch from b=0 on every
        # service restart, advance the epoch counter properly). The skip loop
        # at the top of the epoch (~line 651) fast-forwards through the
        # already-completed batches; with persistent_workers + the warm OS
        # tile cache, the skip is cheap on warm caches. The old cold-cache
        # cost only mattered when the very first epoch's resume hit unread
        # NAS data; that's a one-time tax we accept to preserve epoch progress.
        stored_batch_idx = sd.get("batch_idx", 0)
        skip_batches = stored_batch_idx
        global_step = sd.get("global_step", 0)
        best_val_psnr = sd.get("best_val_psnr", -1e9)
        # Restore persisted L1 window so the spike watchdog is armed from step 1
        # of the resumed run (no 200-step blind spot).
        restored_l1_window = sd.get("l1_window")
        # Free the transient copies created while loading the checkpoint
        # (model + AdamW m/v states). Without this, the first training
        # iteration's activation peak stacks on top of the loader's leftover
        # allocations and briefly spikes ~7 GB over steady-state, OOMing us
        # right after every resume on 2026-05-28.
        del sd
        import gc as _gc
        _gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        ckpt_mgr.log_event("training_resumed", from_step=global_step,
                           from_epoch=start_epoch, from_batch=stored_batch_idx,
                           resume_path=str(resume_path))
        writer.add_text(
            "startup/resume",
            f"Resumed from `{resume_path.name}` — epoch {start_epoch}, "
            f"batch {stored_batch_idx} (skipping ahead, not replaying from b0), "
            f"step {global_step}, best PSNR {best_val_psnr:.2f} dB",
            global_step,
        )
        writer.add_scalar("startup/resume_step", global_step, 0)
        writer.add_scalar("startup/resume_epoch", start_epoch, 0)
        print(f"[resume] loaded {resume_path.name}: epoch={start_epoch}, "
              f"batch={skip_batches}, step={global_step}, best={best_val_psnr:.2f} dB",
              flush=True)
    elif args.salvage_from:
        # Overlay salvaged encoder/predictor/projection weights from a prior run.
        # Decoder + optimizer + scheduler stay fresh. global_step = 0.
        salvage_path = Path(args.salvage_from)
        salvage = torch.load(salvage_path, map_location="cpu", weights_only=False)
        salvage_sd = salvage["model"]
        result = model.load_checkpoint_state(salvage_sd, strict=False)
        missing = [k for k in result.missing_keys if not k.startswith("target_encoder.")]
        unexpected = list(result.unexpected_keys)
        print(f"[salvage] loaded {salvage_path.name} from step "
              f"{salvage.get('salvaged_step')} (epoch {salvage.get('salvaged_epoch')})")
        print(f"[salvage] kept {len(salvage_sd)} keys | missing-on-model: "
              f"{len(missing)} | unexpected: {len(unexpected)}")
        if missing:
            print(f"[salvage] (missing are fresh-init: decoder.* etc) e.g. {missing[:3]}")
        ckpt_mgr.log_event("salvage_loaded", salvage_path=str(salvage_path),
                           salvaged_from_step=salvage.get("salvaged_step"),
                           kept_keys=len(salvage_sd),
                           missing=len(missing), unexpected=len(unexpected))
        writer.add_text("startup/resume",
                        f"Salvage overlay from `{salvage_path.name}` "
                        f"(was step {salvage.get('salvaged_step')}). "
                        f"Decoder + optimizer fresh.",
                        0)
    else:
        writer.add_text("startup/resume", "no resume — starting fresh", 0)

    all_trainable = [p for g in param_groups for p in g["params"] if p.requires_grad]

    config_dict = vars(args)
    consecutive_nan = 0
    nan_skipped_total = 0
    consecutive_spikes = 0  # for watchdog "require N in a row" logic
    consecutive_skips = 0   # anti-spin: high-L1 skip shadows the spike watchdog
    last_val_step = -1      # for step-based validation (decoupled from epochs)
    # High threshold: the early step-200 dip can legitimately skip a long run of
    # hard batches and still recover (the proven run rode through with NO guard).
    # Only a TRULY stuck model (resumed-collapsed) skips this many in a row.
    MAX_CONSECUTIVE_SKIPS = 300
    l1_window: deque = deque(maxlen=args.l1_spike_window)
    # If we restored an L1 window from the resume checkpoint, repopulate the
    # deque so the watchdog activates immediately (no 200-step blind spot).
    if restored_l1_window:
        for v in restored_l1_window[-args.l1_spike_window:]:
            l1_window.append(v)
        print(f"[resume] restored l1_window with {len(l1_window)} values "
              f"(median={statistics.median(l1_window):.4f})", flush=True)
    # V4 Phase 1: track decoder lr_clamp_rate over a sliding window. Abort
    # if rate > lr_clamp_abort_rate sustained for lr_clamp_abort_window steps
    # — the sigmoid logit boundary is the most likely Phase 1 failure mode.
    lr_clamp_window: deque = deque(maxlen=args.lr_clamp_abort_window)
    feat_std_window: deque = deque(maxlen=args.feat_std_abort_window)

    def _current_residual_range(step: int) -> float:
        """Linear ramp from --residual_range_start to --residual_range_end
        across [--residual_range_ramp_start, --residual_range_ramp_end].
        Held flat before/after. v3 decoder collapses if R is large early
        (content-independent residual amplification); we grow R as the
        decoder demonstrates it's using its budget for real content."""
        a, b = args.residual_range_ramp_start, args.residual_range_ramp_end
        r_a, r_b = args.residual_range_start, args.residual_range_end
        if step <= a or b <= a:
            return r_a
        if step >= b:
            return r_b
        frac = (step - a) / (b - a)
        return r_a + frac * (r_b - r_a)

    _has_set_R = hasattr(model.decoder, "set_residual_range")
    print(f"[residual-range schedule] start={args.residual_range_start} "
          f"end={args.residual_range_end} ramp=[{args.residual_range_ramp_start}, "
          f"{args.residual_range_ramp_end}] (decoder supports schedule: {_has_set_R})",
          flush=True)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_t0 = time.time()
        running_loss = {"total": 0.0, "l1": 0.0, "ssim": 0.0, "vgg": 0.0, "hf": 0.0, "jepa": 0.0, "oob": 0.0, "vicreg": 0.0}
        running_count = 0
        # (B) per-tile L1 history for offline hard-tile analysis.
        from collections import defaultdict as _dd
        tile_l1_sum: dict = _dd(float)
        tile_l1_count: dict = _dd(int)

        for batch_idx, batch in enumerate(train_loader):
            if epoch == start_epoch and batch_idx < skip_batches:
                continue
            if batch_idx == 0 and epoch == start_epoch and skip_batches > 0:
                print(f"[resume] skipped to batch_idx={batch_idx}")
                skip_batches = 0

            if _has_set_R:
                model.decoder.set_residual_range(_current_residual_range(global_step))

            low_res = batch["low_res"].to(device, non_blocking=True)
            high_res = batch["high_res"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=args.precision != "fp32"):
                result = model(low_res, high_res)
                pred = result["pred_image"]
                # ---------------- Per-sample outlier masking (A) ----------------
                # Compute per-sample L1 to identify outliers within the batch.
                # Samples with per-sample L1 > args.per_sample_l1_max are dropped
                # from the *pixel* losses (L1, SSIM, VGG, HF, LPIPS, OOB), but
                # JEPA loss stays on the full batch (feature-space, less spike-
                # prone, and the predictor needs full-batch context anyway).
                # This handles the "1 hard sample in 18" case without throwing
                # away the whole batch.
                # Outlier detection only — no gradient needed, stay in bf16.
                with torch.no_grad():
                    per_sample_l1 = (pred.detach() - high_res).abs().mean(dim=[1, 2, 3])
                # (B) Accumulate per-tile L1 for offline hard-tile analysis.
                if "tile_idx" in batch:
                    tis = batch["tile_idx"].tolist() if hasattr(batch["tile_idx"], "tolist") else list(batch["tile_idx"])
                    psl = per_sample_l1.detach().cpu().tolist()
                    for _ti, _l in zip(tis, psl):
                        tile_l1_sum[int(_ti)] += float(_l)
                        tile_l1_count[int(_ti)] += 1
                if args.per_sample_l1_max > 0:
                    keep_mask = (per_sample_l1 < args.per_sample_l1_max)
                    n_dropped = int((~keep_mask).sum().item())
                    if n_dropped > 0:
                        if global_step % args.log_every == 0:
                            print(f"  [per-sample-mask] step={global_step} "
                                  f"dropped {n_dropped}/{pred.shape[0]} samples "
                                  f"(L1 > {args.per_sample_l1_max})", flush=True)
                        writer.add_scalar("train/per_sample_dropped",
                                          n_dropped, global_step)
                else:
                    keep_mask = None
                    n_dropped = 0

                if keep_mask is not None and n_dropped == pred.shape[0]:
                    # All samples are outliers; nothing to learn from this batch.
                    writer.add_scalar("train/full_batch_dropped", 1, global_step)
                    if global_step % args.log_every == 0:
                        print(f"  [full-batch-dropped] step={global_step} "
                              f"all {pred.shape[0]} samples L1 > "
                              f"{args.per_sample_l1_max} — skipping",
                              flush=True)
                    optimizer.zero_grad(set_to_none=True)
                    del result, pred  # free forward graph (backward skipped)
                    continue

                if keep_mask is not None and n_dropped > 0:
                    pred_loss = pred[keep_mask]
                    hr_loss = high_res[keep_mask]
                else:
                    pred_loss = pred
                    hr_loss = high_res
                # Unclamped output (low_res + delta, pre-clamp) for L1 + OOB.
                # The clamp has ZERO gradient outside [0,1]: a pixel pushed past
                # the boundary saturates at 0/1 and can NEVER recover (no grad
                # flows back). At higher LR a single step can overshoot the
                # boundary -> a few blown-out pixels crater PSNR while L1 stays
                # low (only a handful wrong) -> the LR-zone collapse. Computing
                # L1/OOB on the UNCLAMPED sum gives those pixels a restoring
                # gradient toward the in-range target, eliminating the dead zone.
                # SSIM/HF/LPIPS stay on the clamped pred (they need [0,1]).
                pred_un = result.get("pred_unclamped", pred)
                pred_un_loss = (pred_un[keep_mask]
                                if (keep_mask is not None and n_dropped > 0)
                                else pred_un)

                # L1 on the CLAMPED output (real image quality, no fighting with
                # OOB at the [0,1] boundary). OOB owns the gradient outside the
                # range; L1 owns it inside. Previous setup put both on unclamped
                # → opposite-sign gradients on dark/saturated pixels.
                l_l1 = l1_pixel_loss(pred_loss, hr_loss)
                # Skip/watchdog uses the same clamped L1.
                with torch.no_grad():
                    l1_skip_val = l_l1.detach().item()
                l_ssim = ssim_loss(pred_loss, hr_loss)
                # Skip the VGG-perceptual forward entirely when w_vgg == 0.
                # Computing it builds a full VGG-16 autograd graph (several GB
                # of activations held for backward) that contributes ZERO
                # gradient — pure waste. This was a major chunk of the ~24 GB
                # backward-pass memory peak that OOM'd us even at batch=10.
                if args.w_vgg > 0:
                    l_vgg = vgg_loss_fn(pred_loss, hr_loss)
                else:
                    l_vgg = torch.zeros((), device=device)
                # Skip the FFT2 forward entirely when w_hf == 0 (same rationale
                # as the VGG guard above — fft2 builds a non-trivial graph).
                if args.w_hf > 0:
                    l_hf = high_frequency_loss(pred_loss, hr_loss)
                else:
                    l_hf = torch.zeros((), device=device)
                # JEPA: single-layer (v3) or hierarchical (v4 with --hierarchical_jepa).
                if getattr(args, "hierarchical_jepa", False) and "pred_layers" in result:
                    l_jepa = hierarchical_jepa_loss(result["pred_layers"],
                                                    result["target_layers"])
                else:
                    l_jepa = result["jepa_loss"]
                # VICReg variance+covariance anti-collapse on projected features.
                if args.w_vicreg > 0 and "projected" in result:
                    l_vicreg = vicreg_loss(result["projected"])
                else:
                    l_vicreg = torch.zeros((), device=device)
                # OOB on the UNCLAMPED output — this is the explicit restoring
                # penalty for delta that overshoots [0,1] (the LR-zone collapse).
                # On the clamped pred it was always 0 (a wiring bug that left
                # this guardrail disabled the whole time).
                l_oob = out_of_range_loss(pred_un_loss)
                # Optional LPIPS perceptual loss (v4 Phase 1+).
                if lpips_loss_fn is not None:
                    l_lpips = lpips_loss_fn(pred_loss.float().clamp(0, 1), hr_loss.float())
                else:
                    l_lpips = torch.zeros((), device=device)
                # Linear ramp of w_lpips from 0 -> args.w_lpips over l1_warmup_steps.
                # Real-ESRGAN convention: while the decoder still outputs near-noise,
                # the LPIPS gradient is large and uninformative; let L1+SSIM shape the
                # image first, then phase in perceptual signal.
                if args.l1_warmup_steps > 0:
                    w_lpips_eff = args.w_lpips * min(1.0, global_step / args.l1_warmup_steps)
                else:
                    w_lpips_eff = args.w_lpips
                total = (
                    args.w_l1 * l_l1 + args.w_ssim * l_ssim + args.w_vgg * l_vgg
                    + args.w_hf * l_hf + args.w_jepa * l_jepa + args.w_oob * l_oob
                    + w_lpips_eff * l_lpips + args.w_vicreg * l_vicreg
                )
                total_scaled = total / args.grad_accum

            # ---------------- High-L1 gradient skip ----------------
            # Anomalously high batch L1 (>= --skip_high_l1) is treated as
            # "this batch's gradient is likely unreliable; don't update the model
            # from it". We skip backward + optimizer step entirely, keeping
            # Adam momentum clean. Different from the spike watchdog which
            # ABORTS — this just drops bad batches without aborting.
            l1_now_for_skip = l1_skip_val  # clamped-output L1 (see note at l_l1)
            if (args.skip_high_l1 > 0
                    and l1_now_for_skip > args.skip_high_l1):
                consecutive_skips += 1
                writer.add_scalar("train/skipped_batch_l1", l1_now_for_skip, global_step)
                if consecutive_skips % args.log_every == 1 or consecutive_skips <= 3:
                    print(f"  [batch-skip {consecutive_skips}] step={global_step} "
                          f"batch={batch_idx} L1={l1_now_for_skip:.4f} > "
                          f"{args.skip_high_l1} — skipping (Adam preserved)",
                          flush=True)
                # Anti-spin: the high-L1 skip runs BEFORE the spike watchdog and
                # `continue`s, so a collapsed model (every batch > skip_high_l1)
                # would skip forever with global_step frozen (the silent 100%-GPU
                # spin). Bail after MAX_CONSECUTIVE_SKIPS so systemd restarts with
                # a fresh seed + a healthy step_slot instead of hanging.
                if consecutive_skips >= MAX_CONSECUTIVE_SKIPS:
                    ckpt_mgr.log_event("collapse_skip_spiral", global_step=global_step,
                                       consecutive_skips=consecutive_skips,
                                       l1=l1_now_for_skip)
                    raise RuntimeError(
                        f"Collapse: {consecutive_skips} consecutive batches with "
                        f"L1 > {args.skip_high_l1} (last={l1_now_for_skip:.4f}) at "
                        f"step={global_step}. Bailing for systemd restart.")
                optimizer.zero_grad(set_to_none=True)
                # Free the forward graph: backward() is skipped on this path, so
                # these locals would otherwise keep the full activation graph
                # alive into the next iteration's forward -> two graphs
                # co-resident -> ~2x activations -> OOM. Root cause of the
                # sporadic 22.9 GB OOMs that fired one step after a hard batch.
                del result, pred, pred_loss, pred_un, pred_un_loss, total, \
                    total_scaled, l_l1, l_ssim, l_vgg, l_hf, l_jepa, l_oob, \
                    l_lpips, l_vicreg
                continue
            consecutive_skips = 0  # streak broken: this batch is being used

            if torch.isnan(total) or torch.isinf(total):
                consecutive_nan += 1
                nan_skipped_total += 1
                losses = {"l1": l_l1.item(), "ssim": l_ssim.item(), "vgg": l_vgg.item(),
                          "hf": l_hf.item(), "jepa": l_jepa.item()}
                writer.add_scalar("train/nan_skipped_total", nan_skipped_total, global_step)
                writer.add_scalar("train/consecutive_nan", consecutive_nan, global_step)
                print(f"  [nan-skip {consecutive_nan}/{args.max_consecutive_nan}] "
                      f"step={global_step} batch={batch_idx} losses={losses}", flush=True)
                if consecutive_nan >= args.max_consecutive_nan:
                    nan_path = ckpt_mgr.save_nan_checkpoint(
                        model, optimizer, scheduler, scaler, epoch, batch_idx,
                        global_step, best_val_psnr, config_dict, losses,
                        l1_window=list(l1_window), ema=ema,
                    )
                    writer.add_text("events/nan",
                                    f"{consecutive_nan} consecutive NaNs at step {global_step}. "
                                    f"Pre-NaN checkpoint: {nan_path}. Bailing for systemd to restart.",
                                    global_step)
                    raise RuntimeError(
                        f"NaN streak ({consecutive_nan}) at step={global_step} losses={losses}"
                    )
                optimizer.zero_grad(set_to_none=True)
                # Same graph-leak fix as the high-L1 skip path above.
                del result, pred, pred_loss, pred_un, pred_un_loss, total, \
                    total_scaled, l_l1, l_ssim, l_vgg, l_hf, l_jepa, l_oob, \
                    l_lpips, l_vicreg
                continue
            else:
                consecutive_nan = 0

            if use_scaler:
                scaler.scale(total_scaled).backward()
            else:
                total_scaled.backward()

            if (batch_idx + 1) % args.grad_accum == 0:
                if use_scaler:
                    scaler.unscale_(optimizer)
                grad_norm = nn.utils.clip_grad_norm_(all_trainable, args.grad_clip)
                # Skip the step if grads themselves contain non-finite values.
                if not torch.isfinite(grad_norm):
                    writer.add_scalar("train/skipped_step_bad_grad", 1, global_step)
                    optimizer.zero_grad(set_to_none=True)
                    if use_scaler:
                        scaler.update()
                    continue
                if use_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                # Periodic empty_cache to cap reserved-pool fragmentation.
                # Real-data tiles have varying complexity → varying intermediate
                # tensor sizes → the allocator's reserved pool slowly grows past
                # the actual allocated need (3.3 GB) toward the 24 GB ceiling.
                # Flushing the cache every N steps forces reserved back down.
                # Cheap relative to a full step; prevents the slow-OOM creep.
                if (args.empty_cache_every > 0
                        and global_step % args.empty_cache_every == 0
                        and torch.cuda.is_available()):
                    torch.cuda.empty_cache()

                ema_decay = cosine_ema_decay(global_step, total_steps,
                                             args.ema_start, args.ema_end)
                model.update_target_ema(ema_decay)

                # Update v4 EMA shadow (separate from target-encoder EMA above).
                if ema is not None:
                    ema_d = ema.update(model, global_step)
                    if ema_d is not None and global_step % args.log_every == 0:
                        writer.add_scalar("train/ema_decay", ema_d, global_step)

                if global_step % args.recovery_every == 0:
                    # SACRED HEALTHY-SLOT GUARD: only save the step_slot if the
                    # model is currently healthy. Defined as median(l1_window) < 0.20
                    # (vs typical healthy of ~0.08, vs collapse ~0.55+). Prevents
                    # rotation from overwriting healthy slots with collapsed weights
                    # — the exact bug we hit on 2026-05-27 ~19:50 where 6 collapsed
                    # step_slots overwrote all healthy history.
                    save_l1_med = (statistics.median(l1_window) if len(l1_window) >= 5
                                   else l1_now)
                    if save_l1_med < args.save_step_l1_max:
                        ckpt_mgr.save_step(model, optimizer, scheduler, scaler,
                                           epoch, batch_idx, global_step,
                                           best_val_psnr, config_dict,
                                           l1_window=list(l1_window), ema=ema)
                    else:
                        print(f"  [step_slot skip] step={global_step} "
                              f"median_l1={save_l1_med:.4f} > "
                              f"{args.save_step_l1_max} (unhealthy — preserving slot)",
                              flush=True)

            # ---------------- L1-spike watchdog (CONSECUTIVE-spike requirement) ----------------
            # Single-batch L1 spikes are common in SR training: a particular batch
            # with hard texture (text, fine signage, dense grilles) produces L1
            # 3-4x median, but the *next* batch returns to normal. v3-style
            # collapse, by contrast, produces SUSTAINED high L1 over many batches.
            #
            # Watchdog now requires N consecutive spike-level batches (controlled
            # by --l1_spike_consecutive, default 2). This eliminates the
            # false-positive crashes we saw on 2026-05-27 22:00 where the watchdog
            # fired on single hard batches.
            l1_now = l1_skip_val  # clamped-output L1 (see note at l_l1)

            # Did THIS batch trigger either the absolute or ratio check?
            spike_this_batch = False
            spike_reason = ""
            spike_meta = {}

            if args.l1_spike_absolute > 0 and l1_now > args.l1_spike_absolute:
                spike_this_batch = True
                spike_reason = "absolute"
                spike_meta = {"l1": l1_now, "threshold": args.l1_spike_absolute,
                              "trigger": "absolute"}

            if (not spike_this_batch
                    and global_step >= args.l1_spike_warmup
                    and len(l1_window) == l1_window.maxlen):
                median_l1 = statistics.median(l1_window)
                if l1_now > args.l1_spike_ratio * median_l1 and median_l1 > 0:
                    spike_this_batch = True
                    spike_reason = "ratio"
                    spike_meta = {"l1": l1_now, "median_l1": median_l1,
                                  "ratio": l1_now / median_l1, "trigger": "ratio"}

            if spike_this_batch:
                consecutive_spikes += 1
                print(f"  [l1-spike-{spike_reason} {consecutive_spikes}/"
                      f"{args.l1_spike_consecutive}] step={global_step} L1={l1_now:.4f}"
                      + (f" median={spike_meta['median_l1']:.4f} "
                         f"ratio={spike_meta['ratio']:.2f}"
                         if spike_reason == "ratio" else ""),
                      flush=True)
            else:
                consecutive_spikes = 0

            if spike_this_batch and consecutive_spikes >= args.l1_spike_consecutive:
                if spike_reason == "absolute":
                    spike_path = ckpt_mgr.save_nan_checkpoint(
                        model, optimizer, scheduler, scaler, epoch, batch_idx,
                        global_step, best_val_psnr, config_dict,
                        losses=spike_meta,
                        l1_window=list(l1_window), ema=ema,
                    )
                    ckpt_mgr.log_event("l1_spike_absolute", global_step=global_step,
                                       l1=l1_now, threshold=args.l1_spike_absolute,
                                       spike_ckpt=str(spike_path),
                                       consecutive=consecutive_spikes)
                    writer.add_text(
                        "events/l1_spike_absolute",
                        f"L1 absolute spike at step {global_step}: cur={l1_now:.4f} "
                        f"> threshold={args.l1_spike_absolute} for "
                        f"{consecutive_spikes} consecutive batches. Aborting.",
                        global_step,
                    )
                    raise RuntimeError(
                        f"L1 absolute spike: cur={l1_now:.4f} > "
                        f"threshold={args.l1_spike_absolute} for "
                        f"{consecutive_spikes} consecutive batches. "
                        f"Pre-spike snapshot: {spike_path}"
                    )
                else:  # ratio
                    median_l1 = spike_meta["median_l1"]
                    spike_path = ckpt_mgr.save_nan_checkpoint(
                        model, optimizer, scheduler, scaler, epoch, batch_idx,
                        global_step, best_val_psnr, config_dict,
                        losses=spike_meta,
                        l1_window=list(l1_window), ema=ema,
                    )
                    ckpt_mgr.log_event("l1_spike", global_step=global_step,
                                       l1=l1_now, median_l1=median_l1,
                                       ratio=l1_now / median_l1,
                                       spike_ckpt=str(spike_path),
                                       consecutive=consecutive_spikes)
                    writer.add_text(
                        "events/l1_spike",
                        f"L1 spike at step {global_step}: cur={l1_now:.4f} "
                        f"median(last {l1_window.maxlen})={median_l1:.4f} "
                        f"ratio={l1_now/median_l1:.2f} for "
                        f"{consecutive_spikes} consecutive batches. Aborting.",
                        global_step,
                    )
                    raise RuntimeError(
                        f"L1 spike: cur={l1_now:.4f} > {args.l1_spike_ratio} * "
                        f"median={median_l1:.4f} for {consecutive_spikes} "
                        f"consecutive batches. Pre-spike snapshot: {spike_path}"
                    )
            l1_window.append(l1_now)

            running_count += 1
            running_loss["total"] += total.item()
            running_loss["l1"] += l_l1.item()
            running_loss["ssim"] += l_ssim.item()
            running_loss["vgg"] += l_vgg.item()
            running_loss["hf"] += l_hf.item()
            running_loss["jepa"] += l_jepa.item()
            running_loss["oob"] += l_oob.item()
            running_loss["vicreg"] += l_vicreg.item()
            if "lpips" not in running_loss:
                running_loss["lpips"] = 0.0
            running_loss["lpips"] += float(l_lpips.item())

            # lr_clamp_rate (v4 decoder only). Abort if sustained > threshold.
            if "lr_clamp_rate" in result:
                rate = float(result["lr_clamp_rate"].item())
                lr_clamp_window.append(rate)
                if global_step % args.log_every == 0:
                    writer.add_scalar("train/lr_clamp_rate", rate, global_step)
                if (len(lr_clamp_window) == lr_clamp_window.maxlen
                        and global_step >= args.l1_spike_warmup
                        and statistics.mean(lr_clamp_window)
                            > args.lr_clamp_abort_rate):
                    mean_rate = statistics.mean(lr_clamp_window)
                    writer.add_text("events/lr_clamp_abort",
                                    f"lr_clamp_rate mean over last "
                                    f"{lr_clamp_window.maxlen} steps = "
                                    f"{mean_rate:.4f} > "
                                    f"{args.lr_clamp_abort_rate}. Aborting.",
                                    global_step)
                    raise RuntimeError(
                        f"lr_clamp_rate sustained > {args.lr_clamp_abort_rate} "
                        f"({mean_rate:.4f}) for {lr_clamp_window.maxlen} steps "
                        f"— sigmoid logit boundary saturating. Phase 1 watchdog "
                        f"per V4_PLAN §2.3."
                    )

            # feature-collapse abort (Change 4): sustained near-zero per-dim
            # feature std == the JEPA representation has collapsed to a constant.
            if "ctx_std" in result:
                feat_std_window.append(float(result["ctx_std"].item()))
                if (len(feat_std_window) == feat_std_window.maxlen
                        and global_step >= args.l1_spike_warmup
                        and statistics.mean(feat_std_window)
                            < args.feat_std_abort_min):
                    mean_fstd = statistics.mean(feat_std_window)
                    ckpt_mgr.log_event("feature_collapse", global_step=global_step,
                                       mean_ctx_std=mean_fstd)
                    writer.add_text("events/feature_collapse",
                                    f"ctx_std mean over last {feat_std_window.maxlen} "
                                    f"steps = {mean_fstd:.5f} < "
                                    f"{args.feat_std_abort_min} — representation "
                                    f"collapse. Bailing for systemd restart.",
                                    global_step)
                    raise RuntimeError(
                        f"Feature collapse: ctx_std {mean_fstd:.5f} < "
                        f"{args.feat_std_abort_min} for {feat_std_window.maxlen} "
                        f"steps at step={global_step}.")

            if batch_idx % args.log_every == 0:
                with torch.no_grad():
                    cur_psnr = psnr_metric(pred.float().clamp(0, 1), high_res.float()).item()
                lr_dec = optimizer.param_groups[0]["lr"]
                lr_enc = optimizer.param_groups[-1]["lr"]
                cos_sim = result["cos_sim"].item()
                _r_now = getattr(model.decoder, "RESIDUAL_RANGE", None)
                _r_str = f" | R={_r_now:.3f}" if _r_now is not None else ""
                print(
                    f"E{epoch} | b{batch_idx}/{len(train_loader)} | step={global_step} | "
                    f"PSNR={cur_psnr:.2f}dB | L1={l_l1.item():.4f} | SSIM_loss={l_ssim.item():.4f} | "
                    f"VGG={l_vgg.item():.4f} | LPIPS={l_lpips.item():.4f} | "
                    f"HF={l_hf.item():.4f} | JEPA={l_jepa.item():.4f} | "
                    f"OOB={l_oob.item():.4f} | VIC={l_vicreg.item():.4f} | "
                    f"cos={cos_sim:.3f} | lr_dec={lr_dec:.2e} | lr_enc={lr_enc:.2e}{_r_str}",
                    flush=True,
                )
                if _r_now is not None:
                    writer.add_scalar("train/residual_range", _r_now, global_step)
                if torch.cuda.is_available():
                    _al = torch.cuda.memory_allocated() / 1e9
                    _rs = torch.cuda.memory_reserved() / 1e9
                    _pk = torch.cuda.max_memory_allocated() / 1e9
                    print(f"  [MEM] step={global_step} alloc={_al:.2f}GB "
                          f"reserved={_rs:.2f}GB peak={_pk:.2f}GB", flush=True)
                    # When we approach the ceiling, dump the allocator breakdown
                    # ONCE so we can see WHAT category holds the memory (the
                    # offline trace peaks at only ~19.5 GB; the live process
                    # OOMs at ~22.9 GB and we need to see the missing ~3 GB).
                    if _pk > 20.0 and not getattr(main, "_mem_dumped", False):
                        main._mem_dumped = True
                        print(torch.cuda.memory_summary(abbreviated=True),
                              flush=True)
                writer.add_scalar("train/psnr", cur_psnr, global_step)
                writer.add_scalar("train/l1", l_l1.item(), global_step)
                writer.add_scalar("train/ssim_loss", l_ssim.item(), global_step)
                writer.add_scalar("train/vgg", l_vgg.item(), global_step)
                writer.add_scalar("train/hf", l_hf.item(), global_step)
                writer.add_scalar("train/jepa", l_jepa.item(), global_step)
                writer.add_scalar("train/oob", l_oob.item(), global_step)
                writer.add_scalar("train/vicreg", l_vicreg.item(), global_step)
                writer.add_scalar("train/lpips", float(l_lpips.item()), global_step)
                writer.add_scalar("train/cos_sim", cos_sim, global_step)
                writer.add_scalar("train/lr_decoder", lr_dec, global_step)
                writer.add_scalar("train/lr_encoder", lr_enc, global_step)
                # --- collapse diagnostics (Change 4) ---
                _ctx_std = result["ctx_std"].item()
                _pf_std = result["pred_feat_std"].item()
                _do_std = result.get("pred_unclamped_std", torch.zeros(())).item()
                _oob = result.get("oob_frac", torch.zeros(())).item()
                print(f"  [diag] ctx_std={_ctx_std:.4f} pred_feat_std={_pf_std:.4f} "
                      f"dec_out_std={_do_std:.4f} oob_frac={_oob:.4f}", flush=True)
                writer.add_scalar("train/ctx_std", _ctx_std, global_step)
                writer.add_scalar("train/pred_feat_std", _pf_std, global_step)
                writer.add_scalar("train/dec_out_std", _do_std, global_step)
                writer.add_scalar("train/oob_frac", _oob, global_step)

            # --- periodic STEP-based validation (decoupled from epoch end) ---
            # Epoch-end-only validation can be starved forever: a collapse resets
            # batch_idx to 0, so if the model can't complete an epoch it never
            # validates. global_step survives restarts, so poll on it instead —
            # this guarantees a gate signal + best.pt regardless of epoch completion.
            if (args.val_every_steps > 0 and global_step > 0
                    and global_step % args.val_every_steps == 0
                    and global_step != last_val_step):
                last_val_step = global_step
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                vm = validate(model, val_loader, device, vgg_loss_fn, writer=writer,
                              global_step=global_step, log_images=0,
                              amp_dtype=amp_dtype,
                              autocast_enabled=args.precision != "fp32",
                              ema=ema)
                model.train()  # validate() left the model in eval mode
                _bp = vm.get('bilinear_psnr', float('nan'))
                _bl = vm.get('bilinear_lpips', float('nan'))
                print(f"--- [step-val] step={global_step} PSNR={vm['psnr']:.2f}dB "
                      f"(bilinear {_bp:.2f}) L1={vm['l1']:.4f} SSIM={vm['ssim']:.4f} "
                      f"LPIPS_vgg={vm.get('lpips_vgg', float('nan')):.4f} "
                      f"lpips_alex={vm.get('lpips', float('nan')):.4f} (bilinear {_bl:.4f}) "
                      f"[gate: beat bilinear on PSNR & LPIPS] ---", flush=True)
                for k, v in vm.items():
                    writer.add_scalar(f"val/{k}", v, global_step)
                if vm["psnr"] > best_val_psnr:
                    best_val_psnr = vm["psnr"]
                    ckpt_mgr.save_epoch(model, optimizer, scheduler, scaler,
                                        epoch, global_step, vm["psnr"],
                                        best_val_psnr, config_dict,
                                        extra_metrics=vm, ema=ema)
                    print(f"  [step-val] NEW BEST val PSNR {best_val_psnr:.2f} dB "
                          f"-> saved best.pt", flush=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if args.smoke_test and global_step >= 20:
                break

        train_time = time.time() - epoch_t0
        avg = {k: v / max(1, running_count) for k, v in running_loss.items()}
        print(f"--- E{epoch} train: {train_time/60:.1f}m | avg L1={avg['l1']:.4f} ---", flush=True)
        # (B) Dump per-tile L1 stats for offline hard-tile analysis.
        if tile_l1_count:
            tile_stats = {str(ti): {
                "mean": tile_l1_sum[ti] / tile_l1_count[ti],
                "n": tile_l1_count[ti],
            } for ti in tile_l1_count}
            tile_stats_path = (ckpt_mgr.exp_dir / f"tile_l1_e{epoch}.json")
            try:
                json.dump(tile_stats, open(tile_stats_path, "w"))
                # Compute top-5% by mean L1 for quick visibility
                sorted_tiles = sorted(tile_stats.items(),
                                      key=lambda kv: kv[1]["mean"], reverse=True)
                top_n = max(1, len(sorted_tiles) // 20)
                print(f"  [tile-stats] wrote {tile_stats_path.name}; "
                      f"top {top_n} hardest tiles: " +
                      " ".join(f"{ti}({s['mean']:.3f})" for ti, s in sorted_tiles[:top_n]))
            except Exception as e:
                print(f"  [tile-stats] failed to write: {e}")

        # Release the training reserved pool before validation. At batch=20 the
        # train pool is ~20.8 GB reserved; validation lazily loads the LPIPS
        # perceptual nets (~1.5 GB) + runs a fresh-shaped no_grad forward, so
        # without this flush the first end-of-epoch validation can exceed 24 GB.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        val_t0 = time.time()
        metrics = validate(model, val_loader, device, vgg_loss_fn, writer=writer,
                           global_step=global_step, log_images=args.val_log_images,
                           amp_dtype=amp_dtype,
                           autocast_enabled=args.precision != "fp32",
                           ema=ema)
        val_time = time.time() - val_t0
        print(
            f"--- E{epoch} val ({val_time/60:.1f}m): PSNR={metrics['psnr']:.2f}dB "
            f"SSIM={metrics['ssim']:.4f} L1={metrics['l1']:.4f} VGG={metrics['vgg']:.4f} ---",
            flush=True,
        )
        for k, v in metrics.items():
            writer.add_scalar(f"val/{k}", v, global_step)
        writer.add_scalar("val/epoch", epoch, global_step)
        writer.add_text(
            "events/epoch_complete",
            f"Epoch {epoch}: val PSNR={metrics['psnr']:.2f} dB, "
            f"SSIM={metrics['ssim']:.4f}, L1={metrics['l1']:.4f}, "
            f"VGG={metrics['vgg']:.4f}",
            global_step,
        )

        if metrics["psnr"] > best_val_psnr:
            best_val_psnr = metrics["psnr"]
            writer.add_scalar("val/best_psnr", best_val_psnr, global_step)

        ckpt_mgr.save_epoch(model, optimizer, scheduler, scaler,
                            epoch, global_step, metrics["psnr"],
                            best_val_psnr, config_dict,
                            extra_metrics=metrics, ema=ema)

        if args.smoke_test:
            break

    ckpt_mgr.join_pending_saves()
    ckpt_mgr.log_event("training_complete", best_val_psnr=best_val_psnr,
                       final_step=global_step)
    writer.close()
    print(f"Best val PSNR: {best_val_psnr:.2f} dB", flush=True)


if __name__ == "__main__":
    try:
        main()
    except torch.cuda.OutOfMemoryError:
        # Dump the recorded allocation history + a summary so we can see exactly
        # what held memory at the OOM. Snapshot retains peak history even though
        # the stack has unwound. Re-raise so systemd still treats this as failure.
        try:
            import torch as _t
            ts = time.strftime("%H%M%S")
            snap = f"runs/oom_snapshot_{ts}.pickle"
            _t.cuda.memory._dump_snapshot(snap)
            print(f"[mem-debug] OOM snapshot written to {snap}", flush=True)
            print(_t.cuda.memory_summary(), flush=True)
        except Exception as _e:
            print(f"[mem-debug] snapshot dump failed: {_e}", flush=True)
        raise
