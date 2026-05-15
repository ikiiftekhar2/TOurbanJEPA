"""
Phase 4: Full D-JEPA Training

Phase 4a: Train denoising MLP with frozen JEPA (50 epochs).
Phase 4b: Joint fine-tune everything (20 epochs).

Loss: L = Ld (diffusion MSE on VAE latent tokens) + Lp (Smooth L1 prediction).

Usage:
    # Phase 4a (MLP only)
    python -m src.training.train_djepa --phase 4a --data_dir data/ortho

    # Phase 4b (joint, resume from 4a checkpoint)
    python -m src.training.train_djepa --phase 4b --data_dir data/ortho \\
        --resume models/checkpoints/djepa_mlp_best.pt

TensorBoard:
    tensorboard --logdir runs/
"""

import argparse
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.urbanjepa import UrbanJEPA
from src.training.losses import LinearNoiseSchedule
from src.data.ortho_dataset import OrthoDataset


# ---------------------------------------------------------------------------
# EMA schedule: cosine from start → end (I-JEPA paper)
# ---------------------------------------------------------------------------

def ema_cosine(step, total_steps, start=0.996, end=1.0):
    tau = min(step / max(total_steps, 1), 1.0)
    return end - (end - start) * (math.cos(math.pi * tau) + 1.0) / 2.0


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(description="Phase 4: D-JEPA training")
    p.add_argument("--phase", type=str, required=True, choices=["4a", "4b"])
    p.add_argument("--data_dir", type=str, default="data/ortho")
    p.add_argument("--pretrained", type=str,
                   default="models/ijepa/vit_base_patch16_224_imagenet.pt")
    p.add_argument("--jepa_ckpt", type=str,
                   default="models/checkpoints/jepa_best.pt",
                   help="Phase 2 JEPA checkpoint")
    p.add_argument("--checkpoint_dir", type=str, default="models/checkpoints")
    p.add_argument("--resume", type=str, default=None,
                   help="Resume from Phase 4 checkpoint")
    p.add_argument("--log_dir", type=str, default="runs")

    # Hyperparams
    p.add_argument("--epochs", type=int, default=None,
                   help="Override default (4a=50, 4b=20)")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr_mlp", type=float, default=1e-3)
    p.add_argument("--lr_jepa", type=float, default=1e-4)
    p.add_argument("--lr_eta_min", type=float, default=1e-6,
                   help="Minimum LR for cosine annealing (default: 1e-6)")
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--patches_per_epoch", type=int, default=4)
    p.add_argument("--val_patches_per_tile", type=int, default=4)

    # Logging / checkpointing
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--checkpoint_every", type=int, default=5)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--no_early_stop", action="store_true", default=False)

    # Diffusion
    p.add_argument("--diffusion_T", type=int, default=1000)
    p.add_argument("--beta_start", type=float, default=1e-4)
    p.add_argument("--beta_end", type=float, default=2e-2)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Checkpoint save/load
# ---------------------------------------------------------------------------

def save_ckpt(model, optimizer, scheduler, epoch, path, phase, ema_val):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "phase": phase,
        "ema_decay": ema_val,
        "context_encoder": model.context_encoder.state_dict(),
        "target_encoder": model.target_encoder.state_dict(),
        "feature_predictor": model.feature_predictor.state_dict(),
        "projection_head": model.projection_head.state_dict(),
        "denoising_mlp": model.denoising_mlp.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
    }, path)
    print(f"  Saved: {path}")


def load_jepa(model, path, device="cuda"):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.context_encoder.load_state_dict(ckpt["context_encoder"])
    model.target_encoder.load_state_dict(ckpt["target_encoder"])
    model.feature_predictor.load_state_dict(ckpt["feature_predictor"])
    model.projection_head.load_state_dict(ckpt["projection_head"])
    # Also load denoising MLP if present (enables 4a→4b transition)
    if "denoising_mlp" in ckpt:
        model.denoising_mlp.load_state_dict(ckpt["denoising_mlp"])
        print(f"  JEPA + MLP weights from {path}")
    else:
        print(f"  JEPA weights from {path}")


def load_phase4(model, optimizer, scheduler, path, device="cuda"):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.context_encoder.load_state_dict(ckpt["context_encoder"])
    model.target_encoder.load_state_dict(ckpt["target_encoder"])
    model.feature_predictor.load_state_dict(ckpt["feature_predictor"])
    model.projection_head.load_state_dict(ckpt["projection_head"])
    model.denoising_mlp.load_state_dict(ckpt["denoising_mlp"])
    if optimizer:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and ckpt.get("scheduler"):
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt["epoch"] + 1, ckpt.get("phase", "4a"), ckpt.get("ema_decay", 0.9999)


# ---------------------------------------------------------------------------
# VAE
# ---------------------------------------------------------------------------

def load_vae(device="cuda"):
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16
    ).to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False
    print(f"  VAE loaded ({sum(p.numel() for p in vae.parameters()):,} params)")
    return vae


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    phase_id = "mlp" if args.phase == "4a" else "joint"
    epochs = args.epochs or (50 if args.phase == "4a" else 20)

    print(f"Device: {device}")
    print(f"Phase:  {args.phase} ({phase_id}), {epochs} epochs")
    print(f"Data:   {args.data_dir}")

    # --- Datasets ---
    train_ds = OrthoDataset(args.data_dir, split="train", train_ratio=0.9,
                            augment=True, patches_per_epoch=args.patches_per_epoch)
    val_ds = OrthoDataset(args.data_dir, split="val", train_ratio=0.9,
                          augment=False, val_patches_per_tile=args.val_patches_per_tile)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False)
    print(f"Train: {len(train_ds):,} samples, {len(train_loader)} batches")
    print(f"Val:   {len(val_ds):,} samples, {len(val_loader)} batches")

    # --- Model ---
    model = UrbanJEPA(pretrained_path=args.pretrained).to(device)
    load_jepa(model, args.jepa_ckpt, device)

    # VAE
    vae = load_vae(device)
    model.load_vae(vae)

    # Configure trainability
    model.train_for_phase(phase_id)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable: {sum(p.numel() for p in trainable):,} params")

    # --- Optimizer ---
    if args.phase == "4a":
        optimizer = AdamW(model.denoising_mlp.parameters(),
                          lr=args.lr_mlp, weight_decay=args.weight_decay,
                          betas=(0.9, 0.95))
    else:
        optimizer = AdamW([
            {"params": model.denoising_mlp.parameters(), "lr": args.lr_mlp},
            {"params": model.context_encoder.parameters(), "lr": args.lr_jepa},
            {"params": model.feature_predictor.parameters(), "lr": args.lr_jepa},
            {"params": model.projection_head.parameters(), "lr": args.lr_jepa},
        ], weight_decay=args.weight_decay, betas=(0.9, 0.95))

    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=args.lr_eta_min)

    # Resume
    start_epoch = 0
    ema_decay = 0.9999
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        saved_phase = ckpt.get("phase", args.phase)
        # Load model weights
        model.context_encoder.load_state_dict(ckpt["context_encoder"])
        model.target_encoder.load_state_dict(ckpt["target_encoder"])
        model.feature_predictor.load_state_dict(ckpt["feature_predictor"])
        model.projection_head.load_state_dict(ckpt["projection_head"])
        model.denoising_mlp.load_state_dict(ckpt["denoising_mlp"])
        # Only load optimizer/scheduler if same phase (same param groups)
        if saved_phase == args.phase:
            optimizer.load_state_dict(ckpt["optimizer"])
            if scheduler and ckpt.get("scheduler"):
                scheduler.load_state_dict(ckpt["scheduler"])
            start_epoch = ckpt["epoch"] + 1
            status = f"Resumed {saved_phase} from epoch {start_epoch}"
        else:
            status = f"Loaded {saved_phase} weights for {args.phase} (fresh optimizer)"
        ema_decay = ckpt.get("ema_decay", 0.9999)
        model.train_for_phase(phase_id)
        print(status)

    # --- Noise schedule ---
    noise_schedule = LinearNoiseSchedule(
        T=args.diffusion_T, beta_start=args.beta_start,
        beta_end=args.beta_end, device=device, schedule="cosine")

    # --- TensorBoard ---
    run_name = f"phase{args.phase}_{time.strftime('%b%d_%H%M')}"
    writer = SummaryWriter(log_dir=str(Path(args.log_dir) / run_name))
    print(f"TensorBoard: {args.log_dir}/{run_name}")

    # --- Training state ---
    accum = args.grad_accum
    eff_bs = args.batch_size * accum
    steps_per_epoch = len(train_loader) // accum
    total_steps = epochs * steps_per_epoch
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    global_step = start_epoch * steps_per_epoch

    best_val = float("inf")
    best_epoch = -1
    patience_ctr = 0

    print(f"Batch: {args.batch_size} x {accum} accum = {eff_bs} effective")
    print(f"Steps/epoch: {steps_per_epoch}, total: ~{total_steps}")

    for epoch in range(start_epoch, epochs):
        model.train()
        t0 = time.time()
        total_batches = len(train_loader)
        epoch_ld = epoch_lp = 0.0
        accum_count = 0
        optimizer.zero_grad()

        # --- Train loop ---
        for bi, batch in enumerate(train_loader):
            low = batch["low_res"].to(device, non_blocking=True)
            high = batch["high_res"].to(device, non_blocking=True)
            B, N = low.shape[0], model.num_patches

            # ---- JEPA pathway ----
            if args.phase == "4a":
                with torch.no_grad():
                    ctx = model.context_encoder(low)
                    tgt_emb = model.target_encoder(high)
                    high_tok = model.encode_to_latent(high)
                    mask, _ = model.sample_mask(N)
                    mask_b = mask.to(device).unsqueeze(0).expand(B, -1)
                    pred = model.feature_predictor(ctx, mask_b)
                    tgt_masked = tgt_emb.gather(
                        1, mask_b.unsqueeze(-1).expand(-1, -1, model.embed_dim))
                    Lp = model.prediction_loss(pred, tgt_masked)
                tok_masked = high_tok.gather(
                    1, mask_b.unsqueeze(-1).expand(-1, -1, model.token_dim))
                Ld = model.diffusion_loss(tok_masked, pred, noise_schedule)
                loss = Ld  # MLP only
            else:
                ctx = model.context_encoder(low)
                with torch.no_grad():
                    tgt_emb = model.target_encoder(high)
                    high_tok = model.encode_to_latent(high)
                mask, _ = model.sample_mask(N)
                mask_b = mask.to(device).unsqueeze(0).expand(B, -1)
                pred = model.feature_predictor(ctx, mask_b)
                tgt_masked = tgt_emb.gather(
                    1, mask_b.unsqueeze(-1).expand(-1, -1, model.embed_dim))
                Lp = model.prediction_loss(pred, tgt_masked)
                tok_masked = high_tok.gather(
                    1, mask_b.unsqueeze(-1).expand(-1, -1, model.token_dim))
                Ld = model.diffusion_loss(tok_masked, pred, noise_schedule)
                loss = Ld + Lp

            epoch_ld += Ld.item()
            epoch_lp += Lp.item()

            scaled = loss / accum
            if scaler:
                scaler.scale(scaled).backward()
            else:
                scaled.backward()
            accum_count += 1

            if accum_count == accum:
                if scaler:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(trainable, args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(trainable, args.grad_clip)
                    optimizer.step()
                optimizer.zero_grad()
                accum_count = 0
                global_step += 1

                if args.phase == "4b":
                    ema_decay = ema_cosine(global_step, total_steps)
                    model.ema_decay = ema_decay
                    model.update_target_encoder()

            # Log
            if bi > 0 and bi % args.log_every == 0:
                n = bi + 1
                avg_ld = epoch_ld / n
                avg_lp = epoch_lp / n
                lr = optimizer.param_groups[0]["lr"]
                elapsed = time.time() - t0
                eta = (elapsed / n) * (total_batches - n)
                tag = f"EMA={ema_decay:.4f} " if args.phase == "4b" else ""
                print(f"  E{epoch:3d} S{bi:4d}/{total_batches} "
                      f"Ld={Ld.item():.4f} Lp={Lp.item():.4f} "
                      f"avg Ld={avg_ld:.4f} Lp={avg_lp:.4f} "
                      f"lr={lr:.1e} {tag}ETA {eta/60:.0f}m{eta%60:.0f}s")
                writer.add_scalar("train/Ld", Ld.item(), global_step)
                writer.add_scalar("train/Lp", Lp.item(), global_step)
                writer.add_scalar("train/total", Ld.item() + Lp.item(), global_step)
                writer.add_scalar("train/lr", lr, global_step)

                # Quick PSNR on 1 image, 20 DDPM steps (~0.2s)
                with torch.no_grad():
                    ctx1 = model.context_encoder(low[0:1])
                    mask1, _ = model.sample_mask(
                        model.num_patches, mask_ratio_mean=0.5,
                        mask_ratio_std=0.05, mask_ratio_min=0.47)
                    mask_b1 = mask1.to(device).unsqueeze(0)  # (1, N_mask) indices
                    pred1 = model.feature_predictor(ctx1, mask_b1)
                    gen_tok = model.ddpm_sample(pred1, noise_schedule, num_steps=20)
                    gt_tok = model.encode_to_latent(high[0:1])
                    combined = gt_tok.clone()
                    combined[0, mask_b1[0]] = gen_tok[0]
                    img1 = model.decode_from_latent(combined).clamp(0, 1)
                    mse1 = nn.functional.mse_loss(
                        img1, high[0:1], reduction="none").mean()
                    step_psnr = 10 * math.log10(1.0 / mse1.clamp(min=1e-8))
                    writer.add_scalar("train/PSNR_step", step_psnr, global_step)

        # --- End of epoch ---
        n_batches = total_batches
        avg_ld = epoch_ld / n_batches
        avg_lp = epoch_lp / n_batches
        train_time = time.time() - t0
        scheduler.step()

        # --- Validation ---
        model.eval()
        val_ld = val_lp = 0.0
        t_val = time.time()
        total_val = len(val_loader)

        with torch.no_grad():
            for vi, vbatch in enumerate(val_loader):
                low = vbatch["low_res"].to(device)
                high = vbatch["high_res"].to(device)
                B, N = low.shape[0], model.num_patches

                ctx = model.context_encoder(low)
                tgt_emb = model.target_encoder(high)
                high_tok = model.encode_to_latent(high)
                mask, _ = model.sample_mask(N)
                mask_b = mask.to(device).unsqueeze(0).expand(B, -1)
                pred = model.feature_predictor(ctx, mask_b)

                tgt_masked = tgt_emb.gather(
                    1, mask_b.unsqueeze(-1).expand(-1, -1, model.embed_dim))
                tok_masked = high_tok.gather(
                    1, mask_b.unsqueeze(-1).expand(-1, -1, model.token_dim))

                val_ld += model.diffusion_loss(tok_masked, pred, noise_schedule).item()
                val_lp += model.prediction_loss(pred, tgt_masked).item()

                # Progress
                vlog = max(1, total_val // 4)
                if vi > 0 and vi % vlog == 0:
                    e = time.time() - t_val
                    eta = (e / (vi + 1)) * (total_val - vi - 1)
                    print(f"  Val: {vi}/{total_val} | ETA {eta/60:.0f}m{eta%60:.0f}s")

        val_ld /= total_val
        val_lp /= total_val
        val_total = val_ld + val_lp
        val_time = time.time() - t_val

        print(f"--- Epoch {epoch:3d} | "
              f"train Ld={avg_ld:.4f} Lp={avg_lp:.4f} | "
              f"val Ld={val_ld:.4f} Lp={val_lp:.4f} total={val_total:.4f} | "
              f"T={train_time/60:.0f}m{int(train_time)%60:02.0f}s "
              f"V={val_time/60:.0f}m{int(val_time)%60:02.0f}s ---")

        writer.add_scalar("train/epoch_Ld", avg_ld, epoch)
        writer.add_scalar("train/epoch_Lp", avg_lp, epoch)
        writer.add_scalar("val/Ld", val_ld, epoch)
        writer.add_scalar("val/Lp", val_lp, epoch)
        writer.add_scalar("val/total", val_total, epoch)

        # --- PSNR samples (every epoch, first val batch, 50-step DDPM) ---
        psnr_val = 0.0
        with torch.no_grad():
            sample_batch = next(iter(val_loader))
            low_s = sample_batch["low_res"].to(device)
            high_s = sample_batch["high_res"].to(device)
            Bs = low_s.shape[0]

            ctx_s = model.context_encoder(low_s)
            mask_s, _ = model.sample_mask(
                model.num_patches, mask_ratio_mean=0.5,
                mask_ratio_std=0.05, mask_ratio_min=0.47)
            mask_b_s = mask_s.to(device).unsqueeze(0).expand(Bs, -1)
            pred_s = model.feature_predictor(ctx_s, mask_b_s)

            t_sample = time.time()
            generated_tokens = model.ddpm_sample(
                pred_s, noise_schedule, num_steps=50)
            sample_time = time.time() - t_sample

            # Inpaint: masked = generated, unmasked = GT tokens
            high_tok_s = model.encode_to_latent(high_s)
            combined = high_tok_s.clone()
            for b in range(Bs):
                combined[b, mask_b_s[b]] = generated_tokens[b]
            generated_img = model.decode_from_latent(combined)
            generated_img = generated_img.clamp(0, 1)

            mse = nn.functional.mse_loss(
                generated_img, high_s, reduction="none").mean(dim=[1, 2, 3])
            psnr_val = (10 * torch.log10(1.0 / mse.clamp(min=1e-8))).mean().item()

        print(f"  PSNR={psnr_val:.2f} dB | DDPM 50-step ({sample_time:.0f}s)")
        writer.add_scalar("val/PSNR", psnr_val, epoch)
        writer.add_images("val/samples", generated_img, epoch)
        writer.add_images("val/ground_truth", high_s, epoch)

        # Best tracking
        improved = val_total < best_val
        if improved:
            best_val = val_total
            best_epoch = epoch
            patience_ctr = 0
            save_ckpt(model, optimizer, scheduler, epoch,
                      f"{args.checkpoint_dir}/djepa_{phase_id}_best.pt",
                      args.phase, ema_decay)
            print(f"  New best val={val_total:.4f}")
        else:
            patience_ctr += 1

        if not args.no_early_stop and patience_ctr >= args.patience:
            print(f"Early stop (no improvement for {args.patience} epochs). "
                  f"Best: epoch {best_epoch} val={best_val:.4f}")
            break

        if (epoch + 1) % args.checkpoint_every == 0:
            save_ckpt(model, optimizer, scheduler, epoch,
                      f"{args.checkpoint_dir}/djepa_{phase_id}_epoch_{epoch}.pt",
                      args.phase, ema_decay)

    # Final save
    if patience_ctr < args.patience or args.no_early_stop:
        save_ckpt(model, optimizer, scheduler, epoch,
                  f"{args.checkpoint_dir}/djepa_{phase_id}_final.pt",
                  args.phase, ema_decay)

    writer.close()
    print(f"\nPhase {args.phase} done. Best: epoch {best_epoch} val={best_val:.4f}")
    print(f"TensorBoard: tensorboard --logdir {args.log_dir}/{run_name}")


if __name__ == "__main__":
    main()
