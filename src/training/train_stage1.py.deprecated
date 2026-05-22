"""
Phase 2: Cross-Attention Latent Transformer (Stage 1).

Frozen JEPA + frozen VAE + trainable cross-attention transformer.
Predicts high-res VAE latent tokens from low-res latent + JEPA embeddings,
decoded through frozen VAE for pixel-space losses.

Usage:
    python -m src.training.train_stage1 \
        --data_dir data/ortho \
        --jepa_ckpt models/checkpoints/jepa_best.pt \
        --epochs 30 --lr 3e-4 --batch_size 16 --grad_accum 2
"""

import argparse, math, sys, time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from diffusers import AutoencoderKL

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.urbanjepa import UrbanJEPA
from src.models.latent_transformer import LatentCrossAttentionTransformer
from src.models.token_utils import group_latent_tokens, ungroup_latent_tokens
from src.data.ortho_dataset import OrthoDataset
from src.training.losses import VGGPerceptualLoss, high_frequency_loss


VAE_SCALE = 0.18215


def create_lr_lambda(warmup_steps, total_steps, min_factor=1e-6 / 3e-4):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(min_factor, 0.5 * (1 + math.cos(math.pi * progress)))
    return lr_lambda


def get_args():
    p = argparse.ArgumentParser(description="Phase 2: Cross-Attention Latent Transformer")
    p.add_argument("--data_dir", type=str, default="data/ortho")
    p.add_argument("--jepa_ckpt", type=str, default="models/checkpoints/jepa_best.pt")
    p.add_argument("--pretrained", type=str,
                   default="models/ijepa/vit_base_patch16_224_imagenet.pt")
    p.add_argument("--checkpoint_dir", type=str, default="models/checkpoints")
    p.add_argument("--resume", type=str, default=None,
                   help="Resume transformer training from checkpoint")

    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=2,
                   help="Gradient accumulation steps")
    p.add_argument("--warmup_epochs", type=int, default=3)
    p.add_argument("--patches_per_epoch", type=int, default=64,
                   help="Random crops per tile per epoch")
    p.add_argument("--val_patches_per_tile", type=int, default=4)

    # Loss weights
    p.add_argument("--w_latent", type=float, default=0.5)
    p.add_argument("--w_pixel", type=float, default=2.0)
    p.add_argument("--w_percep", type=float, default=0.5)
    p.add_argument("--w_hf", type=float, default=0.3)
    p.add_argument("--latent_only_epochs", type=int, default=3,
                   help="Train with latent MSE only for first N epochs "
                        "(pixel gradients through frozen VAE decode are noisy early)")

    # Transformer architecture
    p.add_argument("--n_layers", type=int, default=8)
    p.add_argument("--d_model", type=int, default=768)
    p.add_argument("--n_heads", type=int, default=12)
    p.add_argument("--ffn_ratio", type=float, default=4.0)
    p.add_argument("--dropout", type=float, default=0.1)

    # Logging
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--checkpoint_every", type=int, default=5)
    p.add_argument("--log_dir", type=str, default="runs/stage1")
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--no_early_stop", action="store_true", default=False)

    return p.parse_args()


def save_checkpoint(transformer, optimizer, scheduler, epoch, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "transformer": transformer.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }, path)
    print(f"  Checkpoint saved: {path}")


def load_checkpoint(transformer, optimizer, scheduler, path):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    transformer.load_state_dict(ckpt["transformer"])
    if optimizer:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt.get("epoch", -1) + 1


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Datasets ---
    train_ds = OrthoDataset(args.data_dir, split="train", train_ratio=0.9, augment=True,
                            patches_per_epoch=args.patches_per_epoch)
    val_ds = OrthoDataset(args.data_dir, split="val", train_ratio=0.9, augment=False,
                          val_patches_per_tile=args.val_patches_per_tile)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True, drop_last=False)
    print(f"Train: {len(train_ds):,} samples, {len(train_loader)} batches")
    print(f"Val:   {len(val_ds):,} samples, {len(val_loader)} batches")

    # --- Frozen JEPA ---
    print(f"Loading JEPA from {args.jepa_ckpt}")
    jepa = UrbanJEPA(pretrained_path=args.pretrained, drop_path_rate=0.0).to(device)
    jepa_ckpt = torch.load(args.jepa_ckpt, map_location="cpu", weights_only=True)
    jepa.context_encoder.load_state_dict(jepa_ckpt["context_encoder"])
    jepa.target_encoder.load_state_dict(jepa_ckpt["target_encoder"])
    jepa.feature_predictor.load_state_dict(jepa_ckpt["feature_predictor"])
    ph_state = jepa_ckpt["projection_head"]
    if "2.weight" in ph_state and "3.weight" not in ph_state:
        ph_state = {k.replace("2.", "3.") if k.startswith("2.") else k: v for k, v in ph_state.items()}
    jepa.projection_head.load_state_dict(ph_state)
    jepa.eval()
    for p in jepa.parameters():
        p.requires_grad = False
    print(f"  JEPA loaded (epoch {jepa_ckpt.get('epoch', '?')}), frozen")

    # --- Frozen VAE ---
    print("Loading SD-VAE (fp32)")
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse", torch_dtype=torch.float32).to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False
    print("  VAE loaded, frozen")

    # --- Trainable Transformer ---
    transformer = LatentCrossAttentionTransformer(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        ffn_ratio=args.ffn_ratio,
        dropout=args.dropout,
    ).to(device)
    trainable_params = sum(p.numel() for p in transformer.parameters())
    print(f"Transformer: {trainable_params:,} trainable params")

    # --- Losses ---
    vgg_loss = VGGPerceptualLoss(device=device).to(device)
    for p in vgg_loss.parameters():
        p.requires_grad = False

    # --- Optimizer & Scheduler ---
    accum_steps = args.grad_accum
    effective_bs = args.batch_size * accum_steps
    effective_batches_per_epoch = len(train_loader) // accum_steps
    total_steps = args.epochs * effective_batches_per_epoch
    warmup_steps = args.warmup_epochs * effective_batches_per_epoch
    print(f"Batch: {args.batch_size} x {accum_steps} accum = {effective_bs} effective")
    print(f"Steps: {total_steps} total, {warmup_steps} warmup")

    optimizer = AdamW(transformer.parameters(), lr=args.lr,
                      weight_decay=args.weight_decay, betas=(0.9, 0.95))
    scheduler = LambdaLR(optimizer, create_lr_lambda(warmup_steps, total_steps))

    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(transformer, optimizer, scheduler, args.resume)
        print(f"Resumed from epoch {start_epoch}")

    # --- Logging ---
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"TensorBoard: {log_dir}")

    # --- Training ---
    print(f"Training {args.epochs} epochs (starting at {start_epoch})")
    print(f"Loss: latent_mse * {args.w_latent} + l1_pixel * {args.w_pixel} "
          f"+ percep * {args.w_percep} + hf * {args.w_hf}")
    print(f"Latent-only first {args.latent_only_epochs} epochs (no pixel losses)\n")

    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    global_step = start_epoch * effective_batches_per_epoch

    best_val_psnr = 0.0
    best_epoch = -1
    patience_counter = 0

    for epoch in range(start_epoch, args.epochs):
        transformer.train()
        epoch_loss = 0.0
        epoch_latent = 0.0
        epoch_pixel = 0.0
        optimizer.zero_grad()
        t_train_start = time.time()
        total_batches = len(train_loader)

        use_pixel_losses = epoch >= args.latent_only_epochs

        for batch_idx, batch in enumerate(train_loader):
            low_res = batch["low_res"].to(device, non_blocking=True)
            high_res = batch["high_res"].to(device, non_blocking=True)

            # --- Frozen encodings ---
            with torch.no_grad():
                # VAE encode → latent tokens
                lr_latent = vae.encode(low_res).latent_dist.mean * VAE_SCALE
                hr_latent = vae.encode(high_res).latent_dist.mean * VAE_SCALE
                lr_tokens = group_latent_tokens(lr_latent)
                hr_tokens = group_latent_tokens(hr_latent)

                # JEPA embeddings
                ctx_emb = jepa.context_encoder(low_res)
                B = low_res.shape[0]
                all_pos = torch.arange(256, device=device).unsqueeze(0).expand(B, -1)
                pred_emb = jepa.feature_predictor(ctx_emb, all_pos)

            # --- Transformer forward ---
            if scaler:
                with torch.amp.autocast("cuda"):
                    pred_hr_tokens = transformer(lr_tokens, ctx_emb, pred_emb)
            else:
                pred_hr_tokens = transformer(lr_tokens, ctx_emb, pred_emb)

            # --- Latent loss (always active) ---
            loss_latent = F.mse_loss(pred_hr_tokens, hr_tokens)

            total_loss = args.w_latent * loss_latent

            if use_pixel_losses:
                # Decode to pixel space (gradients flow through frozen VAE decoder)
                pred_latent_2d = ungroup_latent_tokens(pred_hr_tokens)
                pred_latent_2d = pred_latent_2d / VAE_SCALE
                pred_image = vae.decode(pred_latent_2d).sample.clamp(0, 1)

                loss_pixel = F.l1_loss(pred_image, high_res)
                loss_percep = vgg_loss(pred_image, high_res)
                loss_hf = high_frequency_loss(pred_image, high_res)

                total_loss = (total_loss +
                              args.w_pixel * loss_pixel +
                              args.w_percep * loss_percep +
                              args.w_hf * loss_hf)
            else:
                loss_pixel = torch.tensor(0.0)

            total_loss = total_loss / accum_steps

            if scaler:
                scaler.scale(total_loss).backward()
            else:
                total_loss.backward()

            if (batch_idx + 1) % accum_steps == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(transformer.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(transformer.parameters(), args.grad_clip)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            epoch_loss += loss_latent.item()
            if use_pixel_losses:
                epoch_pixel += loss_pixel.item()

            if batch_idx % args.log_every == 0 and batch_idx > 0:
                avg = epoch_loss / (batch_idx + 1)
                lr = optimizer.param_groups[0]["lr"]
                elapsed = time.time() - t_train_start
                eta = (elapsed / (batch_idx + 1)) * (total_batches - batch_idx - 1)
                pixel_str = f"| L1={loss_pixel.item():.4f}" if use_pixel_losses else ""
                print(f"  Epoch {epoch:3d} | Step {batch_idx:4d}/{total_batches} | "
                      f"latent={loss_latent.item():.4f} {pixel_str} | "
                      f"avg_loss={avg:.4f} | lr={lr:.2e} | "
                      f"ETA {eta/60:.0f}m{eta%60:02.0f}s")
                writer.add_scalar("train/latent_loss", loss_latent.item(), global_step)
                writer.add_scalar("train/lr", lr, global_step)
                if use_pixel_losses:
                    writer.add_scalar("train/pixel_loss", loss_pixel.item(), global_step)

        # --- End of epoch ---
        avg_loss = epoch_loss / total_batches
        train_elapsed = time.time() - t_train_start

        # --- Validation ---
        transformer.eval()
        val_psnr = 0.0
        val_l1 = 0.0
        val_latent_loss = 0.0
        t_val_start = time.time()
        total_val_batches = len(val_loader)

        with torch.no_grad():
            for val_batch_idx, batch in enumerate(val_loader):
                low_res = batch["low_res"].to(device)
                high_res = batch["high_res"].to(device)

                lr_latent = vae.encode(low_res).latent_dist.mean * VAE_SCALE
                hr_latent = vae.encode(high_res).latent_dist.mean * VAE_SCALE
                lr_tokens = group_latent_tokens(lr_latent)
                hr_tokens = group_latent_tokens(hr_latent)

                ctx_emb = jepa.context_encoder(low_res)
                B = low_res.shape[0]
                all_pos = torch.arange(256, device=device).unsqueeze(0).expand(B, -1)
                pred_emb = jepa.feature_predictor(ctx_emb, all_pos)

                if scaler:
                    with torch.amp.autocast("cuda"):
                        pred_hr_tokens = transformer(lr_tokens, ctx_emb, pred_emb)
                else:
                    pred_hr_tokens = transformer(lr_tokens, ctx_emb, pred_emb)

                val_latent_loss += F.mse_loss(pred_hr_tokens, hr_tokens).item()

                # Decode for PSNR
                pred_latent_2d = ungroup_latent_tokens(pred_hr_tokens)
                pred_image = vae.decode(pred_latent_2d / VAE_SCALE).sample.clamp(0, 1)

                val_l1 += F.l1_loss(pred_image, high_res).item()
                mse = F.mse_loss(pred_image, high_res)
                val_psnr += (10.0 * torch.log10(1.0 / mse)).item()

        val_latent_loss /= total_val_batches
        val_l1 /= total_val_batches
        val_psnr /= total_val_batches
        val_elapsed = time.time() - t_val_start

        print(f"--- Epoch {epoch:3d} complete | "
              f"latent_loss={val_latent_loss:.4f} | L1={val_l1:.4f} | "
              f"PSNR={val_psnr:.2f}dB | "
              f"train={train_elapsed/60:.0f}m{int(train_elapsed)%60:02.0f}s "
              f"val={val_elapsed/60:.0f}m{int(val_elapsed)%60:02.0f}s ---")

        writer.add_scalar("val/latent_loss", val_latent_loss, epoch)
        writer.add_scalar("val/pixel_l1", val_l1, epoch)
        writer.add_scalar("val/PSNR", val_psnr, epoch)
        writer.add_scalar("train/epoch_loss", avg_loss, epoch)

        # Early stopping
        improved = val_psnr > best_val_psnr
        if improved:
            best_val_psnr = val_psnr
            best_epoch = epoch
            patience_counter = 0
            best_path = Path(args.checkpoint_dir) / "stage1_best.pt"
            save_checkpoint(transformer, optimizer, scheduler, epoch, best_path)
            print(f"  New best PSNR: {val_psnr:.2f} dB (saved to {best_path})")
        else:
            patience_counter += 1

        if not args.no_early_stop and patience_counter >= args.patience:
            print(f"\nEarly stopping after {args.patience} epochs without PSNR improvement.")
            print(f"Best PSNR={best_val_psnr:.2f} dB at epoch {best_epoch}.")
            break

        if (epoch + 1) % args.checkpoint_every == 0:
            ckpt_path = Path(args.checkpoint_dir) / f"stage1_epoch_{epoch}.pt"
            save_checkpoint(transformer, optimizer, scheduler, epoch, ckpt_path)

    # Final checkpoint
    if patience_counter < args.patience or args.no_early_stop:
        final_path = Path(args.checkpoint_dir) / "stage1_final.pt"
        save_checkpoint(transformer, optimizer, scheduler, epoch, final_path)
        print(f"Final checkpoint: {final_path}")

    writer.close()
    print(f"\nPhase 2 complete. Best PSNR={best_val_psnr:.2f} dB at epoch {best_epoch}.")


if __name__ == "__main__":
    main()
