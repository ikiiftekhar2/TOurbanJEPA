"""
Path B: Direct Latent Regression Training

Maps JEPA context embeddings → VAE latent via a lightweight regressor.
Bypasses diffusion entirely to test whether JEPA embeddings carry enough
information for super-resolution.

Loss: MSE in scaled VAE latent space (L_latent).
Metric: PSNR after VAE decode (evaluated each epoch).

Usage:
    python -m src.training.train_regress --data_dir data/ortho

If this reaches ~22+ dB PSNR, JEPA representations are sufficient and
we invest in a proper spatial diffusion head. If it stays at ~11 dB,
JEPA training itself needs rework.
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
from src.data.ortho_dataset import OrthoDataset


def get_args():
    p = argparse.ArgumentParser(description="Path B: Latent regression training")
    p.add_argument("--data_dir", type=str, default="data/ortho")
    p.add_argument("--pretrained", type=str,
                   default="models/ijepa/vit_base_patch16_224_imagenet.pt")
    p.add_argument("--jepa_ckpt", type=str,
                   default="models/checkpoints/jepa_best.pt",
                   help="Phase 2 JEPA checkpoint")
    p.add_argument("--vae_decoder", type=str, default=None,
                   help="Fine-tuned VAE decoder checkpoint")
    p.add_argument("--checkpoint_dir", type=str, default="models/checkpoints")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--log_dir", type=str, default="runs")

    # Hyperparams
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr_eta_min", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--patches_per_epoch", type=int, default=4)
    p.add_argument("--val_patches_per_tile", type=int, default=4)

    # Regressor
    p.add_argument("--regressor_type", type=str, default="mlp",
                   choices=["mlp", "conv"],
                   help="mlp=per-token only, conv=per-token + conv refinement")
    p.add_argument("--regressor_hidden", type=int, default=512)

    # Logging
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--checkpoint_every", type=int, default=5)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--no_early_stop", action="store_true", default=False)

    return p.parse_args()


def save_ckpt(model, optimizer, scheduler, epoch, path, best_psnr):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "best_psnr": best_psnr,
        "context_encoder": model.context_encoder.state_dict(),
        "target_encoder": model.target_encoder.state_dict(),
        "feature_predictor": model.feature_predictor.state_dict(),
        "projection_head": model.projection_head.state_dict(),
        "latent_regressor": model.latent_regressor.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
    }, path)
    print(f"  Saved: {path}")


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


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Path B: Latent Regression, {args.epochs} epochs")
    print(f"Regressor: {args.regressor_type}, hidden={args.regressor_hidden}")
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
    model = UrbanJEPA(
        pretrained_path=args.pretrained,
        regressor_type=args.regressor_type,
        regressor_hidden=args.regressor_hidden,
    ).to(device)

    # Load JEPA checkpoint
    jepa_ckpt = torch.load(args.jepa_ckpt, map_location=device, weights_only=True)
    model.context_encoder.load_state_dict(jepa_ckpt["context_encoder"])
    model.target_encoder.load_state_dict(jepa_ckpt["target_encoder"])
    model.feature_predictor.load_state_dict(jepa_ckpt["feature_predictor"])
    model.projection_head.load_state_dict(jepa_ckpt["projection_head"])
    print(f"  JEPA weights from {args.jepa_ckpt}")

    # VAE
    vae = load_vae(device)
    if args.vae_decoder:
        print(f"  Loading fine-tuned VAE decoder from {args.vae_decoder}")
        dec_ckpt = torch.load(args.vae_decoder, map_location=device, weights_only=True)
        vae.decoder.load_state_dict(dec_ckpt["decoder"])
        print(f"  VAE decoder fine-tuned weights loaded")
    model.load_vae(vae)

    # Freeze everything except regressor
    model.train_for_phase("regress")
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in trainable)
    print(f"Trainable: {n_params:,} params (regressor only)")

    # --- Optimizer ---
    optimizer = AdamW(
        model.latent_regressor.parameters(),
        lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs,
                                  eta_min=args.lr_eta_min)

    # Resume
    start_epoch = 0
    best_psnr = 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.latent_regressor.load_state_dict(ckpt["latent_regressor"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler and ckpt.get("scheduler"):
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_psnr = ckpt.get("best_psnr", 0.0)
        model.train_for_phase("regress")
        print(f"Resumed from epoch {start_epoch}, best PSNR={best_psnr:.2f}")

    # --- TensorBoard ---
    run_name = f"regress_{args.regressor_type}_{time.strftime('%b%d_%H%M')}"
    writer = SummaryWriter(log_dir=str(Path(args.log_dir) / run_name))
    print(f"TensorBoard: {args.log_dir}/{run_name}")

    steps_per_epoch = len(train_loader)
    global_step = start_epoch * steps_per_epoch
    best_val = float("inf")
    best_epoch = -1
    patience_ctr = 0

    print(f"Batch: {args.batch_size} | Steps/epoch: {steps_per_epoch}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        epoch_loss = 0.0

        for bi, batch in enumerate(train_loader):
            low = batch["low_res"].to(device, non_blocking=True)
            high = batch["high_res"].to(device, non_blocking=True)

            with torch.no_grad():
                target_latent = model.encode_to_latent(high)
                B = target_latent.shape[0]
                target_latent_2d = target_latent.reshape(B, 16, 16, 2, 2, 4)
                target_latent_2d = target_latent_2d.permute(0, 5, 1, 3, 2, 4).contiguous()
                target_latent_2d = target_latent_2d.reshape(B, 4, 32, 32)

            pred_latent = model.forward_regress(low)
            loss = nn.functional.mse_loss(pred_latent, target_latent_2d)

            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            optimizer.step()
            global_step += 1

            if bi > 0 and bi % args.log_every == 0:
                n = bi + 1
                avg_loss = epoch_loss / n
                lr = optimizer.param_groups[0]["lr"]
                elapsed = time.time() - t0
                eta = (elapsed / n) * (steps_per_epoch - n)
                print(f"  E{epoch:3d} S{bi:4d}/{steps_per_epoch} "
                      f"loss={loss.item():.4f} avg={avg_loss:.4f} "
                      f"lr={lr:.1e} ETA {eta/60:.0f}m{eta%60:.0f}s")
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/avg_loss", avg_loss, global_step)
                writer.add_scalar("train/lr", lr, global_step)

        # --- End of epoch ---
        avg_loss = epoch_loss / steps_per_epoch
        train_time = time.time() - t0
        scheduler.step()

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        t_val = time.time()

        with torch.no_grad():
            for vi, vbatch in enumerate(val_loader):
                low = vbatch["low_res"].to(device)
                high = vbatch["high_res"].to(device)
                B = low.shape[0]

                target_latent = model.encode_to_latent(high)
                target_latent_2d = target_latent.reshape(B, 16, 16, 2, 2, 4)
                target_latent_2d = target_latent_2d.permute(0, 5, 1, 3, 2, 4).contiguous()
                target_latent_2d = target_latent_2d.reshape(B, 4, 32, 32)

                pred_latent = model.forward_regress(low)
                val_loss += nn.functional.mse_loss(
                    pred_latent, target_latent_2d).item()

        val_loss /= len(val_loader)
        val_time = time.time() - t_val

        # --- PSNR evaluation ---
        psnr_val = 0.0
        with torch.no_grad():
            sample_batch = next(iter(val_loader))
            low_s = sample_batch["low_res"].to(device)
            high_s = sample_batch["high_res"].to(device)

            pred_img = model.regress_decode(low_s).clamp(0, 1)
            mse = nn.functional.mse_loss(
                pred_img, high_s, reduction="none").mean(dim=[1, 2, 3])
            psnr_val = (10 * torch.log10(1.0 / mse.clamp(min=1e-8))).mean().item()

        print(f"--- Epoch {epoch:3d} | "
              f"train loss={avg_loss:.4f} | val loss={val_loss:.4f} | "
              f"PSNR={psnr_val:.2f} dB | "
              f"T={train_time/60:.0f}m{int(train_time)%60:02.0f}s "
              f"V={val_time/60:.0f}m{int(val_time)%60:02.0f}s ---")

        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/PSNR", psnr_val, epoch)
        writer.add_images("val/predicted", pred_img, epoch)
        writer.add_images("val/ground_truth", high_s, epoch)

        # Best tracking (by PSNR)
        improved = psnr_val > best_psnr
        if improved:
            best_psnr = psnr_val
            best_val = val_loss
            best_epoch = epoch
            patience_ctr = 0
            save_ckpt(model, optimizer, scheduler, epoch,
                      f"{args.checkpoint_dir}/regress_best.pt", best_psnr)
            print(f"  New best PSNR={best_psnr:.2f} dB")
        else:
            patience_ctr += 1

        if not args.no_early_stop and patience_ctr >= args.patience:
            print(f"Early stop (no PSNR improvement for {args.patience} epochs). "
                  f"Best: epoch {best_epoch} PSNR={best_psnr:.2f} dB")
            break

        if (epoch + 1) % args.checkpoint_every == 0:
            save_ckpt(model, optimizer, scheduler, epoch,
                      f"{args.checkpoint_dir}/regress_epoch_{epoch}.pt", best_psnr)

    # Final save
    if patience_ctr < args.patience or args.no_early_stop:
        save_ckpt(model, optimizer, scheduler, epoch,
                  f"{args.checkpoint_dir}/regress_final.pt", best_psnr)

    writer.close()
    print(f"\nPath B done. Best: epoch {best_epoch} PSNR={best_psnr:.2f} dB")
    print(f"TensorBoard: tensorboard --logdir {args.log_dir}/{run_name}")


if __name__ == "__main__":
    main()
