"""
Tier 2 — SD-VAE Decoder Fine-tuning on Ortho Imagery

The SD-VAE (stabilityai/sd-vae-ft-mse) was trained on LAION natural images.
Satellite textures (rooftops, roads, vegetation) may not survive 48x compression.
Fine-tuning the decoder on ortho patches teaches satellite-domain reconstruction
while keeping the encoder (and latent space) frozen.

Loss: L1 pixel + VGG perceptual loss (features from relu1_2, relu2_2, relu3_3)
LR: 1e-5, low to preserve pretrained structure.

After this, re-run Phase 4b with the fine-tuned VAE decoder for lower Ld floor.

Usage:
    python -m src.training.train_vae --data_dir data/ortho --epochs 20
    python -m src.training.train_vae --data_dir data/ortho --resume models/checkpoints/vae_decoder_best.pt
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import vgg16, VGG16_Weights

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.ortho_dataset import OrthoDataset


# ---------------------------------------------------------------------------
# VGG Perceptual Loss
# ---------------------------------------------------------------------------

class VGGPerceptualLoss(nn.Module):
    """L1 distance between VGG16 feature maps at multiple layers."""

    def __init__(self, device="cuda"):
        super().__init__()
        vgg = vgg16(weights=VGG16_Weights.DEFAULT).to(device)
        vgg.eval()
        for p in vgg.parameters():
            p.requires_grad = False

        # Layers whose features we compare (before each pooling)
        self.slices = nn.ModuleList([
            vgg.features[:4],    # relu1_2  (64 ch)
            vgg.features[:9],    # relu2_2  (128 ch)
            vgg.features[:16],   # relu3_3  (256 ch)
        ])
        # ImageNet stats, placed on the correct device in forward
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def forward(self, pred, target):
        d = pred.device
        mean = self.mean.to(d) if self.mean.device != d else self.mean
        std = self.std.to(d) if self.std.device != d else self.std
        pred_n = (pred - mean) / std
        target_n = (target - mean) / std

        loss = 0.0
        for layer in self.slices:
            p_feat = layer(pred_n)
            t_feat = layer(target_n)
            loss += F.l1_loss(p_feat, t_feat)
        return loss / len(self.slices)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(description="Tier 2: Fine-tune SD-VAE decoder on ortho")
    p.add_argument("--data_dir", type=str, default="data/ortho")
    p.add_argument("--checkpoint_dir", type=str, default="models/checkpoints")
    p.add_argument("--log_dir", type=str, default="runs")
    p.add_argument("--resume", type=str, default=None,
                   help="Resume from VAE decoder checkpoint")

    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--perceptual_weight", type=float, default=0.1,
                   help="Weight for VGG perceptual loss (relative to L1)")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--patches_per_epoch", type=int, default=4)
    p.add_argument("--val_patches_per_tile", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=4)

    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--no_early_stop", action="store_true", default=False)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_ckpt(decoder, optimizer, scheduler, epoch, path, best_val_loss):
    torch.save({
        "epoch": epoch,
        "decoder": decoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_val_loss": best_val_loss,
    }, path)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Datasets ---
    train_ds = OrthoDataset(args.data_dir, split="train", train_ratio=0.9,
                            augment=False, patches_per_epoch=args.patches_per_epoch)
    val_ds = OrthoDataset(args.data_dir, split="val", train_ratio=0.9,
                          augment=False, val_patches_per_tile=args.val_patches_per_tile)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
        persistent_workers=True,
    )
    print(f"Train: {len(train_ds):,} samples, {len(train_loader)} batches")
    print(f"Val:   {len(val_ds):,} samples, {len(val_loader)} batches")

    # --- Load SD-VAE ---
    print("Loading SD-VAE...")
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse", torch_dtype=torch.float32
    ).to(device)

    # Freeze encoder, train decoder only
    for p in vae.encoder.parameters():
        p.requires_grad = False
    for p in vae.decoder.parameters():
        p.requires_grad = True

    n_enc = sum(p.numel() for p in vae.encoder.parameters())
    n_dec = sum(p.numel() for p in vae.decoder.parameters())
    print(f"VAE encoder: {n_enc:,} params (frozen)")
    print(f"VAE decoder: {n_dec:,} params (trainable)")

    # --- Perceptual loss ---
    perceptual = VGGPerceptualLoss(device)
    print("VGG perceptual loss ready (relu1_2, relu2_2, relu3_3)")

    # --- Optimizer ---
    optimizer = AdamW(vae.decoder.parameters(), lr=args.lr,
                     weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    start_epoch = 0
    best_val = float("inf")
    best_epoch = -1
    patience_ctr = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        vae.decoder.load_state_dict(ckpt["decoder"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed from epoch {start_epoch}")

    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    accum = args.grad_accum
    eff_bs = args.batch_size * accum
    steps_per_epoch = len(train_loader) // accum
    print(f"Batch: {args.batch_size} x {accum} accum = {eff_bs} effective")
    print(f"Steps/epoch: {steps_per_epoch}")

    # --- TensorBoard ---
    run_name = f"vae_finetune_{time.strftime('%b%d_%H%M')}"
    writer = SummaryWriter(log_dir=str(Path(args.log_dir) / run_name))
    print(f"TensorBoard: {args.log_dir}/{run_name}")

    vae_latent_scale = 0.18215

    for epoch in range(start_epoch, args.epochs):
        # --- Train ---
        vae.decoder.train()
        vae.encoder.eval()
        epoch_l1 = epoch_percep = 0.0
        t0 = time.time()
        total_batches = len(train_loader)
        accum_count = 0
        optimizer.zero_grad()

        for bi, batch in enumerate(train_loader):
            high = batch["high_res"].to(device, non_blocking=True)

            # Encode (frozen, no grad)
            with torch.no_grad():
                latent_dist = vae.encode(high).latent_dist
                latents = latent_dist.sample() * vae_latent_scale

            # Decode (trainable)
            decoded = vae.decode(latents / vae_latent_scale).sample
            decoded = decoded.clamp(0, 1)

            # L1 pixel loss
            l1 = F.l1_loss(decoded, high)

            # Perceptual loss
            percep = perceptual(decoded, high)

            loss = (l1 + args.perceptual_weight * percep) / accum

            if scaler:
                scaler.scale(loss).backward()
                accum_count += 1
                if accum_count == accum:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    accum_count = 0
            else:
                loss.backward()
                accum_count += 1
                if accum_count == accum:
                    optimizer.step()
                    optimizer.zero_grad()
                    accum_count = 0

            epoch_l1 += l1.item()
            epoch_percep += percep.item()

            if bi > 0 and bi % args.log_every == 0:
                n = bi + 1
                avg_l1 = epoch_l1 / n
                avg_percep = epoch_percep / n
                elapsed = time.time() - t0
                eta = (elapsed / n) * (total_batches - n)
                print(f"  E{epoch:3d} S{bi:4d}/{total_batches} "
                      f"L1={l1.item():.5f} percep={percep.item():.5f} "
                      f"avg L1={avg_l1:.5f} percep={avg_percep:.5f} "
                      f"ETA {eta/60:.0f}m{eta%60:.0f}s")

        train_l1 = epoch_l1 / total_batches
        train_percep = epoch_percep / total_batches
        train_time = time.time() - t0
        scheduler.step()

        writer.add_scalar("train/L1", train_l1, epoch)
        writer.add_scalar("train/perceptual", train_percep, epoch)

        # --- Validation ---
        vae.decoder.eval()
        val_l1 = val_percep = 0.0
        t_val = time.time()
        total_val = len(val_loader)

        with torch.no_grad():
            for vb, vbatch in enumerate(val_loader):
                high = vbatch["high_res"].to(device)
                latent_dist = vae.encode(high).latent_dist
                latents = latent_dist.sample() * vae_latent_scale
                decoded = vae.decode(latents / vae_latent_scale).sample
                decoded = decoded.clamp(0, 1)

                val_l1 += F.l1_loss(decoded, high).item()
                val_percep += perceptual(decoded, high).item()

        val_l1 /= total_val
        val_percep /= total_val
        val_total = val_l1 + args.perceptual_weight * val_percep
        val_time = time.time() - t_val

        writer.add_scalar("val/L1", val_l1, epoch)
        writer.add_scalar("val/perceptual", val_percep, epoch)
        writer.add_scalar("val/total", val_total, epoch)

        # PSNR
        with torch.no_grad():
            sample_batch = next(iter(val_loader))
            high_s = sample_batch["high_res"].to(device)
            latent_dist = vae.encode(high_s).latent_dist
            latents = latent_dist.sample() * vae_latent_scale
            recon = vae.decode(latents / vae_latent_scale).sample.clamp(0, 1)
            mse = F.mse_loss(recon, high_s)
            psnr = 10 * torch.log10(1.0 / mse.clamp(min=1e-8))
        writer.add_scalar("val/PSNR", psnr.item(), epoch)
        writer.add_images("val/reconstruction", recon, epoch)
        writer.add_images("val/ground_truth", high_s, epoch)

        print(f"--- Epoch {epoch:3d} | "
              f"train L1={train_l1:.5f} percep={train_percep:.5f} | "
              f"val L1={val_l1:.5f} percep={val_percep:.5f} total={val_total:.5f} | "
              f"PSNR={psnr.item():.2f}dB | "
              f"T={train_time/60:.0f}m{int(train_time)%60:02.0f}s "
              f"V={val_time/60:.0f}m{int(val_time)%60:02.0f}s ---")

        # Best tracking
        improved = val_total < best_val
        if improved:
            best_val = val_total
            best_epoch = epoch
            patience_ctr = 0
            ckpt_path = Path(args.checkpoint_dir) / "vae_decoder_best.pt"
            save_ckpt(vae.decoder, optimizer, scheduler, epoch, ckpt_path, best_val)
        else:
            patience_ctr += 1

        if not args.no_early_stop and patience_ctr >= args.patience:
            print(f"Early stop after {args.patience} epochs without improvement.")
            print(f"Best: epoch {best_epoch} val_total={best_val:.5f}")
            break

    # Final save
    if patience_ctr < args.patience or args.no_early_stop:
        ckpt_path = Path(args.checkpoint_dir) / "vae_decoder_final.pt"
        save_ckpt(vae.decoder, optimizer, scheduler, epoch, ckpt_path, best_val)

    writer.close()
    print(f"\nVAE fine-tuning done. Best: epoch {best_epoch} val_total={best_val:.5f}")
    print(f"TensorBoard: tensorboard --logdir {args.log_dir}/{run_name}")
    print(f"\nNext: re-run Phase 4b with fine-tuned VAE decoder:")
    print(f"  # Update UrbanJEPA to load fine-tuned decoder, then:")
    print(f"  python -m src.training.train_djepa --phase 4b --data_dir data/ortho \\")
    print(f"      --log_dir runs --resume models/checkpoints/djepa_mlp_best.pt \\")
    print(f"      --lr_mlp 1e-3 --lr_jepa 1e-4 --lr_eta_min 1e-5 --epochs 50 \\")
    print(f"      --batch_size 20 --grad_accum 2 --vae_decoder models/checkpoints/vae_decoder_best.pt")


if __name__ == "__main__":
    main()
