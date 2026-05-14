"""
Phase 3: CNN Decoder Validation

Trains a lightweight CNN decoder to reconstruct RGB images from frozen
JEPA target encoder embeddings. This validates that the ViT representations
learned in Phase 2 preserve spatial information usable for image reconstruction.

Success gate: PSNR > 26dB on held-out validation ortho.

Usage:
    python -m src.training.train_decoder --data_dir data/ortho --jepa_ckpt models/checkpoints/jepa_best.pt
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.urbanjepa import UrbanJEPA
from src.models.cnn_decoder import CNNDecoder, compute_psnr
from src.data.ortho_dataset import OrthoDataset


def get_args():
    p = argparse.ArgumentParser(description="Phase 3: CNN decoder validation")
    p.add_argument("--data_dir", type=str, default="data/ortho")
    p.add_argument("--jepa_ckpt", type=str, default="models/checkpoints/jepa_best.pt",
                   help="Path to Phase 2 JEPA checkpoint")
    p.add_argument("--pretrained", type=str,
                   default="models/ijepa/vit_base_patch16_224_imagenet.pt",
                   help="Path to timm ViT-B/16 pretrained weights")
    p.add_argument("--checkpoint_dir", type=str, default="models/checkpoints")
    p.add_argument("--resume", type=str, default=None,
                   help="Resume from decoder checkpoint")

    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--patches_per_epoch", type=int, default=4,
                   help="Random crops per tile per epoch")
    p.add_argument("--val_patches_per_tile", type=int, default=4,
                   help="Grid patches per validation tile")

    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--patience", type=int, default=10,
                   help="Early stopping patience")
    p.add_argument("--no_early_stop", action="store_true", default=False)

    return p.parse_args()


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"JEPA checkpoint: {args.jepa_ckpt}")

    # --- Datasets ---
    train_ds = OrthoDataset(args.data_dir, split="train", train_ratio=0.9, augment=False,
                            patches_per_epoch=args.patches_per_epoch)
    val_ds = OrthoDataset(args.data_dir, split="val", train_ratio=0.9, augment=False,
                          val_patches_per_tile=args.val_patches_per_tile)

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

    # --- Load frozen JEPA ---
    print("Loading JEPA model...")
    model = UrbanJEPA(pretrained_path=args.pretrained).to(device)
    ckpt = torch.load(args.jepa_ckpt, map_location="cpu", weights_only=True)
    model.context_encoder.load_state_dict(ckpt["context_encoder"])
    model.target_encoder.load_state_dict(ckpt["target_encoder"])
    model.feature_predictor.load_state_dict(ckpt["feature_predictor"])
    model.projection_head.load_state_dict(ckpt["projection_head"])

    # Freeze everything
    for p in model.parameters():
        p.requires_grad = False
    model.context_encoder.eval()
    model.target_encoder.eval()
    model.feature_predictor.eval()
    print("JEPA loaded and frozen.")

    # --- CNN Decoder ---
    decoder = CNNDecoder().to(device)
    print(f"Decoder params: {sum(p.numel() for p in decoder.parameters()):,}")

    optimizer = AdamW(decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 0
    best_psnr = -float("inf")
    best_epoch = -1
    patience_counter = 0

    if args.resume:
        dec_ckpt = torch.load(args.resume, map_location="cpu", weights_only=True)
        decoder.load_state_dict(dec_ckpt["decoder"])
        optimizer.load_state_dict(dec_ckpt["optimizer"])
        scheduler.load_state_dict(dec_ckpt["scheduler"])
        start_epoch = dec_ckpt["epoch"] + 1
        best_psnr = dec_ckpt.get("best_psnr", -float("inf"))
        print(f"Resumed decoder from epoch {start_epoch}")

    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    mse_loss_fn = nn.MSELoss()

    print(f"Training {args.epochs} epochs...\n")

    for epoch in range(start_epoch, args.epochs):
        # --- Train ---
        decoder.train()
        epoch_loss = 0.0
        t_train_start = time.time()
        total_train = len(train_loader)

        for batch_idx, batch in enumerate(train_loader):
            low_res = batch["low_res"].to(device, non_blocking=True)
            high_res = batch["high_res"].to(device, non_blocking=True)

            with torch.no_grad():
                # JEPA predictive pathway: low_res → context → predictor → embeddings
                ctx_features = model.context_encoder(low_res)
                B, N, _ = ctx_features.shape
                mask_idx = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
                predicted_embeddings = model.feature_predictor(ctx_features, mask_idx)

            if scaler:
                with torch.amp.autocast("cuda"):
                    reconstructed = decoder(predicted_embeddings)
                    loss = mse_loss_fn(reconstructed, high_res)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                reconstructed = decoder(predicted_embeddings)
                loss = mse_loss_fn(reconstructed, high_res)
                loss.backward()
                optimizer.step()

            optimizer.zero_grad()
            epoch_loss += loss.item()

            if batch_idx % args.log_every == 0 and batch_idx > 0:
                avg = epoch_loss / (batch_idx + 1)
                psnr_val = compute_psnr(reconstructed.detach(), high_res)
                elapsed = time.time() - t_train_start
                eta = (elapsed / (batch_idx + 1)) * (total_train - batch_idx - 1)
                print(f"  Epoch {epoch:3d} | Step {batch_idx:4d}/{total_train} | "
                      f"loss={loss.item():.5f} | avg={avg:.5f} | "
                      f"PSNR={psnr_val:.2f}dB | ETA {eta/60:.0f}m{eta%60:02.0f}s")

        train_loss = epoch_loss / total_train
        train_elapsed = time.time() - t_train_start
        scheduler.step()

        # --- Validation ---
        decoder.eval()
        val_loss = 0.0
        val_psnr = 0.0
        t_val_start = time.time()
        total_val = len(val_loader)

        with torch.no_grad():
            for val_batch_idx, batch in enumerate(val_loader):
                low_res = batch["low_res"].to(device)
                high_res = batch["high_res"].to(device)
                ctx_features = model.context_encoder(low_res)
                B, N, _ = ctx_features.shape
                mask_idx = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
                predicted_embeddings = model.feature_predictor(ctx_features, mask_idx)
                reconstructed = decoder(predicted_embeddings)
                val_loss += mse_loss_fn(reconstructed, high_res).item()
                val_psnr += compute_psnr(reconstructed, high_res)

        val_loss /= total_val
        val_psnr /= total_val
        val_elapsed = time.time() - t_val_start

        print(f"--- Epoch {epoch:3d} complete | "
              f"train_loss={train_loss:.5f} | val_loss={val_loss:.5f} | "
              f"val_psnr={val_psnr:.2f}dB | "
              f"train={train_elapsed/60:.0f}m{int(train_elapsed)%60:02.0f}s "
              f"val={val_elapsed/60:.0f}m{int(val_elapsed)%60:02.0f}s ---")

        # --- Checkpointing ---
        improved = val_psnr > best_psnr
        if improved:
            best_psnr = val_psnr
            best_epoch = epoch
            patience_counter = 0
            best_path = Path(args.checkpoint_dir) / "decoder_best.pt"
            torch.save({
                "epoch": epoch,
                "decoder": decoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_psnr": best_psnr,
                "val_loss": val_loss,
            }, best_path)
            print(f"  New best decoder (PSNR={val_psnr:.2f}dB) saved to {best_path}")
        else:
            patience_counter += 1

        if not args.no_early_stop and patience_counter >= args.patience:
            print(f"\nEarly stopping after {args.patience} epochs without improvement.")
            print(f"Best PSNR={best_psnr:.2f}dB at epoch {best_epoch}.")
            break

    # --- Final ---
    if patience_counter < args.patience or args.no_early_stop:
        final_path = Path(args.checkpoint_dir) / "decoder_final.pt"
        torch.save({
            "epoch": epoch,
            "decoder": decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_psnr": best_psnr,
            "val_loss": val_loss,
        }, final_path)
        print(f"Final decoder: {final_path}")

    gate = "PASSED" if best_psnr > 26.0 else "FAILED"
    print(f"\nPhase 3 complete. Best PSNR={best_psnr:.2f}dB at epoch {best_epoch}. Gate (26dB): {gate}.")


if __name__ == "__main__":
    main()
