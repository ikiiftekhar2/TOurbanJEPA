"""
Phase 2: I-JEPA Domain Fine-Tuning

Trains context_encoder + feature_predictor + projection_head with
prediction loss (Smooth L1) on ortho tile crops. Target encoder is
updated via EMA only — it receives no gradients.

Usage:
    python -m src.training.train_jepa --data_dir data/ortho --epochs 50

Recovery:
    python -m src.training.train_jepa --data_dir data/ortho --resume models/checkpoints/jepa_epoch_40.pt
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.urbanjepa import UrbanJEPA
from src.data.ortho_dataset import OrthoDataset


def get_args():
    p = argparse.ArgumentParser(description="Phase 2: JEPA domain fine-tuning")
    p.add_argument("--data_dir", type=str, default="data/ortho",
                   help="Path to ortho data directory (contains tiles/)")
    p.add_argument("--pretrained", type=str,
                   default="models/ijepa/vit_base_patch16_224_imagenet.pt",
                   help="Path to timm ViT-B/16 pretrained weights")
    p.add_argument("--checkpoint_dir", type=str, default="models/checkpoints",
                   help="Directory for saving checkpoints")
    p.add_argument("--resume", type=str, default=None,
                   help="Resume from checkpoint path")

    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--ema_decay", type=float, default=0.9999)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=4,
                   help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")
    p.add_argument("--patches_per_epoch", type=int, default=4,
                   help="Random crops per tile per epoch (training). Lower = faster epochs.")
    p.add_argument("--val_patches_per_tile", type=int, default=4,
                   help="Grid patches per validation tile. 4/tile × 416 tiles = 1,664 samples "
                        "(SE ≈ σ/40, sufficient for early stopping).")

    p.add_argument("--use_wandb", action="store_true", default=False)
    p.add_argument("--wandb_project", type=str, default="urbanjepa")
    p.add_argument("--wandb_run", type=str, default="phase2-jepa-finetune")
    p.add_argument("--log_every", type=int, default=100,
                   help="Log metrics every N steps")
    p.add_argument("--checkpoint_every", type=int, default=5,
                   help="Save periodic checkpoint every N epochs")
    p.add_argument("--patience", type=int, default=10,
                   help="Early stopping patience (epochs with no val improvement)")
    p.add_argument("--no_early_stop", action="store_true", default=False,
                   help="Disable early stopping")

    return p.parse_args()


def save_checkpoint(model, optimizer, scheduler, epoch, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "context_encoder": model.context_encoder.state_dict(),
        "target_encoder": model.target_encoder.state_dict(),
        "feature_predictor": model.feature_predictor.state_dict(),
        "projection_head": model.projection_head.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }, path)
    print(f"  Checkpoint saved: {path}")


def load_checkpoint(model, optimizer, scheduler, path):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model.context_encoder.load_state_dict(ckpt["context_encoder"])
    model.target_encoder.load_state_dict(ckpt["target_encoder"])
    model.feature_predictor.load_state_dict(ckpt["feature_predictor"])
    model.projection_head.load_state_dict(ckpt["projection_head"])
    if optimizer:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt.get("epoch", -1) + 1


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Data:   {args.data_dir}")
    print(f"Weights:{args.pretrained}")

    # --- Datasets ---
    train_ds = OrthoDataset(args.data_dir, split="train", train_ratio=0.9, augment=True,
                            patches_per_epoch=args.patches_per_epoch)
    val_ds = OrthoDataset(args.data_dir, split="val", train_ratio=0.9, augment=False,
                          val_patches_per_tile=args.val_patches_per_tile)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
    )
    print(f"Train: {len(train_ds):,} samples, {len(train_loader)} batches")
    print(f"Val:   {len(val_ds):,} samples, {len(val_loader)} batches")

    # --- Model ---
    model = UrbanJEPA(
        pretrained_path=args.pretrained,
        ema_decay=args.ema_decay,
    ).to(device)

    model.train_for_phase("jepa")

    # --- Optimizer & Scheduler ---
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable params: {sum(p.numel() for p in trainable):,}")

    optimizer = AdamW(
        trainable,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(model, optimizer, scheduler, args.resume)
        print(f"Resumed from epoch {start_epoch}")

    # --- Logger ---
    if args.use_wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run)
        wandb.config.update(vars(args))

    # --- Training ---
    accum_steps = args.grad_accum
    effective_bs = args.batch_size * accum_steps
    print(f"Batch: {args.batch_size} x {accum_steps} accum = {effective_bs} effective")
    print(f"Training {args.epochs} epochs (starting at {start_epoch})...\n")

    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    global_step = start_epoch * len(train_loader)

    # Early stopping state
    best_val_loss = float("inf")
    best_epoch = -1
    patience_counter = 0

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        t_train_start = time.time()
        total_train_batches = len(train_loader)

        for batch_idx, batch in enumerate(train_loader):
            low_res = batch["low_res"].to(device, non_blocking=True)
            high_res = batch["high_res"].to(device, non_blocking=True)

            # Forward with mixed precision
            if scaler:
                with torch.amp.autocast("cuda"):
                    result = model(low_res, high_res)
                    loss = result["loss"] / accum_steps
                scaler.scale(loss).backward()
            else:
                result = model(low_res, high_res)
                loss = result["loss"] / accum_steps
                loss.backward()

            if (batch_idx + 1) % accum_steps == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(trainable, args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(trainable, args.grad_clip)
                    optimizer.step()

                optimizer.zero_grad()
                model.update_target_encoder()
                global_step += 1

            epoch_loss += result["loss_prediction"]

            if batch_idx % args.log_every == 0 and batch_idx > 0:
                avg = epoch_loss / (batch_idx + 1)
                lr = optimizer.param_groups[0]["lr"]
                elapsed = time.time() - t_train_start
                eta = (elapsed / (batch_idx + 1)) * (total_train_batches - batch_idx - 1)
                print(f"  Epoch {epoch:3d} | Step {batch_idx:4d}/{total_train_batches} | "
                      f"loss={result['loss_prediction']:.4f} | "
                      f"avg={avg:.4f} | lr={lr:.2e} | "
                      f"ETA {eta/60:.0f}m{eta%60:02.0f}s")
                if args.use_wandb:
                    import wandb
                    wandb.log({
                        "train/prediction_loss": result["loss_prediction"],
                        "train/avg_loss": avg,
                        "train/lr": lr,
                        "train/epoch": epoch,
                        "train/step": global_step,
                    })

        # End of epoch
        avg_loss = epoch_loss / total_train_batches
        train_elapsed = time.time() - t_train_start
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        t_val_start = time.time()
        total_val_batches = len(val_loader)
        val_log_every = max(1, total_val_batches // 5)  # ~5 updates during validation

        with torch.no_grad():
            for val_batch_idx, batch in enumerate(val_loader):
                low_res = batch["low_res"].to(device)
                high_res = batch["high_res"].to(device)
                result = model(low_res, high_res)
                val_loss += result["loss_prediction"]

                if val_batch_idx % val_log_every == 0 and val_batch_idx > 0:
                    elapsed = time.time() - t_val_start
                    eta = (elapsed / (val_batch_idx + 1)) * (total_val_batches - val_batch_idx - 1)
                    print(f"  Val progress: {val_batch_idx}/{total_val_batches} batches | "
                          f"ETA {eta/60:.0f}m{eta%60:02.0f}s")

        val_loss /= total_val_batches
        val_elapsed = time.time() - t_val_start

        print(f"--- Epoch {epoch:3d} complete | "
              f"train_loss={avg_loss:.4f} | val_loss={val_loss:.4f} | "
              f"train={train_elapsed/60:.0f}m{int(train_elapsed)%60:02.0f}s "
              f"val={val_elapsed/60:.0f}m{int(val_elapsed)%60:02.0f}s ---")

        if args.use_wandb:
            import wandb
            wandb.log({
                "val/prediction_loss": val_loss,
                "train/epoch_loss": avg_loss,
                "epoch": epoch,
            })

        # Early stopping & best model tracking
        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            best_path = Path(args.checkpoint_dir) / "jepa_best.pt"
            save_checkpoint(model, optimizer, scheduler, epoch, best_path)
            print(f"  New best model (val_loss={val_loss:.4f}) saved to {best_path}")
        else:
            patience_counter += 1

        if not args.no_early_stop and patience_counter >= args.patience:
            print(f"\nEarly stopping triggered after {args.patience} epochs without improvement.")
            print(f"Best val_loss={best_val_loss:.4f} at epoch {best_epoch}.")
            break

        # Periodic checkpoint
        if (epoch + 1) % args.checkpoint_every == 0:
            ckpt_path = Path(args.checkpoint_dir) / f"jepa_epoch_{epoch}.pt"
            save_checkpoint(model, optimizer, scheduler, epoch, ckpt_path)

    # Final checkpoint (if not early-stopped)
    if patience_counter < args.patience or args.no_early_stop:
        final_path = Path(args.checkpoint_dir) / "jepa_final.pt"
        save_checkpoint(model, optimizer, scheduler, epoch, final_path)
        print(f"Final checkpoint: {final_path}")
    else:
        print(f"Best checkpoint (early stopped): {Path(args.checkpoint_dir) / 'jepa_best.pt'}")

    if args.use_wandb:
        import wandb
        wandb.finish()

    print(f"\nPhase 2 complete. Best val_loss={best_val_loss:.4f} at epoch {best_epoch}.")


if __name__ == "__main__":
    main()
