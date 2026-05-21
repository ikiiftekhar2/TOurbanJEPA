"""
Phase 1: I-JEPA Domain Fine-Tuning (v2)

Trains context_encoder + feature_predictor + projection_head with
prediction loss (Smooth L1) on ortho tile crops. Target encoder is
updated via cosine EMA (0.996→1.0) — it receives no gradients.

Usage:
    python -m src.training.train_jepa --data_dir data/ortho --epochs 25

Recovery:
    python -m src.training.train_jepa --data_dir data/ortho --resume models/checkpoints/jepa_epoch_5.pt
"""

import argparse
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.urbanjepa import UrbanJEPA
from src.models.cnn_decoder import CNNDecoder
from src.data.ortho_dataset import OrthoDataset
from src.training.losses import VGGPerceptualLoss


def cosine_ema_decay(step, total_steps, base=0.996, final=1.0):
    """Cosine EMA: fast tracking early, stable late."""
    progress = step / max(total_steps, 1)
    return final - (final - base) * (1 + math.cos(math.pi * progress)) / 2


def create_lr_lambda(warmup_steps, total_steps, min_factor=1e-6 / 5e-4):
    """Warmup + cosine decay LR schedule."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(min_factor, 0.5 * (1 + math.cos(math.pi * progress)))
    return lr_lambda


def get_args():
    p = argparse.ArgumentParser(description="Phase 1 v2: JEPA retrain with cosine EMA + warmup LR")
    p.add_argument("--data_dir", type=str, default="data/ortho",
                   help="Path to ortho data directory (contains tiles/)")
    p.add_argument("--pretrained", type=str,
                   default="models/ijepa/vit_base_patch16_224_imagenet.pt",
                   help="Path to timm ViT-B/16 pretrained weights")
    p.add_argument("--checkpoint_dir", type=str, default="models/checkpoints",
                   help="Directory for saving checkpoints")
    p.add_argument("--resume", type=str, default=None,
                   help="Resume from checkpoint path")

    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--ema_decay", type=float, default=0.9999,
                   help="Ignored when using cosine EMA schedule (default)")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=4,
                   help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")
    p.add_argument("--warmup_epochs", type=int, default=5,
                   help="Linear LR warmup epochs")
    p.add_argument("--patches_per_epoch", type=int, default=32,
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
    p.add_argument("--log_dir", type=str, default="runs/jepa_v2",
                   help="TensorBoard log directory")
    p.add_argument("--drop_path_rate", type=float, default=0.1,
                   help="Stochastic depth rate for ViT blocks (0.0 = disabled)")

    p.add_argument("--alpha", type=float, default=1.0,
                   help="Fixed weight for JEPA loss (only used if --no_uncertainty)")
    p.add_argument("--beta", type=float, default=0.1,
                   help="Fixed weight for L1 loss (only used if --no_uncertainty)")
    p.add_argument("--gamma", type=float, default=0.1,
                   help="Fixed weight for VGG loss (only used if --no_uncertainty)")
    p.add_argument("--no_uncertainty", action="store_true", default=False,
                   help="Use fixed alpha/beta/gamma instead of learned uncertainty weights")

    return p.parse_args()


def save_checkpoint(model, cnn_decoder, optimizer, scheduler, epoch, path,
                    log_var_jepa=None, log_var_l1=None, log_var_percep=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "context_encoder": model.context_encoder.state_dict(),
        "target_encoder": model.target_encoder.state_dict(),
        "feature_predictor": model.feature_predictor.state_dict(),
        "projection_head": model.projection_head.state_dict(),
        "cnn_decoder": cnn_decoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    if log_var_jepa is not None:
        ckpt["log_var_jepa"] = log_var_jepa.data
        ckpt["log_var_l1"] = log_var_l1.data
        ckpt["log_var_percep"] = log_var_percep.data
    torch.save(ckpt, path)
    print(f"  Checkpoint saved: {path}")


def load_checkpoint(model, cnn_decoder, optimizer, scheduler, path,
                    log_var_jepa=None, log_var_l1=None, log_var_percep=None):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model.context_encoder.load_state_dict(ckpt["context_encoder"])
    model.target_encoder.load_state_dict(ckpt["target_encoder"])
    model.feature_predictor.load_state_dict(ckpt["feature_predictor"])

    # Handle projection_head key remapping for old checkpoints
    ph_state = ckpt["projection_head"]
    if "2.weight" in ph_state and "3.weight" not in ph_state:
        ph_state = {k.replace("2.", "3.") if k.startswith("2.") else k: v for k, v in ph_state.items()}
    model.projection_head.load_state_dict(ph_state)

    if "cnn_decoder" in ckpt:
        cnn_decoder.load_state_dict(ckpt["cnn_decoder"])
        print("  Loaded CNN decoder weights")
    if "log_var_jepa" in ckpt and log_var_jepa is not None:
        log_var_jepa.data = ckpt["log_var_jepa"]
        log_var_l1.data = ckpt["log_var_l1"]
        log_var_percep.data = ckpt["log_var_percep"]
        print("  Loaded uncertainty weights")
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
        drop_path_rate=args.drop_path_rate,
    ).to(device)

    cnn_decoder = CNNDecoder().to(device)
    vgg_loss = VGGPerceptualLoss(device=device).to(device)

    print(f"Loss weights: alpha={args.alpha} (JEPA), beta={args.beta} (L1), gamma={args.gamma} (VGG)")

    # --- Optimizer & Scheduler ---
    trainable = (
        [p for p in model.parameters() if p.requires_grad] +
        [p for p in cnn_decoder.parameters() if p.requires_grad]
    )

    # Uncertainty weighting: learnable log-variances for automatic loss balancing
    log_var_jepa = None
    log_var_l1 = None
    log_var_percep = None
    if not args.no_uncertainty:
        log_var_jepa = nn.Parameter(torch.tensor(-1.0, device=device))
        log_var_l1 = nn.Parameter(torch.tensor(0.0, device=device))
        log_var_percep = nn.Parameter(torch.tensor(0.0, device=device))
        trainable += [log_var_jepa, log_var_l1, log_var_percep]
        print("Using learned uncertainty weighting")
    else:
        print(f"Using fixed weights: alpha={args.alpha}, beta={args.beta}, gamma={args.gamma}")

    print(f"Trainable params: {sum(p.numel() for p in trainable):,} (JEPA + CNN decoder + loss weights)")

    accum_steps = args.grad_accum
    effective_bs = args.batch_size * accum_steps
    print(f"Batch: {args.batch_size} x {accum_steps} accum = {effective_bs} effective")

    optimizer = AdamW(
        trainable,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    effective_batches_per_epoch = len(train_loader) // accum_steps
    total_steps = args.epochs * effective_batches_per_epoch
    warmup_steps = args.warmup_epochs * effective_batches_per_epoch
    scheduler = LambdaLR(optimizer, create_lr_lambda(warmup_steps, total_steps))
    print(f"Steps: {total_steps} total, {warmup_steps} warmup")

    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(model, cnn_decoder, optimizer, scheduler, args.resume,
                                       log_var_jepa, log_var_l1, log_var_percep)
        print(f"Resumed from epoch {start_epoch}")

    # --- Logger ---
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"TensorBoard: {log_dir}")

    if args.use_wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run)
        wandb.config.update(vars(args))

    # --- Training ---
    print(f"Training {args.epochs} epochs (starting at {start_epoch})...\n")

    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    global_step = start_epoch * effective_batches_per_epoch

    # Early stopping state
    best_val_loss = float("inf")
    best_epoch = -1
    patience_counter = 0

    for epoch in range(start_epoch, args.epochs):
        model.train()
        cnn_decoder.train()
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
                    L_jepa = result["loss"]
                    all_pred = result["predicted_all"]
                    decoded = cnn_decoder(all_pred)
                    L_l1 = F.l1_loss(decoded, high_res)
                    L_percep = vgg_loss(decoded, high_res)
            else:
                result = model(low_res, high_res)
                L_jepa = result["loss"]
                all_pred = result["predicted_all"]
                decoded = cnn_decoder(all_pred)
                L_l1 = F.l1_loss(decoded, high_res)
                L_percep = vgg_loss(decoded, high_res)

            # Combine losses with uncertainty or fixed weights
            if not args.no_uncertainty:
                precision_jepa = torch.exp(-log_var_jepa)
                precision_l1 = torch.exp(-log_var_l1)
                precision_percep = torch.exp(-log_var_percep)
                total_loss = (precision_jepa * L_jepa +
                              precision_l1 * L_l1 +
                              precision_percep * L_percep +
                              0.5 * (log_var_jepa + log_var_l1 + log_var_percep))
            else:
                total_loss = (args.alpha * L_jepa +
                              args.beta * L_l1 +
                              args.gamma * L_percep)
            total_loss = total_loss / accum_steps

            if scaler:
                scaler.scale(total_loss).backward()
            else:
                total_loss.backward()

            if (batch_idx + 1) % accum_steps == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(trainable, args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(trainable, args.grad_clip)
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()

                decay = cosine_ema_decay(global_step, total_steps)
                with torch.no_grad():
                    for ctx_p, tgt_p in zip(
                        model.context_encoder.parameters(),
                        model.target_encoder.parameters(),
                    ):
                        tgt_p.data.mul_(decay).add_(ctx_p.data, alpha=1 - decay)
                global_step += 1

            epoch_loss += result["loss_prediction"]

            if batch_idx % args.log_every == 0 and batch_idx > 0:
                avg = epoch_loss / (batch_idx + 1)
                lr = optimizer.param_groups[0]["lr"]
                elapsed = time.time() - t_train_start
                eta = (elapsed / (batch_idx + 1)) * (total_train_batches - batch_idx - 1)
                prec_str = ""
                if not args.no_uncertainty:
                    prec_str = (f"| w_J={torch.exp(-log_var_jepa).item():.3f} "
                                f"w_L1={torch.exp(-log_var_l1).item():.3f} "
                                f"w_V={torch.exp(-log_var_percep).item():.3f} ")
                print(f"  Epoch {epoch:3d} | Step {batch_idx:4d}/{total_train_batches} | "
                      f"JEPA={result['loss_prediction']:.4f} | L1={L_l1.item():.4f} "
                      f"{prec_str}| "
                      f"avg={avg:.4f} | lr={lr:.2e} | ema={decay:.4f} | "
                      f"ETA {eta/60:.0f}m{eta%60:02.0f}s")
                writer.add_scalar("train/ema_decay", decay, global_step)
                if args.use_wandb:
                    import wandb
                    wandb.log({
                        "train/prediction_loss": result["loss_prediction"],
                        "train/L1_loss": L_l1.item(),
                        "train/avg_loss": avg,
                        "train/lr": lr,
                        "train/epoch": epoch,
                        "train/step": global_step,
                    })
                writer.add_scalar("train/loss", result["loss_prediction"], global_step)
                writer.add_scalar("train/L1", L_l1.item(), global_step)
                writer.add_scalar("train/loss_avg", avg, global_step)
                writer.add_scalar("train/lr", lr, global_step)
                if not args.no_uncertainty:
                    writer.add_scalar("train/w_jepa", torch.exp(-log_var_jepa).item(), global_step)
                    writer.add_scalar("train/w_l1", torch.exp(-log_var_l1).item(), global_step)
                    writer.add_scalar("train/w_vgg", torch.exp(-log_var_percep).item(), global_step)

        # End of epoch
        avg_loss = epoch_loss / total_train_batches
        train_elapsed = time.time() - t_train_start

        # Validation
        model.eval()
        cnn_decoder.eval()
        val_loss = 0.0
        val_cos_sim = 0.0
        val_l1 = 0.0
        val_psnr = 0.0
        t_val_start = time.time()
        total_val_batches = len(val_loader)
        val_log_every = max(1, total_val_batches // 5)  # ~5 updates during validation

        with torch.no_grad():
            for val_batch_idx, batch in enumerate(val_loader):
                low_res = batch["low_res"].to(device)
                high_res = batch["high_res"].to(device)
                if scaler:
                    with torch.amp.autocast("cuda"):
                        result = model(low_res, high_res)
                        val_decoded = cnn_decoder(result["predicted_all"])
                else:
                    result = model(low_res, high_res)
                    val_decoded = cnn_decoder(result["predicted_all"])
                val_loss += result["loss_prediction"]
                val_cos_sim += result.get("cos_sim", 0.0)
                val_l1 += F.l1_loss(val_decoded, high_res).item()
                mse = F.mse_loss(val_decoded, high_res)
                val_psnr += (10.0 * torch.log10(1.0 / mse)).item()

                if val_batch_idx % val_log_every == 0 and val_batch_idx > 0:
                    elapsed = time.time() - t_val_start
                    eta = (elapsed / (val_batch_idx + 1)) * (total_val_batches - val_batch_idx - 1)
                    print(f"  Val progress: {val_batch_idx}/{total_val_batches} batches | "
                          f"ETA {eta/60:.0f}m{eta%60:02.0f}s")

        val_loss /= total_val_batches
        val_cos_sim /= total_val_batches
        val_l1 /= total_val_batches
        val_psnr /= total_val_batches
        val_elapsed = time.time() - t_val_start

        print(f"--- Epoch {epoch:3d} complete | "
              f"train_loss={avg_loss:.4f} | val_loss={val_loss:.4f} | "
              f"cos_sim={val_cos_sim:.4f} | L1={val_l1:.4f} | PSNR={val_psnr:.2f}dB | "
              f"train={train_elapsed/60:.0f}m{int(train_elapsed)%60:02.0f}s "
              f"val={val_elapsed/60:.0f}m{int(val_elapsed)%60:02.0f}s ---")

        if args.use_wandb:
            import wandb
            wandb.log({
                "val/prediction_loss": val_loss,
                "val/cos_sim": val_cos_sim,
                "val/L1": val_l1,
                "val/PSNR": val_psnr,
                "train/epoch_loss": avg_loss,
                "epoch": epoch,
            })
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/cos_sim", val_cos_sim, epoch)
        writer.add_scalar("val/L1", val_l1, epoch)
        writer.add_scalar("val/PSNR", val_psnr, epoch)
        writer.add_scalar("train/epoch_loss", avg_loss, epoch)

        # Early stopping & best model tracking
        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            best_path = Path(args.checkpoint_dir) / "jepa_best.pt"
            save_checkpoint(model, cnn_decoder, optimizer, scheduler, epoch, best_path,
                            log_var_jepa, log_var_l1, log_var_percep)
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
            save_checkpoint(model, cnn_decoder, optimizer, scheduler, epoch, ckpt_path,
                            log_var_jepa, log_var_l1, log_var_percep)

    # Final checkpoint (if not early-stopped)
    if patience_counter < args.patience or args.no_early_stop:
        final_path = Path(args.checkpoint_dir) / "jepa_final.pt"
        save_checkpoint(model, cnn_decoder, optimizer, scheduler, epoch, final_path,
                        log_var_jepa, log_var_l1, log_var_percep)
        print(f"Final checkpoint: {final_path}")
    else:
        print(f"Best checkpoint (early stopped): {Path(args.checkpoint_dir) / 'jepa_best.pt'}")

    writer.close()
    if args.use_wandb:
        import wandb
        wandb.finish()

    print(f"\nPhase 1 complete. Best val_loss={best_val_loss:.4f} at epoch {best_epoch}.")


if __name__ == "__main__":
    main()
