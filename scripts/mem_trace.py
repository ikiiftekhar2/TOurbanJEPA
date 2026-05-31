"""Memory-leak trace: run the exact Stage-A training step for N steps,
logging torch.cuda.memory_allocated() each step. Flat => no leak; monotonic
climb => leak (and how fast). EMA engaged from step 0 to mimic resume-past-4000.
"""
import argparse, time, torch, torch.nn as nn, torch.nn.functional as F
from torch.optim import AdamW

from src.models.urbanjepa import UrbanJEPA
from src.data.ortho_dataset import OrthoDataset
from torch.utils.data import DataLoader
from src.training.ema import ModelEMA
from src.training.losses import (l1_pixel_loss, ssim_loss, high_frequency_loss,
                                 out_of_range_loss, VGGPerceptualLoss)
from src.training.checkpoint import CheckpointManager


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch_size", type=int, default=14)
    ap.add_argument("--grad_checkpoint", type=int, default=1)
    ap.add_argument("--ema", type=int, default=1)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--ckpt_every", type=int, default=50,
                    help="Mirror train.py recovery_every: real save_step (0=off)")
    ap.add_argument("--with_vgg", type=int, default=1,
                    help="Construct resident VGGPerceptualLoss like train.py")
    args = ap.parse_args()

    dev = torch.device("cuda")
    model = UrbanJEPA(
        backbone_name="dinov2_vitb14",
        pretrained_path="models/pretrained/dinov2_vitb14.pth",
        use_v5_decoder=True,
        use_grad_checkpoint=bool(args.grad_checkpoint),
    ).to(dev)
    model.train()

    pg = model.trainable_param_groups(decoder_lr=7.5e-5, encoder_lr=1e-5)
    opt = AdamW(pg, lr=7.5e-5, weight_decay=0.05, betas=(0.9, 0.95))

    ema = None
    if args.ema:
        ema = ModelEMA(model, decay_start=0.995, decay_end=0.9999,
                       warmup_steps=20000, start_step=0, device="cuda")

    vgg_loss_fn = VGGPerceptualLoss().to(dev) if args.with_vgg else None
    ckpt_mgr = (CheckpointManager("checkpoints", "_memtrace_tmp",
                                  recovery_every=args.ckpt_every, recovery_slots=6)
                if args.ckpt_every > 0 else None)
    cfg_dict = {"trace": True}

    train_ds = OrthoDataset("data/ortho", split="train", augment=True,
                            patches_per_epoch=128, seed=42)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=4, drop_last=True, pin_memory=True)

    print(f"[cfg] steps={args.steps} bs={args.batch_size} "
          f"grad_ckpt={args.grad_checkpoint} ema={args.ema}", flush=True)

    step = 0
    base_alloc = None
    t0 = time.time()
    it = iter(loader)
    while step < args.steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        lr_img = batch["low_res"].to(dev, non_blocking=True)
        hr_img = batch["high_res"].to(dev, non_blocking=True)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            result = model(lr_img, hr_img)
            pred = result["pred_image"]
            l_l1 = l1_pixel_loss(pred, hr_img)
            l_ssim = ssim_loss(pred, hr_img)
            l_hf = high_frequency_loss(pred, hr_img)
            l_jepa = result["jepa_loss"]
            l_oob = out_of_range_loss(pred)
            total = 2.0*l_l1 + 0.5*l_ssim + 0.3*l_hf + 0.1*l_jepa + 0.0*l_oob

        total.backward()
        nn.utils.clip_grad_norm_(
            [p for g in pg for p in g["params"]], 0.15)
        opt.step()
        opt.zero_grad(set_to_none=True)
        model.update_target_ema(0.996)
        if ema is not None:
            ema.update(model, step)
        step += 1

        saved = ""
        if ckpt_mgr is not None and step % args.ckpt_every == 0:
            torch.cuda.synchronize()
            before = torch.cuda.max_memory_allocated() / 1e9
            ckpt_mgr.save_step(model, opt, None, None, 0, 0, step, -1e9, cfg_dict)
            torch.cuda.synchronize()
            after = torch.cuda.max_memory_allocated() / 1e9
            saved = f" | CKPT-SAVE peak {before:.2f}->{after:.2f}GB"

        if step % args.log_every == 0 or saved:
            torch.cuda.synchronize()
            alloc = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            peak = torch.cuda.max_memory_allocated() / 1e9
            if base_alloc is None:
                base_alloc = alloc
            rate = (time.time() - t0) / step
            print(f"step={step:4d} | alloc={alloc:6.3f}GB | "
                  f"reserved={reserved:6.3f}GB | peak={peak:6.3f}GB | "
                  f"delta_from_first={alloc-base_alloc:+.3f}GB | "
                  f"{rate:.2f}s/step | L1={l_l1.item():.4f}{saved}", flush=True)

    print(f"[done] total {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
