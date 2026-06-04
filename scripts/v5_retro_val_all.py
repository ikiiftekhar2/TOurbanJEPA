"""Run retroactive LIVE-weight val on every Stage A step_slot checkpoint
and print a one-row-per-step summary table."""
import glob, os, time
import torch
from torch.utils.data import DataLoader

from src.data.ortho_dataset import OrthoDataset
from src.evaluation.metrics import psnr as psnr_metric, ssim_metric
from src.models.v5_model import build_v5_model
from src.training.losses import LPIPSLoss, l1_pixel_loss


@torch.no_grad()
def run_val(model, loader, device, lpips_loss):
    model.eval()
    s = {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0, "l1": 0.0,
         "psnr_bil": 0.0, "lpips_bil": 0.0}
    n = 0
    for batch in loader:
        lr = batch["low_res"].to(device, non_blocking=True)
        hr = batch["high_res"].to(device, non_blocking=True)
        out = model(lr, hr)
        sr = out["sr"].clamp(0, 1)
        s["psnr"]      += psnr_metric(sr, hr).item()
        s["ssim"]      += ssim_metric(sr, hr).item()
        s["l1"]        += l1_pixel_loss(sr, hr).item()
        s["lpips"]     += lpips_loss(sr, hr).item()
        s["psnr_bil"]  += psnr_metric(lr, hr).item()
        s["lpips_bil"] += lpips_loss(lr, hr).item()
        n += 1
    return {k: v/n for k, v in s.items()}, n


def main():
    device = torch.device("cuda")
    print(f"[device] free={torch.cuda.mem_get_info()[0]/1024**3:.2f} GiB", flush=True)

    val_ds = OrthoDataset(ortho_dir="data/ortho", split="val", augment=False,
                          val_patches_per_tile=1,
                          tile_manifest="data/ortho/metadata/train_textured.txt")
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False,
                            num_workers=2, pin_memory=True)
    print(f"[data] {len(val_ds)} tiles, {len(val_loader)} batches", flush=True)

    model = build_v5_model(
        backbone_name="dinov2_vitb14",
        jepa_pretrained_path="models/pretrained/dinov2_vitb14.pth",
        esrgan_weights_path="models/pretrained/RealESRGAN_x4plus.pth",
    ).to(device)
    lpips = LPIPSLoss(net="vgg").to(device).eval()

    files = sorted(glob.glob("checkpoints/v5_jepa_esrgan_stageA/step_slot_*.pt"))
    rows = []
    for f in files:
        # Load to CPU first — the 2.3 GB ckpt blob contains optimizer/EMA/etc
        # that we don't need on GPU. Then push only model weights to GPU.
        st = torch.load(f, map_location="cpu", weights_only=False)
        step = st["global_step"]
        model.load_checkpoint_state(st["model"], strict=False)
        t0 = time.time()
        m, n = run_val(model, val_loader, device, lpips)
        dt = time.time() - t0
        rows.append((step, m["psnr"], m["lpips"], m["ssim"], m["l1"],
                     m["psnr_bil"], m["lpips_bil"], dt))
        print(f"  step={step:5d}  PSNR={m['psnr']:.3f} (bil={m['psnr_bil']:.3f}, "
              f"d={m['psnr']-m['psnr_bil']:+.3f})  "
              f"LPIPS={m['lpips']:.4f} (bil={m['lpips_bil']:.4f}, "
              f"d={m['lpips']-m['lpips_bil']:+.4f})  in {dt:.1f}s",
              flush=True)
        del st
        torch.cuda.empty_cache()

    rows.sort(key=lambda r: r[0])
    print("\n" + "=" * 90)
    print(f"{'step':>6}  {'PSNR':>6}  {'d_bil':>7}  {'LPIPS':>6}  {'d_bil':>7}  "
          f"{'SSIM':>6}  {'L1':>6}")
    print("-" * 90)
    for step, psnr, lpips_v, ssim, l1, pbil, lbil, _ in rows:
        print(f"{step:6d}  {psnr:6.3f}  {psnr-pbil:+7.3f}  {lpips_v:6.4f}  "
              f"{lpips_v-lbil:+7.4f}  {ssim:6.4f}  {l1:6.4f}")
    print("=" * 90)
    print(f"  bilinear baseline:        PSNR={rows[-1][5]:.3f}  LPIPS={rows[-1][6]:.4f}")
    print(f"  bare RRDBNet baseline:    PSNR=20.749         LPIPS=0.6560 (measured earlier)")


if __name__ == "__main__":
    main()
