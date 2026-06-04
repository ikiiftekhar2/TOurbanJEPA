# UrbanJEPA v5 — JEPA-Conditioned ESRGAN for 20× Satellite Super-Resolution

20× single-image super-resolution from ~3 m/px PlanetScope-equivalent input to ~15 cm/px Toronto aerial ortho. v5 conditions a pretrained Real-ESRGAN x4plus RRDBNet on DINOv2 self-supervised features through three zero-init injection points, and fine-tunes end-to-end on Toronto tiles.

## Result (TL;DR)

A four-way control was run to isolate JEPA conditioning from the underlying fine-tune. All numbers below are scale-matched validation averages over `{16, 18, 20, 22, 24}×` with `match_train_aug_in_val=True`.

| Model                                 | PSNR   | LPIPS  | Notes                                          |
| ------------------------------------- | ------ | ------ | ---------------------------------------------- |
| Bilinear                              | 21.561 | 0.6853 | baseline                                       |
| Bare RRDBNet (pretrained)             | 21.382 | 0.6442 | Real-ESRGAN x4plus, zero fine-tune             |
| **Bare RRDBNet (Toronto fine-tuned)** | **21.389** | **0.4790** | 16k steps, no JEPA — the attribution control |
| **v5 JEPA-Conditioned**               | **21.378** | **0.4642** | 15k steps Toronto fine-tune + JEPA branch    |

**JEPA's contribution beyond fine-tuning alone: PSNR −0.011 dB (tied), LPIPS −0.0148.**

~92% of the headline LPIPS gain (−0.165 of −0.180) was already delivered by Toronto fine-tuning the pretrained RRDBNet. JEPA conditioning adds a small, directionally consistent perceptual sliver (better LPIPS at every individual scale) at the cost of a ~120M-param DINOv2 ViT-B/14 + predictor branch and a forward pass per step. Useful as a research result; borderline-not-worth-it as deployed architecture unless a domain-adapted JEPA backbone (e.g. satellite-pretrained DINOv2 / ExPLoRA) is substituted.

## Architecture

```
PlanetScope-equivalent LR (256×256 bilinear)
        │
        ├──► DINOv2 ViT-B/14 context encoder ──┐
        │                                       │ predictor ──► projected features
        │     DINOv2 EMA target (stop-grad) ────┘
        │
        ▼
  avg_pool 4× → 64×64
        │
        ▼
  Pretrained Real-ESRGAN x4plus RRDBNet
        │ ↑                ↑                ↑
        │ │bottleneck      │up1             │up2
        │ └────────────────┴────────────────┘
        │   3× FeatureInjection (zero-init residual)
        ▼
  256×256 SR @ ~15 cm/px
```

- `src/models/v5_model.py` — top-level `V5Model = JEPABackbone + JEPAConditionedRRDBNet`
- `src/models/jepa_esrgan.py` — pretrained RRDBNet wrapped with 3 zero-init feature injection points (identity-at-init: `max|wrap − rrdb| = 0.00`)
- `src/models/esrgan/` — vendored Real-ESRGAN RRDBNet + x4plus weight loader
- `src/models/urbanjepa.py` — JEPA backbone (context + EMA target + predictor + projection head)
- `src/models/discriminator.py` — U-Net SN discriminator (Stage B, ended up archived)

Identity-at-init means the v5 wrap with zero-init injections produces the same output as the bare pretrained RRDBNet at step 0. JEPA features only contribute as the injection layers learn — no risk of damaging the pretrained backbone before learning has anything useful to inject.

## Training

```bash
# v5 JEPA-Conditioned (Stage A — Stage B GAN fine-tune was net-destructive, archived)
sudo bash scripts/setup_systemd_v5_jepa_esrgan.sh
sudo systemctl start urbanjepa-v5.service

# Bare-RRDBNet fine-tune control (the attribution experiment)
sudo bash scripts/setup_systemd_v5_bare_rrdb_control.sh
sudo systemctl start urbanjepa-v5-bare.service
```

Both services use `Type=simple`, `Restart=on-failure`, and the launcher script reads `checkpoints/<exp>/stage.txt` (v5) or just runs to completion (control). Training log: `runs/<exp>.log`. Systemd log: `runs/systemd_<exp>.log`. TensorBoard: `tensorboard --logdir runs --port 6006`.

Stage A recipe (matched by both v5 and the control, modulo the JEPA branch):
- batch_size 20, patches_per_epoch 32, `epochs=3` (v5) / `4` (control, overshoots v5's best to eliminate "needed more time" objections)
- AdamW(betas=(0.9, 0.95), wd=1e-4), warmup-cosine LR over total_steps, min_lr_ratio=0.05
- rrdbnet_lr 1e-5; (v5 only) injection_lr 1e-4; (v5 only) jepa_encoder_lr / pred_proj_lr small
- L1 (w=1.0) + LPIPS-VGG (w=0.1), no GAN
- bf16 autocast, grad_clip=1.0
- Scale-matched val on `{16, 18, 20, 22, 24}×` with `match_train_aug_in_val=True` every 1000 steps; `best.pt` promoted by `lpips_avg` (lower is better)

## Evaluation

`scripts/v5_four_way_comparison.py` is the truth-of-record. It mutates `od.VAL_SCALE` per scale *inside* the dataloader-building loop (the only correct pattern, because `OrthoDataset` reads `VAL_SCALE` live at `__getitem__` time):

```bash
PYTHONPATH=. python scripts/v5_four_way_comparison.py \
    --v5_ckpt   checkpoints/v5_jepa_esrgan_stageA/best.pt \
    --ctrl_ckpt checkpoints/v5_bare_rrdb_control/best.pt \
    --scales 16 18 20 22 24 --batch_size 2
```

Companion scripts:
- `scripts/v5_three_way_comparison.py` — bilinear / pretrained-RRDB / v5 (older, before the FT control existed)
- `scripts/v5_bare_rrdbnet_baseline.py` — inference-only sanity check on the off-the-shelf x4plus
- `scripts/v5_val_scale_curve.py` — sweep performance over a continuous scale range
- `scripts/v5_retro_val.py` / `v5_retro_val_all.py` — re-validate historical checkpoints against the canonical val protocol

## What changed under `src/training/`

- `train.py` — `--reset_ema_on_resume`, `--rollback_steps`, scale-match val infrastructure, four-bucket param-group LRs for v5 (`jepa_encoder`, `jepa_pred_proj`, `jepa_injection`, `rrdbnet_pretrained`)
- `ema.py` — NaN guard around the EMA update (a 0.9999 decay over an unstable shadow blew up otherwise)
- `checkpoint.py` — manifest-tracked, atomic step/epoch slots; best-by-metric promotion; resume preserves epoch position

## Repo layout

```
src/
  data/ortho_dataset.py        LRU-cached tile loader, realistic degradation
  models/
    urbanjepa.py               JEPABackbone (context + EMA + predictor + projection)
    esrgan/rrdbnet.py          vendored Real-ESRGAN RRDBNet
    esrgan/weight_loader.py    x4plus weight loader
    jepa_esrgan.py             JEPAConditionedRRDBNet (3 zero-init injections)
    v5_model.py                top-level V5Model
    discriminator.py           U-Net SN discriminator (Stage B, archived)
  training/
    losses.py                  L1 + SSIM + VGG + LPIPS-VGG + HF
    ema.py                     target-encoder EMA with NaN guard
    checkpoint.py              manifest-tracked, atomic, async
    train.py                   end-to-end v5 training
  evaluation/metrics.py        PSNR + SSIM helpers
scripts/
  run_v5_jepa_esrgan.sh                v5 staged launcher (A → B → DONE)
  setup_systemd_v5_jepa_esrgan.sh
  run_v5_bare_rrdb_control.sh          bare-RRDB FT control launcher
  setup_systemd_v5_bare_rrdb_control.sh
  v5_bare_rrdbnet_finetune.py          standalone trainer for the control
  v5_four_way_comparison.py            the canonical comparison
  v5_three_way_comparison.py           bilinear / RRDB-pre / v5
  v5_bare_rrdbnet_baseline.py          off-the-shelf inference check
  v5_val_scale_curve.py                continuous-scale val sweep
  v5_retro_val.py, v5_retro_val_all.py retroactive val on old checkpoints
  v5_phase[0-3]_smoketest.py           component-level smoke tests
  v5_check_ema_nan.py                  diagnostic for the EMA blow-up
  v5_gpu_probe.py                      VRAM headroom probe
  check_v5_stage_gate.py               A → B gate (PSNR + LPIPS + dual-bilinear)
configs/                       per-backbone YAML
data/ortho/                    tiles + metadata (gitignored)
runs/                          TensorBoard logs (gitignored)
checkpoints/                   per-experiment manifests + weights (gitignored)
```

## Hardware

Single RTX 3090, 24 GB VRAM. v5 trains at batch_size 20 with bf16; the control trains at batch_size 20. End-to-end Stage A run is ~7 h for v5, ~6.5 h for the control. 12-core CPU, `num_workers=4` for the train loader.

## Lineage

```
v1  →  v2  →  v3 (data loader only)  →  v4 (in-house SR decoders, all collapsed)
                                    └─►  v5 (this branch — pretrained RRDBNet + JEPA injections)
```

v4 retired with full forensics on `origin/v4` (`a5d76e5`); in-house SR decoders kept collapsing at any meaningful loss balance. v5 stops fighting the decoder and just augments a proven pretrained one, which is why it trains stably.

Key v4 lessons that carried into v5:
1. Identity-at-init via zero-init residuals (verified: `max|wrap − rrdb| = 0.00` at step 0)
2. Additive residual + multi-point fusion (3 injections: bottleneck / up1 / up2)
3. Freeze pretrained RRDBNet at startup; unfreeze at very low LR after warmup
4. Filtered tile manifest (`train_textured.txt`) to skip flat ground / cloud tiles
5. **Match train aug distribution in val** — without this the train/val gap looks like overfitting but is actually a pipeline mismatch
