# UrbanJEPA — Progress Checklist & Recovery Notes

## Quick Recovery Commands
```bash
# Activate venv
source /home/ubuntu/urbanjepa-venv/bin/activate

# Project root
cd /mnt/eskeetit/Code-server/UrbanJEPA

# Check tile count
find data/ortho/tiles -name "*.jpg" | wc -l
du -sh data/ortho/

# Run smoke test
python scripts/smoke_test.py

# Restart Phase 2 from best checkpoint
python -m src.training.train_jepa --data_dir data/ortho --resume models/checkpoints/jepa_best.pt
```

## Phase 0: Setup & Data Acquisition
- [x] Repository structure created
- [x] Python venv at `/home/ubuntu/urbanjepa-venv/` with all deps (torch, timm, diffusers, rasterio, matplotlib, tqdm, requests)
- [x] Pretrained ViT-B/16 weights: `models/ijepa/vit_base_patch16_224_imagenet.pt` (timm ImageNet-1K, 86.6M params)
- [x] Pretrained SD-VAE weights: `models/vae/sd_vae_ft_mse.pt` (stabilityai/sd-vae-ft-mse, 4 latent channels)
- [x] **H3MRL ortho tiles**: `data/ortho/tiles/` — 4,154 non-white L20 tiles (4096×4096, ~15cm/px), + 10,212 JGW world files. Copied from H3MRL project.
- [x] `data/ortho/metadata/` — manifest_nonwhite.json, filter_report.csv, input_files_nonwhite.txt
- [x] `scripts/download_ortho.py` — rewritten with H3MRL spiral /export approach (resumable, L18-L21)
- [x] `scripts/import_h3mrl_tiles.py` — copies and processes tiles from H3MRL to UrbanJEPA
- [ ] Planet API tiles (deferred to Phase 6)

## Phase 1: Data Pipeline
- [x] `src/data/ortho_dataset.py` — OrthoDataset with train/val split at tile level. Random 256×256 crops from 4096×4096 tiles (train), deterministic 16×16 grid (val). Downsampling: area avg at scales [18, 20, 22] → 2.7m, 3.0m, 3.3m/px. Augmentations: hflip, vflip, rot90, brightness, contrast, sensor noise, cloud masks.
- [x] `src/data/__init__.py`
- [x] `create_dataloaders()` helper in ortho_dataset.py
- [x] Smoke test passed: `scripts/smoke_test.py` — 3,738 train tiles (~957K possible patches), 416 val tiles (~106K grid patches), L20
- [ ] PlanetDataset for evaluation (`src/data/planet_dataset.py`) — not started
- [ ] `configs/` YAML config files — not started

## Downsampling Strategy (CRITICAL)
- **Inference inputs are ALWAYS PlanetScope ~3m/px** — training scales must cluster tightly around this
- Scales [18, 20, 22]: 2.7m, 3.0m, 3.3m (scale 20 = 2.99m, exact PlanetScope match)
- Training on easier scales (3, 5, 10 = 45cm-1.5m) is wasted — model would learn to expect detail not present at inference
- Augmentations on low-res branch (noise, blur, cloud masks) simulate real Planet artifacts

## Phase 2: Model Components

### Encoder
- [x] `src/models/encoder.py` — UrbanEncoder, ViT-B/16 via timm, 768-dim, 12 layers, loads ImageNet-1K weights
- [x] `src/models/__init__.py`
- [x] `src/__init__.py`

### Predictor (γ)
- [x] `src/models/predictor.py` — FeaturePredictor, uses pretrained ViT-B transformer blocks directly (not forward_features). Learnable mask_token + pos_embed. Predicts embeddings for masked positions from context features.
- [x] **FIXED** — Rewritten to accept token sequences (not images), pos_embed interpolated for 256×256 (was 224×224), pos_embed indexing fixed (4D→3D bug)

### Denoising MLP (εθ)
- [x] `src/models/denoising_mlp.py` — 6 residual blocks, 1024 hidden, AdaLN conditioning, sinusoidal time embedding
- [x] **token_dim=16** (was 4, now 16 after 2×2 VAE latent grouping — 4 channels × 2×2 spatial = 16)
- [x] Verified correct with round-trip encode/decode test (error = 0.00)

### Full Model
- [x] `src/models/urbanjepa.py` — Done. Unified 256-token space (2×2 VAE latent grouping). context_encoder, target_encoder (EMA), feature_predictor, denoising_mlp, projection_head, train_for_phase(), diffusion_loss(), encode/decode_to_latent(), ddpm_sample().
- [x] `src/models/cnn_decoder.py` — Done (Phase 3 CNN decoder for JEPA validation)

### Training Infrastructure
- [x] `src/training/losses.py` — Done. LinearNoiseSchedule with q_sample, p_sample, p_sample_loop (strided DDPM reverse).
- [x] `src/training/train_jepa.py` — Done. Phase 2 training: mixed precision (autocast), gradient accumulation, checkpointing, resume, ETA logging, validation progress logging.
- [x] `src/training/train_djepa.py` — Done. Phase 4a (MLP only) + Phase 4b (joint). Phase transition support, PSNR tracking (step + epoch level), TensorBoard sample images.
- [x] `src/training/__init__.py` — Done (exists, empty)
- [x] `src/training/train_decoder.py` — Done (Phase 3 CNN decoder training + VAE fine-tune support)

### Phase 2 Training Results (2026-05-14)
- [x] Phase 2 training complete — early stopped at epoch 12, best epoch 2
- [x] Best val_loss: 0.0336 (Smooth L1 on 768-dim embeddings) — passes cosine similarity gate
- [x] Checkpoint saved: `models/checkpoints/jepa_best.pt`
- [x] Validation optimized: 4 patches/tile (1,664 samples), val time ~2m40s (was 67 min)
- [x] ETA/progress logging added to both train and val loops
- [x] **Issue identified**: EMA decay 0.9999 too slow → JEPA representation drift after epoch 2. Both train/val rose together (not overfitting). Fix for Phase 4: lower LR or EMA schedule.

## Phase 3: CNN Decoder Validation
- [x] Simple CNN decoder code exists: project embeddings → spatial → transposed convs → 256×256 RGB
- [x] **SKIPPED** — JEPA val_loss=0.0336 was strong enough, proceeded directly to Phase 4

## Phase 4: Full D-JEPA Training
- [x] 4a: Train denoising MLP only (frozen JEPA), 50 epochs. Ld plateaued at 0.165-0.170. Did NOT hit Ld<0.10 gate.
- [x] 4b: Joint fine-tune all components, 20 epochs. **IN PROGRESS** (epoch 4 as of May 14 23:00). Lp stabilized but Ld still at plateau.
- [ ] 4b results & decisions (after training completes or early stops)
- [ ] Tier 1 interventions if plateau persists: cosine noise schedule, importance-weighted t-sampling, lower LR
- [ ] Tier 2 if needed: VAE fine-tune, cross-token attention, more noise samples

## Phase 5: Autoregressive Sampling
- [ ] `sample()` method on UrbanJEPA model
- [ ] Cosine schedule for 64 AR steps
- [ ] 100-step DDPM denoising per token group
- [ ] Classifier-free guidance (CFG) implementation

## Phase 6: Evaluation
- [ ] `src/evaluation/metrics.py` — PSNR, SSIM, FID, edge sharpness
- [ ] `src/evaluation/visualize.py` — Tile map overlays, pair visualizations
- [ ] `src/evaluation/__init__.py` (exists, empty)
- [ ] Planet evaluation script
- [ ] Baselines: bicubic, ESRGAN

## Phase 7: VAE Decoder Fine-Tuning (Optional)
- [ ] Fine-tune SD-VAE decoder on ortho patches with perceptual loss

---

## Key Files Reference

| File | Purpose | Status |
|------|---------|--------|
| `scripts/download_ortho.py` | Download Toronto tiles via spiral /export (resumable) | Done |
| `scripts/import_h3mrl_tiles.py` | Copy/process tiles from H3MRL project | Done |
| `scripts/download_pretrained.py` | Download ViT-B + SD-VAE weights | Done |
| `scripts/smoke_test.py` | End-to-end data pipeline test | Done |
| `scripts/diagnose_phase2.py` | Phase 2 diagnostic analysis | Done |
| `src/data/ortho_dataset.py` | Self-supervised ortho dataset (random 256×256 crops, val_patches_per_tile) | Done |
| `src/models/encoder.py` | ViT-B/16 context/target encoder | Done |
| `src/models/predictor.py` | Feature predictor (γ) — token-sequence transformer | Done |
| `src/models/denoising_mlp.py` | Denoising MLP (εθ) — token_dim=16, 6 blocks, 1024 hidden | Done |
| `src/models/urbanjepa.py` | Full D-JEPA model — unified 256-token space, ddpm_sample, PSNR | Done |
| `src/models/cnn_decoder.py` | Phase 3 CNN decoder (not used — Phase 3 skipped) | Done |
| `src/training/train_jepa.py` | Phase 2 JEPA fine-tuning loop | Done |
| `src/training/train_djepa.py` | Phase 4a/4b D-JEPA training — phase transition, PSNR tracking | Done |
| `src/training/train_decoder.py` | Phase 3 decoder training + VAE fine-tune support | Done |
| `src/training/losses.py` | LinearNoiseSchedule — q_sample, p_sample, p_sample_loop | Done |
| `data/ortho/tiles/` | 4,154 L20 tiles (4096×4096, ~15cm/px) | Done |
| `data/ortho/metadata/` | manifest_nonwhite.json, filter_report.csv | Done |
| `models/checkpoints/jepa_best.pt` | Best Phase 2 checkpoint (epoch 2, val=0.0336) | Done |
| `models/checkpoints/djepa_mlp_best.pt` | Best Phase 4a checkpoint (epoch 4, val_total=0.2022) | Done |

## Bugs Discovered & Fixed (2026-05-14)
1. **Phase transition optimizer mismatch** — 4a optimizer (MLP only) incompatible with 4b (MLP+JEPA dual-LR). Fixed by detecting phase mismatch and loading weights only.
2. **ddpm_sample N=256 hardcoded** — used `self.num_patches` instead of conditioning tensor's actual N. Fixed.
3. **sample_mask returns indices, not boolean** — PSNR code treated indices as boolean mask, `.nonzero()` dropped position 0. Fixed by using indices directly.

## Quick Recovery Commands
```bash
# Activate venv
source /home/ubuntu/urbanjepa-venv/bin/activate

# Project root
cd /mnt/eskeetit/Code-server/UrbanJEPA

# Check tile count
find data/ortho/tiles -name "*.jpg" | wc -l

# Run smoke test
python scripts/smoke_test.py

# Resume Phase 2
python -m src.training.train_jepa --data_dir data/ortho --resume models/checkpoints/jepa_best.pt

# Resume Phase 4b from 4a best
python -m src.training.train_djepa --phase 4b --data_dir data/ortho --log_dir runs \
    --resume models/checkpoints/djepa_mlp_best.pt

# Full 4b with cosine schedule + lower LR (if plateau persists)
python -m src.training.train_djepa --phase 4b --data_dir data/ortho --log_dir runs \
    --resume models/checkpoints/djepa_mlp_best.pt --lr_mlp 2e-4 --lr_jepa 2e-5 --epochs 50

# TensorBoard
tensorboard --logdir runs/
```
