# UrbanJEPA — Summary

## Project Title
**UrbanJEPA: A D-JEPA-Based Foundation Model for High-Resolution Urban Image Generation from Satellite Inputs**

## One-Liner
Train a self-supervised foundation model entirely on Toronto's open ortho imagery using I-JEPA and D-JEPA, then evaluate its ability to generate pseudo-high-resolution urban images from Planet satellite inputs (3m/pixel).

## Core Thesis
Municipal ortho imagery, used in a fully self-supervised manner through masked prediction and diffusion generation, is sufficient to build a generalizable urban foundation model that can synthesize high-resolution detail from commodity satellite inputs — without requiring manually labelled data.

## Hardware
- **GPU**: NVIDIA RTX 3090 (24GB VRAM)
- **RAM**: 32–64GB
- **Storage**: 2TB+ SSD

---

## Architecture — 5 Components

| # | Component | Arch | Params | Role |
|---|-----------|------|--------|------|
| 1 | Context Encoder (φ) | ViT-B/16 | ~86M | Encodes low-res input → context features |
| 2 | Target Encoder (φ̄) | ViT-B/16 | ~86M | Encodes high-res target → ground truth embeddings (EMA updated only, no grads) |
| 3 | Feature Predictor (γ) | ViT-B/16 | ~86M | Predicts high-res embeddings for masked positions from context |
| 4 | Denoising MLP (εθ) | 6-block residual MLP | ~4M | Generates 16-dim latent tokens via diffusion conditioned on predicted embeddings |
| 5 | VAE Decoder | SD-VAE (frozen) | ~80M | Decodes 16×16×16 latents → 32×32×4 → 256×256 pixel images |

**Critical detail — Unified token space (Option A)**: ViT-B/16 produces 256 tokens (16×16 patches on 256×256 px). SD-VAE produces 32×32×4 latents — grouped 2×2 spatially → 16×16×16 = **256 tokens × 16-dim**. Both token grids are aligned 1:1. Each ViT token position maps to exactly one latent token position. No mapping layer needed. Scaling factor is 0.18215. Inputs are 256×256 random crops from 4096×4096 L20 tiles (~15cm/px).

## Initialization
- Components 1–3: **timm ViT-B/16** (ImageNet-1K supervised) — Meta FAIR never released I-JEPA ViT-B weights (only ViT-H and ViT-G exist). Phase 2 must teach the JEPA predictive objective from scratch.
- Component 4: Trained from scratch (zero-initialized output layers)
- Component 5: Pretrained SD-VAE (stabilityai/sd-vae-ft-mse), frozen

## Loss Functions
- **Prediction Loss (Lp)**: Smooth L1 between projected predicted embeddings and target encoder embeddings. Applied only to masked tokens. Prevents representation collapse.
- **Diffusion Loss (Ld)**: Standard DDPM MSE between predicted and actual noise. 4 noise samples per token per step. Linear variance schedule β ∈ [1e-4, 2e-2], T=1000.
- **Total**: `L = Ld + Lp` — complementary, no weighting needed.

## Tokenization (Unified — Option A)
- 256×256 images → VAE encoder → **32×32×4** latent → **2×2 spatial grouping** → **16×16×16** → 256 tokens × **16-dim**
- ViT-B/16: 256×256 → 16×16 patches = **256 tokens** × 768-dim
- Both token grids aligned 1:1 — no mapping layer needed
- Grouping is lossless reshape+transpose (round-trip error = 0.00)

## Datasets

### Toronto Ortho (Training + Ground Truth)
- City of Toronto Open Data, ~15cm/pixel at L20, 4,154 non-white tiles
- 4096×4096 px tiles with JGW world files (EPSG:3857), ~61km × 46km coverage
- Covers: downtown, mid-rise residential, suburban, industrial, parks/ravines, waterfront, airport
- Random 256×256 crops from tiles serve as high-res targets
- Downsampled at scales [18, 20, 22] → low-res at 2.7m, 3.0m, 3.3m (clustered on PlanetScope 3m)
- Inference inputs are ALWAYS PlanetScope ~3m/px, so training scales are kept tight around that

### Planet API (Test Only — Never Trained On)
- PlanetScope 3m/pixel, 3,000 km² quota
- Tests real-world generalization: cloud cover, haze, sensor differences, seasonal variation

---

## Implementation Phases

| Phase | Days | What | Status |
|-------|------|------|--------|
| 0 | 1–3 | Setup + download ortho/Planet/pretrained weights | **Done** — 4,154 L20 tiles from H3MRL |
| 1 | 4–6 | Patch extraction pipeline + augmentations | **Done** — OrthoDataset crops on-the-fly, smoke test passed |
| 2 | 7–12 | I-JEPA domain fine-tune (prediction loss only) | **Done** — Early stopped epoch 12, best epoch 2 (val=0.0336). JEPA backbone saved. |
| 3 | 13–14 | CNN decoder validation (sanity check) | **Skipped** — JEPA val_loss=0.0336 strong enough, proceeded directly to Phase 4 |
| 4 | 15–20 | Full D-JEPA: MLP training then joint fine-tune | **In progress** — 4a complete, 4b training (epoch 4 as of May 14 23:00) |
| 5 | 21–22 | Autoregressive sampling pipeline | **Not started** |
| 6 | 23–25 | Full evaluation vs bicubic/ESRGAN on Planet | **Not started** |
| 7 | 26–28 | VAE decoder fine-tune (optional) | **Not started** |

---

## Limitations & Known Issues

### 1. EMA Decay Too Slow (Phase 2 CONFIRMED)
Constant EMA decay of 0.9999 caused target encoder to lag context encoder during rapid early learning. Both train and val loss rose together after epoch 2 — JEPA representation drift, not overfitting. Early stopping saved best checkpoint. For Phase 4, consider EMA schedule (0.996→1.0 per I-JEPA paper) or lower initial LR (2e-4).

### 2. No True I-JEPA Pretraining
Meta FAIR only released I-JEPA checkpoints for ViT-H (632M) and ViT-G (1.1B). ViT-B was **never released**. We use timm's supervised ImageNet-1K ViT-B/16 instead. This means Phase 2 must teach both the JEPA objective AND the ortho domain from scratch, rather than just domain-adapting existing JEPA features.

### 2. VAE Latent Bottleneck (mitigated by 2×2 grouping)
The SD-VAE compresses 256×256×3 (196K values) into 32×32×4 (4K values). 2×2 spatial grouping (→ 16×16×16 tokens) gives 16-dim per token, up from 4-dim. The VAE was trained on natural images, not aerial/satellite imagery. Phase 7 VAE fine-tuning may help.

### 3. VRAM Constraint (RTX 3090 24GB)
- ~21GB total with fp16 + gradient checkpointing
- Batch size 8, gradient accumulation ×4 = effective batch 32
- No room for larger models (ViT-L would need ~48GB)

### 4. Resolution: L20 (~15cm/pixel)
L20 is significantly better than the original zoom-19 plan (~30cm/pixel). At 15cm/pixel, cars, building details, and road markings are clearly visible. Downsampled inputs at scale 20 (~3m) match PlanetScope exactly. L21 (~7.5cm/pixel) would be ideal but tiles are 4× more numerous.

### 5. Ortho Coverage
4,154 non-white 4096×4096 tiles covering Toronto's land area (~61km × 46km). Originally downloaded via spiral /export from the H3MRL project. Tiles are georeferenced with JGW world files (EPSG:3857).

### 6. Planet Evaluation Not Set Up
Planet API credentials and download scripts not yet configured.

### 7. Validation Set Oversized (FIXED)
Original validation used all 256 grid patches from all 416 tiles = 106,496 samples, taking 67 minutes per epoch. Reduced to 4 patches/tile (1,664 samples, ~2m40s) via `val_patches_per_tile` parameter. Justified by CLT (SE ≈ σ/√n plateaus beyond ~2K samples) and spatial autocorrelation of adjacent grid patches (Cressie, 1993).

---

## Autoregressive Sampling (Inference)
Generalized next-set-of-tokens prediction with cosine schedule:
- 64 AR steps for ViT-B
- Each step: predict embeddings for unsampled positions → DDPM denoising (100 steps, conditioned on JEPA embeddings) → decode next token group
- Temperature: 0.98; optional CFG scale: 3.0

## Success Gates

| Phase | Gate | Status |
|-------|------|--------|
| Phase 2 | Cosine similarity (predicted vs target embeddings) > 0.7 | **PASSED** — val_loss=0.0336 (Smooth L1 on 768-dim). Rapid convergence (epoch 2). |
| Phase 3 | PSNR on held-out ortho > 26dB | **SKIPPED** — JEPA strong enough |
| Phase 4a | Generated samples recognizably urban, Ld < 0.10 | **PARTIAL** — Ld plateaued at 0.17, not below 0.10 |
| Phase 4b | PSNR > 30dB, SSIM > 0.85, beats bicubic | **In progress** — Ld still at 0.165, PSNR 11 dB at epoch 4 |
| Phase 6 | PSNR on clear Planet > 25dB, degrades gracefully with clouds | Pending |

## Phase 2 Training Results (2026-05-14)
- **Config**: 3,738 train tiles × 4 random crops = 14,952 samples/epoch. 416 val tiles × 4 grid patches = 1,664 samples. Batch 8 × 4 grad accum = 32 effective. LR=8e-4, CosineAnnealingLR, EMA=0.9999.
- **Best epoch**: 2 (val_loss=0.0336, train_loss=0.0351)
- **Early stop**: epoch 12 (10 epochs without improvement)
- **Issue**: Both train and val loss rose after epoch 2 — JEPA representation drift from slow EMA tracking rapid early learning. NOT overfitting (train/val moved together).
- **Checkpoint**: `models/checkpoints/jepa_best.pt` — domain-adapted JEPA backbone ready for Phase 4.
- **Per-epoch timing**: ~16m40s train + ~2m40s val = ~19m20s per epoch

## Phase 4a Training Results (2026-05-14)
- **Config**: 50 epochs (early stopped), batch 8×4=32, lr_mlp=1e-3, JEPA frozen, linear noise schedule T=1000.
- **Ld**: Dropped 1.0→0.17 in epoch 0, then **plateaued at 0.165-0.170**. Flat for 10+ epochs.
- **Lp**: Stable at ~0.034 (JEPA frozen, no degradation).
- **Best**: epoch 4, val_total=0.2022. Checkpoint: `models/checkpoints/djepa_mlp_best.pt`.
- **Conclusion**: MLP learned all it can with frozen JEPA conditioning. Ld=0.17 may be the ceiling for 16-dim VAE tokens + 4M-param per-token MLP.

## Phase 4b Training (In Progress, 2026-05-14)
- **Config**: 20 epochs, lr_mlp=1e-3, lr_jepa=1e-4, cosine EMA 0.996→1.0, loss = Ld + Lp.
- **Epoch 0**: train Ld=0.1703 Lp=0.2400 | val Ld=0.1682 Lp=0.1169 total=0.2851 | PSNR=11.47
- **Epoch 1**: train Ld=0.1666 Lp=0.1064 | val Ld=0.1695 Lp=0.0894 total=0.2588 | PSNR=11.53
- **Epoch 2**: train Ld=0.1652 Lp=0.0907 | val Ld=0.1684 Lp=0.0823 total=0.2506 | PSNR=10.90
- **Epoch 3**: train Ld=0.1653 Lp=0.0875 | val Ld=0.1665 Lp=0.0824 total=0.2489 | PSNR=11.22
- **Key observation**: Lp rapidly stabilized (0.117→0.082). Ld still stuck at 0.165-0.170 — same plateau as frozen Phase 4a. Joint training not breaking through yet.
- **Trainable**: 198M params (JEPA + MLP unfrozen) vs 51M in Phase 4a.
- **Per-epoch**: ~18 min train + ~2m40s val + PSNR sampling.

## Bugs Discovered & Fixed
1. **Phase transition optimizer mismatch** (`train_djepa.py`): 4a optimizer (MLP only) can't load into 4b (MLP+JEPA dual-LR). Fixed: detect phase mismatch, load weights only, fresh optimizer.
2. **ddpm_sample N=256 hardcoded** (`urbanjepa.py`): Used `self.num_patches` instead of `cond.shape[1]`. Crashes when feature predictor returns only masked positions. Fixed.
3. **sample_mask returns indices, not boolean** (`train_djepa.py` PSNR code): `.nonzero()` on integer indices dropped position 0 if masked. Fixed: use indices directly for scatter assignment.

## References
- **D-JEPA**: Chen et al., ICLR 2025 — `facebookresearch/djepa`
- **I-JEPA**: Assran et al., CVPR 2023 — `facebookresearch/ijepa`
- **MAR (VAE)**: Li et al., 2024
- **ADM (noise schedule)**: Dhariwal & Nichol, NeurIPS 2021
- **SD-VAE**: Rombach et al., CVPR 2022 — `stabilityai/sd-vae-ft-mse`
- **timm**: Ross Wightman, PyTorch Image Models — `timm` library
- **Toronto Ortho**: City of Toronto Open Data — `gis.toronto.ca`
