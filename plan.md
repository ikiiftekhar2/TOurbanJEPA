# Toronto Urban Foundation Model — Full Project Plan

## Project Title
**UrbanJEPA: A D-JEPA-Based Foundation Model for High-Resolution Urban Image Generation from Satellite Inputs**

## One-Line Summary
Train a self-supervised foundation model entirely on Toronto's open ortho imagery (8cm/pixel) using I-JEPA and D-JEPA, then evaluate its ability to generate pseudo-high-resolution urban images from Planet satellite inputs (3m/pixel).

## Core Thesis
Municipal ortho imagery, used entirely in a self-supervised manner through masked prediction and diffusion generation, is sufficient to build a generalizable urban foundation model that can synthesize high-resolution detail from commodity satellite inputs — without requiring manually labelled data.

---

## CURRENT STATE (2026-05-14 23:00 UTC) — Updated After Phase 4a + Phase 4b Epoch 4

### What's Done
- **Phase 0**: venv at `/home/ubuntu/urbanjepa-venv/`, all packages installed (PyTorch 2.5.1, CUDA 12.1, timm, diffusers, accelerate, etc.)
- **Pretrained weights**: ViT-B/16 (timm ImageNet-1K) at `models/ijepa/vit_base_patch16_224_imagenet.pt`, SD-VAE (stabilityai/sd-vae-ft-mse)
- **Ortho tiles**: 4,154 non-white tiles at `data/ortho/tiles/` — 4096×4096 px, L20 (~15cm/pixel), full Toronto coverage
- **Phase 1**: `src/data/ortho_dataset.py` — OrthoDataset with train/val split, scales [18,20,22], augmentations. 3,738 train tiles, 416 val tiles.
- **Phase 2 complete**: JEPA trained 12 epochs, early stopped. Best: epoch 2, val_loss=0.0336. Checkpoint: `models/checkpoints/jepa_best.pt`. Representation drift after epoch 2 due to slow EMA (0.9999).
- **Phase 3 SKIPPED**: JEPA val_loss=0.0336 was strong enough — CNN decoder sanity check not needed.
- **Phase 4a complete**: Denoising MLP trained 10+ epochs with frozen JEPA. Ld dropped 1.0→0.17 in 1 epoch then **plateaued at 0.165-0.170**. Did NOT hit Ld<0.10 gate. Best checkpoint: `models/checkpoints/djepa_mlp_best.pt` (epoch 4, val_total=0.2022).
- **Phase 4b launched** (2026-05-14 ~22:27 UTC): Joint training — 198M params unfrozen, cosine EMA 0.996→1.0, lr_mlp=1e-3, lr_jepa=1e-4, 20 epochs.
- **PSNR tracking**: Step-level (every 100 batches, 1 image, 20 DDPM steps) and epoch-level (first val batch, 50 DDPM steps). Logged to TensorBoard.
- **Phase 4b current** (epoch 4): Lp stabilized (0.117→0.082), but **Ld still plateaued at 0.165-0.170** — same as Phase 4a. PSNR flat at ~11 dB (50-step DDPM). Joint training not breaking the plateau yet.

### Key Architectural Details
- **Unified token space (Option A)**: VAE latents 32×32×4 → group 2×2 → 256 tokens × 16-dim. ViT-B/16 also produces 256 tokens. 1:1 alignment.
- **Denoising MLP**: 6 residual blocks, 1024 hidden dim, AdaLN conditioning. Per-token (no cross-token attention). ~4M params.
- **Noise schedule**: Linear β∈[1e-4, 2e-2], T=1000. 4 noise samples per token per step.
- **Loss**: Ld (MSE noise prediction on 16-dim VAE latents) + Lp (Smooth L1 on 768-dim JEPA embeddings).
- **Downsampling**: scales [18, 20, 22] → 2.7m, 3.0m, 3.3m/px. Scale 20 = 2.99m matches PlanetScope 3m exactly.

### Bugs Discovered & Fixed (2026-05-14)
1. **Phase transition optimizer mismatch**: Phase 4a saves MLP-only optimizer; Phase 4b creates MLP+JEPA dual-LR optimizer. `load_phase4()` failed when param groups differed. Fixed: detect phase mismatch, load model weights only, start fresh optimizer.
2. **ddpm_sample hardcoded N=256**: Used `self.num_patches` instead of conditioning tensor's actual N. When feature predictor returns only masked positions (~121), shape mismatch crashed. Fixed: use `cond.shape[1]`.
3. **sample_mask returns indices, not boolean**: `.nonzero()` on integer indices drops position 0 if masked, causing N_mask≠N_pred scatter crash. Fixed: use mask indices directly — `combined[b, mask_indices] = generated_tokens[b]`.

### What's NOT Done
- Planet dataset, Planet evaluation, geographic holdout splits
- Autoregressive sampling pipeline (Phase 5)
- VAE decoder fine-tuning (Phase 7)
- Classifier-free guidance (CFG) implementation

### Critical Open Question
**Ld plateau at 0.165-0.170** — same with frozen JEPA (Phase 4a) and joint training (Phase 4b so far). Either:
- More epochs needed for EMA to catch up (cosine 0.996→1.0, still early)
- Or: fundamental ceiling from 16-dim VAE latent space / per-token MLP architecture

### Recovery Commands
```bash
source /home/ubuntu/urbanjepa-venv/bin/activate
cd /mnt/eskeetit/Code-server/UrbanJEPA

# Resume Phase 4b from best checkpoint
python -m src.training.train_djepa --phase 4b --data_dir data/ortho --log_dir runs \
    --resume models/checkpoints/djepa_mlp_best.pt

# Resume Phase 4b from a 4b checkpoint (same phase)
python -m src.training.train_djepa --phase 4b --data_dir data/ortho --log_dir runs \
    --resume models/checkpoints/djepa_joint_epoch_N.pt

# TensorBoard
tensorboard --logdir runs/

# Full 4b with cosine schedule + lower LR (if plateau persists)
python -m src.training.train_djepa --phase 4b --data_dir data/ortho --log_dir runs \
    --resume models/checkpoints/djepa_mlp_best.pt --lr_mlp 5e-4 --lr_jepa 5e-5
```

---

## Hardware & Environment

### Hardware
- GPU: NVIDIA RTX 3090 (25.3GB VRAM)
- RAM: Minimum 32GB system RAM recommended (64GB ideal for data loading)
- Storage: Minimum 2TB SSD (ortho tiles are large)

### Software Stack
```
Python          3.10+
PyTorch         2.2+         (CUDA 12.1)
torchvision     0.17+
diffusers       0.27+        (for SD-VAE decoder)
huggingface_hub 0.22+        (model downloads)
timm            0.9+         (ViT implementations)
rasterio        1.3+         (GeoTIFF reading)
GDAL            3.7+         (coordinate transforms)
geopandas       0.14+        (spatial operations)
numpy           1.26+
Pillow          10+
tqdm            4+
wandb           0.16+        (experiment tracking)
planet          2.3+         (Planet SDK)
requests        2.31+        (ortho tile downloads)
```

### Environment Setup
```bash
conda create -n urbanjepa python=3.10
conda activate urbanjepa
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install diffusers transformers huggingface_hub timm
pip install rasterio geopandas shapely
pip install wandb tqdm Pillow numpy
pip install planet requests
conda install -c conda-forge gdal
```

---

## Datasets

### Dataset 1: Toronto Ortho (Primary — Training + Ground Truth)
- **Source**: City of Toronto Open Data (ArcGIS MapServer /export endpoint)
- **URL**: https://gis.toronto.ca/arcgis/rest/services/basemap/cot_ortho/MapServer
- **Resolution**: ~15cm/pixel (L20, 4096×4096 px tiles)
- **Format**: JPEG tiles + JGW world files, RGB
- **Coverage**: ~2,800 km² bounding box (~61km × 46km), ~4,154 valid land tiles after filtering
- **Role**: Self-supervised training pairs (both input and target generated from this alone). Random 256×256 crops from 4096×4096 tiles serve as high-res targets. Low-res inputs created by area-average downsampling at scales [3,5,10,20].
- **Size on disk**: ~16GB (JPEG compressed 4096×4096 tiles)
- **Coordinate system**: EPSG:3857 (Web Mercator, from JGW world files)
- **Source project**: Tiles originally downloaded for H3MRL project via spiral /export scraper

#### What Toronto Ortho Covers (why this matters for diversity)
- Dense downtown core (CN Tower area, financial district)
- Mid-rise residential (Annex, Rosedale, Forest Hill)
- Suburban sprawl (Scarborough, Etobicoke, North York)
- Industrial zones (Port Lands, Weston Road corridor)
- Parks and ravines (Don Valley, High Park, Rouge Park)
- Waterfront (Lake Ontario shoreline, Toronto Islands)
- Toronto Pearson Airport (large impervious surfaces, runways)
- Mixed-use commercial corridors (Yonge St, Bloor St, Eglinton)

This diversity is a core strength — the model sees every type of urban structure.

### Dataset 2: Planet API (Test Only — No Training)
- **Source**: Planet Labs API
- **API**: Planet SDK for Python
- **Resolution**: 3m/pixel (PlanetScope)
- **Quota**: 3,000 km² area + 100,000 scene tiles + 100,000 basemap tiles
- **Format**: GeoTIFF, 4-band (RGB + NIR) or RGB
- **Role**: Real-world generalization test — never used in training
- **Key challenges introduced**:
  - Cloud cover and partial occlusion
  - Atmospheric haze
  - Sensor-specific spectral response (different from ortho)
  - Seasonal variation (snow, leaf-off trees)
  - Compression artifacts
  - Radiometric differences (sun angle, shadow patterns)
- **Coordinate system**: WGS84 (EPSG:4326) — must reproject to match ortho

#### Planet Quota Strategy
- Reserve ~200 km² for downtown/dense urban (model's hardest test)
- Reserve ~200 km² for suburban (medium difficulty)
- Reserve ~100 km² for industrial/airport (distinctive structure)
- Reserve ~100 km² for parks/waterfront (natural texture test)
- Keep ~2,400 km² unused initially (use only what you need)

---

## Architecture

### Overview: Three Components from D-JEPA Paper

```
Context Encoder  (ϕ)    ← I-JEPA pretrained ViT-B, fine-tuned
Target Encoder   (ϕ̄)    ← I-JEPA pretrained ViT-B, EMA updated only
Feature Predictor (γ)   ← I-JEPA pretrained ViT-B, fine-tuned
Denoising MLP    (εθ)   ← trained from scratch, small network
VAE Decoder             ← MAR's pretrained VAE, optionally fine-tuned
```

### Component 1: Context Encoder (ϕ)
- **Architecture**: Vision Transformer Base (ViT-B)
- **Patch size**: 16×16
- **Embedding dimension**: 768
- **Depth**: 12 transformer blocks
- **Heads**: 12 attention heads
- **Parameters**: ~86M
- **Initialization**: I-JEPA pretrained weights (Meta FAIR, trained on ImageNet)
- **Download**: `https://github.com/facebookresearch/ijepa` (ViT-B/16 checkpoint)
- **NOTE**: Meta FAIR never released ViT-B I-JEPA. Using timm ImageNet-1K ViT-B/16 instead.
- **Training**: Fine-tuned with gradient descent
- **Input**: Low-resolution satellite patch (downsampled ortho or Planet tile)
- **Output**: Context feature embeddings for unmasked (visible) tokens

### Component 2: Target Encoder (ϕ̄)
- **Architecture**: Identical to context encoder (ViT-B)
- **Initialization**: Same I-JEPA pretrained weights as context encoder
- **Training**: NOT trained with gradient descent — updated via Exponential Moving Average (EMA) of context encoder only
- **EMA decay**: α = 0.9999 (constant, no warmup needed — diffusion loss prevents collapse)
- **Update rule**: `ϕ̄ ← 0.9999 * ϕ̄ + 0.0001 * ϕ`
- **Input**: High-resolution ortho patch (all tokens, unmasked)
- **Output**: Target embeddings (ground truth for the feature predictor to match)
- **Note**: Not used at inference time — only during training. Parameters excluded from model parameter count.

### Component 3: Feature Predictor (γ)
- **Architecture**: Vision Transformer Base (ViT-B) — identical structure to encoders
- **Initialization**: I-JEPA pretrained weights
- **Training**: Fine-tuned with gradient descent
- **Input**: Context features from context encoder + positional information about masked tokens
- **Output**: Predicted embeddings (zi) for each masked token position
- **Role**: The bridge between low-res context and high-res prediction

### Component 4: Denoising MLP (εθ)
- **Architecture**: Small MLP with residual blocks
- **Structure per block**: LayerNorm → Linear → SiLU → Linear → residual connection
- **Number of residual blocks**: 6 (for ViT-B scale, matching D-JEPA paper)
- **Hidden dimension**: 1024
- **Conditioning**: zi (predicted embedding) added to time step embedding via AdaLN (Adaptive Layer Norm)
- **Parameters**: ~4M (small by design — one MLP per token, not for entire image)
- **Initialization**: Random (Xavier uniform for linear layers, zero for final output layer)
- **Training**: Trained from scratch with diffusion loss
- **Input**: Noisy token xit, timestep t, predicted embedding zi
- **Output**: Predicted noise ε (used to compute denoising step)

### Component 5: VAE Decoder
- **Source**: MAR paper (Li et al. 2024) — same VAE D-JEPA paper uses
- **Alternative**: `stabilityai/sd-vae-ft-mse` from HuggingFace (Stable Diffusion VAE)
- **Download**:
```python
from diffusers import AutoencoderKL
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
decoder = vae.decoder
```
- **Input**: Latent token vectors (after diffusion sampling)
- **Output**: Pixel-space image patches (actual viewable images)
- **Training**: Initially frozen. Fine-tuned on ortho patches in Stage 3.
- **Latent dimension**: **4 channels** (verified by loading stabilityai/sd-vae-ft-mse — NOT 16 as initially assumed)

### Loss Functions

**Prediction Loss (Lp) — JEPA component**
```
Lp = Σ smooth_L1(uθ(zi), gi)
```
- `zi` = predicted embedding from feature predictor
- `gi` = target embedding from target encoder (stop gradient applied)
- `uθ` = two-layer MLP projection head (~2M params)
- Smooth L1 chosen for stability (following D-JEPA paper)
- Applied only to masked tokens

**Diffusion Loss (Ld) — generative component**
```
Ld = E[||ε - εθ(xit | t, zi)||²]
```
- `ε` = sampled noise from N(0, I)
- `xit` = noise-corrupted token at timestep t
- `εθ` = denoising MLP
- `t` sampled 4 times per token during training to maximize diffusion loss signal
- Linear variance schedule: β range [1e-4, 2e-2], tmax = 1000 steps

**Total Loss**
```
L = Ld + Lp
```
No weighting needed — these losses are complementary, not conflicting.

---

## Masking Strategy

During training, tokens from the high-res target patch are randomly masked:
- Masking ratio sampled from truncated normal distribution
- Mean: 1.0, std: 0.25, lower bound: 0.75
- Result: 75–100% of tokens masked each step
- The context encoder sees the low-res input (no masking on context)
- The feature predictor predicts embeddings for all masked positions
- High masking ratio reduces memory and training time while improving generalization

---

## Tokenization

### Unified Token Space (Option A — implemented 2026-05-14)

ViT and VAE latent share the same 256-token grid via 2×2 spatial grouping:

**ViT-B/16 path (JEPA):**
- Input: 256×256 pixel crop from 4096×4096 L20 tile (~15cm/pixel source)
- Patch size 16×16: 256/16 = 16 → 16×16 = **256 tokens**
- Each token: **768-dimensional** ViT embedding

**VAE latent path (diffusion):**
- Input: 256×256 pixel crop — VAE encode → **32×32×4** latent
- Group 2×2 spatial blocks: each group = 2×2 positions × 4 channels = **16 values**
- Result: 16×16×16 grid → flatten → **256 tokens × 16-dim**
- This is a lossless reshape+transpose (verified: round-trip error = 0.00)

**Why this matters:** Every ViT token position (i, j) maps 1:1 to one latent token position (i, j). The JEPA predictor's embedding at position i directly conditions the denoising MLP for the same position — no mapping layer, no interpolation, no ambiguity. This also makes autoregressive sampling 4× faster (256 tokens to denoise instead of 1024).

---

## Step-by-Step Implementation Plan

---

### Phase 0: Setup and Data Acquisition (Days 1–3)

#### Step 0.1: Repository Structure
```
urbanjepa/
├── data/
│   ├── ortho/              ← raw Toronto ortho tiles
│   ├── ortho_patches/      ← cropped 256x256 patches
│   ├── planet/             ← Planet API downloads
│   └── planet_patches/     ← cropped Planet patches
├── models/
│   ├── ijepa/              ← pretrained I-JEPA weights
│   ├── vae/                ← pretrained VAE weights
│   └── checkpoints/        ← your training checkpoints
├── src/
│   ├── data/
│   │   ├── ortho_dataset.py
│   │   ├── planet_dataset.py
│   │   └── augmentations.py
│   ├── models/
│   │   ├── encoder.py
│   │   ├── predictor.py
│   │   ├── denoising_mlp.py
│   │   └── urbanjepa.py
│   ├── training/
│   │   ├── train_jepa.py
│   │   ├── train_decoder.py
│   │   ├── train_djepa.py
│   │   └── losses.py
│   └── evaluation/
│       ├── metrics.py
│       └── visualize.py
├── scripts/
│   ├── download_ortho.py
│   ├── download_planet.py
│   ├── extract_patches.py
│   └── evaluate.py
├── configs/
│   ├── jepa_config.yaml
│   ├── djepa_config.yaml
│   └── eval_config.yaml
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_training_curves.ipynb
│   └── 03_results_visualization.ipynb
└── PLAN.md
```

#### Step 0.2: Download Toronto Ortho Tiles
The City of Toronto ortho is served via ArcGIS MapServer. Use the REST API to download tiles systematically.

```python
# scripts/download_ortho.py
import requests
import os
from pathlib import Path

ORTHO_URL = "https://gis.toronto.ca/arcgis/rest/services/basemap/cot_ortho/MapServer"

def get_tile_info():
    """Query MapServer for available tiles and extent."""
    response = requests.get(f"{ORTHO_URL}?f=json")
    return response.json()

def download_tile(tile_id, output_dir, zoom_level=19):
    """
    Download a single ortho tile.
    zoom_level=19 gives approximately 8cm/pixel resolution.
    """
    url = f"{ORTHO_URL}/tile/{zoom_level}/{{row}}/{{col}}"
    # ... tile-by-tile download logic
    # Save as GeoTIFF preserving georeferencing

def download_all_toronto(output_dir, workers=4):
    """
    Download all Toronto ortho tiles covering 630km².
    Use multiprocessing for speed.
    Expect this to take 6-24 hours depending on connection.
    """
    pass
```

Key considerations:
- Respect rate limits on the City of Toronto server
- Download during off-peak hours
- Verify each tile's coordinate reference system (EPSG:26917)
- Check for missing or corrupt tiles after download

#### Step 0.3: Download Planet Tiles for Test Areas
```python
# scripts/download_planet.py
import planet

# Test areas — chosen for evaluation diversity
TEST_AREAS = {
    "downtown": {"bbox": [-79.395, 43.640, -79.365, 43.660]},
    "scarborough": {"bbox": [-79.260, 43.740, -79.220, 43.775]},
    "etobicoke": {"bbox": [-79.560, 43.620, -79.520, 43.650]},
    "port_lands": {"bbox": [-79.340, 43.630, -79.310, 43.650]},
    "high_park": {"bbox": [-79.470, 43.640, -79.450, 43.660]},
}

async def download_planet_area(area_name, bbox, client):
    """
    Download PlanetScope PSScene for a test area.
    Filter for:
    - Cloud cover < 5% (clean scenes)
    - Cloud cover 20-50% (tests cloud robustness)
    - Date range: multiple seasons (summer, winter, fall)
    """
    geometry = {
        "type": "Polygon",
        "coordinates": [[
            [bbox[0], bbox[1]], [bbox[2], bbox[1]],
            [bbox[2], bbox[3]], [bbox[0], bbox[3]],
            [bbox[0], bbox[1]]
        ]]
    }
    
    # Get clear scenes
    clear_filter = planet.data_filter.and_filter([
        planet.data_filter.geometry_filter(geometry),
        planet.data_filter.range_filter("cloud_cover", lte=0.05),
        planet.data_filter.date_range_filter(
            "acquired",
            gte="2023-06-01T00:00:00Z",
            lte="2024-01-01T00:00:00Z"
        )
    ])
    
    # Also get cloudy scenes for robustness testing
    cloudy_filter = planet.data_filter.and_filter([
        planet.data_filter.geometry_filter(geometry),
        planet.data_filter.range_filter("cloud_cover", gte=0.2, lte=0.5),
    ])
    
    # Download both
```

#### Step 0.4: Download Pretrained Weights
```python
# scripts/download_pretrained.py

# I-JEPA ViT-B pretrained on ImageNet
# Download from: https://github.com/facebookresearch/ijepa
# Direct checkpoint URL (check repo for latest):
IJEPA_VITB_URL = "https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.b.16.300e.pth"

# MAR VAE (same one D-JEPA paper uses)
# From HuggingFace:
from diffusers import AutoencoderKL
vae = AutoencoderKL.from_pretrained(
    "stabilityai/sd-vae-ft-mse",
    torch_dtype=torch.float16
)
torch.save(vae.decoder.state_dict(), "models/vae/decoder_weights.pt")
```

---

### Phase 1: Data Pipeline (Days 4–6)

#### Step 1.1: Patch Extraction from Ortho
```python
# src/data/ortho_dataset.py
import rasterio
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

PATCH_SIZE = 256          # pixels cropped from 4096x4096 tiles (~15cm/pixel source)
STRIDE = 192              # 75% overlap for training richness
DOWNSAMPLE_SCALES = [18, 20, 22]  # produces 2.7m, 3.0m, 3.3m — clustered around Planet's 3m

class OrthoDataset(Dataset):
    """
    Dataset that generates (low_res, high_res) patch pairs from ortho tiles.
    Low-res patches are created by downsampling the high-res ortho.
    This is fully self-supervised — no external labels needed.
    """
    
    def __init__(self, ortho_dir, patch_size=256, stride=192,
                 downsample_scales=None, augment=True, split="train"):
        self.ortho_dir = Path(ortho_dir)
        self.patch_size = patch_size
        self.stride = stride
        self.downsample_scales = downsample_scales or DOWNSAMPLE_SCALES
        self.augment = augment
        self.split = split
        
        # Index all valid patch locations across all ortho tiles
        self.patch_index = self._build_patch_index()
        
    def _build_patch_index(self):
        """
        Scan all GeoTIFF files and record (filepath, row, col) for each
        valid 256x256 patch location. Excludes patches with >10% nodata.
        Returns list of (filepath, row_start, col_start) tuples.
        """
        index = []
        for tif_path in self.ortho_dir.glob("*.tif"):
            with rasterio.open(tif_path) as src:
                h, w = src.height, src.width
                for r in range(0, h - self.patch_size, self.stride):
                    for c in range(0, w - self.patch_size, self.stride):
                        # Quick nodata check
                        window = rasterio.windows.Window(c, r, 16, 16)
                        sample = src.read(1, window=window)
                        if sample.mean() > 5:  # not empty
                            index.append((tif_path, r, c))
        return index
    
    def _read_patch(self, filepath, row, col):
        """Read a 256x256 RGB patch from a GeoTIFF."""
        with rasterio.open(filepath) as src:
            window = rasterio.windows.Window(col, row, self.patch_size, self.patch_size)
            patch = src.read([1, 2, 3], window=window)  # RGB bands
            patch = patch.astype(np.float32) / 255.0    # normalize to [0,1]
        return patch  # shape: (3, 256, 256)
    
    def _downsample(self, patch, scale):
        """
        Downsample a patch by given scale factor using area averaging
        (matches satellite sensor physics better than bilinear).
        Then upsample back to 256x256 for uniform tensor size.
        """
        import torch.nn.functional as F
        t = torch.from_numpy(patch).unsqueeze(0)  # (1, 3, 256, 256)
        # Downsample
        small = F.avg_pool2d(t, kernel_size=scale, stride=scale)
        # Upsample back (nearest = no extra smoothing introduced)
        upsampled = F.interpolate(small, size=(256, 256), mode="bilinear",
                                  align_corners=False)
        return upsampled.squeeze(0).numpy()
    
    def _augment(self, high_res, low_res):
        """
        Apply synchronized augmentations to both high and low res patches.
        Augmentations that simulate real satellite conditions:
        - Random horizontal flip
        - Random vertical flip
        - Random 90-degree rotation
        - Color jitter (brightness, contrast, saturation ±20%)
        - Gaussian blur on low-res only (simulates atmospheric haze)
        - Random cloud mask on low-res only (simulates cloud occlusion)
        - Gaussian noise on low-res only (simulates sensor noise)
        """
        import torchvision.transforms.functional as TF
        import random
        
        # Geometric (must be identical for both)
        if random.random() > 0.5:
            high_res = np.flip(high_res, axis=2).copy()
            low_res = np.flip(low_res, axis=2).copy()
        if random.random() > 0.5:
            high_res = np.flip(high_res, axis=1).copy()
            low_res = np.flip(low_res, axis=1).copy()
        k = random.randint(0, 3)
        high_res = np.rot90(high_res, k, axes=(1, 2)).copy()
        low_res = np.rot90(low_res, k, axes=(1, 2)).copy()
        
        # Photometric on low-res only (simulate sensor differences)
        lr_tensor = torch.from_numpy(low_res)
        brightness = random.uniform(0.8, 1.2)
        contrast = random.uniform(0.8, 1.2)
        lr_tensor = TF.adjust_brightness(lr_tensor, brightness)
        lr_tensor = TF.adjust_contrast(lr_tensor, contrast)
        
        # Gaussian noise (sensor noise)
        noise = torch.randn_like(lr_tensor) * 0.02
        lr_tensor = (lr_tensor + noise).clamp(0, 1)
        
        # Random cloud mask (simulate occlusion, 20% chance)
        if random.random() > 0.8:
            lr_tensor = self._add_cloud_mask(lr_tensor)
        
        low_res = lr_tensor.numpy()
        return high_res, low_res
    
    def _add_cloud_mask(self, tensor):
        """Paint a random white blob on the tensor to simulate clouds."""
        _, h, w = tensor.shape
        cx = np.random.randint(w // 4, 3 * w // 4)
        cy = np.random.randint(h // 4, 3 * h // 4)
        radius = np.random.randint(20, 80)
        y, x = np.ogrid[:h, :w]
        mask = ((x - cx)**2 + (y - cy)**2 <= radius**2)
        tensor[:, mask] = 0.95 + torch.randn(1) * 0.05  # white-ish cloud
        return tensor
    
    def __len__(self):
        return len(self.patch_index)
    
    def __getitem__(self, idx):
        filepath, row, col = self.patch_index[idx]
        
        # Read high-res patch
        high_res = self._read_patch(filepath, row, col)
        
        # Choose a random downsampling scale
        scale = np.random.choice(self.downsample_scales)
        low_res = self._downsample(high_res, scale)
        
        # Augment
        if self.augment and self.split == "train":
            high_res, low_res = self._augment(high_res, low_res)
        
        return {
            "low_res": torch.from_numpy(low_res).float(),     # (3, 256, 256)
            "high_res": torch.from_numpy(high_res).float(),   # (3, 256, 256)
            "scale": scale,
            "filepath": str(filepath),
            "location": (row, col),
        }
```

#### Step 1.2: Planet Dataset for Evaluation
```python
# src/data/planet_dataset.py
import rasterio
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

class PlanetDataset(Dataset):
    """
    Dataset for Planet 3m/pixel imagery.
    Used ONLY for evaluation — never training.
    Pairs Planet tiles with corresponding ortho regions for ground truth.
    """
    
    def __init__(self, planet_dir, ortho_dir, patch_size=256):
        self.planet_dir = Path(planet_dir)
        self.ortho_dir = Path(ortho_dir)
        self.patch_size = patch_size
        
        # Build pairs: (planet_tile, ortho_region_same_coords)
        self.pairs = self._build_aligned_pairs()
    
    def _build_aligned_pairs(self):
        """
        For each Planet tile, find the overlapping ortho region.
        Reproject Planet WGS84 → ortho UTM Zone 17N.
        Returns list of (planet_path, ortho_path, overlap_bounds).
        """
        pairs = []
        for planet_path in self.planet_dir.glob("*.tif"):
            with rasterio.open(planet_path) as planet_src:
                # Get Planet tile bounds in WGS84
                bounds = planet_src.bounds
                # Find overlapping ortho tiles
                ortho_matches = self._find_ortho_overlap(bounds)
                for ortho_path in ortho_matches:
                    pairs.append((planet_path, ortho_path, bounds))
        return pairs
    
    def _reproject_planet_patch(self, planet_path, target_crs, target_transform,
                                 target_shape):
        """
        Reproject a Planet tile to match ortho CRS and resolution.
        Uses rasterio.warp for accurate geographic alignment.
        """
        import rasterio.warp
        
        with rasterio.open(planet_path) as src:
            data, _ = rasterio.warp.reproject(
                source=rasterio.band(src, [1, 2, 3]),
                destination=np.zeros((3, *target_shape), dtype=np.float32),
                src_crs=src.crs,
                dst_crs=target_crs,
                dst_transform=target_transform,
                resampling=rasterio.warp.Resampling.bilinear
            )
        return data.astype(np.float32) / 255.0
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        planet_path, ortho_path, bounds = self.pairs[idx]
        
        # Read and reproject Planet tile
        with rasterio.open(ortho_path) as ortho_src:
            planet_data = self._reproject_planet_patch(
                planet_path,
                target_crs=ortho_src.crs,
                target_transform=ortho_src.transform,
                target_shape=(self.patch_size, self.patch_size)
            )
            # Read corresponding ortho region as ground truth
            ortho_data = ortho_src.read([1, 2, 3]).astype(np.float32) / 255.0
        
        return {
            "planet": torch.from_numpy(planet_data).float(),   # (3, 256, 256)
            "ortho_gt": torch.from_numpy(ortho_data).float(),  # (3, 256, 256)
            "planet_path": str(planet_path),
            "has_clouds": self._detect_clouds(planet_data),
        }
    
    def _detect_clouds(self, data):
        """Simple brightness-based cloud detection."""
        return float(data.mean(0)[data.mean(0) > 0.85].sum() / data.shape[1]**2)
```

#### Step 1.3: Verify Data Pipeline
```python
# notebooks/01_data_exploration.ipynb

# Key checks to run before training:
# 1. How many patches extracted? (target: >500k for meaningful training)
# 2. Distribution of pixel values (should be roughly [0,1] uniform)
# 3. Nodata/cloud proportion (reject patches with >15% white pixels)
# 4. Geographic coverage map (ensure all neighborhoods represented)
# 5. Pair alignment sanity check (overlay planet + ortho, check lat/lon match)
# 6. Augmentation sanity check (visualize 10 augmented pairs)
# 7. Tokenization sanity check (encode a patch through VAE, decode it back)
```

---

### Phase 2: I-JEPA Fine-Tuning (Days 7–12)

#### Step 2.1: Load I-JEPA Pretrained Weights
```python
# src/models/encoder.py
import torch
import torch.nn as nn
from timm import create_model

class UrbanEncoder(nn.Module):
    """
    ViT-B/16 encoder initialized from I-JEPA pretrained weights.
    Used for both context encoder and target encoder (same architecture).
    """
    
    def __init__(self, pretrained_path, img_size=256, patch_size=16,
                 embed_dim=768, depth=12, num_heads=12):
        super().__init__()
        
        # Build ViT-B
        self.vit = create_model(
            "vit_base_patch16_224",
            pretrained=False,
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            norm_layer=nn.LayerNorm,
        )
        
        # Load I-JEPA weights
        checkpoint = torch.load(pretrained_path, map_location="cpu")
        # I-JEPA checkpoint format: {"target_encoder": state_dict}
        state_dict = checkpoint.get("target_encoder", checkpoint)
        # Remove 'module.' prefix if present (DDP training artifact)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        missing, unexpected = self.vit.load_state_dict(state_dict, strict=False)
        print(f"Loaded I-JEPA weights. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    
    def forward(self, x, mask=None):
        """
        x: (B, 3, H, W) image patch
        mask: optional boolean mask of shape (B, N) where True = masked
        Returns: (B, N, D) token embeddings
        """
        return self.vit.forward_features(x)
```

#### Step 2.2: Feature Predictor
```python
# src/models/predictor.py
import torch
import torch.nn as nn
from timm import create_model

class FeaturePredictor(nn.Module):
    """
    ViT-B predictor that takes context features and predicts
    embeddings for masked (target) token positions.
    
    Key difference from encoder: takes positional queries for masked positions
    as additional input, so it knows WHERE to predict, not just WHAT.
    """
    
    def __init__(self, pretrained_path, embed_dim=768, depth=12,
                 num_heads=12, num_tokens=256):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Predictor transformer (same architecture as encoder)
        self.transformer = create_model(
            "vit_base_patch16_224",
            pretrained=False,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
        )
        
        # Load I-JEPA pretrained weights for predictor too
        checkpoint = torch.load(pretrained_path, map_location="cpu")
        state_dict = checkpoint.get("target_encoder", checkpoint)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.transformer.load_state_dict(state_dict, strict=False)
        
        # Learnable mask token (query for each masked position)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        
        # Positional embeddings for masked token positions
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def forward(self, context_features, mask_positions):
        """
        context_features: (B, N_ctx, D) features from context encoder
        mask_positions: (B, N_mask) indices of masked token positions
        Returns: (B, N_mask, D) predicted embeddings for masked positions
        """
        B, N_mask = mask_positions.shape
        
        # Create mask token queries with positional information
        mask_tokens = self.mask_token.expand(B, N_mask, -1)
        mask_pos = self.pos_embed[:, mask_positions, :].squeeze(0)
        queries = mask_tokens + mask_pos
        
        # Concatenate context features and mask queries, run transformer
        x = torch.cat([context_features, queries], dim=1)
        x = self.transformer(x)
        
        # Return only the predicted (masked) positions
        return x[:, -N_mask:, :]
```

#### Step 2.3: Denoising MLP
```python
# src/models/denoising_mlp.py
import torch
import torch.nn as nn
import math

class SinusoidalTimestepEmbedding(nn.Module):
    """Sinusoidal time step embedding from DDPM paper."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, t):
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=device) / (half - 1)
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return embedding

class AdaLN(nn.Module):
    """Adaptive Layer Norm — conditions MLP on (timestep + embedding)."""
    def __init__(self, hidden_dim, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.scale = nn.Linear(cond_dim, hidden_dim)
        self.shift = nn.Linear(cond_dim, hidden_dim)
        nn.init.zeros_(self.scale.weight); nn.init.ones_(self.scale.bias)
        nn.init.zeros_(self.shift.weight); nn.init.zeros_(self.shift.bias)
    
    def forward(self, x, cond):
        return self.norm(x) * (1 + self.scale(cond)) + self.shift(cond)

class ResidualBlock(nn.Module):
    """Single residual block: AdaLN → Linear → SiLU → Linear + skip."""
    def __init__(self, hidden_dim, cond_dim):
        super().__init__()
        self.adaln = AdaLN(hidden_dim, cond_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.silu = nn.SiLU()
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        nn.init.zeros_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)
    
    def forward(self, x, cond):
        residual = x
        x = self.adaln(x, cond)
        x = self.linear1(x)
        x = self.silu(x)
        x = self.linear2(x)
        return x + residual

class DenoisingMLP(nn.Module):
    """
    Small MLP that predicts noise ε given:
    - noisy token xit (16-dim latent token, 4 VAE channels × 2×2 spatial group)
    - timestep t (embedded)
    - predicted embedding zi from feature predictor (768-dim)
    
    Applied independently per token — very efficient.
    """
    
    def __init__(self, token_dim=16, embed_dim=768, hidden_dim=1024,
                 time_dim=256, num_blocks=6):
        super().__init__()
        
        self.token_dim = token_dim
        cond_dim = time_dim + embed_dim
        
        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalTimestepEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
        )
        
        # Input projection: token → hidden
        self.input_proj = nn.Linear(token_dim, hidden_dim)
        
        # Residual blocks conditioned on (time + jepa embedding)
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, cond_dim) for _ in range(num_blocks)
        ])
        
        # Output projection: hidden → token (noise prediction)
        self.output_proj = nn.Linear(hidden_dim, token_dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
    
    def forward(self, x_noisy, t, z):
        """
        x_noisy: (B, N, token_dim) noisy token
        t:       (B,) integer timestep
        z:       (B, N, embed_dim) predicted JEPA embedding per token
        Returns: (B, N, token_dim) predicted noise
        """
        B, N, _ = x_noisy.shape
        
        # Time embedding: (B, time_dim) → (B, N, time_dim)
        t_emb = self.time_embed(t)
        t_emb = t_emb.unsqueeze(1).expand(B, N, -1)
        
        # Condition = concat(time_emb, jepa_embedding)
        cond = torch.cat([t_emb, z], dim=-1)  # (B, N, time_dim + embed_dim)
        
        # Flatten batch and tokens for MLP
        x = x_noisy.reshape(B * N, -1)
        cond = cond.reshape(B * N, -1)
        
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x, cond)
        x = self.output_proj(x)
        
        return x.reshape(B, N, self.token_dim)
```

#### Step 2.4: Full UrbanJEPA Model
```python
# src/models/urbanjepa.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class UrbanJEPA(nn.Module):
    """
    Full D-JEPA model for urban image super-resolution.
    
    Components:
    - context_encoder (ϕ): processes low-res input
    - target_encoder (ϕ̄): processes high-res target, EMA updated
    - feature_predictor (γ): predicts high-res embeddings from context
    - denoising_mlp (εθ): generates pixels via diffusion
    - projection_head (uθ): two-layer MLP for prediction loss
    """
    
    def __init__(self, context_encoder, target_encoder, feature_predictor,
                 denoising_mlp, vae_encoder, vae_decoder,
                 ema_decay=0.9999):
        super().__init__()
        
        self.context_encoder = context_encoder
        self.target_encoder = target_encoder
        self.feature_predictor = feature_predictor
        self.denoising_mlp = denoising_mlp
        self.vae_encoder = vae_encoder      # frozen
        self.vae_decoder = vae_decoder      # initially frozen
        self.ema_decay = ema_decay
        
        # Projection head for prediction loss (uθ)
        self.projection_head = nn.Sequential(
            nn.Linear(768, 768),
            nn.SiLU(),
            nn.Linear(768, 768),
        )
        
        # Freeze target encoder (only EMA updates)
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        
        # Freeze VAE encoder always (we never train it)
        for param in self.vae_encoder.parameters():
            param.requires_grad = False
    
    @torch.no_grad()
    def update_target_encoder(self):
        """EMA update of target encoder from context encoder."""
        α = self.ema_decay
        for ctx_param, tgt_param in zip(
            self.context_encoder.parameters(),
            self.target_encoder.parameters()
        ):
            tgt_param.data = α * tgt_param.data + (1 - α) * ctx_param.data
    
    def encode_to_latent(self, images):
        """Encode images to VAE latent space tokens."""
        with torch.no_grad():
            latents = self.vae_encoder.encode(images).latent_dist.sample()
            latents = latents * 0.18215  # SD-VAE scaling factor
        # Reshape to token sequence: (B, C, H, W) → (B, H*W, C)
        B, C, H, W = latents.shape
        tokens = latents.permute(0, 2, 3, 1).reshape(B, H * W, C)
        return tokens
    
    def decode_from_latent(self, tokens, spatial_size=32):
        """Decode latent tokens back to pixel space."""
        B, N, C = tokens.shape
        latents = tokens.reshape(B, spatial_size, spatial_size, C)
        latents = latents.permute(0, 3, 1, 2)
        latents = latents / 0.18215
        images = self.vae_decoder(latents)
        return images
    
    def sample_mask(self, N, mask_ratio_mean=1.0, mask_ratio_std=0.25,
                    mask_ratio_min=0.75):
        """
        Sample masking ratio from truncated normal distribution.
        Returns indices of masked and unmasked tokens.
        """
        ratio = torch.normal(mask_ratio_mean, mask_ratio_std, size=(1,)).item()
        ratio = max(mask_ratio_min, min(ratio, 1.0))
        n_mask = int(ratio * N)
        perm = torch.randperm(N)
        mask_idx = perm[:n_mask]
        ctx_idx = perm[n_mask:]
        return mask_idx, ctx_idx
    
    def diffusion_loss(self, tokens, predicted_embeddings, noise_schedule):
        """
        Compute diffusion (denoising) loss Ld.
        Sample t 4 times per token to maximize signal without recomputing
        predicted_embeddings.
        """
        B, N, D = tokens.shape
        total_loss = 0.0
        
        for _ in range(4):  # 4 noise samples per token
            t = torch.randint(0, noise_schedule.T, (B,), device=tokens.device)
            α_bar = noise_schedule.alpha_bar[t]  # (B,)
            
            # Corrupt tokens
            ε = torch.randn_like(tokens)
            α_bar_exp = α_bar[:, None, None]
            x_t = torch.sqrt(α_bar_exp) * tokens + torch.sqrt(1 - α_bar_exp) * ε
            
            # Predict noise
            ε_pred = self.denoising_mlp(x_t, t, predicted_embeddings)
            
            total_loss += F.mse_loss(ε_pred, ε)
        
        return total_loss / 4
    
    def prediction_loss(self, predicted_embeddings, target_embeddings):
        """
        Compute prediction loss Lp (smooth L1 between projected embeddings).
        """
        projected = self.projection_head(predicted_embeddings)
        return F.smooth_l1_loss(projected, target_embeddings.detach())
    
    def forward(self, low_res, high_res, noise_schedule):
        """
        Full forward pass for training.
        
        low_res:  (B, 3, 256, 256) downsampled input
        high_res: (B, 3, 256, 256) ortho ground truth
        Returns: dict with losses and intermediate outputs
        """
        # Encode both to latent space
        low_tokens = self.encode_to_latent(low_res)    # (B, N, 16)
        high_tokens = self.encode_to_latent(high_res)  # (B, N, 16)
        
        B, N, _ = low_tokens.shape
        
        # Sample mask
        mask_idx, ctx_idx = self.sample_mask(N)
        
        # Context: context encoder sees low-res tokens (all of them, no masking)
        ctx_features = self.context_encoder(low_res)  # (B, N_ctx, 768)
        
        # Target: target encoder sees high-res tokens (all, stop gradient)
        with torch.no_grad():
            target_embeddings = self.target_encoder(high_res)  # (B, N, 768)
            target_masked = target_embeddings[:, mask_idx, :]  # (B, N_mask, 768)
        
        # Predict: feature predictor predicts high-res embeddings for masked positions
        predicted_embeddings = self.feature_predictor(
            ctx_features, mask_idx
        )  # (B, N_mask, 768)
        
        # Compute losses
        Lp = self.prediction_loss(predicted_embeddings, target_masked)
        
        masked_tokens = high_tokens[:, mask_idx, :]  # (B, N_mask, 16)
        Ld = self.diffusion_loss(masked_tokens, predicted_embeddings, noise_schedule)
        
        loss = Ld + Lp
        
        return {
            "loss": loss,
            "loss_diffusion": Ld.item(),
            "loss_prediction": Lp.item(),
        }
```

#### Step 2.5: I-JEPA Domain Fine-Tuning Training Loop
```python
# src/training/train_jepa.py
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb

def train_jepa_phase(model, dataloader, config):
    """
    Phase 2: Fine-tune I-JEPA on satellite domain.
    Only prediction loss (Lp) — no diffusion yet.
    Denoising MLP not used in this phase.
    
    Config:
        lr: 8e-4
        weight_decay: 0.05
        epochs: 50
        warmup_epochs: 5
        gradient_clip: 1.0
    """
    
    # Only train context encoder and feature predictor
    # Target encoder is EMA-only (no grad)
    # Denoising MLP not trained yet
    trainable_params = (
        list(model.context_encoder.parameters()) +
        list(model.feature_predictor.parameters()) +
        list(model.projection_head.parameters())
    )
    
    optimizer = AdamW(trainable_params, lr=config.lr,
                      weight_decay=config.weight_decay,
                      betas=(0.9, 0.95))
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)
    
    wandb.init(project="urbanjepa", name="phase2-jepa-finetune")
    
    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, batch in enumerate(dataloader):
            low_res = batch["low_res"].cuda()
            high_res = batch["high_res"].cuda()
            
            # Forward (prediction loss only, no diffusion)
            ctx_features = model.context_encoder(low_res)
            
            with torch.no_grad():
                target_embeddings = model.target_encoder(high_res)
            
            B, N, _ = ctx_features.shape
            mask_idx, _ = model.sample_mask(N)
            
            predicted_embeddings = model.feature_predictor(ctx_features, mask_idx)
            target_masked = target_embeddings[:, mask_idx, :]
            
            loss = model.prediction_loss(predicted_embeddings, target_masked)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, config.gradient_clip)
            optimizer.step()
            
            # EMA update of target encoder
            model.update_target_encoder()
            
            epoch_loss += loss.item()
            
            if batch_idx % 100 == 0:
                wandb.log({
                    "train/prediction_loss": loss.item(),
                    "train/epoch": epoch,
                    "train/step": epoch * len(dataloader) + batch_idx,
                })
        
        scheduler.step()
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save({
                "epoch": epoch,
                "context_encoder": model.context_encoder.state_dict(),
                "target_encoder": model.target_encoder.state_dict(),
                "feature_predictor": model.feature_predictor.state_dict(),
                "projection_head": model.projection_head.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, f"models/checkpoints/jepa_epoch_{epoch}.pt")
        
        print(f"Epoch {epoch}: avg loss = {epoch_loss / len(dataloader):.4f}")
    
    wandb.finish()
    print("Phase 2 complete. JEPA encoders are domain-adapted.")
```

---

### Phase 3: CNN Decoder Validation (Days 13–14)

#### Step 3.1: Simple CNN Decoder
```python
# src/models/cnn_decoder.py
import torch
import torch.nn as nn

class SimpleCNNDecoder(nn.Module):
    """
    Lightweight CNN decoder to validate JEPA embeddings are meaningful.
    Takes predicted embeddings → upsamples → pixel output.
    NOT the final decoder — just a sanity check before committing to diffusion.
    """
    
    def __init__(self, embed_dim=768, spatial_size=16, out_size=256):
        super().__init__()
        
        self.spatial_size = spatial_size  # 16x16 token grid
        
        # Project embeddings to spatial feature map
        self.proj = nn.Linear(embed_dim, 256)
        
        # Upsample from 16x16 to 256x256 via transposed convolutions
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 16→32
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),   # 32→64
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),    # 64→128
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),    # 128→256
            nn.ReLU(),
            nn.Conv2d(16, 3, 3, padding=1),                        # 256→256 RGB
            nn.Sigmoid(),
        )
    
    def forward(self, embeddings):
        """
        embeddings: (B, N, embed_dim) from feature predictor
        Returns: (B, 3, 256, 256) RGB image
        """
        B, N, _ = embeddings.shape
        
        x = self.proj(embeddings)  # (B, N, 256)
        x = x.reshape(B, self.spatial_size, self.spatial_size, 256)
        x = x.permute(0, 3, 1, 2)  # (B, 256, 16, 16)
        
        return self.decoder(x)

def train_cnn_decoder(model, cnn_decoder, dataloader, epochs=20):
    """
    Train only the CNN decoder with frozen JEPA.
    Quick validation step — should converge in 1-2 days.
    If PSNR < 25dB after 20 epochs, JEPA embeddings are not meaningful enough.
    Go back and train JEPA longer.
    """
    optimizer = torch.optim.Adam(cnn_decoder.parameters(), lr=1e-3)
    
    for epoch in range(epochs):
        for batch in dataloader:
            low_res = batch["low_res"].cuda()
            high_res = batch["high_res"].cuda()
            
            with torch.no_grad():
                ctx_features = model.context_encoder(low_res)
                B, N, _ = ctx_features.shape
                mask_idx = torch.arange(N)  # predict all positions
                predicted_embeddings = model.feature_predictor(ctx_features, mask_idx)
            
            output = cnn_decoder(predicted_embeddings)
            loss = nn.functional.mse_loss(output, high_res)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Compute PSNR
        with torch.no_grad():
            psnr = -10 * torch.log10(loss).item()
        print(f"Epoch {epoch}: MSE={loss.item():.4f}, PSNR≈{psnr:.1f}dB")
        
        # Gate: if PSNR > 26dB, JEPA is working, proceed to D-JEPA
        if psnr > 26:
            print("JEPA validation passed. Proceed to Stage 3.")
            break
```

---

### Phase 4: Full D-JEPA Training (Days 15–20)

#### Step 4.1: Noise Schedule
```python
# src/training/losses.py
import torch
import numpy as np

class LinearNoiseSchedule:
    """
    Linear variance schedule from ADM paper (Dhariwal & Nichol 2021).
    Exactly as used in D-JEPA paper.
    β range: [1e-4, 2e-2] over T=1000 steps.
    """
    
    def __init__(self, T=1000, beta_start=1e-4, beta_end=2e-2, device="cuda"):
        self.T = T
        betas = torch.linspace(beta_start, beta_end, T, device=device)
        alphas = 1.0 - betas
        self.alpha_bar = torch.cumprod(alphas, dim=0)  # ᾱ_t
        self.alpha_bar_prev = torch.cat([torch.ones(1, device=device),
                                          self.alpha_bar[:-1]])
        self.betas = betas
        self.alphas = alphas
        
        # Pre-compute values needed for sampling
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar)
        self.posterior_variance = (
            betas * (1 - self.alpha_bar_prev) / (1 - self.alpha_bar)
        )
    
    def q_sample(self, x0, t):
        """Forward process: add noise to x0 at timestep t."""
        sqrt_ab = self.sqrt_alpha_bar[t][:, None, None]
        sqrt_one_minus = self.sqrt_one_minus_alpha_bar[t][:, None, None]
        ε = torch.randn_like(x0)
        return sqrt_ab * x0 + sqrt_one_minus * ε, ε
    
    @torch.no_grad()
    def p_sample(self, model_output, x_t, t, z, temperature=0.98):
        """
        Reverse process: one denoising step.
        temperature controls sample diversity (τ in the paper).
        """
        t_idx = t[0].item()
        β = self.betas[t_idx]
        α = self.alphas[t_idx]
        α_bar = self.alpha_bar[t_idx]
        
        # Predicted x0
        x0_pred = (x_t - torch.sqrt(1 - α_bar) * model_output) / torch.sqrt(α_bar)
        x0_pred = x0_pred.clamp(-1, 1)
        
        # Compute mean of p(x_{t-1} | x_t)
        mean = (torch.sqrt(self.alpha_bar_prev[t_idx]) * β / (1 - α_bar) * x0_pred +
                torch.sqrt(α) * (1 - self.alpha_bar_prev[t_idx]) / (1 - α_bar) * x_t)
        
        if t_idx == 0:
            return mean
        
        # Add noise scaled by temperature
        noise = torch.randn_like(x_t)
        variance = torch.sqrt(self.posterior_variance[t_idx]) * temperature
        return mean + variance * noise
```

#### Step 4.2: D-JEPA Training Loop
```python
# src/training/train_djepa.py
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb

def train_djepa_phase(model, dataloader, noise_schedule, config):
    """
    Phase 4: Full D-JEPA training.
    
    Sub-phase A (days 15-17): Denoising MLP only, JEPA frozen
    Sub-phase B (days 18-20): Joint fine-tune everything, low LR
    
    Config:
        lr_mlp: 1e-3         (MLP learning rate)
        lr_jepa: 1e-4        (JEPA fine-tune learning rate)
        weight_decay: 0.05
        epochs_frozen: 50    (sub-phase A)
        epochs_joint: 20     (sub-phase B)
        gradient_clip: 1.0
        batch_size: 8        (tight VRAM budget)
        grad_accumulation: 4 (effective batch = 32)
    """
    
    # === Sub-phase A: Train MLP only ===
    print("Sub-phase A: Training denoising MLP with frozen JEPA...")
    
    # Freeze all JEPA components
    for param in model.context_encoder.parameters():
        param.requires_grad = False
    for param in model.feature_predictor.parameters():
        param.requires_grad = False
    
    optimizer_mlp = AdamW(
        list(model.denoising_mlp.parameters()),
        lr=config.lr_mlp,
        weight_decay=config.weight_decay
    )
    
    wandb.init(project="urbanjepa", name="phase4a-mlp-training")
    
    accum_steps = config.grad_accumulation
    
    for epoch in range(config.epochs_frozen):
        model.train()
        
        for batch_idx, batch in enumerate(dataloader):
            low_res = batch["low_res"].cuda()
            high_res = batch["high_res"].cuda()
            
            # JEPA forward (no grad needed — frozen)
            with torch.no_grad():
                low_tokens = model.encode_to_latent(low_res)
                high_tokens = model.encode_to_latent(high_res)
                ctx_features = model.context_encoder(low_res)
                target_embeddings = model.target_encoder(high_res)
                
                B, N, _ = ctx_features.shape
                mask_idx, _ = model.sample_mask(N)
                predicted_embeddings = model.feature_predictor(ctx_features, mask_idx)
            
            # Diffusion loss (only MLP has grad)
            masked_tokens = high_tokens[:, mask_idx, :]
            Ld = model.diffusion_loss(masked_tokens, predicted_embeddings,
                                       noise_schedule)
            
            loss = Ld / accum_steps
            loss.backward()
            
            if (batch_idx + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.denoising_mlp.parameters(), config.gradient_clip
                )
                optimizer_mlp.step()
                optimizer_mlp.zero_grad()
            
            if batch_idx % 100 == 0:
                wandb.log({"train/diffusion_loss": Ld.item(), "epoch": epoch})
        
        if epoch % 10 == 0:
            torch.save({
                "epoch": epoch,
                "denoising_mlp": model.denoising_mlp.state_dict(),
                "optimizer_mlp": optimizer_mlp.state_dict(),
            }, f"models/checkpoints/mlp_epoch_{epoch}.pt")
    
    wandb.finish()
    
    # === Sub-phase B: Joint fine-tune everything ===
    print("Sub-phase B: Joint fine-tuning all components...")
    
    # Unfreeze JEPA
    for param in model.context_encoder.parameters():
        param.requires_grad = True
    for param in model.feature_predictor.parameters():
        param.requires_grad = True
    
    optimizer_joint = AdamW([
        {"params": model.denoising_mlp.parameters(), "lr": config.lr_mlp},
        {"params": model.context_encoder.parameters(), "lr": config.lr_jepa},
        {"params": model.feature_predictor.parameters(), "lr": config.lr_jepa},
        {"params": model.projection_head.parameters(), "lr": config.lr_jepa},
    ], weight_decay=config.weight_decay)
    
    wandb.init(project="urbanjepa", name="phase4b-joint-finetune")
    
    for epoch in range(config.epochs_joint):
        model.train()
        
        for batch_idx, batch in enumerate(dataloader):
            low_res = batch["low_res"].cuda()
            high_res = batch["high_res"].cuda()
            
            result = model(low_res, high_res, noise_schedule)
            loss = result["loss"] / accum_steps
            loss.backward()
            
            if (batch_idx + 1) % accum_steps == 0:
                all_params = (
                    list(model.denoising_mlp.parameters()) +
                    list(model.context_encoder.parameters()) +
                    list(model.feature_predictor.parameters())
                )
                torch.nn.utils.clip_grad_norm_(all_params, config.gradient_clip)
                optimizer_joint.step()
                optimizer_joint.zero_grad()
                model.update_target_encoder()
            
            wandb.log({
                "train/total_loss": result["loss"],
                "train/diffusion_loss": result["loss_diffusion"],
                "train/prediction_loss": result["loss_prediction"],
            })
        
        torch.save({
            "epoch": epoch,
            "full_model": model.state_dict(),
            "optimizer": optimizer_joint.state_dict(),
        }, f"models/checkpoints/djepa_joint_epoch_{epoch}.pt")
    
    wandb.finish()
    print("Phase 4 complete. Full D-JEPA trained.")
```

---

### Phase 5: Autoregressive Sampling (Days 21–22)

#### Step 5.1: Generalized Next-Set-of-Tokens Sampling
```python
# src/models/urbanjepa.py (sampling method)

@torch.no_grad()
def sample(self, low_res_input, noise_schedule,
           ar_steps=64, temperature=0.98, cfg_scale=None):
    """
    Generate high-res image from low-res input using
    generalized next-set-of-tokens prediction (Algorithm 1 from D-JEPA paper).
    
    ar_steps: number of autoregressive steps (64 is optimal for ViT-B)
    temperature: controls sample diversity (0.98 recommended)
    cfg_scale: classifier-free guidance scale (None = no CFG)
    
    Returns: (3, 256, 256) generated high-res image
    """
    self.eval()
    device = low_res_input.device
    
    low_res = low_res_input.unsqueeze(0)  # (1, 3, 256, 256)
    
    # Encode context
    ctx_features = self.context_encoder(low_res)
    B, N, _ = ctx_features.shape
    
    # Cosine schedule for how many tokens to reveal each step
    tokens_per_step = self._cosine_schedule(ar_steps, N)
    
    # Start with no tokens sampled
    sampled_tokens = torch.zeros(1, N, 16, device=device)
    sampled_mask = torch.zeros(N, dtype=torch.bool, device=device)  # True = sampled
    
    for step, n_new in enumerate(tokens_per_step):
        # Get unsampled positions
        unsampled_idx = (~sampled_mask).nonzero().squeeze(1)
        
        if len(unsampled_idx) == 0:
            break
        
        # Predict embeddings for all unsampled positions
        predicted_embeddings = self.feature_predictor(ctx_features, unsampled_idx)
        
        # Randomly select n_new positions to sample this step
        perm = torch.randperm(len(unsampled_idx))[:n_new]
        selected_idx = unsampled_idx[perm]
        selected_embeddings = predicted_embeddings[:, perm, :]
        
        # Denoise selected tokens (DDPM sampling, 100 steps)
        x = torch.randn(1, n_new, 16, device=device)
        
        for t in reversed(range(noise_schedule.T)):
            t_batch = torch.full((1,), t, device=device, dtype=torch.long)
            noise_pred = self.denoising_mlp(x, t_batch, selected_embeddings)
            x = noise_schedule.p_sample(noise_pred, x, t_batch,
                                         selected_embeddings, temperature)
        
        # Add to sampled tokens
        sampled_tokens[:, selected_idx, :] = x
        sampled_mask[selected_idx] = True
    
    # Decode to pixels
    image = self.decode_from_latent(sampled_tokens)
    return image.squeeze(0).clamp(0, 1)

def _cosine_schedule(self, T, N):
    """
    Cosine schedule for number of tokens to reveal at each step.
    Returns list of length T where sum = N.
    """
    import numpy as np
    schedule = []
    revealed = 0
    for t in range(T):
        target = int(N * (1 - np.cos(np.pi * (t + 1) / T)) / 2)
        n = target - revealed
        schedule.append(max(1, n))
        revealed = target
    return schedule
```

---

### Phase 6: Evaluation (Days 23–25)

#### Step 6.1: Metrics
```python
# src/evaluation/metrics.py
import torch
import torch.nn.functional as F
import numpy as np
from torchvision.models import inception_v3
from scipy import linalg

def compute_psnr(pred, target):
    """
    Peak Signal-to-Noise Ratio.
    pred, target: (B, 3, H, W) tensors in [0, 1]
    Higher is better. >30dB = good. >35dB = excellent.
    """
    mse = F.mse_loss(pred, target)
    return -10 * torch.log10(mse + 1e-8)

def compute_ssim(pred, target, window_size=11):
    """
    Structural Similarity Index.
    pred, target: (B, 3, H, W) tensors in [0, 1]
    Range [0, 1]. >0.85 = good. >0.92 = excellent.
    """
    # Standard SSIM implementation
    C1, C2 = 0.01**2, 0.03**2
    mu1 = F.avg_pool2d(pred, window_size, 1, window_size//2)
    mu2 = F.avg_pool2d(target, window_size, 1, window_size//2)
    sigma1_sq = F.avg_pool2d(pred**2, window_size, 1, window_size//2) - mu1**2
    sigma2_sq = F.avg_pool2d(target**2, window_size, 1, window_size//2) - mu2**2
    sigma12 = F.avg_pool2d(pred*target, window_size, 1, window_size//2) - mu1*mu2
    ssim = ((2*mu1*mu2 + C1) * (2*sigma12 + C2)) / \
           ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim.mean()

def compute_fid(real_features, fake_features):
    """
    Frechet Inception Distance.
    Lower is better. <10 = good for satellite imagery.
    """
    mu1, sigma1 = real_features.mean(0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(0), np.cov(fake_features, rowvar=False)
    diff = mu1 - mu2
    covmean = linalg.sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))

def compute_edge_sharpness(pred):
    """
    Sobel edge sharpness metric — important for satellite imagery.
    Measures crispness of building edges, roads, etc.
    Higher = sharper edges.
    """
    sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]],
                            dtype=torch.float32).view(1,1,3,3)
    sobel_y = sobel_x.transpose(-1,-2)
    gray = pred.mean(dim=1, keepdim=True)
    Gx = F.conv2d(gray, sobel_x.to(pred.device), padding=1)
    Gy = F.conv2d(gray, sobel_y.to(pred.device), padding=1)
    return torch.sqrt(Gx**2 + Gy**2).mean().item()

def evaluate_model(model, test_loader, noise_schedule, baselines, device):
    """
    Full evaluation against all baselines.
    Saves results to evaluation/results.json
    """
    results = {
        "d_jepa": {"psnr": [], "ssim": [], "edge": []},
        "bicubic": {"psnr": [], "ssim": [], "edge": []},
        "esrgan": {"psnr": [], "ssim": [], "edge": []},
    }
    
    model.eval()
    
    for batch in test_loader:
        low_res = batch["low_res"].to(device)
        high_res = batch["high_res"].to(device)
        has_clouds = batch["has_clouds"]
        
        # D-JEPA generation
        with torch.no_grad():
            generated = model.sample(low_res[0], noise_schedule,
                                      ar_steps=64, temperature=0.98)
        
        # Bicubic baseline
        bicubic = F.interpolate(low_res, size=(256, 256), mode="bicubic")
        
        # Compute metrics for each
        for name, pred in [("d_jepa", generated.unsqueeze(0)),
                            ("bicubic", bicubic)]:
            results[name]["psnr"].append(compute_psnr(pred, high_res).item())
            results[name]["ssim"].append(compute_ssim(pred, high_res).item())
            results[name]["edge"].append(compute_edge_sharpness(pred))
    
    # Aggregate
    for name in results:
        for metric in results[name]:
            vals = results[name][metric]
            results[name][metric] = {
                "mean": np.mean(vals),
                "std": np.std(vals),
            }
    
    return results
```

#### Step 6.2: Baselines to Compare Against
```
Baseline 1: Bicubic interpolation
- torch.nn.functional.interpolate(low_res, size=(256,256), mode='bicubic')
- Simplest possible upscaler
- Expected PSNR: ~24-26dB

Baseline 2: ESRGAN
- pip install basicsr
- Use pretrained ESRGAN weights from BasicSR library
- Expected PSNR: ~28-30dB for natural images

Baseline 3: Simple CNN decoder (your Phase 3 model)
- Tests how much the diffusion component adds over simple decoding
- Expected PSNR: ~26-28dB
```

#### Step 6.3: Planet Evaluation (The Interesting Part)
```python
def evaluate_on_planet(model, planet_loader, noise_schedule):
    """
    Real-world evaluation on Planet 3m imagery.
    Compares generated output against Toronto ortho ground truth.
    
    Key things to measure:
    1. Overall PSNR/SSIM (lower expected than ortho pairs)
    2. Performance vs cloud cover % (does model degrade gracefully?)
    3. Performance by urban type (downtown vs suburban vs industrial)
    4. Uncertainty maps (variance across multiple samples)
    5. Feature alignment (do generated buildings align with ortho buildings?)
    """
    
    results_by_cloud = {"clear": [], "partial": [], "cloudy": []}
    results_by_type = {"downtown": [], "suburban": [], "industrial": [], "park": []}
    
    for batch in planet_loader:
        planet = batch["planet"].cuda()
        ortho_gt = batch["ortho_gt"].cuda()
        cloud_pct = batch["has_clouds"]
        area_type = batch["area_type"]
        
        # Generate multiple samples for uncertainty estimation
        samples = []
        for _ in range(5):
            with torch.no_grad():
                sample = model.sample(planet[0], noise_schedule,
                                       ar_steps=64, temperature=0.98)
            samples.append(sample)
        
        samples_tensor = torch.stack(samples)  # (5, 3, H, W)
        mean_sample = samples_tensor.mean(0)
        uncertainty_map = samples_tensor.std(0)  # high std = uncertain
        
        psnr = compute_psnr(mean_sample.unsqueeze(0), ortho_gt)
        ssim = compute_ssim(mean_sample.unsqueeze(0), ortho_gt)
        
        # Bin by cloud coverage
        if cloud_pct < 0.05:
            results_by_cloud["clear"].append((psnr, ssim))
        elif cloud_pct < 0.3:
            results_by_cloud["partial"].append((psnr, ssim))
        else:
            results_by_cloud["cloudy"].append((psnr, ssim))
        
        results_by_type[area_type].append((psnr, ssim))
    
    return results_by_cloud, results_by_type
```

---

### Phase 7: VAE Decoder Fine-Tuning (Optional, Days 26–28)

If Planet evaluation shows texture artifacts or domain mismatch:

```python
def finetune_vae_decoder(model, ortho_dataloader, epochs=10):
    """
    Fine-tune the pretrained VAE decoder on ortho patches.
    Teaches it satellite-specific textures (rooftops, roads, vegetation).
    
    Only unfreeze the decoder — encoder stays frozen.
    Short training — ortho textures are not that different from natural images.
    
    Loss: pixel reconstruction + perceptual loss (VGG features)
    """
    
    # Unfreeze decoder
    for param in model.vae_decoder.parameters():
        param.requires_grad = True
    
    optimizer = torch.optim.Adam(model.vae_decoder.parameters(), lr=1e-5)
    
    perceptual_loss = PerceptualLoss().cuda()  # VGG-based
    
    for epoch in range(epochs):
        for batch in ortho_dataloader:
            high_res = batch["high_res"].cuda()
            
            # Encode and immediately decode (autoencoder reconstruction)
            with torch.no_grad():
                latents = model.vae_encoder.encode(high_res).latent_dist.sample()
                latents = latents * 0.18215
            
            reconstructed = model.vae_decoder(latents / 0.18215)
            
            # Pixel + perceptual loss
            pixel_loss = F.mse_loss(reconstructed, high_res)
            percep_loss = perceptual_loss(reconstructed, high_res)
            loss = pixel_loss + 0.1 * percep_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"VAE fine-tune epoch {epoch}: loss={loss.item():.4f}")
```

---

## Training Configuration Reference

```yaml
# configs/djepa_config.yaml

model:
  vit_size: base           # ViT-B/16
  embed_dim: 768
  depth: 12
  num_heads: 12
  patch_size: 16
  img_size: 256
  token_dim: 16            # VAE latent token dim (4 channels × 2×2 spatial group)
  mlp_hidden: 1024
  mlp_blocks: 6

masking:
  ratio_mean: 1.0
  ratio_std: 0.25
  ratio_min: 0.75

noise_schedule:
  T: 1000
  beta_start: 0.0001
  beta_end: 0.02

training:
  batch_size: 8            # fits 3090 with gradient accumulation
  grad_accumulation: 4     # effective batch = 32
  lr_jepa: 0.0008
  lr_mlp: 0.001
  lr_joint: 0.0001
  weight_decay: 0.05
  gradient_clip: 1.0
  ema_decay: 0.9999
  
  phase2_epochs: 50        # JEPA fine-tune
  phase3_epochs: 20        # CNN decoder validation
  phase4a_epochs: 50       # MLP training (frozen JEPA)
  phase4b_epochs: 20       # Joint fine-tune

sampling:
  ar_steps: 64             # autoregressive steps
  ddpm_steps: 100          # diffusion denoising steps
  temperature: 0.98
  cfg_scale: 3.0           # classifier-free guidance (optional)

data:
  tile_size: 4096          # L20 tiles at ~15cm/px
  patch_size: 256           # crops from tiles
  downsample_scales: [18, 20, 22]  # 2.7m, 3.0m, 3.3m (clustered on PlanetScope 3m)
  num_workers: 4
  pin_memory: true
  augment: true
```

---

## VRAM Budget (RTX 3090, 24GB)

```
Context encoder (ViT-B, fp16):     ~1.7GB
Target encoder (ViT-B, fp16):      ~1.7GB  (inference only)
Feature predictor (ViT-B, fp16):   ~1.7GB
Denoising MLP (fp16):              ~0.1GB
VAE encoder+decoder (fp16):        ~1.5GB
Batch (8 × 2 × 256×256 × 3):      ~1.5GB
Activations + gradients:           ~10GB
Optimizer states (AdamW):          ~3GB
Total estimated:                   ~21GB
Headroom remaining:                ~3GB ✓
```

Mixed precision (torch.autocast) is mandatory. Gradient checkpointing on ViT if VRAM is tight.

---

## Success Gates

```
After Phase 2 (JEPA fine-tune):
  □ Cosine similarity between predicted and target embeddings > 0.7
  □ Training loss decreasing steadily

After Phase 3 (CNN decoder validation):
  □ PSNR on held-out ortho patches > 26dB  ← if not, train JEPA longer
  □ Output images look like urban scenes (not noise)

After Phase 4a (MLP training):
  □ Generated samples are recognizably urban
  □ Diffusion loss < 0.1

After Phase 4b (Joint fine-tune):
  □ PSNR on ortho pairs > 30dB
  □ SSIM > 0.85
  □ FID < 20 (vs ortho patches)
  □ Beats bicubic on all metrics
  □ Beats CNN decoder on FID and edge sharpness

After Phase 6 (Planet evaluation):
  □ PSNR on clear Planet scenes > 25dB
  □ Building footprints align with ortho ground truth
  □ Model degrades gracefully with cloud cover
  □ Uncertainty maps show high variance at cloud locations
```

---

## Timeline Summary

| Day(s) | Phase | Output |
|--------|-------|--------|
| 1–3 | Setup + Data download | Ortho tiles + Planet tiles on disk |
| 4–6 | Data pipeline | ~1.5M training patches indexed |
| 7–12 | I-JEPA fine-tune | Domain-adapted JEPA checkpoint |
| 13–14 | CNN decoder validation | Sanity check: embeddings → images |
| 15–17 | Denoising MLP training | Working D-JEPA generator |
| 18–20 | Joint fine-tune | Polished full model |
| 21–22 | Sampling + inference | Generation pipeline working |
| 23–25 | Evaluation | Full metrics vs all baselines |
| 26–28 | VAE fine-tune (optional) | Better satellite textures |

**Total: ~4 weeks** on a single RTX 3090.

---

## What You Can Build On Top of This

Once the foundation model is trained:

1. **Change detection**: Feed Planet T1 and T2 through model, diff the outputs
2. **Cloud removal**: Multiple cloudy Planet inputs → single clear output
3. **Uncertainty maps**: Run 5 samples, measure variance → where is the model unsure?
4. **Other cities**: Fine-tune for 1 week on Amsterdam/Melbourne/NYC ortho
5. **Higher resolution**: Upgrade to ViT-L if results plateau, same pipeline
6. **Multi-temporal fusion**: Stack multiple Planet dates as context channels
7. **Semantic segmentation**: Use context encoder features as backbone for downstream tasks

---

## References

- D-JEPA: Chen et al., ICLR 2025
- I-JEPA: Assran et al., CVPR 2023
- MAR (VAE source): Li et al., 2024
- ADM (noise schedule): Dhariwal & Nichol, NeurIPS 2021
- SD-VAE: Rombach et al., CVPR 2022
- Improved DDPM: Nichol & Dhariwal, ICML 2021
- Toronto Ortho: City of Toronto Open Data Portal
- Planet API: Planet Labs PlanetScope documentation

---

## Next Steps If Ld Plateau Persists (Research-Backed Interventions)

**Current state (epoch 4, Phase 4b):** Ld stuck at 0.165-0.170 — same as frozen-JEPA Phase 4a. PSNR ~11 dB. If this holds past epoch 8-10, apply interventions below.

### Tier 1: Low effort, high impact (try first — ~30 lines of code)

**1. Cosine noise schedule** *(Improved DDPM, Nichol & Dhariwal 2021)*
Replace linear betas with cosine schedule. Concentrates noise at middle timesteps where learning is richest. Paper showed 30-40% FID improvement over linear.
```python
# In LinearNoiseSchedule.__init__, replace:
#   betas = torch.linspace(beta_start, beta_end, T)
# With cosine schedule (see Improved DDPM paper, Eq 17)
```
File: `src/training/losses.py`, ~5 lines.

**2. Importance-weighted timestep sampling** *(Improved DDPM 2021)*
Instead of `t ~ Uniform(0, T)`, sample proportional to noise level to focus on informative timesteps.
File: `src/models/urbanjepa.py` `diffusion_loss()`, ~3 lines.

**3. Lower LR, more patience** 
Phase 2 converged in 2 epochs — ortho imagery is redundant. MLP may need finer steps: `--lr_mlp 2e-4 --lr_jepa 2e-5 --epochs 50`.
Command-line only, no code changes.

### Tier 2: Medium effort, good impact

**4. SD-VAE decoder fine-tuning** *(Phase 7 in plan)*
The VAE was trained on LAION natural images. Urban textures may not survive 48× compression. Fine-tune decoder on ortho patches with perceptual loss. Keeps latent space intact, teaches satellite-domain textures.
File: `src/training/train_decoder.py` — extend to support VAE fine-tuning.

**5. Cross-token self-attention in denoiser** *(D-JEPA uses 12 transformer layers, we use 6-block per-token MLP)*
Add 2-4 lightweight transformer layers to denoiser for spatial noise correlation capture. Current per-token MLP has ZERO cross-token communication — noise in VAE latent space is spatially correlated.
File: `src/models/denoising_mlp.py` — add TransformerEncoder after residual blocks.

**6. More noise samples per token**
Increase from 4 to 8 or 16 noise samples per token. More training signal per forward pass. Trade compute for signal.
File: `src/models/urbanjepa.py` `diffusion_loss()`, change loop count.

### Tier 3: Major architecture changes

**7. Replace MLP with transformer denoiser** *(MAR / D-JEPA papers)*
Full transformer operating on token sequence. 4-6 layers, 8 heads, ~20M params. MAR paper shows transformers significantly outperform MLPs for latent diffusion.
File: `src/models/denoising_mlp.py` — major rewrite.

**8. Increase VAE latent capacity**
Switch VAE or train custom one with 8-16 latent channels. More capacity = more detail survives encoding.
File: `src/models/urbanjepa.py` `load_vae()`.

### Decision Tree

```
After epoch 10 of Phase 4b:
├─ Ld < 0.12, PSNR > 15 dB → Continue, on track ✓
├─ Ld 0.12-0.16, PSNR 12-15 dB → Apply Tier 1, restart 4b
└─ Ld > 0.16, PSNR < 12 dB → Apply Tier 1 + Tier 2 (#5), restart 4b
```

---

## Known Limitations & Risks

### 1. No True I-JEPA Pretraining (CRITICAL)
Meta FAIR only released I-JEPA checkpoints for ViT-H (632M params) and ViT-G (1.1B params). **ViT-B was never released.** We initialize from timm's supervised ImageNet-1K ViT-B/16 instead. Phase 2 (50 epochs JEPA fine-tuning) must therefore teach BOTH the JEPA predictive objective AND adapt to the ortho domain from scratch. This is a harder task than the original plan assumed (which expected to domain-adapt existing I-JEPA features).

**Mitigation**: 50 epochs of prediction-only training before introducing diffusion. The supervised ImageNet features are still a reasonable starting point (edges, shapes, textures).

### 2. VAE Latent Channel Bottleneck
The SD-VAE compresses 256×256×3 (196,608 values) into 32×32×4 (4,096 values) — a 48× compression ratio. Fine structural detail may be lost. Mitigated by 2×2 spatial grouping: 32×32×4 latents → 16×16×16 tokens, giving the denoising MLP 16-dim per token (4× the original 4-dim). The VAE was trained on natural images (not aerial/satellite), so domain-specific textures may not survive encoding. Phase 7 VAE fine-tuning may help.

**Mitigation**: Phase 7 VAE decoder fine-tuning on ortho patches can partially recover domain textures. Alternative: use a higher-channel VAE if results are inadequate.

### 3. Resolution: L20 (~15cm/pixel) — Good Compromise
We are using L20 (~15cm/pixel) from the H3MRL pre-downloaded tiles. This is a significant upgrade from the original zoom-19 (~30cm/pixel) plan. At 15cm/pixel, cars, building details, and road markings are clearly visible. Downsampled low-res inputs at scale 3-20× range from 45cm to 3.0m — scale 20 directly matches PlanetScope's 3m/pixel (2.99m, ~1cm error). L21 (~7.5cm/pixel) would be ideal but requires 4× more tiles and storage.

### 4. VRAM Constraint (RTX 3090 24GB)
- Total VRAM usage: ~21GB with fp16 + gradient checkpointing
- Batch size: 8 (gradient accumulation ×4 = effective 32)
- No room for larger models (ViT-L would need ~48GB)
- Cannot increase batch size without offloading

### 5. Tile Hit Rate Uncertainty
The 477K tile grid includes significant water (Lake Ontario) and areas outside Toronto. The server returns 404 for these. Actual land tiles are estimated at 80K-130K based on early download rates (~95% hit rate in early rows suggests good coverage). Final count TBD when download completes.

### 6. Planet API Not Configured
Planet API credentials and download scripts are not set up. This is deferred to Phase 6. The Planet evaluation is the most interesting part — testing generalization to a completely different sensor.

### 7. Masking Ratio May Need Tuning
plan.md specifies masking ratio μ=1.0, σ=0.25, min=0.75 (75-100% masked). The I-JEPA paper uses μ=0.5, σ=0.2 for ViT-B on ImageNet. The higher masking in the plan may be too aggressive for the smaller ortho dataset. This is a tunable hyperparameter.

### 8. EMA Decay May Need Schedule
plan.md uses constant EMA decay 0.9999. The I-JEPA paper uses a cosine schedule from 0.996 to 1.0. A constant 0.9999 is at the extreme high end — the target encoder may track the context encoder too slowly during early training.