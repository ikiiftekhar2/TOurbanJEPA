# UrbanJEPA v2 — Maximum PSNR Plan
 
## Ceiling Analysis
 
```
28.80 dB  — VAE encode(GT).mean → VAE decode   [latent-path hard ceiling]
~33-35 dB — pixel refinement on top             [achievable with Stage 2]
```
 
Target: 26-30 dB. Current best: 20.90 dB (exp4_joint, epoch 0).
 
---
 
## What Went Wrong: Complete Bug & Design Failure Audit
 
### Bug 1: Stochastic VAE Encoding Poisons Training Targets
 
**File:** `src/models/urbanjepa.py`, line 387
 
```python
# CURRENT (BROKEN):
latents = self.vae.encode(images.to(vae_dtype)).latent_dist.sample()
```
 
`encode_to_latent()` calls `.sample()` on the VAE posterior. The SD-VAE encoder
outputs a Gaussian distribution (mean, logvar) per latent pixel. `.sample()` draws
randomly from that Gaussian each call. The same high-res image produces a different
target latent every epoch. The regression model is fitting a moving target.
 
`train_regress.py` line 281 calls this every batch:
```python
target_latent = model.encode_to_latent(high)  # uses .sample() internally
```
 
The irreducible loss floor from posterior variance means the model can never achieve
zero training loss even with a perfect architecture.
 
**Cost:** ~0.3-0.5 dB. The model learns the posterior mean anyway (MSE minimizer
is the conditional expectation), but the noisy gradients slow convergence and prevent
the model from fitting fine structure that sits within the posterior variance.
 
**Fix:**
```python
# CORRECT:
latents = self.vae.encode(images.to(vae_dtype)).latent_dist.mean
```
 
Use `.mean` everywhere. Deterministic, reproducible, and is the MAP estimate of the
latent. No information is lost since the decoder was trained to handle mean inputs.
 
---
 
### Bug 2: fp16 VAE in Regression Training, fp32 in VAE Finetuning
 
**File:** `src/training/train_regress.py`, line 148
 
```python
# Regression training — fp16
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16)
```
 
**File:** `src/training/train_vae.py`, line 154
 
```python
# VAE finetuning — fp32
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float32)
```
 
fp16 has ~3.3 decimal digits of mantissa precision. The VAE latent values are scaled
by 0.18215, producing small values where fp16 quantization truncates the bottom bits.
Both the training target (encode path) and the PSNR measurement (decode path) have
fp16 quantization artifacts that the fp32 VAE finetuning didn't have.
 
The finetuned VAE decoder was saved from an fp32 training run, then loaded into an
fp16 model for regression training. Weight precision is lost at load time.
 
**Cost:** ~0.2-0.3 dB. Small per-pixel, but systematic across every sample.
 
**Fix:** Load VAE in fp32 for all training. Use autocast for forward passes if
VRAM is tight, but store weights and compute targets in fp32.
 
---
 
### Bug 3: Zero Pixel-Space Training Signal
 
**File:** `src/training/train_regress.py`, line 288
 
```python
loss = nn.functional.mse_loss(pred_latent, target_latent_2d)
```
 
The entire training loss is MSE in latent space. VGG perceptual loss is computed only
during validation for logging purposes (line 373). No pixel L1, no perceptual loss,
no frequency loss contributes to gradients during training.
 
Latent MSE and pixel PSNR are correlated but not equivalent. The model can minimize
latent MSE by getting per-channel means right while smearing spatial details that
would be penalized by pixel losses. This is exactly why Path B plateaued: the model
found the latent MSE optimum, which is not the pixel PSNR optimum.
 
**Cost:** ~0.5-1.0 dB. The model has no gradient signal telling it about decoded
image quality. It optimizes a proxy and gets proxy-optimal, not objective-optimal.
 
**Fix:** Multi-loss during training:
```python
loss = 0.5 * mse_latent + 2.0 * l1_pixel + 0.5 * vgg_perceptual + 0.3 * hf_loss
```
 
---
 
### Bug 4: JEPA Phase 2 EMA Collapse
 
**File:** `src/training/train_jepa.py` (fixed EMA decay 0.9999)
 
**Log:** `training_phase2.log`
 
```
Epoch  2: val_loss = 0.0336   ← best
Epoch  5: val_loss = 0.0350
Epoch  8: val_loss = 0.0778
Epoch 10: val_loss = 0.1185
Epoch 12: val_loss = 0.1518   ← 4.5x worse than best
```
 
EMA decay 0.9999 means the target encoder updates by 0.01% per step. With ~1870
steps/epoch (14,952 samples / batch 8), the target encoder's half-life is ~69,000
steps or ~37 epochs. In early training when the context encoder is changing rapidly,
the target encoder lags far behind, producing stale target embeddings. The prediction
loss becomes meaningless and the context encoder drifts into a bad region of feature
space.
 
Best checkpoint was epoch 2, which means the JEPA got approximately 2 useful
epochs of training out of 12 attempted. Everything downstream is built on a
barely-trained backbone.
 
**Cost:** ~1-2 dB. The context encoder's features are suboptimal for any downstream
task. The diagnostic shows context_enc(HR) CNN decoder at 21.46 dB. With proper
training, this should be closer to the target_enc upper bound of 21.97 dB or above.
 
**Fix:** Cosine EMA schedule starting at 0.996 (fast tracking) ramping to 1.0 (stable):
```python
decay = 0.996 + 0.004 * (1 + cos(pi * step/total_steps)) / 2
```
 
---
 
### Design Flaw 1: Per-Token MLP With No Spatial Communication
 
**File:** `src/models/latent_regressor.py`
 
```python
self.mlp = nn.Sequential(
    nn.Linear(embed_dim, hidden_dim),   # 768 → 512
    nn.GELU(),
    nn.Linear(hidden_dim, hidden_dim),  # 512 → 512
    nn.GELU(),
    nn.Linear(hidden_dim, out_dim),     # 512 → 16
)
```
 
Each of 256 tokens is processed independently through the same MLP. Token at
position (0,0) has zero information about what token (0,1) is producing.
In a 16x16 grid, neighboring tokens should produce spatially smooth VAE latents,
but the model has no mechanism to enforce this.
 
`LatentRegressorConv` adds two conv ResBlocks on the 16x16 feature map (3x3 kernel),
giving a 5x5 effective receptive field. That's 31% of the 16x16 grid. Still no
global communication.
 
**Cost:** ~2-3 dB. The single largest architectural bottleneck.
 
---
 
### Design Flaw 2: 768→16 Compression Per Token
 
**File:** `src/models/latent_regressor.py`, line 43
 
The MLP compresses each 768-dim JEPA embedding down to 16-dim VAE token in three
linear layers. That's 48x compression. The JEPA embedding is rich (cos_sim 0.996
with ground truth), but the 16-dim output can only carry 16 numbers worth of
information. The intermediate hidden_dim=512 helps somewhat, but the final
Linear(512, 16) is a severe bottleneck.
 
**Cost:** ~1-2 dB. Information that exists in the JEPA embedding is destroyed
before the VAE decoder can use it.
 
---
 
### Design Flaw 3: Ignoring the Low-Res VAE Latent
 
**File:** `src/models/urbanjepa.py`, line 254
 
```python
def forward_regress(self, low_res):
    ctx_features = self.context_encoder(low_res)  # (B, 256, 768)
    return self.latent_regressor(ctx_features)     # (B, 4, 32, 32)
```
 
Path B predicts the high-res latent purely from JEPA embeddings. The low-res image's
own VAE encoding, which already contains accurate coarse structure, is completely
discarded. The model must hallucinate structural information it could just copy.
 
Path C (U-Net) does use the low-res latent as input with a residual connection, but
the JEPA conditioning is compressed 768→128 via concatenation (see Design Flaw 4).
 
**Cost:** ~1-2 dB.
 
---
 
### Design Flaw 4: JEPA Conditioning via Concatenation (Path C)
 
**File:** `src/models/latent_unet.py`, lines 166-187
 
```python
class JEPAConditioningProjector(nn.Module):
    def __init__(self, embed_dim=768, cond_channels=128):
        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, cond_channels * 2, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(cond_channels * 2, cond_channels, 3, padding=1),
        )
```
 
256 tokens of 768-dim get reshaped to (B, 768, 16, 16), upsampled to (B, 768, 32, 32),
then projected to (B, 128, 32, 32). That's 768→128 compression of the JEPA signal
before the U-Net sees it. Concatenated with the 4-channel latent as input.
 
Cross-attention would let the U-Net pull selectively from full 768-dim JEPA tokens
per spatial position. Concatenation forces a fixed projection that must work for all
positions equally.
 
**Cost:** ~1-2 dB in Path C specifically.
 
---
 
### Design Flaw 5: Path C Loss Weighting (Confirmed)
 
**File:** `src/training/train_unet.py`
 
```python
p.add_argument("--lambda_img", type=float, default=0.1)
# ...
loss = loss_latent + args.lambda_img * loss_img   # L1_latent + 0.1 * L1_pixel
```
 
91% of gradient signal from latent L1. PSNR declined from 19.29 dB (epoch 0) to
18.12 dB (epoch 3) while latent loss improved. The model overfits the proxy metric.
 
---
 
### Design Flaw 6: Predictor Distribution Shift in "Predict All" Mode
 
**File:** `src/models/predictor.py`, `forward()` method
 
The predictor was trained with ~50% masking (sample_mask default mask_ratio_mean=0.5).
Training sequences were ~256 context + ~128 mask = ~384 tokens.
 
`get_jepa_features_all()` feeds 256 context + 256 mask queries = 512 tokens. The
attention patterns, positional embeddings, and internal representations were never
trained on this sequence length. The 4.6 dB gap between target_enc CNN (21.97 dB)
and predictor CNN (17.36 dB) at 0.996 cosine similarity is partly this distribution
shift. High cosine similarity captures semantic alignment but misses structural
details that matter for pixel reconstruction.
 
---
 
## Estimated dB Recovery Per Fix
 
```
Fix                                   Expected gain   Cumulative
------------------------------------------------------------------------
Bug 1: .mean vs .sample               +0.3-0.5 dB    ~21.2-21.4
Bug 2: fp32 VAE                        +0.2-0.3 dB    ~21.5-21.7
Bug 3: Pixel-space training loss       +0.5-1.0 dB    ~22.0-22.7
Bug 4: JEPA retrain w/ cosine EMA      +1.0-2.0 dB    ~23.0-24.7
Flaw 1+2: Cross-attn transformer       +2.0-3.0 dB    ~25.0-27.7
Flaw 3: Residual from LR latent        +1.0-2.0 dB    (included above)
Phase 3: VAE decoder on pred latents   +0.5-1.5 dB    ~26.0-29.0
Phase 4: Pixel refinement              +1.0-2.0 dB    ~27.0-30.0
Phase 5: Joint finetune                +0.3-1.0 dB    ~27.5-30.0
------------------------------------------------------------------------
```
 
These don't simply stack (diminishing returns), but the direction is clear.
 
---
 
## The Rebuild: Five Phases
 
```
Phase 1: JEPA Retrain         (2-3 days)   → fix Bug 4
Phase 2: Cross-Attn Transformer (5-7 days) → fix all design flaws + bugs 1-3
Phase 3: VAE Decoder Finetune  (1-2 days)  → adapt decoder to predicted latents
Phase 4: Pixel Refinement      (2-3 days)  → break VAE ceiling
Phase 5: Joint Finetune        (2-3 days)  → end-to-end polish
```
 
---
 
## Phase 1: JEPA Retraining
 
### Changes From Original
 
| Parameter          | Original Phase 2     | Phase 1 v2           |
|--------------------|----------------------|----------------------|
| EMA decay          | 0.9999 fixed         | cosine 0.996 → 1.0  |
| LR                 | 8e-4                 | 5e-4                 |
| Warmup             | none                 | 5 epochs to 5e-4     |
| patches_per_epoch  | 4 (14,952 samples)   | 32 (119,616 samples) |
| Epochs             | 50 (early stopped 12)| 25                   |
 
### Code Changes to `train_jepa.py`
 
```python
import math
 
def cosine_ema_decay(step, total_steps, base=0.996, final=1.0):
    """Cosine EMA: fast tracking early, stable late."""
    progress = step / max(total_steps, 1)
    return final - (final - base) * (1 + math.cos(math.pi * progress)) / 2
 
# Replace the fixed EMA update in the training loop:
# OLD: model.update_target_encoder()  # uses self.ema_decay = 0.9999
# NEW:
decay = cosine_ema_decay(global_step, total_steps)
with torch.no_grad():
    for ctx_p, tgt_p in zip(
        model.context_encoder.parameters(),
        model.target_encoder.parameters(),
    ):
        tgt_p.data.mul_(decay).add_(ctx_p.data, alpha=1 - decay)
```
 
Add warmup + cosine LR schedule:
```python
warmup_steps = warmup_epochs * steps_per_epoch
 
def lr_lambda(step):
    if step < warmup_steps:
        return step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return 0.5 * (1 + math.cos(math.pi * progress))
 
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
# Step per batch, not per epoch
```
 
### Launch
 
```bash
python -m src.training.train_jepa \
    --data_dir data/ortho \
    --epochs 25 \
    --lr 5e-4 \
    --batch_size 8 \
    --grad_accum 4 \
    --patches_per_epoch 32 \
    --log_dir runs/jepa_v2
```
 
### Success Gate
 
```
val_loss < 0.030                  (was 0.0336)
cos_sim  > 0.997                  (was 0.9957)
target_enc(HR) → CNN > 22 dB     (was 21.97)
```
 
After Phase 1, re-run `scripts/diagnose_phase2.py` with the new checkpoint.
 
---
 
## Phase 2: Cross-Attention Latent Transformer
 
### Architecture
 
```
low_res_image (256x256)
    |
    +-- JEPA context_encoder (frozen)   → ctx_emb   (B, 256, 768)
    +-- JEPA predictor (frozen)         → pred_emb  (B, 256, 768)
    +-- VAE encoder (frozen, fp32, .mean) → lr_lat  (B, 4, 32, 32)
                                              |
                                         2x2 group
                                              |
                                         lr_tokens (B, 256, 16)
                                              |
                                       Linear(16, 768) + pos_embed
                                              |
                                          queries (B, 256, 768)
                                              |
    ctx_emb ---+                              |
               +-- concat --> kv (B, 512, 768)|
    pred_emb --+     + type embeds            |
                                              |
                     8x Transformer Decoder Layer:
                         Self-Attention (queries, 12 heads)
                         Cross-Attention (q=queries, kv=jepa, 12 heads)
                         FFN (768 -> 3072 -> 768, GELU)
                                              |
                                         final_norm
                                              |
                                       Linear(768, 16) [zero-init]
                                              |
                                         residual_out (B, 256, 16)
                                              |
                                      + lr_tokens [skip connection]
                                              |
                                         hr_tokens (B, 256, 16)
                                              |
                                        ungroup 2x2
                                              |
                                         hr_latent (B, 4, 32, 32)
                                              |
                                    VAE decoder (frozen)
                                              |
                                         output_image (B, 3, 256, 256)
```
 
~50M params. d_model=768 matches JEPA dim (zero-loss cross-attention path).
Residual from lr_tokens means output starts as low-res latent at init.
 
### Code: `src/models/latent_transformer.py`
 
```python
import math
import torch
import torch.nn as nn
 
 
class LatentCrossAttentionTransformer(nn.Module):
    def __init__(
        self,
        token_dim=16,
        jepa_dim=768,
        d_model=768,
        n_layers=8,
        n_heads=12,
        ffn_ratio=4.0,
        n_tokens=256,
        dropout=0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_tokens = n_tokens
 
        self.input_proj = nn.Linear(token_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, n_tokens, d_model) * 0.02)
 
        self.ctx_kv_proj = nn.Linear(jepa_dim, d_model)
        self.pred_kv_proj = nn.Linear(jepa_dim, d_model)
        self.ctx_type_embed = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pred_type_embed = nn.Parameter(torch.zeros(1, 1, d_model))
 
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, int(d_model * ffn_ratio), dropout)
            for _ in range(n_layers)
        ])
 
        self.final_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, token_dim)
 
        # Zero-init so residual starts at zero => output = lr_tokens at init
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
        self._init_weights()
 
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) and m is not self.output_proj:
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
 
    def forward(self, lr_tokens, ctx_emb, pred_emb):
        """
        lr_tokens:  (B, 256, 16)  grouped low-res VAE latent
        ctx_emb:    (B, 256, 768) JEPA context encoder output
        pred_emb:   (B, 256, 768) JEPA predictor output
        Returns:    (B, 256, 16)  predicted high-res VAE latent (grouped)
        """
        queries = self.input_proj(lr_tokens) + self.pos_embed
 
        ctx_kv = self.ctx_kv_proj(ctx_emb) + self.ctx_type_embed
        pred_kv = self.pred_kv_proj(pred_emb) + self.pred_type_embed
        kv = torch.cat([ctx_kv, pred_kv], dim=1)  # (B, 512, 768)
 
        x = queries
        for layer in self.layers:
            x = layer(x, kv)
 
        x = self.final_norm(x)
        residual = self.output_proj(x)
        return lr_tokens + residual
 
 
class DecoderLayer(nn.Module):
    """Pre-norm: self-attn -> cross-attn -> FFN, each with residual."""
 
    def __init__(self, d_model, n_heads, ffn_dim, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)
 
        self.norm2 = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)
 
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )
 
    def forward(self, x, kv):
        h = self.norm1(x)
        x = x + self.self_attn(h, h, h, need_weights=False)[0]
 
        h = self.norm2(x)
        kv_n = self.norm_kv(kv)
        x = x + self.cross_attn(h, kv_n, kv_n, need_weights=False)[0]
 
        h = self.norm3(x)
        x = x + self.ffn(h)
        return x
```
 
### Code: `src/models/token_utils.py`
 
```python
import torch
 
def group_latent_tokens(latent_2d, group_size=2):
    """(B, 4, 32, 32) -> (B, 256, 16)"""
    B, C, H, W = latent_2d.shape
    g = group_size
    x = latent_2d.reshape(B, C, H // g, g, W // g, g)
    x = x.permute(0, 2, 4, 3, 5, 1)  # (B, H//g, W//g, g, g, C)
    return x.reshape(B, (H // g) * (W // g), C * g * g)
 
def ungroup_latent_tokens(tokens, grid_h=16, grid_w=16, channels=4, group_size=2):
    """(B, 256, 16) -> (B, 4, 32, 32)"""
    B, N, D = tokens.shape
    g = group_size
    x = tokens.reshape(B, grid_h, grid_w, g, g, channels)
    x = x.permute(0, 5, 1, 3, 2, 4)  # (B, C, grid_h, g, grid_w, g)
    return x.reshape(B, channels, grid_h * g, grid_w * g)
 
def test_roundtrip():
    """Run ONCE before any training. Failure = every experiment is garbage."""
    latent = torch.randn(2, 4, 32, 32)
    tokens = group_latent_tokens(latent)
    recovered = ungroup_latent_tokens(tokens)
    err = (latent - recovered).abs().max().item()
    assert err < 1e-6, f"Roundtrip FAILED: max error = {err}"
    print(f"OK: {latent.shape} -> {tokens.shape} -> {recovered.shape}, err={err:.2e}")
 
if __name__ == '__main__':
    test_roundtrip()
```
 
### Code: `src/training/losses.py` (replace existing)
 
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights
 
 
class VGGPerceptualLoss(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        vgg = vgg16(weights=VGG16_Weights.DEFAULT).to(device).eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.slices = nn.ModuleList([
            vgg.features[:4],   # relu1_2
            vgg.features[:9],   # relu2_2
            vgg.features[:16],  # relu3_3
            vgg.features[:23],  # relu4_3
        ])
        self.register_buffer('mean',
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std',
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
 
    def forward(self, pred, target):
        pred = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std
        loss = 0.0
        for s in self.slices:
            pred = s(pred)
            target = s(target)
            loss += F.l1_loss(pred, target)
        return loss / len(self.slices)
 
 
def high_frequency_loss(pred, target, cutoff_ratio=0.125):
    """L1 on high-freq components via FFT."""
    B, C, H, W = pred.shape
    cy, cx = H // 2, W // 2
    ry, rx = int(H * cutoff_ratio), int(W * cutoff_ratio)
 
    yy, xx = torch.meshgrid(
        torch.arange(H, device=pred.device),
        torch.arange(W, device=pred.device), indexing='ij')
    mask = ((yy - cy).float() / max(ry, 1)) ** 2 + \
           ((xx - cx).float() / max(rx, 1)) ** 2 >= 1.0
    mask = torch.fft.fftshift(mask.float()).unsqueeze(0).unsqueeze(0)
 
    pred_hf = torch.fft.ifft2(torch.fft.fft2(pred, norm='ortho') * mask, norm='ortho').real
    tgt_hf = torch.fft.ifft2(torch.fft.fft2(target, norm='ortho') * mask, norm='ortho').real
    return F.l1_loss(pred_hf, tgt_hf)
```
 
### Code: `src/models/pixel_refiner.py`
 
```python
import torch
import torch.nn as nn
 
 
class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
 
    def forward(self, x):
        return self.relu(x + self.conv2(self.relu(self.conv1(x))))
 
 
class PixelRefiner(nn.Module):
    """
    Pixel-space residual CNN. Takes Stage 1 output + low-res,
    predicts correction to break the VAE ceiling. ~1.5M params.
    """
    def __init__(self, in_channels=6, base_channels=64, n_blocks=8):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1), nn.ReLU(True))
        self.body = nn.Sequential(*[ResBlock(base_channels) for _ in range(n_blocks)])
        self.tail = nn.Conv2d(base_channels, 3, 3, padding=1)
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)
 
    def forward(self, stage1_image, low_res_image):
        x = torch.cat([stage1_image, low_res_image], dim=1)
        x = self.head(x)
        x = self.body(x)
        return (stage1_image + self.tail(x)).clamp(0, 1)
```
 
### Training: The Loss Function (Critical)
 
This is where Path B and Path C both failed. The loss weighting determines everything.
 
```python
# ENCODE TARGETS WITH .mean AND fp32 — fixes Bug 1 and Bug 2
with torch.no_grad():
    ctx_emb = jepa.context_encoder(low_res)
    pred_emb = get_predictor_output(jepa, ctx_emb)
    lr_latent_2d = vae.encode(low_res).latent_dist.mean * 0.18215
    hr_latent_2d = vae.encode(high_res).latent_dist.mean * 0.18215
    lr_tokens = group_latent_tokens(lr_latent_2d)
    hr_tokens = group_latent_tokens(hr_latent_2d)
 
# FORWARD
pred_hr_tokens = transformer(lr_tokens, ctx_emb, pred_emb)
 
# LATENT LOSS
loss_latent = F.mse_loss(pred_hr_tokens, hr_tokens)
 
# DECODE TO PIXEL SPACE
pred_latent_2d = ungroup_latent_tokens(pred_hr_tokens)
pred_image = vae.decode(pred_latent_2d / 0.18215).sample.clamp(0, 1)
 
# PIXEL LOSSES — these drive PSNR, not the latent loss
loss_pixel = F.l1_loss(pred_image, high_res)
loss_percep = vgg_loss(pred_image, high_res)
loss_hf = high_frequency_loss(pred_image, high_res)
 
# COMBINED — pixel-dominant
loss = 0.5 * loss_latent + 2.0 * loss_pixel + 0.5 * loss_percep + 0.3 * loss_hf
```
 
**Why these weights:**
- `loss_pixel` at 2.0: PSNR is computed from pixel error. This must dominate.
- `loss_latent` at 0.5: Stabilizes early training when pixel gradients through
  VAE decode are noisy. Becomes less important after epoch 5.
- `loss_percep` at 0.5: Prevents blur/averaging failure mode of pure L1/MSE.
- `loss_hf` at 0.3: Directly targets edges and textures where remaining dB hide.
 
**If PSNR plateaus below 24 dB after epoch 10:** increase pixel weight to 4.0,
decrease latent weight to 0.2.
 
### Training Config
 
```python
# Optimizer
optimizer = AdamW(transformer.parameters(), lr=3e-4, weight_decay=0.05, betas=(0.9, 0.95))
 
# Schedule: linear warmup 3 epochs, cosine decay to 1e-6
warmup_steps = 3 * steps_per_epoch
def lr_lambda(step):
    if step < warmup_steps:
        return step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return max(1e-6 / 3e-4, 0.5 * (1 + math.cos(math.pi * progress)))
 
scheduler = LambdaLR(optimizer, lr_lambda)
 
# Mixed precision (transformer is 50M params, fp16 saves VRAM)
scaler = torch.amp.GradScaler()
# BUT: VAE encode targets in fp32 (Bug 2 fix), VAE decode in fp32 for stability
 
# Grad clipping
torch.nn.utils.clip_grad_norm_(transformer.parameters(), 1.0)
```
 
### VRAM Budget (24 GB)
 
```
JEPA ctx_enc forward (frozen, no grad):   ~2 GB
JEPA predictor forward (frozen):          ~2 GB
VAE encode x2 (frozen, fp32):             ~2 GB
Transformer fwd+bwd (50M, fp16):          ~8 GB
VAE decode (frozen, fp32, grad through):  ~3 GB
VGG perceptual (frozen):                  ~1 GB
Overhead:                                 ~3 GB
                                          ------
Total:                                    ~21 GB at batch 24
```
 
If OOM: batch 16 with grad_accum 2 (effective 32).
 
### Launch
 
```bash
python -m src.training.train_stage1 \
    --data_dir data/ortho \
    --jepa_ckpt models/checkpoints/jepa_v2_best.pt \
    --batch_size 24 \
    --epochs 30 \
    --lr 3e-4 \
    --patches_per_epoch 128 \
    --num_workers 8 \
    --log_dir runs/stage1
```
 
### Success Gate
 
```
PSNR > 25 dB by epoch 15   → on track
PSNR > 26 dB by epoch 30   → proceed to Phase 3
Plateaus at 23-24 dB        → increase pixel loss weight to 4.0
PSNR < 22 dB after epoch 5  → bug in pipeline, debug before continuing
```
 
### Troubleshooting
 
1. **Gray/mean output after 3+ epochs:** Gradient not flowing through VAE decode.
   Use only latent MSE for first 3 epochs, then add pixel losses.
 
2. **PSNR worse than 20 dB at epoch 0:** The zero-init residual means epoch-0
   output should equal the low-res latent decoded through VAE. If that's below
   20 dB, the low-res VAE latent itself is poor, meaning the VAE encoding of the
   downsampled image loses too much. Check VAE scale factor (0.18215).
 
3. **NaN loss:** fp16 overflow in attention. Add `torch.nn.utils.clip_grad_norm_`
   and ensure softmax inputs aren't exploding. Try d_model/sqrt(n_heads) scaling.
 
4. **OOM mid-epoch:** Reduce batch to 16, add grad_accum 2. Or reduce n_layers
   to 6 for initial testing, then scale back to 8 once VRAM is mapped.
 
---
 
## Phase 3: VAE Decoder Finetune on Predicted Latents
 
Previous VAE finetuning was circular: encode(GT) → decode → compare with GT.
The decoder was already near-optimal for its own encoder's outputs (28.63 → 28.80 dB).
 
Now finetune the decoder on the transformer's predicted latents, which have
systematic biases the pretrained decoder doesn't know about.
 
```python
# Frozen: JEPA, transformer, VAE encoder
# Trainable: VAE decoder only (49M params), fp32
 
optimizer = AdamW(vae.decoder.parameters(), lr=1e-4, weight_decay=0.01)
 
for epoch in range(15):
    for low_res, high_res in train_loader:
        with torch.no_grad():
            # Full Stage 1 pipeline
            ctx_emb = jepa.context_encoder(low_res)
            pred_emb = get_predictor_output(jepa, ctx_emb)
            lr_lat = vae.encode(low_res).latent_dist.mean * 0.18215
            lr_tokens = group_latent_tokens(lr_lat)
            pred_tokens = transformer(lr_tokens, ctx_emb, pred_emb)
            pred_lat = ungroup_latent_tokens(pred_tokens)
 
        # Trainable decode
        pred_image = vae.decode(pred_lat / 0.18215).sample.clamp(0, 1)
        loss = F.l1_loss(pred_image, high_res) + 0.5 * vgg_loss(pred_image, high_res)
 
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(vae.decoder.parameters(), 1.0)
        optimizer.step()
```
 
If gain < 0.5 dB after 15 epochs, the pretrained decoder is already adequate for
the predicted latent distribution. Skip to Phase 4.
 
If you want the "translator" approach instead: train a fresh CNN
(4x32x32 → 3x256x256) from scratch on predicted latents. Same training loop,
replace `vae.decode()` with a custom upsampling network. Only do this if
finetuning gives < 0.3 dB and you suspect the SD decoder architecture itself
is the limiting factor.
 
---
 
## Phase 4: Pixel-Space Refinement
 
Cache Stage 1 + Phase 3 outputs to disk first. Recomputing them every epoch
wastes 90% of GPU time.
 
```python
# Step 1: Cache (run once)
for low_res, high_res in dataloader:
    with torch.no_grad():
        stage1_out = full_pipeline(low_res)  # through finetuned VAE decoder
    save(stage1_out, low_res, high_res)
 
# Step 2: Train refiner (~1.5M params)
refiner = PixelRefiner()
optimizer = AdamW(refiner.parameters(), lr=1e-3)
 
for epoch in range(20):
    for stage1, low_res, high_res in cached_loader:
        refined = refiner(stage1, low_res)
 
        loss = (1.0 * F.l1_loss(refined, high_res)
              + 0.3 * vgg_loss(refined, high_res)
              + 0.3 * high_frequency_loss(refined, high_res))
 
        # Optional: SSIM loss
        # from pytorch_msssim import ssim
        # loss += 0.5 * (1.0 - ssim(refined, high_res, data_range=1.0))
 
        loss.backward()
        optimizer.step()
```
 
---
 
## Phase 5: Joint Finetune
 
Only after all previous phases converge.
 
```
Trainable:
  Transformer (50M)         lr = 1e-5
  JEPA context_encoder (86M) lr = 1e-6
  JEPA predictor (86M)       lr = 1e-6
  VAE decoder (49M)          lr = 5e-6
  Pixel refiner (1.5M)       lr = 1e-4
 
Frozen:
  VAE encoder (NEVER unfreeze, latent space must be stable)
  JEPA target encoder (vestigial, not used in regression pipeline)
```
 
5-10 epochs max. Stop if val PSNR drops for 2 consecutive epochs.
Batch size 8-12 with grad_accum to effective 24.
 
---
 
## Files to Create
 
```
NEW:
  src/models/latent_transformer.py     LatentCrossAttentionTransformer
  src/models/pixel_refiner.py          PixelRefiner
  src/models/token_utils.py            group/ungroup + roundtrip test
  src/training/train_stage1.py         Phase 2 training loop
  src/training/train_stage2.py         Phase 4 with caching
  src/training/train_vae_stage1.py     Phase 3 VAE decoder finetune
  src/training/train_joint.py          Phase 5
 
MODIFY:
  src/training/losses.py               Add VGGPerceptualLoss, high_frequency_loss
  src/training/train_jepa.py           Cosine EMA, warmup LR schedule
  src/models/urbanjepa.py              .mean instead of .sample in encode_to_latent
 
DO NOT MODIFY:
  src/data/ortho_dataset.py            Works as-is
  src/models/encoder.py                Works as-is
  src/models/predictor.py              Works as-is
```
 
---
 
## Non-Negotiable Rules
 
1. **`.mean` not `.sample`** for VAE encoding. Everywhere. Always.
2. **fp32 VAE** for target computation. Use autocast for forward passes if needed,
   but targets must be fp32.
3. **`.clamp(0, 1)`** after every VAE decode. Unclamped values corrupt PSNR.
4. **`* 0.18215` on encode, `/ 0.18215` on decode.** Getting this wrong is silent
   and costs 5+ dB.
5. **`torch.no_grad()`** for all frozen components. Not optional. Storing grad
   graphs for 86M JEPA params that never update will OOM.
6. **Run `token_utils.test_roundtrip()` before first training.** If it fails,
   every experiment is garbage.
7. **Save checkpoints every epoch.** System reboots daily.
8. **Log images to TensorBoard every val epoch.** 4 triplets of (low_res, predicted,
   ground_truth). Visual inspection catches bugs metrics miss.
 
---
 
## Dependencies
 
```bash
pip install pytorch-msssim --break-system-packages   # SSIM loss (optional)
# Everything else already installed
```