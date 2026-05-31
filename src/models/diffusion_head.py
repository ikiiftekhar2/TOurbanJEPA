"""
DiffusionHead — v-prediction DDIM refinement head per V4_PLAN §2.4.

Used in Phase 3+ as a refinement pass on top of the deterministic base
prediction. The base model is frozen; this module is trained to denoise
the residual between base prediction and ground truth.

Architecture:
  - 3-down/3-up UNet, base_channels=128 (effective max channels = base*8 = 1024).
  - Input: concat(noisy_x_t, base_pred) — 6 channels in, 3 channels out.
  - Conditioning: pooled ctx_features (mean of 256 tokens, 768-dim) + sinusoidal
    time embedding -> 768-dim vector -> FiLM (Perez 2018) at every block.
  - Output: v (per Salimans & Ho 2022) — more stable than epsilon at low
    inference step counts (4 steps).

Sampler: DDIM (Song et al. 2021) with deterministic eta=0, 4 steps by default.
Schedule: linear beta in [1e-4, 2e-2], T=1000 training timesteps.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- noise schedule ----------

def make_beta_schedule(T: int = 1000, beta_start: float = 1e-4,
                       beta_end: float = 2e-2) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, T)


def make_schedule_tensors(T: int = 1000) -> dict:
    betas = make_beta_schedule(T)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)         # (T,)
    sqrt_alpha_bars = alpha_bars.sqrt()
    sqrt_one_minus_alpha_bars = (1.0 - alpha_bars).sqrt()
    return {
        "betas": betas,
        "alphas": alphas,
        "alpha_bars": alpha_bars,
        "sqrt_alpha_bars": sqrt_alpha_bars,
        "sqrt_one_minus_alpha_bars": sqrt_one_minus_alpha_bars,
    }


# ---------- time embedding ----------

def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """t: (B,) integer or float timesteps. Returns (B, dim)."""
    assert dim % 2 == 0, "embedding dim must be even"
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
    return torch.cat([args.sin(), args.cos()], dim=-1)


# ---------- UNet blocks ----------

class _ResBlock(nn.Module):
    """Pre-norm residual block with FiLM conditioning."""

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.film = nn.Linear(cond_dim, 2 * out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.act = nn.SiLU()
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        scale, shift = self.film(cond).chunk(2, dim=-1)
        h = self.norm2(h)
        h = h * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.dropout(self.act(h))
        h = self.conv2(h)
        return h + self.skip(x)


class _BottleneckAttention(nn.Module):
    """Single self-attention block at the UNet bottleneck."""

    def __init__(self, channels: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, dropout=dropout,
                                          batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        h = self.norm(tokens)
        a, _ = self.attn(h, h, h, need_weights=False)
        tokens = tokens + a
        return tokens.transpose(1, 2).reshape(B, C, H, W)


# ---------- UNet ----------

class _DiffusionUNet(nn.Module):
    """3-down / 3-up UNet with FiLM conditioning + bottleneck attention.

    Layout for base=128, depth=3, input=(B, 6, 256, 256):

      Input    -> in_conv  -> 128 ch @ 256x256
      Down1    : res 128->256 -> save skip -> downsample -> 256 ch @ 128x128
      Down2    : res 256->512 -> save skip -> downsample -> 512 ch @ 64x64
      Down3    : res 512->1024 -> save skip -> downsample -> 1024 ch @ 32x32
      Mid      : res 1024 -> attn -> res 1024
      Up3      : upsample -> concat skip3 -> res 2048->512
      Up2      : upsample -> concat skip2 -> res 1024->256
      Up1      : upsample -> concat skip1 -> res 512->128
      Output   : groupnorm -> conv 128->3
    """

    def __init__(self, in_ch: int = 6, out_ch: int = 3, base_ch: int = 128,
                 depth: int = 3, cond_dim: int = 768, dropout: float = 0.1,
                 attn_heads: int = 8):
        super().__init__()
        self.in_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        ch = base_ch
        skip_chs = []
        for i in range(depth):
            next_ch = ch * 2
            self.down_blocks.append(_ResBlock(ch, next_ch, cond_dim, dropout))
            skip_chs.append(next_ch)
            self.downs.append(nn.Conv2d(next_ch, next_ch, 4, stride=2, padding=1))
            ch = next_ch

        self.mid_block1 = _ResBlock(ch, ch, cond_dim, dropout)
        self.mid_attn = _BottleneckAttention(ch, num_heads=attn_heads, dropout=dropout)
        self.mid_block2 = _ResBlock(ch, ch, cond_dim, dropout)

        self.ups = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for i in range(depth):
            self.ups.append(nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1))
            sk = skip_chs[-(i + 1)]
            next_ch = ch // 2
            self.up_blocks.append(_ResBlock(ch + sk, next_ch, cond_dim, dropout))
            ch = next_ch

        self.out_norm = nn.GroupNorm(8, ch)
        self.out_conv = nn.Conv2d(ch, out_ch, 3, padding=1)
        # Zero-init final conv so the diffusion model starts as identity-noise:
        # at step 0 the predicted v ≈ 0 -> denoising leaves x_t unchanged on
        # average (avoids destabilizing the base model in early Phase 3 epochs).
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.in_conv(x)
        skips = []
        for blk, dn in zip(self.down_blocks, self.downs):
            x = blk(x, cond)
            skips.append(x)
            x = dn(x)
        x = self.mid_block1(x, cond)
        x = self.mid_attn(x)
        x = self.mid_block2(x, cond)
        for up, blk in zip(self.ups, self.up_blocks):
            x = up(x)
            sk = skips.pop()
            x = torch.cat([x, sk], dim=1)
            x = blk(x, cond)
        return self.out_conv(self.out_norm(x))


# ---------- Top-level DiffusionHead ----------

class DiffusionHead(nn.Module):
    """v-prediction DDIM head conditioned on (base_pred, ctx_features)."""

    def __init__(self, in_channels: int = 3, base_pred_channels: int = 3,
                 ctx_dim: int = 768, cond_dim: int = 768,
                 base_ch: int = 128, depth: int = 3, T: int = 1000,
                 dropout: float = 0.1):
        super().__init__()
        self.T = T
        self.cond_dim = cond_dim
        # Time embedding MLP.
        self.time_mlp = nn.Sequential(
            nn.Linear(cond_dim, cond_dim * 2),
            nn.SiLU(),
            nn.Linear(cond_dim * 2, cond_dim),
        )
        # ctx-features pooled to (B, ctx_dim) and projected to cond_dim.
        self.ctx_proj = nn.Sequential(
            nn.LayerNorm(ctx_dim),
            nn.Linear(ctx_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        # The UNet sees [x_t || base_pred] concatenated on channel dim.
        self.unet = _DiffusionUNet(
            in_ch=in_channels + base_pred_channels, out_ch=in_channels,
            base_ch=base_ch, depth=depth, cond_dim=cond_dim, dropout=dropout,
        )
        sch = make_schedule_tensors(T)
        for k, v in sch.items():
            self.register_buffer(k, v, persistent=False)

    def _condition(self, t: torch.Tensor, ctx_features: torch.Tensor) -> torch.Tensor:
        t_emb = sinusoidal_time_embedding(t, self.cond_dim)
        t_emb = self.time_mlp(t_emb)
        ctx_pool = ctx_features.mean(dim=1)
        ctx_emb = self.ctx_proj(ctx_pool)
        return t_emb + ctx_emb

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, base_pred: torch.Tensor,
                ctx_features: torch.Tensor) -> torch.Tensor:
        """Predict v at noise level t given (x_t, base_pred, ctx)."""
        cond = self._condition(t, ctx_features)
        unet_in = torch.cat([x_t, base_pred], dim=1)
        return self.unet(unet_in, cond)

    # ---------- training ----------

    def training_loss(self, x_0: torch.Tensor, base_pred: torch.Tensor,
                      ctx_features: torch.Tensor) -> torch.Tensor:
        """Sample t ~ U(0, T), predict v, return MSE(v_pred, v_target)."""
        B = x_0.shape[0]
        device = x_0.device
        t = torch.randint(0, self.T, (B,), device=device)
        eps = torch.randn_like(x_0)
        a_bar = self.alpha_bars[t].view(B, 1, 1, 1)
        sa = a_bar.sqrt()
        soma = (1.0 - a_bar).sqrt()
        x_t = sa * x_0 + soma * eps
        v_target = sa * eps - soma * x_0
        v_pred = self(x_t, t, base_pred, ctx_features)
        return F.mse_loss(v_pred, v_target)

    # ---------- sampling ----------

    @torch.no_grad()
    def sample(self, base_pred: torch.Tensor, ctx_features: torch.Tensor,
               num_steps: int = 4, seed: Optional[int] = None,
               clamp_output: bool = True) -> torch.Tensor:
        """4-step DDIM with v-prediction. eta=0 (deterministic)."""
        B, C, H, W = base_pred.shape
        device = base_pred.device
        if seed is not None:
            g = torch.Generator(device=device).manual_seed(seed)
            x = torch.randn(B, C, H, W, generator=g, device=device)
        else:
            x = torch.randn(B, C, H, W, device=device)
        # Build the DDIM step schedule. Pick `num_steps + 1` indices uniformly
        # across the training schedule (descending), then iterate adjacent pairs.
        step_idx = torch.linspace(self.T - 1, 0, num_steps + 1).round().long().to(device)
        for i in range(num_steps):
            t = step_idx[i]
            t_next = step_idx[i + 1]
            t_batch = t.expand(B)
            a_t = self.alpha_bars[t]
            a_next = self.alpha_bars[t_next] if t_next >= 0 else torch.tensor(1.0,
                                                                              device=device)
            v = self(x, t_batch, base_pred, ctx_features)
            sqrt_a_t = a_t.sqrt()
            sqrt_oma_t = (1.0 - a_t).sqrt()
            sqrt_a_next = a_next.sqrt()
            sqrt_oma_next = (1.0 - a_next).sqrt()
            # Recover x_0 and eps from v (Salimans & Ho 2022 Eq. 5/6):
            #   x_0 = sqrt(a_t) * x_t - sqrt(1-a_t) * v
            #   eps = sqrt(a_t) * v   + sqrt(1-a_t) * x_t
            x0_pred = sqrt_a_t * x - sqrt_oma_t * v
            if clamp_output:
                x0_pred = x0_pred.clamp(0.0, 1.0)
            eps_pred = sqrt_a_t * v + sqrt_oma_t * x
            # DDIM update (Song et al. 2021, eta=0):
            x = sqrt_a_next * x0_pred + sqrt_oma_next * eps_pred
        return x.clamp(0.0, 1.0) if clamp_output else x


# ---------- smoke test ----------

def _smoke():
    import argparse, time
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)
    print(f"diffusion_head smoke on {device}, batch={args.batch_size}")

    head = DiffusionHead().to(device)
    n_params = sum(p.numel() for p in head.parameters()) / 1e6
    print(f"  params: {n_params:.1f} M")

    B = args.batch_size
    base_pred = torch.rand(B, 3, 256, 256, device=device)
    ctx = torch.randn(B, 256, 768, device=device)
    x_t = torch.randn(B, 3, 256, 256, device=device)
    t = torch.randint(0, head.T, (B,), device=device)

    # Forward smoke
    t0 = time.time()
    v_pred = head(x_t, t, base_pred, ctx)
    fwd_ms = (time.time() - t0) * 1000
    assert tuple(v_pred.shape) == (B, 3, 256, 256), f"v shape: {v_pred.shape}"
    assert torch.isfinite(v_pred).all(), "NaN/Inf in v"
    print(f"  forward OK: {tuple(v_pred.shape)} in {fwd_ms:.1f} ms")

    # Training-loss smoke
    x_0 = torch.rand(B, 3, 256, 256, device=device)
    loss = head.training_loss(x_0, base_pred, ctx)
    assert torch.isfinite(loss), "NaN loss"
    loss.backward()
    n_grad = sum(int(p.grad is not None and p.grad.abs().sum().item() > 0)
                 for p in head.parameters())
    print(f"  training_loss OK: {loss.item():.4f}, "
          f"{n_grad}/{sum(1 for _ in head.parameters())} params with grad")

    # 4-step DDIM sampling smoke
    head.eval()
    t1 = time.time()
    sampled = head.sample(base_pred, ctx, num_steps=4, seed=0)
    sample_ms = (time.time() - t1) * 1000
    assert tuple(sampled.shape) == (B, 3, 256, 256), f"sample shape: {sampled.shape}"
    assert torch.isfinite(sampled).all(), "NaN/Inf in sample"
    assert sampled.min().item() >= 0.0 and sampled.max().item() <= 1.0, \
        f"sample range: [{sampled.min().item()}, {sampled.max().item()}]"
    print(f"  4-step DDIM sample OK: {tuple(sampled.shape)} in {sample_ms:.1f} ms, "
          f"range [{sampled.min().item():.3f}, {sampled.max().item():.3f}]")

    # At step 0, zero-init out_conv -> v ≈ 0. With v=0 the DDIM update
    # is x_{t-1} = sqrt(a_next) * x_t + sqrt(1-a_next) * (sqrt(1-a_t)/sqrt(1-a_t)) * x_t
    # ... not exactly LR-preserving, but at least finite and bounded.
    print("diffusion_head smoke PASSED")


if __name__ == "__main__":
    _smoke()
