"""
Denoiser (εθ) — predicts noise given noisy tokens, timestep, and JEPA embeddings.

Two architectures:
- DenoisingMLP: per-token MLP, 4M params (legacy, kept for reference)
- TransformerDenoiser: DiT-style transformer with self-attention across tokens (active)

TransformerDenoiser follows DiT (Peebles & Xie 2023) / MAR (Li et al. 2024):
- Learnable position embeddings for spatial layout
- Self-attention captures spatial noise correlations across tokens
- Per-token JEPA embedding added to token features
- AdaLN-Zero: global (time + pooled JEPA) modulates each transformer block
- 6 layers, 8 heads, d_model=512, ~31M params
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Shared: Sinusoidal time embedding
# ---------------------------------------------------------------------------

class SinusoidalTimestepEmbedding(nn.Module):
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
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


# ---------------------------------------------------------------------------
# DiT-style TransformerDenoiser
# ---------------------------------------------------------------------------

def modulate(x, shift, scale):
    """AdaLN modulation: scale and shift the normalized input."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """One DiT block: AdaLN → Self-Attention → AdaLN → FFN, with gated residuals."""

    def __init__(self, d_model, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, batch_first=True, dropout=0.0
        )
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, int(d_model * mlp_ratio)),
            nn.GELU(approximate="tanh"),
            nn.Linear(int(d_model * mlp_ratio), d_model),
        )
        # AdaLN-Zero: 6 params per dim (shift, scale, gate × attn + shift, scale, gate × ffn)
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 6 * d_model),
        )
        # Zero-init: last layer outputs zeros so block is identity at start
        nn.init.zeros_(self.adaLN[1].weight)
        nn.init.zeros_(self.adaLN[1].bias)

    def forward(self, x, c):
        # c: (B, d_model) global condition
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.adaLN(c).chunk(6, dim=1)

        # Self-attention with AdaLN
        x_norm = modulate(self.norm1(x), shift_a, scale_a)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + gate_a.unsqueeze(1) * attn_out

        # FFN with AdaLN
        x_norm = modulate(self.norm2(x), shift_m, scale_m)
        x = x + gate_m.unsqueeze(1) * self.mlp(x_norm)

        return x


class TransformerDenoiser(nn.Module):
    """
    DiT-style transformer denoiser with cross-token self-attention.

    Key differences from old DenoisingMLP:
    - Self-attention: tokens see each other, capturing spatial noise correlations
    - Per-token JEPA projection: semantic information injected into each token
    - AdaLN-Zero: stable training (identity at initialization)
    - 31M params vs 4M — more capacity for satellite texture modeling
    """

    def __init__(self, token_dim=16, embed_dim=768, d_model=512,
                 num_heads=8, num_layers=6, time_dim=256, max_tokens=256):
        super().__init__()
        self.token_dim = token_dim
        self.d_model = d_model

        # Input: project noisy token (16-dim) → d_model
        self.input_proj = nn.Linear(token_dim, d_model)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)

        # Per-token JEPA projection: embed_dim (768) → d_model (512)
        self.jepa_proj = nn.Linear(embed_dim, d_model)
        nn.init.xavier_uniform_(self.jepa_proj.weight)
        nn.init.zeros_(self.jepa_proj.bias)

        # Learnable position embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, max_tokens, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Time embedding (sinusoidal → MLP)
        self.time_embed = nn.Sequential(
            SinusoidalTimestepEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, d_model),
        )

        # Global condition projector: (time_emb_proj + pooled_jepa) → d_model
        # JEPA pooled: average over tokens
        self.cond_proj = nn.Linear(d_model + d_model, d_model)
        nn.init.xavier_uniform_(self.cond_proj.weight)
        nn.init.zeros_(self.cond_proj.bias)

        # DiT blocks
        self.blocks = nn.ModuleList([
            DiTBlock(d_model, num_heads, mlp_ratio=4.0)
            for _ in range(num_layers)
        ])

        # Final output: d_model → token_dim (noise prediction)
        self.final_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.output_proj = nn.Linear(d_model, token_dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x_noisy, t, z):
        """
        x_noisy: (B, N, token_dim) noisy tokens
        t:       (B,) integer timesteps
        z:       (B, N, embed_dim) predicted JEPA embeddings per token
        Returns: (B, N, token_dim) predicted noise
        """
        B, N, _ = x_noisy.shape

        # Project noisy tokens to d_model
        x = self.input_proj(x_noisy)                     # (B, N, d_model)

        # Add per-token JEPA semantic information
        x = x + self.jepa_proj(z)                        # (B, N, d_model)

        # Add position embeddings
        x = x + self.pos_embed[:, :N, :]

        # Global conditioning: time embedding + pooled JEPA
        t_emb = self.time_embed(t)                       # (B, d_model)
        z_pool = z.mean(dim=1)                           # (B, embed_dim) → pool
        z_pool = self.jepa_proj(z_pool)                  # (B, d_model) — reuse projection
        c = self.cond_proj(torch.cat([t_emb, z_pool], dim=-1))  # (B, d_model)

        # DiT blocks
        for block in self.blocks:
            x = block(x, c)

        # Output projection
        x = self.final_norm(x)
        x = self.output_proj(x)                          # (B, N, token_dim)
        return x


# ---------------------------------------------------------------------------
# Legacy: Per-token MLP (kept for reference / compatibility)
# ---------------------------------------------------------------------------

class AdaLN(nn.Module):
    def __init__(self, hidden_dim, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.scale = nn.Linear(cond_dim, hidden_dim)
        self.shift = nn.Linear(cond_dim, hidden_dim)
        nn.init.zeros_(self.scale.weight)
        nn.init.ones_(self.scale.bias)
        nn.init.zeros_(self.shift.weight)
        nn.init.zeros_(self.shift.bias)

    def forward(self, x, cond):
        return self.norm(x) * (1 + self.scale(cond)) + self.shift(cond)


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, cond_dim):
        super().__init__()
        self.adaln = AdaLN(hidden_dim, cond_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.zeros_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x, cond):
        residual = x
        x = self.adaln(x, cond)
        x = self.linear1(x)
        x = F.silu(x)
        x = self.linear2(x)
        return x + residual


class DenoisingMLP(nn.Module):
    """Legacy per-token MLP — 4M params, no cross-token communication."""

    def __init__(self, token_dim=16, embed_dim=768, hidden_dim=1024,
                 time_dim=256, num_blocks=6):
        super().__init__()
        self.token_dim = token_dim
        cond_dim = time_dim + embed_dim

        self.time_embed = nn.Sequential(
            SinusoidalTimestepEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
        )
        self.input_proj = nn.Linear(token_dim, hidden_dim)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)

        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, cond_dim) for _ in range(num_blocks)
        ])
        self.output_proj = nn.Linear(hidden_dim, token_dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x_noisy, t, z):
        B, N, _ = x_noisy.shape
        t_emb = self.time_embed(t).unsqueeze(1).expand(B, N, -1)
        cond = torch.cat([t_emb, z], dim=-1)

        x = x_noisy.reshape(B * N, -1)
        cond = cond.reshape(B * N, -1)

        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x, cond)
        x = self.output_proj(x)
        return x.reshape(B, N, self.token_dim)
