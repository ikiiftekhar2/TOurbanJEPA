"""
Denoising MLP (εθ) — predicts noise given a noisy token, timestep, and JEPA embedding.

Small MLP (6 residual blocks, ~4M params) with AdaLN conditioning.
Applied independently per token for efficiency.
"""

import math
import torch
import torch.nn as nn


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
    """Adaptive Layer Norm — conditions MLP on (timestep + JEPA embedding)."""

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
    """Single residual block: AdaLN -> Linear -> SiLU -> Linear + skip."""

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
        x = nn.functional.silu(x)
        x = self.linear2(x)
        return x + residual


class DenoisingMLP(nn.Module):
    """
    Small MLP that predicts noise ε given:
    - noisy token xit (token_dim-dimensional)
    - timestep t (embedded via sinusoidal)
    - predicted embedding zi from feature predictor

    Applied independently per token, same weights for all tokens.
    """

    def __init__(self, token_dim=4, embed_dim=768, hidden_dim=1024,
                 time_dim=256, num_blocks=6):
        super().__init__()

        self.token_dim = token_dim
        cond_dim = time_dim + embed_dim

        # Time embedding: sinusoidal -> Linear -> SiLU
        self.time_embed = nn.Sequential(
            SinusoidalTimestepEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
        )

        # Input projection: token -> hidden
        self.input_proj = nn.Linear(token_dim, hidden_dim)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)

        # Residual blocks conditioned on (time_emb, jepa_emb)
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, cond_dim) for _ in range(num_blocks)
        ])

        # Output projection: hidden -> token (noise prediction)
        self.output_proj = nn.Linear(hidden_dim, token_dim)
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

        # Time embedding: (B, time_dim) -> (B, N, time_dim)
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
