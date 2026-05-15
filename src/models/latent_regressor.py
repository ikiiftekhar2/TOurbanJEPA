"""
LatentRegressor (Path B) — direct JEPA embeddings → VAE latent regression.

Maps ViT token embeddings (B, 256, 768) to VAE latents (B, 4, 32, 32)
via per-token MLP projection + optional lightweight conv refinement.

This bypasses diffusion entirely to test whether JEPA embeddings carry
enough information for super-resolution. If this works at ~20+ dB PSNR,
we invest in a spatial diffusion head. If not, JEPA training needs rework.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentRegressor(nn.Module):
    """
    Per-token MLP: embed_dim → VAE latent block (4 channels × 2×2).

    Architecture:
        input:  (B, 256, 768)  JEPA context embeddings
        MLP:    768 → 512 → 512 → 16   (per-token, shared weights)
        reshape: (B, 256, 16) → (B, 4, 32, 32)
        output: (B, 4, 32, 32) VAE latent (scaled)

    ~2.1M params — trains in minutes, not hours.
    """

    def __init__(self, embed_dim=768, hidden_dim=512, latent_channels=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.latent_channels = latent_channels
        out_dim = latent_channels * 4  # 4 channels × 2×2 spatial block = 16

        self.norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, jepa_embeddings):
        """
        jepa_embeddings: (B, 256, embed_dim) — context encoder output
        Returns:         (B, latent_channels, 32, 32) — VAE latent (scaled)
        """
        B, N, D = jepa_embeddings.shape

        x = self.norm(jepa_embeddings)                 # (B, 256, 768)
        x = self.mlp(x)                                # (B, 256, 16)

        # Reshape: (B, 256, 16) → (B, 16, 16, 2, 2, 4) → (B, 4, 32, 32)
        x = x.reshape(B, 16, 16, 2, 2, self.latent_channels)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.reshape(B, self.latent_channels, 32, 32)

        return x


class LatentRegressorConv(nn.Module):
    """
    LatentRegressor with conv refinement for spatial consistency.

    Stage 1: Per-token MLP (same as LatentRegressor)
    Stage 2: Lightweight conv refinement (3× ResBlock on 16×16 feature map)

    ~2.5M params.
    """

    def __init__(self, embed_dim=768, hidden_dim=512, latent_channels=4,
                 refine_dim=64):
        super().__init__()
        self.embed_dim = embed_dim
        self.latent_channels = latent_channels
        out_dim = latent_channels * 4  # 16

        self.norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

        # Lightweight conv refinement on 16×16 feature grid
        self.proj_in = nn.Conv2d(out_dim, refine_dim, 3, padding=1)
        self.conv1 = nn.Conv2d(refine_dim, refine_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(refine_dim, refine_dim, 3, padding=1)
        self.proj_out = nn.Conv2d(refine_dim, out_dim, 3, padding=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        for m in [self.proj_in, self.conv1, self.conv2, self.proj_out]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, jepa_embeddings):
        B, N, D = jepa_embeddings.shape

        x = self.norm(jepa_embeddings)
        x = self.mlp(x)                                # (B, 256, 16)

        # Reshape to spatial: (B, 256, 16) → (B, 16, 16, 16)
        spatial = x.reshape(B, 16, 16, -1).permute(0, 3, 1, 2).contiguous()

        # Conv refinement with residual
        h = F.gelu(self.proj_in(spatial))
        h = h + F.gelu(self.conv1(h))                  # ResBlock 1
        h = h + F.gelu(self.conv2(h))                  # ResBlock 2
        refined = spatial + self.proj_out(h)           # Skip connection

        # Reshape to VAE latent: (B, 16, 16, 16) → (B, 4, 32, 32)
        refined = refined.permute(0, 2, 3, 1)          # (B, 16, 16, 16)
        refined = refined.reshape(B, 16, 16, 2, 2, self.latent_channels)
        refined = refined.permute(0, 5, 1, 3, 2, 4).contiguous()
        refined = refined.reshape(B, self.latent_channels, 32, 32)

        return refined
