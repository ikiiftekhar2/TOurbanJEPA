"""
CNN Decoder for Phase 3 JEPA embedding validation.

Decodes ViT-B/16 target encoder embeddings (256 tokens, 768-dim)
back to RGB images (256×256). Trained with frozen JEPA backbone
to verify that learned representations preserve spatial information.

Architecture:
    (B, 256, 768) → reshape → (B, 768, 16, 16)
    → 1×1 proj → (B, 512, 16, 16)
    → upsample ×4 → (B, 32, 256, 256)
    → head → (B, 3, 256, 256)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UpsampleBlock(nn.Module):
    """Bilinear upsample (×2) + Conv + BatchNorm + ReLU."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.conv(x)


class CNNDecoder(nn.Module):
    """
    Decodes ViT token embeddings to RGB images.

    Input:  (B, N, embed_dim) where N = 256, embed_dim = 768
    Output: (B, 3, 256, 256) in [0, 1]
    """

    def __init__(self, embed_dim=768, hidden_dims=(512, 256, 128, 64, 32)):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_tokens = 256
        self.spatial_size = 16  # sqrt(256)

        # Project embed_dim → first hidden dimension
        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, hidden_dims[0], 1),
            nn.BatchNorm2d(hidden_dims[0]),
            nn.ReLU(inplace=True),
        )

        # Progressive upsampling: 16→32→64→128→256
        self.blocks = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.blocks.append(UpsampleBlock(hidden_dims[i], hidden_dims[i + 1]))

        # Output head
        self.head = nn.Sequential(
            nn.Conv2d(hidden_dims[-1], 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 1),
            nn.Sigmoid(),
        )

    def forward(self, tokens):
        """
        tokens: (B, N, D) ViT patch embeddings
        returns: (B, 3, 256, 256) RGB image in [0, 1]
        """
        B, N, D = tokens.shape
        assert N == self.num_tokens, f"Expected {self.num_tokens} tokens, got {N}"
        assert D == self.embed_dim, f"Expected embed_dim={self.embed_dim}, got {D}"

        x = tokens.transpose(1, 2).reshape(B, D, self.spatial_size, self.spatial_size)
        x = self.proj(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


def compute_psnr(pred, target):
    """PSNR for [0,1] normalized images. Higher is better."""
    mse = F.mse_loss(pred, target, reduction="mean")
    if mse == 0:
        return float("inf")
    return 10.0 * torch.log10(1.0 / mse).item()
