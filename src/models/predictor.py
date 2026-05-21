"""
Feature Predictor (γ) — predicts high-res embeddings for masked token positions
from the context encoder's output.

Uses the transformer blocks from a ViT-B/16 (pretrained weights) but operates on
token sequences directly — no patch embedding or CLS token. Takes context features
+ positional queries for masked positions, runs through transformer blocks, and
returns predicted embeddings for masked positions.

Token space: 256 tokens (16×16 grid), matching both ViT patch grid and grouped
VAE latent grid (2×2 spatial groups of 4-channel latents = 16-dim).
"""

import torch
import torch.nn as nn
from timm.models.layers import DropPath
from .encoder import build_vit_base


class FeaturePredictor(nn.Module):
    """
    Transformer predictor for JEPA masked token prediction.

    Uses ViT-B/16 transformer blocks pretrained on ImageNet-1K, but bypasses
    the image-specific frontend (patch_embed, cls_token, pos_embed).
    """

    def __init__(self, pretrained_path=None, embed_dim=768, depth=12,
                 num_heads=12, max_tokens=256, drop_path_rate=0.0):
        super().__init__()

        self.embed_dim = embed_dim

        # Build a full ViT just to extract its transformer blocks
        vit = build_vit_base(img_size=256, patch_size=16)

        if pretrained_path:
            self._load_pretrained_blocks(vit, pretrained_path)

        if drop_path_rate > 0.0:
            for i, block in enumerate(vit.blocks):
                block.drop_path = DropPath(drop_path_rate * i / (depth - 1))

        self.blocks = vit.blocks      # transformer encoder blocks (ModuleList)
        self.norm = vit.norm          # final LayerNorm
        self.pos_drop = vit.pos_drop  # dropout after pos_embed

        # Learnable mask token (query for each masked position)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # Positional embeddings for masked token positions
        self.pos_embed = nn.Parameter(torch.zeros(1, max_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _load_pretrained_blocks(self, vit, path):
        """Load pretrained weights into the ViT, then keep only the blocks and norm."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        state_dict = checkpoint.get("state_dict", checkpoint)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        # Interpolate pos_embed for the full ViT (needed for load_state_dict)
        if "pos_embed" in state_dict:
            src_pe = state_dict["pos_embed"]
            tgt_pe = vit.pos_embed.data
            if src_pe.shape != tgt_pe.shape:
                src_cls = src_pe[:, :1, :]
                src_patches = src_pe[:, 1:, :]
                tgt_cls = tgt_pe[:, :1, :]
                src_grid = int(src_patches.shape[1] ** 0.5)
                tgt_grid = int((tgt_pe.shape[1] - 1) ** 0.5)
                src_patches = src_patches.reshape(1, src_grid, src_grid, -1)
                src_patches = src_patches.permute(0, 3, 1, 2)
                interpolated = torch.nn.functional.interpolate(
                    src_patches, size=(tgt_grid, tgt_grid),
                    mode="bicubic", align_corners=False,
                )
                interpolated = interpolated.permute(0, 2, 3, 1)
                interpolated = interpolated.reshape(1, tgt_grid * tgt_grid, -1)
                state_dict["pos_embed"] = torch.cat([tgt_cls, interpolated], dim=1)

        missing, unexpected = vit.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  Predictor: {len(missing)} missing keys (expected)")
        if unexpected:
            print(f"  Predictor: {len(unexpected)} unexpected keys")

    def forward(self, context_features, mask_positions):
        """
        context_features: (B, N_ctx, D) features from context encoder
        mask_positions: (B, N_mask) indices of masked token positions (0..max_tokens-1)
        Returns: (B, N_mask, D) predicted embeddings for masked token positions
        """
        B, N_mask = mask_positions.shape

        # Create mask token queries with positional information
        mask_tokens = self.mask_token.expand(B, N_mask, -1)  # (B, N_mask, D)
        mask_pos = self.pos_embed[0, mask_positions, :]       # (B, N_mask, D)
        queries = mask_tokens + mask_pos

        # Concatenate context features and mask queries along sequence dimension
        x = torch.cat([context_features, queries], dim=1)     # (B, N_ctx+N_mask, D)

        # Run through transformer blocks (skip patch_embed, cls_token)
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        # Return only the predicted (masked) positions
        return x[:, -N_mask:, :]
