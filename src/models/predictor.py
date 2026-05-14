"""
Feature Predictor (γ) — predicts high-res embeddings for masked token positions
from the context encoder's output.

ViT-B/16 architecture matching the encoder. Takes context features + positional
queries for masked positions. Initialized from ImageNet-1K ViT-B/16 weights.
"""

import torch
import torch.nn as nn
from .encoder import build_vit_base


class FeaturePredictor(nn.Module):
    """
    ViT-B predictor that predicts embeddings for masked target token positions.

    Key difference from encoder: takes positional queries for masked positions
    as additional input, so it knows WHERE to predict, not just WHAT.
    """

    def __init__(self, pretrained_path=None, embed_dim=768, depth=12,
                 num_heads=12, max_tokens=1024):
        super().__init__()

        self.embed_dim = embed_dim

        self.transformer = build_vit_base(img_size=256, patch_size=16)

        if pretrained_path:
            self._load_pretrained(pretrained_path)

        # Learnable mask token (query for each masked position)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # Positional embeddings for masked token positions
        self.pos_embed = nn.Parameter(torch.zeros(1, max_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _load_pretrained(self, path):
        checkpoint = torch.load(path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        missing, unexpected = self.transformer.load_state_dict(state_dict, strict=False)
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
        mask_pos = self.pos_embed[:, mask_positions, :]       # (B, N_mask, D)
        queries = mask_tokens + mask_pos

        # Concatenate context features and mask queries, run transformer
        x = torch.cat([context_features, queries], dim=1)     # (B, N_ctx+N_mask, D)
        x = self.transformer.forward_features(x)
        x = x[:, 1:, :]                                        # remove cls token

        # Return only the predicted (masked) positions
        return x[:, -N_mask:, :]
