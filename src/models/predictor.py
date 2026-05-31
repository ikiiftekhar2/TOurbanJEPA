"""
FeaturePredictor — 6-layer transformer that maps context features (from LR)
to predicted target features (approximating encoder(HR)).

Random init, learnable positional embeddings, predicts all 256 tokens (no mask).
"""

import torch
import torch.nn as nn


class _TransformerBlock(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x


class FeaturePredictor(nn.Module):
    def __init__(self, embed_dim: int = 768, depth: int = 6, num_heads: int = 12,
                 mlp_ratio: float = 4.0, dropout: float = 0.1,
                 num_tokens: int = 256):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList([
            _TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.MultiheadAttention):
            # PyTorch's default xavier_uniform on in_proj_weight is louder than
            # the trunc_normal(0.02) convention used everywhere else in this
            # model — keep init consistent.
            if m.in_proj_weight is not None:
                nn.init.trunc_normal_(m.in_proj_weight, std=0.02)
            if m.in_proj_bias is not None:
                nn.init.zeros_(m.in_proj_bias)
            nn.init.trunc_normal_(m.out_proj.weight, std=0.02)
            if m.out_proj.bias is not None:
                nn.init.zeros_(m.out_proj.bias)

    def forward(self, context_features: torch.Tensor) -> torch.Tensor:
        x = context_features + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)
