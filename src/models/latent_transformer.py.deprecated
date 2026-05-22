"""
Cross-Attention Latent Transformer for Phase 2.

Takes low-res VAE latent tokens + JEPA embeddings, outputs high-res latent tokens
via 8 decoder layers with self-attention + cross-attention + FFN.

Key design:
- Zero-init output projection: at step 0, output = lr_tokens (residual starts at 0)
- d_model=768 matches JEPA dim: cross-attention path is zero-loss at init
- lr_tokens residual: low-res latent already contains coarse structure
"""

import torch
import torch.nn as nn


class DecoderLayer(nn.Module):
    """Pre-norm transformer decoder layer: self-attn -> cross-attn -> FFN."""

    def __init__(self, d_model, n_heads, ffn_dim, dropout=0.1):
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


class LatentCrossAttentionTransformer(nn.Module):
    """
    Phase 2: JEPA-conditioned latent-space transformer.

    Inputs:
        lr_tokens:  (B, 256, 16)  grouped low-res VAE latent
        ctx_emb:    (B, 256, 768) JEPA context encoder output
        pred_emb:   (B, 256, 768) JEPA predictor output (all 256 tokens)

    Output:
        hr_tokens:  (B, 256, 16)  predicted high-res VAE latent (grouped)

    ~50M params.
    """

    def __init__(
        self,
        token_dim=16,
        jepa_dim=768,
        d_model=768,
        n_layers=8,
        n_heads=12,
        ffn_ratio=4.0,
        n_tokens=256,
        dropout=0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_tokens = n_tokens

        # Project lr_tokens (16-dim) -> d_model (768-dim)
        self.input_proj = nn.Linear(token_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, n_tokens, d_model) * 0.02)

        # Project JEPA embeddings to kv space
        self.ctx_kv_proj = nn.Linear(jepa_dim, d_model)
        self.pred_kv_proj = nn.Linear(jepa_dim, d_model)

        # Type embeddings to distinguish context vs predicted in the kv sequence
        self.ctx_type_embed = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pred_type_embed = nn.Parameter(torch.zeros(1, 1, d_model))

        # Transformer decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, int(d_model * ffn_ratio), dropout)
            for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, token_dim)

        # Zero-init: residual starts at zero -> output = lr_tokens at init
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, lr_tokens, ctx_emb, pred_emb):
        B = lr_tokens.shape[0]

        queries = self.input_proj(lr_tokens) + self.pos_embed  # (B, 256, 768)

        ctx_kv = self.ctx_kv_proj(ctx_emb) + self.ctx_type_embed
        pred_kv = self.pred_kv_proj(pred_emb) + self.pred_type_embed
        kv = torch.cat([ctx_kv, pred_kv], dim=1)  # (B, 512, 768)

        x = queries
        for layer in self.layers:
            x = layer(x, kv)

        x = self.final_norm(x)
        residual = self.output_proj(x)  # (B, 256, 16)

        return lr_tokens + residual
