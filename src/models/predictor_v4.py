"""
HierarchicalFeaturePredictor — 8-layer transformer trunk + 4 per-target-layer
projection heads (Reed et al., Scale-MAE 2023, §3.2). Predicts the target
encoder's outputs at layers {3, 6, 9, 12} simultaneously, in feature space.

v3 predictor only predicted the final layer. v4 predicts all 4 hierarchical
JEPA targets, letting the shared encoder learn richer multi-scale
representations (the loss per layer is small individually — 0.025 — but
collectively constrains the encoder geometry across depth).

Output:
    {
        "layers": [t3_pred, t6_pred, t9_pred, t12_pred],  # each (B, 256, D)
        "final":  t12_pred,  # convenience alias for decoder input
    }

Loss recipe (applied externally in losses.py):
    L_jepa = sum_i 0.025 * (0.25 * SmoothL1(layer_i, target_i.detach())
                          + 0.05 * (1 - cos(layer_i, target_i.detach())))
"""

from typing import Dict, List

import torch
import torch.nn as nn


class _PredictorBlock(nn.Module):
    """Standard pre-norm transformer block, matches the v3 _TransformerBlock."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout,
                                          batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x


class _LayerHead(nn.Module):
    """LayerNorm + Linear head, one per hierarchical target layer."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(x))


class HierarchicalFeaturePredictor(nn.Module):
    """8-layer predictor with 4 per-layer heads (predicting target layers 3/6/9/12).

    Args:
        embed_dim: feature dim (768 for ViT-B).
        depth: number of transformer blocks (8 per V4_PLAN §2.2).
        num_heads: attention heads (12 for ViT-B).
        mlp_ratio: MLP expansion ratio (4.0).
        dropout: dropout rate.
        num_tokens: spatial token count (256 for 16x16).
        n_targets: how many target-layer heads to instantiate (4).
    """

    def __init__(self, embed_dim: int = 768, depth: int = 8, num_heads: int = 12,
                 mlp_ratio: float = 4.0, dropout: float = 0.1,
                 num_tokens: int = 256, n_targets: int = 4):
        super().__init__()
        self.n_targets = n_targets
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList([
            _PredictorBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.trunk_norm = nn.LayerNorm(embed_dim)
        self.heads = nn.ModuleList([_LayerHead(embed_dim) for _ in range(n_targets)])
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.MultiheadAttention):
            if m.in_proj_weight is not None:
                nn.init.trunc_normal_(m.in_proj_weight, std=0.02)
            if m.in_proj_bias is not None:
                nn.init.zeros_(m.in_proj_bias)
            nn.init.trunc_normal_(m.out_proj.weight, std=0.02)
            if m.out_proj.bias is not None:
                nn.init.zeros_(m.out_proj.bias)

    def forward(self, context_features: torch.Tensor) -> Dict[str, object]:
        x = context_features + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.trunk_norm(x)
        layers: List[torch.Tensor] = [head(x) for head in self.heads]
        return {"layers": layers, "final": layers[-1]}


if __name__ == "__main__":
    # smoke
    pred = HierarchicalFeaturePredictor()
    x = torch.randn(2, 256, 768)
    out = pred(x)
    assert isinstance(out, dict) and "layers" in out and "final" in out
    assert len(out["layers"]) == 4
    for i, t in enumerate(out["layers"]):
        assert tuple(t.shape) == (2, 256, 768), f"layer {i}: {t.shape}"
    assert torch.equal(out["final"], out["layers"][-1])
    n_params = sum(p.numel() for p in pred.parameters()) / 1e6
    print(f"predictor_v4 smoke OK — 4 heads × (2,256,768), {n_params:.1f}M params")
