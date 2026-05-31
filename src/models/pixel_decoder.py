"""
PixelDecoderWithSkips — JEPA features + LR image + multi-scale encoder skips
→ 256x256 pixel residual added to the bilinear low_res.

Zero-init final conv: at step 0, output = low_res (identity).

The residual is squashed through tanh(.) * RESIDUAL_RANGE so the output is
bounded to [low_res - 0.5, low_res + 0.5]. We DO NOT hard-clamp to [0, 1]
inside forward — clamp's dead-zone kills gradients in saturated pixels and
caused the v3 imagenet run to collapse at step ~7800 (output saturated to
0/1, gradient stopped flowing through the clamp, no recovery possible).
Callers should clamp at inference / for metric reporting; train losses run
on the unclamped output so gradient always flows.
"""

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def tokens_to_grid(tokens: torch.Tensor, side: int = 16) -> torch.Tensor:
    """(B, N, D) → (B, D, side, side) where N = side*side."""
    B, N, D = tokens.shape
    assert N == side * side, f"expected {side*side} tokens, got {N}"
    return tokens.reshape(B, side, side, D).permute(0, 3, 1, 2).contiguous()


class _SelfAttention2D(nn.Module):
    def __init__(self, channels: int, num_heads: int = 8, mlp_ratio: float = 4.0,
                 dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(channels)
        hidden = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, channels),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # (B, HW, C)
        h = self.norm1(tokens)
        a, _ = self.attn(h, h, h, need_weights=False)
        tokens = tokens + a
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens.transpose(1, 2).reshape(B, C, H, W)


class _ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.GELU()

    def forward(self, x):
        h = self.act(self.norm(x))
        h = self.act(self.conv1(h))
        h = self.conv2(h)
        return x + h


class _UpsampleStage(nn.Module):
    """PixelShuffle(2) + residual conv stack. Channels: in_ch → out_ch."""

    def __init__(self, in_ch: int, out_ch: int, n_resblocks: int = 2):
        super().__init__()
        self.pre = nn.Conv2d(in_ch, out_ch * 4, 3, padding=1)
        self.shuffle = nn.PixelShuffle(2)
        self.blocks = nn.Sequential(*[_ResBlock(out_ch) for _ in range(n_resblocks)])

    def forward(self, x):
        x = self.pre(x)
        x = self.shuffle(x)
        x = self.blocks(x)
        return x


class _SkipProjection(nn.Module):
    """Reshape (B, 256, 768) token sequence → (B, target_ch, target_side, target_side).

    Used to project multi-scale encoder features into a spatial map matching the
    decoder's feature map at each upsampling stage, so they can be added in.
    """

    def __init__(self, in_dim: int, out_ch: int, target_side: int, src_side: int = 16):
        super().__init__()
        self.target_side = target_side
        self.src_side = src_side
        self.proj = nn.Conv2d(in_dim, out_ch, 1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # Project channels at low resolution first (768 -> out_ch is much cheaper
        # at 16x16 than at 256x256), then upsample. For the 256x256 skip stage
        # this avoids a (B, 768, 256, 256) intermediate (~2 GB at B=22).
        x = tokens_to_grid(tokens, side=self.src_side)
        x = self.proj(x)
        if self.target_side != self.src_side:
            x = F.interpolate(x, size=(self.target_side, self.target_side),
                              mode="bilinear", align_corners=False)
        return x


class PixelDecoderWithSkips(nn.Module):
    """
    Inputs:
      low_res:       (B, 3, 256, 256)
      pred_features: (B, 256, 768)
      ctx_features:  (B, 256, 768)
      multi_scale:   list of 4 × (B, 256, 768) — from encoder layers [3, 6, 9, 12]

    Output:
      pred_image:    (B, 3, 256, 256), clamped [0, 1]
    """

    UPSAMPLE_CHANNELS = (512, 384, 256, 128)  # one per stage 16→32→64→128→256
    # RESIDUAL_RANGE is now a runtime attribute set per-step from the train
    # loop via set_residual_range(). Starts at the conservative cap of 0.05
    # and ramps up as the model proves stability. Forensic analysis (2026-05-30)
    # showed the decoder learns a content-independent NEGATIVE bias that grows
    # exponentially when given a large residual budget; we cap the budget early
    # and expand it as the upstream features become useful. See `set_residual_range`.
    RESIDUAL_RANGE = 0.05  # default; overridden by set_residual_range each step

    def __init__(self, jepa_dim: int = 768, base_dim: int = 768,
                 lr_stem_ch: int = 64, n_attn_blocks: int = 4,
                 attn_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.jepa_norm = nn.LayerNorm(jepa_dim * 2)
        self.jepa_proj = nn.Sequential(
            nn.Conv2d(jepa_dim * 2, base_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_dim, base_dim, 3, padding=1),
        )

        # LR stem: 256 → 128 → 64 → 32 → 16 (four stride-2 stages)
        chs = [3, lr_stem_ch, lr_stem_ch * 2, lr_stem_ch * 4, base_dim]
        layers = []
        for cin, cout in zip(chs[:-1], chs[1:]):
            layers += [nn.Conv2d(cin, cout, 3, stride=2, padding=1), nn.GELU()]
        self.lr_stem = nn.Sequential(*layers)

        self.fuse = nn.Sequential(
            nn.Conv2d(base_dim * 2, base_dim, 1),
            nn.GELU(),
            nn.Conv2d(base_dim, base_dim, 3, padding=1),
        )

        self.attn_blocks = nn.ModuleList([
            _SelfAttention2D(base_dim, num_heads=attn_heads, dropout=dropout)
            for _ in range(n_attn_blocks)
        ])

        stage_channels = (base_dim,) + self.UPSAMPLE_CHANNELS
        # stage_channels[i] → stage_channels[i+1], spatial 16 * 2^i → 16 * 2^(i+1)
        self.up_stages = nn.ModuleList([
            _UpsampleStage(stage_channels[i], stage_channels[i + 1])
            for i in range(4)
        ])

        # Skip projections: encoder layers [3, 6, 9, 12] → stages [4, 3, 2, 1]
        # multi_scale[3] (layer 12) -> stage 1 (32x32, 512 ch)
        # multi_scale[2] (layer 9 ) -> stage 2 (64x64, 384 ch)
        # multi_scale[1] (layer 6 ) -> stage 3 (128x128, 256 ch)
        # multi_scale[0] (layer 3 ) -> stage 4 (256x256, 128 ch)
        target_sides = (32, 64, 128, 256)
        self.skip_projs = nn.ModuleList([
            _SkipProjection(jepa_dim, self.UPSAMPLE_CHANNELS[i],
                            target_side=target_sides[i])
            for i in range(4)
        ])

        final_ch = self.UPSAMPLE_CHANNELS[-1]
        self.final_norm = nn.GroupNorm(1, final_ch)
        self.final_conv = nn.Conv2d(final_ch, 3, 3, padding=1)
        nn.init.zeros_(self.final_conv.weight)
        nn.init.zeros_(self.final_conv.bias)

    def set_residual_range(self, r: float) -> None:
        """Set the current ±R cap on the tanh-bounded output residual.
        Called per-step from train.py to ramp R as training stabilises."""
        self.RESIDUAL_RANGE = float(r)

    def forward(self, low_res: torch.Tensor, pred_features: torch.Tensor,
                ctx_features: torch.Tensor, multi_scale: Sequence[torch.Tensor]) -> torch.Tensor:
        assert len(multi_scale) == 4, f"expected 4 multi-scale features, got {len(multi_scale)}"

        # JEPA branch
        jepa = torch.cat([ctx_features, pred_features], dim=-1)  # (B, 256, 1536)
        jepa = self.jepa_norm(jepa)
        jepa = tokens_to_grid(jepa, side=16)  # (B, 1536, 16, 16)
        jepa = self.jepa_proj(jepa)            # (B, base_dim, 16, 16)

        # LR branch
        lr_feat = self.lr_stem(low_res)        # (B, base_dim, 16, 16)

        x = self.fuse(torch.cat([jepa, lr_feat], dim=1))

        for blk in self.attn_blocks:
            x = blk(x)

        # multi_scale order is [layer3, layer6, layer9, layer12] (shallow→deep).
        # We want to inject the deepest skip first (matches semantic 32x32 stage),
        # then shallower skips at higher-res stages.
        skip_order = (multi_scale[3], multi_scale[2], multi_scale[1], multi_scale[0])

        for i, up in enumerate(self.up_stages):
            x = up(x)
            skip = self.skip_projs[i](skip_order[i])
            x = x + skip

        raw_residual = self.final_conv(self.final_norm(x))
        # tanh bounds residual to (-RESIDUAL_RANGE, RESIDUAL_RANGE) so output stays
        # within (low_res - 0.5, low_res + 0.5). Gradient never saturates the way a
        # hard clamp would. NO clamp on the output — that's the caller's job.
        bounded = torch.tanh(raw_residual) * self.RESIDUAL_RANGE
        return low_res + bounded
