"""
JEPA → Pixel Decoder. No VAE, no latent space.

Takes JEPA features (ctx_emb + pred_emb, 256 patch tokens each at 768-dim)
and the low-res image, outputs a 256x256 RGB image directly.

Architecture:
  low_res ──→ conv stem ──→ lr_feat (B, 256, 16, 16)
  ctx_emb  ──→ reshape ──→ (B, 768, 16, 16)
  pred_emb ──→ reshape ──→ (B, 768, 16, 16)
  concat → project → (B, 768, 16, 16)
  concat with lr_feat → (B, 768+256, 16, 16) → fuse → (B, 768, 16, 16)
  6× Self-Attention on 16×16 grid (256 tokens, global receptive field)
  Progressive upsampling: 16→32→64→128→256 (768→384→256→192→128)
  Zero-init final conv → output = low_res + residual
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttentionBlock(nn.Module):
    """Self-attention on spatial grid with pre-norm and residual connection."""

    def __init__(self, dim, n_heads=12, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: (B, dim, H, W)
        B, C, H, W = x.shape
        t = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        h = self.norm1(t)
        t = t + self.attn(h, h, h, need_weights=False)[0]
        h = self.norm2(t)
        t = t + self.mlp(h)
        return t.transpose(1, 2).reshape(B, C, H, W)


class UpsampleBlock(nn.Module):
    """PixelShuffle upsample + two residual conv blocks."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Sequential(
            nn.Conv2d(in_ch, out_ch * 4, 3, padding=1),
            nn.PixelShuffle(2),  # 2× upscale
        )
        self.block = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.up(x)
        return self.block(x) + x  # residual within block


class JEPAPixelDecoder(nn.Module):
    """
    Direct pixel-space decoder. No VAE bottleneck.

    Inputs:
      low_res:  (B, 3, 256, 256)   low-resolution image
      ctx_emb:  (B, 256, 768)      JEPA context encoder output
      pred_emb: (B, 256, 768)      JEPA predictor output

    Output:
      (B, 3, 256, 256)             predicted high-res image
    """

    def __init__(
        self,
        jepa_dim=768,
        lr_stem_ch=64,
        base_dim=768,
        n_attn_blocks=6,
        n_heads=12,
        dropout=0.1,
    ):
        super().__init__()
        self.base_dim = base_dim

        # Low-res image → feature stem
        self.lr_stem = nn.Sequential(
            nn.Conv2d(3, lr_stem_ch, 4, stride=2, padding=1),  # 256→128
            nn.ReLU(inplace=True),
            nn.Conv2d(lr_stem_ch, lr_stem_ch * 2, 4, stride=2, padding=1),  # 128→64
            nn.ReLU(inplace=True),
            nn.Conv2d(lr_stem_ch * 2, lr_stem_ch * 4, 4, stride=2, padding=1),  # 64→32
            nn.ReLU(inplace=True),
            nn.Conv2d(lr_stem_ch * 4, lr_stem_ch * 4, 4, stride=2, padding=1),  # 32→16
            nn.ReLU(inplace=True),
        )
        lr_feat_ch = lr_stem_ch * 4

        # JEPA project: concat ctx+pred, project to base_dim
        self.jepa_proj = nn.Sequential(
            nn.Conv2d(jepa_dim * 2, base_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_dim, base_dim, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Fuse JEPA + LR features
        self.fuse = nn.Conv2d(base_dim + lr_feat_ch, base_dim, 3, padding=1)

        # Self-attention on 16×16 grid
        self.attn_blocks = nn.ModuleList([
            SelfAttentionBlock(base_dim, n_heads, dropout)
            for _ in range(n_attn_blocks)
        ])

        # Progressive upsampling: 16→32→64→128→256
        self.up1 = UpsampleBlock(base_dim, 512)      # 16→32,  768→512
        self.up2 = UpsampleBlock(512, 384)           # 32→64,  512→384
        self.up3 = UpsampleBlock(384, 256)           # 64→128, 384→256
        self.up4 = UpsampleBlock(256, 128)           # 128→256, 256→128
        final_ch = 128

        self.final_norm = nn.GroupNorm(8, final_ch)
        self.final_conv = nn.Conv2d(final_ch, 3, 3, padding=1)

        # Zero-init: at step 0, output = low_res (identity)
        nn.init.zeros_(self.final_conv.weight)
        nn.init.zeros_(self.final_conv.bias)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m is not self.final_conv:
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, low_res, ctx_emb, pred_emb):
        """
        Args:
          low_res:  (B, 3, 256, 256)
          ctx_emb:  (B, 256, 768)
          pred_emb: (B, 256, 768)
        Returns:
          output: (B, 3, 256, 256) clamped to [0, 1]
        """
        B = low_res.shape[0]

        # Low-res features: (B, 3, 256, 256) → (B, lr_feat_ch, 16, 16)
        lr_feat = self.lr_stem(low_res)

        # JEPA features: (B, 256, 768) → (B, 768, 16, 16)
        ctx_spatial = ctx_emb.reshape(B, 16, 16, 768).permute(0, 3, 1, 2)
        pred_spatial = pred_emb.reshape(B, 16, 16, 768).permute(0, 3, 1, 2)
        jepa_feat = torch.cat([ctx_spatial, pred_spatial], dim=1)  # (B, 1536, 16, 16)

        # Project JEPA: (B, 1536, 16, 16) → (B, base_dim, 16, 16)
        jepa_proj = self.jepa_proj(jepa_feat)

        # Fuse: concat(JEPA, LR) → (B, base_dim, 16, 16)
        x = self.fuse(torch.cat([jepa_proj, lr_feat], dim=1))

        # Self-attention on 16×16 grid
        for blk in self.attn_blocks:
            x = x + blk(x)  # residual around attention block

        # Progressive upsampling
        x = self.up1(x)  # 16→32
        x = self.up2(x)  # 32→64
        x = self.up3(x)  # 64→128
        x = self.up4(x)  # 128→256

        # Final output: zero-init → residual starts at zero
        x = self.final_norm(x)
        residual = self.final_conv(x)

        return (low_res + residual).clamp(0, 1)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    # Smoke test
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = JEPAPixelDecoder().to(device)
    print(f"Params: {count_params(model):,}")

    B = 4
    low_res = torch.randn(B, 3, 256, 256, device=device)
    ctx_emb = torch.randn(B, 256, 768, device=device)
    pred_emb = torch.randn(B, 256, 768, device=device)

    with torch.no_grad():
        out = model(low_res, ctx_emb, pred_emb)
    print(f"Input:  {low_res.shape} → Output: {out.shape}")
    print(f"Output range: [{out.min().item():.3f}, {out.max().item():.3f}]")

    # Verify zero-init: with zero final_conv, output should ≈ low_res
    diff = (out - low_res).abs().max().item()
    print(f"Max |output - low_res|: {diff:.6f} (should be ~0 due to zero-init)")
