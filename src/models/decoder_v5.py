"""UrbanJEPA v5 pixel decoder.

Rewrite of `PixelDecoderV4` after the v4 architecture deterministically cliff-
dived around step ~400 across every loss / LR / seed configuration we tried
on 2026-05-27. The diagnosis (see memory/feedback-v4-training-instability):

  1. `sigmoid(logit(low_res) + residual)` output suppresses gradients on
     pixels near 0 or 1 by a factor of ~1e-4. The decoder physically cannot
     learn dark or bright regions, so it distorts mid-tones while leaving
     boundaries unchanged and the conflict eventually explodes.

  2. Four cumulative `bilinear_up(attn(avg_pool(dec, 16)))` cross-attention
     stages inject low-frequency noise that fights the high-frequency
     super-resolution signal at every stage.

  3. `GroupNorm(1, channels)` (channel+spatial layernorm) right before a
     zero-init conv produces unstable per-input gradient scales.

v5 fixes (Phase 2 compatible — Phase 2 adds hierarchical JEPA loss + a
separate cross-attn refinement pass on top of decoder output; those don't
depend on the multi-stage cross-attn inside the decoder):

  - **Additive residual output** instead of sigmoid-of-logit.
    `out = (low_res + delta).clamp(0, 1)` with `delta` from a zero-init conv.
    Gradient is 1.0 for all in-range pixels (vs 1e-4 for sigmoid-saturated).
    Still identity at step 0 (delta=0).

  - **Single bottleneck cross-attention** with the deepest encoder layer.
    Upsample stages use simple U-Net-style skips: 1x1 projection of encoder
    tokens, bilinear-upsample to the stage's spatial resolution, concat with
    decoder feature, 1x1 conv to fuse. No avg-pool-then-bilinear-up round-
    trip, no per-stage attention overhead.

  - **GroupNorm(8, channels)** for the final norm — proper per-group norm
    with smoother gradient than GroupNorm(1, ...).
"""

from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoder_v4 import (
    tokens_to_grid,
    _SelfAttention2D,
    _CrossAttentionBridge,
    _ResBlock,
    _UpsampleStage,
)


class _UNetSkipBridge(nn.Module):
    """Projects encoder tokens to decoder-stage channels and spatial size,
    then concatenates with decoder feature and fuses via 1x1 conv.

    Encoder tokens are 16x16 spatial at `enc_dim` channels; decoder feature
    at this stage is `(B, dec_channels, H, W)` with H=W in {32, 64, 128, 256}.
    """

    ENCODER_GRID = 16

    def __init__(self, dec_channels: int, enc_dim: int = 768):
        super().__init__()
        self.proj = nn.Conv2d(enc_dim, dec_channels, kernel_size=1)
        # Concat-then-fuse: input has 2*dec_channels, output dec_channels.
        self.fuse = nn.Conv2d(dec_channels * 2, dec_channels, kernel_size=1)

    def forward(self, dec_feat: torch.Tensor,
                enc_tokens: torch.Tensor) -> torch.Tensor:
        B, C, H, W = dec_feat.shape
        # Tokens -> 16x16 grid -> 1x1 project -> bilinear-up to (H, W)
        enc_grid = enc_tokens.transpose(1, 2).reshape(
            B, -1, self.ENCODER_GRID, self.ENCODER_GRID)
        enc_proj = self.proj(enc_grid)
        if H != self.ENCODER_GRID or W != self.ENCODER_GRID:
            enc_proj = F.interpolate(
                enc_proj, size=(H, W), mode="bilinear", align_corners=False)
        merged = torch.cat([dec_feat, enc_proj], dim=1)
        return self.fuse(merged)


class PixelDecoderV5(nn.Module):
    """Inputs:
        low_res:       (B, 3, 256, 256), in [0, 1]
        pred_features: (B, 256, 768)
        ctx_features:  (B, 256, 768)
        multi_scale:   list of 4 x (B, 256, 768) — encoder layers [3, 6, 9, 12]

    Output dict:
        pred_image:     (B, 3, 256, 256), in [0, 1] via additive residual + clamp
        lr_clamp_rate:  scalar — fraction of LR pixels at the [0, 1] boundary
                        (kept for compat with the train-loop watchdog).
    """

    UPSAMPLE_CHANNELS = (512, 384, 256, 128)  # one per stage 16->32->64->128->256
    LR_EPS = 1e-4

    def __init__(self, jepa_dim: int = 768, base_dim: int = 768,
                 lr_stem_ch: int = 64, n_attn_blocks: int = 4,
                 attn_heads: int = 8, dropout: float = 0.1,
                 cross_attn_heads: int = 8, cross_attn_mlp_ratio: float = 2.0):
        super().__init__()
        # ----- JEPA bottleneck (16x16) -----
        self.jepa_norm = nn.LayerNorm(jepa_dim * 2)
        self.jepa_proj = nn.Sequential(
            nn.Conv2d(jepa_dim * 2, base_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_dim, base_dim, 3, padding=1),
        )

        # ----- LR stem: 256 -> 16 (four stride-2 convs) -----
        chs = [3, lr_stem_ch, lr_stem_ch * 2, lr_stem_ch * 4, base_dim]
        layers = []
        for cin, cout in zip(chs[:-1], chs[1:]):
            layers += [nn.Conv2d(cin, cout, 3, stride=2, padding=1), nn.GELU()]
        self.lr_stem = nn.Sequential(*layers)

        # ----- Bottleneck fusion (16x16 mix of jepa + lr_feat) -----
        self.fuse = nn.Sequential(
            nn.Conv2d(base_dim * 2, base_dim, 1),
            nn.GELU(),
            nn.Conv2d(base_dim, base_dim, 3, padding=1),
        )

        # ----- Self-attention at bottleneck -----
        self.attn_blocks = nn.ModuleList([
            _SelfAttention2D(base_dim, num_heads=attn_heads, dropout=dropout)
            for _ in range(n_attn_blocks)
        ])

        # ----- ONE cross-attention at bottleneck, with encoder layer 12 -----
        # v4 had per-stage cross-attn which compounded low-freq noise.
        # v5 keeps only this one (semantic-rich encoder bridge) and uses
        # cheaper U-Net skips at the upsample stages.
        self.bottleneck_cross = _CrossAttentionBridge(
            dec_channels=base_dim, enc_dim=jepa_dim,
            num_heads=cross_attn_heads, mlp_ratio=cross_attn_mlp_ratio,
            dropout=dropout,
        )

        # ----- Upsampling stages: 16 -> 32 -> 64 -> 128 -> 256 -----
        stage_channels = (base_dim,) + self.UPSAMPLE_CHANNELS
        self.up_stages = nn.ModuleList([
            _UpsampleStage(stage_channels[i], stage_channels[i + 1])
            for i in range(4)
        ])

        # ----- Per-stage U-Net skip bridges -----
        # stage 0 (32x32)  takes multi_scale[3] = encoder layer 12 (semantic)
        # stage 1 (64x64)  takes multi_scale[2] = encoder layer 9
        # stage 2 (128x128) takes multi_scale[1] = encoder layer 6
        # stage 3 (256x256) takes multi_scale[0] = encoder layer 3 (textural)
        self.stage_skips = nn.ModuleList([
            _UNetSkipBridge(dec_channels=self.UPSAMPLE_CHANNELS[i],
                            enc_dim=jepa_dim)
            for i in range(4)
        ])

        # ----- Output head: GroupNorm(8) -> zero-init conv -> additive residual -----
        # Linear gradient through (low_res + delta).clamp(0, 1) for all pixels
        # in [0, 1] — no sigmoid suppression at boundaries (bug #1 in v4).
        final_ch = self.UPSAMPLE_CHANNELS[-1]
        n_groups = 8 if final_ch % 8 == 0 else 1
        self.final_norm = nn.GroupNorm(n_groups, final_ch)
        self.final_conv = nn.Conv2d(final_ch, 3, 3, padding=1)
        nn.init.zeros_(self.final_conv.weight)
        nn.init.zeros_(self.final_conv.bias)

    def forward(self, low_res: torch.Tensor, pred_features: torch.Tensor,
                ctx_features: torch.Tensor,
                multi_scale: Sequence[torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert len(multi_scale) == 4, (
            f"expected 4 multi-scale features, got {len(multi_scale)}")

        # ---- JEPA bottleneck ----
        jepa = torch.cat([ctx_features, pred_features], dim=-1)  # (B, 256, 1536)
        jepa = self.jepa_norm(jepa)
        jepa = tokens_to_grid(jepa, side=16)
        jepa = self.jepa_proj(jepa)                              # (B, base, 16, 16)

        # ---- LR stem ----
        lr_feat = self.lr_stem(low_res)                          # (B, base, 16, 16)

        # ---- Fuse + self-attn ----
        x = self.fuse(torch.cat([jepa, lr_feat], dim=1))
        for blk in self.attn_blocks:
            x = blk(x)

        # ---- Single bottleneck cross-attention (decoder Q vs encoder layer 12) ----
        x = self.bottleneck_cross(x, multi_scale[3])             # (B, base, 16, 16)

        # ---- Upsample with U-Net skips ----
        # multi_scale order is [layer3, layer6, layer9, layer12] (shallow -> deep).
        # stage 0 takes layer 12 (multi_scale[3]); stage 3 takes layer 3 (multi_scale[0]).
        skip_layers = (multi_scale[3], multi_scale[2], multi_scale[1], multi_scale[0])
        for i, up in enumerate(self.up_stages):
            x = up(x)
            x = self.stage_skips[i](x, skip_layers[i])

        # ---- Tanh-bounded additive-residual output: low_res + tanh(delta) ----
        # tanh keeps the gradient alive EVERYWHERE, unlike the old hard clamp
        # which had a zero-gradient dead zone outside [0,1] (a pixel pushed out
        # could never recover). Bound R=1.0 so any in-range LR pixel can still
        # reach the opposite extreme (LR=0 -> ~1, LR=1 -> ~0). Still identity at
        # init: final_conv is zero-init -> delta=0 -> tanh(0)=0 -> out=low_res.
        delta = self.final_conv(self.final_norm(x))              # (B, 3, 256, 256)

        # fp32 for the LR-boundary metric; the residual add is fp32 too.
        lr_fp32 = low_res.float()
        clamp_mask = (lr_fp32 < self.LR_EPS) | (lr_fp32 > 1.0 - self.LR_EPS)
        lr_clamp_rate = clamp_mask.float().mean()

        out_unclamped = lr_fp32 + torch.tanh(delta.float())      # smooth, in (-1, 2)
        out = out_unclamped.clamp(0.0, 1.0)  # metrics/output safety net ONLY
        #                                      (tanh already provides the in-range
        #                                       gradient; clamp is no longer the
        #                                       learning path)

        # pred_unclamped is exposed ONLY for out_of_range_loss: the clamp above
        # has zero gradient outside [0,1], so a penalty on the clamped output is
        # always 0 and cannot pull delta back in range (the dead-gradient
        # collapse). The OOB loss must see this pre-clamp tensor.
        return {"pred_image": out, "pred_unclamped": out_unclamped,
                "lr_clamp_rate": lr_clamp_rate.detach(),
                # Collapse diagnostics: spread of the decoder output (near 0 =>
                # flat/degenerate) and fraction of pixels driven out of [0,1].
                "pred_unclamped_std": out_unclamped.detach().std(),
                "oob_frac": ((out_unclamped < 0.0) | (out_unclamped > 1.0))
                            .float().mean().detach()}


# ---------------- smoke test ----------------

def _smoke():
    import argparse
    import time
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)
    print(f"decoder_v5 smoke on {device}, batch={args.batch_size}")

    dec = PixelDecoderV5().to(device)
    n_params = sum(p.numel() for p in dec.parameters()) / 1e6
    print(f"  params: {n_params:.1f} M")

    B = args.batch_size
    low_res = torch.rand(B, 3, 256, 256, device=device)
    pred_f = torch.randn(B, 256, 768, device=device)
    ctx_f = torch.randn(B, 256, 768, device=device)
    multi = [torch.randn(B, 256, 768, device=device) for _ in range(4)]

    t0 = time.time()
    out = dec(low_res, pred_f, ctx_f, multi)
    fwd_ms = (time.time() - t0) * 1000
    print(f"  forward time: {fwd_ms:.1f} ms")
    print(f"  pred_image: shape={tuple(out['pred_image'].shape)} "
          f"min={out['pred_image'].min():.4f} max={out['pred_image'].max():.4f}")
    print(f"  lr_clamp_rate: {out['lr_clamp_rate'].item():.6f}")

    # Identity check at zero-init (delta should be 0 -> out should equal low_res)
    err = (out["pred_image"] - low_res.float()).abs().max()
    print(f"  identity err at step 0 (should be ~0): {err.item():.6f}")
    assert err < 1e-5, f"Output is not identity at init: max err {err}"
    print("  ✓ identity at init confirmed")

    # Gradient flow check: backward + look at delta gradient magnitude
    target = torch.rand_like(low_res)
    loss = F.l1_loss(out["pred_image"], target)
    loss.backward()
    final_conv_grad_norm = dec.final_conv.weight.grad.norm().item()
    print(f"  final_conv grad norm after one backward: {final_conv_grad_norm:.6e}")
    assert final_conv_grad_norm > 0, "No gradient reached final_conv!"
    print("  ✓ gradient flow through additive-residual output confirmed")


if __name__ == "__main__":
    _smoke()
