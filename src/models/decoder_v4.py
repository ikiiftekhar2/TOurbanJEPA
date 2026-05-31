"""
PixelDecoderV4 — v3's PixelDecoderWithSkips, upgraded per V4_PLAN §2.3.

Two architectural changes vs v3:

1. **Cross-attention bridge to encoder (Perceiver-IO pattern,
   Jaegle et al. 2021, `arXiv:2107.14795`).** Each decoder upsampling stage
   cross-attends to the matching encoder multi-scale layer. v3 used 1x1-conv
   skip projections that *additively* mixed encoder tokens after spatial
   alignment; v4 uses attention so the decoder can pick which encoder
   tokens to read per spatial location, decoupled from a hard alignment.

   To keep memory tractable at the 256x256 output stage (a naive per-pixel
   Q against 256 encoder tokens explodes to 65k×256×heads attention
   matrices), we downsample the decoder query to the encoder-native 16x16
   grid before each cross-attention, then bilinear-upsample the attended
   output back to the decoder's spatial resolution. This keeps cross-attn
   cost uniform (~256 × 256 × C/heads × heads) at every stage while still
   reading from full-resolution encoder K/V. Documented deviation from
   the plan's literal "decoder spatial Q vs encoder K/V" wording —
   memory-driven, behavior is the same in expectation.

2. **Sigmoid output (Ledig SRGAN 2017 §3.1, Wang ESRGAN 2018 §3.1).**
   v3's decoder emitted `low_res + tanh(r) * 0.5` — a soft-clamp that kept
   gradients flowing but artificially bounded the residual to ±0.5. v4 emits
   `sigmoid(logit(clamp(low_res, eps, 1-eps)) + r)`:
     - `r` is unbounded (no tanh ceiling)
     - sigmoid keeps output in (0, 1) without dead-zone gradient pathology
     - eps=1e-4 clamp on LR before logit avoids `log(0)` NaN
   Plus: we expose `lr_clamp_rate` for the train-loop watchdog (V4_PLAN
   §2.3 5%/100-step abort).
"""

from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def tokens_to_grid(tokens: torch.Tensor, side: int = 16) -> torch.Tensor:
    """(B, N, D) -> (B, D, side, side)."""
    B, N, D = tokens.shape
    assert N == side * side, f"expected {side*side} tokens, got {N}"
    return tokens.reshape(B, side, side, D).permute(0, 3, 1, 2).contiguous()


class _SelfAttention2D(nn.Module):
    """Pre-norm self-attention block on a (B, C, H, W) feature map."""

    def __init__(self, channels: int, num_heads: int = 8, mlp_ratio: float = 4.0,
                 dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        # dropout=0.0 inside MHA — under bf16, dropout-zeroed softmax weights
        # don't renormalise cleanly and introduce a low-frequency bias. Keep
        # dropout on the MLP residual where it doesn't interact with softmax.
        self.attn = nn.MultiheadAttention(channels, num_heads, dropout=0.0,
                                          batch_first=True)
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
        tokens = x.flatten(2).transpose(1, 2)
        # bf16 attention (reverted from fp32 2026-05-28): the fp32-attention
        # experiment was chasing a bf16-softmax cliff theory that turned out
        # wrong (the v5 decoder's additive-residual output was the real cliff
        # fix). fp32 here just doubled attention activation memory and was a
        # major contributor to the OOM cascade. bf16 is fine.
        h = self.norm1(tokens)
        a, _ = self.attn(h, h, h, need_weights=False)
        tokens = tokens + a
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens.transpose(1, 2).reshape(B, C, H, W)


class _CrossAttentionBridge(nn.Module):
    """Cross-attention from decoder Q (downsampled to 16x16) to encoder K/V.

    Decoder feature map at spatial (H, W) -> avg-pool to (16, 16) -> Q.
    Encoder tokens are 768-dim, projected to decoder channels for K, V.
    Attention output is reshaped to (B, C, 16, 16) and bilinear-upsampled
    back to (B, C, H, W), then *added* into the decoder features (skip).
    """

    ENCODER_GRID = 16  # ViT-B at img_size=224, patch=14 -> 16x16 = 256 tokens

    def __init__(self, dec_channels: int, enc_dim: int = 768, num_heads: int = 8,
                 mlp_ratio: float = 2.0, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.dec_channels = dec_channels
        self.norm_q = nn.LayerNorm(dec_channels)
        self.norm_kv = nn.LayerNorm(enc_dim)
        self.k_proj = nn.Linear(enc_dim, dec_channels)
        self.v_proj = nn.Linear(enc_dim, dec_channels)
        # dropout=0.0 inside MHA (same reason as _SelfAttention2D above).
        self.attn = nn.MultiheadAttention(dec_channels, num_heads, dropout=0.0,
                                          batch_first=True)
        self.norm_mlp = nn.LayerNorm(dec_channels)
        hidden = int(dec_channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dec_channels, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dec_channels),
            nn.Dropout(dropout),
        )

    def forward(self, dec_feat: torch.Tensor, enc_tokens: torch.Tensor) -> torch.Tensor:
        B, C, H, W = dec_feat.shape
        # Downsample Q to encoder-native 16x16 (if not already there)
        if H == self.ENCODER_GRID and W == self.ENCODER_GRID:
            q_grid = dec_feat
        else:
            q_grid = F.adaptive_avg_pool2d(dec_feat, self.ENCODER_GRID)
        q_tokens = q_grid.flatten(2).transpose(1, 2)  # (B, 256, C)

        # bf16 attention (reverted from fp32 2026-05-28): fp32 here doubled
        # attention activation memory and was a major OOM contributor. The
        # bf16-softmax instability theory was disproven — the v5 decoder's
        # additive-residual output was the real cliff fix.
        q_norm = self.norm_q(q_tokens)
        kv_norm = self.norm_kv(enc_tokens)
        k = self.k_proj(kv_norm)
        v = self.v_proj(kv_norm)
        attn_out, _ = self.attn(q_norm, k, v, need_weights=False)
        q_tokens = q_tokens + attn_out
        q_tokens = q_tokens + self.mlp(self.norm_mlp(q_tokens))

        out_grid = q_tokens.transpose(1, 2).reshape(B, C, self.ENCODER_GRID,
                                                    self.ENCODER_GRID)
        if H != self.ENCODER_GRID or W != self.ENCODER_GRID:
            out_grid = F.interpolate(out_grid, size=(H, W), mode="bilinear",
                                     align_corners=False)
        return dec_feat + out_grid


class _ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm(x))
        h = self.act(self.conv1(h))
        h = self.conv2(h)
        return x + h


def icnr_(tensor: torch.Tensor, upscale_factor: int = 2,
          initializer=nn.init.kaiming_normal_) -> None:
    """ICNR (Aitken et al. 2017) init for PixelShuffle pre-convs. Without it,
    each of the r×r output sub-pixels gets an independent random projection
    and the network starts with a strong checkerboard pattern that gets baked
    into the residual via a checkerboard-as-content local minimum (observed
    failure mode in v4_p1_5 stageA — 22k steps with a content-independent
    checkerboard output). ICNR initialises the pre-conv so all r² sub-pixels
    share the same projection at step 0, making the upsampler nearest-neighbour
    equivalent — bilinear-equivalent after a single 3×3 anti-aliasing filter."""
    out_ch, in_ch, kh, kw = tensor.shape
    r2 = upscale_factor ** 2
    assert out_ch % r2 == 0, f"out_ch={out_ch} not divisible by r²={r2}"
    sub = torch.empty(out_ch // r2, in_ch, kh, kw, device=tensor.device,
                      dtype=tensor.dtype)
    initializer(sub)
    sub = sub.repeat_interleave(r2, dim=0)
    with torch.no_grad():
        tensor.copy_(sub)


class _UpsampleStage(nn.Module):
    """Bilinear-upsample(2x) + 3x3 conv + residual stack.

    Checkerboard-free by construction — no PixelShuffle, no transposed conv.
    Replaces an earlier `nn.PixelShuffle(2)` design that, even with ICNR init
    (Aitken 2017), reliably converged to a content-independent checkerboard
    over the 4-stage cascade (16→256). Visual evidence: step-500 triptychs
    of fresh-start runs on 2026-05-30 showed sharp colored-grid outputs
    identical in pattern to the 22k-step collapsed run.

    Bilinear is what every modern SR architecture uses for the same reason
    (RealESRGAN, BSRGAN, SwinIR, EDSR-style heads). 4x less compute in the
    pre-conv (no x4 channel inflation) and ~3M fewer params per stage.
    """

    def __init__(self, in_ch: int, out_ch: int, n_resblocks: int = 2):
        super().__init__()
        # Bilinear-up is a fixed (parameter-free) op; pre-conv learns the
        # filter, blocks learn the residual content.
        self.pre = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.blocks = nn.Sequential(*[_ResBlock(out_ch) for _ in range(n_resblocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear",
                          align_corners=False)
        x = self.pre(x)
        x = self.blocks(x)
        return x


class PixelDecoderV4(nn.Module):
    """Inputs:
        low_res:       (B, 3, 256, 256), in [0, 1]
        pred_features: (B, 256, 768)
        ctx_features:  (B, 256, 768)
        multi_scale:   list of 4 x (B, 256, 768) — encoder layers [3, 6, 9, 12]

    Output dict:
        pred_image:     (B, 3, 256, 256), in (0, 1) via sigmoid
        lr_clamp_rate:  scalar — fraction of LR pixels at the eps boundary
                        (for the train-loop watchdog).
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

        # ----- Bottleneck fusion (16x16 mix) -----
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

        # ----- Bottleneck-stage cross-attention (decoder Q @ encoder layer 12) -----
        # Acts before upsampling; uses base_dim channels.
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

        # ----- Per-stage cross-attention with encoder multi-scale layers -----
        # multi_scale order is [layer3, layer6, layer9, layer12] (shallow->deep).
        # Stage i (16*2^(i+1) spatial) cross-attends with layer index (3-i):
        #   stage 0 (32x32)  -> multi_scale[3] = layer 12 (deepest, semantic)
        #   stage 1 (64x64)  -> multi_scale[2] = layer 9
        #   stage 2 (128x128)-> multi_scale[1] = layer 6
        #   stage 3 (256x256)-> multi_scale[0] = layer 3 (shallowest, textural)
        self.stage_cross = nn.ModuleList([
            _CrossAttentionBridge(
                dec_channels=self.UPSAMPLE_CHANNELS[i], enc_dim=jepa_dim,
                num_heads=cross_attn_heads, mlp_ratio=cross_attn_mlp_ratio,
                dropout=dropout,
            )
            for i in range(4)
        ])

        # ----- Output head (zero-init residual -> identity at step 0) -----
        final_ch = self.UPSAMPLE_CHANNELS[-1]
        self.final_norm = nn.GroupNorm(1, final_ch)
        self.final_conv = nn.Conv2d(final_ch, 3, 3, padding=1)
        nn.init.zeros_(self.final_conv.weight)
        nn.init.zeros_(self.final_conv.bias)

    @staticmethod
    def _safe_logit(x: torch.Tensor, eps: float) -> torch.Tensor:
        x = x.clamp(eps, 1.0 - eps)
        return torch.log(x) - torch.log1p(-x)

    def forward(self, low_res: torch.Tensor, pred_features: torch.Tensor,
                ctx_features: torch.Tensor,
                multi_scale: Sequence[torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert len(multi_scale) == 4, f"expected 4 multi-scale features, got {len(multi_scale)}"

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

        # ---- Bottleneck-stage cross-attention (decoder Q vs encoder layer 12) ----
        x = self.bottleneck_cross(x, multi_scale[3])             # (B, base, 16, 16)

        # ---- Upsample with per-stage cross-attention ----
        # multi_scale[3] = layer 12 (semantic) -> first upsample stage
        # multi_scale[0] = layer  3 (textural) -> last upsample stage (256x256)
        skip_layers = (multi_scale[3], multi_scale[2], multi_scale[1], multi_scale[0])
        for i, up in enumerate(self.up_stages):
            x = up(x)
            x = self.stage_cross[i](x, skip_layers[i])

        # ---- Sigmoid output: sigmoid(logit(LR) + residual) ----
        residual = self.final_conv(self.final_norm(x))           # (B, 3, 256, 256)

        # In fp16/bf16 the eps=1e-4 may sit at/below precision; cast to fp32 for the
        # logit math so we don't introduce log(0) NaN under autocast.
        lr_fp32 = low_res.float()
        clamp_mask = (lr_fp32 < self.LR_EPS) | (lr_fp32 > 1.0 - self.LR_EPS)
        lr_clamp_rate = clamp_mask.float().mean()
        lr_logit = self._safe_logit(lr_fp32, self.LR_EPS)
        out = torch.sigmoid(lr_logit + residual.float())

        return {"pred_image": out, "lr_clamp_rate": lr_clamp_rate.detach()}


# ---------------- smoke test ----------------

def _smoke():
    import argparse, time
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="run smoke test")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)
    print(f"decoder_v4 smoke on {device}, batch={args.batch_size}")

    dec = PixelDecoderV4().to(device)
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

    img = out["pred_image"]
    cr = out["lr_clamp_rate"]
    assert tuple(img.shape) == (B, 3, 256, 256), f"shape: {img.shape}"
    assert img.min().item() >= 0.0 and img.max().item() <= 1.0, \
        f"sigmoid bounds violated: [{img.min().item():.4f}, {img.max().item():.4f}]"
    assert torch.isfinite(img).all(), "NaN/Inf in pred"
    # At step 0 with zero-init final_conv, residual = 0, output = sigmoid(logit(LR)) ≈ LR.
    identity_err = (img - low_res).abs().max().item()
    print(f"  shape OK: {tuple(img.shape)}")
    print(f"  range OK: [{img.min().item():.4f}, {img.max().item():.4f}]")
    print(f"  identity-at-init err (max abs): {identity_err:.2e}")
    print(f"  lr_clamp_rate: {cr.item():.4e}")
    print(f"  forward time: {fwd_ms:.1f} ms")

    # Backward sanity
    img.sum().backward()
    n_grad = sum((p.grad is not None and p.grad.abs().sum().item() > 0)
                 for p in dec.parameters() if p.requires_grad)
    print(f"  backward OK: {n_grad} params with non-zero grad")

    # Sigmoid clamp-rate behaviour: feed extreme LR (0 and 1) and confirm
    # lr_clamp_rate > 0.
    lr_extreme = torch.cat([torch.zeros(B, 3, 256, 128, device=device),
                            torch.ones(B, 3, 256, 128, device=device)], dim=-1)
    out2 = dec(lr_extreme, pred_f, ctx_f, multi)
    assert out2["lr_clamp_rate"].item() > 0.99, \
        f"expected lr_clamp_rate ~1 for all-extreme input, got {out2['lr_clamp_rate'].item()}"
    print(f"  clamp-rate watchdog OK: {out2['lr_clamp_rate'].item():.4f} for all-extreme LR")
    print("decoder_v4 smoke PASSED")


if __name__ == "__main__":
    _smoke()
