"""
JEPAConditionedRRDBNet — composes a pretrained Real-ESRGAN RRDBNet with
JEPA-derived feature conditioning.

Design
------
The RRDBNet operates at scale=4 (its native pretrained mode). Our data pipeline
delivers `low_res` already bilinear-upsampled to 256×256 (to match `high_res`
spatially for JEPA). To use ESRGAN as intended, we **internally downsample**
the 256×256 LR back to 64×64 with an average pool, run the full 4× RRDBNet
pipeline, and produce a 256×256 SR output. This preserves every pretrained
weight (conv_first, 23 RRDBs, conv_body, conv_up1, conv_up2, conv_hr, conv_last)
in its trained configuration.

JEPA conditioning is added at 3 points in the SR head:

  1. **Bottleneck** (after the RRDB trunk, before the upsample head) at 64×64.
     Inject `ctx_final` (the JEPA context encoder's final features).
  2. **Post-upsample-1** at 128×128. Inject `ctx_multi[2]` (mid-depth ViT layer).
  3. **Post-upsample-2** at 256×256. Inject `ctx_multi[3]` (final-depth ViT layer).

Each injection is a `FeatureInjection` module: project JEPA tokens (B, 256, 768)
down to a small dim, reshape to a 16×16 spatial grid, bilinear-upsample to the
target feature-map resolution, concatenate with the feature map, and fuse via
1×1 conv. The 1×1 fuse is **zero-initialised** so its residual contribution at
step 1 is exactly 0 — guarantees identity-at-init.

Identity-at-init invariant
--------------------------
When all `FeatureInjection.fuse` layers are zero-initialised:

    JEPAConditionedRRDBNet(low_res_256, jepa)  ==  RRDBNet(avg_pool_4x(low_res_256))

This is checked in `scripts/v5_phase1_smoketest.py`.
"""

from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .esrgan.rrdbnet import RRDBNet


class FeatureInjection(nn.Module):
    """Inject JEPA tokens into a 2D feature map at a target spatial size.

    Computes a zero-init residual that is added to the feature map. At init
    the residual is exactly 0, so the parent module's output is unchanged.
    """

    def __init__(self, feat_ch: int, jepa_dim: int = 768, project_dim: int = 64,
                 token_grid_side: int = 16):
        super().__init__()
        self.feat_ch = feat_ch
        self.project_dim = project_dim
        self.token_grid_side = token_grid_side

        # 1×1 over the token sequence (acts as Linear on the 768-d channel).
        self.project = nn.Conv1d(jepa_dim, project_dim, 1)
        self.fuse = nn.Conv2d(feat_ch + project_dim, feat_ch, 1)

        # Project: small Kaiming init.
        nn.init.kaiming_normal_(self.project.weight, a=0, mode="fan_in")
        nn.init.zeros_(self.project.bias)
        # Fuse: zero-init weight + bias → residual contribution is 0 at step 1.
        nn.init.zeros_(self.fuse.weight)
        nn.init.zeros_(self.fuse.bias)

    def forward(self, feat_map: torch.Tensor, jepa_tokens: torch.Tensor) -> torch.Tensor:
        """feat_map: (B, C, H, W). jepa_tokens: (B, N=token_grid_side², D=jepa_dim)."""
        B, N, D = jepa_tokens.shape
        s = self.token_grid_side
        if N != s * s:
            raise ValueError(f"jepa_tokens has {N} tokens, expected {s*s}")

        proj = self.project(jepa_tokens.transpose(1, 2))             # (B, project_dim, N)
        proj = proj.reshape(B, self.project_dim, s, s)               # (B, project_dim, s, s)

        H, W = feat_map.shape[-2:]
        if (H, W) != (s, s):
            proj = F.interpolate(proj, size=(H, W), mode="bilinear", align_corners=False)

        x = torch.cat([feat_map, proj], dim=1)                       # (B, C+project_dim, H, W)
        residual = self.fuse(x)                                      # zero at init
        return feat_map + residual


class JEPAConditionedRRDBNet(nn.Module):
    """Pretrained Real-ESRGAN RRDBNet (4×) conditioned on JEPA tokens.

    Forward
    -------
    Inputs:
      low_res    : (B, 3, 256, 256) — bilinear-upsampled LR (matches dataset)
      ctx_final  : (B, 256, 768)    — JEPA context encoder final features
      ctx_multi  : list of (B, 256, 768) — multi-depth ViT features. Items at
                   indices `up1_multi_idx` and `up2_multi_idx` are injected at
                   the two upsample stages. Default uses indices 2 and 3
                   (extract_layers=(3,6,9,12) → layer 9 and layer 12).

    Output:
      sr         : (B, 3, 256, 256) — super-resolved output.

    Identity-at-init: when all `FeatureInjection.fuse` weights are zero,
    the output equals `inner_rrdbnet(avg_pool_4x(low_res))`.
    """

    def __init__(
        self,
        rrdbnet: RRDBNet,
        jepa_dim: int = 768,
        token_grid_side: int = 16,
        project_dim: int = 64,
        up1_multi_idx: int = 2,
        up2_multi_idx: int = 3,
        downsample_factor: int = 4,
    ):
        super().__init__()
        if rrdbnet.scale != 4:
            raise ValueError(f"Expected RRDBNet(scale=4), got scale={rrdbnet.scale}")
        self.rrdbnet = rrdbnet
        self.downsample_factor = downsample_factor
        self.up1_multi_idx = up1_multi_idx
        self.up2_multi_idx = up2_multi_idx

        feat_ch = rrdbnet.conv_first.out_channels  # 64 for x4plus

        self.inject_bottleneck = FeatureInjection(
            feat_ch, jepa_dim=jepa_dim, project_dim=project_dim,
            token_grid_side=token_grid_side)
        self.inject_up1 = FeatureInjection(
            feat_ch, jepa_dim=jepa_dim, project_dim=project_dim,
            token_grid_side=token_grid_side)
        self.inject_up2 = FeatureInjection(
            feat_ch, jepa_dim=jepa_dim, project_dim=project_dim,
            token_grid_side=token_grid_side)

    def _downsample_lr(self, low_res_256: torch.Tensor) -> torch.Tensor:
        """LR-256 → LR-64 by avg pooling, matching ESRGAN's pretrained 4× input."""
        f = self.downsample_factor
        return F.avg_pool2d(low_res_256, kernel_size=f, stride=f)

    def forward(
        self,
        low_res: torch.Tensor,
        ctx_final: torch.Tensor,
        ctx_multi: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        lr_small = self._downsample_lr(low_res)              # (B, 3, 64, 64)

        # ---- Trunk (pretrained, untouched) ----
        feat = self.rrdbnet.conv_first(lr_small)             # (B, 64, 64, 64)
        body_feat = self.rrdbnet.conv_body(self.rrdbnet.body(feat))
        trunk = feat + body_feat                              # (B, 64, 64, 64)

        # ---- Bottleneck injection (zero-init residual) ----
        trunk = self.inject_bottleneck(trunk, ctx_final)     # (B, 64, 64, 64)

        # ---- Upsample stage 1 ----
        x = F.interpolate(trunk, scale_factor=2.0, mode="nearest")
        x = self.rrdbnet.lrelu(self.rrdbnet.conv_up1(x))     # (B, 64, 128, 128)
        x = self.inject_up1(x, ctx_multi[self.up1_multi_idx])

        # ---- Upsample stage 2 ----
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.rrdbnet.lrelu(self.rrdbnet.conv_up2(x))     # (B, 64, 256, 256)
        x = self.inject_up2(x, ctx_multi[self.up2_multi_idx])

        # ---- Final head (pretrained) ----
        out = self.rrdbnet.conv_last(self.rrdbnet.lrelu(self.rrdbnet.conv_hr(x)))
        return out

    # ------------------------------------------------------------------
    # Parameter grouping for the optimizer.
    # ------------------------------------------------------------------
    def trainable_param_groups(
        self,
        rrdbnet_lr: float,
        injection_lr: float,
    ) -> List[dict]:
        """Two groups: pretrained RRDBNet (low LR) + new injection layers (high LR).

        Used after the initial warmup where the RRDBNet is fully frozen and only
        the injection layers train.
        """
        rrdb_params = list(self.rrdbnet.parameters())
        inj_params = (
            list(self.inject_bottleneck.parameters())
            + list(self.inject_up1.parameters())
            + list(self.inject_up2.parameters())
        )
        return [
            {"params": inj_params, "lr": injection_lr, "name": "jepa_injection"},
            {"params": rrdb_params, "lr": rrdbnet_lr, "name": "rrdbnet_pretrained"},
        ]

    def freeze_rrdbnet(self) -> None:
        for p in self.rrdbnet.parameters():
            p.requires_grad = False

    def unfreeze_rrdbnet(self) -> None:
        for p in self.rrdbnet.parameters():
            p.requires_grad = True
