"""
JEPABackbone — context encoder, EMA target encoder, predictor, projection head.

This is the JEPA-only half of the v5 architecture. The v4 in-house SR decoders
(PixelDecoderWithSkips / V4 / V5) were removed when forensic analysis showed
they all converged to content-independent residual outputs regardless of init
or supervision. v5 composes this backbone with an externally-pretrained SR
decoder (ESRGAN RRDBNet) — see `JEPAConditionedESRGAN` (TBD).

Forward returns:
    {
        'ctx_final':     (B, N, D) final context-encoder features
        'ctx_multi':     list of multi-scale context features (for skip-conditioning)
        'pred_features': (B, N, D) predictor output (predicts target features)
        'projected':     (B, N, D) projected pred features (JEPA target space)
        'jepa_loss':     scalar — SmoothL1(projected, tgt_final.detach())
        'cos_sim':       scalar — mean cosine sim (diagnostic only)
        'ctx_std':       scalar — mean per-dim std of ctx features (collapse diag)
        'pred_feat_std': scalar — same for predictor features
    }

The decoder is a SEPARATE module. To use the backbone for SR, build:

    backbone = JEPABackbone(...)
    decoder  = JEPAConditionedESRGAN(jepa_dim=embed_dim, ...)
    # In train loop:
    feats = backbone(low_res, high_res)   # JEPA forward + features
    pred  = decoder(low_res, feats['ctx_final'], feats['ctx_multi'])
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import load_backbone
from .encoder import MultiScaleEncoder
from .predictor import FeaturePredictor


class _ProjectionHead(nn.Module):
    def __init__(self, dim: int, hidden: int = 768, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )

    def forward(self, x):
        return self.net(x)


class JEPABackbone(nn.Module):
    """JEPA: context encoder + EMA target encoder + predictor + projection head.

    Does NOT include any decoder. Compose with an external SR decoder.
    """

    def __init__(
        self,
        backbone_name: str,
        pretrained_path: Optional[str] = None,
        predictor_depth: int = 6,
        dropout: float = 0.1,
        extract_layers=(3, 6, 9, 12),
        use_grad_checkpoint: bool = False,
    ):
        super().__init__()
        self.backbone_name = backbone_name

        ctx_bb, cfg = load_backbone(backbone_name, pretrained_path, drop_path_rate=dropout)
        tgt_bb, _ = load_backbone(backbone_name, pretrained_path, drop_path_rate=0.0)

        self.context_encoder = MultiScaleEncoder(
            ctx_bb, cfg, extract_layers=extract_layers,
            use_grad_checkpoint=use_grad_checkpoint)
        self.target_encoder = MultiScaleEncoder(tgt_bb, cfg, extract_layers=extract_layers)

        # Target = context at init, frozen, eval mode.
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        self.target_encoder.eval()

        embed_dim = cfg["embed_dim"]
        num_tokens = cfg["num_tokens"]
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens
        self.cfg = cfg

        self.predictor = FeaturePredictor(
            embed_dim=embed_dim, depth=predictor_depth,
            num_heads=cfg["num_heads"], mlp_ratio=4.0,
            dropout=dropout, num_tokens=num_tokens,
        )
        self.projection_head = _ProjectionHead(embed_dim, hidden=embed_dim, dropout=dropout)

    @torch.no_grad()
    def update_target_ema(self, decay: float):
        for ctx_p, tgt_p in zip(self.context_encoder.parameters(),
                                self.target_encoder.parameters()):
            tgt_p.data.mul_(decay).add_(ctx_p.data, alpha=1 - decay)

    def forward(self, low_res: torch.Tensor, high_res: torch.Tensor) -> Dict:
        ctx_final, ctx_multi = self.context_encoder(low_res)
        pred_features = self.predictor(ctx_final)

        with torch.no_grad():
            tgt_final, _ = self.target_encoder(high_res)

        projected = self.projection_head(pred_features)
        jepa_loss = F.smooth_l1_loss(projected, tgt_final.detach())
        cos_sim = F.cosine_similarity(projected, tgt_final.detach(), dim=-1).mean()

        return {
            "ctx_final": ctx_final,
            "ctx_multi": ctx_multi,
            "pred_features": pred_features,
            "projected": projected,
            "jepa_loss": jepa_loss,
            "cos_sim": cos_sim.detach(),
            "ctx_std": ctx_final.detach().float().std(dim=(0, 1)).mean(),
            "pred_feat_std": pred_features.detach().float().std(dim=(0, 1)).mean(),
        }

    def trainable_param_groups(self, encoder_lr: float, predictor_lr: float):
        """Two param groups: encoder (slow LR) + predictor+projection (fast LR).
        Target encoder is excluded (frozen)."""
        return [
            {"params": list(self.predictor.parameters())
                       + list(self.projection_head.parameters()),
             "lr": predictor_lr, "name": "predictor_proj"},
            {"params": list(self.context_encoder.parameters()),
             "lr": encoder_lr, "name": "encoder"},
        ]

    def freeze(self):
        """Freeze everything except target_encoder (already frozen)."""
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def state_dict_for_checkpoint(self):
        """Full state dict INCLUDING target encoder (preserves EMA history)."""
        return self.state_dict()

    def load_checkpoint_state(self, sd, strict: bool = False):
        """Loads checkpoint state. Resets target = context only for legacy ckpts."""
        result = self.load_state_dict(sd, strict=strict)
        has_target = any(k.startswith("target_encoder.") for k in sd.keys())
        if not has_target:
            self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        self.target_encoder.eval()
        return result


# Back-compat alias — old training code imported `UrbanJEPA`. New code should
# use `JEPABackbone` directly.
UrbanJEPA = JEPABackbone
