"""
UrbanJEPA — orchestrates context encoder, target encoder (frozen EMA copy),
predictor, decoder, and projection head.

The pixel-space loss is the primary objective. The JEPA loss (predictor output
vs target encoder output, in projected feature space) is a small regularizer
that keeps the embedding geometry stable.

v4 feature flags (per V4_PLAN §2):
    use_v4_predictor  -> swap in HierarchicalFeaturePredictor (8 layers, 4 heads)
    use_v4_decoder    -> swap in PixelDecoderV4 (cross-attn + sigmoid)
    hierarchical_jepa -> emit `pred_layers` for the hierarchical JEPA loss
"""

import copy
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import load_backbone
from .encoder import MultiScaleEncoder
from .predictor import FeaturePredictor
from .predictor_v4 import HierarchicalFeaturePredictor
from .pixel_decoder import PixelDecoderWithSkips
from .decoder_v4 import PixelDecoderV4
from .decoder_v5 import PixelDecoderV5


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


class UrbanJEPA(nn.Module):
    def __init__(self, backbone_name: str, pretrained_path: str = None,
                 predictor_depth: int = 6,
                 decoder_attn_blocks: int = 4,
                 decoder_base_dim: int = 768,
                 dropout: float = 0.1,
                 extract_layers=(3, 6, 9, 12),
                 # ---- v4 feature flags ----
                 use_v4_predictor: bool = False,
                 use_v4_decoder: bool = False,
                 use_v5_decoder: bool = False,
                 hierarchical_jepa: bool = False,
                 use_grad_checkpoint: bool = False):
        super().__init__()
        self.backbone_name = backbone_name
        self.use_v4_predictor = use_v4_predictor
        self.use_v4_decoder = use_v4_decoder
        self.hierarchical_jepa = hierarchical_jepa
        if hierarchical_jepa and not use_v4_predictor:
            raise ValueError("hierarchical_jepa=True requires use_v4_predictor=True")

        ctx_bb, cfg = load_backbone(backbone_name, pretrained_path, drop_path_rate=dropout)
        tgt_bb, _ = load_backbone(backbone_name, pretrained_path, drop_path_rate=0.0)

        # Only the context encoder is trained, so only it benefits from grad
        # checkpointing. The target encoder runs under no_grad (no activations
        # saved), so checkpointing there is a no-op.
        self.context_encoder = MultiScaleEncoder(
            ctx_bb, cfg, extract_layers=extract_layers,
            use_grad_checkpoint=use_grad_checkpoint)
        self.target_encoder = MultiScaleEncoder(tgt_bb, cfg, extract_layers=extract_layers)

        # Initialize target = context, freeze target.
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        # Belt-and-suspenders: ensure the frozen target encoder runs in eval mode
        # (no dropout, no batchnorm running-stats updates). ViT has no BN, so this
        # is mostly intent documentation, but timm models in other backbones do.
        self.target_encoder.eval()

        embed_dim = cfg["embed_dim"]
        num_tokens = cfg["num_tokens"]

        if use_v4_predictor:
            # v4 plan §2.2: 8 layers, 4 per-layer heads.
            self.predictor = HierarchicalFeaturePredictor(
                embed_dim=embed_dim, depth=8,
                num_heads=cfg["num_heads"], mlp_ratio=4.0,
                dropout=dropout, num_tokens=num_tokens,
                n_targets=len(extract_layers),
            )
        else:
            self.predictor = FeaturePredictor(
                embed_dim=embed_dim, depth=predictor_depth,
                num_heads=cfg["num_heads"], mlp_ratio=4.0,
                dropout=dropout, num_tokens=num_tokens,
            )

        if use_v5_decoder:
            self.decoder = PixelDecoderV5(
                jepa_dim=embed_dim, base_dim=decoder_base_dim,
                n_attn_blocks=decoder_attn_blocks, dropout=dropout,
            )
        elif use_v4_decoder:
            self.decoder = PixelDecoderV4(
                jepa_dim=embed_dim, base_dim=decoder_base_dim,
                n_attn_blocks=decoder_attn_blocks, dropout=dropout,
            )
        else:
            self.decoder = PixelDecoderWithSkips(
                jepa_dim=embed_dim, base_dim=decoder_base_dim,
                n_attn_blocks=decoder_attn_blocks, dropout=dropout,
            )

        self.projection_head = _ProjectionHead(embed_dim, hidden=embed_dim, dropout=dropout)

    @torch.no_grad()
    def update_target_ema(self, decay: float):
        for ctx_p, tgt_p in zip(self.context_encoder.parameters(),
                                self.target_encoder.parameters()):
            tgt_p.data.mul_(decay).add_(ctx_p.data, alpha=1 - decay)

    def forward(self, low_res: torch.Tensor, high_res: torch.Tensor) -> Dict:
        ctx_final, ctx_multi = self.context_encoder(low_res)

        # --- predictor ---
        if self.use_v4_predictor:
            pred_out = self.predictor(ctx_final)
            pred_layers: List[torch.Tensor] = pred_out["layers"]
            pred_features = pred_out["final"]
        else:
            pred_layers = None
            pred_features = self.predictor(ctx_final)

        # --- target encoder pass (frozen) ---
        with torch.no_grad():
            tgt_final, tgt_multi = self.target_encoder(high_res)

        # --- JEPA loss (single-layer or hierarchical) ---
        projected = self.projection_head(pred_features)
        jepa_loss = F.smooth_l1_loss(projected, tgt_final.detach())
        cos_sim = F.cosine_similarity(projected, tgt_final.detach(), dim=-1).mean()

        # --- decoder ---
        decoder_out = self.decoder(low_res, pred_features, ctx_final, ctx_multi)
        if isinstance(decoder_out, dict):
            pred_image = decoder_out["pred_image"]
            lr_clamp_rate = decoder_out.get("lr_clamp_rate")
            pred_unclamped = decoder_out.get("pred_unclamped")
        else:
            # Raw-tensor return (e.g. v3 PixelDecoderWithSkips: `low_res +
            # tanh(r)*RESIDUAL_RANGE`). That tensor IS the unclamped output
            # (the decoder didn't clamp internally), so feed it to BOTH the
            # clamped-loss and unclamped-loss paths so OOB still has teeth.
            pred_image = decoder_out
            lr_clamp_rate = None
            pred_unclamped = decoder_out

        result: Dict = {
            "pred_image": pred_image,
            "jepa_loss": jepa_loss,
            "cos_sim": cos_sim.detach(),
            "ctx_final": ctx_final,
            "pred_features": pred_features,
            # Predictor's projected features — exposed for VICReg variance/cov
            # regularization in the train loop (no-op unless w_vicreg > 0).
            "projected": projected,
            # Representation-collapse diagnostics: mean per-feature-dim std.
            # Near 0 => the encoder/predictor has collapsed to a constant.
            "ctx_std": ctx_final.detach().float().std(dim=(0, 1)).mean(),
            "pred_feat_std": pred_features.detach().float().std(dim=(0, 1)).mean(),
        }
        if lr_clamp_rate is not None:
            result["lr_clamp_rate"] = lr_clamp_rate
        if pred_unclamped is not None:
            result["pred_unclamped"] = pred_unclamped
        # Pass through decoder collapse diagnostics (Change 4).
        if isinstance(decoder_out, dict):
            for _k in ("pred_unclamped_std", "oob_frac"):
                if _k in decoder_out:
                    result[_k] = decoder_out[_k]
        if self.hierarchical_jepa:
            # Returned for the external hierarchical_jepa_loss aggregator in
            # losses.py; the simple jepa_loss above stays for compatibility +
            # cos_sim diagnostics.
            result["pred_layers"] = pred_layers
            result["target_layers"] = [t.detach() for t in tgt_multi]
        return result

    def freeze_encoder_side(self):
        """Freeze context_encoder + predictor + projection_head (in addition to
        target_encoder which is always frozen). Used for the Phase-2-style
        "decoder-only against frozen pretrained features" training recipe."""
        for module in (self.context_encoder, self.predictor, self.projection_head):
            for p in module.parameters():
                p.requires_grad = False
            module.eval()

    def trainable_param_groups(self, decoder_lr: float, encoder_lr: float):
        """Returns parameter groups for AdamW: decoder + predictor + proj at decoder_lr,
        context encoder at encoder_lr. Target encoder is excluded (frozen).
        Groups whose params are all frozen (e.g. after freeze_encoder_side)
        are dropped — AdamW errors on empty param groups."""
        candidates = [
            {"params": list(self.decoder.parameters()), "lr": decoder_lr, "name": "decoder"},
            {"params": list(self.predictor.parameters()), "lr": decoder_lr, "name": "predictor"},
            {"params": list(self.projection_head.parameters()), "lr": decoder_lr, "name": "proj"},
            {"params": list(self.context_encoder.parameters()), "lr": encoder_lr, "name": "encoder"},
        ]
        kept = []
        for g in candidates:
            train_params = [p for p in g["params"] if p.requires_grad]
            if train_params:
                g["params"] = train_params
                kept.append(g)
        return kept

    def state_dict_for_checkpoint(self):
        """Full state dict INCLUDING target encoder.

        Old comment claimed target was "reconstructible from context+EMA" — but
        the target encoder accumulates its own EMA via `update_target_ema`, and
        without persisting it, every systemd restart on L1-spike resets target =
        context, throwing away thousands of EMA steps. The full state dict is
        ~75 MB vs the encoder's 86 MB; rounding cost for correctness."""
        return self.state_dict()

    def load_checkpoint_state(self, sd, strict: bool = False):
        """Loads checkpoint state. Resets target = context ONLY for legacy
        checkpoints that don't have target_encoder weights."""
        result = self.load_state_dict(sd, strict=strict)
        has_target = any(k.startswith("target_encoder.") for k in sd.keys())
        if not has_target:
            self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        self.target_encoder.eval()
        return result
