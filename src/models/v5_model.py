"""
V5Model — the v5 training wrapper.

Composes:
  - JEPABackbone (context+EMA-target encoders, predictor, projection)
  - JEPAConditionedRRDBNet (pretrained Real-ESRGAN trunk + 3 zero-init JEPA
    injection points)

Forward returns a dict carrying both the SR output and JEPA diagnostics, so
the training loop can compute pixel/perceptual losses on `sr` and optionally
add a small JEPA self-supervision term on `jepa_loss`.

Param groups are exposed in 4 buckets so the optimizer can run each at its
own learning rate:
    jepa_encoder         — DINOv2 ViT (carried from v4, low LR)
    jepa_pred_proj       — JEPA predictor + projection head (slightly higher)
    jepa_injection       — NEW zero-init injection layers in the ESRGAN head
    rrdbnet_pretrained   — pretrained RRDBNet (very low LR, frozen at start)
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn

from .esrgan import build_pretrained_x4plus
from .esrgan.rrdbnet import RRDBNet
from .jepa_esrgan import JEPAConditionedRRDBNet
from .urbanjepa import JEPABackbone


_JEPA_PREFIXES = ("context_encoder.", "target_encoder.", "predictor.", "projection_head.")


class V5Model(nn.Module):
    """JEPA backbone + ESRGAN-with-JEPA-injection, one module to train."""

    def __init__(self, jepa: JEPABackbone, esrgan: JEPAConditionedRRDBNet):
        super().__init__()
        self.jepa = jepa
        self.esrgan = esrgan
        # Preserved for CheckpointManager compatibility (was on UrbanJEPA).
        self.backbone_name = jepa.backbone_name

    # ------------------------------------------------------------------ forward
    def forward(self, low_res: torch.Tensor, high_res: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats = self.jepa(low_res, high_res)
        sr = self.esrgan(low_res, feats["ctx_final"], feats["ctx_multi"])
        return {
            "sr": sr,
            "jepa_loss": feats["jepa_loss"],
            "cos_sim": feats["cos_sim"],
            "ctx_std": feats["ctx_std"],
            "pred_feat_std": feats["pred_feat_std"],
        }

    # ------------------------------------------------------------- param groups
    def trainable_param_groups(
        self,
        jepa_encoder_lr: float,
        jepa_pred_proj_lr: float,
        injection_lr: float,
        rrdbnet_lr: float,
    ) -> List[dict]:
        return [
            {"params": list(self.jepa.predictor.parameters())
                       + list(self.jepa.projection_head.parameters()),
             "lr": jepa_pred_proj_lr, "name": "jepa_pred_proj"},
            {"params": list(self.jepa.context_encoder.parameters()),
             "lr": jepa_encoder_lr, "name": "jepa_encoder"},
            {"params": list(self.esrgan.inject_bottleneck.parameters())
                       + list(self.esrgan.inject_up1.parameters())
                       + list(self.esrgan.inject_up2.parameters()),
             "lr": injection_lr, "name": "jepa_injection"},
            {"params": list(self.esrgan.rrdbnet.parameters()),
             "lr": rrdbnet_lr, "name": "rrdbnet_pretrained"},
        ]

    # ------------------------------------------------------------ freeze toggles
    def freeze_rrdbnet(self) -> None:
        self.esrgan.freeze_rrdbnet()

    def unfreeze_rrdbnet(self) -> None:
        self.esrgan.unfreeze_rrdbnet()

    def freeze_jepa(self) -> None:
        for p in self.jepa.parameters():
            p.requires_grad = False

    def unfreeze_jepa(self) -> None:
        # Re-freeze the EMA target encoder; that one is never trained.
        for p in self.jepa.parameters():
            p.requires_grad = True
        for p in self.jepa.target_encoder.parameters():
            p.requires_grad = False

    # ----------------------------------------------------------------- EMA hook
    @torch.no_grad()
    def update_target_ema(self, decay: float) -> None:
        self.jepa.update_target_ema(decay)

    # --------------------------------------------------------- checkpoint API
    def state_dict_for_checkpoint(self) -> dict:
        return self.state_dict()

    def load_checkpoint_state(self, sd: dict, strict: bool = False):
        result = self.load_state_dict(sd, strict=strict)
        for p in self.jepa.target_encoder.parameters():
            p.requires_grad = False
        self.jepa.target_encoder.eval()
        return result

    def load_jepa_from_v4_checkpoint(
        self,
        ckpt_path: Union[str, Path],
        strict: bool = False,
    ):
        """Warm-start ONLY the JEPABackbone half from a v4 best.pt.

        v4's `model` state dict carries `{context_encoder.*, target_encoder.*,
        predictor.*, projection_head.*, decoder.*, ...}`. We keep only the JEPA
        prefixes and route them through `JEPABackbone.load_checkpoint_state`,
        which also re-syncs the target encoder if it's absent.
        """
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        if "model" in ckpt:
            sd = ckpt["model"]
        else:
            sd = ckpt
        jepa_sd = {k: v for k, v in sd.items()
                   if any(k.startswith(p) for p in _JEPA_PREFIXES)}
        if not jepa_sd:
            raise RuntimeError(f"No JEPA-prefixed keys found in {ckpt_path}")
        return self.jepa.load_checkpoint_state(jepa_sd, strict=strict)


def build_v5_model(
    backbone_name: str = "dinov2_vitb14",
    jepa_pretrained_path: Optional[Union[str, Path]] = "models/pretrained/dinov2_vitb14.pth",
    esrgan_weights_path: Union[str, Path] = "models/pretrained/RealESRGAN_x4plus.pth",
    predictor_depth: int = 6,
    dropout: float = 0.0,
    extract_layers=(3, 6, 9, 12),
    up1_multi_idx: int = 2,
    up2_multi_idx: int = 3,
    project_dim: int = 64,
    use_grad_checkpoint: bool = False,
) -> V5Model:
    """Convenience constructor that wires up the full v5 model with pretrained weights."""
    jepa = JEPABackbone(
        backbone_name=backbone_name,
        pretrained_path=str(jepa_pretrained_path) if jepa_pretrained_path else None,
        predictor_depth=predictor_depth,
        dropout=dropout,
        extract_layers=extract_layers,
        use_grad_checkpoint=use_grad_checkpoint,
    )
    rrdbnet: RRDBNet = build_pretrained_x4plus(esrgan_weights_path, eval_mode=False)
    esrgan = JEPAConditionedRRDBNet(
        rrdbnet=rrdbnet,
        jepa_dim=jepa.embed_dim,
        token_grid_side=int(jepa.num_tokens ** 0.5),
        project_dim=project_dim,
        up1_multi_idx=up1_multi_idx,
        up2_multi_idx=up2_multi_idx,
    )
    return V5Model(jepa=jepa, esrgan=esrgan)
