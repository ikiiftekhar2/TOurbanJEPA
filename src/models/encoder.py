"""
MultiScaleEncoder — wraps a ViT backbone and exposes both the final-layer
features and features from intermediate transformer blocks. Early-layer
outputs preserve spatial detail that gets squeezed out by later blocks; the
decoder consumes them as skip connections.
"""

from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint


class MultiScaleEncoder(nn.Module):
    def __init__(self, backbone: nn.Module, backbone_config: Dict,
                 extract_layers: List[int] = (3, 6, 9, 12),
                 use_grad_checkpoint: bool = False):
        super().__init__()
        self.backbone = backbone
        self.config = backbone_config
        self.img_size = backbone_config["img_size"]
        self.embed_dim = backbone_config["embed_dim"]
        self.num_tokens = backbone_config["num_tokens"]
        self.extract_layers = tuple(extract_layers)
        # Gradient checkpointing: recompute each transformer block's activations
        # during backward instead of storing them. ~40% lower peak memory for a
        # 12-layer ViT, at the cost of one extra forward per block (~25% slower).
        self.use_grad_checkpoint = use_grad_checkpoint
        depth = len(backbone.blocks)
        for layer_idx in self.extract_layers:
            if layer_idx < 1 or layer_idx > depth:
                raise ValueError(
                    f"extract_layers contains {layer_idx} but backbone depth is {depth}"
                )

    def _maybe_resize(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] == self.img_size and x.shape[-2] == self.img_size:
            return x
        return F.interpolate(x, size=(self.img_size, self.img_size),
                             mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        x: (B, 3, H, W) — H, W may differ from the backbone img_size; we resize.

        Returns:
          final: (B, 256, D) — output of the last block (CLS stripped)
          multi: list of (B, 256, D) — one per extract_layers entry (CLS stripped)
        """
        x = self._maybe_resize(x)

        # timm ViT: patch_embed → pos_drop(+pos_embed) → blocks → norm
        h = self.backbone.patch_embed(x)
        if hasattr(self.backbone, "_pos_embed"):
            h = self.backbone._pos_embed(h)
        else:
            cls = self.backbone.cls_token.expand(h.shape[0], -1, -1)
            h = torch.cat([cls, h], dim=1)
            h = h + self.backbone.pos_embed
            h = self.backbone.pos_drop(h)

        if hasattr(self.backbone, "patch_drop"):
            h = self.backbone.patch_drop(h)
        if hasattr(self.backbone, "norm_pre"):
            h = self.backbone.norm_pre(h)

        multi_outputs = []
        target_set = set(self.extract_layers)
        # Only checkpoint when grad is actually being tracked (training the
        # context encoder). The target encoder runs under no_grad, where
        # checkpoint would be a wasteful no-op, so we guard on requires_grad.
        do_ckpt = (self.use_grad_checkpoint and self.training
                   and torch.is_grad_enabled())
        for i, block in enumerate(self.backbone.blocks, start=1):
            if do_ckpt:
                h = torch.utils.checkpoint.checkpoint(block, h, use_reentrant=False)
            else:
                h = block(h)
            if i in target_set:
                multi_outputs.append(h[:, 1:, :])

        h = self.backbone.norm(h)
        final = h[:, 1:, :]

        return final, multi_outputs
