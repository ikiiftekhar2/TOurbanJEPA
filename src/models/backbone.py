"""
Swappable ViT backbone loader.

load_backbone(name, pretrained_path) returns (model, config) for one of:
  - imagenet_vitb16 : timm vit_base_patch16_224, img_size=256, 16x16 = 256 tokens
  - dinov2_vitb14   : timm vit_base_patch14_dinov2, img_size=224, 16x16 = 256 tokens
  - explora_vitb14  : same arch as dinov2; weights are DINOv2+fMoW with LoRA merged

All three produce (B, 257, 768) from forward_features (CLS + 256 patch tokens).
"""

import math
from pathlib import Path
from typing import Tuple, Dict, Optional

import torch
import torch.nn.functional as F
import timm


BACKBONE_CONFIGS = {
    "imagenet_vitb16": {
        "timm_name": "vit_base_patch16_224",
        "img_size": 256,
        "patch_size": 16,
        "num_tokens": 256,
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
    },
    "dinov2_vitb14": {
        "timm_name": "vit_base_patch14_dinov2",
        "img_size": 224,
        "patch_size": 14,
        "num_tokens": 256,
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
    },
    "explora_vitb14": {
        "timm_name": "vit_base_patch14_dinov2",  # same architecture
        "img_size": 224,
        "patch_size": 14,
        "num_tokens": 256,
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
    },
}


def _interp_pos_embed(pos_embed: torch.Tensor, new_num_tokens: int) -> torch.Tensor:
    """Bicubic interpolate ViT positional embeddings to new token grid size.

    pos_embed: (1, N+1, D) — N old patch tokens + 1 CLS at index 0
    new_num_tokens: target number of patch tokens (e.g. 256 for 16x16)
    """
    cls = pos_embed[:, :1]
    patch = pos_embed[:, 1:]
    old_n = patch.shape[1]
    if old_n == new_num_tokens:
        return pos_embed
    old_side = int(math.sqrt(old_n))
    new_side = int(math.sqrt(new_num_tokens))
    assert old_side * old_side == old_n, f"pos_embed {old_n} not square"
    assert new_side * new_side == new_num_tokens, f"target {new_num_tokens} not square"
    d = patch.shape[-1]
    patch = patch.reshape(1, old_side, old_side, d).permute(0, 3, 1, 2)
    patch = F.interpolate(patch, size=(new_side, new_side), mode="bicubic", align_corners=False)
    patch = patch.permute(0, 2, 3, 1).reshape(1, new_num_tokens, d)
    return torch.cat([cls, patch], dim=1)


def _strip_register_tokens(state_dict: dict) -> dict:
    """DINOv2-reg checkpoints carry a `register_tokens` param; we don't use registers."""
    return {k: v for k, v in state_dict.items() if not k.startswith("register_tokens")}


def _remap_dinov2_keys(state_dict: dict) -> dict:
    """Normalize DINOv2 / ExPLoRA checkpoint keys to timm's layout.

    DINOv2 release files store keys without prefix (`blocks.0.norm1.weight`).
    ExPLoRA encoder-only checkpoints follow the same convention. Some HuggingFace
    redistributions add a `backbone.` or `teacher.` prefix.
    """
    out = {}
    for k, v in state_dict.items():
        nk = k
        for prefix in ("teacher.", "student.", "backbone.", "module.", "model."):
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
                break
        out[nk] = v
    return out


def _load_state_dict_file(path: str) -> dict:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        for k in ("state_dict", "model", "model_state_dict", "teacher"):
            if k in obj and isinstance(obj[k], dict):
                obj = obj[k]
                break
    return obj


def load_backbone(name: str, pretrained_path: Optional[str] = None,
                  drop_path_rate: float = 0.1) -> Tuple[torch.nn.Module, Dict]:
    """Build a ViT and optionally load pretrained weights.

    Returns (model, config). config keys: img_size, patch_size, num_tokens, embed_dim, depth.
    """
    if name not in BACKBONE_CONFIGS:
        raise ValueError(f"Unknown backbone {name!r}. Options: {list(BACKBONE_CONFIGS)}")
    cfg = BACKBONE_CONFIGS[name]

    model = timm.create_model(
        cfg["timm_name"],
        pretrained=False,
        img_size=cfg["img_size"],
        drop_path_rate=drop_path_rate,
        num_classes=0,
    )

    if pretrained_path is None:
        return model, dict(cfg)

    sd = _load_state_dict_file(pretrained_path)
    sd = _strip_register_tokens(sd)

    if name in ("dinov2_vitb14", "explora_vitb14"):
        sd = _remap_dinov2_keys(sd)

    if "pos_embed" in sd:
        target_tokens = cfg["num_tokens"]
        old_n = sd["pos_embed"].shape[1] - 1
        if old_n != target_tokens:
            sd["pos_embed"] = _interp_pos_embed(sd["pos_embed"], target_tokens)

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[backbone:{name}] missing keys: {len(missing)} — first few: {missing[:5]}")
    if unexpected:
        print(f"[backbone:{name}] unexpected keys: {len(unexpected)} — first few: {unexpected[:5]}")

    return model, dict(cfg)
