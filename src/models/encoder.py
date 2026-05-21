"""
UrbanJEPA Encoder — ViT-B/16 initialized from ImageNet-1K pretrained weights.

NOTE: Meta FAIR never released I-JEPA pretrained ViT-B weights.
We use timm's supervised ViT-B/16 (ImageNet-1K) as initialization instead.
The self-supervised I-JEPA-style fine-tuning (Phase 2) adapts these features
to the ortho domain.
"""

import torch
import torch.nn as nn
from timm.models.layers import DropPath


def build_vit_base(img_size=256, patch_size=16):
    """Build a ViT-B/16 model matching the I-JEPA architecture."""
    import timm
    model = timm.create_model(
        "vit_base_patch16_224",
        pretrained=False,
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        norm_layer=nn.LayerNorm,
    )
    return model


class UrbanEncoder(nn.Module):
    """
    ViT-B/16 encoder for UrbanJEPA.

    Initialized from timm's supervised ImageNet-1K ViT-B/16 weights.
    Used for both context encoder and target encoder (identical architecture).
    """

    def __init__(self, pretrained_path=None, img_size=256, patch_size=16,
                 embed_dim=768, depth=12, num_heads=12, drop_path_rate=0.0):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2

        self.vit = build_vit_base(img_size=img_size, patch_size=patch_size)

        if drop_path_rate > 0.0:
            for i, block in enumerate(self.vit.blocks):
                block.drop_path = DropPath(drop_path_rate * i / (depth - 1))

        if pretrained_path:
            self._load_pretrained(pretrained_path)

    def _load_pretrained(self, path):
        """Load pretrained weights, interpolating pos_embed if needed."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        state_dict = checkpoint.get("state_dict", checkpoint)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        # Interpolate pos_embed if source img_size differs from target
        if "pos_embed" in state_dict:
            src_pe = state_dict["pos_embed"]  # e.g. [1, 197, 768] for 224×224
            tgt_pe = self.vit.pos_embed.data  # e.g. [1, 257, 768] for 256×256
            if src_pe.shape != tgt_pe.shape:
                # Separate cls and patch embeddings
                src_cls = src_pe[:, :1, :]     # (1, 1, D)
                src_patches = src_pe[:, 1:, :]  # (1, N_src, D)
                tgt_cls = tgt_pe[:, :1, :]      # (1, 1, D)

                src_grid = int(src_patches.shape[1] ** 0.5)  # 14
                tgt_grid = int((tgt_pe.shape[1] - 1) ** 0.5)  # 16

                src_patches = src_patches.reshape(1, src_grid, src_grid, -1)
                src_patches = src_patches.permute(0, 3, 1, 2)  # (1, D, 14, 14)

                interpolated = torch.nn.functional.interpolate(
                    src_patches, size=(tgt_grid, tgt_grid),
                    mode="bicubic", align_corners=False,
                )
                interpolated = interpolated.permute(0, 2, 3, 1)  # (1, 16, 16, D)
                interpolated = interpolated.reshape(1, tgt_grid * tgt_grid, -1)  # (1, 256, D)

                state_dict["pos_embed"] = torch.cat([tgt_cls, interpolated], dim=1)

        missing, unexpected = self.vit.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  Encoder: {len(missing)} missing keys (expected for ViT-B init)")
        if unexpected:
            print(f"  Encoder: {len(unexpected)} unexpected keys")

    def forward(self, x):
        """
        x: (B, 3, H, W) input image
        Returns: (B, N, D) token embeddings where N = (H/patch_size)^2
        """
        B = x.shape[0]
        features = self.vit.forward_features(x)  # (B, N+1, D) with cls token

        # Remove cls token — IJEPA uses patch tokens only
        features = features[:, 1:, :]  # (B, N, D)

        return features

    def get_num_tokens(self):
        return self.num_patches
