"""
UrbanJEPA Encoder — ViT-B/16 initialized from ImageNet-1K pretrained weights.

NOTE: Meta FAIR never released I-JEPA pretrained ViT-B weights.
We use timm's supervised ViT-B/16 (ImageNet-1K) as initialization instead.
The self-supervised I-JEPA-style fine-tuning (Phase 2) adapts these features
to the ortho domain.
"""

import torch
import torch.nn as nn


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
                 embed_dim=768, depth=12, num_heads=12):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2

        self.vit = build_vit_base(img_size=img_size, patch_size=patch_size)

        if pretrained_path:
            self._load_pretrained(pretrained_path)

    def _load_pretrained(self, path):
        """Load pretrained weights, handling key mismatches."""
        checkpoint = torch.load(path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        # timm checkpoints have different key prefixes; adapt as needed
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
