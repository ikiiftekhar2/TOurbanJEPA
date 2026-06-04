"""
U-Net discriminator with spectral normalization — from Real-ESRGAN (MIT).

Transcribed from xinntao/Real-ESRGAN `realesrgan/archs/discriminator_arch.py`.
Used in v5 Stage B to apply an adversarial loss on top of the L1+LPIPS recipe,
exactly the way Real-ESRGAN itself does. U-Net (vs PatchGAN) gives a dense
per-pixel realism map and combined with spectral norm trains stably under
bf16 without R1/gradient penalty.

Output is (B, 1, H, W) of raw logits. Combine with BCEWithLogitsLoss; positive
target = real, negative target = fake.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class UNetDiscriminatorSN(nn.Module):
    """U-Net discriminator with spectral norm on every conv except first+last.

    Args:
        num_in_ch: input channels (3 for RGB).
        num_feat:  base feature width (Real-ESRGAN default 64).
        skip_connection: enable U-Net residual skips between encoder and decoder.
    """

    def __init__(self, num_in_ch: int = 3, num_feat: int = 64,
                 skip_connection: bool = True):
        super().__init__()
        self.skip_connection = skip_connection
        sn = spectral_norm

        # Stem (no SN on the first conv — Real-ESRGAN convention).
        self.conv0 = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)

        # Downsample path.
        self.conv1 = sn(nn.Conv2d(num_feat,     num_feat * 2, 4, 2, 1, bias=False))
        self.conv2 = sn(nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False))
        self.conv3 = sn(nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False))

        # Upsample path.
        self.conv4 = sn(nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
        self.conv5 = sn(nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        self.conv6 = sn(nn.Conv2d(num_feat * 2, num_feat,     3, 1, 1, bias=False))

        # Extra refinement convs.
        self.conv7 = sn(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv8 = sn(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))

        # Head (no SN on the last conv).
        self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = F.leaky_relu(self.conv0(x), 0.2, inplace=True)
        x1 = F.leaky_relu(self.conv1(x0), 0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(x1), 0.2, inplace=True)
        x3 = F.leaky_relu(self.conv3(x2), 0.2, inplace=True)

        x3 = F.interpolate(x3, scale_factor=2.0, mode="bilinear", align_corners=False)
        x4 = F.leaky_relu(self.conv4(x3), 0.2, inplace=True)
        if self.skip_connection:
            x4 = x4 + x2

        x4 = F.interpolate(x4, scale_factor=2.0, mode="bilinear", align_corners=False)
        x5 = F.leaky_relu(self.conv5(x4), 0.2, inplace=True)
        if self.skip_connection:
            x5 = x5 + x1

        x5 = F.interpolate(x5, scale_factor=2.0, mode="bilinear", align_corners=False)
        x6 = F.leaky_relu(self.conv6(x5), 0.2, inplace=True)
        if self.skip_connection:
            x6 = x6 + x0

        out = F.leaky_relu(self.conv7(x6), 0.2, inplace=True)
        out = F.leaky_relu(self.conv8(out), 0.2, inplace=True)
        out = self.conv9(out)
        return out
