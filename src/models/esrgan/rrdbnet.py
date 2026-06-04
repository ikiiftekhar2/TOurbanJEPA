"""
RRDBNet — Residual-in-Residual Dense Block network from ESRGAN / Real-ESRGAN.

Transcribed from xinntao/Real-ESRGAN (MIT). Kept dependency-free (no basicsr)
so we can carry only what we need into v5. Architecture matches the
RealESRGAN_x4plus.pth state dict exactly, key-for-key.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _default_init(module: nn.Module, scale: float = 0.1):
    """Kaiming init scaled down — the ESRGAN convention for residual stacks."""
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
            m.weight.data.mul_(scale)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        _default_init(self, scale=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """RRDBNet generator from Real-ESRGAN.

    State-dict keys match RealESRGAN_x4plus.pth (`params_ema`) exactly. For
    scale=4, both `conv_up1` and `conv_up2` apply nearest-neighbour ×2 upsample;
    for scale=2 only `conv_up1` does; for scale=1 neither does.
    """

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        num_feat: int = 64,
        num_block: int = 23,
        num_grow_ch: int = 32,
        scale: int = 4,
    ):
        super().__init__()
        if scale not in (1, 2, 4):
            raise ValueError(f"scale must be 1, 2, or 4 (got {scale})")
        self.scale = scale

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat

        if self.scale >= 2:
            feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2.0, mode="nearest")))
        if self.scale == 4:
            feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2.0, mode="nearest")))

        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out

    def extract_trunk_features(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the post-trunk feature map (B, num_feat, H, W) — the natural
        injection point for JEPA conditioning. Same compute as the first half
        of `forward`, exposed so the wrapper can splice features in."""
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        return feat + body_feat

    def decode_from_trunk(self, feat: torch.Tensor) -> torch.Tensor:
        """Upsample-head half of `forward`, takes the trunk feature map and
        produces the SR output. Pairs with `extract_trunk_features`."""
        if self.scale >= 2:
            feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2.0, mode="nearest")))
        if self.scale == 4:
            feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2.0, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))
