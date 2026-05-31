"""
Loss functions for v3 + v4 training.

All pixel-space loss functions take (pred, target) where both are (B, 3, H, W)
in [0, 1]. Returned values are scalar tensors.

v4 additions:
  - LPIPSLoss          : LPIPS-AlexNet perceptual loss for training (Zhang 2018)
  - hierarchical_jepa_loss : per-target-layer SmoothL1 + (1 - cos), Reed 2023
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
from pytorch_msssim import ssim as _ssim


_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def l1_pixel_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(pred, target)


def out_of_range_loss(pred: torch.Tensor, lo: float = 0.0, hi: float = 1.0) -> torch.Tensor:
    """Mean linear penalty for pixels outside [lo, hi].

    Decoder forward returns unclamped output. This penalty pulls pixels back into
    the valid range and stays differentiable (relu has nonzero gradient outside
    the valid region — unlike clamp, which has zero gradient in the dead zone).
    """
    return F.relu(pred - hi).mean() + F.relu(lo - pred).mean()


def ssim_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 1.0 - _ssim(pred, target, data_range=1.0, size_average=True)


def high_frequency_loss(pred: torch.Tensor, target: torch.Tensor,
                        cutoff_ratio: float = 0.125) -> torch.Tensor:
    """L1 in FFT space restricted to high-frequency components.

    Build a circular low-pass mask in the centered FFT plane with radius
    cutoff_ratio * size, then take L1 on the complement (high-freq region).
    """
    B, C, H, W = pred.shape
    f_pred = torch.fft.fftshift(torch.fft.fft2(pred, norm="ortho"))
    f_targ = torch.fft.fftshift(torch.fft.fft2(target, norm="ortho"))
    yy = torch.linspace(-1, 1, H, device=pred.device).view(H, 1).expand(H, W)
    xx = torch.linspace(-1, 1, W, device=pred.device).view(1, W).expand(H, W)
    r = torch.sqrt(xx ** 2 + yy ** 2)
    high_pass = (r > cutoff_ratio).float()
    high_pass = high_pass[None, None]
    diff = (f_pred - f_targ) * high_pass
    return diff.abs().mean()


class VGGPerceptualLoss(nn.Module):
    """L1 between pred/target VGG16 features at conv1_2, conv2_2, conv3_3, conv4_3."""

    def __init__(self, layer_weights=(1.0, 1.0, 1.0, 1.0)):
        super().__init__()
        vgg = tvm.vgg16(weights=tvm.VGG16_Weights.IMAGENET1K_V1).features.eval()
        for p in vgg.parameters():
            p.requires_grad = False
        # VGG16 indices for the relu after conv1_2 / conv2_2 / conv3_3 / conv4_3
        self.slices = nn.ModuleList([
            nn.Sequential(*list(vgg.children())[:4]),
            nn.Sequential(*list(vgg.children())[4:9]),
            nn.Sequential(*list(vgg.children())[9:16]),
            nn.Sequential(*list(vgg.children())[16:23]),
        ])
        for s in self.slices:
            for p in s.parameters():
                p.requires_grad = False
        self.register_buffer("mean", _IMAGENET_MEAN)
        self.register_buffer("std", _IMAGENET_STD)
        self.layer_weights = tuple(layer_weights)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    @torch.amp.autocast("cuda", enabled=False)
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = self._normalize(pred.float())
        target = self._normalize(target.float())
        loss = torch.zeros((), device=pred.device, dtype=pred.dtype)
        for slc, w in zip(self.slices, self.layer_weights):
            pred = slc(pred)
            with torch.no_grad():
                target = slc(target)
            loss = loss + w * F.l1_loss(pred, target)
        return loss


# ---------- v4 additions ----------

class LPIPSLoss(nn.Module):
    """LPIPS-AlexNet training loss (Zhang et al., 2018, `arXiv:1801.03924`).

    Used as the perceptual loss in v4 Phase 1+ (replaces VGGPerceptualLoss in
    Path A loss recipe — see V4_PLAN §2.5). Returns a scalar mean LPIPS
    distance in [0, ~1+]. Lower = perceptually closer.

    The LPIPS package expects inputs in [-1, 1].
    """

    def __init__(self, net: str = "alex"):
        super().__init__()
        import lpips  # local import to keep optional
        self.net = lpips.LPIPS(net=net, verbose=False)
        for p in self.net.parameters():
            p.requires_grad = False
        self.net.eval()

    @torch.amp.autocast("cuda", enabled=False)
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.float() * 2.0 - 1.0
        target = target.float() * 2.0 - 1.0
        return self.net(pred, target).mean()


def hierarchical_jepa_loss(pred_layers: List[torch.Tensor],
                           target_layers: List[torch.Tensor],
                           smooth_l1_weight: float = 0.25,
                           cos_weight: float = 0.05) -> torch.Tensor:
    """Per-layer SmoothL1 + (1 - cos) over hierarchical predictor / target outputs.

    Per V4_PLAN §2.2:
        L_per_layer = smooth_l1_weight * SmoothL1(pred_i, target_i.detach())
                    + cos_weight       * (1 - cos(pred_i, target_i.detach()))
        L_jepa = sum_i 0.025 * L_per_layer            # outer 0.025 applied here

    Args:
        pred_layers:   list of (B, N, D), one per target layer (length 4)
        target_layers: list of (B, N, D), matching, detached

    Returns:
        scalar loss (already includes the outer per-layer 0.025 weighting,
        so the caller multiplies by 1.0 — i.e. this is the full hierarchical
        JEPA term that goes into the total loss with weight 1.0).

    Total weight in main loss is the sum:
        4 layers * 0.025 = 0.10
    which matches V4_PLAN §2.5 table row "JEPA hierarchical".
    """
    assert len(pred_layers) == len(target_layers), \
        f"layer count mismatch: {len(pred_layers)} vs {len(target_layers)}"
    n = len(pred_layers)
    out = torch.zeros((), device=pred_layers[0].device,
                      dtype=pred_layers[0].dtype)
    per_layer_outer = 0.025
    for p, t in zip(pred_layers, target_layers):
        t = t.detach()
        l_l1 = F.smooth_l1_loss(p, t)
        l_cos = (1.0 - F.cosine_similarity(p, t, dim=-1).mean())
        out = out + per_layer_outer * (smooth_l1_weight * l_l1 + cos_weight * l_cos)
    return out


def vicreg_loss(features: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4,
                var_weight: float = 1.0, cov_weight: float = 0.04) -> torch.Tensor:
    """VICReg variance + covariance regularizer (Bardes et al. 2022).

    Anti-collapse term for the JEPA latent: the model relies on EMA + stop-grad +
    predictor with no explicit force keeping representations from collapsing to a
    constant. This adds one.

    features: (B, N, D) or (B, D); flattened to (M, D). NOT detached — these
    gradients are exactly what push the representation away from collapse.
      variance term  : mean_j relu(gamma - std_j)   (hinge: pushes per-dim std -> gamma)
      covariance term: sum of squared off-diagonal covariances / D
    NO invariance term — JEPA's SmoothL1(projected, target) already plays that role.
    Computed in fp32 (the D x D covariance is precision-sensitive under bf16).
    """
    x = features.reshape(-1, features.shape[-1]).float()
    M, D = x.shape
    x = x - x.mean(dim=0, keepdim=True)
    std = torch.sqrt(x.var(dim=0, unbiased=False) + eps)
    var_loss = F.relu(gamma - std).mean()
    cov = (x.T @ x) / max(1, M - 1)                      # (D, D)
    off_diag = cov - torch.diag(torch.diag(cov))
    # Normalise by D*(D-1) (Bardes 2022) rather than D; the off-diag has ~D²
    # entries so the previous /D made cov_loss scale with D and dominate at
    # high feature dims (D=768 here).
    cov_loss = (off_diag ** 2).sum() / (D * (D - 1))
    return var_weight * var_loss + cov_weight * cov_loss
