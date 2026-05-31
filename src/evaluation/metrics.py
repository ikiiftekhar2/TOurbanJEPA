"""
Reconstruction metrics: PSNR + SSIM.

PSNR computed on [0,1] images with eps for numerical safety.
SSIM via pytorch_msssim.
"""

import torch
import torch.nn.functional as F
from pytorch_msssim import ssim as _ssim


def psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    mse = F.mse_loss(pred.float(), target.float())
    return 10.0 * torch.log10(1.0 / (mse + eps))


def ssim_metric(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return _ssim(pred, target, data_range=1.0, size_average=True)
