import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights


class VGGPerceptualLoss(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        vgg = vgg16(weights=VGG16_Weights.DEFAULT).to(device).eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.slices = nn.ModuleList([
            vgg.features[:4],   # relu1_2
            vgg.features[:9],   # relu2_2
            vgg.features[:16],  # relu3_3
            vgg.features[:23],  # relu4_3
        ])
        self.register_buffer('mean',
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std',
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred, target):
        pred = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std
        loss = 0.0
        for s in self.slices:
            loss += F.l1_loss(s(pred), s(target))
        return loss / len(self.slices)


def high_frequency_loss(pred, target, cutoff_ratio=0.125):
    """L1 on high-freq components via FFT."""
    B, C, H, W = pred.shape
    cy, cx = H // 2, W // 2
    ry, rx = int(H * cutoff_ratio), int(W * cutoff_ratio)

    yy, xx = torch.meshgrid(
        torch.arange(H, device=pred.device),
        torch.arange(W, device=pred.device), indexing='ij')
    mask = ((yy - cy).float() / max(ry, 1)) ** 2 + \
           ((xx - cx).float() / max(rx, 1)) ** 2 >= 1.0
    mask = torch.fft.fftshift(mask.float()).unsqueeze(0).unsqueeze(0)

    pred_hf = torch.fft.ifft2(torch.fft.fft2(pred, norm='ortho') * mask, norm='ortho').real
    tgt_hf = torch.fft.ifft2(torch.fft.fft2(target, norm='ortho') * mask, norm='ortho').real
    return F.l1_loss(pred_hf, tgt_hf)
