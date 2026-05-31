#!/bin/bash
# Download pretrained backbone weights for UrbanJEPA v3.
# Places everything under models/pretrained/.
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p models/pretrained

VENV_PY="${VENV_PY:-/home/ubuntu/urbanjepa-venv/bin/python}"

# ImageNet ViT-B/16 (timm, downloaded via Python so it pulls a clean state_dict)
if [ ! -f models/pretrained/imagenet_vitb16.pt ]; then
    echo "Downloading ImageNet ViT-B/16 via timm..."
    "$VENV_PY" - <<'PY'
import timm, torch
m = timm.create_model("vit_base_patch16_224", pretrained=True)
torch.save(m.state_dict(), "models/pretrained/imagenet_vitb16.pt")
print("saved imagenet_vitb16.pt")
PY
fi

# DINOv2 ViT-B/14
if [ ! -f models/pretrained/dinov2_vitb14.pth ]; then
    echo "Downloading DINOv2 ViT-B/14..."
    wget -nc -O models/pretrained/dinov2_vitb14.pth \
        https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth
fi

# ExPLoRA DINOv2 + fMoW (encoder-only)
if [ ! -f models/pretrained/explora_vitb14.pth ]; then
    echo "Downloading ExPLoRA DINOv2+fMoW..."
    wget -nc -O models/pretrained/explora_vitb14.pth \
        "https://huggingface.co/samarkhanna/ExPLoRA/resolve/main/explora_dinov2_fmow_rgb/explora_dinov2_vit_base_fmow_rgb_encoder_only.pth"
fi

echo "All weights present in models/pretrained/"
ls -la models/pretrained/
