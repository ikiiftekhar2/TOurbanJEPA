#!/usr/bin/env python3
"""
Download pretrained weights for UrbanJEPA.

IMPORTANT: Meta FAIR only released I-JEPA checkpoints for ViT-H and ViT-G.
ViT-B/16 was NOT released. We use timm's supervised ImageNet ViT-B/16 as the
initialization for all encoder components instead.

Downloads:
1. ViT-B/16 (timm, ImageNet-1K supervised) — used to initialize all three ViTs
2. SD-VAE (stabilityai/sd-vae-ft-mse) — latent tokenizer and decoder
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn


MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def download_vit_base(output_dir: Path) -> dict:
    """Download ViT-B/16 pretrained on ImageNet-1K from timm."""
    try:
        import timm
    except ImportError:
        print("ERROR: timm not installed. Run: pip install timm")
        sys.exit(1)

    print("Loading ViT-B/16 (ImageNet-1K supervised)...")
    model = timm.create_model("vit_base_patch16_224", pretrained=True)
    state_dict = model.state_dict()
    del model

    path = output_dir / "vit_base_patch16_224_imagenet.pt"
    torch.save(state_dict, path)
    n_params = sum(v.numel() for v in state_dict.values()) / 1e6
    print(f"  Saved: {path} ({n_params:.1f}M params)")
    print(f"  NOTE: This is supervised ImageNet pretrained, NOT I-JEPA pretrained.")
    print(f"  Meta FAIR never released ViT-B I-JEPA weights. This is the best")
    print(f"  available ViT-B initialization for our encoders.")
    return state_dict


def download_vae(output_dir: Path):
    """Download SD-VAE from HuggingFace."""
    try:
        from diffusers import AutoencoderKL
    except ImportError:
        print("ERROR: diffusers not installed. Run: pip install diffusers")
        sys.exit(1)

    print("Loading SD-VAE (stabilityai/sd-vae-ft-mse)...")
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse",
        torch_dtype=torch.float32,
    )
    path = output_dir / "sd_vae_ft_mse.pt"
    torch.save(vae.state_dict(), path)

    enc_params = sum(p.numel() for p in vae.encoder.parameters()) / 1e6
    dec_params = sum(p.numel() for p in vae.decoder.parameters()) / 1e6
    print(f"  Saved: {path}")
    print(f"  Encoder: {enc_params:.1f}M params, Decoder: {dec_params:.1f}M params")
    return vae


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / "ijepa").mkdir(exist_ok=True)
    (MODELS_DIR / "vae").mkdir(exist_ok=True)

    print("=" * 60)
    print("UrbanJEPA — Pretrained Weight Downloader")
    print("=" * 60)

    print("\n[1/2] ViT-B/16 Encoder Backbone")
    print("-" * 40)
    download_vit_base(MODELS_DIR / "ijepa")

    print("\n[2/2] VAE Tokenizer / Decoder")
    print("-" * 40)
    download_vae(MODELS_DIR / "vae")

    print("\n" + "=" * 60)
    print("All pretrained weights downloaded.")
    print(f"  ViT-B/16:    {MODELS_DIR}/ijepa/vit_base_patch16_224_imagenet.pt")
    print(f"  SD-VAE:      {MODELS_DIR}/vae/sd_vae_ft_mse.pt")
    print("=" * 60)


if __name__ == "__main__":
    main()
