"""
UrbanJEPA: JEPA model for urban image super-resolution.

Unified token space — ViT and VAE latent use the same 256-token grid:
    ViT-B/16 on 256×256 pixels: 16×16 patches = 256 tokens
    SD-VAE on 256×256 pixels: 32×32×4 latents → group 2×2 → 16×16×16 = 256 tokens × 16-dim

JEPA (embedding space):
    low_res → context_encoder → ctx_features       (256 tokens, 768-dim)
    high_res → target_encoder → target_embeddings   (256 tokens, 768-dim, stop grad, EMA)
    ctx_features + mask → feature_predictor → predicted_embeddings
    projection_head(predicted) ↔ target → Smooth L1 loss

Components:
    context_encoder (φ)  — ViT-B/16, trained
    target_encoder (φ̄)  — ViT-B/16, EMA-only
    feature_predictor (γ) — ViT-B/16 + mask tokens, trained
    vae_encoder          — SD-VAE, frozen
    vae_decoder          — SD-VAE, frozen initially
    projection_head (uθ) — 2-layer MLP, trained
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import UrbanEncoder
from .predictor import FeaturePredictor


class UrbanJEPA(nn.Module):
    def __init__(
        self,
        pretrained_path,
        img_size=256,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        token_dim=16,
        ema_decay=0.9999,
        drop_path_rate=0.0,
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2  # 256 (ViT tokens = VAE latent tokens after 2×2 grouping)
        self.token_dim = token_dim  # 16 (4 VAE channels × 2×2 spatial group)
        self.ema_decay = ema_decay

        # --- Encoders ---
        self.context_encoder = UrbanEncoder(
            pretrained_path=pretrained_path,
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            drop_path_rate=drop_path_rate,
        )

        self.target_encoder = UrbanEncoder(
            pretrained_path=pretrained_path,
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            drop_path_rate=drop_path_rate,
        )

        # Target encoder receives NO gradients — updated via EMA only
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        # Initialize target encoder as exact copy of context encoder
        self.target_encoder.load_state_dict(
            self.context_encoder.state_dict()
        )

        # --- Feature Predictor ---
        self.feature_predictor = FeaturePredictor(
            pretrained_path=pretrained_path,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            max_tokens=self.num_patches,
            drop_path_rate=drop_path_rate,
        )

        # --- Projection Head (uθ) ---
        self.projection_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim),
        )

        # --- VAE (loaded separately) ---
        self.vae = None
        self._vae_loaded = False

    def load_vae(self, vae):
        """Load SD-VAE. Converts to fp32 — targets must be fp32 precision."""
        self.vae = vae.to(torch.float32)
        self._vae_loaded = True

    @torch.no_grad()
    def update_target_encoder(self):
        """EMA update: φ̄ ← α·φ̄ + (1-α)·φ"""
        for ctx_p, tgt_p in zip(
            self.context_encoder.parameters(),
            self.target_encoder.parameters(),
        ):
            tgt_p.data = self.ema_decay * tgt_p.data + (1.0 - self.ema_decay) * ctx_p.data

    def sample_mask(self, N, mask_ratio_mean=0.5, mask_ratio_std=0.2, mask_ratio_min=0.3):
        """
        Sample masking ratio from truncated normal distribution.

        I-JEPA paper uses μ=0.5, σ=0.2 for ViT-B on ImageNet.
        Returns (mask_idx, ctx_idx) — indices of masked and unmasked tokens.
        """
        # Clip to [0.3, 1.0] to ensure meaningful context and prediction
        ratio = torch.normal(mask_ratio_mean, mask_ratio_std, size=(1,)).item()
        ratio = max(mask_ratio_min, min(ratio, 1.0))
        n_mask = max(1, int(ratio * N))
        perm = torch.randperm(N)
        mask_idx = perm[:n_mask]
        ctx_idx = perm[n_mask:]
        return mask_idx, ctx_idx

    def prediction_loss(self, predicted_embeddings, target_embeddings):
        """
        Prediction loss Lp: Smooth L1 between projected predicted and target.

        predicted_embeddings: (B, N_mask, D)
        target_embeddings:    (B, N_mask, D) — detached
        """
        projected = self.projection_head(predicted_embeddings)
        return F.smooth_l1_loss(projected, target_embeddings.detach())

    def forward(self, low_res, high_res):
        """
        JEPA forward pass — predicts ALL tokens for downstream pixel loss.

        low_res:  (B, 3, 256, 256) downsampled ortho
        high_res: (B, 3, 256, 256) high-res ortho crop
        Returns: dict with 'loss', 'loss_prediction', 'cos_sim', 'predicted_all'
        """
        B = low_res.shape[0]
        N = self.num_patches
        device = low_res.device

        # Context encoder: process low-res input
        ctx_features = self.context_encoder(low_res)  # (B, N, D)

        # Target encoder: process high-res target (no gradient)
        with torch.no_grad():
            target_embeddings = self.target_encoder(high_res)  # (B, N, D)

        # Sample mask for JEPA loss
        mask_idx, _ = self.sample_mask(N)
        mask_idx = mask_idx.to(device)
        mask_idx_b = mask_idx.unsqueeze(0).expand(B, -1)  # (B, N_mask)

        # Predict ALL 256 tokens from context (needed for pixel loss)
        all_positions = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
        all_predicted = self.feature_predictor(ctx_features, all_positions)  # (B, N, D)

        # Slice masked positions for JEPA loss
        predicted_masked = all_predicted.gather(
            1, mask_idx_b.unsqueeze(-1).expand(-1, -1, self.embed_dim)
        )  # (B, N_mask, D)
        target_masked = target_embeddings.gather(
            1, mask_idx_b.unsqueeze(-1).expand(-1, -1, self.embed_dim)
        )  # (B, N_mask, D)

        # JEPA prediction loss + cosine similarity
        projected = self.projection_head(predicted_masked)
        cos_sim = F.cosine_similarity(projected, target_masked.detach(), dim=-1).mean()
        Lp = F.smooth_l1_loss(projected, target_masked.detach())

        return {
            "loss": Lp,
            "loss_prediction": Lp.item(),
            "cos_sim": cos_sim.item(),
            "predicted_all": all_predicted,  # (B, N, D) for pixel loss
        }

    # ------------------------------------------------------------------
    # VAE latent encoding/decoding
    # ------------------------------------------------------------------

    def encode_to_latent(self, images):
        """
        Encode images to VAE latent tokens matching ViT token count.
        VAE: (B,3,256,256) → (B,4,32,32) latent.
        Group 2×2 spatial blocks → (B,16,16,16) → 256 tokens × 16-dim.
        """
        if not self._vae_loaded:
            raise RuntimeError("VAE not loaded. Call load_vae() first.")
        vae_dtype = next(self.vae.parameters()).dtype
        with torch.no_grad():
            latents = self.vae.encode(images.to(vae_dtype)).latent_dist.mean
            latents = latents * 0.18215  # SD-VAE scaling
        B, C, H, W = latents.shape  # (B, 4, 32, 32)

        # Spatial-to-channel: group 2×2 blocks → (B, 16, 16, 16)
        latents = latents.reshape(B, C, H // 2, 2, W // 2, 2)   # (B,4,16,2,16,2)
        latents = latents.permute(0, 2, 4, 3, 5, 1)              # (B,16,16,2,2,4)
        latents = latents.reshape(B, H // 2, W // 2, C * 4)      # (B,16,16,16)

        tokens = latents.reshape(B, self.num_patches, self.token_dim)  # (B, 256, 16)
        return tokens.float()

    def decode_from_latent(self, tokens, spatial_size=16):
        """
        Decode latent tokens back to pixel space.
        Reverse of encode_to_latent: (B,256,16) → (B,16,16,16) → (B,4,32,32) → VAE decode.
        """
        if not self._vae_loaded:
            raise RuntimeError("VAE not loaded. Call load_vae() first.")
        vae_dtype = next(self.vae.parameters()).dtype
        B, N, C = tokens.shape

        # Reverse the 2×2 spatial grouping
        x = tokens.reshape(B, spatial_size, spatial_size, C)           # (B,16,16,16)
        x = x.reshape(B, spatial_size, spatial_size, 2, 2, C // 4)    # (B,16,16,2,2,4)
        x = x.permute(0, 5, 1, 3, 2, 4)                               # (B,4,16,2,16,2)
        latents = x.reshape(B, C // 4, spatial_size * 2, spatial_size * 2)  # (B,4,32,32)
        latents = latents / 0.18215
        images = self.vae.decode(latents.to(vae_dtype)).sample
        return images.float()
