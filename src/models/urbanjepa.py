"""
UrbanJEPA: Full D-JEPA model for urban image super-resolution.

Unified token space — ViT and VAE latent use the same 256-token grid:
    ViT-B/16 on 256×256 pixels: 16×16 patches = 256 tokens
    SD-VAE on 256×256 pixels: 32×32×4 latents → group 2×2 → 16×16×16 = 256 tokens × 16-dim

Phase 2 (embedding space only):
    low_res → context_encoder → ctx_features       (256 tokens, 768-dim)
    high_res → target_encoder → target_embeddings   (256 tokens, 768-dim, stop grad, EMA)
    ctx_features + mask → feature_predictor → predicted_embeddings
    projection_head(predicted) ↔ target → Smooth L1 loss

Phase 4 (adds diffusion in latent space):
    Same JEPA pipeline + VAE encode (2×2 grouped) + Denoising MLP + VAE decode

Components:
    context_encoder (φ)  — ViT-B/16, trained
    target_encoder (φ̄)  — ViT-B/16, EMA-only
    feature_predictor (γ) — ViT-B/16 + mask tokens, trained
    denoising_mlp (εθ)   — 6-block residual MLP, token_dim=16 (Phase 4)
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
from .denoising_mlp import TransformerDenoiser, DenoisingMLP
from .latent_regressor import LatentRegressor, LatentRegressorConv


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
        mlp_hidden_dim=1024,
        mlp_blocks=6,
        ema_decay=0.9999,
        denoiser_type="transformer",
        denoiser_d_model=512,
        denoiser_heads=8,
        denoiser_layers=6,
        regressor_type="mlp",
        regressor_hidden=512,
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
        )

        self.target_encoder = UrbanEncoder(
            pretrained_path=pretrained_path,
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
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
        )

        # --- Projection Head (uθ) ---
        self.projection_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # --- Denoiser (Phase 4) ---
        # TransformerDenoiser: DiT-style with cross-token self-attention (~31M params)
        # DenoisingMLP: legacy per-token MLP (~4M params), use via --legacy_mlp
        if denoiser_type == "transformer":
            self.denoising_mlp = TransformerDenoiser(
                token_dim=token_dim,
                embed_dim=embed_dim,
                d_model=denoiser_d_model,
                num_heads=denoiser_heads,
                num_layers=denoiser_layers,
            )
        else:
            self.denoising_mlp = DenoisingMLP(
                token_dim=token_dim,
                embed_dim=embed_dim,
                hidden_dim=mlp_hidden_dim,
                num_blocks=mlp_blocks,
            )

        # --- Latent Regressor (Path B) ---
        if regressor_type == "conv":
            self.latent_regressor = LatentRegressorConv(
                embed_dim=embed_dim,
                hidden_dim=regressor_hidden,
            )
        else:
            self.latent_regressor = LatentRegressor(
                embed_dim=embed_dim,
                hidden_dim=regressor_hidden,
            )

        # --- VAE (Phase 4, loaded separately) ---
        self.vae = None
        self._vae_loaded = False

    def load_vae(self, vae):
        """Load SD-VAE for Phase 4. vae is an AutoencoderKL from diffusers."""
        self.vae = vae
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
        Phase 2 forward pass — prediction loss only (no diffusion).

        low_res:  (B, 3, 256, 256) downsampled ortho
        high_res: (B, 3, 256, 256) high-res ortho crop
        Returns: dict with 'loss' and 'loss_prediction'
        """
        B = low_res.shape[0]
        N = self.num_patches

        # Context encoder: process low-res input
        ctx_features = self.context_encoder(low_res)  # (B, N, D)

        # Target encoder: process high-res target (no gradient)
        with torch.no_grad():
            target_embeddings = self.target_encoder(high_res)  # (B, N, D)

        # Sample mask on the token index axis
        mask_idx, _ = self.sample_mask(N)
        mask_idx = mask_idx.to(low_res.device)

        # Expand mask_idx for batch
        mask_idx_b = mask_idx.unsqueeze(0).expand(B, -1)  # (B, N_mask)

        # Predict embeddings for masked positions
        predicted_embeddings = self.feature_predictor(
            ctx_features, mask_idx_b
        )  # (B, N_mask, D)

        # Target embeddings at masked positions
        target_masked = target_embeddings.gather(
            1, mask_idx_b.unsqueeze(-1).expand(-1, -1, self.embed_dim)
        )  # (B, N_mask, D)

        # Compute prediction loss
        Lp = self.prediction_loss(predicted_embeddings, target_masked)

        return {
            "loss": Lp,
            "loss_prediction": Lp.item(),
        }

    # ------------------------------------------------------------------
    # Path B: Direct latent regression (skip diffusion)
    # ------------------------------------------------------------------

    def forward_regress(self, low_res):
        """
        Path B forward: context_encoder → latent_regressor → VAE latent.

        low_res: (B, 3, 256, 256) downsampled ortho
        Returns: (B, 4, 32, 32) predicted VAE latent (scaled)
        """
        ctx_features = self.context_encoder(low_res)  # (B, 256, 768)
        return self.latent_regressor(ctx_features)     # (B, 4, 32, 32)

    def regress_decode(self, low_res):
        """
        Full Path B pipeline: low_res → regressor → VAE decode → pixel image.
        Convenience for evaluation.
        """
        latent = self.forward_regress(low_res)
        return self.decode_from_latent_2d(latent)

    def decode_from_latent_2d(self, latent_2d):
        """
        Decode a (B, 4, 32, 32) latent directly (no token grouping step).

        Unlike decode_from_latent which expects (B, 256, 16) tokens,
        this takes the native VAE latent format.
        """
        if not self._vae_loaded:
            raise RuntimeError("VAE not loaded. Call load_vae() first.")
        vae_dtype = next(self.vae.parameters()).dtype
        latents = latent_2d / 0.18215
        images = self.vae.decode(latents.to(vae_dtype)).sample
        return images.float()

    # ------------------------------------------------------------------
    # Phase 4 methods (stubs — unused in Phase 2)
    # ------------------------------------------------------------------

    def train_for_phase(self, phase):
        """
        Configure which parameters are trainable for the current phase.

        phase='jepa':  context_encoder + feature_predictor + projection_head
        phase='mlp':   denoising_mlp only (everything else frozen)
        phase='joint': all components jointly
        """
        # Freeze everything first
        for p in self.parameters():
            p.requires_grad = False

        if phase == "jepa":
            for p in self.context_encoder.parameters():
                p.requires_grad = True
            for p in self.feature_predictor.parameters():
                p.requires_grad = True
            for p in self.projection_head.parameters():
                p.requires_grad = True

        elif phase == "mlp":
            for p in self.denoising_mlp.parameters():
                p.requires_grad = True

        elif phase == "regress":
            for p in self.latent_regressor.parameters():
                p.requires_grad = True

        elif phase == "regress_joint":
            for p in self.context_encoder.parameters():
                p.requires_grad = True
            for p in self.feature_predictor.parameters():
                p.requires_grad = True
            for p in self.latent_regressor.parameters():
                p.requires_grad = True

        elif phase == "joint":
            for p in self.context_encoder.parameters():
                p.requires_grad = True
            for p in self.feature_predictor.parameters():
                p.requires_grad = True
            for p in self.projection_head.parameters():
                p.requires_grad = True
            for p in self.denoising_mlp.parameters():
                p.requires_grad = True

        elif phase == "all":
            for p in self.parameters():
                p.requires_grad = True

        # Target encoder NEVER receives gradients
        for p in self.target_encoder.parameters():
            p.requires_grad = False

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
            latents = self.vae.encode(images.to(vae_dtype)).latent_dist.sample()
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

    def diffusion_loss(self, tokens, predicted_embeddings, noise_schedule):
        """
        Diffusion loss Ld: MSE between predicted and actual noise.
        4 noise samples per token, importance-weighted timestep sampling.
        """
        B, N, D = tokens.shape
        total_loss = 0.0

        # Importance weights: sqrt(ᾱ_t) * (1 - ᾱ_t) — peaks at middle timesteps
        # where SNR is changing fastest and learning signal is richest
        w = torch.sqrt(noise_schedule.alpha_bar) * (1 - noise_schedule.alpha_bar)
        w = w / w.sum()

        for _ in range(8):
            t = torch.multinomial(w, B, replacement=True).to(tokens.device)
            x_t, eps = noise_schedule.q_sample(tokens, t)
            eps_pred = self.denoising_mlp(x_t, t, predicted_embeddings)
            total_loss += F.mse_loss(eps_pred, eps)

        return total_loss / 8.0

    @torch.no_grad()
    def ddpm_sample(self, cond, noise_schedule, num_steps=50, temperature=0.98):
        """
        Generate latent tokens via DDPM reverse process.

        cond:       (B, N, embed_dim) JEPA predicted features
        num_steps:  reverse steps (strided across full T range for speed)
        Returns:    (B, N, token_dim) denoised VAE latent tokens
        """
        B, N, D = cond.shape[0], cond.shape[1], self.token_dim
        return noise_schedule.p_sample_loop(
            (B, N, D),
            lambda x_t, t, z: self.denoising_mlp(x_t, t, z),
            cond,
            num_steps=num_steps,
            temperature=temperature,
        )
