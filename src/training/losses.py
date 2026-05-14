"""
Linear noise schedule from ADM (Dhariwal & Nichol, NeurIPS 2021).
Used by the Denoising MLP in Phase 4 D-JEPA training.

β ∈ [1e-4, 2e-2], T=1000, linear variance schedule.
Pre-computes alpha_bar values for efficient forward/reverse diffusion.
"""

import torch
import torch.nn as nn


class LinearNoiseSchedule:
    """
    Linear variance schedule for DDPM diffusion.

    T=1000 steps, β linearly spaced from beta_start to beta_end.
    Pre-computes α, ᾱ, √ᾱ, √(1-ᾱ), and posterior variance for fast sampling.
    """

    def __init__(self, T=1000, beta_start=1e-4, beta_end=2e-2, device="cpu"):
        self.T = T

        betas = torch.linspace(beta_start, beta_end, T, device=device)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        alpha_bar_prev = torch.cat(
            [torch.ones(1, device=device), alpha_bar[:-1]]
        )

        self.betas = betas
        self.alphas = alphas
        self.alpha_bar = alpha_bar
        self.alpha_bar_prev = alpha_bar_prev

        self.sqrt_alpha_bar = torch.sqrt(alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)
        self.posterior_variance = betas * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)

    def q_sample(self, x0, t):
        """
        Forward diffusion: corrupt x0 with noise at timestep t.

        x0: (B, N, D) clean tokens
        t:  (B,) integer timesteps
        Returns: (x_t, epsilon) where x_t = √ᾱ_t·x0 + √(1-ᾱ_t)·ε
        """
        sqrt_ab = self.sqrt_alpha_bar[t][:, None, None]
        sqrt_one_minus = self.sqrt_one_minus_alpha_bar[t][:, None, None]
        eps = torch.randn_like(x0)
        x_t = sqrt_ab * x0 + sqrt_one_minus * eps
        return x_t, eps

    def to(self, device):
        """Move all buffers to the given device."""
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bar = self.alpha_bar.to(device)
        self.alpha_bar_prev = self.alpha_bar_prev.to(device)
        self.sqrt_alpha_bar = self.sqrt_alpha_bar.to(device)
        self.sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        return self

    @torch.no_grad()
    def p_sample(self, noise_pred, x_t, t, z, temperature=0.98):
        """
        Single reverse diffusion step.

        noise_pred: (B, N, D) predicted noise from denoising MLP
        x_t:        (B, N, D) noisy tokens at timestep t
        t:          (B,) integer timesteps
        z:          (B, N, embed_dim) JEPA embeddings (unused in base step,
                     included for interface compatibility)
        temperature: sample diversity (τ < 1 reduces variance)
        Returns: x_{t-1}
        """
        t_idx = t[0].item()
        beta_t = self.betas[t_idx]
        alpha_t = self.alphas[t_idx]
        ab_t = self.alpha_bar[t_idx]
        ab_prev = self.alpha_bar_prev[t_idx]

        # Predicted x0 from noise prediction
        x0_pred = (x_t - torch.sqrt(1.0 - ab_t) * noise_pred) / torch.sqrt(ab_t)
        x0_pred = x0_pred.clamp(-1.0, 1.0)

        # Posterior mean
        mean = (
            torch.sqrt(ab_prev) * beta_t / (1.0 - ab_t) * x0_pred
            + torch.sqrt(alpha_t) * (1.0 - ab_prev) / (1.0 - ab_t) * x_t
        )

        if t_idx == 0:
            return mean

        noise = torch.randn_like(x_t)
        variance = torch.sqrt(self.posterior_variance[t_idx]) * temperature
        return mean + variance * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, denoise_fn, cond, num_steps=50, temperature=0.98):
        """
        Full DDPM reverse process with strided timesteps for efficiency.

        shape:       (B, N, D) latent token shape
        denoise_fn:  callable (x_t, t, cond) -> noise_pred
        cond:        (B, N, embed_dim) conditioning from JEPA
        num_steps:   number of reverse steps (uses stride across full T range)
        temperature: sample diversity
        Returns: (B, N, D) denoised latent tokens
        """
        B, N, D = shape
        stride = max(1, self.T // num_steps)
        timesteps = list(range(self.T - 1, -1, -stride))

        x_t = torch.randn(B, N, D, device=cond.device)
        for t in timesteps:
            t_batch = torch.full((B,), t, device=cond.device, dtype=torch.long)
            eps_pred = denoise_fn(x_t, t_batch, cond)
            x_t = self.p_sample(eps_pred, x_t, t_batch, cond, temperature)
        return x_t
