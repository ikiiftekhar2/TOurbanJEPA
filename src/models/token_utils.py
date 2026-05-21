def group_latent_tokens(latent_2d, group_size=2):
    """(B, 4, 32, 32) -> (B, 256, 16)"""
    B, C, H, W = latent_2d.shape
    g = group_size
    x = latent_2d.reshape(B, C, H // g, g, W // g, g)
    x = x.permute(0, 2, 4, 3, 5, 1)
    return x.reshape(B, (H // g) * (W // g), C * g * g)


def ungroup_latent_tokens(tokens, grid_h=16, grid_w=16, channels=4, group_size=2):
    """(B, 256, 16) -> (B, 4, 32, 32)"""
    B, N, D = tokens.shape
    g = group_size
    x = tokens.reshape(B, grid_h, grid_w, g, g, channels)
    x = x.permute(0, 5, 1, 3, 2, 4)
    return x.reshape(B, channels, grid_h * g, grid_w * g)


if __name__ == '__main__':
    import torch
    latent = torch.randn(2, 4, 32, 32)
    tokens = group_latent_tokens(latent)
    recovered = ungroup_latent_tokens(tokens)
    err = (latent - recovered).abs().max().item()
    assert err < 1e-6, f"Roundtrip FAILED: max error = {err}"
    print(f"OK: {latent.shape} -> {tokens.shape} -> {recovered.shape}, err={err:.2e}")
