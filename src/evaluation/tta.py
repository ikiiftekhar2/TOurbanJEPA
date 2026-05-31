"""
8x geometric self-ensemble (Lim et al., EDSR 2017, §4.1) for inference.

Apply 8 geometric transforms (identity, 3 rotations, horizontal flip + 4 rots),
run the model, invert the transform, then average. Costs 8x compute, gains
~0.1-0.3 dB PSNR on SR tasks.
"""

from typing import Callable, List

import torch
import torch.nn.functional as F


def _apply_transform(x: torch.Tensor, t_id: int) -> torch.Tensor:
    """Apply one of 8 dihedral-group transforms to a (..., H, W) tensor.

    t_id encodes (n_rot, do_hflip):
        0: identity                4: hflip
        1: rot90                   5: hflip + rot90
        2: rot180                  6: hflip + rot180
        3: rot270                  7: hflip + rot270
    """
    rot = t_id % 4
    flip = t_id >= 4
    if flip:
        x = torch.flip(x, dims=[-1])
    if rot:
        x = torch.rot90(x, k=rot, dims=[-2, -1])
    return x


def _invert_transform(x: torch.Tensor, t_id: int) -> torch.Tensor:
    """Inverse of `_apply_transform`. Order is reversed: undo rotation, then flip."""
    rot = t_id % 4
    flip = t_id >= 4
    if rot:
        x = torch.rot90(x, k=-rot, dims=[-2, -1])
    if flip:
        x = torch.flip(x, dims=[-1])
    return x


@torch.no_grad()
def tta_predict(model_fn: Callable[[torch.Tensor], torch.Tensor],
                low_res: torch.Tensor, n_transforms: int = 8) -> torch.Tensor:
    """Run `model_fn` under the 8 dihedral transforms and average the outputs.

    Args:
        model_fn: callable taking (B, 3, H, W) → (B, 3, H, W).
        low_res:  (B, 3, H, W) input.
        n_transforms: 1, 4 (rotations only), or 8 (full dihedral).

    Returns:
        (B, 3, H, W) averaged prediction.
    """
    assert n_transforms in (1, 4, 8), f"n_transforms must be 1/4/8, got {n_transforms}"
    outs: List[torch.Tensor] = []
    for t_id in range(n_transforms):
        x = _apply_transform(low_res, t_id)
        y = model_fn(x)
        y = _invert_transform(y, t_id)
        outs.append(y)
    return torch.stack(outs, dim=0).mean(dim=0)


if __name__ == "__main__":
    # smoke
    x = torch.randn(2, 3, 256, 256)

    def identity(z):
        return z.clamp(0, 1)

    # With identity, averaging the 8 transforms-of-identity must recover the
    # input exactly (the dihedral group acts on (H,W); the average of the
    # inverse-transformed identity outputs is just x).
    y = tta_predict(identity, x.clamp(0, 1), n_transforms=8)
    err = (y - x.clamp(0, 1)).abs().max().item()
    assert err < 1e-5, f"identity TTA should recover input, got max err {err}"
    print(f"tta smoke: identity recovery OK (max err {err:.2e})")

    # Check round-trip on each transform individually.
    for t in range(8):
        z = _invert_transform(_apply_transform(x, t), t)
        e = (z - x).abs().max().item()
        assert e < 1e-5, f"transform {t} round-trip failed: {e}"
    print("tta smoke: 8 round-trips OK")
