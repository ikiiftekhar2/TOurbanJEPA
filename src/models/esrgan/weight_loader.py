"""
Load Real-ESRGAN pretrained weights into our RRDBNet.

The official RealESRGAN_x4plus.pth checkpoint is `{"params_ema": {...}}`
where the inner dict has keys that match our RRDBNet exactly. The loader
unwraps `params_ema` (falling back to `params` if EMA is absent) and runs
`load_state_dict(strict=True)` — any mismatch is a bug, not a warning.
"""

from pathlib import Path
from typing import Optional, Union

import torch

from .rrdbnet import RRDBNet


DEFAULT_X4PLUS_PATH = Path("models/pretrained/RealESRGAN_x4plus.pth")


def load_realesrgan_x4plus(
    model: RRDBNet,
    weights_path: Union[str, Path] = DEFAULT_X4PLUS_PATH,
    *,
    prefer_ema: bool = True,
) -> RRDBNet:
    """Loads RealESRGAN_x4plus weights into the given RRDBNet in-place.

    Raises if anything is missing or shape-mismatched — silent partial loads
    have burned us before, so we want strict.
    """
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Pretrained weights not found at {weights_path}")

    raw = torch.load(weights_path, map_location="cpu", weights_only=False)

    if isinstance(raw, dict) and "params_ema" in raw and prefer_ema:
        state = raw["params_ema"]
    elif isinstance(raw, dict) and "params" in raw:
        state = raw["params"]
    else:
        state = raw

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"State-dict mismatch loading {weights_path}\n"
            f"  missing keys:    {missing}\n"
            f"  unexpected keys: {unexpected}"
        )
    return model


def build_pretrained_x4plus(
    weights_path: Union[str, Path] = DEFAULT_X4PLUS_PATH,
    *,
    device: Optional[Union[str, torch.device]] = None,
    eval_mode: bool = True,
) -> RRDBNet:
    """Convenience: instantiate the canonical x4plus RRDBNet and load weights."""
    model = RRDBNet(
        num_in_ch=3, num_out_ch=3,
        num_feat=64, num_block=23, num_grow_ch=32, scale=4,
    )
    load_realesrgan_x4plus(model, weights_path)
    if device is not None:
        model = model.to(device)
    if eval_mode:
        model.eval()
    return model
