"""
EMA (exponential moving average) of model parameters.

Per V4_PLAN §5.2: EMA copy of the trainable model is updated each step with
decay schedule that ramps from `decay_start` toward `decay_end` over the
specified `warmup_steps`, beginning after `start_step`. Eval uses the EMA
weights; the EMA copy is also persisted alongside training weights.

We hold a CPU clone by default to keep GPU memory budget intact at B=23.
On step update, the relevant tensors are temporarily uploaded to GPU for the
in-place lerp. Eval-time `apply_to(model)` swaps weights into a target model.
"""

from __future__ import annotations

import copy
from typing import Dict, Iterator, Optional

import torch
import torch.nn as nn


class ModelEMA:
    """Polyak-averaged shadow weights of `model`.

    Skips:
      - Frozen parameters (requires_grad == False) — e.g. target encoder.
      - Non-floating-point buffers (so we don't accidentally average ints).

    Args:
        model: the live training model.
        decay_start: initial decay value at step `start_step`.
        decay_end: target decay reached after `warmup_steps` steps past start.
        warmup_steps: linear ramp length from decay_start → decay_end.
        start_step: global_step at which EMA begins updating.
        device: where to keep the shadow copy. "cpu" minimises GPU pressure.
    """

    def __init__(self, model: nn.Module, decay_start: float = 0.995,
                 decay_end: float = 0.9999, warmup_steps: int = 0,
                 start_step: int = 0, device: str = "cpu"):
        self.decay_start = float(decay_start)
        self.decay_end = float(decay_end)
        self.warmup_steps = int(warmup_steps)
        self.start_step = int(start_step)
        self.device = torch.device(device)
        self.shadow: Dict[str, torch.Tensor] = {}
        for name, p in self._trainable_named_params(model):
            self.shadow[name] = p.detach().to(self.device).clone()

    @staticmethod
    def _trainable_named_params(model: nn.Module) -> Iterator:
        for name, p in model.named_parameters():
            if p.requires_grad and torch.is_floating_point(p):
                yield name, p

    def current_decay(self, global_step: int) -> float:
        if global_step <= self.start_step:
            return self.decay_start
        if self.warmup_steps <= 0:
            return self.decay_end
        progress = (global_step - self.start_step) / self.warmup_steps
        progress = max(0.0, min(1.0, progress))
        return self.decay_start + (self.decay_end - self.decay_start) * progress

    @torch.no_grad()
    def update(self, model: nn.Module, global_step: int) -> Optional[float]:
        if global_step < self.start_step:
            return None
        d = self.current_decay(global_step)
        for name, p in self._trainable_named_params(model):
            if name not in self.shadow:
                self.shadow[name] = p.detach().to(self.device).clone()
                continue
            live = p.detach().to(self.device, non_blocking=True)
            self.shadow[name].mul_(d).add_(live, alpha=1.0 - d)
        return d

    def apply_to(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """Swap shadow weights into `model`; returns original tensors for restore."""
        backup: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for name, p in self._trainable_named_params(model):
                if name in self.shadow:
                    backup[name] = p.detach().clone()
                    p.copy_(self.shadow[name].to(p.device, dtype=p.dtype))
        return backup

    def restore(self, model: nn.Module, backup: Dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            for name, p in self._trainable_named_params(model):
                if name in backup:
                    p.copy_(backup[name])

    def state_dict(self) -> dict:
        return {
            "shadow": {k: v.cpu() for k, v in self.shadow.items()},
            "decay_start": self.decay_start,
            "decay_end": self.decay_end,
            "warmup_steps": self.warmup_steps,
            "start_step": self.start_step,
        }

    def load_state_dict(self, sd: dict) -> None:
        self.shadow = {k: v.to(self.device) for k, v in sd["shadow"].items()}
        self.decay_start = sd.get("decay_start", self.decay_start)
        self.decay_end = sd.get("decay_end", self.decay_end)
        self.warmup_steps = sd.get("warmup_steps", self.warmup_steps)
        self.start_step = sd.get("start_step", self.start_step)
