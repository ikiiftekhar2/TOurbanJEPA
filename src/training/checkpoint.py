"""
CheckpointManager — async, atomic, manifest-tracked checkpoint system.

Two checkpoint kinds:
  - step: rotating slots, frequent (every N steps), async non-blocking writes
  - epoch: end-of-epoch save, plus a 'best' copy on PSNR improvement

A JSON manifest in the experiment directory tracks every save, so resuming
never requires scanning the filesystem.
"""

import json
import os
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict

import torch


def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _clone_to_cpu(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {k: _clone_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clone_to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_clone_to_cpu(v) for v in obj)
    return obj


def _atomic_write_torch(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _atomic_write_json(obj: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)


class CheckpointManager:
    def __init__(self, checkpoint_dir: str, experiment_name: str,
                 recovery_every: int = 50, recovery_slots: int = 6,
                 rollback_steps: int = 200):
        self.exp_dir = Path(checkpoint_dir) / experiment_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.exp_dir / "manifest.json"
        self.experiment_name = experiment_name
        self.recovery_every = recovery_every
        self.recovery_slots = recovery_slots
        self.rollback_steps = rollback_steps
        self._last_thread: Optional[threading.Thread] = None
        self._manifest_lock = threading.Lock()
        self.manifest = self._load_or_init_manifest()

    def _load_or_init_manifest(self) -> dict:
        if self.manifest_path.exists():
            with open(self.manifest_path) as f:
                return json.load(f)
        m = {
            "experiment": self.experiment_name,
            "created": _now_iso(),
            "best": None,
            "latest_epoch": None,
            "step_checkpoints": [],
            "history": [],
        }
        _atomic_write_json(m, self.manifest_path)
        return m

    def _save_manifest(self):
        with self._manifest_lock:
            _atomic_write_json(self.manifest, self.manifest_path)

    def log_event(self, event_type: str, **kwargs):
        self.manifest["history"].append({
            "event": event_type, "time": _now_iso(), **kwargs,
        })
        if len(self.manifest["history"]) > 2000:
            self.manifest["history"] = self.manifest["history"][-1000:]
        self._save_manifest()

    def _build_state(self, model, optimizer, scheduler, scaler,
                     epoch: int, batch_idx: int, global_step: int,
                     best_val_psnr: float, config: dict,
                     ema=None) -> dict:
        state = {
            "epoch": epoch,
            "batch_idx": batch_idx,
            "global_step": global_step,
            "model": model.state_dict_for_checkpoint(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "scaler": scaler.state_dict() if scaler else None,
            "best_val_psnr": best_val_psnr,
            "config": config,
            "backbone_name": getattr(model, "backbone_name", None),
        }
        # Persist the EMA shadow so it survives systemd restarts. Without this,
        # every spike-restart resets ModelEMA to the live weights, throwing away
        # the entire smoothing history — defeating the point of EMA over a
        # 100k-step run where restarts happen every few hours.
        if ema is not None:
            state["ema"] = ema.state_dict()
        return state

    def save_step(self, model, optimizer, scheduler, scaler,
                  epoch: int, batch_idx: int, global_step: int,
                  best_val_psnr: float, config: dict,
                  l1_window: Optional[list] = None, ema=None):
        """Non-blocking step checkpoint. Clones state to CPU then writes in background.

        l1_window: optional list of recent L1 values; restored on resume so the
        spike watchdog doesn't have a 200-step blind spot after every restart.
        ema: optional ModelEMA; persisted so EMA history survives restarts.
        """
        slot = (global_step // self.recovery_every) % self.recovery_slots
        path = self.exp_dir / f"step_slot_{slot}.pt"

        state = self._build_state(model, optimizer, scheduler, scaler,
                                  epoch, batch_idx, global_step,
                                  best_val_psnr, config, ema=ema)
        if l1_window is not None:
            state["l1_window"] = list(l1_window)
        cpu_state = _clone_to_cpu(state)

        def _worker():
            try:
                _atomic_write_torch(cpu_state, path)
            except Exception as e:
                print(f"[ckpt] async save failed for {path}: {e}")

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        self._last_thread = t

        with self._manifest_lock:
            # Update or insert this slot entry
            slot_entries = [s for s in self.manifest["step_checkpoints"] if s["slot"] != slot]
            slot_entries.append({
                "slot": slot,
                "path": str(path),
                "epoch": epoch,
                "batch_idx": batch_idx,
                "global_step": global_step,
                "saved_at": _now_iso(),
            })
            slot_entries.sort(key=lambda s: s["global_step"])
            self.manifest["step_checkpoints"] = slot_entries
        self._save_manifest()

    def save_epoch(self, model, optimizer, scheduler, scaler,
                   epoch: int, global_step: int, val_psnr: float,
                   best_val_psnr: float, config: dict, extra_metrics: dict = None,
                   ema=None):
        """Blocking save at epoch end. Updates 'best' if val_psnr improved."""
        path = self.exp_dir / f"epoch_{epoch}.pt"
        state = self._build_state(model, optimizer, scheduler, scaler,
                                  epoch, batch_idx=0, global_step=global_step,
                                  best_val_psnr=best_val_psnr, config=config,
                                  ema=ema)
        state["val_psnr"] = val_psnr
        if extra_metrics:
            state["val_metrics"] = extra_metrics
        cpu_state = _clone_to_cpu(state)
        _atomic_write_torch(cpu_state, path)

        with self._manifest_lock:
            self.manifest["latest_epoch"] = {
                "path": str(path),
                "epoch": epoch,
                "global_step": global_step,
                "val_psnr": val_psnr,
                "saved_at": _now_iso(),
            }
            if (self.manifest["best"] is None
                    or val_psnr > self.manifest["best"].get("val_psnr", -1e9)):
                best_path = self.exp_dir / "best.pt"
                shutil.copyfile(path, best_path)
                self.manifest["best"] = {
                    "path": str(best_path),
                    "epoch": epoch,
                    "global_step": global_step,
                    "val_psnr": val_psnr,
                    "saved_at": _now_iso(),
                }
        self._save_manifest()
        self.log_event("epoch_complete", epoch=epoch, val_psnr=val_psnr,
                       global_step=global_step)

    def save_nan_checkpoint(self, model, optimizer, scheduler, scaler,
                            epoch: int, batch_idx: int, global_step: int,
                            best_val_psnr: float, config: dict, losses: dict,
                            l1_window: Optional[list] = None, ema=None):
        path = self.exp_dir / f"pre_nan_step_{global_step}.pt"
        state = self._build_state(model, optimizer, scheduler, scaler,
                                  epoch, batch_idx, global_step,
                                  best_val_psnr, config, ema=ema)
        if l1_window is not None:
            state["l1_window"] = list(l1_window)
        _atomic_write_torch(_clone_to_cpu(state), path)
        self.log_event("nan_detected", global_step=global_step,
                       pre_nan_path=str(path), losses=losses)
        return path

    def get_resume_checkpoint(self) -> Tuple[Optional[Path], Optional[dict]]:
        """Pick the best resume point: latest step checkpoint with rollback applied,
        else falling back to the latest epoch checkpoint."""
        steps = sorted(self.manifest.get("step_checkpoints", []),
                       key=lambda s: s["global_step"])
        if steps:
            latest_step = steps[-1]["global_step"]
            target_step = max(0, latest_step - self.rollback_steps)
            valid = [s for s in steps if s["global_step"] <= target_step]
            chosen = valid[-1] if valid else steps[0]
            p = Path(chosen["path"])
            if p.exists():
                return p, chosen
        latest = self.manifest.get("latest_epoch")
        if latest and Path(latest["path"]).exists():
            return Path(latest["path"]), latest
        return None, None

    def join_pending_saves(self, timeout: float = 30.0):
        if self._last_thread is not None and self._last_thread.is_alive():
            self._last_thread.join(timeout=timeout)
