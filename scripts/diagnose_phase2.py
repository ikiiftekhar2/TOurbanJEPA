"""
Diagnostic: Isolate Phase 2 embedding quality vs Phase 3 CNN decoder capability.

Three tests, each training a fresh CNN decoder for 5 epochs (frozen JEPA):
  A: target_encoder(high_res)   → CNN decoder → PSNR  (upper bound: can ViT→pixels work?)
  B: context_encoder(high_res)  → CNN decoder → PSNR  (does context encoder preserve spatial info?)
  C: feature_predictor(ctx_features, all_256_positions) → CNN decoder → PSNR  (current approach)

Also measures cosine similarity between predicted and target embeddings.
"""
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

# Tee output to log file
LOG_PATH = Path(__file__).resolve().parents[1] / "diagnostic_results.txt"
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()
log_f = open(LOG_PATH, "w")
sys.stdout = Tee(sys.stdout, log_f)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.urbanjepa import UrbanJEPA
from src.models.cnn_decoder import CNNDecoder, compute_psnr
from src.data.ortho_dataset import OrthoDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

# --- Load data (small subset for speed) ---
print("Loading data...")
val_ds = OrthoDataset("data/ortho", split="val", train_ratio=0.9, augment=False,
                       val_patches_per_tile=4)
val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=4,
                         pin_memory=True, persistent_workers=True)
print(f"Val: {len(val_ds)} samples, {len(val_loader)} batches\n")

# --- Load JEPA ---
print("Loading JEPA checkpoint...")
PRETRAINED = "models/ijepa/vit_base_patch16_224_imagenet.pt"
CKPT = "models/checkpoints/jepa_best.pt"

model = UrbanJEPA(pretrained_path=PRETRAINED).to(device)
ckpt = torch.load(CKPT, map_location="cpu", weights_only=True)
model.context_encoder.load_state_dict(ckpt["context_encoder"])
model.target_encoder.load_state_dict(ckpt["target_encoder"])
model.feature_predictor.load_state_dict(ckpt["feature_predictor"])
model.projection_head.load_state_dict(ckpt["projection_head"])

for p in model.parameters():
    p.requires_grad = False
model.eval()
print("JEPA loaded and frozen.\n")


def train_decoder_for_epochs(decoder, embedding_fn, name, epochs=5):
    """Train a fresh CNN decoder using a specific embedding source. Returns best val PSNR."""
    decoder.train()
    opt = AdamW(decoder.parameters(), lr=1e-3, weight_decay=1e-4)
    mse = nn.MSELoss()
    best_psnr = -float("inf")

    for epoch in range(epochs):
        # Train on val set (small, fine for diagnostic)
        epoch_loss = 0.0
        t0 = time.time()

        for batch in val_loader:
            high_res = batch["high_res"].to(device)
            low_res = batch["low_res"].to(device)

            with torch.no_grad():
                embeddings = embedding_fn(model, low_res, high_res)

            opt.zero_grad()
            output = decoder(embeddings)
            loss = mse(output, high_res)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()

        # Validation PSNR
        decoder.eval()
        val_psnr_sum = 0.0
        with torch.no_grad():
            for batch in val_loader:
                high_res = batch["high_res"].to(device)
                low_res = batch["low_res"].to(device)
                embeddings = embedding_fn(model, low_res, high_res)
                output = decoder(embeddings)
                val_psnr_sum += compute_psnr(output, high_res)

        val_psnr = val_psnr_sum / len(val_loader)
        avg_loss = epoch_loss / len(val_loader)
        elapsed = time.time() - t0

        best_psnr = max(best_psnr, val_psnr)
        status = "↑" if val_psnr == best_psnr else " "
        print(f"  [{name}] Epoch {epoch}: loss={avg_loss:.5f}  PSNR={val_psnr:.2f}dB  {status}  ({elapsed:.0f}s)")

        decoder.train()

    return best_psnr


# --- Embedding functions for each test ---
def emb_target_hr(model, low_res, high_res):
    """Test A: target encoder on high-res (upper bound)"""
    return model.target_encoder(high_res)

def emb_context_hr(model, low_res, high_res):
    """Test B: context encoder on high-res"""
    return model.context_encoder(high_res)

def emb_context_lr(model, low_res, high_res):
    """Context encoder on low-res (extra diagnostic)"""
    return model.context_encoder(low_res)

def emb_predictor_all(model, low_res, high_res):
    """Test C: feature predictor predicting ALL 256 positions from context"""
    ctx = model.context_encoder(low_res)
    B, N, D = ctx.shape
    all_pos = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
    return model.feature_predictor(ctx, all_pos)

def emb_predictor_masked(model, low_res, high_res):
    """Feature predictor predicting ~50% positions (matches training distribution)"""
    ctx = model.context_encoder(low_res)
    B, N, D = ctx.shape
    mask_idx, _ = model.sample_mask(N)
    mask_idx = mask_idx.to(device).unsqueeze(0).expand(B, -1)
    return model.feature_predictor(ctx, mask_idx)


# --- Cosine similarity diagnostic ---
print("Computing cosine similarity (predicted vs target embeddings)...")
cos_sims_all = []
cos_sims_masked = []

with torch.no_grad():
    for batch in val_loader:
        high_res = batch["high_res"].to(device)
        low_res = batch["low_res"].to(device)

        ctx = model.context_encoder(low_res)
        tgt = model.target_encoder(high_res)
        B, N, D = ctx.shape

        # Predict ALL positions
        all_pos = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
        pred_all = model.feature_predictor(ctx, all_pos)
        cs_all = F.cosine_similarity(
            model.projection_head(pred_all), tgt, dim=-1
        ).mean().item()
        cos_sims_all.append(cs_all)

        # Predict ~50% positions (training distribution)
        mask_idx, _ = model.sample_mask(N)
        mask_idx = mask_idx.to(device).unsqueeze(0).expand(B, -1)
        pred_masked = model.feature_predictor(ctx, mask_idx)
        tgt_masked = tgt.gather(1, mask_idx.unsqueeze(-1).expand(-1, -1, D))
        cs_masked = F.cosine_similarity(
            model.projection_head(pred_masked), tgt_masked, dim=-1
        ).mean().item()
        cos_sims_masked.append(cs_masked)

        # Smooth L1 (same metric as Phase 2 val_loss)
        sl1_all = F.smooth_l1_loss(model.projection_head(pred_all), tgt).item()
        sl1_masked = F.smooth_l1_loss(
            model.projection_head(pred_masked), tgt_masked
        ).item()

cos_mean_all = sum(cos_sims_all) / len(cos_sims_all)
cos_mean_masked = sum(cos_sims_masked) / len(cos_sims_masked)

print(f"  Predictor → ALL 256 tokens:   cos_sim={cos_mean_all:.4f}  SmoothL1={sl1_all:.5f}")
print(f"  Predictor → ~50% masked only: cos_sim={cos_mean_masked:.4f}  SmoothL1={sl1_masked:.5f}")
print(f"  Plan gate: cos_sim > 0.70  →  {'PASS' if cos_mean_masked > 0.70 else 'FAIL'}")
print()

# --- Run decoder tests ---
print("=" * 72)
print("TRAINING CNN DECODERS (5 epochs each, using validation set for speed)")
print("=" * 72)

results = {}

for name, emb_fn in [
    ("A: target_enc(HR)     ", emb_target_hr),
    ("B: context_enc(HR)    ", emb_context_hr),
    ("C: predictor(all_256) ", emb_predictor_all),
    ("D: predictor(~50%_mask)", emb_predictor_masked),
    ("E: context_enc(LR)    ", emb_context_lr),
]:
    print(f"\n--- {name.strip()} ---")
    decoder = CNNDecoder().to(device)
    best = train_decoder_for_epochs(decoder, emb_fn, name.strip(), epochs=5)
    results[name.strip()] = best
    del decoder
    torch.cuda.empty_cache()

# --- Verdict ---
print("\n" + "=" * 72)
print("DIAGNOSTIC RESULTS")
print("=" * 72)
for name, psnr in results.items():
    gate = "PASS" if psnr > 26 else "FAIL"
    bar = "█" * min(int(psnr), 40)
    print(f"  {name}:  PSNR={psnr:.2f}dB  {bar}  [{gate}]")

print()
print("Interpretation:")
print("  If A > 26dB: ViT embeddings CAN be decoded to pixels → predictor is the bottleneck")
print("  If A < 22dB: ViT representations are too lossy → revisit Phase 2 or CNN decoder capacity")
print("  If B ≈ A: context and target encoders are equally good (expected after EMA)")
print("  If C << A: predictor can't handle full 256-token prediction (distribution shift)")
print("  If D ≈ C: predictor quality is uniform regardless of masking ratio")
print("  If D >> C: predictor breaks at full-sequence prediction → Phase 3 design flaw")

log_f.close()
print(f"\nResults saved to {LOG_PATH}")
