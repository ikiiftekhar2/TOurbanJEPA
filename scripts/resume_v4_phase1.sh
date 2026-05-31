#!/bin/bash
# UrbanJEPA v4 Phase 1 (Tier-1 validation) — auto-resume + TensorBoard.
#
# Mirrors the v3 step-resume strategy: train.py with --resume picks the latest
# step_slot_*.pt via CheckpointManager.get_resume_checkpoint(), applies the
# 500-step rollback, and continues. On crash, systemd restarts this script and
# the same resume path picks up where it left off.
#
# TensorBoard is started in the background on port 6006 (bound to 0.0.0.0)
# and killed when this script exits, so a single systemd unit owns both.
set -euo pipefail

cd "$(dirname "$0")/.."

VENV_PY="${VENV_PY:-/home/ubuntu/urbanjepa-venv/bin/python}"
TB_BIN="${TB_BIN:-/home/ubuntu/urbanjepa-venv/bin/tensorboard}"
TB_PORT="${TB_PORT:-6006}"
TB_HOST="${TB_HOST:-0.0.0.0}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-v4_phase1_tier1}"
EPOCHS="${EPOCHS:-5}"
# batch_size lowered 23 -> 16 -> bumped to 18 to fit LPIPS-VGG in 24 GB.
# Empirical: batch=16 used 21.1 GB; per-sample cost ~0.7 GB. batch=18
# estimated ~22.5 GB with ~1.5 GB headroom for content-dependent LPIPS-VGG
# allocation spikes. Real-ESRGAN runs batch 12-32, so 18 is well in range.
BATCH_SIZE="${BATCH_SIZE:-18}"
# Peak LR halved 3e-4 -> 1.5e-4 after the 5.0x watchdog let the model diverge
# (PSNR 19 -> 5 dB) in the warmup window where lr_dec crossed 1.3-1.5e-4.
# With peak 1.5e-4 we cap below that unstable zone for the v4 architecture.
LR="${LR:-1.5e-4}"
ENCODER_LR="${ENCODER_LR:-1e-5}"
# LPIPS-VGG (matches the production eval metric — Zhang 2018) replaces
# LPIPS-Alex. Halved 1.0 -> 0.5 after w_lpips=1.0 + w_l1=2.0 caused recurring
# L1 cliff-dives at step ~2000: LPIPS contribution (1.0 × 0.567 = 0.57) was
# 3x L1 contribution (2.0 × 0.095 = 0.19), so optimizer drifted into
# "perceptually plausible but pixel-wrong" basin and L1 exploded.
LPIPS_NET="${LPIPS_NET:-vgg}"
# w_lpips back to default 1.0 (Real-ESRGAN convention) now that v5 decoder
# fixes the cliff. The 0.5 was compensation for a broken decoder.
W_LPIPS="${W_LPIPS:-1.0}"
L1_WARMUP_STEPS="${L1_WARMUP_STEPS:-1000}"
# Zero out raw VGGPerceptualLoss — LPIPS-VGG now covers perceptual.
W_VGG="${W_VGG:-0}"
# Random seed on each launch breaks the cursed-batch death-spiral: a resume
# from pre_nan_step_X.pt with the same seed serves the exact batch that
# triggered the spike (data shuffle is deterministic from seed). New seed
# each launch = different shuffle = different next batch = no loop.
# $RANDOM is bash's built-in PRNG, seeded from $$ + time on launch.
SEED="${SEED:-$RANDOM}"
# w_l1 back to default 2.0 (was bumped to 5.0 compensating for broken v4).
W_L1="${W_L1:-2.0}"
# grad_clip tightened 0.5 -> 0.3 to bound per-step movement when LPIPS
# does occasionally spike. Combined with lower w_lpips this should keep
# the loss landscape navigable through the first epoch.
GRAD_CLIP="${GRAD_CLIP:-0.3}"
# L1-spike watchdog reverted to 3.0x / 200-step window. The 5.0x attempt
# (2026-05-27 14:48) silenced the watchdog through real divergence: the
# 0.26 L1 spikes we thought were "benign hard-batch variance" were in fact
# precursors to full collapse (PSNR 19 -> 5 dB over ~300 steps). With the
# peak LR now halved we shouldn't hit the unstable zone at all, but keep
# the original threshold as a safety net.
L1_SPIKE_RATIO="${L1_SPIKE_RATIO:-3.0}"
L1_SPIKE_WINDOW="${L1_SPIKE_WINDOW:-200}"
# Absolute L1 threshold: fires regardless of window state. Added after the
# 2026-05-27 ~20:00 incident where the model collapsed PSNR 19->4 in 50 steps
# AFTER an auto-resume — the window-based watchdog stayed asleep because the
# 200-step l1_window hadn't filled yet on the new process. 0.25 is conservative
# vs typical healthy L1 of 0.07-0.10.
L1_SPIKE_ABSOLUTE="${L1_SPIKE_ABSOLUTE:-0.25}"
EMA_START_STEP="${EMA_START_STEP:-4000}"
EMA_WARMUP_STEPS="${EMA_WARMUP_STEPS:-20000}"
PATCHES_PER_EPOCH="${PATCHES_PER_EPOCH:-128}"

mkdir -p runs

# ----- Start TensorBoard (background, on the same machine) -----
# Use a deterministic logdir for the TB UI; train.py writes to runs/$EXP_NAME
# but we expose all of runs/ so other experiments stay visible.
TB_LOG="runs/tensorboard.log"
echo "[$(date -Iseconds)] launching tensorboard on ${TB_HOST}:${TB_PORT}, logdir=runs/" \
    | tee -a "$TB_LOG"
"$TB_BIN" --logdir runs --host "$TB_HOST" --port "$TB_PORT" \
    --reload_interval 30 >> "$TB_LOG" 2>&1 &
TB_PID=$!
echo "[$(date -Iseconds)] tensorboard pid=$TB_PID" | tee -a "$TB_LOG"

# Ensure TB is killed when this script exits (graceful stop or crash).
cleanup() {
    if kill -0 "$TB_PID" 2>/dev/null; then
        echo "[$(date -Iseconds)] stopping tensorboard pid=$TB_PID" >> "$TB_LOG"
        kill "$TB_PID" 2>/dev/null || true
        wait "$TB_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# ----- Run the v4 Phase 1 training (auto-resume) -----
echo "[$(date -Iseconds)] launching training: experiment=$EXPERIMENT_NAME epochs=$EPOCHS"
"$VENV_PY" -m src.training.train \
    --data_dir data/ortho \
    --backbone dinov2_vitb14 \
    --pretrained_path models/pretrained/dinov2_vitb14.pth \
    --experiment_name "$EXPERIMENT_NAME" \
    --use_v5_decoder \
    --use_lpips --w_lpips "$W_LPIPS" --lpips_net "$LPIPS_NET" \
    --l1_warmup_steps "$L1_WARMUP_STEPS" \
    --w_vgg "$W_VGG" --w_l1 "$W_L1" \
    --seed "$SEED" \
    --match_train_aug_in_val \
    --use_ema --ema_start_step "$EMA_START_STEP" \
        --ema_warmup_steps "$EMA_WARMUP_STEPS" \
    --epochs "$EPOCHS" --batch_size "$BATCH_SIZE" \
    --lr "$LR" --encoder_lr "$ENCODER_LR" \
    --grad_clip "$GRAD_CLIP" \
    --l1_spike_ratio "$L1_SPIKE_RATIO" \
    --l1_spike_window "$L1_SPIKE_WINDOW" \
    --l1_spike_absolute "$L1_SPIKE_ABSOLUTE" \
    --patches_per_epoch "$PATCHES_PER_EPOCH" \
    --precision bf16 \
    --resume 2>&1 | tee -a "runs/${EXPERIMENT_NAME}.log"
