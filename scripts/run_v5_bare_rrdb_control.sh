#!/bin/bash
# Bare-RRDBNet fine-tune control for v5 attribution.
#
# Mirrors v5 Stage A's recipe EXACTLY minus the JEPA branch, so we can
# isolate whether v5's -0.18 LPIPS gain came from JEPA conditioning or
# from Toronto fine-tuning the pretrained RRDBNet alone.
#
# epochs=4 (~15k steps at bs=20, patches_per_epoch=32) deliberately
# overshoots v5's best.pt @ step 15000 so "needed more time" cannot
# explain a control underperformance.

set -euo pipefail
cd "$(dirname "$0")/.."

VENV_PY="${VENV_PY:-/home/ubuntu/urbanjepa-venv/bin/python}"

EXP_NAME="v5_bare_rrdb_control"
BATCH_SIZE="${BATCH_SIZE:-20}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EPOCHS="${EPOCHS:-4}"
PATCHES_PER_EPOCH="${PATCHES_PER_EPOCH:-32}"
WARMUP_STEPS="${WARMUP_STEPS:-1000}"
RRDBNET_LR="${RRDBNET_LR:-1e-5}"
W_L1="${W_L1:-1.0}"
W_LPIPS="${W_LPIPS:-0.1}"
SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-500}"
VAL_EVERY_STEPS="${VAL_EVERY_STEPS:-1000}"
LOG_EVERY="${LOG_EVERY:-25}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
EMPTY_CACHE_EVERY="${EMPTY_CACHE_EVERY:-200}"

mkdir -p runs checkpoints
echo "[$(date -Iseconds)] starting bare-RRDB control: $EXP_NAME" \
    | tee -a "runs/${EXP_NAME}.log"

PYTHONPATH=. exec "$VENV_PY" scripts/v5_bare_rrdbnet_finetune.py \
    --exp_name "$EXP_NAME" \
    --ortho_dir data/ortho \
    --tile_manifest data/ortho/metadata/train_textured.txt \
    --esrgan_weights models/pretrained/RealESRGAN_x4plus.pth \
    --checkpoint_dir checkpoints \
    --log_dir runs \
    --batch_size "$BATCH_SIZE" \
    --val_batch_size "$VAL_BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --patches_per_epoch "$PATCHES_PER_EPOCH" \
    --epochs "$EPOCHS" \
    --warmup_steps "$WARMUP_STEPS" \
    --rrdbnet_lr "$RRDBNET_LR" \
    --w_l1 "$W_L1" \
    --w_lpips "$W_LPIPS" \
    --lpips_net vgg \
    --save_every_steps "$SAVE_EVERY_STEPS" \
    --val_every_steps "$VAL_EVERY_STEPS" \
    --log_every "$LOG_EVERY" \
    --grad_clip "$GRAD_CLIP" \
    --empty_cache_every "$EMPTY_CACHE_EVERY" \
    --scale_match_val_scales 16.0 18.0 20.0 22.0 24.0 \
    --bf16 \
    --resume
