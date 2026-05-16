#!/bin/bash
#
# Auto-resume Path B latent regressor training.
# Finds the latest checkpoint and resumes from it, appending to logs.
# Designed to run as a systemd service on boot.
#
# Usage (manual):  bash scripts/resume_regress.sh
# Usage (service): systemctl start urbanjepa-regress
#
set -euo pipefail

PROJECT_DIR="/mnt/eskeetit/Code-server/UrbanJEPA"
VENV_PYTHON="/home/ubuntu/urbanjepa-venv/bin/python"
LOG_FILE="${PROJECT_DIR}/training_regress_mlp.log"
CKPT_DIR="${PROJECT_DIR}/models/checkpoints"

TOTAL_EPOCHS=20

cd "$PROJECT_DIR"

# --- Find latest checkpoint ---
LATEST=""
LATEST_EPOCH=-1

# Check epoch checkpoints first (best granularity)
for ckpt in "$CKPT_DIR"/regress_epoch_*.pt; do
    if [ -f "$ckpt" ]; then
        fname=$(basename "$ckpt")                    # regress_epoch_4.pt
        epoch_str=${fname#regress_epoch_}            # 4.pt
        epoch_str=${epoch_str%.pt}                    # 4
        epoch_num=$((epoch_str))
        if [ "$epoch_num" -gt "$LATEST_EPOCH" ]; then
            LATEST_EPOCH=$epoch_num
            LATEST=$ckpt
        fi
    fi
done

# Fall back to best.pt if no epoch checkpoints
if [ -z "$LATEST" ] && [ -f "$CKPT_DIR/regress_best.pt" ]; then
    LATEST="$CKPT_DIR/regress_best.pt"
    LATEST_EPOCH="best"
fi

# --- Build command ---
CMD=(
    "$VENV_PYTHON" -u -m src.training.train_regress
    --data_dir data/ortho
    --vae_decoder models/checkpoints/vae_decoder_best.pt
    --batch_size 96
    --epochs 20
    --lr 1e-3
    --regressor_type mlp
    --log_dir runs
    --num_workers 8
    --patches_per_epoch 128
    --log_every 200
    --checkpoint_every 1
)

# --- Check if already complete ---
if [ -n "$LATEST" ] && [ "$LATEST_EPOCH" != "best" ] && [ "$LATEST_EPOCH" -ge "$((TOTAL_EPOCHS - 1))" ]; then
    echo "[$(date)] All ${TOTAL_EPOCHS} epochs complete (latest: epoch ${LATEST_EPOCH}). Nothing to do." | tee -a "$LOG_FILE"
    exit 0
fi

if [ -n "$LATEST" ]; then
    echo "[$(date)] Resuming from epoch ${LATEST_EPOCH} → ${LATEST}" | tee -a "$LOG_FILE"
    CMD+=(--resume "$LATEST")
else
    echo "[$(date)] No checkpoint found — starting fresh" | tee -a "$LOG_FILE"
fi

# --- Launch (append to log) ---
echo "[$(date)] Launching: ${CMD[*]}" >> "$LOG_FILE"
exec "${CMD[@]}" >> "$LOG_FILE" 2>&1
