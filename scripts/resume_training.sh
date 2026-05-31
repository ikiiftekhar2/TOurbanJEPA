#!/bin/bash
# Launch UrbanJEPA v3 training and let the manifest auto-resume from the latest
# step / epoch checkpoint. Customize BACKBONE / EXPERIMENT_NAME / PRETRAINED for
# the active run.
set -euo pipefail

cd "$(dirname "$0")/.."

VENV_PY="${VENV_PY:-/home/ubuntu/urbanjepa-venv/bin/python}"
BACKBONE="${BACKBONE:-imagenet_vitb16}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-backbone_${BACKBONE}}"
PRETRAINED="${PRETRAINED:-models/pretrained/imagenet_vitb16.pt}"

mkdir -p runs

"$VENV_PY" -m src.training.train \
    --data_dir data/ortho \
    --backbone "$BACKBONE" \
    --pretrained_path "$PRETRAINED" \
    --experiment_name "$EXPERIMENT_NAME" \
    --epochs 30 \
    --batch_size 22 \
    --lr 3e-4 \
    --encoder_lr 1e-5 \
    --patches_per_epoch 128 \
    --checkpoint_dir checkpoints \
    --log_dir runs \
    --resume 2>&1 | tee -a "runs/training_${EXPERIMENT_NAME}.log"
