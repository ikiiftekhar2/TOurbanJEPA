#!/bin/bash
# Run the 3-backbone comparison experiment with auto-resume.
# Each experiment manifest is independently resumable; the runner skips any
# experiment whose manifest reports the target epoch complete.
set -euo pipefail

cd "$(dirname "$0")/.."

VENV_PY="${VENV_PY:-/home/ubuntu/urbanjepa-venv/bin/python}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-22}"
PATCHES_PER_EPOCH="${PATCHES_PER_EPOCH:-128}"

mkdir -p runs

"$VENV_PY" scripts/run_backbone_experiment.py \
    --data_dir data/ortho \
    --checkpoint_dir checkpoints \
    --log_dir runs \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --patches_per_epoch "$PATCHES_PER_EPOCH" \
    --python "$VENV_PY" \
    --skip_missing_weights 2>&1 | tee -a runs/backbone_experiment.log
