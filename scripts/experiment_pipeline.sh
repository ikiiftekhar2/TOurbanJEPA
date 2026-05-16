#!/bin/bash
#
# UrbanJEPA Experiment Pipeline — runs 4 experiments sequentially.
# Resume-safe: finds last checkpoint on boot, skips completed experiments.
#
# Usage:
#   Smoke test:  bash scripts/experiment_pipeline.sh smoke
#   Full train:  bash scripts/experiment_pipeline.sh train
#   Systemd:     systemctl start urbanjepa-pipeline
#
set -euo pipefail

PROJECT_DIR="/mnt/eskeetit/Code-server/UrbanJEPA"
VENV_PYTHON="/home/ubuntu/urbanjepa-venv/bin/python"
LOG_DIR="${PROJECT_DIR}/logs"
CKPT_DIR="${PROJECT_DIR}/models/checkpoints"
STATE_FILE="${PROJECT_DIR}/scripts/.pipeline_state"

mkdir -p "$LOG_DIR" "$CKPT_DIR"

# --- Experiment definitions ---
# Format: exp_name|regressor_type|hidden|unfreeze|epochs|lr_eta_min|warmstart_from|description
EXPERIMENTS=(
    "exp1_mlp|mlp|512|0|20|1e-5||MLP baseline (666K)"
    "exp2_conv|conv|512|0|20|1e-5||Conv refinement (2.5M)"
    "exp3_conv_wide|conv|1024|0|30|1e-6||Conv + wider hidden (~5M)"
    "exp4_joint|conv|1024|1|40|1e-6|exp3_conv_wide|Conv + wide + joint — warmstart from exp3 regressor"
)

COMMON_FLAGS=(
    --data_dir data/ortho
    --vae_decoder models/checkpoints/vae_decoder_best.pt
    --batch_size 96
    --lr 1e-3
    --log_dir runs
    --num_workers 8
    --patches_per_epoch 128
    --log_every 200
    --checkpoint_every 1
    --psnr_samples 128
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log() { echo "[$(date '+%a %b %d %H:%M:%S %Z %Y')] $*" | tee -a "$LOG_FILE"; }

is_exp_done() {
    local name="$1"
    # Experiment is done if final checkpoint exists
    [ -f "${CKPT_DIR}/${name}_final.pt" ]
}

find_latest_ckpt() {
    local name="$1"
    local latest=""
    local latest_epoch=-1

    # Check epoch checkpoints
    for ckpt in "${CKPT_DIR}/${name}"_epoch_*.pt; do
        if [ -f "$ckpt" ]; then
            local fname epoch_num
            fname=$(basename "$ckpt")
            epoch_str=${fname#${name}_epoch_}
            epoch_str=${epoch_str%.pt}
            epoch_num=$((epoch_str))
            if [ "$epoch_num" -gt "$latest_epoch" ]; then
                latest_epoch=$epoch_num
                latest=$ckpt
            fi
        fi
    done

    # Fall back to best.pt
    if [ -z "$latest" ] && [ -f "${CKPT_DIR}/${name}_best.pt" ]; then
        latest="${CKPT_DIR}/${name}_best.pt"
        latest_epoch="best"
    fi

    # Legacy fallback: old regress_*.pt naming (from before exp_name)
    if [ -z "$latest" ]; then
        for ckpt in "${CKPT_DIR}"/regress_epoch_*.pt; do
            if [ -f "$ckpt" ]; then
                local fname epoch_num
                fname=$(basename "$ckpt")
                epoch_str=${fname#regress_epoch_}
                epoch_str=${epoch_str%.pt}
                epoch_num=$((epoch_str))
                if [ "$epoch_num" -gt "$latest_epoch" ]; then
                    latest_epoch=$epoch_num
                    latest=$ckpt
                fi
            fi
        done
    fi
    if [ -z "$latest" ] && [ -f "${CKPT_DIR}/regress_best.pt" ]; then
        latest="${CKPT_DIR}/regress_best.pt"
        latest_epoch="legacy_best"
    fi

    echo "$latest_epoch:$latest"
}

# ---------------------------------------------------------------------------
# Smoke test mode — validate each experiment doesn't crash
# ---------------------------------------------------------------------------
smoke_test() {
    local LOG_FILE="${LOG_DIR}/smoke_test.log"
    echo "" >> "$LOG_FILE"
    log "===== SMOKE TEST START ====="

    local passed=0 failed=0

    for exp_def in "${EXPERIMENTS[@]}"; do
        IFS='|' read -r name rtype hidden unfreeze epochs lr_min warmstart_from desc <<< "$exp_def"

        log ""
        log "--- Testing: ${name} (${desc}) ---"

        local CMD=(
            "$VENV_PYTHON" -u -m src.training.train_regress
            --exp_name "$name"
            --regressor_type "$rtype"
            --regressor_hidden "$hidden"
            --epochs 1
            --max_steps 50
            --lr_eta_min "$lr_min"
            --no_early_stop
            "${COMMON_FLAGS[@]}"
        )

        if [ "$unfreeze" = "1" ]; then
            CMD+=(--unfreeze_jepa --lr_jepa 1e-4)
        fi

        log "  CMD: ${CMD[*]}"

        # Run and capture exit code
        local smoke_log="${LOG_DIR}/smoke_${name}.log"
        if "${CMD[@]}" >> "$smoke_log" 2>&1; then
            # Verify PSNR appears in output
            if grep -q "PSNR=" "$smoke_log"; then
                local psnr_line
                psnr_line=$(grep "PSNR=" "$smoke_log" | tail -1)
                log "  PASS — ${psnr_line}"
                passed=$((passed + 1))
            else
                log "  FAIL — no PSNR in output (VAE decode issue?)"
                failed=$((failed + 1))
            fi
        else
            log "  FAIL — crashed (exit code $?). Check ${smoke_log}"
            log "  Last 5 lines:"
            tail -5 "$smoke_log" | while read -r line; do log "    $line"; done
            failed=$((failed + 1))
        fi
    done

    log ""
    log "===== SMOKE TEST DONE: ${passed} passed, ${failed} failed ====="

    if [ "$failed" -gt 0 ]; then
        log "Fix failures before running 'train'."
        exit 1
    fi
    log "All smoke tests passed. Ready for 'train'."
}

# ---------------------------------------------------------------------------
# Train mode — run experiments sequentially, resume-safe
# ---------------------------------------------------------------------------
train() {
    local LOG_FILE="${LOG_DIR}/pipeline_train.log"
    echo "" >> "$LOG_FILE"
    log "===== PIPELINE TRAIN START ====="

    for exp_def in "${EXPERIMENTS[@]}"; do
        IFS='|' read -r name rtype hidden unfreeze epochs lr_min warmstart_from desc <<< "$exp_def"

        # Skip if already done
        if is_exp_done "$name"; then
            log "[${name}] Already done (${name}_final.pt exists). Skipping."
            continue
        fi

        log ""
        log "=== Running: ${name} — ${desc} ==="
        log "  Epochs: ${epochs}, LR eta_min: ${lr_min}"

        # Find checkpoint to resume from
        local ckpt_info
        ckpt_info=$(find_latest_ckpt "$name")
        local ckpt_epoch="${ckpt_info%%:*}"
        local ckpt_path="${ckpt_info#*:}"

        local CMD=(
            "$VENV_PYTHON" -u -m src.training.train_regress
            --exp_name "$name"
            --regressor_type "$rtype"
            --regressor_hidden "$hidden"
            --epochs "$epochs"
            --lr_eta_min "$lr_min"
            "${COMMON_FLAGS[@]}"
        )

        if [ "$unfreeze" = "1" ]; then
            CMD+=(--unfreeze_jepa --lr_jepa 1e-4)
        fi

        if [ -n "$ckpt_path" ]; then
            log "  Resuming from epoch ${ckpt_epoch}: ${ckpt_path}"
            CMD+=(--resume "$ckpt_path")
        else
            log "  Starting fresh (no checkpoint found)"
        fi

        # Warm-start regressor from a previous experiment's checkpoint (first run only)
        if [ -z "$ckpt_path" ] && [ -n "$warmstart_from" ]; then
            local ws_ckpt="${CKPT_DIR}/${warmstart_from}_best.pt"
            if [ ! -f "$ws_ckpt" ]; then
                ws_ckpt="${CKPT_DIR}/${warmstart_from}_final.pt"
            fi
            if [ -f "$ws_ckpt" ]; then
                log "  Warm-starting regressor from ${warmstart_from}: ${ws_ckpt}"
                CMD+=(--warmstart_regressor "$ws_ckpt")
            else
                log "  WARNING: warmstart checkpoint ${warmstart_from} not found, using random init"
            fi
        fi

        log "  Launching: ${CMD[*]}"

        # Run the experiment (this blocks until done or crash)
        local exp_log="${LOG_DIR}/train_${name}.log"
        if "${CMD[@]}" >> "$exp_log" 2>&1; then
            log "[${name}] Completed successfully."
        else
            local exit_code=$?
            log "[${name}] Exited with code ${exit_code}. Check ${exp_log}"
            log "  Last 10 lines:"
            tail -10 "$exp_log" | while read -r line; do log "    $line"; done

            # If it crashed, stop here so next boot can resume this experiment
            log "Pipeline stopping. Next boot will resume ${name}."
            exit 1
        fi

        # Verify final checkpoint was created
        if is_exp_done "$name"; then
            log "[${name}] Final checkpoint verified."
        else
            log "[${name}] WARNING: no final checkpoint found after successful exit."
            log "  Creating milestone marker anyway so pipeline advances."
        fi
    done

    log ""
    log "===== PIPELINE COMPLETE: all experiments done ====="
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
case "${1:-train}" in
    smoke) smoke_test ;;
    train)  train ;;
    *)
        echo "Usage: $0 {smoke|train}"
        exit 1
        ;;
esac
