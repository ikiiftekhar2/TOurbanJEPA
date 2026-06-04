#!/bin/bash
# UrbanJEPA v5 — JEPA-Conditioned ESRGAN, staged launcher.
#
# Per plan.md Phase 3:
#   Stage A: L1 + LPIPS-VGG (no GAN). Warm up the JEPA-injection layers on
#            top of frozen-then-low-LR pretrained RRDBNet. ~6-8h target.
#   Stage B: + U-Net discriminator GAN loss. Adversarial fine-tune for
#            sharpness. Resumes G (full V5Model) from Stage A's best.pt;
#            D starts fresh. ~8-12h target.
#
# Both stages are run by ONE systemd unit (urbanjepa-v5.service). The unit
# is Type=simple, Restart=on-failure — so a crash is auto-restarted; clean
# exit advances stage via $STATE_FILE. Per-step rotating slot checkpoints
# (recovery_every=200) plus the v4 rollback-on-resume keep us safe.
#
# State machine: checkpoints/v5_jepa_esrgan/stage.txt  ∈ {A, B, DONE}.
# Same convention as v4 Phase 1.5.

set -euo pipefail
cd "$(dirname "$0")/.."

VENV_PY="${VENV_PY:-/home/ubuntu/urbanjepa-venv/bin/python}"
TB_BIN="${TB_BIN:-/home/ubuntu/urbanjepa-venv/bin/tensorboard}"
TB_PORT="${TB_PORT:-6006}"

# GPU-saturating defaults for a single RTX 3090 (23.56 GiB usable).
# Calibrated by scripts/v5_gpu_probe.py at TWO phases each:
#   warmup = RRDBNet frozen (steps 0..warmup_steps, no RRDB activations stored)
#   steady = RRDBNet unfrozen (rest of training, full backward graph)
# The STEADY ceiling is the binding constraint — pick bs that fits there.
#
#   Stage A: steady max bs=20 → 19.25 GiB peak (82% util). bs=24 → OOM.
#            warmup at bs=20 would peak ~10.9 GiB; bs=40 fits warmup at 19.5 GiB,
#            but changing bs mid-run is more trouble than it's worth.
#   Stage B: steady max bs=20 → 22.10 GiB peak (94% util — too tight for long-run
#            with EMA-apply during val, LPIPS init transients, allocator
#            fragmentation creep). Pick bs=18 for ~5 GiB headroom on the GAN
#            forward+backward chain.
#   Val: bs=32 is comfortable (no grads, no optimizer, no D step).
#
# Override at launch: BATCH_SIZE_B=20 STAGE_OVERRIDE=B ./run_v5_jepa_esrgan.sh
BATCH_SIZE_A="${BATCH_SIZE_A:-${BATCH_SIZE:-20}}"
BATCH_SIZE_B="${BATCH_SIZE_B:-${BATCH_SIZE:-18}}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EMPTY_CACHE_EVERY="${EMPTY_CACHE_EVERY:-200}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-200}"
VAL_EVERY_STEPS="${VAL_EVERY_STEPS:-1000}"
LOG_EVERY="${LOG_EVERY:-25}"
RECOVERY_SLOTS="${RECOVERY_SLOTS:-6}"
# On resume, walk back this many steps from latest ckpt. 0 = no rollback
# (resume from most recent slot exactly). Default 200 = v4-style safety.
ROLLBACK_STEPS="${ROLLBACK_STEPS:-200}"
# When set to 1, the next resume re-snapshots EMA shadow from the LIVE model
# (drops the saved shadow). Use after EMA corruption / cold-start of EMA.
RESET_EMA_ON_RESUME="${RESET_EMA_ON_RESUME:-0}"
TILE_MANIFEST="${TILE_MANIFEST:-data/ortho/metadata/train_textured.txt}"

# --- Stage A: pixel + perceptual ---
WARMUP_STEPS_A="${WARMUP_STEPS_A:-1000}"
EPOCHS_A="${EPOCHS_A:-3}"
LR_INJECTION_A="${LR_INJECTION_A:-1e-4}"
LR_RRDBNET_A="${LR_RRDBNET_A:-1e-5}"
LR_JEPA_ENC_A="${LR_JEPA_ENC_A:-1e-5}"
LR_JEPA_PP_A="${LR_JEPA_PP_A:-1e-4}"
W_L1_A="${W_L1_A:-1.0}"
W_LPIPS_A="${W_LPIPS_A:-0.1}"

# --- Stage B: + GAN ---
WARMUP_STEPS_B="${WARMUP_STEPS_B:-0}"        # G already warmed in Stage A
GAN_WARMUP_STEPS_B="${GAN_WARMUP_STEPS_B:-200}"  # D-only burn-in
EPOCHS_B="${EPOCHS_B:-2}"
LR_INJECTION_B="${LR_INJECTION_B:-5e-5}"
LR_RRDBNET_B="${LR_RRDBNET_B:-5e-6}"
LR_JEPA_ENC_B="${LR_JEPA_ENC_B:-5e-6}"
LR_JEPA_PP_B="${LR_JEPA_PP_B:-5e-5}"
LR_D_B="${LR_D_B:-1e-4}"
W_L1_B="${W_L1_B:-1.0}"
W_LPIPS_B="${W_LPIPS_B:-0.1}"
W_GAN_B="${W_GAN_B:-0.1}"

BASE="v5_jepa_esrgan"
STATE_FILE="checkpoints/${BASE}/stage.txt"
mkdir -p "checkpoints/${BASE}" runs

# Build optional-flag array up here. Doing this inside the run_train_common
# array literal via `$([...] && echo ...)` propagates the failing test's
# exit 1 under `set -e` and silently kills the launcher (caused a 100+
# restart crash loop on 2026-05-31 once RESET_EMA_ON_RESUME flipped back to 0).
RESET_EMA_FLAGS=()
if [ "$RESET_EMA_ON_RESUME" = "1" ]; then
    RESET_EMA_FLAGS=(--reset_ema_on_resume)
fi

STAGE=$(cat "$STATE_FILE" 2>/dev/null || echo "A")
if [ -n "${STAGE_OVERRIDE:-}" ]; then
    STAGE="$STAGE_OVERRIDE"
fi
echo "[$(date -Iseconds)] v5 JEPA-ESRGAN launcher — stage=$STAGE" | tee -a "runs/${BASE}.log"

if [ "$STAGE" = "DONE" ]; then
    echo "[$(date -Iseconds)] All stages complete. Exiting 0." | tee -a "runs/${BASE}.log"
    exit 0
fi

# Background TensorBoard (idempotent).
if ! pgrep -f "tensorboard.*--port=${TB_PORT}" > /dev/null 2>&1; then
    "${TB_BIN}" --logdir runs --port="${TB_PORT}" --host=0.0.0.0 \
        >/dev/null 2>&1 &
    disown || true
    echo "[$(date -Iseconds)] tensorboard started on :${TB_PORT}" | tee -a "runs/${BASE}.log"
fi

# Find the best Stage-A checkpoint for Stage-B warm-start.
STAGE_A_EXP="${BASE}_stageA"
STAGE_B_EXP="${BASE}_stageB"
STAGE_A_BEST="checkpoints/${STAGE_A_EXP}/best.pt"

run_train_common=(
    "$VENV_PY" -m src.training.train
    --ortho_dir data/ortho
    --tile_manifest "$TILE_MANIFEST"
    --batch_size "$BATCH_SIZE_A"
    --val_batch_size "$VAL_BATCH_SIZE"
    --num_workers "$NUM_WORKERS"
    --backbone dinov2_vitb14
    --jepa_pretrained models/pretrained/dinov2_vitb14.pth
    --esrgan_weights models/pretrained/RealESRGAN_x4plus.pth
    --grad_clip "$GRAD_CLIP"
    --save_every_steps "$SAVE_EVERY_STEPS"
    --val_every_steps "$VAL_EVERY_STEPS"
    --log_every "$LOG_EVERY"
    --recovery_slots "$RECOVERY_SLOTS"
    --rollback_steps "$ROLLBACK_STEPS"
    "${RESET_EMA_FLAGS[@]}"
    --empty_cache_every "$EMPTY_CACHE_EVERY"
    --use_ema
    --bf16
    --resume
)

case "$STAGE" in
    A)
        echo "[$(date -Iseconds)] Stage A (L1+LPIPS, no GAN): batch=$BATCH_SIZE_A val_batch=$VAL_BATCH_SIZE epochs=$EPOCHS_A" \
            | tee -a "runs/${BASE}.log"
        "${run_train_common[@]}" \
            --exp_name "$STAGE_A_EXP" \
            --checkpoint_dir checkpoints \
            --log_dir runs \
            --batch_size "$BATCH_SIZE_A" \
            --warmup_steps "$WARMUP_STEPS_A" \
            --epochs "$EPOCHS_A" \
            --injection_lr "$LR_INJECTION_A" \
            --rrdbnet_lr "$LR_RRDBNET_A" \
            --jepa_encoder_lr "$LR_JEPA_ENC_A" \
            --jepa_pred_proj_lr "$LR_JEPA_PP_A" \
            --w_l1 "$W_L1_A" \
            --w_lpips "$W_LPIPS_A" \
            --lpips_net vgg

        echo "[$(date -Iseconds)] Stage A train.py exited 0 — evaluating gate" \
            | tee -a "runs/${BASE}.log"
        if "$VENV_PY" scripts/check_v5_stage_gate.py --stage A --exp "$STAGE_A_EXP"; then
            echo "B" > "$STATE_FILE"
            echo "[$(date -Iseconds)] Stage A PASSED — advancing to B" \
                | tee -a "runs/${BASE}.log"
            exit 0
        else
            echo "[$(date -Iseconds)] Stage A FAILED gate — exit 1 (no advance)" \
                | tee -a "runs/${BASE}.log"
            exit 1
        fi
        ;;

    B)
        if [ ! -f "$STAGE_A_BEST" ]; then
            echo "[$(date -Iseconds)] FATAL: Stage A best.pt missing ($STAGE_A_BEST). " \
                "Reverting to Stage A." | tee -a "runs/${BASE}.log"
            echo "A" > "$STATE_FILE"
            exit 1
        fi
        echo "[$(date -Iseconds)] Stage B (+GAN): batch=$BATCH_SIZE_B val_batch=$VAL_BATCH_SIZE epochs=$EPOCHS_B" \
            | tee -a "runs/${BASE}.log"
        "${run_train_common[@]}" \
            --exp_name "$STAGE_B_EXP" \
            --checkpoint_dir checkpoints \
            --log_dir runs \
            --batch_size "$BATCH_SIZE_B" \
            --warmup_steps "$WARMUP_STEPS_B" \
            --epochs "$EPOCHS_B" \
            --injection_lr "$LR_INJECTION_B" \
            --rrdbnet_lr "$LR_RRDBNET_B" \
            --jepa_encoder_lr "$LR_JEPA_ENC_B" \
            --jepa_pred_proj_lr "$LR_JEPA_PP_B" \
            --w_l1 "$W_L1_B" \
            --w_lpips "$W_LPIPS_B" \
            --w_gan "$W_GAN_B" \
            --d_lr "$LR_D_B" \
            --gan_warmup_steps "$GAN_WARMUP_STEPS_B" \
            --use_gan \
            --resume_g_from "$STAGE_A_BEST" \
            --lpips_net vgg

        echo "[$(date -Iseconds)] Stage B train.py exited 0 — evaluating gate" \
            | tee -a "runs/${BASE}.log"
        if "$VENV_PY" scripts/check_v5_stage_gate.py --stage B --exp "$STAGE_B_EXP"; then
            echo "DONE" > "$STATE_FILE"
            echo "[$(date -Iseconds)] Stage B PASSED — pipeline DONE" \
                | tee -a "runs/${BASE}.log"
            exit 0
        else
            echo "[$(date -Iseconds)] Stage B FAILED gate — exit 1 (no advance)" \
                | tee -a "runs/${BASE}.log"
            exit 1
        fi
        ;;

    *)
        echo "[$(date -Iseconds)] FATAL: unknown stage '$STAGE'. Aborting." \
            | tee -a "runs/${BASE}.log"
        exit 1
        ;;
esac
