#!/bin/bash
# UrbanJEPA v4 Phase 1.5 — staged training launcher.
#
# Replaces the all-at-once Phase 1 recipe (which deterministically destabilized
# on 2026-05-27 with multiple crash loops + a silent collapse). Trains the v5
# decoder in three stages, each warm-starting from the previous epoch checkpoint:
#
#   Stage A (3 epochs): L1 + SSIM + JEPA + HF, NO perceptual loss.
#       Goal: stable PSNR baseline ~22-24 from clean pixel-fidelity training.
#       Gate: val_psnr >= 21 dB AND val_l1 <= 0.08 AND no_collapse.
#
#   Stage B (2 epochs): add LPIPS-VGG at half weight, lower LR.
#       Goal: introduce perceptual gradient on a well-trained decoder so it
#       doesn't blow up like it did in our all-at-once attempts.
#       Gate: val_lpips_vgg <= 0.50 AND val_psnr >= 20 dB AND no_collapse.
#
#   Stage C (2 epochs): full perceptual recipe, halved LR for fine-tune.
#       Goal: hit final Phase 1 production target.
#       Gate: val_lpips_vgg <= 0.45 AND val_psnr >= 20 dB AND val_ssim >= 0.55.
#
# Safety nets carried forward from the all-at-once attempts (2026-05-27):
#   - --use_v5_decoder (PixelDecoderV5: additive-residual output, single
#     bottleneck cross-attn + U-Net skips, GroupNorm(8)).
#   - --l1_spike_absolute 0.25 (absolute watchdog, no warmup, no window).
#   - --l1_spike_ratio 3.0 --l1_spike_window 200 (relative watchdog).
#   - --save_step_l1_max 0.20 (sacred-slot guard: don't overwrite healthy
#     step_slot weights with collapsed ones).
#   - L1 window persistence (in checkpoint, restored on resume).
#   - --seed $RANDOM (different shuffle each launch — breaks cursed-batch loops).
#   - fp32 attention inside decoder_v4._SelfAttention2D / _CrossAttentionBridge.
#   - --w_vgg 0 (LPIPS-VGG covers perceptual when active, no double-counting).
#
# State machine via checkpoints/v4_p1_5/stage.txt: "A" -> "B" -> "C" -> "DONE".
# On each launch:
#   1. Read current stage from state file (default "A").
#   2. Run train.py with stage-appropriate args + --resume.
#   3. If train.py exits 0 (epochs complete), evaluate gate.
#   4. If gate passes, advance stage. Otherwise systemd restarts the same stage.

set -euo pipefail

cd "$(dirname "$0")/.."

VENV_PY="${VENV_PY:-/home/ubuntu/urbanjepa-venv/bin/python}"
TB_BIN="${TB_BIN:-/home/ubuntu/urbanjepa-venv/bin/tensorboard}"
TB_PORT="${TB_PORT:-6006}"
TB_HOST="${TB_HOST:-0.0.0.0}"

# Debug allocation history (torch.cuda.memory._record_memory_history) is a
# diagnostic for the now-fixed forward-graph OOM leak. Off by default for the
# clean long run — it adds per-alloc bookkeeping overhead. Set UJ_MEM_DEBUG=1
# to re-enable if a memory issue resurfaces.
export UJ_MEM_DEBUG="${UJ_MEM_DEBUG:-0}"

BASE="v4_p1_5"
STATE_FILE="checkpoints/${BASE}/stage.txt"
mkdir -p "checkpoints/${BASE}" runs

STAGE=$(cat "$STATE_FILE" 2>/dev/null || echo "A")
echo "[$(date -Iseconds)] v4 Phase 1.5 launcher — stage=$STAGE"

if [ "$STAGE" = "DONE" ]; then
    echo "[$(date -Iseconds)] All stages complete. Exiting 0."
    exit 0
fi

# ----- Common args (best practices from 2026-05-27 debugging) -----
# Stage-aware batch size (set in the case "$STAGE" block below).
# Stage A: no LPIPS-VGG forward → ~1.9 GB/sample → batch 20 fits in 24 GB.
# Stages B/C: LPIPS-VGG adds ~3 GB of activations → batch 14 (~21 GB).
BATCH_SIZE_DEFAULT="${BATCH_SIZE:-}"
ENCODER_LR="${ENCODER_LR:-1e-5}"
# num_workers lowered 8 -> 4 on 2026-05-28 02:23. The 8 DataLoader workers
# each inherit a ~600 MB CUDA context (forked after model.cuda()), wasting
# ~5 GB of GPU memory they never use. Halving to 4 frees ~2.5 GB — the exact
# headroom the OOM cascade kept running out of.
# num_workers=8 with spawn (workers don't inherit CUDA context → zero GPU cost).
# 2-4 forked workers data-starved the GPU (idle ~80%); spawned 8 keep it fed.
NUM_WORKERS="${NUM_WORKERS:-8}"
DATALOADER_SPAWN="${DATALOADER_SPAWN:-1}"
# empty_cache every 200 steps: re-enabled after removing checkpointing. Without
# checkpointing, allocated is ~16.4 GB (model 15.42 + EMA-GPU 1); the reserved
# pool then creeps toward 23-24 GB and occasionally OOMs. Flushing every 200
# steps caps reserved near allocated. 200 (not 100) keeps the re-malloc
# overhead negligible vs throughput.
EMPTY_CACHE_EVERY="${EMPTY_CACHE_EVERY:-200}"
# Gradient checkpointing RE-ENABLED (2026-05-28 04:51). Without it, the LIVE
# peak hit 22.92 GB allocated (real-data transients + EMA + fragmentation,
# higher than the 15.42 GB synthetic profile) and OOM'd. The earlier "checkpointing
# is slow" conclusion was CONFOUNDED — that slow run had EMA on CPU (the real
# 180s/step bottleneck). With EMA now on GPU, checkpointing gives low allocated
# (~3.3-6 GB → zero OOM risk) at only ~1.5x compute cost. Safe AND workable.
USE_GRAD_CHECKPOINT="${USE_GRAD_CHECKPOINT:-1}"
# grad_clip tightened 0.3 -> 0.15 on 2026-05-27 21:42 after 4 crashes in
# ~85 min of Stage A all in the LR 5-6e-5 window. The crashes were Adam
# overshooting on hard batches; halving the gradient norm cap limits the
# worst-case per-step weight movement. Cost: ~10% slower convergence.
GRAD_CLIP="${GRAD_CLIP:-0.15}"
L1_SPIKE_RATIO="${L1_SPIKE_RATIO:-3.0}"
L1_SPIKE_WINDOW="${L1_SPIKE_WINDOW:-200}"
# 2026-05-27 22:35: thresholds loosened after 10 false-positive crashes in
# Stage A. Healthy L1 max during training was 0.096 (p99). Spike values were
# 0.22-0.31 — definitely abnormal for ONE batch but next batch usually returns
# to ~0.07. The fix: require N consecutive batches above threshold (only
# sustained spikes = real collapse), and raise absolute threshold modestly.
L1_SPIKE_ABSOLUTE="${L1_SPIKE_ABSOLUTE:-0.35}"
L1_SPIKE_CONSECUTIVE="${L1_SPIKE_CONSECUTIVE:-2}"
# Per-batch gradient skip: drop the backward+step for anomalous batches
# instead of letting them poison Adam momentum.
# KEEP AT 0.20 — this is load-bearing. Verified from the step-5700 checkpoint
# config (the model that trained cleanly through the step-200 dip used 0.20).
# At 0.20, the dip's elevated-L1 batches (0.20-0.35) get DROPPED so they can't
# drive Adam into collapse -> L1 stays flat at 0.08-0.14 and recovers. Loosening
# to 0.35 on 2026-05-28 let those batches train through and DETERMINISTICALLY
# collapsed the model (17->9 dB by step ~400) at both bs18 and bs22. The
# "skip-starvation death-spiral" that motivated loosening is a symptom of an
# ALREADY-collapsed model and is now handled by the anti-spin guard in train.py
# (abort after N consecutive skips -> systemd restart + fresh seed).
SKIP_HIGH_L1="${SKIP_HIGH_L1:-0.20}"
# Per-sample outlier rejection: drop INDIVIDUAL samples with L1 > 0.30 from the
# pixel losses (L1/SSIM/VGG/HF/LPIPS/OOB); JEPA stays on full batch. Handles
# the "1 hard tile in 18" case without dumping the whole batch.
# Per-sample masking DISABLED 2026-05-28 02:03 — pred[keep_mask] creates
# a copy that doubles activation memory and causes OOMs at every batch
# size we've tried. Fall back to whole-batch skip via --skip_high_l1.
# Will revisit with a memory-efficient implementation (weighted reduction)
# in a follow-up if outlier rejection turns out to be needed.
PER_SAMPLE_L1_MAX="${PER_SAMPLE_L1_MAX:-0}"
SAVE_STEP_L1_MAX="${SAVE_STEP_L1_MAX:-0.20}"
EMA_START_STEP="${EMA_START_STEP:-4000}"
EMA_WARMUP_STEPS="${EMA_WARMUP_STEPS:-20000}"
PATCHES_PER_EPOCH="${PATCHES_PER_EPOCH:-128}"
# Step-based validation cadence. Lowered to 2500 (from train.py default 7000)
# for fast feedback while confirming the cross-sensor-aug fix closes the
# train/val gap. Raise back toward 7000 once the foundation is trusted.
VAL_EVERY_STEPS="${VAL_EVERY_STEPS:-2500}"
SEED="${SEED:-$RANDOM}"
# Out-of-range penalty. The wiring fix (decoder_v5 exposes pred_unclamped;
# train.py feeds it to out_of_range_loss) makes this actually nonzero now, but
# W_OOB 0.05: with the tanh R=1.0 bound, out = low_res + tanh(delta) can still
# reach (-1, 2), so the decoder CAN drift out of [0,1] (oob_frac spiked to 0.068
# and triggered abort cascades at w_oob=0). A small OOB penalty discourages those
# excursions. (The cascade itself is fixed by judging skip/watchdog on CLAMPED L1
# in train.py; this is the preventive belt-and-suspenders.)
W_OOB="${W_OOB:-0.05}"
# VICReg variance+covariance anti-collapse on the predictor's projected features —
# the JEPA otherwise has NO explicit force against representation collapse.
W_VICREG="${W_VICREG:-0}"  # was 0.02; encoder frozen so no anti-collapse needed

# ----- Stage-specific args -----
case "$STAGE" in
    A)
        EXP_NAME="${BASE}_stageA"
        # batch=20 (2026-05-28 05:45). Root cause of prior OOMs was fixed:
        # PYTORCH_CUDA_ALLOC_CONF now expandable_segments:True ONLY (the
        # max_split_size_mb:128 knob had ballooned reserved to 23.9 GB).
        # mem_trace.py FLAT/no-leak floors: bs20 peak 19.5 GB, bs22 peak 21.0 GB.
        # The earlier bs22 "OOM at 22.8 GB" was NOT footprint — it was the
        # forward-graph leak on the skip paths (train.py high-L1/NaN/full-batch
        # `continue` skipped backward() and kept the graph alive into the next
        # forward -> 2x activations). Fixed 2026-05-28: graph freed before each
        # skip `continue`.
        # PERCEPTUAL LOSS IS LOAD-BEARING FOR STAGE A (verified 2026-05-28 from
        # the 05-27 proven checkpoint config: w_lpips=1.0, lpips_net=vgg). The
        # "Stage A = pixel-only (no perceptual)" design was the bug: L1+SSIM
        # alone cannot hold image structure and DETERMINISTICALLY collapses at
        # the step-200 dip (PSNR 17->9 dB) across every seed/batch/skip config.
        # The proven run had LPIPS-VGG @ 1.0 from the start, learned faster
        # (19.6 dB @ step 101 vs 15 without), and its L1 RECOVERED through the
        # dip instead of running away. Re-enabled below. Batch was a red herring.
        # batch 14 (REVERTED from bs=20 on 2026-05-29 22:10 after empirical test:
        # bs=20 used 22 GB GPU as predicted but throughput was 22 samples/s vs
        # bs=14's 28 samples/s (~22% slower per sample), and peak alloc hit
        # 23.24 GB / 24 GB cap (0.7 GB headroom). The slowdown is allocator
        # thrashing + LPIPS-VGG not amortizing at this batch. bs=14 is the
        # known-fastest at this config.
        BATCH_SIZE="${BATCH_SIZE_DEFAULT:-20}"  # 2026-05-30 22:55: bumped 14->20
        # for Stage A (no LPIPS, no JEPA loss compute, no EMA). Memory headroom
        # at bs=14 was ~7 GB; bs=20 keeps us under 23.5 GB with v3 decoder.
        # lr 7.5e-5 (not the proven run's 1.5e-4): the proven run hit LR-zone
        # spikes ~step 2900 at 1.5e-4; 7.5e-5 avoids that late-phase instability
        # while LPIPS-VGG provides the early-phase stability.
        LR="7.5e-5"
        EPOCHS="3"
        # FROZEN-ENCODER Phase-2-style restart (2026-05-30 16:30). All 4 decoder
        # variants tried today plateaued at PSNR 15-19 by step 350-700 with the
        # joint encoder+predictor+decoder training recipe. v3 PixelDecoderWithSkips
        # peaked at 23.14 then fell back to 17. Removing perceptual losses didn't
        # help in prior attempts. The only untested variable: freeze the encoder
        # side entirely and train decoder ONLY against pretrained DINOv2 features.
        W_L1="2.0"
        W_SSIM="0.3"      # LIGHT structural signal — was 1.0 (dragged PSNR down
                          # via perception-distortion tradeoff once decoder learned)
        W_VGG="0"
        W_HF="0"          # already validated as destabilizer
        W_JEPA="0"        # NO JEPA loss — encoder is frozen, nothing to learn
        # No perceptual losses for this restart. Pure pixel fidelity to test if
        # the frozen-encoder hypothesis escapes the PSNR-17 plateau. If it does,
        # we can re-introduce perceptual losses later (Stage B/C).
        LPIPS_FLAGS=()
        SALVAGE_FLAGS=()
        ;;
    B)
        EXP_NAME="${BASE}_stageB"
        BATCH_SIZE="${BATCH_SIZE_DEFAULT:-14}"  # LPIPS-VGG adds ~3 GB → batch 14 is safe
        LR="7.5e-5"
        EPOCHS="2"
        W_L1="2.0"
        W_SSIM="0.5"
        W_VGG="0"
        W_HF="0"
        W_JEPA="0.1"
        LPIPS_FLAGS=(--use_lpips --lpips_net vgg --w_lpips 0.5 --l1_warmup_steps 0)
        # If this is the FIRST launch of stage B (no checkpoint yet), salvage from stage A
        if [ ! -f "checkpoints/${EXP_NAME}/manifest.json" ] || \
           [ "$(python3 -c "import json; m=json.load(open('checkpoints/${EXP_NAME}/manifest.json')); print(len(m.get('step_checkpoints',[])))" 2>/dev/null)" = "0" ]; then
            STAGE_A_BEST="checkpoints/${BASE}_stageA/best.pt"
            if [ -f "$STAGE_A_BEST" ]; then
                SALVAGE_FLAGS=(--salvage_from "$STAGE_A_BEST")
                echo "[$(date -Iseconds)] Stage B cold start — salvaging encoder/predictor from $STAGE_A_BEST"
            else
                # Fall back to latest epoch from stage A
                STAGE_A_LATEST=$(ls -t checkpoints/${BASE}_stageA/epoch_*.pt 2>/dev/null | head -1 || true)
                if [ -z "$STAGE_A_LATEST" ]; then
                    echo "[$(date -Iseconds)] ERROR: Stage B requires Stage A checkpoint, none found." >&2
                    exit 1
                fi
                SALVAGE_FLAGS=(--salvage_from "$STAGE_A_LATEST")
                echo "[$(date -Iseconds)] Stage B cold start — salvaging from $STAGE_A_LATEST"
            fi
        else
            SALVAGE_FLAGS=()
        fi
        ;;
    C)
        EXP_NAME="${BASE}_stageC"
        BATCH_SIZE="${BATCH_SIZE_DEFAULT:-14}"  # LPIPS-VGG → batch 14
        LR="5e-5"
        EPOCHS="2"
        W_L1="2.0"
        W_SSIM="0.5"
        W_VGG="0"
        W_HF="0"
        W_JEPA="0.1"
        LPIPS_FLAGS=(--use_lpips --lpips_net vgg --w_lpips 1.0 --l1_warmup_steps 0)
        if [ ! -f "checkpoints/${EXP_NAME}/manifest.json" ] || \
           [ "$(python3 -c "import json; m=json.load(open('checkpoints/${EXP_NAME}/manifest.json')); print(len(m.get('step_checkpoints',[])))" 2>/dev/null)" = "0" ]; then
            STAGE_B_BEST="checkpoints/${BASE}_stageB/best.pt"
            if [ -f "$STAGE_B_BEST" ]; then
                SALVAGE_FLAGS=(--salvage_from "$STAGE_B_BEST")
            else
                STAGE_B_LATEST=$(ls -t checkpoints/${BASE}_stageB/epoch_*.pt 2>/dev/null | head -1 || true)
                if [ -z "$STAGE_B_LATEST" ]; then
                    echo "[$(date -Iseconds)] ERROR: Stage C requires Stage B checkpoint, none found." >&2
                    exit 1
                fi
                SALVAGE_FLAGS=(--salvage_from "$STAGE_B_LATEST")
            fi
        else
            SALVAGE_FLAGS=()
        fi
        ;;
    *)
        echo "[$(date -Iseconds)] ERROR: Unknown stage '$STAGE'" >&2
        exit 1
        ;;
esac

LOG_PATH="runs/${BASE}.log"

# ----- TensorBoard (background) -----
TB_LOG="runs/tensorboard.log"
echo "[$(date -Iseconds)] launching tensorboard on ${TB_HOST}:${TB_PORT}, logdir=runs/" \
    | tee -a "$TB_LOG"
"$TB_BIN" --logdir runs --host "$TB_HOST" --port "$TB_PORT" \
    --reload_interval 30 >> "$TB_LOG" 2>&1 &
TB_PID=$!
echo "[$(date -Iseconds)] tensorboard pid=$TB_PID" | tee -a "$TB_LOG"

cleanup() {
    if kill -0 "$TB_PID" 2>/dev/null; then
        echo "[$(date -Iseconds)] stopping tensorboard pid=$TB_PID" >> "$TB_LOG"
        kill "$TB_PID" 2>/dev/null || true
        wait "$TB_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# ----- Run train.py for current stage -----
echo "[$(date -Iseconds)] === STAGE $STAGE start ===" | tee -a "$LOG_PATH"
echo "[$(date -Iseconds)] exp=$EXP_NAME epochs=$EPOCHS lr=$LR seed=$SEED" | tee -a "$LOG_PATH"
echo "[$(date -Iseconds)] w_l1=$W_L1 w_ssim=$W_SSIM w_vgg=$W_VGG w_hf=$W_HF w_jepa=$W_JEPA" | tee -a "$LOG_PATH"
echo "[$(date -Iseconds)] lpips: ${LPIPS_FLAGS[*]:-disabled}" | tee -a "$LOG_PATH"
echo "[$(date -Iseconds)] salvage: ${SALVAGE_FLAGS[*]:-none}" | tee -a "$LOG_PATH"

set +e
"$VENV_PY" -m src.training.train \
    --data_dir data/ortho \
    --backbone dinov2_vitb14 \
    --pretrained_path models/pretrained/dinov2_vitb14.pth \
    --experiment_name "$EXP_NAME" \
    --match_train_aug_in_val \
    --tile_manifest data/ortho/metadata/train_textured.txt \
    --epochs "$EPOCHS" --batch_size "$BATCH_SIZE" \
    --lr "$LR" --encoder_lr "$ENCODER_LR" \
    --num_workers "$NUM_WORKERS" \
    --empty_cache_every "$EMPTY_CACHE_EVERY" \
    $( [ "$USE_GRAD_CHECKPOINT" = "1" ] && echo "--use_grad_checkpoint" ) \
    $( [ "$DATALOADER_SPAWN" = "1" ] && echo "--dataloader_spawn" ) \
    --grad_clip "$GRAD_CLIP" \
    --l1_spike_ratio "$L1_SPIKE_RATIO" \
    --l1_spike_window "$L1_SPIKE_WINDOW" \
    --l1_spike_absolute "$L1_SPIKE_ABSOLUTE" \
    --l1_spike_consecutive "$L1_SPIKE_CONSECUTIVE" \
    --skip_high_l1 "$SKIP_HIGH_L1" \
    --per_sample_l1_max "$PER_SAMPLE_L1_MAX" \
    --save_step_l1_max "$SAVE_STEP_L1_MAX" \
    --w_l1 "$W_L1" --w_ssim "$W_SSIM" --w_vgg "$W_VGG" \
    --w_hf "$W_HF" --w_jepa "$W_JEPA" --w_oob "$W_OOB" --w_vicreg "$W_VICREG" \
    --residual_range_start "${RESIDUAL_RANGE_START:-0.05}" \
    --residual_range_end "${RESIDUAL_RANGE_END:-0.20}" \
    --residual_range_ramp_start "${RESIDUAL_RANGE_RAMP_START:-5000}" \
    --residual_range_ramp_end "${RESIDUAL_RANGE_RAMP_END:-15000}" \
    --seed "$SEED" \
    --patches_per_epoch "$PATCHES_PER_EPOCH" \
    --val_every_steps "$VAL_EVERY_STEPS" \
    --precision bf16 \
    "${LPIPS_FLAGS[@]}" "${SALVAGE_FLAGS[@]}" \
    --resume 2>&1 | tee -a "$LOG_PATH"
TRAIN_EXIT=${PIPESTATUS[0]}
set -e

echo "[$(date -Iseconds)] === STAGE $STAGE train.py exit=$TRAIN_EXIT ===" | tee -a "$LOG_PATH"

# ----- If training exited 0 (epochs complete), evaluate gate -----
if [ "$TRAIN_EXIT" -eq 0 ]; then
    if "$VENV_PY" scripts/check_stage_gate.py --stage "$STAGE" --exp "$EXP_NAME" \
            2>&1 | tee -a "$LOG_PATH"; then
        # Gate passed — advance stage
        case "$STAGE" in
            A) NEXT="B" ;;
            B) NEXT="C" ;;
            C) NEXT="DONE" ;;
        esac
        echo "$NEXT" > "$STATE_FILE"
        echo "[$(date -Iseconds)] === STAGE $STAGE PASSED gate, advancing to $NEXT ===" | tee -a "$LOG_PATH"
    else
        echo "[$(date -Iseconds)] === STAGE $STAGE FAILED gate, will retry on next systemd start ===" | tee -a "$LOG_PATH"
        # Don't advance; systemd Restart=on-failure won't restart on exit 0,
        # but operator can manually restart to retry the same stage.
        exit 1
    fi
fi

# Non-zero exit (crash) — systemd auto-restart will pick up the same stage
exit $TRAIN_EXIT
