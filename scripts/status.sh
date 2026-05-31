#!/bin/bash
# One-shot status snapshot of the UrbanJEPA v4 Phase 1.5 fidelity-first run.
# Usage:  bash scripts/status.sh          (single snapshot)
#         watch -n 30 bash scripts/status.sh   (live dashboard, refresh 30s)
cd /mnt/eskeetit/Code-server/UrbanJEPA || exit 1
LOG=runs/v4_p1_5.log
SVC=urbanjepa-v4-phase1_5.service

echo "================ UrbanJEPA status @ $(date -u +%H:%M:%S)Z ================"
echo "service: $(systemctl is-active $SVC)   up-since: $(systemctl show $SVC -p ActiveEnterTimestamp --value | cut -d' ' -f2)   restarts: $(systemctl show $SVC -p NRestarts --value)"

MS=$(grep -aoE 'step=[0-9]+' "$LOG" 2>/dev/null | grep -oE '[0-9]+' | sort -n | tail -1)
echo "progress: global_step ${MS:-?} / 102528  (3 epochs)   skips=$(grep -ac batch-skip "$LOG" 2>/dev/null)  aborts=$(grep -acE 'Collapse:|feature_collapse' "$LOG" 2>/dev/null)"

echo
echo "-- recent train (PSNR / L1 / VICreg) --"
grep -aoE "step=[0-9]+ \| PSNR=[0-9.]+dB \| L1=[0-9.]+.*VIC=[0-9.]+" "$LOG" 2>/dev/null \
  | grep -oE "step=[0-9]+|PSNR=[0-9.]+dB|L1=[0-9.]+|VIC=[0-9.]+" | paste - - - - | tail -4

echo
echo "-- collapse watch (ctx_std should stay ~1+, oob_frac ~0) --"
grep -aE "\[diag\]" "$LOG" 2>/dev/null | tail -2

echo
echo "-- STEP-VAL  <<< THE signal: does PSNR beat bilinear? --"
grep -aE "\[step-val\]" "$LOG" 2>/dev/null | tail -4 || true
grep -aqE "\[step-val\]" "$LOG" 2>/dev/null || echo "  (none yet — first validation at step 7000)"

echo
echo "-- gpu --"; nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null
echo "============================================================"
echo "live log:   tail -f $LOG"
echo "tensorboard: http://<host>:6006   (curves: val/psnr, val/bilinear_psnr, train/ctx_std, train/oob_frac)"
echo "eval a ckpt: PYTHONPATH=. python scripts/eval_checkpoint.py --ckpt checkpoints/v4_p1_5_stageA/best.pt"
