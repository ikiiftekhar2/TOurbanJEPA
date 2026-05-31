#!/bin/bash
# Install systemd unit for v4 Phase 1 (Tier-1 validation).
# Runs scripts/resume_v4_phase1.sh which auto-resumes training and starts
# TensorBoard on port 6006. On crash, systemd restarts the script and the
# CheckpointManager's --resume picks up from the latest step_slot_*.pt
# (with 500-step rollback).
#
# Once training reaches --epochs 5 the unit exits 0 and does NOT restart.
set -euo pipefail

REPO_DIR="${REPO_DIR:-/mnt/eskeetit/Code-server/UrbanJEPA}"
USER_NAME="${SUDO_USER:-ubuntu}"
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-23}"
TB_PORT="${TB_PORT:-6006}"
SERVICE=/etc/systemd/system/urbanjepa-v4-phase1.service

cat > "$SERVICE" <<EOF
[Unit]
Description=UrbanJEPA v4 Phase 1 Tier-1 validation (auto-resume) + TensorBoard
After=network-online.target mnt-eskeetit.mount
Wants=network-online.target mnt-eskeetit.mount

[Service]
Type=simple
User=$USER_NAME
WorkingDirectory=$REPO_DIR
Environment=EPOCHS=$EPOCHS
Environment=BATCH_SIZE=$BATCH_SIZE
Environment=TB_PORT=$TB_PORT
Environment=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ExecStartPre=/bin/sleep 10
ExecStart=$REPO_DIR/scripts/resume_v4_phase1.sh
Restart=on-failure
RestartSec=30
StandardOutput=append:$REPO_DIR/runs/systemd_v4_phase1.log
StandardError=append:$REPO_DIR/runs/systemd_v4_phase1.log

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable urbanjepa-v4-phase1.service
echo "Installed urbanjepa-v4-phase1.service"
echo "  Epochs:    $EPOCHS"
echo "  BatchSize: $BATCH_SIZE"
echo "  TBPort:    $TB_PORT (logdir=$REPO_DIR/runs)"
echo "  Enabled:   $(systemctl is-enabled urbanjepa-v4-phase1.service)"
echo ""
echo "Start now:        sudo systemctl start urbanjepa-v4-phase1.service"
echo "Live train logs:  tail -f $REPO_DIR/runs/v4_phase1_tier1.log"
echo "Live system logs: journalctl -u urbanjepa-v4-phase1.service -f"
echo "TensorBoard:      http://<host>:$TB_PORT"
