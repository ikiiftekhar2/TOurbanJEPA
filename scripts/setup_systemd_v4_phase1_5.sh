#!/bin/bash
# Install systemd unit for v4 Phase 1.5 (staged training).
# Runs scripts/run_v4_phase1_5.sh which executes one stage at a time per
# launch. On non-zero exit (crash), systemd auto-restarts the same stage.
# On clean exit after gate pass, the stage state file advances; on clean
# exit after gate fail, exit 1 keeps systemd from looping infinitely.
set -euo pipefail

REPO_DIR="${REPO_DIR:-/mnt/eskeetit/Code-server/UrbanJEPA}"
USER_NAME="${SUDO_USER:-ubuntu}"
SERVICE=/etc/systemd/system/urbanjepa-v4-phase1_5.service

cat > "$SERVICE" <<EOF
[Unit]
Description=UrbanJEPA v4 Phase 1.5 staged training (A -> B -> C)
After=network-online.target mnt-eskeetit.mount
Wants=network-online.target mnt-eskeetit.mount

[Service]
Type=simple
User=$USER_NAME
WorkingDirectory=$REPO_DIR
Environment=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ExecStartPre=/bin/sleep 10
ExecStart=$REPO_DIR/scripts/run_v4_phase1_5.sh
Restart=on-failure
RestartSec=30
StandardOutput=append:$REPO_DIR/runs/systemd_v4_p1_5.log
StandardError=append:$REPO_DIR/runs/systemd_v4_p1_5.log

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable urbanjepa-v4-phase1_5.service 2>/dev/null || true
echo "Installed urbanjepa-v4-phase1_5.service"
echo "  Enabled:   $(systemctl is-enabled urbanjepa-v4-phase1_5.service)"
echo ""
echo "Start:            sudo systemctl start urbanjepa-v4-phase1_5.service"
echo "Stop:             sudo systemctl stop urbanjepa-v4-phase1_5.service"
echo "Live train log:   tail -f $REPO_DIR/runs/v4_p1_5.log"
echo "Live system log:  journalctl -u urbanjepa-v4-phase1_5.service -f"
echo "TensorBoard:      http://<host>:6006"
echo "Current stage:    cat $REPO_DIR/checkpoints/v4_p1_5/stage.txt"
