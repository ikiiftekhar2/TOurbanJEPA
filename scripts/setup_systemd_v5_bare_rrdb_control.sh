#!/bin/bash
# Install systemd unit for the bare-RRDBNet fine-tune control.
# One-shot training (4 epochs ~ 15k steps). No staged gates — clean exit = done.

set -euo pipefail

REPO_DIR="${REPO_DIR:-/mnt/eskeetit/Code-server/UrbanJEPA}"
USER_NAME="${SUDO_USER:-ubuntu}"
SERVICE_FILE=/etc/systemd/system/urbanjepa-v5-bare.service

cat > "$SERVICE_FILE" <<EOF
[Unit]
Description=UrbanJEPA v5 bare-RRDBNet fine-tune control
After=network-online.target mnt-eskeetit.mount
Wants=network-online.target mnt-eskeetit.mount

[Service]
Type=simple
User=$USER_NAME
WorkingDirectory=$REPO_DIR
Environment=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ExecStartPre=/bin/sleep 5
ExecStart=$REPO_DIR/scripts/run_v5_bare_rrdb_control.sh
Restart=on-failure
RestartSec=30
StandardOutput=append:$REPO_DIR/runs/systemd_v5_bare.log
StandardError=append:$REPO_DIR/runs/systemd_v5_bare.log

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable urbanjepa-v5-bare.service 2>/dev/null || true
echo "Installed urbanjepa-v5-bare.service"
echo ""
echo "Start:           sudo systemctl start urbanjepa-v5-bare.service"
echo "Stop:            sudo systemctl stop urbanjepa-v5-bare.service"
echo "Status:          sudo systemctl status urbanjepa-v5-bare.service"
echo "Train log:       tail -F $REPO_DIR/runs/v5_bare_rrdb_control.log"
echo "Systemd log:     tail -F $REPO_DIR/runs/systemd_v5_bare.log"
