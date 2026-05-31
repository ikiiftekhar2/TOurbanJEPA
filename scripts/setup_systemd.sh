#!/bin/bash
# Install systemd unit to auto-resume UrbanJEPA v3 training on boot.
# Run with sudo.
set -euo pipefail

REPO_DIR="${REPO_DIR:-/mnt/eskeetit/Code-server/UrbanJEPA}"
USER_NAME="${SUDO_USER:-ubuntu}"
SERVICE=/etc/systemd/system/urbanjepa-v3-train.service

cat > "$SERVICE" <<EOF
[Unit]
Description=UrbanJEPA v3 Training (auto-resume)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$USER_NAME
WorkingDirectory=$REPO_DIR
ExecStart=$REPO_DIR/scripts/resume_training.sh
Restart=on-failure
RestartSec=30
StandardOutput=append:$REPO_DIR/runs/systemd.log
StandardError=append:$REPO_DIR/runs/systemd.log

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable urbanjepa-v3-train.service
echo "Installed urbanjepa-v3-train.service. Start with: sudo systemctl start urbanjepa-v3-train.service"
