#!/bin/bash
# Install systemd unit that auto-runs the 3-backbone comparison on boot.
# Each backbone runs --epochs N (default 3). The runner skips finished experiments,
# and within each experiment training auto-resumes from the manifest.
# Once all 3 finish, the unit exits 0 and does NOT restart.
set -euo pipefail

REPO_DIR="${REPO_DIR:-/mnt/eskeetit/Code-server/UrbanJEPA}"
USER_NAME="${SUDO_USER:-ubuntu}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-23}"
SERVICE=/etc/systemd/system/urbanjepa-v3-comparison.service

cat > "$SERVICE" <<EOF
[Unit]
Description=UrbanJEPA v3 3-backbone comparison ($EPOCHS epochs each, auto-resume)
After=network-online.target mnt-eskeetit.mount
Wants=network-online.target mnt-eskeetit.mount

[Service]
Type=simple
User=$USER_NAME
WorkingDirectory=$REPO_DIR
Environment=EPOCHS=$EPOCHS
Environment=BATCH_SIZE=$BATCH_SIZE
Environment=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ExecStartPre=/bin/sleep 10
ExecStart=$REPO_DIR/scripts/resume_backbone_experiment.sh
Restart=on-failure
RestartSec=30
StandardOutput=append:$REPO_DIR/runs/systemd.log
StandardError=append:$REPO_DIR/runs/systemd.log

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable urbanjepa-v3-comparison.service
echo "Installed urbanjepa-v3-comparison.service ($EPOCHS epochs/backbone)."
echo "Status: $(systemctl is-enabled urbanjepa-v3-comparison.service)"
echo "Start now with: sudo systemctl start urbanjepa-v3-comparison.service"
echo "View live logs:  journalctl -u urbanjepa-v3-comparison.service -f"
