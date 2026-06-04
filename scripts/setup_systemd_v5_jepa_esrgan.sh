#!/bin/bash
# Install systemd unit for v5 JEPA-Conditioned ESRGAN staged training.
#
# One unit, two stages — the launcher script (run_v5_jepa_esrgan.sh) reads
# checkpoints/v5_jepa_esrgan/stage.txt and runs the matching stage.
#
# On crash (any non-zero exit), systemd auto-restarts the same stage; the
# launcher invokes train.py --resume which uses the CheckpointManager's
# rollback (default 200 steps from the latest step_slot_*.pt).
# On clean exit after a passing gate, stage.txt advances (A→B→DONE).
# On clean exit after a failing gate, the launcher exits 1 → systemd
# restarts but the gate keeps failing until the issue is fixed.
#
# Auto-resume on reboot is provided by `WantedBy=multi-user.target` + the
# launcher's --resume behaviour.

set -euo pipefail

REPO_DIR="${REPO_DIR:-/mnt/eskeetit/Code-server/UrbanJEPA}"
USER_NAME="${SUDO_USER:-ubuntu}"
SERVICE_FILE=/etc/systemd/system/urbanjepa-v5.service

cat > "$SERVICE_FILE" <<EOF
[Unit]
Description=UrbanJEPA v5 JEPA-Conditioned ESRGAN staged training (A -> B)
After=network-online.target mnt-eskeetit.mount
Wants=network-online.target mnt-eskeetit.mount

[Service]
Type=simple
User=$USER_NAME
WorkingDirectory=$REPO_DIR
Environment=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ExecStartPre=/bin/sleep 10
ExecStart=$REPO_DIR/scripts/run_v5_jepa_esrgan.sh
Restart=on-failure
RestartSec=30
StandardOutput=append:$REPO_DIR/runs/systemd_v5.log
StandardError=append:$REPO_DIR/runs/systemd_v5.log

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable urbanjepa-v5.service 2>/dev/null || true
echo "Installed urbanjepa-v5.service"
echo "  Enabled:   \$(systemctl is-enabled urbanjepa-v5.service 2>/dev/null || echo n/a)"
echo ""
echo "Start:            sudo systemctl start urbanjepa-v5.service"
echo "Stop:             sudo systemctl stop urbanjepa-v5.service"
echo "Status:           sudo systemctl status urbanjepa-v5.service"
echo "Live system log:  journalctl -u urbanjepa-v5.service -f"
echo "Live train log:   tail -F $REPO_DIR/runs/v5_jepa_esrgan.log"
echo "Current stage:    cat $REPO_DIR/checkpoints/v5_jepa_esrgan/stage.txt"
echo "TensorBoard:      http://<host>:6006"
