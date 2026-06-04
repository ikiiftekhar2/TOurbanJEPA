#!/bin/bash
# scripts/watch_v5.sh — live dashboard for the v5 training service.
#
# Usage:
#   ./scripts/watch_v5.sh           # log+gpu+stage tail (single pane, no tmux)
#   ./scripts/watch_v5.sh tmux      # 3-pane tmux dashboard (recommended)
#   ./scripts/watch_v5.sh journal   # just systemd journal -f
#   ./scripts/watch_v5.sh gpu       # just `watch nvidia-smi`
#   ./scripts/watch_v5.sh tb        # print tensorboard URL
#
# Tail filters keep noise low — only [step], [step_val], [epoch_val], [freeze],
# [gan], [resume], and any error/traceback lines.

set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-tail}"

LAUNCHER_LOG="runs/v5_jepa_esrgan.log"
SYSTEMD_LOG="runs/systemd_v5.log"

TAIL_FILTER='^\[step|^\[step_val|^\[epoch_val|^\[freeze|^\[gan|^\[resume|^\[warm|^\[train|^\[device|^\[data|^\[model|^\[done|Traceback|Error|RuntimeError|OOM|FAIL|NaN|stage='

case "$MODE" in
    tail)
        # Single-pane combined tail: launcher log + per-stage train log lines.
        echo "[watch_v5] tailing launcher + train logs. ctrl-C to exit."
        echo "[watch_v5] filter: $TAIL_FILTER"
        tail -F "$LAUNCHER_LOG" "$SYSTEMD_LOG" 2>/dev/null \
            | grep --line-buffered -E "$TAIL_FILTER"
        ;;

    journal)
        journalctl -u urbanjepa-v5.service -f
        ;;

    gpu)
        watch -n 2 'nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw --format=csv'
        ;;

    stage)
        watch -n 5 'echo "stage.txt: $(cat checkpoints/v5_jepa_esrgan/stage.txt 2>/dev/null || echo init)"; echo; \
            echo "=== Stage A best ===" && python3 -c "import json; m=json.load(open('"'"'checkpoints/v5_jepa_esrgan_stageA/manifest.json'"'"', '"'"'r'"'"')); print(json.dumps(m.get('"'"'best'"'"'), indent=2))" 2>/dev/null || echo "(none yet)"; \
            echo "=== Stage B best ===" && python3 -c "import json; m=json.load(open('"'"'checkpoints/v5_jepa_esrgan_stageB/manifest.json'"'"', '"'"'r'"'"')); print(json.dumps(m.get('"'"'best'"'"'), indent=2))" 2>/dev/null || echo "(none yet)"; \
            echo; echo "=== latest ckpts ==="; ls -lat checkpoints/v5_jepa_esrgan_stage*/*.pt 2>/dev/null | head -8'
        ;;

    tb|tensorboard)
        echo "TensorBoard: http://$(hostname -I | awk '{print $1}'):6006"
        echo "(auto-launched by run_v5_jepa_esrgan.sh on first start)"
        ;;

    tmux)
        SESSION="v5"
        if tmux has-session -t "$SESSION" 2>/dev/null; then
            echo "[watch_v5] attaching to existing tmux session '$SESSION'"
            tmux attach -t "$SESSION"
            exit 0
        fi
        # 3 panes: log tail (top), gpu (bottom-left), stage+ckpts (bottom-right)
        tmux new-session -d -s "$SESSION" \
            "tail -F runs/v5_jepa_esrgan.log runs/systemd_v5.log 2>/dev/null | grep --line-buffered -E '$TAIL_FILTER'"
        tmux split-window -v -t "$SESSION:0" -p 35 \
            "watch -n 2 'nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv'"
        tmux split-window -h -t "$SESSION:0.1" \
            "watch -n 5 'echo STAGE: \$(cat checkpoints/v5_jepa_esrgan/stage.txt 2>/dev/null || echo init); echo; echo CKPTS:; ls -lat checkpoints/v5_jepa_esrgan_stage*/*.pt 2>/dev/null | head -6; echo; echo SYSTEMD:; systemctl is-active urbanjepa-v5.service'"
        tmux select-pane -t "$SESSION:0.0"
        echo "[watch_v5] tmux session '$SESSION' ready."
        echo "  attach: tmux attach -t $SESSION"
        echo "  detach inside session: Ctrl-b d"
        echo "  kill: tmux kill-session -t $SESSION"
        tmux attach -t "$SESSION"
        ;;

    *)
        echo "Unknown mode '$MODE'. Use: tail | tmux | journal | gpu | stage | tb"
        exit 1
        ;;
esac
