#!/bin/bash
#
# Auto-resume experiment pipeline on boot.
# Wired to your existing systemd service — just calls the pipeline.
#
set -euo pipefail
cd /mnt/eskeetit/Code-server/UrbanJEPA
exec bash scripts/experiment_pipeline.sh train
