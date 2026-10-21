#!/bin/bash
# Run FINO main.py inside WSL.
# Usage: wsl_run.sh <log-name> [main.py args...]
#   first arg = log file name under /mnt/e/FINO/logs/
set -u
LOG="/mnt/e/FINO/logs/$1"; shift
cd /opt/fino/FINO
export FINO_WANDB_MODE=offline
export WANDB_SILENT=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${MEM_FRACTION:-0.60}"
echo "START $(date -Is) | args: $*" > "$LOG"
/opt/fino-venv/bin/python main.py "$@" >> "$LOG" 2>&1
echo "exit=$? END $(date -Is)" >> "$LOG"
