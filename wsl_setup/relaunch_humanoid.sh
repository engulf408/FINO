#!/bin/bash
# Restart humanoidmaze-medium run with reduced evaluation budget (kill old PID 14 first).
set -u
kill -9 14 2>/dev/null || true
sleep 2
cd /opt/fino/FINO || exit 1
export FINO_WANDB_MODE=offline
export WANDB_SILENT=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.45
setsid nohup /opt/fino-venv/bin/python main.py \
  --env_name=humanoidmaze-medium-navigate-singletask-v0 --agent.alpha=100 --agent.discount=0.995 \
  --offline_steps=1000000 --online_steps=500000 --eval_interval=100000 --eval_episodes=20 \
  --log_interval=5000 --save_interval=1000000 --seed=0 --save_dir=/mnt/e/FINO/gpu_exp \
  > /mnt/e/FINO/logs/gpu_humanoidmaze_medium_fino.log 2>&1 < /dev/null &
echo "new humanoidmaze pid=$!"
sleep 3
ps -eo pid,etime,cmd --no-headers | grep 'main.py' | grep -v grep | cut -c1-100
