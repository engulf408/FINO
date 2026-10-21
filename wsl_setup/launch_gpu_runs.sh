#!/bin/bash
# Launch full-budget GPU runs (FINO + FQL) detached inside WSL.
set -u
cd /opt/fino/FINO || exit 1

export FINO_WANDB_MODE=offline
export WANDB_SILENT=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.35

COMMON="--env_name=antmaze-large-navigate-singletask-v0 --agent.alpha=10 --agent.q_agg=min \
--offline_steps=1000000 --online_steps=500000 --eval_interval=50000 --eval_episodes=50 \
--log_interval=5000 --save_interval=1000000 --seed=0 --save_dir=/mnt/e/FINO/gpu_exp"

# FINO
setsid nohup /opt/fino-venv/bin/python main.py $COMMON > /mnt/e/FINO/logs/gpu_antmaze_fino.log 2>&1 < /dev/null &
echo $! > /mnt/e/FINO/logs/gpu_antmaze_fino.pid
sleep 3

# FQL baseline
setsid nohup /opt/fino-venv/bin/python main.py --agent=agents/fql.py $COMMON > /mnt/e/FINO/logs/gpu_antmaze_fql.log 2>&1 < /dev/null &
echo $! > /mnt/e/FINO/logs/gpu_antmaze_fql.pid

sleep 2
echo "launched: fino pid=$(cat /mnt/e/FINO/logs/gpu_antmaze_fino.pid) fql pid=$(cat /mnt/e/FINO/logs/gpu_antmaze_fql.pid)"
ps aux | grep 'main.py' | grep -v grep | awk '{print $2, $12, $13, $14, $15}'
