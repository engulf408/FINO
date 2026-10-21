#!/bin/bash
# Launch batch-2 GPU runs (FINO on two more OGBench tasks), detached inside WSL.
set -u
cd /opt/fino/FINO || exit 1

export FINO_WANDB_MODE=offline
export WANDB_SILENT=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.35

COMMON="--agent.alpha=100 --agent.discount=0.995 \
--offline_steps=1000000 --online_steps=500000 --eval_interval=50000 --eval_episodes=50 \
--log_interval=5000 --save_interval=1000000 --seed=0 --save_dir=/mnt/e/FINO/gpu_exp"

# humanoidmaze-medium (README: alpha=100, discount=0.995)
setsid nohup /opt/fino-venv/bin/python main.py --env_name=humanoidmaze-medium-navigate-singletask-v0 $COMMON \
  > /mnt/e/FINO/logs/gpu_humanoidmaze_medium_fino.log 2>&1 < /dev/null &
echo $! > /mnt/e/FINO/logs/gpu_humanoidmaze_medium_fino.pid
sleep 3

# cube-double-play (README: alpha=300)
setsid nohup /opt/fino-venv/bin/python main.py --env_name=cube-double-play-singletask-v0 --agent.alpha=300 \
  --offline_steps=1000000 --online_steps=500000 --eval_interval=50000 --eval_episodes=50 \
  --log_interval=5000 --save_interval=1000000 --seed=0 --save_dir=/mnt/e/FINO/gpu_exp \
  > /mnt/e/FINO/logs/gpu_cube_double_fino.log 2>&1 < /dev/null &
echo $! > /mnt/e/FINO/logs/gpu_cube_double_fino.pid

sleep 2
echo "batch2 launched: humanoidmaze pid=$(cat /mnt/e/FINO/logs/gpu_humanoidmaze_medium_fino.pid) cube pid=$(cat /mnt/e/FINO/logs/gpu_cube_double_fino.pid)"
