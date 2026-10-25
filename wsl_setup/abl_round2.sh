#!/bin/bash
# Round-2 ablations: formula shape/scale, multi-seed robustness, cross-task generality, Q-guided K.
# 7 runs, 2-3 parallel phases.
set -u
cd /opt/fino/FINO || exit 1
export FINO_WANDB_MODE=offline
export WANDB_SILENT=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.30

LOG=/mnt/e/FINO/logs
OUT=/mnt/e/FINO/gpu_exp/ablation2
mkdir -p "$LOG" "$OUT"

COMMON="--env_name=antmaze-large-navigate-singletask-v0 --agent.alpha=10 --agent.q_agg=min \
--offline_steps=1000000 --online_steps=500000 --eval_interval=100000 --eval_episodes=50 \
--log_interval=5000 --save_interval=1000000 --seed=0"

COMMON_S1="--env_name=antmaze-large-navigate-singletask-v0 --agent.alpha=10 --agent.q_agg=min \
--offline_steps=1000000 --online_steps=500000 --eval_interval=100000 --eval_episodes=50 \
--log_interval=5000 --save_interval=1000000 --seed=1"

HUM="--env_name=humanoidmaze-medium-navigate-singletask-v0 --agent.alpha=100 --agent.q_agg=min \
--agent.discount=0.995 --offline_steps=1000000 --online_steps=500000 --eval_interval=100000 \
--eval_episodes=20 --log_interval=5000 --save_interval=1000000 --seed=0"

BCOMMON="--env_name=antmaze-large-navigate-singletask-v0 --agent.alpha=10 --agent.q_agg=min \
--offline_steps=0 --online_steps=500000 --eval_interval=50000 --eval_episodes=50 \
--log_interval=5000 --save_interval=500000 --seed=0 \
--restore_path=/mnt/e/FINO/gpu_exp/FINO/antmaze-large-navigate-singletask-v0/0 --restore_epoch=1000000"

run_bg() {
  local name=$1; shift
  local envs=$1; shift
  env $envs /opt/fino-venv/bin/python main.py "$@" > "$LOG/r2_${name}.log" 2>&1 &
  echo $! > "$LOG/r2_${name}.pid"
  echo "[$(date -Is)] launched ${name} pid=$(cat "$LOG/r2_${name}.pid")"
}

wait_pids() {
  for n in "$@"; do
    pid=$(cat "$LOG/r2_${n}.pid" 2>/dev/null || true)
    [ -n "$pid" ] && wait "$pid" 2>/dev/null
    echo "[$(date -Is)] finished ${n} rc=$?"
  done
}

echo "[$(date -Is)] R2 PHASE1 start"
run_bg A5_quad_peak "FINO_NOISE_SCHEDULE=quad FINO_QUAD_SCALE=0.4365" $COMMON --save_dir="$OUT/A5_quad_peak"
sleep 4
run_bg A6_quad_var "FINO_NOISE_SCHEDULE=quad_var" $COMMON --save_dir="$OUT/A6_quad_var"
sleep 4
run_bg S1_official_seed1 "FINO_NOISE_SCHEDULE=exp" $COMMON_S1 --save_dir="$OUT/S1_official_seed1"
wait_pids A5_quad_peak A6_quad_var S1_official_seed1

echo "[$(date -Is)] R2 PHASE2 start"
run_bg S1_quad_seed1 "FINO_NOISE_SCHEDULE=quad" $COMMON_S1 --save_dir="$OUT/S1_quad_seed1"
sleep 4
run_bg HUM_quad "FINO_NOISE_SCHEDULE=quad" $HUM --save_dir="$OUT/HUM_quad"
sleep 4
run_bg K1_guided "FINO_QGUIDE_K=1" $BCOMMON --save_dir="$OUT/K1_guided"
wait_pids S1_quad_seed1 HUM_quad K1_guided

echo "[$(date -Is)] R2 PHASE3 start"
run_bg K10_guided "FINO_QGUIDE_K=10" $BCOMMON --save_dir="$OUT/K10_guided"
wait_pids K10_guided

echo "[$(date -Is)] R2_ALL_DONE"
