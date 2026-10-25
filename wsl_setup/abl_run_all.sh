#!/bin/bash
# Paper-formula ablation runs (standard plan), executed sequentially in 3 phases.
# Phase 1 (2x full-budget from scratch): A2 quadratic schedule, A3 no noise
# Phase 2 (A4 full-budget eta=0.3) + (B1 resume baseline, official beta)
# Phase 3 (B2 paper beta sign) + (B3 fixed beta)
set -u
cd /opt/fino/FINO || exit 1

export FINO_WANDB_MODE=offline
export WANDB_SILENT=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.35

LOG=/mnt/e/FINO/logs
OUT=/mnt/e/FINO/gpu_exp/ablation
mkdir -p "$LOG" "$OUT"

COMMON="--env_name=antmaze-large-navigate-singletask-v0 --agent.alpha=10 --agent.q_agg=min \
--offline_steps=1000000 --online_steps=500000 --eval_interval=100000 --eval_episodes=50 \
--log_interval=5000 --save_interval=1000000 --seed=0"

BCOMMON="--env_name=antmaze-large-navigate-singletask-v0 --agent.alpha=10 --agent.q_agg=min \
--offline_steps=0 --online_steps=500000 --eval_interval=50000 --eval_episodes=50 \
--log_interval=5000 --save_interval=500000 --seed=0 \
--restore_path=/mnt/e/FINO/gpu_exp/FINO/antmaze-large-navigate-singletask-v0/0 --restore_epoch=1000000"

run_bg() {
  local name=$1; shift
  local envs=$1; shift
  env $envs /opt/fino-venv/bin/python main.py "$@" > "$LOG/abl_${name}.log" 2>&1 &
  echo $! > "$LOG/abl_${name}.pid"
  echo "[$(date -Is)] launched ${name} pid=$(cat "$LOG/abl_${name}.pid")"
}

wait_pids() {
  for n in "$@"; do
    pid=$(cat "$LOG/abl_${n}.pid" 2>/dev/null || true)
    [ -n "$pid" ] && wait "$pid" 2>/dev/null
    echo "[$(date -Is)] finished ${n} rc=$?"
  done
}

echo "[$(date -Is)] PHASE1 start"
run_bg A2_quad_sched "FINO_NOISE_SCHEDULE=quad" $COMMON --save_dir="$OUT/A2_quad_sched"
sleep 5
run_bg A3_none_noise "FINO_NOISE_SCHEDULE=none" $COMMON --save_dir="$OUT/A3_none_noise"
wait_pids A2_quad_sched A3_none_noise

echo "[$(date -Is)] PHASE2 start"
run_bg A4_eta03 "FINO_NOISE_SCHEDULE=exp" $COMMON --agent.noise_scale=0.3 --save_dir="$OUT/A4_eta03"
sleep 5
run_bg B1_resume_official "FINO_BETA_MODE=official" $BCOMMON --save_dir="$OUT/B1_resume_official"
wait_pids A4_eta03 B1_resume_official

echo "[$(date -Is)] PHASE3 start"
run_bg B2_paper_beta "FINO_BETA_MODE=paper" $BCOMMON --save_dir="$OUT/B2_paper_beta"
sleep 5
run_bg B3_fixed_beta "FINO_BETA_MODE=fixed" $BCOMMON --save_dir="$OUT/B3_fixed_beta"
wait_pids B2_paper_beta B3_fixed_beta

echo "[$(date -Is)] ABL_ALL_DONE"
