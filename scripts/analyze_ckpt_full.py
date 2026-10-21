"""Checkpoint-level analysis for the full-budget GPU runs (epoch 1,000,000).

Run inside WSL:
  XLA_PYTHON_CLIENT_PREALLOCATE=false /opt/fino-venv/bin/python \
      /mnt/e/FINO/scripts/analyze_ckpt_full.py
"""
import argparse
import json
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, '/opt/fino/FINO')

from agents import agents  # noqa: E402
from envs.env_utils import make_env_and_datasets  # noqa: E402
from utils.flax_utils import restore_agent  # noqa: E402

OUT = '/mnt/e/FINO/results'
ENV = 'antmaze-large-navigate-singletask-v0'
AGENTS = {'FINO': 'fino', 'FQL': 'fql'}
EXP_ROOT = '/mnt/e/FINO/gpu_exp'
EPOCH = 1_000_000
N_STATES = 256
N_NOISES = 8


def energy_distance(x, y):
    def pdist(a, b):
        return np.sqrt(np.maximum(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1), 0.0))
    return float(2 * pdist(x, y).mean() - pdist(x, x).mean() - pdist(y, y).mean())


def load_agent(agent_name, val_dataset, train_dataset):
    import importlib
    mod = importlib.import_module(f'agents.{agent_name}')
    cfg = mod.get_config()
    example = train_dataset.sample(1)
    agent = agents[agent_name].create(0, example['observations'], example['actions'], cfg)
    ckpt = os.path.join(EXP_ROOT, agent_name.upper(), ENV, '0')
    agent = restore_agent(agent, ckpt, EPOCH)
    return agent


def main():
    env, eval_env, train_dataset, val_dataset = make_env_and_datasets(ENV)
    rng = np.random.default_rng(0)
    idxs = rng.choice(val_dataset.size, N_STATES, replace=False)
    batch = val_dataset.sample(N_STATES, idxs=idxs)
    obs = jnp.asarray(batch['observations'])
    data_actions = np.asarray(batch['actions'])
    act_dim = data_actions.shape[-1]

    results = {}
    for label, agent_name in AGENTS.items():
        print(f'=== {label} ===', flush=True)
        agent = load_agent(agent_name, val_dataset, train_dataset)

        # one-step deterministic eval action (B=256; temperature=0 uses global argmax in code, so also report single-sample)
        one_step = np.asarray(agent.network.select('actor_onestep_flow')(
            obs, jax.random.normal(jax.random.PRNGKey(1), (N_STATES, act_dim))))
        qs = np.asarray(agent.network.select('critic')(obs, actions=jnp.asarray(one_step)))
        q_mean = float(qs.mean())

        keys = jax.random.split(jax.random.PRNGKey(2), N_NOISES)
        t0 = __import__('time').time()
        teacher = np.stack([
            np.asarray(agent.compute_flow_actions(
                obs, noises=jax.random.normal(k, (N_STATES, act_dim))))
            for k in keys
        ], axis=1)
        jax.block_until_ready(teacher)
        t_teacher = __import__('time').time() - t0

        t0 = __import__('time').time()
        for _ in range(10):
            one = np.asarray(agent.network.select('actor_onestep_flow')(
                obs, jax.random.normal(jax.random.PRNGKey(3), (N_STATES, act_dim))))
        jax.block_until_ready(one)
        t_one = (__import__('time').time() - t0) / 10

        one_dist = one_step[:, None, :].repeat(N_NOISES, axis=1)
        results[label] = dict(
            n_states=N_STATES,
            energy_distance_one_step_vs_teacher=round(energy_distance(
                one_dist.reshape(-1, act_dim), teacher.reshape(-1, act_dim)), 6),
            mse_one_step_vs_teacher=round(float(np.mean((one_step - teacher.mean(axis=1)) ** 2)), 6),
            mse_one_step_vs_dataset_action=round(float(np.mean((one_step - data_actions) ** 2)), 6),
            mse_teacher_vs_dataset_action=round(float(np.mean((teacher.mean(axis=1) - data_actions) ** 2)), 6),
            mean_q_argmax_policy=round(q_mean, 4),
            one_step_latency_ms_N256=round(1000 * t_one, 2),
            teacher_10step_latency_ms_N256x8=round(1000 * t_teacher, 2),
        )
        print(json.dumps(results[label], indent=2), flush=True)

    with open(os.path.join(OUT, 'checkpoint_analysis_full_budget.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print('wrote', os.path.join(OUT, 'checkpoint_analysis_full_budget.json'))


if __name__ == '__main__':
    main()
