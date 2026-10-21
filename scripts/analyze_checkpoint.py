"""Checkpoint-level analysis: flow-vs-one-step fidelity and policy quality on validation data.

Loads the saved antmaze checkpoints (FINO and FQL) and reports:
  * the distribution distance between the 10-step flow teacher and the one-step distilled policy,
  * the MSE between one-step policy actions and the dataset actions (behaviour cloning quality),
  * mean Q value of the deterministic (argmax) policy actions,
  * a multi-step vs one-step latency comparison on CPU.

Run after the antmaze runs finish.
"""
import json
import os
import sys
import time

sys.path.insert(0, r'E:\FINO\FINO')

import jax
import jax.numpy as jnp
import numpy as np

from agents import agents
from envs.env_utils import make_env_and_datasets
from utils.flax_utils import restore_agent

OUT = r'E:\FINO\results'
os.makedirs(OUT, exist_ok=True)
ENV = 'antmaze-large-navigate-singletask-v0'
AGENTS = {'FINO': 'fino', 'FQL': 'fql'}
CKPT_EPOCH = 30000
N_STATES = 256
N_NOISES = 8


def energy_distance(x, y):
    """Energy distance between two samples (lower = more similar)."""
    def pdist(a, b):
        return np.sqrt(np.maximum(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1), 0.0))
    dxy = pdist(x, y).mean()
    dxx = pdist(x, x).mean()
    dyy = pdist(y, y).mean()
    return float(2 * dxy - dxx - dyy)


def load_agent(agent_name):
    env, eval_env, train_dataset, val_dataset = make_env_and_datasets(ENV)
    import importlib
    mod = importlib.import_module(f'agents.{agent_name}')
    cfg = mod.get_config()
    example = train_dataset.sample(1)
    agent = agents[agent_name].create(0, example['observations'], example['actions'], cfg)
    ckpt = os.path.join(r'E:\FINO\exp', agent_name.upper(), ENV, '0')
    agent = restore_agent(agent, ckpt, CKPT_EPOCH)
    return agent, val_dataset, env


def main():
    results = {}
    for label, agent_name in AGENTS.items():
        ckpt_dir = os.path.join(r'E:\FINO\exp', agent_name.upper(), ENV, '0', str(CKPT_EPOCH))
        if not os.path.exists(ckpt_dir):
            print(f'skip {label}: no checkpoint at {ckpt_dir}')
            continue
        print(f'=== {label} ===')
        agent, val_dataset, env = load_agent(agent_name)
        rng = np.random.default_rng(0)
        idxs = rng.choice(val_dataset.size, N_STATES, replace=False)
        batch = val_dataset.sample(N_STATES, idxs=idxs)
        obs = jnp.asarray(batch['observations'])
        data_actions = np.asarray(batch['actions'])

        # one-step policy: deterministic argmax over candidate noises (eval behaviour, B=1) and single-noise
        t0 = time.time()
        det_actions = np.asarray(agent.sample_actions(obs, seed=jax.random.PRNGKey(0), temperature=0))
        t_one = time.time() - t0
        one_step = np.asarray(agent.network.select('actor_onestep_flow')(obs, jax.random.normal(jax.random.PRNGKey(1), obs.shape[:-1] + (data_actions.shape[-1],))))
        qs = np.asarray(agent.network.select('critic')(obs, actions=jnp.asarray(one_step)))
        q_mean = float(qs.mean())

        # 10-step flow teacher with several noises
        keys = jax.random.split(jax.random.PRNGKey(2), N_NOISES)
        t0 = time.time()
        teacher = np.stack([
            np.asarray(agent.compute_flow_actions(obs, noises=jax.random.normal(k, obs.shape[:-1] + (data_actions.shape[-1],))))
            for k in keys
        ], axis=1)  # (N, K, A)
        t_teacher = time.time() - t0

        one_dist = one_step[:, None, :].repeat(N_NOISES, axis=1)
        ed = energy_distance(one_dist.reshape(-1, data_actions.shape[-1]), teacher.reshape(-1, data_actions.shape[-1]))
        mse_teacher = float(np.mean((one_step - teacher.mean(axis=1)) ** 2))
        mse_data = float(np.mean((one_step - data_actions) ** 2))
        mse_data_teacher = float(np.mean((teacher.mean(axis=1) - data_actions) ** 2))

        results[label] = dict(
            n_states=N_STATES,
            energy_distance_one_step_vs_10step_teacher=round(ed, 6),
            mse_one_step_vs_teacher=round(mse_teacher, 6),
            mse_one_step_vs_dataset_action=round(mse_data, 6),
            mse_teacher_vs_dataset_action=round(mse_data_teacher, 6),
            mean_q_argmax_policy=round(q_mean, 4),
            one_step_latency_ms_N256=round(1000 * t_one, 1),
            teacher_10step_latency_ms_N256x8=round(1000 * t_teacher, 1),
        )
        print(json.dumps(results[label], indent=2))

    with open(os.path.join(OUT, 'checkpoint_analysis.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print('wrote', os.path.join(OUT, 'checkpoint_analysis.json'))


if __name__ == '__main__':
    main()
