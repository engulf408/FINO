"""GPU benchmark for FINO/FQL on the WSL stack (jax 0.12, RTX 5070).

Run inside WSL:  /opt/fino-venv/bin/python /mnt/e/FINO/scripts/bench_gpu.py
"""
import os
import sys
import time

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '.60')

REPO = '/opt/fino/FINO'
sys.path.insert(0, REPO)

import numpy as np

import jax
import jax.numpy as jnp

from agents import agents  # noqa: E402
from agents import fino as fino_mod  # noqa: E402
from agents import fql as fql_mod  # noqa: E402

print('jax', jax.__version__, '| devices:', jax.devices())

rng = np.random.default_rng(0)
N, ob_dim, act_dim = 2048, 29, 8
data = dict(
    observations=rng.standard_normal((N, ob_dim)).astype(np.float32),
    actions=rng.uniform(-1, 1, (N, act_dim)).astype(np.float32),
    rewards=rng.standard_normal((N,)).astype(np.float32),
    terminals=np.zeros((N,), np.float32),
    masks=np.ones((N,), np.float32),
    next_observations=rng.standard_normal((N, ob_dim)).astype(np.float32),
)


def make_agent(name, module):
    cfg = module.get_config()
    cfg.ob_dims = (ob_dim,)
    cfg.action_dim = act_dim
    return agents[name].create(0, data['observations'], data['actions'], cfg)


def bench(name, module, n=20):
    agent = make_agent(name, module)
    batch = {k: jnp.asarray(v[:256]) for k, v in data.items()}
    t0 = time.time()
    agent, _ = agent.update(batch)
    jax.block_until_ready(agent.network.params)
    t_first = time.time() - t0
    t0 = time.time()
    for _ in range(n):
        agent, _ = agent.update(batch)
    jax.block_until_ready(agent.network.params)
    t = (time.time() - t0) / n
    print(f'{name.upper():5s}: first(compile)={t_first:6.2f}s | steady={t*1000:7.2f} ms/update | '
          f'1.5M steps ≈ {t*1_500_000/3600:5.2f} h')
    return t


if __name__ == '__main__':
    t_fino = bench('fino', fino_mod)
    t_fql = bench('fql', fql_mod)
    print()
    print('== env + dataset load test ==')
    t0 = time.time()
    from envs.env_utils import make_env_and_datasets
    env, eval_env, train_dataset, val_dataset = make_env_and_datasets('antmaze-large-navigate-singletask-v0')
    print(f'dataset loaded in {time.time()-t0:.1f}s | train size={train_dataset.size:,} | obs={train_dataset["observations"].shape} | val={val_dataset.size:,}')
    ob, _ = env.reset()
    t0 = time.time()
    for _ in range(200):
        ob, r, te, tr, info = env.step(env.action_space.sample())
        if te or tr:
            ob, _ = env.reset()
    print(f'env step: {(time.time()-t0)/200*1000:.2f} ms/step')
    print('BENCH_DONE')
