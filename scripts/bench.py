"""Quick benchmark: FINO update speed + env step speed on this machine (CPU)."""
import time
import sys
import numpy as np

sys.path.insert(0, r'E:\FINO\FINO')
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict
from agents.fino import FINOAgent
from agents.fql import FQLAgent

rng = np.random.default_rng(0)
N, ob_dim, act_dim = 1024, 29, 8
data = dict(
    observations=rng.standard_normal((N, ob_dim)).astype(np.float32),
    actions=rng.uniform(-1, 1, (N, act_dim)).astype(np.float32),
    rewards=rng.standard_normal((N,)).astype(np.float32),
    terminals=np.zeros((N,), np.float32),
    masks=np.ones((N,), np.float32),
    next_observations=rng.standard_normal((N, ob_dim)).astype(np.float32),
)


def make_agent(cls, module):
    cfg = module.get_config()
    cfg.ob_dims = (ob_dim,)
    cfg.action_dim = act_dim
    return cls.create(0, data['observations'], data['actions'], cfg)


def bench(cls, module, n=10):
    agent = make_agent(cls, module)
    batch = {k: jnp.asarray(v[:256]) for k, v in data.items()}
    t0 = time.time()
    agent, info = agent.update(batch)
    jax.block_until_ready(agent.network.params)
    t_first = time.time() - t0
    t0 = time.time()
    for _ in range(n):
        agent, info = agent.update(batch)
    jax.block_until_ready(agent.network.params)
    t = (time.time() - t0) / n
    print(f'{cls.__name__}: first(compile)={t_first:.2f}s, steady={t*1000:.1f} ms/update')
    return t


if __name__ == '__main__':
    from agents import fino as fino_mod
    from agents import fql as fql_mod
    print('jax', jax.__version__, jax.devices())
    bench(FINOAgent, fino_mod, 10)
    bench(FQLAgent, fql_mod, 5)

    import ogbench
    env = ogbench.make_env_and_datasets('pointmaze-medium-navigate-singletask-v0', env_only=True)
    ob, _ = env.reset()
    t0 = time.time()
    for _ in range(200):
        ob, r, te, tr, info = env.step(env.action_space.sample())
        if te or tr:
            ob, _ = env.reset()
    print(f'pointmaze env step: {(time.time()-t0)/200*1000:.2f} ms/step')

    env2 = ogbench.make_env_and_datasets('antmaze-large-navigate-singletask-v0', env_only=True)
    ob, _ = env2.reset()
    t0 = time.time()
    for _ in range(200):
        ob, r, te, tr, info = env2.step(env2.action_space.sample())
        if te or tr:
            ob, _ = env2.reset()
    print(f'antmaze env step: {(time.time()-t0)/200*1000:.2f} ms/step')
