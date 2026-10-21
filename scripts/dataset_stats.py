"""Compute dataset statistics for the OGBench datasets used in the reproduction."""
import json
import os

import numpy as np

DATA = os.path.expanduser(r'~\.ogbench\data')
OUT = r'E:\FINO\results'
os.makedirs(OUT, exist_ok=True)

NAMES = ['pointmaze-medium-navigate-v0', 'antmaze-large-navigate-v0']

rows = []
for name in NAMES:
    p = os.path.join(DATA, name + '.npz')
    if not os.path.exists(p):
        print('missing', p)
        continue
    f = np.load(p)
    obs = f['observations']
    act = f['actions']
    term = f['terminals']
    ep_len = int(np.argmax(term > 0) + 1)
    rows.append(dict(
        dataset=name,
        transitions=len(obs),
        obs_dim=obs.shape[1],
        action_dim=act.shape[1],
        obs_dtype=str(obs.dtype),
        episode_length=ep_len,
        num_episodes=int((term > 0).sum()),
        file_mb=round(os.path.getsize(p) / 1e6, 1),
        action_min=round(float(act.min()), 3),
        action_max=round(float(act.max()), 3),
        obs_mean_first_dim=round(float(obs[:, 0].mean()), 3),
    ))
    f.close()

with open(os.path.join(OUT, 'dataset_stats.json'), 'w', encoding='utf-8') as fh:
    json.dump(rows, fh, indent=2)
for r in rows:
    print(r)
