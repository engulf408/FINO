# -*- coding: utf-8 -*-
"""Patch the WSL copy of FINO with env-var switches for the paper-formula ablations.

Modifies /opt/fino/FINO/agents/fino.py (noise schedule + beta update) and main.py (log beta/entropy).
Keeps .orig backups. Idempotent (skips if already patched).
"""
import io
import os
import shutil

ROOT = '/opt/fino/FINO'
FINO = os.path.join(ROOT, 'agents', 'fino.py')
MAIN = os.path.join(ROOT, 'main.py')

# ---------------------------------------------------------------- fino.py
s = io.open(FINO, encoding='utf-8').read()
if 'FINO_NOISE_SCHEDULE' in s:
    print('fino.py already patched; skip')
else:
    shutil.copy2(FINO, FINO + '.orig')
    # top import
    if '\nimport os\n' not in s:
        s = s.replace('\nimport jax\n', '\nimport os\n\nimport jax\n', 1)
    # global switches after imports (right before first class/def)
    anchor = 'class FINO'
    assert anchor in s
    switches = (
        "_SCHED = os.environ.get('FINO_NOISE_SCHEDULE', 'exp')  # [ablation] exp | quad | none\n"
        "_BETA_MODE = os.environ.get('FINO_BETA_MODE', 'official')  # [ablation] official | paper | fixed\n\n\n"
    )
    s = s.replace(anchor, switches + anchor, 1)
    old = ("        normal_noise = jax.random.normal(nor_rng, shape=(batch_size, action_dim))\n"
           "        noise_scale = jnp.exp(-(t * -10 + 10)) * self.config['noise_scale']\n"
           "        normal_samples = x_t + noise_scale * normal_noise")
    new = (
        "        normal_noise = jax.random.normal(nor_rng, shape=(batch_size, action_dim))\n"
        "        # [ablation patch] selectable noise schedule:\n"
        "        #   exp  = official eta*exp(10(t-1))\n"
        "        #   quad = paper Eq.(7) sigma_t = sqrt((eta^2-2*eta)t^2 + 2*eta*t)\n"
        "        #   none = no injection\n"
        "        _eta = self.config['noise_scale']\n"
        "        if _SCHED == 'quad':\n"
        "            _sigma_sq = (_eta ** 2 - 2.0 * _eta) * t ** 2 + 2.0 * _eta * t\n"
        "            noise_scale = jnp.sqrt(jnp.maximum(_sigma_sq, 0.0))\n"
        "        elif _SCHED == 'none':\n"
        "            noise_scale = jnp.zeros_like(t)\n"
        "        else:\n"
        "            noise_scale = jnp.exp(-(t * -10 + 10)) * _eta\n"
        "        normal_samples = x_t + noise_scale * normal_noise"
    )
    assert old in s, 'noise block not found'
    s = s.replace(old, new, 1)
    old_b = "        new_beta = max(0, self.beta - 0.1 * (-self.config['action_dim'] - entropy))"
    new_b = (
        "        # [ablation patch] selectable beta update direction\n"
        "        if _BETA_MODE == 'paper':\n"
        "            new_beta = max(0.0, self.beta - 0.1 * (entropy + self.config['action_dim']))\n"
        "        elif _BETA_MODE == 'fixed':\n"
        "            new_beta = self.beta\n"
        "        else:\n"
        "            new_beta = max(0, self.beta - 0.1 * (-self.config['action_dim'] - entropy))"
    )
    assert old_b in s, 'beta line not found'
    s = s.replace(old_b, new_b, 1)
    io.open(FINO, 'w', encoding='utf-8').write(s)
    print('fino.py patched')

# ---------------------------------------------------------------- main.py
s = io.open(MAIN, encoding='utf-8').read()
if 'beta.csv' in s:
    print('main.py already patched; skip')
else:
    shutil.copy2(MAIN, MAIN + '.orig')
    old = ("                if hasattr(agent, 'beta'):\n"
           "                    eval_metrics['evaluation/beta'] = agent.beta\n")
    new = ("                if hasattr(agent, 'beta'):\n"
           "                    eval_metrics['evaluation/beta'] = agent.beta\n"
           "                    # [ablation patch] persist beta/entropy (CsvLogger header is fixed)\n"
           "                    _bf = os.path.join(FLAGS.save_dir, 'beta.csv')\n"
           "                    _write_header = not os.path.exists(_bf)\n"
           "                    with open(_bf, 'a', encoding='utf-8') as _f:\n"
           "                        if _write_header:\n"
           "                            _f.write('step,beta,entropy\\n')\n"
           "                        _f.write(f'{i},{float(agent.beta)},{float(entropy)}\\n')\n")
    assert old in s, 'eval beta block not found'
    s = s.replace(old, new, 1)
    io.open(MAIN, 'w', encoding='utf-8').write(s)
    print('main.py patched')

print('DONE')
