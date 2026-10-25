# -*- coding: utf-8 -*-
"""Round-2 ablation patches: quad scale, variance-read quadratic, Q-guided K override."""
import io
import os
import shutil

ROOT = '/opt/fino/FINO'
FINO = os.path.join(ROOT, 'agents', 'fino.py')
s = io.open(FINO, encoding='utf-8').read()

# module-level globals
if 'FINO_QUAD_SCALE' not in s:
    old = "_SCHED = os.environ.get('FINO_NOISE_SCHEDULE', 'exp')  # [ablation] exp | quad | none\n_BETA_MODE = os.environ.get('FINO_BETA_MODE', 'official')  # [ablation] official | paper | fixed\n"
    new = old + ("_QUAD_SCALE = float(os.environ.get('FINO_QUAD_SCALE', '1.0') or 1.0)\n"
                 "_QGUIDE_K = int(os.environ.get('FINO_QGUIDE_K', '0') or 0)\n")
    assert old in s
    s = s.replace(old, new, 1)
    # quad branch: apply scale; add quad_var branch
    old_b = ("        if _SCHED == 'quad':\n"
             "            _sigma_sq = (_eta ** 2 - 2.0 * _eta) * t ** 2 + 2.0 * _eta * t\n"
             "            noise_scale = jnp.sqrt(jnp.maximum(_sigma_sq, 0.0))\n")
    new_b = ("        if _SCHED == 'quad':\n"
             "            _sigma_sq = (_eta ** 2 - 2.0 * _eta) * t ** 2 + 2.0 * _eta * t\n"
             "            noise_scale = _QUAD_SCALE * jnp.sqrt(jnp.maximum(_sigma_sq, 0.0))\n"
             "        elif _SCHED == 'quad_var':\n"
             "            _sigma_sq = (_eta ** 2 - 2.0 * _eta) * t ** 2 + 2.0 * _eta * t\n"
             "            noise_scale = _QUAD_SCALE * jnp.maximum(_sigma_sq, 0.0)\n")
    assert old_b in s
    s = s.replace(old_b, new_b, 1)
    # Q-guided K override
    oldk = "        num_samples = min(10, (self.config['action_dim'] + 1) // 2)"
    newk = "        num_samples = _QGUIDE_K if _QGUIDE_K > 0 else min(10, (self.config['action_dim'] + 1) // 2)"
    assert oldk in s
    s = s.replace(oldk, newk, 1)
    io.open(FINO, 'w', encoding='utf-8').write(s)
    print('round-2 patches applied')
else:
    print('already patched')

# sanity: show patched lines
out = io.open(FINO, encoding='utf-8').read()
for key in ['_QUAD_SCALE', 'quad_var', '_QGUIDE_K']:
    print(key, 'present:', key in out)
