#!/bin/bash
# FINO reproduction - WSL setup step 3: full code stack for the official repository
set -e
echo "=== [1/3] install code stack (Tsinghua pip mirror) ==="
/opt/fino-venv/bin/pip install -q flax optax orbax-checkpoint distrax chex ml_collections absl-py tqdm \
  scikit-learn pillow matplotlib pandas wandb imageio moviepy \
  "gymnasium==0.29.1" "mujoco==3.3.0" "dm_control==1.0.20" ogbench 2>&1 | tail -8

echo "=== [2/3] import smoke test ==="
/opt/fino-venv/bin/python - <<'PY'
import importlib
mods = ['jax','flax','optax','orbax.checkpoint','distrax','gymnasium','mujoco','ogbench','ml_collections','sklearn','matplotlib','wandb']
for m in mods:
    try:
        mod = importlib.import_module(m)
        print('OK ', m, getattr(mod, '__version__', ''))
    except Exception as e:
        print('FAIL', m, type(e).__name__, str(e)[:160])
print('SETUP3_DONE')
PY
