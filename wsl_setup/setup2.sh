#!/bin/bash
# FINO reproduction - WSL setup step 2: venv + latest JAX with CUDA 12 plugin, GPU smoke test
set -e
echo "=== [1/4] create venv /opt/fino-venv ==="
if [ ! -x /opt/fino-venv/bin/python ]; then
  python3 -m venv /opt/fino-venv
fi
/opt/fino-venv/bin/python -m pip install --upgrade -q pip setuptools wheel

echo "=== [2/4] configure pip mirror (Tsinghua) ==="
/opt/fino-venv/bin/pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
/opt/fino-venv/bin/pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn

echo "=== [3/4] install jax[cuda12] latest (large download) ==="
/opt/fino-venv/bin/pip install "jax[cuda12]" 2>&1 | tail -5

echo "=== [4/4] GPU smoke test ==="
/opt/fino-venv/bin/python - <<'PY'
import jax
print('jax', jax.__version__)
print('devices:', jax.devices())
x = jax.numpy.ones((1024, 1024))
y = (x @ x).sum()
jax.block_until_ready(y)
print('matmul ok, sum =', float(y))
print('SETUP2_DONE')
PY
