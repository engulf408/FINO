#!/bin/bash
# FINO reproduction - WSL step 5: patch distrax guard + GPU benchmark
set -e
echo "=== [1/2] patch utils/networks.py: optional distrax (jax>=0.12 incompatibility) ==="
/opt/fino-venv/bin/python - <<'PY'
import io
p = '/opt/fino/FINO/utils/networks.py'
s = io.open(p, encoding='utf-8').read()
if 'FINO-WSL-PATCH: optional distrax' in s:
    print('already patched')
else:
    old = "import distrax\nimport flax.linen as nn"
    new = ("try:  # FINO-WSL-PATCH: optional distrax\n"
           "    import distrax\n"
           "except Exception:  # jax>=0.12 removed jax.interpreters.xla.pytype_aval_mappings used by distrax 0.1.5\n"
           "    distrax = None\n"
           "import flax.linen as nn")
    assert old in s, 'import block not found'
    s = s.replace(old, new, 1)
    old_cls = ("class TransformedWithMode(distrax.Transformed):\n"
               '    """Transformed distribution with mode calculation."""\n'
               "\n"
               "    def mode(self):\n"
               "        return self.bijector.forward(self.distribution.mode())\n")
    new_cls = ("if distrax is not None:\n"
               "    class TransformedWithMode(distrax.Transformed):\n"
               '        """Transformed distribution with mode calculation."""\n'
               "\n"
               "        def mode(self):\n"
               "            return self.bijector.forward(self.distribution.mode())\n"
               "else:  # pragma: no cover - FINO/FQL do not use the Gaussian Actor\n"
               "    class TransformedWithMode:\n"
               "        def __init__(self, *args, **kwargs):\n"
               "            raise RuntimeError('distrax is unavailable with this JAX version; '\n"
               "                               'the Gaussian Actor is not used by FINO/FQL.')\n")
    assert old_cls in s, 'TransformedWithMode block not found'
    s = s.replace(old_cls, new_cls, 1)
    io.open(p, 'w', encoding='utf-8').write(s)
    print('patched', p)
PY

echo "=== [2/2] import + GPU benchmark ==="
cd /opt/fino/FINO
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=.60 \
  /opt/fino-venv/bin/python /mnt/e/FINO/scripts/bench_gpu.py
echo "SETUP5_DONE"
