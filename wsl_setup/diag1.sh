#!/bin/bash
# FINO reproduction - WSL diagnostics: which API changed under jax 0.12 / flax 0.12 / orbax 0.12
cd /opt/fino/FINO || exit 1
/opt/fino-venv/bin/python - <<'PY'
import sys, types, traceback

# --- stub distrax so we can test the rest of the code base ---
stub = types.ModuleType('distrax')
class _T:  # minimal stand-in for distrax.Transformed
    def __init__(self, *a, **k): pass
stub.Transformed = _T
stub.Block = lambda *a, **k: None
class _Tanh: pass
stub.Tanh = _Tanh
class _MND:
    def __init__(self, *a, **k): pass
stub.MultivariateNormalDiag = _MND
sys.modules['distrax'] = stub

def check(label, fn):
    try:
        fn()
        print('OK  ', label)
    except Exception as e:
        print('FAIL', label, '->', type(e).__name__, str(e).splitlines()[0][:200])

check('import flax', lambda: __import__('flax'))
import flax, flax.core
check('flax.core.FrozenDict', lambda: flax.core.FrozenDict(a=1))
check('flax.struct.PyTreeNode', lambda: __import__('flax.struct', fromlist=['x']))
check('flax.training.orbax_utils', lambda: __import__('flax.training.orbax_utils', fromlist=['x']))
import orbax, orbax.checkpoint
check('orbax.checkpoint.PyTreeCheckpointer', lambda: getattr(orbax.checkpoint, 'PyTreeCheckpointer'))
check('orbax.checkpoint.StandardCheckpointer', lambda: getattr(orbax.checkpoint, 'StandardCheckpointer'))

check('import utils.networks', lambda: __import__('utils.networks', fromlist=['x']))
check('import utils.flax_utils', lambda: __import__('utils.flax_utils', fromlist=['x']))
check('import utils.datasets', lambda: __import__('utils.datasets', fromlist=['x']))
check('import agents.fino', lambda: __import__('agents.fino', fromlist=['x']))
check('import agents.fql', lambda: __import__('agents.fql', fromlist=['x']))

# ModuleDict / TrainState / MLP init under new flax
def test_networks():
    import jax, jax.numpy as jnp
    from utils.networks import Value, ActorVectorField
    from utils.flax_utils import ModuleDict, TrainState
    import optax
    obs = jnp.zeros((4, 29)); act = jnp.zeros((4, 8)); t = jnp.zeros((4, 1))
    critic = Value(hidden_dims=(64, 64), layer_norm=True, num_ensembles=2)
    actor_bc = ActorVectorField(hidden_dims=(64, 64), action_dim=8, layer_norm=False)
    net_def = ModuleDict(dict(critic=critic, actor_bc_flow=actor_bc))
    params = net_def.init(jax.random.PRNGKey(0), critic=(obs, act), actor_bc_flow=(obs, act, t))['params']
    tx = optax.adam(1e-3)
    state = TrainState.create(net_def, params, tx=tx)
    q = state.select('critic')(obs, actions=act)
    print('     critic out shape:', q.shape)
check('Value/ModuleDict/TrainState init+call', test_networks)
print('DIAG_DONE')
PY
