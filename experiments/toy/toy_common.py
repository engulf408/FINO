"""FINO 2D/6D 玩具实验公共工具。

严格复用 E:\\FINO\\FINO\\utils\\networks.py 中的 ActorVectorField / Value / MLP，
算法（flow matching 目标、noise injection、one-step distill、Q-guided sampling、
熵自适应 beta）参照 E:\\FINO\\FINO\\agents\\fino.py。

本文件只写入 experiments/toy 目录，不修改仓库源码。
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np

REPO = Path(r"E:\FINO\FINO")
OUT = Path(r"E:\FINO\experiments\toy")
OUT.mkdir(parents=True, exist_ok=True)
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# networks.py 顶部 import distrax，但其中只有未使用的 Gaussian Actor 需要 distrax。
# 若 Lead 的安装尚未带上 distrax，提供一个最小占位模块，保证可以复用
# ActorVectorField / Value / MLP；这不影响任何被调用的代码路径。
try:  # pragma: no cover
    import distrax  # noqa: F401
except ImportError:  # pragma: no cover
    class _Unavailable:
        def __init__(self, *args, **kwargs):
            raise ImportError("distrax not installed; Gaussian Actor is unused in toy experiments.")

    _stub = types.ModuleType("distrax")
    _stub.Transformed = _Unavailable
    _stub.Block = _Unavailable
    _stub.Tanh = _Unavailable
    _stub.MultivariateNormalDiag = _Unavailable
    sys.modules["distrax"] = _stub

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402
from flax.training.train_state import TrainState  # noqa: E402

from utils.networks import ActorVectorField, Value  # noqa: E402

jax.config.update("jax_enable_x64", False)

# ---------------------------------------------------------------------------
# 合成数据：state s in R^2，决定 4 个高斯模态的中心；action 各向异性噪声
# ---------------------------------------------------------------------------

STATE_CENTERS_2D = np.array([[1.5, 0.0], [-1.5, 0.0]], dtype=np.float32)

OFFSETS_2D = np.array(
    [
        [0.7, 0.7],
        [0.7, -0.7],
        [-0.7, 0.7],
        [-0.7, -0.7],
    ],
    dtype=np.float32,
)
SIGMA_2D = np.array([0.16, 0.07], dtype=np.float32)

# 6D 版本：4 个模态中心由固定的 2x6 线性映射 + 固定 6D 偏移给出。
_W6_RNG = np.random.RandomState(1234)
W6 = (_W6_RNG.normal(size=(2, 6)) * 0.55).astype(np.float32)
OFFSETS_6D = np.array(
    [
        [0.9, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.8, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.7, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.6, 0.0, 0.0],
    ],
    dtype=np.float32,
)
SIGMA_6D = np.array([0.14, 0.11, 0.09, 0.08, 0.07, 0.06], dtype=np.float32)


def make_dataset(n: int, seed: int, action_dim: int = 2) -> dict:
    """合成 state-conditioned 多模态离线数据。

    返回 dict: observations (n,2), actions (n,A), clusters (n,), modes (n,),
    centers (n,A)（该样本所属模态的真实中心）。
    """
    rng = np.random.default_rng(seed)
    clusters = rng.integers(0, 2, size=n)
    observations = STATE_CENTERS_2D[clusters] + 0.15 * rng.normal(size=(n, 2)).astype(np.float32)

    if action_dim == 2:
        modes = rng.integers(0, 4, size=n)
        centers = observations + OFFSETS_2D[modes]
        sigma = SIGMA_2D
    elif action_dim == 6:
        modes = rng.integers(0, 4, size=n)
        centers = observations @ W6 + OFFSETS_6D[modes]
        sigma = SIGMA_6D
    else:  # pragma: no cover
        raise ValueError(action_dim)

    actions = centers + sigma[None, :] * rng.normal(size=(n, action_dim)).astype(np.float32)
    return dict(
        observations=observations.astype(np.float32),
        actions=actions.astype(np.float32),
        clusters=clusters.astype(np.int32),
        modes=modes.astype(np.int32),
        centers=centers.astype(np.float32),
    )


def save_dataset(path: Path, train: dict, test: dict) -> None:
    np.savez(
        path,
        **{f"train_{k}": v for k, v in train.items()},
        **{f"test_{k}": v for k, v in test.items()},
    )


def mode_centers_for(states: np.ndarray, action_dim: int) -> np.ndarray:
    """(N,2) states -> (N,4,A) 4 个模态中心。两维与 6D 数据保持一致。"""
    if action_dim == 2:
        return states[:, None, :] + OFFSETS_2D[None, :, :]
    return states[:, None, :] @ W6[None, :, :] + OFFSETS_6D[None, :, :]


def gt_conditional_velocity(
    states: np.ndarray, x_t: np.ndarray, t: float, action_dim: int, sigma: np.ndarray
) -> np.ndarray:
    """BC flow 的 MSE 最优目标 E[x1 - x0 | s, x_t, t]（4 个高斯模态的后验加权）。

    x0 ~ N(0, I)，x1 | s, mode m ~ N(center_m, diag(sigma^2))，
    x_t = (1-t) x0 + t x1。返回 (N,A)。
    """
    centers = mode_centers_for(states, action_dim)  # (N,M,A)
    tt = float(t)
    var_t = (1 - tt) ** 2 + tt**2 * sigma**2  # (A,)
    diff = x_t[:, None, :] - tt * centers  # (N,M,A)
    logp = -0.5 * np.sum(diff**2 / var_t[None, None, :], axis=-1)  # (N,M)
    logp -= logp.max(axis=1, keepdims=True)
    w = np.exp(logp)
    w /= w.sum(axis=1, keepdims=True)
    # E[x1 | x_t] for each mode
    ex1_m = centers + (tt * sigma[None, None, :] ** 2 / var_t[None, None, :]) * diff
    ex1 = np.sum(w[:, :, None] * ex1_m, axis=1)  # (N,A)
    if tt > 0.999:
        # t=1: x_t = x1，x0 与 x1 独立 => E[vel]=x1
        return x_t
    ex0 = (x_t - tt * ex1) / (1 - tt)
    return ex1 - ex0


# ---------------------------------------------------------------------------
# 网络与训练
# ---------------------------------------------------------------------------


def make_flow(key, ob_dim: int, act_dim: int, hidden=(128, 128), with_time: bool = True):
    """ActorVectorField；with_time=True 对应 actor_bc_flow（输入含 t），
    False 对应 actor_onestep_flow（输入只有 s 和 noise，与 fino.py::create 一致）。"""
    module = ActorVectorField(hidden_dims=hidden, action_dim=act_dim, layer_norm=False)
    if with_time:
        params = module.init(
            key,
            jnp.zeros((1, ob_dim), jnp.float32),
            jnp.zeros((1, act_dim), jnp.float32),
            jnp.zeros((1, 1), jnp.float32),
        )
    else:
        params = module.init(
            key,
            jnp.zeros((1, ob_dim), jnp.float32),
            jnp.zeros((1, act_dim), jnp.float32),
        )
    return module, params


def make_critic(key, ob_dim: int, act_dim: int, hidden=(128, 128), num_ensembles=2):
    module = Value(hidden_dims=hidden, layer_norm=True, num_ensembles=num_ensembles)
    params = module.init(
        key, jnp.zeros((1, ob_dim), jnp.float32), jnp.zeros((1, act_dim), jnp.float32)
    )
    return module, params


def noise_scale_schedule(t, noise_scale: float = 0.1, schedule: str = "exp"):
    """注入噪声 std 的 schedule（对应 agents/fino.py L57）。

    - "exp"  : eta * exp(-(t*-10+10)) = eta*exp(10(t-1))，官方代码所用；
    - "quad" : sqrt((eta^2 - 2*eta) t^2 + 2*eta*t)，论文 Eq.(7) 的二次型 schedule
               （用于小消融；端点 0 和 eta 与指数 schedule 相同）；
    - "const": 常数 eta（额外参照）。
    """
    if schedule == "exp":
        return jnp.exp(-(t * -10 + 10)) * noise_scale
    if schedule == "quad":
        var = (noise_scale**2 - 2 * noise_scale) * t**2 + 2 * noise_scale * t
        return jnp.sqrt(jnp.maximum(var, 0.0))
    if schedule == "const":
        return jnp.full_like(t, noise_scale)
    raise ValueError(schedule)


def bc_flow_loss(apply_fn, params, observations, actions, key, inject_noise, noise_scale=0.1,
                 schedule="exp"):
    """与 agents/fino.py::actor_loss 中 bc_flow_loss 相同的写法。"""
    batch_size, action_dim = actions.shape
    x_rng, t_rng, n_rng = jax.random.split(key, 3)
    x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
    x_1 = actions
    t = jax.random.uniform(t_rng, (batch_size, 1))
    x_t = (1 - t) * x_0 + t * x_1
    vel = x_1 - x_0
    if inject_noise:
        normal_noise = jax.random.normal(n_rng, shape=(batch_size, action_dim))
        scale = noise_scale_schedule(t, noise_scale, schedule)
        normal_samples = x_t + scale * normal_noise
    else:
        normal_samples = x_t
    pred = apply_fn(params, observations, normal_samples, t)
    return jnp.mean((pred - vel) ** 2)


def train_flow(
    observations: np.ndarray,
    actions: np.ndarray,
    *,
    inject_noise: bool,
    seed: int,
    steps: int = 2500,
    batch_size: int = 256,
    lr: float = 3e-4,
    hidden=(128, 128),
    noise_scale: float = 0.1,
    schedule: str = "exp",
    log_every: int = 500,
):
    """训练 BC flow，返回 (module, params, loss_curve)。"""
    ob_dim, act_dim = observations.shape[1], actions.shape[1]
    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)
    module, params = make_flow(init_key, ob_dim, act_dim, hidden)
    state = TrainState.create(apply_fn=module.apply, params=params, tx=optax.adam(lr))

    obs_j = jnp.asarray(observations)
    act_j = jnp.asarray(actions)
    n = observations.shape[0]

    @jax.jit
    def step(state, obs, act, key):
        def loss_fn(params):
            return bc_flow_loss(
                state.apply_fn, params, obs, act, key, inject_noise, noise_scale, schedule
            )

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        return state.apply_gradients(grads=grads), loss

    losses = []
    data_key = key
    for i in range(steps):
        data_key, batch_key, step_key = jax.random.split(data_key, 3)
        idx = jax.random.randint(batch_key, (batch_size,), 0, n)
        state, loss = step(state, obs_j[idx], act_j[idx], step_key)
        losses.append(float(loss))
        if log_every and (i + 1) % log_every == 0:
            print(f"    step {i + 1:5d}/{steps}  bc_flow_loss={float(loss):.5f}", flush=True)
    return module, state.params, losses


def sample_euler(apply_fn, params, observations, noises, steps: int = 10):
    """agents/fino.py::compute_flow_actions 的 Euler 多步采样（不含 clip）。"""
    actions = noises
    for i in range(steps):
        t = jnp.full((observations.shape[0], 1), i / steps, jnp.float32)
        vels = apply_fn(params, observations, actions, t)
        actions = actions + vels / steps
    return actions


def make_euler_sampler(apply_fn, params, steps: int = 10):
    @jax.jit
    def _sample(obs, noises):
        return sample_euler(apply_fn, params, obs, noises, steps)

    return _sample


def make_flow_sampler(apply_fn, params):
    """one-step 采样：actor_onestep_flow(observations, noises)。"""

    @jax.jit
    def _sample(obs, noises):
        return apply_fn(params, obs, noises)

    return _sample


def train_onestep_distill(
    teacher_apply,
    teacher_params,
    observations: np.ndarray,
    actions: np.ndarray,
    *,
    seed: int,
    steps: int = 3000,
    batch_size: int = 256,
    lr: float = 3e-4,
    flow_steps: int = 10,
    hidden=(128, 128),
    log_every: int = 500,
):
    """agents/fino.py::actor_loss 中 distill_loss 的玩具版：
    actor_onestep_flow(s, noise) 回归教师多步 flow 的输出。"""
    ob_dim, act_dim = observations.shape[1], actions.shape[1]
    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)
    module, params = make_flow(init_key, ob_dim, act_dim, hidden, with_time=False)
    state = TrainState.create(apply_fn=module.apply, params=params, tx=optax.adam(lr))
    obs_j = jnp.asarray(observations)
    n = observations.shape[0]

    @jax.jit
    def step(state, obs, key):
        noise = jax.random.normal(key, (obs.shape[0], act_dim))
        target = sample_euler(teacher_apply, teacher_params, obs, noise, flow_steps)

        def loss_fn(params):
            pred = state.apply_fn(params, obs, noise)
            return jnp.mean((pred - target) ** 2)

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        return state.apply_gradients(grads=grads), loss

    losses = []
    data_key = key
    for i in range(steps):
        data_key, batch_key, step_key = jax.random.split(data_key, 3)
        idx = jax.random.randint(batch_key, (batch_size,), 0, n)
        state, loss = step(state, obs_j[idx], step_key)
        losses.append(float(loss))
        if log_every and (i + 1) % log_every == 0:
            print(f"    step {i + 1:5d}/{steps}  distill_loss={float(loss):.5f}", flush=True)
    return module, state.params, losses


def train_critic_regression(
    observations: np.ndarray,
    actions: np.ndarray,
    targets: np.ndarray,
    *,
    seed: int,
    steps: int = 2500,
    batch_size: int = 256,
    lr: float = 1e-3,
    hidden=(128, 128),
):
    """用监督回归拟合玩具 Q 函数（Value 是 2-ensemble，取 mean 回归）。"""
    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)
    module, params = make_critic(init_key, observations.shape[1], actions.shape[1], hidden)
    state = TrainState.create(apply_fn=module.apply, params=params, tx=optax.adam(lr))
    obs_j, act_j, tgt_j = jnp.asarray(observations), jnp.asarray(actions), jnp.asarray(targets)
    n = observations.shape[0]

    @jax.jit
    def step(state, obs, act, tgt):
        def loss_fn(params):
            qs = state.apply_fn(params, obs, actions=act)
            q = qs.mean(axis=0)
            return jnp.mean((q - tgt) ** 2)

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        return state.apply_gradients(grads=grads), loss

    losses = []
    data_key = key
    for i in range(steps):
        data_key, batch_key = jax.random.split(data_key)
        idx = jax.random.randint(batch_key, (batch_size,), 0, n)
        state, loss = step(state, obs_j[idx], act_j[idx], tgt_j[idx])
        losses.append(float(loss))
    return module, state.params, losses


# ---------------------------------------------------------------------------
# 指标
# ---------------------------------------------------------------------------


def _pairwise_sq_dists(x, y):
    return np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1)


def energy_distance(x: np.ndarray, y: np.ndarray, max_n: int = 1024) -> float:
    """unbiased energy distance（越小越接近）。"""
    x = np.asarray(x, np.float64)[:max_n]
    y = np.asarray(y, np.float64)[:max_n]
    d_xy = np.sqrt(np.maximum(_pairwise_sq_dists(x, y), 0)).mean()
    d_xx = np.sqrt(np.maximum(_pairwise_sq_dists(x, x), 0)).mean()
    d_yy = np.sqrt(np.maximum(_pairwise_sq_dists(y, y), 0)).mean()
    return float(2 * d_xy - d_xx - d_yy)


def mmd_rbf(x: np.ndarray, y: np.ndarray, max_n: int = 1024) -> float:
    """RBF-kernel MMD^2，带宽取合并样本距离的中位数。"""
    x = np.asarray(x, np.float64)[:max_n]
    y = np.asarray(y, np.float64)[:max_n]
    z = np.concatenate([x, y], axis=0)
    d2 = _pairwise_sq_dists(z, z)
    med = np.median(d2[d2 > 0])
    gamma = 1.0 / max(med, 1e-8)
    k_xx = np.exp(-gamma * _pairwise_sq_dists(x, x))
    k_yy = np.exp(-gamma * _pairwise_sq_dists(y, y))
    k_xy = np.exp(-gamma * _pairwise_sq_dists(x, y))
    return float(k_xx.mean() + k_yy.mean() - 2 * k_xy.mean())


def mode_assignment(actions: np.ndarray, states: np.ndarray, action_dim: int) -> np.ndarray:
    """把动作分配给最近的模态中心，返回 (N,) 模态 index。"""
    centers = mode_centers_for(states, action_dim)  # (N,M,A)
    d = np.sum((actions[:, None, :] - centers) ** 2, axis=-1)
    return np.argmin(d, axis=1)


def mode_histogram(actions, states, action_dim):
    m = mode_assignment(actions, states, action_dim)
    counts = np.bincount(m, minlength=4).astype(np.float64)
    return counts / counts.sum()


def hist_entropy(p: np.ndarray) -> float:
    p = np.asarray(p, np.float64)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def local_lipschitz(
    apply_fn, params, observations, times, act_dim: int = 2, delta: float = 0.05, key=None
):
    """||v(x+δ)-v(x)|| / ||δ|| 的平均值（action 空间有限差分，δ 为高斯方向）。"""
    if key is None:
        key = jax.random.PRNGKey(0)
    n = observations.shape[0]
    obs = jnp.asarray(observations)
    tt = jnp.full((n, 1), times, jnp.float32)
    x = jax.random.normal(key, (n, act_dim))
    d = jax.random.normal(jax.random.PRNGKey(1), (n, act_dim))
    d = d / jnp.linalg.norm(d, axis=-1, keepdims=True)
    v0 = apply_fn(params, obs, x, tt)
    v1 = apply_fn(params, obs, x + delta * d, tt)
    ratio = jnp.linalg.norm(v1 - v0, axis=-1) / delta
    return float(jnp.mean(ratio))


def knn_entropy(samples: np.ndarray, k: int = 3) -> float:
    """Kozachenko-Leonenko kNN 熵估计，samples: (n, d)。用于替换 sklearn GMM
    （环境未装 sklearn 时；仓库用的是 GMM，见 RESULTS.md 说明）。"""
    from math import lgamma, pi

    x = np.asarray(samples, np.float64)
    n, d = x.shape
    d2 = _pairwise_sq_dists(x, x)
    np.fill_diagonal(d2, np.inf)
    eps = np.sqrt(np.sort(d2, axis=1)[:, k - 1])  # 到第 k 近邻的距离
    eps = np.maximum(eps, 1e-12)
    log_cd = 0.5 * d * np.log(pi) - lgamma(0.5 * d + 1)
    from scipy.special import digamma  # noqa: F401  (scipy 由 jax 依赖自带)

    return float(
        digamma(n) - digamma(k) + log_cd + d * np.mean(np.log(eps))
    )


def save_json(path: Path, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_csv(path: Path, rows, fieldnames) -> None:
    import csv

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def mean_std(values):
    a = np.asarray(values, np.float64)
    return float(a.mean()), float(a.std(ddof=0))
