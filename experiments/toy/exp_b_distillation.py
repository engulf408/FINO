"""实验 B：one-step distillation（对应 agents/fino.py::actor_loss 的 distill_loss）。

先训练一个 10-step BC flow 教师（带 injected noise），再把
actor_onestep_flow(s, noise) 蒸馏到教师的多步 Euler 输出上。
评估：一步 vs 多步采样的动作分布差异（energy distance / MMD）、与真实数据的距离、
推理时间（CPU，jitted，warm-up 后计时）。
输出：CSV/JSON/PNG。全部写入 E:\\FINO\\experiments\\toy\\。
"""
from __future__ import annotations

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from toy_common import (
    OFFSETS_2D,
    OUT,
    energy_distance,
    hist_entropy,
    make_dataset,
    make_euler_sampler,
    make_flow,
    make_flow_sampler,
    mean_std,
    mmd_rbf,
    mode_histogram,
    sample_euler,
    save_csv,
    save_json,
    train_flow,
    train_onestep_distill,
)

SEED = 100
TEACHER_STEPS = 3000
DISTILL_STEPS = 3000
FLOW_STEPS = 10
N_EVAL = 2048


def balanced_states(test, n_per_cluster=1024):
    idx = []
    for c in range(2):
        idx.append(np.where(test["clusters"] == c)[0][:n_per_cluster])
    return np.concatenate(idx)


def distribution_metrics(actions, real_actions, states, action_dim=2):
    emd = energy_distance(actions, real_actions)
    mmd = mmd_rbf(actions, real_actions)
    hist = mode_histogram(actions, states, action_dim)
    hist_r = mode_histogram(real_actions, states, action_dim)
    return dict(
        emd=emd,
        mmd=mmd,
        mode_entropy=hist_entropy(hist),
        mode_entropy_real=hist_entropy(hist_r),
        mode_kl=float(np.sum(hist * np.log((hist + 1e-9) / (hist_r + 1e-9)))),
        mode_max_frac=float(hist.max()),
        mode_hist=hist,
        mode_hist_real=hist_r,
    )


def timeit(jitted_fn, *args, n=30):
    jax.block_until_ready(jitted_fn(*args))  # warmup / compile
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        out = jitted_fn(*args)
        jax.block_until_ready(out)
        ts.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(ts)), float(np.mean(ts))


def analytic_q_2d(states, actions):
    a_pref = states + OFFSETS_2D[0]
    return -(np.sum((actions - a_pref) ** 2, axis=-1)) / 0.2 + 3.0


def train_onestep_with_q(
    teacher_apply, teacher_params, observations, actions, *, alpha: float,
    seed: int, steps: int = 1500, batch_size: int = 256, lr: float = 3e-4,
    flow_steps: int = 10,
):
    """FINO actor_loss 的 toy 版：loss = alpha * distill_loss + q_loss。

    真实代码里 q_loss 由 critic 提供；这里用解析 toy Q（偏好 mode 0）代替，
    以扫描 alpha（蒸馏项权重）对"蒸馏保真度 vs Q 提升"的权衡。
    """
    act_dim = actions.shape[1]
    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)
    module, params = make_flow(init_key, observations.shape[1], act_dim, with_time=False)
    state = TrainState.create(apply_fn=module.apply, params=params, tx=optax.adam(lr))
    obs_j = jnp.asarray(observations)
    n = observations.shape[0]

    @jax.jit
    def step(state, obs, key):
        noise = jax.random.normal(key, (obs.shape[0], act_dim))
        target = sample_euler(teacher_apply, teacher_params, obs, noise, flow_steps)

        def loss_fn(params):
            pred = state.apply_fn(params, obs, noise)
            distill = jnp.mean((pred - target) ** 2)
            a_pref = obs + jnp.asarray(OFFSETS_2D[0])
            q = -jnp.sum((pred - a_pref) ** 2, axis=-1) / 0.2 + 3.0
            return alpha * distill - q.mean(), distill

        (loss, distill), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        return state.apply_gradients(grads=grads), distill

    data_key = key
    for _ in range(steps):
        data_key, batch_key, step_key = jax.random.split(data_key, 3)
        idx = jax.random.randint(batch_key, (batch_size,), 0, n)
        state, _ = step(state, obs_j[idx], step_key)
    return module, state.params


def main():
    print("[data] 载入/合成 2D 数据 ...", flush=True)
    train = make_dataset(30000, seed=10, action_dim=2)
    test = make_dataset(10000, seed=11, action_dim=2)

    print(f"[train] teacher BC flow (inject noise, {TEACHER_STEPS} steps) ...", flush=True)
    teacher, teacher_params, teacher_losses = train_flow(
        train["observations"], train["actions"],
        inject_noise=True, seed=SEED, steps=TEACHER_STEPS,
        batch_size=256, lr=3e-4, hidden=(128, 128),
    )

    print(f"[train] one-step distill ({DISTILL_STEPS} steps) ...", flush=True)
    student, student_params, distill_losses = train_onestep_distill(
        teacher.apply, teacher_params, train["observations"], train["actions"],
        seed=SEED + 1, steps=DISTILL_STEPS, batch_size=256, lr=3e-4,
        flow_steps=FLOW_STEPS, hidden=(128, 128),
    )

    idx = balanced_states(test)
    states = test["observations"][idx]
    real = test["actions"][idx]
    rng = np.random.default_rng(2024)
    noises = jnp.asarray(rng.normal(size=(len(idx), 2)).astype(np.float32))
    s_j = jnp.asarray(states)

    teacher_sampler = make_euler_sampler(teacher.apply, teacher_params, FLOW_STEPS)
    student_sampler = make_flow_sampler(student.apply, student_params)

    teacher_actions = np.asarray(teacher_sampler(s_j, noises))
    student_actions = np.asarray(student_sampler(s_j, noises))

    m_teacher = distribution_metrics(teacher_actions, real, states)
    m_student = distribution_metrics(student_actions, real, states)
    m_student_vs_teacher = distribution_metrics(student_actions, teacher_actions, states)
    one_vs_teacher_mse = float(np.mean((student_actions - teacher_actions) ** 2))
    per_dim_mse = np.mean((student_actions - teacher_actions) ** 2, axis=0)

    # flow_steps 敏感性（教师）
    step_rows = []
    for steps in [1, 2, 5, 10, 20, 50]:
        sampler = make_euler_sampler(teacher.apply, teacher_params, steps)
        acts = np.asarray(sampler(s_j, noises))
        mm = distribution_metrics(acts, real, states)
        ms, _ = timeit(sampler, s_j, noises, n=10)
        # 与 10 步教师的差异也记录
        mm_t = distribution_metrics(acts, teacher_actions, states)
        step_rows.append(dict(
            flow_steps=steps, emd_to_data=mm["emd"], mmd_to_data=mm["mmd"],
            mode_entropy=mm["mode_entropy"], mode_kl=mm["mode_kl"],
            emd_to_teacher10=mm_t["emd"], ms=ms,
        ))
        print(f"  flow_steps={steps:2d}  emd_to_data={mm['emd']:.4f}  emd_to_teacher10={mm_t['emd']:.4f}  {ms:.2f} ms", flush=True)

    # 推理时间对比（N=4096）
    n_time = 4096
    rng2 = np.random.default_rng(2025)
    idx_time = rng2.integers(0, len(test["actions"]), n_time)
    s_time = jnp.asarray(test["observations"][idx_time])
    n_time_j = jnp.asarray(rng2.normal(size=(n_time, 2)).astype(np.float32))
    one_ms, _ = timeit(student_sampler, s_time, n_time_j, n=30)
    ten_ms, _ = timeit(teacher_sampler, s_time, n_time_j, n=30)
    ten_ms_per_sample = ten_ms / n_time
    one_ms_per_sample = one_ms / n_time

    rows = [
        dict(model="teacher_flow10", emd_to_data=m_teacher["emd"], mmd_to_data=m_teacher["mmd"],
             mode_entropy=m_teacher["mode_entropy"], mode_kl=m_teacher["mode_kl"],
             mode_max_frac=m_teacher["mode_max_frac"],
             emd_to_teacher10=0.0, mse_to_teacher10=0.0),
        dict(model="student_onestep", emd_to_data=m_student["emd"], mmd_to_data=m_student["mmd"],
             mode_entropy=m_student["mode_entropy"], mode_kl=m_student["mode_kl"],
             mode_max_frac=m_student["mode_max_frac"],
             emd_to_teacher10=m_student_vs_teacher["emd"], mse_to_teacher10=one_vs_teacher_mse),
    ]
    save_csv(OUT / "exp_b_distill_metrics.csv", rows, list(rows[0].keys()))
    save_csv(OUT / "exp_b_flow_steps.csv", step_rows, list(step_rows[0].keys()))

    # ---- 附带小消融：FINO actor_loss 中 distill 权重 alpha 的作用 ----
    # 真实 loss = bc_flow + alpha * distill + q_loss；这里只保留 one-step 项：
    # loss = alpha * distill + q_loss，用解析 toy Q 观察 alpha 对
    # "蒸馏保真度 vs Q 提升/多样性坍缩" 的权衡。
    print("[alpha] 扫描 distill 权重 alpha（toy Q，偏好 mode 0）...", flush=True)
    alpha_rows, alpha_curves = [], {}
    for alpha in [1.0, 10.0, 100.0, 1000.0]:
        smod, sparams = train_onestep_with_q(
            teacher.apply, teacher_params, train["observations"], train["actions"],
            alpha=alpha, seed=SEED + 100, steps=1500, flow_steps=FLOW_STEPS,
        )
        ssampler = make_flow_sampler(smod.apply, sparams)
        acts = np.asarray(ssampler(s_j, noises))
        hist = mode_histogram(acts, states, 2)
        row = dict(
            alpha=alpha,
            distill_mse_to_teacher=float(np.mean((acts - teacher_actions) ** 2)),
            mean_q=float(np.mean(analytic_q_2d(states, acts))),
            emd_to_teacher=energy_distance(acts, teacher_actions),
            emd_to_data=energy_distance(acts, real),
            mode_entropy=hist_entropy(hist),
            mode_max_frac=float(hist.max()),
        )
        alpha_rows.append(row)
        alpha_curves[alpha] = acts
        print(f"  alpha={alpha:7.0f}  Q={row['mean_q']:+.3f}  "
              f"distill_mse={row['distill_mse_to_teacher']:.5f}  "
              f"EMD_teacher={row['emd_to_teacher']:.4f}  mode_H={row['mode_entropy']:.3f}", flush=True)
    teacher_q = float(np.mean(analytic_q_2d(states, teacher_actions)))
    save_csv(OUT / "exp_b_alpha_sweep.csv", alpha_rows, list(alpha_rows[0].keys()))
    print(f"  teacher baseline: Q={teacher_q:+.3f}", flush=True)

    summary = dict(
        config=dict(seed=SEED, teacher_steps=TEACHER_STEPS, distill_steps=DISTILL_STEPS,
                    flow_steps=FLOW_STEPS, n_eval=N_EVAL),
        teacher=dict({k: (v.tolist() if hasattr(v, "tolist") else v)
                      for k, v in m_teacher.items() if k not in ("mode_hist", "mode_hist_real")}),
        student=dict({k: (v.tolist() if hasattr(v, "tolist") else v)
                      for k, v in m_student.items() if k not in ("mode_hist", "mode_hist_real")}),
        student_vs_teacher=dict({k: (v.tolist() if hasattr(v, "tolist") else v)
                                 for k, v in m_student_vs_teacher.items()
                                 if k not in ("mode_hist", "mode_hist_real")}),
        one_vs_teacher_mse=one_vs_teacher_mse,
        per_dim_mse=per_dim_mse.tolist(),
        timing=dict(
            n=n_time,
            one_step_ms=one_ms, one_step_ms_per_sample=one_ms_per_sample,
            teacher10_ms=ten_ms, teacher10_ms_per_sample=ten_ms_per_sample,
            speedup=ten_ms / one_ms,
        ),
        alpha_sweep=dict(rows=alpha_rows, teacher_mean_q=teacher_q,
                         note="alpha 为 distill_loss 权重；此处 loss = alpha*distill + q_toy_loss"),
        flow_steps_sensitivity=step_rows,
        teacher_final_loss=teacher_losses[-1],
        distill_final_loss=distill_losses[-1],
    )
    save_json(OUT / "exp_b_summary.json", summary)
    print(f"[summary] distill mse={one_vs_teacher_mse:.5f}  "
          f"emd(student vs teacher)={m_student_vs_teacher['emd']:.4f}  "
          f"emd(student vs data)={m_student['emd']:.4f}  (teacher vs data {m_teacher['emd']:.4f})", flush=True)
    print(f"[timing] one-step {one_ms:.2f}ms vs 10-step {ten_ms:.2f}ms (N={n_time}), speedup={ten_ms / one_ms:.2f}x", flush=True)

    # ---------------- 图 ----------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    ax.scatter(real[:, 0], real[:, 1], s=5, alpha=0.25, label="offline data")
    ax.scatter(teacher_actions[:, 0], teacher_actions[:, 1], s=5, alpha=0.35, label="teacher 10-step")
    ax.set_title("data vs teacher flow samples")
    ax.legend(markerscale=2)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.scatter(teacher_actions[:, 0], teacher_actions[:, 1], s=5, alpha=0.25, label="teacher 10-step")
    ax.scatter(student_actions[:, 0], student_actions[:, 1], s=5, alpha=0.35, label="student 1-step")
    ax.set_title(f"teacher vs student (EMD={m_student_vs_teacher['emd']:.4f})")
    ax.legend(markerscale=2)
    ax.grid(alpha=0.3)

    ax = axes[2]
    var_names = ["emd", "mmd", "mode_kl"]
    t_vals = [m_teacher[v] for v in var_names]
    s_vals = [m_student[v] for v in var_names]
    x = np.arange(len(var_names))
    ax.bar(x - 0.18, t_vals, 0.36, label="teacher 10-step")
    ax.bar(x + 0.18, s_vals, 0.36, label="student 1-step")
    ax.set_xticks(x, ["EMD to data", "MMD to data", "mode KL"], rotation=10)
    ax.set_title("sample distribution quality")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "fig_b_distill.png", dpi=150)

    # 第二步图：flow steps 与时间/质量
    fig, ax1 = plt.subplots(figsize=(6.5, 4))
    steps = [r["flow_steps"] for r in step_rows]
    ax1.plot(steps, [r["emd_to_data"] for r in step_rows], "-o", color="tab:blue", label="EMD to data")
    ax1.plot(steps, [r["emd_to_teacher10"] for r in step_rows], "-s", color="tab:red", label="EMD to 10-step teacher")
    ax1.set_xscale("log")
    ax1.set_xlabel("Euler steps")
    ax1.set_ylabel("energy distance")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_b_steps.png", dpi=150)

    # alpha 扫描图
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    alphas = [r["alpha"] for r in alpha_rows]
    axes[0].plot(alphas, [r["mean_q"] for r in alpha_rows], "-o", color="tab:red", label="student Q")
    axes[0].axhline(teacher_q, ls="--", color="gray", label=f"teacher Q={teacher_q:.2f}")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("alpha (distill loss weight)")
    axes[0].set_ylabel("mean toy Q")
    axes[0].set_title("alpha -> Q / diversity tradeoff")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(alphas, [r["emd_to_teacher"] for r in alpha_rows], "-o", label="EMD to teacher")
    axes[1].plot(alphas, [r["emd_to_data"] for r in alpha_rows], "-s", label="EMD to data")
    axes[1].plot(alphas, [r["mode_entropy"] for r in alpha_rows], "-^", label="mode entropy")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("alpha (distill loss weight)")
    axes[1].set_title("distillation fidelity / diversity")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_b_alpha.png", dpi=150)
    print("[done] 实验 B 完成。", flush=True)


if __name__ == "__main__":
    main()
