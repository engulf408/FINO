"""实验 A：BC flow matching 的 injected noise 作用（对应 agents/fino.py::actor_loss 的 bc_flow_loss）。

训练两个设置：
  (1) no-inject : normal_samples = x_t
  (2) inject    : normal_samples = x_t + exp(-(t*-10+10)) * 0.1 * noise
每个设置 3 个随机种子。评估：
  - 与解析最优目标 E[x1-x0|s,x_t,t] 的向量场 MSE（按 t 分开）
  - 多步 Euler 采样的动作分布质量（per-cluster energy distance / MMD / 模态覆盖）
  - 流场在 t≈0（噪声端）与 t≈1（数据端）的局部 Lipschitz 敏感性
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

from toy_common import (
    OUT,
    SIGMA_2D,
    bc_flow_loss,
    energy_distance,
    gt_conditional_velocity,
    hist_entropy,
    local_lipschitz,
    make_dataset,
    make_euler_sampler,
    mean_std,
    mmd_rbf,
    mode_assignment,
    mode_histogram,
    save_csv,
    save_dataset,
    save_json,
    train_flow,
)

SEEDS = [0, 1, 2]
STEPS = 2500
FLOW_STEPS = 10
SETTINGS = {
    # 官方代码 agents/fino.py L56-58 的指数 schedule
    "no_inject": dict(inject=False, schedule="exp"),
    "inject_exp": dict(inject=True, schedule="exp"),
    # 论文 Eq.(7) 二次型 schedule 的小消融（Lead 要求）
    "inject_quad": dict(inject=True, schedule="quad"),
}
T_GRID = np.array([0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0])


def sample_states(test, n_per_cluster=512):
    idx = []
    for c in range(2):
        ids = np.where(test["clusters"] == c)[0]
        idx.append(ids[:n_per_cluster])
    return np.concatenate(idx)


def eval_vector_field(module, params, test, seed=777, n=3000):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(test["actions"]), n)
    s = test["observations"][idx]
    a = test["actions"][idx]
    x0 = rng.normal(size=(n, 2)).astype(np.float32)
    s_j = jnp.asarray(s)
    rows = []
    for t in T_GRID:
        x_t = ((1 - t) * x0 + t * a).astype(np.float32)
        gt = gt_conditional_velocity(s, x_t, float(t), 2, SIGMA_2D)
        pred = module.apply(params, s_j, jnp.asarray(x_t), jnp.full((n, 1), t, jnp.float32))
        rows.append(
            dict(t=float(t), vf_mse=float(np.mean((np.asarray(pred) - gt) ** 2)))
        )
    return rows


def eval_sampling(module, params, test, seed=888, flow_steps=FLOW_STEPS):
    rng = np.random.default_rng(seed)
    idx = sample_states(test)
    s = test["observations"][idx]
    a_real = test["actions"][idx]
    cl = test["clusters"][idx]
    noises = jnp.asarray(rng.normal(size=(len(idx), 2)).astype(np.float32))
    sampler = make_euler_sampler(module.apply, params, flow_steps)
    t0 = time.perf_counter()
    actions = np.asarray(sampler(jnp.asarray(s), noises))
    t_ms = (time.perf_counter() - t0) * 1000.0

    out = {}
    emds, mmds = [], []
    for c in range(2):
        mask = cl == c
        emds.append(energy_distance(actions[mask], a_real[mask]))
        mmds.append(mmd_rbf(actions[mask], a_real[mask]))
    out["emd_cluster0"] = emds[0]
    out["emd_cluster1"] = emds[1]
    out["emd_mean"] = float(np.mean(emds))
    out["mmd_mean"] = float(np.mean(mmds))

    # 模态覆盖：把动作分配到最近模态中心，与真实数据分布对比
    hist = mode_histogram(actions, s, 2)
    hist_real = mode_histogram(a_real, s, 2)
    out["mode_hist"] = hist.tolist()
    out["mode_hist_real"] = hist_real.tolist()
    out["mode_entropy"] = hist_entropy(hist)
    out["mode_entropy_real"] = hist_entropy(hist_real)
    out["mode_kl_to_real"] = float(
        np.sum(hist * np.log((hist + 1e-9) / (hist_real + 1e-9)))
    )
    out["mode_max_frac"] = float(hist.max())
    out["sample_ms"] = t_ms

    out["lipschitz_t01"] = local_lipschitz(
        module.apply, params, jnp.asarray(test["observations"][idx[:1024]]),
        times=0.05, act_dim=2, delta=0.05,
    )
    out["lipschitz_t09"] = local_lipschitz(
        module.apply, params, jnp.asarray(test["observations"][idx[:1024]]),
        times=0.9, act_dim=2, delta=0.05,
    )
    out["lipschitz_t099"] = local_lipschitz(
        module.apply, params, jnp.asarray(test["observations"][idx[:1024]]),
        times=0.99, act_dim=2, delta=0.05,
    )
    out["actions"] = actions
    out["states"] = s
    return out


def main():
    print("[data] 合成 2D state-conditioned 多模态数据集 ...", flush=True)
    train = make_dataset(30000, seed=10, action_dim=2)
    test = make_dataset(10000, seed=11, action_dim=2)
    save_dataset(OUT / "toy_data_2d.npz", train, test)
    print(f"  train actions range {train['actions'].min():.2f}..{train['actions'].max():.2f}", flush=True)

    all_rows = []
    metric_rows = []
    vf_curves = {k: [] for k in SETTINGS}
    sample_store = {}
    for name, cfg in SETTINGS.items():
        for seed in SEEDS:
            print(f"[train] {name} seed={seed} steps={STEPS}", flush=True)
            module, params, losses = train_flow(
                train["observations"], train["actions"],
                inject_noise=cfg["inject"], seed=seed, steps=STEPS,
                batch_size=256, lr=3e-4, hidden=(128, 128),
                schedule=cfg["schedule"],
            )
            vf_rows = eval_vector_field(module, params, test)
            for r in vf_rows:
                all_rows.append(dict(setting=name, seed=seed, t=r["t"], vf_mse=r["vf_mse"]))
            vf_curves[name].append([r["vf_mse"] for r in vf_rows])
            sm = eval_sampling(module, params, test)
            sample_store[(name, seed)] = sm
            row = dict(setting=name, seed=seed, final_bc_flow_loss=losses[-1])
            for k in (
                "emd_cluster0", "emd_cluster1", "emd_mean", "mmd_mean",
                "mode_entropy", "mode_entropy_real", "mode_kl_to_real", "mode_max_frac",
                "lipschitz_t01", "lipschitz_t09", "lipschitz_t099", "sample_ms",
            ):
                row[k] = sm[k]
            row["mode_hist"] = "|".join(f"{v:.3f}" for v in sm["mode_hist"])
            metric_rows.append(row)
            print(
                f"  final_loss={losses[-1]:.4f}  emd={sm['emd_mean']:.4f}  "
                f"mode_entropy={sm['mode_entropy']:.3f}  lip(0.9)={sm['lipschitz_t09']:.3f}",
                flush=True,
            )

    # ---------------- 汇总 ----------------
    fields = list(metric_rows[0].keys())
    save_csv(OUT / "exp_a_metrics.csv", metric_rows, fields)
    save_csv(
        OUT / "exp_a_vf_by_t.csv", all_rows, ["setting", "seed", "t", "vf_mse"]
    )

    summary = {}
    for name in SETTINGS:
        rows = [r for r in metric_rows if r["setting"] == name]
        s = {}
        for k in (
            "emd_mean", "mmd_mean", "mode_entropy", "mode_kl_to_real", "mode_max_frac",
            "lipschitz_t01", "lipschitz_t09", "lipschitz_t099",
            "emd_cluster0", "emd_cluster1",
        ):
            m, sd = mean_std([r[k] for r in rows])
            s[k] = dict(mean=m, std=sd)
        curve = np.asarray(vf_curves[name])  # (seeds, T)
        s["vf_curve"] = {
            "t": T_GRID.tolist(),
            "mean": curve.mean(axis=0).tolist(),
            "std": curve.std(axis=0).tolist(),
        }
        summary[name] = s
        print(f"[summary] {name}: emd={s['emd_mean']['mean']:.4f}±{s['emd_mean']['std']:.4f}", flush=True)

    # 关键对比：t 分段的向量场误差（inject_exp / inject_quad 相对 no_inject）
    seg = {
        "t<=0.1": T_GRID <= 0.1,
        "0.25<=t<=0.75": (T_GRID >= 0.25) & (T_GRID <= 0.75),
        "t>=0.9": T_GRID >= 0.9,
        "t==1.0": T_GRID == 1.0,
    }
    comparisons = {}
    base = np.asarray(vf_curves["no_inject"])
    for k, mask in seg.items():
        b = float(base[:, mask].mean())
        entry = dict(no_inject=b)
        for name in ("inject_exp", "inject_quad"):
            cur = np.asarray(vf_curves[name])
            entry[name] = float(cur[:, mask].mean())
            entry[f"{name}_ratio"] = entry[name] / b if b else None
        comparisons[k] = entry
    summary["vf_segment_comparison"] = comparisons
    # 解析噪声 schedule
    from toy_common import noise_scale_schedule

    summary["injected_noise_scale_schedule"] = {
        "exp": [float(noise_scale_schedule(t, 0.1, "exp")) for t in T_GRID],
        "quad": [float(noise_scale_schedule(t, 0.1, "quad")) for t in T_GRID],
        "t_grid": T_GRID.tolist(),
    }
    summary["config"] = dict(
        seeds=SEEDS, train_steps=STEPS, flow_steps=FLOW_STEPS,
        lr=3e-4, batch_size=256, hidden=[128, 128], noise_scale=0.1,
        data="s~2 clusters + 4 action modes (anisotropic), 30k train / 10k test",
    )
    save_json(OUT / "exp_a_summary.json", summary)

    # ---------------- 图 ----------------
    COLORS = {"no_inject": "tab:blue", "inject_exp": "tab:orange", "inject_quad": "tab:green"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    for name, color in COLORS.items():
        s = summary[name]["vf_curve"]
        m, sd = np.asarray(s["mean"]), np.asarray(s["std"])
        tr = T_GRID[1:]
        ax.plot(tr, m[1:], "-o", ms=3, color=color, label=name)
        ax.fill_between(tr, m[1:] - sd[1:], m[1:] + sd[1:], color=color, alpha=0.2)
    ax.set_yscale("log")
    ax.set_xlabel("flow time t")
    ax.set_ylabel("MSE vs analytic E[x1-x0|s,x_t,t]")
    ax.set_title("Vector-field error vs t (mean±std, 3 seeds)")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    labels = ["energy dist\n(per cluster mean)", "mode KL to real"]
    idx = np.arange(len(labels))
    for j, (name, color) in enumerate(COLORS.items()):
        vals = [summary[name]["emd_mean"]["mean"], summary[name]["mode_kl_to_real"]["mean"]]
        ax.bar(idx + (j - 1) * 0.27, vals, 0.27, label=name, color=color)
    ax.set_xticks(idx, labels)
    ax.set_ylabel("lower is better")
    ax.set_title("Sample distribution quality (10-step Euler)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[2]
    keys = ("lipschitz_t01", "lipschitz_t09", "lipschitz_t099")
    x = np.arange(3)
    for j, (name, color) in enumerate(COLORS.items()):
        vals = [summary[name][k]["mean"] for k in keys]
        ax.bar(x + (j - 1) * 0.27, vals, 0.27, label=name, color=color)
    ax.set_xticks(x, ["t=0.05", "t=0.9", "t=0.99"])
    ax.set_ylabel("local Lipschitz proxy")
    ax.set_title("Flow sensitivity near noise end vs data end")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "fig_a_noise_injection.png", dpi=150)

    # 采样散点对比图（一个 state cluster）
    fig, axes = plt.subplots(1, 4, figsize=(17, 4))
    for ax, name in zip(axes[:3], COLORS):
        sm = sample_store[(name, 0)]
        ax.scatter(sm["actions"][:, 0], sm["actions"][:, 1], s=6, alpha=0.4, label="flow samples")
        ax.scatter(test["actions"][:1024, 0], test["actions"][:1024, 1], s=6, alpha=0.25, label="data")
        ax.set_title(f"{name} (seed 0) actions")
        ax.legend(markerscale=2)
        ax.grid(alpha=0.3)
    sm = sample_store[("inject_exp", 0)]
    axes[3].bar(np.arange(4) - 0.2, sm["mode_hist"], 0.4, label="inject_exp samples")
    axes[3].bar(np.arange(4) + 0.2, sm["mode_hist_real"], 0.4, label="data")
    axes[3].set_xticks(np.arange(4), [f"mode{i}" for i in range(4)])
    axes[3].set_title("mode histogram")
    axes[3].legend()
    axes[3].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "fig_a_samples.png", dpi=150)
    print("[done] 实验 A 完成。", flush=True)


if __name__ == "__main__":
    main()
