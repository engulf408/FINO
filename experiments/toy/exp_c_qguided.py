"""实验 C：Q-guided 采样的行为验证（对应 agents/fino.py::sample_actions）。

包含三部分：
  C1. 2D 玩具：action_dim=2 时 num_samples = min(10,(2+1)//2)=1，Q-guided 选择退化为
      "只有 1 个候选"，选择不改变任何动作；另用 K 覆盖实验说明机制潜力。
  C2. 6D 玩具：action_dim=6 时 num_samples=3，用学习到的 toy critic（Value 集成）
      复现 sample_actions 的 to_env / to_eval 两个分支，对比 first/random/argmax/
      categorical 选择在 toy Q 上的表现，并量化"批处理 to_eval 的全局 argmax 语义"。
  C3. 熵自适应 beta：复现 update_entropy 的更新规则，用 kNN 熵估计替代 sklearn GMM，
      观察 beta 轨迹与 entropy-beta 权衡。
输出 CSV/JSON/PNG。全部写入 E:\\FINO\\experiments\\toy\\。
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp

from toy_common import (
    OFFSETS_2D,
    OFFSETS_6D,
    OUT,
    W6,
    energy_distance,
    hist_entropy,
    knn_entropy,
    make_dataset,
    make_euler_sampler,
    make_flow_sampler,
    mode_assignment,
    save_csv,
    save_dataset,
    save_json,
    train_critic_regression,
    train_flow,
    train_onestep_distill,
)

SEED = 300
FLOW_STEPS = 10


def NUM_SAMPLES_FORMULA(d: int) -> int:
    """与 fino.py L163 完全一致：num_samples = min(10, (action_dim + 1) // 2)。"""
    return min(10, (d + 1) // 2)


def balanced_states(test, n_per_cluster=256):
    idx = []
    for c in range(2):
        idx.append(np.where(test["clusters"] == c)[0][:n_per_cluster])
    return np.concatenate(idx)


def analytic_q_2d(states, actions):
    a_pref = states + OFFSETS_2D[0]
    return -(np.sum((actions - a_pref) ** 2, axis=-1)) / 0.2 + 3.0


def analytic_q_6d(states, actions):
    a_pref = states @ W6 + OFFSETS_6D[0]
    return -(np.sum((actions - a_pref) ** 2, axis=-1)) / 0.8 + 6.0


def onestep_actions(sampler, states, noises):
    """states (S,2), noises (K,S,d) -> actions (K,S,d)，一次 jit 调用。"""
    K, S, d = noises.shape
    s_rep = np.repeat(states[None, :, :], K, axis=0).reshape(K * S, 2)
    acts = sampler(jnp.asarray(s_rep), jnp.asarray(noises.reshape(K * S, d)))
    return np.asarray(acts).reshape(K, S, d)


def critic_q(critic, params, states, actions):
    """actions (K,S,d) -> q (K,S)，与 fino.py 默认 q_agg='mean' 一致。"""
    K, S, d = actions.shape
    s_rep = np.repeat(states[None, :, :], K, axis=0).reshape(K * S, 2)
    qs = critic.apply(params, jnp.asarray(s_rep), actions=jnp.asarray(actions.reshape(K * S, d)))
    return np.asarray(qs).mean(axis=0).reshape(K, S)


def gather(actions, idx):
    """actions (K,S,d), idx (S,) -> (S,d)"""
    return actions[idx, np.arange(actions.shape[1]), :]


def select_methods(q, beta, seed=0):
    """返回 {method: idx (S,)}，复现 fino.py::sample_actions 的选择语义。

    - first: 取第 0 个噪声样本（无选择）
    - random: 在 K 个样本里随机选
    - argmax_perstate: 对每个 state 取 Q 最大的候选（等价于 sample_actions 在
      batch=1 且 temperature=0 时的行为，evaluation.py 正是逐条 observation 调用）
    - argmax_batched_code: fino.py 原代码 `actions[jnp.argmax(q)]` 在 batch>1 时的
      全局 argmax 语义（返回被选中的那一行样本，所有 state 共享同一抽样 index）
    - env_beta: 原代码 to_env 分支：logits = q/|q|.mean() * beta，categorical(axis=0)
    """
    K, S = q.shape
    idx = {}
    idx["first"] = np.zeros(S, dtype=np.int64)
    rng = np.random.default_rng(seed)
    idx["random"] = rng.integers(0, K, size=S)
    idx["argmax_perstate"] = np.argmax(q, axis=0).astype(np.int64)
    flat = int(np.argmax(q))
    # actions[flat] 的形状是 (S,d)，即所有 state 都取 "第 flat//S 个样本"
    idx["argmax_batched_code"] = np.full(S, flat // S, dtype=np.int64)
    logits = (q / np.abs(q).mean()) * beta
    cat = np.asarray(jax.random.categorical(jax.random.PRNGKey(seed), jnp.asarray(logits), axis=0))
    idx["env_beta"] = cat.astype(np.int64)
    return idx


def summarize_selection(actions_sel, states, q_analytic_fn, action_dim):
    q_true = q_analytic_fn(states, actions_sel)
    modes = mode_assignment(actions_sel, states, action_dim)
    hist = np.bincount(modes, minlength=4).astype(np.float64)
    hist /= hist.sum()
    return dict(
        mean_q=float(np.mean(q_true)),
        preferred_frac=float(np.mean(modes == 0)),
        mode_entropy=hist_entropy(hist),
        mode_hist="|".join(f"{v:.3f}" for v in hist),
    )


# ---------------------------------------------------------------------------
# C1: 2D, num_samples == 1
# ---------------------------------------------------------------------------


def part_c1(train2d, test2d):
    print("[C1] 2D action_dim=2: num_samples=1 的退化验证 ...", flush=True)
    teacher, tparams, _ = train_flow(
        train2d["observations"], train2d["actions"],
        inject_noise=True, seed=SEED, steps=2000, batch_size=256, lr=3e-4, hidden=(128, 128),
    )
    student, sparams, _ = train_onestep_distill(
        teacher.apply, tparams, train2d["observations"], train2d["actions"],
        seed=SEED + 1, steps=2000, batch_size=256, lr=3e-4, flow_steps=FLOW_STEPS,
    )
    sampler = make_flow_sampler(student.apply, sparams)
    idx = balanced_states(test2d, 256)
    states = test2d["observations"][idx]
    S = len(states)
    rng = np.random.default_rng(7)
    noises = jnp.asarray(rng.normal(size=(1, S, 2)).astype(np.float32))
    actions = onestep_actions(sampler, states, np.asarray(noises))
    q = analytic_q_2d(states, actions[0])

    # 用任意 Q 检验：K=1 时 argmax/env 选择恒等
    idxs = select_methods(q[None, :], beta=10.0)
    identity = {m: float(np.mean(idxs[m] == 0)) for m in idxs}
    q_change = {}
    for m in idxs:
        sel = gather(actions, idxs[m])
        q_change[m] = float(np.mean(np.abs(analytic_q_2d(states, sel) - q)))

    rows = [
        dict(part="C1_2d", action_dim=2, num_samples=1, method=m,
             identity_fraction=identity[m], mean_abs_q_change=q_change[m])
        for m in idxs
    ]

    # K 覆盖实验：若强行把 K 调大（不是官方公式），机制是否有效？
    K_override = [1, 2, 3, 5, 10]
    rng = np.random.default_rng(8)
    for K in K_override:
        noises = jnp.asarray(rng.normal(size=(K, S, 2)).astype(np.float32))
        acts = onestep_actions(sampler, states, np.asarray(noises))
        q = analytic_q_2d(states, acts)  # (K,S)
        for m in ("first", "random", "argmax_perstate"):
            idxs = select_methods(q, beta=10.0, seed=K)
            sel = gather(acts, idxs[m])
            summ = summarize_selection(sel, states, analytic_q_2d, 2)
            rows.append(dict(part="C1_2d_K_override", action_dim=2, num_samples=K, method=m,
                             identity_fraction=np.nan, mean_abs_q_change=np.nan,
                             mean_q=summ["mean_q"], preferred_frac=summ["preferred_frac"],
                             mode_entropy=summ["mode_entropy"]))
    return rows, sampler, states, actions


# ---------------------------------------------------------------------------
# C2 + C3: 6D
# ---------------------------------------------------------------------------


def build_critic(train6d, seed=SEED + 5):
    rng = np.random.default_rng(seed)
    n = len(train6d["observations"])
    idx = rng.choice(n, size=4096, replace=False)
    s = train6d["observations"][idx]
    a_pref = s @ W6 + OFFSETS_6D[0]
    # 70% 覆盖 preferred 附近的动作空间，30% 用离线数据动作，保证 critic 覆盖
    mask = rng.random(len(s)) < 0.7
    a = np.where(
        mask[:, None],
        a_pref + 1.0 * rng.normal(size=(len(s), 6)).astype(np.float32),
        train6d["actions"][idx],
    ).astype(np.float32)
    q = analytic_q_6d(s, a).astype(np.float32)
    critic, cparams, losses = train_critic_regression(
        s, a, q, seed=seed, steps=3000, batch_size=256, lr=1e-3, hidden=(128, 128)
    )
    # 测试集 fit
    idx_t = rng.choice(len(train6d["observations"]), size=2048, replace=False)
    st = train6d["observations"][idx_t]
    at = np.where(
        (rng.random(len(st)) < 0.5)[:, None],
        (st @ W6 + OFFSETS_6D[0]) + 1.0 * rng.normal(size=(len(st), 6)).astype(np.float32),
        train6d["actions"][idx_t],
    ).astype(np.float32)
    qt = analytic_q_6d(st, at)
    pred = np.asarray(critic.apply(cparams, jnp.asarray(st), actions=jnp.asarray(at))).mean(axis=0)
    mae = float(np.mean(np.abs(pred - qt)))
    r2 = float(1 - np.mean((pred - qt) ** 2) / np.var(qt))
    return critic, cparams, dict(mae=mae, r2=r2, q_min=float(qt.min()), q_max=float(qt.max()),
                                 target_std=float(qt.std()), final_loss=losses[-1])


def part_c2c3(train6d, test6d):
    print("[C2] 6D action_dim=6: 训练 BC flow / one-step / critic ...", flush=True)
    teacher, tparams, _ = train_flow(
        train6d["observations"], train6d["actions"],
        inject_noise=True, seed=SEED + 10, steps=2500, batch_size=256, lr=3e-4, hidden=(128, 128),
    )
    student, sparams, _ = train_onestep_distill(
        teacher.apply, tparams, train6d["observations"], train6d["actions"],
        seed=SEED + 11, steps=2500, batch_size=256, lr=3e-4, flow_steps=FLOW_STEPS,
    )
    critic, cparams, critic_meta = build_critic(train6d)
    print(f"  critic MAE={critic_meta['mae']:.4f} R2={critic_meta['r2']:.4f}", flush=True)

    sampler = make_flow_sampler(student.apply, sparams)
    teacher_sampler = make_euler_sampler(teacher.apply, tparams, FLOW_STEPS)
    idx = balanced_states(test6d, 256)
    states = test6d["observations"][idx]
    real = test6d["actions"][idx]
    S = len(states)
    K_formula = NUM_SAMPLES_FORMULA(6)

    rows = []
    # 官方公式 K=3 的主结果
    rng = np.random.default_rng(11)
    noises = rng.normal(size=(K_formula, S, 6)).astype(np.float32)
    actions = onestep_actions(sampler, states, noises)
    q = critic_q(critic, cparams, states, actions)
    idxs = select_methods(q, beta=10.0)
    for m, ix in idxs.items():
        sel = gather(actions, ix)
        summ = summarize_selection(sel, states, analytic_q_6d, 6)
        summ["method"] = m
        summ["K"] = K_formula
        summ["mean_q_first"] = float(np.mean(analytic_q_6d(states, actions[0])))
        summ["q_improve_vs_first"] = summ["mean_q"] - summ["mean_q_first"]
        rows.append(dict(part="C2_6d", action_dim=6, K=K_formula, method=m,
                         mean_q=summ["mean_q"], q_improve=summ["q_improve_vs_first"],
                         preferred_frac=summ["preferred_frac"], mode_entropy=summ["mode_entropy"],
                         mode_hist=summ["mode_hist"]))

    # 教师多步 vs one-step 的分布差距（6D）
    t_actions = np.asarray(teacher_sampler(
        jnp.asarray(np.repeat(states[None], K_formula, 0).reshape(-1, 2)),
        jnp.asarray(noises.reshape(-1, 6)),
    )).reshape(K_formula, S, 6)
    student_vs_teacher_emd = energy_distance(actions[0], t_actions[0])
    student_vs_data_emd = energy_distance(actions[0], real)

    # K 扫描
    K_scan = [1, 2, 3, 5, 10]
    rng = np.random.default_rng(12)
    for K in K_scan:
        noises = rng.normal(size=(K, S, 6)).astype(np.float32)
        acts = onestep_actions(sampler, states, noises)
        q = critic_q(critic, cparams, states, acts)
        for m in ("first", "random", "argmax_perstate", "env_beta"):
            ix = select_methods(q, beta=10.0, seed=K)[m]
            sel = gather(acts, ix)
            summ = summarize_selection(sel, states, analytic_q_6d, 6)
            rows.append(dict(part="C2_6d_K_scan", action_dim=6, K=K, method=m,
                             mean_q=summ["mean_q"],
                             q_improve=summ["mean_q"] - float(np.mean(analytic_q_6d(states, acts[0]))),
                             preferred_frac=summ["preferred_frac"],
                             mode_entropy=summ["mode_entropy"], mode_hist=summ["mode_hist"]))

    # C3a: beta 扫描（env 分支，K=3）
    beta_rows = []
    rng = np.random.default_rng(13)
    noises = rng.normal(size=(K_formula, S, 6)).astype(np.float32)
    acts = onestep_actions(sampler, states, noises)
    q = critic_q(critic, cparams, states, acts)
    for beta in [0.1, 0.5, 2.0, 10.0, 50.0]:
        ix = select_methods(q, beta=beta, seed=5)["env_beta"]
        sel = gather(acts, ix)
        summ = summarize_selection(sel, states, analytic_q_6d, 6)
        beta_rows.append(dict(part="C3_beta_sweep", beta=beta, mean_q=summ["mean_q"],
                              preferred_frac=summ["preferred_frac"],
                              mode_entropy=summ["mode_entropy"], mode_hist=summ["mode_hist"]))
    # 确定性 argmax 作为 beta 很大（近似 inf）的参照
    ix = np.argmax(q, axis=0)
    sel = gather(acts, ix)
    summ = summarize_selection(sel, states, analytic_q_6d, 6)
    beta_rows.append(dict(part="C3_beta_sweep", beta=1e6, mean_q=summ["mean_q"],
                          preferred_frac=summ["preferred_frac"],
                          mode_entropy=summ["mode_entropy"], mode_hist=summ["mode_hist"]))

    # C3b: update_entropy 的 beta 轨迹（kNN 熵替代 GMM）
    print("[C3] 复现 update_entropy 的 beta 自适应轨迹 ...", flush=True)
    n_repeats, n_states = 200, 96
    idx_e = balanced_states(test6d, n_states // 2)
    st_e = test6d["observations"][idx_e]
    Se = len(st_e)
    rng = np.random.default_rng(14)
    all_actions = []  # n_repeats 次 sample_actions(env mode) 的候选
    noises = rng.normal(size=(n_repeats, K_formula, Se, 6)).astype(np.float32)
    for r in range(n_repeats):
        acts_r = onestep_actions(sampler, st_e, noises[r])
        q_r = critic_q(critic, cparams, st_e, acts_r)
        all_actions.append((acts_r, q_r))
    beta = 10.0
    beta_hist, entropy_hist = [beta], []
    sel_actions = None
    for it in range(30):
        sel_actions = np.zeros((n_repeats, Se, 6), np.float32)
        for r in range(n_repeats):
            acts_r, q_r = all_actions[r]
            ix = select_methods(q_r, beta=beta, seed=1000 + r)["env_beta"]
            sel_actions[r] = gather(acts_r, ix)
        Hs = [knn_entropy(sel_actions[:, i, :], k=3) for i in range(Se)]
        H = float(np.mean(Hs))
        entropy_hist.append(H)
        beta = max(0.0, beta - 0.1 * (-6.0 - H))
        beta_hist.append(beta)
        if it % 5 == 0:
            print(f"  iter {it:2d} entropy={H:.3f} beta={beta:.2f}", flush=True)
    final_q = float(np.mean(analytic_q_6d(st_e, sel_actions.mean(axis=0))))
    beta_rows.append(dict(part="C3_beta_adaptive_final", beta=beta, mean_q=final_q,
                          preferred_frac=np.nan, mode_entropy=entropy_hist[-1], mode_hist=""))

    meta = dict(
        num_samples_formula=dict(action_dim_2=NUM_SAMPLES_FORMULA(2),
                                 action_dim_6=NUM_SAMPLES_FORMULA(6)),
        critic=critic_meta,
        student_vs_teacher_emd_6d=student_vs_teacher_emd,
        student_vs_data_emd_6d=student_vs_data_emd,
        c2_main=rows[:len(idxs)],
        beta_adaptive=dict(beta_hist=beta_hist, entropy_hist=entropy_hist,
                           target_entropy=-6.0,
                           note="repo 用 sklearn GMM(3) 估熵；此处用 kNN(KL) 估计"),
    )
    return rows, beta_rows, meta


def main():
    print("[data] 合成 2D / 6D 玩具数据 ...", flush=True)
    train2d = make_dataset(30000, seed=10, action_dim=2)
    test2d = make_dataset(10000, seed=11, action_dim=2)
    train6d = make_dataset(30000, seed=20, action_dim=6)
    test6d = make_dataset(10000, seed=21, action_dim=6)
    save_dataset(OUT / "toy_data_6d.npz", train6d, test6d)

    c1_rows, _, _, _ = part_c1(train2d, test2d)
    c2_rows, c3_rows, meta = part_c2c3(train6d, test6d)

    rows = c1_rows + c2_rows + c3_rows
    fields = ["part", "action_dim", "K", "num_samples", "method", "mean_q",
              "q_improve", "preferred_frac", "mode_entropy", "identity_fraction",
              "mean_abs_q_change", "beta", "mode_hist"]
    norm = [{k: r.get(k) for k in fields} for r in rows]
    save_csv(OUT / "exp_c_qguided.csv", norm, fields)
    save_json(OUT / "exp_c_summary.json", dict(rows=norm, meta=meta))

    print("[summary] C1: 2D num_samples=%d" % meta["num_samples_formula"]["action_dim_2"], flush=True)
    for r in c1_rows[:5]:
        print(f"  {r['part']} method={r['method']} identity={r['identity_fraction']:.3f} "
              f"|dQ|={r['mean_abs_q_change']:.5f}", flush=True)
    print("[summary] C2: 6D num_samples=%d" % meta["num_samples_formula"]["action_dim_6"], flush=True)
    for r in c2_rows:
        print(f"  method={r['method']:20s} mean_q={r['mean_q']:.3f} "
              f"q_improve={r['q_improve']:+.3f} pref_frac={r['preferred_frac']:.3f} "
              f"mode_H={r['mode_entropy']:.3f}", flush=True)
    print(f"  critic R2={meta['critic']['r2']:.3f}  "
          f"student-vs-teacher EMD={meta['student_vs_teacher_emd_6d']:.4f}", flush=True)
    print(f"[summary] C3: beta {meta['beta_adaptive']['beta_hist'][0]:.2f} -> "
          f"{meta['beta_adaptive']['beta_hist'][-1]:.2f}; entropy "
          f"{meta['beta_adaptive']['entropy_hist'][0]:.3f} -> "
          f"{meta['beta_adaptive']['entropy_hist'][-1]:.3f}", flush=True)

    # ---------------- 图 ----------------
    K_override = [1, 2, 3, 5, 10]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    for m, color in [("first", "gray"), ("random", "tab:green"), ("argmax_perstate", "tab:blue")]:
        ys = []
        for K in K_override:
            rr = [r for r in c1_rows if r["part"] == "C1_2d_K_override"
                  and r["num_samples"] == K and r["method"] == m]
            ys.append(rr[0]["mean_q"] if rr else np.nan)
        ax.plot(K_override, ys, "-o", color=color, label=m)
    ax.set_xlabel("K (candidates per state, NOT the official formula)")
    ax.set_ylabel("toy Q of selected action")
    ax.set_title("C1: action_dim=2 -> official K=1 (dashed)")
    ax.axvline(1, ls="--", color="k", alpha=0.5)
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    for m, color in [("first", "gray"), ("random", "tab:green"),
                     ("argmax_perstate", "tab:blue"), ("env_beta", "tab:orange")]:
        ys = []
        for K in K_override:
            rr = [r for r in c2_rows if r["part"] == "C2_6d_K_scan"
                  and r["K"] == K and r["method"] == m]
            ys.append(rr[0]["mean_q"] if rr else np.nan)
        ax.plot(K_override, ys, "-o", color=color, label=m)
    ax.axvline(3, ls="--", color="k", alpha=0.5, label="official K=3")
    ax.set_xlabel("K candidates")
    ax.set_ylabel("toy Q of selected action")
    ax.set_title("C2: 6D Q-guided selection")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    bh = meta["beta_adaptive"]["beta_hist"]
    eh = meta["beta_adaptive"]["entropy_hist"]
    ax.plot(range(len(bh)), bh, "-o", ms=3, color="tab:red", label="beta")
    ax.set_xlabel("update_entropy iteration")
    ax.set_ylabel("beta", color="tab:red")
    ax2 = ax.twinx()
    ax2.plot(range(len(eh)), eh, "-s", ms=3, color="tab:blue", label="entropy")
    ax2.axhline(-6.0, ls="--", color="tab:blue", alpha=0.5)
    ax2.set_ylabel("estimated entropy (nats)", color="tab:blue")
    ax.set_title("C3: entropy-adaptive beta")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_c_qguided.png", dpi=150)

    # beta sweep 图
    fig, ax = plt.subplots(figsize=(6.5, 4))
    betas = [r["beta"] for r in c3_rows if r["part"] == "C3_beta_sweep"]
    qs = [r["mean_q"] for r in c3_rows if r["part"] == "C3_beta_sweep"]
    hs = [r["mode_entropy"] for r in c3_rows if r["part"] == "C3_beta_sweep"]
    xs = np.arange(len(betas))
    labels = ["0.1", "0.5", "2", "10", "50", "argmax"]
    ax.plot(xs, qs, "-o", color="tab:red", label="mean toy Q")
    ax.set_xticks(xs, labels)
    ax.set_xlabel("beta (env-mode categorical temperature)")
    ax.set_ylabel("mean toy Q", color="tab:red")
    ax2 = ax.twinx()
    ax2.plot(xs, hs, "-s", color="tab:blue", label="mode entropy")
    ax2.set_ylabel("mode entropy", color="tab:blue")
    ax.set_title("C3: beta controls Q vs diversity tradeoff (6D, K=3)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_c_beta_sweep.png", dpi=150)
    print("[done] 实验 C 完成。", flush=True)


if __name__ == "__main__":
    main()
