"""实验 A 的附带消融：指数 schedule 下 noise_scale = eta 的强度扫描。

官方 eta=0.1（agents/fino.py L357）。本脚本扫描 eta ∈ {0, 0.1, 0.3, 1.0}，
观察"注入噪声强度 -> 流场平滑性 / 采样分布质量"的权衡，检验 eta=0.1 在
当前 2D 玩具数据尺度下是否足够强。
输出 CSV/JSON/PNG。
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from toy_common import (
    OUT,
    energy_distance,
    hist_entropy,
    make_dataset,
    mean_std,
    mode_histogram,
    save_csv,
    save_json,
    train_flow,
)
from exp_a_noise_injection import eval_sampling, eval_vector_field

SEEDS = [0, 1, 2]
ETAS = [0.0, 0.1, 0.3, 1.0]
STEPS = 2500


def main():
    train = make_dataset(30000, seed=10, action_dim=2)
    test = make_dataset(10000, seed=11, action_dim=2)
    rows = []
    for eta in ETAS:
        for seed in SEEDS:
            print(f"[train] eta={eta} seed={seed}", flush=True)
            module, params, losses = train_flow(
                train["observations"], train["actions"],
                inject_noise=eta > 0.0, seed=seed, steps=STEPS,
                batch_size=256, lr=3e-4, hidden=(128, 128),
                noise_scale=eta, schedule="exp",
            )
            vf = eval_vector_field(module, params, test)
            vf_by_t = {round(r["t"], 2): r["vf_mse"] for r in vf}
            sm = eval_sampling(module, params, test)
            rows.append(dict(
                eta=eta, seed=seed, final_loss=losses[-1],
                emd_mean=sm["emd_mean"], mmd_mean=sm["mmd_mean"],
                mode_entropy=sm["mode_entropy"], mode_kl=sm["mode_kl_to_real"],
                mode_max_frac=sm["mode_max_frac"],
                lipschitz_t01=sm["lipschitz_t01"], lipschitz_t09=sm["lipschitz_t09"],
                lipschitz_t099=sm["lipschitz_t099"],
                vf_mse_t09=vf_by_t[0.9], vf_mse_t099=vf_by_t[0.99],
                vf_mse_t1=vf_by_t[1.0],
            ))
            print(f"  emd={sm['emd_mean']:.4f} lip(.9)={sm['lipschitz_t09']:.3f} "
                  f"lip(.99)={sm['lipschitz_t099']:.3f} vf(.99)={vf_by_t[0.99]:.4f}", flush=True)

    save_csv(OUT / "exp_a2_eta_sweep.csv", rows, list(rows[0].keys()))
    summary = {}
    for eta in ETAS:
        sub = [r for r in rows if r["eta"] == eta]
        summary[str(eta)] = {
            k: dict(zip(("mean", "std"), mean_std([r[k] for r in sub])))
            for k in ("emd_mean", "mmd_mean", "mode_entropy", "mode_kl", "mode_max_frac",
                      "lipschitz_t01", "lipschitz_t09", "lipschitz_t099",
                      "vf_mse_t09", "vf_mse_t099", "vf_mse_t1")
        }
    summary["config"] = dict(seeds=SEEDS, etas=ETAS, steps=STEPS, schedule="exp",
                             note="eta=0 即不注入噪声；其余按官方 exp schedule")
    save_json(OUT / "exp_a2_eta_sweep.json", summary)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    xs = np.arange(len(ETAS))
    labels = [str(e) for e in ETAS]
    for ax, keys, title, ylabel in [
        (axes[0], ["emd_mean", "mmd_mean"], "sample quality vs eta", "distance (lower better)"),
        (axes[1], ["lipschitz_t09", "lipschitz_t099"], "flow smoothness vs eta", "local Lipschitz proxy"),
        (axes[2], ["vf_mse_t09", "vf_mse_t099", "vf_mse_t1"],
         "data-end vector-field MSE vs eta", "MSE"),
    ]:
        for key in keys:
            m = [summary[str(e)][key]["mean"] for e in ETAS]
            sd = [summary[str(e)][key]["std"] for e in ETAS]
            ax.errorbar(xs, m, yerr=sd, fmt="-o", capsize=3, label=key)
        ax.set_xticks(xs, labels)
        ax.set_xlabel("eta (official = 0.1)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_a2_eta_sweep.png", dpi=150)
    print("[done] eta 扫描完成。", flush=True)


if __name__ == "__main__":
    main()
