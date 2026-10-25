"""Analyze the paper-formula ablations: noise schedule (A2/A3/A4) and beta update (B1/B2/B3).

Reads runs from E:\\FINO\\gpu_exp\\ablation and the official FINO full-budget baseline,
writes summary tables + figures into E:\\FINO\\results.
"""
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
import numpy as np
import pandas as pd

ABL = r'E:\FINO\gpu_exp\ablation'
BASE = r'E:\FINO\gpu_exp\FINO\antmaze-large-navigate-singletask-v0\0'
OUT = r'E:\FINO\results'
ENV = 'antmaze-large-navigate-singletask-v0'
os.makedirs(OUT, exist_ok=True)

A_RUNS = [
    ('官方指数 schedule（基线）', 'official', BASE),
    ('论文二次型 Eq.(7)', 'A2_quad_sched', os.path.join(ABL, 'A2_quad_sched', 'FINO', ENV, '0')),
    ('无噪声 η=0', 'A3_none_noise', os.path.join(ABL, 'A3_none_noise', 'FINO', ENV, '0')),
    ('指数 schedule η=0.3', 'A4_eta03', os.path.join(ABL, 'A4_eta03', 'FINO', ENV, '0')),
]
B_RUNS = [
    ('官方 β（恢复基线）', 'B1_resume_official', os.path.join(ABL, 'B1_resume_official', 'FINO', ENV, '0')),
    ('论文 Eq.(9) 符号', 'B2_paper_beta', os.path.join(ABL, 'B2_paper_beta', 'FINO', ENV, '0')),
    ('固定 β=10', 'B3_fixed_beta', os.path.join(ABL, 'B3_fixed_beta', 'FINO', ENV, '0')),
]


def load(run_dir):
    ev = os.path.join(run_dir, 'eval.csv')
    tr = os.path.join(run_dir, 'train.csv')
    bt = os.path.join(run_dir, 'beta.csv')
    e = pd.read_csv(ev) if os.path.exists(ev) else None
    t = pd.read_csv(tr) if os.path.exists(tr) else None
    b = pd.read_csv(bt) if os.path.exists(bt) else None
    return t, e, b


def summarize(runs, group):
    rows = []
    for label, name, d in runs:
        t, e, b = load(d)
        if e is None or len(e) == 0:
            rows.append(dict(group=group, variant=name, label=label, status='missing'))
            continue
        rows.append(dict(
            group=group, variant=name, label=label,
            eval_points=len(e),
            success_final=round(float(e['evaluation/success'].iloc[-1]), 3),
            success_peak=round(float(e['evaluation/success'].max()), 3),
            success_last5=round(float(e['evaluation/success'].tail(5).mean()), 3),
            return_final=round(float(e['evaluation/episode.return'].iloc[-1]), 1),
            minutes=round(float(t['time/total_time'].iloc[-1]) / 60, 1) if t is not None else None,
            beta_last=round(float(b['beta'].iloc[-1]), 2) if b is not None and len(b) else None,
            entropy_last=round(float(b['entropy'].iloc[-1]), 2) if b is not None and len(b) else None,
        ))
    return rows


def plot_schedule_sigma():
    eta = 0.1
    t = np.linspace(0, 1, 200)
    exp = eta * np.exp(10 * (t - 1))
    quad = np.sqrt(np.maximum(0, (eta ** 2 - 2 * eta) * t ** 2 + 2 * eta * t))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    ax = axes[0]
    ax.plot(t, exp, label='官方实现：η·e^{10(t−1)}', color='tab:orange')
    ax.plot(t, quad, label='论文 Eq.(7)：√((η²−2η)t²+2ηt)', color='tab:blue')
    ax.set_xlabel('t（0=噪声端，1=数据端）')
    ax.set_ylabel('注入噪声标准差 σ_t')
    ax.set_title('η = 0.1：两种噪声 schedule')
    ax.grid(alpha=.3)
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.plot(t, quad / (exp + 1e-12), color='tab:red')
    ax.set_yscale('log')
    ax.set_xlabel('t')
    ax.set_ylabel('σ_quad / σ_exp（对数）')
    ax.set_title('论文公式 / 实现公式 的噪声强度比')
    ax.grid(alpha=.3)
    fig.tight_layout()
    p = os.path.join(OUT, 'fig_ablation_schedule_sigma.png')
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print('wrote', p)


def plot_success(runs, fname, title):
    fig, ax = plt.subplots(figsize=(11, 4.2))
    for label, name, d in runs:
        t, e, _ = load(d)
        if e is None:
            continue
        ax.plot(e['step'], e['evaluation/success'], marker='o', ms=2.5, label=label)
    ax.axvline(1_000_000, color='gray', ls='--', lw=1, label='离线→在线 (1M)')
    ax.set_xlabel('update step')
    ax.set_ylabel('evaluation success')
    ax.set_ylim(-0.02, 1.03)
    ax.set_title(title)
    ax.grid(alpha=.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    p = os.path.join(OUT, fname)
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print('wrote', p)


def plot_beta(runs):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.0))
    ax = axes[0]
    for label, name, d in runs:
        _, e, b = load(d)
        if e is not None:
            ax.plot(e['step'], e['evaluation/success'], marker='o', ms=2.5, label=label)
    ax.set_xlabel('update step（在线阶段从 1M 起算）')
    ax.set_ylabel('evaluation success')
    ax.set_ylim(-0.02, 1.03)
    ax.set_title('β 消融：成功率')
    ax.grid(alpha=.3)
    ax.legend(fontsize=8)
    ax = axes[1]
    for label, name, d in runs:
        _, _, b = load(d)
        if b is not None and len(b):
            ax.plot(b['step'], b['beta'], marker='o', ms=2.5, label=label)
    ax.set_xlabel('step')
    ax.set_ylabel('β')
    ax.set_title('β 轨迹')
    ax.grid(alpha=.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(OUT, 'fig_ablation_beta.png')
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print('wrote', p)


def main():
    rows = summarize(A_RUNS, 'A_noise_schedule') + summarize(B_RUNS, 'B_beta')
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, 'ablation_summary.csv'), index=False)
    with open(os.path.join(OUT, 'ablation_summary.md'), 'w', encoding='utf-8') as f:
        f.write('# 论文公式逻辑消融（antmaze-large，GPU 全预算）\n\n')
        cols = list(df.columns)
        f.write('| ' + ' | '.join(cols) + ' |\n|' + '|'.join(['---'] * len(cols)) + '|\n')
        for _, r in df.iterrows():
            f.write('| ' + ' | '.join(str(r[c]) for c in cols) + ' |\n')
    print(df.to_string(index=False))

    plot_schedule_sigma()
    plot_success(A_RUNS, 'fig_ablation_schedule_success.png',
                 'A: noise-schedule ablation (antmaze-large, full budget)')
    plot_beta(B_RUNS)


if __name__ == '__main__':
    main()
