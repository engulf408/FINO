# -*- coding: utf-8 -*-
"""Summarize round-2 ablations (schedule scale, seeds, cross-task, Q-guided K) and draw figures.

Run inside WSL with /opt/fino-venv/bin/python (has matplotlib + pandas).
Figures use English labels; the markdown summary is Chinese.
"""
import json
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

BASE = '/mnt/e/FINO'
OUT = os.path.join(BASE, 'results')
os.makedirs(OUT, exist_ok=True)

ENV_A = 'antmaze-large-navigate-singletask-v0'
ENV_H = 'humanoidmaze-medium-navigate-singletask-v0'
AB1 = os.path.join(BASE, 'gpu_exp', 'ablation')
AB2 = os.path.join(BASE, 'gpu_exp', 'ablation2')
MAIN = os.path.join(BASE, 'gpu_exp', 'FINO')

SCHEDULE = [
    ('official exp (eta=0.1)', os.path.join(MAIN, ENV_A, '0')),
    ('no noise (eta=0)', os.path.join(AB1, 'A3_none_noise', 'FINO', ENV_A, '0')),
    ('exp eta=0.3', os.path.join(AB1, 'A4_eta03', 'FINO', ENV_A, '0')),
    ('Eq.(7) unscaled', os.path.join(AB1, 'A2_quad_sched', 'FINO', ENV_A, '0')),
    ('Eq.(7) peak-matched', os.path.join(AB2, 'A5_quad_peak', 'FINO', ENV_A, '0')),
    ('Eq.(7) variance-read', os.path.join(AB2, 'A6_quad_var', 'FINO', ENV_A, '0')),
]
SEEDS = [
    ('official seed0', os.path.join(MAIN, ENV_A, '0')),
    ('official seed1', os.path.join(AB2, 'S1_official_seed1', 'FINO', ENV_A, '1')),
    ('Eq.(7) seed0', os.path.join(AB1, 'A2_quad_sched', 'FINO', ENV_A, '0')),
    ('Eq.(7) seed1', os.path.join(AB2, 'S1_quad_seed1', 'FINO', ENV_A, '1')),
]
CROSS = [
    ('humanoid official', os.path.join(MAIN, ENV_H, '0')),
    ('humanoid Eq.(7)', os.path.join(AB2, 'HUM_quad', 'FINO', ENV_H, '0')),
]
KGROUP = [
    ('K=1', os.path.join(AB2, 'K1_guided', 'FINO', ENV_A, '0')),
    ('K=4 (official)', os.path.join(AB1, 'B1_resume_official', 'FINO', ENV_A, '0')),
    ('K=10', os.path.join(AB2, 'K10_guided', 'FINO', ENV_A, '0')),
]


def load(d):
    ev = os.path.join(d, 'eval.csv')
    tr = os.path.join(d, 'train.csv')
    bt = os.path.join(d, 'beta.csv')
    e = pd.read_csv(ev) if os.path.exists(ev) else None
    t = pd.read_csv(tr) if os.path.exists(tr) else None
    b = pd.read_csv(bt) if os.path.exists(bt) else None
    return e, t, b


def summarize(group, runs):
    rows = []
    for label, d in runs:
        e, t, b = load(d)
        if e is None or len(e) == 0:
            rows.append(dict(group=group, variant=label, status='missing'))
            continue
        rows.append(dict(
            group=group, variant=label,
            final=round(float(e['evaluation/success'].iloc[-1]), 3),
            peak=round(float(e['evaluation/success'].max()), 3),
            last5=round(float(e['evaluation/success'].tail(5).mean()), 3),
            evals=len(e),
            minutes=round(float(t['time/total_time'].iloc[-1]) / 60, 1) if t is not None else None,
            beta_last=round(float(b['beta'].iloc[-1]), 2) if b is not None and len(b) else None,
        ))
    return rows


def curves(runs, fname, title, vline=1_000_000):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={'width_ratios': [2.1, 1]})
    ax = axes[0]
    for label, d in runs:
        e, _, _ = load(d)
        if e is not None:
            ax.plot(e['step'], e['evaluation/success'], marker='o', ms=2.3, label=label)
    if vline:
        ax.axvline(vline, color='gray', ls='--', lw=1, label='offline→online')
    ax.set_xlabel('update step')
    ax.set_ylabel('evaluation success')
    ax.set_ylim(-0.02, 1.03)
    ax.set_title(title)
    ax.grid(alpha=.3)
    ax.legend(fontsize=7.2, ncol=2)
    ax = axes[1]
    labels, vals = [], []
    for label, d in runs:
        e, _, _ = load(d)
        if e is not None:
            labels.append(label)
            vals.append(float(e['evaluation/success'].iloc[-1]))
    bars = ax.barh(range(len(vals)), vals, color='tab:blue')
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_xlabel('final success')
    ax.set_title('final success')
    ax.grid(axis='x', alpha=.3)
    fig.tight_layout()
    p = os.path.join(OUT, fname)
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print('wrote', p)


def main():
    rows = (summarize('noise_schedule', SCHEDULE) + summarize('seeds', SEEDS)
            + summarize('cross_task', CROSS) + summarize('qguided_K', KGROUP))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, 'ablation2_summary.csv'), index=False)
    with open(os.path.join(OUT, 'ablation2_summary.md'), 'w', encoding='utf-8') as f:
        f.write('# 补充消融结果（Round 2，2026-10-26）\n\n')
        cols = list(df.columns)
        f.write('| ' + ' | '.join(cols) + ' |\n|' + '|'.join(['---'] * len(cols)) + '|\n')
        for _, r in df.iterrows():
            f.write('| ' + ' | '.join(str(r[c]) for c in cols) + ' |\n')
    print(df.to_string(index=False))

    curves(SCHEDULE, 'fig_ablation_schedule_scale.png',
           'Noise schedule scale: unscaled Eq.(7) vs normalized variants')
    curves(SEEDS, 'fig_ablation_seeds.png', 'Seed robustness: official vs Eq.(7)', vline=1_000_000)
    curves(CROSS, 'fig_ablation_cross_task.png', 'Cross-task: humanoidmaze official vs Eq.(7)', vline=1_000_000)
    curves(KGROUP, 'fig_ablation_k.png', 'Q-guided candidates K (online 500k from 1M ckpt)', vline=0)


if __name__ == '__main__':
    main()
