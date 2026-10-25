"""Summarize the three full-budget GPU tasks (antmaze-large, cube-double-play, humanoidmaze-medium)
and produce PPT/report-ready figures.

Run after the extended runs finish:
    python make_extended_results.py
"""
import json
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = r'E:\FINO\gpu_exp'
OUT = r'E:\FINO\results'
os.makedirs(OUT, exist_ok=True)

TASKS = [
    ('antmaze-large', 'antmaze-large-navigate-singletask-v0'),
    ('cube-double-play', 'cube-double-play-singletask-v0'),
    ('humanoidmaze-medium', 'humanoidmaze-medium-navigate-singletask-v0'),
]
AGENT = 'FINO'


def load(agent, env):
    d = os.path.join(ROOT, agent, env, '0')
    t = os.path.join(d, 'train.csv')
    e = os.path.join(d, 'eval.csv')
    if not os.path.exists(e):
        return None, None
    return (pd.read_csv(t) if os.path.exists(t) else None), pd.read_csv(e)


def main():
    rows = []
    curves = {}
    for short, env in TASKS:
        train, ev = load(AGENT, env)
        if ev is None:
            print('missing', env)
            continue
        curves[short] = ev
        total = int(train['step'].max()) if train is not None else None
        rows.append(dict(
            task=short, env=env, agent=AGENT,
            steps=total,
            eval_points=len(ev),
            success_peak=round(float(ev['evaluation/success'].max()), 3),
            success_final=round(float(ev['evaluation/success'].iloc[-1]), 3),
            success_last5_mean=round(float(ev['evaluation/success'].tail(5).mean()), 3),
            return_final=round(float(ev['evaluation/episode.return'].iloc[-1]), 1),
            minutes=round(float(train['time/total_time'].iloc[-1]) / 60, 1) if train is not None else None,
        ))

    # also FQL antmaze for reference
    train_f, ev_f = load('FQL', 'antmaze-large-navigate-singletask-v0')
    if ev_f is not None:
        rows.append(dict(task='antmaze-large (FQL)', env='antmaze-large-navigate-singletask-v0', agent='FQL',
                         steps=int(train_f['step'].max()), eval_points=len(ev_f),
                         success_peak=round(float(ev_f['evaluation/success'].max()), 3),
                         success_final=round(float(ev_f['evaluation/success'].iloc[-1]), 3),
                         success_last5_mean=round(float(ev_f['evaluation/success'].tail(5).mean()), 3),
                         return_final=round(float(ev_f['evaluation/episode.return'].iloc[-1]), 1),
                         minutes=round(float(train_f['time/total_time'].iloc[-1]) / 60, 1)))

    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'extended_summary.csv'), index=False)
    with open(os.path.join(OUT, 'extended_summary.md'), 'w', encoding='utf-8') as f:
        f.write('# 全预算 GPU 任务汇总（FINO：3 个 OGBench 任务 + FQL 参考）\n\n')
        cols = list(pd.DataFrame(rows).columns)
        f.write('| ' + ' | '.join(cols) + ' |\n|' + '|'.join(['---'] * len(cols)) + '|\n')
        for r in rows:
            f.write('| ' + ' | '.join(str(r[c]) for c in cols) + ' |\n')
    print(pd.DataFrame(rows).to_string(index=False))

    # ---- figure 1: success curves for 3 tasks + FQL antmaze ----
    fig, ax = plt.subplots(figsize=(13, 4.6))
    colors = {'antmaze-large': 'tab:orange', 'cube-double-play': 'tab:green',
              'humanoidmaze-medium': 'tab:purple'}
    for short, ev in curves.items():
        ax.plot(ev['step'], ev['evaluation/success'], marker='o', ms=2.5,
                label=f'FINO {short}', color=colors.get(short))
    if ev_f is not None:
        ax.plot(ev_f['step'], ev_f['evaluation/success'], marker='s', ms=2, ls='--',
                label='FQL antmaze-large', color='tab:blue')
    ax.axvline(1_000_000, color='gray', ls='--', lw=1, label='offline→online (1M)')
    ax.axvspan(1_000_000, 1_500_000, color='gray', alpha=0.07)
    ax.set_xlabel('update step')
    ax.set_ylabel('evaluation success rate')
    ax.set_ylim(-0.02, 1.05)
    ax.set_title('Full-budget GPU reproduction: FINO on 3 OGBench tasks (RTX 5070)', fontsize=12)
    ax.grid(alpha=.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    p1 = os.path.join(OUT, 'fig_gpu_extended_curves.png')
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    print('wrote', p1)

    # ---- figure 2: final success bars ----
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    labels = [f"{r['task']}\n({r['agent']})" for _, r in df.iterrows()]
    vals = [r['success_final'] for _, r in df.iterrows()]
    bars = ax.bar(labels, vals, color=['tab:orange', 'tab:green', 'tab:purple', 'tab:blue'][:len(vals)])
    for b, r in zip(bars, [r for _, r in df.iterrows()]):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015,
                f"{r['success_final']:.2f}\n(last5: {r['success_last5_mean']:.2f})",
                ha='center', fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('final success rate')
    ax.set_title('Full-budget results: final evaluation success', fontsize=12)
    ax.grid(axis='y', alpha=.3)
    fig.tight_layout()
    p2 = os.path.join(OUT, 'fig_gpu_task_summary.png')
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    print('wrote', p2)

    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
