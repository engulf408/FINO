"""Aggregate FINO reproduction runs into summary tables and figures.

Usage: python plot_results.py
Reads E:/FINO/exp/<AGENT>/<env>/<seed>/eval.csv and train.csv.
Writes E:/FINO/results/ figures + summary.md / summary.csv.
"""
import json
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EXP_ROOT = r'E:\FINO\exp'
OUT = r'E:\FINO\results'
os.makedirs(OUT, exist_ok=True)

RUNS = [
    ('FINO', 'pointmaze-medium-navigate-singletask-v0', 'pointmaze', 'pointmaze', 'PointMaze-medium navigate (smoke, 8k offline + 4k online)'),
    ('FINO', 'antmaze-large-navigate-singletask-v0', 'antmaze_fino', 'antmaze', 'AntMaze-large navigate (paper task, FINO, 30k offline + 10k online)'),
    ('FQL', 'antmaze-large-navigate-singletask-v0', 'antmaze_fql', 'antmaze', 'AntMaze-large navigate (paper task, FQL baseline, 30k offline + 10k online)'),
]


def load_run(agent, env):
    d = os.path.join(EXP_ROOT, agent, env, '0')
    train_p = os.path.join(d, 'train.csv')
    eval_p = os.path.join(d, 'eval.csv')
    if not os.path.exists(train_p):
        return None, None, d
    train = pd.read_csv(train_p)
    ev = pd.read_csv(eval_p) if os.path.exists(eval_p) else None
    return train, ev, d


def plot_training(tag, train, title):
    steps = train['step']
    has_distill = 'training/actor/distill_loss' in train.columns
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    ax = axes[0, 0]
    ax.plot(steps, train['training/actor/bc_flow_loss'], label='BC flow loss (train)')
    if 'validation/actor/bc_flow_loss' in train.columns:
        ax.plot(steps, train['validation/actor/bc_flow_loss'], label='BC flow loss (val)')
    ax.set_title('Flow-matching BC loss'); ax.set_xlabel('update step'); ax.legend(); ax.grid(alpha=.3)

    ax = axes[0, 1]
    if has_distill:
        ax.plot(steps, train['training/actor/distill_loss'], label='distill loss (train)')
        if 'validation/actor/distill_loss' in train.columns:
            ax.plot(steps, train['validation/actor/distill_loss'], label='distill loss (val)')
        ax.set_title('One-step distillation loss')
    else:
        ax.plot(steps, train['training/actor/q_loss'] if 'training/actor/q_loss' in train.columns else train['training/actor/actor_loss'], label='actor loss')
        ax.set_title('Actor loss (no distillation in FQL)')
    ax.set_xlabel('update step'); ax.legend(); ax.grid(alpha=.3)

    ax = axes[1, 0]
    ax.plot(steps, train['training/critic/critic_loss'], label='critic loss (train)')
    if 'validation/critic/critic_loss' in train.columns:
        ax.plot(steps, train['validation/critic/critic_loss'], label='critic loss (val)')
    ax.set_title('Critic TD loss'); ax.set_xlabel('update step'); ax.legend(); ax.grid(alpha=.3)

    ax = axes[1, 1]
    ax.plot(steps, train['training/actor/q'], label='actor Q (train)')
    if 'validation/actor/q' in train.columns:
        ax.plot(steps, train['validation/actor/q'], label='actor Q (val)')
    ax.set_title('Q value of policy actions'); ax.set_xlabel('update step'); ax.legend(); ax.grid(alpha=.3)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    path = os.path.join(OUT, f'fig_training_{tag}.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print('wrote', path)


def plot_eval(tag, ev, title):
    if ev is None or len(ev) == 0:
        return
    steps = ev['step']
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
    axes[0].plot(steps, ev['evaluation/success'], marker='o', ms=3)
    axes[0].set_title('Evaluation success rate'); axes[0].set_xlabel('update step')
    axes[0].set_ylim(-0.02, 1.02); axes[0].grid(alpha=.3)
    axes[1].plot(steps, ev['evaluation/episode.return'], marker='o', ms=3, color='tab:orange')
    axes[1].set_title('Evaluation episode return'); axes[1].set_xlabel('update step')
    axes[1].grid(alpha=.3)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    path = os.path.join(OUT, f'fig_eval_{tag}.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print('wrote', path)


def plot_comparison(evs):
    """FINO vs FQL on antmaze: success & return."""
    if len(evs) < 2:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
    for label, ev in evs.items():
        axes[0].plot(ev['step'], ev['evaluation/success'], marker='o', ms=3, label=label)
        axes[1].plot(ev['step'], ev['evaluation/episode.return'], marker='o', ms=3, label=label)
    axes[0].set_title('Evaluation success rate (antmaze-large)'); axes[0].set_xlabel('update step')
    axes[0].set_ylim(-0.02, 1.02); axes[0].legend(); axes[0].grid(alpha=.3)
    axes[1].set_title('Evaluation episode return (antmaze-large)'); axes[1].set_xlabel('update step')
    axes[1].legend(); axes[1].grid(alpha=.3)
    fig.tight_layout()
    path = os.path.join(OUT, 'fig_eval_antmaze_compare.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print('wrote', path)


def main():
    rows = []
    evs = {}
    for agent, env, tag, group, title in RUNS:
        train, ev, d = load_run(agent, env)
        if train is None:
            print('skip (no data):', agent, env)
            continue
        plot_training(f'{tag}_{agent.lower()}', train, f'{title} [{agent}]')
        plot_eval(f'{tag}_{agent.lower()}', ev, f'{title} [{agent}]')
        if group == 'antmaze' and ev is not None and len(ev) > 0:
            evs[agent] = ev
        flags = {}
        fp = os.path.join(d, 'flags.json')
        if os.path.exists(fp):
            with open(fp) as f:
                flags = json.load(f)
        total_t = float(train['time/total_time'].iloc[-1])
        last_step = float(train['step'].iloc[-1])
        rows.append(dict(
            agent=agent, env=env, seed=0,
            offline_steps=flags.get('offline_steps'), online_steps=flags.get('online_steps'),
            updates=len(train), total_seconds=round(total_t, 1),
            ms_per_update=round(1000 * total_t / last_step, 1),
            bc_flow_loss_first=round(float(train['training/actor/bc_flow_loss'].iloc[0]), 4),
            bc_flow_loss_last=round(float(train['training/actor/bc_flow_loss'].iloc[-1]), 4),
            critic_loss_last=round(float(train['training/critic/critic_loss'].iloc[-1]), 4),
            q_mean_last=round(float(train['training/actor/q'].iloc[-1]), 2),
            eval_success_max=(float(ev['evaluation/success'].max()) if ev is not None else None),
            eval_return_last=(float(ev['evaluation/episode.return'].iloc[-1]) if ev is not None else None),
        ))
    plot_comparison(evs)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, 'summary.csv'), index=False)
    with open(os.path.join(OUT, 'summary.md'), 'w', encoding='utf-8') as f:
        f.write('# FINO 复现实验结果汇总（CPU、预算缩减版）\n\n')
        f.write('> 论文每任务预算为 1M 离线 + 500k 在线；这里每任务仅 30k + 10k（约 3%）。\n')
        f.write('> `eval_success_max` 为所有评估点的最高成功率；0 表示缩减预算下策略未学会任务。\n\n')
        cols = list(df.columns)
        f.write('| ' + ' | '.join(cols) + ' |\n')
        f.write('|' + '|'.join(['---'] * len(cols)) + '|\n')
        for _, row in df.iterrows():
            f.write('| ' + ' | '.join(str(row[c]) for c in cols) + ' |\n')
        f.write('\n')
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
