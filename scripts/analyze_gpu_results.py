"""Analyze full-budget GPU runs (1M offline + 500k online) under E:/FINO/gpu_exp.

Usage:
    python analyze_gpu_results.py [--exp-root E:/FINO/gpu_exp] [--out E:/FINO/results]
Produces:
    results/fig_gpu_eval_<env>.png        eval success/return comparison (FINO vs FQL if present)
    results/fig_gpu_training_<agent>_<env>.png
    results/gpu_summary.csv / gpu_summary.md
"""
import argparse
import json
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_ENV = 'antmaze-large-navigate-singletask-v0'
OFFLINE_STEPS = 1_000_000
TOTAL_STEPS = 1_500_000


def load(exp_root, agent, env):
    d = os.path.join(exp_root, agent, env, '0')
    tp, ep = os.path.join(d, 'train.csv'), os.path.join(d, 'eval.csv')
    if not os.path.exists(tp):
        return None, None, d
    train = pd.read_csv(tp)
    ev = pd.read_csv(ep) if os.path.exists(ep) else None
    return train, ev, d


def eval_figure(rows, env, out):
    """rows: dict agent -> eval dataframe"""
    if not rows:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for agent, ev in rows.items():
        color = {'FINO': 'tab:orange', 'FQL': 'tab:blue'}.get(agent)
        axes[0].plot(ev['step'], ev['evaluation/success'], marker='o', ms=3, label=agent, color=color)
        axes[1].plot(ev['step'], ev['evaluation/episode.return'], marker='o', ms=3, label=agent, color=color)
    for ax, title, ylab in zip(
            axes, ['Evaluation success rate', 'Evaluation episode return'],
            ['success rate', 'episode return']):
        ax.axvline(OFFLINE_STEPS, color='gray', ls='--', lw=1, label='offline→online')
        ax.axvspan(OFFLINE_STEPS, TOTAL_STEPS, color='gray', alpha=0.08)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('update step')
        ax.set_ylabel(ylab)
        ax.grid(alpha=.3)
        ax.legend(fontsize=8)
    axes[0].set_ylim(-0.02, 1.02)
    fig.suptitle(f'{env} — full budget (1M offline + 500k online)', fontsize=12)
    fig.tight_layout()
    path = os.path.join(out, f'fig_gpu_eval_{env}.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print('wrote', path)
    return path


def training_figure(train, agent, env, out):
    steps = train['step']
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.5))
    ax = axes[0, 0]
    ax.plot(steps, train['training/actor/bc_flow_loss'], label='BC flow (train)')
    if 'validation/actor/bc_flow_loss' in train:
        ax.plot(steps, train['validation/actor/bc_flow_loss'], label='BC flow (val)')
    ax.set_title('Flow-matching BC loss'); ax.legend(); ax.grid(alpha=.3)

    ax = axes[0, 1]
    if 'training/actor/distill_loss' in train:
        ax.plot(steps, train['training/actor/distill_loss'], label='distill (train)')
    ax.set_title('One-step distillation loss'); ax.legend(); ax.grid(alpha=.3)

    ax = axes[1, 0]
    ax.plot(steps, train['training/critic/critic_loss'], label='critic TD (train)')
    ax.set_title('Critic TD loss'); ax.legend(); ax.grid(alpha=.3)

    ax = axes[1, 1]
    ax.plot(steps, train['training/actor/q'], label='actor Q')
    ax.set_title('Q value of policy actions'); ax.legend(); ax.grid(alpha=.3)

    for ax in axes.ravel():
        ax.axvline(OFFLINE_STEPS, color='gray', ls='--', lw=1)
        ax.set_xlabel('update step')
    fig.suptitle(f'{agent} — {env} (full budget)', fontsize=12)
    fig.tight_layout()
    path = os.path.join(out, f'fig_gpu_training_{agent}_{env}.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print('wrote', path)
    return path


def summarize(agent, env, train, ev, run_dir):
    def at(step):
        if ev is None or len(ev) == 0:
            return None
        sub = ev[ev['step'] <= step]
        return float(sub['evaluation/success'].iloc[-1]) if len(sub) else None

    def ret_at(step):
        if ev is None or len(ev) == 0:
            return None
        sub = ev[ev['step'] <= step]
        return float(sub['evaluation/episode.return'].iloc[-1]) if len(sub) else None

    row = dict(
        agent=agent, env=env, run_dir=run_dir,
        offline_steps=int(train['step'].max()) if train is not None else None,
        updates_logged=len(train) if train is not None else 0,
        total_minutes=round(float(train['time/total_time'].iloc[-1]) / 60, 1) if train is not None else None,
        ms_per_update=round(1000 * float(train['time/total_time'].iloc[-1]) / float(train['step'].iloc[-1]), 2)
        if train is not None else None,
        eval_points=len(ev) if ev is not None else 0,
        success_peak=float(ev['evaluation/success'].max()) if ev is not None and len(ev) else None,
        success_at_1M=at(1_000_000),
        success_final=at(1_500_000),
        return_at_1M=ret_at(1_000_000),
        return_final=ret_at(1_500_000),
        bc_flow_last=round(float(train['training/actor/bc_flow_loss'].iloc[-1]), 4) if train is not None else None,
        critic_loss_last=round(float(train['training/critic/critic_loss'].iloc[-1]), 4) if train is not None else None,
        q_last=round(float(train['training/actor/q'].iloc[-1]), 2) if train is not None else None,
    )
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp-root', default=r'E:\FINO\gpu_exp')
    ap.add_argument('--out', default=r'E:\FINO\results')
    ap.add_argument('--env', default=DEFAULT_ENV)
    ap.add_argument('--agents', nargs='*', default=['FINO', 'FQL'])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows, evs = [], {}
    for agent in args.agents:
        train, ev, d = load(args.exp_root, agent, args.env)
        if train is None:
            print('skip', agent, '(no data)')
            continue
        training_figure(train, agent, args.env, args.out)
        if ev is not None and len(ev):
            evs[agent] = ev
        rows.append(summarize(agent, args.env, train, ev, d))

    if evs:
        eval_figure(evs, args.env, args.out)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.out, 'gpu_summary.csv')
    md_path = os.path.join(args.out, 'gpu_summary.md')
    if len(df):
        old = pd.read_csv(csv_path) if os.path.exists(csv_path) else None
        if old is not None:
            df = pd.concat([old[~old.set_index(['agent', 'env']).index.isin(df.set_index(['agent', 'env']).index)], df])
        df.to_csv(csv_path, index=False)
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write('# GPU 全预算复现结果汇总（1M 离线 + 500k 在线）\n\n')
            cols = list(df.columns)
            f.write('| ' + ' | '.join(cols) + ' |\n')
            f.write('|' + '|'.join(['---'] * len(cols)) + '|\n')
            for _, r in df.iterrows():
                f.write('| ' + ' | '.join(str(r[c]) for c in cols) + ' |\n')
    print(df.to_string(index=False))
    print('wrote', csv_path, md_path)


if __name__ == '__main__':
    main()
