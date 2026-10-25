# -*- coding: utf-8 -*-
"""Add extended-task results to the report and update the coverage accounting."""
import io

P = r'E:\FINO\docs\02_reproduction_report.md'
s = io.open(P, encoding='utf-8').read()


def rep(old, new):
    global s
    assert old in s, 'ANCHOR NOT FOUND: ' + old[:90]
    s = s.replace(old, new, 1)


# ---------- §7.5 extended results ----------
anchor = '| README 全量 45（OGBench）+10（D4RL） | ≈ 55 天 | **5–9 天**（单卡） |\n'
ext = '''
### 7.5 扩展任务结果（GPU 全预算，2026-10-21）

在 antmaze-large 之外，又用同一套 GPU 环境按论文预算跑了 2 个 OGBench 默认任务（FINO）：

| 任务 | agent | 最终 success | 峰值 | 最后 5 次均值 | 耗时 / 评估协议 |
|---|---|---:|---:|---:|---|
| antmaze-large | FINO | **1.00** | 1.00 | 0.98 | 150.3 min / 31×50 eps |
| humanoidmaze-medium | FINO | **1.00** | 1.00 | 0.96 | 127.4 min / 16×20 eps\\* |
| cube-double-play | FINO | **0.72** | 0.92 | 0.84 | 144.2 min / 31×50 eps |
| antmaze-large（参考） | FQL | 0.92 | 0.98 | 0.88 | 136.6 min / 31×50 eps |

\\* humanoidmaze-medium 为节省时间采用每 100k 步 × 20 episodes；其余按论文协议（每 50k 步 × 50 episodes）。

![三个任务的评估曲线](../results/fig_gpu_extended_curves.png)

**结论**：FINO 在 3 个 OGBench 任务上均学到有效策略（antmaze-large 与 humanoidmaze-medium 达到 100%，
cube-double-play 72%/峰值 92%）；其中 antmaze-large 与 FQL 的同预算对比方向与论文一致（100% vs 92%）。
全部为单 seed（seed=0），只用于方向性验证，不作显著性结论。

'''
rep(anchor, anchor + ext)

# ---------- §0 bullet 3 ----------
rep('扩展任务 humanoidmaze-medium / cube-double-play 正在 GPU 上运行；',
    '扩展任务 humanoidmaze-medium 与 cube-double-play 也已完成全预算运行（100% / 72%）；')

# ---------- §12 coverage rows ----------
rep("| OGBench 主任务（FINO） | 35 | 1（antmaze-large task1：**最终成功率 100%**） | 2（cube-double-play ~75%、humanoidmaze-medium ~75%） | 1/35 完整；含进行中 ≈ 8.6% |",
    "| OGBench 主任务（FINO） | 35 | **3**（antmaze-large 100%、humanoidmaze-medium 100%、cube-double-play 72%） | 0 | 3/35 ≈ 8.6% |")
rep('- `antmaze-large-navigate-singletask-v0`：**FINO 100% vs FQL 92%**（§7.2，31 个评估点 × 50 episodes）；\n'
    '- 进行中（约 75%）：`cube-double-play-singletask-v0`（FINO，当前成功率 ~0.5）、\n'
    '  `humanoidmaze-medium-navigate-singletask-v0`（FINO，当前成功率 0.6–0.95，在线阶段仍有提升）。',
    '- `antmaze-large-navigate-singletask-v0`：**FINO 100% vs FQL 92%**（§7.2，31 个评估点 × 50 episodes）；\n'
    '- `humanoidmaze-medium-navigate-singletask-v0`：**FINO 100%**（峰值 100%，127.4 min）；\n'
    '- `cube-double-play-singletask-v0`：**FINO 72%**（峰值 92%，最后 5 次均值 84%，144.2 min）。')
rep('**结论**：论文主表 45 个任务中，本复现完整完成 **1 个任务、2 个 agent**（约 2.2% 的任务覆盖），\n'
    '另有 2 个 OGBench 任务接近完成（含进行中约 7%）。覆盖的是"最核心的验证性任务"，',
    '**结论**：论文主表 45 个任务中，本复现完整完成 **3 个 OGBench 任务**（其中 antmaze-large 含 FINO+FQL 同预算对比，\n'
    '约 6.7% 的任务覆盖；按 OGBench 35 个任务计约 8.6%）。覆盖的是"最核心的验证性任务"，')

# ---------- §10 deliverables ----------
rep('| `results/` | CPU/GPU 训练评估曲线、`summary.csv`、`summary.md`、`gpu_summary.csv`、`checkpoint_analysis*.json` |',
    '| `results/` | CPU/GPU 训练评估曲线、`summary*`、`gpu_summary*`、`extended_summary*`、`checkpoint_analysis*.json` |')

io.open(P, 'w', encoding='utf-8').write(s)
print('extended results added to report, lines:', len(s.splitlines()))
