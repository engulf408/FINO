# -*- coding: utf-8 -*-
"""Append an experiment-coverage section to the report (45-task accounting)."""
import io

P = r'E:\FINO\docs\02_reproduction_report.md'
s = io.open(P, encoding='utf-8').read()
anchor = '---\n\n## 11. 参考'
assert anchor in s
section = '''---

## 12. 实验覆盖盘点：与原论文的对比

论文 README 提供 45 个任务命令：**OGBench 35 个 singletask 任务**（7 个环境 × 5 个 task）+ **D4RL 10 个任务**；
论文主表比较 FINO 与多个 baseline，并包含噪声注入、Q-guided 采样、熵自适应 β、flow steps、α 等消融实验。

本复现的实际覆盖（截至 2026-10-21 18:20）：

| 类别 | 论文规模 | 本复现已完成（全预算） | 进行中 | 覆盖率 |
|---|---:|---:|---:|---:|
| OGBench 主任务（FINO） | 35 | 1（antmaze-large task1：**最终成功率 100%**） | 2（cube-double-play ~75%、humanoidmaze-medium ~75%） | 1/35 完整；含进行中 ≈ 8.6% |
| OGBench 主任务（FQL baseline） | 35 | 1（antmaze-large task1：**92%**） | 0 | 1/35 ≈ 2.9% |
| D4RL 任务（FINO） | 10 | 0 | 0 | 0% |
| 论文消融实验 | 约 5 组 | 2D 玩具机制验证（非论文环境，只验证机制方向） | — | 机制级，非论文环境 |
| 多 seed / 方差 | 3–5 seeds | 1 seed（seed=0） | — | 未做 |
| 像素环境 / 视频 | 有 | 0 | — | 未做 |

**已完成的核心全预算实验**（GPU，论文预算 1M 离线 + 500k 在线、每 50k 步 × 50 episodes）：

- `antmaze-large-navigate-singletask-v0`：**FINO 100% vs FQL 92%**（§7.2，31 个评估点 × 50 episodes）；
- 进行中（约 75%）：`cube-double-play-singletask-v0`（FINO，当前成功率 ~0.5）、
  `humanoidmaze-medium-navigate-singletask-v0`（FINO，当前成功率 0.6–0.95，在线阶段仍有提升）。

**结论**：论文主表 45 个任务中，本复现完整完成 **1 个任务、2 个 agent**（约 2.2% 的任务覆盖），
另有 2 个 OGBench 任务接近完成（含进行中约 7%）。覆盖的是"最核心的验证性任务"，
而不是论文的全量表格；要接近论文规模还需要：

1. 再选 3–5 个代表任务（如 humanoidmaze-large、antmaze-giant、puzzle-4x4），约 **10–20 GPU 小时**；
2. 每个任务 3–5 个 seed，约 **30–60 GPU 小时**；
3. 论文消融（`--agent.noise_scale=0`、`flow_steps`、`q_agg`、`alpha` 扫描），约 **20–40 GPU 小时**；
4. 总计约 **50–100 GPU 小时**（本机 RTX 5070 约 3–5 天；云单卡 A100 约 2 天）；
5. 全量 45 任务 × 多 seed 约 **400–900 GPU 小时**。

'''
s = s.replace(anchor, section + anchor, 1)
io.open(P, 'w', encoding='utf-8').write(s)
print('coverage section added, lines:', len(s.splitlines()))
