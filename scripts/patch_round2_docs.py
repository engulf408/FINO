# -*- coding: utf-8 -*-
"""Consistency fix for the scale factor in the optimization deck, plus docs/README round-2 updates."""
import io

# ---------------- optimization deck: 250x -> 340x (t=0.5)
P = r'E:\FINO\course\build_course_ppt.py'
s = io.open(P, encoding='utf-8').read()
old = "    '噪声 schedule：代码 σ_t=0.1·e^{10(t−1)}，t=0.5 时 σ≈0.0009；论文 Eq.(7) 二次型在 t≈0.53 处 σ≈0.229（相差约 250×）',"
new = "    '噪声 schedule：代码 σ_t=0.1·e^{10(t−1)}，t=0.5 时 σ≈0.0007；论文 Eq.(7) 二次型在 t≈0.5 处 σ≈0.229（相差约 340×）',"
assert old in s
s = s.replace(old, new, 1)
io.open(P, 'w', encoding='utf-8').write(s)
print('optimization deck scale factor fixed')

# ---------------- docs/03 append round-2
P = r'E:\FINO\docs\03_formula_ablation_results.md'
s = io.open(P, encoding='utf-8').read()
if '## 5. Round 2' not in s:
    s += '''
## 5. Round 2：公式"形状 vs 幅度"、多 seed、跨任务、Q-guided K（2026-10-26）

| 组 | 变体 | 最终 success | 峰值 | last-5 | 说明 |
|---|---|---:|---:|---:|---|
| 噪声 schedule | 官方指数 η=0.1 | 1.00 | 1.00 | 0.98 | 基线 |
| | 无噪声 η=0 | 0.96 | 0.98 | 0.88 | 去掉注入 |
| | 指数 η=0.3 | 0.96 | 0.96 | 0.75 | 幅度增大无收益 |
| | **Eq.(7) 未缩放** | **0.04** | 0.62 | 0.01 | 中段 σ=0.229 |
| | **Eq.(7) 峰值匹配**（×0.4365，峰值 0.1） | **0.94** | 0.94 | 0.76 | 形状可用 |
| | **Eq.(7) 方差读法**（σ=α²，峰值 0.053） | **1.00** | 1.00 | 0.86 | 读法歧义 |
| seed 稳健性 | 官方 seed0 / seed1 | 1.00 / 0.96 | 1.00 / 1.00 | 0.98 / 0.91 | 官方稳定 |
| | Eq.(7) seed0 / seed1 | 0.04 / 0.12 | 0.62 / 0.52 | 0.01 / 0.03 | 崩溃可重复 |
| 跨任务 | humanoidmaze 官方 | 1.00 | 1.00 | 0.96 | 基线 |
| | humanoidmaze Eq.(7) 未缩放 | **0.00** | 0.00 | 0.00 | 跨任务崩溃 |
| Q-guided K | K=1 / K=4 / K=10 | 0.90 / 0.94 / 0.96 | 0.96 / 1.00 / 1.00 | 0.80 / 0.96 / 0.97 | 多候选略优 |

![schedule scale](../results/fig_ablation_schedule_scale.png)

**结论（更新）**：
1. **失败在幅度、不在形状**：Eq.(7) 未缩放时峰值噪声 0.229（官方实现的约 340 倍）→ 0.04；把峰值归一化到 0.1 → 0.94；按方差读法（峰值 0.053）→ 1.00。
2. **崩溃可重复、跨任务**：seed1 0.12、humanoidmaze 0.00；官方 seed1 0.96 → 结论不是单 seed 偶然。
3. **Q-guided K**：K=1/4/10 = 0.90/0.94/0.96，多候选选择有轻微正向收益，不是决定性因素。
4. 对论文公式的解读：公式可作为"噪声方差应沿路径呈二次型"的设计依据，但必须做幅度归一化/标定；论文给定的 η=0.1 与公式直接组合会破坏训练。

复现命令：见 §4，round-2 新增 `FINO_QUAD_SCALE`、`FINO_NOISE_SCHEDULE=quad_var`、`FINO_QGUIDE_K` 三个开关；
一键脚本 `wsl_setup/abl_round2.sh`，分析脚本 `scripts/analyze_ablation_round2.py`。
'''
    io.open(P, 'w', encoding='utf-8').write(s)
    print('docs/03 updated with round-2')
else:
    print('docs/03 already has round-2')
