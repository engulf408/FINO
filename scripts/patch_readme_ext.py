# -*- coding: utf-8 -*-
"""Update README with extended-task results."""
import io

P = r'E:\FINO\README.md'
s = io.open(P, encoding='utf-8').read()
old9 = '> **FINO 最终成功率 100%（1.00），FQL 92%（0.92）**；FINO 在线微调阶段 0.82 → 1.00；单任务 2.3–2.5 小时。'
new9 = (old9 + '\n>\n> 扩展任务（同样全预算）：**humanoidmaze-medium 100%、cube-double-play 72%（峰值 92%）**；'
        'FINO 已在 3 个 OGBench 任务上完成论文预算复现（3/45 ≈ 6.7%）。')
old11 = ('> 详细数据：[`results/gpu_summary.md`](results/gpu_summary.md)、[`gpu_exp/`](gpu_exp/)、'
         '[`docs/02_reproduction_report.md`](docs/02_reproduction_report.md) §7.2。')
new11 = ('> 详细数据：[`results/gpu_summary.md`](results/gpu_summary.md)、'
         '[`results/extended_summary.md`](results/extended_summary.md)、[`gpu_exp/`](gpu_exp/)、'
         '[`docs/02_reproduction_report.md`](docs/02_reproduction_report.md) §7.2 / §7.5。')
assert old9 in s and old11 in s
s = s.replace(old9, new9, 1).replace(old11, new11, 1)
io.open(P, 'w', encoding='utf-8').write(s)
print('README updated ok')
