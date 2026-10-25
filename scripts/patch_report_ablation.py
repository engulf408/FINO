# -*- coding: utf-8 -*-
"""Insert completed ablation results into the optimization-course report outline + README."""
import io

REPORT = r'E:\FINO\course\report_outline.md'
s = io.open(REPORT, encoding='utf-8').read()
anchor = '### 6.5 实现审计发现（原创，已可写）'
assert anchor in s
section = '''### 6.4b 消融结果（已完成，2026-10-24，antmaze-large）

**批次 A：噪声 schedule**（从零 1M 离线 + 500k 在线；评估每 100k×50 episodes）

| 变体 | 最终 success | 峰值 | last-5 | BC flow loss（末值） | 说明 |
|---|---:|---:|---:|---:|---|
| 官方指数 η=0.1（基线） | **1.00** | 1.00 | 0.98 | 0.285 | 最优 |
| 无噪声 η=0 | 0.96 | 0.98 | 0.88 | 0.257 | 离线拟合最好，但未超过官方 |
| 指数 η=0.3 | 0.96 | 0.96 | 0.75 | 0.303 | 噪声加大无收益 |
| 论文 Eq.(7) 二次型 | **0.04** | 0.62 | 0.01 | **0.422** | 在线阶段崩溃 |

**批次 B：β 更新**（从 1M checkpoint 恢复，在线 500k；评估每 50k×50）

| 变体 | 最终 success | 峰值 | last-5 | β 末值 |
|---|---:|---:|---:|---:|
| 官方 β（恢复基线） | 0.94 | 1.00 | 0.960 | 12.91 |
| 论文 Eq.(9) 印刷符号 | 0.96 | 1.00 | 0.960 | 7.07 |
| 固定 β=10 | 0.94 | 1.00 | 0.944 | 10.00 |

![公式消融](../results/fig_ablation_schedule_success.png)

**结论**：
1. **论文 Eq.(7) 二次型 schedule 按字面实现会破坏训练**（最终 0.04，在线阶段从 1M 的 0.62 崩到 0.02–0）；原因是路径中段注入噪声约 0.229，是官方实现的 ~340 倍，BC flow 拟合显著变差（loss 0.42 vs 0.26–0.29）、critic 崩溃（q_max −10.4）。
2. **官方指数 schedule 是最优的**（1.00）；无噪声与 η=0.3 均为 0.96 —— 噪声不是越强越好，官方 schedule 的关键作用是"数据端小方差"的支持集扩展。
3. **β 更新符号方向影响很小**（0.94 vs 0.96），固定 β=10 略低（0.94）——对偶更新有收益但方向不敏感，鲁棒性较好。
4. 单 seed，但 0.04 vs 1.00 是量级差异，足以支持"公式逻辑需要修正"的结论。

'''
s = s.replace(anchor, section + anchor, 1)

# update the TODO checklist
s = s.replace('- [ ] 跑批次 A（噪声 schedule/η，4 组）并出表/图 → 填 §6.2\n'
              '- [ ] 跑批次 B（β 三种变体）→ 填 §6.3\n'
              '- [ ] 跑批次 C（K=1/4/10）→ 填 §6.4\n',
              '- [x] 批次 A（噪声 schedule/η，4 组）已完成（2026-10-24）→ 见 §6.4b\n'
              '- [x] 批次 B（β 三种变体）已完成 → 见 §6.4b\n'
              '- [ ] 批次 C（K=1/4/10，可选，未做）→ 可作为后续工作\n')
io.open(REPORT, 'w', encoding='utf-8').write(s)
print('report outline updated, lines:', len(s.splitlines()))

# ---------------------------------------------------------------- README
P = r'E:\FINO\README.md'
r = io.open(P, encoding='utf-8').read()
old = '> 扩展任务（同样全预算）：**humanoidmaze-medium 100%、cube-double-play 72%（峰值 92%）**；'
new = ('> 公式逻辑消融（2026-10-24）：论文 Eq.(7) 二次型 schedule 按字面实现会崩溃（最终 0.04），'
       '官方指数 schedule 最优（1.00）；β 符号方向影响很小（0.94 vs 0.96）。\n>\n'
       '> 扩展任务（同样全预算）：**humanoidmaze-medium 100%、cube-double-play 72%（峰值 92%）**；')
assert old in r
r = r.replace(old, new, 1)
io.open(P, 'w', encoding='utf-8').write(r)
print('README updated')
