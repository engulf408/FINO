# -*- coding: utf-8 -*-
"""Update report outline and README with round-2 findings."""
import io

# ---------------- report outline: extend the 6.4b conclusion
P = r'E:\FINO\course\report_outline.md'
s = io.open(P, encoding='utf-8').read()
anchor = '### 6.5 实现审计发现（原创，已可写）'
assert anchor in s
extra = '''**Round 2 补充（2026-10-26，antmaze 全预算 + humanoid 跨任务）**：

| 变体 | 最终 | 说明 |
|---|---:|---|
| Eq.(7) 未缩放（seed0/seed1） | 0.04 / 0.12 | 崩溃可重复 |
| Eq.(7) 峰值匹配（峰值=0.1） | 0.94 | 形状可用 |
| Eq.(7) 方差读法（σ=α²，峰值 0.053） | 1.00 | 读法歧义 |
| 官方 seed0 / seed1 | 1.00 / 0.96 | 官方稳定 |
| humanoid 官方 / Eq.(7) | 1.00 / 0.00 | 跨任务崩溃 |
| Q-guided K=1 / 4 / 10 | 0.90 / 0.94 / 0.96 | 多候选略优 |

**结论修正**：Eq.(7) 的失败是"幅度/归一化"问题而非"形状"问题——归一化后恢复到 0.94–1.00；
按论文给定的 η=0.1 字面实现会崩溃，且崩溃在 seed=1 与 humanoidmaze 上都可复现。

'''
s = s.replace(anchor, extra + anchor, 1)
io.open(P, 'w', encoding='utf-8').write(s)
print('report outline updated')

# ---------------- README
P = r'E:\FINO\README.md'
r = io.open(P, encoding='utf-8').read()
old = ('> 公式逻辑消融（2026-10-24）：论文 Eq.(7) 二次型 schedule 按字面实现会崩溃（最终 0.04），'
       '官方指数 schedule 最优（1.00）；β 符号方向影响很小（0.94 vs 0.96）。')
new = ('> 公式逻辑消融（2026-10-24/26）：Eq.(7) 未缩放崩溃（0.04；seed1 0.12；humanoid 0.00），'
       '但峰值归一化后恢复（0.94–1.00）→ 问题在幅度而非形状；β 符号影响很小（0.94 vs 0.96）；'
       'Q-guided K=1/4/10 为 0.90/0.94/0.96。')
assert old in r
r = r.replace(old, new, 1)
io.open(P, 'w', encoding='utf-8').write(r)
print('README updated')
