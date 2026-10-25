# -*- coding: utf-8 -*-
"""Swap the misplaced page numbers on RL deck slides 11/12 and update the limitation card."""
import io

P = r'E:\FINO\course\build_rl_deck.py'
s = io.open(P, encoding='utf-8').read()

# swap the two footer numbers (each appears once)
assert s.count("footer(s, 11)") == 1 and s.count("footer(s, 12)") == 1
s = s.replace("footer(s, 12)", "footer(s, __TMP__)", 1)
s = s.replace("footer(s, 11)", "footer(s, 12)", 1)
s = s.replace("footer(s, __TMP__)", "footer(s, 11)", 1)

old = ("('⚠️ 局限', '3/45 任务、单 seed、无 D4RL、无真实环境消融；论文 schedule 与 β 符号与实现不一致，'\n"
       "              '严格对齐需要补做消融。', AMBER_D, AMBER_L),")
new = ("('⚠️ 局限', '3/45 任务、单 seed、无 D4RL；公式消融已补做（A/B 组），Q-guided K 与多 seed 未做；'\n"
       "              '主结果仍只能下\"方法有效、与论文同量级\"的结论。', AMBER_D, AMBER_L),")
assert old in s
s = s.replace(old, new, 1)
io.open(P, 'w', encoding='utf-8').write(s)
print('RL deck footers and limitation card fixed')
