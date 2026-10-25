# -*- coding: utf-8 -*-
"""Update RL deck coverage/conclusion with round-2 results."""
import io

P = r'E:\FINO\course\build_rl_deck.py'
s = io.open(P, encoding='utf-8').read()


def rep(old, new):
    global s
    assert old in s, 'ANCHOR NOT FOUND: ' + old[:90]
    s = s.replace(old, new, 1)


# coverage table + summary
rep("           ('消融 / 多 seed', '多组', '未做')],",
    "           ('消融 / 多 seed', '多组', '已补')],")
rep("text(s, 1.0, 4.02, 5.35, 0.3, '合计完成：3/45 ≈ 7%（单 seed）', size=10, color=AMBER_D, bold=True)",
    "text(s, 1.0, 4.02, 5.35, 0.3, '合计完成：3/45 ≈ 7%；主实验单 seed，公式/机制消融已补', size=9.4, color=AMBER_D, bold=True)")

# coverage bottom impact text
rep("     '这些是\"论文描述与实现不一致\"或\"实验记录缺陷\"，不影响主结果的成立（我们以代码为准复现）；'\n"
    "     '但说明：① 若要严格对齐论文公式，需要做 schedule/β 的消融；② 单 seed + 3/45 覆盖，只能下\"方法有效、与论文同量级\"的结论，不能下\"完整复现论文\"的结论。',",
    "     '这些是\"论文描述与实现不一致\"或\"实验记录缺陷\"，不影响主结果的成立（我们以代码为准复现）；'\n"
    "     '已补消融：公式幅度/形状、seed=1、humanoid 跨任务、Q-guided K、beta 变体；仍只能下\"方法有效、与论文同量级\"的结论，'\n"
    "     '不能下\"完整复现论文\"的结论（覆盖 3/45，无 D4RL）。',")

# conclusion analysis card
rep("('🔍 分析结论', '训练损失与 Q 值收敛正常；蒸馏保真（ED≈0.006）；公式消融显示论文 Eq.(7) 字面实现会崩溃'\n"
    "                 '（0.04），官方实现最优（1.00），β 符号影响很小（0.94 vs 0.96）。', BLUE, BLUE_L),",
    "('🔍 分析结论', '训练损失与 Q 值收敛正常；蒸馏保真（ED≈0.006）；Eq.(7) 未缩放会崩溃（0.04/0.12/0.00），'\n"
    "                 '但峰值归一化后恢复（0.94–1.00）→ 问题在幅度而非形状；K=1/4/10 为 0.90/0.94/0.96。', BLUE, BLUE_L),")

# conclusion limitation card
rep("('⚠️ 局限', '3/45 任务、单 seed、无 D4RL；公式消融已补做（A/B 组），Q-guided K 与多 seed 未做；'\n"
    "              '主结果仍只能下\"方法有效、与论文同量级\"的结论。', AMBER_D, AMBER_L),",
    "('⚠️ 局限', '3/45 任务、D4RL 未覆盖、部分任务单 seed；公式幅度/形状、seed=1、跨任务、K 已补做；'\n"
    "              '仍只能下\"方法有效、与论文同量级\"的结论。', AMBER_D, AMBER_L),")

io.open(P, 'w', encoding='utf-8').write(s)
print('RL deck coverage/conclusion updated')
