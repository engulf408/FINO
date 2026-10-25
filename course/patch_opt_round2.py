# -*- coding: utf-8 -*-
"""Update optimization deck slide-9 with round-2 findings."""
import io

P = r'E:\FINO\course\build_course_ppt.py'
s = io.open(P, encoding='utf-8').read()


def rep(old, new):
    global s
    assert old in s, 'ANCHOR NOT FOUND: ' + old[:90]
    s = s.replace(old, new, 1)


rep("""bullets(s, 7.2, 2.2, 5.1, 2.2, [
    '论文 Eq.(7) 二次型 schedule：最终 0.04（在线崩溃），BC flow loss 0.42 vs 0.26',
    '官方指数 schedule：1.00；无噪声 0.96；η=0.3 0.96 → 噪声不是越大越好',
    'β 论文 Eq.(9) 符号：0.96 vs 官方 0.94；固定 β=10：0.94 → 符号影响很小',
    '2D 机制：蒸馏 EMD 0.0004 / 5.38×，Q-guided d=6 +0.63，β 方向一致'], size=8.8, gap=3, bullet_color=GREEN)""",
    """bullets(s, 7.2, 2.2, 5.1, 2.2, [
    'Eq.(7) 未缩放：0.04（在线崩溃）；峰值 0.229 = 官方 340×，BC flow loss 0.42',
    '峰值归一化（0.1）→ 0.94；方差读法（σ=α²）→ 1.00：形状可用，缺归一化',
    '崩溃可重复：seed1 0.12；humanoidmaze 0.00；官方 seed1 0.96',
    'Q-guided K=1/4/10：0.90/0.94/0.96；β 符号 0.94 vs 0.96（影响小）'], size=8.5, gap=2, bullet_color=GREEN)""")
rep("""text(s, 1.0, 5.2, 11.2, 1.05,
     '这些结果把"论文公式 vs 实现"的差异变成了可复现实验：按论文 Eq.(7) 实现会破坏训练（0.04 vs 1.00），'
     '官方指数 schedule 才有效；β 符号方向影响很小。对优化方法的启示：噪声方差 schedule 是关键超参，'
     'β 对偶步长的鲁棒性较好；单 seed 但效应量级差异显著（0.04 vs 1.00）。', size=10, color=INK, line=1.35)""",
    """text(s, 1.0, 5.2, 11.2, 1.05,
     '这些结果把"论文公式 vs 实现"的差异变成了可复现实验：按论文给定 η 字面实现 Eq.(7) 会破坏训练（0.04/0.12/0.00），'
     '但把峰值归一化到 0.1 后恢复到 0.94–1.00 → 问题在噪声幅度的归一化，而非二次型形状；'
     'β 符号方向影响很小。对优化方法的启示：噪声方差 schedule 的关键是"幅度标定"，β 对偶步长鲁棒。', size=9.6, color=INK, line=1.35)""")

io.open(P, 'w', encoding='utf-8').write(s)
print('optimization deck slide 9 updated')
