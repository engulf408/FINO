# -*- coding: utf-8 -*-
"""Update the optimization deck originality slide with the completed formula ablations."""
import io

P = r'E:\FINO\course\build_course_ppt.py'
s = io.open(P, encoding='utf-8').read()


def rep(old, new):
    global s
    assert old in s, 'ANCHOR NOT FOUND: ' + old[:100]
    s = s.replace(old, new, 1)


rep("header(s, '07 · 原创性', '原创分析：实现审计与 2D 机制证据', 9)",
    "header(s, '07 · 原创性', '原创分析：实现审计与公式消融', 9)")
rep("text(s, 7.2, 1.83, 5.1, 0.35, '② 2D 机制实验（定量验证）', size=13, bold=True, color=GREEN)",
    "text(s, 7.2, 1.83, 5.1, 0.35, '② 公式逻辑消融（真实任务，全预算）', size=13, bold=True, color=GREEN)")
rep("""bullets(s, 7.2, 2.2, 5.1, 2.2, [
    '一步蒸馏：学生 vs 教师 EMD 0.0004；推理 5.38× 加速',
    'Q-guided：d=6 时 Q 提升 +0.63；d=2 时 K=1 严格退化为恒等',
    'β 自适应：H≈−2.14 → 30 次更新 β 从 10 升到 21.3，方向与代码一致',
    '噪声强度 η 扫描：Lipschitz 1.02→0.545，EMD 0.0293→0.0393（平滑↑、保真↓的权衡）'], size=8.8, gap=3, bullet_color=GREEN)""",
    """bullets(s, 7.2, 2.2, 5.1, 2.2, [
    '论文 Eq.(7) 二次型 schedule：最终 0.04（在线崩溃），BC flow loss 0.42 vs 0.26',
    '官方指数 schedule：1.00；无噪声 0.96；η=0.3 0.96 → 噪声不是越大越好',
    'β 论文 Eq.(9) 符号：0.96 vs 官方 0.94；固定 β=10：0.94 → 符号影响很小',
    '2D 机制：蒸馏 EMD 0.0004 / 5.38×，Q-guided d=6 +0.63，β 方向一致'], size=8.8, gap=3, bullet_color=GREEN)""")
rep("""text(s, 1.0, 5.2, 11.2, 1.05,
     '噪声方差 schedule、β 对偶步长与目标熵、支持集大小（K）都是论文未充分指定的优化超参；'
     '我们的审计给出它们的实际取值与偏差，2D 实验给出方向性证据。真实任务上的 A/B/C 消融（噪声 schedule、β 变体、K=1/4/10，'
     '各 200k 在线步）已设计好，可作为后续工作补入，不阻塞本报告的结论。', size=10, color=INK, line=1.35)""",
    """text(s, 1.0, 5.2, 11.2, 1.05,
     '这些结果把"论文公式 vs 实现"的差异变成了可复现实验：按论文 Eq.(7) 实现会破坏训练（0.04 vs 1.00），'
     '官方指数 schedule 才有效；β 符号方向影响很小。对优化方法的启示：噪声方差 schedule 是关键超参，'
     'β 对偶步长的鲁棒性较好；单 seed 但效应量级差异显著（0.04 vs 1.00）。', size=10, color=INK, line=1.35)""")
rep("notes(s, '原创性：一是论文与代码的审计差异（噪声 schedule 差约250倍、beta符号相反等）；'\n"
    "         '二是2D机制实验的定量证据（蒸馏、Q-guided、beta、噪声强度）。真实任务消融已设计，作为后续工作。')",
    "notes(s, '原创性：一是论文与代码的审计差异；二是真实任务上的公式消融——论文 Eq.(7) 字面实现崩溃（0.04），'\n"
    "         '官方指数 schedule 最优（1.00），beta 符号影响很小（0.94 vs 0.96）。')")

io.open(P, 'w', encoding='utf-8').write(s)
print('optimization deck patched')
