# -*- coding: utf-8 -*-
"""Add a formula-logic ablation slide to the RL deck and update the coverage/conclusion numbering."""
import io

P = r'E:\FINO\course\build_rl_deck.py'
s = io.open(P, encoding='utf-8').read()


def rep(old, new):
    global s
    assert old in s, 'ANCHOR NOT FOUND: ' + old[:100]
    s = s.replace(old, new, 1)


# ---------------- shift numbers of existing slides 10/11 -> 11/12
rep("header(s, '06 · 覆盖面与问题', '覆盖 3/45 任务；复现中发现 4 个实现问题', 10)",
    "header(s, '06 · 覆盖面与问题', '覆盖 3/45 任务；复现中发现 4 个实现问题', 11)")
rep("footer(s, 10)", "footer(s, 11)")
rep("header(s, '07 · 结论', '结论与展望', 11)", "header(s, '07 · 结论', '结论与展望', 12)")
rep("footer(s, 11)", "footer(s, 12)")

# ---------------- conclusion card 2: include ablation result
rep("('🔍 分析结论', '训练损失与 Q 值收敛正常；一步蒸馏保真（ED≈0.006）；2D 实验与真实任务结论一致；'\n"
    "                 'FINO 的收益主要来自在线阶段的支持集扩展。', BLUE, BLUE_L),",
    "('🔍 分析结论', '训练损失与 Q 值收敛正常；蒸馏保真（ED≈0.006）；公式消融显示论文 Eq.(7) 字面实现会崩溃'\n"
    "                 '（0.04），官方实现最优（1.00），β 符号影响很小（0.94 vs 0.96）。', BLUE, BLUE_L),")

# ---------------- insert new slide 10 before coverage block
marker = "# ---------------------------------------------------------------- 10 coverage & issues"
assert marker in s
new_slide = '''# ---------------------------------------------------------------- 10 formula-logic ablation
s = prs.slides.add_slide(BLANK)
header(s, '05 · 原创消融', '公式逻辑验证：论文公式 vs 实现（antmaze-large，全预算）', 10)
card(s, LM, 1.6, 6.1, 1.75)
text(s, 1.0, 1.7, 5.6, 0.28, 'A. 噪声 schedule（从零 1M+500k）', size=10.8, bold=True, color=NAVY)
add_table(s, 1.0, 2.02, 5.6, [0.46, 0.18, 0.18, 0.18],
          ['变体', '最终', '峰值', 'last-5'],
          [('官方指数 η=0.1', '1.00', '1.00', '0.98'),
           ('无噪声 η=0', '0.96', '0.98', '0.88'),
           ('指数 η=0.3', '0.96', '0.96', '0.75'),
           ('论文二次型 Eq.(7)', '0.04', '0.62', '0.01')],
          font_size=8.4, header_size=8.4, row_h=0.28,
          aligns=[PP_ALIGN.LEFT, PP_ALIGN.CENTER, PP_ALIGN.CENTER, PP_ALIGN.CENTER])
card(s, 6.95, 1.6, 5.63, 1.75)
text(s, 7.2, 1.7, 5.1, 0.28, 'B. β 更新（1M ckpt 起在线 500k）', size=10.8, bold=True, color=NAVY)
add_table(s, 7.2, 2.02, 5.1, [0.44, 0.18, 0.18, 0.20],
          ['变体', '最终', '峰值', 'β 末值'],
          [('官方 β', '0.94', '1.00', '12.9'),
           ('论文 Eq.(9) 符号', '0.96', '1.00', '7.1'),
           ('固定 β=10', '0.94', '1.00', '10.0')],
          font_size=8.4, header_size=8.4, row_h=0.3,
          aligns=[PP_ALIGN.LEFT, PP_ALIGN.CENTER, PP_ALIGN.CENTER, PP_ALIGN.CENTER])
add_image(s, os.path.join(RES, 'fig_ablation_schedule_success.png'), LM, 3.42, 6.35, 2.12)
card(s, 7.35, 3.42, 5.23, 2.12, fill=RED_L, line_color=RGBColor(0xE8, 0xC4, 0xC4))
text(s, 7.55, 3.52, 4.8, 0.28, '分析', size=10.8, bold=True, color=RED)
bullets(s, 7.55, 3.84, 4.8, 1.6, [
    '论文 Eq.(7) 字面实现：在线期崩溃（1M 时 0.62 → 1.1M 后 0.02–0）',
    'BC flow loss 0.42 vs 官方 0.29 / 无噪声 0.26 → 中段噪声（约 340×）破坏拟合',
    '官方指数 schedule 最优；无噪声、η=0.3 均 0.96 → 噪声不是越强越好',
    'β 符号方向影响很小（0.94 vs 0.96）；固定 β 略低'], size=8.2, gap=2, bc=RED)
text(s, LM, 5.75, CW, 0.9,
     '结论：论文公式与实现不一致不仅是文档问题——按 Eq.(7) 实现会破坏训练（0.04 vs 1.00）；'
     '以实现为准的复现策略成立，同时说明论文的公式逻辑需要修正。单 seed，但效应量级差异显著。',
     size=10.2, color=INK, line=1.3)
footer(s, 10)
notes(s, '公式逻辑消融：论文二次型 schedule 按字面实现会崩溃（0.04），官方指数 schedule 最优（1.00）；'
         '无噪声和 eta=0.3 都是 0.96，说明噪声不是越强越好；beta 符号方向影响很小（0.94 vs 0.96）。')

'''
s = s.replace(marker, new_slide + marker, 1)
io.open(P, 'w', encoding='utf-8').write(s)
print('RL deck patched with ablation slide; lines:', len(s.splitlines()))
